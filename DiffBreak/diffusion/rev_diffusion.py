import math
import yaml
import numpy as np
import torch
from torchvision import transforms

from ..utils import ClassifierWrapper
from .diffusers import *
from .utils import *
from .grad_diff import DiffGrad, DiffGradEngine


class LossWrapper(torch.nn.Module):
    def __init__(self, loss_fn, dm):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.dm = dm
        self.label = None

    def forward(self, inputs, labels=None, **kwargs):
        if labels is None:
            assert self.label is not None
        else:
            self.label = labels.detach()

        return (
            self.loss_fn(inputs, self.label, **kwargs)
            .mean(0, keepdims=True)
            .view(-1, 1)
        )


class RevDiffusion(torch.nn.Module):
    def __init__(
        self,
        model,
        scaler,
        diffusion_steps,
        betas,
        timestep_map,
        image_size,
        effective_size,
        num_diffusion_timesteps=1000,
        precision=3,
        preprocess_diffusion_t=lambda t: t,
        batch_size=1,
        cond_args={},
    ):
        super().__init__()
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.precision = precision
        self.a = (1.0 - betas).cumprod(dim=0)
        self.t = diffusion_steps
        self.preprocess_diffusion_t = preprocess_diffusion_t
        dt = round(
            1 / self.num_diffusion_timesteps,
            precision,
        )
        self.batch_size = batch_size

        self.rev_vpsde = RevVPSDE(
            model,
            scaler,
            effective_size,
            betas,
            precision,
            timestep_map,
            preprocess_diffusion_t,
            cond_args,
        )

        solver_args = [self.rev_vpsde, dt]

        self.solvers = {
            "sdeint": SDEINT(*solver_args),
            "sdeint_adjoint": SDEINT_ADJOINT(*solver_args),
            "odeint": ODEINT(*solver_args),
            "odeint_adjoint": ODEINT_ADJOINT(*solver_args),
            "ddpm": DDPMSolver(*solver_args),
        }

        self.upscaler = transforms.Resize(
            (effective_size, effective_size), antialias=None
        )
        self.downscaler = transforms.Resize((image_size, image_size), antialias=None)

    def eval(self):
        self.rev_vpsde = self.rev_vpsde.eval()
        for s, solver in self.solvers.items():
            self.solvers[s] = solver.eval()
        return super().eval()

    def to(self=None, device=None):
        self.rev_vpsde = self.rev_vpsde.to(device)
        self.a = self.a.to(device)
        self.device = device
        for s, solver in self.solvers.items():
            self.solvers[s] = solver.to(device)
        return super().to(device)

    def select_solver(self, diffusion_type, adjoint, ode):
        assert diffusion_type in ["ddpm", "vpsde"]
        if diffusion_type == "ddpm":
            assert not adjoint
            return self.solvers["ddpm"]
        mtd, adj_mtd = (
            ("odeint", "odeint_adjoint") if ode else ("sdeint", "sdeint_adjoint")
        )
        return self.solvers[mtd if not adjoint else adj_mtd]

    def forward(
        self,
        allimg,
        t_start=None,
        t_end=0.0,
        steps=2,
        with_noise=True,
        adjoint=True,
        guide=None,
        bm=None,
        ode=False,
        diffusion_type="vpsde",
        batch_size=None,
        no_score=False,
    ):
        assert steps >= 2
        if steps > 2:
            assert (t_start - t_end) * self.num_diffusion_timesteps == steps - 1

        self.rev_vpsde.no_score = no_score
        solver = self.select_solver(diffusion_type, adjoint, ode)
        batch_size = self.batch_size if batch_size is None else batch_size
        t = self.preprocess_diffusion_t(int(t_start * self.num_diffusion_timesteps))
        factor = self.a[t - 1].sqrt() if with_noise else 1.0
        noise_factor = (1 - self.a[t - 1]).sqrt() if with_noise else 0.0
        allouts = []

        n_batch = int(np.ceil(allimg.shape[0] / batch_size))
        for step in range(n_batch):
            img = allimg[step * batch_size : (step + 1) * batch_size]
            allouts.extend(
                list(
                    solver(
                        img * factor + torch.randn_like(img) * noise_factor,
                        t_start=t_start,
                        t_end=t_end,
                        steps=steps,
                        guide=guide,
                        bm=bm[step] if isinstance(bm, list) else bm,
                        ode=ode,
                        no_score=no_score,
                    ).view(-1, *img.shape[1:])
                )
            )

        self.rev_vpsde.no_score = False
        return torch.cat(allouts, dim=0).view(-1, *allimg.shape[1:])


class DBPWrapper(ClassifierWrapper):
    def __initialize_diffusion_engine(
        self,
        dm,
        diffusion_type,
        diffusion_steps,
        ode=False,
        repeats=None,
        eval_batch_sz=32,
        deterministic=False,
        basic_adjoint_method=False,
        with_intermediate=False,
        dm_eval_batch_sz=None,
        power=0.5,
        power_2=0.25,
        diffusion_iterations=1,
        bpda=False,
        blind=False,
        forward_diff_only=False,
    ):
        self.ode = ode
        self.eval_batch_sz = eval_batch_sz if not deterministic else 1
        self.dm_eval_batch_sz = (
            self.eval_batch_sz
            if dm_eval_batch_sz is None or deterministic
            else dm_eval_batch_sz
        )
        assert self.dm_eval_batch_sz <= self.eval_batch_sz
        self.dm = DiffGradEngine(
            dm,
            self.model_fn,
            LossWrapper(self.orig_loss, dm) if self.orig_loss is not None else None,
            diffusion_type,
            diffusion_steps,
            ode,
            repeats=repeats,
            deterministic=deterministic,
            basic_adjoint_method=basic_adjoint_method,
            with_intermediate=with_intermediate,
            power=power,
            power_2=power_2,
            bpda=bpda,
            blind=blind,
            forward_diff_only=forward_diff_only,
            verbose=self.verbose,
        )
        self.diff_eval_args = {
            "diffusion_type": diffusion_type,
            "with_noise": False,
            "adjoint": False,
            "ode": self.ode,
            "t_start": self.dm.t_start / self.dm.dm.num_diffusion_timesteps,
            "steps": 2,
            "batch_size": self.eval_batch_sz,
        }

        assert isinstance(diffusion_iterations, int) and diffusion_iterations >= 1
        self.diffusion_iterations = diffusion_iterations

    def __init__(
        self,
        model_fn,
        model_loss,
        eval_mode="batch",
        verbose=0,
        **diffusion_args,
    ):
        super().__init__(
            model_fn,
            model_loss,
            eval_mode=eval_mode,
            verbose=verbose,
        )

        self.__initialize_diffusion_engine(**diffusion_args)

    def to(self, device):
        self.dm = self.dm.to(device)
        return super().to(device)

    def eval(self):
        self.dm = self.dm.eval()
        return super().eval()

    def get_loss_fn(self):
        return self.dm.model_loss

    def preprocess_forward(self, x, steps=None):
        x = self.dm.dm.upscaler(x)
        guide = x.detach().requires_grad_()
        guide.grad = torch.zeros_like(guide)
        for di in range(self.diffusion_iterations):
            x = (
                self.dm(
                    x,
                    guide=guide,
                    steps=steps,
                    disable=False,
                    diffusion_iteration=self.diffusion_iterations - 1 - di,
                    total_diffusion_iterations=self.diffusion_iterations,
                )
                + 1
            ) / 2
        return self.dm.dm.downscaler(x).clamp(self.clip_min, self.clip_max)

    def preprocess_eval_iter(self, x, guide=None):
        self.dm.setup_batch(
            self.eval_batch_sz, self.dm_eval_batch_sz, x.shape[1:], x.device
        )
        x = DiffGrad.add_noise(
            2 * x - 1,
            self.dm.dm,
            t=self.diff_eval_args["t_start"],
            noise=self.dm.noise,
        )
        return (
            self.dm.dm(x, bm=self.dm.bm, guide=guide, **self.diff_eval_args) + 1
        ) / 2

    def preprocess_eval(self, x):
        x = self.dm.dm.upscaler(x)
        guide = x.detach().clone()
        for _ in range(self.diffusion_iterations):
            x = self.preprocess_eval_iter(x, guide=guide)
        return self.dm.dm.downscaler(x).detach()

    def eval_sample(self, x, y_init):
        ##change batch_size
        if self.eval_mode == "single":
            orig_batch_size = self.diff_eval_args["batch_size"]
            orig_dm_eval_batch_sz = self.dm_eval_batch_sz
            self.diff_eval_args["batch_size"] = 1
            self.eval_batch_sz = 1

        init_success_rate, init_score = super().eval_sample(x, y_init)

        ##restore batch_size
        if self.eval_mode == "single":
            self.diff_eval_args["batch_size"] = orig_batch_size
            self.eval_batch_sz = orig_batch_size
            self.dm_eval_batch_sz = orig_dm_eval_batch_sz

        return init_success_rate, init_score
