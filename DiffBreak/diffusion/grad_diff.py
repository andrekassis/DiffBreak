import gc
from tqdm import trange
import numpy as np
import torch

from .utils import NoiseGenerator, BrownianPath


class DiffGrad(torch.autograd.Function):
    @staticmethod
    def add_noise(x, dm, t, noise):
        t = dm.preprocess_diffusion_t(int(t * dm.num_diffusion_timesteps))
        return x * dm.a[t - 1].sqrt() + noise * (1 - dm.a[t - 1]).sqrt()

    @staticmethod
    def get_layer_grad(
        x,
        guide,
        x_o,
        noise,
        dm,
        diffusion_type,
        bm,
        repeats,
        grads,
        guide_grads,
        grad_fn,
        step,
        dt,
        ode,
        with_intermediate=False,
        diffusion_iteration=0,
        total_diffusion_iterations=1,
        power=0.5,
        power_2=0.25,
        factor=1,
        pbar=None,
    ):
        multiplier = (1 / (diffusion_iteration + 1) ** power_2) / factor
        not_first_iter = step > 0 or diffusion_iteration > 0
        loss_factor = (
            ((1 / (step + 1) ** power) * multiplier * float(not_first_iter))
            if with_intermediate
            else 0.0
        )
        guide = guide.detach().requires_grad_() if guide is not None else guide
        multiplier = 1.0 if not with_intermediate else multiplier

        x_next = dm(
            x,
            diffusion_type=diffusion_type,
            t_start=round(dt * (step + 1), dm.precision),
            t_end=round(dt * step, dm.precision),
            steps=2,
            with_noise=False,
            adjoint=False,
            guide=guide,
            bm=bm,
            ode=ode,
        )
        g_loss, lab_loss = (
            grad_fn(x_next, x_o=x_o, t=round(dt * step, dm.precision), noise=noise)
            if with_intermediate
            else (torch.zeros_like(x_next), 0)
        )
        grads = (
            grads if not_first_iter else (grads * multiplier)
        ) + g_loss * loss_factor
        guide = guide if guide is not None else torch.zeros_like(x_next)

        if pbar is not None and with_intermediate:
            pbar.set_description(f"Backprop - step: {step+1}, loss: {lab_loss}")

        grads, guide_grads_add = torch.autograd.grad(
            x_next,
            [x, guide],
            grad_outputs=[grads, grads],
            materialize_grads=True,
        )
        return grads, guide_grads + guide_grads_add

    @staticmethod
    def get_factor(
        steps, total_diffusion_iterations, power, power_2, with_intermediate
    ):
        factor_1 = float(
            1
            if not with_intermediate
            else np.array([1 / (step + 1) ** power for step in range(steps)]).sum()
        )
        factor_2 = float(
            np.array(
                [1 / (d + 1) ** power_2 for d in range(total_diffusion_iterations)]
            ).sum()
        )
        return factor_1 * factor_2

    @staticmethod
    def diff_grad(
        x_o,
        noise,
        xd,
        bm,
        repeats,
        dm,
        diffusion_type,
        grad_fn,
        ode=False,
        t_start=1,
        grads=0,
        guide=None,
        with_intermediate=False,
        diffusion_iteration=0,
        total_diffusion_iterations=1,
        power=0.5,
        power_2=0.25,
        pbar=None,
    ):
        torch.cuda.empty_cache()
        gc.collect()

        steps, guide_grads = xd.shape[0] // repeats, 0.0
        factor = DiffGrad.get_factor(
            steps, total_diffusion_iterations, power, power_2, with_intermediate
        )
        for step in range(steps):
            with torch.enable_grad():
                x = (
                    xd[(steps - step - 1) * repeats : (steps - step) * repeats]
                    .detach()
                    .to(bm[0]._w0.device)
                    .requires_grad_()
                )
                grads, guide_grads = DiffGrad.get_layer_grad(
                    x,
                    guide,
                    x_o,
                    noise,
                    dm,
                    diffusion_type,
                    bm,
                    repeats,
                    grads,
                    guide_grads,
                    grad_fn,
                    t_start - steps + step,
                    round(
                        1 / dm.num_diffusion_timesteps,
                        dm.precision,
                    ),
                    ode,
                    with_intermediate=with_intermediate,
                    diffusion_iteration=diffusion_iteration,
                    total_diffusion_iterations=total_diffusion_iterations,
                    power=power,
                    power_2=power_2,
                    factor=factor,
                    pbar=pbar,
                )

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.reset(total=steps)
            pbar.set_description("Backprop")
        torch.cuda.empty_cache()
        gc.collect()
        return grads, guide_grads

    @staticmethod
    def forward(
        ctx,
        x,
        guide,
        dm,
        diffusion_type,
        t_start,
        bm,
        noise,
        ode,
        repeats,
        grad_fn,
        steps,
        with_intermediate,
        diffusion_iteration,
        total_diffusion_iterations,
        power,
        power_2,
        bpda,
        pbar,
    ):
        x_batch = DiffGrad.add_noise(
            2 * x - 1, dm, t=t_start / dm.num_diffusion_timesteps, noise=noise
        )

        with torch.no_grad():
            xd = dm(
                x_batch,
                diffusion_type=diffusion_type,
                with_noise=False,
                adjoint=False,
                guide=guide.detach(),
                bm=bm,
                ode=ode,
                t_start=round(
                    t_start / dm.num_diffusion_timesteps,
                    dm.precision,
                ),
                t_end=round(
                    (t_start - steps) / dm.num_diffusion_timesteps,
                    dm.precision,
                ),
                steps=(steps + 1) if not bpda else 2,
            )

        (
            ctx.shape,
            ctx.diffusion_type,
            ctx.diffusion_iteration,
            ctx.total_diffusion_iterations,
            ctx.noise,
            ctx.bm,
            ctx.dm,
            ctx.ode,
            ctx.t_start,
            ctx.grad_fn,
            ctx.repeats,
            ctx.with_intermediate,
            ctx.power,
            ctx.power_2,
            ctx.bpda,
            ctx.pbar,
        ) = (
            x.shape,
            diffusion_type,
            diffusion_iteration,
            total_diffusion_iterations,
            noise,
            bm,
            dm,
            ode,
            t_start,
            grad_fn,
            repeats,
            with_intermediate,
            power,
            power_2,
            bpda,
            pbar,
        )
        if not bpda:
            ctx.saved_inputs = (
                x.detach(),
                torch.concat((x_batch, xd), dim=0)[: -x_batch.shape[0]].detach().cpu(),
                guide,
            )
        torch.cuda.empty_cache()
        gc.collect()
        return xd[-x_batch.shape[0] :].detach().requires_grad_()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.bpda:
            g = grad_output.view(ctx.shape[0], -1, *ctx.shape[1:]).mean(1)
            torch.cuda.empty_cache()
            gc.collect()
            return g, *([None] * 17)

        x, xd, guide = ctx.saved_inputs
        g, gg = DiffGrad.diff_grad(
            x,
            ctx.noise,
            xd,
            ctx.bm,
            ctx.noise.shape[0],
            ctx.dm,
            ctx.diffusion_type,
            ctx.grad_fn,
            ode=ctx.ode,
            t_start=ctx.t_start,
            grads=grad_output,
            guide=guide,
            with_intermediate=ctx.with_intermediate,
            diffusion_iteration=ctx.diffusion_iteration,
            total_diffusion_iterations=ctx.total_diffusion_iterations,
            power=ctx.power,
            power_2=ctx.power_2,
            pbar=ctx.pbar,
        )
        with torch.enable_grad():
            x = x.requires_grad_()
            x_batch = DiffGrad.add_noise(
                2 * x - 1,
                ctx.dm,
                t=ctx.t_start / ctx.dm.num_diffusion_timesteps,
                noise=ctx.noise,
            )
            g = torch.autograd.grad(x_batch, [x], grad_outputs=g)[0]
        guide.grad += gg
        g = (
            (g + guide.grad)
            if ctx.diffusion_iteration == ctx.total_diffusion_iterations - 1
            else g
        )
        if ctx.diffusion_iteration == ctx.total_diffusion_iterations - 1:
            guide.grad = None
        del xd
        del x_batch
        del ctx.saved_inputs
        torch.cuda.empty_cache()
        gc.collect()
        return g, *([None] * 17)


class DiffGradEngine(torch.nn.Module):
    def __init__(
        self,
        dm,
        model_fn,
        model_loss,
        diffusion_type,
        diffusion_steps,
        ode,
        repeats=32,
        deterministic=False,
        basic_adjoint_method=False,
        with_intermediate=False,
        power=0.5,
        power_2=0.25,
        bpda=False,
        blind=False,
        forward_diff_only=False,
        verbose=0,
    ):
        super().__init__()
        assert diffusion_type in ["ddpm", "vpsde"]
        self.diffusion_type = diffusion_type
        self.dm = dm
        self.model_fn = model_fn
        self.model_loss = model_loss
        self.t_start = diffusion_steps
        self.ode = ode
        self.repeats = self.dm.batch_size if repeats is None else repeats
        self.diff_gradient = DiffGrad.apply
        self.deterministic = deterministic
        self.bm, self.noise = None, None
        self.repeats = 1 if self.deterministic else self.repeats
        self.dm.batch_size = 1 if self.deterministic else self.dm.batch_size
        self.with_intermediate = with_intermediate and (model_loss is not None)

        assert (
            float(bpda)
            + float(blind)
            + float(forward_diff_only)
            + float(basic_adjoint_method)
            + float(with_intermediate)
            <= 1
        )
        if self.diffusion_type == "ddpm":
            assert not basic_adjoint_method

        self.basic_adjoint_method = basic_adjoint_method
        self.power = power if with_intermediate else 0
        self.power_2 = power_2
        self.bpda = bpda
        self.blind = blind
        self.forward_diff_only = forward_diff_only

        self.pbar = (
            trange(self.t_start, disable=verbose == 0)
            if not self.basic_adjoint_method
            else None
        )
        if self.pbar is not None:
            self.pbar.set_description("Backprop")
        assert self.dm.batch_size <= self.repeats

    @staticmethod
    def _setup_bm(
        bm_class,
        state_size,
        dt,
        batch_size=32,
        single=False,
        device="cuda",
        length=None,
    ):
        kwargs = {"length": length} if bm_class == NoiseGenerator else {"dt": dt}
        return bm_class(
            t0=torch.tensor(0.0, device=device),
            w0=torch.randn(
                [
                    batch_size if not single else 1,
                    state_size,
                ],
                device=device,
            ).repeat(
                1 if not single else batch_size,
                1,
            ),
            **kwargs,
        )

    def _setup_bm_batch(self, state_size, batch_size, dm_batch_size, device="cuda"):
        bm_args = {
            "bm_class": (
                BrownianPath if self.diffusion_type != "ddpm" else NoiseGenerator
            ),
            "state_size": state_size,
            "device": device,
            "single": self.deterministic,
            "length": self.t_start,
            "dt": round(
                1 / self.dm.num_diffusion_timesteps,
                self.dm.precision,
            ),
        }
        bm = [
            self._setup_bm(batch_size=dm_batch_size, **bm_args)
            for bm_c in range(batch_size // dm_batch_size)
        ]
        return (
            (bm + [self._setup_bm(batch_size=batch_size % dm_batch_size, **bm_args)])
            if batch_size % dm_batch_size > 0
            else bm
        )

    def setup_batch(self, batch_size, dm_batch_size, target_shape, device):
        if not self.deterministic or (self.bm is None and self.noise is None):
            noise = torch.randn((batch_size, *target_shape)).to(device)
            bm = self._setup_bm_batch(
                int(np.prod(target_shape)), batch_size, dm_batch_size, device
            )
            self.bm, self.noise = bm, noise.detach()

    def forward(
        self,
        x,
        guide,
        steps=None,
        disable=False,
        diffusion_iteration=0,
        total_diffusion_iterations=1,
    ):
        self.setup_batch(self.repeats, self.dm.batch_size, x.shape[1:], x.device)
        steps = self.t_start if steps is None else steps

        if self.forward_diff_only or self.blind or self.basic_adjoint_method:
            torch.cuda.empty_cache()
            gc.collect()

        if self.forward_diff_only:
            return DiffGrad.add_noise(
                2 * x - 1,
                self.dm,
                t=self.t_start / self.dm.num_diffusion_timesteps,
                noise=self.noise,
            )

        if self.blind:
            return 2 * x - 1

        if self.basic_adjoint_method:
            out = self.dm(
                (2 * x - 1).repeat(self.repeats // x.shape[0], 1, 1, 1),
                guide=guide,
                with_noise=True,
                adjoint=True,
                bm=self.bm,
                ode=self.ode,
                t_start=round(
                    self.t_start / self.dm.num_diffusion_timesteps,
                    self.dm.precision,
                ),
                t_end=round(
                    self.t_start / self.dm.num_diffusion_timesteps,
                    self.dm.precision,
                )
                - round(
                    1 / self.dm.num_diffusion_timesteps,
                    self.dm.precision,
                )
                * steps,
                steps=2,
            )
            torch.cuda.empty_cache()
            gc.collect()
            return out

        return self.diff_gradient(
            x,
            guide,
            self.dm,
            self.diffusion_type,
            self.t_start,
            self.bm,
            self.noise,
            self.ode,
            self.repeats,
            self.grad_fn,
            steps,
            self.with_intermediate and not disable,
            diffusion_iteration,
            total_diffusion_iterations,
            self.power,
            self.power_2,
            self.bpda,
            self.pbar,
        )

    def grad_fn(self, y, x_o, t, noise):
        y = y.detach().requires_grad_()
        loss = self.model_loss(
            self.model_fn(self.dm.downscaler((y + 1) / 2)),
            x_t=y,
            t=t,
            x=x_o,
            noise=noise,
        ).mean()
        return (
            torch.autograd.grad(loss, [y])[0],
            loss.detach().item(),
        )

    def eval(self):
        self.dm.eval()
        self.model_fn.eval()
        return self

    def to(self, device):
        self.dm = self.dm.to(device)
        self.model_fn = self.model_fn.to(device)
        self.device = device
        return self
