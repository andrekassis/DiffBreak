from argparse import Namespace
from math import exp
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchsde._core import base_sde, methods
from torchsde.settings import LEVY_AREA_APPROXIMATIONS
from torchsde import sdeint_adjoint, BrownianInterval
from torchdiffeq import odeint_adjoint
from torchdiffeq._impl.fixed_grid import Euler
from torchdiffeq._impl.misc import (
    _flat_to_shape,
    _PerturbFunc,
    _ReverseFunc,
    _TupleFunc,
)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, reduction="sum"):
    size_average = reduction != "none"
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class RoundFloat(float):
    def __new__(self, value, precision):
        return float.__new__(self, round(float(value), precision))

    def __init__(self, value, precision):
        float.__init__(round(value, precision))
        self.precision = precision
        self.is_leaf = True

    def __add__(self, other):
        assert not isinstance(other, torch.Tensor)
        precision = self.precision if self.precision != 5 else 3
        return RoundFloat(super().__add__(other), precision)

    def __radd__(self, other):
        assert not isinstance(other, torch.Tensor)
        precision = self.precision if self.precision != 5 else 3
        return RoundFloat(super().__add__(other), precision)

    def __min__(self, other):
        if self <= other:
            return self
        return other

    def __neg__(self):
        return RoundFloat(super().__neg__(), self.precision)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return RoundFloatTensor(
                torch.stack([torch.tensor(a) for a in args[0]]),
                [a.precision for a in args[0]],
            )
        return torch.Tensor.__torch_function__(func, types, args, kwargs)


class RoundFloatTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, precisions, *args, **kwargs):
        x = torch.tensor(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x)
        precisions = (
            precisions
            if len(x.shape) > 0
            else (precisions[0] if isinstance(precisions, list) else precisions)
        )
        x = (
            [torch.round(xi, decimals=precisions[i]) for i, xi in enumerate(x)]
            if len(x.shape) > 0
            else torch.round(x, decimals=precisions)
        )
        obj = super().__new__(cls, x, *args, **kwargs)
        obj.__init__(x, precisions)
        return obj

    def __init__(self, x, precisions):
        self.precisions = precisions if isinstance(precisions, list) else [precisions]

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            return (
                RoundFloatTensor(
                    self.data.detach().cpu().numpy().tolist()[idx], self.precisions[idx]
                )
                .to(self.device)
                .to(self.dtype)
            )
        return RoundFloat(
            super().__getitem__(idx).detach().item(), self.precisions[idx]
        )

    def __iter__(self):
        for ip, p in enumerate(super().__iter__()):
            yield RoundFloat(p.detach().item(), self.precisions[ip])

    def to(self, *args, **kwargs):
        new_obj = RoundFloatTensor(
            torch.ones(
                (len(self.precisions) if isinstance(self.precisions, list) else 1)
            ),
            self.precisions,
        )
        tempTensor = super().to(*args, **kwargs)
        new_obj.data = tempTensor.data
        new_obj.requires_grad = tempTensor.requires_grad
        return new_obj


class DDPMSolver(torch.nn.Module):
    def __init__(
        self,
        sde,
        dt,
    ):
        super().__init__()

        self.model, self.preprocess_diffusion_t, config = (
            sde.model,
            sde.preprocess_diffusion_t,
            sde.cond_args,
        )
        self.guide_mode, self.ptb, self.guide_scale = (
            config.get("guide_mode"),
            config.get("ptb", 4.0),
            config.get("guide_scale", 1000.0),
        )
        if self.guide_mode is not None:
            assert isinstance(self.ptb, float)
            assert isinstance(self.guide_scale, float)
            assert self.guide_mode in ["CONSTANT", "MSE", "SSIM"]
            self.criterion = (
                ssim
                if self.guide_mode in ["CONSTANT", "SSIM"]
                else (lambda x, y: -1 * torch.nn.functional.mse_loss(x, y))
            )
            self.diffuse_cond_reference = self.guide_mode == "MSE"

        self.betas = sde.betas
        self.num_diffusion_timesteps = len(self.betas)
        self.timestep_map = sde.timestep_map
        alphas = 1.0 - self.betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            (
                torch.tensor(1.0).to(self.alphas_cumprod.dtype).view(-1),
                self.alphas_cumprod[:-1],
            ),
            dim=0,
        )
        self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_cumprod).sqrt()
        self.sqrt_recipm1_alphas_cumprod = (1.0 / self.alphas_cumprod - 1).sqrt()

        posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.logvar = torch.log(
            torch.cat((posterior_variance[1].view(-1), self.betas[1:]), dim=0)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat(
                (posterior_variance[1].view(-1), posterior_variance[1:]),
                dim=0,
            )
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def process_x_cond(self, x, t, noise):
        x = 2 * x - 1
        if self.diffuse_cond_reference:
            t = self.preprocess_diffusion_t(int(t.float().mean().detach().item()))
            return (
                x * self.alphas_cumprod[t - 1].sqrt()
                + noise * (1 - self.alphas_cumprod[t - 1]).sqrt()
            )
        return x

    def get_output_and_var(self, x, t, ode=False, no_score=False):
        target_shape = (-1, *([1] * (len(x.shape) - 1)))
        model_output, model_var_values = self.model(x, self.timestep_map[t])
        model_output = model_output if not no_score else torch.zeros_like(model_output)
        if model_var_values is not None:
            min_log = self.posterior_log_variance_clipped[t].float().view(*target_shape)
            max_log = torch.log(self.betas)[t].float().view(*target_shape)
            log_var = (model_var_values + 1) / 2 * max_log + (
                1 - (model_var_values + 1) / 2
            ) * min_log
        else:
            log_var = self.logvar[t].float().view(*target_shape)
        if ode:
            model_output = model_output * 0.5
        return model_output, log_var

    def to(self, device):
        self.model = self.model.to(device)
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        self.logvar = self.logvar.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(
            device
        )
        self.timestep_map = self.timestep_map.to(device)
        return super().to(device)

    def eval(self):
        self.model = self.model.eval()
        return super().eval()

    def cond_fn(self, x, t, cond_noise, guide=None):
        if self.guide_mode is None:
            return 0.0
        assert guide is not None
        retain_graph = x.requires_grad
        with torch.enable_grad():
            if not retain_graph:
                x = x.detach().requires_grad_()
            guide_t = self.process_x_cond(guide, t, cond_noise)
            selected = self.criterion(x, guide_t.repeat(x.shape[0], 1, 1, 1))
            return (
                torch.autograd.grad(
                    selected.sum(),
                    x,
                    create_graph=retain_graph,
                    retain_graph=retain_graph,
                )[0]
                * self.compute_scale(x, t)
            ).float()

    def compute_scale(self, x, t):
        if self.guide_scale == "CONSTANT":
            return self.guide_scale
        m = self.ptb * 2 / 255.0 / 3.0 / self.guide_scale
        alpha_bar = self.alphas_cumprod[t].float().view(-1, *([1] * (len(x.shape) - 1)))
        return torch.sqrt(1 - alpha_bar) / (m * torch.sqrt(alpha_bar))

    def step(self, x, t, noise, cond_noise, guide=None, ode=False, no_score=False):
        target_shape = (-1, *([1] * (len(x.shape) - 1)))
        model_output, log_var = self.get_output_and_var(
            x, t, ode=ode, no_score=no_score
        )
        pred_xstart = (
            self.sqrt_recip_alphas_cumprod[t].float().view(*target_shape) * x
            - self.sqrt_recipm1_alphas_cumprod[t].float().view(*target_shape)
            * model_output
        ).clamp(-1, 1)
        model_mean = (
            self.posterior_mean_coef1[t].float().view(*target_shape) * pred_xstart
            + self.posterior_mean_coef2[t].float().view(*target_shape) * x
        )
        model_mean = model_mean.float() + torch.exp(log_var) * self.cond_fn(
            x,
            t,
            cond_noise,
            guide=guide,
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        if ode:
            noise = torch.zeros_like(noise)
        return model_mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    def forward(
        self,
        img,
        t_start,
        t_end,
        steps,
        guide=None,
        bm=None,
        ode=False,
        no_score=False,
    ):
        t_start = int(t_start * self.num_diffusion_timesteps)
        t_end = int(t_end * self.num_diffusion_timesteps)
        outputs = []
        for i in list(range(t_end, t_start))[::-1]:
            t = torch.tensor([i] * img.shape[0], device=img.device)
            if bm is None:
                noise, cond_noise = torch.randn_like(img), torch.randn(
                    (1, *img.shape[1:]), device=img.device
                ).to(img.dtype)
            else:
                noise, cond_noise = bm(i, img.shape[-1])
            img = self.step(
                img,
                t,
                noise=noise,
                cond_noise=cond_noise,
                guide=guide,
                ode=ode,
                no_score=no_score,
            )
            outputs.append(img)
        if steps == 2:
            outputs = [img]
        return torch.cat(outputs).view(-1, *img.shape[1:])


class SDE(torch.nn.Module):
    def __init__(self, sde, dt):
        super().__init__()
        self.dt = dt
        self.sde = sde

    def get_ts(self, t_start, t_end, steps, dtype, device):
        last_precision = 5 if 0.99999 < 1 - t_end else self.sde.precision
        t_end = np.min([0.99999, 1 - t_end])
        ts = (
            ([1 - t_start + t * self.dt for t in range(steps - 1)] + [t_end])
            if steps > 2
            else [1 - t_start, t_end]
        )

        return (
            RoundFloatTensor(ts, [self.sde.precision] * (steps - 1) + [last_precision])
            .to(device)
            .to(dtype)
        )

    def forward(
        self, x, t_start, t_end, steps, guide=None, bm=None, no_score=False, **kwargs
    ):
        pass

    def to(self, device):
        self.sde = self.sde.to(device)
        return super().to(device)

    def eval(self):
        self.sde = self.sde.eval()
        return super().eval()


class SDEINT(SDE):
    def __init__(self, sde, dt):
        super().__init__(sde, dt)
        sde = base_sde.ForwardSDE(self.sde)
        self.solver = methods.select(method="euler", sde_type=sde.sde_type)(
            sde=sde,
            bm=Namespace(levy_area_approximation=LEVY_AREA_APPROXIMATIONS.none),
            dt=dt,
            adaptive=False,
            rtol=1e-5,
            atol=1e-4,
            dt_min=1e-5,
            options={},
        )

    def forward(
        self, x, t_start, t_end, steps, guide=None, bm=None, no_score=False, **kwargs
    ):
        ts = self.get_ts(t_start, t_end, steps, x.dtype, x.device)
        if bm is None:
            bm = BrownianInterval(
                t0=ts[0],
                t1=ts[-1],
                size=(x.shape[0], np.prod(x.shape[1:])),
                dtype=x.dtype,
                device=x.device,
                dt=self.dt,
                levy_area_approximation=LEVY_AREA_APPROXIMATIONS.none,
            )
        self.solver.bm = bm
        return self.solver.integrate(x.view(x.shape[0], -1), ts, ())[0][1:].view(
            -1, *x.shape[1:]
        )


class SDEINT_ADJOINT(SDE):
    def __init__(self, sde, dt):
        super().__init__(sde, dt)

    def forward(
        self, x, t_start, t_end, steps, guide=None, bm=None, no_score=False, **kwargs
    ):
        ts = self.get_ts(t_start, t_end, steps, x.dtype, x.device)
        return sdeint_adjoint(
            self.sde, x.view(x.shape[0], -1), ts, method="euler", bm=bm, dt=self.dt
        )[1:].view(-1, *x.shape[1:])


class ODE(torch.nn.Module):
    def __init__(self, sde, dt):
        super().__init__()
        self.sde = sde
        self.dt = dt
        self.grid_constructor = self._grid_constructor_from_step_size(dt)

    def _grid_constructor_from_step_size(self, step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()

            t_infer = torch.tensor(
                [
                    round(
                        ti + start_time.detach().item(),
                        self.sde.precision,
                    )
                    for ti in (
                        torch.arange(0, niters, dtype=t.dtype, device=t.device)
                        * step_size
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                ],
                device=t.device,
                dtype=t.dtype,
            )
            t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    def get_ts(self, t_start, t_end, steps, dtype, device):
        t_end = np.max([1e-5, t_end])
        ts = (
            ([t_start - t * self.dt for t in range(steps - 1)] + [t_end])
            if steps > 2
            else [t_start, t_end]
        )
        return torch.tensor(ts, dtype=dtype, device=device)

    def forward(
        self, x, t_start, t_end, steps, guide=None, bm=None, no_score=False, **kwargs
    ):
        pass

    def to(self, device):
        self.sde = self.sde.to(device)
        return super().to(device)

    def eval(self):
        self.sde = self.sde.eval()
        return super().eval()


class ODEINT(ODE):
    def __init__(self, sde, dt):
        super().__init__(sde, dt)
        self.solver = Euler(
            func=None,
            y0=torch.tensor(1.0).float(),
            atol=1e-3,
            grid_constructor=self.grid_constructor,
        )

    def forward(
        self, x, t_start, t_end, steps, guide=None, bm=None, no_score=False, **kwargs
    ):
        shapes = [x.view(x.shape[0], -1).shape]
        self.solver.y0 = x.view(-1)
        self.solver.func = _PerturbFunc(
            _ReverseFunc(_TupleFunc(self.sde, shapes), mul=-1.0)
        )
        ts = self.get_ts(t_start, t_end, steps, x.dtype, x.device)
        return _flat_to_shape(self.solver.integrate(-ts), (len(ts),), shapes)[0][
            1:
        ].view(-1, *x.shape[1:])

    def to(self, device):
        self.solver.device = device
        return super().to(device)


class ODEINT_ADJOINT(ODE):
    def __init__(self, sde, dt):
        super().__init__(sde, dt)
        self.args = {
            "atol": 1e-3,
            "rtol": 1e-3,
            "options": dict(grid_constructor=self.grid_constructor),
        }

    def _grid_constructor_from_step_size(self, step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[-1]
            end_time = t[0]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()

            t_infer = torch.tensor(
                [
                    round(
                        ti + start_time.detach().item(),
                        self.sde.precision,
                    )
                    for ti in (
                        torch.arange(0, niters, dtype=t.dtype, device=t.device)
                        * step_size
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                ],
                device=t.device,
                dtype=t.dtype,
            )

            t_infer = torch.flip(t_infer, [-1])
            if start_time > 0:
                t_infer[-1] = t[-1]
            else:
                t_infer[0] = end_time

            return t_infer

        return _grid_constructor

    def forward(
        self, x, t_start, t_end, steps, guide=None, bm=None, no_score=False, **kwargs
    ):
        ts = self.get_ts(t_start, t_end, steps, x.dtype, x.device)
        return odeint_adjoint(
            self.sde, (x.view(x.shape[0], -1),), ts, method="euler", **self.args
        )[0][1:].view(-1, *x.shape[1:])
