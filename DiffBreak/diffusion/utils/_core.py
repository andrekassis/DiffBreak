import torch
from torchsde._brownian import brownian_base, brownian_interval


class RevVPSDE(torch.nn.Module):
    def __init__(
        self,
        model,
        scaler,
        image_size,
        betas,
        precision,
        timestep_map,
        preprocess_diffusion_t,
        cond_args={},
    ):
        super().__init__()
        beta_min = float(betas[0].detach().item()) * len(betas)
        beta_max = float(betas[-1].detach().item()) * len(betas)
        (
            self.model,
            self.scaler,
            self.betas,
            self.timestep_map,
            self.beta_0,
            self.beta_1,
            self.img_shape,
            self.preprocess_diffusion_t,
            self.precision,
            self.cond_args,
        ) = (
            model,
            scaler,
            betas,
            timestep_map,
            beta_min,
            beta_max,
            (3, image_size, image_size),
            preprocess_diffusion_t,
            precision,
            cond_args,
        )

        alphas_cumprod_cont = lambda t: torch.exp(
            -0.5 * (beta_max - beta_min) * t**2 - beta_min * t
        )
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1.0 / torch.sqrt(
            1.0 - alphas_cumprod_cont(t)
        )

        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def diffusion_fn(self, t, x):
        return self.vpsde_fn(t, x)[1]

    def drift_fn(self, t, x, ode=False, no_score=False):
        """Create the drift and diffusion functions for the reverse SDE"""
        drift, diffusion = self.vpsde_fn(t, x)
        score = (
            self.model(x.view(-1, *self.img_shape), self.scaler(t))[0]
            if not self.no_score
            else torch.zeros_like(x)
        )

        std_inv_neg = self.sqrt_1m_alphas_cumprod_neg_recip_cont(t)[:, None, None, None]
        score = (score * std_inv_neg).view(x.shape[0], -1)

        self.saved_t = t
        self.saved_diff = diffusion

        return (
            (drift - diffusion[:, None] ** 2 * score)
            if not ode
            else (drift - 0.5 * diffusion[:, None] ** 2 * score)
        )

    def f(self, t, x):
        assert not torch.isnan(x).any()
        return -self.drift_fn(
            1 - torch.tensor(t, device=x.device, dtype=x.dtype).expand(x.shape[0]), x
        )

    def g(self, t, x):
        assert not torch.isnan(x).any()
        return self.diffusion_fn(
            1 - torch.tensor(t, device=x.device, dtype=x.dtype).expand(x.shape[0]), x
        )[:, None].expand(x.shape)

    def forward(self, t, states):
        return (self.drift_fn(t.expand(states[0].shape[0]), states[0], ode=True),)

    def to(self, device):
        self.model.to(device)
        return super().to(device)

    def eval(self):
        self.model.eval()
        return super().eval()


class BrownianPath(brownian_base.BaseBrownian):
    def __init__(self, t0, w0, dt, window_size=8):
        t1 = t0 + 1
        self._w0 = w0
        self._interval = brownian_interval.BrownianInterval(
            t0=t0,
            t1=t1,
            size=w0.shape,
            dtype=w0.dtype,
            device=w0.device,
            cache_size=None,
            dt=dt,
        )
        super(BrownianPath, self).__init__()

    def __call__(self, t, tb=None, return_U=False, return_A=False):
        out = self._interval(t, tb, return_U=return_U, return_A=return_A)
        if tb is None and not return_U and not return_A:
            out = out + self._w0
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(interval={self._interval})"

    @property
    def dtype(self):
        return self._interval.dtype

    @property
    def device(self):
        return self._interval.device

    @property
    def shape(self):
        return self._interval.shape

    @property
    def levy_area_approximation(self):
        return self._interval.levy_area_approximation


class NoiseGenerator:
    def __init__(self, t0, w0, length):
        self.device = w0.device
        self.seeds_0 = torch.randint(
            low=0, high=100000000, size=(length,), device=self.device
        )
        self.seeds_1 = torch.randint(
            low=0, high=100000000, size=(length,), device=self.device
        )
        self.generator0 = torch.Generator(device=self.device)
        self.generator1 = torch.Generator(device=self.device)
        self._w0 = w0

    def __call__(self, seed, image_size):
        shape = (self._w0.shape[0], 3, image_size, image_size)
        seed0 = int(self.seeds_0[seed].detach().item())
        seed1 = int(self.seeds_1[seed].detach().item())
        noise = torch.randn(
            shape, generator=self.generator0.manual_seed(seed0), device=self.device
        )
        cond_noise = torch.randn(
            (1, *shape[1:]),
            generator=self.generator1.manual_seed(seed1),
            device=self.device,
        )
        return noise, cond_noise
