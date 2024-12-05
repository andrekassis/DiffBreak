import torch
import numpy as np


class CE(torch.nn.Module):
    def __init__(self, targeted=True, negate=False):
        super().__init__()
        self.targeted = targeted
        self.negate = 1 if not negate else -1
        self.fn = torch.nn.CrossEntropyLoss(reduction="none")

    def is_targeted(self, targeted):
        self.targeted = targeted

    def is_negated(self, negate):
        self.negate = 1 if not negate else -1

    def forward(self, logits, labels, *args, **kwargs):
        factor = 1 if self.targeted else -1
        return self.fn(logits, labels.repeat(logits.shape[0])) * factor * self.negate


class DLR(torch.nn.Module):
    def __init__(self, targeted=True, negate=False):
        super().__init__()
        self.targeted = targeted
        self.negate = 1 if not negate else -1

    def is_targeted(self, targeted):
        self.targeted = targeted

    def is_negated(self, negate):
        self.negate = 1 if not negate else -1

    def forward(self, logits, labels, *args, **kwargs):
        labels = labels.repeat(logits.shape[0] // labels.shape[0])

        x_sorted, ind_sorted = logits.sort(dim=1)
        u = torch.arange(logits.shape[0])

        if not self.targeted:
            ind = (ind_sorted[:, -1] == labels).float()
            return (
                -(
                    logits[u, labels]
                    - x_sorted[:, -2] * ind
                    - x_sorted[:, -1] * (1.0 - ind)
                )
                / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
                * self.negate
            )

        y_orig = self.y_orig.repeat(logits.shape[0] // self.y_orig.shape[0])
        return (
            -(logits[u, y_orig] - logits[u, labels])
            / (x_sorted[:, -1] - 0.5 * (x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)
            * self.negate
        )


class MarginLoss(torch.nn.Module):
    def __init__(self, kappa="inf", negate=False, targeted=True):
        super().__init__()
        self.kappa = float(kappa)
        self.targeted = targeted
        self.negate = 1 if not negate else -1

    def is_targeted(self, targeted):
        self.targeted = targeted

    def is_negated(self, negate):
        self.negate = 1 if not negate else -1

    def forward(self, logits, labels, *args, **kwargs):
        factor = 1 if self.targeted else -1
        labels = labels.repeat(logits.shape[0] // labels.shape[0])
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1))
        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max
        return ((correct_logits - max_incorrect_logits) * factor).clamp(
            max=self.kappa
        ) * self.negate


class DiffAttack(torch.nn.Module):
    def __init__(self, out_loss, t_interval=1, negate=False, targeted=True, **kwargs):
        super().__init__()
        self.t_interval = t_interval
        self.mse = torch.nn.MSELoss(reduction="none")
        assert (
            isinstance(out_loss, CE)
            or isinstance(out_loss, MarginLoss)
            or isinstance(out_loss, DLR)
        )
        self.loss = out_loss
        self.negate = 1 if not negate else -1
        self.dm = None

        indicator = isinstance(out_loss, MarginLoss) or isinstance(out_loss, DLR)
        self.indicator = 1 if indicator else -1

    def is_targeted(self, targeted):
        self.loss.is_targeted(targeted)

    def is_negated(self, negate):
        self.negate = 1 if not negate else -1
        self.loss.is_negated(negate)

    def add_noise(self, x, t, noise):
        t = self.dm.preprocess_diffusion_t(t)
        return x * self.dm.a[t - 1].sqrt() + noise * (1 - self.dm.a[t - 1]).sqrt()

    def forward(self, logits, labels, t=0, x=None, x_t=None, noise=None):
        if t is not None:
            t = int(t * self.dm.num_diffusion_timesteps)
        out_loss = self.loss(logits, labels).mean() if t == 0 else 0.0
        if x is None or t % self.t_interval != 0.0:
            return out_loss
        x_t_o = self.add_noise(2 * x.detach() - 1, t, noise)
        mse = self.mse(x_t, x_t_o) * self.negate * self.indicator
        return out_loss + mse.view(mse.shape[0], -1).mean()
