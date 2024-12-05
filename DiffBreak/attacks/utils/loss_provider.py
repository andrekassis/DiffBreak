import torch
import lpips
from torchvision import transforms


class ColourWrapper(torch.nn.Module):
    def __init__(self, grayscale, lossclass, args, kwargs):
        super().__init__()
        self.grayscale = grayscale
        self.to_greyscale = (
            (lambda x: transforms.Grayscale()(x).repeat(1, 3, 1, 1))
            if grayscale
            else (lambda x: x)
        )
        self.add_module("loss", lossclass(*args, **kwargs))

    def forward(self, input, target):
        return self.loss.forward(self.to_greyscale(input), self.to_greyscale(target))


class NormalizedLoss(torch.nn.Module):
    def __init__(self, loss_fn, **kwargs):
        super().__init__()
        self.loss_fn = loss_fn

    def eval(self):
        self.loss_fn = self.loss_fn.eval()
        return super().eval()

    def to(self, device):
        self.loss_fn = self.loss_fn.to(device)
        return super().to(device)

    def forward(self, x, y):
        return self.loss_fn(x, y, normalize=True)


class Loss(torch.nn.Module):
    def __init__(
        self,
        loss_class,
        colorspace="RGB",
        **loss_kwargs,
    ):
        assert "reduction" in list(loss_kwargs.keys())
        assert loss_kwargs["reduction"] in ["sum", "mean", "none"]
        assert colorspace in ["RGB", "LA"]

        super().__init__()
        self.loss = ColourWrapper(
            colorspace == "LA",
            loss_class,
            (),
            loss_kwargs,
        )

    def eval(self):
        for param in self.loss.parameters():
            param.requires_grad = False
        self.loss = self.loss.eval()
        return super().eval()

    def to(self, device):
        self.loss = self.loss.to(device)
        return super().to(device)

    def forward(self, x, y):
        return self.loss(x, y)


class LpipsVGG(Loss):
    def __init__(
        self,
        colorspace="RGB",
        reduction="none",
    ):
        lps_loss = lpips.LPIPS(
            net="vgg",
        ).eval()
        super().__init__(
            NormalizedLoss,
            loss_fn=lps_loss,
            colorspace=colorspace,
            reduction=reduction,
        )
