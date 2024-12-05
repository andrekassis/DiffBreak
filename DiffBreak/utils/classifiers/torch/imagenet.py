import torch
import torchvision.models as models

from .classifier import Classifier


class _Wrapper_ResNet(Classifier):
    def __init__(self, model, softmaxed=False):
        super().__init__(model=model, softmaxed=softmaxed)
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1)
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1)

    def forward(self, x):
        return super().forward((x - self.mu) / self.sigma)

    def to(self, device):
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)
        return super().to(device)


##classifiers
class RESNET18(_Wrapper_ResNet):
    def __init__(self):
        super().__init__(
            model=models.resnet18(pretrained=True),
            softmaxed=False,
        )


class RESNET50(_Wrapper_ResNet):
    def __init__(self):
        super().__init__(
            model=models.resnet50(pretrained=True),
            softmaxed=False,
        )


class RESNET101(_Wrapper_ResNet):
    def __init__(self):
        super().__init__(
            model=models.resnet101(pretrained=True),
            softmaxed=False,
        )


class WIDERESNET_50_2(_Wrapper_ResNet):
    def __init__(self):
        super().__init__(
            model=models.wide_resnet50_2(pretrained=True),
            softmaxed=False,
        )


class DEIT_S(_Wrapper_ResNet):
    def __init__(self):
        super().__init__(
            model=torch.hub.load(
                "facebookresearch/deit:main", "deit_small_patch16_224", pretrained=True
            ),
            softmaxed=False,
        )
