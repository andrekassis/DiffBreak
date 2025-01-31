import torch

from .loss_provider import LpipsVGG


class LossAndModelWrapper(torch.nn.Module):
    def __init__(self, classifier, loss_fn):
        super().__init__()
        self.classifier = classifier
        self.loss_fn = loss_fn

    def forward(self, examples, labels, *args, **kwargs):
        return self.loss_fn(self.classifier.forward(examples), labels, *args, **kwargs)
