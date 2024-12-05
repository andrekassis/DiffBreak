import torch


class Classifier(torch.nn.Module):
    def __init__(self, model, softmaxed=False):
        super().__init__()
        model.requires_grad_(False)
        self.model = (
            torch.nn.Sequential(*list(model.children())[:-1]) if softmaxed else model
        )

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

    def eval(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return super().eval()

    def forward(self, x):
        return self.model(x)
