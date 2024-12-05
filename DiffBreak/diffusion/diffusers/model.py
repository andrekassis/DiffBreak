import torch


def default_scaler(t):
    return (t.float() * 1000).long()


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.requires_grad_(False)

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

    def eval(self):
        self.model = self.model.eval()
        return super().eval()

    def forward(self, x, t):
        return self.model(x, t), None
