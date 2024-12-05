import numpy as np
import torch
import torchvision
import torch.nn.functional as F


class CifarAlexNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 2 * 2, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        return self.classifier(x)


class AlexNetFeatureModel(torch.nn.Module):
    def __init__(self, alexnet_model):
        super().__init__()
        self.model = alexnet_model.eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        assert len(self.model.features) == 13
        self.layer1 = torch.nn.Sequential(self.model.features[:2])
        self.layer2 = torch.nn.Sequential(self.model.features[2:5])
        self.layer3 = torch.nn.Sequential(self.model.features[5:8])
        self.layer4 = torch.nn.Sequential(self.model.features[8:10])
        self.layer5 = torch.nn.Sequential(self.model.features[10:12])
        self.layer6 = self.model.features[12]

    def features(self, x):
        x = (x - self.mean[None, :, None, None].to(x.device)) / self.std[
            None, :, None, None
        ].to(x.device)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)
        x_layer5 = self.layer5(x_layer4)

        return (x_layer1, x_layer2, x_layer3, x_layer4, x_layer5)

    def classifier(self, last_layer):
        x = self.layer6(last_layer)
        if isinstance(self.model, CifarAlexNet):
            x = x.view(x.size(0), 256 * 2 * 2)
        else:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
        return self.model.classifier(x)

    def forward(self, x):
        return self.classifier(self.features(x)[-1])

    def features_logits(
        self,
        x,
    ):
        features = self.features(x)
        logits = self.classifier(features[-1])
        return features, logits


class LPIPSDistance(torch.nn.Module):
    """
    Calculates the square root of the Learned Perceptual Image Patch Similarity
    (LPIPS) between two images, using a given neural network.
    """

    def __init__(
        self,
        model=None,
        activation_distance="l2",
        include_image_as_activation=False,
    ):
        """
        Constructs an LPIPS distance metric. The given network should return a
        tuple of (activations, logits). If a network is not specified, AlexNet
        will be used. activation_distance can be 'l2' or 'cw_ssim'.
        """

        super().__init__()

        if model is None:
            alexnet_model = torchvision.models.alexnet(pretrained=True)
            self.model = AlexNetFeatureModel(alexnet_model)
        else:
            self.model = model

        self.activation_distance = activation_distance
        self.include_image_as_activation = include_image_as_activation

        self.eval()

    def features(self, image):
        features = self.model.features(image)
        if self.include_image_as_activation:
            features = (image,) + features
        return features

    def forward(self, image1, image2):
        features1 = self.features(image1)
        features2 = self.features(image2)

        if self.activation_distance == "l2":
            return (
                normalize_flatten_features(features1)
                - normalize_flatten_features(features2)
            ).norm(dim=1)
        else:
            raise ValueError(
                f'Invalid activation_distance "{self.activation_distance}"'
            )


def get_lpips_model(
    lpips_model_spec,
    model=None,
):
    if lpips_model_spec == "alexnet":
        lpips_model = AlexNetFeatureModel(torchvision.models.alexnet(pretrained=True))
    elif lpips_model_spec == "alexnet_cifar":
        lpips_model = AlexNetFeatureModel(CifarAlexNet())
        state = torch.hub.load_state_dict_from_url(
            "https://perceptual-advex.s3.us-east-2.amazonaws.com/" "alexnet_cifar.pt",
            progress=True,
        )
        lpips_model.load_state_dict(state["model"])
    else:
        lpips_model = lpips_model_spec
    return lpips_model
