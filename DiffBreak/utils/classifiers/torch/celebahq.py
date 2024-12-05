# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import logging
import numpy as np
import torch
import torch.nn as nn

from ...cache import load_classifier_from_cache
from .classifier import Classifier
from ...logs import get_logger

logger = get_logger()

softmax = torch.nn.Softmax(dim=1)


def lerp_clip(a, b, t):
    return a + (b - a) * torch.clamp(t, 0.0, 1.0)


class WScaleLayer(nn.Module):
    def __init__(self, size, fan_in, gain=np.sqrt(2), bias=True):
        super(WScaleLayer, self).__init__()
        self.scale = gain / np.sqrt(fan_in)  # No longer a parameter
        if bias:
            self.b = nn.Parameter(torch.randn(size))
        else:
            self.b = 0
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale
        # modified to remove warning
        if type(self.b) == nn.Parameter and len(x_size) == 4:
            x = x + self.b.view(1, -1, 1, 1).expand(
                x_size[0], self.size, x_size[2], x_size[3]
            )
        if type(self.b) == nn.Parameter and len(x_size) == 2:
            x = x + self.b.view(1, -1).expand(x_size[0], self.size)
        return x


class WScaleConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        bias=True,
        gain=np.sqrt(2),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        fan_in = in_channels * kernel_size * kernel_size
        self.wscale = WScaleLayer(out_channels, fan_in, gain=gain, bias=bias)

    def forward(self, x):
        return self.wscale(self.conv(x))


class WScaleLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, gain=np.sqrt(2)):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.wscale = WScaleLayer(out_channels, in_channels, gain=gain, bias=bias)

    def forward(self, x):
        return self.wscale(self.linear(x))


class FromRGB(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, act=nn.LeakyReLU(0.2), bias=True
    ):
        super().__init__()
        self.conv = WScaleConv2d(
            in_channels, out_channels, kernel_size, padding=0, bias=bias
        )
        self.act = act

    def forward(self, x):
        return self.act(self.conv(x))


class Downscale2d(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        self.downsample = nn.AvgPool2d(kernel_size=factor, stride=factor)

    def forward(self, x):
        return self.downsample(x)


class DownscaleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        conv0_channels,
        conv1_channels,
        kernel_size,
        padding,
        bias=True,
        act=nn.LeakyReLU(0.2),
    ):
        super().__init__()
        self.downscale = Downscale2d()
        self.conv0 = WScaleConv2d(
            in_channels,
            conv0_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv1 = WScaleConv2d(
            conv0_channels,
            conv1_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.act = act

    def forward(self, x):
        x = self.act(self.conv0(x))
        # conv2d_downscale2d applies downscaling before activation
        # the order matters here! has to be conv -> bias -> downscale -> act
        x = self.conv1(x)
        x = self.downscale(x)
        x = self.act(x)
        return x


class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        group_size = min(self.group_size, x.shape[0])
        s = x.shape
        y = x.view([group_size, -1, s[1], s[2], s[3]])
        y = y.float()
        y = y - torch.mean(y, dim=0, keepdim=True)
        y = torch.mean(y * y, dim=0)
        y = torch.sqrt(y + 1e-8)
        y = torch.mean(
            torch.mean(torch.mean(y, dim=3, keepdim=True), dim=2, keepdim=True),
            dim=1,
            keepdim=True,
        )
        y = y.type(x.type())
        y = y.repeat(group_size, 1, s[2], s[3])
        return torch.cat([x, y], dim=1)


class PredictionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        dense0_feat,
        dense1_feat,
        out_feat,
        pool_size=2,
        act=nn.LeakyReLU(0.2),
        use_mbstd=True,
    ):
        super().__init__()
        self.use_mbstd = use_mbstd  # attribute classifiers don't have this
        if self.use_mbstd:
            self.mbstd_layer = MinibatchStdLayer()
        # MinibatchStdLayer adds an additional feature dimension
        self.conv = WScaleConv2d(
            in_channels + int(self.use_mbstd), dense0_feat, kernel_size=3, padding=1
        )
        self.dense0 = WScaleLinear(dense0_feat * pool_size * pool_size, dense1_feat)
        self.dense1 = WScaleLinear(dense1_feat, out_feat, gain=1)
        self.act = act

    def forward(self, x):
        if self.use_mbstd:
            x = self.mbstd_layer(x)
        x = self.act(self.conv(x))
        x = x.view([x.shape[0], -1])
        x = self.act(self.dense0(x))
        x = self.dense1(x)
        return x


class D(nn.Module):
    def __init__(
        self,
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=128,  # Input resolution. Overridden based on dataset.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        fixed_size=False,  # True = load fromrgb_lod0 weights only
        use_mbstd=True,  # False = no mbstd layer in PredictionBlock
        **kwargs,
    ):  # Ignore unrecognized keyword args.
        super().__init__()

        self.resolution_log2 = resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.register_buffer("lod_in", torch.from_numpy(np.array(0.0)))

        res = resolution_log2

        setattr(self, "fromrgb_lod0", FromRGB(num_channels, nf(res - 1), 1))

        for i, res in enumerate(range(resolution_log2, 2, -1), 1):
            lod = resolution_log2 - res
            block = DownscaleConvBlock(
                nf(res - 1), nf(res - 1), nf(res - 2), kernel_size=3, padding=1
            )
            setattr(self, "%dx%d" % (2**res, 2**res), block)
            fromrgb = FromRGB(3, nf(res - 2), 1)
            if not fixed_size:
                setattr(self, "fromrgb_lod%d" % i, fromrgb)

        res = 2
        pool_size = 2**res
        block = PredictionBlock(
            nf(res + 1 - 2), nf(res - 1), nf(res - 2), 1, pool_size, use_mbstd=use_mbstd
        )
        setattr(self, "%dx%d" % (pool_size, pool_size), block)
        self.downscale = Downscale2d()
        self.fixed_size = fixed_size

    def forward(self, img):
        x = self.fromrgb_lod0(img)
        for i, res in enumerate(range(self.resolution_log2, 2, -1), 1):
            lod = self.resolution_log2 - res
            x = getattr(self, "%dx%d" % (2**res, 2**res))(x)
            if not self.fixed_size:
                img = self.downscale(img)
                y = getattr(self, "fromrgb_lod%d" % i)(img)
                x = lerp_clip(x, y, self.lod_in - lod)
        res = 2
        pool_size = 2**res
        out = getattr(self, "%dx%d" % (pool_size, pool_size))(x)
        return out


def max_res_from_state_dict(state_dict):
    for i in range(3, 12):
        if "%dx%d.conv0.conv.weight" % (2**i, 2**i) not in state_dict:
            break
    return 2 ** (i - 1)


def from_state_dict(state_dict, fixed_size=False, use_mbstd=True):
    res = max_res_from_state_dict(state_dict)
    d = D(num_channels=3, resolution=res, fixed_size=fixed_size, use_mbstd=use_mbstd)
    d.load_state_dict(state_dict)
    return d


def downsample(images, size=256):
    if images.shape[2] > size:
        factor = images.shape[2] // size
        assert factor * size == images.shape[2]
        images = images.view(
            [
                -1,
                images.shape[1],
                images.shape[2] // factor,
                factor,
                images.shape[3] // factor,
                factor,
            ]
        )
        images = images.mean(dim=[3, 5])
        return images
    else:
        assert images.shape[-1] == 256
        return images


def get_logit(net, im):
    im_256 = downsample(im)
    logit = net(im_256)
    return logit


def get_softmaxed(net, im):
    logit = get_logit(net, im)
    logits = torch.cat([logit, -logit], dim=1)
    softmaxed = softmax(torch.cat([logit, -logit], dim=1))[:, 1]
    return logits, softmaxed


def load_attribute_classifier(attribute, ckpt_path=None):
    if ckpt_path is None:
        base_path = "pretrained/celebahq"
        attribute_pkl = os.path.join(base_path, attribute, "net_best.pth")
        ckpt = torch.load(attribute_pkl)
    else:
        ckpt = torch.load(ckpt_path)

    if "valacc" in ckpt.keys():
        print("Validation acc on raw images: %0.5f" % ckpt["valacc"])
    detector = from_state_dict(ckpt["state_dict"], fixed_size=True, use_mbstd=False)
    return detector


class CelebHQWrapper(torch.nn.Module):
    def __init__(self, classifier_name, ckpt_path=None):
        super().__init__()
        self.net = load_attribute_classifier(classifier_name, ckpt_path)

    def forward(self, ims):
        out = (ims - 0.5) / 0.5
        return get_softmaxed(self.net, out)[0]

    def to(self, device):
        self.net.to(device)
        return super().to(device)

    def eval(self):
        self.net.eval()
        return super().eval()


class NET_BEST(Classifier):
    def __init__(self, attribute=None):
        atts = [
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young",
        ]

        if attribute is None:
            logger.error(
                "celeba-hq classifiers must be initialized with an 'attribute' argument. \
                           Available attributes are {atts}"
            )
            exit(1)
        assert attribute in atts
        path = os.path.join(
            load_classifier_from_cache("torch/celebahq"), attribute, "net_best.pth"
        )
        model = CelebHQWrapper(attribute, ckpt_path=path)
        super().__init__(model=model, softmaxed=False)
