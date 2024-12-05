# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import math
from collections import OrderedDict
from platformdirs import user_cache_dir
import torch
import torch.nn.functional as F
import torch.nn as nn
from robustbench import load_model
from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish

from ...cache import load_classifier_from_cache
from .classifier import Classifier


def update_state_dict(state_dict, idx_start=9):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]
        new_state_dict[name] = v
    return new_state_dict


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        num_input_channels = 3
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = (x - self.mean.to(x.device)) / self.std.to(x.device)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """Based on code from https://github.com/yaodongyu/TRADES"""

    def __init__(
        self,
        depth=28,
        num_classes=10,
        widen_factor=10,
        sub_block1=False,
        dropRate=0.0,
        bias_last=True,
    ):
        super(WideResNet, self).__init__()

        num_input_channels = 3
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(
                n, nChannels[0], nChannels[1], block, 1, dropRate
            )
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x):
        out = (x - self.mean.to(x.device)) / self.std.to(x.device)
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def WideResNet_70_16():
    return WideResNet(depth=70, widen_factor=16, dropRate=0.0)


def WideResNet_70_16_dropout():
    return WideResNet(depth=70, widen_factor=16, dropRate=0.3)


###classifiers
class WIDERESNET_28_10(Classifier):
    def __init__(self):
        model_dir = os.path.join(
            user_cache_dir(), "DiffBreak/hub/checkpoints/classifiers/torch/cifar10/wrn"
        )
        model = load_model(
            model_name="Standard",
            dataset="cifar10",
            threat_model="Linf",
            model_dir=model_dir,
        )
        super().__init__(model=model, softmaxed=False)


class WRN_28_10_AT0(Classifier):
    def __init__(self):
        model_dir = os.path.join(
            user_cache_dir(), "DiffBreak/hub/checkpoints/classifiers/torch/cifar10/wrn"
        )
        model = load_model(
            model_name="Gowal2021Improving_28_10_ddpm_100m",
            dataset="cifar10",
            threat_model="Linf",
            model_dir=model_dir,
        )
        super().__init__(model=model, softmaxed=False)


class WRN_28_10_AT1(Classifier):
    def __init__(self):
        model_dir = os.path.join(
            user_cache_dir(), "DiffBreak/hub/checkpoints/classifiers/torch/cifar10/wrn"
        )
        model = load_model(
            model_name="Gowal2020Uncovering_28_10_extra",
            dataset="cifar10",
            threat_model="Linf",
            model_dir=model_dir,
        )
        super().__init__(model=model, softmaxed=False)


class WRN_70_16_AT0(Classifier):
    def __init__(self):
        model_dir = os.path.join(
            user_cache_dir(), "DiffBreak/hub/checkpoints/classifiers/torch/cifar10/wrn"
        )
        model = load_model(
            model_name="Gowal2021Improving_70_16_ddpm_100m",
            dataset="cifar10",
            threat_model="Linf",
            model_dir=model_dir,
        )
        super().__init__(model=model, softmaxed=False)


class WRN_70_16_AT1(Classifier):
    def __init__(self):
        model_dir = os.path.join(
            user_cache_dir(), "DiffBreak/hub/checkpoints/classifiers/torch/cifar10/wrn"
        )
        model = load_model(
            model_name="Rebuffi2021Fixing_70_16_cutmix_extra",
            dataset="cifar10",
            threat_model="Linf",
            model_dir=model_dir,
        )
        super().__init__(model=model, softmaxed=False)


class WRN_70_16_L2_AT1(Classifier):
    def __init__(self):
        model_dir = os.path.join(
            user_cache_dir(), "DiffBreak/hub/checkpoints/classifiers/torch/cifar10/wrn"
        )
        model = load_model(
            model_name="Rebuffi2021Fixing_70_16_cutmix_extra",
            dataset="cifar10",
            threat_model="L2",
            model_dir=model_dir,
        )
        super().__init__(model=model, softmaxed=False)


class WIDERESNET_70_16(Classifier):
    def __init__(self):
        model_args = {
            "num_classes": 10,
            "depth": 70,
            "width": 16,
            "activation_fn": Swish,
        }
        model = DMWideResNet(**model_args)
        path = load_classifier_from_cache("torch/cifar10/wideresnet-70-16")
        ckpt = torch.load(path, weights_only=True)["model_state_dict"]
        ckpt = update_state_dict(ckpt)
        model.load_state_dict(ckpt)
        super().__init__(model=model, softmaxed=False)


class RESNET_50(Classifier):
    def __init__(self):
        model = ResNet50()
        path = load_classifier_from_cache("torch/cifar10/resnet50")
        ckpt = torch.load(path)
        ckpt = update_state_dict(ckpt, idx_start=7)
        model.load_state_dict(ckpt)
        super().__init__(model=model, softmaxed=False)


class WRN_70_16_DROPOUT(Classifier):
    def __init__(self):
        model = WideResNet_70_16_dropout()
        path = load_classifier_from_cache("torch/cifar10/wrn-70-16-dropout")
        ckpt = torch.load(path)
        ckpt = update_state_dict(ckpt, idx_start=7)
        model.load_state_dict(ckpt)
        super().__init__(model=model, softmaxed=False)
