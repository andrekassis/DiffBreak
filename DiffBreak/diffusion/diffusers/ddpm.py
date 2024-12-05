import os
import shutil
from platformdirs import user_cache_dir
import numpy as np
import torch
from diffusers import UNet2DModel

from ...utils.cache import load_dm_from_cache
from .model import ModelWrapper


class DDPMModelWrapper(ModelWrapper):
    def forward(self, x, t):
        return self.model(x, t)["sample"], None


def DDPMModel(
    image_size,
):
    scaler = lambda t: (t.float() * 1000).round()
    return (
        DDPMModelWrapper(
            UNet2DModel.from_pretrained(
                load_dm_from_cache("ddpm_model")
            ).requires_grad_(False)
        ),
        scaler,
    )
