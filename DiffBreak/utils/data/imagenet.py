import os
import pickle
from platformdirs import user_cache_dir
import numpy as np

from .data import PILData
from ..cache import load_dataset_from_cache


class ImageNetData(PILData):
    def __init__(self, **kwargs):
        data_base = load_dataset_from_cache("imagenet")
        with open(data_base, "rb") as f:
            data = pickle.load(f)
        x, y = data
        cache_dir = os.path.join(user_cache_dir(), "DiffBreak", "hub")
        x_out = []
        for sample in x:
            x_out.append(os.path.join(cache_dir, str(sample)))
        x = np.array(x_out)
        data = (x, y)
        super().__init__(data, num_classes=1000, image_size=256, crop_sz=224, **kwargs)