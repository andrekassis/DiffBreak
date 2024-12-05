import logging
import os
import pickle
from platformdirs import user_cache_dir
import numpy as np

from .data import PILData
from ..cache import load_dataset_from_cache
from ..logs import get_logger

logger = get_logger()

available_attributes = [
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


class CelebAHQData(PILData):
    def __init__(self, attribute=None, **kwargs):
        if attribute is None:
            logger.error(
                "celeba-hq dataset must be initialized with an 'attribute' argument. \
                           Available attributes are {available_attributes}"
            )
            exit(1)
        assert attribute in available_attributes

        data_base = load_dataset_from_cache("celeba-hq/celebahq__" + attribute)
        with open(data_base, "rb") as f:
            data = pickle.load(f)
        x, y = data
        cache_dir = os.path.join(user_cache_dir(), "DiffBreak", "hub")
        x_out = []
        for sample in x:
            x_out.append(os.path.join(cache_dir, str(sample)))
        x = np.array(x_out)
        data = (x, y)
        super().__init__(data, num_classes=2, image_size=256, crop_sz=None, **kwargs)
