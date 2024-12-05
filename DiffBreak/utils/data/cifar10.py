import pickle

from .data import NumpyData
from ..cache import load_dataset_from_cache


class Cifar10Data(NumpyData):
    def __init__(self, **kwargs):
        data_base = load_dataset_from_cache("cifar10", index_only=True)
        with open(data_base, "rb") as f:
            data = pickle.load(f)
        super().__init__(data, num_classes=10, image_size=32, crop_sz=None, **kwargs)
