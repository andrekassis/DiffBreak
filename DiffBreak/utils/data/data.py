import logging
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataloader import Dataset

from ..logs import get_logger

logger = get_logger()


class Data(Dataset):
    def __init__(self, data, num_classes, image_size, crop_sz=None, **kwargs):
        super().__init__()
        self.data, self.num_classes, self.target_size = data, num_classes, image_size

        self.crop = (
            transforms.CenterCrop(crop_sz)
            if crop_sz is not None
            else torch.nn.Identity()
        )
        self.resize = transforms.Resize(self.target_size)
        self.crop_sz = crop_sz if crop_sz is not None else self.target_size
        self.dataset_size = -1

    def setup(self, total_samples=256, balanced_splits=False):
        assert len(self.data[0]) >= total_samples
        if balanced_splits:
            self.data = self.get_balanced_set(self.data, total_samples)
        self.dataset_size = len(self.data[0])

    def get_balanced_set(self, data, total_samples):
        subsets = [list(np.where(data[1] == label)[0]) for label in np.unique(data[1])]
        subset_lens = [
            min(len(subset), int(np.ceil(total_samples / self.num_classes)))
            for subset in subsets
        ]
        selected = [random.sample(subset, l) for subset, l in zip(subsets, subset_lens)]
        remainder = [
            list(set(subset) - set(selected_set))
            for subset, selected_set in zip(subsets, selected)
        ]
        remainder = [rem for rem in remainder if len(rem) > 0]
        remaineder = [r for rem in remainder for r in rem]
        idx = np.concatenate(
            [np.array(selected_set) for selected_set in selected], axis=0
        )
        needed = total_samples - len(idx)
        if needed > 0:
            remainder = np.array(random.sample(remainder, needed))
            idx = np.concatenate((idx, remainder), axis=0)
        idx = np.random.permutation(idx)[:total_samples]
        return (
            np.array(data[0])[idx].tolist(),
            data[1][idx],
        )

    @staticmethod
    def load_label(y, num_classes):
        return (
            torch.nn.functional.one_hot(
                torch.tensor(y.astype(np.int64)), num_classes=num_classes
            )
            .view(num_classes)
            .long()
        )

    def _post_process(self, x):
        return x

    def load_input(self, x):
        return self._post_process(self.crop(self.resize(self._do_load(x)))).view(
            3, self.crop_sz, self.crop_sz
        )

    def __len__(self):
        if self.dataset_size == -1:
            logger.error(
                "You must first execute the setup method before using the dataset"
            )
            exit(0)
        return self.dataset_size

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        if self.dataset_size == -1:
            logger.error(
                "You must first execute the setup method before using the dataset"
            )
            exit(0)

        return self.load_input(self.data[0][idx]), self.load_label(
            self.data[1][idx], self.num_classes
        )


class PILData(Data):
    def _do_load(self, x):
        return Image.open(x).convert("RGB")

    def _post_process(self, x):
        return transforms.ToTensor()(x)


class NumpyData(Data):
    def _do_load(self, x):
        return torch.from_numpy(x).permute(2, 0, 1) / 255.0
