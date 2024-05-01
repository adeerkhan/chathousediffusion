import random
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T

from .utils import convert_image_to_fn, exists

# dataset classes


class Dataset(Dataset):
    def __init__(
        self,
        folder_image,
        folder_mask,
        folder_text,
        image_size,
        exts=["jpg", "jpeg", "png", "tiff"],
        augment_flip=False,
        convert_image_to=None,
    ):
        super().__init__()
        # self.folder = folder
        self.image_size = image_size
        self.augment_flip = augment_flip
        self.image_paths = [
            p for ext in exts for p in Path(f"{folder_image}").glob(f"**/*.{ext}")
        ]
        self.mask_paths = [
            p for ext in exts for p in Path(f"{folder_mask}").glob(f"**/*.{ext}")
        ]
        self.image_paths.sort(key=lambda x: int(x.stem.split("_")[0]))
        self.mask_paths.sort(key=lambda x: int(x.stem.split("_")[0]))
        self.text_path = next(Path(f"{folder_text}").glob(f"**/*.csv"))
        texts = pd.read_csv(self.text_path)
        self.texts = [p for p in zip(texts["0"], texts["1"])]
        self.texts.sort(key=lambda x: int(x[0].replace(".png", "").split("/")[-1]))
        assert (
            len(self.image_paths) == len(self.mask_paths) == len(self.texts)
        ), "number of images, masks and texts should be the same"
        maybe_convert_fn = (
            partial(convert_image_to_fn, convert_image_to)
            if exists(convert_image_to)
            else nn.Identity()
        )

        self.transform = T.Compose(
            [
                T.Lambda(maybe_convert_fn),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(image_path)
        mask = Image.open(mask_path)
        img = self.transform(img)
        mask = self.transform(mask)
        if self.augment_flip and random.random() > 0.5:
            img = T.RandomHorizontalFlip(p=1)(img)
            mask = T.RandomHorizontalFlip(p=1)(mask)
        if self.augment_flip and random.random() > 0.5:
            img = T.RandomVerticalFlip(p=1)(img)
            mask = T.RandomVerticalFlip(p=1)(mask)
        text = self.texts[index][1]
        return img, mask, text