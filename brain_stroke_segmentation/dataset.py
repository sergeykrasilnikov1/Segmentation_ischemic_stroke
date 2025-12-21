"""Dataset class for brain stroke segmentation."""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from brain_stroke_segmentation.utils import extract_red_mask_from_path


class StrokeDataset(Dataset):
    """Dataset for brain stroke CT images."""

    def __init__(
        self,
        image_paths: list[Path | str],
        mask_paths: list[Path | str | None],
        img_height: int,
        img_width: int,
        transforms: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to masks (can contain None for normal images)
            img_height: Target image height
            img_width: Target image width
            transforms: Optional transform function
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) if p is not None else None for p in mask_paths]
        self.img_height = img_height
        self.img_width = img_width
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        img = cv2.imread(str(self.image_paths[idx]))
        if img is None:
            img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)

        mask_path = self.mask_paths[idx]
        mask = extract_red_mask_from_path(mask_path, self.img_width, self.img_height)
        if mask is None:
            mask = np.zeros((self.img_height, self.img_width), dtype=np.float32)

        if self.transforms:
            transformed = self.transforms(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        return img, mask
