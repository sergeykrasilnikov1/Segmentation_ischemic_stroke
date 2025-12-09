"""Data augmentation and transforms."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(is_training: bool = True) -> A.Compose:
    """
    Get data transforms for training or validation.

    Args:
        is_training: Whether to apply training augmentations

    Returns:
        Albumentations compose object
    """
    if is_training:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

