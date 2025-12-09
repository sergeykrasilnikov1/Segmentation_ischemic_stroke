"""Data loading and preprocessing utilities."""

import os
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split
from typing import Tuple

from brain_stroke_segmentation.dataset import StrokeDataset
from brain_stroke_segmentation.transforms import get_transforms


def load_and_preprocess_data(dataset_path: Path | str) -> Tuple[list[Path], list[Path | None]]:
    """
    Scan dataset folders and pair PNG with OVERLAY having the same filename.

    Args:
        dataset_path: Root path to the dataset

    Returns:
        Tuple of (image_paths, mask_paths) lists
    """
    dataset_path = Path(dataset_path)
    images_paths: list[Path] = []
    masks_paths: list[Path | None] = []

    for class_dir in ["Bleeding", "Ischemia", "Normal"]:
        png_dir = dataset_path / class_dir / "PNG"
        overlay_dir = dataset_path / class_dir / "OVERLAY"

        if not png_dir.exists():
            continue

        png_files = sorted(glob(str(png_dir / "*.png")))

        for png_file in png_files:
            filename = Path(png_file).name
            overlay_file = overlay_dir / filename
            if overlay_dir.exists() and overlay_file.exists():
                images_paths.append(Path(png_file))
                masks_paths.append(overlay_file)
            else:
                images_paths.append(Path(png_file))
                masks_paths.append(None)

    return images_paths, masks_paths


def create_datasets(
    dataset_path: Path | str,
    img_height: int,
    img_width: int,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[StrokeDataset, StrokeDataset]:
    """
    Create train and validation datasets.

    Args:
        dataset_path: Path to dataset root
        img_height: Target image height
        img_width: Target image width
        test_size: Fraction of data for validation
        random_state: Random seed for splitting

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    image_paths, mask_paths = load_and_preprocess_data(dataset_path)

    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=test_size, random_state=random_state
    )

    train_dataset = StrokeDataset(
        train_images, train_masks, img_height, img_width, get_transforms(is_training=True)
    )
    val_dataset = StrokeDataset(
        val_images, val_masks, img_height, img_width, get_transforms(is_training=False)
    )

    return train_dataset, val_dataset


def download_data(output_path: Path | str) -> None:
    """
    Download data from open sources.

    Args:
        output_path: Path where to save the data
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        import dvc.repo
        repo = dvc.repo.Repo()
        repo.pull()
        print("Data downloaded successfully via DVC")
        return
    except Exception as e:
        print(f"DVC pull failed: {e}")

    try:
        import kagglehub
        dataset_path = kagglehub.dataset_download("ayushtibrewal/brain-stroke-images")
        print(f"Data downloaded from Kaggle to: {dataset_path}")
    except ImportError:
        print("kagglehub not available. Please install: pip install kagglehub")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Please download data manually or configure DVC remote")

