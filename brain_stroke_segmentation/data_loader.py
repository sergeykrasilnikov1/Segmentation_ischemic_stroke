"""Data loading and preprocessing utilities."""

from glob import glob
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split

from brain_stroke_segmentation.dataset import StrokeDataset
from brain_stroke_segmentation.transforms import get_transforms


def load_and_preprocess_data(dataset_path: Path | str) -> Tuple[list[Path], list[Path | None]]:
    """
    Scan dataset folders and pair images with masks.

    Args:
        dataset_path: Root path to the dataset

    Returns:
        Tuple of (image_paths, mask_paths) lists
    """
    dataset_path = Path(dataset_path)
    images_paths: list[Path] = []
    masks_paths: list[Path | None] = []

    if not dataset_path.exists():
        print(f"Warning: Dataset path does not exist: {dataset_path}")
        return images_paths, masks_paths

    print(f"Loading data from: {dataset_path}")

    stroke_cropped = dataset_path / "stroke_cropped"
    stroke_noncropped = dataset_path / "stroke_noncropped"

    if stroke_cropped.exists() or stroke_noncropped.exists():
        for base_path in [stroke_cropped, stroke_noncropped]:
            if not base_path.exists():
                continue

            print(f"Searching in: {base_path}")
            print(f"Contents: {list(base_path.iterdir())[:5]}")

            cropped_dirs = [
                base_path / "CROPPED" / "TRAIN_CROP",
                base_path / "CROPPED" / "TEST_CROP",
            ]
            noncropped_dirs = [
                base_path / "NON_CROPPED" / "TRAIN",
                base_path / "NON_CROPPED" / "TEST",
            ]

            all_search_dirs = cropped_dirs + noncropped_dirs
            print(f"Search directories: {[str(d) for d in all_search_dirs]}")

            for search_dir in all_search_dirs:
                if not search_dir.exists():
                    print(f"  Directory does not exist: {search_dir}")
                    continue

                print(f"  Checking directory: {search_dir}")
                print(f"  Contents: {list(search_dir.iterdir())[:5]}")

                for class_dir in ["STROKE", "NORMAL", "Bleeding", "Ischemia", "Normal"]:
                    class_path = search_dir / class_dir
                    if not class_path.exists():
                        continue

                    print(f"    Found class directory: {class_path}")
                    image_files = (
                        sorted(glob(str(class_path / "*.jpg")))
                        + sorted(glob(str(class_path / "*.png")))
                        + sorted(glob(str(class_path / "*.jpeg")))
                    )

                    if image_files:
                        print(f"    Found {len(image_files)} images in {class_path}")

                        for img_file in image_files:
                            img_path = Path(img_file)
                            images_paths.append(img_path)

                            overlay_file = None
                            overlay_dir = search_dir / "OVERLAY" / class_dir
                            if overlay_dir.exists():
                                filename = img_path.stem
                                for ext in [".png", ".jpg", ".jpeg"]:
                                    potential_overlay = overlay_dir / f"{filename}{ext}"
                                    if potential_overlay.exists():
                                        overlay_file = potential_overlay
                                        break

                            masks_paths.append(overlay_file)
                    else:
                        print(f"    No image files found in {class_path}")
    else:
        for class_dir in ["Bleeding", "Ischemia", "Normal"]:
            png_dir = dataset_path / class_dir / "PNG"
            overlay_dir = dataset_path / class_dir / "OVERLAY"

            if not png_dir.exists():
                continue

            png_files = sorted(glob(str(png_dir / "*.png")))
            print(f"Found {len(png_files)} PNG files in {png_dir}")

            for png_file in png_files:
                filename = Path(png_file).name
                overlay_file = overlay_dir / filename
                if overlay_dir.exists() and overlay_file.exists():
                    images_paths.append(Path(png_file))
                    masks_paths.append(overlay_file)
                else:
                    images_paths.append(Path(png_file))
                    masks_paths.append(None)

    print(f"Total images loaded: {len(images_paths)}")
    if len(images_paths) == 0:
        print("Error: No images found! Check dataset structure.")
        print("Tried paths: stroke_cropped, stroke_noncropped, or Bleeding/Ischemia/Normal/PNG")

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

    if len(image_paths) == 0:
        raise ValueError(
            f"No images found in dataset path: {dataset_path}. "
            f"Please check that data is downloaded and structure is correct."
        )

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
        import shutil

        import kagglehub

        dataset_path = kagglehub.dataset_download("ayushtibrewal/brain-stroke-images")
        print(f"Data downloaded from Kaggle to: {dataset_path}")

        kaggle_data_path = Path(dataset_path)
        if kaggle_data_path.exists():
            if output_path.exists() and any(output_path.iterdir()):
                print(f"Data already exists at {output_path}, skipping copy")
                return

            print(f"Copying data from {kaggle_data_path} to {output_path}...")
            print(f"Kaggle data structure: {list(kaggle_data_path.iterdir())[:10]}")

            if kaggle_data_path.is_dir():
                brain_stroke_dir = kaggle_data_path / "Brain_Stroke_CT_Dataset"
                if brain_stroke_dir.exists():
                    print("Found Brain_Stroke_CT_Dataset subdirectory")
                    for item in brain_stroke_dir.iterdir():
                        dest = output_path / item.name
                        if item.is_dir():
                            shutil.copytree(item, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest)
                else:
                    for item in kaggle_data_path.iterdir():
                        dest = output_path / item.name
                        if item.is_dir():
                            shutil.copytree(item, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest)
            print(f"Data copied successfully to {output_path}")
            print(f"Output structure: {list(output_path.iterdir())[:10]}")
        else:
            print(f"Warning: Kaggle dataset path does not exist: {dataset_path}")
    except ImportError:
        print("kagglehub not available. Please install: pip install kagglehub")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Please download data manually or configure DVC remote")
