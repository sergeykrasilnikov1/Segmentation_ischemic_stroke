"""CLI commands for training and inference."""

import fire
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from brain_stroke_segmentation.train import train_model
from brain_stroke_segmentation.data_loader import create_datasets, download_data
from brain_stroke_segmentation.inference import StrokeInference
from brain_stroke_segmentation.onnx_converter import convert_to_onnx
from brain_stroke_segmentation.tensorrt_converter import convert_to_tensorrt


def train(config_path: str = "configs", config_name: str = "config") -> None:
    """
    Train the stroke segmentation model.

    Args:
        config_path: Path to configs directory
        config_name: Name of config file
    """
    config_dir = Path(config_path).absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)

    try:
        import dvc.repo
        repo = dvc.repo.Repo()
        repo.pull()
    except Exception as e:
        print(f"DVC pull failed: {e}. Trying to download data...")
        download_data(cfg.data.dataset_path)

    train_dataset, val_dataset = create_datasets(
        dataset_path=cfg.data.dataset_path,
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    use_gpu = cfg.train.use_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        learning_rate=cfg.train.learning_rate,
        epochs=cfg.train.epochs,
        patience=cfg.train.patience,
        checkpoint_dir=cfg.train.checkpoint_dir,
        device=device,
        mlflow_uri=cfg.logging.mlflow_uri if hasattr(cfg.logging, "mlflow_uri") else None,
        log_every_n_steps=cfg.train.log_every_n_steps,
    )

    # Copy best model to models directory
        import shutil
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
    best_model_source = Path(cfg.train.checkpoint_dir) / "best_model.pth"
    best_model_dest = models_dir / "best_model.pth"
    if best_model_source.exists():
        shutil.copy2(best_model_source, best_model_dest)
        print(f"Best model copied to {best_model_dest}")

    if cfg.production.convert_onnx:
        try:
            best_model_path = Path(cfg.train.checkpoint_dir) / "best_model.pth"
            onnx_path = Path(cfg.production.onnx_output_path)
            convert_to_onnx(
                model_path=best_model_path,
                output_path=onnx_path,
                img_height=cfg.data.img_height,
                img_width=cfg.data.img_width,
                encoder_name=cfg.model.encoder_name,
            )
        except Exception as e:
            print(f"Warning: ONNX conversion failed: {e}")
            print("Model training completed, but ONNX conversion skipped.")

    print("Training completed!")


def infer(
    image_path: str,
    config_path: str = "configs",
    config_name: str = "config",
    model_path: str | None = None,
    output_path: str | None = None,
    threshold: float | None = None,
) -> None:
    """
    Run inference on new images.

    Args:
        image_path: Path to input image file or directory
        config_path: Path to configs directory
        config_name: Name of config file
        model_path: Override model path from config
        output_path: Override output path from config
        threshold: Override threshold from config
    """
    config_dir = Path(config_path).absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)

    try:
        import dvc.repo
        repo = dvc.repo.Repo()
        repo.pull()
    except Exception as e:
        print(f"DVC pull failed: {e}")

    model_path_override = model_path if model_path else cfg.infer.model_path
    output_path_override = output_path if output_path else cfg.infer.output_path
    threshold_override = threshold if threshold is not None else cfg.infer.threshold

    inference = StrokeInference(
        model_path=model_path_override,
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width,
        device=cfg.infer.device,
        encoder_name=cfg.model.encoder_name,
    )

    input_path = Path(image_path)
    output_path_final = Path(output_path_override)
    output_path_final.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        rgb, prob, binary = inference.predict(input_path, threshold=threshold_override)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(prob, cmap="gray")
        axes[1].set_title("Probability")
        axes[1].axis("off")
        axes[2].imshow(binary, cmap="gray")
        axes[2].set_title("Binary Mask")
        axes[2].axis("off")
        plt.savefig(output_path_final / f"{input_path.stem}_prediction.png")
        plt.close()
        print(f"Prediction saved to {output_path_final / f'{input_path.stem}_prediction.png'}")

    elif input_path.is_dir():
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
        results = inference.predict_batch(image_files, threshold=threshold_override)
        print(f"Processed {len(results)} images. Results saved to {output_path_final}")

    else:
        raise ValueError(f"Input path must be a file or directory: {input_path}")


def main():
    """Main entry point."""
    fire.Fire({"train": train, "infer": infer})


if __name__ == "__main__":
    main()

