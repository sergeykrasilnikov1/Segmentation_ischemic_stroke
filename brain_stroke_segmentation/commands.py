"""CLI commands for training and inference."""

import fire
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from brain_stroke_segmentation.lightning_module import StrokeSegmentationModule
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

    model = StrokeSegmentationModule(
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        learning_rate=cfg.train.learning_rate,
        mlflow_uri=cfg.logging.mlflow_uri,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.train.checkpoint_dir),
        filename="best-{epoch:02d}-{val_dice:.4f}",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_dice",
        mode="max",
        patience=cfg.train.patience,
        verbose=True,
    )

    logger = pl.loggers.MLFlowLogger(
        experiment_name="brain_stroke_segmentation",
        tracking_uri=cfg.logging.mlflow_uri,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if cfg.train.use_gpu else "cpu",
        devices=cfg.train.num_devices,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=cfg.train.log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader)

    if cfg.production.convert_onnx:
        best_model_path = checkpoint_callback.best_model_path
        onnx_path = Path(cfg.production.onnx_output_path)
        convert_to_onnx(
            model_path=best_model_path,
            output_path=onnx_path,
            img_height=cfg.data.img_height,
            img_width=cfg.data.img_width,
            encoder_name=cfg.model.encoder_name,
        )

    print("Training completed!")


def infer(config_path: str = "configs", config_name: str = "config") -> None:
    """
    Run inference on new images.

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
        print(f"DVC pull failed: {e}")

    inference = StrokeInference(
        model_path=cfg.infer.model_path,
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width,
        device=cfg.infer.device,
        encoder_name=cfg.model.encoder_name,
    )

    input_path = Path(cfg.infer.input_path)
    output_path = Path(cfg.infer.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        rgb, prob, binary = inference.predict(input_path, threshold=cfg.infer.threshold)
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
        plt.savefig(output_path / f"{input_path.stem}_prediction.png")
        plt.close()

    elif input_path.is_dir():
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
        results = inference.predict_batch(image_files, threshold=cfg.infer.threshold)
        print(f"Processed {len(results)} images. Results saved to {output_path}")

    else:
        raise ValueError(f"Input path must be a file or directory: {input_path}")


def main():
    """Main entry point."""
    fire.Fire({"train": train, "infer": infer})


if __name__ == "__main__":
    main()

