"""PyTorch Lightning module for training."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
import mlflow.pytorch

from brain_stroke_segmentation.model import build_model
from brain_stroke_segmentation.metrics import (
    CombinedLoss,
    calculate_dice_score,
    calculate_iou_score,
    calculate_sensitivity,
    calculate_specificity,
    calculate_accuracy,
)
from brain_stroke_segmentation.utils import get_git_commit_id


class StrokeSegmentationModule(pl.LightningModule):
    """PyTorch Lightning module for stroke segmentation."""

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        learning_rate: float = 1e-4,
        mlflow_uri: str = "http://127.0.0.1:8080",
    ):
        """
        Initialize the Lightning module.

        Args:
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights
            learning_rate: Learning rate
            mlflow_uri: MLflow tracking URI
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.mlflow_uri = mlflow_uri

        self.model = build_model(encoder_name, encoder_weights)
        self.criterion = CombinedLoss()

        mlflow.set_tracking_uri(mlflow_uri)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, masks = batch
        masks = masks.unsqueeze(1)

        logits = self(images)
        loss = self.criterion(logits, masks)

        probs = torch.sigmoid(logits)
        dice_score = calculate_dice_score(probs, masks)
        iou_score = calculate_iou_score(probs, masks)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_dice", dice_score, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", iou_score, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        images, masks = batch
        masks = masks.unsqueeze(1)

        logits = self(images)
        loss = self.criterion(logits, masks)

        probs = torch.sigmoid(logits)
        dice_score = calculate_dice_score(probs, masks)
        iou_score = calculate_iou_score(probs, masks)
        sensitivity = calculate_sensitivity(probs, masks)
        specificity = calculate_specificity(probs, masks)
        accuracy = calculate_accuracy(probs, masks)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", dice_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", iou_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_sensitivity", sensitivity, on_step=False, on_epoch=True)
        self.log("val_specificity", specificity, on_step=False, on_epoch=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)

        return {
            "val_loss": loss,
            "val_dice": dice_score,
            "val_iou": iou_score,
            "val_sensitivity": sensitivity,
            "val_specificity": specificity,
            "val_accuracy": accuracy,
        }

    def on_train_start(self) -> None:
        """Log hyperparameters and git commit to MLflow."""
        if self.logger and hasattr(self.logger, "experiment"):
            try:
                git_commit = get_git_commit_id()
                self.logger.log_hyperparams({"git_commit_id": git_commit})
            except Exception:
                pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_dice"},
        }

