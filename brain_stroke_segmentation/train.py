"""Training utilities for stroke segmentation using pure PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple, Optional
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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Tuple of (average_loss, average_dice, average_iou)
    """
    model.train()
    running_loss = running_dice = running_iou = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        dice_score = calculate_dice_score(probs, masks)
        iou_score = calculate_iou_score(probs, masks)

        running_loss += loss.item()
        running_dice += dice_score
        running_iou += iou_score

        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Dice": f"{dice_score:.4f}",
                "IoU": f"{iou_score:.4f}",
            }
        )

    N = len(dataloader)
    return running_loss / N, running_dice / N, running_iou / N


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    """
    Validate for one epoch.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run on

    Returns:
        Tuple of (loss, dice, iou, sensitivity, specificity, accuracy)
    """
    model.eval()
    running_loss = running_dice = running_iou = 0.0
    running_sensitivity = running_specificity = running_accuracy = 0.0
    progress_bar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            probs = torch.sigmoid(logits)
            dice_score = calculate_dice_score(probs, masks)
            iou_score = calculate_iou_score(probs, masks)
            sensitivity = calculate_sensitivity(probs, masks)
            specificity = calculate_specificity(probs, masks)

            pred_binary = (probs > 0.5).float()
            accuracy = (pred_binary == masks).float().mean().item()

            running_loss += loss.item()
            running_dice += dice_score
            running_iou += iou_score
            running_sensitivity += sensitivity
            running_specificity += specificity
            running_accuracy += accuracy

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Dice": f"{dice_score:.4f}",
                    "IoU": f"{iou_score:.4f}",
                }
            )

    N = len(dataloader)
    return (
        running_loss / N,
        running_dice / N,
        running_iou / N,
        running_sensitivity / N,
        running_specificity / N,
        running_accuracy / N,
    )


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    encoder_name: str = "efficientnet-b4",
    encoder_weights: str = "imagenet",
    learning_rate: float = 1e-4,
    epochs: int = 50,
    patience: int = 15,
    checkpoint_dir: str = "models/checkpoints",
    device: Optional[torch.device] = None,
    mlflow_uri: Optional[str] = None,
    log_every_n_steps: int = 10,
) -> Tuple[nn.Module, Dict]:
    """
    Train the model using pure PyTorch.

    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        encoder_name: Encoder backbone name
        encoder_weights: Pretrained weights
        learning_rate: Learning rate
        epochs: Number of epochs
        patience: Early stopping patience
        checkpoint_dir: Directory to save checkpoints
        device: Device to run on (auto-detect if None)
        mlflow_uri: MLflow tracking URI (optional)
        log_every_n_steps: Log every N steps

    Returns:
        Tuple of (best_model, history_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print("Building model...")
    model = build_model(encoder_name=encoder_name, encoder_weights=encoder_weights)
    model = model.to(device)
    print(f"\nModel built successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7, verbose=True
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_iou": [],
        "val_iou": [],
        "val_sensitivity": [],
        "val_specificity": [],
        "val_accuracy": [],
    }

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_path / "best_model.pth"

    best_dice = 0.0
    patience_counter = 0

    # Setup MLflow if URI provided
    mlflow_active = False
    if mlflow_uri:
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            experiment = mlflow.get_experiment_by_name("brain_stroke_segmentation")
            if experiment is None:
                mlflow.create_experiment("brain_stroke_segmentation")
            mlflow.set_experiment("brain_stroke_segmentation")
            mlflow_active = True

            # Log hyperparameters
            git_commit = get_git_commit_id()
            mlflow.log_params(
                {
                    "encoder_name": encoder_name,
                    "encoder_weights": encoder_weights,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "patience": patience,
                    "git_commit_id": git_commit,
                }
            )
            print("MLflow logging enabled")
        except Exception as e:
            print(f"MLflow not available ({e}), continuing without MLflow")
            mlflow_active = False

    print("Starting training...")
    print("=" * 50)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)

        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_dice, val_iou, val_sensitivity, val_specificity, val_accuracy = (
            validate_epoch(model, val_loader, criterion, device)
        )

        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        history["val_sensitivity"].append(val_sensitivity)
        history["val_specificity"].append(val_specificity)
        history["val_accuracy"].append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Dice: {val_dice:.4f}, Val   IoU: {val_iou:.4f}")
        print(
            f"Val Sensitivity: {val_sensitivity:.4f}, Val Specificity: {val_specificity:.4f}"
        )
        print(f"Val Accuracy: {val_accuracy:.4f}")

        # Log to MLflow
        if mlflow_active:
            try:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_dice": train_dice,
                        "val_dice": val_dice,
                        "train_iou": train_iou,
                        "val_iou": val_iou,
                        "val_sensitivity": val_sensitivity,
                        "val_specificity": val_specificity,
                        "val_accuracy": val_accuracy,
                    },
                    step=epoch,
                )
            except Exception as e:
                print(f"Warning: MLflow logging failed: {e}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Dice: {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    print(f"\nTraining completed! Best Dice: {best_dice:.4f}")
    print(f"Best model saved to: {best_model_path}")

    return model, history
