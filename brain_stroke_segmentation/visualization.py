"""Visualization utilities for training metrics."""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def plot_training_history(history: Dict[str, List[float]], output_path: Path | str = None) -> None:
    """
    Plot training history metrics.

    Args:
        history: Dictionary with training metrics
        output_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].plot(history.get("train_loss", []), label="Training Loss", linewidth=2)
    axes[0, 0].plot(history.get("val_loss", []), label="Validation Loss", linewidth=2)
    axes[0, 0].set_title("Model Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.get("train_dice", []), label="Training Dice", linewidth=2)
    axes[0, 1].plot(history.get("val_dice", []), label="Validation Dice", linewidth=2)
    axes[0, 1].set_title("Dice Coefficient")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(history.get("train_iou", []), label="Training IoU", linewidth=2)
    axes[0, 2].plot(history.get("val_iou", []), label="Validation IoU", linewidth=2)
    axes[0, 2].set_title("IoU Score")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("IoU")
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 0].plot(history.get("val_sensitivity", []), label="Validation Sensitivity", linewidth=2)
    axes[1, 0].set_title("Sensitivity (Recall)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Sensitivity")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(history.get("val_specificity", []), label="Validation Specificity", linewidth=2)
    axes[1, 1].set_title("Specificity")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Specificity")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[1, 2].plot(history.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
    axes[1, 2].set_title("Binary Accuracy")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Accuracy")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()

