"""Metrics for segmentation evaluation."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice loss.

        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred_prob: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            y_pred_prob: Predicted probabilities
            y_true: Ground truth labels

        Returns:
            Dice loss value
        """
        y_pred = y_pred_prob.view(-1)
        y_true = y_true.view(-1)
        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(self):
        """Initialize combined loss with BCE and Dice components."""
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, y_pred_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.

        Args:
            y_pred_logits: Predicted logits
            y_true: Ground truth labels

        Returns:
            Combined loss value
        """
        prob = torch.sigmoid(y_pred_logits)
        return 0.5 * self.bce(y_pred_logits, y_true) + 0.5 * self.dice(prob, y_true)


def calculate_dice_score(
    y_pred_prob: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-6
) -> float:
    """Calculate Dice coefficient."""
    y_pred = (y_pred_prob > 0.5).float()
    intersection = (y_pred * y_true).sum()
    dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dice.item()


def calculate_iou_score(
    y_pred_prob: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-6
) -> float:
    """Calculate IoU (Intersection over Union) score."""
    y_pred = (y_pred_prob > 0.5).float()
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_sensitivity(y_pred_prob: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculate sensitivity (recall)."""
    y_pred = (y_pred_prob > 0.5).float()
    true_positives = (y_pred * y_true).sum()
    actual_positives = y_true.sum()
    if actual_positives == 0:
        return 0.0
    return (true_positives / actual_positives).item()


def calculate_specificity(y_pred_prob: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculate specificity."""
    y_pred = (y_pred_prob > 0.5).float()
    true_negatives = ((1 - y_pred) * (1 - y_true)).sum()
    actual_negatives = (1 - y_true).sum()
    if actual_negatives == 0:
        return 0.0
    return (true_negatives / actual_negatives).item()


def calculate_accuracy(y_pred_prob: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculate pixel-wise accuracy."""
    y_pred = (y_pred_prob > 0.5).float()
    accuracy = (y_pred == y_true).float().mean()
    return accuracy.item()
