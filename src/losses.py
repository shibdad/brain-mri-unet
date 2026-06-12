"""Loss functions and evaluation metrics for binary segmentation.

The tumor occupies a small fraction of each slice, so a plain pixel-wise loss is
dominated by the background. We combine BCE (stable pixel-wise gradient) with a
Dice loss (directly optimizes overlap) — a standard, robust pairing for medical
segmentation.

All functions take **logits** (raw model output) unless noted, and apply the
sigmoid internally so callers never double-activate.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Soft Dice loss = 1 - Dice coefficient, computed on probabilities."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten per batch element so the overlap is computed over all pixels.
        probs = probs.reshape(probs.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


@torch.no_grad()
def dice_coef(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    """Hard Dice coefficient on thresholded predictions (evaluation metric)."""
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


@torch.no_grad()
def iou_score(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1.0
) -> float:
    """Intersection-over-Union (Jaccard) on thresholded predictions."""
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
