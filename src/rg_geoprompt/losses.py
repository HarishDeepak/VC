"""Segmentation loss.

One loss for the whole project: pixel-wise cross-entropy with
ignore_index=255. Boundary pixels (255) are excluded from the loss
EVERYWHERE — Potsdam and Darmstadt alike. Clutter (5) IS in the loss
(it is a real class with logits channel 5); it is only excluded from mIoU.

SOURCE: notebook CELL18 — confirmed working.
"""
import torch
import torch.nn.functional as F

from .constants import BOUNDARY_ID


def seg_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over 6 classes, boundary pixels ignored.

    Args:
        logits: [B, 6, H, W] — full resolution (no .logits attribute,
                no interpolation needed for DINOv2/GeoPrompt models)
        labels: [B, H, W] int64 in {0..5, 255}
    """
    return F.cross_entropy(logits, labels, ignore_index=BOUNDARY_ID)


# Backwards-compatible alias matching the notebook name
seg_loss_dino = seg_loss
