"""Evaluation metrics with anti-swap guards.

THE TWO METRICS ARE DIFFERENT AND MUST NEVER BE MERGED OR SWAPPED:

    compute_miou : 5 classes (0-4). ignore_ids MUST be (5, 255)
                   — clutter AND boundary excluded. ISPRS convention.
    compute_f1   : 6 classes (0-5). ignore_ids MUST be (255,)
                   — clutter INCLUDED, only boundary excluded. Brief req.

Both functions hard-assert their ignore sets so an accidental swap fails
loudly instead of silently corrupting results.

SOURCE: notebook CELL14 / CELL-R — metric logic confirmed correct.
(The notebook cell's duplicate definition + auto-run at the bottom is NOT
ported; evaluation is always an explicit call from a notebook.)
"""
from typing import Callable, Optional, Tuple

import torch

from .constants import (DEVICE, F1_IGNORE_IDS, F1_NUM_CLASSES,
                        MIOU_IGNORE_IDS, MIOU_NUM_CLASSES)


def _default_logits_fn(model, imgs):
    """Models in this project return a plain [B, 6, H, W] tensor."""
    return model(imgs)


def segformer_logits_fn(model, imgs):
    """Adapter for HuggingFace SegFormer: extract .logits and upsample
    from 1/4 resolution to label resolution."""
    import torch.nn.functional as F
    out = model(pixel_values=imgs).logits
    return F.interpolate(out, size=imgs.shape[-2:],
                         mode="bilinear", align_corners=False)


@torch.no_grad()
def compute_miou(model, loader,
                 num_classes: int = MIOU_NUM_CLASSES,
                 ignore_ids: Tuple[int, ...] = MIOU_IGNORE_IDS,
                 device: torch.device = DEVICE,
                 logits_fn: Optional[Callable] = None
                 ) -> Tuple[torch.Tensor, float]:
    """Per-class IoU + mIoU over classes 0-4 (ISPRS Potsdam convention).

    Returns:
        (iou_per_class [5], miou float)
    """
    assert num_classes == MIOU_NUM_CLASSES and tuple(sorted(ignore_ids)) == (5, 255), (
        "mIoU must use 5 classes with ignore_ids=(5, 255). "
        "If you want clutter included, you want compute_f1, not this.")
    logits_fn = logits_fn or _default_logits_fn
    model.eval()
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = logits_fn(model, imgs).argmax(dim=1)
        valid = torch.ones_like(lbls, dtype=torch.bool)
        for ig in ignore_ids:
            valid &= (lbls != ig)
        for cls in range(num_classes):
            pm = (preds == cls) & valid
            lm = (lbls == cls) & valid
            intersection[cls] += (pm & lm).sum().cpu()
            union[cls] += (pm | lm).sum().cpu()
    iou = intersection / (union + 1e-6)
    return iou, iou.mean().item()


@torch.no_grad()
def compute_f1(model, loader,
               num_classes: int = F1_NUM_CLASSES,
               ignore_ids: Tuple[int, ...] = F1_IGNORE_IDS,
               device: torch.device = DEVICE,
               logits_fn: Optional[Callable] = None) -> torch.Tensor:
    """Per-class F1 over ALL 6 classes including clutter (brief requirement).

    Returns:
        f1 tensor [6]. Report both f1.mean() (all 6) and f1[:5].mean().
    """
    assert num_classes == F1_NUM_CLASSES and tuple(ignore_ids) == (255,), (
        "F1 must use 6 classes (clutter included) with ignore_ids=(255,). "
        "If you want clutter excluded, you want compute_miou, not this.")
    logits_fn = logits_fn or _default_logits_fn
    model.eval()
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = logits_fn(model, imgs).argmax(dim=1)
        valid = torch.ones_like(lbls, dtype=torch.bool)
        for ig in ignore_ids:
            valid &= (lbls != ig)
        for cls in range(num_classes):
            pc = (preds == cls) & valid
            lc = (lbls == cls) & valid
            tp[cls] += (pc & lc).sum().cpu()
            fp[cls] += (pc & ~lc).sum().cpu()
            fn[cls] += (~pc & lc).sum().cpu()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return 2 * prec * rec / (prec + rec + 1e-6)


def f1_from_arrays(preds, labels, num_classes: int = F1_NUM_CLASSES,
                   ignore_id: int = 255) -> torch.Tensor:
    """F1 from raw prediction/label arrays (numpy or tensor) — used for
    Darmstadt OSM pseudo-GT evaluation where there is no DataLoader of
    (img, lbl) pairs. Same rule: 6 classes, only 255 excluded.
    """
    preds = torch.as_tensor(preds)
    labels = torch.as_tensor(labels)
    valid = labels != ignore_id
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for cls in range(num_classes):
        pc = (preds == cls) & valid
        lc = (labels == cls) & valid
        tp[cls] += (pc & lc).sum()
        fp[cls] += (pc & ~lc).sum()
        fn[cls] += (~pc & lc).sum()
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return 2 * prec * rec / (prec + rec + 1e-6)
