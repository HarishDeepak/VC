"""Training augmentations.

Spatial transforms apply identically to image AND label.
Photometric transforms (jitter, resolution bridge) apply to the image ONLY.
The label mask is NEVER blurred, resized or jittered.

SOURCE: notebook CELL6 — confirmed working (DINOv2+LoRA trained with this).
"""
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .constants import RESOLUTION_BRIDGE_PROB, RESOLUTION_FACTOR

_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)


def resolution_bridge(img: torch.Tensor,
                      factor: int = RESOLUTION_FACTOR) -> torch.Tensor:
    """Simulate lower-GSD imagery (e.g. 20cm DOP20) on a high-res patch.

    Downsample by ``factor`` then bilinearly upsample back to the original
    size. The result keeps the spatial dimensions but loses high-frequency
    detail — the appearance the model will face on Darmstadt DOP20.

    Args:
        img:    float image tensor [3, H, W] in [0, 1].
        factor: 2 if source data is 10cm GSD, 4 if 5cm GSD.
                Verify with rasterio before trusting the default.

    Returns:
        Blurred image tensor [3, H, W]. The label is intentionally NOT
        touched here — semantic content does not change with resolution.
    """
    H, W = img.shape[1], img.shape[2]
    small = F.interpolate(img.unsqueeze(0), scale_factor=1.0 / factor,
                          mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(H, W),
                         mode="bilinear", align_corners=False).squeeze(0)


def train_augment(img: torch.Tensor, lbl: torch.Tensor):
    """Full training augmentation: flips, 90° rotations, color jitter,
    and (30% of the time) the resolution bridge.

    Args:
        img: float tensor [3, H, W] in [0, 1]
        lbl: int64 tensor [H, W] with values in {0..5, 255}

    Returns:
        (img, lbl) — augmented pair. lbl only undergoes spatial transforms.
    """
    if random.random() > 0.5:
        img, lbl = TF.hflip(img), TF.hflip(lbl)
    if random.random() > 0.5:
        img, lbl = TF.vflip(img), TF.vflip(lbl)
    k = random.randint(0, 3)
    if k > 0:
        img = torch.rot90(img, k, dims=[1, 2])
        lbl = torch.rot90(lbl, k, dims=[0, 1])

    img = _jitter(img)                       # image only — never the label

    if random.random() < RESOLUTION_BRIDGE_PROB:
        img = resolution_bridge(img)         # image only — never the label

    return img, lbl
