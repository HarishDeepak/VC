"""DINOv2-base + LoRA backbone and the conv-head segmentation model.

SOURCE: notebook CELL-R (session recovery) — the CONFIRMED-correct version.

⚠ Why CELL-R and not CELL16: the exported CELL16 builds a 5-class head
(NUM_CLASSES=5 in that cell). The trained 84.6% checkpoint is 6-class —
CELL-R rebuilds with 6 outputs and loads it strict=True successfully, and
a 5-class head would crash cross_entropy on clutter pixels (label 5 with
only 5 logits). num_classes is therefore PINNED to 6 here.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (DEVICE, DINOV2_DIM, DINOV2_NAME, DINOV2_PATCH,
                        LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS,
                        NUM_CLASSES_TOTAL)


def build_lora_backbone(device: torch.device = DEVICE):
    """DINOv2-base wrapped with LoRA adapters (r=16, Q+V only).

    The base weights are frozen by peft (98.65% of backbone params);
    only the LoRA A/B matrices remain trainable (~589K params).
    """
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    backbone = AutoModel.from_pretrained(DINOV2_NAME)
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                          target_modules=LORA_TARGETS,
                          lora_dropout=LORA_DROPOUT, bias="none")
    return get_peft_model(backbone, lora_cfg).to(device)


class DINOv2SegModel(nn.Module):
    """LoRA-adapted DINOv2 encoder + lightweight conv decode head.

    Input 512×512, patch 14 → 36×36 = 1296 patch tokens (CLS dropped).
    Output: logits [B, 6, 512, 512] — ALWAYS 6 channels to match the
    trained checkpoint and to keep clutter (label 5) in-range for the loss.
    """

    def __init__(self, backbone, num_classes: int = NUM_CLASSES_TOTAL):
        super().__init__()
        assert num_classes == NUM_CLASSES_TOTAL, (
            "Output head must be 6-class (incl. clutter) to match the "
            "trained checkpoint — see module docstring.")
        self.backbone = backbone
        self.patch_size = DINOV2_PATCH
        self.hidden_dim = DINOV2_DIM
        self.decode_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pixel_values.shape
        out = self.backbone(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, 1:, :]          # drop CLS
        h, w = H // self.patch_size, W // self.patch_size
        feat = tokens[:, :h * w, :].permute(0, 2, 1)
        feat = feat.reshape(B, self.hidden_dim, h, w)      # [B, 768, 36, 36]
        logits = self.decode_head(feat)                    # [B, 6, 36, 36]
        return F.interpolate(logits, size=(H, W),
                             mode="bilinear", align_corners=False)


def strip_compile_prefix(state: dict) -> dict:
    """Remove torch.compile's '_orig_mod.' key prefix from a state dict."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def load_dinov2_lora(ckpt_path: Path, device: torch.device = DEVICE
                     ) -> DINOv2SegModel:
    """Rebuild DINOv2SegModel and load the trained checkpoint (strict).

    Use ``utils.ensure_checkpoint()`` first if the file may only exist on
    the HuggingFace backup repo.
    """
    model = DINOv2SegModel(build_lora_backbone(device)).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = strip_compile_prefix(state)
    model.load_state_dict(state, strict=True)
    return model
