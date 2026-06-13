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
                        DINOV2_REG_NAME, DINOV2_REG_TOKENS, DINOV2_STRIDE,
                        LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGETS,
                        NUM_CLASSES_TOTAL)


def _apply_stride_patch(backbone, patch_size: int, stride: int) -> None:
    """Monkey-patch DINOv2 PatchEmbed and fix pos-encoding interpolation.

    HF's interpolate_pos_encodings calculates the output grid as
    ``height // patch_size``, which is wrong when stride != patch_size.
    We replace it with the correct formula: (H - patch_size) // stride + 1.
    """
    import math

    # backbone (PeftModel) → base_model (LoraModel) → model (Dinov2Model)
    dinov2 = backbone.base_model.model
    dinov2.embeddings.patch_embeddings.projection.stride = (stride, stride)

    pos_emb_module = dinov2.embeddings

    def fixed_interpolate(embeddings_input: torch.Tensor,
                          height: int, width: int) -> torch.Tensor:
        num_patches = embeddings_input.shape[1] - 1  # exclude CLS
        num_positions = pos_emb_module.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return pos_emb_module.position_embeddings
        cls_pe = pos_emb_module.position_embeddings[:, 0]
        patch_pe = pos_emb_module.position_embeddings[:, 1:]
        dim = embeddings_input.shape[-1]
        # Actual output grid for this stride (not height // patch_size)
        h_out = (height - patch_size) // stride + 1
        w_out = (width - patch_size) // stride + 1
        orig = int(math.sqrt(num_positions))
        patch_pe = F.interpolate(
            patch_pe.reshape(1, orig, orig, dim).permute(0, 3, 1, 2),
            scale_factor=(h_out / orig + 1e-4, w_out / orig + 1e-4),
            mode="bicubic", align_corners=False,
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((cls_pe.unsqueeze(0), patch_pe), dim=1)

    pos_emb_module.interpolate_pos_encodings = fixed_interpolate


def build_lora_backbone(device: torch.device = DEVICE,
                        use_registers: bool = False,
                        patch_stride: int = DINOV2_STRIDE):
    """DINOv2-base wrapped with LoRA adapters (r=16, Q+V only).

    Args:
        use_registers: load the register-token variant (DINOV2_REG_NAME).
                       Eliminates high-norm artifact tokens in homogeneous
                       regions. Verify the HF model ID before enabling.
        patch_stride:  stride of the PatchEmbed conv. Default 14 = standard
                       non-overlapping patches (36×36 grid for 512×512 input).
                       Set to 7 for overlapping patches (72×72 grid). Requires
                       AMP + gradient checkpointing in the training loop.
    """
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model

    model_name = DINOV2_REG_NAME if use_registers else DINOV2_NAME
    backbone = AutoModel.from_pretrained(model_name)
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                          target_modules=LORA_TARGETS,
                          lora_dropout=LORA_DROPOUT, bias="none")
    backbone = get_peft_model(backbone, lora_cfg).to(device)

    if patch_stride != DINOV2_PATCH:
        _apply_stride_patch(backbone, DINOV2_PATCH, patch_stride)
        print(f"✓ stride patched {DINOV2_PATCH}→{patch_stride} "
              f"(grid: {512//patch_stride if patch_stride==7 else 36}×…)")

    return backbone


class DINOv2SegModel(nn.Module):
    """LoRA-adapted DINOv2 encoder + lightweight conv decode head.

    Default: patch 14, stride 14 → 36×36 tokens. With stride=7 → 72×72.
    Output: logits [B, 6, 512, 512] — ALWAYS 6 channels.
    """

    def __init__(self, backbone, num_classes: int = NUM_CLASSES_TOTAL,
                 patch_stride: int = DINOV2_STRIDE,
                 n_reg_tokens: int = 0):
        super().__init__()
        assert num_classes == NUM_CLASSES_TOTAL, (
            "Output head must be 6-class (incl. clutter) to match the "
            "trained checkpoint — see module docstring.")
        self.backbone = backbone
        self.patch_size = DINOV2_PATCH
        self.patch_stride = patch_stride
        self.n_reg_tokens = n_reg_tokens
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
        # Skip CLS + register tokens; patch tokens follow
        tokens = out.last_hidden_state[:, 1 + self.n_reg_tokens:, :]
        h = (H - self.patch_size) // self.patch_stride + 1
        w = (W - self.patch_size) // self.patch_stride + 1
        feat = tokens[:, :h * w, :].permute(0, 2, 1)
        feat = feat.reshape(B, self.hidden_dim, h, w)
        logits = self.decode_head(feat)
        return F.interpolate(logits, size=(H, W),
                             mode="bilinear", align_corners=False)


def strip_compile_prefix(state: dict) -> dict:
    """Remove torch.compile's '_orig_mod.' key prefix from a state dict."""
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def load_dinov2_lora(ckpt_path: Path, device: torch.device = DEVICE,
                     use_registers: bool = False,
                     patch_stride: int = DINOV2_STRIDE) -> DINOv2SegModel:
    """Rebuild DINOv2SegModel and load the trained checkpoint (strict).

    Use ``utils.ensure_checkpoint()`` first if the file may only exist on
    the HuggingFace backup repo. Pass use_registers/patch_stride matching
    the checkpoint's training config.
    """
    n_reg = DINOV2_REG_TOKENS if use_registers else 0
    backbone = build_lora_backbone(device, use_registers=use_registers,
                                   patch_stride=patch_stride)
    model = DINOv2SegModel(backbone, patch_stride=patch_stride,
                           n_reg_tokens=n_reg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = strip_compile_prefix(state)
    model.load_state_dict(state, strict=True)
    return model
