"""RG-GeoPrompt-PEFT model: DINOv2+LoRA visual stream + CLIP cosine head.

SOURCE: notebook CELL28 — implemented but the model has NOT yet been
trained to completion (GeoPrompt result is TBD). Architecture is fixed by
the handoff:

    - DINOv2 backbone frozen + LoRA r=16 on Q,V (trainable)
    - CLIP ViT-B/32 text encoder fully frozen (prototypes precomputed)
    - text projection: 2-layer MLP 512→768→768 with GELU, L2-normalized out
    - cosine similarity head between L2-normalized visual features and
      L2-normalized projected text prototypes
    - learnable temperature τ, init 0.07, clamped ≥ 0.01

Memory note (NOT yet verified on T4): the spec upsamples 768-dim features
to full 512×512 BEFORE the cosine product → ~800MB activations per image
plus the autograd graph. If training OOMs, set ``lowres_similarity=True``:
cosine similarity is then computed at 36×36 token resolution and the
6-channel logits are upsampled instead. This is mathematically the same
classifier applied before vs after interpolation — predictions differ only
through interpolation order — and cuts activation memory by ~200×. Default
behavior follows the handoff spec exactly.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (DEVICE, DINOV2_DIM, DINOV2_PATCH,
                        NUM_CLASSES_TOTAL, TAU_INIT, TAU_MIN)
from .models_dino_lora import build_lora_backbone, strip_compile_prefix


class RGGeoPromptSegModel(nn.Module):
    """Open-vocabulary segmentation via cosine similarity to text prototypes.

    Classification IS the dot product between visual features and projected
    text prototypes — change the text, change the predictions. No conv head.
    """

    def __init__(self, backbone, text_embeddings: torch.Tensor,
                 num_classes: int = NUM_CLASSES_TOTAL,
                 lowres_similarity: bool = False):
        super().__init__()
        assert text_embeddings.shape == (num_classes, 512), (
            f"expected [{num_classes}, 512] CLIP prototypes, "
            f"got {tuple(text_embeddings.shape)}")
        self.backbone = backbone
        self.patch_size = DINOV2_PATCH
        self.hidden_dim = DINOV2_DIM
        self.num_classes = num_classes
        self.lowres_similarity = lowres_similarity

        # Frozen CLIP text prototypes [6, 512] — buffer, never trained
        self.register_buffer("text_emb", text_embeddings.float())

        # 2-layer MLP: CLIP (512) → DINOv2 (768). Linear is insufficient —
        # the two embedding spaces have different geometry (Talk2DINO).
        self.text_proj = nn.Sequential(
            nn.Linear(512, 768),
            nn.GELU(),
            nn.Linear(768, 768),
        )

        # Learnable temperature, initialized at CLIP's training value
        self.tau = nn.Parameter(torch.ones(1) * TAU_INIT)

    def project_text(self) -> torch.Tensor:
        """Projected, L2-normalized text prototypes [6, 768]."""
        return F.normalize(self.text_proj(self.text_emb), dim=-1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pixel_values.shape

        # Visual stream
        out = self.backbone(pixel_values=pixel_values)
        tokens = out.last_hidden_state[:, 1:, :]               # drop CLS
        h, w = H // self.patch_size, W // self.patch_size
        feat = tokens[:, :h * w, :].permute(0, 2, 1)
        feat = feat.reshape(B, self.hidden_dim, h, w)          # [B,768,36,36]

        text_norm = self.project_text()                        # [6, 768]
        tau = self.tau.clamp(min=TAU_MIN)

        if self.lowres_similarity:
            # OOM fallback: similarity at token resolution, upsample logits
            feat_norm = F.normalize(feat, dim=1)
            logits = torch.einsum("bchw,kc->bkhw", feat_norm, text_norm) / tau
            return F.interpolate(logits, size=(H, W),
                                 mode="bilinear", align_corners=False)

        # Spec behavior: upsample features first, then cosine similarity
        feat = F.interpolate(feat, size=(H, W),
                             mode="bilinear", align_corners=False)
        feat_norm = F.normalize(feat, dim=1)                   # [B,768,H,W]
        feat_flat = feat_norm.permute(0, 2, 3, 1)              # [B,H,W,768]
        logits = feat_flat @ text_norm.T                       # [B,H,W,6]
        return logits.permute(0, 3, 1, 2) / tau                # [B,6,H,W]


def build_geoprompt_from_dinov2_ckpt(text_embeddings: torch.Tensor,
                                     dinov2_ckpt: Path,
                                     device: torch.device = DEVICE,
                                     lowres_similarity: bool = False
                                     ) -> RGGeoPromptSegModel:
    """Build RGGeoPromptSegModel initialized from the DINOv2+LoRA checkpoint.

    Loads backbone (+LoRA) weights only, strict=False — the old conv
    decode_head is intentionally dropped; text_proj and τ start fresh.
    """
    backbone = build_lora_backbone(device)
    model = RGGeoPromptSegModel(backbone, text_embeddings,
                                lowres_similarity=lowres_similarity).to(device)
    state = torch.load(dinov2_ckpt, map_location=device)
    state = strip_compile_prefix(state)
    backbone_only = {k: v for k, v in state.items() if "decode_head" not in k}
    missing, unexpected = model.load_state_dict(backbone_only, strict=False)
    print(f"✓ backbone loaded into GeoPrompt "
          f"(missing={len(missing)} new layers, unexpected={len(unexpected)})")
    return model


def load_geoprompt(ckpt_path: Path, text_embeddings: torch.Tensor,
                   device: torch.device = DEVICE,
                   lowres_similarity: bool = False) -> RGGeoPromptSegModel:
    """Rebuild RGGeoPromptSegModel and load a trained GeoPrompt checkpoint."""
    backbone = build_lora_backbone(device)
    model = RGGeoPromptSegModel(backbone, text_embeddings,
                                lowres_similarity=lowres_similarity).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = strip_compile_prefix(state)
    model.load_state_dict(state, strict=True)
    return model
