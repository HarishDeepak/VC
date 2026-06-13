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

from .constants import (DEVICE, DINOV2_DIM, DINOV2_PATCH, DINOV2_STRIDE,
                        DINOV2_REG_TOKENS, NUM_CLASSES_TOTAL, TAU_INIT, TAU_MIN)
from .models_dino_lora import build_lora_backbone, strip_compile_prefix


class RGGeoPromptSegModel(nn.Module):
    """Open-vocabulary segmentation via cosine similarity to text prototypes.

    Classification IS the dot product between visual features and projected
    text prototypes — change the text, change the predictions. No conv head.
    """

    def __init__(self, backbone, text_embeddings: torch.Tensor,
                 num_classes: int = NUM_CLASSES_TOTAL,
                 lowres_similarity: bool = False,
                 patch_stride: int = DINOV2_STRIDE,
                 n_reg_tokens: int = 0):
        super().__init__()
        assert text_embeddings.shape == (num_classes, 512), (
            f"expected [{num_classes}, 512] CLIP prototypes, "
            f"got {tuple(text_embeddings.shape)}")
        self.backbone = backbone
        self.patch_size = DINOV2_PATCH
        self.patch_stride = patch_stride
        self.n_reg_tokens = n_reg_tokens
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

    def set_darmstadt_mode(self, multi_emb: torch.Tensor,
                           ortho_scale: float = 0.3) -> None:
        """Switch to Darmstadt inference: max-pool over per-prompt embeddings.

        Args:
            multi_emb:   [6, N, 512] per-prompt CLIP embeddings (not averaged).
                         Produced by prompts.encode_per_prompt(DARMSTADT_PROMPTS).
            ortho_scale: Penalty weight subtracting the clutter score from the
                         building score (0 = disabled). Gemini recommendation: 0.2–0.5.
        """
        assert multi_emb.ndim == 3 and multi_emb.shape[0] == 6 and multi_emb.shape[2] == 512
        device = self.text_emb.device
        self.register_buffer("text_emb_multi", multi_emb.float().to(device))
        self._ortho_scale = float(ortho_scale)

    def _project_text_multi(self) -> torch.Tensor:
        """Project per-prompt embeddings [6, N, 512] → [6*N, 768] normalized."""
        K, N, D = self.text_emb_multi.shape
        flat = self.text_emb_multi.reshape(K * N, D)     # [6*N, 512]
        return F.normalize(self.text_proj(flat), dim=-1)  # [6*N, 768]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pixel_values.shape

        # Visual stream
        out = self.backbone(pixel_values=pixel_values)
        # Skip CLS + register tokens (n_reg_tokens=0 for standard DINOv2)
        tokens = out.last_hidden_state[:, 1 + self.n_reg_tokens:, :]
        h = (H - self.patch_size) // self.patch_stride + 1
        w = (W - self.patch_size) // self.patch_stride + 1
        assert tokens.shape[1] >= h * w, (
            f"Expected >= {h*w} tokens from DINOv2 for {H}x{W} input "
            f"(patch {self.patch_size}, stride {self.patch_stride}), "
            f"got {tokens.shape[1]}.")
        feat = tokens[:, :h * w, :].permute(0, 2, 1)
        feat = feat.reshape(B, self.hidden_dim, h, w)          # [B,768,h,w]

        tau = self.tau.clamp(min=TAU_MIN)
        darmstadt = hasattr(self, "text_emb_multi")

        if darmstadt:
            K = self.num_classes
            N = self.text_emb_multi.shape[1]
            text_flat = self._project_text_multi()             # [6*N, 768]

            if self.lowres_similarity:
                feat_norm = F.normalize(feat, dim=1)           # [B,768,h,w]
                sim = torch.einsum("bchw,kc->bkhw",
                                   feat_norm, text_flat) / tau # [B,6*N,h,w]
                sim = sim.reshape(B, K, N, h, w)
                logits = sim.max(dim=2).values                 # [B,6,h,w]
                logits = F.interpolate(logits, size=(H, W),
                                       mode="bilinear", align_corners=False)
            else:
                feat = F.interpolate(feat, size=(H, W),
                                     mode="bilinear", align_corners=False)
                feat_norm = F.normalize(feat, dim=1)           # [B,768,H,W]
                feat_flat = feat_norm.permute(0, 2, 3, 1)      # [B,H,W,768]
                sim = (feat_flat @ text_flat.T) / tau          # [B,H,W,6*N]
                sim = sim.reshape(B, H, W, K, N)
                logits = sim.max(dim=4).values                 # [B,H,W,6]
                logits = logits.permute(0, 3, 1, 2)            # [B,6,H,W]

            # Orthogonal projection: penalise building score by clutter score.
            # Prevents red terracotta roofs falling to clutter class.
            scale = getattr(self, "_ortho_scale", 0.0)
            if scale > 0:
                logits = logits.clone()
                logits[:, 1] = logits[:, 1] - scale * logits[:, 5]
            return logits

        # ── Standard (training) path ──────────────────────────────────────
        text_norm = self.project_text()                        # [6, 768]

        if self.lowres_similarity:
            feat_norm = F.normalize(feat, dim=1)
            logits = torch.einsum("bchw,kc->bkhw", feat_norm, text_norm) / tau
            return F.interpolate(logits, size=(H, W),
                                 mode="bilinear", align_corners=False)

        feat = F.interpolate(feat, size=(H, W),
                             mode="bilinear", align_corners=False)
        feat_norm = F.normalize(feat, dim=1)                   # [B,768,H,W]
        feat_flat = feat_norm.permute(0, 2, 3, 1)              # [B,H,W,768]
        logits = feat_flat @ text_norm.T                       # [B,H,W,6]
        return logits.permute(0, 3, 1, 2) / tau                # [B,6,H,W]


def build_geoprompt_from_dinov2_ckpt(text_embeddings: torch.Tensor,
                                     dinov2_ckpt: Path,
                                     device: torch.device = DEVICE,
                                     lowres_similarity: bool = False,
                                     use_registers: bool = False,
                                     patch_stride: int = DINOV2_STRIDE,
                                     ) -> RGGeoPromptSegModel:
    """Build RGGeoPromptSegModel initialized from the DINOv2+LoRA checkpoint.

    Loads backbone (+LoRA) weights only, strict=False — the old conv
    decode_head is intentionally dropped; text_proj and τ start fresh.
    """
    n_reg = DINOV2_REG_TOKENS if use_registers else 0
    backbone = build_lora_backbone(device, use_registers=use_registers,
                                   patch_stride=patch_stride)
    model = RGGeoPromptSegModel(backbone, text_embeddings,
                                lowres_similarity=lowres_similarity,
                                patch_stride=patch_stride,
                                n_reg_tokens=n_reg).to(device)
    state = torch.load(dinov2_ckpt, map_location=device)
    state = strip_compile_prefix(state)
    backbone_only = {k: v for k, v in state.items() if "decode_head" not in k}
    missing, unexpected = model.load_state_dict(backbone_only, strict=False)
    print(f"✓ backbone loaded into GeoPrompt "
          f"(missing={len(missing)} new layers, unexpected={len(unexpected)})")
    return model


def load_geoprompt(ckpt_path: Path, text_embeddings: torch.Tensor,
                   device: torch.device = DEVICE,
                   lowres_similarity: bool = False,
                   use_registers: bool = False,
                   patch_stride: int = DINOV2_STRIDE) -> RGGeoPromptSegModel:
    """Rebuild RGGeoPromptSegModel and load a trained GeoPrompt checkpoint."""
    n_reg = DINOV2_REG_TOKENS if use_registers else 0
    backbone = build_lora_backbone(device, use_registers=use_registers,
                                   patch_stride=patch_stride)
    model = RGGeoPromptSegModel(backbone, text_embeddings,
                                lowres_similarity=lowres_similarity,
                                patch_stride=patch_stride,
                                n_reg_tokens=n_reg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state = strip_compile_prefix(state)
    model.load_state_dict(state, strict=True)
    return model
