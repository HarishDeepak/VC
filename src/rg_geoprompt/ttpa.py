"""Test-Time Prompt Adaptation (TTPA) + prediction-collapse detection.

SOURCE: handoff spec (§12 snippet) — not yet run on Kaggle.

Rules fixed by the handoff:
    - Only the text projection MLP (``text_proj``) adapts. Visual backbone
      (DINOv2 + LoRA) stays frozen — adapting it would destroy the learned
      Potsdam representations.
    - Loss: entropy minimization + KL regularization to the pre-adaptation
      prediction (weight 0.5). The KL anchor prevents prediction collapse.
    - MAX 2 steps, LR=1e-5. More steps or higher LR causes collapse.
    - Collapse fallback: steps=1, lr=5e-6, kl_weight=1.0.

Additions over the handoff snippet:
    - text_proj weights are snapshotted and RESTORED after prediction, so
      adaptation on one batch never leaks into the next batch or into any
      later training. (The original snippet left adapted weights in place
      and set requires_grad=False on ALL params at the end.)
    - ``detect_collapse()`` implements the entropy-uniformity check.
"""
import copy
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .constants import TTPA_KL_WEIGHT, TTPA_LR, TTPA_STEPS


def ttpa_predict(geo_model, imgs: torch.Tensor,
                 n_steps: int = TTPA_STEPS,
                 lr: float = TTPA_LR,
                 kl_weight: float = TTPA_KL_WEIGHT,
                 restore: bool = True
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapt text_proj on one batch of unlabeled patches, then predict.

    Args:
        geo_model: RGGeoPromptSegModel (trained on Potsdam).
        imgs:      [B, 3, 512, 512] Darmstadt patches, already on DEVICE.
        n_steps:   max 2 (handoff). More steps → prediction collapse.
        lr:        1e-5 (handoff). NOT 1e-4.
        kl_weight: 0.5 (handoff). Raise to 1.0 if collapse is detected.
        restore:   if True (default), text_proj is reset to its
                   pre-adaptation state after the final prediction —
                   each batch adapts independently.

    Returns:
        (final_logits [B, 6, H, W], ref_probs [B, 6, H, W])
        ref_probs are the zero-shot (pre-adaptation) probabilities, kept
        for before/after comparison figures.
    """
    geo_model.eval()

    # snapshot text_proj for restoration
    snapshot = copy.deepcopy(geo_model.text_proj.state_dict()) if restore else None

    # only text_proj adapts — backbone, LoRA, tau stay frozen
    for name, p in geo_model.named_parameters():
        p.requires_grad = ("text_proj" in name)
    opt = torch.optim.Adam(
        [p for p in geo_model.parameters() if p.requires_grad], lr=lr)

    with torch.no_grad():
        ref_probs = torch.softmax(geo_model(imgs), dim=1).detach()

    for _ in range(n_steps):
        opt.zero_grad()
        probs = torch.softmax(geo_model(imgs), dim=1)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=1).mean()
        kl_div = F.kl_div(torch.log(probs + 1e-6), ref_probs,
                          reduction="batchmean")
        (entropy + kl_weight * kl_div).backward()
        opt.step()

    with torch.no_grad():
        final = geo_model(imgs)

    if restore:
        geo_model.text_proj.load_state_dict(snapshot)
    for p in geo_model.parameters():
        p.requires_grad = False

    return final, ref_probs


def entropy_map(logits: torch.Tensor) -> torch.Tensor:
    """Per-pixel Shannon entropy of the softmax distribution.

    Returns:
        [B, H, W] tensor. Low inside coherent regions, high at boundaries
        is the healthy pattern.
    """
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-6)).sum(dim=1)


def detect_collapse(logits: torch.Tensor,
                    dominant_frac_thresh: float = 0.90,
                    low_entropy_thresh: float = 0.05
                    ) -> Dict[str, float]:
    """Heuristic prediction-collapse check after TTPA.

    Collapse signature (handoff): entropy uniformly LOW everywhere AND one
    class dominating nearly all pixels.

    Returns dict with:
        dominant_class       — argmax class covering the most pixels
        dominant_fraction    — fraction of pixels assigned to it
        mean_entropy         — batch mean per-pixel entropy
        collapsed            — 1.0 if both collapse criteria fire, else 0.0

    If collapsed: rerun ttpa_predict with n_steps=1, lr=5e-6, kl_weight=1.0.
    """
    preds = logits.argmax(dim=1)
    counts = torch.bincount(preds.flatten(), minlength=logits.shape[1]).float()
    dominant = int(counts.argmax().item())
    frac = (counts[dominant] / counts.sum()).item()
    ent = entropy_map(logits).mean().item()
    collapsed = float(frac >= dominant_frac_thresh and ent <= low_entropy_thresh)
    return {"dominant_class": dominant, "dominant_fraction": frac,
            "mean_entropy": ent, "collapsed": collapsed}
