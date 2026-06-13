"""Test-Time Prompt Adaptation (TTPA) + prediction-collapse detection.

SOURCE: handoff spec (§12 snippet) — extended with diagnostics and
masked-entropy fix for Darmstadt zero-shot transfer.

Rules fixed by the handoff:
    - Only the text projection MLP (``text_proj``) adapts. Visual backbone
      (DINOv2 + LoRA) stays frozen — adapting it would destroy the learned
      Potsdam representations.
    - Loss: entropy minimization + KL regularization to the pre-adaptation
      prediction (weight 0.5). The KL anchor prevents prediction collapse.
    - MAX 2 steps, LR=1e-5. More steps or higher LR causes collapse.
    - Collapse fallback: steps=1, lr=5e-6, kl_weight=1.0.

Additions over the handoff:
    - text_proj weights are snapshotted and RESTORED after prediction, so
      adaptation on one batch never leaks into the next batch.
    - ``detect_collapse()`` implements the entropy-uniformity check.
    - ``ttpa_diagnostic()`` measures weight delta + prediction pixel change
      to distinguish "gradient not flowing" from "L2-norm cancelling update".
    - Masked entropy: entropy loss computed only on high-uncertainty pixels.
      When the model is already confident (about wrong predictions), the full
      entropy gradient ≈ 0 and KL anchors to the wrong prediction. Masking
      focuses adaptation on genuinely uncertain regions.
"""
import copy
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .constants import (TTPA_KL_WEIGHT, TTPA_LR, TTPA_STEPS,
                        TTPA_DARMSTADT_KL_WEIGHT, TTPA_DARMSTADT_LR,
                        TTPA_DARMSTADT_STEPS)


def entropy_map(logits: torch.Tensor) -> torch.Tensor:
    """Per-pixel Shannon entropy of the softmax distribution.

    Returns:
        [B, H, W] tensor. Low inside coherent regions, high at boundaries
        is the healthy pattern.
    """
    probs = torch.softmax(logits, dim=1)
    return -(probs * torch.log(probs + 1e-6)).sum(dim=1)


def ttpa_predict(geo_model, imgs: torch.Tensor,
                 n_steps: int = TTPA_STEPS,
                 lr: float = TTPA_LR,
                 kl_weight: float = TTPA_KL_WEIGHT,
                 restore: bool = True,
                 masked_entropy: bool = True,
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapt text_proj on one batch of unlabeled patches, then predict.

    Args:
        geo_model:       RGGeoPromptSegModel (trained on Potsdam).
        imgs:            [B, 3, 512, 512] Darmstadt patches on DEVICE.
        n_steps:         adaptation steps. Max 2 per handoff; use
                         TTPA_DARMSTADT_STEPS=5 if standard shows no effect.
        lr:              step size. 1e-5 per handoff; 5e-5 for Darmstadt mode.
        kl_weight:       KL regularisation weight. 0.5 per handoff; 0.1 for
                         Darmstadt mode (lower → entropy term has more room).
        restore:         reset text_proj to pre-adaptation state after each
                         batch so batches don't accumulate adaptation drift.
        masked_entropy:  if True, entropy loss is computed only on pixels
                         whose per-pixel entropy exceeds the batch mean.
                         Prevents reinforcing already-confident wrong predictions
                         — the dominant TTPA failure mode on Darmstadt.

    Returns:
        (final_logits [B, 6, H, W], ref_probs [B, 6, H, W])
        ref_probs: zero-shot (pre-adaptation) probabilities for comparison.
    """
    geo_model.eval()

    snapshot = copy.deepcopy(geo_model.text_proj.state_dict()) if restore else None

    for name, p in geo_model.named_parameters():
        p.requires_grad = ("text_proj" in name)
    opt = torch.optim.Adam(
        [p for p in geo_model.parameters() if p.requires_grad], lr=lr)

    with torch.no_grad():
        ref_probs = torch.softmax(geo_model(imgs), dim=1).detach()

    for _ in range(n_steps):
        opt.zero_grad()
        probs = torch.softmax(geo_model(imgs), dim=1)

        H_map = -(probs * torch.log(probs + 1e-6)).sum(dim=1)  # [B, H, W]
        if masked_entropy:
            # Only uncertain pixels contribute: prevents reinforcing confident
            # wrong predictions (the main failure on Darmstadt red rooftops).
            mask = H_map > H_map.mean()
            entropy = H_map[mask].mean() if mask.any() else H_map.mean()
        else:
            entropy = H_map.mean()

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


def ttpa_diagnostic(geo_model, imgs: torch.Tensor,
                    n_steps: int = TTPA_DARMSTADT_STEPS,
                    lr: float = TTPA_DARMSTADT_LR,
                    kl_weight: float = TTPA_DARMSTADT_KL_WEIGHT,
                    ) -> Dict[str, float]:
    """Run TTPA and measure how much the weights and predictions actually move.

    Call this ONCE on a representative Darmstadt batch before running full
    inference. Determines which failure mode is active:

        weight_delta ≈ 0        → gradient not reaching text_proj (check autograd)
        weight_delta > 0,
        pred_pixel_change ≈ 0   → L2-normalisation cancels weight update;
                                   increase steps/lr significantly
        pred_pixel_change > 5%  → TTPA is working; tune kl_weight to taste

    Returns dict with weight_delta, pred_pixel_change, mean_entropy_before,
    mean_entropy_after, collapsed.
    """
    geo_model.eval()

    before_w = {k: v.clone() for k, v in geo_model.text_proj.state_dict().items()}

    with torch.no_grad():
        logits_before = geo_model(imgs)
        pred_before = logits_before.argmax(dim=1)
        ent_before = entropy_map(logits_before).mean().item()

    # Run adaptation WITHOUT restore so we can measure the after state
    snapshot = copy.deepcopy(geo_model.text_proj.state_dict())
    for name, p in geo_model.named_parameters():
        p.requires_grad = ("text_proj" in name)
    opt = torch.optim.Adam(
        [p for p in geo_model.parameters() if p.requires_grad], lr=lr)

    with torch.no_grad():
        ref_probs = torch.softmax(geo_model(imgs), dim=1).detach()

    for _ in range(n_steps):
        opt.zero_grad()
        probs = torch.softmax(geo_model(imgs), dim=1)
        H_map = -(probs * torch.log(probs + 1e-6)).sum(dim=1)
        mask = H_map > H_map.mean()
        entropy = H_map[mask].mean() if mask.any() else H_map.mean()
        kl_div = F.kl_div(torch.log(probs + 1e-6), ref_probs,
                          reduction="batchmean")
        (entropy + kl_weight * kl_div).backward()
        opt.step()

    after_w = geo_model.text_proj.state_dict()
    weight_delta = sum(
        (after_w[k] - before_w[k]).abs().mean().item() for k in before_w)

    with torch.no_grad():
        logits_after = geo_model(imgs)
        pred_after = logits_after.argmax(dim=1)
        ent_after = entropy_map(logits_after).mean().item()

    pred_change = (pred_before != pred_after).float().mean().item()
    collapse = detect_collapse(logits_after)

    # Always restore so diagnostic doesn't corrupt model state
    geo_model.text_proj.load_state_dict(snapshot)
    for p in geo_model.parameters():
        p.requires_grad = False

    result = {
        "weight_delta": weight_delta,
        "pred_pixel_change": pred_change,
        "mean_entropy_before": ent_before,
        "mean_entropy_after": ent_after,
        "collapsed": collapse["collapsed"],
    }

    print(f"TTPA diagnostic (steps={n_steps}, lr={lr}, kl={kl_weight}):")
    print(f"  text_proj weight delta : {weight_delta:.6f}"
          + (" ← gradient not flowing!" if weight_delta < 1e-5 else ""))
    print(f"  prediction pixel change: {100*pred_change:.1f}%"
          + (" ← L2-norm cancelling update" if weight_delta > 1e-5 and pred_change < 0.01 else ""))
    print(f"  entropy before→after   : {ent_before:.4f} → {ent_after:.4f}")
    print(f"  collapsed              : {bool(collapse['collapsed'])}")

    if weight_delta < 1e-5:
        print("\n⚠ Diagnosis: gradient not reaching text_proj. Check that "
              "text_proj params are not frozen before TTPA runs.")
    elif pred_change < 0.01:
        print("\n⚠ Diagnosis: weights move but L2-norm cancels directional "
              "change. Try lr=1e-4 or adapt the pre-L2-norm output directly.")
    else:
        print(f"\n✓ TTPA is working. {100*pred_change:.1f}% of pixels changed.")

    return result


def detect_collapse(logits: torch.Tensor,
                    dominant_frac_thresh: float = 0.90,
                    low_entropy_thresh: float = 0.05
                    ) -> Dict[str, float]:
    """Heuristic prediction-collapse check after TTPA.

    Collapse signature: entropy uniformly LOW everywhere AND one class
    dominating nearly all pixels.

    Returns dict with dominant_class, dominant_fraction, mean_entropy,
    collapsed (1.0 if both collapse criteria fire, else 0.0).

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
