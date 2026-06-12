# CLAUDE.md — RG-GeoPrompt-PEFT

Read this first at session start. It is the ground truth; do not contradict it.

## What this project is
Open-vocabulary remote sensing segmentation (Fraunhofer IGD Praktikum, SoSe 2026).
Train on ISPRS Potsdam (labeled, 5–10cm GSD) → zero-shot transfer to Darmstadt
DOP20 (unlabeled, 20cm GSD). Runs on Kaggle T4 (16GB). Method: DINOv2-base
(frozen) + LoRA r=16 on Q,V + frozen CLIP ViT-B/32 text prototypes + 2-layer MLP
text projection (512→768→768, GELU) + cosine similarity head + learnable τ
(init 0.07, clamp ≥0.01) + resolution-bridge augmentation + TTPA at inference.

## Confirmed results — NEVER retrain these
| Model | Trainable | mIoU (5 cls) |
|---|---|---|
| SegFormer-B0 | 3.7M (100%) | 82.1% |
| DINOv2+LoRA r=16 | 788K (0.9%) | 84.6% |
| RG-GeoPrompt-PEFT | ~1.2M (~1.35%) | TBD |

Published OVRSIS numbers (TPOVSeg 38–44%, SegEarth-OV ~47%) are ZERO-SHOT
transfer to Potsdam — a different task. Never mix them with our supervised
numbers in one table.

## Non-negotiable rules
1. **Metrics — never swap or merge:**
   - mIoU: 5 classes (0–4), `ignore_ids=(5, 255)` — clutter AND boundary out.
   - F1: 6 classes (0–5), `ignore_ids=(255,)` — clutter IN, boundary out.
   - Boundary 255 excluded from loss/metrics everywhere, Potsdam & Darmstadt.
   - `metrics.py` hard-asserts these — do not relax the asserts.
2. **Split:** tile-level only. VAL_TILE=6_15 (484 patches), 5 tiles train (2420).
   Random split FORBIDDEN (50% patch overlap → leakage).
3. **Architecture is frozen by the handoff** — backbone frozen, LoRA Q+V r=16,
   CLIP frozen, cosine head. The model head is ALWAYS 6-class.
4. **Model classes:** use `models_dino_lora.py` (6-class — the old notebook
   CELL16 had a 5-class-head bug; the recovery-cell version is canonical).
5. **TTPA:** adapt `text_proj` ONLY. Max 2 steps, LR=1e-5, KL weight 0.5.
   More steps/LR → prediction collapse. Fallback: 1 step, 5e-6, KL 1.0.
6. **text_embeddings.pt:** reuse if on disk. Only `prompts.load_or_encode_
   text_embeddings()` may create it; pass `force=True` only on prompt changes.
7. **Resolution bridge:** 30% of train patches, factor from
   `constants.RESOLUTION_FACTOR` (2 if 10cm, 4 if 5cm — verify via NB01 GSD
   cell). Label mask is NEVER blurred/resized.
8. **Darmstadt eval is patch-based.** Never stitch predictions into tiles.
   OSM pseudo-GT: rasterize → erode 3×3 (borders→255) → F1. Always call it
   "OSM pseudo-GT F1".
9. **HF token from Kaggle Secrets `HF_TOKEN`** — never hardcode. Repo:
   `HarishDeepak/geo-prompt-peft-checkpoints` (private).
10. **DOP20 is RGBI** — always take only the first 3 bands for the model.

## Code layout
- `src/rg_geoprompt/` — paths, constants, datasets, augment, prompts,
  models_dino_lora, models_geoprompt, losses, metrics, ttpa, osm_eval, utils.
- `notebooks/01..03, 99` — preprocess/verify, train+eval, Darmstadt, recovery.
- Notebooks stay thin: `from rg_geoprompt.X import Y`. Logic lives in modules.
- Anything tagged `# SOURCE: handoff spec — not yet run on Kaggle` is a
  reviewed draft (TTPA run, Darmstadt inference, OSM eval, DOP20 slicing).

## Current state / next steps
- Done: data pipeline, SegFormer, DINOv2+LoRA, CLIP prototypes (saved).
- Next: train GeoPrompt (NB02, ~45 min; expect τ→~0.04), F1 for all 3 models,
  download DOP20 locally → slice → upload `darmstadt-dop20` → NB03.
- GeoPrompt full-res cosine head is memory-heavy and unverified on T4; OOM
  fallback flag `lowres_similarity=True` exists in `models_geoprompt.py`.

## Working style (user preference)
Teach-as-you-go: cell-by-cell with confirmation checkpoints. Concise output
confirmations. Flag irreversible actions and security issues immediately.
Honest assessment over optimism. Do not invent results.
