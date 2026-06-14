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

### Potsdam supervised (5-class mIoU, clutter+boundary excluded)
| Model | Trainable | mIoU | Best epoch |
|---|---|---|---|
| SegFormer-B0 | 3.7M (100%) | 82.1% | — |
| DINOv2+LoRA r=16 | 788K (0.9%) | **85.26%** | 15/20 |
| RG-GeoPrompt-PEFT | ~1.2M (~1.35%) | **84.9%** | 10/10 |

### Darmstadt zero-shot (OSM pseudo-GT F1, 6-class, boundary excluded)
| Class | ZS F1 | TTPA F1 |
|---|---|---|
| Impervious | 0.5156 | 0.5154 |
| Building | 0.5810 | 0.5801 |
| Low Veg | 0.0458 | 0.0459 |
| Tree | 0.0931 | 0.0931 |
| Car | 0.0000 | 0.0000 |
| Clutter | 0.0000 | 0.0000 |
| **MEAN** | **0.2059** | **0.2058** |

TTPA params confirmed: steps=5, lr=3e-4, kl_weight=0.05, masked_entropy=True.
TTPA had negligible effect on F1 (~7% pixel change per patch, no global improvement).

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
5. **TTPA:** adapt `text_proj` ONLY. Darmstadt confirmed params: 5 steps,
   LR=3e-4, KL weight=0.05, masked_entropy=True. (5e-5 too low — L2-norm
   cancels updates. 3e-4 gives 7.2% pixel change, no collapse confirmed.)
   Fallback: 1 step, 5e-6, KL 1.0.
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
- Done: data pipeline, SegFormer (82.1%), DINOv2+LoRA (85.26%), GeoPrompt (84.9%).
- Done: Darmstadt DOP20 inference — 1296 patches, ZS + TTPA. OSM pseudo-GT F1 computed.
- Done: Stitched full-tile visualisations (4 tiles, RGB|ZS|TTPA|Diff|Compare PNGs).
- TODO: HF backup of stitched PNGs + patch-level figures (rate-limited; retry with
  `create_commit([CommitOperationAdd(...)])` to batch all files in one commit).
- TODO: F1 scores for SegFormer and DINOv2+LoRA models (not yet computed).
- GeoPrompt ran with `lowres_similarity=True` (cosine at 36×36 token grid); τ → ~0.052.

## Working style (user preference)
Teach-as-you-go: cell-by-cell with confirmation checkpoints. Concise output
confirmations. Flag irreversible actions and security issues immediately.
Honest assessment over optimism. Do not invent results.
