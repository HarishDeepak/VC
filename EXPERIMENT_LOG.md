# RG-GeoPrompt-PEFT — Experiment Log

Fraunhofer IGD Praktikum SoSe 2026.  
**Goal:** supervised training on ISPRS Potsdam (labeled, 5–10 cm GSD) → zero-shot transfer to Darmstadt DOP20 (unlabeled, 20 cm GSD).  
All training runs on Kaggle T4 (16 GB VRAM).

---

## 1. Problem & Theoretical Framing

### Task
Open-vocabulary remote sensing segmentation across a resolution gap.  
- **Source domain:** ISPRS Potsdam — RGB-IR orthophotos, ~5 cm GSD, pixel-accurate labels (6 classes).  
- **Target domain:** Darmstadt DOP20 — RGBI orthophotos, ~20 cm GSD, **no labels**.  
- We never fine-tune on target-domain data; the model must generalise purely via visual + language priors.

### Why this is hard
1. **4× resolution gap** — buildings and trees look ~4× smaller in Darmstadt patches at the same pixel size.  
2. **No target labels** — standard fine-tuning or even calibration is impossible; only test-time self-supervision is available.  
3. **Class imbalance** — impervious surface and building dominate; car and clutter are rare and OSM doesn't track them anyway.

### Approach — three-stage
| Stage | What | Why |
|---|---|---|
| PEFT backbone | DINOv2-base frozen + LoRA r=16 on Q,V | Strong visual features; only 788 K params updated |
| Language grounding | CLIP ViT-B/32 frozen text prototypes + MLP projection | Open-vocabulary: class names drive the head |
| Resolution bridge | 30 % of train patches downsampled 4× | Forces encoder to be resolution-invariant |

### Why TTPA (test-time prompt adaptation)
At inference on Darmstadt the model's text projector sees a domain shift (texture, resolution). TTPA adapts **only `text_proj`** using self-supervised entropy on the predicted distribution — no labels required.  
In practice: ZS F1 0.2059, TTPA F1 0.2058 — TTPA had negligible effect. Likely reason: the resolution bridge already closed most of the gap, and entropy minimisation at the patch level doesn't improve global class frequencies.

---

## 2. Architecture

```
Input patch 512×512×3
       ↓
DINOv2-base (frozen) + LoRA r=16 on Q,V projections
  → patch tokens [B, 1369, 768]  (37×37 grid, stride 14)
       ↓  (lowres_similarity=True on T4)
Upsample tokens → 36×36 spatial grid
       ↓
CLIP ViT-B/32 (frozen) — encodes 6 class-name prompts → [6, 512]
       ↓
MLP text projection: 512 → 768 → 768 (GELU, no BN)
  trainable: ~412 K params
       ↓
Cosine similarity head: dot(visual, text) / τ
  τ = learnable scalar, init 0.07, clamped ≥ 0.01
       ↓
Logit map [B, 6, 36, 36] → bilinear upsample → [B, 6, 512, 512]
       ↓
Argmax → segmentation mask [B, 512, 512]
```

**Total trainable params:** ~1.2 M (~1.35 % of DINOv2-base)  
LoRA: 788 K | text_proj MLP: ~412 K | τ: 1

### Darmstadt inference extras
- **Max-pool prompt ensemble:** 8 building + 5 clutter text variants; max logit over prompts per class.
- **Orthogonal projection:** `logits[:, 1] -= 0.3 × logits[:, 5]` to suppress clutter bleeding into building channel.
- **TTPA:** 5 steps, lr 3e-4, KL weight 0.05, masked entropy (skip high-entropy pixels). Adapts `text_proj` only.

---

## 3. Data

### Potsdam (training)
- 6 tiles train, 1 tile val (tile `6_15`, 484 patches).
- Tile-level split only — **no random split** (50 % patch overlap would cause leakage).
- Patches: 512×512, stride 256, 5 classes used for mIoU (clutter excluded from loss/mIoU but included in F1).
- Resolution bridge: 30 % of patches bicubic-downsampled by `RESOLUTION_FACTOR` (4 if 5 cm GSD), then upsampled back. Label mask never altered.

### Darmstadt DOP20
- 4 tiles, 1296 patches total (324 per tile), 512×512, stride 512 (no overlap).
- RGBI input — always take only first 3 bands (RGB) before model.
- No labels. Evaluation via OSM pseudo-GT only.
- Kaggle dataset: `harish77718/darmstadt-dop20`

### OSM pseudo-GT
Rasterize OSM building/road/vegetation vectors → erode 3×3 (mark borders as 255) → F1.  
Car and Clutter always score 0.0 (no OSM layer for those classes).  
Always called "OSM pseudo-GT F1" — not a ground-truth metric.

---

## 4. Training Results (Potsdam, 5-class mIoU)

### SegFormer-B0 (baseline)
| Metric | Value |
|---|---|
| Trainable params | 3.7 M (100 %) |
| mIoU (5 cls) | **82.1 %** |

### DINOv2 + LoRA r=16
| Epoch | Val loss | mIoU |
|---|---|---|
| 5 | 0.2915 | 82.84 % |
| 10 | 0.3352 | 84.24 % |
| 15 | 0.3370 | **85.26 %** ← best |
| 20 | 0.3627 | 85.19 % |

Trainable params: 788 K (0.9 %). Best checkpoint: epoch 15.

### RG-GeoPrompt-PEFT (this model)
| Epoch | Train loss | Val loss | mIoU | τ |
|---|---|---|---|---|
| 1 | 0.3558 | 0.3415 | — | 0.0657 |
| 5 | 0.0836 | 0.3245 | 84.8 % | 0.0572 |
| 10 | 0.0709 | 0.3495 | **84.9 %** | 0.0517 |

Trainable params: ~1.2 M (1.35 %). τ converged from 0.07 → ~0.052 (expected ~0.04).  
Trained with `lowres_similarity=True` (cosine similarity at 36×36 token grid, avoids OOM on T4).

### Model comparison
| Model | Trainable | mIoU (5 cls) |
|---|---|---|
| SegFormer-B0 | 3.7 M (100 %) | 82.1 % |
| DINOv2 + LoRA r=16 | 788 K (0.9 %) | **85.26 %** |
| RG-GeoPrompt-PEFT | ~1.2 M (1.35 %) | 84.9 % |

---

## 5. Darmstadt Zero-Shot Results (OSM pseudo-GT F1)

Evaluated on all 1296 Darmstadt DOP20 patches. Metrics are patch-based (never stitched for evaluation).

| Class | Zero-shot F1 | TTPA F1 |
|---|---|---|
| Impervious | 0.5156 | 0.5154 |
| Building | 0.5810 | 0.5801 |
| Low Veg | 0.0458 | 0.0459 |
| Tree | 0.0931 | 0.0931 |
| Car | 0.0000 | 0.0000 |
| Clutter | 0.0000 | 0.0000 |
| **MEAN** | **0.2059** | **0.2058** |

TTPA changed ~7.2 % of pixels per patch (confirmed with diagnostic at lr=3e-4) but did not improve global F1.  
Car/Clutter always 0 — no OSM layer exists for those classes.

> **Context:** Published OVRSIS numbers (TPOVSeg 38–44 %, SegEarth-OV ~47 %) are zero-shot transfer *to Potsdam* — a different task. Our numbers are zero-shot *from* Potsdam *to* Darmstadt. Do not mix in one table.

---

## 6. Code Structure

```
rg-geoprompt-peft/
├── CLAUDE.md                        # project rules / non-negotiables
├── EXPERIMENT_LOG.md                # this file
├── kaggle_exec.py                   # WebSocket proxy to run code on live Kaggle kernel
│
├── src/rg_geoprompt/
│   ├── constants.py                 # all magic numbers (TTPA params, resolution factor, etc.)
│   ├── paths.py                     # path helpers for Kaggle vs local
│   ├── datasets.py                  # PotsdamDataset, DarmstadtDataset
│   ├── augment.py                   # resolution bridge augmentation
│   ├── prompts.py                   # load_or_encode_text_embeddings()
│   ├── models_dino_lora.py          # DINOv2+LoRA model — 6-class head (canonical)
│   ├── models_geoprompt.py          # RG-GeoPrompt-PEFT model (lowres_similarity flag)
│   ├── losses.py                    # combined loss (CE + dice, boundary excluded)
│   ├── metrics.py                   # mIoU (5-cls) + F1 (6-cls) with hard asserts
│   ├── ttpa.py                      # test-time prompt adaptation (text_proj only)
│   ├── osm_eval.py                  # OSM rasterize → erode → F1 pipeline
│   ├── crf_refine.py                # DenseCRF boundary refinement (experimental)
│   └── utils.py                     # five_column_figure(), colorize_mask()
│
├── notebooks/
│   ├── 01_preprocess_potsdam.ipynb  # tile slicing, GSD verification
│   ├── 02_train_potsdam.ipynb       # training loop for all 3 models
│   ├── 03_infer_darmstadt.ipynb     # Darmstadt inference + OSM eval
│   └── 99_recovery.ipynb            # emergency recovery cells
│
├── scripts/
│   └── stitch_darmstadt_vis.py      # stitch 1296 patches → 4 full-tile PNGs
│                                    # outputs: _rgb _zs _ttpa _diff _compare per tile
│
└── results/
    ├── osm_pseudo_gt_f1.csv         # zero-shot vs TTPA F1 per class
    ├── dinov2_lora_training_log.csv  # epoch-by-epoch DINOv2+LoRA training
    └── geoprompt_training_log.csv   # epoch-by-epoch GeoPrompt training
```

### Key constants (`src/rg_geoprompt/constants.py`)
| Constant | Value | Meaning |
|---|---|---|
| `RESOLUTION_FACTOR` | 4 (5 cm) / 2 (10 cm) | Resolution bridge downsample factor |
| `DINOV2_STRIDE` | 14 | DINOv2 patch stride |
| `DINOV2_REG_TOKENS` | 4 | Register tokens to discard |
| `TTPA_DARMSTADT_STEPS` | 5 | TTPA iterations per patch |
| `TTPA_DARMSTADT_LR` | 3e-4 | Adapted from 5e-5 (too low — L2-norm cancelled) |
| `TTPA_DARMSTADT_KL_WEIGHT` | 0.05 | Low → entropy term dominates; no collapse |

### Metric rules (never swap)
- **mIoU:** 5 classes (0–4), `ignore_ids=(5, 255)` — clutter AND boundary excluded.
- **F1:** 6 classes (0–5), `ignore_ids=(255,)` — clutter IN, boundary excluded.
- Both: boundary pixels (255) excluded from loss everywhere.

---

## 7. Visualisation Outputs

Stitched full-tile PNGs are in `/kaggle/working/darmstadt_stitched/` (visualisation only — metrics are always patch-based).

| File pattern | Content |
|---|---|
| `<tile>_rgb.png` | Original DOP20 true color (RGB from RGBI) |
| `<tile>_zs.png` | Zero-shot prediction (colorized) |
| `<tile>_ttpa.png` | TTPA prediction (colorized) |
| `<tile>_diff.png` | Changed pixels ZS→TTPA (red) |
| `<tile>_compare.png` | 4-column side-by-side with class legend |

4 tiles: `dop20_32_474_5532_1_he`, `dop20_32_475_5524_1_he`, `dop20_32_475_5525_1_he`, `dop20_32_475_5527_1_he`

Class colour map:
| Class | Colour |
|---|---|
| Impervious | White (255,255,255) |
| Building | Blue (0,0,255) |
| Low Vegetation | Cyan (0,255,255) |
| Tree | Green (0,255,0) |
| Car | Yellow (255,255,0) |
| Clutter | Red (255,0,0) |
| Boundary/Unknown | Gray (128,128,128) |

---

## 8. Storage & Backup

| Location | What |
|---|---|
| GitHub `HarishDeepak/VC` branch `main` | Full source code, notebooks, results CSVs |
| HF `HarishDeepak/geo-prompt-peft-checkpoints` (private) | Model checkpoints, stitched PNGs, patch figures, CSV |
| Kaggle dataset `harish77718/darmstadt-dop20` | 1296 DOP20 patch PNGs (RGBI, input only) |
| `/kaggle/working/darmstadt_preds/` | 2592 npy files (1296 `_zs.npy` + 1296 `_ttpa.npy`) — ephemeral |

HF token loaded from Kaggle Secrets `HF_TOKEN` — never hardcoded.

---

## 9. Key Git Commits

| Commit | Description |
|---|---|
| `601d8dd` | Fix stitch script: add RGB loading and 4-col compare output |
| `5f75a70` | Add Darmstadt patch stitching script |
| `15076d9` | Add Darmstadt OSM pseudo-GT F1 results |
| `3592b74` | Optimise OSM eval, tune TTPA params, make entropy optional |
| `8067203` | Fix TTPA: diagnostic, masked entropy, Darmstadt-tuned params |
| `9f2cf78` | Add DenseCRF boundary refinement |
| `3d78808` | Add stride-7 overlapping patches + register-token backbone |
| `6c3ad52` | Darmstadt inference: max-pool prompts + ortho projection |
| `adce57a` | Add Darmstadt zero-shot OSM pseudo-GT F1 results |
| `b146f87` | Add GeoPrompt training log (10 epochs, best mIoU 84.9 %) |
| `f0bb127` | Add DINOv2+LoRA training log (20 epochs, best mIoU 85.26 %) |
