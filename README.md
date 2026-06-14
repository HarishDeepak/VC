# RG-GeoPrompt-PEFT

Open-vocabulary aerial image segmentation via parameter-efficient fine-tuning.
Fraunhofer IGD Praktikum, SoSe 2026.

**Goal:** Train on labeled ISPRS Potsdam imagery → zero-shot transfer to unlabeled Darmstadt DOP20.

---

## Results

### Supervised — Potsdam val (5-class mIoU, clutter + boundary excluded)

| Model | Trainable params | mIoU |
|---|---|---|
| SegFormer-B0 | 3.7M (100%) | 82.1% |
| DINOv2 + LoRA r=16 | 788K (0.9%) | **85.26%** (ep 15) |
| RG-GeoPrompt-PEFT | ~1.2M (1.35%) | 84.9% (ep 10) |

> mIoU: 5 classes (0–4), `ignore_ids=(5,255)`. F1: 6 classes (0–5), `ignore_ids=(255,)`.  
> Supervised Potsdam numbers — **not comparable** to zero-shot published baselines (TPOVSeg, SegEarth-OV).

### Zero-shot transfer — Darmstadt DOP20 (OSM pseudo-GT F1, 6-class)

| Class | Zero-shot F1 | TTPA F1 |
|---|---|---|
| Impervious | 0.5156 | 0.5154 |
| Building | 0.5810 | 0.5801 |
| Low Veg | 0.0458 | 0.0459 |
| Tree | 0.0931 | 0.0931 |
| Car | 0.0000 | 0.0000 |
| Clutter | 0.0000 | 0.0000 |
| **MEAN** | **0.2059** | **0.2058** |

> OSM pseudo-GT: rasterize → erode 3×3 → F1. Car/Clutter always 0 (no OSM layer).  
> TTPA (5 steps, lr=3e-4) changed ~7% of pixels per patch but did not improve global F1.

---

## Method

**DINOv2-base** (frozen) backbone with **LoRA r=16** adapters on Q+V attention layers.
Classification via **cosine similarity** between per-pixel DINOv2 features and CLIP text prototypes,
bridged by a trainable 2-layer MLP (512→768→768, GELU).

Key components:
- **Resolution bridge** — 30% of training patches are downsampled/upsampled to simulate 20cm GSD
- **TTPA** — test-time prompt adaptation of the text projection only (5 steps, LR=3e-4, masked entropy)
- **Learnable temperature τ** — init 0.07, expected to converge to ~0.04

---

## Datasets

| Dataset | Resolution | Labels | Role |
|---|---|---|---|
| ISPRS Potsdam | 5–10 cm/px | 6 classes | Training + validation |
| Darmstadt DOP20 | 20 cm/px (RGBI) | None | Zero-shot target |

Split: tile-level only. VAL_TILE=`6_15` (484 patches), 5 tiles train (2420 patches).
Random patch split is forbidden — 50% patch overlap causes data leakage.

---

## Setup

```bash
# Local
pip install torch torchvision transformers peft open_clip_torch huggingface_hub rasterio

# Kaggle — handled by notebook bootstrap (git clone + pip install cells)
```

No `setup.py` — add `src/` to `sys.path` (notebooks do this automatically).

---

## Running on Kaggle

1. Open `notebooks/02_train_potsdam.ipynb` on Kaggle with T4 GPU
2. Attach dataset: `harish77718/ovrsis-potsdam-team1`
3. Add Kaggle Secret: `HF_TOKEN` (write access to `HarishDeepak/geo-prompt-peft-checkpoints`)
4. Run all cells top to bottom
5. After training: run the HF backup cell before closing the session

For session recovery: use `notebooks/99_recovery.ipynb` — pulls checkpoints from HF automatically.

---

## File Structure

```
src/rg_geoprompt/
├── paths.py              # Kaggle vs local path detection
├── constants.py          # hyperparams, TTPA params, metric rules
├── datasets.py           # Potsdam + Darmstadt data loading
├── augment.py            # resolution bridge + spatial/color augmentation
├── prompts.py            # CLIP text encoding, prompt ensemble per class
├── models_dino_lora.py   # DINOv2 + LoRA backbone (6-class head)
├── models_geoprompt.py   # full GeoPrompt model (lowres_similarity flag)
├── losses.py             # cross-entropy with boundary masking
├── metrics.py            # mIoU (5-cls) and F1 (6-cls) with hard assertions
├── ttpa.py               # test-time prompt adaptation (text_proj only)
├── osm_eval.py           # rasterize OSM → erode → F1 pipeline
├── crf_refine.py         # DenseCRF boundary refinement (experimental)
└── utils.py              # checkpoints, HF backup, five_column_figure, colorize

notebooks/
├── 01_preprocess_potsdam.ipynb   # tile slicing, GSD verification
├── 02_train_potsdam.ipynb        # training all 3 models + Potsdam eval
├── 03_infer_darmstadt.ipynb      # Darmstadt inference + TTPA + OSM eval
└── 99_recovery.ipynb             # cold-start after Kaggle session restart

scripts/
└── stitch_darmstadt_vis.py       # stitch 1296 patches → 4-tile PNGs (RGB|ZS|TTPA|Diff)

results/
├── osm_pseudo_gt_f1.csv          # zero-shot vs TTPA F1 per class on Darmstadt
├── dinov2_lora_training_log.csv  # DINOv2+LoRA epoch log (best ep15: 85.26%)
└── geoprompt_training_log.csv    # GeoPrompt epoch log (best ep10: 84.9%)
```

---

## Checkpoints

Stored at [`HarishDeepak/geo-prompt-peft-checkpoints`](https://huggingface.co/HarishDeepak/geo-prompt-peft-checkpoints) (private).

| File | Description |
|---|---|
| `dinov2_lora_best.pth` | DINOv2 + LoRA best checkpoint |
| `geoprompt_best.pth` | RG-GeoPrompt-PEFT best checkpoint |
| `text_embeddings.pt` | CLIP text prototypes (6 classes × 512-dim) |
