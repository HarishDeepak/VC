# RG-GeoPrompt-PEFT

Open-vocabulary aerial image segmentation via parameter-efficient fine-tuning.
Fraunhofer IGD Praktikum, SoSe 2026.

**Goal:** Train on labeled ISPRS Potsdam imagery → zero-shot transfer to unlabeled Darmstadt DOP20.

---

## Results

| Model | Trainable params | mIoU (5-cls, Potsdam val) |
|---|---|---|
| SegFormer-B0 | 3.7M (100%) | 82.1% |
| DINOv2 + LoRA r=16 | 788K (0.9%) | 84.6% |
| RG-GeoPrompt-PEFT | ~1.2M (1.35%) | TBD |

> mIoU excludes clutter (class 5) and boundary (255). F1 includes clutter, excludes boundary.
> These are **supervised** Potsdam numbers — not comparable to zero-shot published baselines.

---

## Method

**DINOv2-base** (frozen) backbone with **LoRA r=16** adapters on Q+V attention layers.
Classification via **cosine similarity** between per-pixel DINOv2 features and CLIP text prototypes,
bridged by a trainable 2-layer MLP (512→768→768, GELU).

Key components:
- **Resolution bridge** — 30% of training patches are downsampled/upsampled to simulate 20cm GSD
- **TTPA** — test-time prompt adaptation of the text projection only (2 steps, LR=1e-5)
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
├── constants.py          # hyperparams, metric rules (single source of truth)
├── datasets.py           # Potsdam + Darmstadt data loading
├── augment.py            # resolution bridge + spatial/color augmentation
├── prompts.py            # CLIP text encoding, 5-prompt ensemble per class
├── models_dino_lora.py   # DINOv2 + LoRA backbone
├── models_geoprompt.py   # full GeoPrompt model
├── losses.py             # cross-entropy with boundary masking
├── metrics.py            # mIoU (5-cls) and F1 (6-cls) with hard assertions
├── ttpa.py               # test-time prompt adaptation
├── osm_eval.py           # OSM pseudo-GT evaluation for Darmstadt
└── utils.py              # checkpoints, HF backup, logging, visualization

notebooks/
├── 01_preprocess_potsdam.ipynb   # data verification, GSD check
├── 02_train_potsdam.ipynb        # GeoPrompt training + eval
├── 03_infer_darmstadt.ipynb      # zero-shot + TTPA + OSM eval
└── 99_recovery.ipynb             # cold-start after Kaggle session restart
```

---

## Checkpoints

Stored at [`HarishDeepak/geo-prompt-peft-checkpoints`](https://huggingface.co/HarishDeepak/geo-prompt-peft-checkpoints) (private).

| File | Description |
|---|---|
| `dinov2_lora_best.pth` | DINOv2 + LoRA best checkpoint |
| `geoprompt_best.pth` | RG-GeoPrompt-PEFT best checkpoint |
| `text_embeddings.pt` | CLIP text prototypes (6 classes × 512-dim) |
