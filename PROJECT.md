# RG-GeoPrompt-PEFT — Project Overview

> **Who is this for?** Anyone — student, researcher, or curious reader — who wants to understand
> what this project does, why it exists, and how it works, from first principles.

---

## 1. The Big Picture — What Problem Are We Solving?

Imagine you have a satellite or aerial photograph of a city. Every single pixel in that image
belongs to something: a rooftop, a road, a tree, a car, a patch of grass. The task of
automatically labelling every pixel with the correct category is called
**semantic segmentation**.

This is genuinely useful:
- Urban planners can measure how much green space a city has
- Emergency services can map flooded roads after a disaster
- Climate researchers can track urban heat islands

The challenge: **AI models need labeled training data**, and labeling aerial images is
extremely expensive. A human expert has to go through millions of pixels and say "this is a
road, this is a building..." For one city alone this takes months. For every new city you want
to analyze, you have to do it again.

**Our goal:** Train a model on one labeled city (Potsdam, Germany) and have it work on a
completely different, unlabeled city (Darmstadt, Germany) — without any Darmstadt labels at
all. This is called **zero-shot transfer**.

---

## 2. The Dataset

### Training City — ISPRS Potsdam
- **What:** Aerial imagery of Potsdam, Germany
- **Resolution:** 5–10 cm per pixel (you can see individual roof tiles)
- **Labels:** 6 classes for every pixel
  | ID | Class | What it looks like |
  |---|---|---|
  | 0 | Impervious surface | Roads, parking lots, pavements |
  | 1 | Building | Rooftops |
  | 2 | Low vegetation | Grass, bushes |
  | 3 | Tree | Tree canopies |
  | 4 | Car | Individual vehicles |
  | 5 | Clutter | Everything else (fences, chimneys, etc.) |
- **Split:** 5 tiles for training (2,420 patches), 1 tile for validation (484 patches)
  - Important: patches have 50% overlap, so we must split by whole **tiles** not random patches
    (otherwise near-identical patches end up in both train and validation — "data leakage")

### Target City — Darmstadt DOP20
- **What:** Aerial imagery of Darmstadt, Germany
- **Resolution:** 20 cm per pixel (4× coarser than Potsdam)
- **Labels:** None — this is the zero-shot challenge
- **Evaluation:** We use OpenStreetMap building/road data as a noisy proxy for ground truth
  (called "OSM pseudo-GT"). It's imperfect but gives an honest signal.

The **resolution gap** (5–10cm vs 20cm) is a real obstacle — a model trained on sharp
Potsdam images will struggle with blurrier Darmstadt images unless we specifically prepare it.

---

## 3. Why Standard Approaches Fall Short

A traditional approach:
1. Take a large pretrained network
2. Fine-tune ALL its weights on Potsdam
3. Hope it works on Darmstadt

Problems:
- Fine-tuning all weights (~86M parameters) makes the model overfit to Potsdam's exact
  visual style. It becomes brittle to the resolution and color differences in Darmstadt.
- The model has no concept of *what* a building is — it just memorized pixel patterns.
  If Darmstadt buildings look slightly different, performance collapses.
- You cannot change the classes without retraining everything from scratch.

---

## 4. Our Approach — Three Ideas in One

### Idea 1: Parameter-Efficient Fine-Tuning (PEFT) with LoRA

Instead of retraining 86 million weights, we use **LoRA** (Low-Rank Adaptation).

Think of it like this: the pretrained model already "knows" what the visual world looks like.
We don't want to erase that. Instead, we add tiny trainable "correction layers"
(only 788K parameters — less than 1% of the total) that adapt its knowledge to aerial imagery
without overwriting the general visual understanding.

Result: the model stays general enough to handle Darmstadt's different appearance, while still
learning Potsdam-specific patterns.

### Idea 2: Vision Foundation Model — DINOv2

We don't start from scratch. We start from **DINOv2**, a model trained by Meta on 142 million
diverse images using self-supervised learning. It has already learned extremely rich visual
representations — edges, textures, shapes, objects — without ever needing human labels.

DINOv2 breaks an image into 14×14 pixel patches and produces a 768-dimensional feature vector
for each patch, capturing what's "interesting" about that patch. We freeze DINOv2 completely
and only train the LoRA adapters.

### Idea 3: Text-Guided Classification with CLIP

This is the core novelty — the "GeoPrompt" part.

**CLIP** (Contrastive Language-Image Pretraining, by OpenAI) is a model that learned to
understand both images and text together. It can tell that a photo of a dog and the words
"a photo of a dog" belong together.

We use CLIP's text encoder to turn class descriptions into numerical vectors:
```
"an aerial view of a building rooftop from above"  →  [0.12, -0.34, 0.88, ...]  (512 numbers)
"a satellite photo of a road or pavement"          →  [0.67,  0.21, -0.45, ...]  (512 numbers)
```

We create **5 different text prompts per class** (5 viewpoints on the same concept) and
average them for robustness. These text vectors are computed once and frozen forever —
they never change during training.

**Classification by similarity:** Instead of learning "if the pixel looks like X, predict
class Y", we ask: "which text description does this pixel's visual feature most resemble?"
This is done via cosine similarity — a geometric measure of how aligned two vectors are.

The critical consequence: **to change what classes the model looks for, you just change the
text**. No retraining needed. This is what "open-vocabulary" means.

---

## 5. Full Architecture — How It All Fits Together

```
Input image [512×512 pixels, RGB]
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  DINOv2-base  (FROZEN — 86M params, never updated)     │
│  + LoRA r=16 adapters on Query & Value attention layers │
│    (TRAINABLE — 788K params, ~0.9% of total)           │
└─────────────────────────────────────────────────────────┘
        │
        │  Produces: 36×36 patch tokens, each 768-dimensional
        │  (36 = 512÷14, rounded down)
        ▼
┌─────────────────────────────────────────────────────────┐
│  Upsample to 512×512  (bilinear interpolation)         │
│  L2-normalize each feature vector                       │
└─────────────────────────────────────────────────────────┘
        │
        │  Visual features: [512×512×768], each pixel has
        │  a unit-length feature vector
        ▼
        ╔═══════════════════════════════════════╗
        ║  COSINE SIMILARITY HEAD               ║
        ║                                       ║
        ║  For each pixel, compute similarity   ║
        ║  with each of the 6 text prototypes   ║
        ║  → 6 similarity scores                ║
        ║  → divide by temperature τ            ║
        ║  → softmax → class probabilities      ║
        ╚═══════════════════════════════════════╝
                    ▲
                    │ text prototypes [6×768]
                    │
        ┌──────────────────────────────┐
        │  Text Projection MLP         │
        │  512 → 768 → 768  (GELU)     │
        │  TRAINABLE (~590K params)    │
        └──────────────────────────────┘
                    ▲
                    │ CLIP text embeddings [6×512]
                    │
        ┌──────────────────────────────┐
        │  CLIP ViT-B/32  (FROZEN)     │
        │  Encodes text prompts once   │
        └──────────────────────────────┘

Output: [512×512×6] — one class score per pixel per class
```

**Temperature τ:** A learnable scalar (initialized at 0.07, same as CLIP's training value)
that scales the similarity scores before softmax. As training progresses, τ should drift
downward to ~0.04, meaning the model becomes more "decisive" — higher contrast between
the best-matching class and the others. We watch this as a training health indicator.

**Total trainable parameters:** ~1.2M out of ~87M total (~1.35%). Everything else is frozen.

---

## 6. The Resolution Bridge

Potsdam is 5–10cm/pixel. Darmstadt is 20cm/pixel. The model trained on sharp Potsdam images
will see blurry Darmstadt images at test time. Without preparation, this hurts performance.

**Resolution bridge augmentation:** During training, we randomly take 30% of Potsdam patches,
downsample them by a factor of 2–4× and upsample back to 512×512. This artificially simulates
what a lower-resolution image looks like after upsampling — exactly what Darmstadt looks like.

Crucially: **only the image is blurred, never the label mask**. The semantic content doesn't
change just because the image gets blurrier.

---

## 7. Test-Time Prompt Adaptation (TTPA)

Even with the resolution bridge, there's a domain gap between Potsdam and Darmstadt (different
color profiles, building styles, vegetation). TTPA is a lightweight trick to close this gap at
inference time, without any Darmstadt labels.

**How it works:**
1. Feed a Darmstadt patch to the frozen model → get an initial prediction
2. For just 2 gradient steps, update only the Text Projection MLP (not the backbone!)
3. Loss = entropy minimization + KL divergence penalty
   - Entropy minimization: push the model to be more confident on this patch
   - KL penalty: don't deviate too far from the original prediction (prevents collapse)
4. Use the adapted text projection for this patch's final prediction
5. Reset the text projection back to its original state for the next patch

**Why only text projection?** The backbone has learned Potsdam visual features — we don't
want to corrupt that. The text projection maps CLIP space to DINOv2 space — adapting it
slightly adjusts how text and visual features are aligned for Darmstadt's visual style.

**Collapse risk:** If TTPA runs too aggressively (too many steps, too high LR), the model
collapses — predicting the same class for every pixel (usually "road"). We detect this with
entropy checks and fall back to gentler parameters.

---

## 8. Training Pipeline — Step by Step

### Stage A: SegFormer Baseline (already done ✓)
- Standard semantic segmentation model, fully fine-tuned on Potsdam
- 3.7M trainable parameters
- Result: **82.1% mIoU** — our sanity check that the dataset works

### Stage B: DINOv2 + LoRA (currently training ⟳)
- Frozen DINOv2 + LoRA adapters + simple conv decode head
- Only 788K trainable parameters (0.9%)
- Expected result: **~84.6% mIoU** — better than SegFormer with 5× fewer trained params
- Saves `dinov2_lora_best.pth` — used as GeoPrompt's starting point

### Stage C: GeoPrompt — RG-GeoPrompt-PEFT (pending)
- Takes Stage B's backbone, discards the conv head, adds CLIP cosine head
- Trains for 10 epochs, LR=1e-4
- Adds text projection MLP (~590K params) + learnable temperature τ
- Total trainable: ~1.2M (1.35%)
- Expected result: TBD — may be slightly below 84.6% on supervised Potsdam, but gains
  open-vocabulary transfer ability that the other models don't have

### Stage D: Darmstadt Transfer (after checkpoints saved)
- Attach Darmstadt DOP20 patches as a Kaggle dataset
- Run zero-shot inference (no adaptation) and TTPA inference
- Compare against OSM pseudo-GT

---

## 9. Evaluation Metrics — Two Different Measures

This project uses **two different metrics for two different purposes**. They must never be mixed.

### mIoU — Supervised Potsdam Performance
- **Classes:** 5 (Impervious, Building, Low Veg, Tree, Car) — Clutter excluded
- **Formula:** For each class, IoU = true positives / (TP + FP + FN), then average
- **Why exclude clutter?** Clutter is a catch-all "everything else" class — it's visually
  incoherent and its performance doesn't reflect real segmentation quality
- Used for: comparing our 3 models on Potsdam validation

### F1 Score — Full Report Including Clutter
- **Classes:** 6 (all including Clutter) — only boundary pixels (255) excluded
- **Formula:** Harmonic mean of precision and recall per class
- **Why include clutter here?** The praktikum brief requires reporting all 6 classes
- Used for: the final report table; Clutter F1 will look bad (~20–30%) and that's expected

### Important Rule
Our supervised Potsdam mIoU (82–85%) **cannot be compared** to published zero-shot results
from papers like SegEarth-OV (~47% on Potsdam). They're testing different things:
- We: trained ON Potsdam, tested ON Potsdam (supervised)
- Papers: trained elsewhere, tested ON Potsdam (zero-shot)

These go in two separate tables in the report, always.

---

## 10. Why This Matters

| Property | SegFormer | DINOv2+LoRA | GeoPrompt (ours) |
|---|---|---|---|
| Trained params | 3.7M (100%) | 788K (0.9%) | 1.2M (1.35%) |
| Change classes without retraining? | No | No | **Yes** |
| Zero-shot transfer to new city? | Poor | Moderate | **Designed for this** |
| Open-vocabulary? | No | No | **Yes** |

The practical value: a city council with Darmstadt DOP20 imagery can get a reasonable
segmentation map with *no labeling work*, just by running GeoPrompt with a new city's
aerial imagery and the same text prompts. If they want to add a new class ("solar panel"),
they write one text description instead of relabeling thousands of patches.

---

## 11. Project File Structure

```
rg-geoprompt-peft/
├── src/rg_geoprompt/          ← all logic lives here, notebooks stay thin
│   ├── paths.py               ← Kaggle vs local path detection
│   ├── constants.py           ← all hyperparams, metric rules (single source of truth)
│   ├── datasets.py            ← Potsdam + Darmstadt data loading
│   ├── augment.py             ← resolution bridge + spatial/color augmentation
│   ├── prompts.py             ← CLIP text encoding, 5-prompt ensemble per class
│   ├── models_dino_lora.py    ← DINOv2 + LoRA backbone (Stage B model)
│   ├── models_geoprompt.py    ← full GeoPrompt model (Stage C model)
│   ├── losses.py              ← cross-entropy with boundary masking
│   ├── metrics.py             ← mIoU (5-cls) and F1 (6-cls) with hard assertions
│   ├── ttpa.py                ← Test-Time Prompt Adaptation
│   ├── osm_eval.py            ← OSM download → rasterize → erode → F1
│   └── utils.py               ← checkpoints, HF backup, logging, visualization
│
├── notebooks/
│   ├── 01_preprocess_potsdam.ipynb   ← data verification, GSD check (no GPU)
│   ├── 02_train_potsdam.ipynb        ← Stage C GeoPrompt training + all eval
│   ├── 03_infer_darmstadt.ipynb      ← zero-shot + TTPA + OSM eval (draft)
│   └── 99_recovery.ipynb             ← cold-start after Kaggle session restart
│
├── configs/default.yaml       ← human-readable config reference
├── CLAUDE.md                  ← ground truth rules for this project (AI-readable)
├── RISKS.md                   ← 15 known failure modes with detection + fixes
├── runbook.md                 ← step-by-step Kaggle session workflow
└── PROJECT.md                 ← this file
```

---

## 12. Current Status

| Item | Status |
|---|---|
| Potsdam data pipeline | Done ✓ |
| SegFormer baseline | Done ✓ — 82.1% mIoU |
| DINOv2+LoRA training | **Running now ⟳** — 1/20 epochs |
| GeoPrompt training | Pending (starts after DINOv2 finishes) |
| HuggingFace checkpoint backup | Pending (first save) |
| Darmstadt DOP20 download | Not started (local step needed) |
| Darmstadt inference + TTPA | Not started |
| OSM pseudo-GT evaluation | Not started |
