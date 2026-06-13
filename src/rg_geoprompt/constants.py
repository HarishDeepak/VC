"""Project-wide constants. Single source of truth.

CRITICAL metric rules (Praktikum brief — never change, never merge):
    mIoU : 5 classes (0-4).  Clutter (5) AND boundary (255) excluded.
    F1   : 6 classes (0-5).  Clutter INCLUDED. Only boundary (255) excluded.
Boundary pixels (255) are excluded from loss, mIoU and F1 EVERYWHERE,
for both Potsdam and Darmstadt. No exceptions.
"""
import torch

# ── Classes ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Impervious", "Building", "Low Veg", "Tree", "Car", "Clutter"]
CLASS_COLORS = [
    (255, 255, 255),  # 0 impervious — white
    (0, 0, 255),      # 1 building   — blue
    (0, 255, 255),    # 2 low veg    — cyan
    (0, 255, 0),      # 3 tree       — green
    (255, 255, 0),    # 4 car        — yellow
    (255, 0, 0),      # 5 clutter    — red
]
NUM_CLASSES_TOTAL = 6        # model output channels — ALWAYS 6 (incl. clutter)
BOUNDARY_ID = 255            # ignore_index for loss and all metrics

# Metric ignore sets — see module docstring. Do not swap these.
MIOU_NUM_CLASSES = 5
MIOU_IGNORE_IDS = (5, 255)   # clutter + boundary
F1_NUM_CLASSES = 6
F1_IGNORE_IDS = (255,)       # boundary only

# ── Data / split ───────────────────────────────────────────────────────────
VAL_TILE = "6_15"            # tile-level split. Random split is FORBIDDEN
TRAIN_TILES = ["5_14", "5_15", "6_13", "6_14", "7_13"]
PATCH_SIZE = 512
PATCH_STRIDE = 256
BATCH_SIZE = 4
NUM_WORKERS = 2

# ── Resolution-bridge augmentation ─────────────────────────────────────────
# 2 if Potsdam data is 10cm GSD, 4 if 5cm GSD.
# VERIFY with rasterio (notebook 01, GSD check cell) before trusting this.
RESOLUTION_FACTOR = 2        # ⚠ unverified default — confirm with rasterio
RESOLUTION_BRIDGE_PROB = 0.30

# ── Architecture (fixed by handoff — do not change) ────────────────────────
DINOV2_NAME = "facebook/dinov2-base"
# Register-token variant — eliminates high-norm artifact tokens in background
# regions. Verify HF model ID before switching: facebook/dinov2-with-registers-base
DINOV2_REG_NAME = "facebook/dinov2-with-registers-base"
DINOV2_REG_TOKENS = 4        # extra tokens inserted after CLS in register variant
DINOV2_PATCH = 14
DINOV2_STRIDE = 14           # change to 7 for overlapping-patch training
DINOV2_DIM = 768
CLIP_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_DIM = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = ["query", "value"]
LORA_DROPOUT = 0.1
TAU_INIT = 0.07              # learnable temperature init (CLIP's value)
TAU_MIN = 0.01               # clamp floor

# ── Training (GeoPrompt) ───────────────────────────────────────────────────
GEO_EPOCHS = 10
GEO_LR = 1e-4
EVAL_EVERY = 5               # epochs between mIoU eval + checkpoint

# ── TTPA (fixed by handoff — more steps causes prediction collapse) ────────
TTPA_STEPS = 2               # max 2
TTPA_LR = 1e-5               # NOT 1e-4
TTPA_KL_WEIGHT = 0.5
# Collapse fallback (RISKS.md): steps=1, lr=5e-6, kl_weight=1.0

# ── OSM pseudo-GT (Darmstadt) ──────────────────────────────────────────────
OSM_EROSION_KERNEL = 3       # 3x3 morphological erosion → border pixels = 255
OSM_TAGS = {"building": True, "highway": True,
            "natural": "wood", "landuse": "grass"}
DOP20_CRS_EPSG = 25832       # UTM 32N — reproject OSM (4326) to this

# ── Infrastructure ─────────────────────────────────────────────────────────
HF_REPO_ID = "HarishDeepak/geo-prompt-peft-checkpoints"
HF_SECRET_NAME = "HF_TOKEN"  # Kaggle Secrets key — NEVER hardcode a token

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
