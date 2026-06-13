"""Path resolution that works both on Kaggle and locally.

On Kaggle:
    data  -> /kaggle/input/...
    work  -> /kaggle/working
Locally (dry runs, no GPU expected):
    data  -> ./data/...   (you place a small subset here)
    work  -> ./outputs

Every other module imports paths from here. Nothing else in the
codebase hardcodes /kaggle/... .
"""
from pathlib import Path

# ── Environment detection ──────────────────────────────────────────────────
ON_KAGGLE: bool = Path("/kaggle/input").exists()

# ── Potsdam (training) data ────────────────────────────────────────────────
# CONFIRMED path of the preprocessed patch dataset on Kaggle.
_KAGGLE_POTSDAM = Path(
    "/kaggle/input/datasets/harish77718/ovrsis-potsdam-team1/processed_potsdam"
)
_LOCAL_POTSDAM = Path("data/processed_potsdam")

ROOT: Path = _KAGGLE_POTSDAM if ON_KAGGLE else _LOCAL_POTSDAM
IMG_DIR: Path = ROOT / "images"
LABEL_DIR: Path = ROOT / "labels"

# ── Darmstadt DOP20 (zero-shot target) ─────────────────────────────────────
# Kaggle mounts user datasets under /kaggle/input/datasets/<user>/<slug>.
# The kaggle CLI --dir-mode zip flattens the images/ folder to the dataset
# root, so patches and transforms.json sit directly under DARMSTADT_ROOT.
_KAGGLE_DARMSTADT = Path(
    "/kaggle/input/datasets/harish77718/darmstadt-dop20"
)
_LOCAL_DARMSTADT = Path("data/darmstadt_dop20")

DARMSTADT_ROOT: Path = _KAGGLE_DARMSTADT if ON_KAGGLE else _LOCAL_DARMSTADT
# On Kaggle patches are at root (zip flattened); locally they're in images/
DARMSTADT_IMG_DIR: Path = (
    DARMSTADT_ROOT if ON_KAGGLE else DARMSTADT_ROOT / "images"
)
DARMSTADT_NDVI_DIR: Path = DARMSTADT_ROOT / "ndvi"    # optional NDVI masks

# ── Working / output directory ─────────────────────────────────────────────
WORK_DIR: Path = Path("/kaggle/working") if ON_KAGGLE else Path("outputs")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── Canonical artifact paths (single source of truth for filenames) ────────
SEGFORMER_CKPT = WORK_DIR / "segformer_b0_potsdam_baseline.pth"
DINOV2_LORA_CKPT = WORK_DIR / "dinov2_lora_best.pth"
GEOPROMPT_CKPT = WORK_DIR / "geoprompt_best.pth"
TEXT_EMBEDDINGS_PT = WORK_DIR / "text_embeddings.pt"
DINOV2_LOG_CSV = WORK_DIR / "dinov2_training_log.csv"
GEOPROMPT_LOG_CSV = WORK_DIR / "geoprompt_training_log.csv"


def describe() -> str:
    """Human-readable summary of the resolved environment, for notebook cells."""
    lines = [
        f"ON_KAGGLE        : {ON_KAGGLE}",
        f"Potsdam ROOT     : {ROOT}   (exists: {ROOT.exists()})",
        f"  images         : {IMG_DIR.exists()}",
        f"  labels         : {LABEL_DIR.exists()}",
        f"Darmstadt ROOT   : {DARMSTADT_ROOT}   (exists: {DARMSTADT_ROOT.exists()})",
        f"WORK_DIR         : {WORK_DIR}",
    ]
    return "\n".join(lines)
