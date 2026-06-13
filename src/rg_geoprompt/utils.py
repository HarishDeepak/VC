"""Utilities: seeding, visualization, checkpoints, HuggingFace backup/recovery.

HF backup logic is a port of notebook CELL23/CELL-R — confirmed working.
Token ALWAYS comes from Kaggle Secrets ("HF_TOKEN"); never hardcode.
"""
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from . import paths
from .constants import CLASS_COLORS, CLASS_NAMES, DEVICE, HF_REPO_ID, HF_SECRET_NAME


# ── Reproducibility ─────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Visualization ───────────────────────────────────────────────────────────
def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Class-ID mask [H, W] → RGB uint8 [H, W, 3]. Boundary (255) → gray."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in enumerate(CLASS_COLORS):
        out[mask == cls] = color
    out[mask == 255] = (128, 128, 128)
    return out


def five_column_figure(rgb: np.ndarray, zs_pred: np.ndarray,
                       ttpa_pred: np.ndarray, entropy: np.ndarray,
                       osm_mask: Optional[np.ndarray] = None,
                       title: str = "", save_path: Optional[Path] = None):
    """Report figure: RGB | zero-shot pred | TTPA pred | entropy | OSM.

    Args:
        rgb:      [H, W, 3] uint8 image
        zs_pred:  [H, W] class IDs (zero-shot, no TTPA)
        ttpa_pred:[H, W] class IDs (after TTPA)
        entropy:  [H, W] float per-pixel entropy
        osm_mask: [H, W] eroded OSM pseudo-GT (optional — column hidden if None)
    """
    import matplotlib.pyplot as plt

    cols = 5 if osm_mask is not None else 4
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes[0].imshow(rgb); axes[0].set_title("DOP20 RGB")
    axes[1].imshow(colorize_mask(zs_pred)); axes[1].set_title("Zero-shot")
    axes[2].imshow(colorize_mask(ttpa_pred)); axes[2].set_title("TTPA")
    im = axes[3].imshow(entropy, cmap="magma"); axes[3].set_title("Entropy")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    if osm_mask is not None:
        axes[4].imshow(colorize_mask(osm_mask)); axes[4].set_title("OSM pseudo-GT")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


def print_f1_report(f1: torch.Tensor, name: str) -> None:
    """Pretty per-class F1 table (all 6 classes + both means)."""
    print(f"\n{'=' * 52}\n  F1 Report: {name}\n{'=' * 52}")
    for i, (cls, score) in enumerate(zip(CLASS_NAMES, f1)):
        bar = "█" * int(score * 28)
        note = "  ← excluded from mIoU" if i == 5 else ""
        print(f"  [{i}] {cls:20s}: {score:.4f}  {bar}{note}")
    print(f"\n  Mean F1 (all 6 incl. clutter) : {f1.mean():.4f}")
    print(f"  Mean F1 (5 classes, no clutter): {f1[:5].mean():.4f}")


# ── HuggingFace backup / recovery ───────────────────────────────────────────
def _hf_login():
    """Login via Kaggle Secrets. Raises a clear error off-Kaggle."""
    from huggingface_hub import login
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(HF_SECRET_NAME)
    except ImportError:
        token = os.environ.get(HF_SECRET_NAME)  # local: export HF_TOKEN=...
        if not token:
            raise RuntimeError(
                f"Set the {HF_SECRET_NAME} env var locally, or run on "
                f"Kaggle with the {HF_SECRET_NAME} secret attached.")
    login(token=token, add_to_git_credential=False)


def hf_backup(files: Optional[Iterable[Path]] = None) -> None:
    """Upload artifacts to the private HF repo (end-of-session backup).

    Default: every file > 10KB plus all CSVs in WORK_DIR (CELL23 behavior).
    """
    from huggingface_hub import HfApi
    _hf_login()
    api = HfApi()
    api.create_repo(HF_REPO_ID, private=True, exist_ok=True)
    if files is None:
        files = [p for p in sorted(paths.WORK_DIR.iterdir())
                 if p.is_file() and (p.stat().st_size > 1e4
                                     or p.suffix == ".csv")]
    for f in files:
        f = Path(f)
        if not f.exists():
            print(f"  ✗ skipped (not found): {f}")
            continue
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=HF_REPO_ID, repo_type="model")
        print(f"  ✓ {f.name:50s} {f.stat().st_size / 1e6:.1f} MB")
    print(f"\nAll files → https://huggingface.co/{HF_REPO_ID}")


def ensure_checkpoint(filename: str, required: bool = True) -> Optional[Path]:
    """Return a local path to ``filename``: WORK_DIR copy if present,
    otherwise download from the HF backup repo.

    Args:
        required: if False, return None instead of raising when the file
                  exists nowhere (e.g. geoprompt_best.pth before training).
    """
    local = paths.WORK_DIR / filename
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download
        _hf_login()
        return Path(hf_hub_download(HF_REPO_ID, filename, repo_type="model"))
    except Exception as e:
        if required:
            raise FileNotFoundError(
                f"{filename} not in {paths.WORK_DIR} and not on HF "
                f"({HF_REPO_ID}): {e}") from e
        print(f"  ({filename} not available yet — skipping)")
        return None


# ── Training-log helper (append-safe) ───────────────────────────────────────
def append_log_row(csv_path: Path, header: list, row: list) -> None:
    """Append a row, writing the header only if the file does not exist.

    Replaces the notebook pattern of opening the log with mode='w' at the
    top of the training cell, which wiped history on every rerun/restart.
    """
    import csv
    csv_path = Path(csv_path)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
