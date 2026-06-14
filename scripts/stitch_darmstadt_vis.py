"""Stitch Darmstadt patch predictions into full-tile visualisation PNGs.

Run in a Kaggle cell (predictions must be in /kaggle/working/darmstadt_preds/):

    %run /kaggle/working/VC/rg-geoprompt-peft/scripts/stitch_darmstadt_vis.py

Or locally with PRED_DIR / IMG_DIR / OUT_DIR overridden via env vars.

Output (visualisation ONLY — never used for metrics):
    /kaggle/working/darmstadt_stitched/<tile_id>_rgb.png    (original DOP20)
    /kaggle/working/darmstadt_stitched/<tile_id>_zs.png     (zero-shot pred)
    /kaggle/working/darmstadt_stitched/<tile_id>_ttpa.png   (TTPA pred)
    /kaggle/working/darmstadt_stitched/<tile_id>_diff.png   (changed pixels)
    /kaggle/working/darmstadt_stitched/<tile_id>_compare.png (4-col side-by-side)
"""
import os
import re
import glob
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PRED_DIR = Path(os.environ.get("PRED_DIR", "/kaggle/working/darmstadt_preds"))
IMG_DIR  = Path(os.environ.get("IMG_DIR",  "/kaggle/input/datasets/harish77718/darmstadt-dop20"))
OUT_DIR  = Path(os.environ.get("OUT_DIR",  "/kaggle/working/darmstadt_stitched"))
PATCH    = 512

CLASS_NAMES  = ["Impervious", "Building", "Low Veg", "Tree", "Car", "Clutter"]
CLASS_COLORS = [
    (255, 255, 255),  # impervious — white
    (0,   0,   255),  # building   — blue
    (0,   255, 255),  # low veg    — cyan
    (0,   255,   0),  # tree       — green
    (255, 255,   0),  # car        — yellow
    (255,   0,   0),  # clutter    — red
]

def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        out[mask == i] = c
    out[mask == 255] = (128, 128, 128)
    return out


def parse_stem(stem: str):
    """'dop20_32_474_5532_1_he_y1024_x512' → (tile_id, y, x)"""
    m = re.match(r"^(.+)_y(\d+)_x(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse stem: {stem}")
    return m.group(1), int(m.group(2)), int(m.group(3))


def stitch_tile(patches: dict, fill=255, dtype=np.uint8) -> np.ndarray:
    """patches: {(y,x): [H,W] or [H,W,3] array} → stitched canvas"""
    sample = next(iter(patches.values()))
    max_y = max(y for y, x in patches) + PATCH
    max_x = max(x for y, x in patches) + PATCH
    shape = (max_y, max_x) if sample.ndim == 2 else (max_y, max_x, sample.shape[2])
    canvas = np.full(shape, fill, dtype=dtype)
    for (y, x), patch in patches.items():
        canvas[y:y+PATCH, x:x+PATCH] = patch
    return canvas


def save_stitched(tile_id: str, rgb: np.ndarray, zs: np.ndarray,
                  ttpa: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    legend = [mpatches.Patch(color=tuple(c/255 for c in col), label=name)
              for name, col in zip(CLASS_NAMES, CLASS_COLORS)]
    legend += [mpatches.Patch(color=(0.5,0.5,0.5), label="Boundary/Unknown")]

    changed_pct = 100 * (zs != ttpa).mean()
    diff_rgb = np.zeros((*zs.shape, 3), dtype=np.uint8)
    diff_rgb[zs != ttpa] = (255, 80, 80)   # red = changed

    panels = [
        ("DOP20 RGB",                    rgb,          False),
        ("Zero-shot",                    colorize(zs), True),
        ("TTPA",                         colorize(ttpa), True),
        (f"Diff ({changed_pct:.1f}% px)", diff_rgb,    False),
    ]

    # Save individual files
    for name, mask, has_legend in [("rgb", rgb, False), ("zs", colorize(zs), True),
                                    ("ttpa", colorize(ttpa), True), ("diff", diff_rgb, False)]:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(mask); ax.axis("off")
        ax.set_title(f"{tile_id} — {dict(rgb='DOP20 RGB', zs='Zero-shot', ttpa='TTPA', diff=f'Diff ({changed_pct:.1f}% changed)')[name]}", fontsize=10)
        if has_legend:
            ax.legend(handles=legend, loc="upper right", fontsize=7,
                      title="Classes", framealpha=0.85, bbox_to_anchor=(1,1))
        path = out_dir / f"{tile_id}_{name}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.name}")

    # Save 4-column comparison
    fig, axes = plt.subplots(1, 4, figsize=(32, 10))
    for ax, (label, img, has_leg) in zip(axes, panels):
        ax.imshow(img); ax.axis("off"); ax.set_title(label, fontsize=11)
        if has_leg:
            ax.legend(handles=legend, loc="lower right", fontsize=6,
                      title="Classes", framealpha=0.8)
    fig.suptitle(tile_id, fontsize=12)
    fig.tight_layout()
    path = out_dir / f"{tile_id}_compare.png"
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def load_rgb_patch(img_dir: Path, stem: str) -> np.ndarray:
    """Load RGB patch from TIFF/JPG, take first 3 bands, return uint8 HWC array."""
    for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            img = np.array(Image.open(p))
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            return img[:, :, :3]
    return np.zeros((PATCH, PATCH, 3), dtype=np.uint8)  # black if missing


def main():
    ttpa_files = sorted(PRED_DIR.glob("*_ttpa.npy"))
    if not ttpa_files:
        raise FileNotFoundError(f"No *_ttpa.npy in {PRED_DIR}")
    print(f"Found {len(ttpa_files)} TTPA patches")

    # Group by tile
    tiles: dict = {}
    for f in ttpa_files:
        stem = f.stem.replace("_ttpa", "")
        tile_id, y, x = parse_stem(stem)
        tiles.setdefault(tile_id, {})[( y, x)] = stem

    print(f"Tiles: {sorted(tiles.keys())}")

    for tile_id, patch_stems in tiles.items():
        print(f"\nStitching {tile_id} ({len(patch_stems)} patches)...")
        zs_patches   = {}
        ttpa_patches = {}
        rgb_patches  = {}
        for (y, x), stem in patch_stems.items():
            zs_patches[(y, x)]   = np.load(PRED_DIR / f"{stem}_zs.npy")
            ttpa_patches[(y, x)] = np.load(PRED_DIR / f"{stem}_ttpa.npy")
            rgb_patches[(y, x)]  = load_rgb_patch(IMG_DIR, stem)

        rgb_full  = stitch_tile(rgb_patches,  fill=0, dtype=np.uint8)
        zs_full   = stitch_tile(zs_patches)
        ttpa_full = stitch_tile(ttpa_patches)
        save_stitched(tile_id, rgb_full, zs_full, ttpa_full, OUT_DIR)

    print(f"\nAll stitched tiles saved to {OUT_DIR}")
    print("View with:")
    print("  from IPython.display import Image, display")
    print("  import glob")
    print("  for f in sorted(glob.glob(str(OUT_DIR / '*_compare.png'))): display(Image(f, width=1200))")


if __name__ == "__main__":
    main()
