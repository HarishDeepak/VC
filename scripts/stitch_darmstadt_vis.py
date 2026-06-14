"""Stitch Darmstadt patch predictions into full-tile visualisation PNGs.

Run in a Kaggle cell (predictions must be in /kaggle/working/darmstadt_preds/):

    %run /kaggle/working/VC/rg-geoprompt-peft/scripts/stitch_darmstadt_vis.py

Or locally with PRED_DIR / OUT_DIR overridden via env vars.

Output (visualisation ONLY — never used for metrics):
    /kaggle/working/darmstadt_stitched/<tile_id>_zs.png
    /kaggle/working/darmstadt_stitched/<tile_id>_ttpa.png
    /kaggle/working/darmstadt_stitched/<tile_id>_diff.png   (pixels that changed)
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


def stitch_tile(patches: dict) -> np.ndarray:
    """patches: {(y,x): [H,W] array} → stitched canvas"""
    max_y = max(y for y, x in patches) + PATCH
    max_x = max(x for y, x in patches) + PATCH
    canvas = np.full((max_y, max_x), 255, dtype=np.uint8)
    for (y, x), patch in patches.items():
        canvas[y:y+PATCH, x:x+PATCH] = patch
    return canvas


def save_stitched(tile_id: str, zs: np.ndarray, ttpa: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    legend = [mpatches.Patch(color=tuple(c/255 for c in col), label=name)
              for name, col in zip(CLASS_NAMES, CLASS_COLORS)]
    legend += [mpatches.Patch(color=(0.5,0.5,0.5), label="Boundary/Unknown")]

    diff = (zs != ttpa).astype(np.uint8) * 200  # white where changed

    for name, mask in [("zs", zs), ("ttpa", ttpa), ("diff", diff)]:
        img = colorize(mask) if name != "diff" else np.stack([diff]*3, axis=-1)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)
        ax.axis("off")
        label = {"zs": "Zero-shot", "ttpa": "TTPA", "diff": "Changed pixels (ZS→TTPA)"}[name]
        ax.set_title(f"{tile_id} — {label}", fontsize=10)
        if name != "diff":
            ax.legend(handles=legend, loc="upper right", fontsize=7,
                      title="Classes", framealpha=0.85, bbox_to_anchor=(1,1))
        path = out_dir / f"{tile_id}_{name}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.name}")


def main():
    ttpa_files = sorted(PRED_DIR.glob("*_ttpa.npy"))
    if not ttpa_files:
        raise FileNotFoundError(f"No *_ttpa.npy in {PRED_DIR}")
    print(f"Found {len(ttpa_files)} TTPA patches")

    # Group by tile
    tiles: dict = {}
    for f in ttpa_files:
        stem_ttpa = f.stem.replace("_ttpa", "")
        tile_id, y, x = parse_stem(stem_ttpa)
        if tile_id not in tiles:
            tiles[tile_id] = {}
        zs_path = PRED_DIR / f"{stem_ttpa}_zs.npy"
        tiles[tile_id][(y, x)] = (np.load(f), np.load(zs_path))

    print(f"Tiles: {sorted(tiles.keys())}")

    for tile_id, patches in tiles.items():
        print(f"\nStitching {tile_id} ({len(patches)} patches)...")
        zs_patches   = {k: v[1] for k, v in patches.items()}
        ttpa_patches = {k: v[0] for k, v in patches.items()}
        zs_full   = stitch_tile(zs_patches)
        ttpa_full = stitch_tile(ttpa_patches)
        save_stitched(tile_id, zs_full, ttpa_full, OUT_DIR)

    print(f"\nAll stitched tiles saved to {OUT_DIR}")
    print("View with:")
    print("  from IPython.display import Image, display")
    print("  import glob")
    print("  for f in sorted(glob.glob(str(OUT_DIR / '*.png'))): display(Image(f, width=900))")


if __name__ == "__main__":
    main()
