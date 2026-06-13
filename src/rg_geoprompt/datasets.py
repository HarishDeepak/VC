"""Datasets: Potsdam patches (confirmed) and Darmstadt DOP20 (draft).

PotsdamDataset and the tile-level split are ports of notebook CELL6/CELL8 —
confirmed working. Darmstadt code is built from the handoff spec and has not
yet run on Kaggle because DOP20 has not been downloaded.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from . import paths
from .augment import train_augment
from .constants import (BATCH_SIZE, DOP20_CRS_EPSG, NUM_WORKERS,
                        PATCH_SIZE, PATCH_STRIDE, VAL_TILE)


# ════════════════════════════════════════════════════════════════════════════
# Potsdam (training) — SOURCE: notebook CELL6/CELL8/CELL9, confirmed working
# ════════════════════════════════════════════════════════════════════════════
class PotsdamDataset(Dataset):
    """Pre-sliced 512×512 ISPRS Potsdam patches.

    Returns:
        image : float32 tensor [3, 512, 512], values in [0, 1]
        label : int64 tensor [512, 512], values in {0,1,2,3,4,5,255}
    """

    def __init__(self, img_dir: Path, label_dir: Path,
                 file_stems: List[str], augment: bool = False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.stems = file_stems
        self.augment = augment

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = Image.open(self.img_dir / f"{stem}.png").convert("RGB")
        img_tensor = TF.to_tensor(img)  # /255 → float32 [3, H, W]
        lbl = Image.open(self.label_dir / f"{stem}.png")
        lbl_tensor = torch.from_numpy(np.array(lbl, dtype=np.int64))
        if self.augment:
            img_tensor, lbl_tensor = train_augment(img_tensor, lbl_tensor)
        return img_tensor, lbl_tensor


def tile_id_from_stem(stem: str) -> str:
    """'top_potsdam_6_15_y256_x512' -> '6_15'."""
    parts = stem.split("_")
    return f"{parts[2]}_{parts[3]}"


def tile_level_split(img_dir: Path = None, val_tile: str = VAL_TILE
                     ) -> Tuple[List[str], List[str]]:
    """Tile-level train/val split. Random split is FORBIDDEN: with a
    256-pixel stride, patches from the same tile overlap by 50%, so a
    random split leaks near-identical patches into validation.

    Returns:
        (train_stems, val_stems) — expected 2420 / 484 for val_tile='6_15'.
    """
    img_dir = Path(img_dir) if img_dir else paths.IMG_DIR
    all_stems = [f.stem for f in sorted(img_dir.glob("*.png"))]
    train_stems = [s for s in all_stems if tile_id_from_stem(s) != val_tile]
    val_stems = [s for s in all_stems if tile_id_from_stem(s) == val_tile]
    return train_stems, val_stems


def build_potsdam_loaders(batch_size: int = BATCH_SIZE,
                          num_workers: int = NUM_WORKERS):
    """Standard train/val DataLoaders (confirmed configuration).

    Returns:
        (train_loader, val_loader)
    """
    train_stems, val_stems = tile_level_split()
    train_ds = PotsdamDataset(paths.IMG_DIR, paths.LABEL_DIR,
                              train_stems, augment=True)
    val_ds = PotsdamDataset(paths.IMG_DIR, paths.LABEL_DIR,
                            val_stems, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=False)
    return train_loader, val_loader


# ════════════════════════════════════════════════════════════════════════════
# Darmstadt DOP20 — SOURCE: handoff spec — not yet run on Kaggle
# ════════════════════════════════════════════════════════════════════════════
class DarmstadtPatchDataset(Dataset):
    """Pre-sliced 512×512 Darmstadt DOP20 RGB patches (no labels).

    Evaluation is patch-based by design — predictions are NEVER stitched
    back into full tiles (avoids seam artifacts; see handoff §8).

    Returns:
        image : float32 tensor [3, 512, 512] in [0, 1]
        stem  : str — patch identifier, used to match OSM pseudo-GT masks
    """

    def __init__(self, img_dir: Path = None, stems: Optional[List[str]] = None):
        self.img_dir = Path(img_dir) if img_dir else paths.DARMSTADT_IMG_DIR
        if stems is None:
            stems = [f.stem for f in sorted(self.img_dir.glob("*.png"))]
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img = Image.open(self.img_dir / f"{stem}.png").convert("RGB")
        return TF.to_tensor(img), stem


def slice_dop20_tile(tif_path: Path, out_img_dir: Path,
                     out_ndvi_dir: Optional[Path] = None,
                     patch: int = PATCH_SIZE, stride: int = PATCH_STRIDE
                     ) -> int:
    """Slice one DOP20 GeoTIFF into 512×512 RGB PNG patches.

    SOURCE: handoff spec — not yet run on Kaggle.
    Intended to run LOCALLY (rasterio needed), before uploading the patch
    folder as Kaggle dataset 'darmstadt-dop20'.

    DOP20 is 4-channel RGBI — only the first 3 bands (RGB) are kept,
    because DINOv2 expects 3-channel input. If ``out_ndvi_dir`` is given,
    an NDVI mask (NDVI = (NIR - R) / (NIR + R)) is saved per patch as a
    free weak vegetation label.

    Patch filenames encode tile stem + pixel offsets:
        '<tile_stem>_y<row>_x<col>.png'
    and each patch's affine geotransform is appended to
    ``out_img_dir/transforms.json`` ({stem: [a,b,c,d,e,f]}) so OSM
    rasterization on Kaggle works without the original GeoTIFFs.

    Returns:
        number of patches written.
    """
    import json
    import rasterio  # local-only dependency for this step
    from rasterio.transform import Affine

    out_img_dir.mkdir(parents=True, exist_ok=True)
    if out_ndvi_dir is not None:
        out_ndvi_dir.mkdir(parents=True, exist_ok=True)

    tf_json = out_img_dir / "transforms.json"
    transforms = json.loads(tf_json.read_text()) if tf_json.exists() else {}

    n = 0
    with rasterio.open(tif_path) as src:
        assert src.count >= 3, (
            f"{tif_path.name}: expected >= 3 bands (RGBI), got {src.count}")
        # .jgw world files don't embed CRS; Hessen DOP20 is always EPSG:25832
        if src.crs is None and tif_path.suffix.lower() in (".jpg", ".jpeg"):
            pass  # CRS assumed EPSG:25832 from Hessen tile naming convention
        else:
            assert src.crs is not None and src.crs.to_epsg() == DOP20_CRS_EPSG, (
                f"{tif_path.name}: expected EPSG:{DOP20_CRS_EPSG}, "
                f"got {src.crs}. Reproject before slicing.")
        data = src.read()                      # [bands, H, W], uint8/uint16
        rgb = data[:3]                          # RGBI → RGB only
        nir = data[3].astype(np.float32) if data.shape[0] >= 4 else None
        H, W = rgb.shape[1], rgb.shape[2]
        for y in range(0, H - patch + 1, stride):
            for x in range(0, W - patch + 1, stride):
                tile = rgb[:, y:y + patch, x:x + patch]
                arr = np.transpose(tile, (1, 2, 0))
                if arr.dtype != np.uint8:       # DOP20 sometimes 16-bit
                    arr = (arr / arr.max() * 255).astype(np.uint8)
                stem = f"{tif_path.stem}_y{y}_x{x}"
                Image.fromarray(arr).save(out_img_dir / f"{stem}.png")
                patch_tf = src.transform * Affine.translation(x, y)
                transforms[stem] = list(patch_tf)[:6]
                if nir is not None and out_ndvi_dir is not None:
                    r = rgb[0, y:y + patch, x:x + patch].astype(np.float32)
                    nr = nir[y:y + patch, x:x + patch]
                    ndvi = (nr - r) / (nr + r + 1e-6)
                    np.save(out_ndvi_dir / f"{stem}.npy",
                            ndvi.astype(np.float16))
                n += 1
    tf_json.write_text(json.dumps(transforms))
    return n
