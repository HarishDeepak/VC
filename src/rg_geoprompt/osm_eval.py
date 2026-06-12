"""OSM pseudo-ground-truth for Darmstadt evaluation.

SOURCE: handoff spec — not yet run on Kaggle (DOP20 not downloaded; OSM
rasterization never executed). The erosion function is the confirmed
handoff snippet; everything else is a clean implementation of the spec.

Pipeline:
    1. Download OSM features for Darmstadt via osmnx (buildings, highways,
       woods, grass).
    2. Reproject EPSG:4326 → EPSG:25832 to match DOP20.
    3. Rasterize onto each DOP20 patch's geotransform → class mask.
    4. Erode with a 3×3 kernel: uncertain border pixels become 255.
       This is the ONLY place where boundary-label CREATION differs from
       Potsdam. The 255-EXCLUSION rule in metrics is unchanged.
    5. Per-patch F1 (6 classes, ignore 255) → average over patches.
       Report explicitly as "OSM pseudo-GT F1" — OSM is noisy.

OSM → Potsdam class mapping (handoff):
    building=True              → 1 (Building)
    highway=*                  → 0 (Impervious)
    natural=wood / forest      → 3 (Tree)
    landuse=grass / grassland  → 2 (Low Veg)
Pixels covered by no OSM feature get 255 (unknown — excluded), NOT clutter:
OSM absence is missing data, not evidence of clutter.
"""
from typing import Dict, Optional

import numpy as np

from .constants import DOP20_CRS_EPSG, OSM_EROSION_KERNEL, OSM_TAGS


def download_osm_darmstadt(place: str = "Darmstadt, Germany",
                           tags: Optional[dict] = None):
    """Download OSM features and reproject to EPSG:25832.

    Returns:
        GeoDataFrame in EPSG:25832. Requires osmnx (run locally or
        `pip install osmnx` on Kaggle with internet enabled).
    """
    import osmnx as ox
    gdf = ox.features_from_place(place, tags or OSM_TAGS)
    return gdf.to_crs(epsg=DOP20_CRS_EPSG)


def _osm_class_geoms(gdf):
    """Split the GeoDataFrame into per-class geometry lists.

    Rasterization order matters: later classes overwrite earlier ones.
    We burn in order grass(2) → wood(3) → highway(0) → building(1) so
    buildings (most reliable OSM layer) win conflicts.
    """
    out = {}
    if "landuse" in gdf.columns or "natural" in gdf.columns:
        grass = gdf[(gdf.get("landuse") == "grass") |
                    (gdf.get("natural") == "grassland")]
        out[2] = list(grass.geometry.dropna())
        wood = gdf[(gdf.get("natural") == "wood") |
                   (gdf.get("landuse") == "forest")]
        out[3] = list(wood.geometry.dropna())
    if "highway" in gdf.columns:
        hw = gdf[gdf["highway"].notna()]
        # buffer line geometries to plausible road width (4 m half-width)
        out[0] = [g.buffer(4.0) if g.geom_type == "LineString" else g
                  for g in hw.geometry.dropna()]
    if "building" in gdf.columns:
        out[1] = list(gdf[gdf["building"].notna()].geometry.dropna())
    return out


def rasterize_osm_patch(gdf, transform, shape=(512, 512)) -> np.ndarray:
    """Rasterize OSM vectors onto one patch's pixel grid.

    Args:
        gdf:       GeoDataFrame in EPSG:25832 (from download_osm_darmstadt).
        transform: affine geotransform of THIS patch (derive from the tile
                   transform shifted by the patch's y/x pixel offset:
                   ``tile_transform * Affine.translation(x, y)``).
        shape:     patch shape, (512, 512).

    Returns:
        int64 array [H, W] with values in {0,1,2,3, 255}. 255 = no OSM
        coverage (unknown). Classes 4 (Car) and 5 (Clutter) never appear —
        OSM has no such layers; they remain 255 and are excluded.
    """
    from rasterio import features

    mask = np.full(shape, 255, dtype=np.int64)
    geoms = _osm_class_geoms(gdf)
    for cls in (2, 3, 0, 1):                 # burn order — buildings last
        if cls in geoms and geoms[cls]:
            features.rasterize(
                ((g, cls) for g in geoms[cls]),
                out=mask, transform=transform,
                default_value=cls,
            )
    return mask


def erode_osm_labels(osm_label: np.ndarray,
                     kernel_size: int = OSM_EROSION_KERNEL) -> np.ndarray:
    """Mark uncertain OSM border pixels as 255 via morphological erosion.

    SOURCE: handoff §12 — confirmed spec snippet (3×3 kernel, 1 iteration).
    If OSM misalignment is visible, increase kernel to 5×5 by raising the
    iteration count / structure size (see RISKS.md).
    """
    from scipy.ndimage import binary_erosion
    eroded = osm_label.copy()
    iterations = max(1, (kernel_size - 1) // 2)
    for cls in range(6):
        cls_mask = (osm_label == cls)
        if not cls_mask.any():
            continue
        eroded_mask = binary_erosion(cls_mask, iterations=iterations)
        eroded[~eroded_mask & cls_mask] = 255   # border → ignore
    return eroded


def osm_pseudo_gt_f1(pred_masks: Dict[str, np.ndarray],
                     osm_masks: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Average per-patch F1 of predictions vs eroded OSM masks.

    Patch-based by design: predictions are NEVER stitched into full tiles.

    Args:
        pred_masks: {stem: [512,512] int array of predicted class IDs}
        osm_masks:  {stem: [512,512] int array, already ERODED (255 borders)}

    Returns:
        {"per_class_f1": tensor [6], "mean_f1": float, "n_patches": int}
        Report as "OSM pseudo-GT F1" — never as ground-truth F1.
    """
    import torch
    from .metrics import f1_from_arrays

    stems = sorted(set(pred_masks) & set(osm_masks))
    if not stems:
        raise ValueError("no overlapping patch stems between preds and OSM")
    f1s = torch.stack([f1_from_arrays(pred_masks[s], osm_masks[s])
                       for s in stems])
    per_class = f1s.mean(dim=0)
    return {"per_class_f1": per_class,
            "mean_f1": per_class.mean().item(),
            "n_patches": len(stems)}
