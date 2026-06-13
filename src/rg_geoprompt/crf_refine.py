"""SAM-encoder-driven DenseCRF boundary refinement.

Pipeline (encoder-only — no SAM decoder, no AMG, no majority-voting):
  1. SAM-ViT-B image encoder → [1, 256, 64, 64] features (at 1024px native)
  2. Upsample to [H, W, 256]
  3. PCA → [H, W, n_pca] (default 5), scaled to uint8
  4. DenseCRF: DINOv2 logits as unary, SAM-PCA features as bilateral term
  5. Return refined prediction [H, W]

Why encoder-only: AMG + majority-voting causes pathological over-segmentation
on aerial imagery (rooftop HVAC, shadow boundaries) and shadow snapping.
Using SAM latent features as the CRF bilateral kernel avoids rigid mask
boundaries while exploiting SAM's deep geometric awareness.

Memory: SAM-ViT-B at 1024px ≈ 3–4 GB VRAM. Run sequentially after DINOv2
inference (unload DINOv2 first) or concurrently if VRAM allows.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

SAM_VIT_B_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
SAM_VIT_B_FILENAME = "sam_vit_b_01ec64.pth"


def ensure_sam_checkpoint(work_dir: Path) -> Path:
    """Download SAM-ViT-B checkpoint if not present. Returns local path."""
    dst = work_dir / SAM_VIT_B_FILENAME
    if dst.exists():
        print(f"✓ SAM checkpoint present: {dst}")
        return dst
    import urllib.request
    print(f"Downloading SAM-ViT-B (~375 MB) …")
    urllib.request.urlretrieve(SAM_VIT_B_URL, dst)
    print(f"✓ SAM checkpoint saved: {dst}")
    return dst


def load_sam_encoder(ckpt_path: Path,
                     device: torch.device = torch.device("cuda")) -> torch.nn.Module:
    """Load SAM-ViT-B and return only the image encoder (frozen, eval mode).

    The mask decoder and prompt encoder are discarded to save VRAM.
    Runs in float16 on GPU.
    """
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt_path))
    encoder = sam.image_encoder.to(device).eval()
    if device.type == "cuda":
        encoder = encoder.half()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"✓ SAM-ViT-B encoder loaded "
          f"({sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M params)")
    return encoder


def _sam_preprocess(rgb_np: np.ndarray, img_size: int = 1024) -> torch.Tensor:
    """Resize + normalize a uint8 HxWx3 numpy image for SAM. Returns [1,3,H,W]."""
    from segment_anything.utils.transforms import ResizeLongestSide
    transform = ResizeLongestSide(img_size)
    resized = transform.apply_image(rgb_np)                  # HxWx3 uint8
    tensor = torch.as_tensor(resized, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)            # [1,3,H,W]
    # SAM pixel mean/std from original training
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    pixel_std  = torch.tensor([58.395,  57.12,  57.375]).view(1, 3, 1, 1)
    tensor = (tensor - pixel_mean) / pixel_std
    # Pad to square img_size × img_size
    h, w = tensor.shape[-2:]
    pad_h = img_size - h
    pad_w = img_size - w
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    return tensor


@torch.no_grad()
def extract_sam_features(encoder: torch.nn.Module,
                         rgb_np: np.ndarray,
                         out_size: tuple = (512, 512)) -> np.ndarray:
    """Run SAM image encoder on a uint8 HxWx3 RGB patch.

    Returns float32 numpy array [out_H, out_W, 256] — encoder features
    bilinearly upsampled to out_size.
    """
    device = next(encoder.parameters()).device
    x = _sam_preprocess(rgb_np).to(device)
    if next(encoder.parameters()).dtype == torch.float16:
        x = x.half()
    feats = encoder(x)                          # [1, 256, 64, 64]
    feats = feats.float()
    feats = F.interpolate(feats, size=out_size,
                          mode="bilinear", align_corners=False)
    return feats[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 256]


def _pca_to_uint8(feats: np.ndarray, n_components: int = 5) -> np.ndarray:
    """PCA-reduce [H, W, D] → [H, W, n_components] scaled to uint8."""
    from sklearn.decomposition import PCA
    H, W, D = feats.shape
    flat = feats.reshape(-1, D)
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(flat)            # [H*W, n_components]
    reduced -= reduced.min(0)
    mx = reduced.max(0)
    mx[mx < 1e-8] = 1.0
    reduced = (reduced / mx * 255).astype(np.uint8)
    return reduced.reshape(H, W, n_components)


def refine_with_sam_crf(logits_np: np.ndarray,
                        sam_feats: np.ndarray,
                        n_classes: int = 6,
                        n_pca: int = 5,
                        iters: int = 5,
                        sxy_smooth: float = 3.0,
                        compat_smooth: float = 3.0,
                        sxy_bil: float = 60.0,
                        srgb_bil: float = 13.0,
                        compat_bil: float = 10.0) -> np.ndarray:
    """DenseCRF refinement using SAM features as the bilateral appearance term.

    Args:
        logits_np:   [n_classes, H, W] float32 raw logits from DINOv2 model.
        sam_feats:   [H, W, 256] float32 from extract_sam_features().
        n_pca:       PCA components passed to pydensecrf bilateral (3–5 recommended).
        iters:       mean-field inference iterations (5 is standard).
        sxy_smooth:  spatial sigma for smoothness kernel.
        sxy_bil:     spatial sigma for bilateral kernel.
        srgb_bil:    feature sigma for bilateral kernel.
        compat_*:    label compatibility weights.

    Returns:
        [H, W] int32 refined prediction map.
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    from scipy.special import softmax

    n_cls, H, W = logits_np.shape
    probs = softmax(logits_np, axis=0).astype(np.float32)     # [C, H, W]
    U = unary_from_softmax(probs)                              # [C, H*W]

    feats_u8 = _pca_to_uint8(sam_feats, n_pca)               # [H, W, n_pca]

    d = dcrf.DenseCRF2D(W, H, n_cls)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_smooth, compat=compat_smooth)
    d.addPairwiseBilateral(sxy=sxy_bil, srgb=srgb_bil,
                           rgbim=feats_u8, compat=compat_bil)
    Q = d.inference(iters)
    return np.argmax(np.array(Q).reshape(n_cls, H, W), axis=0).astype(np.int32)
