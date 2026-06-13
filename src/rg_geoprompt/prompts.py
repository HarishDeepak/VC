"""CLIP prompt ensemble and text prototype encoding.

The 5×6 prompt ensemble and the encode-average-renormalize logic are ports
of notebook CELL27 — confirmed run on Kaggle (text_embeddings.pt exists).

KEY RULE (handoff): text_embeddings.pt must be REUSED if it already exists.
CLIP encoding only reruns when the file is missing — that is exactly what
``load_or_encode_text_embeddings()`` enforces. Delete the file manually
(or pass force=True) only when you deliberately change the prompts.
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from . import paths
from .constants import CLIP_NAME, CLIP_PRETRAINED, DEVICE

# 5 viewpoint-specific prompts per class (GeoPriorCLIP / RemoteCLIP-inspired).
# Mandatory keywords per handoff: "aerial view", "top-down",
# "satellite photo", "from above".
PROMPT_ENSEMBLE = {
    0: ["An aerial view of impervious surfaces including paved roads and concrete.",
        "A top-down satellite photo of grey asphalt streets and concrete walkways.",
        "Aerial view of concrete walkways and paved urban surfaces.",
        "Satellite image of road network and parking lots from above.",
        "Top-down view of sealed urban ground surfaces."],
    1: ["A satellite photo of building rooftops viewed from directly above.",
        "An aerial image showing geometric roofs of urban houses.",
        "Top-down view of building structures in a city.",
        "Satellite photo of flat and pitched rooftops in urban area.",
        "Aerial view of densely packed building blocks."],
    2: ["An aerial view of low vegetation including flat green grass and lawns.",
        "A satellite photo of flat green areas and shrubs.",
        "Top-down view of ground-level plants and low vegetation.",
        "Aerial image of grass fields and low plant cover.",
        "Satellite view of green lawn surfaces in urban setting."],
    3: ["A top-down view of a tree canopy showing textured green foliage.",
        "An aerial image of tall trees and dense forest canopies.",
        "Satellite photo of tree crowns casting shadows.",
        "Top-down view of dense vegetation canopy.",
        "Aerial view of park trees and woodland from above."],
    4: ["A satellite photo of cars parked on a road viewed from above.",
        "An aerial view of rectangular vehicles in an urban area.",
        "Top-down satellite view of cars and vans on streets.",
        "Aerial image of parked vehicles in parking lots.",
        "Satellite view of small metallic objects on roads."],
    5: ["An aerial view of undefined urban clutter and background objects.",
        "A satellite photo of unclassified terrain and mixed surfaces.",
        "Top-down view of miscellaneous urban features.",
        "Aerial image of undefined mixed land cover areas.",
        "Satellite view of background objects and undefined surfaces."],
}

# Darmstadt-specific per-prompt ensemble for max-pool inference.
# Building prompts cover red terracotta (dominant in Darmstadt) AND gray flat
# roofs (Potsdam distribution) — max-pool picks whichever fires strongest.
# Clutter prompts are explicitly ground-level to avoid absorbing red rooftops.
DARMSTADT_PROMPTS = {
    0: ["An aerial view of impervious surfaces including paved roads and concrete.",
        "A top-down satellite photo of grey asphalt streets and concrete walkways.",
        "Aerial view of concrete walkways and paved urban surfaces.",
        "Satellite image of road network and parking lots from above.",
        "Top-down view of sealed urban ground surfaces."],
    1: ["Aerial satellite view of a building with a red terracotta pitched roof.",
        "Top-down view of a house with bright red clay roof tiles.",
        "Aerial view of a residential building featuring a red ceramic tiled rooftop.",
        "Satellite imagery of a building with a dark gray flat concrete roof.",
        "Aerial view of a building roof constructed with red brick or terracotta materials.",
        "Top-down view of building structures in a German city from above.",
        "Satellite photo of flat and pitched rooftops in urban area.",
        "Aerial photograph of rooftops from above including red and gray surfaces."],
    2: ["An aerial view of low vegetation including flat green grass and lawns.",
        "A satellite photo of flat green areas and shrubs.",
        "Top-down view of ground-level plants and low vegetation.",
        "Aerial image of grass fields and low plant cover.",
        "Satellite view of green lawn surfaces in urban setting."],
    3: ["A top-down view of a tree canopy showing textured green foliage.",
        "An aerial image of tall trees and dense forest canopies.",
        "Satellite photo of tree crowns casting shadows.",
        "Top-down view of dense vegetation canopy.",
        "Aerial view of park trees and woodland from above."],
    4: ["A satellite photo of cars parked on a road viewed from above.",
        "An aerial view of rectangular vehicles in an urban area.",
        "Top-down satellite view of cars and vans on streets.",
        "Aerial image of parked vehicles in parking lots.",
        "Satellite view of small metallic objects on roads."],
    5: ["Aerial view of unpaved dirt, mud, or brown soil on the ground.",
        "Top-down view of a body of water, river, or dark background surface.",
        "Satellite image of miscellaneous ground debris and urban background clutter.",
        "Aerial view of a paved ground surface with scattered miscellaneous objects.",
        "Top-down satellite photo of bare earth, gravel, or construction ground."],
}


def encode_ensemble_prompts(prompt_dict: dict = None,
                            device: torch.device = DEVICE) -> torch.Tensor:
    """Encode the prompt ensemble with frozen CLIP ViT-B/32.

    Per class: encode 5 prompts → L2-normalize each → average → re-normalize.
    Re-normalization after averaging is mandatory: the mean of unit vectors
    is not itself a unit vector.

    Returns:
        Tensor [6, 512] of L2-normalized class prototypes (CPU not enforced).
    """
    import open_clip  # imported lazily — only needed when actually encoding

    prompt_dict = prompt_dict or PROMPT_ENSEMBLE
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_NAME, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_NAME)

    embeddings = []
    for cls_id in range(6):
        tokens = tokenizer(prompt_dict[cls_id]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)   # [5, 512]
            emb = F.normalize(emb, dim=-1)
            emb = emb.mean(dim=0)                  # [512]
            emb = F.normalize(emb, dim=-1)         # re-normalize after avg
        embeddings.append(emb)
    return torch.stack(embeddings)                 # [6, 512]


def encode_per_prompt(prompt_dict: dict = None,
                      device: torch.device = DEVICE) -> torch.Tensor:
    """Encode prompts individually — no averaging — for max-pool inference.

    Returns:
        Tensor [6, N, 512] where N = number of prompts per class.
        Classes with fewer prompts are padded by repeating the last entry.
    """
    import open_clip

    prompt_dict = prompt_dict or DARMSTADT_PROMPTS
    n_max = max(len(v) for v in prompt_dict.values())

    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_NAME, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_NAME)

    per_class = []
    for cls_id in range(6):
        prompts = prompt_dict[cls_id]
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)   # [N, 512]
            emb = F.normalize(emb, dim=-1)
        # Pad to n_max by repeating last entry
        if len(prompts) < n_max:
            pad = emb[-1:].expand(n_max - len(prompts), -1)
            emb = torch.cat([emb, pad], dim=0)
        per_class.append(emb)
    return torch.stack(per_class)                  # [6, N_max, 512]


def load_or_encode_text_embeddings(path: Path = None, force: bool = False,
                                   device: torch.device = DEVICE
                                   ) -> torch.Tensor:
    """Load text_embeddings.pt if present; encode with CLIP only if missing.

    This is the ONLY entry point notebooks should use — it enforces the
    reuse rule so prompt encoding never silently reruns and overwrites.

    Args:
        path:  override the canonical path (default: WORK_DIR/text_embeddings.pt)
        force: set True only when prompts changed and re-encoding is intended.

    Returns:
        Tensor [6, 512] on ``device``.
    """
    path = Path(path) if path else paths.TEXT_EMBEDDINGS_PT
    if path.exists() and not force:
        te = torch.load(path, map_location=device)
        print(f"✓ text embeddings loaded from {path} (no CLIP rerun)")
        return te
    te = encode_ensemble_prompts(device=device)
    torch.save(te.cpu(), path)
    print(f"✓ text embeddings encoded and saved to {path}")
    return te.to(device)
