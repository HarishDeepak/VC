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


# 12-class open-vocabulary set for Hessen DOP20 zero-shot inference.
# Classes chosen for CLIP discriminability at 20cm GSD aerial view.
# 'car' excluded — too small at 20cm (~4×2px). 'farm road' excluded — CLIP
# cannot separate road use from road texture at this resolution.
HESSEN_PROMPTS = {
    0: ["Aerial satellite view of a residential building with a red terracotta pitched roof.",
        "Top-down view of a house with bright red clay roof tiles in a German city.",
        "Aerial view of a residential rooftop with red ceramic tiles from above.",
        "Satellite image of dense German residential buildings with red tiled rooftops.",
        "Top-down photo of red-roofed houses in an urban neighbourhood."],
    1: ["Aerial satellite view of a commercial building with a flat grey concrete roof.",
        "Top-down view of an industrial warehouse with a large flat metal roof.",
        "Satellite image of a large building with a grey or silver flat rooftop.",
        "Aerial view of an office block with a wide flat roof and rooftop equipment.",
        "Top-down photo of a department store or shopping centre with flat grey roof."],
    2: ["Aerial view of a paved road or street with lane markings.",
        "Top-down satellite photo of grey asphalt streets and concrete walkways.",
        "Satellite image of an urban road network with sidewalks.",
        "Aerial view of a wide boulevard with tarmac surface.",
        "Top-down view of a paved urban street with road markings."],
    3: ["Aerial satellite view of a large parking lot with white lane markings.",
        "Top-down view of an open-air car park filled with parked cars.",
        "Satellite image of a parking area with regularly spaced vehicles.",
        "Aerial view of grey paved surface with lane markings and parked cars.",
        "Top-down photo of an off-street parking lot with vehicle spaces."],
    4: ["Aerial satellite view of a railway track with parallel steel rails.",
        "Top-down view of a train line with metal tracks and gravel ballast.",
        "Satellite image of railway lines and rail infrastructure.",
        "Aerial view of parallel rail tracks running through an urban area.",
        "Top-down photo of a train line with track bed and rail sleepers."],
    5: ["A top-down view of a tree canopy showing textured green foliage.",
        "An aerial image of tall trees and dense forest canopies.",
        "Satellite photo of tree crowns casting shadows on the ground.",
        "Top-down view of dense green vegetation canopy.",
        "Aerial view of park trees and woodland from above."],
    6: ["An aerial view of low green grass and flat lawns in a park or garden.",
        "Top-down satellite photo of flat green grass surfaces.",
        "Aerial image of grass fields and low plant cover.",
        "Satellite view of green lawn surfaces in an urban setting.",
        "Top-down view of mowed grass and garden greenery."],
    7: ["An aerial view of a large agricultural crop field.",
        "Top-down satellite photo of cultivated farmland with row crops.",
        "Aerial image of a field with uniform crop texture.",
        "Satellite view of agricultural land with planting rows.",
        "Top-down view of an arable field from above."],
    8: ["Aerial view of a greenhouse with a glass or translucent plastic roof.",
        "Top-down satellite image of greenhouse structures with bright reflective roofs.",
        "Aerial photo of agricultural greenhouses arranged in rows.",
        "Satellite view of large white or semi-transparent greenhouse panels.",
        "Top-down view of greenhouse growing structures from above."],
    9: ["Aerial view of bare brown soil or gravel at a construction site.",
        "Top-down satellite image of exposed earth or sandy ground.",
        "Aerial photo of bare unpaved ground without vegetation.",
        "Satellite view of a brownfield site or bare agricultural land.",
        "Top-down view of loose gravel or dirt surface."],
    10: ["Aerial satellite view of a river or water body with dark water surface.",
        "Top-down view of a lake or stream with blue or grey water.",
        "Satellite image of a river channel with flowing water.",
        "Aerial view of a pond or canal with dark reflective water surface.",
        "Top-down photo of open water including lakes rivers and reservoirs."],
    11: ["An aerial view of undefined urban clutter and background objects.",
        "A satellite photo of unclassified terrain and mixed surfaces.",
        "Top-down view of miscellaneous urban features and debris.",
        "Aerial image of undefined mixed land cover areas.",
        "Satellite view of background objects and undefined urban surfaces."],
}

# Human-readable class names matching HESSEN_PROMPTS keys (for colourmap / legend)
HESSEN_CLASS_NAMES = [
    "residential building",
    "commercial/industrial building",
    "road/street",
    "parking lot",
    "railway track",
    "tree canopy",
    "grass/lawn",
    "agricultural field",
    "greenhouse",
    "bare soil/gravel",
    "water body",
    "clutter",
]

# Colour palette for HESSEN_PROMPTS (RGB 0-255)
HESSEN_PALETTE = [
    (210,  80,  70),  # 0 residential building — terracotta red
    (130, 130, 130),  # 1 commercial building — grey
    ( 60,  60,  60),  # 2 road — dark grey
    (180, 180, 180),  # 3 parking lot — light grey
    (100,  50, 200),  # 4 railway — purple
    ( 34, 139,  34),  # 5 tree canopy — forest green
    (144, 238, 144),  # 6 grass/lawn — light green
    (255, 220, 100),  # 7 agricultural field — yellow
    (200, 230, 255),  # 8 greenhouse — pale blue
    (139,  90,  43),  # 9 bare soil — brown
    (  0, 100, 200),  # 10 water — blue
    (200, 150, 200),  # 11 clutter — lavender
]


def encode_ensemble_prompts(prompt_dict: dict = None,
                            device: torch.device = DEVICE) -> torch.Tensor:
    """Encode the prompt ensemble with frozen CLIP ViT-B/32.

    Per class: encode prompts → L2-normalize each → average → re-normalize.
    Works for any number of classes — keys of prompt_dict determine N.

    Returns:
        Tensor [N, 512] of L2-normalized class prototypes.
    """
    import open_clip  # imported lazily — only needed when actually encoding

    prompt_dict = prompt_dict or PROMPT_ENSEMBLE
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_NAME, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_NAME)

    n_classes = max(prompt_dict.keys()) + 1
    embeddings = []
    for cls_id in range(n_classes):
        tokens = tokenizer(prompt_dict[cls_id]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            emb = emb.mean(dim=0)
            emb = F.normalize(emb, dim=-1)         # re-normalize after avg
        embeddings.append(emb)
    return torch.stack(embeddings)                 # [N, 512]


def encode_per_prompt(prompt_dict: dict = None,
                      device: torch.device = DEVICE) -> torch.Tensor:
    """Encode prompts individually — no averaging — for max-pool inference.

    Works for any number of classes — keys of prompt_dict determine N.

    Returns:
        Tensor [N, M, 512] where M = max prompts per class (shorter padded).
    """
    import open_clip

    prompt_dict = prompt_dict or DARMSTADT_PROMPTS
    n_classes = max(prompt_dict.keys()) + 1
    n_max = max(len(v) for v in prompt_dict.values())

    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_NAME, pretrained=CLIP_PRETRAINED)
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_NAME)

    per_class = []
    for cls_id in range(n_classes):
        prompts = prompt_dict[cls_id]
        tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)   # [M, 512]
            emb = F.normalize(emb, dim=-1)
        if len(prompts) < n_max:
            pad = emb[-1:].expand(n_max - len(prompts), -1)
            emb = torch.cat([emb, pad], dim=0)
        per_class.append(emb)
    return torch.stack(per_class)                  # [N, N_max, 512]


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
