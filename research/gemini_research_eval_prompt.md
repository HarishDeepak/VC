# Gemini Deep Research — RG-GeoPrompt-PEFT Evaluation Prompt

## How to use this file

1. Go to [gemini.google.com](https://gemini.google.com) → select **Gemini 2.5 Pro**
2. Click **"Deep Research"** before sending (bottom of input box)
3. Attach these files from this repo:
   - `EXPERIMENT_LOG.md`
   - `results/osm_pseudo_gt_f1.csv`
   - `research/gemini_deep_dive.md`
4. Paste the prompt below

If you don't have Gemini Advanced:
- Use **Perplexity Pro → Academic mode** for Q1–Q5 (needs real citations)
- Use **Claude Opus (/fast)** for Q6–Q9 (reasoning + verdict)

---

## Prompt

You are an expert reviewer for top-tier remote sensing and computer vision venues
(CVPR, ICCV, ISPRS, TGRS, IGARSS). I am a student doing a Fraunhofer IGD
practikum (SoSe 2026) and I need a rigorous, citation-backed evaluation of our
method's novelty, standing, and weaknesses. Be a tough reviewer — do not soften
your assessment. I have attached three files:
  - EXPERIMENT_LOG.md — full technical description of our method and results
  - osm_pseudo_gt_f1.csv — our zero-shot Darmstadt F1 scores
  - gemini_deep_dive.md — a prior architecture analysis you can reference

Read all three before answering. Then answer every question below in order,
each with its own section heading. Use real paper citations where possible.

═══════════════════════════════════════════════
CONTEXT SUMMARY (also in EXPERIMENT_LOG.md)
═══════════════════════════════════════════════

Task: Supervised training on ISPRS Potsdam (5–10 cm/px, 6 classes, labeled) →
zero-shot transfer to Darmstadt DOP20 (20 cm/px RGBI, no labels). 4× resolution
gap. Runs on T4 16 GB.

Architecture:
- DINOv2-base (frozen) + LoRA r=16 on Q,V only → 788K trainable (0.9%)
- CLIP ViT-B/32 (frozen) text prototypes → 2-layer MLP projection (512→768→768,
  GELU, ~412K trainable) → cosine similarity head → learnable temperature τ
- Similarity at 36×36 token resolution (logits upsampled to 512×512)
- Total trainable: ~1.2M / ~87M total (~1.35%)

Key training design choices:
- Resolution bridge: 30% of train patches downsampled 4× + upsampled back.
  Label mask never altered.
- Loss: CE + dice, boundary pixels (255) excluded

Inference (Darmstadt only):
- Max-pool prompt ensemble: 8 building + 5 clutter text variants, max logit per class
- Orthogonal projection: logits[building] -= 0.3 × logits[clutter]
- TTPA: 5 steps, lr=3e-4, KL=0.05, masked entropy. Adapts text_proj only,
  resets per patch.

Results — Potsdam supervised mIoU (5-class, clutter+boundary excluded):
  SegFormer-B0 (100% params):    82.1%
  DINOv2 + LoRA r=16 (0.9%):    85.26%  ← best epoch 15/20
  RG-GeoPrompt-PEFT (1.35%):    84.9%   ← best epoch 10/10

Results — Darmstadt zero-shot (OSM pseudo-GT F1, 6-class):
  Impervious: 0.516 | Building: 0.581 | Low Veg: 0.046 | Tree: 0.093
  Car: 0.000 | Clutter: 0.000 | MEAN: 0.206
  TTPA produced no meaningful improvement (MEAN: 0.2058 vs 0.2059 ZS).

Known published OVRSIS baselines (zero-shot TO Potsdam — different direction):
  TPOVSeg: 38–44% mIoU | SegEarth-OV: ~47% mIoU | GeoRSCLIP, RemoteCLIP variants

Known limitations we did not solve:
  - No crisp boundary/shape segmentation (ViT 14×14 patch grid)
  - TTPA statistically failed
  - Car + Clutter F1 = 0.0 (OSM has no layer for these)
  - Evaluation is OSM pseudo-GT only (noisy proxy, not true labels)

═══════════════════════════════════════════════
QUESTIONS — answer each with a section heading
═══════════════════════════════════════════════

Q1 — PEFT EFFICIENCY CLAIM
DINOv2+LoRA r=16 achieves 85.26% mIoU on Potsdam val with only 788K trainable
params (0.9%), outperforming fully fine-tuned SegFormer-B0 (82.1%, 3.7M params).
Search for published PEFT results on ISPRS Potsdam or comparable aerial datasets
(Vaihingen, LoveDA, iSAID). Is our 85.26% competitive? Is beating a fully
fine-tuned model with <1% parameters a novel and publishable finding by itself,
or has this already been shown?

Q2 — ZERO-SHOT TRANSFER QUALITY
Our mean OSM pseudo-GT F1 of 0.206 on Darmstadt after training on Potsdam.
Search for papers doing cross-city, cross-resolution zero-shot transfer in aerial
segmentation — particularly Potsdam→other city or Vaihingen→other city. What do
published methods achieve on comparable setups? Is 0.206 on a noisy pseudo-GT
meaningful, and how should we interpret building F1 of 0.581 specifically?

Q3 — RESOLUTION BRIDGE NOVELTY
Our resolution bridge (downsampling 30% of training patches by 4× to simulate
target domain resolution) is our most lightweight domain adaptation technique.
Search: has this specific augmentation been used for aerial remote sensing domain
adaptation? How does it compare to proper domain adaptation methods (adversarial
feature alignment, style transfer, FDA — Fourier Domain Adaptation)? Is it a
contribution or just a data augmentation trick?

Q4 — TTPA ANALYSIS AND FAILURE
Test-Time Prompt Adaptation adapting only the text projection MLP at inference
(5 steps, lr=3e-4, masked entropy). Ours had negligible effect (ΔF1 < 0.001).
Search: what TTPA/TTA methods have worked in segmentation? Specifically: has
adapting CLIP/text-side parameters at test time been tried in vision-language
segmentation? Why might entropy minimization on text projections fail when
vision-language alignment is already strong? What would you recommend instead?

Q5 — BOUNDARY SEGMENTATION GAP
Our predictions are class blobs — no instance-level shapes, no crisp boundaries.
ViT patch size 14×14 means lowest-resolution boundary at 14px. For aerial
segmentation this means building footprints look like rounded blobs rather than
rectangular polygons. How serious is this for OVRSIS research?
Search for: boundary-aware segmentation in aerial imagery, integration of SAM
(Segment Anything Model) with semantic segmentation for boundary refinement in
remote sensing, DenseCRF post-processing in aerial contexts, DINO stride-7 for
denser tokens. Would adding SAM boundary refinement turn this into a clearly
stronger paper? Is anyone combining DINOv2+CLIP cosine head + SAM?

Q6 — RESEARCH POSITIONING IN OVRSIS
The OVRSIS field includes: SegEarth-OV, TPOVSeg, GeoRSCLIP, RemoteCLIP,
EarthVLP, GeoChat, SkySense, GRAFT. Search for the most recent papers (2024–2025)
in open-vocabulary or zero-shot aerial segmentation.
Then answer:
  a) Is the combination of DINOv2+LoRA + CLIP cosine head novel, or has it been
     published? (search "DINOv2 CLIP cosine segmentation remote sensing")
  b) Is a resolution-bridge augmentation for cross-GSD transfer discussed anywhere?
  c) Where does our work sit: workshop paper level, main conference level, or
     below publishable threshold?
  d) What venue would be most appropriate (IGARSS, ISPRS Annals, CVPR workshop,
     ECCV workshop)?

Q7 — STRONGEST AND WEAKEST CONTRIBUTIONS
Given everything above, what is the single most defensible novel contribution in
our method that a reviewer could not dismiss? And what is the single biggest
weakness a reviewer would immediately flag in a rebuttal?

Q8 — ONE-MONTH IMPROVEMENT PLAN
If we had 4 more weeks and one T4 GPU, what single addition would most move the
needle on research quality — not on Potsdam mIoU (that's already strong) but on
the zero-shot transfer story and publishability? Consider: SAM boundary
integration, better TTPA (feature-level vs text-level), pseudo-label
self-training on Darmstadt, proper domain adaptation, Darmstadt human annotation
of 50 patches for real evaluation, or something else entirely.

Q9 — HONEST VERDICT
One paragraph, no softening: Is this a genuine research contribution to the
OVRSIS/PEFT-for-remote-sensing field, or is it a well-executed engineering
project that assembles existing pieces without a clear novel hypothesis? Would you
accept this at IGARSS 2026 as-is?
