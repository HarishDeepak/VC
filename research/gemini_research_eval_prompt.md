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

Q1 — Is our efficiency result novel?
85.26% mIoU with 1% params beating SegFormer is solid but not novel. The field already has papers doing this better (Earth-Adapter etc). Good result, not a new idea.

Q2 — Is our zero-shot F1 of 0.206 good?
No. SOTA zero-shot methods hit ~40-47% on comparable tasks. 0.206 is poor. Building at 0.581 is the only bright spot — makes sense because buildings are big enough to span multiple tokens. Car fails because a car is literally smaller than one 14×14 token at 20cm resolution.

Q3 — Is the resolution bridge novel?
No, it's just a data augmentation trick. Real domain adaptation (FDA, adversarial alignment) is far more principled. Worse, our version introduces label noise because we blur the image but leave the sharp label mask unchanged — the model gets penalized for boundaries that no longer exist visually.

Q4 — Why did TTPA completely fail?
Because entropy minimization was designed for normal classifiers, not fortraining — forcing it toward confident predictions just collapses it ontothe dominant class. Papers like CLIPTTA already proved this and proposed better approaches (soft contrastive loss instead).

Q5 — How bad is the boundary problem?
Very bad for any practical use. A 14×14 pixel patch = every boundary is at minimum 14px wide and blobby. SAM (Segment Anything) is the standard fix — use it to get crisp
shapes, then assign class labels from our model. Stride-7 inference (alre) also helps without retraining.

Q6 — Where do we sit in the OVRSIS field?
Workshop paper level at best. The DINOv2+CLIP cosine head combo already e-Seg, SegEarth-OV). Resolution bridge isn't novel. Best venue: CVPR Earthn Workshop or ISPRS Annals short paper.

Q7 — What's our best and worst part?

Q8 — What to do in 4 more weeks?                                                                                                                                               Two cheap fixes: (1) Randomized Histogram Matching during training — matcmstadt patches, fixes the dark/shadow gap, beats CycleGAN-level resultswith zero extra params. (2) Stride-7 inference patch — already partly in our code, doubles spatial resolution, fixes blob boundaries. No retraining needed.                    
onest verdict?
Solid engineering practicum, not a research paper yet. Would be rejected from IGARSS 2026 as-is. To be publishable it needs to be reframed as "here's why standard VLM methods fail on cross-resolution aerial data" + offer the RHM and stride-7 fixes

Q10 — What about the dark/shadow gap in Darmstadt?
Darmstadt is much darker than Potsdam. This radiometric gap is a separatetogram matching (pairing each Potsdam training patch with a randomDarmstadt patch and matching colors) is a proven fix — simple, fast, no trainable params, shown to improve cross-domain building F1 by 15-20% in the literature.



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


## answers 
# **Rigorous Evaluation of Parameter-Efficient Fine-Tuning and Vision-Language Alignment for Open-Vocabulary Remote Sensing Semantic Segmentation**

## **Q1 — PEFT EFFICIENCY CLAIM**

### **Theoretical Foundations of Parameter-Efficient Fine-Tuning in Geospatial Contexts**

The deployment of large-scale Vision Foundation Models (VFMs) such as DINOv2 has fundamentally altered the paradigm of feature extraction in remote sensing. However, the sheer parameter count of these models—often exceeding 86 million parameters for base variants and 300 million for large variants—renders Full Fine-Tuning (FFT) computationally prohibitive. Furthermore, in the context of spatially constrained datasets like ISPRS Potsdam, FFT is prone to severe overfitting and catastrophic forgetting of the generalized semantic manifolds learned during self-supervised pre-training.1  
The proposed architecture utilizes Low-Rank Adaptation (LoRA) with a rank of 16, restricted strictly to the Query (![][image1]) and Value (![][image2]) projections of the frozen DINOv2-base model. In conjunction with a 2-layer Multi-Layer Perceptron (MLP) for text prototype projection, this configuration yields approximately 1.2 million trainable parameters, constituting a mere 1.35% of the total architecture. Achieving an 85.26% mean Intersection over Union (mIoU) on the ISPRS Potsdam validation set under these constraints is objectively a robust empirical result.

### **Comparative Benchmarking Against the State-of-the-Art**

To contextualize this performance, the baseline SegFormer-B0 architecture, which undergoes FFT and utilizes 3.7 million parameters, achieves an 82.1% mIoU. The superiority of a PEFT approach over a fully fine-tuned lightweight baseline is not an anomaly. It reflects a well-documented phenomenon in modern representation learning: frozen self-supervised backbones retain a highly generalized, robust semantic manifold that provides superior feature embeddings compared to a lightweight model forced to learn spatial hierarchies from scratch.3  
However, the underlying claim that outperforming a fully fine-tuned model with fewer than 1% trainable parameters constitutes a novel, publishable finding is fundamentally incorrect within the context of the 2025–2026 remote sensing literature. The field has already extensively demonstrated the superiority of PEFT mechanisms over FFT in aerial semantic segmentation.  
Recent architectures have pioneered advanced PEFT methods specifically tailored for remote sensing. For instance, the Earth-Adapter framework introduces a novel Frequency-Guided Mixture of Adapters (MoA) approach.1 Earth-Adapter utilizes Discrete Fourier Transformation (DFT) to divide features into distinct frequency components, isolating artifact-related information from semantic signals. This specialized PEFT method achieves state-of-the-art results on ISPRS Potsdam and Vaihingen benchmarks using between 2.4M and 9.6M trainable parameters (roughly 1.5% to 3.0% of the network), significantly outperforming standard NLP-derived LoRA configurations.1 Earth-Adapter explicitly demonstrates that PEFT methods not only require fewer parameters but consistently outperform fully fine-tuned baselines by mitigating high-frequency artifact disturbances inherent in remote sensing imagery.2  
Similarly, the CrossEarth-Gate architecture introduces a Fisher-guided adaptive tuning engine that leverages PEFT to achieve highly efficient cross-domain adaptation.5 Another contemporary approach, SpectralX, adapts VFMs for spectral imagery using an Attribute-oriented Mixture of Adapter (AoMoA), proving that sub-2% parameter updates are the industry standard for foundation model deployment.7

| Model Architecture | Adaptation Mechanism | Trainable Parameters | Percentage of Total | Potsdam mIoU Performance | Reference |
| :---- | :---- | :---- | :---- | :---- | :---- |
| SegFormer-B0 | Full Fine-Tuning | \~3.7M | 100% | 82.10% | Proposed Baseline |
| DINOv2-Base | Full Fine-Tuning | \~86.6M | 100% | \~49.80% (Cross-Domain) | 4 |
| DINOv2-Large | Earth-Adapter | \~2.4M \- 9.6M | \~1.5% \- 3.0% | State-of-the-Art | 1 |
| DINOv2-Base | Proposed LoRA (r=16) | \~1.2M | \~1.35% | 85.26% (Supervised) | Proposed Method |

### **Assessment of Methodological Novelty**

The finding that a fractional parameter update outperforms a 100% update on a much smaller model is an expected outcome of utilizing a vastly superior feature extractor (DINOv2) rather than a novel methodological discovery. The 85.26% mIoU demonstrates sound engineering execution and validates the specific implementation of the LoRA and cosine similarity head within strict hardware constraints. However, reviewers at top-tier venues (e.g., CVPR, ICCV, IEEE TGRS) would immediately dismiss the assertion of novelty regarding the efficiency-to-performance ratio. The current frontier of PEFT in remote sensing focuses on domain-specific adaptations—such as frequency routing or spatially aware adapters—rather than the baseline application of standard LoRA matrices to vision transformers.

## **Q2 — ZERO-SHOT TRANSFER QUALITY**

### **Analysis of the Zero-Shot Transfer Metrics**

The transition from supervised training on ISPRS Potsdam (5–10 cm/px resolution) to zero-shot inference on Darmstadt DOP20 (20 cm/px resolution) introduces a severe domain shift. This shift encompasses geographical layout variations, radiometric profile differences, and a four-fold degradation in spatial resolution. The proposed method yields a mean F1 score of 0.206 across a 6-class pseudo-ground truth derived from OpenStreetMap (OSM) layers.  
Evaluating zero-shot transfer quality requires a nuanced understanding of both the physical token geometry of the architecture and the inherent epistemic limitations of the evaluation data. A mean F1 of 0.206 is objectively poor when compared to intra-domain supervised benchmarks, and it requires strict contextualization against current Open-Vocabulary Remote Sensing Image Segmentation (OVRSIS) literature.  
Established state-of-the-art OVRSIS models demonstrate significantly higher zero-shot capabilities. For example, SegEarth-OV achieves approximately 40.9% mIoU on the Potsdam dataset in a zero-shot capacity.8 SegEarth-OV utilizes a universal upsampler (SimFeatUp) and a global bias alleviation operation that subtracts inherent global context from patch features, significantly enhancing local semantic fidelity without task-specific post-training.9 Similarly, TPOV-Seg reports mIoU scores ranging from 38% to 47% utilizing textually enhanced prompt tuning.11 The proposed method's performance falls substantially below these established baselines, indicating a failure to generalize linguistic-visual alignments across spatial resolutions.

### **The Statistical Mechanics of Pseudo-Ground Truth Evaluation**

The reliance on OSM pseudo-ground truth introduces massive epistemic uncertainty into the evaluation protocol. OSM data is heavily reliant on crowdsourced annotations, resulting in notorious spatial misalignments, varying levels of topological completeness, and a lack of granular semantic delineation. Consequently, an F1 score computed against OSM layers must be interpreted as a noisy proxy rather than an absolute measure of segmentation fidelity.  
The catastrophic failure on the "Car" and "Clutter" classes (F1 \= 0.000) is a direct consequence of OSM's ontological structure. OSM rarely features comprehensive, up-to-date cadastral layers for transient objects like vehicles or ambiguous land-use categories like clutter. Attempting to evaluate a zero-shot model on semantic classes that are systematically absent or structurally deficient in the target validation data is a methodological flaw that artificially deflates the mean F1 score.

### **Geometric Interpretation of Class-Specific Performance**

Despite the low overall mean, the "Building" class achieves an F1 score of 0.581. This performance is physically and geometrically logical when analyzing the VFM token structure. Buildings in the target Darmstadt DOP20 imagery possess distinct spectral signatures, rigid multi-pixel structural constraints, and strong visual contrast against adjacent impervious surfaces. Even at the degraded 20 cm/px resolution, a standard residential structure occupies a sufficient number of pixels to span multiple 14x14 ViT tokens, allowing the DINOv2 backbone to extract coherent spatial semantics.  
Conversely, the failure on the "Car" class can be mathematically explained by token geometry. At 20 cm/px resolution, a standard vehicle measuring 4.5 meters by 1.8 meters occupies roughly 22 by 9 pixels. The DINOv2 patch size is fixed at 14x14 pixels. Therefore, a vehicle is represented by one or, at most, two tokens, often split across token boundaries. The max-pool prompt ensemble mechanism and the inherent spatial compression of the ViT backbone completely destroy the sub-patch spatial signal required to isolate such small features. The text projection MLP cannot align a linguistic prototype with a visual token that is overwhelmingly contaminated by the surrounding asphalt background. Thus, the zero-shot transfer quality is strictly bounded by the physical sampling frequency of the architecture.

| Model / Methodology | Evaluation Task | Zero-Shot Metric | Reference |
| :---- | :---- | :---- | :---- |
| Proposed Method | Potsdam → Darmstadt (OSM) | Mean F1: 0.206 | Proposed Method |
| SegEarth-OV | Zero-Shot to Potsdam | mIoU: 40.9% | 10 |
| TPOV-Seg | Zero-Shot to Potsdam | mIoU: \~38.0% \- 47.0% | 11 |
| ClearCLIP (ViT-B) | Zero-Shot OVRSIS | mIoU: \~40.9% | 10 |

## **Q3 — RESOLUTION BRIDGE NOVELTY**

### **Disentangling Domain Adaptation from Data Augmentation**

The proposed pipeline attempts to bridge the 4× spatial resolution gap (from 5 cm/px in Potsdam to 20 cm/px in Darmstadt) by randomly downsampling 30% of the training patches by a factor of 4 and subsequently upsampling them back to their native dimensions. Crucially, this operation is applied solely to the RGB arrays; the corresponding semantic label masks are never altered. While positioned as a lightweight domain adaptation technique, a rigorous taxonomic classification reveals that it is exclusively a standard data augmentation strategy, not a domain adaptation method.  
True domain adaptation mathematically entails the alignment of a source domain marginal distribution ![][image3] with a target domain marginal distribution ![][image4], or the learning of domain-invariant feature representations. Methodologies in the remote sensing literature actively utilize target domain statistics. For example, Adversarial Discriminative Domain Adaptation (ADDA) employs adversarial losses to force feature extractors to confuse a domain discriminator, explicitly closing the semantic gap.13  
When addressing resolution gaps specifically, the literature demonstrates far more sophisticated approaches. Fourier Domain Adaptation (FDA) executes a frequency-space swap, substituting the low-frequency amplitude spectrum of high-resolution source images with that of low-resolution target images. This perfectly replicates the target's illumination and sensor characteristics while preserving the high-frequency semantic structure of the source.14 Architectures like SRDA-Net (Super-Resolution Domain Adaptation Network) integrate a multi-task model for super-resolution and segmentation, explicitly addressing the resolution adaptation problem when moving from low-resolution to high-resolution domains by generating pixel-level domain classifications.13

### **The Sub-Pixel Alignment Fallacy and Label Noise**

The technique of altering the Ground Sample Distance (GSD) via downsampling and upsampling during training is intended to force the network to learn scale-invariant features.17 However, applying this transformation to the input image while leaving the dense pixel-level label mask unchanged introduces severe label noise.  
When a 5 cm/px image is downsampled by a factor of 4, the high-frequency boundaries of objects are permanently destroyed through spatial aliasing and interpolation smoothing. Upsampling the blurred image back to the original resolution creates a feature map where the visual boundaries no longer align with the crisp, high-frequency polygon boundaries of the unmodified label mask. The network is subsequently penalized by the Cross-Entropy and Dice losses for failing to predict sharp boundaries that physically no longer exist in the input data. This forces the model to learn conflicting spatial representations.

### **Assessment of Contribution**

The "resolution bridge" as implemented is not a novel contribution to the field of cross-GSD remote sensing transfer. Scale-Aware Adversarial Learning frameworks utilize dual discriminators—one for feature semantics and a novel scale discriminator coupled with a scale attention module—to explicitly perform joint cross-location and cross-scale land-cover classification.19 Relying on naive bilinear degradation without explicit scale-conditioned feature alignment is insufficient for a top-tier venue. Reviewers would immediately flag this as a heuristic augmentation trick that introduces unnecessary label noise rather than a theoretically grounded adaptation mechanism.

## **Q4 — TTPA ANALYSIS AND FAILURE**

### **The Misalignment of Entropy Minimization and Contrastive Manifolds**

The Test-Time Prompt Adaptation (TTPA) strategy implemented in the proposed method adapts only the text projection MLP during inference utilizing a masked entropy minimization objective over 5 steps. The empirical results demonstrate that this approach yielded negligible improvements, with the mean zero-shot F1 score shifting insignificantly from 0.2059 to 0.2058. This statistical failure is not anomalous; it is the predictable consequence of applying classification-derived optimization objectives to contrastively trained vision-language manifolds.  
Standard test-time adaptation methods formulated for closed-set classification, such as TENT, optimize network parameters by minimizing the Shannon entropy of the output probability distribution.20 This encourages the model to output highly confident, one-hot-like predictions. However, Vision-Language Models (VLMs) like CLIP are not trained via standard cross-entropy over a closed label set; they are optimized via an InfoNCE contrastive loss that aligns continuous, high-dimensional visual and textual embeddings into a shared metric space.21  
When entropy minimization is forcibly applied to the text projections of a VLM, the objective blindly pushes the logits toward the nearest prototype, disregarding the underlying geometry of the contrastive space. As rigorously proven by the authors of CLIPTTA (Robust Contrastive Vision-Language Test-Time Adaptation), this misalignment inherently triggers pseudo-label drift and catastrophic class collapse.21 The optimization degrades the learned text-vision alignment by warping the projection MLP to overfit to the most prominent target-domain features (such as impervious surfaces or massive building structures), effectively destroying the open-vocabulary generalization the CLIP model was originally prized for.

### **Recommended Test-Time Adaptation Frameworks for Dense Prediction**

To rectify this failure, test-time adaptation must be fundamentally redesigned to respect both the spatial density of remote sensing segmentation tasks and the contrastive nature of the underlying foundation models. Two contemporary frameworks in the 2025–2026 literature represent the correct theoretical approach:

1. **Multi-Level and Multi-Prompt (MLMP) Adaptation**: The MLMP framework specifically addresses Open-Vocabulary Semantic Segmentation (OVSS) by integrating features from intermediate vision-encoder layers (e.g., intermediate ViT blocks) to capture complementary, shift-resilient cues. Crucially, MLMP avoids isolated parameter updates by minimizing entropy across multiple diverse text-prompt templates at both the global class token and local pixel-wise levels.20 This enforces consistency across linguistic perspectives, stabilizing the adaptation process against the collapse observed in the proposed method's isolated MLP adaptation.  
2. **Soft Contrastive Loss (CLIPTTA)**: The CLIPTTA methodology completely discards standard entropy minimization in favor of a soft contrastive image-text loss that mathematically mirrors CLIP's original pre-training objective. By analyzing the training dynamics and batch-aware gradients, CLIPTTA provides a theoretically sound mechanism to update VLM parameters during inference without risking pseudo-label drift, offering a highly robust alternative to the failed masked entropy approach.23

## **Q5 — BOUNDARY SEGMENTATION GAP**

### **The Interpolation Fallacy and ViT Token Mechanics**

The inability of the proposed architecture to produce crisp, instance-level boundaries is a severe limitation that critically impairs its utility for operational OVRSIS applications, such as cadastral mapping or precise infrastructure monitoring. The predictions manifest as amorphous blobs because of the fundamental geometry of the Vision Transformer (ViT) employed by DINOv2.  
The ViT-Base/14 architecture dissects the input image into non-overlapping 14x14 pixel patches. Each patch is linearly projected into a single, high-dimensional token. As detailed in the provided architectural forensic analysis, practitioners frequently attempt to recover lost spatial resolution by bilinearly upsampling the high-dimensional feature tokens prior to computing the cosine similarity dot product.26 This is a mathematical fallacy.  
Because the tokenization process compresses all sub-patch spatial variance into a singular vector, there is zero sub-token spatial information embedded within those channels to interpolate. The distributive property of the dot product operation (which is a linear mapping) proves that interpolating features before the projection head yields the exact same spatial gradient and boundary delineation as projecting the low-resolution tokens first and upsampling the resulting single-channel logits.26 Consequently, boundaries are mathematically constrained to the rigid 14x14 pixel grid, meaning a building footprint will always possess a minimum boundary resolution of 14 pixels.

### **State-of-the-Art Solutions for Boundary Refinement**

In the current remote sensing landscape, the reliance on coarse patch-level predictions without secondary boundary refinement is unacceptable. The integration of boundary-aware modules and secondary foundation models is standard practice.  
The Segment Anything Model (SAM) has emerged as the definitive solution for class-agnostic boundary extraction in aerial imagery. Architectures such as MoBaNet successfully fuse DINOv2 semantic encoders with lightweight multimodal pipelines to achieve boundary-aware predictions.27 Furthermore, frameworks like SaLIP construct a unified cascade where SAM initially generates precise, prompt-driven spatial segmentations, followed by a CLIP or DINOv2-driven zero-shot semantic assignment to those pre-defined masks.28 This explicitly decouples the task of spatial delineation (handled by SAM's high-resolution CNN decoders) from the task of semantic classification (handled by the VLM).  
Alternatively, if incorporating SAM introduces prohibitive computational latency, modifying the inherent stride of the DINOv2 patch embedding layer offers a deterministic solution. By monkey-patching the convolutional projection to utilize a stride of 7 while maintaining a kernel size of 14, the tokens physically overlap. This doubles the sampling frequency across both spatial dimensions without altering the pre-trained weights, directly allowing the frozen backbone to capture sub-patch spatial fidelity and significantly sharpening the resulting semantic blobs.26 Incorporating either SAM cascade refinement or overlapping inference is mandatory to elevate this research to a clearly stronger, publishable tier.

## **Q6 — RESEARCH POSITIONING IN OVRSIS**

### **Assessment of Methodological Novelty**

The combination of a frozen DINOv2 backbone equipped with LoRA adapters and a CLIP-initialized cosine similarity head is a logical, well-engineered composition of contemporary techniques. However, it is not inherently novel in the 2025–2026 OVRSIS landscape. The literature demonstrates a massive proliferation of methodologies explicitly uniting generic vision foundation models with vision-language models to achieve open-vocabulary segmentation.  
**a) Novelty of the Architecture:** Models such as RSKT-Seg explicitly fuse features from DINOv2 and Remote-CLIP to effectively transfer remote sensing knowledge to open-vocabulary segmentation.29 SegEarth-OV achieves training-free open-vocabulary segmentation by utilizing universal upsamplers (SimFeatUp) and executing global bias alleviation to correct the distorted target shapes typical of patch-based VLMs.8 Similarly, TPOV-Seg focuses heavily on textually enhanced prompt tuning to stabilize vision-language alignment in remote sensing contexts.11 The specific assembly of DINOv2 and CLIP via a linear or MLP projection head represents an incremental engineering optimization rather than a foundational algorithmic breakthrough.  
**b) Resolution-Bridge Augmentation:** As established in the analysis of Q3, downsampling augmentations are pervasive across computer vision and remote sensing literature to achieve scale invariance.17 The use of spatial degradation to simulate target domain resolution is a fundamental augmentation strategy, but it completely lacks the theoretical sophistication of modern scale-aware domain adaptation frameworks.19  
**c) Publishable Level:** The work, in its current state, sits strictly at the workshop paper level. It falls below the publishable threshold for main-conference tracks at premier venues. The lack of a theoretically novel algorithmic contribution, coupled with the profound failure of the TTPA mechanism and the absence of boundary refinement, precludes acceptance in highly competitive main-track arenas.  
**d) Appropriate Venues:** The paper is optimally positioned for an Earth Vision or Remote Sensing Workshop at CVPR/ECCV, or as a short paper at the ISPRS Annals. A submission to IGARSS is viable, but it requires a significant reframing of the narrative to focus on the empirical constraints of PEFT under domain shifts, rather than claiming novel architectural discoveries.

## **Q7 — STRONGEST AND WEAKEST CONTRIBUTIONS**

### **The Most Defensible Novel Contribution**

The single most defensible element of this research is the rigorous forensic profiling and architectural optimization of the PEFT pipeline that achieves 85.26% supervised mIoU with only \~1.2M trainable parameters on hardware constrained to 16 GB of VRAM. The decision to restrict LoRA exclusively to the DINOv2 Query and Value projections, combined with the lightweight MLP for text prototype projection, demonstrates a highly sophisticated understanding of GPU memory constraints, autograd caching mechanisms, and the mathematical boundaries of similarity scaling.26 This extreme parameter efficiency—validating the capacity of a sub-2% trainable matrix to successfully map deep self-supervised visual features into a dense geospatial semantic space—is a practical empirical success that reviewers cannot dismiss.

### **The Single Biggest Weakness**

The most glaring weakness that a reviewer will immediately exploit during a rebuttal is the catastrophic collapse of the zero-shot inference pipeline, manifesting in an F1 score of 0.000 for multiple classes and an inability to define operational object boundaries. A reviewer will forcefully point out that a framework claiming to solve Open-Vocabulary segmentation cannot statistically fail on standard target classes like "Car" simply due to rigid token geometries and naive max-pooling prompt ensembles. Furthermore, the reliance on a demonstrably flawed resolution bridge that introduces dense label noise, paired with a mathematically misaligned test-time entropy minimization scheme that yields zero improvement, severely undermines the paper's scientific rigor.

## **Q8 — ONE-MONTH IMPROVEMENT PLAN**

Given the strict constraints of a four-week timeline and a single NVIDIA T4 GPU, attempting to train a secondary diffusion model or a heavy generative adversarial network for domain adaptation is unfeasible. The single dual-pronged addition that will most dramatically move the needle on zero-shot transfer quality and domain adaptation combines **Randomized Histogram Matching (RHM)** with **Overlapping Patch Inference**.

### **Step 1: Mitigating the Radiometric Domain Gap via RHM**

The domain gap between Potsdam and Darmstadt is heavily influenced by radiometric differences, illumination shifts, and shadow intensities. RHM is a fast, computationally trivial, real-time unsupervised training augmentation technique. During the Potsdam training phase, RHM dynamically pairs each source domain image with a randomly selected, unlabeled target domain image (Darmstadt) and matches their histograms online.30 This effectively normalizes the radiometric domain shift without requiring any trainable parameters or complex GAN architectures, significantly improving the robustness of the DINOv2 embeddings to the target domain's unique illumination profiles.

### **Step 2: Mitigating the Boundary Gap via Stride-7 Monkey Patching**

To address the geometric boundary gap without training a new model or running a VRAM-heavy SAM cascade, the implementation should monkey-patch the DINOv2 embedding layer at inference time. By changing the convolutional stride from (14,14) to (7,7) while maintaining the 14x14 kernel, the tokens physically overlap. This incurs a higher computational cost during the forward pass but easily fits within the transient \~41 MB VRAM footprint demonstrated in the "Mathematically Correct Full-Resolution Path" analysis.26 It physically doubles the spatial sampling frequency, allowing the model to capture sub-patch spatial fidelity and directly mitigating the blob-like boundary artifacts that currently plague the zero-shot predictions.  
Implementing this dual-pronged approach requires minimal code modification, demands near-zero additional VRAM during training, and directly attacks the two most prominent sources of failure: radiometric domain shift and spatial boundary degradation.

## **Q9 — HONEST VERDICT**

This work represents a highly competent, exceptionally well-executed engineering practicum, but it falls short of being a genuine, hypothesis-driven research contribution to the OVRSIS field. The project successfully pieces together existing, powerful foundation models (DINOv2, CLIP, LoRA) and achieves commendable supervised results through brute-force parameter efficiency and memory profiling. However, it systematically fails to address the unique complexities of remote sensing imagery. The resolution bridge is a naive data augmentation trick that introduces label noise; the boundary delineation ignores basic spatial sampling theory; and the test-time prompt adaptation blindly applies classification-based entropy objectives to a contrastive manifold, resulting in inevitable statistical failure. As it currently stands, I would firmly reject this submission for IGARSS 2026\. To achieve publication, the manuscript must be entirely reframed not as a novel architecture, but as a rigorous empirical diagnostic of *why* standard vision-language adaptations fail under severe geospatial resolution and radiometric shifts, subsequently offering computationally constrained solutions (such as RHM and stride modification) to mitigate those failures.

## **Q10 — ILLUMINATION AND SHADOW DOMAIN GAP**

### **The Radiometric Barrier in Geospatial Transfer**

The observation that the Darmstadt DOP20 imagery exhibits significantly darker profiles with pronounced shadow prevalence compared to the Potsdam training dataset introduces a severe radiometric domain shift. In aerial segmentation, illumination disparities are not merely superficial aesthetic differences; they represent fundamental alterations to the statistical distribution of the feature space. Shadows obscure fine-grained textures, alter the perceived spectral reflectance of impervious surfaces, and cause the self-supervised representations of the frozen DINOv2 backbone to drift away from the textual prototypes learned during the supervised phase. When combined with the 4× resolution gap, this illumination shift compounds the feature degradation, leading to catastrophic misclassification.

### **Efficacy of Histogram Matching in Remote Sensing**

Addressing this radiometric discrepancy alongside the resolution gap is mandatory to formulate a coherent domain adaptation narrative. The literature extensively documents the success of histogram matching as an elegant, computationally efficient mechanism for cross-domain aerial segmentation. Studies utilizing Histogram Matching Augmentation or Hybrid Object-Based Augmentation and Histogram Matching demonstrate dramatic improvements in cross-domain building segmentation. By normalizing radiometric discrepancies without modifying the underlying backbone architecture or requiring target-domain labels, researchers have improved F1 scores across disparate satellite collections by margins exceeding 15-20%.32  
More specifically, Randomized Histogram Matching (RHM) has proven exceptionally effective for unsupervised domain adaptation in overhead imagery.30 By stochastically pairing source imagery with unlabeled target imagery and performing online distribution matching, models become inherently robust to variations in sensor hardware and seasonal lighting.30 RHM achieves performance parity with, and often surpasses, complex adversarial domain adaptation techniques (such as CycleGANs) while requiring a fraction of the computational overhead.31 Integrating a lightweight, stochastic histogram matching protocol into the training pipeline would fundamentally strengthen the paper. It would demonstrate to reviewers a sophisticated understanding of the multifaceted nature of geospatial domain gaps—proving that the authors recognize that true adaptation requires solving both the geometric scale (via overlapping inference) and the radiometric distribution (via RHM), thereby transforming a weak "resolution bridge" into a comprehensive, dual-pronged domain adaptation strategy.

#### **Works cited**

1. Earth-Adapter: Bridge the Geospatial Domain Gaps with a Frequency-Guided Mixture of Adapters, accessed June 14, 2026, [https://ojs.aaai.org/index.php/AAAI/article/download/42498/46459](https://ojs.aaai.org/index.php/AAAI/article/download/42498/46459)  
2. Earth-Adapter: Bridge the Geospatial Domain Gaps with Mixture of Frequency Adaptation \- arXiv, accessed June 14, 2026, [https://arxiv.org/html/2504.06220v3](https://arxiv.org/html/2504.06220v3)  
3. Earth-Adapter: Bridge the Geospatial Domain Gaps with a Frequency-Guided Mixture of Adapters \- arXiv, accessed June 14, 2026, [https://arxiv.org/html/2504.06220v4](https://arxiv.org/html/2504.06220v4)  
4. Earth-Adapter: Bridge the Geospatial Domain Gaps with Mixture of Frequency Adaptation, accessed June 14, 2026, [https://www.researchgate.net/publication/390602165\_Earth-Adapter\_Bridge\_the\_Geospatial\_Domain\_Gaps\_with\_Mixture\_of\_Frequency\_Adaptation](https://www.researchgate.net/publication/390602165_Earth-Adapter_Bridge_the_Geospatial_Domain_Gaps_with_Mixture_of_Frequency_Adaptation)  
5. CrossEarth-Gate: Fisher-Guided Adaptive Tuning Engine for Efficient Adaptation of Cross-Domain Remote Sensing Semantic Segmentation \- arXiv, accessed June 14, 2026, [https://arxiv.org/html/2511.20302v1](https://arxiv.org/html/2511.20302v1)  
6. CrossEarth-Gate: Fisher-Guided Adaptive ... \- CVF Open Access, accessed June 14, 2026, [https://openaccess.thecvf.com/content/CVPR2026/supplemental/Cao\_CrossEarth-Gate\_Fisher-Guided\_Adaptive\_CVPR\_2026\_supplemental.pdf](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Cao_CrossEarth-Gate_Fisher-Guided_Adaptive_CVPR_2026_supplemental.pdf)  
7. SpectralX: Parameter-efficient Domain Generalization for Spectral Remote Sensing Foundation Models \- GitHub, accessed June 14, 2026, [https://github.com/YuxiangZhang-BIT/SpectralX](https://github.com/YuxiangZhang-BIT/SpectralX)  
8. SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images \- CVPR 2025 Open Access Repository \- The Computer Vision Foundation, accessed June 14, 2026, [https://openaccess.thecvf.com/content/CVPR2025/html/Li\_SegEarth-OV\_Towards\_Training-Free\_Open-Vocabulary\_Segmentation\_for\_Remote\_Sensing\_Images\_CVPR\_2025\_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.html)  
9. \[2508.18067\] Annotation-Free Open-Vocabulary Segmentation for Remote-Sensing Images, accessed June 14, 2026, [https://arxiv.org/abs/2508.18067](https://arxiv.org/abs/2508.18067)  
10. SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images, accessed June 14, 2026, [https://openaccess.thecvf.com/content/CVPR2025/papers/Li\_SegEarth-OV\_Towards\_Training-Free\_Open-Vocabulary\_Segmentation\_for\_Remote\_Sensing\_Images\_CVPR\_2025\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.pdf)  
11. TPOV-Seg: Textually Enhanced Prompt Tuning of Vision-Language Models for Open-Vocabulary Remote Sensing Semantic Segmentation | Request PDF \- ResearchGate, accessed June 14, 2026, [https://www.researchgate.net/publication/396840233\_TPOV-Seg\_Textually\_Enhanced\_Prompt\_Tuning\_of\_Vision-Language\_Models\_for\_Open-Vocabulary\_Remote\_Sensing\_Semantic\_Segmentation](https://www.researchgate.net/publication/396840233_TPOV-Seg_Textually_Enhanced_Prompt_Tuning_of_Vision-Language_Models_for_Open-Vocabulary_Remote_Sensing_Semantic_Segmentation)  
12. HG-RSOVSSeg: Hierarchical Guidance Open-Vocabulary Semantic Segmentation Framework of High-Resolution Remote Sensing Images \- MDPI, accessed June 14, 2026, [https://www.mdpi.com/2072-4292/18/2/213](https://www.mdpi.com/2072-4292/18/2/213)  
13. (PDF) Tackling Dual Gaps in Remote Sensing Segmentation: Task-Oriented Super-Resolution for Domain Adaptation \- ResearchGate, accessed June 14, 2026, [https://www.researchgate.net/publication/386363498\_Tackling\_Dual\_Gaps\_in\_Remote\_Sensing\_Segmentation\_Task-Oriented\_Super-Resolution\_for\_Domain\_Adaptation](https://www.researchgate.net/publication/386363498_Tackling_Dual_Gaps_in_Remote_Sensing_Segmentation_Task-Oriented_Super-Resolution_for_Domain_Adaptation)  
14. Full article: Enhanced Wheat Head Detection in Images Using Fourier Domain Adaptation and Random Guided Filter \- Taylor & Francis, accessed June 14, 2026, [https://www.tandfonline.com/doi/full/10.1080/07038992.2024.2367479](https://www.tandfonline.com/doi/full/10.1080/07038992.2024.2367479)  
15. Domain Adaptive Semantic Segmentation of Remote Sensing Images \- Encyclopedia.pub, accessed June 14, 2026, [https://encyclopedia.pub/entry/54947](https://encyclopedia.pub/entry/54947)  
16. Super-resolution domain adaptation networks for semantic segmentation via pixel and output level aligning \- Frontiers, accessed June 14, 2026, [https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.974325/full](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.974325/full)  
17. UNIVERSITY OF CALIFORNIA, MERCED ... \- eScholarship.org, accessed June 14, 2026, [https://escholarship.org/content/qt0c07w8bq/qt0c07w8bq\_noSplash\_9c8deac4a84fe34e0239ea4d81b529e8.pdf](https://escholarship.org/content/qt0c07w8bq/qt0c07w8bq_noSplash_9c8deac4a84fe34e0239ea4d81b529e8.pdf)  
18. Dual-domain representation alignment for unsupervised height, accessed June 14, 2026, [https://research.utwente.nl/files/521983829/1-s2.0-S0924271625004186-main.pdf](https://research.utwente.nl/files/521983829/1-s2.0-S0924271625004186-main.pdf)  
19. Scale Aware Adaptation for Land-Cover Classification in Remote Sensing Imagery \- CVF Open Access, accessed June 14, 2026, [https://openaccess.thecvf.com/content/WACV2021/papers/Deng\_Scale\_Aware\_Adaptation\_for\_Land-Cover\_Classification\_in\_Remote\_Sensing\_Imagery\_WACV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/WACV2021/papers/Deng_Scale_Aware_Adaptation_for_Land-Cover_Classification_in_Remote_Sensing_Imagery_WACV_2021_paper.pdf)  
20. Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation \- arXiv, accessed June 14, 2026, [https://arxiv.org/html/2505.21844v2](https://arxiv.org/html/2505.21844v2)  
21. \[2507.14312\] CLIPTTA: Robust Contrastive Vision-Language Test-Time Adaptation \- arXiv, accessed June 14, 2026, [https://arxiv.org/abs/2507.14312](https://arxiv.org/abs/2507.14312)  
22. CLIPTTA: Robust Contrastive Vision-Language Test-Time Adaptation \- OpenReview, accessed June 14, 2026, [https://openreview.net/pdf/4d1e7879dbb8e6fe543f0387ccdbbdd18b0a1cdb.pdf](https://openreview.net/pdf/4d1e7879dbb8e6fe543f0387ccdbbdd18b0a1cdb.pdf)  
23. ClipTTA: Robust Contrastive Vision-Language Test-Time Adaptation \- arXiv, accessed June 14, 2026, [https://arxiv.org/html/2507.14312v2](https://arxiv.org/html/2507.14312v2)  
24. GitHub \- dosowiechi/MLMP: Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation, accessed June 14, 2026, [https://github.com/dosowiechi/MLMP](https://github.com/dosowiechi/MLMP)  
25. Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation \- arXiv, accessed June 14, 2026, [https://arxiv.org/html/2505.21844v1](https://arxiv.org/html/2505.21844v1)  
26. gemini\_deep\_dive.md  
27. sauryeo/MoBaNet: MoBaNet is a multimodal remote sensing semantic segmentation framework for multimodal data. \- GitHub, accessed June 14, 2026, [https://github.com/sauryeo/MoBaNet](https://github.com/sauryeo/MoBaNet)  
28. Test-Time Adaptation with SaLIP: A Cascade of SAM and CLIP for Zero-shot Medical Image Segmentation \- CVF Open Access, accessed June 14, 2026, [https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/papers/Aleem\_Test-Time\_Adaptation\_with\_SaLIP\_A\_Cascade\_of\_SAM\_and\_CLIP\_CVPRW\_2024\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/papers/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.pdf)  
29. UniVCD: A New Method for Unsupervised Change Detection in the Open-Vocabulary Era, accessed June 14, 2026, [https://arxiv.org/html/2512.13089v1](https://arxiv.org/html/2512.13089v1)  
30. Randomized Histogram Matching: A Simple Augmentation for Unsupervised Domain Adaptation in Overhead Imagery \- ResearchGate, accessed June 14, 2026, [https://www.researchgate.net/publication/376348367\_Randomized\_Histogram\_Matching\_A\_Simple\_Augmentation\_for\_Unsupervised\_Domain\_Adaptation\_in\_Overhead\_Imagery](https://www.researchgate.net/publication/376348367_Randomized_Histogram_Matching_A_Simple_Augmentation_for_Unsupervised_Domain_Adaptation_in_Overhead_Imagery)  
31. Histogram Matching: Techniques & Applications \- Emergent Mind, accessed June 14, 2026, [https://www.emergentmind.com/topics/histogram-matching-approach](https://www.emergentmind.com/topics/histogram-matching-approach)  
32. Hybrid Object-Based Augmentation and Histogram Matching for Cross-Domain Building Segmentation in Remote Sensing \- MDPI, accessed June 14, 2026, [https://www.mdpi.com/2076-3417/16/1/543](https://www.mdpi.com/2076-3417/16/1/543)  
33. Combine Histogram Matching and Domain Adaptation to Cope with Temporal Transfer Learning for the Semantic Segmentation of VHR Images \- IEEE Xplore, accessed June 14, 2026, [https://ieeexplore.ieee.org/document/9883611/](https://ieeexplore.ieee.org/document/9883611/)  
34. Activities \- ORCID, accessed June 14, 2026, [https://orcid.org/0000-0001-9847-0243](https://orcid.org/0000-0001-9847-0243)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAXCAYAAAA7kX6CAAABH0lEQVR4XuXSP0sDQRCG8QlqQFQQDEpQEMXef2ChFhZCEDRVQD+EVkICYmNhk0qsBcs0wVorBcEilU0asbAQgjZWttFnbnePZTi4Xl/4cZeZ3b297In8iYyjjEkMmF5mNnCPC9RwjnfsxIPiFHCMZyyY3j4+sGzqyaS6uJXtJE0JXVzahq70hUPb8BnFg6f3SQbREve0uVA0GcOjmInTeENb3CJZmUcP1+JeK8kqvnESChk5QN9f04SJjbgYZRi36GAiboSthifqoe/6qyYcxZr/nUb3rIetKy7iFLM4QhUv2E5Hm4yIe/FPnKEibns3mBH3yQ2lozMyhT1soYlNX1/Hir/PzZW4HTzhTqLzy4v+9T94xZLp5aYo0YH/t/wCMlMslf4Xx2IAAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAYCAYAAADKx8xXAAAAzUlEQVR4XmNgGBlAC4iPA/EtIH4AxOeB2BlJ3huIvwLxMyC+AMRWSHIMLEC8BoivArEIsgQUlABxOBAzokuAwCQgfgjEkmji2kDcBcSsaOJwUM4AcZIxkhhIMUgTSDNO4AvE/4DYBU0sF4mPFYA0gDQGQfliQDwNiPnhKnAAkBNBTgU5GRQIlUBsi6ICBwAFCihw5gCxORA3MuAIRXRAtkZBID4NxLuBeCYQy6BK4wbcQLyHARJAoMgmCSwE4lUMeCIbF1BlwJ7kRsHgBgCXKx9vlSioFgAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAXCAYAAABEQGxzAAACmElEQVR4Xu2XTYhOURjHH6F8hZjIRxFJJGMSUmzkM7GQQlmRhSxGMwulNCiJErFgJRZiIRsRJYoFOxTZUFgQGyUUC/z/PefMPeeZc9/33Gs26v7q18x9znnvez6fc16Rhob/jlVwDxxiCwaRGfA4HGcLWjEKXoHf4B/4Gb6Hr51n4Nj+2koXvARHmvgEeBu+EX3HV9jnyk64GH0KN7p4O1bC03C4LWjHQfgRzgpibOAjeFOKxvPFF0W/qIzl8Cc8FsRmwrtwYRDLYZjo4K0x8ZbwQ9fhLTgiiHM5cfbewSkuxsay3hhfKQE7fwe+gpNFl8x50SVUh/USD2pbpsG3Eo8o6YAv4RMp1jHr2HoptsPfcBc8CZfGxZVg+9iGebagjBXwO1xt4tvgJykaw9nj7Gzqr1EOZ4YzxPeuNWVV4Wp4ADfYgjIOwB+wG26Fu0XX+0M4P6jHGXsMlwSxMobCa/ALXGDK6nAZ9thgCr9/uPlni+4VGu4lD+PP4GJbYODe2y/aISaHw1FpDJMMl+UNeEh0v0yNaijsEAe+LX7/nDXxFLkd2iLaodESJwcLO34ELnPPc0STED9nye4Q9w9HcbMtSJCz5Hi+nJLi3PDJgX8t7CRXx0T3zAHbWxRHZC859tqeP2Vw5O5JOilwtNeJHoJhevXJgTNl0y4H6IXoht8pAw9wT1ZS4LLgqf1L9IbwAe6IaqRJpe2jojcMvofv89mSI8+GMO6/o9eVeebCC6LZ9LkUsxXCbcE9njPolck5WHNgMuoMntkRzmLqrGGi4JLjZwYdLpur0vrqk8N0uC945iyck4HLstbVpypll9Mq8LZ+XzRlM2uyM4uiGkrty2lV/vXnA2/5PHzHw0nuf0utnw8NDQ3V+QsSHnQQAE49MgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAXCAYAAACrggdNAAACfUlEQVR4Xu2XT4iOQRzHf0L5r1DaHIgTSUmUwkFaJA6S5SZy4EIclItNOVAiF052L+LgJkJKceCguGgvFA6066CE4oDvp5nZd97xPM/7zLurqOdTn/ZtZpp5nt/M7/fMmjU0/NdslAflhLRjHGHuQ3Jr2tGJafKa/CJ/yRH5Tr7yXpSzRkc7VspBOTVpnyPvyNfm5vgsT/m+s74Nn8ttvr0Tk+UVuSbtqMNJ+UEujtp4yMfylrVegEWuyvVhUAFr5Xd5JmpbJO/JFVFbXZjvuv0ZxEomyZvytpwStbP97OJb2ePbWIBxM8KgAlj8rhyS8+VseVkujAdlwHwEdkvaUcUC+cbaIwvz5Ev51NyDAWPScUXskT/lfnnOujw+ESfkpbSxinXyq9yUtPfJYWs9ELvILm0fHVEOO8ROMW9v0tcNrPlATk87yiAK3+QRuUseMHf+H8ll0Th27olcHbWVMVHekJ/k8qSvG1bJZ+aC1ZGQTxSEJeZyB+PcCtD+wtwCVZCLR829FAWjv63XQZ5QqgliKkWIoMSwJmuH3K4k5FOd81r3pXaaeymOSlwwYvjOHTZXTRlPYAkw1bEoZ7NeinwimjvSjgLqHD++P+fNPSyEgsHfmH3WKj4E9Jj/zUPv9r9jso4f+ZR+n8og8iRrUaHgyG2WF6z9exIKBjsW2hkbEn6mufwluMCxCwGJqVUo2HK+7j/M3STey71tI4opKumnzd1EmIf5QhWdKx/69rDGcd8XWGpu9zvtAMFP1x036nx8c4jzqQx2mDGs/VdgAa4sVdekHOJ8KoOXyb4m5VJ2oc1hgxyQH+V9c6U8LeMwpgttLv/0vx4NDQ1j5zffOHMoWNpd2gAAAABJRU5ErkJggg==>