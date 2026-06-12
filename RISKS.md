# RISKS.md — Failure Modes, Detection, and Fixes
**RG-GeoPrompt-PEFT** | Companion to CLAUDE.md and runbook.md
Each entry: what happens → how to detect → how to fix. Items marked ⚠️ AUDIT are new findings from the June 2026 notebook audit, not in the original handoff.

---

## R1 — ⚠️ AUDIT: 5-class model head vs 6-class checkpoint (CELL16 bug)

**What happens:** The old notebook's CELL16 builds `DINOv2SegModel` with `NUM_CLASSES=5`. The trained 84.6% checkpoint has a **6-class** head. Loading crashes with a shape mismatch, or — worse — if a fresh 5-class model is trained, `cross_entropy` crashes the moment a clutter pixel (label 5) appears.

**Detect:** `load_state_dict` size-mismatch error on `decode_head`; or `CUDA error: device-side assert` during loss on a 5-class model.

**Fix:** Already fixed in this repo. `models_dino_lora.py` hard-asserts `num_classes == 6` and `load_dinov2_lora()` loads strict=True. Never copy CELL16 from the old notebook. Use `99_recovery.ipynb` to rebuild.

---

## R2 — ⚠️ AUDIT: Full-resolution cosine head OOM on T4 (unverified)

**What happens:** The spec-faithful `RGGeoPromptSegModel` upsamples 768-dim features to 512×512 *before* the cosine product → ~800 MB per image of intermediate activations. Batch 4 on a 16 GB T4 may OOM. **This model has never completed a training run — memory fit is unverified.**

**Detect:** `torch.cuda.OutOfMemoryError` in the first training step of `02_train_potsdam.ipynb` Stage C.

**Fix (in order):**
1. Reduce batch size 4 → 2 (keep LR; effective change is small over 10 epochs).
2. Build the model with `lowres_similarity=True` — cosine at 36×36, then upsample the 6-channel logits. Mathematically a mild approximation; document it in the report if used.
3. Enable AMP (`torch.cuda.amp.autocast`) for the forward pass only.

---

## R3 — Wrong resolution-bridge factor (GSD unverified)

**What happens:** `RESOLUTION_FACTOR=2` assumes 10 cm Potsdam data. If the patches were sliced from original 5 cm tiles, the correct factor is 4. Wrong factor → model trained to bridge to 40 cm or 10 cm instead of 20 cm → suboptimal Darmstadt transfer.

**Detect:** Run the rasterio GSD check on a source tile (`01_preprocess_potsdam.ipynb`, Cell 2) **before** any GeoPrompt training. `src.res == (0.1, 0.1)` → factor 2; `(0.05, 0.05)` → factor 4.

**Fix:** Edit `RESOLUTION_FACTOR` in `constants.py` (single source of truth — nothing else needs changing) and retrain Stage C only (~2–3 h).

---

## R4 — TTPA prediction collapse

**What happens:** Entropy minimization drives all pixels to one dominant class (usually impervious) — trivially minimal entropy.

**Detect:** `ttpa.detect_collapse()` runs automatically after TTPA inference: flags if dominant class fraction ≥ 0.90 **and** mean entropy ≤ 0.05. Also visible by eye: entropy map uniformly dark, prediction map one color.

**Fix:** Fallback parameters (already in `constants.py` as `TTPA_FALLBACK`): steps 2 → 1, LR 1e-5 → 5e-6, KL weight 0.5 → 1.0. If it still collapses, report zero-shot (no-TTPA) results as primary and TTPA as a negative finding — that is a legitimate result.

---

## R5 — ⚠️ AUDIT: TTPA state leakage between batches

**What happens:** The original handoff snippet adapted `text_proj` in place; weights adapted on patch *i* leaked into patch *i+1*, making results order-dependent and non-reproducible.

**Detect:** Re-running TTPA inference with shuffled patch order gives different metrics.

**Fix:** Already fixed — `ttpa.py` snapshots `text_proj` state before each batch and restores it after. Do not "optimize away" the snapshot/restore.

---

## R6 — GeoPrompt mIoU below DINOv2+LoRA (84.6%)

**What happens:** The cosine similarity head may underperform the conv head on supervised Potsdam.

**Detect:** Stage C validation mIoU < 84.6%.

**Fix:** Not a failure — an expected tradeoff. Report honestly: open-vocabulary capability (text-defined classes, cross-city transfer) is bought at some supervised accuracy. Do **not** silently swap back to the conv head; that breaks the brief's open-vocabulary requirement.

---

## R7 — DOP20 format surprises (RGBI, size, CRS)

**What happens:** DOP20 GeoTIFFs are 4-band RGBI. Feeding 4 channels into DINOv2 crashes; tiles are 5000×5000 and can exhaust RAM if loaded carelessly.

**Detect:** Channel-count assertion in `datasets.slice_dop20_tile()`; rasterio `src.count == 4`.

**Fix:** Already handled — slicer takes bands [1,2,3] only and streams windowed reads. NIR (band 4) is extracted separately for NDVI. Process one tile at a time. CRS must be EPSG:25832; the slicer asserts this.

---

## R8 — OSM misalignment / noise

**What happens:** OSM masks offset from imagery by meters to tens of meters; missing buildings; coarse vegetation polygons. Raw OSM as GT would systematically penalize a correct model.

**Detect:** Visual overlay check (`utils.five_column_figure`, column 5) — building outlines should sit on rooftops. If shifted, suspect CRS handling.

**Fix:** (1) Verify reprojection to EPSG:25832 happened (`osm_eval.py` does this; don't bypass). (2) Default 3×3 erosion marks boundaries 255; if alignment is poor, raise to 5×5 via the `kernel_size` arg. (3) Always label the metric "OSM pseudo-GT F1" in tables — never "ground truth".

---

## R9 — Kaggle session restart / variable loss

**What happens:** Session dies; all in-memory state gone.

**Detect:** Obvious.

**Fix:** Cold start via `99_recovery.ipynb` (~5 min): bootstrap → imports → loaders → rebuild model → `ensure_checkpoint()` pulls from local or HF Hub → resume. Training resume is guarded by `START_EPOCH` read from the log CSV — it will not retrain completed epochs.

---

## R10 — ⚠️ AUDIT: Training log wipe on restart

**What happens:** The old loop opened the log with `mode="w"` — every restart erased history, breaking both resume logic and the report's training curves.

**Fix:** Already fixed — `utils.append_log_row()` is append-only with header-once semantics. Never reintroduce `open(log, "w")` in a training loop.

---

## R11 — ⚠️ AUDIT: Accidental recomputation of text embeddings

**What happens:** Old CELL27 re-encoded CLIP prompts on every run, silently overwriting `text_embeddings.pt` (and burning time loading CLIP).

**Fix:** Already fixed — `prompts.load_or_encode_text_embeddings()` loads the cached file if present; pass `force=True` only when prompts themselves change. If you edit `PROMPT_ENSEMBLE`, you **must** pass `force=True` and retrain Stage C.

---

## R12 — Metric definition swap (mIoU vs F1)

**What happens:** The two metrics have different class sets — mIoU: 5 classes, ignore (5, 255); F1: 6 classes, ignore (255,). Swapping them invalidates every number in the report.

**Fix:** Already guarded — `metrics.py` hard-asserts the ignore sets and class counts per function and will raise rather than compute the wrong thing. Constants live only in `constants.py`.

---

## R13 — Clutter F1 looks bad (~20–30%)

**What happens:** Mean 6-class F1 appears poor next to the 5-class mIoU.

**Fix:** Expected. Clutter is a visually incoherent catch-all (5.7% of pixels). Report both mean-F1-with-clutter (brief requirement) and the per-class table so readers see the other 5 classes are strong. One discussion paragraph; not a problem to "solve".

---

## R14 — Supervised vs zero-shot comparison trap

**What happens:** Putting our 84.6% supervised mIoU next to SegEarth-OV's 47% zero-shot Potsdam number implies a 37-point superiority that does not exist — different evaluation settings.

**Fix:** Two separate tables, always (Table 1 supervised Potsdam, Table 2 zero-shot transfer). The report must state the setting in each caption. CLAUDE.md rule #9 enforces this for any future drafting.

---

## R15 — HF token exposure

**What happens:** Token hardcoded in a cell → leaked on notebook share. (Happened once already; that token was revoked.)

**Fix:** `utils.hf_backup()` reads Kaggle Secret `HF_TOKEN` (env var `HF_TOKEN` locally). There is no code path that accepts a token string argument. Keep it that way.

---

## Quick triage table

| Symptom | Likely risk | First action |
|---|---|---|
| Shape mismatch loading checkpoint | R1 | Use 99_recovery, never CELL16 |
| OOM in Stage C step 1 | R2 | batch 2, then `lowres_similarity=True` |
| Darmstadt preds all one class | R4 | TTPA fallback params |
| TTPA results change with patch order | R5 | Verify snapshot/restore intact |
| OSM buildings offset from rooftops | R8 | Check CRS, raise erosion to 5×5 |
| Training restarts from epoch 1 | R9/R10 | Check log CSV exists & append mode |
| device-side assert in loss | R1 | 6-class head check |
