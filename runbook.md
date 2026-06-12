# Runbook — Fresh Kaggle Session

Follow top to bottom. Total cold-start time ≈ 5–8 minutes.

## 0. One-time setup (already done — verify only)
- Kaggle Secret `HF_TOKEN` attached to the notebook (Add-ons → Secrets).
- Datasets attached: `harish77718/ovrsis-potsdam-team1` (Potsdam patches),
  `rg-geoprompt-src` (this repo's `src/` folder, uploaded as a dataset),
  and — once it exists — `darmstadt-dop20`.
- Accelerator: **GPU T4** for notebooks 02/03/99; none needed for 01.

## 1. Updating the code dataset (whenever `src/` changes)
1. Zip `src/` locally (or push the folder as-is).
2. Kaggle → Datasets → `rg-geoprompt-src` → New Version → upload `src/`.
3. In the notebook: ⟳ refresh the dataset version under "+ Add Data".
The bootstrap cell in every notebook finds it at
`/kaggle/input/rg-geoprompt-src/src` automatically (falls back to `../src`
locally).

## 2. Cold start after any restart
Open **`99_recovery.ipynb`** and run all cells in order:
1. Bootstrap (paths + package import) — prints the resolved environment.
2. Pip installs (peft, open_clip_torch).
3. Recovery cell — rebuilds loaders, loads `text_embeddings.pt`
   (local → HF → only then re-encode), loads `dinov2_lora_best.pth`
   (strict, `_orig_mod.` stripped), loads `geoprompt_best.pth` if it exists.
4. Optional 5-batch mIoU smoke test (~84–85% expected).

Then continue in 02 or 03. Never rerun SegFormer or DINOv2+LoRA training.

## 3. Training RG-GeoPrompt (`02_train_potsdam.ipynb`)
Run cells in order:
1. Bootstrap → 2. installs → 3. loaders (2420/484 patches — verify the
   printout) → 4. text embeddings (must say "loaded from disk", not
   "encoded", unless this is the very first run) → 5. build model from
   DINOv2 checkpoint (forward smoke test prints `(1, 6, 512, 512)`).
2. Training cell (~45 min). Watch: τ should drift from 0.07 toward ~0.04;
   val loss should not explode. If CUDA OOM → rebuild the model with
   `lowres_similarity=True` and note it in the report.
3. If the session died mid-training: recover via 99, set `START_EPOCH` to
   the last completed epoch (see `geoprompt_training_log.csv`), load the
   latest `geoprompt_ckpt_epoch*.pth`, rerun the training cell.
4. Evaluation cells: mIoU (5 cls) then F1 (6 cls) for GeoPrompt, then F1
   for DINOv2+LoRA (and optionally SegFormer).
5. `hf_backup()` cell → CELL-END checklist (all ✓ before closing).
6. Belt-and-braces: download `geoprompt_best.pth` + CSVs from the Kaggle
   Output tab.

## 4. Darmstadt (`03_infer_darmstadt.ipynb`)
**Local one-time prep:** download 4–6 DOP20 tiles (gds.hessen.de →
Luftbildinformationen → DOP20), run the Step-0 cell locally
(`RUN_LOCAL_PREPROCESS = True`) — it slices RGBI→RGB 512² patches and
writes `transforms.json`. Upload the folder as Kaggle dataset
`darmstadt-dop20`, attach to the notebook.

**On Kaggle:** run cells in order: bootstrap → installs → load model +
patches → zero-shot + TTPA inference (collapse check prints a % — must be
~0; if >10%, rerun with `n_steps=1, lr=5e-6, kl_weight=1.0`) → OSM
download/rasterize/erode → pseudo-GT F1 (CSV written) → 5-column figures →
backup. Visually verify OSM building masks align with buildings before
trusting the F1 numbers.

## 5. End of every session — non-negotiable
1. Run the HF backup cell (uploads all checkpoints, logs, embeddings).
2. Run the CELL-END checklist; resolve any ✗.
3. Optionally download key `.pth`/`.csv` files from the Output tab
   (third backup layer).
