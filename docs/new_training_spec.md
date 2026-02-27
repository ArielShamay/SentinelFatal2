# SentinelFatal2 — New Training Notebook Specification
## `docs/new_training_spec.md`
### SSOT (Single Source of Truth) for Notebook: `09_e2e_cv_v3.ipynb`

**Version:** 1.0  
**Date:** 2026-02-27  
**Author:** Specification Agent  
**Status:** Authoritative — do not deviate without updating this document

---

## Table of Contents

1. [Title and Purpose](#1-title-and-purpose)
2. [Background: Why This Run Exists (Post-Mortem)](#2-background-why-this-run-exists-post-mortem)
3. [Critical Rules for the Notebook-Writing Agent](#3-critical-rules-for-the-notebook-writing-agent)
4. [Repository and Environment Setup](#4-repository-and-environment-setup)
5. [Data Strategy](#5-data-strategy)
6. [Notebook Cell Map](#6-notebook-cell-map)
7. [Fold Loop Detailed Specification](#7-fold-loop-detailed-specification)
8. [Per-Epoch Checkpoint Management](#8-per-epoch-checkpoint-management)
9. [Timeout Guard Implementation Pattern](#9-timeout-guard-implementation-pattern)
10. [Resume Protocol](#10-resume-protocol)
11. [Expected Metrics and Intervention Guide](#11-expected-metrics-and-intervention-guide)
12. [GitHub Export (Cell EXPORT)](#12-github-export-cell-export)
13. [Gates and Stop Conditions Summary](#13-gates-and-stop-conditions-summary)
14. [File and Directory Layout](#14-file-and-directory-layout)
15. [Differences from Old Notebook (08_e2e_cv_v2.ipynb)](#15-differences-from-old-notebook-08_e2e_cv_v2ipynb)
16. [Checklist for Handoff](#16-checklist-for-handoff)

---

## 1. Title and Purpose

**Notebook filename:** `notebooks/09_e2e_cv_v3.ipynb`  
**Runtime platform:** Google Colab (GPU)  
**Primary goal:** Execute a complete, stable, resumable 5-Fold End-to-End Cross-Validation (E2E CV) for the SentinelFatal2 fetal distress detection model.

### What This Notebook Produces

- Out-of-Fold (OOF) AUC score with 95% bootstrap confidence interval across all 552 CTU-UHB recordings.
- Per-fold AUC, sensitivity, specificity, and clinical threshold values.
- A final comparison CSV summarising model performance under Config A.
- Best `finetune.pt` checkpoints for each fold, zipped and pushed to GitHub.
- All logs (finetune loss curves, validation AUCs per epoch) for each fold.

### What This Notebook Does NOT Do

- It does **not** run Config B or Config C. Config A is the only configuration used.
- It does **not** perform a hyperparameter sweep or A/B/C selection loop.
- It does **not** rely on Google Drive for data — all data is fetched from GitHub via `git clone`.

### Architecture Summary

SentinelFatal2 uses a **PatchTST foundation model** pre-trained on CTG (Cardiotocography) signals from 687 recordings (552 CTU-UHB + 135 FHRMA). The pre-trained backbone is then fine-tuned with a binary `ClassificationHead` on 5-fold CV splits of the 552 CTU-UHB recordings. A downstream Logistic Regression with PCA is fitted on extracted features to produce the final recording-level score. Clinical threshold selection enforces a specificity constraint of ≥ 0.65.

**Key model dimensions (Config A):**
- Input shape per recording window: `(batch, 2, 1800)` — 2 channels (FHR + UC), 1800 samples at fs=4 Hz
- Backbone: PatchTST with `d_model=128`, `num_layers=3`, `n_heads=4`, `ffn_dim=256`, `dropout=0.2`
- Patch parameters: `patch_len=48`, `patch_stride=24`, `n_patches=73`
- Classification head input dimension: `d_in = n_patches × d_model × n_channels = 73 × 128 × 2 = 18688`

---

## 2. Background: Why This Run Exists (Post-Mortem)

### 2.1 The Failure of `08_e2e_cv_v2.ipynb`

The previous notebook (`notebooks/08_e2e_cv_v2.ipynb`) was launched in Google Colab with the intent of running a complete 5-fold E2E CV. It ran successfully through pre-training and began finetune training for Fold 0 (Config A of the A/B/C selection loop). It then **froze permanently at epoch 66** of the finetune phase. No Python exception was raised; no error appeared in the logs. The Colab kernel eventually timed out and the session ended, losing all model weights that had not been committed to GitHub.

From the training logs: epoch 66 completed normally (`train_loss=0.678251`, `val_auc=0.659091`, elapsed 115.2s). The hang occurred *after* epoch 66 completed, during the post-epoch validation or SWA update step.

### 2.2 Root Cause Analysis

Three independent failure modes combined to produce the crash.

**Root Cause 1 — SWA + Multi-Worker DataLoader Deadlock (Primary)**

Stochastic Weight Averaging (SWA) was configured to begin at epoch 50. At approximately epoch 66, the combination of: (a) SWA model updates, (b) a multi-worker DataLoader, and (c) recording-level per-epoch AUC validation — produced a **silent deadlock**. Worker processes spawned by the DataLoader attempted to synchronize across the SWA update boundary, and the main process entered a blocking wait from which it never returned. No CUDA OOM was triggered, no Python exception was raised.

Observable sign: from epoch ~20 onward, per-epoch time increased from ~88s to ~115s (SWA + progressive unfreezing). The deadlock struck at epoch 66 — 16 epochs after SWA started.

The fix is two-fold: (a) restrict DataLoader to `num_workers=0` or at most 2, and (b) perform recording-level AUC validation only every 5 epochs, reducing the number of times DataLoader and SWA must synchronize.

**Root Cause 2 — Memory Bloat Over 66 Epochs**

The notebook printed per-batch output on every batch of every epoch. Over 66 epochs × ~1900 batches per epoch (at stride=60), this generated millions of lines of output retained in Jupyter's output buffer. Additionally, accumulated DataLoader worker buffers were never explicitly flushed between epochs. The cumulative memory pressure — both RAM and GPU VRAM — degraded performance and likely contributed to the blocking condition.

The fix: (a) eliminate per-batch printing entirely — only epoch-level summaries, and (b) after every epoch call `sys.stdout.flush()`, `gc.collect()`, `torch.cuda.empty_cache()`.

**Root Cause 3 — No Mid-Run Checkpoint Commit to GitHub**

The old notebook saved epoch checkpoints locally to `/content/SentinelFatal2/checkpoints/e2e_cv_v2/`. When the Colab session died, all checkpoints were lost (~5 hours of compute wasted). No useful artifacts were preserved on GitHub from that run.

The fix: save a checkpoint after **every epoch** to the local Colab filesystem (enables session-reconnect resume). Commit completed fold OOF CSVs to GitHub immediately after each fold (survives runtime recycle). Push all final artifacts in one batch after all 5 folds complete.

### 2.3 Additional Structural Issue: Config A/B/C Loop

The old notebook began by running three full finetune pipelines (Config A, B, C) in series before starting cross-validation. Each required ~3–4 hours. The session expired (likely ~12 hours) before Config A itself finished training, so no useful A/B/C comparison was ever obtained.

The fix: **this notebook is locked to Config A only**. No config selection loop. Config A is hardcoded via `config_overrides`. Expected total runtime drops to 6–8 hours.

### 2.4 What Is Preserved on GitHub (Pre-Run State)

These artifacts are available on `origin/master` and must NOT be regenerated if they pass sanity checks:

- `checkpoints/e2e_cv_v2/shared_pretrain/best_pretrain.pt` — shared pre-trained backbone (skip pre-training if present and val_mse < 0.015)
- `logs/e2e_cv_v2/shared_pretrain_loss.csv` — pre-training loss curve
- `data/splits/*.csv` — all split files (do not regenerate; these are deterministic)
- `results/e2e_cv_v2/` — any partially completed fold OOF CSVs from previous runs (skip those folds)

> **Note on checkpoint path:** The old notebook saved the shared pretrain checkpoint to `checkpoints/e2e_cv_v2/shared_pretrain/best_pretrain.pt`. The new notebook must look for it there first, then fall back to `checkpoints/pretrain/best_pretrain.pt` if not found.

---

## 3. Critical Rules for the Notebook-Writing Agent

These rules are **non-negotiable**. Any notebook that violates them is incorrect and must be rewritten before execution.

> **⚠️ Pre-condition:** Rules 6, 8, and 11 require changes to `src/train/finetune.py`. These changes are minimal and surgical. A dedicated section (Section 3.5) describes exactly what to add. The notebook writer MUST apply these changes to the source file before writing the notebook cells that call `finetune_train()`.

1. **Config A ONLY.** No Config B, no Config C, no selection loop. The `config_overrides` dict passed to every `finetune_train()` call must be `{'loss': 'cross_entropy'}`. Config A uses Weighted Cross-Entropy (not Focal Loss). The positive class weight of **3.9** is handled by the existing finetune code when `loss='cross_entropy'` is selected — do not double-apply it.

2. **`train_stride=120` to halve training windows.** `train_stride` is a finetune-section config key read by `finetune.py` to set the training DataLoader window stride. Pass `config_overrides={'loss': 'cross_entropy', 'train_stride': 120}`. At stride=120, the training DataLoader has ~60K windows (was ~117K at stride=60), cutting per-epoch training time roughly in half.

   Also pass `'val_stride': 120` to halve the stride used inside `compute_recording_auc()` for per-epoch validation AUC. Note: the _val DataLoader_ itself uses a separate key `window_stride` (defaults to `pretrain.window_stride=900`) which is already sparse and fast — no need to change it.

3. **DataLoader `num_workers` must be 0.** In `finetune.py`, `num_workers` is hardcoded as `min(2, os.cpu_count() or 0)`. On Colab T4 (multi-core), this evaluates to **2**. It is NOT a config key. The only way to override it without modifying source files is to **monkey-patch `torch.utils.data.DataLoader`** in the notebook before calling `finetune_train()` (see Section 7.d). This is the **single most critical safeguard** against the epoch-66 deadlock.

4. **Per-epoch memory flush after every epoch.** The callback or wrapping layer must call `sys.stdout.flush()`, `gc.collect()`, and `torch.cuda.empty_cache()` at the end of every epoch, without exception.

5. **No per-batch logging.** Only epoch-level summaries may appear in cell output. The `quiet=True` parameter in `finetune_train()` suppresses internal per-batch printing — always pass it.

6. **Recording-level AUC validation every 5 epochs only.** Currently `finetune.py` calls `compute_recording_auc()` unconditionally every epoch. This must be changed by adding a `val_every_n_epochs: int = 1` parameter to `train()` that skips the AUC call on non-multiple epochs. See Section 3.5 for the exact 3-line change.

7. **Timeout guard on validation — 180 seconds hard cap.** The recording-level AUC computation must be wrapped in a `threading.Timer` (see Section 9). If it does not complete within 180 seconds: skip updating `best_finetune.pt`, log a warning, return control to the epoch loop. Training must not crash.

8. **Save one checkpoint per epoch AND support resume.** `finetune.py` already saves `epoch_{n:03d}.pt` when `save_epoch_ckpts=True`. However, it does NOT support resuming from a checkpoint — the loop always starts at epoch 0. A `resume_from_epoch: int = 0` parameter must be added to `train()`. See Section 3.5.

9. **Keep only last 3 + best.** After saving `epoch_{n:03d}.pt`, delete epoch checkpoints older than the 3 most recent. Never delete `best_finetune.pt`.

10. **Resume-aware from first cell.** At the start of each fold, scan the checkpoint directory for the latest `epoch_*.pt`. If found, load it and begin the epoch loop from `resume_epoch + 1`. If a fold's `fold{k}_oof_scores.csv` already exists, skip the entire fold.

11. **patience=15 (changed from 25).** Pass `'patience': 15` in `config_overrides`. Note: the EMA beta value used for smoothing is **hardcoded at 0.8** in `finetune.py` (`ema_beta = 0.8`) — it cannot be changed via config. The early stopping counter increments on every epoch where `smooth_auc` does not improve (every epoch, not just validation epochs).

12. **No mid-run GitHub pushes except post-fold OOF CSV mini-commits.** Mid-training epoch checkpoints stay local only. After each fold completes, commit only the OOF CSV.

13. **All data from GitHub, not Google Drive.** Data via `git clone` or zip download from the repo. No `google.colab.drive` mounting for data.

---

## 3.5 Required Modifications to `src/train/finetune.py`

The following three parameters must be added to `train()` before writing the notebook. They are **surgical additions** (~20 lines total) that do not break any existing behavior when default values are used.

### 3.5.1 Add `val_every_n_epochs` (for every-5-epoch AUC evaluation)

Current signature (line ~308):
```python
def train(config_path, device_str="cpu", max_batches=0,
          processed_root=None, train_csv=None, val_csv=None,
          pretrain_checkpoint=None, checkpoint_dir="checkpoints/finetune",
          log_path="logs/finetune_loss.csv", quiet=True,
          config_overrides=None, save_epoch_ckpts=True) -> None:
```

New signature:
```python
def train(config_path, device_str="cpu", max_batches=0,
          processed_root=None, train_csv=None, val_csv=None,
          pretrain_checkpoint=None, checkpoint_dir="checkpoints/finetune",
          log_path="logs/finetune_loss.csv", quiet=True,
          config_overrides=None, save_epoch_ckpts=True,
          val_every_n_epochs: int = 1,
          resume_from_epoch: int = 0,
          per_epoch_callback=None) -> None:
```

**Change 1 — skip AUC on non-validation epochs** (find the `compute_recording_auc` call in the epoch loop, ~line 522):
```python
# BEFORE:
val_auc = compute_recording_auc(model, val_loader, device, stride_val)

# AFTER:
if val_every_n_epochs > 1 and (epoch % val_every_n_epochs) != 0:
    val_auc = prev_val_auc  # carry forward last AUC without re-computing
else:
    val_auc = compute_recording_auc(model, val_loader, device, stride_val)
prev_val_auc = val_auc
```
Also initialise `prev_val_auc = 0.0` before the epoch loop.

**Change 2 — resume support** (find `for epoch in range(max_epochs)`, ~line 490):
```python
# BEFORE:
for epoch in range(max_epochs):

# AFTER:
for epoch in range(resume_from_epoch, max_epochs):
```

**Change 3 — per-epoch callback** (add at the very end of the epoch body, after checkpoint/early-stop logic):
```python
if per_epoch_callback is not None:
    per_epoch_callback(epoch, float(train_loss), float(val_auc))
```

### 3.5.2 How the Notebook Uses These Parameters

```python
def _epoch_callback(epoch, train_loss, val_auc):
    """Called by finetune.train() after every epoch."""
    _prune_old_checkpoints(ckpt_dir, keep=3)   # keep last 3 + best
    _maybe_push_to_github(epoch)               # push only on milestone epochs
    if _session_time_remaining() < TIMEOUT_SECONDS:
        raise TimeoutError(f"Session timeout guard triggered at epoch {epoch}")

finetune.train(
    ...,
    val_every_n_epochs=VAL_EVERY_N_EPOCHS,  # = 5
    resume_from_epoch=resume_from,           # detected from latest epoch_*.pt
    per_epoch_callback=_epoch_callback,
)
```

### 3.5.3 Verification After Modifying `finetune.py`

After applying the changes, run this smoke test **before** writing the fold loop:
```python
import src.train.finetune as ft
import inspect
sig = inspect.signature(ft.train)
assert 'val_every_n_epochs' in sig.parameters, "Missing val_every_n_epochs"
assert 'resume_from_epoch'  in sig.parameters, "Missing resume_from_epoch"
assert 'per_epoch_callback' in sig.parameters, "Missing per_epoch_callback"
print("finetune.py signature OK")
```

---

## 4. Repository and Environment Setup

### 4.1 Constants

Define at the top of Cell SEED:

```python
REPO_URL    = "https://github.com/ArielShamay/SentinelFatal2.git"
BRANCH      = "master"
REPO_DIR    = "/content/SentinelFatal2"   # local path in Colab
CONFIG_PATH = f"{REPO_DIR}/config/train_config.yaml"
SEED        = 42
N_FOLDS     = 5
N_BOOTSTRAP = 10_000
SPEC_CONSTRAINT = 0.65
AT_CANDIDATES   = [0.30, 0.35, 0.40, 0.45]
N_FEATURES      = 12
PATIENCE        = 15
EMA_BETA        = 0.8
VAL_EVERY_N_EPOCHS = 5
TIMEOUT_SECONDS = 180
```

Detect `IS_COLAB` by checking `'google.colab' in sys.modules or os.path.exists('/content')`.

### 4.2 Repository Clone / Pull Pattern

Cell REPO must:

1. Check whether `REPO_DIR` already exists.
2. If **not**: `git clone --branch {BRANCH} {REPO_URL} {REPO_DIR}`.
3. If **yes**: remove any stale `.git/*.lock` files, then `git -C {REPO_DIR} fetch origin {BRANCH}`, then `git -C {REPO_DIR} reset --hard origin/{BRANCH}`. This guarantees a clean state without risking an interrupted-merge error.
4. Run `pip install -q -r {REPO_DIR}/requirements.txt`.
5. `sys.path.insert(0, REPO_DIR)`.

The `git reset --hard` pattern (rather than `git pull`) is chosen because a partial previous run may have modified working-tree files.

### 4.3 Data Download

After cloning, check whether `data/processed/ctu_uhb/` contains `.npy` files. If it does not (because large files are gitignored), download `data_processed.zip` from the raw GitHub URL:

```
https://raw.githubusercontent.com/ArielShamay/SentinelFatal2/master/data_processed.zip
```

Use `wget -q -O {zip_path} {url}`, then `unzip -q {zip_path} -d {REPO_DIR}/data/`, then delete the zip. This is the same pattern used in Cell 4 of the old notebook (`08_e2e_cv_v2.ipynb`), which was verified to work.

### 4.4 Dependency Installation

Run `pip install -q -r requirements.txt` after clone. The `--quiet` flag prevents pip's verbose output from bloating cell output. Verify that `torch`, `scikit-learn`, `pandas`, `numpy`, `pyyaml` are importable after install. If any are missing, raise a hard error before proceeding.

### 4.5 Determinism Settings

Cell SEED must call:

```python
import random, numpy as np, torch
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

### 4.6 `config_overrides` Key Format

The `load_config()` function (`src/model/patchtst.py`, lines 36–55) accepts an `overrides` dict with **flat keys only** (no dotted paths). The routing logic is:

```python
for k, v in overrides.items():
    if k in cfg['model']:        # e.g. d_model, n_layers, n_heads, dropout
        cfg['model'][k] = v
    else:                        # everything else → finetune section
        cfg['finetune'][k] = v
```

For Config A, the correct `config_overrides` dict is:

```python
CONFIG_A_OVERRIDES = {
    'loss':               'cross_entropy',  # → cfg['finetune']['loss']
    'class_weight':       [1.0, 3.9],       # → cfg['finetune']['class_weight']
    'patience':           15,               # → cfg['finetune']['patience'] (was 25)
    'train_stride':       120,              # → cfg['finetune']['train_stride'] (training DataLoader)
    'val_stride':         120,              # → cfg['finetune']['val_stride'] (compute_recording_auc stride)
    # NOTE: 'window_stride' NOT overridden — val DataLoader already uses
    #       pretrain.window_stride=900 by default (sparse, fast, no change needed)
}
```

**Critical naming rules:**
- `train_stride` — controls the window stride used when building the **training** DataLoader
- `val_stride` — controls the stride passed to `compute_recording_auc()` (recording-level AUC)
- `window_stride` — controls the **val** DataLoader stride (defaults to `pretrain.window_stride=900` when not set in `cfg['finetune']`; already fast, do NOT override)
- **Never** use dotted keys like `'finetune.patience'` — they will be silently ignored or placed under a key literally named `'finetune.patience'`

### 4.7 Device Selection

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    raise RuntimeError("No GPU detected. Change runtime to GPU in Colab settings.")
```

---

## 5. Data Strategy

### 5.1 Primary Data Source

Processed `.npy` files are committed to the repository (or stored as `data_processed.zip` in the repo). No separate preprocessing step is needed. The notebook pulls data via `git clone` + optional zip extraction (Section 4.3).

### 5.2 Expected Data Counts

Cell SANITY must assert:

| Dataset | Expected Count | Path |
|---------|---------------|------|
| CTU-UHB processed | 552 `.npy` files (excluding `.gitkeep`) | `data/processed/ctu_uhb/` |
| FHRMA processed | 135 `.npy` files (excluding `.gitkeep`) | `data/processed/fhrma/` |
| pretrain.csv rows | 687 | `data/splits/pretrain.csv` |
| train.csv rows | 441 | `data/splits/train.csv` |
| val.csv rows | 56 | `data/splits/val.csv` |
| test.csv rows | 55 | `data/splits/test.csv` |

Raise a hard stop (`assert` with descriptive message) if any count does not match.

### 5.3 Constructing `train_val_test.csv`

1. Load `train.csv`, `val.csv`, `test.csv` (from `data/splits/`).
2. Concatenate into one DataFrame.
3. Drop duplicates on the `id` column.
4. Assert `len(df) == 552`.
5. Save to `data/splits/train_val_test.csv` (overwrite if exists).
6. Print confirmation: "Built train_val_test.csv: 552 recordings".

The CSV must have the same columns as the individual split files (typically `id`, `target` or equivalent label column). Verify the exact column names from `data/splits/train.csv`.

### 5.4 Processed Data Format

Each `.npy` file: shape `(C, L)` where `C=2` (FHR + UC), `L` varies per recording (sampled at `fs=4` Hz). Preprocessing (resampling, normalization) is already applied. The notebook must not re-run preprocessing.

---

## 6. Notebook Cell Map

The notebook `09_e2e_cv_v3.ipynb` must contain exactly the following cells, in order.

---

### Cell GPU_DIAG — GPU Diagnostic + Stale Process Cleanup

Query GPU using `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader`. Print a summary line. Query compute-app PIDs via `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`. Kill any non-kernel Python PIDs. Call `torch.cuda.empty_cache()`. Print free/total VRAM after cleanup.

> This replicates the working pattern from the first cell of `08_e2e_cv_v2.ipynb`.

---

### Cell SEED — Seeds, Constants, Determinism

Set all seeds (Section 4.5). Define all constants (Section 4.1). Detect `IS_COLAB`. Detect `DEVICE` (Section 4.7). Print confirmation.

---

### Cell REPO — Clone / Pull + pip install

Clone or pull as in Section 4.2. Install requirements. Print git commit hash (`git -C {REPO_DIR} rev-parse --short HEAD`).

---

### Cell GPU_PREFLIGHT — GPU Smoke Test

```python
x = torch.randn(4, 2, 1800, device=DEVICE)
assert x.shape == (4, 2, 1800)
del x; torch.cuda.empty_cache()
print("[OK] GPU smoke test passed")
```

Raise a hard stop if this fails.

---

### Cell MODULE_RELOAD — Force-Reload src.* Modules

Delete all `sys.modules` entries starting with `src.` or `scripts.`, then re-import:

```python
from src.train.finetune    import train as finetune_train, load_pretrained_checkpoint
from src.train.pretrain    import pretrain
from src.train.utils       import compute_recording_auc
from src.model.patchtst    import PatchTST, load_config
from src.model.heads       import ClassificationHead
from scripts.run_e2e_cv_v2 import (
    generate_cv_splits, run_fold,
    extract_features_for_split, fit_lr_model, predict_lr,
    at_sweep, clinical_threshold, bootstrap_auc_ci,
    N_BOOTSTRAP, SEED as CV_SEED
)
```

Verify that `finetune_train` signature has `config_overrides` and `save_epoch_ckpts` parameters (using `inspect.signature`). Raise an assertion error if not — this catches stale cached code.

---

### Cell DATA_DOWN — Data Download + Directory Setup

1. Check whether CTU-UHB `.npy` files exist and count them (exclude `.gitkeep`).
2. If count < 552, download and extract `data_processed.zip` (Section 4.3).
3. Construct `train_val_test.csv` (Section 5.3).
4. Create all output directories:

```
checkpoints/e2e_cv_v2/shared_pretrain/
checkpoints/e2e_cv_v2/fold{k}/finetune/   for k in 0..4
logs/e2e_cv_v2/
results/e2e_cv_v2/
results/e2e_cv_v2/fold{k}_splits/          for k in 0..4
```

---

### Cell SANITY — Full Data Integrity Check

Run all count assertions (Section 5.2). Additionally:
- Verify label distribution: `target.value_counts()` shows 113 positives (acidemia) and 439 negatives in `train_val_test.csv`.
- For each ID in `train_val_test.csv`, verify `data/processed/ctu_uhb/{id}.npy` exists.
- Print a checklist with PASS/FAIL for each check.
- Raise a hard stop on any failure.

---

### Cell PRETRAIN — Shared Pre-Training (Skip if Checkpoint Exists)

Check for `checkpoints/e2e_cv_v2/shared_pretrain/best_pretrain.pt`. Also check the fallback path `checkpoints/pretrain/best_pretrain.pt`.

If either exists: print "Pre-training checkpoint found — skipping." Set `SHARED_PRETRAIN_CKPT` to the found path.

If neither exists: run pre-training:

```python
pretrain(
    config_path    = CONFIG_PATH,
    device_str     = DEVICE,
    max_batches    = 0,
    processed_root = f"{REPO_DIR}/data/processed",
    pretrain_csv   = f"{REPO_DIR}/data/splits/pretrain.csv",
    checkpoint_dir = f"{REPO_DIR}/checkpoints/e2e_cv_v2/shared_pretrain",
    log_path       = f"{REPO_DIR}/logs/e2e_cv_v2/shared_pretrain_loss.csv",
)
SHARED_PRETRAIN_CKPT = f"{REPO_DIR}/checkpoints/e2e_cv_v2/shared_pretrain/best_pretrain.pt"
```

After pre-training, call `gc.collect()` and `torch.cuda.empty_cache()`.

---

### Cell G1_GATE — Pre-Train Quality Gate

Load `logs/e2e_cv_v2/shared_pretrain_loss.csv` (or `logs/pretrain_loss.csv` as fallback). Find the minimum `val_loss` column. If val_mse < 0.015: print "G1 GATE PASSED". If ≥ 0.015: print a prominent warning (do not hard-stop). The decision to continue is the operator's.

---

### Cell CV_SPLITS — Generate 5-Fold Splits

```python
all_csv     = Path(REPO_DIR) / "data/splits/train_val_test.csv"
cv_splits   = generate_cv_splits(all_csv, n_folds=N_FOLDS, seed=SEED)
```

Print, for each fold k:
- Number of train / val / test recordings
- Acidemia fraction in each subset

Save each fold's split CSVs to `results/e2e_cv_v2/fold{k}_splits/`. Store `cv_splits` for FOLD_LOOP.

---

### Cell FOLD_LOOP — Main 5-Fold Training + Evaluation

This is the longest cell. Full specification in Section 7. After all 5 folds, print the fold-level summary table.

---

### Cell AT_SUMMARY — AT Sweep Results Across Folds

Load `results/e2e_cv_v2/fold{k}_at_sweep.csv` for each fold. Print a consolidated table: fold × AT_candidate → val_auc, sensitivity, specificity. Highlight the best AT per fold. Save to `results/e2e_cv_v2/at_sweep_summary.csv`.

---

### Cell BOOTSTRAP_CI — Global OOF AUC + 95% CI

1. Load all 5 `fold{k}_oof_scores.csv`. Concatenate into global OOF of 552 rows.
2. Save `results/e2e_cv_v2/global_oof_predictions.csv`.
3. `global_auc = roc_auc_score(y_true, y_score)`.
4. `ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP, seed=SEED)`.
5. Apply `clinical_threshold(y_true, y_score, spec_constraint=SPEC_CONSTRAINT)` for global operating point.
6. Print formatted summary block.
7. Build per-fold summary DataFrame and save to `results/e2e_cv_v2/per_fold_summary.csv`.
8. Save final report to `results/e2e_cv_v2/final_cv_report_v3.csv`.
9. Build comparison table (see Section 12.5) and save to `results/e2e_cv_v2/comparison_table_v3.csv`.

---

### Cell REPRO_TRACK — Reproducibility Run on 441/56/55 (Optional)

Using `SHARED_PRETRAIN_CKPT`, fine-tune once on canonical `train.csv`/`val.csv`, evaluate on `test.csv`. The AT sweep and threshold are locked on `val.csv` (56 recordings) — never on `test.csv`. Report AUC, sensitivity, specificity, and compare against the prior best (AUC=0.839, Sens=0.818, Spec=0.773, AT=0.40, threshold=0.284) from `results/final_report.md`.

Save to `results/e2e_cv_v2/repro_comparison_v3.csv`. Skip this cell if the session is running low on time.

---

### Cell EXPORT — Push Final Results to GitHub

Full specification in Section 12. Runs only after all 5 fold OOF CSVs and `final_cv_report_v3.csv` exist.

---

## 7. Fold Loop Detailed Specification

The FOLD_LOOP cell iterates `k = 0` to `4`. Below is the precise logic for each iteration.

### 7.a Fold-Level Skip

```
if results/e2e_cv_v2/fold{k}_oof_scores.csv exists:
    print "Fold {k}: OOF CSV found — skipping (already complete)."
    continue
```

### 7.b Write Fold Split CSVs

Save `cv_splits[k]` train/val/test DataFrames to:
- `results/e2e_cv_v2/fold{k}_splits/fold{k}_train.csv`
- `results/e2e_cv_v2/fold{k}_splits/fold{k}_val.csv`
- `results/e2e_cv_v2/fold{k}_splits/fold{k}_test.csv`

### 7.c Checkpoint Scan — Resume Epoch Detection

Scan `checkpoints/e2e_cv_v2/fold{k}/finetune/` for `epoch_*.pt` files. Parse the integer suffix from each filename. If none found: `resume_from = None`, `resume_epoch = 0`. If found: select the highest-numbered one, set `resume_from = {that file path}`, `resume_epoch = {that epoch number}`.

Print: "Fold {k}: Resuming from epoch {resume_epoch}" or "Fold {k}: Starting fresh from epoch 0."

### 7.d The `num_workers=0` Monkey-Patch

Before calling `finetune_train()`, apply the monkey-patch to force `num_workers=0`:

```python
import torch.utils.data as _data
_orig_DataLoader = _data.DataLoader
def _patched_DataLoader(*args, **kwargs):
    kwargs['num_workers'] = 0
    kwargs['pin_memory']  = False
    return _orig_DataLoader(*args, **kwargs)
_data.DataLoader = _patched_DataLoader
```

Restore original after `finetune_train()` returns:

```python
_data.DataLoader = _orig_DataLoader
```

This is the safest approach since `num_workers` is not a config key. Alternatively, if `finetune_train()` already accepts `num_workers` as a parameter (verify by inspecting the signature), pass it directly.

### 7.e Finetune Training Call

With the modifications from Section 3.5, the call is:

```python
def _epoch_callback(epoch, train_loss, val_auc):
    """Called by finetune_train() at the end of every epoch."""
    # 1. Prune old epoch checkpoints (keep last 3 + best)
    _prune_old_checkpoints(fold_ckpt_dir, keep=3)
    # 2. Flush memory
    gc.collect()
    torch.cuda.empty_cache()
    # 3. Timeout guard
    if _session_time_remaining() < TIMEOUT_SECONDS:
        print(f"[TIMEOUT GUARD] Fold {k} epoch {epoch} — saving and exiting")
        raise TimeoutError(f"Timeout guard at epoch {epoch}")

def _detect_resume_epoch(ckpt_dir):
    import glob, re
    pts = sorted(glob.glob(f"{ckpt_dir}/epoch_*.pt"))
    if not pts:
        return 0
    m = re.search(r'epoch_(\d+)\.pt', pts[-1])
    return int(m.group(1)) + 1 if m else 0

resume_from = _detect_resume_epoch(fold_ckpt_dir)
print(f"Fold {k}: resuming from epoch {resume_from}")

finetune_train(
    config_path          = CONFIG_PATH,
    device_str           = DEVICE,
    max_batches          = 0,
    processed_root       = f"{REPO_DIR}/data/processed",
    train_csv            = str(fold_train_csv),
    val_csv              = str(fold_val_csv),
    pretrain_checkpoint  = str(SHARED_PRETRAIN_CKPT),
    checkpoint_dir       = fold_ckpt_dir,
    log_path             = f"{REPO_DIR}/logs/e2e_cv_v2/fold{k}_finetune_loss.csv",
    quiet                = True,
    save_epoch_ckpts     = True,
    config_overrides     = CONFIG_A_OVERRIDES,   # defined in Cell CONSTANTS
    val_every_n_epochs   = VAL_EVERY_N_EPOCHS,   # = 5
    resume_from_epoch    = resume_from,
    per_epoch_callback   = _epoch_callback,
)
```

Wrap in `try/except`. On `TimeoutError`: print graceful exit, commit any new OOF CSV, and `break` the fold loop. On `KeyboardInterrupt`: same. On any other exception: log full traceback with `traceback.print_exc()` and `continue` to the next fold.

> **Requires:** The three changes in Section 3.5 must be applied to `src/train/finetune.py` before this cell executes.

### 7.f Per-Epoch Side-Channel Logic

All per-epoch side effects are handled through the `per_epoch_callback` added in Section 3.5. The callback defined in Section 7.e handles:

1. **Checkpoint pruning** — `_prune_old_checkpoints(fold_ckpt_dir, keep=3)` deletes `epoch_*.pt` files older than the 3 most recent. Never deletes `best_finetune.pt`.
2. **Memory flush** — `gc.collect()` + `torch.cuda.empty_cache()` after every epoch.
3. **Timeout guard** — raises `TimeoutError` when less than `TIMEOUT_SECONDS=180` of session time remains, allowing the fold loop to catch it and push a mid-fold checkpoint to GitHub before the session dies.

The recording-level AUC (every 5 epochs) is handled inside `finetune_train()` itself via `val_every_n_epochs=5`. The callback needs no AUC logic.

```python
def _prune_old_checkpoints(ckpt_dir, keep=3):
    import glob, os
    pts = sorted(glob.glob(f"{ckpt_dir}/epoch_*.pt"))
    for old in pts[:-keep]:     # keep the last `keep` only
        os.remove(old)

def _session_time_remaining():
    """Returns estimated remaining Colab session seconds."""
    # Simple heuristic: Colab T4 sessions last ~4h = 14400s from kernel start
    import time
    elapsed = time.time() - _SESSION_START
    return max(0.0, 14400 - elapsed)

_SESSION_START = __import__('time').time()  # record kernel start time in Cell SEED
```

### 7.g AT Sweep

After `finetune_train()` returns:

1. Load the best model checkpoint (`best_finetune.pt`) from fold's checkpoint dir.
2. Construct `PatchTST` + `ClassificationHead` and load weights using `load_pretrained_checkpoint()`.
3. Call `at_sweep(model, fold_val_csv, processed_root, train_csv=fold_train_csv, device=DEVICE, inference_stride=24, n_features=N_FEATURES, lr_C=0.1, use_pca=True)`.
4. This returns `(best_at, best_val_auc, results_dict)`.
5. Save AT sweep results to `results/e2e_cv_v2/fold{k}_at_sweep.csv`.

### 7.h Feature Extraction, LR, Clinical Threshold

```
X_tr, y_tr, _      = extract_features_for_split(model, fold_train_csv, processed_root, best_at, 24, DEVICE, N_FEATURES)
X_vl, y_vl, _      = extract_features_for_split(model, fold_val_csv,   processed_root, best_at, 24, DEVICE, N_FEATURES)
X_te, y_te, te_ids = extract_features_for_split(model, fold_test_csv,  processed_root, best_at, 24, DEVICE, N_FEATURES)
X_train_full = concat(X_tr, X_vl)
y_train_full = concat(y_tr, y_vl)
sc, pca, lr_m = fit_lr_model(X_train_full, y_train_full, C=0.1, use_pca=True)
test_scores   = predict_lr(X_te, sc, pca, lr_m)
val_scores    = predict_lr(X_vl, sc, pca, lr_m)
test_auc      = roc_auc_score(y_te, test_scores)
threshold_primary, sensitivity, specificity = clinical_threshold(y_vl, val_scores, spec_constraint=SPEC_CONSTRAINT)
```

Note: `clinical_threshold` is applied on `val_scores` (not test scores) to avoid test-set leakage. The threshold is then applied to `test_scores` for binary predictions.

### 7.i Save OOF Scores CSV

Save `results/e2e_cv_v2/fold{k}_oof_scores.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Recording ID |
| `y_true` | int | Ground truth (0=normal, 1=acidemia) |
| `y_score` | float | LR probability score for acidemia |
| `y_pred` | int | Binary prediction at `threshold_primary` |
| `best_at` | float | AT threshold selected in 7.g |
| `threshold_primary` | float | Clinical threshold from 7.h |
| `fold` | int | Fold index k |

### 7.j Post-Fold Mini-Commit

After saving the OOF CSV, commit it to GitHub:

```python
sh(f'git -C {REPO_DIR} add results/e2e_cv_v2/fold{k}_oof_scores.csv')
sh(f'git -C {REPO_DIR} commit -m "[auto] fold {k} OOF scores — best_epoch={best_epoch}"')
sh(f'git -C {REPO_DIR} push origin master', check=False)
```

Use the `GITHUB_TOKEN` remote URL (same pattern as Cell EXPORT). If push fails, print a warning but continue — the file is still on local disk.

### 7.k Per-Fold Summary Print

```
═══════════════════════════════════════════════════
 FOLD {k} COMPLETE
 Test AUC:      {test_auc:.4f}
 Sensitivity:   {sensitivity:.4f}   (val threshold, spec ≥ 0.65)
 Specificity:   {specificity:.4f}
 Best AT:       {best_at:.2f}
 Threshold:     {threshold_primary:.4f}
 Best Epoch:    {best_epoch}
═══════════════════════════════════════════════════
```

---

## 8. Per-Epoch Checkpoint Management

### 8.1 File Contents

> **Note:** The contents of `epoch_{n:03d}.pt` are determined entirely by `src/train/finetune.py` — the notebook does NOT create or write these files directly. Before writing the fold loop, read `finetune.py` to confirm what keys the checkpoint dict contains (typically `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_ema_auc`). The notebook only needs to:
> 1. **Read** the epoch number from the filename (not from inside the file) to detect `resume_from_epoch`.
> 2. **Count** existing `epoch_*.pt` files for pruning.
> 3. **Load** `best_finetune.pt` (not an epoch checkpoint) after training completes, for feature extraction.

For reference, `epoch_{n:03d}.pt` is expected to contain:
- `model_state_dict`: full model state including ClassificationHead
- `optimizer_state_dict`
- `epoch`: current epoch number (integer)
- `best_ema_auc`: current best smoothed AUC (float)
- `ema_auc`: current EMA AUC value (float)

Verify the actual keys by running: `torch.load(ckpt_path, map_location='cpu').keys()` before writing resume logic.

### 8.2 Pruning Logic

After saving `epoch_{n:03d}.pt`:
1. List all `epoch_*.pt` in the directory, parse epoch numbers.
2. Sort descending. Keep the top 3.
3. Delete all others.
4. Never delete `best_finetune.pt`.

Example: saving `epoch_010.pt` when `epoch_006.pt`, `epoch_007.pt`, `epoch_008.pt`, `epoch_009.pt` exist → keep `epoch_008.pt`, `epoch_009.pt`, `epoch_010.pt` → delete `epoch_006.pt`, `epoch_007.pt`.

### 8.3 Disk Budget Estimate

| Item | Size estimate |
|------|--------------|
| Pre-trained checkpoint | ~60 MB |
| 3 live epoch checkpoints per running fold | ~180 MB |
| 5 best_finetune.pt (one per fold) | ~300 MB |
| Processed .npy data | ~2–4 GB |
| Logs + CSVs | ~50 MB |
| **Total** | **~3–5 GB** (well under Colab's ~40 GB) |

---

## 9. Timeout Guard Implementation Pattern

> **⚠️ Scope note:** With the Section 3.5 modifications, `compute_recording_auc()` is called **inside** `finetune_train()` — the notebook never calls it directly during the fold loop. The session-level timeout ("is there < 180 s of session time left?") is handled by `_session_time_remaining()` in the `per_epoch_callback` (Section 7.f).
>
> The `threading.Timer` pattern below applies ONLY to cells that call `compute_recording_auc()` directly (e.g., Cell REPRO_TRACK, Cell BOOTSTRAP_CI post-processing). Do **not** apply it inside the FOLD_LOOP cell — the callback handles all timeout logic there.

### 9.1 Why `threading.Timer`, Not `signal.alarm`

On Google Colab (Linux), `signal.alarm` works but only in the main thread. Jupyter notebooks run cell code in the main thread, so `signal.alarm` is technically usable — but it is dangerous because an alarm in the middle of a PyTorch DataLoader operation can leave CUDA in an inconsistent state. `threading.Timer` with cooperative cancellation is safer.

### 9.2 Implementation Pattern

```python
import threading, time

def run_with_timeout(func, args, timeout_seconds=180):
    """
    Runs func(*args) in a thread. Returns the result if it completes
    within timeout_seconds, or None if it times out.
    """
    result_container = [None]
    stop_event       = threading.Event()
    exception_container = [None]

    def _target():
        try:
            result_container[0] = func(*args, stop_event=stop_event)
        except Exception as e:
            exception_container[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        stop_event.set()   # signal cooperative cancellation
        return None        # timed out
    if exception_container[0]:
        raise exception_container[0]
    return result_container[0]
```

The function passed as `func` must accept a `stop_event` keyword argument and call `stop_event.is_set()` periodically (e.g., between recordings). If `stop_event` is set, it must return `None` immediately.

> **Note:** `compute_recording_auc` from `src/train/utils.py` likely does NOT currently accept a `stop_event` argument. The notebook writer must either (a) create a wrapper that checks the stop event between recordings, or (b) run the validation in a simple thread with join timeout and accept that the thread may continue running in the background after timeout (it will be garbage-collected when the next cell runs and reassigns the model variable).

### 9.3 Handling Timeout in Epoch Loop

```python
rec_auc = run_with_timeout(compute_recording_auc, [model, val_csv, processed_root, ...], timeout_seconds=TIMEOUT_SECONDS)

if rec_auc is None:
    print(f"  └─ Validation TIMED OUT (>{TIMEOUT_SECONDS}s) — skipping best-ckpt update.")
    consecutive_timeouts += 1
    if consecutive_timeouts >= 3:
        print(f"  CRITICAL: 3 consecutive timeouts. Check num_workers setting.")
else:
    consecutive_timeouts = 0
    ema_auc = EMA_BETA * ema_auc + (1 - EMA_BETA) * rec_auc
    print(f"  └─ RecAUC: {rec_auc:.4f} | EMA: {ema_auc:.4f}")
    if ema_auc > best_ema_auc:
        best_ema_auc = ema_auc
        best_epoch   = current_epoch
        epochs_no_improve = 0
        # save best_finetune.pt
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {current_epoch}")
            break
```

---

## 10. Resume Protocol

### 10.1 What Survives What

| Event | What survives | What is lost |
|-------|--------------|--------------|
| Colab reconnect (same VM) | All of `/content/` — epoch checkpoints, OOF CSVs | Nothing |
| Runtime recycle (new VM) | Only what's committed to GitHub (OOF CSVs via mini-commits, pre-train ckpt) | In-progress fold epoch checkpoints |
| Kernel restart (same session) | Files on disk | Python variables (cv_splits, model, etc.) |

### 10.2 Full Top-to-Bottom Resume Flow

On re-run after any interruption:

1. **GPU_DIAG, SEED:** Re-initialise constants. No state dependency.
2. **REPO:** `git reset --hard origin/master` restores source files. Does NOT touch `checkpoints/` or `results/` (gitignored during training).
3. **MODULE_RELOAD:** Reloads source modules.
4. **DATA_DOWN:** Idempotent — skips zip download if `.npy` files already present.
5. **SANITY:** Re-checks counts. Should pass if data is intact.
6. **PRETRAIN:** Skips if `best_pretrain.pt` found. It is committed to GitHub, so always present after clone.
7. **G1_GATE:** Reads existing log CSV. Passes silently.
8. **CV_SPLITS:** Deterministic from `SEED=42` — same splits every run.
9. **FOLD_LOOP:**
   - Folds with complete OOF CSVs: **skipped entirely** (committed to GitHub via mini-commit).
   - Current incomplete fold: scans for latest `epoch_*.pt` on local disk; resumes from that epoch (if VM was reconnected) or restarts from epoch 0 (if VM was recycled).

### 10.3 Warning for Runtime Recycle

Add this prominent comment in the FOLD_LOOP cell:

```
# ⚠️  RUNTIME RECYCLE WARNING
# If the Colab VM was recycled (new VM assigned), the /content/ filesystem is WIPED.
# epoch_*.pt files on local disk are GONE. Training for in-progress folds restarts
# from epoch 0. Completed fold OOF CSVs are safe (committed to GitHub).
# To minimize lost work: keep Colab tab open and avoid "Disconnect and delete runtime".
```

---

## 11. Expected Metrics and Intervention Guide

### 11.1 Epoch Timing Benchmarks

| Scenario | Expected Time | Action Required |
|----------|--------------|-----------------|
| Standard training epoch (no val, stride=120) | 50–70 s | None |
| Validation epoch (every 5th, recording-level AUC) | 90–110 s | None |
| SWA epochs (50–100) add overhead | +5–10 s | Normal |
| Any epoch > 180 s with no new output | **Deadlock suspect** | Interrupt kernel; restart with `num_workers=0` confirmed |
| Epoch < 40 s | Data issue | Verify DataLoader returns non-empty batches |

At stride=120, a single fold (150 max epochs × ~60 s/epoch) takes ≤ 2.5 hours. With early stopping at patience=15, expect 60–90 epochs → **1–1.5 hours per fold** → **5–7.5 hours total** for all 5 folds.

### 11.2 Loss Trajectory

| Epoch Range | Expected Train Loss | Warning |
|------------|--------------------|--------------------|
| 1–10 | 0.65–0.75 | Normal — model initializing |
| 10–30 | 0.50–0.65 (declining) | If stuck > 0.70 at epoch 20: check class_weight=3.9 applied |
| 30–60 | 0.35–0.50 (learning) | If > 0.60 at epoch 40: check LR scheduler |
| 60–100 | 0.25–0.40 (converging) | Loss < 0.15 possible overfitting |
| 100+ | 0.20–0.35 (plateau) | EMA should stabilize |

### 11.3 Recording-Level Validation AUC (Every 5 Epochs)

| Validation Epoch | Expected RecAUC | Warning |
|-----------------|-----------------|---------|
| 1–4 (training epochs 5–20) | 0.55–0.65 | Normal early phase |
| 5–10 (training epochs 25–50) | 0.65–0.75 | Should be improving |
| 11–20 (training epochs 55–100) | 0.70–0.82 | Target range |
| 20+ (epochs 100+) | 0.75–0.85 peak | EMA may plateau → early stop |

**Critical warning:** If RecAUC is ≤ 0.52 (near-random) after 5 validation epochs (= training epoch 25), training is not converging. Likely causes:
- Class weight not applied (positive weight must be 3.9 for `loss='cross_entropy'`)
- Pretrain checkpoint not loading correctly (verify `load_pretrained_checkpoint` call)
- `train_stride=120` produced empty or single-class batches for some recordings

**Normal oscillation:** Window-level AUC (reported internally) oscillates 0.55–0.74 between epochs. This is expected — do not use it for early stopping.

### 11.4 Gate Checkpoints During Run

After **Fold 0 completes:**
- G3 gate: if `fold0_test_auc < 0.65`, print a critical warning and pause for human confirmation via `input("Press Enter to continue to Fold 1, or Ctrl+C to abort: ")`. This prevents wasting 4+ hours on a broken model.

After **all 5 folds complete:**
- G4a: `mean(fold AUCs) >= 0.70`
- G4b: `std(fold AUCs) < 0.10`

---

## 12. GitHub Export (Cell EXPORT)

### 12.1 Prerequisites Check

Before running any git operations, Cell EXPORT must verify:

```python
from pathlib import Path
ROOT_P = Path(REPO_DIR)

missing = []
for k in range(N_FOLDS):
    p = ROOT_P / f"results/e2e_cv_v2/fold{k}_oof_scores.csv"
    if not p.exists(): missing.append(str(p.name))
if not (ROOT_P / "results/e2e_cv_v2/final_cv_report_v3.csv").exists():
    missing.append("final_cv_report_v3.csv")

assert not missing, f"Missing files before export: {missing}"

try:
    from google.colab import userdata as _ud
    GITHUB_TOKEN = _ud.get('GITHUB_TOKEN')
    assert GITHUB_TOKEN, "GITHUB_TOKEN is empty"
except Exception as e:
    raise RuntimeError(f"GITHUB_TOKEN not found in Colab secrets: {e}")
```

### 12.2 Files to Commit

**Results CSVs:**
- `results/e2e_cv_v2/fold{0..4}_oof_scores.csv`
- `results/e2e_cv_v2/fold{0..4}_at_sweep.csv`
- `results/e2e_cv_v2/per_fold_summary.csv`
- `results/e2e_cv_v2/final_cv_report_v3.csv`
- `results/e2e_cv_v2/comparison_table_v3.csv`
- `results/e2e_cv_v2/at_sweep_summary.csv`

**Logs:**
- `logs/e2e_cv_v2/fold{0..4}_finetune_loss.csv`

**Best checkpoints (zipped):**
- Zip all 5 `checkpoints/e2e_cv_v2/fold{k}/finetune/best_finetune.pt` into `checkpoints/e2e_cv_v2/best_models_v3.zip`:

```python
import zipfile
with zipfile.ZipFile(f"{REPO_DIR}/checkpoints/e2e_cv_v2/best_models_v3.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    for k in range(N_FOLDS):
        src = ROOT_P / f"checkpoints/e2e_cv_v2/fold{k}/finetune/best_finetune.pt"
        if src.exists():
            zf.write(src, f"fold{k}_best_finetune.pt")
```

### 12.3 Git Push Sequence

```python
import datetime
sh(f'git config --global user.email "colab-auto@sentinelfatal2"')
sh(f'git config --global user.name "Colab TrainingAgent"')
# Set token in remote URL
sh(f'git -C {REPO_DIR} remote set-url origin "https://{GITHUB_TOKEN}@github.com/ArielShamay/SentinelFatal2.git"')
# Stage files (force-add .pt/.zip which may be gitignored)
sh(f'git -C {REPO_DIR} add results/e2e_cv_v2/ logs/e2e_cv_v2/')
sh(f'git -C {REPO_DIR} add -f checkpoints/e2e_cv_v2/best_models_v3.zip')
# Commit
date_str = datetime.date.today().isoformat()
sh(f'git -C {REPO_DIR} commit -m "[auto] E2E CV v3 final results — {date_str}"')
# Push
rc = sh(f'git -C {REPO_DIR} push origin master', check=False)
# ALWAYS reset remote to remove token from URL
sh(f'git -C {REPO_DIR} remote set-url origin "https://github.com/ArielShamay/SentinelFatal2.git"')
```

If `git push` fails: print the error and then:

```python
print("\nFALLBACK: use Colab file browser to manually download:")
print(f"  File → Download → {REPO_DIR}/results/e2e_cv_v2/")
print(f"  File → Download → {REPO_DIR}/checkpoints/e2e_cv_v2/best_models_v3.zip")
```

### 12.4 Token Security

- Never hardcode the token in the notebook.
- Always retrieve from `google.colab.userdata.get('GITHUB_TOKEN')`.
- Always reset the remote URL back to HTTPS-without-token after push, even on failure.

### 12.5 Comparison Table

Cell BOOTSTRAP_CI must also build and save `results/e2e_cv_v2/comparison_table_v3.csv`:

| method | n | auc | ci_lo | ci_hi | notes |
|--------|---|-----|-------|-------|-------|
| Paper (benchmark) | 55 | 0.826 | — | — | from original paper |
| Baseline Stage2 LR (test-55) | 55 | 0.812 | 0.630 | 0.953 | from results/final_report.md |
| Best post-hoc (AT=0.40, Youden) | 55 | 0.839 | — | — | from results/final_report.md |
| RepRo Track v3 (441/56/55) | 55 | {repro_auc} | {r_lo} | {r_hi} | from REPRO_TRACK cell |
| E2E CV v3 (552, 5-fold) | 552 | {global_auc} | {ci_lo} | {ci_hi} | OOF across all 5 folds |

---

## 13. Gates and Stop Conditions Summary

| Gate ID | When Evaluated | Condition | PASS | FAIL |
|---------|---------------|-----------|------|------|
| **G1** | Cell G1_GATE | `val_mse < 0.015` | Continue | Prominent warning (human decides) |
| **G2** | Cell SANITY | All data counts & file existence checks | Continue | Hard stop with descriptive error |
| **G3** | After Fold 0 | `fold0_test_auc ≥ 0.65` | Continue to Fold 1 | `input()` pause for human confirmation |
| **G4a** | Cell BOOTSTRAP_CI | `mean(fold AUCs) ≥ 0.70` | Note in report | Flag "below target" |
| **G4b** | Cell BOOTSTRAP_CI | `std(fold AUCs) < 0.10` | Note in report | Flag "high variance" |

**G1 fail:** Backbone not converged. Options: increase `pretrain.max_epochs` in `train_config.yaml` and re-run Cell PRETRAIN; or check `logs/e2e_cv_v2/shared_pretrain_loss.csv` for anomalies.

**G2 fail:** Data missing or corrupt. Fix the data counts and re-run Cell SANITY before any training.

**G3 fail:** Model not learning. Diagnose in order: (1) verify `class_weight_positive=3.9` is applied; (2) verify `load_pretrained_checkpoint()` loaded weights correctly (print a parameter checksum); (3) try stride=60 for Fold 0 as a diagnostic (more data per epoch, slower but higher AUC ceiling).

**G4a/G4b fail:** High variance likely from imbalanced fold splits. Re-examine `generate_cv_splits()` output for class distribution per fold.

---

## 14. File and Directory Layout

```
SentinelFatal2/                                   [repo root]
├── config/
│   └── train_config.yaml                         [repo] read-only
├── data/
│   ├── processed/
│   │   ├── ctu_uhb/       *.npy                  [repo] 552 recordings
│   │   └── fhrma/         *.npy                  [repo] 135 recordings
│   └── splits/
│       ├── pretrain.csv                          [repo] 687 rows
│       ├── train.csv                             [repo] 441 rows
│       ├── val.csv                               [repo] 56 rows
│       ├── test.csv                              [repo] 55 rows
│       └── train_val_test.csv                    [new]  552 rows — Cell DATA_DOWN
├── checkpoints/
│   ├── pretrain/
│   │   └── best_pretrain.pt                      [repo fallback] — if shared_pretrain not found
│   └── e2e_cv_v2/
│       ├── shared_pretrain/
│       │   └── best_pretrain.pt                  [repo or new] — Cell PRETRAIN
│       ├── fold0/finetune/
│       │   ├── epoch_NNN.pt                      [new] rolling (last 3 kept)
│       │   └── best_finetune.pt                  [new] best EMA-AUC checkpoint
│       ├── fold1/finetune/                       [new] same structure
│       ├── fold2/finetune/                       [new]
│       ├── fold3/finetune/                       [new]
│       ├── fold4/finetune/                       [new]
│       └── best_models_v3.zip                    [new] Cell EXPORT — 5 best_finetune.pt zipped
├── logs/
│   ├── pretrain_loss.csv                         [repo fallback]
│   └── e2e_cv_v2/
│       ├── shared_pretrain_loss.csv              [repo or new] — Cell PRETRAIN
│       ├── fold0_finetune_loss.csv               [new]
│       ├── fold1_finetune_loss.csv               [new]
│       ├── fold2_finetune_loss.csv               [new]
│       ├── fold3_finetune_loss.csv               [new]
│       └── fold4_finetune_loss.csv               [new]
├── results/
│   └── e2e_cv_v2/
│       ├── fold0_splits/
│       │   ├── fold0_train.csv                   [new]
│       │   ├── fold0_val.csv                     [new]
│       │   └── fold0_test.csv                    [new]
│       ├── fold1_splits/ … fold4_splits/         [new] same structure
│       ├── fold0_oof_scores.csv                  [new] id,y_true,y_score,y_pred,best_at,threshold_primary,fold
│       ├── fold1_oof_scores.csv                  [new]
│       ├── fold2_oof_scores.csv                  [new]
│       ├── fold3_oof_scores.csv                  [new]
│       ├── fold4_oof_scores.csv                  [new]
│       ├── fold0_at_sweep.csv                    [new] AT × val_auc table
│       ├── fold1_at_sweep.csv … fold4_at_sweep.csv [new]
│       ├── at_sweep_summary.csv                  [new] Cell AT_SUMMARY
│       ├── per_fold_summary.csv                  [new] Cell BOOTSTRAP_CI
│       ├── global_oof_predictions.csv            [new] all 552 OOF scores concatenated
│       ├── final_cv_report_v3.csv                [new] global AUC, CI, clinical threshold
│       ├── comparison_table_v3.csv               [new] Cell BOOTSTRAP_CI
│       └── repro_comparison_v3.csv               [new, optional] Cell REPRO_TRACK
└── notebooks/
    ├── 08_e2e_cv_v2.ipynb                        [repo] old notebook — do not modify
    └── 09_e2e_cv_v3.ipynb                        [new]  this notebook
```

---

## 15. Differences from Old Notebook (`08_e2e_cv_v2.ipynb`)

| # | Change | Rationale |
|---|--------|-----------|
| 1 | **Removed Config A/B/C selection loop** | Primary time-sink; saves 8–10 hours; Config A performs well |
| 2 | **Removed Config B and C code** | No dead code; simpler notebook |
| 3 | **`num_workers=0` (monkey-patch)** | **Primary deadlock fix** — eliminates SWA+DataLoader sync issue |
| 4 | **`val_stride=120`, `train_stride=120`** | Halves windows per epoch (~117K→~60K); ~60s/epoch vs ~115s/epoch |
| 5 | **Recording-level AUC every 5 epochs only** | Reduces DataLoader/SWA sync frequency by 5× during most dangerous phase |
| 6 | **180-second timeout guard on validation** | Prevents infinite hang even if DataLoader does produce a deadlock |
| 7 | **Per-batch logging eliminated (`quiet=True`)** | Removes millions of output lines; reduces RAM/buffer bloat |
| 8 | **Per-epoch memory flush** | `gc.collect()` + `cuda.empty_cache()` prevents gradual VRAM fragmentation |
| 9 | **`patience=15` (was 25)** | Faster early stopping; expected 60–90 epochs/fold vs up to 150 |
| 10 | **Epoch checkpoint every epoch** | Enables session-reconnect resume from last epoch (not epoch 0) |
| 11 | **Checkpoint pruning (keep last 3 + best)** | Prevents disk quota exhaustion |
| 12 | **Fold-level skip on existing OOF CSV** | Re-runs skip completed folds |
| 13 | **Post-fold mini-commit (OOF CSV only)** | Completed fold results survive runtime recycle |
| 14 | **G3 gate at Fold 0 with `input()` pause** | Prevents wasting 4+ hours on non-converging model |
| 15 | **Shared pretrain checkpoint path fixed** | Now checks `checkpoints/e2e_cv_v2/shared_pretrain/` first |
| 16 | **Module reload cell** | Prevents stale cached modules in long sessions |
| 17 | **GPU_PREFLIGHT smoke test** | Verifies CUDA working before 6-hour run |
| 18 | **Post-fold summary print block** | Easy visual tracking of per-fold progress |

---

## 16. Checklist for Handoff

### Before Pressing "Run All"

**Environment:**
- [ ] Runtime type = GPU (T4 or A100 preferred). Runtime → Change runtime type → GPU.
- [ ] `GITHUB_TOKEN` in Colab secrets (🔑 icon in sidebar). Token must have `repo` write scope.
- [ ] No RAM warning visible. If present: RAM indicator → Connect to a new runtime.

**Repository state:**
- [ ] `https://github.com/ArielShamay/SentinelFatal2` accessible from Colab.
- [ ] `checkpoints/e2e_cv_v2/shared_pretrain/best_pretrain.pt` committed to master (verify in GitHub UI). If absent, Cell PRETRAIN runs full pre-training (~1–2 hours extra).
- [ ] All `data/splits/*.csv` files committed to master.
- [ ] Processed `.npy` data accessible (either committed or available via `data_processed.zip`).

**Pre-run validation:**
- [ ] Run Cell SANITY standalone **before** "Run All". Fix any count mismatches before starting.
- [ ] Verify `train_val_test.csv` has 552 rows and 113+ positives.

### Expected Performance Summary

| Metric | Expected Value |
|--------|---------------|
| Total wall-clock time | ~6–8 hours (T4 GPU) |
| Time per fold (finetune) | ~60–90 minutes |
| Per-epoch time (standard) | 50–70 s |
| Per-epoch time (val epoch) | 90–110 s |
| Peak GPU VRAM | 6–10 GB |
| Peak RAM | 8–14 GB |
| Disk usage | ~3–5 GB |
| Fold 0 AUC (expected) | 0.68–0.82 |
| Global OOF AUC (target) | ≥ 0.70 |

### Recovery Plan (Session Dies Mid-Fold)

1. Do NOT close/disconnect the Colab tab if the VM is still alive (reconnect instead).
2. If VM is still available: open the notebook, press "Run All" — epoch checkpoints on local disk will resume training from the last saved epoch.
3. If VM is recycled (new VM, filesystem wiped): open notebook, press "Run All" — completed fold OOF CSVs are on GitHub (via mini-commits), incomplete fold restarts from epoch 0.
4. Never manually edit `fold{k}_oof_scores.csv` — the fold-level skip logic depends on the file existing and containing valid data.

---

*End of specification. This document is the authoritative SSOT for `notebooks/09_e2e_cv_v3.ipynb`.  
Any deviation from the rules in Section 3 or the fold loop logic in Section 7 requires updating this document before implementation.*
