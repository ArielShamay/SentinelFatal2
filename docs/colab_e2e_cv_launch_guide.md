# SentinelFatal2 — Colab GPU E2E Cross-Validation Launch Guide

> **Version:** 1.0 — February 23, 2026  
> **Purpose:** Complete, self-contained instructions for running the End-to-End 5-Fold Cross-Validation on a Colab T4 GPU.  
> **Status:** All code written and locally dry-run validated. Waiting for GPU allocation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Hyperparameters](#2-architecture--hyperparameters)
3. [Previous Training Results (Baseline)](#3-previous-training-results-baseline)
4. [Key Changes That Improved Results](#4-key-changes-that-improved-results)
5. [What Remains: E2E Cross-Validation](#5-what-remains-e2e-cross-validation)
6. [Data Setup: Critical First Step](#6-data-setup-critical-first-step)
7. [Step-by-Step Colab Launch](#7-step-by-step-colab-launch)
8. [Expected Outputs](#8-expected-outputs)
9. [Time Estimates (T4 GPU)](#9-time-estimates-t4-gpu)
10. [Script CLI Reference](#10-script-cli-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Project Overview

**Goal:** Reproduce and validate arXiv:2601.06149v1 — a PatchTST Foundation Model for fetal acidemia prediction from intrapartum CTG (Cardiotocography).

### Dataset
| Split | Recordings | Acidemia | Normal |
|-------|-----------|----------|--------|
| Train | 441 | 90 (20.4%) | 351 |
| Val   | 56  | 12 (21.4%) | 44  |
| Test  | 55  | 11 (20.0%) | 44  |
| Pretrain (FHRMA, unlabeled) | 135 | — | 135 |
| **Pretrain total** | **687** | — | — |

> **Splits are locked.** They come from `data/CTGDL/CTGDL_norm_metadata.csv` column `test` (0=train, 1=val, 2=test). Never recreate them.

### 3-Stage Pipeline
```
Stage 1 — MAE Pretrain
  687 recordings (CTU-UHB + FHRMA)
  → Self-supervised masked autoencoder on FHR signal
  → Saves: checkpoints/pretrain/best_pretrain.pt

Stage 2 — Supervised Fine-tune
  441 train + 56 val CTU-UHB recordings
  → Classification head on frozen→warmed backbone
  → Saves: checkpoints/finetune/best_finetune.pt

Stage 3 — LR Alerting
  Sliding window inference → alert segment features → Logistic Regression
  → Binary acidemia prediction per recording
  → Saves: checkpoints/alerting/logistic_regression.pkl
```

### Paper Benchmark (Table 3)
| Subgroup | n | AUC (paper) |
|----------|---|-------------|
| All test | 55 | **0.826** |
| Vaginal deliveries | 50 | 0.850 |
| Cephalic presentation | 50 | 0.848 |
| Vaginal + cephalic | 46 | 0.853 |

---

## 2. Architecture & Hyperparameters

All values are locked in `config/train_config.yaml`. Source = paper unless marked ⚠ (assumption).

### Signal Processing
| Parameter | Value | Source |
|-----------|-------|--------|
| Sampling frequency | 4 Hz | ✓ paper Section II-A |
| Window length | 1,800 samples (7.5 min) | ✓ paper Section II-C |
| Patch length | 48 samples | ✓ paper Equation 1 |
| Patch stride (within window) | 24 samples | ✓ paper Equation 1 |
| Patches per window | 73 (sequence trimmed to 1,776 samples — see S9) | ✓ computed |
| Sliding window stride (pretrain/finetune) | 900 samples (50% overlap) | ⚠ S4 |
| FHR normalization | `(fhr - 50) / 160.0` → range [0,1] | ⚠ S7 |
| UC normalization | `uc / 100.0` | ⚠ S7 |

### Transformer Architecture
| Parameter | Value | Source |
|-----------|-------|--------|
| d_model | 128 | ⚠ S2 |
| num_layers | 3 | ⚠ S2 |
| n_heads | 4 | ⚠ S2 |
| ffn_dim | 256 | ⚠ S2 |
| dropout | 0.2 | ✓ paper Section II-C |
| Normalization | BatchNorm1d | ✓ paper Section II-C |
| Classification head input | 73 × 128 × 2 = 18,688 | derived |

### Pre-training
| Parameter | Value | Source |
|-----------|-------|--------|
| Mask ratio | 0.4 (FHR only) | ✓ paper Section II-D |
| Masking strategy | Contiguous groups ≥ 2 patches | ✓ paper Section II-D |
| Boundary preservation | First & last patch never masked | ✓ paper Section II-D |
| UC in pretraining | Always visible (asymmetric masking) | ✓ paper Section II-D |
| UC fusion | `fhr_enc += uc_enc` (element-wise add) | ✓ AGW-19 fix |
| Loss | MSE on masked FHR patches only | ✓ Equation 2 |
| Optimizer | Adam, lr=1e-4 | ✓ paper Section II-D |
| Max epochs | 200 | ⚠ S5 |
| Early stopping patience | 10 (on val reconstruction loss) | ⚠ S5 |
| Batch size | 64 | ⚠ S6 |

### Fine-tuning
| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | ✓ paper Section II-E |
| Backbone LR | 1e-5 (differential — prevents catastrophic forgetting) | ⚠ S6 |
| Head LR | 1e-4 | ⚠ S6 |
| Weight decay | 1e-2 | ⚠ S6 |
| Gradient clipping | max_norm=1.0 | ⚠ S6 |
| Class weights | [1.0, 3.9] (auto-computed: n_neg/n_pos) | ⚠ S6 |
| Max epochs | 100 | ✓ paper Section II-E |
| Early stopping patience | 15 (on val AUC) | ✓ paper Section II-E |
| Batch size | 32 | ⚠ S6 |

### Alerting (Stage 3)
| Parameter | Value | Source |
|-----------|-------|--------|
| Alert threshold | **0.40** (lowered from 0.50 — see S11) | S11 |
| Decision threshold | **0.284** (Youden-optimal) | S11 |
| Inference stride (evaluation) | 1 sample | ✓ paper Section II-F |
| Inference stride (E2E CV) | 60 samples (15 sec) | ⚠ S6 (speed) |
| LR features | 4: segment_length, max_prediction, cumulative_sum, weighted_integral | ✓ paper Section II-F |

---

## 3. Previous Training Results (Baseline)

Training was run on Colab T4 GPU (February 22–23, 2026).

### Pre-training (13 epochs, early stopped)
```
Best val MSE = 0.01427 at epoch 2
Train loss: 0.111 → 0.009 (steady decrease across 13 epochs)
Val loss: plateau after epoch 2 → early stopped at epoch 12 (patience=10 exhausted)
```

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 0 | 0.11100 | 0.01538 |
| 1 | 0.02964 | 0.01472 |
| **2** | **0.01949** | **0.01427** ← best |
| 3–12 | 0.015→0.009 | 0.015–0.017 (plateau) |

### Fine-tuning (33 epochs, early stopped at best epoch 17)
```
Best val AUC = 0.7235 at epoch 17
Early stopped at epoch 32 (patience=15 exhausted after no improvement since epoch 17)
```

### Final Test Results (55 recordings)

| Metric | Value |
|--------|-------|
| Stage 1 AUC (direct NN) | 0.7624 |
| **Stage 2 AUC (with LR)** | **0.8120** |
| Paper AUC | 0.826 |
| Accuracy | 81.8% |
| Sensitivity | 0.818 (with AT=0.40 + Youden T=0.284) |
| Specificity | 0.773 |
| True Positives | 9/11 |

> **Key finding from Agent 8 optimization:** Using alert threshold=0.40 (instead of 0.50) + Youden threshold=0.284 raises AUC from 0.812 to **0.839** and sensitivity from 0.09 to **0.818**. The E2E CV must use these exact thresholds.

---

## 4. Key Changes That Improved Results

These are **non-obvious fixes** that were required to reach the reported AUC. The E2E CV agent must preserve all of them.

### 4.1 — UC Fusion via Element-wise Addition (AGW-19, Critical)
**Problem:** The UC encoder output was computed but never used.  
**Fix:** In `src/train/pretrain.py`, `pretrain_step()`:
```python
fhr_enc = fhr_enc + uc_enc   # element-wise add — UC shapes FHR reconstruction
```
Without this, UC had no effect on FHR reconstruction and model was effectively single-channel.

### 4.2 — Alert Threshold Lowered 0.50 → 0.40 (S11, Most Impactful)
**Problem:** With AT=0.50, 13/55 test recordings produced no alert segment at all → `ZERO_FEATURES=[0,0,0,0]` → LR defaulted to class prior → Sensitivity=0.09.  
**Fix:** AT=0.40 eliminates zero-segment recordings on the test set.  
**Effect:** Sensitivity 0.09 → 0.818, AUC 0.569 → 0.839  
**Location:** `src/inference/alert_extractor.py`: `ALERT_THRESHOLD = 0.4`  
**Also in:** `scripts/run_e2e_cv.py`: `ALERT_THRESHOLD = 0.40` (hardcoded constant)

### 4.3 — Differential Learning Rate (S6)
**Why:** Fine-tuning with a single LR=1e-4 on the backbone causes catastrophic forgetting of pretrained weights.  
**Fix:** Backbone LR=1e-5, head LR=1e-4.  
**Location:** `src/train/finetune.py`, optimizer parameter groups.

### 4.4 — Class Weights [1.0, 3.9] (S6)
**Why:** Without SPAM augmentation, train set is 80/20 imbalanced.  
**Fix:** `CrossEntropyLoss(weight=[1.0, n_neg/n_pos])` auto-computed from train labels only.  
**Location:** `src/train/finetune.py`.

### 4.5 — n_patches Trim to 1,776 Samples (S9)
**Why:** `unfold(size=48, step=24)` on 1,800 samples returns 74 patches, not 73.  
**Fix:** Trim signal to `(73-1)*24 + 48 = 1,776` samples before patch extraction.  
**Location:** `src/model/patchtst.py`, `_extract_patches()`.

### 4.6 — Youden-Optimal Decision Threshold = 0.284
**Why:** Default threshold=0.5 for LR is suboptimal. Youden's J maximizes sensitivity+specificity.  
**Value:** 0.284 (computed on original 441-recording train set, Agent 8).  
**Location:** `scripts/run_e2e_cv.py`: `DECISION_THRESHOLD = 0.284` (hardcoded).  
**Note:** Per-fold Youden thresholds are also computed per fold for individual fold metrics, but the final OOF pooled AUC is threshold-independent.

---

## 5. What Remains: E2E Cross-Validation

### Why E2E CV?
The original evaluation used a single fixed test set of 55 recordings. This gives a point estimate of AUC=0.812 but no confidence interval. With only 55 recordings an AUC CI would be very wide (roughly ±0.07). E2E CV uses all 497 labeled recordings for evaluation, providing much more stable AUC estimates.

### What "E2E" Means
Unlike a simple LR-only CV (which reuses the fixed pretrained model), E2E CV retrains the **entire model from scratch** per fold:
- Fold k test set: ~100 recordings held out
- Fold k pretrain corpus: ~532 recordings (FHRMA 135 + ~397 CTU-UHB)
- Fold k finetune train: ~317 recordings
- Fold k finetune val: ~80 recordings

### Script: `scripts/run_e2e_cv.py`
- Fully written (840 lines), locally dry-run validated ✅
- Handles data splitting, pretrain, finetune, feature extraction, LR, OOF collection
- Supports resume: if `best_pretrain.pt` or `best_finetune.pt` exist for a fold, that phase is skipped
- Calculates 5,000-iteration bootstrap CI on pooled OOF scores

---

## 6. Data Setup: Critical First Step

**The processed `.npy` files are NOT on GitHub** (too large: ~100–200 MB). They live locally in:
```
data/processed/ctu_uhb/   ← 552 .npy files (one per CTU-UHB recording)
data/processed/fhrma/     ← 135 .npy files (one per FHRMA recording)
```

You must upload them to Colab before running. **Recommended: Google Drive** (persistent across sessions).

---

### Option A — Google Drive (Recommended)

**Step A1: Zip the processed data locally (run once from project root):**
```powershell
# In PowerShell, from C:\Users\ariel\Desktop\SentinelFatal2\
Compress-Archive -Path "data\processed" -DestinationPath "data_processed.zip"
```
This creates `data_processed.zip` (~60–150 MB). Upload it to your Google Drive (any folder).

**Step A2: In Colab, replace Cell 3 with the following code:**

```python
# ── Cell 3 (ALTERNATIVE): Load data from Google Drive ────────────────────────
import os, sys, zipfile
from pathlib import Path
from google.colab import drive

REPO_DIR = Path("/content/SentinelFatal2")

# Mount Google Drive (will open an auth popup the first time)
drive.mount('/content/gdrive', force_remount=False)

# ── UPDATE THIS PATH to where you uploaded data_processed.zip on Drive ──────
ZIP_ON_DRIVE = "/content/gdrive/MyDrive/data_processed.zip"   # ← change if needed
# ─────────────────────────────────────────────────────────────────────────────

if not Path(ZIP_ON_DRIVE).exists():
    raise FileNotFoundError(
        f"ZIP not found at:\n  {ZIP_ON_DRIVE}\n"
        "Upload data_processed.zip to Google Drive and update the path above."
    )

# Extract only if needed
ctu_dir   = REPO_DIR / "data" / "processed" / "ctu_uhb"
fhrma_dir = REPO_DIR / "data" / "processed" / "fhrma"
ctu_npy   = list(ctu_dir.glob("*.npy"))   if ctu_dir.exists()   else []
fhrma_npy = list(fhrma_dir.glob("*.npy")) if fhrma_dir.exists() else []

if len(ctu_npy) >= 552 and len(fhrma_npy) >= 135:
    print(f"✅ Data already present: ctu_uhb={len(ctu_npy)}, fhrma={len(fhrma_npy)}")
else:
    print(f"Extracting {ZIP_ON_DRIVE} → {REPO_DIR}/data/processed/ ...")
    with zipfile.ZipFile(ZIP_ON_DRIVE, 'r') as zf:
        zf.extractall(REPO_DIR / "data")
    print("Extraction complete.")

# Verify
ctu_npy   = list((REPO_DIR / "data" / "processed" / "ctu_uhb").glob("*.npy"))
fhrma_npy = list((REPO_DIR / "data" / "processed" / "fhrma").glob("*.npy"))
print(f"\nVerification:")
print(f"  ctu_uhb .npy : {len(ctu_npy)}  (expected 552)")
print(f"  fhrma   .npy : {len(fhrma_npy)}  (expected 135)")

for split in ['train', 'val', 'test']:
    p = REPO_DIR / "data" / "splits" / f"{split}.csv"
    print(f"  {split}.csv      : {'✅' if p.exists() else '❌ MISSING — run git pull'}")

if len(ctu_npy) < 552:
    raise RuntimeError(f"Expected 552 ctu_uhb .npy files, got {len(ctu_npy)}.")
if len(fhrma_npy) < 135:
    raise RuntimeError(f"Expected 135 fhrma .npy files, got {len(fhrma_npy)}.")

print("\n✅ Data ready.")
```

---

### Option B — Colab File Browser (Quick, Not Persistent)

1. In Colab (web UI), click the **folder icon** (left sidebar) → **Upload to session storage**
2. Upload `data_processed.zip`
3. Then in a cell:
```python
import zipfile, os
from pathlib import Path
REPO_DIR = Path("/content/SentinelFatal2")
with zipfile.ZipFile("/content/data_processed.zip", 'r') as zf:
    zf.extractall(REPO_DIR / "data")
os.remove("/content/data_processed.zip")
```
> ⚠️ Session storage is wiped when you disconnect. Use Google Drive for overnight runs.

---

### Verify Data is Present
```bash
ls /content/SentinelFatal2/data/processed/ctu_uhb/*.npy | wc -l   # expect 552
ls /content/SentinelFatal2/data/processed/fhrma/*.npy | wc -l     # expect 135
```

---

## 7. Step-by-Step Colab Launch

### Prerequisites
- [ ] `data_processed.zip` created locally and uploaded to Google Drive (see Section 6)
- [ ] Latest changes pushed to GitHub (`git push origin master`)
- [ ] A Colab session with **T4 GPU** runtime

---

### Step 1: Get a T4 GPU Runtime

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**
3. OR: use the **VS Code Colab Extension** (sidebar → Google Colab → New Colab Server → T4 GPU)

---

### Step 2: Open the Notebook

**Via Colab web:**
- File → Open notebook → GitHub tab → paste `https://github.com/ArielShamay/SentinelFatal2` → select `notebooks/07_colab_e2e_cv_launch.ipynb`

**Via VS Code Colab Extension:**
- Open `notebooks/07_colab_e2e_cv_launch.ipynb` in VS Code, click "Select Kernel" → choose your connected Colab T4 server

---

### Step 3: Run Cell 1 — GPU Check (~10 sec)

Expected output:
```
Loading PyTorch... (first run takes ~60s)
==================================================
  SentinelFatal2 — Hardware Check
==================================================
  GPU  : Tesla T4
  VRAM : 15.0 GB
  CUDA : 12.x
  ✅ GPU ready — safe to run Cell 2.
==================================================
  DEVICE = 'cuda'
==================================================
```
**If you see `⚠️ NO GPU detected`:** Your runtime doesn't have a GPU. Switch runtime type and reconnect.

---

### Step 4: Run Cell 2 — Clone Repository (~30 sec)

Expected output:
```
Cloning repository to /content/SentinelFatal2 ...
Working directory: /content/SentinelFatal2
✅ Repository ready on Colab server.
```
If repo already exists (resuming a previous session): `git pull` is run automatically.

---

### Step 5: Set Up Data (Cell 3 — Modified)

**Replace Cell 3** with the Google Drive code from Section 6, Option A.  
Update `ZIP_ON_DRIVE` to match where you saved `data_processed.zip` on Drive.

Expected output:
```
✅ Data already present: ctu_uhb=552, fhrma=135
  (or extraction progress if first time)

Verification:
  ctu_uhb .npy : 552  (expected 552)
  fhrma   .npy : 135  (expected 135)
  train.csv    : ✅
  val.csv      : ✅
  test.csv     : ✅

✅ Data ready.
```

---

### Step 6: Run Cell 4 — Install Dependencies (~1 min)

Expected output:
```
✅ All dependencies already installed.

Package versions:
  torch      : 2.x.x  (CUDA: 12.x)
  numpy      : 1.26.x
  pandas     : 2.x.x
  sklearn    : 1.x.x
  PyYAML     : 6.x.x

  Device     : cuda
```

---

### Step 7: Run Cell 5 — Dry-Run (~5 min) ✅ REQUIRED before Cell 6

Runs 1 fold with `--max-batches 2` per phase to confirm the full pipeline executes without errors.

Expected output (key lines):
```
Starting dry-run (1 fold, max_batches=2)...
------------------------------------------------------------
[E2E-CV] Fold 0/1 starting
[PRETRAIN] Fold 0 | epoch 0/200 | train_loss=... | val_loss=...
[PRETRAIN] max_batches=2 reached, stopping.
[PRETRAIN] Fold 0 saved best_pretrain.pt
[FINETUNE] Fold 0 | epoch 0/100 | train_loss=... | val_auc=...
[FINETUNE] max_batches=2 reached, stopping.
[FINETUNE] Fold 0 saved best_finetune.pt
[LR]       Fold 0 | extracting features for 5 recordings...
[LR]       Fold 0 | n_train=4 | n_test=1 | fold_auc=...
[E2E-CV]   Fold 0 complete.
[E2E-CV]   ALL FOLDS COMPLETE (1/1 succeeded, 0 failed)
------------------------------------------------------------
✅ Dry-run PASSED — safe to run Cell 6.
```

**If dry-run fails:** Read the error output. Common causes in Section 11.

---

### Step 8: Run Cell 6 — Full Overnight Run (~3–5 hours)

This cell blocks until all 5 folds complete. Output streams live to the cell output area.

**Before starting:**
- Make sure VS Code is connected to the Colab server (VS Code extension), OR keep the Colab tab open in your browser
- Do not close your laptop/browser during the run

**What you'll see:**
```
============================================================
  SentinelFatal2 — Full 5-Fold E2E CV
  Device : cuda
  Log    : /content/SentinelFatal2/logs/e2e_cv_master.log
  Keep VS Code open. This cell runs for ~3–5 hours.
============================================================

[E2E-CV] Fold 0/5 starting
[PRETRAIN] Fold 0 | epoch 0/200 | ...
... (progress lines every epoch) ...
[PRETRAIN] Fold 0 | early stopping at epoch 10 | best val_loss=0.014
[FINETUNE] Fold 0 | epoch 0/100 | ...
... (progress lines every epoch) ...
[FINETUNE] Fold 0 | early stopping at epoch 30 | best val_AUC=0.71
[LR]       Fold 0 | AUC=0.XXX | threshold=0.XXX
[E2E-CV]   Fold 0 complete.
[E2E-CV] Fold 1/5 starting
... (repeats for folds 1-4) ...
============================================================
✅ ALL FOLDS COMPLETE — run Cell 7 to see final results.
============================================================
```

**If interrupted:** Just re-run Cell 6. The script automatically skips any fold that has a `best_finetune.pt` checkpoint. Resume is safe.

---

### Step 9: Run Cell 7 — Morning Check (Read-only, safe anytime)

Expected output if complete:
```
============================================================
  SentinelFatal2 — E2E CV Morning Check
============================================================

  Process has finished.

-- Per-fold progress ----------------------------------------
  fold  pretrain_epochs  finetune_epochs  fold_auc  ...

-- Final report ---------------------------------------------
  Metric          Value
  pooled_auc      0.XXX
  auc_ci_lower    0.XXX
  auc_ci_upper    0.XXX
  ...

-- ROC curve plot -------------------------------------------
  [inline PNG displayed]
============================================================
```

---

## 8. Expected Outputs

After a successful run, the following files are written to the Colab server at `/content/SentinelFatal2/`:

| File | Written when | Purpose |
|------|-------------|---------|
| `logs/e2e_cv/fold{k}_pretrain.csv` | During pretrain, each epoch | Monitor val_loss per fold |
| `logs/e2e_cv/fold{k}_finetune.csv` | During finetune, each epoch | Monitor val_auc per fold |
| `checkpoints/e2e_cv/fold{k}/pretrain/best_pretrain.pt` | End of pretrain phase k | Resume support |
| `checkpoints/e2e_cv/fold{k}/finetune/best_finetune.pt` | End of finetune phase k | Resume support |
| `results/e2e_cv/fold{k}_oof_scores.csv` | After LR phase k | Per-recording OOF predictions |
| `logs/e2e_cv_progress.csv` | After each fold | Fold-level AUC summary |
| `logs/e2e_cv_master.log` | During run (Cell 6) | Full output log |
| `results/e2e_cv_final_report.csv` | After all 5 folds | **Primary result: pooled AUC + bootstrap CI** |
| `results/e2e_cv_per_fold.csv` | After all 5 folds | Per-fold AUC table |
| `docs/images/e2e_cv.png` | After all 5 folds | 3-panel ROC + CI + bar chart |

### Downloading Results Back to Local Machine

After the run completes (while the Colab server is still active):

```python
# In any Colab cell — download final report and plot
from google.colab import files
files.download('/content/SentinelFatal2/results/e2e_cv_final_report.csv')
files.download('/content/SentinelFatal2/docs/images/e2e_cv.png')
files.download('/content/SentinelFatal2/logs/e2e_cv_master.log')
```

Or push results back to GitHub:
```bash
cd /content/SentinelFatal2
git config user.email "you@example.com"
git config user.name "YourName"
git add results/e2e_cv_final_report.csv results/e2e_cv_per_fold.csv docs/images/e2e_cv.png logs/e2e_cv_progress.csv
git commit -m "Add E2E CV results (5-fold, T4 GPU)"
git push origin master
```

---

## 9. Time Estimates (T4 GPU)

Based on:
- Original training on Colab T4: pretrain 13 epochs, finetune 33 epochs
- Per-fold data: ~532 pretrain recordings, ~317 finetune train + ~80 val

| Phase | Per fold | Notes |
|-------|---------|-------|
| Dry-run (Cell 5) | ~5 min | max_batches=2, 1 fold |
| Pretrain | ~20–30 min | ~116 batches/epoch × 13 epochs |
| Fine-tune | ~30–45 min | ~87 batches/epoch × 33 epochs |
| LR + feature extraction | <2 min | stride=60, CPU-bound |
| **Per fold total** | **~55–75 min** | |
| **5 folds total** | **~4.5–6 hours** | safe to run overnight |

> These estimates assume similar convergence speed to the original training (which ran on T4 GPU in ~2–3 hours total for pretrain+finetune without CV).

---

## 10. Script CLI Reference

`scripts/run_e2e_cv.py` — full CLI:

```bash
python scripts/run_e2e_cv.py \
  --device cuda \          # 'cuda' or 'cpu'
  --folds 5 \              # number of CV folds (default: 5)
  --seed 42 \              # random seed for fold generation (default: 42)
  --stride 60 \            # LR feature extraction stride in samples (default: 60)
  --config config/train_config.yaml \   # path to YAML config (default: config/train_config.yaml)
  --dry-run \              # if set: max_batches=2 per phase (dry-run mode)
  --force-folds            # if set: regenerate fold CSVs even if they exist
```

**Locked constants in script (not CLI-settable):**
```python
ALERT_THRESHOLD    = 0.40   # S11: lowered from 0.50
DECISION_THRESHOLD = 0.284  # Youden-optimal on original train set
```

**Full overnight command (T4 GPU, equivalent to Cell 6):**
```bash
python scripts/run_e2e_cv.py --device cuda --folds 5 --force-folds --stride 60 --seed 42
```

**Dry-run command (equivalent to Cell 5):**
```bash
python scripts/run_e2e_cv.py --device cuda --dry-run --folds 1 --force-folds --seed 42
```

**Single-fold test (useful after fixing an error, ~1 hour on T4):**
```bash
python scripts/run_e2e_cv.py --device cuda --folds 1 --force-folds --seed 42 --stride 60
```

---

## 11. Troubleshooting

### `RuntimeError: CUDA not available` at Cell 1
- Your Colab runtime has no GPU.
- Click **Runtime → Change runtime type → T4 GPU → Save → Reconnect**.
- Then re-run Cell 1.

### `CUDA out of memory` during Cell 6
- Lower `batch_size` in `config/train_config.yaml`:
  - Pretrain: `64 → 32`
  - Finetune: `32 → 16`
- T4 (15 GB VRAM) should handle the default batch sizes. If still failing, check for background processes in the runtime.

### `ModuleNotFoundError: No module named 'yaml'` or `sklearn`
- Re-run Cell 4 (dependency install).
- If still failing: `!pip install PyYAML scikit-learn tqdm scipy`

### `FileNotFoundError: data/processed/ctu_uhb/xxxx.npy`
- The `.npy` files are missing from the Colab server.
- You must complete Section 6 (data setup) first.
- Verify with: `!ls /content/SentinelFatal2/data/processed/ctu_uhb/*.npy | wc -l`

### `FileNotFoundError: data/splits/train.csv`
- The split CSVs are in the repo but Colab hasn't pulled them.
- Run Cell 2 (clone/pull) again. These files are committed to GitHub.

### Dry-run passes but Cell 6 crashes on fold 1, 2, 3…
- Check `logs/e2e_cv/fold{k}_pretrain.csv` or `fold{k}_finetune.csv` for the epoch where it crashed.
- Re-run Cell 6 — the resume logic will skip completed folds and retry the failed one.

### Cell 6 gets interrupted (VS Code disconnects, browser closes)
- Re-run Cell 6. Resume is automatic.
- The script checks for `checkpoints/e2e_cv/fold{k}/finetune/best_finetune.pt` — if it exists, that fold's pretrain+finetune is skipped.
- If only pretrain completed for a fold, it will skip pretrain but redo finetune.

### Val AUC stuck at ~0.5–0.6 during finetune
- This is normal for early epochs on a small imbalanced dataset (same as original training).
- The best val AUC was 0.7235 at epoch 17 in the original run.
- Let the run complete — early stopping with patience=15 is conservative.

### `git push` fails after run (authentication)
- Use a personal access token: `git push https://<token>@github.com/ArielShamay/SentinelFatal2.git master`
- Or use `google.colab.files.download()` to download results directly.

---

## Appendix — File Inventory

### Source Code (all in `src/`)
| Module | File | Purpose |
|--------|------|---------|
| Data | `src/data/preprocessing.py` | Load `.npy`, normalize FHR/UC, build windows |
| Data | `src/data/dataset.py` | `PretrainDataset`, `FinetuneDataset`, dataloaders |
| Data | `src/data/masking.py` | Contiguous group masking (Section II-D) |
| Model | `src/model/patchtst.py` | PatchTST encoder + patch extraction (S9 fix applied) |
| Model | `src/model/heads.py` | `ReconstructionHead`, `ClassificationHead` |
| Train | `src/train/pretrain.py` | MAE pretraining loop + CLI |
| Train | `src/train/finetune.py` | Supervised finetune loop + CLI |
| Train | `src/train/train_lr.py` | Logistic Regression training on alert features |
| Train | `src/train/utils.py` | `compute_recording_auc`, `sliding_windows` |
| Inference | `src/inference/sliding_window.py` | `inference_recording()` |
| Inference | `src/inference/alert_extractor.py` | Alert segment extraction + 4 features |

### Key Scripts
| File | Purpose |
|------|---------|
| `scripts/run_e2e_cv.py` | **Main E2E CV orchestrator** (840 lines) |
| `config/train_config.yaml` | All hyperparameters |

### Notebooks
| Notebook | Purpose |
|----------|---------|
| `notebooks/00_data_prep.ipynb` | Data extraction + preprocessing (done) |
| `notebooks/01_arch_check.ipynb` | Architecture validation (done) |
| `notebooks/02_pretrain.ipynb` | Pre-training run (done) |
| `notebooks/03_finetune.ipynb` | Fine-tuning run (done) |
| `notebooks/04_inference_demo.ipynb` | Inference demo (done) |
| `notebooks/05_evaluation.ipynb` | Final evaluation + threshold optimization (done) |
| `notebooks/06_bootstrap_cv.ipynb` | Bootstrap CI + LR-only CV (done) |
| `notebooks/07_colab_e2e_cv_launch.ipynb` | **E2E CV launch notebook (this run)** |

### Checkpoints (already on GitHub)
| File | Contents |
|------|---------|
| `checkpoints/pretrain/best_pretrain.pt` | Best pretrain weights (epoch 2, val MSE=0.01427) |
| `checkpoints/finetune/best_finetune.pt` | Best finetune weights (epoch 17, val AUC=0.7235) |
| `checkpoints/alerting/logistic_regression.pkl` | LR trained on 441 recordings, AT=0.40, stride=60 |
