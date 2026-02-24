# S14 — Colab Run Guide

## Background

The S14 iteration targets improving E2E 5-Fold Cross-Validation AUC from **0.565** (baseline) to **>= 0.78**.

### What Was Wrong (S13 Baseline = AUC 0.565)

| Problem | Evidence | Fix in S14 |
|---------|----------|------------|
| Backbone nearly frozen | lr_backbone=1e-5 caused <0.1% weight change over 33 epochs | lr_backbone=5e-5 (5x increase) |
| Noisy val_AUC caused bad checkpoints | Raw AUC oscillated wildly between epochs | EMA smoothing (beta=0.8) for early stopping |
| No LR warmup | Pretrained weights damaged in first epochs | 5-epoch linear warmup (0 -> target LR) |
| Feature scale mismatch in LR | segment_length (~minutes) vs max_prediction (~0-1) | StandardScaler before LogisticRegression |
| LR regularization too loose | C=0.5 with 4 features on ~300 samples | C=0.1 (tighter) |
| Coarse feature extraction stride | stride=60 = 15-sec steps (misses short patterns) | stride=24 = 1 patch step (6-sec) |
| Only 4 features from longest segment | Ignores how much of the recording is alerting | +2 record-level features (6 total) |

### S14 Changes Summary

```
config/train_config.yaml     lr_backbone: 1e-5 -> 5e-5, add lr_warmup_epochs: 5
src/train/finetune.py        5-epoch warmup, EMA val_AUC (beta=0.8), skip scheduler during warmup
scripts/run_e2e_cv.py        StandardScaler + LR(C=0.1), 2 record-level features
notebooks/07_colab_e2e_cv.   stride 60 -> 24
alert_extractor.py           ZERO_FEATURES updated (6 keys)
```

### Feature Vector (6 features)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `segment_length` | Longest alert segment | Duration in minutes |
| 2 | `max_prediction` | Longest alert segment | Peak P(acidemia) score |
| 3 | `cumulative_sum` | Longest alert segment | Integral of scores (score * seconds) |
| 4 | `weighted_integral` | Longest alert segment | Integral of (score-0.5)^2 |
| 5 | `n_alert_segments` | **Entire recording** | Total number of alert segments |
| 6 | `alert_fraction` | **Entire recording** | Fraction of windows above threshold |

---

## Step-by-Step Instructions

### Prerequisites

- VS Code with Google Colab extension installed
- Google account with Colab access
- Repository pushed to GitHub (commit `b7700be` or later)

### Step 1 — Connect to Colab GPU

1. Open `notebooks/07_colab_e2e_cv_launch.ipynb` in VS Code
2. Click **Select Kernel** (top-right) -> **Colab** -> **New Colab Server**
3. Select **T4 GPU** runtime
4. Run **Cell 1** — verify output shows:
   ```
   GPU  : Tesla T4
   VRAM : 15.6 GB
   DEVICE = 'cuda'
   ```

### Step 2 — Pull Latest Code

Run **Cell 2**. It clones or updates the repo on the Colab server.

Expected output:
```
Repository already on server — pulling latest changes...
```

Verify the S14 commit appears in the pull output (files like `finetune.py`, `run_e2e_cv.py`, `train_config.yaml`).

### Step 3 — Extract Data

Run **Cell 3**. Extracts `data_processed.zip` into `data/processed/`.

Expected:
```
ctu_uhb .npy : 552
fhrma   .npy : 135
```

### Step 4 — Install Dependencies

Run **Cell 4**. Usually says "All dependencies already installed."

### Step 5 — Dry-Run (2 minutes)

Run **Cell 5**. This runs 2 folds with `max_batches=2` to verify the full pipeline works.

**What to check:**
- No import errors (StandardScaler, Pipeline)
- Warmup tag appears: `[warmup]` in epoch logs
- `smooth_auc` column appears in logs
- Feature shape shows `(N, 6)` not `(N, 4)`
- Ends with: `Dry-run PASSED`

### Step 6 — Full Run (60-90 minutes)

Run **Cell 6**. This is the real run.

**What Cell 6 does automatically:**
1. Clears old fold artifacts (checkpoints, results, logs from AUC=0.565 run)
2. Preserves shared `best_pretrain.pt`
3. Runs 5-fold E2E CV with all S14 improvements

**Expected timeline on T4 GPU:**

| Phase | Per Fold | Total (5 folds) |
|-------|----------|-----------------|
| Fine-tune (150 epochs max, patience=25) | ~8-15 min | ~40-75 min |
| Feature extraction (stride=24) | ~3-5 min | ~15-25 min |
| LR fit + OOF | < 1 sec | < 5 sec |
| **Total** | **~12-20 min** | **~60-90 min** |

**Keep VS Code open the entire time.** If VS Code disconnects, the run stops. But Cell 6 supports auto-resume — just re-run it and completed folds will be skipped.

**Live monitoring during the run:**
- Each fold prints per-epoch training logs with `val_auc`, `smooth_auc`, `lr_bb`, `lr_hd`
- Warmup epochs (0-4) show `[warmup]` tag
- After fine-tuning, feature extraction logs show progress
- After each fold, a per-fold AUC is printed

### Step 7 — Check Results

Run **Cell 7** (or Cell 8) after the run completes.

**What to look for:**

```
AUC         = X.XXX  [95% CI: X.XXX - X.XXX]
Sensitivity = X.XXX  [95% CI: X.XXX - X.XXX]
Specificity = X.XXX  [95% CI: X.XXX - X.XXX]
```

**Success criteria:**
- AUC >= 0.78 (target)
- AUC >= 0.70 (minimum acceptable)
- All 5 folds completed
- No zero-feature errors (alert_fraction > 0 for positives)

---

## Troubleshooting

### "No module named 'src'"
Cell 2 didn't run or kernel restarted. Re-run Cell 2.

### CUDA out of memory
Unlikely with T4 (15.6 GB) and batch_size=32. If it happens, reduce `batch_size` in config.

### Fold fails with "only one class"
Stratification should prevent this. If it happens, it's a data issue — check fold CSVs.

### Run interrupted
Re-run Cell 6. Completed folds are auto-detected and skipped.

### AUC still low (< 0.65)
Possible next steps:
- Try C in [0.01, 0.1, 1.0] grid search (inner CV)
- Increase d_model from 128 to 256 (more capacity)
- Try CosineAnnealingWarmRestarts scheduler
- Ensemble multiple thresholds

---

## File Reference

| File | Role |
|------|------|
| `config/train_config.yaml` | All hyperparameters (source of truth) |
| `src/train/finetune.py` | Fine-tuning loop (warmup + EMA) |
| `scripts/run_e2e_cv.py` | E2E CV orchestrator (fold generation, training, features, LR, aggregation) |
| `src/inference/alert_extractor.py` | Alert segment extraction + feature computation |
| `src/inference/sliding_window.py` | Sliding window inference |
| `notebooks/07_colab_e2e_cv_launch.ipynb` | Colab launcher notebook |
| `logs/e2e_cv_progress.csv` | Live per-fold results (updated after each fold) |
| `results/e2e_cv_final_report.csv` | Final aggregated report |
| `docs/images/e2e_cv.png` | ROC curves + AUC comparison visualization |
