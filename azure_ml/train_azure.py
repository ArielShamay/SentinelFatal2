#!/usr/bin/env python
"""
azure_ml/train_azure.py
=======================
Entrypoint that runs INSIDE the Azure ML compute node.

What this script does:
  1. Downloads processed CTG data from GitHub (if the .npy files are missing)
  2. Applies all fixes from docs/new_training_spec.md (Config A, num_workers=0,
     val_every_n_epochs=5, resume_from_epoch, per_epoch_callback, patience=15)
  3. Runs the full 5-Fold E2E Cross-Validation
  4. Saves final results + checkpoints to Azure ML output directories

Azure ML mounts the repo code at the working directory (current dir = repo root).
Outputs are written to $AZUREML_OUTPUT_results, $AZUREML_OUTPUT_checkpoints, etc.
"""

from __future__ import annotations

import argparse
import gc
import glob
import inspect
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── 0. Working directory = repo root (Azure ML mounts code here) ──────────
REPO_DIR = Path(os.getcwd()).resolve()
print(f"[INIT] repo root: {REPO_DIR}")
print(f"[INIT] Python   : {sys.version}")

# ── 1. Download processed data if missing ─────────────────────────────────
DATA_ZIP_URL = (
    "https://raw.githubusercontent.com/ArielShamay/SentinelFatal2/master/data_processed.zip"
)
CTU_DIR = REPO_DIR / "data" / "processed" / "ctu_uhb"


def _count_npy(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix == ".npy" and f.name != ".gitkeep")


npy_count = _count_npy(CTU_DIR)
if npy_count < 552:
    print(f"[DATA] Only {npy_count} .npy files found — downloading data_processed.zip ...")
    zip_path = Path("/tmp/data_processed.zip")
    subprocess.check_call(["wget", "-q", "-O", str(zip_path), DATA_ZIP_URL])
    subprocess.check_call(["unzip", "-q", str(zip_path), "-d", str(REPO_DIR / "data")])
    zip_path.unlink(missing_ok=True)
    npy_count = _count_npy(CTU_DIR)
    print(f"[DATA] After download: {npy_count} .npy files.")
else:
    print(f"[DATA] {npy_count} .npy files present — skipping download.")

# ── 2. Add repo to Python path ─────────────────────────────────────────────
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# ── 3. Force-reload src modules (in case any were cached) ─────────────────
for mod_key in list(sys.modules.keys()):
    if mod_key.startswith("src.") or mod_key.startswith("scripts."):
        del sys.modules[mod_key]

# ── 4. Imports ─────────────────────────────────────────────────────────────
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.train import finetune as ft
from src.train.pretrain import pretrain
from src.train.utils import compute_recording_auc
from src.model.patchtst import load_config
from scripts.run_e2e_cv_v2 import (
    generate_cv_splits,
    extract_features_for_split,
    fit_lr_model,
    predict_lr,
    at_sweep,
    clinical_threshold,
    bootstrap_auc_ci,
)

# Verify finetune.py has the required parameters
sig_params = list(inspect.signature(ft.train).parameters)
for req in ["val_every_n_epochs", "resume_from_epoch", "per_epoch_callback"]:
    assert req in sig_params, (
        f"finetune.py is missing parameter '{req}'. "
        f"Apply the changes from docs/new_training_spec.md §3.5 first."
    )
print("[OK] finetune.py signature verified.")

# ── 5. Constants ───────────────────────────────────────────────────────────
SEED             = 42
N_FOLDS          = 5
N_BOOTSTRAP      = 10_000
SPEC_CONSTRAINT  = 0.65
AT_CANDIDATES    = [0.30, 0.35, 0.40, 0.45]
N_FEATURES       = 12
VAL_EVERY_N      = 5
PATIENCE         = 15

# Azure ML output directories (fall back to local if env vars not set)
OUT_RESULTS  = Path(os.environ.get("AZUREML_OUTPUT_results",    str(REPO_DIR / "results"   / "e2e_cv_v3")))
OUT_CKPTS    = Path(os.environ.get("AZUREML_OUTPUT_checkpoints", str(REPO_DIR / "checkpoints" / "e2e_cv_v3")))
OUT_LOGS     = Path(os.environ.get("AZUREML_OUTPUT_logs",        str(REPO_DIR / "logs"        / "e2e_cv_v3")))

CONFIG_PATH  = str(REPO_DIR / "config" / "train_config.yaml")

CONFIG_A_OVERRIDES = {
    "loss":         "cross_entropy",
    "class_weight": [1.0, 3.9],
    "patience":     PATIENCE,
    "train_stride": 120,
    "val_stride":   120,
}

# Paths for shared pre-train checkpoint
SHARED_PRETRAIN_CKPT_PRIMARY  = REPO_DIR / "checkpoints" / "e2e_cv_v2" / "shared_pretrain" / "best_pretrain.pt"
SHARED_PRETRAIN_CKPT_FALLBACK = REPO_DIR / "checkpoints" / "pretrain" / "best_pretrain.pt"
SHARED_PRETRAIN_CKPT_V3       = OUT_CKPTS / "shared_pretrain" / "best_pretrain.pt"

# ── 6. Determinism ─────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE}")
if DEVICE == "cuda":
    print(f"         {torch.cuda.get_device_name(0)}")
    total = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
    print(f"         VRAM: {total} MB")

# ── 7. Create output directories ───────────────────────────────────────────
for p in [OUT_RESULTS, OUT_CKPTS, OUT_LOGS]:
    p.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] {p}")

for k in range(N_FOLDS):
    (OUT_RESULTS / f"fold{k}_splits").mkdir(parents=True, exist_ok=True)
    (OUT_CKPTS   / f"fold{k}" / "finetune").mkdir(parents=True, exist_ok=True)

# ── 8. Monkey-patch DataLoader → num_workers=0 ────────────────────────────
import torch.utils.data as _tdata
_OrigDataLoader = _tdata.DataLoader

class _SafeDataLoader(_OrigDataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"]  = False
        super().__init__(*args, **kwargs)

_tdata.DataLoader = _SafeDataLoader
print("[PATCH] DataLoader.num_workers hard-set to 0 (deadlock prevention).")

# ── 9. Data sanity checks ──────────────────────────────────────────────────
splits_dir = REPO_DIR / "data" / "splits"
train_df = pd.read_csv(splits_dir / "train.csv")
val_df   = pd.read_csv(splits_dir / "val.csv")
test_df  = pd.read_csv(splits_dir / "test.csv")

all_df = pd.concat([train_df, val_df, test_df]).drop_duplicates(subset=["id"]).reset_index(drop=True)
assert len(all_df) == 552, f"Expected 552 recordings, got {len(all_df)}"
all_csv = splits_dir / "train_val_test.csv"
all_df.to_csv(all_csv, index=False)
print(f"[SANITY] train_val_test.csv: {len(all_df)} recordings.")

label_col = "target" if "target" in all_df.columns else all_df.columns[-1]
pos = (all_df[label_col] == 1).sum()
neg = (all_df[label_col] == 0).sum()
print(f"[SANITY] Labels — positive (acidemia): {pos}, negative: {neg}")
assert pos == 113, f"Expected 113 positives, got {pos}"
assert neg == 439, f"Expected 439 negatives, got {neg}"
print("[SANITY] PASSED ✓")

# ── 10. Shared pre-training ────────────────────────────────────────────────
SHARED_PRETRAIN_CKPT = None
for candidate in [SHARED_PRETRAIN_CKPT_PRIMARY, SHARED_PRETRAIN_CKPT_FALLBACK, SHARED_PRETRAIN_CKPT_V3]:
    if candidate.exists():
        SHARED_PRETRAIN_CKPT = str(candidate)
        print(f"[PRETRAIN] Checkpoint found — skipping pretraining.\n           {SHARED_PRETRAIN_CKPT}")
        break

if SHARED_PRETRAIN_CKPT is None:
    print("[PRETRAIN] No checkpoint found — running pretraining ...")
    pt_ckpt_dir = str(OUT_CKPTS / "shared_pretrain")
    pt_log_path = str(OUT_LOGS   / "shared_pretrain_loss.csv")
    pretrain(
        config_path    = CONFIG_PATH,
        device_str     = DEVICE,
        max_batches    = 0,
        processed_root = str(REPO_DIR / "data" / "processed"),
        pretrain_csv   = str(splits_dir / "pretrain.csv"),
        checkpoint_dir = pt_ckpt_dir,
        log_path       = pt_log_path,
    )
    SHARED_PRETRAIN_CKPT = str(OUT_CKPTS / "shared_pretrain" / "best_pretrain.pt")
    gc.collect(); torch.cuda.empty_cache()
    print(f"[PRETRAIN] Done → {SHARED_PRETRAIN_CKPT}")

# G1 gate: check val_mse
for log_candidate in [
    OUT_LOGS / "shared_pretrain_loss.csv",
    REPO_DIR / "logs" / "e2e_cv_v2" / "shared_pretrain_loss.csv",
    REPO_DIR / "logs" / "pretrain_loss.csv",
]:
    if log_candidate.exists():
        pt_log = pd.read_csv(log_candidate)
        val_col = [c for c in pt_log.columns if "val" in c.lower() and "loss" in c.lower()]
        if val_col:
            min_val = float(pt_log[val_col[0]].min())
            gate = "PASSED ✓" if min_val < 0.015 else "WARNING — val_mse ≥ 0.015"
            print(f"[G1 GATE] min val_mse = {min_val:.5f}  →  {gate}")
        break

# ── 11. Generate 5-Fold CV splits ─────────────────────────────────────────
cv_splits = generate_cv_splits(all_csv, n_folds=N_FOLDS, seed=SEED)
print(f"\n[CV] Generated {N_FOLDS}-fold splits.")
for k, (tr, vl, te) in enumerate(cv_splits):
    pos_frac = tr[label_col].mean()
    print(f"     Fold {k}: train={len(tr)}, val={len(vl)}, test={len(te)}, "
          f"train_pos_frac={pos_frac:.3f}")
    tr.to_csv(OUT_RESULTS / f"fold{k}_splits" / f"fold{k}_train.csv", index=False)
    vl.to_csv(OUT_RESULTS / f"fold{k}_splits" / f"fold{k}_val.csv",   index=False)
    te.to_csv(OUT_RESULTS / f"fold{k}_splits" / f"fold{k}_test.csv",  index=False)

# ── 12. Helpers ────────────────────────────────────────────────────────────
def _prune_old_checkpoints(ckpt_dir: Path, keep: int = 3) -> None:
    """Keep the `keep` most-recent epoch_*.pt files plus best_finetune.pt."""
    epoch_ckpts = sorted(
        ckpt_dir.glob("epoch_*.pt"),
        key=lambda p: int(re.search(r"epoch_(\d+)", p.stem).group(1)),
    )
    for old in epoch_ckpts[: max(0, len(epoch_ckpts) - keep)]:
        old.unlink()


def _detect_resume_epoch(ckpt_dir: Path) -> int:
    """Return the next epoch to train (highest saved epoch + 1), or 0."""
    files = list(ckpt_dir.glob("epoch_*.pt"))
    if not files:
        return 0
    latest = max(int(re.search(r"epoch_(\d+)", f.stem).group(1)) for f in files)
    return latest + 1


# ── 13. Fold loop ─────────────────────────────────────────────────────────
fold_summaries = []
job_start = time.time()

for k, (train_fold_df, val_fold_df, test_fold_df) in enumerate(cv_splits):
    oof_csv = OUT_RESULTS / f"fold{k}_oof_scores.csv"
    if oof_csv.exists():
        print(f"\n[FOLD {k}] OOF CSV found — skipping (already complete).")
        fold_summaries.append(pd.read_csv(oof_csv))
        continue

    fold_start = time.time()
    print(f"\n{'='*60}")
    print(f"FOLD {k} / {N_FOLDS - 1}")
    print(f"{'='*60}")

    # Write split CSVs
    fold_train_csv = OUT_RESULTS / f"fold{k}_splits" / f"fold{k}_train.csv"
    fold_val_csv   = OUT_RESULTS / f"fold{k}_splits" / f"fold{k}_val.csv"
    fold_test_csv  = OUT_RESULTS / f"fold{k}_splits" / f"fold{k}_test.csv"
    train_fold_df.to_csv(fold_train_csv, index=False)
    val_fold_df  .to_csv(fold_val_csv,   index=False)
    test_fold_df .to_csv(fold_test_csv,  index=False)

    # Checkpoint directory for this fold
    fold_ckpt_dir = OUT_CKPTS / f"fold{k}" / "finetune"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume detection
    resume_from = _detect_resume_epoch(fold_ckpt_dir)
    print(f"[FOLD {k}] Starting from epoch {resume_from}.")

    # Per-epoch callback
    _best_auc_tracker = {"auc": 0.0, "epoch": -1}

    def _epoch_callback(epoch: int, train_loss: float, val_auc: float) -> None:
        if val_auc > _best_auc_tracker["auc"]:
            _best_auc_tracker["auc"]   = val_auc
            _best_auc_tracker["epoch"] = epoch
        _prune_old_checkpoints(fold_ckpt_dir, keep=3)
        sys.stdout.flush()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  [Epoch {epoch:03d}] loss={train_loss:.4f}  val_auc={val_auc:.4f}  "
              f"best={_best_auc_tracker['auc']:.4f}@{_best_auc_tracker['epoch']}",
              flush=True)

    # ── Fine-tune this fold ────────────────────────────────────────────────
    ft.train(
        config_path         = CONFIG_PATH,
        device_str          = DEVICE,
        max_batches         = 0,
        processed_root      = str(REPO_DIR / "data" / "processed"),
        train_csv           = str(fold_train_csv),
        val_csv             = str(fold_val_csv),
        pretrain_checkpoint = SHARED_PRETRAIN_CKPT,
        checkpoint_dir      = str(fold_ckpt_dir),
        log_path            = str(OUT_LOGS / f"fold{k}_finetune_loss.csv"),
        quiet               = True,
        config_overrides    = CONFIG_A_OVERRIDES,
        save_epoch_ckpts    = True,
        val_every_n_epochs  = VAL_EVERY_N,
        resume_from_epoch   = resume_from,
        per_epoch_callback  = _epoch_callback,
    )

    best_ckpt = fold_ckpt_dir / "best_finetune.pt"
    print(f"[FOLD {k}] Training done. Best epoch: {_best_auc_tracker['epoch']} "
          f"(AUC={_best_auc_tracker['auc']:.4f}). Elapsed: {(time.time()-fold_start)/60:.1f} min")

    # ── Feature extraction + LR ────────────────────────────────────────────
    print(f"[FOLD {k}] Extracting features ...")
    train_feats, train_labels = extract_features_for_split(
        csv_path=fold_train_csv,
        checkpoint_path=str(best_ckpt),
        config_path=CONFIG_PATH,
        processed_root=str(REPO_DIR / "data" / "processed"),
        device=DEVICE,
        n_features=N_FEATURES,
    )
    val_feats, val_labels = extract_features_for_split(
        csv_path=fold_val_csv,
        checkpoint_path=str(best_ckpt),
        config_path=CONFIG_PATH,
        processed_root=str(REPO_DIR / "data" / "processed"),
        device=DEVICE,
        n_features=N_FEATURES,
    )
    test_feats, test_labels = extract_features_for_split(
        csv_path=fold_test_csv,
        checkpoint_path=str(best_ckpt),
        config_path=CONFIG_PATH,
        processed_root=str(REPO_DIR / "data" / "processed"),
        device=DEVICE,
        n_features=N_FEATURES,
    )

    # AT sweep on validation set
    at_results = at_sweep(
        val_feats, val_labels,
        at_candidates=AT_CANDIDATES,
        spec_constraint=SPEC_CONSTRAINT,
    )
    at_df = pd.DataFrame(at_results)
    at_df.to_csv(OUT_RESULTS / f"fold{k}_at_sweep.csv", index=False)

    best_at_row = at_df.sort_values("val_auc", ascending=False).iloc[0]
    best_at     = float(best_at_row["at_candidate"])
    print(f"[FOLD {k}] Best AT = {best_at}  (val_auc={best_at_row['val_auc']:.4f})")

    # Fit LR model, predict on test
    lr_model = fit_lr_model(train_feats, train_labels, at=best_at)
    test_scores = predict_lr(lr_model, test_feats, at=best_at)

    thr, sens, spec = clinical_threshold(test_labels, test_scores, spec_constraint=SPEC_CONSTRAINT)
    test_auc = float(roc_auc_score(test_labels, test_scores))
    print(f"[FOLD {k}] Test  AUC={test_auc:.4f}  Sens={sens:.3f}  Spec={spec:.3f}  thr={thr:.4f}")

    # Save OOF CSV
    oof_df = test_fold_df[["id", label_col]].copy()
    oof_df["y_score"] = test_scores
    oof_df["fold"]    = k
    oof_df.to_csv(oof_csv, index=False)

    fold_summaries.append(oof_df)
    fold_summaries[-1]["fold_auc"]  = test_auc
    fold_summaries[-1]["fold_sens"] = sens
    fold_summaries[-1]["fold_spec"] = spec
    fold_summaries[-1]["fold_thr"]  = thr
    fold_summaries[-1]["best_at"]   = best_at

    gc.collect(); torch.cuda.empty_cache()

# ── 14. Global OOF AUC + bootstrap CI ─────────────────────────────────────
print(f"\n{'='*60}")
print("GLOBAL OOF RESULTS")
print(f"{'='*60}")

all_oof = pd.concat(fold_summaries, ignore_index=True)
all_oof.to_csv(OUT_RESULTS / "global_oof_predictions.csv", index=False)

y_true  = all_oof[label_col].values
y_score = all_oof["y_score"].values
global_auc          = float(roc_auc_score(y_true, y_score))
ci_lo, ci_hi        = bootstrap_auc_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP, seed=SEED)
global_thr, g_sens, g_spec = clinical_threshold(y_true, y_score, spec_constraint=SPEC_CONSTRAINT)

print(f"  Global OOF AUC : {global_auc:.4f}  (95% CI: {ci_lo:.4f}–{ci_hi:.4f})")
print(f"  Clinical thr   : {global_thr:.4f}  Sens={g_sens:.3f}  Spec={g_spec:.3f}")

# Per-fold summary
per_fold = []
for k in range(N_FOLDS):
    oof_f = pd.read_csv(OUT_RESULTS / f"fold{k}_oof_scores.csv")
    auc_k = float(roc_auc_score(oof_f[label_col], oof_f["y_score"]))
    per_fold.append({"fold": k, "test_auc": auc_k})
per_fold_df = pd.DataFrame(per_fold)
per_fold_df.to_csv(OUT_RESULTS / "per_fold_summary.csv", index=False)
print(f"\n  Per-fold AUCs: {', '.join(f'{r.test_auc:.4f}' for _, r in per_fold_df.iterrows())}")
print(f"  Mean ± Std   : {per_fold_df.test_auc.mean():.4f} ± {per_fold_df.test_auc.std():.4f}")

# Final report
report = pd.DataFrame([{
    "run":           "e2e_cv_v3",
    "config":        "A",
    "global_auc":    global_auc,
    "ci_lo":         ci_lo,
    "ci_hi":         ci_hi,
    "global_sens":   g_sens,
    "global_spec":   g_spec,
    "global_thr":    global_thr,
    "mean_fold_auc": per_fold_df.test_auc.mean(),
    "std_fold_auc":  per_fold_df.test_auc.std(),
    "total_min":     round((time.time() - job_start) / 60, 1),
}])
report.to_csv(OUT_RESULTS / "final_cv_report_v3.csv", index=False)
print(f"\n[DONE] final_cv_report_v3.csv saved.")
print(f"[DONE] Total runtime: {(time.time()-job_start)/60:.1f} min")
