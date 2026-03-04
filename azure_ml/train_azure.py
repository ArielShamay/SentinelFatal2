#!/usr/bin/env python
"""
azure_ml/train_azure.py
=======================
Entrypoint that runs INSIDE the Azure ML compute node.

What this script does:
  1. Gets processed CTG data — either from an Azure ML Data Asset input (fast,
     same-region Blob download) or falls back to downloading from GitHub.
  2. Applies all fixes from docs/new_training_spec.md (Config A, num_workers=0,
     val_every_n_epochs=5, resume_from_epoch, per_epoch_callback, patience=15)
  3. Runs the full 5-Fold E2E Cross-Validation
  4. Saves final results + checkpoints to Azure ML output directories

Azure ML mounts the repo code at the working directory (current dir = repo root).
Outputs are written to $AZUREML_OUTPUT_results, $AZUREML_OUTPUT_checkpoints, etc.

Usage (Azure ML passes --data automatically from the job input):
    python azure_ml/train_azure.py --data /path/to/data_processed.zip
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── 0. Working directory = repo root (Azure ML mounts code here) ──────────
REPO_DIR = Path(os.getcwd()).resolve()
print(f"[INIT] repo root: {REPO_DIR}")
print(f"[INIT] Python   : {sys.version}")

# ── 1. Get processed data ──────────────────────────────────────────────────
# Parse --data argument (Azure ML passes the downloaded zip path here)
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--data", default=None,
                 help="Local path to data_processed.zip (provided by Azure ML data input)")
_args, _ = _ap.parse_known_args()

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
    if _args.data:
        # Azure ML Data Asset: zip was downloaded to a local path before the job started
        zip_path = Path(_args.data)
        print(f"[DATA] Using Azure ML data input: {zip_path}  "
              f"({zip_path.stat().st_size // (1024*1024)} MB)")
    else:
        # Fallback: download from GitHub
        print(f"[DATA] --data not provided; downloading data_processed.zip from GitHub ...")
        zip_path = Path("/tmp/data_processed.zip")
        subprocess.check_call(["wget", "-q", "-O", str(zip_path), DATA_ZIP_URL])

    print(f"[DATA] Extracting {zip_path.name} ...")
    subprocess.check_call(["unzip", "-q", "-o", str(zip_path), "-d", str(REPO_DIR / "data")])
    if not _args.data:
        zip_path.unlink(missing_ok=True)

    npy_count = _count_npy(CTU_DIR)
    print(f"[DATA] {npy_count} .npy files ready.")
else:
    print(f"[DATA] {npy_count} .npy files present — skipping extraction.")

# ── 2. Add repo to Python path ─────────────────────────────────────────────
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# ── 3. Force-reload src modules (in case any were cached) ─────────────────
for mod_key in list(sys.modules.keys()):
    if mod_key.startswith("src.") or mod_key.startswith("scripts."):
        del sys.modules[mod_key]

# ── 4. Imports ─────────────────────────────────────────────────────────────
import random
import shutil as _shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.train import finetune as ft
from src.train.pretrain import pretrain
from src.model.patchtst import PatchTST, load_config
from src.model.heads import ClassificationHead
from scripts.run_e2e_cv_v2 import (
    generate_cv_splits,
    extract_features_for_split,
    fit_lr_model,
    predict_lr,
    at_sweep,
    clinical_threshold,
    bootstrap_auc_ci,
)

# Verify finetune.py has all required parameters (spec §3.5)
sig_params = list(inspect.signature(ft.train).parameters)
for req in ["val_every_n_epochs", "resume_from_epoch", "per_epoch_callback", "resume_checkpoint"]:
    assert req in sig_params, (
        f"finetune.py is missing parameter '{req}'. "
        f"Apply the changes from docs/new_training_spec.md section 3.5 first."
    )
print("[OK] finetune.py signature verified.")

# ── 5. Constants ───────────────────────────────────────────────────────────
SEED            = 42
N_FOLDS         = 5
N_BOOTSTRAP     = 10_000
SPEC_CONSTRAINT = 0.65
AT_CANDIDATES   = [0.30, 0.35, 0.40, 0.45]
N_FEATURES      = 12
VAL_EVERY_N     = 5
PATIENCE        = 20                   # ← 15→20: allows recovery from unfreeze AUC drops

# Azure ML output directories (fall back to local if env vars not set)
OUT_RESULTS = Path(os.environ.get("AZUREML_OUTPUT_results",    str(REPO_DIR / "results"     / "e2e_cv_v3")))
OUT_CKPTS   = Path(os.environ.get("AZUREML_OUTPUT_checkpoints", str(REPO_DIR / "checkpoints" / "e2e_cv_v3")))
OUT_LOGS    = Path(os.environ.get("AZUREML_OUTPUT_logs",        str(REPO_DIR / "logs"        / "e2e_cv_v3")))

# Azure ML always uploads ./outputs/ at job end — persistent across compute dealloc
PERSISTENT_OUT = Path("./outputs")
(PERSISTENT_OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
(PERSISTENT_OUT / "results").mkdir(parents=True, exist_ok=True)

CONFIG_PATH = str(REPO_DIR / "config" / "train_config.yaml")

CONFIG_A_OVERRIDES = {
    "loss":         "cross_entropy",
    "class_weight": [1.0, 3.9],
    "patience":     PATIENCE,
    "train_stride": 120,
    "val_stride":   60,   # ← 120→60: denser validation signal for better model selection
}

# Shared pre-train checkpoint candidates (checked in order)
SHARED_PRETRAIN_CKPT_PRIMARY  = REPO_DIR / "checkpoints" / "e2e_cv_v2" / "shared_pretrain" / "best_pretrain.pt"
SHARED_PRETRAIN_CKPT_FALLBACK = REPO_DIR / "checkpoints" / "pretrain" / "best_pretrain.pt"
SHARED_PRETRAIN_CKPT_V3       = OUT_CKPTS / "shared_pretrain" / "best_pretrain.pt"

PROCESSED_ROOT = REPO_DIR / "data" / "processed"

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

# ── 8. Monkey-patch DataLoader -> num_workers=0 (spec Rule 3) ─────────────
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

all_df = (
    pd.concat([train_df, val_df, test_df])
    .drop_duplicates(subset=["id"])
    .reset_index(drop=True)
)
assert len(all_df) == 552, f"Expected 552 recordings, got {len(all_df)}"
all_csv = splits_dir / "train_val_test.csv"
all_df.to_csv(all_csv, index=False)
print(f"[SANITY] train_val_test.csv: {len(all_df)} recordings.")

label_col = "target" if "target" in all_df.columns else all_df.columns[-1]
pos = int((all_df[label_col] == 1).sum())
neg = int((all_df[label_col] == 0).sum())
print(f"[SANITY] Labels — positive (acidemia): {pos}, negative: {neg}")
assert pos == 113, f"Expected 113 positives, got {pos}"
assert neg == 439, f"Expected 439 negatives, got {neg}"
print("[SANITY] PASSED")

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
    pt_log_path = str(OUT_LOGS  / "shared_pretrain_loss.csv")
    pretrain(
        config_path    = CONFIG_PATH,
        device_str     = DEVICE,
        max_batches    = 0,
        processed_root = str(PROCESSED_ROOT),
        pretrain_csv   = str(splits_dir / "pretrain.csv"),
        checkpoint_dir = pt_ckpt_dir,
        log_path       = pt_log_path,
    )
    SHARED_PRETRAIN_CKPT = str(OUT_CKPTS / "shared_pretrain" / "best_pretrain.pt")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[PRETRAIN] Done -> {SHARED_PRETRAIN_CKPT}")

# G1 gate: check val_mse < 0.015
for log_candidate in [
    OUT_LOGS / "shared_pretrain_loss.csv",
    REPO_DIR / "logs" / "e2e_cv_v2" / "shared_pretrain_loss.csv",
    REPO_DIR / "logs" / "pretrain_loss.csv",
]:
    if log_candidate.exists():
        pt_log = pd.read_csv(log_candidate)
        val_cols = [c for c in pt_log.columns if "val" in c.lower() and "loss" in c.lower()]
        if val_cols:
            min_val = float(pt_log[val_cols[0]].min())
            gate = "PASSED" if min_val < 0.015 else "WARNING — val_mse >= 0.015"
            print(f"[G1 GATE] min val_mse = {min_val:.5f}  ->  {gate}")
        break

# ── 11. Generate 5-Fold CV splits ─────────────────────────────────────────
# generate_cv_splits returns List[Dict] with keys: train_ids, val_ids, test_ids, df_all
cv_splits = generate_cv_splits(all_csv, n_folds=N_FOLDS, seed=SEED)
print(f"\n[CV] Generated {N_FOLDS}-fold splits.")

for k, split in enumerate(cv_splits):
    df_all = split["df_all"]
    tr_df  = df_all[df_all["id"].isin(split["train_ids"])]
    vl_df  = df_all[df_all["id"].isin(split["val_ids"])]
    te_df  = df_all[df_all["id"].isin(split["test_ids"])]
    pos_frac = float(tr_df[label_col].mean())
    print(f"     Fold {k}: train={len(tr_df)}, val={len(vl_df)}, test={len(te_df)}, "
          f"train_pos_frac={pos_frac:.3f}")
    fold_split_dir = OUT_RESULTS / f"fold{k}_splits"
    fold_split_dir.mkdir(parents=True, exist_ok=True)
    tr_df.to_csv(fold_split_dir / f"fold{k}_train.csv", index=False)
    vl_df.to_csv(fold_split_dir / f"fold{k}_val.csv",   index=False)
    te_df.to_csv(fold_split_dir / f"fold{k}_test.csv",  index=False)
    # Store CSV paths in split dict for use in fold loop below
    split["train_csv"] = fold_split_dir / f"fold{k}_train.csv"
    split["val_csv"]   = fold_split_dir / f"fold{k}_val.csv"
    split["test_csv"]  = fold_split_dir / f"fold{k}_test.csv"

# ── 12. Helpers ────────────────────────────────────────────────────────────

def _prune_old_checkpoints(ckpt_dir: Path, keep: int = 3) -> None:
    """Keep the `keep` most-recent epoch_*.pt files; delete the rest."""
    epoch_ckpts = sorted(
        ckpt_dir.glob("epoch_*.pt"),
        key=lambda p: int(re.search(r"epoch_(\d+)", p.stem).group(1)),
    )
    for old in epoch_ckpts[: max(0, len(epoch_ckpts) - keep)]:
        old.unlink()


def _detect_resume_epoch(ckpt_dir: Path):
    """Return (next_epoch_to_train, latest_ckpt_path_or_None).

    If epoch_49.pt exists, returns (50, '/path/to/epoch_49.pt').
    If no epoch_*.pt files exist, returns (0, None).
    """
    files = list(ckpt_dir.glob("epoch_*.pt"))
    if not files:
        return 0, None
    latest = max(files, key=lambda f: int(re.search(r"epoch_(\d+)", f.stem).group(1)))
    epoch_num = int(re.search(r"epoch_(\d+)", latest.stem).group(1)) + 1
    return epoch_num, str(latest)


def _load_best_model(ckpt_path: str, device: str) -> torch.nn.Module:
    """Load PatchTST + ClassificationHead from a finetune checkpoint."""
    cfg   = load_config(CONFIG_PATH)
    model = PatchTST(cfg)
    d_in  = (
        int(cfg["data"]["n_patches"])
        * int(cfg["model"]["d_model"])
        * int(cfg["data"]["n_channels"])
    )
    model.replace_head(ClassificationHead(
        d_in=d_in,
        n_classes=int(cfg["finetune"]["n_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ))
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


# ── 13. Fold loop ─────────────────────────────────────────────────────────
fold_summaries = []
job_start = time.time()

for k, split in enumerate(cv_splits):
    oof_csv = OUT_RESULTS / f"fold{k}_oof_scores.csv"
    if oof_csv.exists():
        print(f"\n[FOLD {k}] OOF CSV found — skipping (already complete).")
        fold_summaries.append(pd.read_csv(oof_csv))
        continue

    fold_start = time.time()
    print(f"\n{'='*60}")
    print(f"FOLD {k} / {N_FOLDS - 1}")
    print(f"{'='*60}")

    fold_train_csv = split["train_csv"]
    fold_val_csv   = split["val_csv"]
    fold_test_csv  = split["test_csv"]

    # Checkpoint directory for this fold
    fold_ckpt_dir = OUT_CKPTS / f"fold{k}" / "finetune"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume detection: returns (next_epoch, ckpt_path_or_None)
    resume_from, resume_ckpt_path = _detect_resume_epoch(fold_ckpt_dir)
    if resume_from > 0:
        print(f"[FOLD {k}] Resuming from epoch {resume_from} (ckpt: {resume_ckpt_path})")
    else:
        print(f"[FOLD {k}] Starting fresh from epoch 0.")

    # Per-epoch callback: prune checkpoints + memory flush + progress log
    _best_auc_tracker = {"auc": 0.0, "epoch": -1}

    def _epoch_callback(epoch: int, train_loss: float, val_auc: float) -> None:
        if val_auc > _best_auc_tracker["auc"]:
            _best_auc_tracker["auc"]   = val_auc
            _best_auc_tracker["epoch"] = epoch
        _prune_old_checkpoints(fold_ckpt_dir, keep=3)
        gc.collect()
        torch.cuda.empty_cache()
        sys.stdout.flush()
        print(
            f"  [Epoch {epoch:03d}] loss={train_loss:.4f}  val_auc={val_auc:.4f}  "
            f"best={_best_auc_tracker['auc']:.4f}@{_best_auc_tracker['epoch']}",
            flush=True,
        )

    # ── Fine-tune this fold ────────────────────────────────────────────────
    ft.train(
        config_path         = CONFIG_PATH,
        device_str          = DEVICE,
        max_batches         = 0,
        processed_root      = str(PROCESSED_ROOT),
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
        resume_checkpoint   = resume_ckpt_path,   # load weights from latest epoch_*.pt
        per_epoch_callback  = _epoch_callback,
    )

    best_ckpt = fold_ckpt_dir / "best_finetune.pt"
    print(
        f"[FOLD {k}] Training done. Best epoch: {_best_auc_tracker['epoch']} "
        f"(AUC={_best_auc_tracker['auc']:.4f}). Elapsed: {(time.time()-fold_start)/60:.1f} min"
    )

    # Persist to ./outputs/ immediately (survives compute node deallocation)
    _out_fold_ckpt = PERSISTENT_OUT / "checkpoints" / f"fold{k}"
    _out_fold_ckpt.mkdir(parents=True, exist_ok=True)
    if best_ckpt.exists():
        _shutil.copy(str(best_ckpt), str(_out_fold_ckpt / "best_finetune.pt"))
        print(f"[FOLD {k}] Checkpoint persisted -> ./outputs/checkpoints/fold{k}/")
    else:
        print(f"[FOLD {k}] WARNING: best_finetune.pt not found — fold evaluation may fail.")

    if not best_ckpt.exists():
        print(f"[FOLD {k}] best_finetune.pt not found — skipping evaluation.")
        continue

    # ── Load best model for feature extraction ────────────────────────────
    try:
        model = _load_best_model(str(best_ckpt), DEVICE)
        print(f"[FOLD {k}] Loaded best model from {best_ckpt.name}")
    except Exception as exc:
        print(f"[FOLD {k}] ERROR loading model: {exc} — skipping evaluation.")
        continue

    # ── AT sweep on validation set (model object, correct API) ────────────
    try:
        best_at, best_val_auc, at_results_dict = at_sweep(
            model, fold_val_csv, PROCESSED_ROOT,
            train_csv=fold_train_csv, device=DEVICE,
            inference_stride=24, n_features=N_FEATURES,
            lr_C=0.1, use_pca=True,
        )
        print(f"[FOLD {k}] AT sweep -> best_at={best_at:.2f}  val_auc={best_val_auc:.4f}")
        at_df = pd.DataFrame(
            [{"fold": k, "at": at, "val_auc": v} for at, v in at_results_dict.items()]
        )
        at_df.to_csv(OUT_RESULTS / f"fold{k}_at_sweep.csv", index=False)
    except Exception as exc:
        print(f"[FOLD {k}] AT sweep failed: {exc} — defaulting to AT=0.40")
        best_at, best_val_auc = 0.40, 0.0

    # ── Feature extraction (model object, correct API) ────────────────────
    try:
        X_tr, y_tr, _      = extract_features_for_split(
            model, fold_train_csv, PROCESSED_ROOT, best_at, 24, DEVICE, N_FEATURES)
        X_vl, y_vl, _      = extract_features_for_split(
            model, fold_val_csv,   PROCESSED_ROOT, best_at, 24, DEVICE, N_FEATURES)
        X_te, y_te, te_ids = extract_features_for_split(
            model, fold_test_csv,  PROCESSED_ROOT, best_at, 24, DEVICE, N_FEATURES)
    except Exception as exc:
        print(f"[FOLD {k}] Feature extraction failed: {exc} — skipping.")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        continue

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── LR model: train on training set ONLY (val not reused after AT selection) ──
    # C=0.01: grid winner from v5 (grid_best_configs.csv rank 1)
    scaler, pca, lr_m = fit_lr_model(X_tr, y_tr, C=0.01, use_pca=True)

    # Predict on val (threshold selection) and test (evaluation)
    # Threshold selected on val — NOT on test (avoids leakage)
    val_scores  = predict_lr(X_vl, scaler, pca, lr_m)
    test_scores = predict_lr(X_te, scaler, pca, lr_m)

    thr, sens, spec = clinical_threshold(y_vl, val_scores, spec_constraint=SPEC_CONSTRAINT)
    test_auc = float(roc_auc_score(y_te, test_scores))
    y_pred_te = (test_scores >= thr).astype(int)

    print(
        f"[FOLD {k}] Test  AUC={test_auc:.4f}  Sens={sens:.3f}  "
        f"Spec={spec:.3f}  thr={thr:.4f}  AT={best_at:.2f}"
    )

    # G3 gate: log warning after fold 0 (no interactive input in Azure ML)
    if k == 0 and test_auc < 0.65:
        print(
            f"[G3 GATE] WARNING: fold0 test AUC={test_auc:.4f} < 0.65 — "
            f"check class weights, pretrain checkpoint, and stride settings."
        )

    # ── Save OOF CSV (matching notebook column format) ────────────────────
    oof_df = pd.DataFrame({
        "id":                te_ids,
        "y_true":            y_te.tolist(),
        "y_score":           test_scores.tolist(),
        "y_pred":            y_pred_te.tolist(),
        "best_at":           best_at,
        "threshold_primary": thr,
        "fold":              k,
    })
    oof_df.to_csv(oof_csv, index=False)
    fold_summaries.append(oof_df)

    # Per-fold summary (spec §7.k)
    _best_ep = _best_auc_tracker["epoch"]
    print(f"\n{'='*54}")
    print(f" FOLD {k} COMPLETE")
    print(f"   Test AUC:    {test_auc:.4f}")
    print(f"   Sensitivity: {sens:.4f}  (val-threshold, spec>={SPEC_CONSTRAINT})")
    print(f"   Specificity: {spec:.4f}")
    print(f"   Best AT:     {best_at:.2f}")
    print(f"   Threshold:   {thr:.4f}")
    print(f"   Best Epoch:  {_best_ep}  (val_auc={_best_auc_tracker['auc']:.4f})")
    print(f"{'='*54}")

# Restore original DataLoader after fold loop
_tdata.DataLoader = _OrigDataLoader

# ── 14. Global OOF AUC + bootstrap CI ─────────────────────────────────────
print(f"\n{'='*60}")
print("GLOBAL OOF RESULTS")
print(f"{'='*60}")

if not fold_summaries:
    print("[ERROR] No fold results available — exiting.")
    sys.exit(1)

all_oof = pd.concat(fold_summaries, ignore_index=True)
all_oof.to_csv(OUT_RESULTS / "global_oof_predictions.csv", index=False)

y_true  = all_oof["y_true"].values
y_score = all_oof["y_score"].values
global_auc          = float(roc_auc_score(y_true, y_score))
ci_lo, ci_hi        = bootstrap_auc_ci(y_true, y_score, n_bootstrap=N_BOOTSTRAP, seed=SEED)
global_thr, g_sens, g_spec = clinical_threshold(y_true, y_score, spec_constraint=SPEC_CONSTRAINT)

print(f"  Global OOF AUC : {global_auc:.4f}  (95% CI: {ci_lo:.4f}-{ci_hi:.4f})")
print(f"  Clinical thr   : {global_thr:.4f}  Sens={g_sens:.3f}  Spec={g_spec:.3f}")

# Per-fold AUC summary
per_fold = []
for k in range(N_FOLDS):
    oof_f_path = OUT_RESULTS / f"fold{k}_oof_scores.csv"
    if oof_f_path.exists():
        oof_f = pd.read_csv(oof_f_path)
        auc_k = float(roc_auc_score(oof_f["y_true"], oof_f["y_score"]))
        per_fold.append({"fold": k, "test_auc": auc_k})
per_fold_df = pd.DataFrame(per_fold)
per_fold_df.to_csv(OUT_RESULTS / "per_fold_summary.csv", index=False)
print(f"\n  Per-fold AUCs: {', '.join(f'{r.test_auc:.4f}' for _, r in per_fold_df.iterrows())}")
print(f"  Mean +/- Std  : {per_fold_df.test_auc.mean():.4f} +/- {per_fold_df.test_auc.std():.4f}")

# G4 quality gates (spec §11)
mean_auc = per_fold_df.test_auc.mean()
std_auc  = per_fold_df.test_auc.std()
print(f"[G4a] {'PASS' if mean_auc >= 0.70 else 'FAIL'} — mean fold AUC = {mean_auc:.4f} (threshold 0.70)")
print(f"[G4b] {'PASS' if std_auc  <  0.10 else 'FAIL'} — std  fold AUC = {std_auc:.4f}  (threshold 0.10)")

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
    "mean_fold_auc": mean_auc,
    "std_fold_auc":  std_auc,
    "total_min":     round((time.time() - job_start) / 60, 1),
}])
report.to_csv(OUT_RESULTS / "final_cv_report_v3.csv", index=False)
print(f"\n[DONE] final_cv_report_v3.csv saved.")
print(f"[DONE] Total runtime: {(time.time()-job_start)/60:.1f} min")

# Persist CSVs to ./outputs/ (survives compute node deallocation)
for _csv_src in [
    OUT_RESULTS / "final_cv_report_v3.csv",
    OUT_RESULTS / "global_oof_predictions.csv",
    OUT_RESULTS / "per_fold_summary.csv",
]:
    if _csv_src.exists():
        _shutil.copy(str(_csv_src), str(PERSISTENT_OUT / "results" / _csv_src.name))
print("[DONE] Results persisted to ./outputs/results/")

# ── 15. REPRO_TRACK — Canonical 441/56/55 split for direct comparison ──────
# Mirrors Cell REPRO_TRACK in notebooks/09_e2e_cv_v3.ipynb.
# Trains on the fixed canonical split (not cross-validated) to produce a
# test-set AUC comparable with prior reported results (benchmark AUC 0.826).
print(f"\n{'='*60}")
print("REPRO_TRACK — Canonical split (train=441 / val=56 / test=55)")
print(f"{'='*60}")
try:
    repro_ckpt_dir = OUT_CKPTS / "repro_track" / "finetune"
    repro_ckpt_dir.mkdir(parents=True, exist_ok=True)

    ft.train(
        config_path         = CONFIG_PATH,
        device_str          = DEVICE,
        max_batches         = 0,
        processed_root      = str(PROCESSED_ROOT),
        train_csv           = str(splits_dir / "train.csv"),
        val_csv             = str(splits_dir / "val.csv"),
        pretrain_checkpoint = SHARED_PRETRAIN_CKPT,
        checkpoint_dir      = str(repro_ckpt_dir),
        log_path            = str(OUT_LOGS / "repro_track_loss.csv"),
        quiet               = True,
        save_epoch_ckpts    = False,
        config_overrides    = CONFIG_A_OVERRIDES,
        val_every_n_epochs  = VAL_EVERY_N,
    )

    best_ckpt_r = repro_ckpt_dir / "best_finetune.pt"
    if not best_ckpt_r.exists():
        raise FileNotFoundError("REPRO_TRACK: best_finetune.pt not found after training")

    # Persist REPRO_TRACK checkpoint to ./outputs/
    _repro_out = PERSISTENT_OUT / "checkpoints" / "repro_track"
    _repro_out.mkdir(parents=True, exist_ok=True)
    _shutil.copy(str(best_ckpt_r), str(_repro_out / "best_finetune.pt"))
    print("[REPRO_TRACK] Checkpoint persisted -> ./outputs/checkpoints/repro_track/")

    model_r = _load_best_model(str(best_ckpt_r), DEVICE)

    best_at_r, _, _ = at_sweep(
        model_r, splits_dir / "val.csv", PROCESSED_ROOT,
        train_csv=splits_dir / "train.csv", device=DEVICE,
        inference_stride=24, n_features=N_FEATURES,
        lr_C=0.1, use_pca=True,
    )

    X_tr_r, y_tr_r, _ = extract_features_for_split(
        model_r, splits_dir / "train.csv", PROCESSED_ROOT, best_at_r, 24, DEVICE, N_FEATURES)
    X_vl_r, y_vl_r, _ = extract_features_for_split(
        model_r, splits_dir / "val.csv",   PROCESSED_ROOT, best_at_r, 24, DEVICE, N_FEATURES)
    X_te_r, y_te_r, _ = extract_features_for_split(
        model_r, splits_dir / "test.csv",  PROCESSED_ROOT, best_at_r, 24, DEVICE, N_FEATURES)

    del model_r
    gc.collect()
    torch.cuda.empty_cache()

    # LR on training set only (same policy as main pipeline — no val leakage)
    sc_r, pca_r, lr_r = fit_lr_model(X_tr_r, y_tr_r, C=0.01, use_pca=True)

    val_sc_r  = predict_lr(X_vl_r, sc_r, pca_r, lr_r)
    test_sc_r = predict_lr(X_te_r, sc_r, pca_r, lr_r)

    repro_auc          = float(roc_auc_score(y_te_r, test_sc_r))
    r_lo, r_hi         = bootstrap_auc_ci(y_te_r, test_sc_r, N_BOOTSTRAP, SEED)
    r_thr, r_sens, r_spec = clinical_threshold(y_vl_r, val_sc_r, spec_constraint=SPEC_CONSTRAINT)

    print(f"\n[REPRO_TRACK] Canonical test (n=55):")
    print(f"   AUC:         {repro_auc:.4f}  [{r_lo:.4f}, {r_hi:.4f}]  95% CI")
    print(f"   Sensitivity: {r_sens:.4f}  Specificity: {r_spec:.4f}")
    print(f"   Threshold:   {r_thr:.4f}   Best AT:     {best_at_r:.2f}")
    print(f"   Prior best:  AUC=0.839  (AT=0.40, Youden threshold)")

    pd.DataFrame([{
        "auc": repro_auc, "ci_lo": r_lo, "ci_hi": r_hi,
        "sensitivity": r_sens, "specificity": r_spec,
        "threshold": r_thr, "best_at": best_at_r, "prior_auc": 0.839,
    }]).to_csv(OUT_RESULTS / "repro_comparison_v3.csv", index=False)
    print("[REPRO_TRACK] Saved repro_comparison_v3.csv")

    # Append RepRo row to comparison table (matches notebook spec §12.5)
    cmp_path = OUT_RESULTS / "comparison_table_v3.csv"
    comparison = pd.DataFrame([
        {"method": "Paper (benchmark)",              "n": 55,  "auc": 0.826, "ci_lo": None,  "ci_hi": None},
        {"method": "Baseline Stage2 LR (test-55)",   "n": 55,  "auc": 0.812, "ci_lo": 0.630, "ci_hi": 0.953},
        {"method": "Best post-hoc (AT=0.40, Youden)","n": 55,  "auc": 0.839, "ci_lo": None,  "ci_hi": None},
        {"method": "RepRo Track v3 (441/56/55)",      "n": 55,  "auc": repro_auc, "ci_lo": r_lo, "ci_hi": r_hi},
        {"method": "E2E CV v3 (552, 5-fold OOF)",    "n": 552, "auc": global_auc, "ci_lo": ci_lo, "ci_hi": ci_hi},
    ])
    comparison.to_csv(cmp_path, index=False)
    print("[REPRO_TRACK] Saved comparison_table_v3.csv")
    print(comparison[["method", "n", "auc", "ci_lo", "ci_hi"]].to_string(index=False))

    # Persist REPRO_TRACK CSVs to ./outputs/
    for _csv_src in [
        OUT_RESULTS / "repro_comparison_v3.csv",
        OUT_RESULTS / "comparison_table_v3.csv",
    ]:
        if _csv_src.exists():
            _shutil.copy(str(_csv_src), str(PERSISTENT_OUT / "results" / _csv_src.name))
    print("[REPRO_TRACK] CSVs persisted to ./outputs/results/")

except Exception as _repro_exc:
    import traceback
    print(f"[REPRO_TRACK] Failed: {_repro_exc}")
    traceback.print_exc()
    print("[REPRO_TRACK] Skipping — main 5-fold CV results are unaffected.")

# ── 16. POST-TRAINING GRID SEARCH ──────────────────────────────────────────
print(f"\n{'='*60}")
print("POST-TRAINING GRID SEARCH — AT × LR_C × n_features × threshold")
print(f"{'='*60}")

AT_GRID     = [0.30, 0.35, 0.40, 0.45, 0.50]
LR_C_GRID   = [0.01, 0.1, 1.0]
NFEAT_GRID  = [4, 12]
THRESH_GRID = ["clinical", "youden"]
grid_rows   = []

from sklearn.metrics import roc_curve as _roc_curve  # imported once, outside all loops

for k, split in enumerate(cv_splits):
    fold_ckpt_path = PERSISTENT_OUT / "checkpoints" / f"fold{k}" / "best_finetune.pt"
    if not fold_ckpt_path.exists():
        print(f"[GRID] Fold {k}: checkpoint not found — skipping.")
        continue

    print(f"\n[GRID] Fold {k}: loading checkpoint for grid search...")
    try:
        model_g = _load_best_model(str(fold_ckpt_path), DEVICE)
    except Exception as _e:
        print(f"[GRID] Fold {k}: load error: {_e}")
        continue

    # Pre-extract 12 features at each AT (reuse across all LR_C/nfeat/thresh combos)
    feat_cache = {}
    for at_val in AT_GRID:
        try:
            _Xtr12, _ytr, _ = extract_features_for_split(
                model_g, split["train_csv"], PROCESSED_ROOT, at_val, 24, DEVICE, 12)
            _Xvl12, _yvl, _ = extract_features_for_split(
                model_g, split["val_csv"],   PROCESSED_ROOT, at_val, 24, DEVICE, 12)
            _Xte12, _yte, _ids = extract_features_for_split(
                model_g, split["test_csv"],  PROCESSED_ROOT, at_val, 24, DEVICE, 12)
            feat_cache[at_val] = (_Xtr12, _ytr, _Xvl12, _yvl, _Xte12, _yte, _ids)
            print(f"[GRID]   Fold {k} AT={at_val:.2f}: features extracted "
                  f"(tr={len(_ytr)}, vl={len(_yvl)}, te={len(_yte)})")
        except Exception as _e:
            print(f"[GRID]   Fold {k} AT={at_val:.2f}: extraction failed: {_e}")

    del model_g
    gc.collect()
    torch.cuda.empty_cache()

    # Full grid sweep over (AT, n_features, LR_C, threshold_method)
    for at_val in AT_GRID:
        if at_val not in feat_cache:
            continue
        Xtr12, ytr, Xvl12, yvl, Xte12, yte, te_ids_g = feat_cache[at_val]
        for n_feat in NFEAT_GRID:
            Xtr = Xtr12[:, :n_feat]
            Xvl = Xvl12[:, :n_feat]
            Xte = Xte12[:, :n_feat]
            for lr_c in LR_C_GRID:
                # LR fit on training set only (no val leakage); threshold selected on val
                try:
                    sc_g, pca_g, lr_g = fit_lr_model(Xtr, ytr, C=lr_c, use_pca=(n_feat > 3))
                    vs_g = predict_lr(Xvl, sc_g, pca_g, lr_g)
                    ts_g = predict_lr(Xte, sc_g, pca_g, lr_g)
                    test_auc_g = float(roc_auc_score(yte, ts_g))
                except Exception as _lr_exc:
                    print(f"[GRID]   fold={k} at={at_val:.2f} n={n_feat} C={lr_c}: "
                          f"LR failed — {_lr_exc}")
                    continue
                # Per-threshold evaluation — independent try/except so one failure
                # doesn't suppress the other threshold method's result
                for thr_method in THRESH_GRID:
                    try:
                        if thr_method == "clinical":
                            thr_g, sens_g, spec_g = clinical_threshold(
                                yvl, vs_g, SPEC_CONSTRAINT)
                        else:  # youden
                            fpr_y, tpr_y, thr_cands = _roc_curve(yvl, vs_g)
                            best_idx = int(np.argmax(tpr_y - fpr_y))
                            thr_g  = float(thr_cands[best_idx])
                            sens_g = float(tpr_y[best_idx])
                            spec_g = float(1 - fpr_y[best_idx])
                        ypred_g = (ts_g >= thr_g).astype(int)
                        test_sens_g = float(ypred_g[yte == 1].mean()) if yte.sum() > 0 else 0.0
                        test_spec_g = float((1 - ypred_g[yte == 0]).mean()) if (1 - yte).sum() > 0 else 0.0
                        grid_rows.append({
                            "fold":             k,
                            "at":               at_val,
                            "n_features":       n_feat,
                            "lr_C":             lr_c,
                            "threshold_method": thr_method,
                            "val_threshold":    round(thr_g, 4),
                            "test_auc":         round(test_auc_g, 4),
                            "test_sens":        round(test_sens_g, 3),
                            "test_spec":        round(test_spec_g, 3),
                        })
                    except Exception as _thr_exc:
                        print(f"[GRID]   fold={k} at={at_val:.2f} n={n_feat} C={lr_c} "
                              f"thr={thr_method}: skipped — {_thr_exc}")

    n_fold_rows = len([r for r in grid_rows if r["fold"] == k])
    print(f"[GRID] Fold {k}: {n_fold_rows} combos done.")

# Grid search summary
if grid_rows:
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUT_RESULTS / "grid_search_results.csv", index=False)
    _shutil.copy(str(OUT_RESULTS / "grid_search_results.csv"),
                 str(PERSISTENT_OUT / "results" / "grid_search_results.csv"))

    group_cols = ["at", "n_features", "lr_C", "threshold_method"]
    best_combos = (
        grid_df.groupby(group_cols)["test_auc"]
        .mean()
        .reset_index()
        .sort_values("test_auc", ascending=False)
    )
    best_combos.to_csv(OUT_RESULTS / "grid_best_configs.csv", index=False)
    _shutil.copy(str(OUT_RESULTS / "grid_best_configs.csv"),
                 str(PERSISTENT_OUT / "results" / "grid_best_configs.csv"))

    print(f"\n[GRID] {len(grid_df)} total evaluations across {len(cv_splits)} folds.")
    print("\n[GRID] TOP 10 CONFIGURATIONS (mean test AUC across folds):")
    print(best_combos.head(10).to_string(index=False))
    best_row = best_combos.iloc[0]
    print(
        f"\n[GRID] WINNER: AT={best_row['at']:.2f} | n_feat={int(best_row['n_features'])} | "
        f"C={best_row['lr_C']} | thr={best_row['threshold_method']} | "
        f"mean_AUC={best_row['test_auc']:.4f}"
    )
else:
    print("[GRID] No results — fold checkpoints were missing.")

print(f"\n[DONE] All done. Total runtime: {(time.time()-job_start)/60:.1f} min")
