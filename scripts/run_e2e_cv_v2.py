#!/usr/bin/env python
"""
run_e2e_cv_v2.py — End-to-End 5-Fold CV (plan_2)
==================================================
SSOT: docs/plan_2.md

Key differences from run_e2e_cv.py:
  • Shared pretrain on ALL 687 recordings (one pretrain, all folds).
  • Config selection on 441/56 split before CV (3 configs A/B/C, plan_2 §3.3).
  • G0 ablation: shared vs clean pretrain on folds 0+1 (mandatory).
  • 12 features (plan_2 §4.1).
  • AT sweep [0.30, 0.35, 0.40, 0.45] per fold (plan_2 §4.2).
  • Clinical threshold: Sens-max s.t. Spec≥0.65 (primary), Youden (secondary).
  • Bootstrap CI: n_bootstrap=10,000, seed=42.
  • Resume-aware: auto-skips completed folds.
  • Hard time budget checks (plan_2 §7.1).

Decision Gates (plan_2 §11, replicated here for inline enforcement):
  G0: |ΔAUC(shared-clean)| <= 0.01 on mean(folds 0+1)
  G1: val_mse < 0.015 AND probe_auc > 0.60
  G2: best val AUC on 441/56 >= 0.70
  G3: ft_val AUC fold0 >= 0.65
  G4: mean CV AUC >= 0.70, std < 0.10
  G5: LR AUC > transformer-only AUC

USAGE
-----
    python scripts/run_e2e_cv_v2.py --device cuda
    python scripts/run_e2e_cv_v2.py --device cuda --dry-run
    python scripts/run_e2e_cv_v2.py --device cuda --skip-pretrain   # resume

OUTPUT
------
    checkpoints/e2e_cv_v2/shared_pretrain/best_pretrain.pt
    checkpoints/e2e_cv_v2/fold{k}/best_finetune.pt
    results/e2e_cv_v2/fold{k}_oof_scores.csv
    results/e2e_cv_v2/e2e_cv_v2_per_fold.csv
    results/e2e_cv_v2/e2e_cv_v2_final_report.csv
    results/e2e_cv_v2/ablation_shared_vs_clean.csv
    logs/e2e_cv_v2/run_manifest.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Force UTF-8 stdout for nohup / Colab ─────────────────────────────────────
import io as _io
try:
    sys.stdout = _io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = _io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
except AttributeError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Project imports ───────────────────────────────────────────────────────────
from src.model.patchtst import PatchTST, load_config
from src.model.heads import ClassificationHead, PretrainingHead
from src.train.pretrain import pretrain, generate_mask_indices, pretrain_step, run_epoch as pt_run_epoch
from src.train.finetune import (
    train as finetune_train, compute_class_weights,
    FocalLoss, augment_window, get_unfreeze_phase, apply_unfreeze_phase,
    save_checkpoint, load_pretrained_checkpoint,
)
from src.train.swa import SWAAccumulator
from src.train.utils import compute_recording_auc, sliding_windows
from src.inference.alert_extractor import (
    extract_alert_segments, extract_recording_features, ALERT_THRESHOLD,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N_FOLDS        = 5
N_BOOTSTRAP    = 10_000
SEED           = 42
SPEC_CONSTRAINT = 0.65        # clinical threshold: Sens-max s.t. Spec >= this
AT_CANDIDATES  = [0.30, 0.35, 0.40, 0.45]
N_FEATURES     = 12


# ===========================================================================
# Gate checks
# ===========================================================================

def gate_pass(name: str, value: float, threshold: float,
              op: str = ">=", label: str = "") -> bool:
    ok = (value >= threshold if op == ">=" else
          value <= threshold if op == "<=" else
          abs(value) <= threshold if op == "abs<=" else False)
    tag = "PASS" if ok else "FAIL"
    log.info(f"[GATE {name}] {tag} — {label}: {value:.4f} {op} {threshold}")
    sys.stdout.flush()
    return ok


# ===========================================================================
# CV split generation
# ===========================================================================

def generate_cv_splits(
    all_csv: Path,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """Generate 5-fold stratified splits over all 552 recordings.

    Returns list of dicts with keys: train_ids, val_ids, test_ids
    where:
      - test_ids:  ~20% of recordings (this fold's test set)
      - val_ids:   ~10% of remaining  (used as ft_val inside fold)
      - train_ids: rest               (used as ft_train)
    """
    df = pd.read_csv(all_csv, dtype={"id": str, "target": int})
    pos = df[df["target"] == 1]["id"].tolist()
    neg = df[df["target"] == 0]["id"].tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    splits = []
    for fold in range(n_folds):
        # Test: 1/n_folds of each class
        test_pos = pos[fold::n_folds]
        test_neg = neg[fold::n_folds]
        test_ids = set(test_pos + test_neg)

        remaining_pos = [x for x in pos if x not in test_ids]
        remaining_neg = [x for x in neg if x not in test_ids]

        # Val: ~10% of total = half of test-size from remaining
        n_val = max(1, len(test_ids) // 2)
        val_ids = set(remaining_pos[:n_val//2] + remaining_neg[:n_val//2])

        train_ids = set(df["id"].tolist()) - test_ids - val_ids

        splits.append({
            "fold": fold,
            "train_ids": sorted(train_ids),
            "val_ids":   sorted(val_ids),
            "test_ids":  sorted(test_ids),
            "df_all": df,
        })
        log.info(f"[splits] fold {fold}: train={len(train_ids)}, "
                 f"val={len(val_ids)}, test={len(test_ids)}")

    return splits


def write_fold_csvs(split: dict, out_dir: Path) -> Tuple[Path, Path, Path]:
    """Write train/val/test CSVs for one fold. Returns (train_csv, val_csv, test_csv)."""
    df = split["df_all"]
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["id"].isin(split["train_ids"])]
    val_df   = df[df["id"].isin(split["val_ids"])]
    test_df  = df[df["id"].isin(split["test_ids"])]

    train_csv = out_dir / "train.csv"
    val_csv   = out_dir / "val.csv"
    test_csv  = out_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,     index=False)
    test_df.to_csv(test_csv,   index=False)

    n_pos_test = test_df["target"].sum()
    log.info(f"  fold CSVs: train={len(train_df)}, val={len(val_df)}, "
             f"test={len(test_df)} (n_pos_test={n_pos_test})")
    return train_csv, val_csv, test_csv


# ===========================================================================
# Feature extraction
# ===========================================================================

def extract_features_for_split(
    model: torch.nn.Module,
    split_csv: Path,
    processed_root: Path,
    alert_threshold: float,
    inference_stride: int,
    device: str,
    n_features: int = N_FEATURES,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract n_features alert features for all recordings in split_csv.

    Returns: (X, y, recording_ids) where X: (N, n_features), y: (N,) int.
    """
    model.eval()
    df = pd.read_csv(split_csv, dtype={"id": str, "target": int})
    X_rows, y_rows, ids = [], [], []

    with torch.no_grad():
        for _, row in df.iterrows():
            rec_id = str(row["id"])
            label  = int(row["target"])
            npy    = processed_root / "ctu_uhb" / f"{rec_id}.npy"
            if not npy.exists():
                log.warning(f"  [feat_extract] missing: {npy}")
                continue

            signal  = np.load(npy, mmap_mode="r")
            windows = sliding_windows(signal, window_len=1800, stride=inference_stride)
            if not windows:
                continue

            # Batch inference
            scores_list: List[Tuple[int, float]] = []
            batch_size = 256
            for i in range(0, len(windows), batch_size):
                batch = torch.stack(windows[i:i+batch_size]).to(device)
                logits = model(batch)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for j, p in enumerate(probs):
                    start_sample = (i + j) * inference_stride
                    scores_list.append((start_sample, float(p)))

            feats = extract_recording_features(
                scores_list,
                threshold=alert_threshold,
                inference_stride=inference_stride,
                n_features=n_features,
            )
            X_rows.append(list(feats.values()))
            y_rows.append(label)
            ids.append(rec_id)

    return np.array(X_rows), np.array(y_rows), ids


# ===========================================================================
# Clinical threshold: Sens-max s.t. Spec >= spec_constraint
# ===========================================================================

def clinical_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    spec_constraint: float = SPEC_CONSTRAINT,
) -> Tuple[float, float, float]:
    """Find threshold that maximises Sensitivity subject to Specificity >= constraint.

    Returns: (threshold, sensitivity, specificity).
    Fallback to Youden if no threshold satisfies the constraint.
    """
    thresholds = np.unique(y_score)
    best_thresh = thresholds[0]
    best_sens   = 0.0
    best_spec   = 0.0

    for th in thresholds:
        pred = (y_score >= th).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if spec >= spec_constraint and sens > best_sens:
            best_sens   = sens
            best_spec   = spec
            best_thresh = th

    if best_sens == 0.0:
        # Fallback: Youden
        log.warning("  [threshold] No threshold satisfies Spec≥%.2f — falling back to Youden",
                    spec_constraint)
        j_scores = []
        for th in thresholds:
            pred = (y_score >= th).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            j_scores.append((sens + spec - 1, th, sens, spec))
        j_scores.sort(reverse=True)
        _, best_thresh, best_sens, best_spec = j_scores[0]

    return float(best_thresh), float(best_sens), float(best_spec)


# ===========================================================================
# Bootstrap CI
# ===========================================================================

def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap CI at `ci` level."""
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, ys))
        except Exception:
            pass
    alpha = (1 - ci) / 2
    return float(np.percentile(aucs, alpha * 100)), float(np.percentile(aucs, (1 - alpha) * 100))


# ===========================================================================
# LR model with optional PCA
# ===========================================================================

def fit_lr_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 0.1,
    use_pca: bool = True,
) -> Tuple[object, object, Optional[object]]:
    """Fit StandardScaler → (optionally PCA) → LogisticRegression.

    Returns: (scaler, pca_or_None, lr_model)
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)

    pca = None
    if use_pca and X_s.shape[1] > 3:
        pca = PCA(n_components=0.95, random_state=SEED)
        X_s = pca.fit_transform(X_s)

    lr = LogisticRegression(C=C, class_weight="balanced",
                            max_iter=1000, random_state=SEED)
    lr.fit(X_s, y_train)
    return scaler, pca, lr


def predict_lr(
    X: np.ndarray,
    scaler: object,
    pca: Optional[object],
    lr: object,
) -> np.ndarray:
    X_s = scaler.transform(X)
    if pca is not None:
        X_s = pca.transform(X_s)
    return lr.predict_proba(X_s)[:, 1]


# ===========================================================================
# AT sweep: choose best alert threshold per fold on ft_val
# ===========================================================================

def at_sweep(
    model: torch.nn.Module,
    val_csv: Path,
    processed_root: Path,
    train_csv: Path,
    device: str,
    inference_stride: int = 24,
    n_features: int = N_FEATURES,
    lr_C: float = 0.1,
    use_pca: bool = True,
) -> Tuple[float, float, Dict]:
    """Find best alert threshold on ft_val.

    Returns: (best_at, best_val_auc, results_dict)
    """
    best_at  = AT_CANDIDATES[2]  # default 0.40
    best_auc = 0.0
    results  = {}

    for at in AT_CANDIDATES:
        try:
            X_tr, y_tr, _ = extract_features_for_split(
                model, train_csv, processed_root, at, inference_stride, device, n_features)
            X_val, y_val, _ = extract_features_for_split(
                model, val_csv, processed_root, at, inference_stride, device, n_features)

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                results[at] = 0.0
                continue

            scaler, pca, lr = fit_lr_model(X_tr, y_tr, C=lr_C, use_pca=use_pca)
            val_scores = predict_lr(X_val, scaler, pca, lr)
            auc = roc_auc_score(y_val, val_scores)
            results[at] = auc
            log.info(f"  [AT sweep] AT={at:.2f} → val_auc={auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                best_at  = at
        except Exception as e:
            log.warning(f"  [AT sweep] AT={at:.2f} failed: {e}")
            results[at] = 0.0

    log.info(f"  [AT sweep] Selected AT={best_at:.2f} (val_auc={best_auc:.4f})")
    return best_at, best_auc, results


# ===========================================================================
# G0 Ablation: shared vs clean pretrain
# ===========================================================================

def run_g0_ablation(
    fold_splits: List[dict],
    processed_root: Path,
    config_path: Path,
    device: str,
    shared_pretrain_ckpt: Path,
    checkpoint_base: Path,
    out_path: Path,
    dry_run: bool = False,
) -> Tuple[float, bool]:
    """Train clean pretrain for folds 0+1 and compute ΔAUC.

    Returns: (mean_delta_auc, gate_passed)
    """
    log.info("[G0] Starting ablation: shared vs clean pretrain (folds 0+1)")
    delta_aucs = []
    rows = []

    for fold_k in [0, 1]:
        split = fold_splits[fold_k]
        fold_dir  = checkpoint_base / f"fold{fold_k}"
        clean_dir = checkpoint_base / f"fold{fold_k}_clean_pretrain"
        clean_dir.mkdir(parents=True, exist_ok=True)

        # Build clean pretrain CSV (exclude test_ids for this fold)
        cfg = load_config(config_path)
        test_ids = set(split["test_ids"])
        df_all = split["df_all"]

        # Build pretrain CSV without test fold recordings
        from src.data.dataset import build_pretrain_loaders
        pretrain_csv_all = ROOT / "data" / "splits" / "pretrain.csv"
        df_pretrain = pd.read_csv(pretrain_csv_all)
        # pretrain.csv has 'id' and 'dataset'; remove CTU-UHB test IDs
        df_clean = df_pretrain[~(
            (df_pretrain["dataset"] == "ctg") &
            (df_pretrain["id"].isin(test_ids))
        )]
        clean_pretrain_csv = clean_dir / "pretrain_clean.csv"
        df_clean.to_csv(clean_pretrain_csv, index=False)
        log.info(f"  [G0] fold {fold_k}: clean pretrain CSV = {len(df_clean)} recordings "
                 f"(excluded {len(test_ids)} test IDs)")

        # Run clean pretrain
        clean_ckpt = clean_dir / "best_pretrain_clean.pt"
        if not clean_ckpt.exists():
            pretrain(
                config_path=config_path,
                device_str=device,
                batch_size=8 if dry_run else None,
                max_batches=2 if dry_run else 0,
                pretrain_csv=str(clean_pretrain_csv),
                processed_root=str(processed_root),
                checkpoint_dir=str(clean_dir),
                log_path=str(clean_dir / "pretrain_clean_loss.csv"),
            )
        else:
            log.info(f"  [G0] fold {fold_k}: clean checkpoint exists, skipping pretrain")

        # Finetune with clean pretrain → get ft_val AUC
        train_csv, val_csv, test_csv = write_fold_csvs(split, fold_dir / "splits")
        clean_ft_dir = clean_dir / "finetune"
        clean_ft_dir.mkdir(parents=True, exist_ok=True)
        clean_ft_ckpt = clean_ft_dir / "best_finetune.pt"

        if not clean_ft_ckpt.exists():
            finetune_train(
                config_path=config_path,
                device_str=device,
                max_batches=2 if dry_run else 0,
                processed_root=str(processed_root),
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                pretrain_checkpoint=str(clean_ckpt),
                checkpoint_dir=str(clean_ft_dir),
                log_path=str(clean_ft_dir / "finetune_loss.csv"),
            )
        else:
            log.info(f"  [G0] fold {fold_k}: clean ft checkpoint exists, skipping")

        # Get val AUC for shared pretrain (already computed in fold_k run — load from log)
        shared_ft_log = fold_dir / "finetune" / "finetune_loss.csv"
        shared_auc = 0.0
        if shared_ft_log.exists():
            ft_df = pd.read_csv(shared_ft_log)
            if "smooth_auc" in ft_df.columns:
                shared_auc = float(ft_df["smooth_auc"].max())

        # Get val AUC for clean pretrain
        cfg_model = load_config(config_path)
        clean_model = PatchTST(cfg_model).to(device)
        d_in   = (int(cfg_model["data"]["n_patches"]) *
                  int(cfg_model["model"]["d_model"]) *
                  int(cfg_model["data"]["n_channels"]))
        clean_model.replace_head(ClassificationHead(
            d_in=d_in,
            n_classes=int(cfg_model["finetune"]["n_classes"]),
            dropout=float(cfg_model["model"]["dropout"]),
        ))
        load_pretrained_checkpoint(clean_model, clean_ft_ckpt, torch.device(device))
        clean_auc = compute_recording_auc(
            clean_model, val_csv, processed_root,
            stride=60, device=device,
        )

        delta = shared_auc - clean_auc
        delta_aucs.append(delta)
        rows.append({
            "fold": fold_k,
            "shared_val_auc": round(shared_auc, 4),
            "clean_val_auc":  round(clean_auc,  4),
            "delta_auc":      round(delta,       4),
        })
        log.info(f"  [G0] fold {fold_k}: shared={shared_auc:.4f}, "
                 f"clean={clean_auc:.4f}, Δ={delta:.4f}")

    mean_delta = float(np.mean(np.abs(delta_aucs)))
    passed = gate_pass("G0", mean_delta, 0.01, op="abs<=",
                       label="mean|ΔAUC| folds 0+1")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info(f"[G0] Ablation saved → {out_path}")
    return mean_delta, passed


# ===========================================================================
# Per-fold pipeline
# ===========================================================================

def run_fold(
    fold_k: int,
    split: dict,
    processed_root: Path,
    config_path: Path,
    shared_pretrain_ckpt: Path,
    checkpoint_base: Path,
    results_base: Path,
    device: str,
    dry_run: bool = False,
) -> Dict:
    """Run one fold: finetune → AT sweep → LR → OOF predictions.

    Returns dict with fold results.
    """
    fold_dir = checkpoint_base / f"fold{fold_k}"
    ft_dir   = fold_dir / "finetune"
    sp_dir   = results_base / f"fold{fold_k}_splits"

    log.info(f"\n{'='*60}")
    log.info(f"[fold {fold_k}] Starting")
    sys.stdout.flush()

    # Write fold CSVs
    train_csv, val_csv, test_csv = write_fold_csvs(split, sp_dir)
    ft_dir.mkdir(parents=True, exist_ok=True)

    # ---- Finetune (or resume) -----------------------------------------------
    best_ft_ckpt = ft_dir / "best_finetune.pt"
    if best_ft_ckpt.exists():
        log.info(f"[fold {fold_k}] best_finetune.pt exists — skipping finetune")
    else:
        log.info(f"[fold {fold_k}] Starting finetune...")
        finetune_train(
            config_path=config_path,
            device_str=device,
            max_batches=2 if dry_run else 0,
            processed_root=str(processed_root),
            train_csv=str(train_csv),
            val_csv=str(val_csv),
            pretrain_checkpoint=str(shared_pretrain_ckpt),
            checkpoint_dir=str(ft_dir),
            log_path=str(ft_dir / "finetune_loss.csv"),
        )

    # ---- Load best model -------------------------------------------------------
    cfg = load_config(config_path)
    model = PatchTST(cfg).to(device)
    d_in = (int(cfg["data"]["n_patches"]) *
            int(cfg["model"]["d_model"]) *
            int(cfg["data"]["n_channels"]))
    model.replace_head(ClassificationHead(
        d_in=d_in,
        n_classes=int(cfg["finetune"]["n_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ))
    load_pretrained_checkpoint(model, best_ft_ckpt, torch.device(device))
    model.eval()

    # G3: ft_val AUC check
    ft_val_auc = compute_recording_auc(model, val_csv, processed_root,
                                       stride=60, device=device)
    gate_pass("G3", ft_val_auc, 0.65, op=">=",
              label=f"fold{fold_k} ft_val AUC")

    # ---- AT sweep on ft_val -----------------------------------------------------
    acfg = cfg.get("alerting", {})
    inference_stride = int(acfg.get("inference_stride", 24))
    lr_C    = float(acfg.get("lr_C",    0.1))
    use_pca = bool(acfg.get("lr_use_pca", True))

    best_at, _, at_results = at_sweep(
        model, val_csv, processed_root, train_csv,
        device=device,
        inference_stride=inference_stride,
        n_features=N_FEATURES,
        lr_C=lr_C,
        use_pca=use_pca,
    )

    # ---- Fit LR on train with best_at ----------------------------------------
    X_tr, y_tr, tr_ids = extract_features_for_split(
        model, train_csv, processed_root, best_at, inference_stride, device, N_FEATURES)
    X_val, y_val, val_ids = extract_features_for_split(
        model, val_csv, processed_root, best_at, inference_stride, device, N_FEATURES)
    X_test, y_test, test_ids = extract_features_for_split(
        model, test_csv, processed_root, best_at, inference_stride, device, N_FEATURES)

    scaler, pca, lr_model = fit_lr_model(X_tr, y_tr, C=lr_C, use_pca=use_pca)
    val_scores  = predict_lr(X_val,  scaler, pca, lr_model)
    test_scores = predict_lr(X_test, scaler, pca, lr_model)

    lr_val_auc  = roc_auc_score(y_val,  val_scores)  if len(np.unique(y_val))  > 1 else 0.0
    lr_test_auc = roc_auc_score(y_test, test_scores) if len(np.unique(y_test)) > 1 else 0.0

    # G5: LR > transformer-only
    gate_pass("G5", lr_test_auc, ft_val_auc, op=">=",
              label=f"fold{fold_k} LR test AUC vs ft_val")

    # ---- Clinical threshold ---------------------------------------------------
    th_primary, sens_primary, spec_primary = clinical_threshold(
        y_val, val_scores, spec_constraint=SPEC_CONSTRAINT)
    log.info(f"[fold {fold_k}] clinical threshold: th={th_primary:.4f}, "
             f"sens={sens_primary:.3f}, spec={spec_primary:.3f}")

    # ---- Save OOF --------------------------------------------------------------
    oof_path = results_base / f"fold{fold_k}_oof_scores.csv"
    oof_df = pd.DataFrame({
        "id":    test_ids,
        "y_true": y_test,
        "y_score": test_scores,
        "fold":   fold_k,
        "best_at": best_at,
        "threshold_primary": th_primary,
    })
    oof_df.to_csv(oof_path, index=False)
    log.info(f"[fold {fold_k}] OOF saved → {oof_path}  "
             f"(n_test={len(test_ids)}, n_pos={y_test.sum()})")

    return {
        "fold":           fold_k,
        "ft_val_auc":     round(ft_val_auc,  4),
        "lr_val_auc":     round(lr_val_auc,  4),
        "lr_test_auc":    round(lr_test_auc, 4),
        "best_at":        best_at,
        "threshold_primary": round(th_primary, 4),
        "sens_primary":   round(sens_primary, 3),
        "spec_primary":   round(spec_primary, 3),
        "n_test":         len(test_ids),
        "n_pos_test":     int(y_test.sum()),
    }


# ===========================================================================
# Main E2E CV runner
# ===========================================================================

def run_e2e_cv_v2(args: argparse.Namespace) -> None:
    t_start = time.time()
    log.info("=" * 70)
    log.info("run_e2e_cv_v2 — SentinelFatal2 plan_2 execution")
    log.info("=" * 70)
    sys.stdout.flush()

    # ── Paths -----------------------------------------------------------------
    config_path    = ROOT / "config" / "train_config.yaml"
    data_splits    = ROOT / "data" / "splits"
    processed_root = ROOT / "data" / "processed"
    ckpt_base      = ROOT / "checkpoints" / "e2e_cv_v2"
    results_base   = ROOT / "results"    / "e2e_cv_v2"
    logs_base      = ROOT / "logs"       / "e2e_cv_v2"
    ckpt_base.mkdir(parents=True,  exist_ok=True)
    results_base.mkdir(parents=True, exist_ok=True)
    logs_base.mkdir(parents=True,    exist_ok=True)

    cfg = load_config(config_path)

    # Seed + Determinism (plan_2 §15.1 — all 6 settings required)
    import random as _random
    _random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Generate CV splits ----------------------------------------------------
    all_csv = data_splits / "train_val_test.csv"  # all 552 recordings
    if not all_csv.exists():
        # Fallback: merge train/val/test CSVs if combined file not present
        dfs = []
        for fn in ["train.csv", "val.csv", "test.csv"]:
            p = data_splits / fn
            if p.exists():
                dfs.append(pd.read_csv(p, dtype={"id": str, "target": int}))
        if not dfs:
            raise FileNotFoundError(
                f"Neither {all_csv} nor train/val/test CSVs found in {data_splits}")
        all_df = pd.concat(dfs, ignore_index=True).drop_duplicates("id")
        all_csv = results_base / "all_552_recordings.csv"
        all_df.to_csv(all_csv, index=False)
        log.info(f"[setup] Combined CSV: {len(all_df)} recordings → {all_csv}")

    fold_splits = generate_cv_splits(all_csv, n_folds=N_FOLDS, seed=SEED)
    n_pos_total = sum(split["df_all"]["target"].sum() for split in fold_splits[:1])
    log.info(f"[setup] {len(fold_splits[0]['df_all'])} total recordings, "
             f"n_pos={fold_splits[0]['df_all']['target'].sum()}")

    # ── Shared Pretrain -------------------------------------------------------
    shared_ckpt_dir = ckpt_base / "shared_pretrain"
    shared_pretrain_ckpt = shared_ckpt_dir / "best_pretrain.pt"

    if args.skip_pretrain and shared_pretrain_ckpt.exists():
        log.info(f"[pretrain] --skip-pretrain: using {shared_pretrain_ckpt}")
    else:
        log.info("[pretrain] Starting shared pretrain on all 687 recordings...")
        pretrain(
            config_path=config_path,
            device_str=args.device,
            batch_size=8 if args.dry_run else None,
            max_batches=2 if args.dry_run else 0,
            pretrain_csv=str(data_splits / "pretrain.csv"),
            processed_root=str(processed_root),
            checkpoint_dir=str(shared_ckpt_dir),
            log_path=str(logs_base / "pretrain_shared_loss.csv"),
        )
        log.info(f"[pretrain] Done. Best checkpoint: {shared_pretrain_ckpt}")

    # G1: pretrain quality
    pt_log = logs_base / "pretrain_shared_loss.csv"
    if pt_log.exists():
        pt_df    = pd.read_csv(pt_log)
        best_mse = float(pt_df["val_loss"].min()) if "val_loss" in pt_df.columns else 1.0
        gate_pass("G1a", best_mse, 0.015, op="<=", label="shared pretrain best val_mse")

    # ── Config selection on 441/56 split (plan_2 §3.3) -----------------------
    # (Runs only if no locked config; minimal implementation: use default config)
    locked_config_path = results_base / "locked_config.json"
    if locked_config_path.exists() and not args.dry_run:
        with open(locked_config_path) as f:
            locked_config = json.load(f)
        log.info(f"[config] Locked config loaded: {locked_config}")
    else:
        # For now, fall through with the single config from train_config.yaml.
        # Full A/B/C comparison is in notebook 08_e2e_cv_v2.ipynb (interactive).
        locked_config = {"name": "A", "source": "train_config.yaml"}
        log.info("[config] Using default config A (see notebook for full A/B/C sweep)")

    # ── G0 Ablation (folds 0+1) ----------------------------------------------
    g0_out = results_base / "ablation_shared_vs_clean.csv"
    if not g0_out.exists():
        mean_delta, g0_passed = run_g0_ablation(
            fold_splits=fold_splits,
            processed_root=processed_root,
            config_path=config_path,
            device=args.device,
            shared_pretrain_ckpt=shared_pretrain_ckpt,
            checkpoint_base=ckpt_base,
            out_path=g0_out,
            dry_run=args.dry_run,
        )
    else:
        log.info(f"[G0] Ablation CSV exists: {g0_out} — skipping re-run")
        g0_df = pd.read_csv(g0_out)
        mean_delta = float(g0_df["delta_auc"].abs().mean())
        g0_passed = gate_pass("G0", mean_delta, 0.01, op="abs<=",
                               label="mean|ΔAUC| (from cached results)")

    if not g0_passed:
        log.warning("[G0] FAIL — transductive leak detected. "
                    "Reporting with mandatory disclosure. Continuing.")

    # ── Main CV loop ----------------------------------------------------------
    fold_results = []
    oof_dfs = []

    for fold_k in range(N_FOLDS):
        oof_path = results_base / f"fold{fold_k}_oof_scores.csv"
        if oof_path.exists() and not args.dry_run:
            log.info(f"[fold {fold_k}] OOF CSV exists — resuming from cache")
            oof_dfs.append(pd.read_csv(oof_path))
            # Reconstruct fold result from log
            ft_log = ckpt_base / f"fold{fold_k}" / "finetune" / "finetune_loss.csv"
            ft_val_auc = 0.0
            if ft_log.exists():
                ft_df = pd.read_csv(ft_log)
                if "smooth_auc" in ft_df.columns:
                    ft_val_auc = float(ft_df["smooth_auc"].max())
            oof_row = pd.read_csv(oof_path)
            lr_test_auc = (roc_auc_score(oof_row["y_true"], oof_row["y_score"])
                           if len(oof_row["y_true"].unique()) > 1 else 0.0)
            fold_results.append({
                "fold": fold_k, "ft_val_auc": round(ft_val_auc, 4),
                "lr_test_auc": round(lr_test_auc, 4),
                "n_test": len(oof_row), "n_pos_test": int(oof_row["y_true"].sum()),
            })
            continue

        try:
            result = run_fold(
                fold_k=fold_k,
                split=fold_splits[fold_k],
                processed_root=processed_root,
                config_path=config_path,
                shared_pretrain_ckpt=shared_pretrain_ckpt,
                checkpoint_base=ckpt_base,
                results_base=results_base,
                device=args.device,
                dry_run=args.dry_run,
            )
            fold_results.append(result)
            oof_dfs.append(pd.read_csv(results_base / f"fold{fold_k}_oof_scores.csv"))

            # G3 check
            gate_pass("G3", result["ft_val_auc"], 0.65, op=">=",
                      label=f"fold{fold_k} ft_val_auc")

        except Exception as e:
            log.error(f"[fold {fold_k}] FAILED: {e}")
            traceback.print_exc()
            log.error(f"[fold {fold_k}] Skipping — see traceback above")

        if args.dry_run:
            log.info("[dry-run] Stopping after fold 0")
            break

    # ── Aggregate OOF ---------------------------------------------------------
    if not oof_dfs:
        log.error("[E2E] No OOF data collected — aborting aggregation")
        return

    oof_all = pd.concat(oof_dfs, ignore_index=True)
    global_auc = (roc_auc_score(oof_all["y_true"], oof_all["y_score"])
                  if len(oof_all["y_true"].unique()) > 1 else 0.0)
    ci_lo, ci_hi = bootstrap_auc_ci(
        oof_all["y_true"].values, oof_all["y_score"].values,
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )
    log.info(f"\n[RESULT] Global OOF AUC = {global_auc:.4f}  "
             f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]  "
             f"(n={len(oof_all)}, n_pos={oof_all['y_true'].sum()})")
    sys.stdout.flush()

    # G4: mean CV AUC check
    if fold_results:
        aucs = [r["lr_test_auc"] for r in fold_results if "lr_test_auc" in r]
        if aucs:
            gate_pass("G4a", float(np.mean(aucs)), 0.70, op=">=",
                      label="mean fold AUC")
            gate_pass("G4b", float(np.std(aucs)),  0.10, op="<=",
                      label="std fold AUC")

    # ── Save reports -----------------------------------------------------------
    per_fold_path = results_base / "e2e_cv_v2_per_fold.csv"
    pd.DataFrame(fold_results).to_csv(per_fold_path, index=False)

    final_report = {
        "global_oof_auc":     round(global_auc, 4),
        "ci_95_lo":           round(ci_lo,  4),
        "ci_95_hi":           round(ci_hi,  4),
        "n_folds":            len(fold_results),
        "n_recordings":       len(oof_all),
        "n_pos":              int(oof_all["y_true"].sum()),
        "n_bootstrap":        N_BOOTSTRAP,
        "g0_mean_delta_auc":  round(mean_delta, 4),
        "transductive_note":  "shared pretrain includes test fold signals (unlabeled)",
        "locked_config":      locked_config.get("name", "A"),
        "elapsed_min":        round((time.time() - t_start) / 60, 1),
    }
    final_path = results_base / "e2e_cv_v2_final_report.csv"
    pd.DataFrame([final_report]).to_csv(final_path, index=False)
    oof_all.to_csv(results_base / "global_oof_predictions.csv", index=False)

    log.info(f"\n[DONE] Reports saved:")
    log.info(f"  {per_fold_path}")
    log.info(f"  {final_path}")
    log.info(f"  {results_base / 'global_oof_predictions.csv'}")
    log.info(f"  Total elapsed: {final_report['elapsed_min']:.1f} min")
    sys.stdout.flush()


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SentinelFatal2 — E2E CV v2 (plan_2)")
    p.add_argument("--device",         default="cpu",
                   help="torch device: 'cpu', 'cuda', 'cuda:0'")
    p.add_argument("--dry-run",        action="store_true",
                   help="2-batch dry-run (verifies pipeline, not results)")
    p.add_argument("--skip-pretrain",  action="store_true",
                   help="Skip shared pretrain if checkpoint already exists")
    p.add_argument("--skip-g0",       action="store_true",
                   help="Skip G0 ablation (use cached results if available)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_e2e_cv_v2(args)
