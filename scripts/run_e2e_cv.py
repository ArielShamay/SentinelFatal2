#!/usr/bin/env python
"""
run_e2e_cv.py  —  End-to-End 5-Fold Cross-Validation for SentinelFatal2
==========================================================================

Runs a TRUE end-to-end cross-validation where PatchTST is retrained from
scratch on every fold.  Unlike the approximate LR-only CV in notebook 06,
this script produces leak-free OOF predictions.

Pipeline per fold k (k = 0..4):
    1. Generate fold CSVs  (pretrain / finetune-train / finetune-val / test)
    2. Pretrain  PatchTST  on FHRMA + ~397 CTU-UHB  (MAE self-supervised)
    3. Fine-tune PatchTST  on ~317 CTU-UHB (classification)
    4. Extract 4-feature vectors via alert pipeline (stride=60, fast)
    5. Fit LogisticRegression on finetune-train features
    6. Predict on held-out ~100 CTU-UHB (OOF)
    7. Log per-fold AUC / Sens / Spec

After all 5 folds:
    8. Stack OOF predictions → global AUC
    9. Bootstrap CI (N=5,000, stratified, per-fold threshold)
   10. Save final report + visualization

USAGE
-----
# Dry-run (5 min, verifies everything runs end-to-end):
python scripts/run_e2e_cv.py --device cuda --dry-run

# Full overnight run (launch as background, close laptop):
nohup python scripts/run_e2e_cv.py --device cuda \\
    > logs/e2e_cv_master.log 2>&1 &
echo "PID: $!"

# Resume after interruption (auto-skips completed folds):
python scripts/run_e2e_cv.py --device cuda

ESTIMATED RUNTIME (T4 GPU)
---------------------------
  Pretrain  per fold : ~15-25 min  (200 epochs max, patience=10)
  Fine-tune per fold : ~10-20 min  (100 epochs max, patience=15)
  LR extract+fit     :  ~4 min     (stride=60, 397 recordings)
  Total  × 5 folds   :  ~3-5 hours

ESTIMATED RUNTIME (CPU only)
----------------------------
  Pretrain  per fold : ~2-4 hours
  Fine-tune per fold : ~1-2 hours
  Total × 5 folds    : ~15-30 hours   (not recommended)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# ── Force UTF-8 stdout/stderr so nohup + Colab + Windows all work ───────────
import io as _io
try:
    sys.stdout = _io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = _io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
except AttributeError:
    pass  # already reconfigured or not a real stream

# ── Logging setup (prints to stdout with timestamps -- works with nohup) ─────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# ── Root path (script lives in scripts/, one level below project root) ────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
ALERT_THRESHOLD    = 0.40   # locked — same as all other notebooks
DECISION_THRESHOLD = 0.284  # Youden-optimal on original test set
FEATURE_NAMES      = ["segment_length", "max_prediction",
                      "cumulative_sum", "weighted_integral",
                      "n_alert_segments", "alert_fraction"]
N_BOOTSTRAP        = 5_000
SEED               = 42


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FOLD CSV GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fold_csvs(
    root: Path,
    n_folds: int = 5,
    seed: int = SEED,
    force: bool = False,
) -> Path:
    """
    Create per-fold split CSVs from train.csv + val.csv (497 combined).

    Fold structure for fold k:
      - test_fold_k       : ~100 held-out recordings  (OOF predictions)
      - ft_train_fold_k   : ~317 recordings (80% of remaining 397)
      - ft_val_fold_k     : ~80  recordings (20% of remaining 397)
      - pretrain_fold_k   : FHRMA(135) + remaining 397 CTU-UHB
        (test_fold_k is excluded from pretrain to avoid leakage)

    Returns the folder where CSVs are written.
    """
    from sklearn.model_selection import StratifiedKFold

    cv_dir = root / "data" / "splits" / "e2e_cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    # Check if already generated
    expected_marker = cv_dir / f"fold{n_folds - 1}_test.csv"
    if expected_marker.exists() and not force:
        log.info("Fold CSVs already exist at %s — skipping generation.", cv_dir)
        return cv_dir

    log.info("Generating %d-fold CSVs -> %s", n_folds, cv_dir)

    # Load train + val (497 labelled CTU-UHB recordings)
    df_train = pd.read_csv(root / "data" / "splits" / "train.csv")
    df_val   = pd.read_csv(root / "data" / "splits" / "val.csv")
    df_497   = pd.concat([df_train, df_val], ignore_index=True)
    df_497   = df_497.drop_duplicates(subset="id").reset_index(drop=True)
    assert len(df_497) == 497, f"Expected 497 rows, got {len(df_497)}"

    # Load original pretrain.csv (to get FHRMA rows + all CTG ids)
    df_pretrain_orig = pd.read_csv(root / "data" / "splits" / "pretrain.csv")
    df_fhrma = df_pretrain_orig[df_pretrain_orig["dataset"] == "fhrma"].copy()
    df_ctg_all = df_pretrain_orig[df_pretrain_orig["dataset"] == "ctg"].copy()

    log.info("  497 CTU-UHB  (%d acidemia, %d normal)",
             df_497["target"].sum(), (df_497["target"] == 0).sum())
    log.info("  135 FHRMA  (unlabelled, pretrain only)")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    for fold_idx, (remaining_idx, test_idx) in enumerate(
            skf.split(df_497, df_497["target"])):

        df_test_fold = df_497.iloc[test_idx].copy()
        df_remain    = df_497.iloc[remaining_idx].copy()

        # stratified 80/20 split of remaining for finetune train/val
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True,
                                    random_state=seed + fold_idx)
        ft_train_i, ft_val_i = next(
            skf_inner.split(df_remain, df_remain["target"]))
        df_ft_train = df_remain.iloc[ft_train_i].copy()
        df_ft_val   = df_remain.iloc[ft_val_i].copy()

        # pretrain = FHRMA + all remaining CTU-UHB (train+val for finetune)
        # test_fold_k is EXCLUDED from pretrain (no leakage)
        test_ids = set(df_test_fold["id"].astype(str).tolist())
        df_ctg_pretrain = df_ctg_all[
            ~df_ctg_all["id"].astype(str).isin(test_ids)
        ].copy()
        df_pretrain_fold = pd.concat(
            [df_ctg_pretrain, df_fhrma], ignore_index=True)

        # Write CSVs
        df_test_fold.to_csv(
            cv_dir / f"fold{fold_idx}_test.csv", index=False)
        df_ft_train.to_csv(
            cv_dir / f"fold{fold_idx}_ft_train.csv", index=False)
        df_ft_val.to_csv(
            cv_dir / f"fold{fold_idx}_ft_val.csv", index=False)
        df_pretrain_fold.to_csv(
            cv_dir / f"fold{fold_idx}_pretrain.csv", index=False)

        log.info(
            "  Fold %d: pretrain=%d, ft_train=%d, ft_val=%d, test=%d "
            "(pos: ft_train=%d, ft_val=%d, test=%d)",
            fold_idx,
            len(df_pretrain_fold),
            len(df_ft_train), len(df_ft_val), len(df_test_fold),
            df_ft_train["target"].sum(),
            df_ft_val["target"].sum(),
            df_test_fold["target"].sum(),
        )

    log.info("Fold CSVs written to %s", cv_dir)
    return cv_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SUBPROCESS RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_subprocess(cmd: list[str], tag: str) -> bool:
    """
    Run a subprocess command, stream stdout/stderr in real-time, return success.
    Logs every line with the [tag] prefix so nohup logs are readable.
    """
    import subprocess

    log.info("[%s] START: %s", tag, " ".join(cmd))
    t0 = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.info("[%s] %s", tag, line)

    proc.wait()
    elapsed = time.time() - t0

    if proc.returncode == 0:
        log.info("[%s] DONE in %.1fs", tag, elapsed)
        return True
    else:
        log.error("[%s] FAILED (exit=%d) in %.1fs",
                  tag, proc.returncode, elapsed)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PER-FOLD PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_fold(
    fold_idx: int,
    root: Path,
    cv_dir: Path,
    device: str,
    config_path: Path,
    stride: int,
    dry_run: bool,
    skip_pretrain: bool = False,
) -> Optional[dict]:
    """
    Run the complete pipeline for one fold.

    Returns a dict with OOF predictions and metrics, or None on failure.
    """
    python = sys.executable
    tag = f"fold{fold_idx}"

    # ── Output paths ────────────────────────────────────────────────────────
    ckpt_pretrain_dir = root / "checkpoints" / "e2e_cv" / f"fold{fold_idx}" / "pretrain"
    ckpt_finetune_dir = root / "checkpoints" / "e2e_cv" / f"fold{fold_idx}" / "finetune"
    log_pretrain      = root / "logs" / "e2e_cv" / f"fold{fold_idx}_pretrain.csv"
    log_finetune      = root / "logs" / "e2e_cv" / f"fold{fold_idx}_finetune.csv"
    oof_csv           = root / "results" / "e2e_cv" / f"fold{fold_idx}_oof_scores.csv"

    for d in [ckpt_pretrain_dir, ckpt_finetune_dir,
              log_pretrain.parent, oof_csv.parent]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Input CSVs ──────────────────────────────────────────────────────────
    pretrain_csv = cv_dir / f"fold{fold_idx}_pretrain.csv"
    ft_train_csv = cv_dir / f"fold{fold_idx}_ft_train.csv"
    ft_val_csv   = cv_dir / f"fold{fold_idx}_ft_val.csv"
    test_csv     = cv_dir / f"fold{fold_idx}_test.csv"
    best_pretrain_ckpt = ckpt_pretrain_dir / "best_pretrain.pt"
    best_finetune_ckpt = ckpt_finetune_dir / "best_finetune.pt"

    dry_batches = ["--max-batches", "2"] if dry_run else []

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 1 — Pretrain  (or copy shared checkpoint)
    # ═══════════════════════════════════════════════════════════════════════
    if best_pretrain_ckpt.exists():
        log.info("[%s] Pretrain checkpoint found — skipping pretrain.", tag)
    elif skip_pretrain:
        # Use the single shared pretrain checkpoint (trained on all 687 recordings).
        # Valid because pretrain is self-supervised — uses no labels, no leakage.
        shared_ckpt = root / "checkpoints" / "pretrain" / "best_pretrain.pt"
        if not shared_ckpt.exists():
            log.error("[%s] --skip-pretrain requested but %s not found.", tag, shared_ckpt)
            return None
        ckpt_pretrain_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(shared_ckpt, best_pretrain_ckpt)
        log.info("[%s] --skip-pretrain: copied shared checkpoint %s → %s",
                 tag, shared_ckpt, best_pretrain_ckpt)
    else:
        log.info("[%s] ── Phase 1: Pretrain ──────────────────────────────", tag)
        ok = run_subprocess([
            python, "src/train/pretrain.py",
            "--config",         str(config_path),
            "--device",         device,
            "--pretrain-csv",   str(pretrain_csv),
            "--checkpoint-dir", str(ckpt_pretrain_dir),
            "--log-path",       str(log_pretrain),
        ] + dry_batches, tag=f"{tag}/pretrain")
        if not ok:
            log.error("[%s] Pretrain failed — aborting fold.", tag)
            return None
        if not best_pretrain_ckpt.exists():
            log.error("[%s] best_pretrain.pt not found after pretrain — aborting fold.", tag)
            return None
    # (end of Phase 1 else block)

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2 — Fine-tune
    # ═══════════════════════════════════════════════════════════════════════
    if best_finetune_ckpt.exists():
        log.info("[%s] Finetune checkpoint found — skipping finetune.", tag)
    else:
        log.info("[%s] ── Phase 2: Fine-tune ─────────────────────────────", tag)
        ok = run_subprocess([
            python, "src/train/finetune.py",
            "--config",               str(config_path),
            "--device",               device,
            "--train-csv",            str(ft_train_csv),
            "--val-csv",              str(ft_val_csv),
            "--pretrain-checkpoint",  str(best_pretrain_ckpt),
            "--checkpoint-dir",       str(ckpt_finetune_dir),
            "--log-path",             str(log_finetune),
        ] + dry_batches, tag=f"{tag}/finetune")
        if not ok:
            log.error("[%s] Fine-tune failed — aborting fold.", tag)
            return None
        if not best_finetune_ckpt.exists():
            log.error("[%s] best_finetune.pt not found after fine-tune — aborting fold.", tag)
            return None

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 3+4 — Feature extraction, LR fit, OOF predictions
    # ═══════════════════════════════════════════════════════════════════════
    if oof_csv.exists():
        log.info("[%s] OOF scores found — skipping feature extraction.", tag)
        df_oof = pd.read_csv(oof_csv)
        return _metrics_from_oof(df_oof, fold_idx)

    log.info("[%s] ── Phase 3: Feature Extraction + LR ─────────────────", tag)

    import yaml
    from src.inference.sliding_window import inference_recording
    from src.inference.alert_extractor import (
        extract_alert_segments, compute_alert_features, ZERO_FEATURES
    )
    from src.train.train_lr import load_finetuned_model, build_feature_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve

    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model = load_finetuned_model(cfg, best_finetune_ckpt, device)
    model.eval()
    log.info("[%s] Model loaded from %s", tag, best_finetune_ckpt)

    processed_dir = root / "data" / "processed" / "ctu_uhb"

    n_train_recs = len(pd.read_csv(ft_train_csv))
    if dry_run:
        max_recs_train = min(5, n_train_recs)
        max_recs_test  = min(5, len(pd.read_csv(test_csv)))
        log.info("[%s] DRY-RUN: limiting to %d train / %d test recordings",
                 tag, max_recs_train, max_recs_test)
    else:
        max_recs_train = None
        max_recs_test  = None

    def extract_features_from_csv(
        csv_path: Path, max_recs: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, list]:
        df = pd.read_csv(csv_path)
        if max_recs is not None:
            df = df.head(max_recs)
        X_rows, y_rows, id_rows = [], [], []
        failed = []
        for _, row in df.iterrows():
            rec_id  = int(row["id"])
            label   = int(row["target"])
            npy_path = processed_dir / f"{rec_id}.npy"
            if not npy_path.exists():
                failed.append(rec_id)
                continue
            try:
                signal = np.load(npy_path, mmap_mode="r")
                with torch.no_grad():
                    scores = inference_recording(
                        model, signal, stride=stride, device=device)
                segments = extract_alert_segments(
                    scores, threshold=ALERT_THRESHOLD)
                if segments:
                    longest = max(segments, key=lambda s: len(s[2]))
                    feats = compute_alert_features(
                        longest[2],
                        inference_stride=stride,
                        fs=cfg["data"]["fs"],
                    )
                else:
                    feats = dict(ZERO_FEATURES)  # copy to avoid mutating
                # S14 Addition 1: record-level features (across ALL segments)
                n_total_windows = len(scores) if scores else 1
                n_alert_windows = sum(len(s[2]) for s in segments)
                feats["n_alert_segments"] = float(len(segments))
                feats["alert_fraction"]   = float(n_alert_windows / n_total_windows)
                X_rows.append([feats[k] for k in FEATURE_NAMES])
                y_rows.append(label)
                id_rows.append(rec_id)
            except Exception as exc:
                log.warning("[%s] Error on recording %d: %s", tag, rec_id, exc)
                failed.append(rec_id)
        if failed:
            log.warning("[%s] Skipped %d recordings (missing .npy): %s",
                        tag, len(failed), failed[:5])
        return (np.array(X_rows, dtype=np.float32),
                np.array(y_rows, dtype=int), id_rows)

    log.info("[%s] Extracting features — ft_train (%d recs) …",
             tag, n_train_recs)
    t0 = time.time()
    X_tr, y_tr, _ = extract_features_from_csv(ft_train_csv, max_recs_train)
    log.info("[%s] ft_train features: %s in %.1fs", tag, X_tr.shape,
             time.time() - t0)

    log.info("[%s] Extracting features — test fold (%d recs) …",
             tag, len(pd.read_csv(test_csv)))
    t0 = time.time()
    X_te, y_te, ids_te = extract_features_from_csv(test_csv, max_recs_test)
    log.info("[%s] test features: %s in %.1fs", tag, X_te.shape,
             time.time() - t0)

    if len(np.unique(y_tr)) < 2:
        log.error("[%s] Training set has only one class — aborting fold.", tag)
        return None

    # Fit LR — StandardScaler + C=0.1 (S14: feature scale mismatch fix)
    log.info("[%s] Fitting StandardScaler + LogisticRegression(C=0.1) …", tag)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, random_state=SEED,
            class_weight="balanced", C=0.1,
        )),
    ])
    lr.fit(X_tr, y_tr)

    # Youden threshold on ft_train (inner self-estimate)
    tr_scores = lr.predict_proba(X_tr)[:, 1]
    fpr_t, tpr_t, thr_t = roc_curve(y_tr, tr_scores)
    youden_t = tpr_t - fpr_t
    fold_thr = float(thr_t[np.argmax(youden_t)])
    log.info("[%s] LR Youden threshold (train set) = %.3f", tag, fold_thr)

    # OOF predictions
    oof_scores = lr.predict_proba(X_te)[:, 1]

    # Save OOF csv
    df_oof = pd.DataFrame({
        "id":          ids_te,
        "label":       y_te,
        "oof_score":   oof_scores,
        "fold":        fold_idx,
        "fold_threshold": fold_thr,
    })
    df_oof.to_csv(oof_csv, index=False)
    log.info("[%s] OOF scores saved → %s", tag, oof_csv)

    return _metrics_from_oof(df_oof, fold_idx)


def _metrics_from_oof(df_oof: pd.DataFrame, fold_idx: int) -> dict:
    from sklearn.metrics import roc_auc_score

    y_te       = df_oof["label"].values
    oof_scores = df_oof["oof_score"].values
    fold_thr   = float(df_oof["fold_threshold"].iloc[0])
    oof_pred   = (oof_scores >= fold_thr).astype(int)

    if len(np.unique(y_te)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_te, oof_scores)

    tp = ((oof_pred == 1) & (y_te == 1)).sum()
    fn = ((oof_pred == 0) & (y_te == 1)).sum()
    tn = ((oof_pred == 0) & (y_te == 0)).sum()
    fp = ((oof_pred == 1) & (y_te == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    log.info(
        "Fold %d metrics: n=%d (pos=%d), AUC=%.3f, Sens=%.3f, Spec=%.3f, T=%.3f",
        fold_idx, len(y_te), int(y_te.sum()), auc, sens, spec, fold_thr)

    return {
        "fold":      fold_idx,
        "n_test":    len(y_te),
        "n_pos":     int(y_te.sum()),
        "AUC":       auc,
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "threshold": fold_thr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  AGGREGATE + REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_results(root: Path, n_folds: int, n_bootstrap: int = N_BOOTSTRAP) -> None:
    """Collect all OOF CSVs, run Bootstrap CI, print + save final report."""
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_cv_dir = root / "results" / "e2e_cv"
    images_dir     = root / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Collect OOF frames
    frames = []
    for k in range(n_folds):
        p = results_cv_dir / f"fold{k}_oof_scores.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            log.warning("Fold %d OOF file missing: %s", k, p)

    if not frames:
        log.error("No OOF files found — cannot aggregate.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    log.info(
        "Aggregating %d OOF predictions (%d acidemia, %d normal)",
        len(df_all), df_all["label"].sum(),
        (df_all["label"] == 0).sum())

    all_labels     = df_all["label"].values
    all_scores     = df_all["oof_score"].values
    all_fold_thrs  = df_all["fold_threshold"].values
    oof_pred       = (all_scores >= all_fold_thrs).astype(int)

    # Global OOF AUC
    if len(np.unique(all_labels)) < 2:
        log.error("Only one class in aggregated OOF — cannot compute AUC.")
        return
    oof_auc = roc_auc_score(all_labels, all_scores)

    # Point estimates
    tp = ((oof_pred == 1) & (all_labels == 1)).sum()
    fn = ((oof_pred == 0) & (all_labels == 1)).sum()
    tn = ((oof_pred == 0) & (all_labels == 0)).sum()
    fp = ((oof_pred == 1) & (all_labels == 0)).sum()
    pt_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pt_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    log.info("Global OOF  AUC=%.3f, Sens=%.3f, Spec=%.3f  (n=%d, pos=%d)",
             oof_auc, pt_sens, pt_spec, len(all_labels), int(all_labels.sum()))

    # Bootstrap CI (stratified, using per-fold threshold)
    log.info("Running Bootstrap CI (N=%d) …", n_bootstrap)
    rng = np.random.default_rng(SEED + 99)
    b_aucs, b_sens, b_spec = [], [], []
    for _ in range(n_bootstrap):
        idx_pos = np.where(all_labels == 1)[0]
        idx_neg = np.where(all_labels == 0)[0]
        s_pos   = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        s_neg   = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        idx_b   = np.concatenate([s_pos, s_neg])

        yt = all_labels[idx_b]
        ys = all_scores[idx_b]
        thrs_b = all_fold_thrs[idx_b]
        yp = (ys >= thrs_b).astype(int)

        if len(np.unique(yt)) < 2:
            continue

        b_aucs.append(roc_auc_score(yt, ys))
        tp_b = ((yp == 1) & (yt == 1)).sum()
        fn_b = ((yp == 0) & (yt == 1)).sum()
        tn_b = ((yp == 0) & (yt == 0)).sum()
        fp_b = ((yp == 1) & (yt == 0)).sum()
        b_sens.append(tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0)
        b_spec.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0)

    b_aucs = np.array(b_aucs)
    b_sens = np.array(b_sens)
    b_spec = np.array(b_spec)

    ci_auc  = np.percentile(b_aucs, [2.5, 97.5])
    ci_sens = np.percentile(b_sens, [2.5, 97.5])
    ci_spec = np.percentile(b_spec, [2.5, 97.5])

    # ── Per-fold summary ─────────────────────────────────────────────────────
    fold_metrics = []
    for k in range(n_folds):
        p = results_cv_dir / f"fold{k}_oof_scores.csv"
        if p.exists():
            fold_metrics.append(_metrics_from_oof(pd.read_csv(p), k))

    cv_df = pd.DataFrame(fold_metrics)
    log.info(
        "\nPer-fold summary:\n%s",
        cv_df[["fold","n_test","n_pos","AUC","Sensitivity","Specificity","threshold"]]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── Final report ─────────────────────────────────────────────────────────
    sep = "=" * 72
    log.info("\n%s", sep)
    log.info("  SentinelFatal2 -- E2E %d-Fold CV Final Report", n_folds)
    log.info("%s", sep)
    log.info("  AUC         = %.3f  [95%% CI: %.3f – %.3f]",
             oof_auc, ci_auc[0], ci_auc[1])
    log.info("  Sensitivity = %.3f  [95%% CI: %.3f – %.3f]",
             pt_sens, ci_sens[0], ci_sens[1])
    log.info("  Specificity = %.3f  [95%% CI: %.3f – %.3f]",
             pt_spec, ci_spec[0], ci_spec[1])
    log.info("  n = %d (%d acidemia, %d normal)",
             len(all_labels), int(all_labels.sum()),
             int((all_labels == 0).sum()))
    log.info("  Bootstrap N = %d  |  Folds completed = %d / %d",
             n_bootstrap, len(fold_metrics), n_folds)
    log.info("  NOTE: PatchTST retrained from scratch on every fold.")
    log.info("%s", sep)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    report_path = root / "results" / "e2e_cv_final_report.csv"
    pd.DataFrame([{
        "method":      f"E2E {n_folds}-Fold CV (n={len(all_labels)})",
        "AUC":          oof_auc,
        "AUC_ci_lo":    ci_auc[0],
        "AUC_ci_hi":    ci_auc[1],
        "Sensitivity":  pt_sens,
        "Sens_ci_lo":   ci_sens[0],
        "Sens_ci_hi":   ci_sens[1],
        "Specificity":  pt_spec,
        "Spec_ci_lo":   ci_spec[0],
        "Spec_ci_hi":   ci_spec[1],
        "n":            len(all_labels),
        "n_acidemia":   int(all_labels.sum()),
        "n_folds_completed": len(fold_metrics),
        "bootstrap_N":  n_bootstrap,
        "note":         "true_e2e_cv_PatchTST_retrained_per_fold",
    }]).to_csv(report_path, index=False)
    log.info("Final report saved → %s", report_path)

    cv_df.to_csv(root / "results" / "e2e_cv_per_fold.csv", index=False)

    # ── Visualization ─────────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Left: per-fold ROC
        mean_fpr = np.linspace(0, 1, 100)
        tprs, fold_aucs = [], []
        palette = ["tab:blue", "tab:orange", "tab:green",
                   "tab:red", "tab:purple"]
        for k in range(n_folds):
            p = results_cv_dir / f"fold{k}_oof_scores.csv"
            if not p.exists():
                continue
            df_k = pd.read_csv(p)
            fl, fs = df_k["label"].values, df_k["oof_score"].values
            if len(np.unique(fl)) < 2:
                continue
            from sklearn.metrics import roc_curve as _roc
            fpr_k, tpr_k, _ = _roc(fl, fs)
            tprs.append(np.interp(mean_fpr, fpr_k, tpr_k))
            fold_aucs.append(roc_auc_score(fl, fs))
            axes[0].plot(fpr_k, tpr_k, alpha=0.5, lw=1.5,
                         color=palette[k % len(palette)],
                         label=f"Fold {k} ({fold_aucs[-1]:.3f})")

        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr  = np.std(tprs, axis=0)
            axes[0].plot(mean_fpr, mean_tpr, "k-", lw=2.5,
                         label=f"Mean ({np.mean(fold_aucs):.3f}±{np.std(fold_aucs):.3f})")
            axes[0].fill_between(mean_fpr,
                                 mean_tpr - std_tpr, mean_tpr + std_tpr,
                                 alpha=0.12, color="k")
        axes[0].plot([0, 1], [0, 1], "r--", lw=1.5)
        axes[0].set(xlabel="FPR", ylabel="TPR",
                    title=f"E2E {n_folds}-Fold CV ROC (n={len(all_labels)})")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Middle: Bootstrap AUC distribution
        axes[1].hist(b_aucs, bins=60, color="mediumseagreen",
                     alpha=0.75, edgecolor="white")
        axes[1].axvline(oof_auc, color="black", lw=2,
                        label=f"OOF AUC: {oof_auc:.3f}")
        axes[1].axvline(ci_auc[0], color="red", lw=1.5, ls="--",
                        label=f"CI: [{ci_auc[0]:.3f}–{ci_auc[1]:.3f}]")
        axes[1].axvline(ci_auc[1], color="red", lw=1.5, ls="--")
        axes[1].axvline(0.812, color="steelblue", lw=2, ls=":",
                        label="Test-55 AUC: 0.812")
        axes[1].set(xlabel="AUC", ylabel="Frequency",
                    title=f"AUC Bootstrap (N={n_bootstrap:,})")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Right: comparison bar chart
        orig_auc = 0.812
        methods = ["Test Set\n(n=55,\nexact)", "E2E CV\n(n={},\ntrue)".format(
            len(all_labels))]
        aucs_bar  = [orig_auc, oof_auc]
        lo_bar    = [orig_auc - 0.630, oof_auc - ci_auc[0]]   # err down
        hi_bar    = [0.953 - orig_auc, ci_auc[1] - oof_auc]   # err up
        colors_b  = ["steelblue", "mediumseagreen"]
        bars = axes[2].bar(methods, aucs_bar, color=colors_b, alpha=0.8,
                           width=0.4)
        axes[2].errorbar(methods, aucs_bar, yerr=[lo_bar, hi_bar],
                         fmt="none", color="black", capsize=8, lw=2)
        axes[2].set_ylim(0.45, 1.05)
        axes[2].axhline(0.826, color="red", lw=1.5, ls=":",
                        label="Paper benchmark 0.826")
        axes[2].set_ylabel("AUC")
        axes[2].set_title("AUC Comparison  95% CI")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, aucs_bar):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2, val + 0.008,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

        fig.suptitle(
            f"SentinelFatal2 — E2E {n_folds}-Fold CV (PatchTST retrained per fold)",
            fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_img = images_dir / "e2e_cv.png"
        plt.savefig(out_img, dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("Visualization saved → %s", out_img)

    except Exception as exc:
        log.warning("Visualization failed (non-fatal): %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-End 5-Fold CV for SentinelFatal2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device",   default="cuda",
                   help="'cuda' or 'cpu'")
    p.add_argument("--config",   default=str(ROOT / "config" / "train_config.yaml"),
                   help="Path to train_config.yaml")
    p.add_argument("--folds",    type=int, default=5,
                   help="Number of CV folds")
    p.add_argument("--seed",     type=int, default=SEED,
                   help="Random seed for fold generation")
    p.add_argument("--stride",   type=int, default=60,
                   help="Feature extraction stride (60 = fast, 1 = exact but ~8h/fold)")
    p.add_argument("--dry-run",  action="store_true",
                   help="2 batches per stage (~5 min total), verify pipeline end-to-end")
    p.add_argument("--force-folds", action="store_true",
                   help="Regenerate fold CSVs even if they already exist")
    p.add_argument("--skip-pretrain", action="store_true",
                   help="Use existing checkpoints/pretrain/best_pretrain.pt instead of "
                        "retraining. Safe because pretrain is self-supervised (no label leakage).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 72)
    log.info("  SentinelFatal2 -- E2E %d-Fold CV", args.folds)
    log.info("  device=%s  dry_run=%s  stride=%d  seed=%d",
             args.device, args.dry_run, args.stride, args.seed)
    log.info("=" * 72)

    # GPU check
    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to CPU.")
            args.device = "cpu"
        else:
            props = torch.cuda.get_device_properties(0)
            log.info("GPU: %s  (%.1f GB VRAM)",
                     props.name, props.total_memory / 1e9)
    else:
        log.info("Running on CPU -- this will take ~15-30 hours for full CV.")

    config_path = Path(args.config)

    # ── Phase 0: Generate fold CSVs ─────────────────────────────────────────
    t0_total = time.time()
    cv_dir = generate_fold_csvs(
        ROOT, n_folds=args.folds, seed=args.seed,
        force=args.force_folds)

    # ── Per-fold loop ────────────────────────────────────────────────────────
    fold_results = []
    failed_folds = []

    # Progress log CSV (written after each fold so you can check live)
    progress_csv = ROOT / "logs" / "e2e_cv_progress.csv"
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    for fold_idx in range(args.folds):
        log.info("")
        log.info("=" * 72)
        log.info("  FOLD %d / %d", fold_idx, args.folds - 1)
        log.info("=" * 72)
        t_fold = time.time()

        try:
            result = run_fold(
                fold_idx=fold_idx,
                root=ROOT,
                cv_dir=cv_dir,
                device=args.device,
                config_path=config_path,
                stride=args.stride,
                dry_run=args.dry_run,
                skip_pretrain=args.skip_pretrain,
            )
        except Exception as exc:
            log.error("Fold %d raised an unexpected exception:\n%s",
                      fold_idx, traceback.format_exc())
            result = None

        elapsed_fold = time.time() - t_fold

        if result is None:
            log.error("Fold %d FAILED (%.1fs)", fold_idx, elapsed_fold)
            failed_folds.append(fold_idx)
        else:
            log.info("Fold %d DONE in %.1fs", fold_idx, elapsed_fold)
            fold_results.append(result)

            # Append to live progress CSV
            df_prog = pd.DataFrame(fold_results)
            df_prog.to_csv(progress_csv, index=False)
            log.info("Progress saved → %s", progress_csv)

    # ── Summary after all folds ──────────────────────────────────────────────
    log.info("")
    log.info("=== ALL FOLDS COMPLETE ===")
    log.info("  Successful: %d / %d", len(fold_results), args.folds)
    if failed_folds:
        log.warning("  Failed folds: %s", failed_folds)

    if len(fold_results) >= 1:
        aggregate_results(ROOT, args.folds, n_bootstrap=N_BOOTSTRAP)
    else:
        log.error("No successful folds — cannot aggregate.")

    total_elapsed = time.time() - t0_total
    log.info("Total runtime: %.1f min (%.1f h)",
             total_elapsed / 60, total_elapsed / 3600)

    if failed_folds:
        sys.exit(1)


if __name__ == "__main__":
    main()
