"""
train_lr.py — Stage 2 Logistic Regression Training for SentinelFatal2
======================================================================
Source: arXiv:2601.06149v1, Section II-F
SSOT:   docs/work_plan.md, Part ו, שלב 5.3

Trains a Logistic Regression (LR) on 4 alert features derived from
sliding-window NN inference scores on the Train set.

Pipeline
--------
  For each recording in train.csv:
    1. Run inference_recording() with stride=INFERENCE_STRIDE_REPRO (1).
    2. Extract alert segments via extract_alert_segments() (threshold=0.5).
    3. Select the LONGEST alert segment (max feature contribution).
       If no alert segment exists -> use ZERO_FEATURES (assumption S10).
    4. Compute 4 features via compute_alert_features().

  Fit LogisticRegression(max_iter=1000) on the 4-feature matrix.
  Save {'model': lr, 'stride': INFERENCE_STRIDE_REPRO} to
       checkpoints/alerting/logistic_regression.pkl

CRITICAL RULES (V5.4, V5.6):
  - ONLY train.csv is read.  val.csv and test.csv are NEVER touched.
  - LR training stride MUST equal INFERENCE_STRIDE_REPRO (1).
  - Stage 7 evaluation MUST use the same stride=1 (stored in .pkl).

Assumption S10 (zero-features for no-alert recordings) is logged in
deviation_log.md.

Usage
-----
  # Full run (all train recordings, stride=1 — always):
  python src/train/train_lr.py --config config/train_config.yaml --device cuda

  # Dry-run subset (first N recordings, stride=1 — takes longer on CPU):
  python src/train/train_lr.py --config config/train_config.yaml \\
      --device cpu --max-recordings 5

Outputs
-------
  checkpoints/alerting/logistic_regression.pkl  — {'model', 'stride', 'n_train'}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Project root on sys.path (AGW-20 pattern)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model.patchtst import PatchTST, load_config
from src.model.heads import ClassificationHead
from src.inference.sliding_window import (
    INFERENCE_STRIDE_REPRO,
    inference_recording,
)
from src.inference.alert_extractor import (
    ALERT_THRESHOLD,
    extract_alert_segments,
    compute_alert_features,
    ZERO_FEATURES,
)


# ---------------------------------------------------------------------------
# Feature extraction for one recording
# ---------------------------------------------------------------------------

def _features_for_recording(
    model: torch.nn.Module,
    npy_path: Path,
    stride: int,
    device: str,
) -> Dict[str, float]:
    """Run inference on one recording and return 4-feature dict.

    If no alert segment is found, returns ZERO_FEATURES (assumption S10).
    The LONGEST alert segment is used when multiple segments exist.
    """
    signal = np.load(npy_path, mmap_mode="r")    # (2, T) float32

    scores = inference_recording(model, signal, stride=stride, device=device)

    segments = extract_alert_segments(scores, threshold=ALERT_THRESHOLD)

    if not segments:
        return dict(ZERO_FEATURES)

    # Select longest segment (most informative)
    longest = max(segments, key=lambda seg: len(seg[2]))
    _, _, seg_scores = longest

    return compute_alert_features(seg_scores, inference_stride=stride, fs=4.0)


# ---------------------------------------------------------------------------
# Build feature matrix from a split CSV (train only!)
# ---------------------------------------------------------------------------

def build_feature_matrix(
    model: torch.nn.Module,
    split_csv: Path,
    processed_dir: Path,
    stride: int,
    device: str,
    max_recordings: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from a split CSV for LR training.

    Args:
        model:          PatchTST with ClassificationHead in eval mode.
        split_csv:      Path to train.csv (MUST NOT be val/test).
        processed_dir:  Path to data/processed/ directory.
        stride:         Inference stride (1 for official evaluation).
        device:         Torch device string.
        max_recordings: If set, limit to first N recordings (dry-run only).

    Returns:
        X: (N, 4) float32 feature matrix.
        y: (N,)   int binary labels.
    """
    df = pd.read_csv(split_csv, dtype={"id": str, "target": int})
    if max_recordings is not None:
        df = df.head(max_recordings)

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    model.eval()
    n = len(df)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        rec_id = str(row.id)
        label = int(row.target)

        npy_path = processed_dir / "ctu_uhb" / f"{rec_id}.npy"
        if not npy_path.exists():
            print(f"[train_lr] WARNING: {npy_path} not found, skipping {rec_id}")
            continue

        feats = _features_for_recording(model, npy_path, stride=stride, device=device)

        feat_vec = [
            feats["segment_length"],
            feats["max_prediction"],
            feats["cumulative_sum"],
            feats["weighted_integral"],
        ]
        X_rows.append(feat_vec)
        y_rows.append(label)

        if idx % 50 == 0 or idx == n:
            print(f"[train_lr] {idx}/{n} recordings processed")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=int)
    return X, y


# ---------------------------------------------------------------------------
# Checkpoint validation (AGW-27: guard for Stage 7 — Agent 7 should call this)
# ---------------------------------------------------------------------------

def validate_lr_checkpoint(
    pkl_path: Path,
    expected_stride: int = INFERENCE_STRIDE_REPRO,
    min_n_train: int = 397,  # 90% of 441 train recordings
) -> tuple[bool, str]:
    """Validate a saved LR checkpoint before Stage 7 evaluation.

    AGW-27 guard: fail-fast if the checkpoint is a dry-run artifact
    (wrong stride or too few training recordings).

    Args:
        pkl_path:        Path to logistic_regression.pkl.
        expected_stride: Must equal INFERENCE_STRIDE_REPRO (1).
        min_n_train:     Minimum acceptable n_train (default 397 = 90% of 441).

    Returns:
        (ok, message) — ok=True if valid, False with explanation otherwise.
    """
    import joblib as _joblib

    if not Path(pkl_path).exists():
        return False, f"checkpoint not found: {pkl_path}"

    payload = _joblib.load(pkl_path)

    for key in ("model", "stride", "n_train", "feature_names"):
        if key not in payload:
            return False, f"missing key '{key}' in checkpoint"

    if payload["stride"] != expected_stride:
        return False, (
            f"stride={payload['stride']} != expected {expected_stride}. "
            "Checkpoint was built with wrong stride — regenerate with stride=1 (INFERENCE_STRIDE_REPRO)."
        )

    if payload["n_train"] < min_n_train:
        return False, (
            f"n_train={payload['n_train']} < min_n_train={min_n_train}. "
            "Checkpoint looks like a dry-run artifact — regenerate on full train set (441)."
        )

    return True, (
        f"LR checkpoint valid (stride={payload['stride']}, "
        f"n_train={payload['n_train']}, "
        f"features={payload['feature_names']})"
    )


# ---------------------------------------------------------------------------
# Load fine-tuned model
# ---------------------------------------------------------------------------

def load_finetuned_model(
    config: dict,
    checkpoint_path: Path,
    device: str,
) -> torch.nn.Module:
    """Load PatchTST + ClassificationHead from a fine-tuning checkpoint.

    Uses strict=False to tolerate any extra/missing keys (AGW-21 pattern).
    """
    model = PatchTST(config)
    d_in = (
        config["data"]["n_patches"]
        * config["model"]["d_model"]
        * config["data"]["n_channels"]
    )
    model.replace_head(ClassificationHead(d_in=d_in))

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    print(f"[train_lr] Loaded checkpoint: {checkpoint_path}")
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Stage-2 Logistic Regression on 4 alert features"
    )
    p.add_argument(
        "--config",
        default="config/train_config.yaml",
        help="Path to train_config.yaml",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to fine-tuned checkpoint (default: checkpoints/finetune/best_finetune.pt)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )
    p.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Limit to first N recordings (subset run; None = all train)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .pkl path (default: checkpoints/alerting/logistic_regression.pkl)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # Resolve paths
    # -----------------------------------------------------------------------
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent   # …/SentinelFatal2

    config = load_config(config_path)

    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else (
        project_root / "checkpoints" / "finetune" / "best_finetune.pt"
    )
    processed_dir = project_root / "data" / "processed"
    train_csv = project_root / "data" / "splits" / "train.csv"
    out_pkl = Path(args.out).resolve() if args.out else (
        project_root / "checkpoints" / "alerting" / "logistic_regression.pkl"
    )

    # Sanity checks
    for path, name in [
        (config_path, "config"),
        (ckpt_path, "finetune checkpoint"),
        (train_csv, "train.csv"),
        (processed_dir, "processed_dir"),
    ]:
        if not path.exists():
            print(f"[train_lr] ERROR: {name} not found: {path}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"[train_lr] Loading model from {ckpt_path}")
    model = load_finetuned_model(config, ckpt_path, args.device)

    # -----------------------------------------------------------------------
    # Build feature matrix — TRAIN ONLY (V5.4)
    # -----------------------------------------------------------------------
    # Stride is ALWAYS INFERENCE_STRIDE_REPRO=1 — maximum precision (V5.6).
    # Do NOT add stride overrides; accuracy over speed.
    stride = INFERENCE_STRIDE_REPRO

    print(
        f"[train_lr] Building features from train.csv "
        f"(stride={stride} [REPRO], max_recordings={args.max_recordings})"
    )
    t0 = time.time()
    X_train, y_train = build_feature_matrix(
        model,
        train_csv,
        processed_dir,
        stride=stride,
        device=args.device,
        max_recordings=args.max_recordings,
    )
    elapsed = time.time() - t0
    print(
        f"[train_lr] Feature matrix: X={X_train.shape}, y={y_train.shape} "
        f"({elapsed:.1f}s)"
    )
    print(
        f"[train_lr] Class distribution: "
        f"normal={int((y_train == 0).sum())}, "
        f"acidemia={int((y_train == 1).sum())}"
    )

    # Guard: need at least 2 classes
    if len(np.unique(y_train)) < 2:
        print(
            "[train_lr] WARNING: only one class in feature matrix "
            "(dry-run with too few recordings?). LR skipped."
        )
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Fit Logistic Regression
    # -----------------------------------------------------------------------
    # Assumption: max_iter=1000, default C=1.0, solver='lbfgs' (S10)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    print("[train_lr] Fitting LogisticRegression ...")
    lr.fit(X_train, y_train)
    train_acc = lr.score(X_train, y_train)
    print(f"[train_lr] Train accuracy: {train_acc:.4f}")

    # -----------------------------------------------------------------------
    # Save checkpoint with stride metadata (V5.5)
    # -----------------------------------------------------------------------
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": lr,
        "stride": INFERENCE_STRIDE_REPRO,   # always 1 — locked for Stage 7 evaluation
        "n_train": len(y_train),
        "feature_names": [
            "segment_length",
            "max_prediction",
            "cumulative_sum",
            "weighted_integral",
        ],
    }
    joblib.dump(payload, out_pkl)
    print(f"[train_lr] Saved LR checkpoint: {out_pkl}")
    print("[train_lr] Done.")


if __name__ == "__main__":
    main()
