#!/usr/bin/env python
"""
scripts/evaluate_clinical_rules.py — Local Standalone Evaluation of Clinical Rules (v8)
========================================================================================
Purpose:
  Evaluate the 11 clinical features (the "clinical brain" of v8) in isolation —
  completely independent of PatchTST, Azure, or any GPU.

  This script answers: "Do the clinical rules have predictive power on their own?"

Outputs:
  1. Sanity check — readable feature dump for 3 Acidemia + 3 Normal cases.
  2. Standalone LR ROC AUC (5-Fold CV) over all recordings.

Run locally:
    python scripts/evaluate_clinical_rules.py

No GPU required. No Azure connection. No pretrained model needed.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── Root path setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Project imports ──────────────────────────────────────────────────────────
from src.features.clinical_extractor import (
    extract_clinical_features,
    CLINICAL_FEATURE_NAMES,
    N_CLINICAL_FEATURES,
)

# ── Suppress noisy warnings ──────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

# ── Data paths ────────────────────────────────────────────────────────────────
PROCESSED_DIR = ROOT / "data" / "processed" / "ctu_uhb"
SPLITS_DIR    = ROOT / "data" / "splits"
SEED          = 42


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Load all labeled recordings
# ════════════════════════════════════════════════════════════════════════════

def load_all_labeled_recordings() -> pd.DataFrame:
    """Merge train / val / test splits into a single labeled DataFrame.

    Expected columns: id (str), target (int: 1=Acidemia, 0=Normal).
    Only keeps recordings for which a .npy file actually exists.
    """
    dfs = []
    for fname in ("train.csv", "val.csv", "test.csv"):
        csv_path = SPLITS_DIR / fname
        if csv_path.exists():
            df = pd.read_csv(csv_path, dtype={"id": str, "target": int})
            dfs.append(df[["id", "target"]])

    if not dfs:
        raise FileNotFoundError(
            f"No split CSVs found in {SPLITS_DIR}. "
            "Expected train.csv / val.csv / test.csv."
        )

    full_df = pd.concat(dfs, ignore_index=True).drop_duplicates("id")
    log.info(f"[load] Found {len(full_df)} labeled recordings "
             f"({full_df['target'].sum()} Acidemia, "
             f"{(full_df['target']==0).sum()} Normal)")

    # Verify .npy files exist
    missing = [r for r in full_df["id"] if not (PROCESSED_DIR / f"{r}.npy").exists()]
    if missing:
        log.warning(f"[load] {len(missing)} recording(s) missing from {PROCESSED_DIR} — skipping")
        full_df = full_df[~full_df["id"].isin(missing)]

    log.info(f"[load] {len(full_df)} recordings ready for feature extraction")
    return full_df.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Extract clinical features for all recordings
# ════════════════════════════════════════════════════════════════════════════

def extract_all_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run extract_clinical_features on every recording.

    Returns:
        X   — float array (N, 11)
        y   — int array   (N,)      1=Acidemia, 0=Normal
        ids — list of recording IDs (N,)
    """
    X_rows, y_rows, ids = [], [], []
    total = len(df)

    log.info(f"[extract] Starting clinical feature extraction for {total} recordings …")

    for idx, row in df.iterrows():
        rec_id = str(row["id"])
        label  = int(row["target"])
        npy    = PROCESSED_DIR / f"{rec_id}.npy"

        try:
            signal = np.load(npy, mmap_mode="r")   # shape (2, T), normalized
        except Exception as exc:
            log.warning(f"[extract] Cannot read {npy}: {exc}")
            continue

        feats = extract_clinical_features(signal, fs=4.0)   # list of 11 floats

        X_rows.append(feats)
        y_rows.append(label)
        ids.append(rec_id)

        if (idx + 1) % 100 == 0:
            log.info(f"[extract]   … {idx + 1}/{total} done")

    log.info(f"[extract] Done — extracted features for {len(X_rows)} recordings.")
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int), ids


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Clinical Sanity Check
# ════════════════════════════════════════════════════════════════════════════

# Friendly display labels for each of the 11 features
FEATURE_DISPLAY = {
    "baseline_bpm":               ("Baseline FHR",            "bpm"),
    "is_tachycardia":             ("Tachycardia detected",     "0/1"),
    "is_bradycardia":             ("Bradycardia detected",     "0/1"),
    "variability_amplitude_bpm":  ("Variability amplitude",   "bpm"),
    "variability_category":       ("Variability category",    "1=minimal 2=moderate 3=marked"),
    "n_late_decelerations":       ("Late decelerations",      "count"),
    "n_variable_decelerations":   ("Variable decelerations",  "count"),
    "n_prolonged_decelerations":  ("Prolonged decelerations", "count"),
    "max_deceleration_depth_bpm": ("Max decel depth",         "bpm"),
    "sinusoidal_detected":        ("Sinusoidal pattern",      "0/1"),
    "tachysystole_detected":      ("Tachysystole (UC)",       "0/1"),
}

VARIABILITY_CAT_NAMES = {1.0: "minimal (<5 bpm)", 2.0: "moderate (5–25 bpm)", 3.0: "marked (>25 bpm)"}


def print_sanity_check(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    n_per_class: int = 3,
    rng_seed: int = SEED,
) -> None:
    """Print readable clinical features for n random Acidemia + n Normal cases."""
    rng = np.random.default_rng(rng_seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    sample_pos = rng.choice(pos_idx, size=min(n_per_class, len(pos_idx)), replace=False)
    sample_neg = rng.choice(neg_idx, size=min(n_per_class, len(neg_idx)), replace=False)

    sep = "─" * 66

    print("\n")
    print("=" * 66)
    print("  CLINICAL SANITY CHECK — Feature Values per Recording")
    print("=" * 66)

    for group_label, indices in [("🔴  ACIDEMIA (Positive)", sample_pos),
                                  ("🟢  NORMAL  (Negative)", sample_neg)]:
        print(f"\n{group_label}")
        print(sep)

        for i in indices:
            rec_id = ids[i]
            feats  = X[i]
            print(f"\n  Recording ID: {rec_id}")
            for j, feat_name in enumerate(CLINICAL_FEATURE_NAMES):
                val = feats[j]
                label_text, unit = FEATURE_DISPLAY[feat_name]
                # Extra annotation for variability category
                if feat_name == "variability_category":
                    cat_name = VARIABILITY_CAT_NAMES.get(val, f"cat={val}")
                    print(f"    {label_text:<30} {val:>6.1f}   ({cat_name})")
                else:
                    print(f"    {label_text:<30} {val:>6.1f}   [{unit}]")
        print(f"\n{sep}")

    print()


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — Standalone Logistic Regression (5-Fold CV)
# ════════════════════════════════════════════════════════════════════════════

def evaluate_standalone_lr(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    rng_seed: int = SEED,
) -> float:
    """Train a Logistic Regression using only the 11 clinical features.

    Uses stratified k-fold CV. Returns mean ROC AUC across folds.
    Prints per-fold results and a summary table.
    """
    from sklearn.linear_model      import LogisticRegression
    from sklearn.metrics           import roc_auc_score
    from sklearn.preprocessing     import StandardScaler
    from sklearn.model_selection   import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)

    fold_aucs    = []
    fold_details = []

    print("=" * 66)
    print(f"  STANDALONE LR EVALUATION — {n_splits}-Fold Stratified CV")
    print(f"  Input features: 11 clinical rules (NO PatchTST, NO GPU)")
    print("=" * 66)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Normalize
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # Train
        lr = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",   # important: ~25% positive rate
            solver="lbfgs",
            C=1.0,
            random_state=rng_seed,
        )
        lr.fit(X_tr_sc, y_tr)

        # Evaluate
        y_proba = lr.predict_proba(X_te_sc)[:, 1]
        auc = roc_auc_score(y_te, y_proba)

        n_pos_test = y_te.sum()
        n_neg_test = (y_te == 0).sum()
        fold_aucs.append(auc)
        fold_details.append((fold, n_pos_test, n_neg_test, auc))

        print(f"  Fold {fold}:  test={len(y_te)} "
              f"(+{n_pos_test} / -{n_neg_test})   AUC = {auc:.4f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))

    print()
    print("─" * 66)
    print(f"  Mean AUC : {mean_auc:.4f}")
    print(f"  Std  AUC : {std_auc:.4f}")
    print(f"  Min  AUC : {min(fold_aucs):.4f}")
    print(f"  Max  AUC : {max(fold_aucs):.4f}")
    print("─" * 66)

    # Interpretation guidance
    print()
    if mean_auc >= 0.75:
        verdict = "EXCELLENT — clinical rules carry strong standalone signal."
    elif mean_auc >= 0.65:
        verdict = "GOOD — clinical rules add meaningful signal to PatchTST."
    elif mean_auc >= 0.55:
        verdict = "MODERATE — some signal; hybrid benefit is plausible."
    else:
        verdict = "WEAK — features may not be well calibrated or data is too noisy."

    print(f"  Verdict:  {verdict}")
    print()

    return mean_auc


# ════════════════════════════════════════════════════════════════════════════
# Step 5 — Feature importance (Logistic Regression coefficients)
# ════════════════════════════════════════════════════════════════════════════

def print_feature_importance(X: np.ndarray, y: np.ndarray) -> None:
    """Fit one LR on the full dataset and print signed coefficients."""
    from sklearn.linear_model  import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X)
    lr       = LogisticRegression(max_iter=2000, class_weight="balanced",
                                  solver="lbfgs", C=1.0, random_state=SEED)
    lr.fit(X_sc, y)
    coefs    = lr.coef_[0]
    ordering = np.argsort(np.abs(coefs))[::-1]

    print("=" * 66)
    print("  FEATURE IMPORTANCE (LR coefficients — full dataset)")
    print("  Positive = associated with Acidemia | Negative = Normal")
    print("=" * 66)
    for rank, idx in enumerate(ordering, 1):
        name  = CLINICAL_FEATURE_NAMES[idx]
        coef  = coefs[idx]
        bar   = "+" * int(abs(coef) * 6) if coef > 0 else "-" * int(abs(coef) * 6)
        print(f"  {rank:2d}. {name:<35}  coef={coef:+.3f}  {bar}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   SentinelFatal2 — Clinical Rules Standalone Evaluation     ║")
    print("║   v8 Clinical Brain — Local CPU Only                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Step 1: Load data ────────────────────────────────────────────────────
    log.info("Step 1/4 — Loading labeled recordings …")
    df = load_all_labeled_recordings()

    # ── Step 2: Extract clinical features ───────────────────────────────────
    log.info("Step 2/4 — Extracting 11 clinical features per recording …")
    X, y, ids = extract_all_features(df)

    log.info(f"Feature matrix: X={X.shape}, y={y.shape} "
             f"(prevalence={y.mean()*100:.1f}% Acidemia)")

    # ── Step 3: Sanity check ─────────────────────────────────────────────────
    log.info("Step 3/4 — Printing clinical sanity check …")
    print_sanity_check(X, y, ids, n_per_class=3)

    # ── Step 4: Standalone LR AUC ────────────────────────────────────────────
    log.info("Step 4/4 — Computing standalone LR AUC (5-Fold CV) …")
    mean_auc = evaluate_standalone_lr(X, y, n_splits=5)

    # ── Step 5: Feature importance ───────────────────────────────────────────
    print_feature_importance(X, y)

    # ── Final summary ────────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║   FINAL RESULT:  Clinical Rules Standalone AUC = {mean_auc:.4f}    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
