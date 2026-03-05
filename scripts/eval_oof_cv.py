#!/usr/bin/env python
"""
scripts/eval_oof_cv.py — 5-fold Out-of-Fold evaluation on all 552 recordings
=============================================================================
The most reliable evaluation of the hybrid system (PatchTST + clinical rules).

Every recording is predicted exactly once by a model that was NEVER trained on it.
5-fold OOF gives 10x more test data than the single REPRO_TRACK split, producing
a much tighter confidence interval and a truly unbiased AUC estimate.

FOLD SPLITS
-----------
Calls generate_cv_splits() with seed=42 (same as v7 Azure training) to reproduce
the exact test/val/train assignments that each fold's checkpoint was trained with.
Old fold CSVs in data/splits/e2e_cv/ are ignored (they are from an earlier run).

CACHE
-----
Each fold uses its own checkpoint, but all 5 checkpoints have the same file size
and mtime (downloaded together). The cache key therefore includes the fold index:
  fold{k}_{ckpt_size}_{ckpt_mtime}/

First run: builds 5 x ~18 min = ~90 min of inference. All subsequent runs: <2 min.

ABLATION
--------
- 23 features: PatchTST (12) + Clinical (11)  -- the hybrid
- 12 features: PatchTST only
- 11 features: Clinical only

LABEL MODES
-----------
Default: pH < 7.15  (original CTU-UHB label — includes respiratory acidemia)
--metabolic: (pH < 7.15) AND (BDecf >= 8)  — metabolic acidemia only.
  Respiratory acidemia (CO2 accumulation during contractions, BDecf < 8) is
  relabeled as negative. This matches the clinically dangerous phenotype that
  produces CTG changes. Uses data/processed/ctu_uhb_clinical_full.csv.
  Missing BDecf cases with pH < 7.10 are kept positive (conservative fallback).
  Outputs saved to results/oof_cv_evaluation_metabolic/.

Usage:
    python scripts/eval_oof_cv.py
    python scripts/eval_oof_cv.py --no-cache       # force rebuild all caches
    python scripts/eval_oof_cv.py --metabolic       # metabolic acidemia labels
"""

from __future__ import annotations

import argparse
import sys
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse all helpers from local_eval_cpu — no duplication
from local_eval_cpu import (
    load_model,
    build_raw_cache,
    load_cache,
    save_cache,
    features_from_cache,
    fit_lr,
    predict_lr,
    clinical_threshold,
    bootstrap_auc_ci,
    PROCESSED_ROOT,
    AT_CANDIDATES,
    N_PATCHTST,
    N_CLINICAL_FEATURES,
    ALL_FEATURE_NAMES,
    PATCHTST_FEATURE_NAMES,
    INFERENCE_STRIDE,
)

# ── Constants ──────────────────────────────────────────────────────────────────
SPLITS_DIR   = ROOT / "data" / "splits"
CKPT_DIR     = ROOT / "checkpoints" / "e2e_cv_v7"
OUT_DIR_BASE = ROOT / "results"
OOF_CACHE    = ROOT / "results" / "oof_cv_evaluation" / "cache"  # shared cache
SPLITS_OUT_BASE = ROOT / "results" / "oof_cv_evaluation" / "splits"
N_FOLDS      = 5
SEED         = 42
N_FEATURES   = N_PATCHTST + N_CLINICAL_FEATURES  # 23

# Metabolic label settings
CLINICAL_CSV     = ROOT / "data" / "processed" / "ctu_uhb_clinical_full.csv"
BDECF_THRESHOLD  = 8.0   # mmol/L — standard cutoff for significant metabolic acidosis

# Extended hyperparameter grid — searched jointly (AT x C)
AT_CANDIDATES_OOF = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
C_CANDIDATES      = [0.01, 0.1, 1.0, 10.0]

# 2 global threshold-free features appended to the 23-feature vector
GLOBAL_FEATURE_NAMES = ["overall_mean_prob", "overall_std_prob"]
N_FEATURES_25        = N_FEATURES + len(GLOBAL_FEATURE_NAMES)  # 25
ALL_FEATURE_NAMES_25 = ALL_FEATURE_NAMES + GLOBAL_FEATURE_NAMES


# ── Fold split generation (same logic as run_e2e_cv_v2.py) ────────────────────

def generate_cv_splits(all_df: pd.DataFrame, n_folds: int = 5, seed: int = 42):
    """
    Stratified k-fold split — exact replica of run_e2e_cv_v2.generate_cv_splits.
    Inlined to avoid importing a large Azure-dependent module.

    Returns list of dicts: {fold, train_ids, val_ids, test_ids}
    """
    pos = all_df[all_df["target"] == 1]["id"].tolist()
    neg = all_df[all_df["target"] == 0]["id"].tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    splits = []
    for fold in range(n_folds):
        test_pos = pos[fold::n_folds]
        test_neg = neg[fold::n_folds]
        test_ids = set(test_pos + test_neg)

        remaining_pos = [x for x in pos if x not in test_ids]
        remaining_neg = [x for x in neg if x not in test_ids]

        n_val    = max(1, len(test_ids) // 2)
        val_ids  = set(remaining_pos[:n_val // 2] + remaining_neg[:n_val // 2])
        train_ids = set(all_df["id"].tolist()) - test_ids - val_ids

        splits.append({
            "fold":      fold,
            "train_ids": sorted(train_ids),
            "val_ids":   sorted(val_ids),
            "test_ids":  sorted(test_ids),
        })
    return splits


def load_metabolic_labels(clinical_csv: pathlib.Path, bdecf_th: float = 8.0) -> dict:
    """Return {rec_id (str): new_label (int)} using metabolic acidemia criterion.

    Positive (1) = (pH < 7.15) AND (BDecf >= bdecf_th)
               OR  (pH < 7.10) AND BDecf is missing   [conservative fallback]

    Respiratory acidemia (BDecf < bdecf_th) is relabeled to 0.
    Returns a dict covering all 552 recordings.
    """
    df = pd.read_csv(clinical_csv)
    df["record_id"] = df["record_id"].astype(str)
    labels = {}
    for _, row in df.iterrows():
        ph = row["pH"]
        bd = row["BDecf"]
        bd_missing = pd.isna(bd)
        if bd_missing:
            # fallback: only keep clearly acidemic cases with missing BDecf
            new_label = int(ph < 7.10)
        else:
            new_label = int((ph < 7.15) and (float(bd) >= bdecf_th))
        labels[str(row["record_id"])] = new_label
    return labels


def override_cache_labels(cache: dict, new_labels: dict) -> dict:
    """Update in-memory cache entry labels from new_labels dict.

    Recordings not in new_labels keep their original label (should not happen
    for properly filtered splits, but safe to ignore).
    Returns the same cache dict (mutated in place for efficiency).
    """
    for rec_id, entry in cache.items():
        if rec_id in new_labels:
            entry["label"] = new_labels[rec_id]
    return cache


def ids_to_csv(ids, all_df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write a subset of all_df (filtered to ids) as a CSV for build_raw_cache."""
    id_set = set(str(i) for i in ids)
    sub = all_df[all_df["id"].astype(str).isin(id_set)].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    sub[["id", "target"]].to_csv(path, index=False)


# ── Global threshold-free feature augmentation ────────────────────────────────

def augment_global_features(X: np.ndarray, cache: dict, ids: list) -> np.ndarray:
    """Append overall_mean_prob and overall_std_prob to feature matrix X.

    These are threshold-free statistics over ALL PatchTST windows, complementing
    the existing recording_mean_above_th (which only counts windows above AT).
    Computed from cache in memory — no model re-inference needed.
    """
    rows = []
    for rec_id in ids:
        probs = [p for _, p in cache[rec_id]["scores"]]
        if probs:
            rows.append([float(np.mean(probs)), float(np.std(probs))])
        else:
            rows.append([0.5, 0.0])
    return np.hstack([X, np.array(rows, dtype=np.float32)])


# ── Per-fold cache paths (fold-specific key to avoid collision) ────────────────

def _fold_cache_file(fold_k: int, ckpt_path: pathlib.Path, split_name: str) -> pathlib.Path:
    st  = ckpt_path.stat()
    key = f"fold{fold_k}_{st.st_size}_{int(st.st_mtime)}"
    return OOF_CACHE / key / f"{split_name}_raw.pkl"


def get_fold_cache(
    fold_k: int,
    split_name: str,
    split_csv: pathlib.Path,
    ckpt_path: pathlib.Path,
    model,
    force_rebuild: bool,
) -> dict:
    cache_file = _fold_cache_file(fold_k, ckpt_path, split_name)
    if not force_rebuild:
        data = load_cache(cache_file)
        if data is not None and len(data) > 0:
            print(f"   fold{fold_k}/{split_name}: {len(data)} recordings from cache")
            return data
    label = f"fold{fold_k}/{split_name}"
    return build_raw_cache(model, split_csv, cache_file, label)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold OOF evaluation on all 552 recordings")
    parser.add_argument("--no-cache",  action="store_true", help="Force rebuild all caches")
    parser.add_argument("--metabolic", action="store_true",
                        help=f"Relabel: positive = pH<7.15 AND BDecf>={BDECF_THRESHOLD}. "
                             "Treats respiratory acidemia as normal. "
                             "Outputs to results/oof_cv_evaluation_metabolic/")
    args = parser.parse_args()

    # ── Resolve output directories based on label mode ────────────────────────
    label_mode = "metabolic" if args.metabolic else "original"
    OUT_DIR    = OUT_DIR_BASE / (
        "oof_cv_evaluation_metabolic" if args.metabolic else "oof_cv_evaluation"
    )
    SPLITS_OUT = OUT_DIR / "splits"

    # ── Phase 0: compose all-recordings CSV + generate fold splits ────────────
    tv_csv = SPLITS_DIR / "train_val_test.csv"
    if not tv_csv.exists():
        print("Composing train_val_test.csv from train+val+test...")
        dfs = [pd.read_csv(SPLITS_DIR / f, dtype={"id": str, "target": int})
               for f in ("train.csv", "val.csv", "test.csv")]
        pd.concat(dfs, ignore_index=True)[["id", "target"]].to_csv(tv_csv, index=False)
        print(f"  Written: {tv_csv.relative_to(ROOT)} ({sum(len(d) for d in dfs)} recordings)")

    all_df = pd.read_csv(tv_csv, dtype={"id": str, "target": int})

    # ── Apply metabolic label override if requested ───────────────────────────
    new_labels: dict | None = None
    if args.metabolic:
        if not CLINICAL_CSV.exists():
            print(f"ERROR: clinical CSV not found: {CLINICAL_CSV.relative_to(ROOT)}")
            sys.exit(1)
        new_labels = load_metabolic_labels(CLINICAL_CSV, BDECF_THRESHOLD)
        # Override targets in all_df for stratified fold generation
        all_df["target"] = all_df["id"].map(new_labels).fillna(all_df["target"]).astype(int)
        orig_pos = pd.read_csv(tv_csv, dtype={"id": str, "target": int})["target"].sum()
        print(f"\nLabel mode: METABOLIC  (pH<7.15 AND BDecf>={BDECF_THRESHOLD})")
        print(f"  Original positives (pH<7.15)  : {orig_pos}")
        print(f"  Metabolic positives after remap: {all_df['target'].sum()}")
        print(f"  Reclassified respiratory->neg  : {int(orig_pos) - int(all_df['target'].sum())}")
    else:
        print(f"\nLabel mode: ORIGINAL  (pH<7.15)")

    n_total = len(all_df)
    n_pos   = int(all_df["target"].sum())
    n_neg   = n_total - n_pos
    print(f"\nDataset: {n_total} recordings  ({n_pos} positive / {n_neg} negative)")

    cv_splits = generate_cv_splits(all_df, n_folds=N_FOLDS, seed=SEED)
    print(f"Generated {N_FOLDS} stratified folds:")
    for s in cv_splits:
        k = s["fold"]
        print(f"  fold{k}: train={len(s['train_ids'])}, val={len(s['val_ids'])}, "
              f"test={len(s['test_ids'])}")

    # Write split CSVs (for reproducibility / audit)
    SPLITS_OUT.mkdir(parents=True, exist_ok=True)
    for s in cv_splits:
        k = s["fold"]
        ids_to_csv(s["train_ids"], all_df, SPLITS_OUT / f"fold{k}_train.csv")
        ids_to_csv(s["val_ids"],   all_df, SPLITS_OUT / f"fold{k}_val.csv")
        ids_to_csv(s["test_ids"],  all_df, SPLITS_OUT / f"fold{k}_test.csv")

    # ── Phase 1: per-fold cache + evaluation ─────────────────────────────────
    # Accumulators
    oof_records: list[dict] = []        # one entry per recording
    fold_summaries: list[dict] = []
    all_coef_rows: list[np.ndarray] = []

    t_total = time.time()

    for s in cv_splits:
        k    = s["fold"]
        ckpt = CKPT_DIR / f"fold{k}" / "best_finetune.pt"
        if not ckpt.exists():
            print(f"\n[SKIP] fold{k}: checkpoint not found at {ckpt.relative_to(ROOT)}")
            continue

        print(f"\n{'='*60}")
        print(f"  FOLD {k}   (ckpt: {ckpt.relative_to(ROOT)})")
        print(f"{'='*60}")

        train_csv = SPLITS_OUT / f"fold{k}_train.csv"
        val_csv   = SPLITS_OUT / f"fold{k}_val.csv"
        test_csv  = SPLITS_OUT / f"fold{k}_test.csv"

        # Check if all caches exist before loading model
        all_cached = all(
            _fold_cache_file(k, ckpt, name).exists()
            for name in ("train", "val", "test")
        ) and not args.no_cache

        if all_cached:
            print("  All caches found — skipping model load.")
            model = None
        else:
            print("  Building caches (inference pass)...")
            t0    = time.time()
            model = load_model(ckpt)
            print(f"  Model loaded in {time.time()-t0:.1f}s")

        cache_tr = get_fold_cache(k, "train", train_csv, ckpt, model, args.no_cache)
        cache_vl = get_fold_cache(k, "val",   val_csv,   ckpt, model, args.no_cache)
        cache_te = get_fold_cache(k, "test",  test_csv,  ckpt, model, args.no_cache)

        if model is not None:
            del model
            import gc; gc.collect()

        # Apply metabolic label override if requested (in-memory, does not touch cache files)
        if new_labels is not None:
            override_cache_labels(cache_tr, new_labels)
            override_cache_labels(cache_vl, new_labels)
            override_cache_labels(cache_te, new_labels)

        # ── Joint (AT, C) grid search (instant from cache) ────────────────────
        n_combos = len(AT_CANDIDATES_OOF) * len(C_CANDIDATES)
        print(f"\n  Joint (AT x C) grid search: {n_combos} combos on val={len(cache_vl)}...")
        best_at, best_C, best_val_auc = 0.40, 0.1, 0.0
        for at in AT_CANDIDATES_OOF:
            X_tr_at, y_tr_at, tr_ids_at = features_from_cache(cache_tr, at)
            X_vl_at, y_vl_at, vl_ids_at = features_from_cache(cache_vl, at)
            if len(np.unique(y_tr_at)) < 2 or len(np.unique(y_vl_at)) < 2:
                continue
            # augment with global threshold-free features
            X_tr_aug = augment_global_features(X_tr_at, cache_tr, tr_ids_at)
            X_vl_aug = augment_global_features(X_vl_at, cache_vl, vl_ids_at)
            for C in C_CANDIDATES:
                sc_g, lr_g = fit_lr(X_tr_aug, y_tr_at, C=C)
                vauc = roc_auc_score(y_vl_at, predict_lr(X_vl_aug, sc_g, lr_g))
                print(f"    AT={at:.2f} C={C:5}: val_auc={vauc:.4f}")
                if vauc > best_val_auc:
                    best_val_auc, best_at, best_C = vauc, at, C
        print(f"  Best: AT={best_at:.2f}, C={best_C}, val_auc={best_val_auc:.4f}")

        # ── Final LR: train on train+val, evaluate on test ────────────────────
        # Hyperparameters locked above on val-only; merging train+val for more data
        # is standard practice (select-then-retrain) — no leakage.
        cache_tv = {**cache_tr, **cache_vl}   # 387 + 54 = 441 recordings

        X_tr_at, y_tr_at, tr_ids_at   = features_from_cache(cache_tr,  best_at)
        X_vl_at, y_vl_at, vl_ids_at   = features_from_cache(cache_vl,  best_at)
        X_tv_at, y_tv_at, tv_ids_at   = features_from_cache(cache_tv,  best_at)
        X_te,    y_te,    te_ids       = features_from_cache(cache_te,  best_at)

        # Augment all splits with 2 global features
        X_tr_aug = augment_global_features(X_tr_at, cache_tr, tr_ids_at)
        X_vl_aug = augment_global_features(X_vl_at, cache_vl, vl_ids_at)
        X_tv_aug = augment_global_features(X_tv_at, cache_tv, tv_ids_at)
        X_te_aug = augment_global_features(X_te,    cache_te, te_ids)

        # 25-feature model (train+val)
        sc25, lr25 = fit_lr(X_tv_aug, y_tv_at, C=best_C)
        te_scores_25 = predict_lr(X_te_aug, sc25, lr25)

        # 23-feature baseline (train+val, no global features) for comparison
        sc23, lr23 = fit_lr(X_tv_at, y_tv_at, C=best_C)
        te_scores_23 = predict_lr(X_te, sc23, lr23)

        # Ablation — 12 features (PatchTST only, train+val)
        sc12, lr12 = fit_lr(X_tv_at[:, :N_PATCHTST], y_tv_at, C=best_C)
        te_scores_12 = predict_lr(X_te[:, :N_PATCHTST], sc12, lr12)

        # Ablation — 11 features (clinical only, train+val)
        sc11, lr11 = fit_lr(X_tv_at[:, N_PATCHTST:], y_tv_at, C=best_C)
        te_scores_11 = predict_lr(X_te[:, N_PATCHTST:], sc11, lr11)

        _ok = len(np.unique(y_te)) > 1
        fold_auc_25 = roc_auc_score(y_te, te_scores_25) if _ok else float("nan")
        fold_auc_23 = roc_auc_score(y_te, te_scores_23) if _ok else float("nan")
        fold_auc_12 = roc_auc_score(y_te, te_scores_12) if _ok else float("nan")
        fold_auc_11 = roc_auc_score(y_te, te_scores_11) if _ok else float("nan")

        print(f"\n  Fold {k} test AUCs:  hybrid25={fold_auc_25:.4f}  hybrid23={fold_auc_23:.4f}  "
              f"patchtst={fold_auc_12:.4f}  clinical={fold_auc_11:.4f}")

        # Collect OOF predictions
        for rec_id, y_true, s25, s23, s12, s11 in zip(
            te_ids, y_te, te_scores_25, te_scores_23, te_scores_12, te_scores_11
        ):
            oof_records.append({
                "id": rec_id, "y_true": int(y_true), "fold": k,
                "oof_score_25": float(s25),
                "oof_score_23": float(s23),
                "oof_score_12": float(s12),
                "oof_score_11": float(s11),
            })

        # Collect LR25 coefficients (for averaged feature importance)
        all_coef_rows.append(lr25.coef_[0].copy())

        fold_summaries.append({
            "fold": k,
            "n_test": len(y_te),
            "n_pos":  int(y_te.sum()),
            "n_neg":  int((y_te == 0).sum()),
            "auc_hybrid25":  round(fold_auc_25, 4),
            "auc_hybrid23":  round(fold_auc_23, 4),
            "auc_patchtst":  round(fold_auc_12, 4),
            "auc_clinical":  round(fold_auc_11, 4),
            "best_at":  best_at,
            "lr_c":     best_C,
            "val_auc_grid": round(best_val_auc, 4),
        })

    if not oof_records:
        print("\nNo OOF predictions collected — check checkpoints.")
        sys.exit(1)

    # ── Phase 3: aggregate OOF ────────────────────────────────────────────────
    oof_df = pd.DataFrame(oof_records).sort_values("id").reset_index(drop=True)
    y_all      = oof_df["y_true"].values
    scores_25  = oof_df["oof_score_25"].values   # primary: 25-feature hybrid
    scores_23  = oof_df["oof_score_23"].values   # comparison: 23-feature hybrid
    scores_12  = oof_df["oof_score_12"].values
    scores_11  = oof_df["oof_score_11"].values

    oof_auc_25 = roc_auc_score(y_all, scores_25)
    oof_auc_23 = roc_auc_score(y_all, scores_23)
    oof_auc_12 = roc_auc_score(y_all, scores_12)
    oof_auc_11 = roc_auc_score(y_all, scores_11)
    ci_lo, ci_hi = bootstrap_auc_ci(y_all, scores_25, n=10000)

    th, sens_oof, spec_oof = clinical_threshold(y_all, scores_25)
    y_pred = (scores_25 >= th).astype(int)
    tp = int(((y_pred == 1) & (y_all == 1)).sum())
    tn = int(((y_pred == 0) & (y_all == 0)).sum())
    fp = int(((y_pred == 1) & (y_all == 0)).sum())
    fn = int(((y_pred == 0) & (y_all == 1)).sum())

    elapsed = time.time() - t_total

    # ── Feature importance (averaged across folds, 25-feature model) ─────────
    avg_coefs = np.mean(all_coef_rows, axis=0) if all_coef_rows else np.zeros(N_FEATURES_25)
    fi = sorted(zip(ALL_FEATURE_NAMES_25, avg_coefs), key=lambda x: abs(x[1]), reverse=True)

    # ── Print results ─────────────────────────────────────────────────────────
    n_folds_done = len(fold_summaries)
    n_oof        = len(oof_df)

    SEP = "=" * 66
    DIV = "-" * 66
    print(f"\n{SEP}")
    print(f"  5-FOLD OOF EVALUATION  ({n_oof} recordings, {n_folds_done} folds)")
    print(f"  Label mode: {label_mode.upper()}  "
          + (f"(pH<7.15 AND BDecf>={BDECF_THRESHOLD})" if args.metabolic else "(pH<7.15)"))
    print(f"{SEP}")
    print(f"  Hybrid AUC   (25 feat) : {oof_auc_25:.4f}  [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Hybrid AUC   (23 feat) : {oof_auc_23:.4f}  (prev baseline)")
    print(f"  PatchTST only (12 feat): {oof_auc_12:.4f}")
    print(f"  Clinical only (11 feat): {oof_auc_11:.4f}")
    print(f"  Delta 25-feat vs 23-feat: {oof_auc_25-oof_auc_23:+.4f}")
    print(f"  Delta hybrid  vs AI-only: {oof_auc_25-oof_auc_12:+.4f}")
    print(f"  Paper target            : 0.826")
    print(DIV)
    print(f"  Sensitivity : {sens_oof:.3f}   (clinical threshold, Spec>=0.65)")
    print(f"  Specificity : {spec_oof:.3f}")
    print(f"  Threshold   : {th:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(DIV)
    print(f"  Per-fold breakdown:")
    for fs in fold_summaries:
        print(f"    fold{fs['fold']}: 25f={fs['auc_hybrid25']:.4f}  23f={fs['auc_hybrid23']:.4f}  "
              f"pt={fs['auc_patchtst']:.4f}  clin={fs['auc_clinical']:.4f}  "
              f"(n={fs['n_test']}, AT={fs['best_at']:.2f}, C={fs['lr_c']}, "
              f"val_grid={fs['val_auc_grid']:.4f})")
    print(DIV)
    print(f"  Total elapsed: {elapsed/60:.1f} min")
    print(SEP)

    FI_SEP = "-" * 58
    print(f"\n{FI_SEP}")
    print(f"  Feature Importance -- 25-feat model (avg |coeff|, {n_folds_done} folds)")
    print(FI_SEP)
    for i, (name, coef) in enumerate(fi, 1):
        bar   = '|' * min(int(abs(coef) * 25), 30)
        sign  = '+' if coef >= 0 else '-'
        group = 'Global' if name in GLOBAL_FEATURE_NAMES else ('AI' if name in PATCHTST_FEATURE_NAMES else 'Clinical')
        print(f"  {i:>2}. {name:<35} {sign}{abs(coef):.4f}  {bar}  [{group}]")

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # OUT_DIR is local var (label-mode-specific)

    oof_df.to_csv(OUT_DIR / "oof_predictions.csv", index=False)

    pd.DataFrame(fold_summaries).to_csv(OUT_DIR / "fold_summary.csv", index=False)

    pd.DataFrame([{
        "label_mode":       label_mode,
        "n_recordings":     n_oof,
        "n_folds":          n_folds_done,
        "oof_auc_hybrid25": round(oof_auc_25, 4),
        "oof_auc_hybrid23": round(oof_auc_23, 4),
        "oof_auc_patchtst": round(oof_auc_12, 4),
        "oof_auc_clinical": round(oof_auc_11, 4),
        "delta_25_vs_23":   round(oof_auc_25 - oof_auc_23, 4),
        "delta_hybrid":     round(oof_auc_25 - oof_auc_12, 4),
        "ci_lo":    round(ci_lo, 4),
        "ci_hi":    round(ci_hi, 4),
        "ci_width": round(ci_hi - ci_lo, 4),
        "sens":  round(sens_oof, 3),
        "spec":  round(spec_oof, 3),
        "threshold": round(float(th), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "repro_track_auc": 0.8285,   # single-split reference
        "paper_target":    0.826,
    }]).to_csv(OUT_DIR / "global_summary.csv", index=False)

    pd.DataFrame([
        {"rank": i+1, "feature": name, "avg_coeff": round(float(c), 5),
         "avg_abs_coeff": round(abs(float(c)), 5),
         "group": "Global" if name in GLOBAL_FEATURE_NAMES else
                  ("AI" if name in PATCHTST_FEATURE_NAMES else "Clinical")}
        for i, (name, c) in enumerate(fi)
    ]).to_csv(OUT_DIR / "feature_importance_avg.csv", index=False)

    print(f"\nResults saved to: {OUT_DIR.relative_to(ROOT)}/")
    print(f"  oof_predictions.csv     ({n_oof} rows)")
    print(f"  fold_summary.csv        ({n_folds_done} rows)")
    print(f"  global_summary.csv")
    print(f"  feature_importance_avg.csv")
    print(f"\nTo re-run (instant from cache):")
    print(f"  python scripts/eval_oof_cv.py")
    print(f"  python scripts/eval_oof_cv.py --no-cache  # force rebuild")


if __name__ == "__main__":
    main()
