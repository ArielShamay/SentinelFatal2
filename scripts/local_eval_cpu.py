#!/usr/bin/env python
"""
local_eval_cpu.py — Hybrid evaluation (PatchTST + clinical rules) on CPU
=========================================================================
Runs the full 23-feature pipeline locally without GPU or Azure.
Uses a single best_finetune.pt checkpoint on the 441/56/55 canonical split.

CACHING STRATEGY
----------------
PatchTST inference (the bottleneck) is AT-independent — the same window
probabilities are used regardless of alert threshold. So we run inference
once per recording and cache the raw scores to disk. On subsequent runs
(or when sweeping AT/C), all feature extraction is near-instant.

Cache location: results/local_eval_hybrid/cache/{ckpt_key}/
  train_raw.pkl  — {rec_id: {"scores": [(start, prob), ...], "clinical": [11 floats], "label": int}}
  val_raw.pkl
  test_raw.pkl

The cache key is derived from the checkpoint's size+mtime, so it auto-
invalidates if you swap in a different checkpoint file.

Features (23 total):
  - 12 PatchTST alert features  (from alert_extractor, AT-dependent)
  - 11 clinical rule features   (from clinical_extractor, AT-independent)

Usage:
    python scripts/local_eval_cpu.py
    python scripts/local_eval_cpu.py --ckpt checkpoints/e2e_cv_v7/fold0/best_finetune.pt
    python scripts/local_eval_cpu.py --no-cache   # force rebuild cache
    python scripts/local_eval_cpu.py --lr-c 0.1   # skip inner C sweep
"""

from __future__ import annotations

import argparse
import pickle
import sys
import pathlib
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.patchtst import PatchTST, load_config
from src.model.heads import ClassificationHead
from src.train.finetune import load_pretrained_checkpoint
from src.inference.alert_extractor import extract_recording_features
from src.train.utils import sliding_windows
from src.features.clinical_extractor import (
    extract_clinical_features,
    N_CLINICAL_FEATURES,
    CLINICAL_FEATURE_NAMES,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_CKPTS = [
    ROOT / 'checkpoints' / 'e2e_cv_v7' / 'fold0' / 'best_finetune.pt',
    ROOT / 'checkpoints' / 'e2e_cv_v7' / 'fold1' / 'best_finetune.pt',
    ROOT / 'checkpoints' / 'e2e_cv_v2' / 'config_A' / 'best_finetune.pt',
    ROOT / 'checkpoints' / 'e2e_cv_v2' / 'fold0' / 'finetune' / 'best_finetune.pt',
    ROOT / 'checkpoints' / 'finetune'  / 'best_finetune.pt',
]

SPLITS_DIR     = ROOT / 'data' / 'splits'
PROCESSED_ROOT = ROOT / 'data' / 'processed'
CONFIG_PATH    = ROOT / 'config' / 'train_config.yaml'
OUT_DIR        = ROOT / 'results' / 'local_eval_hybrid'
CACHE_BASE     = OUT_DIR / 'cache'

AT_CANDIDATES    = [0.30, 0.35, 0.40, 0.45]
N_PATCHTST       = 12
N_FEATURES       = N_PATCHTST + N_CLINICAL_FEATURES   # 23
INFERENCE_STRIDE = 24
BATCH_SIZE       = 64
SPEC_CONSTRAINT  = 0.65
DEVICE           = 'cpu'

PATCHTST_FEATURE_NAMES = [
    "segment_length", "max_prediction", "cumulative_sum", "weighted_integral",
    "n_alert_segments", "alert_fraction", "mean_prediction", "std_prediction",
    "max_pred_all_segments", "total_alert_duration",
    "recording_max_score", "recording_mean_above_th",
]
ALL_FEATURE_NAMES = PATCHTST_FEATURE_NAMES + CLINICAL_FEATURE_NAMES  # 23


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _ckpt_key(ckpt_path: pathlib.Path) -> str:
    """Stable cache key from checkpoint size + mtime (auto-invalidates on change)."""
    st = ckpt_path.stat()
    return f"{st.st_size}_{int(st.st_mtime)}"


def _cache_path(ckpt_path: pathlib.Path, split_name: str) -> pathlib.Path:
    return CACHE_BASE / _ckpt_key(ckpt_path) / f"{split_name}_raw.pkl"


def load_cache(cache_file: pathlib.Path) -> dict | None:
    """Load cache if it exists. Returns None if missing or corrupted."""
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"   [cache] Could not load {cache_file.name}: {e} — will rebuild.")
        return None


def save_cache(cache_file: pathlib.Path, data: dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(ckpt_path: pathlib.Path) -> torch.nn.Module:
    cfg   = load_config(CONFIG_PATH, overrides={'loss': 'cross_entropy'})
    model = PatchTST(cfg).to(DEVICE)
    d_in  = (int(cfg['data']['n_patches']) *
             int(cfg['model']['d_model'])  *
             int(cfg['data']['n_channels']))
    model.replace_head(ClassificationHead(
        d_in=d_in,
        n_classes=int(cfg['finetune']['n_classes']),
        dropout=float(cfg['model']['dropout']),
    ))
    # Load the full fine-tuned checkpoint (backbone + classification head).
    # NOTE: do NOT use load_pretrained_checkpoint() here — that function strips
    # the head weights (designed for pretrain→finetune hand-off only).
    state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print(f"[load_checkpoint] Loaded full finetune checkpoint ({len(state_dict)} tensors) from {ckpt_path}")
    model.eval()
    return model


# ── Phase 1: build raw cache (inference + clinical, AT-independent) ────────────

def build_raw_cache(
    model: torch.nn.Module,
    split_csv: pathlib.Path,
    cache_file: pathlib.Path,
    split_label: str,
) -> dict:
    """
    Run PatchTST inference and clinical feature extraction for every recording
    in split_csv. Results are AT-independent and saved to cache_file.

    Cache format:
        {rec_id: {"scores": List[Tuple[int,float]], "clinical": List[float], "label": int}}
    """
    df = pd.read_csv(split_csv, dtype={'id': str, 'target': int})
    cache: dict = {}
    n_total = len(df)
    t_split = time.time()

    with torch.no_grad():
        for i, (_, row) in enumerate(df.iterrows(), 1):
            rec_id = str(row['id'])
            label  = int(row['target'])

            npy = PROCESSED_ROOT / 'ctu_uhb' / f'{rec_id}.npy'
            if not npy.exists():
                npy = PROCESSED_ROOT / f'{rec_id}.npy'
            if not npy.exists():
                continue

            signal  = np.load(npy, mmap_mode='r')
            windows = sliding_windows(signal, window_len=1800, stride=INFERENCE_STRIDE)
            if not windows:
                continue

            # PatchTST inference — raw window scores (AT-independent)
            scores_list: list[tuple[int, float]] = []
            for b in range(0, len(windows), BATCH_SIZE):
                batch  = torch.stack(windows[b:b + BATCH_SIZE]).to(DEVICE)
                logits = model(batch)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for j, p in enumerate(probs):
                    scores_list.append(((b + j) * INFERENCE_STRIDE, float(p)))

            # Clinical features (also AT-independent)
            clinical_feats = extract_clinical_features(signal, fs=4.0)

            cache[rec_id] = {
                "scores":   scores_list,
                "clinical": clinical_feats,
                "label":    label,
            }

            # Progress every 50 recordings
            if i % 50 == 0 or i == n_total:
                elapsed = time.time() - t_split
                rate    = i / elapsed
                eta     = (n_total - i) / rate if rate > 0 else 0
                print(f"   {split_label}: {i}/{n_total}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    save_cache(cache_file, cache)
    print(f"   Cached {len(cache)}/{n_total} recordings -> {cache_file.relative_to(ROOT)}")
    return cache


def get_raw_cache(
    model: torch.nn.Module | None,
    split_csv: pathlib.Path,
    cache_file: pathlib.Path,
    split_label: str,
    force_rebuild: bool = False,
) -> dict:
    """Return raw cache, loading from disk if available or building if needed."""
    if not force_rebuild:
        cached = load_cache(cache_file)
        if cached is not None:
            print(f"   {split_label}: loaded {len(cached)} recordings from cache "
                  f"({cache_file.name})")
            return cached

    if model is None:
        raise RuntimeError(
            f"Cache missing for {split_label} and no model provided to rebuild it."
        )
    print(f"   {split_label}: building cache ({cache_file.name})...")
    return build_raw_cache(model, split_csv, cache_file, split_label)


# ── Phase 2: apply AT threshold to cached data (instant) ──────────────────────

def features_from_cache(
    cache: dict,
    at: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Apply alert threshold to cached raw scores → 23-feature matrix.
    This is near-instant (no model inference needed).
    """
    X_rows, y_rows, ids = [], [], []
    for rec_id, entry in cache.items():
        patchtst_feats = extract_recording_features(
            entry["scores"],
            threshold=at,
            inference_stride=INFERENCE_STRIDE,
            n_features=N_PATCHTST,
        )
        X_rows.append(list(patchtst_feats.values()) + entry["clinical"])
        y_rows.append(entry["label"])
        ids.append(rec_id)
    return np.array(X_rows), np.array(y_rows), ids


# ── LR helpers ─────────────────────────────────────────────────────────────────

def fit_lr(X_tr: np.ndarray, y_tr: np.ndarray, C: float = 0.1):
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(Xs, y_tr)
    return sc, lr


def predict_lr(X: np.ndarray, sc: StandardScaler, lr: LogisticRegression) -> np.ndarray:
    return lr.predict_proba(sc.transform(X))[:, 1]


# ── Clinical threshold ─────────────────────────────────────────────────────────

def clinical_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    thresholds = np.unique(y_score)
    best_th, best_sens, best_spec = thresholds[0], 0.0, 0.0
    for th in thresholds:
        pred = (y_score >= th).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if spec >= SPEC_CONSTRAINT and sens > best_sens:
            best_th, best_sens, best_spec = th, sens, spec

    if best_sens == 0.0:
        print('[threshold] No threshold meets Spec>=0.65 — falling back to Youden')
        j = []
        for th in thresholds:
            pred = (y_score >= th).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            s  = tp / max(tp + fn, 1)
            sp = tn / max(tn + fp, 1)
            j.append((s + sp - 1, th, s, sp))
        j.sort(reverse=True)
        _, best_th, best_sens, best_spec = j[0]
    return float(best_th), float(best_sens), float(best_spec)


# ── Feature importance ─────────────────────────────────────────────────────────

def print_feature_importance(lr: LogisticRegression) -> None:
    coefs = sorted(zip(ALL_FEATURE_NAMES, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n{'-'*58}")
    print(f"  Feature Importance  (|LR coefficient|, descending)")
    print(f"{'-'*58}")
    for i, (name, coef) in enumerate(coefs, 1):
        bar   = '|' * min(int(abs(coef) * 25), 30)
        sign  = '+' if coef >= 0 else '-'
        group = 'AI' if name in PATCHTST_FEATURE_NAMES else 'Clinical'
        print(f"  {i:>2}. {name:<35} {sign}{abs(coef):.4f}  {bar}  [{group}]")


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                     n: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid CTG evaluation on CPU (with caching)")
    parser.add_argument('--ckpt',        type=str,   default=None,
                        help='Path to best_finetune.pt (default: auto-detect v7)')
    parser.add_argument('--lr-c',        type=float, default=None,
                        help='Fix LR C value. Default: inner sweep [0.01,0.1,1.0] on val')
    parser.add_argument('--no-cache',    action='store_true',
                        help='Force rebuild of feature cache (ignore existing files)')
    parser.add_argument('--split-train', type=str,   default=None)
    parser.add_argument('--split-val',   type=str,   default=None)
    parser.add_argument('--split-test',  type=str,   default=None)
    args = parser.parse_args()

    # ── Find checkpoint ────────────────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = pathlib.Path(args.ckpt).resolve()
    else:
        ckpt_path = next((p for p in DEFAULT_CKPTS if p.exists()), None)

    if ckpt_path is None or not ckpt_path.exists():
        print('No checkpoint found!')
        print('  python scripts/download_run_artifacts.py --run-id <run-id>')
        sys.exit(1)

    size_mb = ckpt_path.stat().st_size / (1024 ** 2)
    print(f'Checkpoint : {ckpt_path.relative_to(ROOT)}  ({size_mb:.1f} MB)')
    print(f'Cache key  : {_ckpt_key(ckpt_path)}')

    # ── Split files ────────────────────────────────────────────────────────────
    train_csv = pathlib.Path(args.split_train) if args.split_train else SPLITS_DIR / 'train.csv'
    val_csv   = pathlib.Path(args.split_val)   if args.split_val   else SPLITS_DIR / 'val.csv'
    test_csv  = pathlib.Path(args.split_test)  if args.split_test  else SPLITS_DIR / 'test.csv'
    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            print(f'Missing split file: {p}')
            sys.exit(1)

    n_tr   = len(pd.read_csv(train_csv))
    n_val  = len(pd.read_csv(val_csv))
    n_test = len(pd.read_csv(test_csv))
    print(f'Splits     : train={n_tr}, val={n_val}, test={n_test}')
    print(f'Features   : {N_PATCHTST} PatchTST + {N_CLINICAL_FEATURES} clinical = {N_FEATURES} total')

    # ── Phase 1: build/load raw feature cache ─────────────────────────────────
    cache_tr_file = _cache_path(ckpt_path, 'train')
    cache_vl_file = _cache_path(ckpt_path, 'val')
    cache_te_file = _cache_path(ckpt_path, 'test')

    all_cached = (cache_tr_file.exists() and
                  cache_vl_file.exists() and
                  cache_te_file.exists() and
                  not args.no_cache)

    if all_cached:
        print('\nCache found — skipping model inference.')
        model = None
    else:
        print('\nBuilding feature cache (runs model inference once per recording)...')
        print(f'  Estimated time: ~10-20 min on CPU for {n_tr+n_val+n_test} recordings')
        t0    = time.time()
        model = load_model(ckpt_path)
        print(f'  Model loaded in {time.time()-t0:.1f}s')

    print('\nLoading / building caches:')
    cache_tr = get_raw_cache(model, train_csv, cache_tr_file, 'train', args.no_cache)
    cache_vl = get_raw_cache(model, val_csv,   cache_vl_file, 'val',   args.no_cache)
    cache_te = get_raw_cache(model, test_csv,  cache_te_file, 'test',  args.no_cache)

    # Free model memory
    if model is not None:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Phase 2: AT sweep (instant from cache) ─────────────────────────────────
    print(f'\nAT sweep on val={n_val} (instant from cache)...')
    at_results  = {}
    best_at     = 0.40
    best_auc_at = 0.0
    c_sweep     = args.lr_c if args.lr_c is not None else 0.1

    for at in AT_CANDIDATES:
        X_tr, y_tr, _  = features_from_cache(cache_tr, at)
        X_vl, y_vl, _  = features_from_cache(cache_vl, at)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_vl)) < 2:
            at_results[at] = 0.0
            continue

        sc_at, lr_at = fit_lr(X_tr, y_tr, C=c_sweep)
        val_auc = roc_auc_score(y_vl, predict_lr(X_vl, sc_at, lr_at))
        at_results[at] = val_auc
        print(f'  AT={at:.2f} -> val_auc={val_auc:.4f}')

        if val_auc > best_auc_at:
            best_auc_at = val_auc
            best_at     = at

    print(f'\nBest AT = {best_at:.2f}  (val_auc={best_auc_at:.4f})')

    # ── Phase 3: select best C (instant from cache) ────────────────────────────
    X_tr, y_tr, _      = features_from_cache(cache_tr, best_at)
    X_vl, y_vl, _      = features_from_cache(cache_vl, best_at)
    X_te, y_te, te_ids = features_from_cache(cache_te, best_at)
    print(f'Features extracted: train={len(y_tr)}, val={len(y_vl)}, test={len(y_te)}, '
          f'shape={X_tr.shape[1]}')

    if args.lr_c is not None:
        best_C = args.lr_c
        print(f'\nUsing C={best_C} (--lr-c)')
    else:
        print('\nInner C sweep on val [0.01, 0.1, 1.0]...')
        best_C, best_C_auc = 0.1, 0.0
        for C in [0.01, 0.1, 1.0]:
            sc_c, lr_c = fit_lr(X_tr, y_tr, C=C)
            vauc = roc_auc_score(y_vl, predict_lr(X_vl, sc_c, lr_c))
            print(f'  C={C}: val_auc={vauc:.4f}')
            if vauc > best_C_auc:
                best_C_auc, best_C = vauc, C
        print(f'Best C = {best_C}')

    # ── Phase 4: final LR + test evaluation ────────────────────────────────────
    sc, lr      = fit_lr(X_tr, y_tr, C=best_C)
    val_scores  = predict_lr(X_vl, sc, lr)
    test_scores = predict_lr(X_te, sc, lr)

    th, sens_v, spec_v = clinical_threshold(y_vl, val_scores)
    print(f'\nVal threshold: {th:.4f}  sens={sens_v:.3f}  spec={spec_v:.3f}')

    test_auc     = roc_auc_score(y_te, test_scores) if len(np.unique(y_te)) > 1 else 0.0
    ci_lo, ci_hi = bootstrap_auc_ci(y_te, test_scores, n=10000)

    y_pred   = (test_scores >= th).astype(int)
    tp = int(((y_pred == 1) & (y_te == 1)).sum())
    tn = int(((y_pred == 0) & (y_te == 0)).sum())
    fp = int(((y_pred == 1) & (y_te == 0)).sum())
    fn = int(((y_pred == 0) & (y_te == 1)).sum())
    test_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f'\n{"="*60}')
    print(f'  HYBRID SYSTEM RESULTS  (PatchTST + Clinical Rules)')
    print(f'  Checkpoint : {ckpt_path.name}')
    print(f'{"="*60}')
    print(f'  AUC         : {test_auc:.4f}  [95% CI: {ci_lo:.4f}, {ci_hi:.4f}]')
    print(f'  Sensitivity : {test_sens:.3f}')
    print(f'  Specificity : {test_spec:.3f}')
    print(f'  Threshold   : {th:.4f}')
    print(f'  AT          : {best_at:.2f}')
    print(f'  LR C        : {best_C}')
    print(f'  TP={tp}  TN={tn}  FP={fp}  FN={fn}')
    print(f'{"="*60}')
    print(f'  v7 REPRO_TRACK AUC : 0.7934  (12 features, PatchTST only)')
    print(f'  v6 OOF AUC         : 0.6329  (12 features, PatchTST only)')
    print(f'  Paper target       : 0.826')

    # ── Feature importance ─────────────────────────────────────────────────────
    print_feature_importance(lr)

    # ── Save results ───────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'id': te_ids, 'y_true': y_te, 'y_score': test_scores,
        'y_pred': y_pred, 'threshold': th,
    }).to_csv(OUT_DIR / 'predictions_test.csv', index=False)

    pd.DataFrame([{
        'ckpt':       str(ckpt_path.relative_to(ROOT)),
        'n_features': N_FEATURES,
        'best_at':    best_at,
        'lr_c':       best_C,
        'test_auc':   round(test_auc, 4),
        'ci_lo':      round(ci_lo, 4),
        'ci_hi':      round(ci_hi, 4),
        'test_sens':  round(test_sens, 3),
        'test_spec':  round(test_spec, 3),
        'threshold':  round(float(th), 4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }]).to_csv(OUT_DIR / 'summary.csv', index=False)

    pd.DataFrame([
        {'at': k, 'val_auc': round(v, 4)} for k, v in sorted(at_results.items())
    ]).to_csv(OUT_DIR / 'at_sweep.csv', index=False)

    coefs = sorted(zip(ALL_FEATURE_NAMES, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
    pd.DataFrame([
        {'rank': i+1, 'feature': name, 'coeff': round(c, 5),
         'abs_coeff': round(abs(c), 5),
         'group': 'AI' if name in PATCHTST_FEATURE_NAMES else 'Clinical'}
        for i, (name, c) in enumerate(coefs)
    ]).to_csv(OUT_DIR / 'feature_importance.csv', index=False)

    print(f'\nResults saved to: {OUT_DIR.relative_to(ROOT)}/')
    print(f'  predictions_test.csv, summary.csv, at_sweep.csv, feature_importance.csv')
    print(f'\nTo re-run with different AT/C (uses cache, instant):')
    print(f'  python scripts/local_eval_cpu.py --lr-c 1.0')
    print(f'  python scripts/local_eval_cpu.py --no-cache   # force rebuild')


if __name__ == '__main__':
    main()
