#!/usr/bin/env python
"""
local_eval_cpu.py — הרצת AT-sweep + LR + threshold evaluation מקומית על CPU
=============================================================================
משתמש ב-best_finetune.pt הנמצא בתיקייה ומריץ את כל ה-pipeline
ללא GPU, ללא Colab.

שימוש:
    python scripts/local_eval_cpu.py
    python scripts/local_eval_cpu.py --ckpt checkpoints/e2e_cv_v2/config_A/best_finetune.pt

פלט:
    הדפסה לטרמינל + שמירה ב- results/local_eval_cpu/
"""

import argparse, sys, pathlib, time
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

# ── Paths ─────────────────────────────────────────────────────────────────────
# עדיפות: e2e_cv_v2 > finetune (ישן)
DEFAULT_CKPTS = [
    ROOT / 'checkpoints' / 'e2e_cv_v2' / 'config_A' / 'best_finetune.pt',
    ROOT / 'checkpoints' / 'e2e_cv_v2' / 'fold0'    / 'finetune' / 'best_finetune.pt',
    ROOT / 'checkpoints' / 'e2e_cv'    / 'fold0'    / 'finetune' / 'best_finetune.pt',
    ROOT / 'checkpoints' / 'finetune'  / 'best_finetune.pt',
]

SPLITS_DIR     = ROOT / 'data' / 'splits'
PROCESSED_ROOT = ROOT / 'data' / 'processed'
CONFIG_PATH    = ROOT / 'config' / 'train_config.yaml'
OUT_DIR        = ROOT / 'results' / 'local_eval_cpu'

AT_CANDIDATES  = [0.30, 0.35, 0.40, 0.45]
N_FEATURES     = 6
INFERENCE_STRIDE = 24
BATCH_SIZE     = 64
SPEC_CONSTRAINT = 0.65

DEVICE = 'cpu'


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_path: pathlib.Path) -> torch.nn.Module:
    cfg = load_config(CONFIG_PATH, overrides={'loss': 'cross_entropy'})
    model = PatchTST(cfg).to(DEVICE)
    d_in = (int(cfg['data']['n_patches']) *
            int(cfg['model']['d_model'])  *
            int(cfg['data']['n_channels']))
    model.replace_head(ClassificationHead(
        d_in=d_in,
        n_classes=int(cfg['finetune']['n_classes']),
        dropout=float(cfg['model']['dropout']),
    ))
    load_pretrained_checkpoint(model, ckpt_path, torch.device(DEVICE))
    model.eval()
    return model


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(model, split_csv: pathlib.Path,
                     alert_threshold: float, n_features: int = N_FEATURES):
    df = pd.read_csv(split_csv, dtype={'id': str, 'target': int})
    X_rows, y_rows, ids = [], [], []

    with torch.no_grad():
        for _, row in df.iterrows():
            rec_id = str(row['id'])
            label  = int(row['target'])

            # Try ctu_uhb first, then root of processed
            npy = PROCESSED_ROOT / 'ctu_uhb' / f'{rec_id}.npy'
            if not npy.exists():
                npy = PROCESSED_ROOT / f'{rec_id}.npy'
            if not npy.exists():
                continue

            signal  = np.load(npy, mmap_mode='r')
            windows = sliding_windows(signal, window_len=1800, stride=INFERENCE_STRIDE)
            if not windows:
                continue

            scores_list = []
            for i in range(0, len(windows), BATCH_SIZE):
                batch  = torch.stack(windows[i:i + BATCH_SIZE]).to(DEVICE)
                logits = model(batch)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                for j, p in enumerate(probs):
                    scores_list.append(((i + j) * INFERENCE_STRIDE, float(p)))

            feats = extract_recording_features(
                scores_list,
                threshold=alert_threshold,
                inference_stride=INFERENCE_STRIDE,
                n_features=n_features,
            )
            X_rows.append(list(feats.values()))
            y_rows.append(label)
            ids.append(rec_id)

    return np.array(X_rows), np.array(y_rows), ids


# ── LR ────────────────────────────────────────────────────────────────────────
def fit_lr(X_tr, y_tr, C=0.1):
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(Xs, y_tr)
    return sc, lr

def predict_lr(X, sc, lr):
    return lr.predict_proba(sc.transform(X))[:, 1]


# ── Clinical threshold ────────────────────────────────────────────────────────
def clinical_threshold(y_true, y_score):
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
        # Youden fallback
        print('[threshold] אין threshold שעומד ב-Spec≥0.65 — fallback ל-Youden')
        j = []
        for th in thresholds:
            pred = (y_score >= th).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            s  = tp / max(tp + fn, 1)
            sc_val = tn / max(tn + fp, 1)
            j.append((s + sc_val - 1, th, s, sc_val))
        j.sort(reverse=True)
        _, best_th, best_sens, best_spec = j[0]

    return float(best_th), float(best_sens), float(best_spec)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None,
                        help='נתיב לצ\'קפוינט. ברירת מחדל: חיפוש אוטומטי')
    args = parser.parse_args()

    # ── Find checkpoint ───────────────────────────────────────────────────────
    if args.ckpt:
        ckpt_path = pathlib.Path(args.ckpt)
    else:
        ckpt_path = None
        for p in DEFAULT_CKPTS:
            if p.exists():
                ckpt_path = p
                break

    if ckpt_path is None or not ckpt_path.exists():
        print('❌ לא נמצא צ\'קפוינט!')
        print('הורד את הקובץ מ-Colab והנח אותו ב:')
        print(f'  {ROOT / "checkpoints" / "e2e_cv_v2" / "config_A" / "best_finetune.pt"}')
        print('\nאו הרץ ב-Colab:')
        print('  from google.colab import files')
        print('  files.download("/content/SentinelFatal2/checkpoints/e2e_cv_v2/config_A/best_finetune.pt")')
        sys.exit(1)

    size_mb = ckpt_path.stat().st_size / (1024**2)
    print(f'✅ צ\'קפוינט: {ckpt_path.relative_to(ROOT)}  ({size_mb:.1f} MB)')

    # Check splits
    train_csv = SPLITS_DIR / 'train.csv'
    val_csv   = SPLITS_DIR / 'val.csv'
    test_csv  = SPLITS_DIR / 'test.csv'
    for p in [train_csv, val_csv, test_csv]:
        assert p.exists(), f'❌ חסר קובץ split: {p}'

    n_val  = len(pd.read_csv(val_csv))
    n_test = len(pd.read_csv(test_csv))
    n_tr   = len(pd.read_csv(train_csv))
    print(f'✅ splits: train={n_tr}, val={n_val}, test={n_test}')

    # ── Load model ────────────────────────────────────────────────────────────
    print('\n⏳ טוען מודל על CPU...')
    t0 = time.time()
    model = load_model(ckpt_path)
    print(f'   מודל נטען תוך {time.time()-t0:.1f}s')

    # ── AT sweep on val=56 ────────────────────────────────────────────────────
    print(f'\n⏳ AT sweep על val={n_val} (זה לוקח ~5-10 דקות על CPU)...')
    at_results = {}
    best_at, best_auc_at = 0.40, 0.0

    for at in AT_CANDIDATES:
        print(f'  AT={at:.2f}...', end=' ', flush=True)
        t1 = time.time()
        X_tr, y_tr, _ = extract_features(model, train_csv, at)
        X_vl, y_vl, _ = extract_features(model, val_csv,   at)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_vl)) < 2:
            print('דלג (אין שתי מחלקות)')
            at_results[at] = 0.0
            continue

        sc, lr = fit_lr(X_tr, y_tr)
        val_auc = roc_auc_score(y_vl, predict_lr(X_vl, sc, lr))
        at_results[at] = val_auc
        elapsed = time.time() - t1
        print(f'val_auc={val_auc:.4f}  ({elapsed:.0f}s)')

        if val_auc > best_auc_at:
            best_auc_at = val_auc
            best_at     = at

    print(f'\n✅ Best AT = {best_at:.2f}  (val_auc={best_auc_at:.4f})')

    # ── Extract with best AT + final LR ──────────────────────────────────────
    print(f'\n⏳ חילוץ features עם AT={best_at:.2f} לכל ה-splits...')
    X_tr, y_tr, _      = extract_features(model, train_csv, best_at)
    X_vl, y_vl, _      = extract_features(model, val_csv,   best_at)
    X_te, y_te, te_ids = extract_features(model, test_csv,  best_at)
    print(f'   train={len(y_tr)}  val={len(y_vl)}  test={len(y_te)}')

    sc, lr = fit_lr(X_tr, y_tr)
    val_scores  = predict_lr(X_vl, sc, lr)
    test_scores = predict_lr(X_te, sc, lr)

    # ── Threshold on val=56 ───────────────────────────────────────────────────
    th, sens_v, spec_v = clinical_threshold(y_vl, val_scores)
    print(f'\n   Threshold (val): {th:.4f}  |  sens={sens_v:.3f}  spec={spec_v:.3f}')

    # ── Final metrics on test=55 ──────────────────────────────────────────────
    test_auc   = roc_auc_score(y_te, test_scores) if len(np.unique(y_te)) > 1 else 0.0
    y_pred     = (test_scores >= th).astype(int)
    tp = int(((y_pred == 1) & (y_te == 1)).sum())
    tn = int(((y_pred == 0) & (y_te == 0)).sum())
    fp = int(((y_pred == 1) & (y_te == 0)).sum())
    fn = int(((y_pred == 0) & (y_te == 1)).sum())
    test_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f'\n{"="*55}')
    print(f'  תוצאות Config A — test-55  (CPU evaluation)')
    print(f'{"="*55}')
    print(f'  AUC         : {test_auc:.4f}   (קודם: 0.839)')
    print(f'  Sensitivity : {test_sens:.3f}   (קודם: 0.818)')
    print(f'  Specificity : {test_spec:.3f}   (קודם: 0.773)')
    print(f'  Threshold   : {th:.4f}  (קודם: 0.284)')
    print(f'  AT used     : {best_at:.2f}')
    print(f'  TP={tp}  TN={tn}  FP={fp}  FN={fn}')
    print(f'{"="*55}')

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pred = OUT_DIR / 'predictions_test55.csv'
    pd.DataFrame({
        'id': te_ids, 'y_true': y_te, 'y_score': test_scores,
        'y_pred': y_pred, 'best_at': best_at, 'threshold': th,
    }).to_csv(out_pred, index=False)

    out_summary = OUT_DIR / 'summary.csv'
    pd.DataFrame([{
        'ckpt': str(ckpt_path.relative_to(ROOT)),
        'test_auc': round(test_auc, 4),
        'test_sens': round(test_sens, 3),
        'test_spec': round(test_spec, 3),
        'threshold': round(float(th), 4),
        'best_at': best_at,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }]).to_csv(out_summary, index=False)

    at_df = pd.DataFrame([{'at': k, 'val_auc': round(v, 4)} for k, v in sorted(at_results.items())])
    at_df.to_csv(OUT_DIR / 'at_sweep.csv', index=False)

    print(f'\n✅ תוצאות נשמרו ב: {OUT_DIR.relative_to(ROOT)}')
    print(f'   {out_pred.name}')
    print(f'   {out_summary.name}')
    print(f'   at_sweep.csv')


if __name__ == '__main__':
    main()
