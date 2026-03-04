# SentinelFatal2 — סיכום מלא של כל האימונים

> **פרויקט:** זיהוי דיסטרס עוברי (CTG) באמצעות PatchTST Foundation Model
> **מטרה:** AUC ≥ 0.70 ב-5-fold cross-validation על 552 recordings
> **ייחוס:** arXiv:2601.06149v1 (Fridman & Ben Shachar) — AUC 0.826 על n=55
> **עדכון אחרון:** 2026-03-04 (לאחר Training Run v6)

---

## תוכן עניינים

1. [ארכיטקטורה ותשתית](#1-ארכיטקטורה-ותשתית)
2. [נקודת ייחוס — 0.839 Benchmark](#2-נקודת-ייחוס--0839-benchmark)
3. [שלב 0 — אימונים מקומיים ב-Colab](#3-שלב-0--אימונים-מקומיים-ב-colab-פב-2022-23-2026)
4. [Training Run v3 — Azure](#4-training-run-v3--azure)
5. [Training Run v4 — Azure](#5-training-run-v4--azure)
6. [Training Run v5 — Azure](#6-training-run-v5--azure)
7. [Training Run v6 — Azure](#7-training-run-v6--azure)
8. [השוואה מסכמת](#8-השוואה-מסכמת)
9. [מיקום Checkpoints](#9-מיקום-checkpoints)
10. [צבר בעיות ופתרונות](#10-צבר-בעיות-ופתרונות)
11. [הצעד הבא](#11-הצעד-הבא)

---

## 1. ארכיטקטורה ותשתית

### Pipeline בן 3 שלבים

```
שלב 1 — MAE Pre-training
  687 recordings (CTU-UHB + FHRMA, 13,687 windows)
  → Masked autoencoder על FHR — channel-asymmetric: FHR מוסתר, UC תמיד נראה
  → Masking curriculum: 0.20 → 0.30 → 0.40 (אפוקים 0/20/50+)
  → 300 אפוקים מקסימום, patience=50, CosineAnnealingWarmRestarts
  → שמירה: checkpoints/pretrain/best_pretrain.pt

שלב 2 — Fine-tuning (per fold)
  5-Fold CV על 552 recordings
  → Progressive unfreezing: 4 שלבים באפוקים 0/5/15/30
  → EarlyStopping על val_AUC, SWA (אפוקים 50–100), EMA smoothing
  → שמירה: checkpoints/e2e_cv/fold{k}/finetune/best_finetune.pt

שלב 3 — Alert Extraction + LR
  Sliding window → 12 alert segment features → Logistic Regression
  → AT sweep [0.30, 0.35, 0.40, 0.45, 0.50] על val set
  → Threshold: clinical (Sens-max s.t. Spec ≥ 0.65) או Youden
```

### ארכיטקטורת PatchTST

| פרמטר | ערך | מקור |
|--------|-----|-------|
| d_model | 128 | ⚠ S2 (לא פורסם) |
| num_layers | 3 | ⚠ S2 |
| n_heads | 4 | ⚠ S2 |
| ffn_dim | 256 | ⚠ S2 |
| dropout | 0.2 | ✓ מאמר |
| patch_len | 48 דגימות (12 שנ') | ✓ מאמר |
| patch_stride | 24 דגימות | ✓ מאמר |
| n_patches | 73 (חלון נחתך ל-1,776 דגימות) | ✓ מחושב (S9) |
| context window | 1,800 דגימות (7.5 דקות @ 4 Hz) | ✓ מאמר |

### תשתית Azure ML

| רכיב | ערך |
|------|-----|
| Subscription | Azure for Students (`02b4b69d-...`) — $100 קרדיט |
| Resource Group | `sentinelfatal2-rg` (France Central) |
| Workspace | `sentinelfatal2-aml` |
| Compute | `gpu-t4-cluster` — Standard_NC4as_T4_v3 (NVIDIA T4, 16 GB VRAM) |
| Environment | `sentinelfatal2-env:3` — PyTorch 2.2+cu118, NumPy<2.0 |
| Data Asset | `ctg-processed:1` — data_processed.zip (~23 MB, Azure Blob) |
| עלות עד כה | ~$7–8 מתוך $100 (v3+v4+v5+v6) |

### Dataset

| Split | Recordings | Acidemia | Normal |
|-------|-----------|----------|--------|
| Train | 441 | 90 (20.4%) | 351 |
| Val   | 56  | 12 (21.4%) | 44  |
| Test  | 55  | 11 (20.0%) | 44  |
| Pretrain (FHRMA) | 135 | — | 135 |
| **Pretrain total** | **687** | — | — |

> ⚠ **SPaM dataset חסר:** המאמר השתמש ב-984 recordings לpretrain (בניגוד ל-687 שלנו). SPaM הוסר מגישה ציבורית; פנייה ל-ctg.challenge2017@gmail.com.

---

## 2. נקודת ייחוס — 0.839 Benchmark

לפני הריצות ב-Azure, הושגה תוצאה של **AUC=0.839** בריצה מקומית (notebook).
**חשוב:** זו **לא** תוצאה השוואתית אמינה — זה single-split post-hoc:

| פרמטר | benchmark 0.839 | Azure runs v3-v6 |
|--------|----------------|-----------------|
| Split | 441/56/55 (קבוע) | 5-fold CV על 552 |
| LR training data | train+val = 441+56=497 | per-fold train only (~415) |
| N_features | 4 | 12 |
| Threshold | Youden-optimal (0.284) | clinical (Spec ≥ 0.65) |
| CI (95%) | [0.630, 0.953] — רוחב 0.32! | [0.57–0.69] מבוסס 552 recordings |

ה-CI הרחב (±0.16) מדגים שהתוצאה **לא מהימנה** — n=55 קטן מדי לאסטימציה יציבה.

**השוואה מהמאמר (Table 3):**

| תת-קבוצה | n | AUC מאמר |
|----------|---|----------|
| All test | 55 | **0.826** |
| Vaginal deliveries | 50 | 0.850 |
| Cephalic presentation | 50 | 0.848 |

---

## 3. שלב 0 — אימונים מקומיים ב-Colab (פב' 22-23, 2026)

### מה עשינו

הריצות הראשונות של הפרויקט — single-split על 441/56/55. **לא E2E cross-validation.**
מטרה: לאמת שהפייפליין עובד ולהגיע לAUC קרוב ל-0.826 של המאמר.

**כלים:** Google Colab T4 GPU (~15 GB VRAM)
**Script:** `scripts/run_e2e_cv.py` (תומך גם single-fold)
**Notebook:** `notebooks/07_colab_e2e_cv_launch.ipynb`

### תוצאות Pre-training

```
Max epochs: 200, Early stopping patience: 10 (val reconstruction loss)
Best val MSE = 0.01427 at epoch 2
Train loss: 0.111 → 0.009 (ירידה רציפה על 13 אפוקים)
Val loss: plateau לאחר epoch 2 → early stopped at epoch 12
```

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 0 | 0.11100 | 0.01538 |
| 1 | 0.02964 | 0.01472 |
| **2** | **0.01949** | **0.01427** ← best |
| 3–12 | 0.015→0.009 | 0.015–0.017 (plateau) |

### תוצאות Fine-tuning (single split)

```
Max epochs: 100, Early stopping patience: 15 (val AUC)
Best val AUC = 0.7235 at epoch 17
Early stopped at epoch 32 (patience=15 אחרי epoch 17)
```

### תוצאות Test (55 recordings — single split)

| קונפיגורציה | AUC | Sens | Spec | TP/11 | threshold |
|-------------|-----|------|------|-------|-----------|
| Baseline AT=0.50, T=0.50 | 0.812 | 0.091 | 1.000 | 1/11 | 0.50 |
| **Old LR + AT=0.40 + Youden T=0.284** | **0.839** | **0.818** | **0.773** | **9/11** | **0.284** |
| Final LR (497 recordings) + AT=0.40 + CV-T | 0.717 | 0.636 | 0.818 | 7/11 | 0.199 |

> **הגילוי הגדול:** שינוי AT מ-0.50 ל-0.40 שינה את כל התוצאות — Sensitivity קפץ מ-0.09 ל-0.818!
> עם AT=0.50, 13/55 הקלטות לא ייצרו alert segment → ZERO_FEATURES → LR defaulted לclass prior → Sensitivity=0.09

### LR-only Cross-Validation (Colab)

לאחר האימון הראשוני, הרצנו 5-fold CV על LR בלבד (ה-backbone היה קבוע):

| שיטה | n | AUC | CI |
|------|---|-----|-----|
| 5-Fold CV OOF (n=552) | 552 | 0.6595 | [0.597, 0.715] |
| CV improved (n=497) | 497 | 0.6720 | [0.615, 0.729] |

> **מגבלה:** אלו LR-only CVs — ה-backbone אומן פעם אחת על כל הdata ולא per-fold. זו **לא** E2E CV.

### תיקונים קריטיים שהתגלו בשלב זה

| תיקון | קוד | השפעה |
|-------|-----|-------|
| UC Fusion (`fhr_enc += uc_enc`) | `src/train/pretrain.py` | ללא זה UC לא היה משפיע כלל |
| AT=0.40 (מ-0.50) | `src/inference/alert_extractor.py` | Sensitivity: 0.09 → 0.818 |
| Differential LR (backbone=1e-5, head=1e-4) | `src/train/finetune.py` | מניעת catastrophic forgetting |
| Class weights [1.0, 3.9] | `src/train/finetune.py` | מאזן 80/20 imbalance |
| n_patches trim ל-1,776 (S9) | `src/model/patchtst.py` | מבטיח 73 patches בדיוק |

### Checkpoints (מקומיים)

| קובץ | תוכן |
|------|------|
| `checkpoints/pretrain/best_pretrain.pt` | Pretrain weights — epoch 2, val_MSE=0.01427 |
| `checkpoints/finetune/best_finetune.pt` | Finetune weights — epoch 17, val_AUC=0.7235 |
| `checkpoints/alerting/logistic_regression.pkl` | LR על 441 recordings, AT=0.50, T=0.50 |
| `checkpoints/alerting/logistic_regression_at040.pkl` | LR על 441 recordings, AT=0.40, T=0.284 (הטוב ביותר) |
| `checkpoints/e2e_cv/fold0/pretrain/best_pretrain.pt` | Dry-run fold 0 (ניסיון בלבד) |
| `checkpoints/e2e_cv/fold0/finetune/best_finetune.pt` | Dry-run fold 0 (ניסיון בלבד) |

---

## 4. Training Run v3 — Azure

**Job name:** `frosty_kite_kdt326xr9h`
**תאריך:** 2026-02-27
**משך:** 162.8 דקות
**סטטוס:** Completed ✅
**Artifacts:** `logs/e2e_cv_v3/azure_job/artifacts/`

### מה עשינו

האימון הראשון ב-Azure ML. E2E pipeline מלא:
pre-training על 687 recordings + 5-fold fine-tuning על 552 recordings.

**הגדרות עיקריות (Config A):**
- Loss: CrossEntropy, class_weight=[1.0, 3.9]
- Patience: 15, train_stride=120, val_stride=60
- LR: head=1e-3, backbone=0 → 1e-5 → 3e-5 → 5e-5 (progressive unfreezing)
- LR meta-classifier: C=0.1, **train+val combined**, 12 features

### תוצאות לפי Fold

| Fold | best@epoch | Val AUC | Test AUC | AT | זמן |
|------|-----------|---------|----------|----|-----|
| 0 | 35 | 0.7394 | 0.5865 | 0.40 | ~30 min |
| 1 | 10 | 0.6036 | 0.6527 | 0.30 | ~25 min |
| 2 | 15 | 0.6557 | 0.6324 | 0.45 | ~30 min |
| 3 | 20 | 0.7010 | 0.5997 | 0.30 | ~28 min |
| 4 | 10 | 0.6859 | 0.6061 | 0.40 | ~23 min |

### תוצאות סופיות

| מדד | ערך |
|-----|-----|
| **Global OOF AUC** | **0.6013** (95% CI: 0.5429–0.6597) |
| Mean fold AUC | 0.6155 ± 0.027 |
| Clinical threshold | Sens=0.478, Spec=0.651 |
| REPRO_TRACK AUC (n=55) | 0.6529 [0.471, 0.821] |
| G4a (mean ≥ 0.70) | ❌ FAIL |
| G4b (std < 0.10) | ✅ PASS |

### בעיות שזוהו

1. **Val AUC >> Test AUC** בכל fold (פערים 5%–19%) — overfitting/distribution shift
2. **LR אומן על train+val** (leakage) — ה-val שימש גם לבחירת AT וגם לאימון LR
3. **Data download מ-GitHub** בזמן ריצה (לא יעיל, ~5 min overhead)
4. **אין grid search** — C=0.1 נבחר שרירותית

### שינויים לv4

- patience: 15 → **20**
- הוספת **grid search** (AT × C × n_features × threshold, 300 קומבינציות)
- **Data Asset** (`ctg-processed:1`) — אין download מGitHub
- Code snapshot מינימלי (0.6MB)

### Checkpoints (Azure Blob)

| מיקום | תוכן |
|--------|------|
| Azure ML Studio → `frosty_kite_kdt326xr9h` → outputs/checkpoints/fold{0-4}/ | best_finetune.pt per fold |

---

## 5. Training Run v4 — Azure

**Job name:** `ashy_wing_4n7bmybg1c`
**תאריך:** 2026-02-28
**משך:** 197.2 דקות
**סטטוס:** Completed ✅
**Artifacts:** `logs/e2e_cv_v4/azure_job/artifacts/`

### מה עשינו

שיפורים על v3: patience גדולה יותר + grid search מקיף. **הפעלנו מחדש pretrain** כי לא נמצא checkpoint קיים על ה-compute.

**שינויים מv3:**
- Patience: 15 → **20**
- Grid search: 300 קומבינציות (AT=5 × C=3 × n_feat=4 × thr=2 × 5 folds)
- Data Asset במקום GitHub download

### תוצאות לפי Fold

| Fold | best@epoch | Val AUC | Test AUC | AT | זמן |
|------|-----------|---------|----------|----|-----|
| 0 | 10 | 0.7476 | 0.6146 | 0.40 | 21.1 min |
| 1 | 35 | 0.7449 | 0.6458 | 0.30 | 38.9 min |
| 2 | 15 | 0.7311 | 0.6542 | 0.35 | 35.2 min |
| **3** | **0** ⚠️ | 0.6955 | 0.6854 | 0.40 | **11.9 min** |
| 4 | 15 | 0.7682 | 0.5768 | 0.45 | 24.5 min |

### תוצאות סופיות

| מדד | ערך |
|-----|-----|
| **Global OOF AUC** | **0.6385** (95% CI: 0.5777–0.6979) |
| Mean fold AUC | 0.6354 ± 0.041 |
| REPRO_TRACK AUC (n=55) | 0.5930 [0.418, 0.764] |
| G4a (mean ≥ 0.70) | ❌ FAIL |
| G4b (std < 0.10) | ✅ PASS |
| **Grid winner** | AT=0.40, n_feat=12, C=1.0, Youden → **mean_AUC=0.6358** |

### בעיות שזוהו

1. **Fold 3 קרס — best_epoch=0** ⚠️
   - AUC ראשוני גבוה (0.6955@0) לפני אימון backbone
   - Epoch 5 (unfreezing): AUC צנח ל-0.6214 (ירידה 10.6%) — **LR shock**
   - `lr_warmup_epochs: 5` היה בconfig אבל **לא היה ממומש בקוד!**
   - Paradox: fold 3 השיג test AUC הגבוה ביותר (0.6854) — frozen backbone לא overfitted

2. **Val AUC >> Test AUC** בכל fold (פערים 1.1%–19.1%)
   - Val שימש **3 פעמים**: AT selection + LR training + threshold selection

3. **C=1.0 hardcoded** אחרי שgrid winner v4 היה C=1.0 — ייגרום לבעיות בריצות עתידיות

### שינויים לv5

- **מימוש LR warmup** — ramp ליניארי על 5 אפוקים בכל unfreeze
- **LR על train בלבד** (הסרת val leakage)
- **Augmentation לPretrain** (Gaussian noise + amplitude scaling)

### Checkpoints (Azure Blob)

| מיקום | תוכן |
|--------|------|
| Azure ML Studio → `ashy_wing_4n7bmybg1c` → outputs/checkpoints/fold{0-4}/ | best_finetune.pt per fold |

---

## 6. Training Run v5 — Azure

**Job name:** `jovial_spinach_6vfchv3bk3`
**תאריך:** 2026-03-02 → 03-03
**משך:** 183.0 דקות (CV core: 117.9 min)
**סטטוס:** Completed ✅
**Artifacts:** `logs/e2e_cv_v5/azure_job/artifacts/`

### מה עשינו

שלושה שינויים ממוקדים על בסיס פוסט-מורטם v4:

#### שינוי 1 — LR Warmup לאחר Unfreezing [`src/train/finetune.py`]

**הבעיה:** Backbone "נפתח" באפוק 5 → LR קפץ מ-0 ל-1e-5 בבת אחת → שוק.
**הפתרון:** Ramp ליניארי על 5 אפוקים: `LR_bb = target × (epoch/5)`.

```python
# v4 — LR קפץ מיד
apply_unfreeze_phase(model, n_top, lr_bb, lr_hd, optimizer)

# v5 — backbone LR=0 + warmup הדרגתי
apply_unfreeze_phase(model, n_top, 0.0, lr_hd, optimizer)
# אחרי כל אפוק: optimizer.param_groups[0]['lr'] = lr_target * (elapsed+1)/warmup_epochs
```

#### שינוי 2 — הסרת Val Leakage [`azure_ml/train_azure.py`]

**הבעיה:** LR אומן על train+val — val שימש 3 פעמים.
**הפתרון:** LR מאומן על train בלבד, C=1.0 (grid winner v4).

```python
# v4: leakage
scaler, pca, lr_m = fit_lr_model(np.concatenate([X_tr, X_vl]), y_tv, C=0.1)

# v5: train only
scaler, pca, lr_m = fit_lr_model(X_tr, y_tr, C=1.0)
```

#### שינוי 3 — Augmentation לPretrain [`src/data/dataset.py` + `src/train/pretrain.py`]

**הבעיה:** Pretrain על signal נקי — SPaM dataset (297 recordings) לא זמין.
**הפתרון:**

```python
# FHR noise (σ=0.01, p=0.5) ≈ 1.6 bpm
# UC noise  (σ=0.005, p=0.5) ≈ 0.5 mmHg
# Amplitude scaling [0.95, 1.05] (p=0.3)
```

### תוצאות לפי Fold

| Fold | best@epoch | Val AUC | Test AUC | AT | זמן |
|------|-----------|---------|----------|----|-----|
| 0 | 10 | 0.7682 | 0.5716 | 0.35 | 21.1 min |
| **1** | **0** ⚠️ | 0.7599 | 0.6112 | 0.35 | **12.0 min** |
| 2 | 15 | 0.7709 | 0.6744 | 0.45 | 24.5 min |
| **3** | **10** ✅ | 0.6872 | 0.6854 | 0.35 | **24.5 min** |
| 4 | 10 | 0.7133 | 0.5042 | 0.30 | 21.0 min |

### תוצאות סופיות

| מדד | ערך |
|-----|-----|
| **Global OOF AUC** | **0.5870** (95% CI: 0.5258–0.6480) |
| Mean fold AUC | 0.6094 ± 0.075 |
| **REPRO_TRACK AUC (n=55)** | **0.7872** [0.636, 0.917] ↑↑ (מ-0.5930) |
| G4a (mean ≥ 0.70) | ❌ FAIL |
| G4b (std < 0.10) | ✅ PASS |
| **Grid winner** | AT=0.40, n_feat=12, C=0.01, clinical → **mean_AUC=0.6426** |

### ניתוח — מה עבד ומה לא

**מה עבד ✅:**
- **Fold 3 תוקן:** best_epoch עלה מ-0 ל-10, זמן מ-11.9→24.5 min. Warmup עבד!
- **REPRO_TRACK קפץ ל-0.7872:** pretrain augmentation + train-only LR → פיצ'רים טובים יותר

**מה נכשל ❌:**

1. **Fold 1 קרס (best@0)** — אותה בעיה, fold אחר:
   - AUC ראשוני 0.7599@0; warmup גרם לירידה (0.5789@5-9)
   - Patience=20 ספורה **מאפוק 0**, לא מרגע ה-unfreeze → אוזלת לפני התאוששות

2. **C=1.0 לא נכון לפיצ'רים החדשים:**
   - Hardcoded C=1.0 (grid winner v4) — אבל pretrain חדש שינה את הפיצ'רים
   - Grid search v5 גילה: C=0.01 עדיף → OOF 0.6426 vs 0.5870 עם C=1.0

**Timeline — הדגמה לfold 3 תוקן בv5:**

```
v4 — fold 3 (ללא warmup):
  Epoch 0:  val_auc=0.6955  ← best
  Epoch 5:  unfreeze, LR קפץ מ-0 ל-1e-5 → שוק: val_auc=0.6214
  Epoch 20: early stop (patience=20)  ← best_epoch=0

v5 — fold 3 (עם warmup):
  Epoch 0:  val_auc=0.7051  ← best
  Epoch 5:  unfreeze, backbone LR=0 → warmup מתחיל
  Epoch 10: val_auc=0.7682  ← warmup הסתיים, AUC קפץ → NEW BEST
  ...       best_epoch=10  ✅
```

### שינויים לv6

- **Patience מתאפסת בכל unfreeze** — fold 1 יקבל budget חדש אחרי כל שלב
- **C=0.01 בmain pipeline** — align עם grid winner v5

### Checkpoints (Azure Blob)

| מיקום | תוכן |
|--------|------|
| Azure ML Studio → `jovial_spinach_6vfchv3bk3` → outputs/checkpoints/fold{0-4}/ | best_finetune.pt per fold |

---

## 7. Training Run v6 — Azure

**Job name:** `crimson_dog_stjh6zsmqb`
**תאריך:** 2026-03-03
**משך:** 212.8 דקות
**סטטוס:** Completed ✅
**Artifacts:** `logs/e2e_cv_v6/azure_job/artifacts/`

### מה עשינו

שני שינויים ממוקדים על בסיס פוסט-מורטם v5:

#### שינוי 1 — Patience מתאפסת בכל Unfreeze [`src/train/finetune.py`]

**הבעיה:** `patience_ctr` אוזל בזמן ה-warmup dip (5-9 אפוקים לאחר unfreeze) → fold collapse.
**הפתרון:** `patience_ctr = 0` בתוך בלוק ה-phase-change:

```python
if phase_key != last_phase_key:
    if n_top != 0:
        _warmup_start_epoch  = epoch
        _warmup_target_lr_bb = lr_bb
        apply_unfreeze_phase(model, n_top, 0.0, lr_hd, optimizer,
                             weight_decay=float(ftcfg["weight_decay"]))
        patience_ctr = 0   # ← שינוי v6: budget חדש אחרי כל unfreeze
        print(f"  [warmup] Starting {_warmup_epochs}-epoch backbone LR ramp "
              f"(target={lr_bb:.1e})")
```

**למה לא מאפסים best_smooth_auc:** best global נשמר. רק ה-patience budget מתאפס — המודל מקבל סיכוי הוגן בכל שלב מבלי לזרוק את הcheckpoint הטוב ביותר.

#### שינוי 2 — C=0.01 בmain Pipeline LR [`azure_ml/train_azure.py`]

**הבעיה:** Grid winner v5 היה C=0.01, אך main pipeline השתמש ב-C=1.0 → OOF=0.5870 במקום 0.6426.

```python
# v5 (שגוי — hardcoded C מv4):
scaler, pca, lr_m = fit_lr_model(X_tr, y_tr, C=1.0, use_pca=True)

# v6 (מתוקן — grid winner v5):
scaler, pca, lr_m = fit_lr_model(X_tr, y_tr, C=0.01, use_pca=True)
# גם REPRO_TRACK קיבל את אותו שינוי
```

### תוצאות לפי Fold

| Fold | best@epoch | Val AUC | Test AUC | זמן | הערה |
|------|-----------|---------|----------|-----|------|
| **0** | **55** ✅ | 0.7490 | 0.6102 | 53.1 min | patience reset עבד! (v5: best@10) |
| **1** | **0** ⚠️ | 0.7695 | 0.5909 | 31.6 min | עדיין collapse, אך 31.6 vs 12 min |
| 2 | 15 | 0.7078 | 0.6621 | 39.1 min | ללא שינוי |
| **3** | **45** ✅ | 0.6845 | 0.6472 | 31.6 min | patience reset עבד! (v5: best@10) |
| **4** | **40** ✅ | 0.7462 | 0.6562 | 42.8 min | patience reset עבד! (v5: best@10) |

### תוצאות סופיות

| מדד | ערך | השוואה לv5 |
|-----|-----|------------|
| **Global OOF AUC** | **0.6329** (CI: 0.5729–0.6908) | ↑ מ-0.5870 (+0.046) |
| Mean fold AUC | 0.6333 ± 0.031 | ↑ מ-0.6094 ±0.075 |
| **REPRO_TRACK AUC (n=55)** | **0.5165** [0.322, 0.713] | ↓↓ מ-0.7872 (רגרסיה!) |
| G4a (mean ≥ 0.70) | ❌ FAIL | — |
| G4b (std < 0.10) | ✅ PASS | ✅ |
| **Grid winner** | AT=0.45, n_feat=12, C=0.1, clinical → **mean_AUC=0.6371** | |
| Runtime | 212.8 min | ↑ (יותר epochs = יותר זמן) |

### ניתוח — מה עבד ומה לא

**מה עבד ✅:**
- **Patience reset עבד על folds 0, 3, 4:** בוגרות לאפוקים 55, 45, 40 (במקום 10)
- **OOF AUC שוב עלה:** 0.5870 → 0.6329 (תיקון C+patience)
- **Std ירד:** 0.075 → 0.031 (מדדי יציבות טובים יותר)

**מה נכשל ❌:**

1. **Fold 1 עדיין collapse (best@0):**
   - AUC ראשוני גבוה מאוד (0.7695@0) — הgolden reference גבוה מדי
   - Backbone לא מצליח לעלות על frozen head גם עם patience חדשה (31.6 min, 2.6× v5)
   - **בעיית עומק:** ה-fold הזה אולי מסיג לא טובה מbackbone fine-tuning

2. **REPRO_TRACK קרס ל-0.5165 (רגרסיה מסיבית):**
   - C=0.01 היה grid winner **בv5** — אך בv6 הפיצ'רים שונים (pretrain אחיד, fold שונה)
   - Grid winner של v6 הוא C=0.1, לא C=0.01!
   - **מסקנה קריטית:** C חייב להיות נבחר מgrid search בכל ריצה — hardcoding C הוא שגיאה מבנית

3. **OOF (0.6329) עדיין פחות מv4 (0.6385):**
   - Fold 1 collapse (test 0.5909) גורר את הממוצע למטה
   - Grid winner (0.6371) > main pipeline (0.6329) — פער קטן, לא קריטי

**Timeline — fold 1 collapse pattern (קבוע לאורך כל הריצות):**

```
v4 — fold 1 (best@35): עבד
v5 — fold 1 (best@0): collapse — patience אזלה לפני warmup
v6 — fold 1 (best@0): collapse — גם עם patience reset, backbone < frozen head

AUC@epoch 0 (frozen head): 0.7695 — גבוה מדי
AUC אחרי warmup: ~0.6–0.65 — backbone מנסה ולא מגיע
```

### Checkpoints (Azure Blob + Local)

| מיקום | תוכן |
|--------|------|
| Azure ML Studio → `crimson_dog_stjh6zsmqb` → outputs/checkpoints/fold{0-4}/ | best_finetune.pt per fold |
| Azure ML Studio → `crimson_dog_stjh6zsmqb` → outputs/checkpoints/repro_track/ | REPRO_TRACK checkpoint |

---

## 8. השוואה מסכמת

### תוצאות לאורך כל הריצות

| ריצה | תאריך | OOF AUC | Grid best OOF | REPRO_TRACK | G4a | G4b | Runtime | עלות |
|------|-------|---------|--------------|-------------|-----|-----|---------|------|
| **Colab** | פב' 22-23 | — | — | **0.839** (single split!) | — | — | ~3 שעות | $0 |
| **LR-only CV** | פב' 23 | 0.6595 (n=552) | — | — | — | — | — | $0 |
| **v3** | פב' 27 | 0.6013 | — | 0.6529 | ❌ | ✅ | 162.8 min | ~$2.5 |
| **v4** | פב' 28 | **0.6385** | 0.6358 | 0.5930 | ❌ | ✅ | 197.2 min | ~$3.0 |
| **v5** | מר' 2-3 | 0.5870 | **0.6426** | **0.7872** | ❌ | ✅ | 183.0 min | ~$2.8 |
| **v6** | מר' 3 | 0.6329 | 0.6371 | 0.5165 | ❌ | ✅ | 212.8 min | ~$3.2 |
| **יעד** | — | **≥ 0.70** | — | ~0.826 | ✅ | ✅ | — | — |

### Grid winner לפי ריצה

| ריצה | AT | n_feat | C | threshold | mean_AUC |
|------|-----|--------|---|-----------|---------|
| v3 | — | — | — | — | — |
| v4 | 0.40 | 12 | **1.0** | Youden | 0.6358 |
| v5 | 0.40 | 12 | **0.01** | clinical | 0.6426 |
| v6 | 0.45 | 12 | **0.1** | clinical | 0.6371 |

> ⚠️ **מסקנה קריטית:** C האופטימלי משתנה בכל ריצה (1.0 → 0.01 → 0.1). **אין לעולם להכניס C hardcoded!** הפיצ'רים משתנים עם כל pretrain checkpoint חדש.

### Best Epochs לפי fold (התפתחות לאורך הריצות)

| Fold | v4 | v5 | v6 | מגמה |
|------|----|----|-----|------|
| 0 | 10 | 10 | **55** ✅ | patience reset עבד |
| 1 | 35 | 0 ⚠️ | 0 ⚠️ | collapse עקבי בv5+v6 |
| 2 | 15 | 15 | 15 | יציב |
| 3 | 0 ⚠️ | **10** ✅ | **45** ✅ | warmup תיקן, patience הוסיפה |
| 4 | 15 | 10 | **40** ✅ | patience reset שיפרה |

### Timeline — תיאור הבעיות שנפתרו ושלא

```
v3: val leakage (LR על train+val)
      ↓ תוקן בv5
v4: fold 3 collapse (LR shock, warmup לא ממומש)
      ↓ תוקן בv5 (warmup)
v5: fold 1 collapse (patience אוזלת בזמן warmup)
      ↓ תוקן חלקית בv6 (patience reset) — fold 1 עדיין collapse
v5: C hardcoded אחרי grid search
      ↓ "תוקן" בv6 אבל C=0.01 לא היה נכון לv6 → REPRO crashed
v6: C חייב לבוא מinner CV, לא מgrid search חיצוני
      ← כאן אנחנו עכשיו
```

---

## 9. מיקום Checkpoints

### Checkpoints מקומיים (בדיסק — `checkpoints/`)

| קובץ | ריצה | תוכן | AUC/MSE |
|------|------|------|---------|
| `checkpoints/pretrain/best_pretrain.pt` | Colab | Pretrain weights, epoch 2 | val_MSE=0.01427 |
| `checkpoints/finetune/best_finetune.pt` | Colab | Finetune weights, epoch 17 | val_AUC=0.7235 |
| `checkpoints/alerting/logistic_regression.pkl` | Colab | LR על 441 recordings, AT=0.50 | AUC=0.812 |
| `checkpoints/alerting/logistic_regression_at040.pkl` | Colab | LR על 441 recordings, AT=0.40 | **AUC=0.839** |
| `checkpoints/e2e_cv/fold0/pretrain/best_pretrain.pt` | Dry-run | Fold 0 pretrain (בדיקה בלבד) | — |
| `checkpoints/e2e_cv/fold0/finetune/best_finetune.pt` | Dry-run | Fold 0 finetune (בדיקה בלבד) | — |
| `checkpoints/pretrain_audit/best_pretrain.pt` | Audit | Architecture validation run | — |
| `checkpoints/finetune_audit/best_finetune.pt` | Audit | Architecture validation run | — |

> ⚠️ **Checkpoints מקומיים = ריצת Colab בלבד** (single split 441/56/55, לא E2E CV)

### Checkpoints Azure ML (על Azure Blob Storage)

גישה דרך Azure ML Studio: `ml.azure.com` → Workspace → Jobs → [job_name] → Outputs + Logs → outputs/checkpoints/

| Job Name | ריצה | מה יש שם |
|----------|------|----------|
| `frosty_kite_kdt326xr9h` | v3 | fold0–4/finetune/best_finetune.pt |
| `ashy_wing_4n7bmybg1c` | v4 | fold0–4/finetune/best_finetune.pt |
| `jovial_spinach_6vfchv3bk3` | v5 | fold0–4/finetune/best_finetune.pt |
| `crimson_dog_stjh6zsmqb` | v6 | fold0–4/finetune/best_finetune.pt + repro_track/ |

> **הערה:** Checkpoints נמחקים מAzure Blob לאחר 90 יום (ייתכן). ניתן להוריד דרך Studio לפני שיפוגו.

### Artifacts מורדים (בדיסק — `logs/`)

| תיקייה | ריצה | תוכן |
|---------|------|------|
| `logs/e2e_cv_v3/azure_job/artifacts/` | v3 | CSVs, logs (ללא .pt checkpoints) |
| `logs/e2e_cv_v4/azure_job/artifacts/` | v4 | CSVs, logs, grid_search_results.csv |
| `logs/e2e_cv_v5/azure_job/artifacts/` | v5 | CSVs, logs, grid_best_configs.csv |
| `logs/e2e_cv_v6/azure_job/artifacts/` | v6 | CSVs, logs, grid_best_configs.csv |

---

## 10. צבר בעיות ופתרונות

| בעיה | התגלה | פתרון | מצב |
|------|-------|-------|-----|
| Data download מGitHub | v3 | Data Asset `ctg-processed:1` | ✅ נפתר בv3 |
| LR trained on train+val (leakage) | v3 | train only בv5 | ✅ נפתר בv5 |
| `lr_warmup_epochs` לא ממומש | v4 | warmup ליניארי בv5 | ✅ נפתר בv5 |
| Fold collapse (best@0) — fold 3 בv4 | v4 | warmup תיקן אותו | ✅ תוקן בv5 |
| Patience לא מתאפסת בunfreeze — fold 1 | v5 | reset בv6 | ⚠️ חלקי (fold 0,3,4 ✅, fold 1 עדיין ❌) |
| C hardcoded ≠ grid winner | v5 | — | ❌ **לא נפתר** — C=0.01 לv6 גרם לרגרסיה ב-REPRO |
| Fold 1 persistent collapse (best@0) | v5/v6 | — | ❌ **לא נפתר** — backbone < frozen head ב-fold זה |
| SPaM dataset חסר | תמיד | pretrain augmentation (חלקי) | ⚠️ פיצוי חלקי |
| Val>>Test gap | תמיד | train-only LR (חלקי) | ⚠️ עדיין קיים |
| C אינו יציב בין ריצות | v5/v6 | צריך inner-CV per fold | ❌ **לא ממומש** |

### בעיית ה-C — הסבר מורחב

הבעיה הכי עדינה בפרויקט: ה-C האופטימלי של LR אינו קבוע בין ריצות:

- v4: grid winner C=1.0 → hardcoded C=1.0 בv5 → **OOF=0.5870** (רגרסיה!)
- v5: grid winner C=0.01 → hardcoded C=0.01 בv6 → **REPRO=0.5165** (קריסה!)
- v6: grid winner C=0.1 → ?

**הסיבה:** הפיצ'רים (embedding space) משתנים עם כל pretrain חדש. C שהיה optimal ל-feature space אחד לא בהכרח optimal לאחר.

**הפתרון הנכון:** Inner loop per-fold — כל fold בוחר C על val set שלו. כך אין leakage וC מתאים לdata.

---

## 11. הצעד הבא

### בעיות פתוחות ל-v7

**עדיפות גבוהה:**
1. **פתרון בעיית C:** Inner CV per fold (C ∈ [0.01, 0.1, 1.0]) — כל fold בוחר C על val שלו
2. **Fold 1 collapse:** לחקור למה ה-frozen head כל-כך חזק (0.7695) וה-backbone לא מצליח לעלות עליו. אפשרויות:
   - Regularization חזק יותר על head בשלב frozen
   - Grad clipping חמור יותר בזמן warmup
   - הנחה: fold 1 data distribution שונה — בדוק class distribution

**עדיפות בינונית:**
3. **Val>>Test gap:** בחינת test-time augmentation או calibration
4. **Hyperparameter search:** d_model=256 (S2), layers=4 — ייתכן שיפור קל

### פקודת הגשה לv7

```bash
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 python -u azure_ml/setup_and_submit.py --dedicated
```

---

## 12. נספח — פרטים טכניים מלאים (שאלות ותשובות)

> **מטרה:** תשובות מדויקות לשאלות טכניות שנדרשות לתכנון v7.
> כל התשובות מאומתות ישירות מהקוד (ולא מהערכות).

---

### ש1 — מהם בדיוק 12 הפיצ'רים של ה-LR?

**קוד:** `src/inference/alert_extractor.py` → `extract_recording_features()`

**חשוב:** יש **שני סוגי פיצ'רים** — פיצ'רים של **ה-segment הארוך ביותר** ופיצ'רים של **ההקלטה כולה**.

| # | שם פיצ'ר | חישוב | בסיס |
|---|----------|--------|------|
| 1 | `segment_length` | אורך ה-segment הארוך ביותר (דקות): `len × (stride/fs) / 60` | Segment ארוך |
| 2 | `max_prediction` | מקסימום P(acidemia) בסגמנט הארוך | Segment ארוך |
| 3 | `cumulative_sum` | `Σ(score) × dt` (שטח מתחת לעקומה בזמן) | Segment ארוך |
| 4 | `weighted_integral` | `Σ((score − 0.5)² × dt)` (סטייה ממצב ניטרלי) | Segment ארוך |
| 5 | `n_alert_segments` | מספר ה-alert segments הנפרדים | כל ההקלטה |
| 6 | `alert_fraction` | `n_alert_windows / n_total_windows` | כל ההקלטה |
| 7 | `mean_prediction` | ממוצע P(acidemia) בסגמנט הארוך | Segment ארוך |
| 8 | `std_prediction` | סטיית תקן של הציונים בסגמנט הארוך | Segment ארוך |
| 9 | `max_pred_all_segments` | מקסימום על פני **כל** ה-segments | כל ה-segments |
| 10 | `total_alert_duration` | סך זמן ה-alert (דקות) — כל הסגמנטים | כל ה-segments |
| 11 | `recording_max_score` | מקסימום ציון בכל ההקלטה (ללא threshold) | כל ההקלטה |
| 12 | `recording_mean_above_th` | ממוצע ציוני כל החלונות שמעל AT=0.40 | כל ההקלטה |

**פרמטרים:**
- `inference_stride = 24` דגימות → `dt = 24/4 = 6` שניות לחלון
- AT = 0.40 (כלומר, חלון "alert" הוא כל חלון עם score > 0.40)
- Feature 4: `weighted_integral` — המשקל `(score − 0.5)²` מגביר את הרגישות לציונים גבוהים (>0.5) ומכחיש ציונים נמוכים (<0.5)

**אם אין alert segments:** כל 12 הפיצ'רים = 0.0 (ZERO_FEATURES)

---

### ש2 — מה זה REPRO_TRACK ומאין הגיעו 55 ההקלטות?

**REPRO_TRACK = נקודת ייחוס, לא independent test set.**

```
הdata כולל 552 הקלטות CTU-UHB:
  ├── data/splits/train.csv → 441 הקלטות
  ├── data/splits/val.csv   →  56 הקלטות
  └── data/splits/test.csv  →  55 הקלטות (= REPRO_TRACK test set)

5-Fold CV: נוצרת stratified 5-fold על כל 552 ההקלטות יחד
REPRO_TRACK: מאמן על train(441) + val(56) = 497, בודק על test(55)
```

**התשובה הברורה:**
1. **האם 55 ההקלטות הן held-out עצמאי?** **לא לגמרי.** הן חלק מ-552, ולכן בחלק מה-CV folds הן נמצאות ב-train/val — לא ב-test
2. **האם הן אותן 55 הקלטות של Colab?** **כן בדיוק.** אלו אותן ההקלטות שנבדקו ב-Colab (AUC=0.839 / 0.812)
3. **מה ה-REPRO_TRACK בא להוכיח?** השוואה ישירה עם המאמר ועם תוצאות Colab — על אותו split קנוני

**לכן אי-היציבות של REPRO_TRACK גבוהה בהגדרה:** n=55, acidemia=11 → CI רחב של ±0.16 מתקבל מהnoise הסטטיסטי בלבד. תנודות של 0.5165↔0.7872↔0.5930 בין ריצות נובעות בעיקרן מה-noise הזה ולא משינויי מודל.

---

### ש3 — התפלגות מחלקות ב-Fold 1 — האם יש חוסר איזון חריג?

**תשובה:** **לא. כל ה-folds זהים כמעט לחלוטין** — ה-fold collapse ב-fold 1 אינו נובע מחוסר איזון.

| Fold | Train | n_pos (acidemia) | n_neg | pos_frac | class_weight[1] |
|------|-------|-----------------|-------|---------|----------------|
| 0 | 387 | **63** | 324 | 16.3% | 5.14 |
| **1** | 387 | **63** | 324 | 16.3% | 5.14 |
| 2 | 387 | **63** | 324 | 16.3% | 5.14 |
| 3 | 388 | 64 | 324 | 16.5% | 5.06 |
| 4 | 389 | 64 | 325 | 16.5% | 5.08 |

*מקור: `[class_weights]` בstd_log.txt של v6*

**מה בכל זאת שונה בין fold 1 לשאר?** ה-111 הקלטות ב-**test set** של fold 1 הן אחרות. הבעיה איננה באימון — הbacobne אינו מצליח לעלות על ה-frozen head (0.7695@0) על **ה-val set** של אותו fold. ה-val set (54 הקלטות) ייחודי לכל fold, וב-fold 1 המודל הקפוא (ללא backbone fine-tuning) מניב AUC גבוה במיוחד — כך שה-backbone לא מוסיף מידע.

**מסקנה:** הבעיה ב-fold 1 היא מבנית — תת-קבוצת ה-validation של fold זה כבר "נפתרת" טוב על ידי הראש הלינארי לבדו (frozen pre-trained features מספיקות). Fine-tuning לא מסייע — אפילו פוגע.

---

### ש4 — ה-Progressive Unfreezing — מה זה n_top ואילו שכבות נפתחות?

**קוד:** `src/train/finetune.py`, `unfreeze_phases` + `apply_unfreeze_phase()`

```python
# ארכיטקטורה: 3 Transformer layers (num_layers=3)
unfreeze_phases = [
    [epoch_start, n_top, lr_backbone, lr_head]

    [0,   0,  0.0,    1e-3],   # Phase 1: epochs 0–4   → backbone קפוא לחלוטין
    [5,   1,  1e-5,   5e-4],   # Phase 2: epochs 5–14  → שכבה 3 (האחרונה) נפתחת
    [15,  2,  3e-5,   3e-4],   # Phase 3: epochs 15–29 → שכבות 2+3 נפתחות
    [30, -1,  5e-5,   1e-4],   # Phase 4: epochs 30+   → הכל נפתח (כולל patch_embed)
]
```

| Phase | אפוקים | n_top | מה נפתח | backbone LR | head LR |
|-------|--------|-------|----------|------------|---------|
| 1 | 0–4 | 0 | כלום — backbone קפוא | 0.0 | 1e-3 |
| 2 | 5–14 | 1 | `encoder.layers[-1]` (שכבה 3) | 1e-5 | 5e-4 |
| 3 | 15–29 | 2 | `encoder.layers[-2:]` (שכבות 2+3) | 3e-5 | 3e-4 |
| 4 | 30+ | -1 | **הכל** — patch_embed + כל 3 שכבות | 5e-5 | 1e-4 |

**מה n_top מספר לנו:**
- `n_top=0` → backbone.parameters() `requires_grad = False`
- `n_top=k>0` → `model.encoder.layers[-k:]` ↔ `requires_grad = True` (כי הhead תמיד trainable)
- `n_top=-1` → `model.parameters()` כולם `requires_grad = True` (כולל `patch_embed`)

**עם LR warmup (v5+):** בכל unfreeze, backbone LR מתחיל מ-0.0 ועולה ב-ramp ליניארי על 5 אפוקים לערך היעד שבטבלה. **עם patience reset (v6+):** `patience_ctr = 0` בכל unfreeze, כלומר המודל מקבל budget חדש של 20 אפוקים.

---

### ש5 — Logistic Regression — Solver ו-max_iter

**קוד:** `scripts/run_e2e_cv_v2.py` → `fit_lr_model()`

```python
lr = LogisticRegression(
    C=C,
    class_weight="balanced",   # auto: n_neg/n_pos per class
    max_iter=1000,
    random_state=42
)
```

| פרמטר | ערך | הערה |
|--------|-----|------|
| **Solver** | **`lbfgs`** (ברירת מחדל sklearn) | L-BFGS — אלגוריתם gradient quasi-Newton |
| max_iter | **1000** | מספיק ל-C ∈ {0.01, 0.1, 1.0} על feature space זה |
| class_weight | **"balanced"** | sklearn מחשב אוטומטית: `n_samples / (n_classes × bincount[i])` |
| PCA לפני LR | `PCA(n_components=0.95)` | שומר על 95% מהשונות — מוריד noise ב-12 features |

**Pipeline מלא:** `X` → `StandardScaler()` → `PCA(n_components=0.95)` → `LogisticRegression(C=C, ...)`

**השפעת C על lbfgs:** lbfgs מתכנס בדרך כלל היטב על feature spaces קטנים כמו זה (12 features → PCA → ~4-8 components). הבעיה של C=0.01 ב-v6 לא הייתה convergence אלא under-regularization (features v6 אינם זהים לfeatures v5 ו-C=0.01 גרם ל-LR להיות conservative מדי).

---

### ש6 — האוגמנטציות — רק Pretrain או גם Fine-tuning?

**תשובה קצרה: שניהם, אבל באופן שונה ובמקורות שונים.**

**Pretrain augmentation** (נוסף ב-v5):
```python
# src/data/dataset.py → PretrainDataset
# מופעל רק על train batches, לא על val
gaussian_noise:  FHR σ=0.01 (p=0.5), UC σ=0.005 (p=0.5)
random_scaling:  amplitude × U[0.95, 1.05] (p=0.3)
```

**Fine-tuning augmentation** (היה קיים מ-v3/v4 — Config A, plan_2 §3.2.1):
```python
# src/train/finetune.py → run_epoch() → augment_window()
# if training and aug_cfg:  ← רק בtraining, לא בvalidation
gaussian_noise:   {sigma_fhr: 0.01, sigma_uc: 0.005, p: 0.5}
random_scaling:   {scale_min: 0.95, scale_max: 1.05,  p: 0.3}
temporal_jitter:  {max_shift: 50,                      p: 0.5}   ← נוסף ב-fine-tune
channel_dropout:  {p: 0.1}                                        ← נוסף ב-fine-tune
cutout:           {min_len: 48, max_len: 96,           p: 0.2}   ← נוסף ב-fine-tune
```

**סיכום:**
- Fine-tuning augmentation: **היה תמיד** (v3/v4/v5/v6), 5 סוגים, **רק בtrain epochs**
- Pretrain augmentation: **חדש בv5**, 2 סוגים בלבד, **רק בtrain batches**
- Validation/Test: **ללא שום augmentation** בשום שלב

---

### ש7 — EMA Smoothing — מהו ה-Alpha (Beta)?

**קוד:** `src/train/finetune.py`

```python
ema_beta = 0.8      # ← hardcoded, לא configurable

# epoch 0:
smooth_auc = val_auc

# epoch > 0:
smooth_auc = 0.8 * smooth_auc + 0.2 * val_auc
```

**תכונות ה-EMA עם beta=0.8:**

| מאפיין | ערך | משמעות |
|--------|-----|--------|
| beta (momentum) | **0.8** | משקל ה-history |
| alpha (learning rate) | 0.2 | משקל הנקודה החדשה |
| effective lookback | **1/(1−0.8) = 5 אפוקים** | EMA "זוכרת" ~5 אפוקים אחורה |
| half-life | ~3.1 אפוקים | מחצית ה-weight מה-5 אפוקים האחרונים |

**ה-EarlyStopping בודק `smooth_auc > best_smooth_auc`** — ולא את `val_auc` הגולמי.

**השלכה על ה-fold collapse:** אם `val_auc` יורד מ-0.7599 ל-0.5789 ב-epoch 5, אזי:
```
smooth_5 = 0.8×smooth_4 + 0.2×0.5789 ≈ 0.8×0.7599 + 0.2×0.5789 = 0.7237
smooth_6 = 0.8×0.7237 + 0.2×0.5789 = 0.6947
smooth_7 = 0.8×0.6947 + 0.2×0.5789 = 0.6715
...
```
גם אם `val_auc` חוזר ל-0.70 ב-epoch 10, ה-`smooth_auc` עדיין נמוך מ-0.7599 ולכן ה-patience_ctr ממשיך לעלות. בv6 עם patience reset, patience_ctr = 0 ב-epoch 5, כך שיש 20 אפוקים נוספים, אבל ה-best_smooth_auc עדיין 0.7237 מ-epoch 0 — וה-backbone צריך להשיג smooth_auc > 0.7237 כדי לשמור checkpoint.

---

### ש8 — מה זה G4a ו-G4b? מאין הגיעו?

**קוד:** `scripts/run_e2e_cv_v2.py`, שורה 23 + `azure_ml/train_azure.py`

```python
# G4 quality gates (spec §11)
G0: |ΔAUC(shared-clean)| <= 0.01 on mean(folds 0+1)
G1: val_mse < 0.015 AND probe_auc > 0.60
G2: best val AUC on 441/56 >= 0.70
G3: ft_val AUC fold0 >= 0.65
G4: mean CV AUC >= 0.70, std < 0.10   ← המפתחים
G5: LR AUC > transformer-only AUC
```

**G4 הוא gate 4 מתוך 6 gates שהוגדרו ב-`docs/plan_2.md §11`** — מסמך תכנון פנימי של הפרויקט, לא דרישה מהמאמר.

| Gate | שם | תנאי | מה זה בוחן |
|------|-----|------|------------|
| **G4a** | AUC performance | `mean_fold_AUC ≥ 0.70` | האם המודל מגיע לקרבת נייר המחקר (0.826) |
| **G4b** | AUC consistency | `std_fold_AUC < 0.10` | האם האימון יציב על כל קבוצות-החוצה |

**ביסוס G4a (0.70):** הנייר דיווח AUC=0.826 על single split. עם 5-fold CV על dataset קטן ללא SPaM, 0.70 מוגדר כ-"קרוב מספיק להוכיח שהשיטה עובדת". בפועל, עדיין לא הגענו ל-G4a (best = 0.6385 ב-v4).

**G4b (std<0.10):** עמדנו בו בכל הריצות — גם v5 עם std=0.075.

---

### ש9 — איך מוחלט מי מנצח — Clinical Threshold או Youden?

**יש שני מנגנונים שונים שיש לא לבלבל ביניהם:**

#### מנגנון A — Main Pipeline (sections 12-15, per fold)

```python
# scripts/run_e2e_cv_v2.py → clinical_threshold()
SPEC_CONSTRAINT = 0.65

def clinical_threshold(y_true, y_score, spec_constraint=0.65):
    # מחפש: max(Sensitivity) s.t. Specificity ≥ 0.65
    # Fallback: אם לא נמצא threshold עם Spec≥0.65 → Youden's J
```

**תהליך per fold:**
1. LR מאומן על train, מחזיר ציונים על val
2. `clinical_threshold()` בוחר threshold על val (clinical primary, Youden fallback)
3. זה ה-threshold שמשמש לOOF predictions (ומשם OOF AUC)
4. ה-**global OOF AUC הגלובלי אינו תלוי בthreshold** — הוא חושב ב-pooled ROC על כל הציונים

#### מנגנון B — Post-hoc Grid Search (section 16, for analysis)

```python
# בודק כל קומבינציה (AT, n_feat, C, threshold_method) על TEST set
# threshold_method ∈ ["clinical", "youden"]
# מדרג לפי mean_test_AUC
# "grid winner" = הקומבינציה הטובה ביותר
```

**חשוב:** Grid search בוחר threshold_method **לפי test AUC** — ולא כחוק לתקופות הבאות. זה ניתוח post-hoc בלבד.

**מה נראה בהיסטוריה:**
- v4 grid winner: Youden (כי val set שימש לleakage, clinical לא התאים)
- v5 grid winner: clinical (אחרי תיקון leakage, clinical ≥ Youden)
- v6 grid winner: clinical

**לסיכום:** בפועל, ה-OOF AUC הגלובלי (שהוא המדד העיקרי) תמיד משתמש ב-**clinical** עם **Youden fallback**. הgrid search רק בודק בדיעבד — ולא משנה את ה-OOF AUC שכבר חושב.

---

## נספח — ביצועים טכניים Azure ML

| בעיה | פתרון |
|------|-------|
| `.amlignore` לא עובד מלא | `_build_code_snapshot()` — temp dir עם רק src/, azure_ml/, scripts/, config/, data/splits/ |
| `VisualStudioCodeCredential` תלוי | ThreadPoolExecutor עם timeout=20s + fallback לDeviceCodeCredential |
| `torch` מPyPI = CPU-only | `torch==2.2.0+cu118` מ-PyTorch wheel index |
| NumPy≥2.0 שובר `from_numpy()` | `"numpy>=1.26,<2.0"` בconda_env.yml |
| `num_workers>0` → deadlock | hardcoded `num_workers=0` בכל DataLoader |
| Live logs לא נגישים בריצה | Azure לא מאפשר REST access ל-std_log בזמן ריצה — להמתין לסיום |

---

*מסמך מוזג מהמקורות: `docs/colab_e2e_cv_launch_guide.md`, `docs/deviation_log.md`, `results/final_model_comparison.csv`, `results/final_cv_report.csv`, logs v3/v4/v5/v6*
*נוצר ועודכן: 2026-03-04*
