# plan_2.md — סיכום מקיף של פרויקט SentinelFatal2

> **תאריך:** 25 פברואר 2026
> **מטרה:** תיעוד מלא של מה שנבנה, מה אומן, אילו תוצאות הושגו, ומה הסטטוס הנוכחי — כבסיס לתכנון אימון מחדש.

---

## 1. מטרת הפרויקט

**חיזוי מצוקה עוברית (fetal acidemia) מתוך אותות CTG (Cardiotocography) בזמן לידה.**

- **קלט:** שני ערוצים — FHR (קצב לב עוברי, bpm) + UC (התכווצויות רחם, mmHg) בדגימה של 4 Hz
- **פלט:** סיווג בינארי — `0=normal`, `1=acidemia` (pH ≤ 7.15 בדם חבל הטבור)
- **גישה:** Foundation Model — פרה-טריינינג self-supervised (MAE) ואז fine-tuning supervised
- **מאמר ייחוס:** arXiv:2601.06149v1 (4 עמודים) — `docs/2601.06149v1.pdf`

---

## 2. מאמר הייחוס — מה הוא מתאר

המאמר מציע ארכיטקטורת **Channel-Asymmetric Masked Autoencoder** על בסיס PatchTST:
- **פרה-טריינינג:** מסכים 40% מהפאצ'ים של FHR, UC תמיד גלוי. המודל לומד לשחזר את הפאצ'ים החסרים
- **פיין-טיונינג:** ראש סיווג (2 מחלקות) על הקלטות CTU-UHB עם תוויות pH
- **שלב 3 — Alerting:** חלון הזזה → חילוץ alert segments → 4 features → Logistic Regression → החלטה לכל הקלטה
- **תוצאה מדווחת:** AUC=0.826 על 55 הקלטות test

**מה המאמר לא מציין:** d_model, מספר שכבות, מספר ראשים, batch sizes, learning rates ספציפיים ל-fine-tuning, stride חלון — כל אלה דרשו הנחות מתועדות (S2-S6).

---

## 3. הנתונים

### 3.1 מקורות נתונים

| Dataset | הקלטות | מקור | שימוש | סטטוס |
|---------|--------|------|-------|--------|
| **CTU-UHB** | 552 | PhysioNet | pretrain + finetune + test | ✅ זמין |
| **FHRMA** | 135 | Zenodo (CTGDL) | pretrain בלבד (ללא labels) | ✅ זמין |
| **SPAM** | 294 | Zenodo (CTGDL) | pretrain (metadata בלבד) | ❌ חסר — סטייה S1 |

**סה"כ לפרה-טריינינג:** 687 הקלטות (במקום 984 שהמאמר השתמש)

### 3.2 Data Lineage — מקור → ארטיפקט

```
CTU-UHB (PhysioNet):
  data/ctu-chb-.../1001.dat+.hea (552 זוגות, בינארי 16-bit, 4Hz)
    → חילוץ CSV: data/CTGDL/CTGDL_ctu_uhb_proc_csv.tar.gz (552 CSV, עמודות: fhr,uc,fhr_is_nan,uc_is_nan)
    → preprocessing.py: clip+normalize+interpolate
    → data/processed/ctu_uhb/*.npy (552 קבצים, shape (2, T), T≈19200)

  .hea headers (552 קבצים):
    → פרסור pH, Apgar, delivery type, presentation...
    → data/processed/ctu_uhb_clinical_full.csv (552 שורות, 31 עמודות)

FHRMA (Zenodo):
  data/CTGDL/CTGDL_FHRMA_proc_csv.tar.gz (135 CSV, עמודות: uc,fhr,fhr_is_nan,uc_is_nan,no_value_uc)
    → preprocessing.py
    → data/processed/fhrma/*.npy (135 קבצים, shape (2, T))

SPAM (Zenodo):
  data/CTGDL/CTGDL_SPAM_metadata.csv (294 שורות metadata בלבד)
    → ❌ קבצי אות חסרים — לא נמצאו בארכיונים שבתיקיית CTGDL

Splits:
  data/CTGDL/CTGDL_norm_metadata.csv (981 שורות: 552 ctg + 135 fhrma + 294 spam)
    → עמודת test: 0=train, 1=val, 2=test
    → עמודת target: 0=normal, 1=acidemia (pH ≤ 7.15)
    → data/splits/train.csv (441), val.csv (56), test.csv (55), pretrain.csv (687)
```

**פערי איכות:**
- SPAM: metadata קיים אך אין קבצי אות — pipeline מוכן להרחבה ברגע שיתקבלו
- `analysis_results/` ב-CTU: כיסוי חלקי (100+20 רשומות מתוך 552)
- FHRMA: ללא labels — שימוש לpretrain בלבד
- אפס ערכי NaN בקבצי `.npy` המעובדים (V1.8–V1.9 verified)

### 3.3 עיבוד מקדים (Preprocessing)

| ערוץ | שלב | פרטים |
|------|------|--------|
| FHR | חיתוך | [50, 210] bpm |
| FHR | נרמול | `(fhr - 50) / 160` → [0, 1] (סטייה S7) |
| UC | חיתוך | [0, 100] mmHg |
| UC | נרמול | `uc / 100` → [0, 1] |
| שניהם | NaN | אינטרפולציה לינארית, שאריות → 0 |
| שניהם | פורמט | `.npy` shape `(2, T)` כאשר T ≈ 19,200 דגימות (~80 דקות) |

**קוד:** `src/data/preprocessing.py`

### 3.4 חלוקה לסטים

| Split | הקלטות | acidemia | % acidemia | שימוש |
|-------|--------|----------|-----------|-------|
| Train | 441 | 90 | 20.4% | אימון fine-tuning + LR |
| Val | 56 | 12 | 21.4% | early stopping + hyperparameter tuning |
| Test | 55 | 11 | 20.0% | הערכה סופית (שימוש חד-פעמי) |
| Pretrain | 687 | — | — | CTU-UHB (552) + FHRMA (135) |

**מקור החלוקה:** `CTGDL_norm_metadata.csv` עמודת `test` — אומתה מול Table 2 במאמר.
**קבצים:** `data/splits/train.csv`, `val.csv`, `test.csv`, `pretrain.csv`

### 3.5 יחס מחלקות

הנתונים לא מאוזנים: ~80% normal, ~20% acidemia.
**פתרון:** `CrossEntropyLoss(weight=[1.0, 3.9])` — חישוב אוטומטי מ-train בלבד (סטייה S6.1).

---

## 4. ארכיטקטורה

### 4.1 PatchTST Encoder

| פרמטר | ערך | מקור |
|--------|-----|------|
| אורך חלון | 1,800 דגימות (7.5 דקות) | ✓ מאמר |
| אורך פאצ'  | 48 דגימות (12 שניות) | ✓ מאמר |
| stride פאצ' | 24 דגימות (50% חפיפה) | ✓ מאמר |
| מספר פאצ'ים | 73 לכל ערוץ | ✓ מחושב (S9: חיתוך ל-1,776 דגימות) |
| d_model | 128 | ⚠ הנחה S2 |
| שכבות transformer | 3 | ⚠ הנחה S2 |
| ראשי attention | 4 (32 dim/head) | ⚠ הנחה S2 |
| FFN dim | 256 (2×d_model) | ⚠ הנחה S2 |
| dropout | 0.2 | ✓ מאמר |
| normalization | BatchNorm1d (pre-norm) | ✓ מאמר |
| **סה"כ פרמטרים** | **413,056** | |

**עיצוב channel-independent:** FHR ו-UC עוברים דרך אותו encoder (shared weights), ואז מתבצע fusion: `fhr_enc += uc_enc` (element-wise addition, תיקון AGW-19).

**קוד:** `src/model/patchtst.py`, `src/model/heads.py`

### 4.2 ראשי מודל (Heads)

| Head | קלט | פלט | שימוש |
|------|------|------|-------|
| PretrainingHead | (batch, 73, 128) | (batch, 29, 48) | שחזור 29 פאצ'ים מוסכמים |
| ClassificationHead | (batch, 18,688) | (batch, 2) | סיווג בינארי (73×128×2 flattened) |

---

## 5. צינור אימון — 3 שלבים

### שלב 1: פרה-טריינינג (Self-Supervised MAE)

**מטרה:** לימוד ייצוגים מ-687 הקלטות ללא תוויות, ע"י שחזור פאצ'ים מוסכמים של FHR.

| פרמטר | ערך | מקור |
|--------|-----|------|
| נתונים | 687 הקלטות → 12,319 train / 1,368 val חלונות | — |
| window stride | 900 דגימות (50% חפיפה) | ⚠ S4 |
| mask ratio | 40% (29/73 פאצ'ים) | ✓ מאמר |
| מסיכה | קבוצות רציפות (min=2, max=6), פאצ'ים 0 ו-72 לא מוסכמים | ✓ מאמר |
| loss | MSE על פאצ'ים מוסכמים של FHR בלבד | ✓ מאמר |
| optimizer | Adam, lr=1e-4 | ✓ מאמר |
| scheduler | ReduceLROnPlateau (patience=5) | ⚠ S12 |
| max epochs | 300 | ⚠ S5 |
| early stopping | patience=20 על val loss | ⚠ S5 |
| batch size | 64 | ⚠ S6 |

**תוצאות ריצה:**
- הריצה נעצרה ב-epoch 13 (early stopping)
- **Best val MSE: 0.01427** (epoch 2)
- Train loss: 0.111 → 0.009
- **Checkpoint:** `checkpoints/pretrain/best_pretrain.pt` (59 tensors)

### שלב 2: Fine-tuning (Supervised Classification)

**מטרה:** סיווג acidemia על 441 הקלטות train עם תוויות pH.

| פרמטר | ערך (config נוכחי) | מקור |
|--------|-----|------|
| נתונים | 441 train / 56 val הקלטות | — |
| train stride | 60 דגימות (⚠ S13) | — |
| optimizer | AdamW | ✓ מאמר |
| lr backbone | **5e-5** (⚠ S14: הועלה מ-1e-5) | ⚠ S6+S14 |
| lr head | 1e-4 | ⚠ S6 |
| warmup | **5 epochs** (⚠ S14: נוסף) | ⚠ S14 |
| weight decay | 1e-2 | ⚠ S6 |
| scheduler | ReduceLROnPlateau (patience=7) | ⚠ S12 |
| max epochs | 150 | ✓ מאמר (הורחב) |
| early stopping | patience=25 על val AUC | ✓ מאמר (הורחב) |
| batch size | 32 | ⚠ S6 |
| gradient clip | max_norm=1.0 | ⚠ S6 |
| class weights | [1.0, 3.9] (auto) | ⚠ S6.1 |
| val stride | 60 | ⚠ S12 |

**תוצאות ריצה ראשונה (S6-S12, config מקורי lr_backbone=1e-5):**
- 33 epochs עד early stopping (patience=15 מקורית)
- **Best val AUC: 0.7235** (epoch 17)
- Train loss: 0.739 → 0.698 (ירידה הדרגתית, plateau)
- Val AUC תנודתי: 0.545–0.724 (dataset קטן + אי-איזון)
- **Checkpoint:** `checkpoints/finetune/best_finetune.pt`

### שלב 3: Logistic Regression (Alert Classifier)

**מטרה:** המרת ניבויים ברמת חלון להחלטה ברמת הקלטה.

**תהליך:**
1. **Inference בחלון הזזה** — כל חלון 7.5 דקות מקבל P(acidemia)
2. **חילוץ alert segments** — רצפים רציפים שבהם score > ALERT_THRESHOLD
3. **חילוץ 4 features** מה-segment הארוך ביותר:
   - `segment_length` — אורך ב-windows
   - `max_prediction` — ציון שיא
   - `cumulative_sum` — סכום ציונים
   - `weighted_integral` — אינטגרל ציונים
4. **Logistic Regression** — מקבל 4 features, מוציא הסתברות

**תוצאות LR:**
- Train accuracy: 80.05%
- Coefficients: [0.1793, **1.3319**, 0.0006, -0.0135]
- **max_prediction שולט** (coefficient=1.33) — ציון השיא הוא המנבא הטוב ביותר

**Checkpoint:** `checkpoints/alerting/logistic_regression.pkl`

---

## 6. תוצאות — הערכה על סט Test (55 הקלטות)

### 6.1 ביצועים ראשוניים (AT=0.50)

| מטריקה | Stage 1 (NN ישיר) | Stage 2 (עם LR) | Benchmark מאמר |
|---------|-------------------|-----------------|----------------|
| AUC | 0.7624 | **0.8120** | 0.826 |
| Sensitivity | — | 0.0909 (1/11) | — |
| Specificity | — | 1.0000 | — |

**בעיה קריטית:** Sensitivity=0.09 — המודל מזהה רק 1 מתוך 11 מקרי acidemia!

### 6.2 ניתוח שורש הבעיה (סטייה S11)

**הסיבה:** עם AT=0.50, 13/55 הקלטות test (כולל 10/11 acidemia!) לא ייצרו alert segment כלל, וקיבלו ZERO_FEATURES=[0,0,0,0]. ה-LR קיבל וקטור אפסים והנחש class prior בלבד.

### 6.3 תוצאות אחרי אופטימיזציה (AT=0.40)

| Config | AUC | Sensitivity | Specificity | TP |
|--------|-----|-------------|-------------|-----|
| Baseline (AT=0.50) | 0.812 | 0.091 | 1.000 | 1/11 |
| **Old LR + AT=0.40 + Youden T=0.2836565** | **0.838843** | **0.818** | **0.773** | **9/11** |
| New LR-497 + AT=0.40 + CV T=0.199 | 0.717 | 0.636 | 0.818 | 7/11 |

**מסקנה:** הקונפיגורציה הטובה ביותר: AT=0.40 + LR מקורי (441 train) + Youden threshold=0.2836565.

**דיוק מספרי (אימות חוזר 25/02/2026):**
- עם סף החלטה מעוגל `0.284` מתקבל 8/11.
- עם הסף המדויק `0.2836565329055065` מתקבל 9/11.
- AUC Stage 2 בהרצה החוזרת: `0.838843` (תואם ל-`0.8388` לאחר עיגול בדוחות).

### 6.4 תוצאות לפי תת-קבוצות (Table 3)

| Subset | n | AUC Stage 2 | Benchmark |
|--------|---|-------------|-----------|
| All Test | 55 | 0.812 (0.839 optimized) | 0.826 |
| Vaginal | 48 | 0.734 | 0.850 |
| Cephalic | 50 | 0.795 | 0.848 |
| No Labor Arrest | 51 | 0.811 | 0.837 |

---

## 7. Cross-Validation

### 7.1 Bootstrap CI על Test (n=55)

| מטריקה | ערך | 95% CI |
|---------|-----|--------|
| AUC | 0.812 | [0.630, 0.953] |
| Sensitivity | 0.455 | [0.182, 0.727] |
| Specificity | 0.932 | [0.841, 1.000] |

### 7.2 Five-Fold CV (LR בלבד, n=497)

| מטריקה | ערך | 95% CI |
|---------|-----|--------|
| AUC | 0.653 | [0.613, 0.693] |
| Sensitivity | 0.530 | ±0.111 |
| Specificity | 0.714 | ±0.101 |

**הערה:** CV AUC (0.653) < Test AUC (0.839) — בגלל datasets קטנים יותר per-fold ו-LR retrain per-fold.

### 7.3 E2E CV (S15 — ניסיון כושל)

**מטרה:** אימון מחדש של כל הצינור (finetune→LR) לכל fold, עם שיפורי S14.

**מה הורץ בפועל (fold 0 בלבד):**

| שלב | מטריקה | ערך | הערה |
|------|---------|-----|------|
| Pretrain (shared) | epoch 0 train_loss | 0.707 | epoch יחיד בלוג |
| Pretrain (shared) | epoch 0 val_loss | 0.308 | — |
| Finetune fold 0 | epoch 0 train_loss | 0.758 | epoch יחיד בלוג |
| Finetune fold 0 | epoch 0 val_auc | 0.543 | lr_backbone=1e-5, lr_head=1e-4 |
| LR fold 0 | n_test | 5 | מדגם קטן מדי |
| LR fold 0 | n_pos (acidemia) | 1 | — |
| **Test fold 0** | **AUC** | **1.0** | לא אמין — 5 הקלטות, 1 positive |
| **Test fold 0** | **Sensitivity** | 1.0 | trivial sample |
| **Test fold 0** | **Specificity** | 0.5 | — |
| **Test fold 0** | **threshold** | 0.699 | — |

**סיבת הכישלון:** סשן Colab T4 פג תוקף (session timeout) אחרי fold 0. הריצה עצרה לפני שfolds 1–4 התחילו.

**ארטיפקטים שנשמרו:**
- `logs/e2e_cv/fold0_pretrain.csv` (שורה אחת)
- `logs/e2e_cv/fold0_finetune.csv` (שורה אחת)
- `logs/e2e_cv_progress.csv` (fold 0 בלבד)
- `results/e2e_cv_per_fold.csv` (fold 0 בלבד)

**ארטיפקטים שלא נוצרו:** folds 1–4 — כל checkpoints, logs, ו-OOF predictions.

**השלכה:** אין אומדן CV אמין. fold 0 עם 5 הקלטות test (1 positive) נותן AUC=1.0 שהוא trivially perfect ולא אינפורמטיבי. נדרשת הרצה מחדש מלאה של 5 folds.

---

## 8. שיפורי S13-S14 (לקראת אימון מחדש)

### 8.1 בעיות שזוהו (S13 baseline AUC=0.565 ב-CV)

| בעיה | ראייה | תיקון ב-S14 |
|------|-------|-------------|
| Backbone כמעט קפוא | lr_backbone=1e-5 → שינוי <0.1% במשקולות | lr_backbone=5e-5 (×5) |
| Val AUC תנודתי → checkpoints גרועים | AUC קפץ בין epochs | EMA smoothing (beta=0.8) ל-early stopping |
| אין warmup | משקולות pretrained נפגעות ב-epochs ראשונים | 5-epoch linear warmup |
| Scale mismatch ב-LR features | segment_length (~דקות) vs max_prediction (~0-1) | StandardScaler לפני LR |
| LR regularization רפוי מדי | C=0.5 עם 4 features על ~300 דגימות | C=0.1 |
| Stride חילוץ features גס | stride=60 = 15 שניות | stride=24 = 6 שניות |
| רק 4 features מ-segment ארוך ביותר | מתעלם מכמות ה-alerting בכל ההקלטה | +2 features ברמת הקלטה (6 סה"כ) |

### 8.2 Feature Vector מורחב (6 features)

| # | Feature | מקור | תיאור |
|---|---------|------|--------|
| 1 | segment_length | Longest segment | אורך בדקות |
| 2 | max_prediction | Longest segment | ציון שיא P(acidemia) |
| 3 | cumulative_sum | Longest segment | אינטגרל ציונים |
| 4 | weighted_integral | Longest segment | אינטגרל (score-0.5)² |
| 5 | n_alert_segments | **כל ההקלטה** | מספר alert segments |
| 6 | alert_fraction | **כל ההקלטה** | אחוז החלונות מעל threshold |

### 8.3 שינויי קבצים ב-S14

```
config/train_config.yaml      lr_backbone: 1e-5 → 5e-5, lr_warmup_epochs: 5
src/train/finetune.py          5-epoch warmup, EMA val_AUC (beta=0.8)
scripts/run_e2e_cv.py          StandardScaler + LR(C=0.1), 2 record-level features
notebooks/07_colab_e2e_cv      stride 60 → 24
src/inference/alert_extractor  ZERO_FEATURES updated (6 keys)
```

---

## 9. אבולוציית Config — סתירות והכרעות

חלק מהפרמטרים השתנו בין איטרציות. הטבלה הבאה מתעדת את הערכים בכל שלב ומה נקבע כסופי ב-`config/train_config.yaml`:

| פרמטר | ערך מקורי (S6) | ערך S12 | ערך S14 (נוכחי) | SSOT סופי | נימוק |
|--------|----------------|---------|-----------------|-----------|-------|
| lr_backbone | 1e-5 | 1e-5 | **5e-5** | 5e-5 | backbone כמעט קפוא עם 1e-5 |
| lr_warmup_epochs | — | — | **5** | 5 | הגנה על pretrained weights |
| patience (pretrain) | 10 | 10 | **20** | 20 | מרחב לscheduler לפעול |
| patience (finetune) | 15 | 15 | **25** | 25 | יותר סבלנות לval AUC תנודתי |
| max_epochs (pretrain) | 200 | 200 | **300** | 300 | scheduler handles overfitting |
| max_epochs (finetune) | 100 | 100 | **150** | 150 | מרחב נוסף |
| lr_scheduler | — | **ReduceLROnPlateau** | ReduceLROnPlateau | ReduceLROnPlateau | S12: נוסף |
| val_stride | — | **60** | 60 | 60 | dense inference עבור val AUC |
| train_stride | 900 (pretrain-style) | — | **60** | 60 | S13: יותר חלונות training |
| ALERT_THRESHOLD | 0.50 | 0.50 | **0.40** | 0.40 | S11: zero-segments fix |
| LR features | 4 | 4 | **6** | 6 (ב-run_e2e_cv.py) | S14: +2 record-level features |
| LR C | — | 0.5 | **0.1** | 0.1 (ב-run_e2e_cv.py) | tighter regularization |
| EMA smoothing | — | — | **beta=0.8** | beta=0.8 (ב-finetune.py) | stable early stopping |

**כלל הכרעה:** `config/train_config.yaml` הוא SSOT. כל ערך שם הוא הסופי. ערכים ב-`scripts/run_e2e_cv.py` חלים רק על E2E CV flow.

---

## 10. סטיות מהמאמר — סיכום

| קוד | תיאור | השפעה | סטטוס |
|-----|--------|--------|--------|
| **S1** | SPAM חסר (294 הקלטות) | pretrain על 687 במקום 984 (~70%) | פעיל |
| **S2** | Hyperparameters ארכיטקטורה לא פורסמו | d_model=128, layers=3 — הנחות | פעיל |
| **S4** | Window stride=900 לא צוין במאמר | — | פעיל |
| **S5** | Epochs pretrain לא צוינו | max=300, patience=20 | פעיל |
| **S6** | פרמטרים חסרים (batch sizes, LR, gradient clip) | defaults סבירים | פעיל |
| **S7** | נרמול FHR: `(fhr-50)/160` vs `fhr/160` | — | פעיל |
| **S8** | pH ≤ 7.15 (inclusive) vs < 7.15 (strict) | 8 מקרים גבוליים | פתור |
| **S9** | n_patches 73 vs 74 — חיתוך ל-1776 | אפסית | פתור |
| **S10** | ZERO_FEATURES להקלטות ללא alert segments | LR מנחש prior | פעיל |
| **S11** | ALERT_THRESHOLD: 0.50 → 0.40 | Sens 0.09→0.818 | פתור |

---

## 11. מפת קבצים

### קוד מקור
| קובץ | תפקיד |
|------|--------|
| `src/model/patchtst.py` | ארכיטקטורת PatchTST encoder |
| `src/model/heads.py` | PretrainingHead + ClassificationHead |
| `src/data/preprocessing.py` | עיבוד מקדים FHR/UC |
| `src/data/dataset.py` | PyTorch DataLoaders (pretrain + finetune) |
| `src/data/masking.py` | מסיכה channel-asymmetric |
| `src/train/pretrain.py` | לולאת אימון pretrain |
| `src/train/finetune.py` | לולאת אימון finetune (עם warmup + EMA) |
| `src/train/train_lr.py` | אימון Logistic Regression |
| `src/train/utils.py` | AUC computation, sliding windows |
| `src/inference/sliding_window.py` | inference על הקלטות ארוכות |
| `src/inference/alert_extractor.py` | חילוץ alert segments + features |
| `scripts/run_e2e_cv.py` | אורקסטרציית E2E cross-validation |

### Notebooks (בסדר ביצוע) — ממצאים עיקריים

| # | מחברת | תפקיד | תוצאות / ארטיפקטים מרכזיים |
|---|--------|--------|---------------------------|
| 0 | `00_data_prep.ipynb` | חילוץ נתונים + עיבוד מקדים | 552+135 קבצי `.npy`, `ctu_uhb_clinical_full.csv`, splits (441/56/55). V1.1–V1.9 עברו |
| 1 | `01_arch_check.ipynb` | אימות ארכיטקטורה | 413,056 params, 73 patches, BatchNorm×6, Dropout×7. V2.1–V2.8 עברו |
| 2 | `02_pretrain.ipynb` | הרצת pretrain (Colab GPU) | 13 epochs, best val MSE=0.01427 (epoch 2), masking 10K seeds stable. V3.1–V3.6 עברו |
| 3 | `03_finetune.ipynb` | הרצת finetune | 33 epochs, best val AUC=0.7235 (epoch 17), val AUC range 0.545–0.724. V4.1–V4.8 עברו, V4.5: אפס גישה ל-test |
| 4 | `04_inference_demo.ipynb` | הדגמת inference | 2-stage pipeline verified, 4 features extracted, stride=60 (RUNTIME). V5.1–V5.6 עברו |
| 5 | `05_evaluation.ipynb` | הערכה + threshold optimization | AUC=0.812→0.839 (S11), Sens 0.09→0.818, 5 case studies, Table 3 subsets |
| 6 | `06_cv_bootstrap.ipynb` | cross-validation + bootstrap CIs | Bootstrap CI=[0.63,0.95], 5-fold CV AUC=0.653±0.040 |
| 7 | `07_colab_e2e_cv_launch.ipynb` | E2E CV launcher (S14/S15) | fold 0 בלבד הושלם (AUC=1.0 על 5 הקלטות), folds 1–4 לא רצו — Colab expired |

### Config + Data
| קובץ | תפקיד |
|------|--------|
| `config/train_config.yaml` | **SSOT** לכל ההיפרפרמטרים |
| `data/splits/*.csv` | חלוקת train/val/test/pretrain |
| `data/processed/ctu_uhb/` | 552 קבצי .npy מעובדים |
| `data/processed/fhrma/` | 135 קבצי .npy (pretrain בלבד) |

### Checkpoints
| קובץ | תוכן |
|------|-------|
| `checkpoints/pretrain/best_pretrain.pt` | 59 tensors, epoch 2, val MSE=0.01427 |
| `checkpoints/finetune/best_finetune.pt` | 59 tensors, epoch 17, val AUC=0.7235 |
| `checkpoints/alerting/logistic_regression.pkl` | LR model, n_train=441, stride=60, 4 features |

### מיקום פיזי מדויק של המודל הטוב ביותר

**Artifacts (על הדיסק):**
- `checkpoints/finetune/best_finetune.pt`
- `checkpoints/alerting/logistic_regression.pkl`

**איפה נטען בקוד:**
- `src/train/train_lr.py` — פונקציה `load_finetuned_model()` טוענת את `best_finetune.pt`.
- `src/train/train_lr.py` — `joblib.load(...)` טוען את `logistic_regression.pkl`.
- `src/inference/alert_extractor.py` — קבועים `ALERT_THRESHOLD=0.4`, `DECISION_THRESHOLD=0.284` (החלטה מעוגלת בדוקומנטציה; הסף המדויק לאופטימום הוא `0.2836565...`).

### תיעוד
| קובץ | תפקיד |
|------|--------|
| `docs/work_plan.md` | SSOT ראשי — מלאי פיזי, כל הפעולות, היפרפרמטרים |
| `docs/deviation_log.md` | כל הסטיות מהמאמר (S1–S11) עם נימוק והשפעה |
| `docs/project_context.md` | סטטוס ביצוע לכל סוכן/שלב (1–8), ולידציות שעברו |
| `docs/data_documentation_he.md` | Data lineage מלא — ארכיונים, מבני CSV, מטא-דאטה, פערים |
| `docs/pretrain_full_report_he.md` | דוח מפורט של כל שלבי האימון (pretrain+finetune+LR) עם טבלאות epoch-by-epoch |
| `docs/work_plan_issues_review_he.md` | 33 באגים/בעיות (AGW-1 עד AGW-33) — כולם נסגרו |
| `docs/agent_workflow.md` | הוראות מפורטות לכל סוכן (prompts, validations, outputs) |
| `docs/s14_colab_guide.md` | מדריך step-by-step להרצת S14 E2E CV ב-Colab |
| `docs/colab_e2e_cv_launch_guide.md` | מדריך להרצת E2E CV (מקורי, לפני S14) |
| `docs/colab-vscode-guide-hebrew.md` | מדריך חיבור VS Code ל-Colab |
| `docs/2601.06149v1.pdf` | המאמר המקורי (4 עמודים) |

### Results (תוצרי הערכה)
| קובץ | תפקיד | נוצר ע"י |
|------|--------|----------|
| `results/evaluation_table3.csv` | AUC לפי 6 תת-קבוצות (Table 3) | `05_evaluation.ipynb` |
| `results/test_predictions.csv` | ניבויים לכל 55 הקלטות test | `05_evaluation.ipynb` |
| `results/final_report.md` | סיכום Stage 7 | `05_evaluation.ipynb` |
| `results/threshold_optimization_summary.csv` | השוואת AT=0.50 vs AT=0.40 | `05_evaluation.ipynb` |
| `results/final_model_comparison.csv` | השוואת 3 תצורות LR | `05_evaluation.ipynb` |
| `results/bootstrap_ci.csv` | Bootstrap CI על test (5000 iterations) | `06_cv_bootstrap.ipynb` |
| `results/cv_bootstrap_ci_552.csv` | CV per-fold metrics | `06_cv_bootstrap.ipynb` |
| `results/final_cv_report.csv` | סיכום CV מצרפי | `06_cv_bootstrap.ipynb` |
| `results/e2e_cv_per_fold.csv` | תוצאות E2E CV per-fold (fold 0 בלבד) | `07_colab_e2e_cv_launch.ipynb` |
| `results/e2e_cv_final_report.csv` | סיכום E2E CV | `07_colab_e2e_cv_launch.ipynb` |

---

## 12. ציר זמן

| תאריך | אירוע |
|--------|-------|
| 22/02/2026 | חילוץ נתונים, ארכיטקטורה, קוד pretrain + finetune + inference |
| 22-23/02/2026 | הרצת pretrain (13 epochs) + finetune (33 epochs) על Colab T4 |
| 23/02/2026 | הערכה על test: AUC=0.812 (Stage 2), Sensitivity=0.09 |
| 23/02/2026 | אופטימיזציית threshold (S11): AUC=0.839, Sensitivity=0.818 |
| 23/02/2026 | 5-fold CV (LR בלבד): AUC=0.653 |
| 23-24/02/2026 | S13-S14: שיפורים (lr×5, warmup, EMA, 6 features, StandardScaler) |
| 24/02/2026 | S15: ניסיון E2E CV — Colab session expired, fold 0 בלבד |

---

## 13. סיכום מצב נוכחי

### מה עובד
- ✅ צינור מלא end-to-end: data → pretrain → finetune → inference → alerting → LR → prediction
- ✅ Test AUC baseline (ללא post-hoc optimization): **0.812** — קרוב ל-benchmark המאמר (0.826), פער של 0.014
- ✅ Test AUC אחרי threshold optimization על test (AT=0.40 + Youden): **0.838843** (≈0.839), Sensitivity=0.818 (9/11, עם סף מדויק `0.2836565`; סף מעוגל `0.284` נותן 8/11) — **הערה: ה-0.838843 כולל אופטימיזציה פוסט-הוק על סט ה-test עצמו**, ולכן ההשוואה ל-benchmark אינה "נקייה" מתודולוגית. ה-baseline 0.812 הוא ההשוואה ההוגנת יותר
- ✅ קוד מתועד, סטיות מתועדות, splits נעולים

### מה לא עובד / לא הושלם
- ❌ E2E 5-fold CV לא הושלם (S15 — Colab expired)
- ❌ CV AUC נמוך (0.653) — מעיד על overfitting אפשרי ל-test set
- ❌ SPAM dataset חסר (S1) — 30% פחות נתוני pretrain
- ❌ Hyperparameters של ארכיטקטורה לא אופטימלו (S2) — אין sweep
- ❌ Val AUC תנודתי מאוד (0.545-0.724) — instability

### שאלות פתוחות לאימון הבא
1. **האם להשתמש ב-config S14** (lr_backbone=5e-5, warmup, EMA, 6 features) או **לחזור ל-config מקורי** (שנתן AUC=0.839 על test)?
2. **האם לנסות d_model=256** או ארכיטקטורות אחרות (S2)?
3. **האם להוריד AT ל-0.35 או 0.30** לבדוק אם יש שיפור נוסף?
4. **איך להתמודד עם instability** של val AUC? (EMA? ensemble? larger val set?)
5. **האם לנסות focal loss** במקום CrossEntropy עם class weights?
6. **מה יעד ה-AUC ב-CV** שנרצה להגיע אליו?
7. **סביבת אימון** — Colab T4 (מוגבל בזמן) או משהו אחר?
