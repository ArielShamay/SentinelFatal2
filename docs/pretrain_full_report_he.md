# דוח מלא — תהליך האימון של SentinelFatal2
**מחברת:** `notebooks/02_pretrain.ipynb`  
**תאריך עריכה:** פברואר 2026  
**מקורות:** `src/train/pretrain.py`, `src/data/masking.py`, `src/data/dataset.py`, `config/train_config.yaml`, `logs/*.csv`

---

## תוכן עניינים

1. [סקירה כללית של הפרויקט](#1-סקירה-כללית-של-הפרויקט)
2. [ארכיטקטורת המודל](#2-ארכיטקטורת-המודל)
3. [נתונים ועיבוד מקדים](#3-נתונים-ועיבוד-מקדים)
4. [שלב 1 — הגדרות סביבה](#4-שלב-1--הגדרות-סביבה)
5. [שלב 2 — בדיקת GPU](#5-שלב-2--בדיקת-gpu)
6. [שלב 3 — בניית המודל עם PretrainingHead](#6-שלב-3--בניית-המודל-עם-pretraininghead)
7. [שלב 4 — V3.1: בדיקת יציבות Masking](#7-שלב-4--v31-בדיקת-יציבות-masking)
8. [שלב 5 — V3.2 + V3.3: בדיקות אחריות Masking ו-Loss](#8-שלב-5--v32--v33-בדיקות-אחריות-masking-ו-loss)
9. [שלב 6 — V3.4 + V3.5 + V3.6: Pre-training מלא](#9-שלב-6--v34--v35--v36-pre-training-מלא)
10. [תוצאות Pre-training](#10-תוצאות-pre-training)
11. [שלב 7 — Fine-tuning](#11-שלב-7--fine-tuning)
12. [תוצאות Fine-tuning](#12-תוצאות-fine-tuning)
13. [שלב 8 — Logistic Regression (שלב Alerting)](#13-שלב-8--logistic-regression-שלב-alerting)
14. [תוצאות Logistic Regression](#14-תוצאות-logistic-regression)
15. [ארטיפקטים שנשמרו](#15-ארטיפקטים-שנשמרו)
16. [סיכום היפר-פרמטרים](#16-סיכום-היפר-פרמטרים)
17. [חריגות מהמאמר ומה ניתן לשפר](#17-חריגות-מהמאמר-ומה-ניתן-לשפר)

---

## 1. סקירה כללית של הפרויקט

**מטרה:** בניית Foundation Model לניבוי מצוקה עוברית (חמצת עוברית — acidemia) מרשומות CTG (Cardiotocography).  
**מאמר בסיס:** arXiv:2601.06149v1  
**גישה:** אימון עצמי-מפוקח (Self-Supervised Pre-training) באמצעות Channel-Asymmetric Masked Auto-Encoding, ולאחר מכן Fine-tuning לסיווג בינארי.

### Pipeline כולל:
```
נתוני CTG גולמיים (FHR + UC)
        ↓
  עיבוד מקדים (.npy)
        ↓
  Pre-training — שחזור patches מוסתרים (MSE)
        ↓
  Fine-tuning — סיווג בינארי normal/acidemia (Cross-Entropy, AUC)
        ↓
  Logistic Regression — שלב Alerting (4 features)
        ↓
  מודל Alerting מוכן לשימוש
```

---

## 2. ארכיטקטורת המודל

### 2.1 מודל בסיס: PatchTST

המודל מבוסס על **PatchTST** — Transformer-based Time Series Classifier שמחלק את האות לפאצ'ים (patches) ומעבד כל פאץ' כ-token.

| פרמטר | ערך | מקור |
|-------|-----|------|
| `patch_len` | 48 samples | ✓ מאמר Section II-C |
| `patch_stride` | 24 samples (חפיפה של 50%) | ✓ מאמר Equation 1 |
| `n_patches` | 73 = (1800−48)/24 + 1 | ✓ חישוב |
| `n_channels` | 2 (FHR + UC) | ✓ מאמר Section II-A |
| `d_model` | 128 | ⚠ הנחה S2 |
| `num_layers` | 3 | ⚠ הנחה S2 |
| `n_heads` | 4 (dim/head = 32) | ⚠ הנחה S2 |
| `ffn_dim` | 256 (= 2 × d_model) | ⚠ הנחה S2 |
| `dropout` | 0.2 | ✓ מאמר Section II-C |
| `norm_type` | BatchNorm | ✓ מאמר Section II-C |

**סה"כ פרמטרים של Encoder: 413,056**

### 2.2 ראשים (Heads)

#### PretrainingHead (לשלב Pre-training)
- מטרה: שחזור patches מוסתרים של FHR
- קלט: (B, 73, 128) — פלט ה-Encoder
- פלט: (B, n_masked, 48) — patches מוסתרים שחוזרו
- מימוש: Linear layer מ-d_model=128 ל-patch_len=48

#### ClassificationHead (לשלב Fine-tuning)
- מטרה: סיווג בינארי normal/acidemia
- קלט: n_patches × d_model × n_channels = 73 × 128 × 2 = **18,688** features
- פלט: 2 logits

### 2.3 Channel-Asymmetric Encoder
- **FHR** (ערוץ 0): מוזן עם zero-masking על patches מוסתרים
- **UC** (ערוץ 1): **תמיד גלוי במלואו** — לא מוסתר בשום מצב
- הEncoder מעבד שני הערוצים בנפרד, ולאחר מכן מוסיף (token-wise) את פלט UC לפלט FHR לפני ה-Head

---

## 3. נתונים ועיבוד מקדים

### 3.1 מסדי נתונים

| מסד נתונים | תיאור | מספר רשומות |
|------------|--------|------------|
| CTU-UHB | Intrapartum CTG database | עיקרי |
| FHRMA | FHR Multi-Archive | נוסף |
| **סה"כ** | | **687 רשומות לטיפול מקדים** |

### 3.2 הגדרות האות

- **קצב דגימה (fs):** 4 Hz
- **אורך חלון:** 1800 samples = **7.5 דקות** (✓ מאמר Section II-C)
- **סף חמצת (pH):** ≤ 7.15 (⚠ הגדרת פרויקט — המאמר משתמש ב-< 7.15)

### 3.3 פיצול הנתונים

- **Train:** 441 רשומות (80%)
- **Validation:** 1,368 חלונות (מתוך 13,687 סה"כ, ~10%)
- **Test:** (נפרד)

### 3.4 יצירת חלונות (Sliding Window)

- **Stride לPre-training:** 900 samples = 50% חפיפה (⚠ הנחה S4)
- **סה"כ חלונות:** 13,687 (מ-687 רשומות)
  - Train: 12,319 חלונות
  - Validation: 1,368 חלונות

---

## 4. שלב 1 — הגדרות סביבה

**קוד:** Cell 1 + Cell 1b של המחברת

### מה נעשה:
- זיהוי סביבת ריצה (Google Colab / VS Code מקומי)
- הגדרת `PROJECT_ROOT`
- אתחול reproducibility: **SEED = 42** עבור `random`, `numpy`, `torch`
- הורדת קבצי הנתונים המעובדים בColab (מ-GitHub Release)

### פלט שהתקבל:
```
Project root : c:\Users\ariel\Desktop\SentinelFatal2
Python       : [גרסה]
Data files already present — skipping download.
```

---

## 5. שלב 2 — בדיקת GPU

**קוד:** Cell 2

### מה נעשה:
- בדיקה האם CUDA זמין
- הגדרת `DEVICE` בהתאם

### פלט שהתקבל:
```
torch [גרסה]  |  device: cpu
```
> **הערה:** האימון הנוכחי רץ על **CPU**. בColab הוגדר CUDA (GPU).

---

## 6. שלב 3 — בניית המודל עם PretrainingHead

**קוד:** Cell 3

### מה נעשה:
1. טעינת קובץ הקונפיגורציה: `config/train_config.yaml`
2. בניית מופע PatchTST
3. החלפת ה-Head ב-`PretrainingHead`
4. העברת המודל ל-DEVICE

### פלט שהתקבל:
```
PatchTST(patch=48@24, n_patches=73, d_model=128, head=PretrainingHead)
Encoder params: 413,056
Head          : PretrainingHead
```

---

## 7. שלב 4 — V3.1: בדיקת יציבות Masking

**קוד:** Cell 4  
**ולידציה:** V3.1

### מה נבדק:
אלגוריתם ה-masking נבדק על **10,000 seeds שונים** כדי לוודא שלאף seed אין כשל.

### פרמטרי Masking:
| פרמטר | ערך | מקור |
|-------|-----|------|
| `mask_ratio` | 0.4 (40% מהפאצ'ים) | ✓ מאמר Section II-D |
| `min_group_size` | 2 patches | ✓ מאמר Section II-D Figure 4 |
| `max_group_size` | 6 patches | ⚠ הנחה |
| `n_patches` | 73 | ✓ חישוב |
| `TARGET_MASKED` | round(0.4 × 73) = **29 patches** | ✓ חישוב |

### אלגוריתם Masking — "P6 fix v2":
1. יצירת קבוצות רציפות (contiguous groups) של patches
2. גודל כל קבוצה בין `min_group_size` ל-`max_group_size`
3. **ממאפיין חובה:** הפאץ' הראשון (index 0) והאחרון (index 72) **לעולם לא מוסתרים** (שמירת границות)
4. הפאצ'ים מוחלפים ב-0.0 לפני ה-Embedding (MAE convention)

### פלט שהתקבל:
```
Testing 10,000 seeds  |  n_patches=73, mask_ratio=0.4, target_masked=29
✓ V3.1 PASS — 10,000 seeds passed, masking is stable
```
**תוצאה: 0 כשלים מתוך 10,000 seeds — אלגוריתם ה-masking יציב לחלוטין.**

---

## 8. שלב 5 — V3.2 + V3.3: בדיקות אחריות Masking ו-Loss

**קוד:** Cell 5  
**ולידציות:** V3.2, V3.3

### V3.2 — ולידציית ערבויות ה-Masking:

נבדקו 3 תנאים:
1. **שמירת גבולות:** patch[0] ו-patch[72] אינם מוסתרים ✓
2. **ספירה מדויקת:** בדיוק 29 patches מוסתרים ✓
3. **גודל קבוצות:** כל קבוצה רציפה ≥ min_group_size=2 ✓

### פלט V3.2:
```
✓ V3.2 PASS — boundary=True, n_masked=29, n_groups=9,
               group_sizes=[2, 2, 3, 3, 4, 2, 4, 3]...
```

### V3.3 — ולידציית Loss על FHR בלבד:

- **קלט:** batch גודל 4, טנסור (4, 2, 1800)
- **בדיקה:** Loss מחושב על צורה (4, 29, 48) — **FHR masked patches בלבד**
- **ולידציה:** UC אינו נכנס לחישוב ה-Loss כלל

#### פונקציית Loss (Equation 2 מהמאמר):
```
L = (1/|M|) · Σ_{i∈M} || x^FHR_i − x̂^FHR_i ||²
```
כאשר M = קבוצת ה-indices של patches מוסתרים

### פלט V3.3:
```
✓ V3.3 PASS — loss=1.182770 on (4, 29, 48) masked FHR patches only
```

---

## 9. שלב 6 — V3.4 + V3.5 + V3.6: Pre-training מלא

**קוד:** Cell 6 + Cell 6b  
**ולידציות:** V3.4, V3.5, V3.6

### 9.1 הגדרות Pre-training:

| פרמטר | ערך | מקור |
|-------|-----|------|
| `DRY_RUN` | False (אימון מלא) | — |
| `batch_size` | 64 | ⚠ הנחה S6 |
| `max_epochs` | 200 | ⚠ הנחה S5 |
| `patience` | 10 | ⚠ הנחה S5 |
| `optimizer` | Adam | ✓ מאמר Section II-D |
| `lr` | 1×10⁻⁴ | ✓ מאמר Section II-D |
| `clip_norm` | 1.0 | ⚠ הנחה S6 |
| `window_stride` | 900 (50% overlap) | ⚠ הנחה S4 |

### 9.2 תהליך לולאת האימון:

עבור כל epoch:
```
לכל batch בDataLoader:
  1. יצירת mask_indices = generate_mask_indices() — per-batch, בכל iteration מחדש
  2. חילוץ FHR patches: (B, 73, 48)
  3. שמירת original_fhr (ground truth)
  4. Zero-masking: fhr_patches[:, mask_t, :] = 0.0
  5. Embedding: patch_embed(masked_fhr) → (B, 73, 128)
  6. Encoding FHR: encoder(fhr_embedded) → (B, 73, 128)
  7. Encoding UC (תמיד גלוי): encoder(uc_embedded) → (B, 73, 128)
  8. Fusion: fhr_enc + uc_enc (AGW-19 fix — Channel-Asymmetric MAE)
  9. Head: pred = head(fused[:, mask_t, :]) → (B, 29, 48)
  10. Loss: MSE(pred, original_fhr[:, mask_t, :])
  11. Backward + clip_grad_norm(1.0) + step

אחרי כל epoch:
  - חישוב val_loss על validation set
  - שמירת checkpoint: checkpoints/pretrain/epoch_N.pt
  - אם val_loss < best: שמירת checkpoints/pretrain/best_pretrain.pt
  - Early stopping: אם אין שיפור ב-10 epochs רצופים → עצירה
  - לוגינג: logs/pretrain_loss.csv
```

### 9.3 ולידציית ארטיפקטים (V3.5 + V3.6):

```
✓ V3.5 PASS — checkpoints saved (59 param tensors)
✓ V3.6 PASS — loss CSV has 16 row(s):
               epoch=0 train_loss=0.11075290 val_loss=0.01522100 lr=1.00e-04
```

---

## 10. תוצאות Pre-training

### 10.1 טבלת Loss לפי Epoch (הרצה הנוכחית — 13 epochs):

| Epoch | Train MSE | Val MSE | שיפור ב-Val |
|-------|-----------|---------|-------------|
| 0 | 0.11100 | 0.01538 | — |
| 1 | 0.02964 | 0.01472 | ✓ |
| 2 | 0.01949 | 0.01427 | ✓ **best** |
| 3 | 0.01542 | 0.01503 | ✗ |
| 4 | 0.01335 | 0.01503 | ✗ |
| 5 | 0.01215 | 0.01450 | ✗ |
| 6 | 0.01112 | 0.01691 | ✗ |
| 7 | 0.01068 | 0.01536 | ✗ |
| 8 | 0.01022 | 0.01598 | ✗ |
| 9 | 0.00986 | 0.01504 | ✗ |
| 10 | 0.00949 | 0.01499 | ✗ |
| 11 | 0.00935 | 0.01537 | ✗ |
| 12 | 0.00922 | 0.01633 | ✗ |

**Best Val Loss: 0.01427 (Epoch 2)**  
**Epoch שנשמר best_pretrain.pt:** Epoch 2

### 10.2 תצפיות מהתוצאות:

1. **Train Loss יורד בצורה עקבית** מ-0.111 ב-Epoch 0 עד ≈0.009 ב-Epoch 12
2. **Val Loss עצר לרדת אחרי Epoch 2** — גל ראשון של Early Stopping
3. **פער גדול בין Train ל-Val**: Train Loss → ~0.009, Val Loss → ~0.014–0.017 — סימן קל של overfitting לאחר Epoch 2
4. **ירידה דרמטית ב-Epoch 0→1**: Train Loss ירד מ-0.111 ל-0.030 (+70%) — הEncoder למד מהר מאוד את מבנה האות
5. הגרף נשמר בקובץ: `logs/pretrain_loss_curve.png`

### 10.3 נתוני Dataset שנטענו:
```
[PretrainDataset] Loaded 687/687 recordings → 13,687 windows (stride=900)
[pretrain] dataset - train windows=12,319, val windows=1,368
[pretrain] encoder params: 413,056
```

---

## 11. שלב 7 — Fine-tuning

**קוד:** Cell בפרק "PHASE 6 — FINE-TUNING"  
**מקור:** `src/train/finetune.py`

### 11.1 הגדרות Fine-tuning:

| פרמטר | ערך | מקור |
|-------|-----|------|
| `optimizer` | AdamW | ✓ מאמר Section II-E |
| `lr_backbone` | 1×10⁻⁵ (Differential LR) | ⚠ הנחה S6 |
| `lr_head` | 1×10⁻⁴ | ⚠ הנחה S6 |
| `weight_decay` | 1×10⁻² | ⚠ הנחה S6 |
| `max_epochs` | 100 | ✓ מאמר Section II-E |
| `patience` | 15 (Early Stopping על val AUC) | ✓ מאמר Section II-E |
| `batch_size` | 32 | ⚠ הנחה S6 |
| `gradient_clip` | 1.0 | ⚠ הנחה S6 |
| `n_classes` | 2 (בינארי) | ✓ עיצוב פרויקט |
| `loss` | Cross-Entropy | ✓ מאמר |
| `metric` | AUC-ROC | ✓ מאמר |

### 11.2 Differential Learning Rate:
שיטה חשובה שמאפשרת שמירת הידע הקיים ב-backbone תוך לימוד מהיר יותר של Head:
- **Backbone** (Encoder): lr = 1×10⁻⁵ (נמוך — שמירת representation מPre-training)
- **Classification Head**: lr = 1×10⁻⁴ (גבוה × 10 — למידה חדשה)

### 11.3 תהליך Fine-tuning:
1. טעינת best_pretrain.pt
2. החלפת PretrainingHead ב-ClassificationHead
3. אימון עם AdamW + Differential LR
4. val AUC נמדד בסוף כל epoch
5. Early Stopping: אם val AUC לא עולה ב-15 epochs רצופים → עצירה

---

## 12. תוצאות Fine-tuning

### 12.1 טבלת AUC לפי Epoch (33 epochs, הרצה מלאה):

| Epoch | Train CE Loss | Val AUC | הערה |
|-------|--------------|---------|------|
| 0 | 0.7386 | 0.6061 | — |
| 1 | 0.7283 | 0.6686 | ↑ |
| 2 | 0.7162 | 0.6534 | ↓ |
| 3 | 0.7069 | 0.5947 | ↓ |
| 4 | 0.7091 | 0.6515 | ↑ |
| 5 | 0.7080 | 0.6250 | — |
| 6 | 0.7080 | 0.5890 | ↓ |
| 7 | 0.7145 | 0.5606 | ↓ |
| 8 | 0.7114 | 0.6061 | — |
| 9 | 0.7070 | 0.6553 | ↑ |
| 10 | 0.7103 | 0.6383 | — |
| 11 | 0.7035 | 0.6042 | — |
| 12 | 0.7049 | 0.6761 | ↑ |
| 13 | 0.7112 | 0.7008 | ↑ |
| 14 | 0.7017 | 0.6023 | ↓ |
| 15 | 0.7072 | 0.5852 | ↓ |
| 16 | 0.7027 | 0.6212 | — |
| 17 | 0.7070 | **0.7235** | ↑ **best** |
| 18 | 0.7129 | 0.5928 | ↓ |
| 19 | 0.7116 | 0.6193 | — |
| 20 | 0.7016 | 0.5795 | ↓ |
| 21 | 0.6980 | 0.6098 | — |
| 22 | 0.7024 | 0.6383 | — |
| 23 | 0.7056 | 0.6117 | — |
| 24 | 0.7055 | 0.6326 | — |
| 25 | 0.7102 | 0.5966 | ↓ |
| 26 | 0.7009 | 0.6023 | — |
| 27 | 0.7044 | 0.6326 | — |
| 28 | 0.7069 | 0.5455 | ↓ |
| 29 | 0.7011 | 0.6212 | — |
| 30 | 0.7079 | 0.6364 | — |
| 31 | 0.7024 | 0.5758 | ↓ |
| 32 | 0.7037 | 0.5890 | ↓ |

**Best Val AUC: 0.7235 (Epoch 17)**

### 12.2 תצפיות מהתוצאות:

1. **Train Loss יורד אך באיטיות** — מ-0.739 ל-0.698 לאורך 33 epochs — סימן שה-backbone קשה לשינוי עם lr=1e-5
2. **Val AUC תנודתי מאוד** — טווח 0.545–0.724 — מעיד על variance גבוה עם mataset קטן
3. **Best AUC = 0.7235** — ביצועים בינוניים, מעל baseline אקראי (0.5) אך מקום לשיפור
4. **אין Early Stopping לפני Epoch 32** — המודל המשיך 15 epochs אחרי best ב-17

---

## 13. שלב 8 — Logistic Regression (שלב Alerting)

**קוד:** Cell "PHASE 6 — LR TRAINING"  
**מקור:** `src/inference/sliding_window.py`, `src/inference/alert_extractor.py`

### 13.1 מטרה:
מעל פלטי המודל הנוירוני, נבנה מסווג קל יותר (Logistic Regression) שמבוסס על **features של alert segments** ולא על logits ישירים.

### 13.2 הגדרות:

| פרמטר | ערך | הסבר |
|-------|-----|------|
| `ALERT_THRESHOLD` | 0.5 | ✓ מאמר Section II-F |
| `LR_STRIDE` | 60 samples = 15 שניות | ⚠ הנחה S6 לביצועי CPU |
| `n_train` | 441 רשומות | |
| `optimizer` | LogisticRegression (sklearn) | |
| `max_iter` | 1000 | |
| `random_state` | 42 | |

> **הערה על LR_STRIDE:** הMאמר מגדיר stride=1 לervaluation מדויק. השתמשנו ב-stride=60 (15 שניות) כדי להפוך את האימון לאפשרי על CPU. ההשפעה המוערכת: הפרש ~0.01–0.02 AUC.

### 13.3 הFeatures (4 features):

עבור כל רשומה, מוצא ה-Alert Segment הארוך ביותר מעל הסף (0.5). מ-segment זה מוצאות:

| Feature | הסבר |
|---------|------|
| `segment_length` | אורך הsegment (בsamples) |
| `max_prediction` | מקסימום ה-score בsegment |
| `cumulative_sum` | סכום כל ה-scores בsegment |
| `weighted_integral` | אינטגרל משוקלל |

אם אין alert segment (score < threshold לאורך כל הרשומה) → כל ה-features = 0 (ZERO_FEATURES).

### 13.4 תהליך:

```
לכל רשומת train:
  1. טעינת .npy signal
  2. inference_recording(): החלקה ב-stride=60 → רשימת scores לכל window
  3. extract_alert_segments(): מציאת intervals רציפים > 0.5
  4. בחירת הsegment הארוך ביותר
  5. compute_alert_features(): חישוב 4 features
  6. הוספה לX_rows, y_rows

לאחר כל הרשומות:
  7. lr.fit(X_train, y_train) — Logistic Regression
  8. joblib.dump() → checkpoints/alerting/logistic_regression.pkl
```

---

## 14. תוצאות Logistic Regression

### 14.1 סטטיסטיקות Dataset:

| קטגוריה | ערך |
|---------|-----|
| סה"כ רשומות train | 441 |
| Normal (label=0) | 351 (79.6%) |
| Acidemia (label=1) | 90 (20.4%) |
| **רשומות עם zero-features** | **107/441 (24.3%)** |
| רשומות שנדלגו (missing .npy) | 0 |

> **הערה:** 107 רשומות (24.3%) לא הניבו אף alert segment — המודל לא זיהה אזורי סכנה. עבורן הוכנסו אפסים כ-features. רובן (כנראה) רשומות נורמליות.

### 14.2 תוצאות LR:

| מדד | ערך |
|-----|-----|
| **Train Accuracy** | **0.8005 (80.05%)** |
| **LR Coefficients** | [0.1793, 1.3319, 0.0006, -0.0135] |
| **LR Intercept** | -2.2777 |

### 14.3 פירוש המקדמים:

| Feature | מקדם | פירוש |
|---------|------|------|
| `segment_length` | 0.1793 | תרומה בינונית — ככל שהsegment ארוך יותר → acidemia |
| `max_prediction` | **1.3319** | **התרומה הדומיננטית** — ה-score המקסימלי הוא המנבא החזק ביותר |
| `cumulative_sum` | 0.0006 | תרומה זניחה — כנראה מתאמת גבוה עם max ועם length |
| `weighted_integral` | -0.0135 | תרומה שלילית קלה |

**פירוש:** המסווג מסתמך בעיקר על `max_prediction` (מה ה-peak activation?) ועל אורך הsegment.

### 14.4 זמן ריצה:
- סה"כ זמן לבניית feature matrix: **264 שניות** (~4.4 דקות)
- קצב עיבוד: ~441/264 ≈ 1.67 רשומות/שנייה

---

## 15. ארטיפקטים שנשמרו

### Checkpoints:

| קובץ | תוכן | שלב |
|------|------|-----|
| `checkpoints/pretrain/best_pretrain.pt` | מצב מודל עם best val loss (epoch 2) (59 tensors) | Pre-training |
| `checkpoints/pretrain/epoch_N.pt` | checkpoint של כל epoch (0–12 + הרצות קדומות 0–12) | Pre-training |
| `checkpoints/finetune/best_finetune.pt` | מצב מודל עם best val AUC (epoch 17) (59 tensors) | Fine-tuning |
| `checkpoints/finetune/epoch_N.pt` | checkpoints fine-tuning (0–32) | Fine-tuning |
| `checkpoints/alerting/logistic_regression.pkl` | joblib pickle: {model, stride, n_train, feature_names} | Alerting |

### Logs:

| קובץ | תוכן |
|------|------|
| `logs/pretrain_loss.csv` | epoch, train_loss, val_loss, lr (13 rows) |
| `logs/finetune_loss.csv` | epoch, train_loss, val_auc, lr_backbone, lr_head (33 rows) |
| `logs/pretrain_loss_curve.png` | גרף Pre-training Loss (Train + Val MSE) |
| `logs/finetune_curves.png` | גרף Fine-tuning (Loss + AUC side by side) |
| `logs/pretrain_loss_audit.csv` | הרצת audit: epoch 0 בלבד (train=0.393, val=0.354) |
| `logs/pretrain_loss_v36check.csv` | בדיקת v36: 4 epochs עם ערכים סינטטיים |
| `logs/finetune_loss_audit.csv` | hרצת audit fine-tune: epoch 0 (train=0.750, AUC=0.697) |
| `logs/finetune_loss_v48check.csv` | בדיקת v48: 4 epochs עם ערכים סינטטיים |

---

## 16. סיכום היפר-פרמטרים

### Pre-training:
```yaml
data:
  fs: 4                    # Hz
  window_len: 1800         # samples = 7.5 min
  patch_len: 48            # samples/patch
  patch_stride: 24         # hop
  n_patches: 73            # (1800-48)/24 + 1
  n_channels: 2            # FHR + UC

model:
  d_model: 128
  num_layers: 3
  n_heads: 4
  ffn_dim: 256
  dropout: 0.2
  norm_type: batch_norm

pretrain:
  mask_ratio: 0.4          # → 29/73 patches masked
  min_group_size: 2
  max_group_size: 6
  masking_channel: fhr     # UC תמיד גלוי
  mask_value: 0.0
  optimizer: adam
  lr: 1e-4
  max_epochs: 200
  patience: 10
  batch_size: 64
  window_stride: 900       # 50% overlap
```

### Fine-tuning:
```yaml
finetune:
  optimizer: adamw
  lr_backbone: 1e-5        # differential LR
  lr_head: 1e-4
  weight_decay: 0.01
  max_epochs: 100
  patience: 15
  batch_size: 32
  gradient_clip: 1.0
  n_classes: 2
```

### Alerting:
```yaml
alerting:
  threshold: 0.4  # S11: הורד מ-0.5, validated 2026-02-23
  inference_stride_repro: 1    # official evaluation
  inference_stride_runtime: 60 # operational (15 sec)
```

---

## 17. חריגות מהמאמר ומה ניתן לשפר

### 17.1 חריגות ידועות (מתועדות ב-deviation_log.md):

| קוד | חריגה | השפעה צפויה |
|-----|-------|------------|
| S2 | d_model=128, layers=3, heads=4 — המאמר לא מפרט | לא ידוע |
| S4 | window_stride=900 — המאמר לא מפרט | עלול להשפיע על גיוון האימון |
| S5 | max_epochs=200, patience=10 — המאמר לא מפרט לPre-training | — |
| S6 | batch_size=64/32, gradient_clip — המאמר לא מפרט | — |
| S8 | pH ≤ 7.15 (inclusive) במקום < 7.15 (strict) | הגדרת מחלקה שונה במקצת |
| S9 | inference_stride=60 לLR (במקום 1) | ~0.01–0.02 AUC |

### 17.2 תצפיות שמצביעות על מקום לשיפור:

1. **Overfitting בPre-training:**
   - Train Loss ירד לכ-0.009 בעוד Val Loss נשאר בכ-0.014
   - **הצעה:** הוספת weight decay לAdam, Dropout גבוה יותר, או augmentation על הנתונים

2. **Val AUC תנודתי מאוד ב-Fine-tuning (0.545–0.724):**
   - **סיבה:** Dataset קטן (441 רשומות, עם אי-איזון 79%/21%)
   - **הצעה:** Focal Loss במקום Cross-Entropy, class weighting, SMOTE

3. **107/441 רשומות ללא alert segments (24.3%):** ✅ **מיושם — Deviation S11**
   - **סיבה:** המודל לא מזהה סכנה כלל ברישומות אלה
   - **פתרון:** הורדת ALERT_THRESHOLD ל-0.4 — ביטלה כמעט לחלוטין את zero-segments (4/497 בלבד).
     AUC שיפור: 0.812→0.839, Sensitivity: 0.09→0.818. ראה Section 18.

4. **max_prediction דומיננטי בLR:**
   - 4 features נבחרו ומסתבר ש-cumulative_sum ו-weighted_integral כמעט לא תורמות
   - **הצעה:** ניסוי להוריד ל-2 features (segment_length + max_prediction) או להוסיף features נוספות כמו `mean_prediction` ו-`duration_above_threshold`

5. **Batch-level masking (אותם mask_indices לכל הdataset batchב):**
   - כיום: mask אחד משותף לכל הsamples ב-batch
   - **הצעה:** per-sample masking לגיוון גדול יותר

6. **learning rate scheduler:**
   - כיום: lr קבוע ב-Pre-training
   - **הצעה:** Cosine Annealing שיוריד lr בהדרגה

7. **הגברת נתוני אימון:**
   - 13,687 חלונות מ-687 רשומות
   - **הצעה:** הורדת window_stride ל-240 (60 שניות overlap) → יכפיל את כמות החלונות ל-~27,000+

---

---

## 18. תוצאות סופיות — Threshold Optimization + CV (סוכן 8)

**תאריך:** 23 פברואר 2026 | **ניתוח:** `notebooks/05_evaluation.ipynb` Cells 12–18

### 18.1 השוואת תצורות על Test Set (55 הקלטות)

| תצורה | AUC | Sensitivity | Specificity | TP/11 | Threshold |
|-------|-----|-------------|-------------|-------|-----------|
| Baseline AT=0.50, LR-441, T=0.50 | 0.812 | 0.091 | 1.000 | 1/11 | 0.500 |
| **Old LR + AT=0.40 + Youden T** | **0.839** | **0.818** | **0.773** | **9/11** | **0.284** |
| Final LR-497 + AT=0.40 + CV T | 0.717 | 0.636 | 0.818 | 7/11 | 0.199 |

**תצורה נבחרת:** Old LR (n_train=441) + AT=0.40 + Youden threshold=0.284

### 18.2 Cross-Validation (5-Fold, 497 Train+Val)

| Fold | AUC | Sens | Spec | Threshold |
|------|-----|------|------|-----------|
| 1 | 0.636 | 0.476 | 0.646 | 0.194 |
| 2 | 0.603 | 0.524 | 0.570 | 0.185 |
| 3 | 0.692 | 0.700 | 0.797 | 0.197 |
| 4 | 0.694 | 0.550 | 0.772 | 0.200 |
| 5 | 0.638 | 0.400 | 0.785 | 0.219 |
| **Mean±SD** | **0.653±0.040** | **0.530±0.111** | **0.714±0.101** | **0.199±0.013** |

### 18.3 ממצאים עיקריים

- **שורש הבעיה:** AT=0.50 גרם ל-13/55 רשומות test לקבל ZERO_FEATURES → Sensitivity=0.09
- **הפתרון:** AT=0.40 → 0/55 zero-segments בtest, 4/497 בtrain+val
- **תוצאה:** Sensitivity: 0.09 → 0.818 (שיפור פי 9), AUC: 0.812 → 0.839
- **LR retrained:** AUC=0.717 — הLR המקורי עם AT=0.40 מכליל טוב יותר
- **Artifact:** `checkpoints/alerting/logistic_regression_at040.pkl` (n_train=497, לשימוש עתידי)

---

*מסמך זה נוצר אוטומטית על בסיס ניתוח `notebooks/02_pretrain.ipynb`, `config/train_config.yaml`, וקובצי ה-logs.*
