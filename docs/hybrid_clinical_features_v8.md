# v8 — Hybrid Rule-Based + AI Feature System

**תאריך:** מרץ 2026
**גרסה:** Training Run v8
**מחבר:** Claude (Sonnet 4.6) + Ariel Shamay

---

## תוכן עניינים

1. [מדוע הוספנו זאת — הבעיה שפותרים](#1-מדוע-הוספנו-זאת--הבעיה-שפותרים)
2. [הרעיון הכללי — ארכיטקטורת ה-Hybrid](#2-הרעיון-הכללי--ארכיטקטורת-ה-hybrid)
3. [זרימת הנתונים המלאה](#3-זרימת-הנתונים-המלאה)
4. [המודולים הקליניים החדשים](#4-המודולים-הקליניים-החדשים)
5. [טבלת ה-Features (23 בסה"כ)](#5-טבלת-ה-features-23-בסהכ)
6. [מיקומי הקוד בפרויקט](#6-מיקומי-הקוד-בפרויקט)
7. [איך זה משתלב בצינור האימון הקיים](#7-איך-זה-משתלב-בצינור-האימון-הקיים)
8. [חוקים קליניים — מה כל מודול מזהה](#8-חוקים-קליניים--מה-כל-מודול-מזהה)
9. [מה ה-Grid Search בודק עכשיו](#9-מה-ה-grid-search-בודק-עכשיו)
10. [מה ציפינו לקבל — תחזיות ביצועים](#10-מה-ציפינו-לקבל--תחזיות-ביצועים)
11. [מקור הקוד — SentinelFetal](#11-מקור-הקוד--sentinelfetal)

---

## 1. מדוע הוספנו זאת — הבעיה שפותרים

### הבעיה הקודמת (v3–v7)

ה-LR Classifier (Logistic Regression) שמקבל את ההחלטה הסופית עבד עם **12 features** שכולם גזורים מה-PatchTST:

```
PatchTST → anomaly scores per window → segment statistics → 12 features → LR → P(חמצון)
```

**הבעיה:** כל 12 ה-features מתארים *כמה* הסיגנל נראה חריג — אך **לא מה הוא**. הדוגמאות:

| תופעה CTG | מה ה-PatchTST רואה | מה קורה בפועל |
|-----------|---------------------|----------------|
| Early deceleration | Anomaly score גבוה | **תקין לחלוטין** — לחץ ראש |
| Late deceleration  | Anomaly score גבוה | **פתולוגי** — אי-ספיקה שלייתית |
| שינת עובר (variability נמוכה) | Anomaly score בינוני | **תקין** — שלב שינה עוברי |
| Absent variability + decelerations | Anomaly score גבוה | **קטגוריה 3 — מסוכן** |

ה-LR לא יכול להבדיל בין שני המקרים הראשונים — שניהם מייצרים anomaly score דומה.

### הפתרון (v8)

מוסיפים **11 features קליניים** שמחושבים ישירות מהסיגנל הגולמי לפי חוקים רפואיים מוגדרים:

```
PatchTST → 12 features ⎤
                         ├→ concat(23) → LR → P(חמצון)
Rules    → 11 features ⎦
```

עכשיו ה-LR יודע: "יש 3 late decelerations + variability נמוכה" → אלה שני signals מאד ספציפיים שאין ל-PatchTST גישה אליהם במפורש.

---

## 2. הרעיון הכללי — ארכיטקטורת ה-Hybrid

```
┌─────────────────────────────────────────────────────────────────┐
│                    קובץ .npy לכל הקלטה                          │
│              shape: (2, T) — מנורמל [0,1] ב-4 Hz               │
│         Channel 0: FHR (דופק עוברי)  Channel 1: UC (צירים)      │
└──────────────────────┬──────────────────────────────────────────┘
                       │
           ┌───────────┴───────────────┐
           │                           │
           ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────────────────┐
│   PatchTST Model     │   │      Clinical Rule Engine            │
│  (Transformer)       │   │   (src/features/clinical_extractor)  │
│                      │   │                                      │
│  sliding window      │   │  denormalize:                        │
│  stride=24 samples   │   │    FHR → bpm (×160+50)              │
│  ↓                   │   │    UC  → mmHg (×100)                 │
│  P(acidemia) per     │   │  ↓                                   │
│  window              │   │  baseline.py → baseline_bpm          │
│  ↓                   │   │  variability.py → amplitude, cat     │
│  extract_recording_  │   │  decelerations.py → counts, depth   │
│  features()          │   │  sinusoidal.py → detected (0/1)     │
│  ↓                   │   │  tachysystole.py → detected (0/1)   │
│  12 features         │   │  ↓                                   │
└──────────┬───────────┘   │  11 features                         │
           │               └──────────────┬───────────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          │
                          ▼
               ┌─────────────────────┐
               │  concat → 23 feats  │
               │  StandardScaler     │
               │  PCA (optional)     │
               │  LogisticRegression │
               │  ↓                  │
               │  P(acidemia) ∈[0,1] │
               └─────────────────────┘
```

---

## 3. זרימת הנתונים המלאה

### שלב 1: Pre-processing (נעשה לפני האימון)
**קובץ:** [`src/data/preprocessing.py`](../src/data/preprocessing.py)

כל הקלטה CTG מעובדת ונשמרת כקובץ `.npy`:
- FHR: ניקוי artifacts (< 50 או > 220 bpm → NaN) → interpolation → normalization: `(fhr - 50) / 160`
- UC: ניקוי flat regions → interpolation → normalization: `uc / 100`
- Output: shape `(2, T)` כאשר `T = duration_seconds × 4`

**חשוב:** כל המודולים הקליניים מבצעים **denormalization** לפני העיבוד:
```python
fhr_bpm = signal[0] * 160.0 + 50.0   # חזרה ל-bpm
uc_mmhg = signal[1] * 100.0           # חזרה ל-mmHg
```

---

### שלב 2: Pretrain
**קובץ:** [`src/train/pretrain.py`](../src/train/pretrain.py)

PatchTST לומד לשחזר חלקים מוסתרים של סיגנל FHR. זה **לא משתנה** ב-v8. ה-pretraining לא מודע ל-clinical features.

---

### שלב 3: Finetune (5-fold CV)
**קובץ:** [`src/train/finetune.py`](../src/train/finetune.py)

לכל fold, ה-PatchTST מתאמן על binary classification (acidemia: כן/לא). זה **לא משתנה** ב-v8.

---

### שלב 4: Feature Extraction — השינוי המרכזי של v8
**קובץ:** [`scripts/run_e2e_cv_v2.py`](../scripts/run_e2e_cv_v2.py) — פונקציה `extract_features_for_split()` (שורה 212)

```python
for _, row in df.iterrows():
    signal = np.load(npy, mmap_mode="r")   # (2, T) normalized

    # ── שלב 4א: PatchTST features (12) ──────────────────────────────
    windows = sliding_windows(signal, window_len=1800, stride=24)
    scores_list = []
    for batch in ...:
        probs = softmax(model(batch))[:, 1]  # P(acidemia) per window
        scores_list.append((start_sample, prob))

    patchtst_feats = extract_recording_features(
        scores_list, threshold=best_at, n_features=12
    )  # → dict של 12 features

    # ── שלב 4ב: Clinical features (11) — חדש ב-v8 ──────────────────
    clinical_feats = extract_clinical_features(signal, fs=4.0)
    # → רשימה של 11 floats

    # ── שלב 4ג: concat (23) ─────────────────────────────────────────
    combined = list(patchtst_feats.values()) + clinical_feats
    X_rows.append(combined)   # X matrix: (N_recordings, 23)
```

---

### שלב 5: LR Training + Evaluation
**קובץ:** [`azure_ml/train_azure.py`](../azure_ml/train_azure.py)

```python
# Inner-CV לבחירת C (נשאר מ-v7)
best_c = select_lr_c_inner_cv(X_tr, y_tr, X_vl, y_vl)

# LR על X matrix (N, 23) — עובד עם כל מספר features אוטומטית
scaler, pca, lr_m = fit_lr_model(X_tr, y_tr, C=best_c, use_pca=True)

# Grid search בוחן: n_feat ∈ [4, 12, 23]
Xtr = Xtr_full[:, :n_feat]   # 4=רק ראשוניים, 12=רק PatchTST, 23=הכל
```

---

## 4. המודולים הקליניים החדשים

### מבנה הקבצים

```
SentinelFatal2/
├── src/
│   ├── rules/                          ← NEW — חוקים קליניים
│   │   ├── __init__.py
│   │   ├── baseline.py                 ← חישוב FHR baseline
│   │   ├── variability.py              ← חישוב variability
│   │   ├── decelerations.py            ← זיהוי decelerations
│   │   ├── sinusoidal.py               ← זיהוי דפוס סינוסואידלי
│   │   └── tachysystole.py             ← זיהוי tachysystole
│   └── features/                       ← NEW — מאגד features
│       ├── __init__.py
│       └── clinical_extractor.py       ← ממשק מרכזי
```

---

## 5. טבלת ה-Features (23 בסה"כ)

### Features 1–12: PatchTST Anomaly Features (לא שונו)

| # | שם | תיאור |
|---|-----|--------|
| 1 | `segment_length` | משך הסגמנט הארוך ביותר עם anomaly score גבוה (דקות) |
| 2 | `max_prediction` | שיא P(acidemia) בסגמנט הארוך |
| 3 | `cumulative_sum` | אינטגרל של scores × זמן בסגמנט |
| 4 | `weighted_integral` | Σ(score − 0.5)² × dt |
| 5 | `n_alert_segments` | מספר סגמנטים שחורגים מ-threshold |
| 6 | `alert_fraction` | אחוז חלונות מעל threshold |
| 7 | `mean_prediction` | ממוצע scores בסגמנט הארוך |
| 8 | `std_prediction` | סטיית תקן scores בסגמנט |
| 9 | `max_pred_all_segments` | שיא גלובלי בכל הסגמנטים |
| 10 | `total_alert_duration` | סה"כ זמן התראה (דקות) |
| 11 | `recording_max_score` | הציון הגבוה ביותר בהקלטה |
| 12 | `recording_mean_above_th` | ממוצע כל הציונים מעל threshold |

---

### Features 13–23: Clinical Rule Features (חדש ב-v8)

| # | שם Feature | מודול | ערך תקין | משמעות קלינית |
|---|------------|-------|----------|----------------|
| 13 | `baseline_bpm` | `baseline.py` | 110–160 bpm | Baseline דופק עוברי |
| 14 | `is_tachycardia` | `baseline.py` | 0.0 | 1 אם baseline > 160 bpm |
| 15 | `is_bradycardia` | `baseline.py` | 0.0 | 1 אם baseline < 110 bpm |
| 16 | `variability_amplitude_bpm` | `variability.py` | 6–25 bpm | עוצמת variability |
| 17 | `variability_category` | `variability.py` | 2.0 (moderate) | 0=absent, 1=minimal, 2=moderate, 3=marked |
| 18 | `n_late_decelerations` | `decelerations.py` | 0 | מספר late decels (פתולוגי בחזרה) |
| 19 | `n_variable_decelerations` | `decelerations.py` | נמוך | מספר variable decels |
| 20 | `n_prolonged_decelerations` | `decelerations.py` | 0 | מספר decelerations > 2 דקות |
| 21 | `max_deceleration_depth_bpm` | `decelerations.py` | — | עומק ה-deceleration העמוקה |
| 22 | `sinusoidal_detected` | `sinusoidal.py` | 0.0 | 1 אם נמצא דפוס סינוסואידלי |
| 23 | `tachysystole_detected` | `tachysystole.py` | 0.0 | 1 אם >5 צירים/10 דקות |

---

## 6. מיקומי הקוד בפרויקט

### קבצים חדשים (v8)

#### [`src/rules/baseline.py`](../src/rules/baseline.py)
**מה עושה:** מחשב את ה-FHR Baseline לפי תקן ישראלי (ACOG).

**אלגוריתם:**
1. חלוני sliding window של 2 דקות (50% overlap)
2. חלון "יציב" = variability (max−min) < 25 bpm ועם ≥80% ערכים תקינים
3. Baseline = ממוצע כל החלונות היציבים, מעוגל ל-5 bpm הקרובים
4. Fallback: חציון גלובלי אם אין חלונות יציבים

```python
from src.rules.baseline import calculate_baseline, BaselineResult

result: BaselineResult = calculate_baseline(fhr_bpm_array, fs=4.0)
# result.baseline_bpm    → e.g. 135.0
# result.is_tachycardia  → 0.0 or 1.0
# result.is_bradycardia  → 0.0 or 1.0
```

---

#### [`src/rules/variability.py`](../src/rules/variability.py)
**מה עושה:** מחשב את ה-FHR variability (תנודתיות) ומסווג לאחת מ-4 קטגוריות.

**אלגוריתם:**
1. חלוני 1 דקה (50% overlap)
2. Amplitude = max − min לכל חלון
3. ממוצע amplitudes = variability estimate

**סיווג:**

| ערך | קטגוריה (מספר) | משמעות |
|-----|----------------|---------|
| ≤ 2 bpm | 0 — Absent | חמור — עלול להצביע על חמצת |
| 3–5 bpm | 1 — Minimal | דאגה — המשך מעקב |
| 6–25 bpm | 2 — Moderate | **תקין** — מצב עוברי תקין |
| > 25 bpm | 3 — Marked | מוגבר — יכול להצביע על stress |

```python
from src.rules.variability import calculate_variability, VariabilityResult

result: VariabilityResult = calculate_variability(fhr_bpm_array, fs=4.0)
# result.amplitude_bpm  → e.g. 22.7
# result.category       → e.g. 2.0 (moderate)
```

---

#### [`src/rules/decelerations.py`](../src/rules/decelerations.py)
**מה עושה:** מזהה ומסווג ירידות בדופק (decelerations) לפי סוגן.

**הגדרות (מסמך עמדה ישראלי):**
- **Deceleration:** ירידה ≥ 15 bpm מה-baseline למשך ≥ 15 שניות
- **Early:** ירידה הדרגתית (descent ≥ 30 שניות), הנדיר ≤ 15 שניות אחרי שיא הציר — **תקין**
- **Late:** ירידה הדרגתית (descent ≥ 30 שניות), הנדיר > 15 שניות אחרי שיא הציר — **פתולוגי**
- **Variable:** ירידה פתאומית (descent < 30 שניות) — עצב ב-cord
- **Prolonged:** ≥ 2 דקות — **חמור**

**Early decelerations לא נספרות** (הן תקינות) — רק late, variable ו-prolonged.

**אלגוריתם:**
1. Baseline rolling median (חלון 2 דקות)
2. זיהוי dip events: FHR < baseline − 15 bpm למשך ≥ 15 שניות
3. לכל event: מדידת descent time → סיווג early/late/variable
4. Late vs Early: תזמון הנדיר ביחס לשיא הציר (UC signal)
5. Prolonged: משך ≥ 120 שניות

```python
from src.rules.decelerations import detect_decelerations, DecelerationSummary

result: DecelerationSummary = detect_decelerations(fhr_bpm, uc_mmhg, fs=4.0)
# result.n_late_decelerations       → e.g. 2.0
# result.n_variable_decelerations   → e.g. 5.0
# result.n_prolonged_decelerations  → e.g. 0.0
# result.max_deceleration_depth_bpm → e.g. 35.0
```

---

#### [`src/rules/sinusoidal.py`](../src/rules/sinusoidal.py)
**מה עושה:** מזהה דפוס סינוסואידלי — תנודות חלקות ומחזוריות ב-FHR.

**הגדרה:** גלים חלקים בתדר **3–5 מחזורים/דקה (0.05–0.083 Hz)**, אמפליטודה **5–15 bpm**, למשך **> 20 דקות**.

**משמעות קלינית:** ממצא פתולוגי קשה — מצביע על **אנמיה עוברית חמורה** (Rh isoimmunization, vasa previa). קטגוריה 3 מיידית.

**אלגוריתם:**
1. FFT על הסיגנל המלא
2. מציאת תדר דומיננטי בתחום 0.05–0.083 Hz
3. בדיקת: dominance ratio > 0.15, amplitude 5–15 bpm
4. בדיקת duration: ≥50% מחלוני 20 דקות מקיימים את התנאים

```python
from src.rules.sinusoidal import detect_sinusoidal_pattern, SinusoidalResult

result: SinusoidalResult = detect_sinusoidal_pattern(fhr_bpm, fs=4.0)
# result.sinusoidal_detected → 0.0 or 1.0
```

---

#### [`src/rules/tachysystole.py`](../src/rules/tachysystole.py)
**מה עושה:** מזהה tachysystole — פעילות צירים מוגזמת.

**הגדרה:** > 5 צירים ל-10 דקות, בממוצע על חלון של 30 דקות.

**משמעות:** הצירים התכופים מונעים מהעובר להתאושש בין הצירים → חמצת. קשור ל-late decelerations חוזרות.

**אלגוריתם:**
1. `scipy.signal.find_peaks` על UC signal (30 דקות אחרונות)
2. מינימום מרחק: 60 שניות בין שיאים
3. Prominence threshold: אחוזון 75 של ה-UC
4. ספירה: > 5 צירים/10 דקות → tachysystole

```python
from src.rules.tachysystole import detect_tachysystole, TachysystoleResult

result: TachysystoleResult = detect_tachysystole(uc_mmhg, fs=4.0)
# result.tachysystole_detected  → 0.0 or 1.0
# result.contractions_per_10min → e.g. 6.2
```

---

#### [`src/features/clinical_extractor.py`](../src/features/clinical_extractor.py)
**מה עושה:** מאגד את כל 5 המודולים ומחזיר רשימה של 11 features ממוינים.

**נקודות חשובות:**
- מקבל **סיגנל מנורמל** `(2, T)` — בדיוק כמו שנשמר ב-.npy
- מבצע denormalization פנימי לפני קריאת המודולים
- כל מודול עטוף ב-`try/except` → fallback לערכי ברירת מחדל בטוחים

```python
from src.features.clinical_extractor import (
    extract_clinical_features,
    CLINICAL_FEATURE_NAMES,   # רשימת שמות ל-11 features
    N_CLINICAL_FEATURES,      # = 11
)

signal = np.load("data/processed/ctu_uhb/1001.npy")  # (2, T) normalized
feats: List[float] = extract_clinical_features(signal, fs=4.0)
# len(feats) == 11, בסדר של CLINICAL_FEATURE_NAMES
```

**ערכי ברירת מחדל בטוחים** (כאשר מודול נכשל):

| Feature | ברירת מחדל | הסבר |
|---------|-----------|-------|
| `baseline_bpm` | 130.0 | אמצע הטווח התקין |
| `is_tachycardia` | 0.0 | הנחה שאין |
| `is_bradycardia` | 0.0 | הנחה שאין |
| `variability_amplitude_bpm` | 15.0 | אמצע Moderate |
| `variability_category` | 2.0 | Moderate (תקין) |
| `n_late_decelerations` | 0.0 | הנחה שאין |
| `n_variable_decelerations` | 0.0 | הנחה שאין |
| `n_prolonged_decelerations` | 0.0 | הנחה שאין |
| `max_deceleration_depth_bpm` | 0.0 | הנחה שאין |
| `sinusoidal_detected` | 0.0 | הנחה שאין |
| `tachysystole_detected` | 0.0 | הנחה שאין |

ברירות המחדל הן **"תקין"** — כך שאם מודול קורס, הוא לא מוסיף רעש שלילי לסיגנל.

---

### קבצים שהשתנו (v8)

#### [`scripts/run_e2e_cv_v2.py`](../scripts/run_e2e_cv_v2.py)

**שינוי 1 — import:**
```python
# הוסף בראש הקובץ
from src.features.clinical_extractor import (
    extract_clinical_features,
    N_CLINICAL_FEATURES,
    CLINICAL_FEATURE_NAMES,
)
```

**שינוי 2 — N_FEATURES:**
```python
# לפני (v7):
N_FEATURES = 12

# אחרי (v8):
N_FEATURES = 12 + N_CLINICAL_FEATURES   # = 23
```

**שינוי 3 — extract_features_for_split() שורות 249–257:**
```python
# לפני (v7):
feats = extract_recording_features(
    scores_list, threshold=alert_threshold, inference_stride=inference_stride, n_features=n_features
)
X_rows.append(list(feats.values()))

# אחרי (v8):
patchtst_feats = extract_recording_features(
    scores_list, threshold=alert_threshold, inference_stride=inference_stride, n_features=12
)
clinical_feats = extract_clinical_features(signal, fs=4.0)  # signal כבר טעון שורה 233
combined = list(patchtst_feats.values()) + clinical_feats   # 12 + 11 = 23
X_rows.append(combined)
```

---

#### [`azure_ml/train_azure.py`](../azure_ml/train_azure.py)

**שינוי 1 — N_FEATURES (שורה 128):**
```python
N_FEATURES = 23   # 12 PatchTST + 11 clinical rule features (v8)
```

**שינוי 2 — NFEAT_GRID (שורה 723):**
```python
# לפני:
NFEAT_GRID = [4, 12]

# אחרי:
NFEAT_GRID = [4, 12, 23]   # 23 = all features
```

**שינוי 3 — גרסה (שורות 133–135 + 581, 608):**
```python
# e2e_cv_v7 → e2e_cv_v8 בכל המיקומים
OUT_RESULTS = Path(...  "e2e_cv_v8" ...)
OUT_CKPTS   = Path(...  "e2e_cv_v8" ...)
OUT_LOGS    = Path(...  "e2e_cv_v8" ...)
```

---

## 7. איך זה משתלב בצינור האימון הקיים

### מה **לא** השתנה

- `src/train/pretrain.py` — הפרה-טריינינג זהה לחלוטין
- `src/train/finetune.py` — הפיין-טיונינג זהה לחלוטין (כולל תיקוני v6/v7)
- `src/model/patchtst.py` — המודל עצמו לא השתנה
- `azure_ml/conda_env.yml` — אין תלויות חדשות (scipy כבר קיים)
- `azure_ml/setup_and_submit.py` — הגשה לא השתנתה
- `select_lr_c_inner_cv()` — inner-CV לבחירת C נשאר (מ-v7)
- `at_sweep()` — AT selection עובד אוטומטית עם N_FEATURES=23

### מה השתנה — נקודת השינוי היחידה

**כל השינוי בפועל הוא בשלב 4 בלבד** — Feature Extraction.
האימון, ה-pretrain, ה-finetune, ה-LR — כולם זהים.
רק **ה-X matrix** שנכנס ל-LR גדל מ-(N,12) ל-(N,23).

זה מבטיח:
- **אין סיכון regression** ב-pretrain/finetune
- **LR עם L2 regularization** יאפס features לא-אינפורמטיביים אוטומטית
- **backward compatibility:** `n_feat=12` ב-grid search = בדיוק v7 (ablation)

---

## 8. חוקים קליניים — מה כל מודול מזהה

### מה CTG מספר לנו

CTG (Cardiotocography) מצייר שני ערוצים:
1. **FHR (Fetal Heart Rate)** — דופק העובר ב-bpm
2. **UC (Uterine Contractions)** — עוצמת הצירים ב-mmHg

רופא שמסתכל על CTG מחפש דפוסים ספציפיים שמוגדרים פורמלית במסמך עמדה ישראלי (יולי 2023, מבוסס ACOG/FIGO):

### סיווג 3 קטגוריות (התקן הישראלי)

| קטגוריה | הגדרה | ניהול |
|---------|-------|-------|
| **1 — תקין** | Baseline 110–160, variability 6–25, ללא decelerations מאוחרות | מעקב שגרתי |
| **2 — ביניים** | לא עונה על 1 ולא על 3 | הערכה קלינית + התערבות |
| **3 — פתולוגי** | דפוס סינוסואידלי **או** absent variability + חזרתי: late/variable decels/bradycardia | לידה מיידית |

### הדפוסים שאנחנו מזהים

```
┌─────────────────────┬─────────────┬─────────────────────────────┐
│ דפוס                 │ Feature     │ Feature ב-X matrix         │
├─────────────────────┼─────────────┼─────────────────────────────┤
│ Baseline < 110      │ is_bradycardia=1  │ col 14               │
│ Baseline > 160      │ is_tachycardia=1  │ col 13               │
│ Variability < 5     │ variability_category≤1 │ col 16          │
│ Late decelerations  │ n_late_decelerations>0 │ col 17          │
│ Variable decels     │ n_variable_decelerations │ col 18        │
│ Prolonged decel     │ n_prolonged_decelerations>0 │ col 19    │
│ עומק deceleration   │ max_deceleration_depth_bpm │ col 20     │
│ Sinusoidal pattern  │ sinusoidal_detected=1 │ col 21           │
│ Tachysystole        │ tachysystole_detected=1 │ col 22         │
└─────────────────────┴─────────────┴─────────────────────────────┘
```

---

## 9. מה ה-Grid Search בודק עכשיו

**קובץ:** [`azure_ml/train_azure.py`](../azure_ml/train_azure.py) — שורה 762

```
NFEAT_GRID = [4, 12, 23]
```

### מה כל ערך בודק

| n_feat | עמודות X | משמעות |
|--------|---------|---------|
| **4** | cols 0–3 | segment_length, max_pred, cum_sum, weighted_integral — ה-4 features הבסיסיים |
| **12** | cols 0–11 | כל 12 ה-PatchTST features — בדיוק v7 (ablation) |
| **23** | cols 0–22 | כל 23 features כולל clinical — השיטה המלאה של v8 |

**מה נוכל ללמוד מה-grid:**
- אם `n_feat=23` מנצח → ה-clinical features עוזרים
- אם `n_feat=12` מנצח → ה-PatchTST כבר מכיל את כל המידע שצריך
- אם `n_feat=4` מנצח → אנחנו overfitting עם יותר מדי features

ה-grid גם בודק `AT × LR_C × threshold_method` → 5 × 3 × 3 × 2 = 90 קומבינציות לכל fold.
תוצאות נשמרות ב-`results/e2e_cv_v8/grid_best_configs.csv`.

---

## 10. מה ציפינו לקבל — תחזיות ביצועים

| גרסה | OOF AUC | מה השתנה |
|------|---------|---------|
| v3 (baseline) | 0.6013 | ריצה ראשונה |
| v4 | 0.6385 | class weights, patience=20 |
| v5 | 0.5870 ↓ | LR warmup, C salah hardcoded |
| v6 | 0.6329 | patience_ctr reset |
| v7 | ??? (Running) | best_smooth_auc reset, inner-CV C |
| **v8** | **0.66–0.70?** | + 11 clinical features |

### למה ציפינו לשיפור

1. **Features אורתוגונליים:** ה-11 features קליניים לא מתואמים עם ה-12 הקיימים. ה-LR מקבל מידע חדש שלא היה לו.

2. **בסיס מחקרי:** מחקרים מ-2024–2025 ([Frontiers 2025](https://doi.org/10.3389/fdgth.2025.1638424), [Nature npj 2024](https://doi.org/10.1038/s44294-024-00033-z)) מראים שגישות hybrid עקביות עולות על pure DL ב-CTG.

3. **ה-L2 מגן מפני overfitting:** אם features קליניים לא עוזרים — ה-regularization יאפס אותם. אין downside.

4. **הגדרה פורמלית:** Late decelerations מוגדרות בצורה מדויקת ומדידה. PatchTST עשוי ללמוד לגלות "משהו חריג" אבל לא בהכרח "nadir מאוחר ביחס לשיא הציר".

### סיכון עיקרי

PatchTST גדול מספיק (3 שכבות, 128 dim) עשוי כבר להכיל **implicitly** את כל המידע הקליני — ואז ה-11 features לא יוסיפו. זה יתגלה אם `n_feat=12` ינצח ב-grid.

---

## 11. מקור הקוד — SentinelFetal

המודולים הקליניים פורטו מהפרויקט הקודם [SentinelFetal](https://github.com/ArielShamay/SentinelFetal):

- **גרסת המקור:** Phase 13 (עם הרפיית thresholds — depth 12 bpm, duration 12 שניות)
- **גרסת v8:** חזרנו לתקן הרשמי (15 bpm, 15 שניות) לדיוק קליני

**מה הוסר** בפורט:
- תלויות `api.config`, `case_context`, audit trail
- logging framework ייעודי
- strict mode / runtime configuration
- Docker / FastAPI integrations

**מה נשמר:**
- כל האלגוריתמיקה: sliding windows, FFT, peak detection, descent timing
- dataclasses עם safe defaults
- error handling לכל מודול

---

## בדיקת Smoke Test

לאחר הגשת v8 ב-Azure ML, ניתן לאמת locally:

```bash
cd SentinelFatal2

python -c "
import numpy as np
from pathlib import Path
from src.features.clinical_extractor import extract_clinical_features, CLINICAL_FEATURE_NAMES

npy = next(Path('data/processed/ctu_uhb').glob('*.npy'))
sig = np.load(npy)
feats = extract_clinical_features(sig)

for k, v in zip(CLINICAL_FEATURE_NAMES, feats):
    print(f'{k:40s}: {v:.3f}')
print(f'Total features: {len(feats)}')
"
```

**תוצאה צפויה על הקלטה 1001:**
```
baseline_bpm                            : 135.000
is_tachycardia                          : 0.000
is_bradycardia                          : 0.000
variability_amplitude_bpm               : 22.660
variability_category                    : 2.000
n_late_decelerations                    : 1.000
n_variable_decelerations                : 3.000
n_prolonged_decelerations               : 0.000
max_deceleration_depth_bpm              : 55.447
sinusoidal_detected                     : 0.000
tachysystole_detected                   : 0.000
Total features: 11
```

---

*מסמך זה תואר על ידי Claude Sonnet 4.6 עבור Training Run v8 — SentinelFatal2*
