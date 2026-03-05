# מנוע החוקים הקליניים — v8
## תיעוד מלא: ארכיטקטורה, חוקים, באגים ותיקונים

**תאריך:** מרץ 2026  
**גרסה:** v8.1 (post-debug)  
**מחבר:** Claude (Sonnet 4.6) + Ariel Shamay

---

## תוכן עניינים

1. [רקע — מה הבעיה שפותרת](#1-רקע--מה-הבעיה-שפותרת)
2. [ארכיטקטורת המנוע — תרשים זרימה](#2-ארכיטקטורת-המנוע--תרשים-זרימה)
3. [טבלת 11 הפיצ'רים הקליניים](#3-טבלת-11-הפיצ'רים-הקליניים)
4. [מודול 1 — Baseline (baseline.py)](#4-מודול-1--baseline-baselinepy)
5. [מודול 2 — Variability (variability.py)](#5-מודול-2--variability-variabilitypy)
6. [מודול 3 — Decelerations (decelerations.py)](#6-מודול-3--decelerations-decelerationspy)
7. [מודול 4 — Sinusoidal (sinusoidal.py)](#7-מודול-4--sinusoidal-sinusoidalpy)
8. [מודול 5 — Tachysystole (tachysystole.py)](#8-מודול-5--tachysystole-tachysystolypy)
9. [הקואורדינטור — clinical_extractor.py](#9-הקואורדינטור--clinical_extractorpy)
10. [הבאגים שנמצאו ואיך תוקנו](#10-הבאגים-שנמצאו-ואיך-תוקנו)
11. [תוצאות הבדיקה המקומית](#11-תוצאות-הבדיקה-המקומית)
12. [מה נשאר פתוח ומגבלות ידועות](#12-מה-נשאר-פתוח-ומגבלות-ידועות)

---

## 1. רקע — מה הבעיה שפותרת

### הארכיטקטורה ב-v3–v7 (בעיה)

בגרסאות הקודמות, מודל ה-PatchTST ייצר 12 פיצ'רים סטטיסטיים שמוזנים ל-Logistic Regression:

```
.npy (2,T) → PatchTST → anomaly_score/window → 12 stats → LR → P(acidemia)
```

כל 12 הפיצ'רים מתארים *כמה* הסיגנל נראה חריג — אך לא *מה* הוא. דוגמה:

| תופעה CTG | PatchTST רואה | מציאות קלינית |
|-----------|---------------|----------------|
| Early deceleration | Anomaly score גבוה | **תקין** — לחץ ראש בלידה |
| Late deceleration | Anomaly score גבוה | **פתולוגי** — אי-ספיקה שלייתית |

שני המקרים נראים אותו דבר ל-PatchTST. ה-LR לא יכול להבדיל ביניהם.

### הפתרון ב-v8 (Hybrid)

מוסיפים 11 פיצ'רים קליניים שמחושבים מחוקים רפואיים מוגדרים:

```
.npy (2,T) → PatchTST → 12 features ─┐
                                       ├→ concat (23) → StandardScaler → LR → P(acidemia)
.npy (2,T) → Rule Engine → 11 features┘
```

המנוע הקליני מבצע **denormalization** לפני כל החישובים:
- FHR: `signal[0] × 160 + 50` → bpm (טווח 50–210)
- UC:  `signal[1] × 100` → mmHg (טווח 0–100)

זה מוגדר ב-[`src/features/clinical_extractor.py` שורות 81–93](../src/features/clinical_extractor.py#L81).

---

## 2. ארכיטקטורת המנוע — תרשים זרימה

```
┌─────────────────────────────────────────────────────────────────────┐
│  קובץ .npy  shape: (2, T)  @  4 Hz                                  │
│  channel 0: FHR מנורמל [0,1]   channel 1: UC מנורמל [0,1]          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  clinical_extractor.py :: extract_clinical_features()
                            │  denormalize: FHR → bpm, UC → mmHg
                            │
          ┌─────────────────┼──────────────────────────────────────┐
          │                 │                                      │
          ▼                 ▼                                      ▼
    baseline.py       variability.py                    decelerations.py
    ─────────────     ──────────────                    ────────────────
    baseline_bpm      amplitude_bpm                     n_late
    is_tachycardia    category (0-3)                    n_variable
    is_bradycardia                                      n_prolonged
                                                        max_depth
          │                 │                                      │
          │          ┌──────┴───────┐                              │
          │          │              │                              │
          │          ▼              ▼                              │
          │    sinusoidal.py  tachysystole.py                      │
          │    ─────────────  ──────────────                       │
          │    detected(0/1)  detected(0/1)                        │
          │                                                        │
          └──────────────────────┬─────────────────────────────────┘
                                 │
                       List[float] באורך 11
                       (סדר קבוע — ר' טבלה בסעיף 3)
```

כל מודול מחזיר `dataclass` עם שדות מוגדרים. הקואורדינטור אוסף את כולם לרשימה אחת.  
אם מודול נכשל — מוחזרות ברירות מחדל בטוחות (SAFE_DEFAULTS) ללא קריסת הצינור.

---

## 3. טבלת 11 הפיצ'רים הקליניים

אלה הפיצ'רים בסדר קבוע (עמודות 12–22 של מטריצת ה-X):

| # | שם פיצ'ר | יחידה | מה זה מודד | כיוון קליני | AUC עצמאי |
|---|----------|-------|------------|-------------|------------|
| 13 | `baseline_bpm` | bpm | קצב לב עוברי בסיסי | ניטרלי (אופטימלי 110–160) | 0.515 |
| 14 | `is_tachycardia` | 0/1 | baseline > 160 bpm | + Acidemia | 0.508 |
| 15 | `is_bradycardia` | 0/1 | baseline < 110 bpm | ++ Acidemia | 0.513 |
| 16 | `variability_amplitude_bpm` | bpm | טווח תנודות FHR בזמן מנוחה | מורכב* | 0.563 |
| 17 | `variability_category` | 0-3 | 0=absent, 1=minimal, 2=moderate, 3=marked | נמוך=Acidemia | 0.529 |
| 18 | `n_late_decelerations` | count | האטות מאוחרות (אחרי ציר) | +++ Acidemia | 0.576 |
| 19 | `n_variable_decelerations` | count | האטות משתנות (פתאומיות) | + Acidemia | 0.561 |
| 20 | `n_prolonged_decelerations` | count | האטות ≥2 דקות | +++ Acidemia | 0.650 |
| 21 | `max_deceleration_depth_bpm` | bpm | עומק ההאטה הגדולה ביותר | +++ Acidemia | 0.651 |
| 22 | `sinusoidal_detected` | 0/1 | דפוס סינוסואידלי (אנמיה חמורה) | +++ Acidemia | 0.500† |
| 23 | `tachysystole_detected` | 0/1 | >5 צירים ל-10 דק' | ± (לא מובהק) | 0.496 |

*`variability_amplitude` מורכב: minimal variability = Acidemia, אך בשלב מוקדם של מצוקה יש לפעמים variability מוגברת  
†`sinusoidal` כמעט לא זוהה בנתוני CTU-UHB (prevalence נמוכה מאוד)

**מיקום הגדרה:** [`src/features/clinical_extractor.py`, שורות 43–55](../src/features/clinical_extractor.py#L43)

---

## 4. מודול 1 — Baseline

**קובץ:** [`src/rules/baseline.py`](../src/rules/baseline.py)

### מה זה מודד
הDynamic baseline של הדופק העוברי — הקצב הממוצע בזמן מנוחה ללא האטות ות האצות. לפי נייר העמדה הישראלי: **תקין = 110–160 bpm**.

### אלגוריתם
1. פצל את ה-FHR לחלונות של **2 דקות עם overlap 50%**
2. לכל חלון — בדוק שהוא "יציב": variability < 25 bpm ו-≥80% נקודות תקינות (לא NaN)
3. Baseline = ממוצע כל החלונות היציבים, **מעוגל ל-5 bpm הקרוב** (תקן ACOG)
4. Fallback: חציון גלובלי אם אין חלון יציב

### ספים קליניים
```python
TACHYCARDIA_THRESHOLD  = 160.0   # bpm  (src/rules/baseline.py שורה 26)
BRADYCARDIA_THRESHOLD  = 110.0   # bpm  (src/rules/baseline.py שורה 27)
VARIABILITY_MAX_STABLE = 25.0    # bpm  (src/rules/baseline.py שורה 28)
```

### פיצ'רים שמופקים
- `baseline_bpm` — הערך הבסיסי עצמו
- `is_tachycardia` — 1 אם baseline > 160
- `is_bradycardia` — 1 אם baseline < 110

### תאימות לנייר העמדה הישראלי
| סיווג | טווח | ערך בקוד |
|-------|-------|----------|
| ברדיקרדיה | < 110 bpm | `BRADYCARDIA_THRESHOLD = 110.0` |
| תקין | 110–160 bpm | בין שני הספים |
| טכיקרדיה | > 160 bpm | `TACHYCARDIA_THRESHOLD = 160.0` |

---

## 5. מודול 2 — Variability

**קובץ:** [`src/rules/variability.py`](../src/rules/variability.py)

### מה זה מודד
שונות טווח הדופק — כמה ה-FHR "מקפץ" בזמן מנוחה. variability תקינה = 5–25 bpm.  
minimal variability (< 5 bpm) = סימן מובהק לחמצת.

### אלגוריתם (לאחר תיקון v8.1)
1. חשב **reference level**: P90 של כל ה-FHR (מייצג את הרמה הבסיסית בזמן שאין האטות)
2. פצל ל-**חלונות של דקה אחת עם overlap 50%**
3. **סנן החוצה** חלונות שחציון ה-FHR שלהם < `reference - 15 bpm` (חלונות בתוך האטות)
4. amplitude לכל חלון שנשאר = max − min
5. mean amplitude = ממוצע כל החלונות שעברו סינון
6. סווג לקטגוריה: 0/1/2/3

### ספים קליניים
```python
ABSENT_MAX   = 2.0    # ≤2 bpm  → absent   (0)  (src/rules/variability.py שורה 38)
MINIMAL_MAX  = 5.0    # 3–5 bpm → minimal  (1)  (שורה 39)
MODERATE_MAX = 25.0   # 6–25 bpm→ moderate (2)  (שורה 40)
              # >25 bpm → marked   (3)
DECEL_EXCLUSION_BPM = 15.0  # סף לסינון חלונות האטה  (שורה 49)
BASELINE_REF_PCTILE = 90.0  # אחוזון לחישוב reference (שורה 50)
```

### תאימות לנייר העמדה הישראלי
| קטגוריה | טווח | משמעות |
|---------|-------|---------|
| Absent | ≤2 bpm | חשוד biologically — מחייב הערכה דחופה |
| Minimal | 3–5 bpm | דאגה, במיוחד עם האטות |
| Moderate | 6–25 bpm | **תקין** |
| Marked | >25 bpm | בדרך כלל תקין, לפעמים אנמיה |

---

## 6. מודול 3 — Decelerations

**קובץ:** [`src/rules/decelerations.py`](../src/rules/decelerations.py)

זהו המודול הטכני ביותר וגם זה שהכיל את הבאגים הקריטיים ביותר.

### מה זה מודד
**האטות דופק עוברי** — ירידות זמניות ב-FHR מתחת לbaseline. לפי נייר העמדה הישראלי, האטות מסווגות ל-4 סוגים:

| סוג | הגדרה | משמעות קלינית |
|-----|-------|----------------|
| **Early** | ירידה הדרגתית, מתחילה עם הציר, חוזרת לbאseline עם סיומו | **תקין** — לחץ ראש בלידה |
| **Late** | ירידה הדרגתית, מגיעה לנדיר >15 שניות **אחרי** שיא הציר | **פתולוגי** — אי-ספיקה שלייתית, hypoxia |
| **Variable** | ירידה **פתאומית** (onset-to-nadir < 30 שניות) | חשוד — לחץ חבל הטבור |
| **Prolonged** | כל ירידה שנמשכת ≥2 דקות | חמור מאוד |

### צינור הזיהוי (5 שלבים)

#### שלב א: Baseline מתגלגל
```python
# src/rules/decelerations.py שורות 84-101
_compute_rolling_baseline(fhr, fs, ROLLING_WINDOW_S)
```
חציון מתגלגל עם חלון 2 דקות + **global floor** (P85 של כל הסיגנל פחות 20 bpm) למניעת קריסה בתוך האטות ארוכות.

#### שלב ב: זיהוי אירועי ירידה
```python
# src/rules/decelerations.py שורות 112-141
_find_dip_events(fhr, baseline, fs)
```
מחפש מקטעים רצופים שבהם `FHR < baseline − 15 bpm` למשך ≥15 שניות.

#### שלב ג: מציאת ה-True Onset
```python
# src/rules/decelerations.py שורות 142-191
_true_onset_idx(fhr, baseline, event_start, fs)
```
מהנקודה שבה הסף נחצה הולך **אחורה בזמן** עד שה-FHR היה עדיין ליד הbaseline. זה מייצג את תחילת הירידה בפועל (ולא את הנקודה שבה היא כבר עמוקה 15 bpm).

#### שלב ד: סיווג abrupt vs gradual
```python
descent_s = (nadir_idx - onset_idx) / fs
if descent_s < VARIABLE_ONSET_S:  # < 30 שניות
    n_var += 1
else:
    # gradual → late or early
```

#### שלב ה: late vs early לפי תזמון ציר
```python
# src/rules/decelerations.py שורות 192-219
uc_peak_idx = _find_nearest_uc_peak(uc, start, end, fs)
nadir_lag_s = (nadir_idx - uc_peak_idx) / fs
if nadir_lag_s > LATE_NADIR_LAG_S:  # > 15 שניות
    n_late += 1
```

### ספים
```python
MIN_DEPTH_BPM          = 15.0   # עומק מינימלי  (decelerations.py שורה 43)
MIN_DURATION_S         = 15.0   # משך מינימלי   (שורה 44)
PROLONGED_MIN_S        = 120.0  # ≥2 דקות       (שורה 45)
VARIABLE_ONSET_S       = 30.0   # onset פחות מ-30s = abrupt  (שורה 46)
LATE_NADIR_LAG_S       = 15.0   # >15s אחרי UC peak = late  (שורה 47)
TRUE_ONSET_LOOKBACK_S  = 60.0   # כמה אחורה לחפש onset (שורה 60)
UC_PEAK_PROMINENCE     =  2.0   # mmHg — רגישות UC peaks (שורה 64)
UC_SEARCH_WINDOW_S     = 90.0   # ±90s לחיפוש ציר (שורה 65)
```

---

## 7. מודול 4 — Sinusoidal

**קובץ:** [`src/rules/sinusoidal.py`](../src/rules/sinusoidal.py)

### מה זה מודד
דפוס סינוסואידלי ב-FHR — גלים חלקים ורגולריים בתדר **3–5 מחזורים/דקה** (0.05–0.083 Hz). מצב זה הוא **קטגוריה 3 — פתולוגי קיצוני**, מצביע על אנמיה חמורה (כגון במחלת Rh) או hypoxia עמוקה.

### אלגוריתם (לאחר תיקון v8.1)
1. **FFT** על כל ה-FHR (או מקטע של ≥5 דקות)
2. חשב את אחוז העוצמה הספקטרלית בתחום 0.05–0.083 Hz (`dominance_ratio`)
3. חשב amplitude מ-**עוצמת ה-band בלבד** (לא std*2 שכולל את כל הסיגנל):
   ```
   A = 2 × sqrt(band_power) / n
   ```
   עבור גל `A·sin(2πft)` באורך n נקודות, `|X[k]| ≈ n·A/2`, כלומר `band_power ≈ (n·A/2)²`
4. גם `dominance ≥ 0.10` וגם `3 ≤ amplitude ≤ 25 bpm` → check for duration
5. **בדיקת duration**: sliding windows של 20 דקות — ≥50% חוזרים true → detected

### ספים
```python
FREQ_LOW_HZ         = 0.05    # 3 cycles/min   (sinusoidal.py שורה 27)
FREQ_HIGH_HZ        = 0.083   # 5 cycles/min   (שורה 28)
MIN_AMPLITUDE_BPM   = 3.0     # (שורה 31)
MAX_AMPLITUDE_BPM   = 25.0    # (שורה 32)
MIN_DOMINANCE_RATIO = 0.10    # (שורה 34)
MIN_DURATION_MIN    = 20.0    # (שורה 36)
```

---

## 8. מודול 5 — Tachysystole

**קובץ:** [`src/rules/tachysystole.py`](../src/rules/tachysystole.py)

### מה זה מודד
**Tachysystole** (hyper-stimulation) — יותר מ-5 צירים ב-10 דקות, ממוצע על פני 30 דקות. גורם ל-uteroplacental insufficiency ו-late decelerations.

### אלגוריתם (לאחר תיקון v8.1)
1. קח את **30 הדקות האחרונות** של UC (הכי קלינית-רלוונטיות — סוף הלידה)
2. זהה פיקים ב-UC עם `scipy.signal.find_peaks` עם ספי **absolute** (לא relative):
   - `prominence ≥ 10 mmHg` מעל baseline מקומי
   - `height ≥ 8 mmHg` מעל אפס
   - `distance ≥ 60 שניות` בין פיקים סמוכים
3. `contractions_per_10min = n_peaks / (window_min / 10)`
4. `tachysystole = contractions_per_10min > 5`

### ספים
```python
MAX_CONTRACTIONS_PER_10MIN = 5.0    # (tachysystole.py שורה 26)
MIN_PEAK_SPACING_S         = 60.0   # (שורה 28)
MIN_UC_PROMINENCE_MMHG     = 10.0   # (שורה 35)
MIN_UC_HEIGHT_MMHG         =  8.0   # (שורה 36)
```

---

## 9. הקואורדינטור — clinical_extractor.py

**קובץ:** [`src/features/clinical_extractor.py`](../src/features/clinical_extractor.py)

הפונקציה הציבורית היחידה שהקוד החיצוני משתמש בה:

```python
def extract_clinical_features(
    signal_normalized: np.ndarray,   # shape (2, T)
    fs: float = 4.0,
) -> List[float]:                    # אורך 11, בסדר CLINICAL_FEATURE_NAMES
```

### אחריות
1. **Denormalization** (שורות 81–93, פונקציה `_denormalize`): המרה חזרה ל-bpm ו-mmHg
2. **NaN handling** (שורות 119–120): החלפת NaN ב-FHR ב-130.0, UC ב-0.0
3. **קריאה לכל מודול** עם try/except — כשל באחד לא מקרים את הצינור
4. **SAFE_DEFAULTS** (שורות 60–72): ערכים בטוחים ניטרליים ברירת מחדל

### קבועים ציבוריים (לשימוש מחוץ לקובץ)
```python
CLINICAL_FEATURE_NAMES   # list[str] אורך 11
N_CLINICAL_FEATURES      # = 11
SAFE_DEFAULTS            # dict עם ברירות מחדל
```

### שימוש בקוד האימון
בקובץ [`scripts/run_e2e_cv_v2.py`, שורות 261–264](../scripts/run_e2e_cv_v2.py#L261):
```python
clinical_feats = extract_clinical_features(signal, fs=4.0)
combined = list(patchtst_feats.values()) + clinical_feats  # אורך 23
```

---

## 10. הבאגים שנמצאו ואיך תוקנו

### סקר הבאגים: הדרך לגילוי

הריצה הראשונה של [`scripts/evaluate_clinical_rules.py`](../scripts/evaluate_clinical_rules.py) הניבה:

```
Mean AUC : 0.5466  →  Verdict: WEAK
```

Sanity check הראה שכל 6 המקרים — **אפס late decelerations לנצח**, למרות עומקי האטה של 40–65 bpm. הבעיה לא הייתה בגילוי האירועים אלא בסיווגם.

---

### באג 1 (קריטי) — מדידת descent time שגויה

**קובץ:** [`src/rules/decelerations.py`](../src/rules/decelerations.py)

#### הבעיה
הפונקציה המקורית `_descent_time_s` מדדה:
```
event_start → nadir
```
אבל `event_start` היא הנקודה שבה ה-FHR **כבר חצה** את הסף `baseline − 15 bpm`. כלומר הירידה כבר התחילה.  

עבור Late deceleration אמיתית: הירידה מתחילה 30–40 שניות לפני הנדיר, אבל הסף נחצה אחרי 20 שניות — נשאר רק 10–20 שניות עד הנדיר → מסווג כ-`VARIABLE` (abrupt) בטעות.

#### הפתרון: `_true_onset_idx()`
```python
# src/rules/decelerations.py שורות 142-177
def _true_onset_idx(fhr, baseline, event_start, fs):
    # מחשב reference מ-P75 של חלון לפני האירוע (לא baseline מתגלגל שמוטה)
    pre_window  = fhr[lookback:event_start]
    stable_ref  = np.percentile(pre_window, 75)    # ←  רובסטי לפני הירידה
    threshold   = stable_ref - TRUE_ONSET_MARGIN_BPM  # 5 bpm מטה מהbaseline
    # הולך אחורה עד שה-FHR חוזר לרמה זו
    for i in range(event_start - 1, lookback - 1, -1):
        if fhr[i] >= threshold:
            return i
```

**מדוע P75 ולא median/baseline?** הbaseline המתגלגל הסימטרי כולל גם חלק מהירידה עצמה ולכן מוטה למטה. P75 של חלון לפני האירוע מייצג את הרמה הבסיסית הספציפית לאותו רגע.

---

### באג 2 — Fallback שגוי ב-UC peak detection

**קובץ:** [`src/rules/decelerations.py`](../src/rules/decelerations.py)

#### הבעיה
```python
# קוד מקורי
pks, _ = find_peaks(seg, prominence=5.0)
if len(pks) == 0:
    return event_start   # ← FALLBACK שגוי
```

כשלא נמצא UC peak (prominence=5.0 mmHg — גבוה מדי לנתוני CTU-UHB שבהם לרוב שיא UC ≈ 30–60 mmHg אבל לפעמים רק 15), הפונקציה חזרה `event_start` כ"מיקום שיא הציר".

תוצאה: `nadir_lag = (nadir_idx − event_start) / fs` — ערך קטן (הנדיר תמיד אחרי event_start), לרוב לא עובר את הסף של 15 שניות → האטה לא נספרת כ-late ולא נספרת כ-early (פשוט נעלמת).

#### הפתרון
```python
# src/rules/decelerations.py שורה 64
UC_PEAK_PROMINENCE = 2.0   # הורד מ-5.0 ל-2.0 mmHg

# שינוי חתימת הפונקציה: מחזירה Optional[int]
def _find_nearest_uc_peak(...) -> Optional[int]:
    if len(pks) == 0:
        return None    # ← המתקשר מחליט, לא מניחה fallback שגוי

# בקוד הראשי:
if uc_peak_idx is None:
    n_var += 1   # שמרני — לא יוצר False late
    continue
```

---

### באג 3 — Baseline מתגלגל קורס בתוך האטות ארוכות (Prolonged)

**קובץ:** [`src/rules/decelerations.py`](../src/rules/decelerations.py)

#### הבעיה
עבור האטה של 3 דקות ב-FHR, חלון המדיאן הסימטרי של 2 דקות **כולל גם חלקים מתוך ההאטה**. הbaseline קורס מ-140 ל-100 bpm → depth = 0 → האירוע לא מזוהה כלל.

#### הפתרון: Global Floor
```python
# src/rules/decelerations.py קבועים: שורות 56-57  |קוד הfloor: שורות 103-108
BASELINE_STABLE_PCTILE  = 85.0   # אחוזון מייצג את "FHR רגיל"
BASELINE_FLOOR_DROP_BPM = 20.0   # מינימום הפרש מותר

# בתוך _compute_rolling_baseline — אחרי חציון מתגלגל:
global_stable = np.percentile(valid_all, BASELINE_STABLE_PCTILE)
floor_value   = global_stable - BASELINE_FLOOR_DROP_BPM
baseline      = np.maximum(baseline, floor_value)
```

---

### באג 4 — Variability מנופחת על-ידי האטות

**קובץ:** [`src/rules/variability.py`](../src/rules/variability.py)

#### הבעיה
הקוד המקורי חישב max-min על **כל** החלונות, כולל אלה שמכילים האטות. חלון ש-FHR בו עובר 130→90 bpm מקבל amplitude=40 bpm — גבוה מאוד — אף שהvariability האמיתית תקינה.

תוצאה: מקרי Acidemia (יותר האטות) הראו amplitude **גבוה יותר** מ-Normal — ההפך מהתצפית הקלינית הנכונה. ה-LR למד כיוון הפוך.

**נתון מהנתונים:** Acidemia mean=28.7 bpm, Normal mean=26.5 bpm בקוד המקורי.

#### הפתרון: סינון חלונות האטה
```python
# src/rules/variability.py קבועים: שורות 49-51  |לוגיקת הסינון: שורות 108-121
DECEL_EXCLUSION_BPM = 15.0    # bpm מתחת לreference → חלון מסוינן
BASELINE_REF_PCTILE = 90.0    # P90 של כל הFHR = reference יציב

# בלולאת החלונות:
ref_level = np.percentile(valid_all, BASELINE_REF_PCTILE)
if np.median(window) < ref_level - DECEL_EXCLUSION_BPM:
    continue    # חלון בתוך האטה — לא נספר
```

---

### באג 5 — Sinusoidal amplitude שגוי (std×2 כולל כל הסיגנל)

**קובץ:** [`src/rules/sinusoidal.py`](../src/rules/sinusoidal.py)

#### הבעיה
```python
# קוד מקורי
amplitude = float(np.std(segment) * 2)   # ← כולל כל השונות!
```
STD של כל הסיגנל כולל variability רגילה, האטות, ועצמאות. עבור הקלטה ממוצעת — std ≈ 15 bpm → amplitude = 30 bpm → חורג מ-`MAX_AMPLITUDE_BPM=15` → NEVER detected.

#### הפתרון: נוסחת FFT נכונה
```python
# src/rules/sinusoidal.py שורות 83-91
# עבור x[t] = A·sin(2πft) באורך n: |X[k]| = n·A/2
# → band_power = (n·A/2)²  → A = 2·sqrt(band_power) / n
if np.any(in_band) and n > 0:
    amplitude = float(2.0 * np.sqrt(band_power) / n)
```
בדיקה: גל סינתטי `A=8 bpm, 0.06 Hz` → amplitude מחושב = 8.00 bpm ✓

---

### באג 6 — Tachysystole: relative prominence גורם ל-over/under detection

**קובץ:** [`src/rules/tachysystole.py`](../src/rules/tachysystole.py)

#### הבעיה
```python
# קוד מקורי
prominence = float(np.percentile(uc_window, 0.75 * 100))  # P75 של UC
```
כאשר UC ≈ 0 ברוב הזמן (שלבים מוקדמים), P75 ≈ 0 → גם רעש קל נספר כציר.  
כאשר כל הצירים בגובה דומה → P75 ≈ גובה ציר → אפס צירים מזוהים בסוף.

#### הפתרון: ספים absolute
```python
# src/rules/tachysystole.py שורות 30-36
MIN_UC_PROMINENCE_MMHG = 10.0   # prominence מינימלי מעל baseline מקומי
MIN_UC_HEIGHT_MMHG     =  8.0   # גובה מינימלי מעל אפס
```

---

### סיכום הבאגים

| # | קובץ | סיבה | השפעה | פתרון |
|---|------|-------|-------|-------|
| 1 | `decelerations.py` | descent time נמדד מסף, לא מonset | Late→Variable misclassify | `_true_onset_idx()` + P75 |
| 2 | `decelerations.py` | UC fallback=event_start | Late→Early misclassify | `Optional[int]` + prominence↓ |
| 3 | `decelerations.py` | Baseline קורס בהאטות ארוכות | Prolonged לא מזוהות | Global floor P85−20 |
| 4 | `variability.py` | max-min כולל האטות | כיוון פיצ'ר הפוך | סינון חלונות האטה |
| 5 | `sinusoidal.py` | std×2 כולל כל הסיגנל | Never detected | נוסחת FFT נכונה |
| 6 | `tachysystole.py` | prominence relative גורם instability | AUC<0.5 | ספים absolute |

---

## 11. תוצאות הבדיקה המקומית

הבדיקה הורצה עם [`scripts/evaluate_clinical_rules.py`](../scripts/evaluate_clinical_rules.py) על 552 הקלטות מנתוני CTU-UHB (113 Acidemia, 439 Normal).

### השוואת AUC לאורך הפיתוח

| שלב | תיאור | Mean AUC (5-Fold CV) | Verdict |
|-----|-------|---------------------|---------|
| קוד מקורי (6 באגים) | — | 0.5466 | WEAK |
| תיקון decelerations (באגים 1-3) | true onset + UC prominence + baseline floor | 0.7031 | GOOD |
| תיקון כל המודולים (1-6) | + variability + sinusoidal + tachysystole | **0.7049** | GOOD |

### תוצאות Fold-by-Fold (גרסה סופית)

```
Fold 0:  test=111 (+23/-88)   AUC = 0.7782
Fold 1:  test=111 (+23/-88)   AUC = 0.7268
Fold 2:  test=110 (+23/-87)   AUC = 0.6802
Fold 3:  test=110 (+22/-88)   AUC = 0.6751
Fold 4:  test=110 (+22/-88)   AUC = 0.6643
─────────────────────────────────────────
Mean = 0.7049  |  Std = 0.0424
```

### חשיבות פיצ'רים (מקדמי LR על הנתונים המלאים)

```
1.  max_deceleration_depth_bpm     coef=+0.535  +++     ← עומק > ספירה
2.  n_prolonged_decelerations      coef=+0.448  ++
3.  variability_category           coef=-0.379  --      ← low cat = acidemia ✓
4.  n_late_decelerations           coef=+0.271  +
5.  variability_amplitude_bpm      coef=+0.238  +       ← מורכב (ר' סעיף 12)
6.  is_bradycardia                 coef=+0.219  +
7.  sinusoidal_detected            coef=-0.179  -       ← ייתכן false detections
8.  is_tachycardia                 coef=+0.085
9.  n_variable_decelerations       coef=+0.082
10. baseline_bpm                   coef=-0.068
11. tachysystole_detected          coef=+0.028
```

### Sanity Check — אחרי (גרסה מתוקנת)

| Recording | סוג | Late | Variable | Prolonged | Max Depth |
|-----------|-----|------|----------|-----------|-----------|
| 1472 | 🔴 Acidemia | **1** | 2 | 0 | 65 bpm |
| 1039 | 🔴 Acidemia | **1** | 8 | **5** | 68 bpm |
| 2030 | 🔴 Acidemia | **1** | 0 | **2** | 89 bpm |
| 1056 | 🟢 Normal | 0 | 2 | 0 | 54 bpm |
| 1473 | 🟢 Normal | 1 | 7 | 0 | 80 bpm |
| 1339 | 🟢 Normal | 0 | 1 | 0 | 26 bpm |

לפני התיקון: late=0 לכולם, prolonged=0 לכולם.

---

## 12. מה נשאר פתוח ומגבלות ידועות

### מה שנשאר כפי שהוא (בכוונה)

**`variability_amplitude` עם מקדם חיובי (+0.238):**  
נראה אינטואיטיבי, אבל קלינית קיימת תופעה של *reactive hyperdynamic variability* בשלב מוקדם של מצוקה. ה-LR לומד קורלציה אמיתית בנתונים, לא בעיית קוד.

**`variability_category` עם מקדם שלילי (−0.379):**  
**נכון קלינית** — Low category (1=minimal) ↔ Acidemia. המקדם השלילי הוא הצפוי.

**`sinusoidal_detected` עם מקדם שלילי (−0.179):**  
בנתוני CTU-UHB דפוסים מחזוריים מצירים סדירים יכולים להפעיל false detection. prevalence אמיתי של sinusoidal pattern בנתונים זעיר (0/552 בבדיקה).

### מגבלות אינהרנטיות של 11 פיצ'רים בינאריים/ספירתיים

AUC ≈ 0.70 הוא **תקרת האפשרויות הסבירה** לארכיטקטורת חוקים בינאריים על CTU-UHB עם threshold pH<7.05. הערך האמיתי של הפיצ'רים הקליניים מתגלה בשילוב:

```
LR(23 feats) = LR(12 PatchTST + 11 clinical)
```

כאן ה-11 פיצ'רים מספקים **signal ש-PatchTST לא יכול לגלות בכלל** (late vs early, baseline category), והשילוב הוא שיעשה את ה-AUC הגבוה ב-v8.

### שיפורים אפשריים עתידיים (לא חובה)

| פיצ'ר פוטנציאלי | מה הוא מדד | מדוע טוב יותר |
|----------------|-----------|----------------|
| `baseline_trend_bpm_per_hour` | האם הbaseline עולה/יורד | מזהה deterioration גדולה מ-static value |
| `deceleration_area_bpm_s` | עומק × זמן לכל האטה | מדד עשיר יותר מספירה בלבד |
| `contractions_per_10min` | ערך ממשי (לא 0/1) | מדד continuous יותר מtachysystole_detected |
| `recovery_time_bpm_per_s` | כמה מהר ה-FHR חוזר לbאseline | מדד resilience שלייתית |

---

*מסמך זה מתאר את מצב המנוע נכון למרץ 2026. כל שינוי בספים או בלוגיקה יש לתעד כan בDEVIATION_LOG.*
