# תוכנית עבודה: Foundation Model לניבוי מצוקה עוברית מ-CTG

> **גרסה:** 2.0 — מעודכן לנתונים פיזיים בפועל (22 פברואר 2026)
> **מקור מאמר:** arXiv:2601.06149v1 — נמצא ב-`docs/2601.06149v1.pdf`
> **מטרה:** מסמך זה הוא **מקור האמת היחיד (SSOT)** לכלל ההחלטות בפרויקט.

---

## חלק א — מלאי נתונים פיזי (Physical Inventory)

> **מצב נכון ל-22.2.2026 — נסרק ואומת מול הדיסק.**
> כל קובץ שאינו ברשימה זו — **אינו קיים**.

---

### א.1 — CTU-UHB Intrapartum CTG Database (PhysioNet)

**מיקום:** `data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/`

| סוג קובץ | כמות | פורמט | תוכן |
|----------|------|--------|-------|
| `.dat` | 552 | WFDB binary | אותות FHR + UC גולמיים, 4 Hz |
| `.hea` | 552 | WFDB text header | **מקור קליני מלא**: pH, Deliv. type, Presentation, NoProgress, I.stage, II.stage |
| `metadata_summary.csv` | 1 (**100 שורות — חלקי!**) | CSV | pH, delivery_type, presentation, stage2_min (100 מתוך 552 בלבד) |
| `signal_quality_stats.csv` | 1 | CSV | אחוז חסרים, ממוצעים, ארטיפקטים לכל ערוץ |

**מה יש:** 552 הקלטות שלמות, pH לכולן (בקבצי `.hea`), מטה-דאטה קלינית מלאה **בקבצי `.hea` בלבד**.
**פורמט .dat/.hea:** נדרש `wfdb` Python package לקריאת אות; קבצי `.hea` ניתנים לקריאה ישירה כטקסט.

> ⚠ **אזהרה P1:** `metadata_summary.csv` מכיל רק 100 שורות, **לא 552**. **אסור להשתמש בו כמקור לתתי-קבוצות הערכה.** יש לבנות `data/processed/ctu_uhb_clinical_full.csv` מ-552 קבצי `.hea` (ראה שלב 1.6).

---

### א.2 — CTGDL Archives (Zenodo)

**מיקום:** `data/CTGDL/`

| קובץ | גודל | תוכן | סטטוס |
|------|------|-------|-------|
| `CTGDL_ctu_uhb_csv.tar.gz` | 15 MB | CTU-UHB — raw CSVs (fname: `ctgdl_ctu_uhb_XXXX.csv`) | ✓ קיים, לא פרוס |
| `CTGDL_ctu_uhb_proc_csv.tar.gz` | 26 MB | CTU-UHB — **processed** CSVs | ✓ קיים, לא פרוס |
| `CTGDL_FHRMA_ano_csv.tar.gz` | 46 MB | FHRMA — raw/annotated CSVs | ✓ קיים, לא פרוס |
| `CTGDL_FHRMA_proc_csv.tar.gz` | 6.8 MB | FHRMA — **processed** CSVs | ✓ קיים, לא פרוס |
| `CTGDL_norm_metadata.csv` | 66 KB | **מטה-דאטה מרכזי** — 981 שורות | ✓ קיים, קריא |
| `CTGDL_FHEMA_metadata.csv` | 19 KB | מטה-דאטה FHRMA — 135 שורות | ✓ קיים |
| `CTGDL_SPAM_metadata.csv` | 34 KB | מטה-דאטה SPAM — 294 שורות | ✓ קיים (metadata בלבד) |
| `CTGDL - Data Collection.pdf` | 578 KB | תיעוד איסוף נתונים | ✓ קיים |
| `CTGDL_spam_dataset_read_matlab_signal_and_preprocess.ipynb` | 5.9 MB | מחברת עיבוד SPAM מ-.mat | ✓ קיים (חסרים קבצי .mat) |

#### א.2.1 — CTGDL_norm_metadata.csv — THE key file

**פורמט עמודות:** `pid, id, sig_len, sig_min, fname, stage2_idx, stage2min, dataset, target, test`

**תוכן מאומת (ריצת ניתוח Python):**

| dataset | שורות | target=1 (acidemia) | target=0 | test=0 (train) | test=1 (val) | test=2 (test) |
|---------|-------|---------------------|----------|-----------------|--------------|----------------|
| `ctg` (CTU-UHB) | 552 | 113 | 439 | 441 | 56 | 55 |
| `fhrma` (FHRMA) | 135 | 0 | 135 | 85 | 0 | 50 |
| `spam` (SPAM) | 294 | 3 | 291 | 234 | 0 | 60 |
| **סה"כ** | **981** | **116** | **865** | **760** | **56** | **165** |

**ספליטי CTU-UHB — אומתו מול Table 2 במאמר:**

| Split | n | acidemia | % acidemia | אימות vs מאמר |
|-------|---|----------|------------|----------------|
| train (test=0) | 441 | 90 | 20.4% | ✓ |
| val (test=1) | 56 | 12 | 21.4% | ✓ |
| test (test=2) | 55 | 11 | 20.0% | ✓ |

> **מסקנה קריטית:** הספליטים **כבר מוגדרים** ב-`CTGDL_norm_metadata.csv` עמודת `test`. **אסור ליצור ספליטים חדשים.** השתמש בעמודה הקיימת.

---

### א.3 — FHRMA-master

**מיקום:** `data/FHRMA-master/`

> **שימו לב:** תיקייה זו היא כלי MATLAB לניתוח FHR (Feature Selection + Morphological Analysis), **לא** 135 ההקלטות של CTGDL-FHRMA.
> קבצי ה-`.fhr` בתיקייה זו (66 train + 90 test) שייכים ל-FHRMA Feature Selection dataset — **שונה** מ-135 הקלטות ה-CTGDL.
> **לפרה-טריינינג: השתמש ב-CTGDL_FHRMA_proc_csv.tar.gz, לא ב-FHRMA-master.**

| תת-תיקייה | תוכן |
|-----------|-------|
| `FHRMAdataset/traindata/` | 66 קבצי `.fhr` (לFHRMA tool) |
| `FHRMAdataset/testdata/` | 90 קבצי `.fhr` (לFHRMA tool) |
| `FSdataset/` | 1,066 קבצי `.fhrm` לFSdataset |
| `*.m` / `*.mat` | קוד MATLAB ומודלים אימונים |

**שימוש בפרויקט זה:** אפס — אינו חלק מה-pipeline.

---

### א.4 — SPAM — חסר

**מה קיים:** `CTGDL_SPAM_metadata.csv` בלבד (294 שורות, עמודות: id, fs, fname, stage2_idx, sig_len, sig_min, stage2min)
**מה חסר:** קבצי האותות (`ctgdl_spam_001.csv` עד `ctgdl_spam_294.csv`) ו-קבצי `.mat` המקוריים
**סיבה:** Dataset הוסר מגישה ציבורית. יש לפנות: `ctg.challenge2017@gmail.com`

> **הבהרה: 297 vs 294** — המאמר מציין 297 הקלטות SPAM, אך ה-metadata מכיל 294. הסיבה: 3 הקלטות הוצאו במפורש מ-CTGDL לפי script הקיים בנתונים (030, 630, 104 — כולן עם TOCO=0 לאורך כל ההקלטה). לפיכך **294 הוא המספר הנכון** של הקלטות SPAM הניתנות לשימוש.

**SPAM positive cases (target=1):** 3 הקלטות עם `stage2min == 0` = ניתוח קיסרי ללא שלב שני. אלה מסומנות כ-positive לצורך augmentation.

---

### א.5 — סיכום מלאי

| מקור | הקלטות | אותות זמינים | pH/labels | שימוש בפרויקט |
|------|---------|--------------|-----------|----------------|
| CTU-UHB (PhysioNet .dat/.hea) | 552 | ✓ | ✓ | Fine-tuning + Pretraining |
| CTU-UHB (CTGDL processed CSVs) | 552 | ✓ (same) | דרך norm_metadata | Fine-tuning + Pretraining |
| FHRMA (CTGDL processed CSVs) | 135 | ✓ | ✗ | Pretraining בלבד |
| SPAM | 294 | ✗ | ✗ | **חסר — לא ניתן לשימוש** |
| **סה"כ זמין לפרה-טריינינג** | **687** | **✓** | — | — |
| **סה"כ לפי המחקר** | **984** | — | — | — |

**סטייה ממחקר:** 297 הקלטות SPAM חסרות → פרה-טריינינג על **687 במקום 984** (70% מהנדרש).
→ ראה חלק ד, סטייה S1.

---

## חלק ב — Single Source of Truth (SSOT)

### ב.1 — מקורות סמכות לפי נושא

| נושא | מקור | עמודה/סעיף |
|------|-------|------------|
| ארכיטקטורה + פרמטרים | `docs/2601.06149v1.pdf` | Section II-C, Equation 1, Figure 3 |
| ספליטים (train/val/test) | `data/CTGDL/CTGDL_norm_metadata.csv` | עמודת `test` + `dataset='ctg'` |
| תיוג acidemia | `data/CTGDL/CTGDL_norm_metadata.csv` | עמודת `target` (1=acidemia) |
| אימות תיוג acidemia | קבצי `.hea` (552) → `data/processed/ctu_uhb_clinical_full.csv` | שדה `#pH` <= 7.15 |
| רשימת IDs לפרה-טריינינג | `data/CTGDL/CTGDL_norm_metadata.csv` | שורות dataset='ctg' OR 'fhrma', עמודת `fname` |
| שמות קבצי אות | `data/CTGDL/CTGDL_norm_metadata.csv` | עמודת `fname` |
| Preprocessing | `docs/2601.06149v1.pdf` | Section II-B |
| Masking | `docs/2601.06149v1.pdf` | Section II-D, Equation 2, Figure 4 |
| Alerting protocol | `docs/2601.06149v1.pdf` | Section II-F, Figure 5 |
| Evaluation metrics | `docs/2601.06149v1.pdf` | Table 3, Figure 6 |
| Colab workflow | `docs/colab-vscode-guide-hebrew.md` | — |

### ב.2 — כלל עבודה

> **כל החלטה טכנית**, פרמטר, הגדרת נתונים, או שינוי ב-pipeline **חייבת הסמכה** מאחד ממקורות אלה.
> אם אין הסמכה — זוהי **הנחה** שחייבת להיות מתועדת ב-`docs/deviation_log.md` לפני יישום.

---

## חלק ג — תקציר הפרויקט

### ג.1 — קלטים

- **אות:** CTG דו-ערוצי: FHR (קצב לב עוברי, bpm) + UC (התכווצויות רחמיות, mmHg)
- **תדירות דגימה:** 4 Hz
- **חלון עיבוד:** 1,800 דגימות = **7.5 דקות**
- **מספר patches לחלון:** (1800 − 48) / 24 + 1 = **73 patches**
- **Fine-tuning corpus:** 552 הקלטות CTU-UHB
- **Pretraining corpus (בפועל):** 687 הקלטות (CTU-UHB + FHRMA)

### ג.2 — פלטים

1. **ציון סיכון רציף** (0–1) לכל חלון הזזה
2. **זיהוי קטעי-אזעקה** — אזורים שבהם ציון > 0.5 (Stage 1)
3. **החלטה בינארית** (0/1) על בסיס 4 features של קטע-האזעקה → Logistic Regression (Stage 2)

### ג.3 — קריטריוני הצלחה (מהמחקר — Table 3)

| תת-קבוצה | n | AUC שהושג במאמר | Accuracy |
|-----------|---|-----------------|----------|
| כלל Test | 55 | **0.826** | 78.6% |
| לידות וגינליות | 50 | **0.850** | 80.0% |
| מצגת ראש | 50 | **0.848** | 80.0% |
| **וגינלי + ראש (cephalic)** | **46** | **0.853** | **80.4%** |
| ללא עצירת לידה | 47 | **0.837** | 83.0% |
| וגינלי+ראש+ללא עצירה | 43 | **0.837** | 83.7% |
| Benchmark קודם (להכות) | — | 0.68–0.75 | — |

> **אזהרה:** עם פרה-טריינינג על 687 (≈1,706 שעות, לא 984=2,444 שעות), ה-AUC הצפוי עשוי להיות מעט נמוך יותר מהמאמר.

---

## חלק ד — פערים וסטיות מהמחקר

### S1 — SPAM חסר (קריטי)

| | |
|-|-|
| **תיאור** | 294 הקלטות SPAM הניתנות לשימוש (297 פחות 3 שהוצאו) חסרות לפרה-טריינינג ולאוגמנטציה |
| **סיבה** | Dataset הוסר מגישה ציבורית; נדרשת הסכמת שימוש |
| **השפעה** | פרה-טריינינג על 687 הקלטות (≈1,706 שעות) במקום 984 (2,444 שעות = 70% מהנדרש); ללא 3 הקלטות קיסריות כ-augmentation חיובי ב-fine-tuning |
| **טיפול** | פנייה ל-`ctg.challenge2017@gmail.com` (מייל נוסח בשיחה קודמת). pipeline כתוב להרחבה ל-984 ברגע שהנתונים יתקבלו |

### S6 — פרמטרים שחסרים לחלוטין מהמאמר (נדרשת החלטה לפני אימון)

> אלה פרמטרים שהמאמר **אינו מציין בכלל** ואין להם אסמכתא. יש להחליט לפני תחילת כל שלב אימון ולתעד ב-`docs/deviation_log.md`.

| פרמטר | ערך מוצע | נימוק | שלב רלוונטי |
|--------|-----------|-------|-------------|
| **Batch size (pretraining)** | 64 | מקובל ל-MAE על time series קצרים; מאזן GPU mem ו-BN stability | שלב 3 |
| **Batch size (fine-tuning)** | 32 | dataset קטן (441 recordings) → batch קטן יותר | שלב 4 |
| **LR fine-tuning** | 1×10⁻⁴ | זהה ל-pretraining; LR נמוך יותר (1e-5) אם overfitting מהיר | שלב 4 |
| **Weight decay (AdamW)** | 1×10⁻² | ברירת מחדל AdamW; עוזר לרגולריזציה על dataset קטן | שלב 4 |
| **Inference stride — repro_mode** | **1 דגימה** | נדרש להשוואה מדויקת למאמר; הערכה רשמית (שלב 7) | שלב 5,7 |
| **Inference stride — runtime_mode** | 60 דגימות (15 שניות) | תפעולי בלבד; גרף קריא במהירות | שלב 5 (demo) |
| **Class imbalance (ללא SPAM)** | class_weight={'0':1.0, '1':4.0} | יחס 80/20 → weight הפוך; חלופה: oversampling minority | שלב 4 |
| **LR scheduler pretraining** | ללא (fixed lr) | פשטות; אם loss לא מתכנס — הוסף cosine decay | שלב 3 |
| **Gradient clipping** | max_norm=1.0 | סטנדרטי ל-Transformer; מונע gradient explosion | שלבים 3-4 |
| **Differential LR (fine-tuning)** | backbone=1e-5, head=1e-4 | מונע catastrophic forgetting; backbone מבוסס pretraining שבריר מ-head חדש | שלב 4 |

**החלטה קריטית לגבי class imbalance (S6.1):**
מאחר שנתוני SPAM חסרים (הפתרון המקורי של המחקר), יש שתי חלופות:
- **אפשרות א (מומלצת):** `class_weight` ב-CrossEntropyLoss — פשוט, אין שינוי בנתונים
- **אפשרות ב:** Oversampling של הקלטות positive ב-dataloader — מגדיל את exposure למקרים נדירים

> תעד את הבחירה ב-`docs/deviation_log.md` לפני הרצת שלב 4.

---

### S7 — נרמול FHR: פרשנות מסקנה (P9)

| | |
|-|-|
| **תיאור** | המאמר מציין "FHR divided by 160" ללא הבהרה האם זה `fhr/160` (range [0.3125, 1.3125]) או `(fhr-50)/160` (range [0,1]) |
| **החלטה** | `(fhr-50)/160.0` — מיפוי [50,210] → [0,1]; זה הפרשון המסקנה המתמטית |
| **ביסוס** | מיפוי לינארי תקין על טווח הclip [50,210]; `fhr/160` יתן range חלקי לא מנורמל |
| **בדיקת רגישות מומלצת** | לאחר ריצה ראשונה: השווה val AUC עם `fhr/160` לעומת `(fhr-50)/160` — אם הפרש > 0.02, תעד |

### S2 — Hyperparameters ארכיטקטורה לא פורסמו

| פרמטר | ערך הנחה | מקור | ביסוס |
|--------|-----------|-------|-------|
| d_model | 128 | הנחה | PatchTST מקורי; dataset קטן → מודל קטן |
| num_layers | 3 | הנחה | PatchTST מקורי |
| n_heads | 4 | הנחה | 128/4=32 dim per head, סטנדרטי |
| ffn_dim | 256 | הנחה | פי 2 מ-d_model |

> **פעולה נדרשת:** hyperparameter sweep קטן על Val set לאחר שלב 2.

### S3 — ספליטים — פתור

**הספליטים קיימים ב-`CTGDL_norm_metadata.csv`** (עמודת `test`). אין צורך ביצירה חדשה.

### S4 — Stride חלון בפרה-טריינינג

| | |
|-|-|
| **תיאור** | המאמר לא מציין stride לחלון הזזה על ההקלטה הארוכה |
| **החלטה** | stride = 900 דגימות = 50% חפיפה |
| **ביסוס** | TS-MAE / Ti-MAE literature, 2024 |

### S5 — Epochs פרה-טריינינג

| | |
|-|-|
| **תיאור** | המאמר מציין 100 epochs ל-fine-tuning; epochs לפרה-טריינינג לא מצוינים |
| **החלטה** | עד 200 epochs עם early stopping (patience=10) על validation reconstruction loss |
| **ביסוס** | MAE ביו-רפואי literature: 100–200 epochs |

---

## חלק ה — Hyperparameters Reference Card

> **כל פרמטר מסומן:** ✓ **מאמר** (מה שכתוב מפורשות) | ⚠ **הנחה** (לתעד ולבדוק)

### ה.1 — Data / Windowing

| פרמטר | ערך | מקור |
|--------|-----|-------|
| תדירות דגימה | 4 Hz | ✓ מאמר Section II-A |
| אורך חלון | 1,800 דגימות (7.5 דקות) | ✓ מאמר Section II-C |
| Stride חלון (pretraining) | 900 דגימות (50% overlap) | ⚠ הנחה S4 |
| Stride חלון (inference — repro) | **1 דגימה** | ✓ מאמר Section II-F — "sliding window"; נדרש להערכה רשמית |
| Stride חלון (inference — runtime) | 60 דגימות (15 שניות) | ⚠ הנחה S6 — לשימוש תפעולי בלבד, **לא** לשלב 7 |
| pH threshold acidemia | <= 7.15 (umbilical artery pH) | ✓ פרויקט (S8), מאמר Section II-A |

### ה.2 — Patch Tokenization

| פרמטר | ערך | מקור |
|--------|-----|-------|
| Patch length | 48 דגימות | ✓ מאמר Section II-C, Eq. 1 |
| Patch stride | 24 דגימות | ✓ מאמר Section II-C |
| מספר patches לחלון | 73 | ✓ מחושב: (1800-48)/24 + 1 = 73 |
| סוג patch embedding | Linear projection + Positional embedding | ✓ Equation 1: x_d = W_P · x_p + W_pos |
| Channel independence | FHR ו-UC עוברים encoder נפרד (shared weights) | ✓ מאמר Section II-C |

### ה.3 — Encoder Architecture (Transformer)

| פרמטר | ערך | מקור |
|--------|-----|-------|
| d_model (embedding dim) | 128 | ⚠ הנחה S2 |
| num_layers | 3 | ⚠ הנחה S2 |
| n_heads | 4 | ⚠ הנחה S2 |
| ffn_dim | 256 | ⚠ הנחה S2 |
| Dropout | 0.2 | ✓ מאמר Section II-C |
| Normalization | Batch Normalization | ✓ מאמר Section II-C |

### ה.4 — Pre-training

| פרמטר | ערך | מקור |
|--------|-----|-------|
| Masking ratio | 0.4 (40% מ-FHR patches) | ✓ מאמר Section II-D |
| Masking strategy | רצועות רציפות (contiguous groups ≥ 2) | ✓ מאמר Section II-D, Figure 4 |
| Boundary preservation | הpatch הראשון והאחרון לא מוסתרים | ✓ מאמר Section II-D |
| Asymmetric masking | UC נגלה תמיד; FHR בלבד מוסתר | ✓ מאמר Section II-D |
| Masking type | Zero masking (החלפה ב-0) | ✓ מאמר Section II-D |
| Loss | MSE על patches מוסתרים בלבד: L=(1/|M|)·Σ_{i∈M}\|\|x^FHR_i − x̂^FHR_i\|\|² | ✓ Equation 2 |
| Optimizer | Adam, lr = 1×10⁻⁴ | ✓ מאמר Section II-D |
| Epochs | עד 200, early stopping patience=10 על val reconstruction loss | ⚠ הנחה S5 (המאמר לא מציין; early stopping לא הוזכר לפרה-טריינינג) |
| Corpus | CTU-UHB (552) + FHRMA (135) = 687 (≈1,706 שעות) | ⚠ סטייה S1 (מקור: 984=2,444 שעות) |

### ה.5 — Fine-tuning

| פרמטר | ערך | מקור |
|--------|-----|-------|
| Initialization | backbone מ-pretrained checkpoint | ✓ מאמר Section II-E |
| Classification head | Linear layer → 2 מחלקות | ✓ מאמר Section II-E |
| Optimizer | AdamW | ✓ מאמר Section II-E |
| Loss | Cross-entropy | ✓ מאמר Section II-E |
| Max epochs | 100 | ✓ מאמר Section II-E |
| Early stopping | על AUC של Val set | ✓ מאמר Section II-E |
| Dropout | 0.2 (encoder + head) | ✓ מאמר Section II-E |
| SPAM augmentation | לא זמין (סטייה S1) | ⚠ סטייה |
| Train set | 441 הקלטות CTU-UHB (test=0) | ✓ CTGDL_norm_metadata.csv |

### ה.6 — Alerting Protocol (Stage 2)

| פרמטר | ערך | מקור |
|--------|-----|-------|
| Alert threshold | 0.5 | ✓ מאמר Section II-F |
| Alert segment | קטע רציף שבו NN score > 0.5 | ✓ מאמר Section II-F |
| Feature 1 | segment length (דקות) | ✓ מאמר Section II-F |
| Feature 2 | max prediction (מקסימום score בקטע) | ✓ מאמר Section II-F |
| Feature 3 | cumulative sum (סכום scores בקטע) | ✓ מאמר Section II-F |
| Feature 4 | weighted integral: Σ_t (p_t − 0.5)² | ✓ מאמר Section II-F |
| קטע נבחר | קטע האזעקה הארוך/משמעותי ביותר | ✓ מאמר Section II-F |
| Stage 2 classifier | Logistic Regression, אמון על Train | ✓ מאמר Section II-F |

---

## חלק ו — מפת שלבים (Roadmap)

> **מוסכמה:** כל שלב מתאר INPUT → ACTIONS → OUTPUT → VALIDATION.
> שלב לא **מושלם** עד שכל הבדיקות עוברות.

---

### שלב 0 — חילוץ ארכיונים והכנת תיקיות

**מטרה:** כל הנתונים נגישים כקבצים CSV בתיקיות מסודרות.

**INPUT:**
- `data/CTGDL/CTGDL_ctu_uhb_proc_csv.tar.gz`
- `data/CTGDL/CTGDL_FHRMA_proc_csv.tar.gz`
- `data/CTGDL/CTGDL_norm_metadata.csv`

**ACTIONS:**
1. **חלץ CTU-UHB processed:**
   ```bash
   tar -xzf data/CTGDL/CTGDL_ctu_uhb_proc_csv.tar.gz -C data/raw/ctu_uhb/
   ```
   → תיקייה `data/raw/ctu_uhb/` עם 552 קבצי CSV בשם `ctgdl_ctu_uhb_XXXX.csv`

2. **חלץ FHRMA processed:**
   ```bash
   tar -xzf data/CTGDL/CTGDL_FHRMA_proc_csv.tar.gz -C data/raw/fhrma/
   ```
   → תיקייה `data/raw/fhrma/` עם 135 קבצי CSV

3. **בדוק שמות העמודות בקבצים המחולצים:**
   - CTU-UHB: לפחות עמודות FHR ו-UC (הערכים בטווח הגולמי)
   - FHRMA: אותה מבנה

4. **צור תיקיות יעד:**
   ```
   data/processed/ctu_uhb/    ← אחרי preprocessing שלב 1
   data/processed/fhrma/      ← אחרי preprocessing שלב 1
   data/splits/               ← קבצי splits (שלב 1)
   ```

5. **בדוק ספליטים מ-`CTGDL_norm_metadata.csv`:**
   ```python
   import pandas as pd
   meta = pd.read_csv('data/CTGDL/CTGDL_norm_metadata.csv')
   ctg = meta[meta['dataset'] == 'ctg']
   assert len(ctg[ctg['test'] == 0]) == 441  # train
   assert len(ctg[ctg['test'] == 1]) == 56   # val
   assert len(ctg[ctg['test'] == 2]) == 55   # test
   print("Splits OK")
   ```

**OUTPUT:**
- `data/raw/ctu_uhb/` — 552 CSVs
- `data/raw/fhrma/` — 135 CSVs

**VALIDATION:**
- [ ] 552 קבצים ב-`data/raw/ctu_uhb/`
- [ ] 135 קבצים ב-`data/raw/fhrma/`
- [ ] ספליטים: 441/56/55 ✓

---

### שלב 1 — עיבוד מקדים (Preprocessing) ויצירת קבצי Splits

**מטרה:** אותות מנורמלים ונקיים; קבצי splits מוקפאים; dataset classes מוכן.

**INPUT:**
- `data/raw/ctu_uhb/` + `data/raw/fhrma/` (מ-שלב 0)
- `data/CTGDL/CTGDL_norm_metadata.csv` (ספליטים + labels)
- `data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0/*.hea` (מקור קליני מלא — pH, Deliv. type, Presentation, NoProgress)
- `docs/2601.06149v1.pdf` Section II-B

> **הבהרה P1:** `metadata_summary.csv` חלקי (100 שורות בלבד). מקור הסמכות הקליני הוא קבצי `.hea` (552 קבצים).

**ACTIONS:**

#### 1.1 — עיבוד FHR (מ-Section II-B)
```python
# per recording, per sample:
fhr = raw_fhr.copy()
fhr[fhr < 50] = np.nan      # הסר outliers נמוכים
fhr[fhr > 220] = np.nan     # הסר outliers גבוהים
fhr = pd.Series(fhr).interpolate(method='linear').values  # interpolation לינארית
fhr = np.clip(fhr, 50, 210) # clip לטווח תקין
fhr = (fhr - 50) / 160.0   # נרמול → [0, 1]: (50-50)/160=0, (210-50)/160=1
```

#### 1.2 — עיבוד UC (מ-Section II-B)
```python
# זיהוי אזורים שטוחים (ארטיפקטים):
window = 120  # 30 שניות × 4 Hz
rolling_std = pd.Series(uc_raw).rolling(window=window, center=True).std()
flat_mask = (rolling_std < 1e-5) & (uc_raw < 80) & (~np.isnan(uc_raw))
uc = uc_raw.copy()
uc[flat_mask] = np.nan      # סמן ארטיפקטים כחסרים

uc = np.clip(uc, 0, 100)    # clip לטווח תקין
uc[~np.isnan(uc)] /= 100.0  # נרמול → [0, 1] (מחלק ב-100)
uc = np.nan_to_num(uc, nan=0.0)  # מלא NaN ב-0
```

#### 1.3 — פורמט שמירה
```python
# שמור כ-numpy array לכל הקלטה:
# shape: (2, T) — [0]=FHR, [1]=UC, T=מספר דגימות
np.save(f'data/processed/ctu_uhb/{record_id}.npy', np.stack([fhr, uc]))
```

#### 1.4 — יצירת קבצי Splits (מ-CTGDL_norm_metadata)
```python
import pandas as pd
meta = pd.read_csv('data/CTGDL/CTGDL_norm_metadata.csv')
ctg = meta[meta['dataset'] == 'ctg']

# שמור splits כקבצי CSV עם id + target + fname
ctg[ctg['test'] == 0][['id', 'target', 'fname']].to_csv('data/splits/train.csv', index=False)
ctg[ctg['test'] == 1][['id', 'target', 'fname']].to_csv('data/splits/val.csv', index=False)
ctg[ctg['test'] == 2][['id', 'target', 'fname']].to_csv('data/splits/test.csv', index=False)

# כל הקלטות לפרה-טריינינג (CTU-UHB + FHRMA)
pretrain = meta[meta['dataset'].isin(['ctg', 'fhrma'])][['id', 'dataset', 'fname']]
pretrain.to_csv('data/splits/pretrain.csv', index=False)
```

#### 1.5 — אימות labels מול pH
```python
# טען ctu_uhb_clinical_full.csv שנוצר בשלב 1.6 (ראה למטה)
clinical = pd.read_csv('data/processed/ctu_uhb_clinical_full.csv')
ph_acidemia = (clinical['pH'] <= 7.15).astype(int)
# השווה ל-target של norm_metadata לכל record_id — חייבים להתאים 100%
merged = ctg_meta.merge(clinical[['record_id','pH']], on='record_id')
mismatches = merged[merged['target'] != (merged['pH'] <= 7.15).astype(int)]
assert len(mismatches) == 0, f"pH label mismatch: {mismatches}"
```

#### 1.6 — בניית clinical metadata מ-552 קבצי `.hea` (P1 fix)

> **חיוני:** `metadata_summary.csv` חלקי (100 שורות). קבצי `.hea` הם מקור הסמכות.

```python
import re, os, csv

HEA_DIR = 'data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0/'
FIELDS = {
    'pH':          r'#pH\s+([\d.]+)',
    'BDecf':       r'#BDecf\s+([\d.]+)',
    'Apgar1':      r'#Apgar1\s+(\d+)',
    'Apgar5':      r'#Apgar5\s+(\d+)',
    'gest_weeks':  r'#Gest\. weeks\s+(\d+)',
    'weight_g':    r'#Weight\(g\)\s+(\d+)',
    'presentation': r'#Presentation\s+(\d+)',
    'induced':     r'#Induced\s+(\d+)',
    'stage1_min':  r'#I\.stage\s+(\d+)',
    'NoProgress':  r'#NoProgress\s+(\d+)',   # ← P2 fix: שדה הנכון לעצירת לידה
    'stage2_min':  r'#II\.stage\s+(\d+)',
    'delivery_type': r'#Deliv\. type\s+(\d+)',
    'pos_stage2':  r'#Pos\. II\.st\.\s+(\d+)',
}

rows = []
for fname in sorted(os.listdir(HEA_DIR)):
    if not fname.endswith('.hea'):
        continue
    record_id = fname.replace('.hea', '')
    content = open(os.path.join(HEA_DIR, fname), encoding='utf-8', errors='replace').read()
    row = {'record_id': record_id}
    for field, pattern in FIELDS.items():
        m = re.search(pattern, content)
        row[field] = m.group(1) if m else None
    rows.append(row)

with open('data/processed/ctu_uhb_clinical_full.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['record_id'] + list(FIELDS.keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Built clinical metadata: {len(rows)} records")
# בדוק שיש 552
assert len(rows) == 552
```

**שדות קריטיים לתתי-קבוצות הערכה (Table 3):**
| עמודה | ערכים | שימוש |
|-------|-------|-------|
| `delivery_type` | 1=ווגינלי, 2=קיסרי | סינון לידות וגינליות |
| `presentation` | 1=ראש (cephalic), 2=עכוז, ... | סינון מצגת ראש |
| `NoProgress` | 0=ללא עצירה, 1=עצירת לידה | סינון ללא עצירת לידה (P2 fix) |

**OUTPUT:**
- `data/processed/ctu_uhb/XXXX.npy` — 552 קבצים, shape (2, T)
- `data/processed/fhrma/XX.npy` — 135 קבצים, shape (2, T)
- `data/processed/ctu_uhb_clinical_full.csv` — **552 שורות**, כל השדות הקליניים מ-.hea (P1 fix)
- `data/splits/train.csv` — 441 שורות (id, target, fname)
- `data/splits/val.csv` — 56 שורות
- `data/splits/test.csv` — 55 שורות
- `data/splits/pretrain.csv` — 687 שורות (כל CTU-UHB + FHRMA)

**VALIDATION:**
- [ ] FHR כל ערכים בטווח [0, 1]
- [ ] UC כל ערכים בטווח [0, 1]
- [ ] acidemia train: 90/441 = 20.4% ✓ (מאמר Table 2: 20.4%)
- [ ] acidemia val: 12/56 = 21.4% ✓ (מאמר Table 2: 21.4%)
- [ ] acidemia test: 11/55 = 20.0% ✓ (מאמר Table 2: 20.0%; שים לב: Table 3 מציין 12 — כנראה שגיאת הדפסה)
- [ ] `ctu_uhb_clinical_full.csv` מכיל בדיוק 552 שורות (1 לכל .hea) ✓
- [ ] labels CTU-UHB תואמים pH <= 7.15 ב-ctu_uhb_clinical_full.csv ✓
- [ ] `ctu_uhb_clinical_full.csv` מכיל עמודת `NoProgress` (לא `labor_arrest`) ✓
- [ ] אין חפיפת IDs בין train/val/test ✓

**הערות נאמנות למחקר:** Section II-B. פרמטרים קריטיים: rolling window=120 דגימות, std threshold=1e-5, clip FHR ל-[50,210], נרמול FHR=(FHR-50)/160 → [0,1], UC=UC/100 → [0,1].

---

### שלב 2 — ארכיטקטורה: PatchTST Channel-Independent

**מטרה:** קוד ארכיטקטורה עובד עם שתי heads (שחזור / סיווג), בדיקת dimensions מלאה.

**INPUT:**
- `docs/2601.06149v1.pdf` Section II-C, Equation 1, Figure 3
- `config/train_config.yaml` (hyperparameters)

**ACTIONS:**

#### 2.1 — Patch Tokenization
```python
# input: (batch, channels=2, seq_len=1800)
# patch: non-overlapping strides of 48 with hop 24
# output per channel: (batch, n_patches=73, d_model=128)
patch_len = 48
patch_stride = 24
# n_patches = (1800 - 48) / 24 + 1 = 73
```

#### 2.2 — Channel-Independent Encoder (נפרד לFHR ולUC, משקלים משותפים)
```python
# מבנה per channel:
# 1) Linear projection: patch_len → d_model  (W_P)
# 2) Add positional embedding (W_pos) [learnable]
# 3) Transformer encoder:
#    - L=3 layers, each:
#      - Multi-head self-attention (n_heads=4, d_model=128)
#      - Feed-forward (ffn_dim=256)
#      - Residual + Batch Normalization
#      - Dropout=0.2
# output per channel: (batch, 73, 128)
```

#### 2.3 — Representation Concatenation
```python
# שרשר FHR repr ו-UC repr:
# fhr_repr: (batch, 73, 128) → flatten → (batch, 73*128=9344)
# uc_repr:  (batch, 73, 128) → flatten → (batch, 9344)
# concat:   (batch, 18688)
```

#### 2.4 — Pre-training Head (שחזור)
```python
# input: (batch, 73, 128) [FHR encoder output only, per patch]
# output: (batch, n_masked_patches, patch_len=48)
# loss: MSE על masked patches בלבד
```

#### 2.5 — Fine-tuning Head (סיווג)
```python
# לפי המאמר Section II-E: "a linear layer mapping the flattened encoder output to two classes"
# כלומר: שכבה לינארית אחת בלבד, ללא hidden layer נוסף
# input: (batch, 18688) [concat FHR+UC flattened]
# dropout: 0.2 (לפני השכבה הלינארית — ממאמר Section II-D)
# output: Linear(18688, 2) → softmax → probability of acidemia
#
# ⚠ הנחה S2: גודל 18688 = 73*128*2 תלוי ב-d_model=128 שהוא הנחה
# ⚠ אם d_model שונה, גודל הכניסה לhead משתנה בהתאם
```

**OUTPUT:**
- `src/model/patchtst.py` — ארכיטקטורה מלאה
- `src/model/heads.py` — PretrainingHead, ClassificationHead
- `config/train_config.yaml` — כל hyperparameters
- `notebooks/01_arch_check.ipynb` — בדיקת dimensions

**VALIDATION:**
- [ ] Input (2, 1800) → 73 patches per channel ✓
- [ ] Encoder output shape: (batch, 73, 128) ✓
- [ ] Pre-training head: reconstructs 48*n_masked values ✓
- [ ] Classification head: outputs probability vector (2,) ✓
- [ ] FHR ו-UC encoder משתמשים **באותם משקלים** (shared backbone) ✓

**הערות נאמנות למחקר:** Section II-C, Equation 1 (`x_d = W_P · x_p + W_pos`), Figure 3.

---

### שלב 3 — פרה-טריינינג: Channel-Asymmetric Masking

**מטרה:** backbone מאומן self-supervised על 687 הקלטות; loss יורד; FHR נשחזר מ-UC + FHR חלקי.

**INPUT:**
- `data/splits/pretrain.csv` — 687 IDs
- `data/processed/ctu_uhb/` + `data/processed/fhrma/`
- `src/model/patchtst.py`

**ACTIONS:**

#### 3.1 — Dataset & DataLoader
```python
# לכל הקלטה: sliding window, stride=900
# output per window: (2, 1800) — FHR + UC normalized
```

#### 3.2 — Asymmetric Masking

> **P6 fix v2:** אלגוריתם דו-שלבי דטרמיניסטי.
> שלב א: פירוק `target_masked` לסכום קבוצות חוקיות (כל אחת ≥2).
> שלב ב: שיבוץ הקבוצות על אינדקסים חוקיים (1..n-2) ללא חפיפה.
> אם שיבוץ נכשל — retry מלא (לא greedy המשכי).

```python
def _random_partition(total, min_size=2, max_size=6):
    """שלב א: פירוק total לרשימת קבוצות, כל אחת בטווח [min_size, max_size]."""
    groups = []
    remaining = total
    while remaining > 0:
        if remaining < min_size:
            # לא ייתכן: total לא מתחלק → הרחב קבוצה אחרונה
            groups[-1] += remaining
            remaining = 0
        else:
            g = random.randint(min_size, min(max_size, remaining))
            # וודא שמה שנשאר לא יהיה 1 (שלא ניתן לפירוק)
            if remaining - g == 1:
                g = remaining  # מצרף הכל לקבוצה אחת
            groups.append(g)
            remaining -= g
    return groups

def apply_masking(fhr_patches, mask_ratio=0.4, max_retries=100):
    """
    P6 fix v2: deterministic two-phase contiguous group masking.
    מבטיח:
      - boundary preservation: mask[0] = mask[-1] = False
      - כל קבוצה רציפה ≥ 2 patches
      - sum(mask) == target_masked בדיוק
      - לא נתקע בלולאה אינסופית (max_retries)
    """
    n = len(fhr_patches)  # 73
    target_masked = round(mask_ratio * n)  # 29
    eligible_range = n - 2  # אינדקסים 1..71

    for attempt in range(max_retries):
        # שלב א: פירוק target_masked לקבוצות חוקיות
        groups = _random_partition(target_masked, min_size=2, max_size=6)

        # שלב ב: שיבוץ על אינדקסים חוקיים ללא חפיפה
        mask = np.zeros(n, dtype=bool)
        # דרוש: מציאת מיקום לכל קבוצה בתוך [1, n-2]
        # נדגום מיקומים רנדומליים ונבדוק חוקיות
        positions = []
        success = True
        random.shuffle(groups)
        for g_len in groups:
            # מצא מיקומי start חוקיים שלא חופפים
            valid_starts = []
            for s in range(1, n - 1 - g_len + 1):
                if not any(mask[s:s + g_len]):
                    valid_starts.append(s)
            if not valid_starts:
                success = False
                break
            start = random.choice(valid_starts)
            mask[start:start + g_len] = True
            positions.append((start, g_len))

        if success and mask.sum() == target_masked:
            break
    else:
        raise RuntimeError(f"Masking failed after {max_retries} retries")

    # Assertions
    assert not mask[0] and not mask[-1], "Boundary violation"
    assert mask.sum() == target_masked, f"Count: {mask.sum()} != {target_masked}"
    # בדוק שאין קבוצות באורך 1
    diffs = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]
    assert all((e - s) >= 2 for s, e in zip(starts, ends)), "Group < 2"

    fhr_patches[mask] = 0.0  # zero masking
    return fhr_patches, np.where(mask)[0]

# ⚠ בדיקת יציבות נדרשת לפני נעילת המימוש:
# for seed in range(10_000):
#     random.seed(seed); np.random.seed(seed)
#     dummy = np.random.rand(73, 48)
#     apply_masking(dummy.copy())
# print("10,000 seeds passed — masking is stable")
```

#### 3.3 — Loss Function
```python
# Equation 2 — MSE על masked FHR patches בלבד:
loss = F.mse_loss(pred_fhr[mask_indices], original_fhr[mask_indices])
# UC: לא נכנס ל-loss בכלל
```

#### 3.4 — Training Loop
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(max_epochs=200):
    for batch in dataloader:
        # apply masking to FHR
        # forward pass
        # compute MSE loss on masked patches only
        # backward + step

    # Early stopping: אם val reconstruction loss לא ירד ב-10 epochs רצופים
    if no_improvement >= patience(10):
        break

    # שמור checkpoint כל epoch לGoogle Drive
    torch.save(model.state_dict(), f'checkpoints/pretrain/epoch_{epoch}.pt')
```

**OUTPUT:**
- `checkpoints/pretrain/epoch_N.pt` — checkpoint כל epoch
- `checkpoints/pretrain/best_pretrain.pt` — מודל עם lowest val reconstruction loss
- `logs/pretrain_loss.csv` — train/val loss per epoch

**VALIDATION:**
- [ ] **תנאי קדם:** בדיקת יציבות masking — 10,000 seeds ללא כשל (ראה קוד בדיקה ב-3.2) ✓
- [ ] Loss יורד לאורך epochs ✓
- [ ] UC אינו בחישוב loss — loss מחושב רק על M (masked FHR patches) ✓
- [ ] masking: groups ≥ 2, אין boundary, יחס ~40% ✓
- [ ] Checkpoint נשמר ב-Google Drive ✓

**הערות נאמנות למחקר:** Section II-D, Equation 2, Figure 4. **Corpus:** 687 (סטייה S1 — 297 SPAM חסרים).

---

### שלב 4 — Fine-tuning לסיווג Acidemia

**מטרה:** מודל עם backbone פרה-טריינד + classification head מאומן על CTU-UHB Train; מוערך על Val.

**INPUT:**
- `checkpoints/pretrain/best_pretrain.pt`
- `data/splits/train.csv` (441 הקלטות), `data/splits/val.csv` (56)
- `data/processed/ctu_uhb/`
- `docs/2601.06149v1.pdf` Section II-E

**ACTIONS:**

#### 4.1 — אתחול מודל
```python
model = PatchTST(...)
model.load_state_dict(torch.load('checkpoints/pretrain/best_pretrain.pt'))
model.replace_head(ClassificationHead(d_in=18688, n_classes=2, dropout=0.2))
```

#### 4.2 — Dataset וטיפול ב-Imbalance

> **P8 fix:** החלטת imbalance נעולה. **אפשרות א נבחרת: class_weight.**

```python
# CTU-UHB train set: כל הקלטה → sliding window → קטעים של 1800 דגימות
# label: target מ-splits/train.csv (acidemia=1, normal=0)
# P8 fix: CLASS WEIGHT נבחר (לא oversampling) — ראה S6.1 בחלק ד
#
# חישוב אוטומטי מ-Train בלבד (לא מ-Val/Test):
import torch
n_neg, n_pos = train_df['target'].value_counts().sort_index()  # ~351, ~90
class_weights = torch.tensor([1.0, n_neg / n_pos])  # [1.0, ~3.9]
# הגדרת criterion: CrossEntropyLoss(weight=class_weights)
```

#### 4.3 — הגדרת יחידת אימון ו-Validation (P7 fix)

> **P7 fix:** יחידת אימון = **window** (1800 דגימות). יחידת הערכה (AUC) = **הקלטה**.
> AUC על Val set מחושב per-recording בלבד — **לא** per-window.

```python
# Aggregation function — נעולה:
# recording_score = max(window_scores_for_recording)
#
# נימוק: המאמר מציין "sliding window → continuous risk score per recording"
# ו-AUC מחושב per-recording (label = 0/1 לכל הקלטה).
# max aggregation שמרנית: מזהה הקלטה כ-positive אם לפחות חלון אחד חרג.

def compute_recording_auc(model, split_csv, processed_dir, stride=1):
    """
    מחזיר AUC per-recording.
    P7: aggregation קבועה = max score per recording.
    """
    df = pd.read_csv(split_csv)
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        signal = np.load(f'{processed_dir}/{row.id}.npy')
        scores = [s for _, s in inference_recording(model, signal, stride)]
        recording_score = max(scores)  # aggregation קבועה: max
        y_true.append(row.target)
        y_pred.append(recording_score)
    return roc_auc_score(y_true, y_pred)
```

#### 4.4 — Training Loop
```python
# Differential LR: backbone LR נמוך יותר למניעת catastrophic forgetting
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},  # backbone: עדין
    {'params': model.head.parameters(),     'lr': 1e-4},  # head: רגיל
], weight_decay=1e-2)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

best_val_auc = 0
patience_counter = 0

for epoch in range(100):
    # train loop — per window
    model.train()
    for batch in train_loader:
        # forward, compute loss, backward, step
        pass

    # val loop — per RECORDING (P7 fix: aggregation=max)
    model.eval()
    val_auc = compute_recording_auc(model, 'data/splits/val.csv',
                                     'data/processed/ctu_uhb',
                                     stride=INFERENCE_STRIDE_REPRO)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'checkpoints/finetune/best_finetune.pt')
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= 15:
        break
```

#### 4.4 — הערת SPAM Augmentation
המחקר מוסיף הקלטות SPAM (stage2=0 = ניתוח קיסרי אינטראפרטום) כ-positive examples.
**כרגע לא זמין** — ראה סטייה S1.
כשנתוני SPAM יתקבלו: הוסף שורות עם `target=1` ל-`train.csv` לפני שלב זה.

**OUTPUT:**
- `checkpoints/finetune/best_finetune.pt`
- `logs/finetune_val_auc.csv` — AUC per epoch

**VALIDATION:**
- [ ] Test set לא נגעה בשום שלב עד שלב 6 ✓
- [ ] Val AUC עולה; אם לא — בדוק data leakage, preprocessing, hyperparameters ✓
- [ ] Val AUC → צפוי 0.75–0.85 (תלוי ב-backbone quality) ✓

**הערות נאמנות למחקר:** Section II-E. פרמטרים: AdamW, cross-entropy, 100 epochs, early stopping on val AUC, dropout=0.2.

---

### שלב 5 — פייפליין Inference ומנגנון Alerting

**מטרה:** הרצת מודל על הקלטה שלמה → ציון רציף → זיהוי קטעי-אזעקה → LR → החלטה בינארית.

**INPUT:**
- `checkpoints/finetune/best_finetune.pt`
- `data/splits/train.csv` (לאימון LR)
- `data/processed/ctu_uhb/`
- `docs/2601.06149v1.pdf` Section II-F, Figure 5

**ACTIONS:**

#### 5.1 — Stage 1: Sliding Window Inference

> **P4 fix:** שני מצבים בלבד. הערכה רשמית (שלב 7) — repro_mode בלבד.

```python
INFERENCE_STRIDE_REPRO    = 1   # repro_mode: להשוואה מדויקת למאמר
INFERENCE_STRIDE_RUNTIME  = 60  # runtime_mode: הדמיה מהירה בלבד

def inference_recording(model, signal, stride=INFERENCE_STRIDE_REPRO):
    """
    signal: (2, T) — FHR + UC
    stride: REPRO=1 (הערכה רשמית), RUNTIME=60 (הדמיה בלבד)
    returns: list of (start_sample, score)
    """
    T = signal.shape[1]
    scores = []
    for start in range(0, T - 1800 + 1, stride):
        window = signal[:, start:start+1800]
        score = model(window).softmax(-1)[..., 1].item()  # P(acidemia)
        scores.append((start, score))
    return scores
# שלב 7: תמיד קרא עם stride=INFERENCE_STRIDE_REPRO
```

#### 5.2 — Alert Segment Extraction
```python
def extract_alert_segments(scores, threshold=0.5):
    """מחזיר רשימת segments רציפים שבהם score > 0.5"""
    alert_mask = [s > threshold for _, s in scores]
    segments = []
    # חיפוש רצועות רציפות של True
    # החזר: [(start_time, end_time, scores_in_segment), ...]
    return segments
```

#### 5.3 — Stage 2: Feature Extraction + LR

> **P5 fix v2:** כל feature שמכיל זמן/אינטגרל מנורמל ל-stride בנוסחה.
> `dt = stride / fs` — שניות לכל צעד inference.
> **כלל:** LR מאומן ומוערך **באותו stride בלבד**. שלב 7 = repro_mode (stride=1).
> אם runtime_mode רצוי — אמן LR נפרד עליו.

```python
def compute_alert_features(segment_scores, inference_stride=1, fs=4):
    """
    4 features בדיוק מ-Section II-F, מנורמלים ליחידות זמן.
    P5 fix v2: כל feature אינטגרלי מוכפל ב-dt.

    segment_scores: רשימת scores (p_t) ב-window-level
    inference_stride: stride (דגימות) — חייב להיות זהה לאימון LR!
    fs: תדירות דגימה (4 Hz)
    """
    p = np.array(segment_scores)
    dt = inference_stride / fs  # שניות לכל צעד; repro: 0.25s, runtime: 15s

    return {
        # segment_length: זמן כולל בדקות
        'segment_length': len(p) * dt / 60,
        # max_prediction: ללא תלות ב-stride (ערך נקודתי)
        'max_prediction': np.max(p),
        # cumulative_sum: אינטגרל ← מנורמל ב-dt
        'cumulative_sum': np.sum(p) * dt,
        # weighted_integral: אינטגרל ← מנורמל ב-dt
        'weighted_integral': np.sum((p - 0.5) ** 2) * dt
    }

# ⚠ כלל קריטי: LR training ו-evaluation חייבים להשתמש באותו stride!
INFERENCE_STRIDE_REPRO = 1   # הערכה רשמית (שלב 7) — חובה
# INFERENCE_STRIDE_RUNTIME = 60  # הדמיה בלבד — LR נפרד אם נדרש

# אמן LR על Train set בלבד (stride=REPRO)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_features, y_train)  # X=features, y=acidemia label
# שמור stride כ-metadata:
import joblib
joblib.dump({'model': lr, 'stride': INFERENCE_STRIDE_REPRO},
            'checkpoints/alerting/logistic_regression.pkl')
```

**OUTPUT:**
- `src/inference/sliding_window.py`
- `src/inference/alert_extractor.py`
- `checkpoints/alerting/logistic_regression.pkl`
- `notebooks/04_inference_demo.ipynb` — הדמיה על הקלטה אחת

**VALIDATION:**
- [ ] Figure 5 במאמר: ציון רציף דומה לדוגמה (pH=7.02) ✓
- [ ] LR אמון על Train בלבד ✓
- [ ] 4 features מחושבים נכון ✓

---

### שלב 6 — סביבת Colab ב-VS Code

**מטרה:** GPU זמין, נתונים נגישים, checkpoints נשמרים ב-Drive, הרצה חוזרת מובטחת.

**INPUT:**
- `docs/colab-vscode-guide-hebrew.md`

**ACTIONS:**
1. התקן `Google Colab` extension ב-VS Code (v0.3.0+)
2. Activity Bar → Sign in → Connect to Colab Runtime
3. הרץ `Mount Google Drive` → בדוק גישה ל-`/content/drive/MyDrive/SentinelFatal2/`
4. העתק את `data/`, `src/`, `config/`, `checkpoints/` ל-Google Drive לפני אימון
5. בדוק GPU: `import torch; print(torch.cuda.is_available())`  — חייב להחזיר True
6. ודא שכל checkpoint נשמר ל-Drive path (לא ל-`/content/` הזמני)

**מבנה Notebooks:**

| Notebook | שלב | תוכן |
|----------|-----|-------|
| `notebooks/00_data_prep.ipynb` | שלב 0+1 | חילוץ + preprocessing + splits |
| `notebooks/01_arch_check.ipynb` | שלב 2 | בדיקת dimensions |
| `notebooks/02_pretrain.ipynb` | שלב 3 | פרה-טריינינג loop |
| `notebooks/03_finetune.ipynb` | שלב 4 | fine-tuning loop |
| `notebooks/04_inference_demo.ipynb` | שלב 5 | הדמיה + alerting |
| `notebooks/05_evaluation.ipynb` | שלב 7 | הערכה סופית |

**Reproducibility:**
```python
# ראש כל notebook:
import torch, random, numpy as np
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = True
```

**VALIDATION:**
- [ ] GPU זמין (T4 לפחות) ✓
- [ ] Drive mount פעיל ✓
- [ ] checkpoint נשמר ל-Drive אחרי כל epoch ✓

---

### שלב 7 — הערכה סופית (Test Set)

> **כלל:** שלב זה מורץ **פעם אחת בלבד**, רק לאחר שכל hyperparameters נעולים ו-fine-tuning הסתיים.

**INPUT:**
- `checkpoints/finetune/best_finetune.pt`
- `checkpoints/alerting/logistic_regression.pkl`
- `data/splits/test.csv` (55 הקלטות)
- `data/processed/ctu_uhb/`
- `data/processed/ctu_uhb_clinical_full.csv` (לתת-קבוצות — **552 שורות** מ-.hea; P1+P2 fix)
- `docs/2601.06149v1.pdf` Table 3, Figure 6

**ACTIONS:**

#### 7.1 — AUC לפי תת-קבוצות (Table 3 מהמאמר — ערכים מדויקים)

> **הבהרה חשובה:** Table 2 במאמר מציין 11 acidemia ב-test (20.0%), אך Table 3 מציין 12 (21.4%). הנתונים ב-`CTGDL_norm_metadata.csv` מאשרים **11** acidemia בתת-קבוצת test=2. האי-התאמה ב-Table 3 כנראה שגיאת הדפסה במאמר. **השתמש ב-11 מהמטה-דאטה** לחישובי splits; השווה לערכי AUC המדווחים.

| תת-קבוצה | n | Acidemia | Prevalence | AUC (benchmark) | Accuracy | סינון מ-`ctu_uhb_clinical_full.csv` |
|-----------|---|----------|------------|------------------|----------|--------------------------------------|
| כלל Test | 55 | 12* | 21.4%* | **0.826** | 0.786 | — |
| לידות וגינליות | 50 | 10 | 20.0% | **0.850** | 0.800 | `delivery_type == 1` |
| מצגת ראש | 50 | 10 | 20.0% | **0.848** | 0.800 | `presentation == 1` |
| וגינלי + ראש | 46 | 9 | 19.6% | **0.853** | 0.804 | `delivery_type==1 AND presentation==1` |
| ללא עצירת לידה | 47 | 8 | 17.0% | **0.837** | 0.830 | `NoProgress == 0` (P2 fix) |
| וגינלי+ראש+ללא עצירה | 43 | 7 | 16.3% | **0.837** | 0.837 | `delivery_type==1 AND presentation==1 AND NoProgress==0` |

\* מספר ה-acidemia ב-Table 3 של המאמר; הנתון ב-CTGDL_norm_metadata מראה 11 (20.0%).

#### 7.2 — עקומות ROC
```python
from sklearn.metrics import roc_auc_score, roc_curve
for subset, ids in subsets.items():
    auc = roc_auc_score(y_true[ids], y_pred[ids])
    # plot ROC curve
```

#### 7.3 — Case Studies (Figure 6)
```python
# הרץ inference על 5 מקרים נבחרים ממאמר:
# - 2 False Positives
# - 1 True Positive (pH <= 7.15)
# - 1 True Negative
# - 1 ניתוח קיסרי
# עבור כל מקרה: גרף ציון רציף + סמן stage 2
```

**OUTPUT:**
- `results/evaluation_table3.csv`
- `results/roc_curves.png`
- `results/case_studies/` — גרפים לכל מקרה
- `results/final_report.md`

**VALIDATION:**
- [ ] AUC Test ≥ 0.75 (כלל) — **קריטריון הצלחה מינימלי** (מכה את baseline) ✓
- [ ] AUC Test ≥ 0.826 (כלל) — **יעד benchmark** מהמאמר (ייתכן פחות עם 687 במקום 984) ✓
- [ ] AUC וגינלי+ראש ≥ 0.853 — **יעד benchmark** (P3: reference, לא pass/fail קשיח) ✓
- [ ] Test set לא נגע בשום checkpoint/hyperparameter selection קודם ✓
- [ ] הדיווח כולל ספירות תתי-קבוצה בפועל (לא רק AUC) ✓

> **P3 note:** ספירות תתי-קבוצה בדאטה המקומי עשויות להיות שונות מהמאמר (ידועה אי-התאמה של 11 vs 12 acidemia ב-test). יש לדווח על ספירות בפועל, לא להכריז כישלון אם AUC נמוך בשל הבדל בסינון. קריטריון הצלחה ראשי: AUC כלל test vs baseline 0.68–0.75.

---

## חלק ז — חוקי ממשל (Governance)

### ז.1 — קבועים שאסור לשנות

| קבוע | ערך | אסמכתא |
|------|-----|---------|
| אורך חלון | 1,800 דגימות | Section II-C |
| pH threshold | <= 7.15 | Deviation S8 (project policy) |
| Splits | 441/56/55 | CTGDL_norm_metadata.csv עמודת `test` |
| Patch length | 48 | Equation 1 |
| Patch stride (within window) | 24 | Equation 1 |
| n_patches per window | 73 | מחושב |
| Masking ratio | 0.4 | Section II-D |
| Masking: groups ≥ 2 | כן | Section II-D |
| Masking: boundary preservation | כן | Section II-D |
| Alert threshold | 0.5 | Section II-F |
| LR features | 4 בדיוק | Section II-F |
| Adam lr pretrain | 1e-4 | Section II-D |
| Dropout | 0.2 | Section II-E |

### ז.2 — מניעת דליפת מידע

```
pretrain → train → val ↑ (early stopping signal)
                 → test ✗ (אסור לגעת עד שלב 7)
LR: אמון על train features בלבד
```

### ז.3 — תיעוד סטיות

כל סטייה מהמאמר מתועדת ב-`docs/deviation_log.md` בפורמט:

```markdown
## S[N] — שם הסטייה
- **תיאור:**
- **סיבה:**
- **השפעה צפויה:**
- **טיפול:**
```

סטיות קיימות: S1 (SPAM חסר), S2 (hyperparameters הנחה), S4 (stride הנחה), S5 (epochs הנחה), S6 (פרמטרים חסרים), S7 (נרמול FHR פרשנות).

> **P10 fix:** קובץ `docs/deviation_log.md` **קיים** עם כל הסטיות הנוכחיות.

---

## חלק ח — מבנה תיקיות הפרויקט

```
SentinelFatal2/
├── data/
│   ├── CTGDL/                          ← ארכיבים + metadata (קיים)
│   │   ├── CTGDL_norm_metadata.csv     ← SSOT: splits + labels
│   │   ├── CTGDL_FHEMA_metadata.csv
│   │   ├── CTGDL_SPAM_metadata.csv
│   │   ├── CTGDL_ctu_uhb_csv.tar.gz
│   │   ├── CTGDL_ctu_uhb_proc_csv.tar.gz
│   │   ├── CTGDL_FHRMA_ano_csv.tar.gz
│   │   └── CTGDL_FHRMA_proc_csv.tar.gz
│   ├── ctu-chb-intrapartum-*/          ← PhysioNet raw .dat/.hea (קיים)
│   │   └── analysis_results/
│   │       ├── metadata_summary.csv    ← pH + clinical metadata
│   │       └── signal_quality_stats.csv
│   ├── raw/                            ← אחרי חילוץ ארכיבים (שלב 0)
│   │   ├── ctu_uhb/                    ← 552 CSVs
│   │   └── fhrma/                      ← 135 CSVs
│   ├── processed/                      ← אחרי preprocessing (שלב 1)
│   │   ├── ctu_uhb/                    ← 552 .npy files
│   │   └── fhrma/                      ← 135 .npy files
│   └── splits/                         ← קבצי splits (שלב 1)
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── pretrain.csv
├── docs/
│   ├── 2601.06149v1.pdf                ← מאמר (SSOT)
│   ├── work_plan.md                    ← מסמך זה
│   └── deviation_log.md               ← תיעוד סטיות
├── src/
│   ├── model/
│   │   ├── patchtst.py
│   │   └── heads.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   └── inference/
│       ├── sliding_window.py
│       └── alert_extractor.py
├── config/
│   └── train_config.yaml
├── notebooks/
│   ├── 00_data_prep.ipynb
│   ├── 01_arch_check.ipynb
│   ├── 02_pretrain.ipynb
│   ├── 03_finetune.ipynb
│   ├── 04_inference_demo.ipynb
│   └── 05_evaluation.ipynb
├── checkpoints/
│   ├── pretrain/
│   ├── finetune/
│   └── alerting/
├── logs/
└── results/
```

---

## נספח — Open-Source Resources

> אלה **לא** חלק מ-SSOT — משאבים חיצוניים לעיון בלבד.

| פרויקט | קישור | מה ניתן ללמוד |
|--------|--------|----------------|
| **PatchCTG** | arXiv:2411.07796, GitHub: jaleedkhan/PatchCTG | PatchTST על CTG, pipeline בפייתון |
| **PatchTST מקורי** | arXiv:2211.14730, GitHub: yuqinie98/PatchTST | קוד self-supervised pretraining, masking |
| **HeartGPT** | GitHub: harryjdavies/HeartGPT | GPT pretraining לאותות לב |
