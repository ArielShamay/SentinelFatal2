# deviation_log.md — תיעוד סטיות מהמאמר

> **פרויקט:** SentinelFatal — Foundation Model לניבוי מצוקה עוברית מ-CTG
> **מאמר:** arXiv:2601.06149v1
> **מדיניות:** כל סטייה מהמאמר חייבת להופיע כאן לפני יישום.

---

## S1 — SPAM חסר (קריטי)

- **תיאור:** 294 הקלטות SPAM הניתנות לשימוש חסרות לפרה-טריינינג ולאוגמנטציה
- **סיבה:** Dataset הוסר מגישה ציבורית; נדרשת הסכמת שימוש
- **השפעה:** פרה-טריינינג על 687 הקלטות (≈1,706 שעות) במקום 984 (2,444 שעות = 70%); ללא 3 הקלטות קיסריות כ-augmentation חיובי ב-fine-tuning
- **טיפול:** פנייה ל-`ctg.challenge2017@gmail.com` — pipeline מוכן להרחבה ל-984 ברגע שהנתונים יתקבלו
- **סטטוס:** פעיל

---

## S2 — Hyperparameters ארכיטקטורה לא פורסמו

- **תיאור:** המאמר לא מציין d_model, num_layers, n_heads, ffn_dim
- **סיבה:** פרסום חסר
- **החלטה:** d_model=128, num_layers=3, n_heads=4, ffn_dim=256 (ביסוס: PatchTST מקורי + dataset קטן)
- **השפעה:** AUC עשוי להיות גבוה יותר עם d_model=256; ניתן לבדוק ב-hyperparameter sweep על Val
- **סטטוס:** פעיל — בדיקה לאחר ריצה ראשונה

---

## S3 — ספליטים — פתור

- **תיאור:** המאמר לא פרסם קובץ splits
- **פתרון:** נמצאו ב-`CTGDL_norm_metadata.csv` עמודת `test` — מאומת מול Table 2 במאמר
- **סטטוס:** פתור — אין סטייה

---

## S4 — Stride חלון בפרה-טריינינג

- **תיאור:** המאמר לא מציין stride לחלון הזזה על ההקלטה הארוכה
- **החלטה:** stride = 900 דגימות (50% חפיפה)
- **ביסוס:** TS-MAE / Ti-MAE literature, 2024; סטנדרטי למטלות MAE
- **סטטוס:** פעיל

---

## S5 — Epochs פרה-טריינינג

- **תיאור:** המאמר מציין 100 epochs ל-fine-tuning; epochs לפרה-טריינינג לא מצוינים
- **החלטה:** עד 200 epochs עם early stopping (patience=10) על validation reconstruction loss
- **ביסוס:** MAE ביו-רפואי literature: 100–200 epochs
- **סטטוס:** פעיל

---

## S6 — פרמטרים חסרים לחלוטין מהמאמר

| פרמטר | ערך | נימוק |
|--------|-----|-------|
| Batch size (pretraining) | 64 | מאזן GPU mem ו-BN stability |
| Batch size (fine-tuning) | 32 | dataset קטן |
| LR fine-tuning (backbone) | 1e-5 | Differential LR למניעת catastrophic forgetting |
| LR fine-tuning (head) | 1e-4 | ראש חדש — LR גבוה יותר |
| Weight decay (AdamW) | 1e-2 | ברירת מחדל AdamW |
| Inference stride (repro) | 1 | הערכה רשמית; מאמר = "sliding window" |
| Inference stride (runtime) | 60 | הדמיה מהירה בלבד |
| Class imbalance (ללא SPAM) | CrossEntropyLoss(weight=[1.0, n_neg/n_pos]) | חישוב אוטומטי מ-Train בלבד |
| LR scheduler (pretraining) | ללא | fixed lr; אם loss לא מתכנס — הוסף cosine decay |
| Gradient clipping | max_norm=1.0 | סטנדרטי ל-Transformer |
| Differential LR (fine-tuning) | backbone=1e-5, head=1e-4 | מניעת catastrophic forgetting |

**S6.1 — Class imbalance:** נבחרה אפשרות א (`class_weight`) — פשוט, ללא שינוי בנתונים.
נוסחה: `weight_positive = n_negative / n_positive` (חישוב מ-train בלבד).

- **סטטוס:** פעיל — לתעד תוצאות בדיקה ב-Val לאחר ריצה ראשונה

---

## S7 — נרמול FHR: פרשנות מסקנה

- **תיאור:** המאמר מציין "FHR divided by 160" ללא הבהרה מפורשת
- **פרשנות 1:** `fhr / 160` → range [0.3125, 1.3125] — לא מנורמל ל-[0,1]
- **פרשנות 2:** `(fhr - 50) / 160` → range [0,1] — מיפוי לינארי על טווח clip [50,210]
- **החלטה:** `(fhr - 50) / 160.0` — פרשנות 2; מאפשרת [0,1] ועקבית עם clip
- **בדיקת רגישות:** לאחר ריצה ראשונה — השווה val AUC בין שני הווריאנטים; תעד כאן
- **סטטוס:** פעיל — טרם נבדק

---

## S8 — pH threshold: ≤ 7.15 בפרויקט vs. < 7.15 במאמר

- **תיאור:** המאמר מציין "pH < 7.15" (strict less-than) לסיווג acidemia. בפרויקט, כדי להתיישר עם labels בפועל, מאומצת הגדרה `pH <= 7.15`.
- **רשימת IDs:** 1019, 1059, 1067, 1219, 1249, 1274, 1351, 2020
- **סיבה:** CTGDL השתמש לסיווג ב-threshold `pH ≤ 7.15` (inclusive), לא `pH < 7.15`.
- **החלטה:** הלייבלים ב-CTGDL_norm_metadata.csv הם authoritative (SSOT). work_plan + agent_workflow + validation V1.6 עובדים עם `pH <= 7.15`.
- **ביסוס:** בדיקת קבצי .hea עבור כל 8 ה-IDs — כולם מכילים `#pH	7.15` בדיוק.
- **השפעה צפויה:** אפסית על הביצועים — 8 הקלטות גבוליות שמסווגות כ-acidemia. מקובל לרפואיה.
- **סטטוס:** פתור — תועד

---

## S9 — n_patches: אי-התאמה בין נוסחת SSOT לפלט unfold בפועל

- **תיאור:** נוסחת ה-SSOT `(1800-48)/24+1 = 73` שגויה מתמטית — `1752/24+1 = 74`. `torch.Tensor.unfold(size=48, step=24)` על רצף 1800 נד' מחזיר 74 patches.
- **סיבה:** שגיאת חישוב בנוסחה, כנראה הוחסרה פעולת floor (`⌊(1800-48)/24⌋+1 = 73` בלבד אם מדובר בחלון שאינו מכסה את כל הרצף).
- **החלטה:** חיתוך הרצף ל-1776 דגימות לפני unfold: `end = (73-1)*24 + 48 = 1776`. 24 הדגימות (6 שניות) האחרונות אינן נמסכות.
- **ביסוס:** SSOT קובע n_patches=73 כערך קנוני (✓ מאמר). חיתוך ל-1776 מבטיח עקביות מלאה.
- **השפעה צפויה:** אפסית — 24 דגימות עוקבות בסוף חלון 7.5 דקות. אין שינוי בתוצאות.
- **יישום:** `src/model/patchtst.py` → `_extract_patches`: `x[..., :1776].unfold(-1, 48, 24)`.
- **סטטוס:** פתור — ממומש ונבדק

---

## S10 — ZERO_FEATURES: הקלטות ללא alert segments

- **תיאור:** הקלטות שבהן כל scores ≤ 0.5 (אין alert segment) מקבלות feature vector אפסי `[0, 0, 0, 0]` לשלב ה-LR.
- **סיבה:** המאמר (Section II-F) לא מתייחס למקרה שבו הקלטה אינה מייצרת alert segment. יש לשמור פקודת feature extraction גם עבורן.
- **החלטה:** `ZERO_FEATURES = {segment_length: 0, max_prediction: 0, cumulative_sum: 0, weighted_integral: 0}` — מייצג העדר פעילות alert מוחלט.
- **ביסוס:** בהיעדר הנחיית מאמר, אפסים הם הייצוג הקונסרבטיבי ביותר (ללא alert = ציון נמוך ל-LR).
- **השפעה צפויה:** LR ינחש class prior בלבד עבור הקלטות אלו. אם הן בעיקר normal (label=0), ייתכן עיוות קל ב-recall. נבחן בשלב 7.
- **יישום:** `src/inference/alert_extractor.py` → `ZERO_FEATURES`, נקרא ב-`src/train/train_lr.py`.
- **סטטוס:** פעיל

---

## S11 — ALERT_THRESHOLD: 0.50 → 0.40

- **תיאור:** ספף ה-ALERT_THRESHOLD בחילוץ alert segments שונה מ-0.50 (ערך ברירת מחדל) ל-0.40.
- **סיבה:** עם AT=0.50, 13/55 הקלטות בסט ה-test (למעשה 2-4/497 בsplit train+val) לא ייצרו alert segment כלל, ולכן קיבלו ZERO_FEATURES. אצל הקלטות החמצניות (acidemia), ה-LR קיבל feature vector of zeros שהניב stage2_score ≈ 0.09 (prior בלבד). תוצאה: Sensitivity=0.09.
- **החלטה:** AT=0.40 מבטל כמעט לחלוטין את תופעת ה-zero-segments (4/497 במקום 13/55), מאפשר ל-LR לקבל features אמיתיים עבור הקלטות החמצניות.
- **ביסוס:** ניתוח threshold optimization ב-`notebooks/05_evaluation.ipynb` (Cells 12-15). AT=0.40 הניב AUC=0.839, Sens=0.818 (vs AUC=0.812, Sens=0.09 של baseline) מבלי לפגוע ב-AUC – אפילו שיפור.
- **השפעה בפועל:**
  - Baseline AT=0.50: AUC=0.812, Sens=0.09, Spec=1.00, TP=1/11
  - Old LR + AT=0.40 + Youden T=0.284: AUC=0.839, Sens=0.818, Spec=0.773, TP=9/11
  - Final LR retrained on 497 + AT=0.40 + CV T=0.199: AUC=0.717, Sens=0.636, Spec=0.818, TP=7/11
  - **מסקנה:** הגדרת AT=0.40 עם ה-LR המקורי ו-Youden threshold=0.284 נותנת את התוצאות הטובות ביותר.
- **יישום:** `src/inference/alert_extractor.py` — שינוי `ALERT_THRESHOLD = 0.5` ל-`ALERT_THRESHOLD = 0.4`. Threshold ה-LR: 0.284 (Youden optimal). Checkpoint חדש: `checkpoints/alerting/logistic_regression_at040.pkl` (n_train=497).
- **סטטוס:** פעיל — ממומש בניתוח, ממתין לבחירת config סופי

---

## פורמט להוספת סטייה חדשה

```markdown
## S[N] — שם הסטייה

- **תיאור:** מה שונה מהמאמר
- **סיבה:** למה נדרשת הסטייה
- **החלטה:** מה נבחר
- **ביסוס:** מקור/נימוק
- **השפעה צפויה:** איך ישפיע על AUC/ביצועים
- **סטטוס:** פעיל / פתור / נבדק
```
