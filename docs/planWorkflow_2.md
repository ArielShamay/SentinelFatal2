# planWorkflow_2 — תוכנית ביצוע מחולקת לסוכנים (Colab Free + T4)

> תאריך: 25 פברואר 2026  
> SSOT לביצוע: `docs/plan_2.md`  
> קונטקסט: `docs/plan_2_context.md`  
> סביבת אימון: Google Colab Free (GPU T4, סשן עד ~12 שעות, ללא background execution)

> הבהרה: במסמך הזה `work_plan` מתייחס לתוכנית הביצוע המעודכנת שב-`docs/plan_2.md`.

---

## 0) עקרונות מחייבים

1. אסור לסטות מהלוגיקה של `plan_2.md` (כולל MVP, Gates, ו-DoD).
2. כל היפר-פרמטר נבחר רק על `ft_train/ft_val` (לא על test fold).
3. אם מתגלה בעיה בזמן ריצה: עוצרים, מתעדים, וחוזרים ל-SSOT לפני שינוי.
4. קודם משלימים MVP (Phase A), ורק אחר כך Extended (Phase B/C).
5. כל סוכן חייב להוציא ארטיפקטים ולוגים ברורים, ולא לעבוד "מהזיכרון".

---

## 1) סטטוס ZIP והעברת נתונים ל-Colab

### 1.1 תוצאה בפועל (נבדק)

- הקובץ `data_processed.zip` קיים בריפו המקומי.
- הקובץ קיים גם ב-`origin/master` (נתיב: `data_processed.zip`).
- בדיקת תוכן מקומית:  
  - `processed/ctu_uhb/*.npy` (552 + `.gitkeep`)  
  - `processed/fhrma/*.npy` (135 + `.gitkeep`)  
  - `processed/ctu_uhb_clinical_full.csv`

### 1.2 מדיניות פעולה

- ברירת מחדל: להשתמש ב-ZIP שכבר נמצא ב-GitHub כדי לחסוך זמן העלאה.
- אם בדיקה ב-Colab נכשלת (404/403/ZIP פגום): מעלים ZIP חדש לחשבון GitHub ואז מורידים מחדש.

### 1.3 כתובת הורדה מומלצת ל-Colab

```text
https://raw.githubusercontent.com/ArielShamay/SentinelFatal2/master/data_processed.zip
```

---

## 2) מגבלות Colab Free שיש לבנות סביבן

1. זמן סשן מוגבל (~12 שעות) → חובה `resume` ואחסון ביניים.
2. GPU לא מובטח תמיד → אם אין GPU/T4, לא מתחילים אימון כבד.
3. אין background execution → לא סוגרים חלון/חיבור באמצע.
4. אחסון `/content` זמני → שומרים תוצרים ל-Drive או דוחפים ל-GitHub בסוף כל בלוק.

---

## 3) חלוקת סוכנים (מבט על)

| Agent | אחריות עיקרית | Output מחייב |
|---|---|---|
| A0 | Orchestrator / SSOT Guard | `logs/e2e_cv_v2/run_manifest.md` |
| A1 | Data Pack & GitHub Transport | `logs/e2e_cv_v2/data_pack_check.md` |
| A2 | Notebook Builder | `notebooks/08_e2e_cv_v2.ipynb` |
| A3 | Colab Runtime Bootstrap | `logs/e2e_cv_v2/runtime_preflight.md` |
| A4 | Trainer Phase A (Session 1) | pretrain + config lock + fold0 artifacts |
| A5 | Trainer Phase A2 (Session 2) | G0 ablation + folds 1-4 + OOF merged |
| A6 | Runtime Monitor / Incident Response | `logs/e2e_cv_v2/incidents.md` |
| A7 | Evaluator & Reporter | `results/e2e_cv_v2/*` + final report |

---

## 4) סדר הרצה מחייב

1. A0 → A1 → A2 → A3
2. A4 (Session 1)
3. A5 + A6 (במקביל בזמן ריצות)
4. A7
5. A0 סוגר DoD-MVP

### 4.1 Gates מחייבים מתוך plan_2

| Gate | PASS | FAIL Trigger | פעולה |
|---|---|---|---|
| G0 | `|ΔAUC(shared-clean)| <= 0.01` על folds 0+1 | `|ΔAUC| > 0.01` | ממשיכים עם disclosure מחמיר |
| G1 | `val_mse < 0.015` וגם `probe_auc > 0.60` | MSE גבוה או probe חלש | תיקון pretrain לפני המשך |
| G2 | Best val AUC (441/56) `>= 0.70` | כל configs `< 0.65` | חזרה לבדיקת pretrain/config |
| G3 | ft_val AUC fold0 `>= 0.65` | fold0 ft_val `< 0.55` | עצירה ואבחון לפני folds נוספים |
| G4 | mean CV AUC `>= 0.70` ו-std `< 0.10` | mean נמוך/שונות גבוהה | בדיקת folds חריגים/rollback |
| G5 | LR AUC > transformer-only AUC | LR לא מוסיף ערך | rollback ל-feature/model פשוט יותר |

---

## 5) פרומפטים מוכנים לסוכנים

העתק-הדבק לכל סוכן את הבלוק שלו בדיוק כמו שהוא.

### 5.1 Prompt — A0 Orchestrator / SSOT Guard

```text
Role:
אתה סוכן מתזמר. המשימה שלך היא לוודא שכל ביצוע תואם ל-SSOT:
1) docs/plan_2.md
2) docs/plan_2_context.md

Goals:
- לבנות run manifest מסודר ולחלק עבודה לסוכנים A1..A7.
- לנעול מראש את גבולות Phase A (MVP) מול Phase B/C.
- למנוע סטיות מתודולוגיות (test-time tuning, stride mismatch, וכו').

Must Read:
- docs/plan_2.md
- docs/plan_2_context.md
- docs/colab-vscode-guide-hebrew.md

Tasks:
1. צור logs/e2e_cv_v2/run_manifest.md עם:
   - מטרות Primary endpoint
   - Gates G0..G5
   - רשימת ארטיפקטים חובה ל-DoD-MVP
2. קבע שאם זמן סשן < 90 דקות: fallback ל-Config A לפי plan_2.
3. פרסם הוראה שאסור לבחור hyperparams על test fold.
4. אשר ש-A6 מנטר טרמינל בזמן אמת בכל ריצת אימון.

Hard Stop Rules:
- אם סוכן מציע סטייה מה-SSOT בלי הצדקה כתובה → עצור והחזר לתיקון.

Deliverable:
- logs/e2e_cv_v2/run_manifest.md
```

### 5.2 Prompt — A1 Data Pack & GitHub Transport

```text
Role:
אתה סוכן דאטה ואחסון. המטרה: לוודא שקובץ ZIP ל-Colab זמין ותקין.

Primary Path:
- להשתמש ב-data_processed.zip שכבר קיים ב-GitHub אם נגיש ותקין.

Tasks:
1. ודא שהקובץ נגיש מהקישור:
   https://raw.githubusercontent.com/ArielShamay/SentinelFatal2/master/data_processed.zip
2. בדוק integrity אחרי הורדה (zip opens + expected folders/files):
   - processed/ctu_uhb (552 npy)
   - processed/fhrma (135 npy)
   - processed/ctu_uhb_clinical_full.csv
3. אם לא נגיש/פגום:
   - צור ZIP חדש מקומי מתוך data/processed
   - העלה ל-GitHub (או ל-Release אם גדול מדי)
   - עדכן URL במניפסט
4. תעד גודל קובץ + checksum + URL סופי.

Commands (example):
- PowerShell local zip:
  Compress-Archive -Path "data\\processed" -DestinationPath "data_processed.zip" -Force

Deliverable:
- logs/e2e_cv_v2/data_pack_check.md
  עם status: PASS/FAIL, URL סופי, checksum, ותוכן הארכיון.
```

### 5.3 Prompt — A2 Notebook Builder

```text
Role:
אתה סוכן שבונה notebook חדש ספציפי לאימון plan_2.

Output Required:
- notebooks/08_e2e_cv_v2.ipynb

Rules:
- notebook חדש בלבד (לא למחזר 07).
- תואם ל-plan_2 בלבד.
- כל תא חייב לוג ברור והדפסת סטטוס.

Mandatory Notebook Sections:
1. Runtime + seed setup
2. Clone/Pull repo
3. GPU preflight (torch cuda + nvidia-smi)
4. Download data_processed.zip from GitHub
5. Extract to data/processed + sanity counts (552/135)
6. Pretrain shared (או טעינת checkpoint קיים)
7. Config selection על 441/56 (A/B/C או fallback A לפי זמן)
8. Fold 0 pilot
9. G0 ablation (shared vs clean on folds 0+1)
10. Folds 1..4 עם resume
11. Merge OOF + bootstrap CI + final tables
12. Export artifacts (Drive/GitHub/download)

Enforcement:
- אסור metrics רשמיים עם stride=1.
- אסור threshold tuning על test fold.
- חובה guard clauses אם data counts לא תואמים.

Deliverable:
- notebooks/08_e2e_cv_v2.ipynb
```

### 5.4 Prompt — A3 Colab Runtime Bootstrap

```text
Role:
אתה סוכן הפעלה ראשונית ב-Colab Free.

Must Follow:
- docs/colab-vscode-guide-hebrew.md

Tasks:
1. התחבר ל-Colab kernel.
2. ודא GPU פעיל (עדיפות T4).
3. הפעל preflight:
   - torch.cuda.is_available() == True
   - זיכרון GPU מספיק
   - דיסק מספיק ל-unzip + checkpoints
4. הרץ clone/pull + pip install לפי requirements.
5. הורד ZIP מ-GitHub וחלץ לנתיב הנכון.
6. כתוב runtime_preflight.md עם כל הערכים.

If GPU missing:
- אל תתחיל אימון כבד.
- תבצע רק שלבי הכנה ותדווח BLOCKED.

Deliverable:
- logs/e2e_cv_v2/runtime_preflight.md
```

### 5.5 Prompt — A4 Trainer Phase A (Session 1)

```text
Role:
אתה סוכן אימון Session 1.

Scope (Phase A / MVP first):
1. Shared pretrain (או שימוש ב-best_pretrain קיים אם מאושר במניפסט)
2. Config lock on 441/56:
   - A/B/C comparison
   - או fallback ל-A אם time budget rule הופעל
3. Fold 0 pilot end-to-end

Time Rules:
- Pretrain hard limit: 60 min
- Config selection hard limit: 90 min
- Fold0 hard limit: 60 min

Must Track:
- val_mse, probe_auc, val_auc(EMA), grad_norm, lr schedule

Red Flags:
- epoch נסגר מהר מדי באופן לא סביר (למשל שניות בודדות עם דאטה מלא)
- val_auc < 0.55 אחרי 30 epochs
- loss=NaN/inf
- n_test fold קטן מדי/לא תואם

If Red Flag:
1. עצור ריצה
2. תעד אירוע ב-incidents.md
3. בדוק plan_2.md סעיפים 2/3/7/11 + plan_2_context.md
4. החל תיקון מינימלי לפי SSOT והמשך

Deliverables:
- checkpoints/e2e_cv_v2/shared_pretrain/*
- logs/e2e_cv_v2/config_selection_441_56.csv
- artifacts של fold0
```

### 5.6 Prompt — A5 Trainer Phase A2 (Session 2)

```text
Role:
אתה סוכן Session 2 להשלמת CV מלא.

Order (Mandatory):
1. הרץ G0 Ablation Gate על folds 0+1 (shared vs clean pretrain)
2. כתוב results/e2e_cv_v2/ablation_shared_vs_clean.csv
3. רק אז השלם folds 1..4
4. מיזוג OOF וחישוב CI

Must Enforce:
- feature extraction stride=24 (train/test זהה)
- no test-time tuning
- decision threshold ראשי רק מתוך ft_val

Resume:
- אם fold כבר הושלם → דלג אוטומטית והמשך fold הבא

If Session Time Low (<2h):
- לא מתחילים אבלציה/ניסוי חדש
- משלימים רק MVP

Deliverables:
- results/e2e_cv_v2/fold*_oof_scores.csv
- results/e2e_cv_v2/global_oof_predictions.csv
- results/e2e_cv_v2/e2e_cv_v2_per_fold.csv
```

### 5.7 Prompt — A6 Runtime Monitor / Incident Response

```text
Role:
אתה סוכן ניטור בזמן אמת של פלט טרמינל notebook.

What To Monitor Continuously:
- זמני epoch (קיצוניים לכאן/לכאן)
- שימוש GPU/VRAM
- loss/auc trends
- early stopping מוקדם מדי
- הודעות קראש: OOM, FileNotFound, CUDA error, kernel restart
- סימני overfitting: train עולה, val יורד לאורך זמן

Incident Protocol (mandatory):
1. PAUSE: לעצור את ההרצה הנוכחית.
2. LOG: לרשום timestamp + תא notebook + שגיאה.
3. MAP: לקשר את הבעיה לסעיף רלוונטי ב-plan_2.
4. DECIDE:
   - אם פתרון כבר מוגדר ב-SSOT -> ליישם.
   - אם לא מוגדר -> לא לאלתר; להסלים ל-A0 להחלטה.
5. RESUME: לחזור מה-checkpoint האחרון בלבד.

Deliverable:
- logs/e2e_cv_v2/incidents.md
```

### 5.8 Prompt — A7 Evaluator & Reporter

```text
Role:
אתה סוכן הערכה וסגירה.

Tasks:
1. חשב מדדי Stability Track (552 CV): AUC, CI, Sens/Spec/PPV/NPV.
2. חשב Reproducibility Track על 441/56/55.
3. ודא דיווח disclosure על transductive pretrain + תוצאת G0.
4. בדוק DoD-MVP לפי סעיף 15.1 ב-plan_2.
5. הפק דו"ח סופי מרוכז + טבלאות השוואה מול baseline/paper.

Must Output:
- results/e2e_cv_v2/e2e_cv_v2_final_report.csv
- results/e2e_cv_v2/e2e_cv_v2_per_fold.csv
- docs/plan_2_execution_report.md

Fail Condition:
- אם חסר אחד מארטיפקטי DoD-MVP -> הדוח מסומן INCOMPLETE.
```

---

## 6) צ'קליסט תפעולי ל-Notebook החדש (08_e2e_cv_v2.ipynb)

1. תא בדיקת GPU:
   - `torch.cuda.is_available()`
   - `!nvidia-smi`
2. תא הורדת ZIP מ-GitHub (או skip אם כבר קיים ב-`/content`).
3. תא אימות counts:
   - CTU-UHB == 552
   - FHRMA == 135
4. תא pretrain/logs/checkpoint.
5. תא config lock 441/56.
6. תא fold0 pilot.
7. תא G0 ablation.
8. תא loop folds 1..4 (resume-aware).
9. תא bootstrap + final tables.
10. תא export תוצרים (Drive/GitHub/Download).

---

## 7) Runbook קצר לתקלות "פיזיות" בזמן ריצה

### 7.1 אימון מסתיים מהר מדי

- חשד: דאטה לא נטען / דטלואדר ריק / split שגוי.
- פעולה:
  1. בדוק counts.
  2. בדוק מספר batches בפועל לכל epoch.
  3. בדוק נתיבי `data/processed`.

### 7.2 Overfitting ברור

- סימן: `train_loss` יורד ו-`val_auc` נשחק עקבית.
- פעולה:
  1. לעצור.
  2. לבדוק התאמה ל-config הנעול.
  3. ליישם רק פתרונות שמוגדרים ב-plan_2 (regularization/rollback/features gate).

### 7.3 GPU נעלם/Kernel reset

- פעולה:
  1. הרצה מחדש של bootstrap.
  2. resume מה-fold האחרון שהושלם.
  3. לא למחוק logs/checkpoints קיימים.

### 7.4 תוצאה חשודה (AUC לא הגיוני)

- פעולה:
  1. לא לפרסם כ-final.
  2. לבדוק n_test/n_pos בכל fold.
  3. לבדוק שלא בוצע tuning על test fold.

---

## 8) קריטריון סיום

Phase A נחשב "בוצע" רק אם:

1. כל סעיפי DoD-MVP ב-`plan_2.md` סעיף 15.1 מסומנים.
2. קיימים כל ארטיפקטי חובה תחת `checkpoints/e2e_cv_v2/`, `logs/e2e_cv_v2/`, `results/e2e_cv_v2/`.
3. `incidents.md` קיים (גם אם ריק עם הצהרת "No incidents").
4. יש דו"ח סופי עם שני מסלולים: Stability + Reproducibility.
