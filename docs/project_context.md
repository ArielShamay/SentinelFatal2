# project_context.md — מעקב ביצוע סוכנים

> **פרויקט:** SentinelFatal2 — Foundation Model לניבוי מצוקה עוברית מ-CTG  
> **מאמר:** arXiv:2601.06149v1  
> **מסמך SSOT:** `docs/work_plan.md`  
> **מסמך בעיות:** `docs/work_plan_issues_review_he.md`  
> **מסמך סטיות:** `docs/deviation_log.md`  
> **מסמך דאטה:** `docs/data_documentation_he.md`

---

## הוראות לסוכנים

כל סוכן שמבצע פרומפט מ-`agent_workflow.md` **חייב** לעדכן את הקטע שלו כאן:

1. **לפני תחילת עבודה:** סמן `סטטוס: בביצוע 🔄` וכתוב תאריך התחלה.
2. **בסיום:** סמן `סטטוס: הושלם ✅` ומלא את כל השדות.
3. **אם נמצאו בעיות:** תעד גם ב-`docs/work_plan_issues_review_he.md` בפורמט המתואר שם.
4. **אם נדרשה סטייה מהמאמר:** תעד ב-`docs/deviation_log.md` לפי הפורמט שם.

---

## סוכן 1 — חילוץ נתונים ועיבוד מקדים (שלבים 0+1)

| שדה | ערך |
|-----|-----|
| **סטטוס** | הושלם ✅ |
| **תאריך התחלה** | 22 פברואר 2026 |
| **תאריך סיום** | 22 פברואר 2026 |
| **קבצים שנוצרו** | O1.1: `data/raw/ctu_uhb/` (552 CSV), O1.2: `data/raw/fhrma/` (135 CSV), O1.3: `data/processed/ctu_uhb/` (552 .npy), O1.4: `data/processed/fhrma/` (135 .npy), O1.5: `data/processed/ctu_uhb_clinical_full.csv` (552 שורות), O1.6: `data/splits/train.csv` (441), O1.7: `data/splits/val.csv` (56), O1.8: `data/splits/test.csv` (55), O1.9: `data/splits/pretrain.csv` (687), O1.10: `src/data/preprocessing.py`, O1.11: `notebooks/00_data_prep.ipynb` |
| **קבצים ששונו** | `docs/deviation_log.md` (הוסף S8) |
| **בדיקות שעברו** | V1.1 ✅, V1.2 ✅, V1.3 ✅, V1.4 ✅, V1.5 ✅, V1.6 ✅, V1.7 ✅, V1.8 ✅, V1.9 ✅, G1.5 ✅ |
| **בעיות שנמצאו** | G1.2: anomaly שדווחה בתוכנית (982 שורות) לא אומתה — למעשה 981 שורות כצפוי. pH threshold boundary (deviation S8): 8 הקלטות עם pH=7.15 בדיוק מסומנות target=1 — תועד ב-deviation_log.md כS8 |
| **סטיות שתועדו** | S8 — pH ≤ 7.15 (inclusive) בפרויקט, בעוד שבמאמר מוגדר pH < 7.15. תועד ב-`docs/deviation_log.md`. |
| **הערות** | הכול רץ בלי שגיאות. שתי ארכיבות חולצו עם --strip-components=1. עמודות CSV: `fhr` ו-`uc` בשני ה-datasets. |

---

## סוכן 2 — ארכיטקטורת PatchTST (שלב 2)

| שדה | ערך |
|-----|-----|
| **סטטוס** | הושלם ✅ |
| **תאריך התחלה** | 22 פברואר 2026 |
| **תאריך סיום** | 22 פברואר 2026 |
| **קבצים שנוצרו** | O2.1: `config/train_config.yaml`, O2.2: `src/model/patchtst.py`, O2.3: `src/model/heads.py`, O2.4: `src/__init__.py` (קיים), O2.5: `src/model/__init__.py`, O2.6: `notebooks/01_arch_check.ipynb` |
| **קבצים ששונו** | `docs/deviation_log.md` (הוסף S9), `docs/work_plan_issues_review_he.md` (הוסף AGW-13) |
| **בדיקות שעברו** | V2.1 ✅, V2.2 ✅, V2.3 ✅, V2.4 ✅, V2.5 ✅, V2.6 ✅, V2.7 ✅, V2.8 ✅, G2.4 ✅, AGW-3 ✅ |
| **בעיות שנמצאו** | AGW-13: נוסחת n_patches ב-SSOT שגויה (74 במקום 73); תוקן ע"י חיתוך ל-1776 דגימות. תועד כ-S9 ב-deviation_log.md. |
| **סטיות שתועדו** | S9 — נוסחת n_patches (ראה deviation_log.md). max_group_size=6 — הנחה ללא מקור (מתועד ב-config). |
| **הערות** | backbone params: 413,056. Classification head מחזיר logits (softmax ב-training loop). Encoder: pre-norm BatchNorm1d, 3 layers, 4 heads, ffn=256, dropout=0.2. |
| **ביקורת עמיתים** | AGW-14 (קריטי) ✅, AGW-15 (בינוני) ✅, AGW-16 (בינוני) ✅ — כל ביקורות נסגרו. מחברת הורצה מחדש לחלוטין; כל V2.1–V2.8 עוברות עם outputs חיים. ראה `docs/work_plan_issues_review_he.md` לפרטי פתרון. |

---

## סוכן 3 — פייפליין פרה-טריינינג (שלב 3)

| שדה | ערך |
|-----|-----|
| **סטטוס** | הושלם ✅ |
| **תאריך התחלה** | 22 פברואר 2026 |
| **תאריך סיום** | 22 פברואר 2026 |
| **קבצים שנוצרו** | O3.1: `src/data/dataset.py` (PretrainDataset + build_pretrain_loaders), O3.2: `src/data/masking.py` (apply_masking + _random_partition), O3.3: `src/train/pretrain.py` (training loop + CLI), O3.4: `src/train/__init__.py`, O3.5: `notebooks/02_pretrain.ipynb` |
| **קבצים ששונו** | `docs/deviation_log.md` (תיעוד סטיית path), `docs/work_plan_issues_review_he.md` (הוסף Agent 3) |
| **בדיקות שעברו** | V3.1 ✅ (10,000 seeds), V3.2 ✅ (boundary+groups), V3.3 ✅ (loss on FHR only), V3.4 ✅ (forward→backward→step), V3.5 ✅ (checkpoint saved), V3.6 ✅ (val loss + early stopping), G3.5 ✅ (dry-run 2 batches CPU) |
| **בעיות שנמצאו** | AGW-17 path resolution (תוקן). ביקורת עמיתים: AGW-18 (Unicode prints), AGW-19 (UC לא תרם לשחזור FHR — קריטי), AGW-20 (cwd-dependent path) — כולם נסגרו, ראה work_plan_issues_review_he.md. |
| **סטיות שתועדו** | S4 (stride=900), S5 (epochs=200+patience=10), S6 (batch_size=64, gradient_clip=1.0) — כל אלו ידועות ומתועדות מקודם |
| **הערות** | Per-batch masking. UC fusion: element-wise addition `fhr_enc += uc_enc` (AGW-19 fix). Path: `Path(config_path).resolve().parent.parent` (AGW-20). 12,319 train + 1,368 val windows. Dry-run post-review: train~0.62, val~0.40. |

---

## סוכן 4 — פייפליין Fine-tuning (שלב 4)

| שדה | ערך |
|-----|-----|
| **סטטוס** | הושלם ✅ |
| **תאריך התחלה** | 22 פברואר 2026 |
| **תאריך סיום** | 22 פברואר 2026 |
| **קבצים שנוצרו** | O4.1: `src/data/dataset.py` (הוסף FinetuneDataset + build_finetune_loaders), O4.2: `src/train/finetune.py` (training loop + CLI + differential LR), O4.3: `src/train/utils.py` (compute_recording_auc + sliding_windows), O4.4: `notebooks/03_finetune.ipynb` |
| **קבצים ששונו** | `docs/work_plan_issues_review_he.md` (הוסף Agent 4 + AGW-21, AGW-22, AGW-23), `docs/project_context.md` (עדכון Agent 4), `notebooks/03_finetune.ipynb` (תיקון V4.5) |
| **בדיקות שעברו** | V4.1 ✅ (class_weights=[1.0, 3.9]), V4.2 ✅ (single batch training), V4.3 ✅ (AUC in [0,1]), V4.4 ✅ (max aggregation verified), V4.5 ✅ (NO test.csv — רפרודוצבילי), V4.6 ✅ (differential LR), V4.7 ✅ (gradient clipping), V4.8 ✅ (early stopping patience=15), G4.6 ✅ (dry-run 2 batches CPU) |
| **בעיות שנמצאו** | AGW-21: Pretrained checkpoint loading conflict (PretrainingHead vs ClassificationHead) — תוקן עם strict=False + filter. AGW-22: Type hint bug log_path (str vs Path) — תוקן עם Path() conversion. AGW-23 (ביקורת): V4.5 בדקה את הנוטבוק עצמו — תוקן על ידי הסרת הנוטבוק מרשימת קבצים לבדיקה (בודק רק src/). |
| **סטיות שתועדו** | S6.1 (class_weight נבחר על oversampling) — מתועד באופן רשמי ב-deviation_log.md |
| **הערות** | Class imbalance: weights=[1.0, 3.9]. Differential LR: backbone=1e-5, head=1e-4. AUC per-recording: max(window_scores). Dataset: 8109 train windows (441 recordings), 1037 val windows (56 recordings). Dry-run: train_loss=0.75, val_auc=0.697. Test data: אפס גישה (V4.5 verified רפרודוצבילי). Checkpoints: best_finetune.pt (1.8MB). |

---

## סוכן 5 — Inference ומנגנון Alerting (שלב 5)

| שדה | ערך |
|-----|-----|
| **סטטוס** | הושלם ✅ |
| **תאריך התחלה** | 22 פברואר 2026 |
| **תאריך סיום** | 22 פברואר 2026 |
| **קבצים שנוצרו** | O5.1: `src/inference/sliding_window.py`, O5.2: `src/inference/__init__.py`, O5.3: `src/inference/alert_extractor.py`, O5.4: `src/train/train_lr.py`, O5.5: `notebooks/04_inference_demo.ipynb`, checkpoint: `checkpoints/alerting/logistic_regression.pkl` |
| **קבצים ששונו** | `docs/project_context.md` (עדכון סוכן 5), `docs/work_plan_issues_review_he.md` (הוסף Agent 5) |
| **בדיקות שעברו** | V5.1 ✅, V5.2 ✅, V5.3 ✅, V5.4 ✅, V5.5 ✅, V5.6 ✅, G5.4 ✅ |
| **בעיות שנמצאו** | AGW-24 (בוטל): `--inference-stride` flag נוסף ואז הוסר — stride נעול ל-1 (INFERENCE_STRIDE_REPRO) ללא פשרות. AGW-25: `test.csv` בdocstring (false positive בlint) — תיעוד בלבד. AGW-26 ✅: תא V5.4 עודכן לבדיקת regex I/O במקום raw substring. AGW-27 ✅: pkl פגום נמחק; נוספה `validate_lr_checkpoint()` עם fail-fast guard; pkl רשמי יוצר ע"י Agent 6. |
| **סטיות שתועדו** | S10: recordings ללא alert segments מקבלות ZERO_FEATURES=[0,0,0,0]. מתועד ב-`deviation_log.md`. |
| **הערות** | Dry-run: 5 הקלטות, stride=1 (REPRO — מקסימום דיוק), אימות end-to-end. Official training (Agent 6): Colab GPU, stride=1, כל 441 הקלטות. Alert threshold=0.5 (LOCKED). 4 features: segment_length, max_prediction, cumulative_sum, weighted_integral. "test.csv" בdocstring של train_lr.py — תיעוד בלבד, אין גישה לנתוני test. |

---

## סוכן 6 — סביבת Colab והרצת אימון (שלב 6 + הרצות 3-4)

| שדה | ערך |
|-----|-----|
| **סטטוס** | טרם התחיל ⬜ |
| **תאריך התחלה** | — |
| **תאריך סיום** | — |
| **קבצים שנוצרו** | — |
| **קבצים ששונו** | — |
| **בדיקות שעברו** | — |
| **בעיות שנמצאו** | — |
| **סטיות שתועדו** | — |
| **הערות** | — |

---

## סוכן 7 — הערכה סופית (שלב 7)

| שדה | ערך |
|-----|-----|
| **סטטוס** | טרם התחיל ⬜ |
| **תאריך התחלה** | — |
| **תאריך סיום** | — |
| **קבצים שנוצרו** | — |
| **קבצים ששונו** | — |
| **בדיקות שעברו** | — |
| **בעיות שנמצאו** | — |
| **סטיות שתועדו** | — |
| **הערות** | — |

---

## סיכום כללי

| סוכן | שלבים | סטטוס |
|-------|-------|--------|
| 1 | 0+1 — חילוץ + עיבוד מקדים | ✅ |
| 2 | 2 — ארכיטקטורה | ✅ |
| 3 | 3 — פרה-טריינינג קוד | ✅ |
| 4 | 4 — Fine-tuning קוד | ✅ |
| 5 | 5 — Inference + Alerting | ✅ |
| 6 | 6 — Colab + הרצת אימון | ⬜ |
| 7 | 7 — הערכה סופית | ⬜ |
