# plan_2.md — תוכנית אימון מקיפה: SentinelFatal2 v2

> **תאריך:** 25 פברואר 2026  
> **מטרה:** להגיע לביצועים מקסימליים, ולהוכיח יציבות על מדגם גדול משמעותית מ-55 הקלטות  
> **סביבת אימון:** Google Colab — GPU T4 (15GB VRAM, סשן עד ~12 שעות)  
> **מסמך בסיס:** `docs/plan_2_context.md`

---

## 0. תקציר מנהלים

### הבעיה
המודל הנוכחי הגיע ל-AUC=0.812 (ו-0.839 עם אופטימיזציית threshold פוסט-הוק), אבל **נבדק רק על 55 הקלטות (11 acidemia)**. רווח הסמך הוא [0.630, 0.953] — רחב מדי כדי לסמוך על התוצאה. ה-CV היחיד שרץ (LR בלבד) נתן AUC=0.653, ו-E2E CV נכשל (Colab timeout אחרי fold 0).

### המטרה

**Primary endpoint (מחייב):**
- CV AUC ≥ 0.75 על 552 הקלטות עם CI width < 0.15
- Sensitivity ≥ 0.60 תחת Specificity ≥ 0.65 בנקודת החלטה

**Aspirational (לא תנאי PASS):**
- AUC ≥ 0.81 (שמירת baseline) — רצוי אך צפוי שה-CV יהיה נמוך יותר (approximately unbiased)
- AUC ≥ 0.85 (שיפור על baseline) — יתרון משמעותי אם מושג

**הקשר:**
- **להוכיח יציבות** — E2E Cross-Validation על **כל 552 ההקלטות** של CTU-UHB (113 acidemia), עם רווחי סמך צרים
- **להראות** שהמודל מגיע למקסימום הפוטנציאל שלו בכל שלב (pretrain, finetune, alerting)

### האסטרטגיה בשלושה משפטים
> **Pretrain חזק אחד** → **Finetune מותאם ל-fold** → **Alerting pipeline משופר** → **5 folds על 552 הקלטות** → **Bootstrap CI**

### MVP — גרעין מינימלי מחייב

גם אם Colab נופל או הזמן נגמר, **חייבים להשלים לפחות את זה:**

1. Shared pretrain (קיים או חדש) — checkpoint אחד
2. Finetune 5 folds עם **config יחיד קבוע מראש** (נבחר על split 441/56)
3. LR עם 6 features + StandardScaler
4. OOF predictions לכל 552 הקלטות + Bootstrap CI
5. טבלת per-fold metrics + comparison table

**כל מה שמעבר ל-MVP** (SWA, focal loss, 12 features, TTA, LightGBM, attention pooling) הוא **אופציונלי** ומסומן בתוכנית כ-Phase B/C.

---

## 1. אסטרטגיית ניצול נתונים

### 1.1 מאגר הנתונים המלא

| מקור | הקלטות | Labels | שימוש |
|------|--------|--------|-------|
| CTU-UHB | 552 | ✅ pH (113 acidemia) | pretrain + finetune + test (via CV) |
| FHRMA | 135 | ❌ ללא labels | pretrain בלבד |
| SPAM | 294 | ❌ חסר אותות | לא זמין |
| **סה"כ לפרה-טריינינג** | **687** | | |
| **סה"כ עם labels** | **552** | | |

### 1.2 שינוי מרכזי: שימוש בכל 552 ההקלטות ל-CV

במקום לשמור 55 הקלטות בצד כ-test קבוע (שנותנות CI רחב מדי), **נשתמש בכל 552 ההקלטות** בתוך Cross-Validation:

| שיטה | n לבדיקה | acidemia בבדיקה | אמינות |
|-------|---------|-----------------|--------|
| **Test קבוע (ישן)** | 55 | 11 | ❌ CI = [0.63, 0.95] |
| **5-fold CV על 552 (חדש)** | ~110 per fold × 5 = **552** | ~22 per fold | ✅ CI צפוי צר בהרבה |

**יתרונות:**
- כל הקלטה נבדקת **בדיוק פעם אחת** (OOF predictions)
- 552 ניבויים במקום 55 → Bootstrap CI צר פי ~3
- 113 acidemia cases נבדקים (במקום 11!) → אומדן Sensitivity אמין
- אין "בזבוז" של 55 הקלטות עם labels

### 1.3 חלוקת Folds

```
כל 552 CTU-UHB → Stratified 5-Fold (seed=42)
  
  Fold k (k = 0..4):
    ├── test_fold_k       : ~110 הקלטות (~22 acidemia)  ← OOF predictions
    ├── ft_train_fold_k   : ~353 הקלטות (~73 acidemia)  ← 80% מ-remaining
    ├── ft_val_fold_k     : ~89  הקלטות (~18 acidemia)  ← 20% מ-remaining
    └── pretrain (shared) : 135 FHRMA + כל 552 CTU-UHB  ← self-supervised
                            (transductive — ראה הערה ב-2.1)
```

**שיפור מול run_e2e_cv.py הקיים:** הסקריפט הנוכחי משתמש ב-497 (train+val) בלבד ולא כולל את 55 ה-test המקוריים. התוכנית החדשה משתמשת ב**כל 552** — תוספת של 55 הקלטות (11 acidemia נוספים!) לסך כל המדגם.

---

## 2. אסטרטגיית Pretrain — מיצוי מקסימום

### 2.1 אסטרטגיית Pretrain משותף (Shared Pretrain)

**הרציונל:** Pretrain הוא self-supervised (MSE על FHR) — אין שימוש ב-labels. ההדלפה היחידה האפשרית היא שהמודל "רואה" את צורת האות של הקלטות test. זו הדלפה **שולית** לעומת החיסכון האדיר בזמן:

| אסטרטגיה | עלות | הערה |
|-----------|------|------|
| Pretrain נפרד לכל fold | ~25 דקות × 5 = ~2 שעות | "נקי" מתודולוגית |
| **Pretrain משותף אחד** | **~25 דקות × 1** | חוסך 100 דקות, הדלפה שולית |

**ההחלטה:** נאמן **pretrain משותף אחד** על כל 687 ההקלטות, ונשתמש בו לכל 5 ה-folds.

> **🔴 Ablation חובה (לא אופציונלי):** יש להריץ לפחות 2 folds (folds 0+1) עם pretrain נקי שאינו כולל את ה-test fold המקביל, ולחשב ΔAUC = AUC_shared − AUC_clean. אם |ΔAUC| > 0.01 — יש לדווח שהתוצאה רגישה לטרנסדוקציה ולסמן זאת כמגבלה מרכזית בדוח. רק אם |ΔAUC| ≤ 0.01 ניתן לדווח שהדליפה שולית.

> **ℹ️ הערה מתודולוגית (transductive pretraining):** Shared pretrain כולל את אותות ה-CTU-UHB שישמשו כ-test fold, כך שהמודל "רואה" את צורת האות של הקלטות test בשלב ה-MAE. זה **לא** מהווה דליפת labels (ה-pretrain לא משתמש ב-pH כלל), אבל זה שימוש במידע לא-מתוייג מה-test. זה פרקטיקה מקובלת בספרות foundation model (ראה BERT, GPT, MAE המקורי). יש לציין זאת בדוח הסופי.

> **Policy (מחייב לדוח הסופי):** הדיווח יכלול שני מסלולים נפרדים:
> 1. **Stability Track** — CV על 552 לפי התוכנית (עם disclosure מלא על transductive pretrain).  
> 2. **Reproducibility Track** — הרצה על split ההיסטורי 441/56/55 ללא שום טיוב על test-55.  
> לא מציגים מספר יחיד כ"final truth" בלי שני המסלולים יחד.

### 2.2 שיפורי Pretrain

הבעיה: Pretrain הנוכחי עצר ב-epoch 13, best ב-epoch 2. **זה מצביע על כך שהמודל לא למד ייצוגים עמוקים מספיק.**

#### 2.2.1 Cosine Annealing במקום ReduceLROnPlateau

| פרמטר | ישן | חדש | סיבה |
|--------|-----|-----|------|
| scheduler | ReduceLROnPlateau | **CosineAnnealingWarmRestarts** | מאפשר יציאה מ-local minima |
| T_0 | — | **50 epochs** | מחזור ראשון |
| T_mult | — | **2** | מחזורים הולכים וגדלים (50, 100, 200) |
| η_min | 1e-6 | **1e-6** | לא לרדת לאפס |
| max_epochs | 300 | **300** | מספיק ל-~2.5 מחזורים |
| patience | 20 | **50** | נותן מספיק מרחב ל-scheduler לעבוד |

**למה?** ReduceLROnPlateau הוריד LR מהר מדי → המודל נתקע. Cosine annealing מאפשר "חימום מחדש" שעוזר לצאת ממינימומים מקומיים.

#### 2.2.2 Progressive Masking

במקום mask ratio קבוע של 40% מההתחלה:

```
Epochs 0-20:   mask_ratio = 0.20  (קל — המודל לומד patterns בסיסיים)
Epochs 20-50:  mask_ratio = 0.30  (בינוני)  
Epochs 50+:    mask_ratio = 0.40  (קשה — כמו במאמר)
```

**למה?** Curriculum learning — מתחיל קל ומקשה בהדרגה. מאפשר למודל לבנות ייצוגים בסיסיים לפני שדורשים ממנו לשחזר 40% מהאות.

#### 2.2.3 שמירת Checkpoint כל T_0/2 epochs

במקום לשמור רק best, נשמור checkpoints כל 25 epochs. זה מאפשר:
- לבחור את ה-checkpoint שנותן את ה-finetune AUC הטוב ביותר (pretrain-selection via downstream probe)
- Snapshot ensemble של pretrain checkpoints

#### 2.2.4 Pretrain Quality Probe

**אחרי** סיום pretrain, לפני שמתחילים finetune:

```python
# Linear probe: freeze encoder, train linear classifier
for param in model.parameters():
    param.requires_grad = False
    
probe = nn.Linear(18688, 2)  # frozen encoder → 2 classes
# Train for 10 epochs on train set
# If probe AUC > 0.60 → pretrain learned useful features
# If probe AUC < 0.55 → pretrain needs more epochs/different config
```

**מטרה:** וידוא שה-pretrain אכן למד ייצוגים שימושיים **לפני** שמשקיעים זמן ב-finetune.

### 2.3 ניטור Pretrain

| מטריקה | מה עוקבים | מטרה | פעולה אם חריגה |
|---------|----------|------|----------------|
| val_mse | MSE על פאצ'ים מוסכמים | יורד בהדרגה | אם עולה 10 epochs רצוף → בעיה |
| train_mse | MSE על train | יורד מונוטונית | אם plateau ארוך → LR נמוך מדי |
| lr | learning rate נוכחי | עוקב cosine schedule | וידוא schedule עובד |
| probe_auc | Linear probe AUC | > 0.60 | אם < 0.55: epoch לא מספיק, נמשיך |
| recon_viz | ויזואליזציית שחזור | FHR משוחזר דומה למקור | בדיקה ויזואלית כל 50 epochs |

---

## 3. אסטרטגיית Finetune — מיצוי מקסימום

### 3.1 שיפורי ארכיטקטורה

#### 3.1.1 Progressive Unfreezing (הפשרה הדרגתית)

במקום differential LR מההתחלה, נשתמש בהפשרה הדרגתית:

```
Phase 1 (epochs 0-5):   FREEZE backbone, train HEAD בלבד  (lr=1e-3)
Phase 2 (epochs 5-15):  UNFREEZE top-1 encoder layer      (lr_backbone=1e-5, lr_head=5e-4)
Phase 3 (epochs 15-30): UNFREEZE top-2 encoder layers      (lr_backbone=3e-5, lr_head=3e-4)
Phase 4 (epochs 30+):   UNFREEZE all (including embed)     (lr_backbone=5e-5, lr_head=1e-4)
```

**כלל מימוש מחייב:** ה-unfreeze חייב להיות **דינמי לפי `num_layers` בפועל** (ולא אינדקסים קשיחים).  
לדוגמה:
- Config A/B (3 layers): top-1=`encoder.2`, top-2=`encoder.2+encoder.1`
- Config C (4 layers): top-1=`encoder.3`, top-2=`encoder.3+encoder.2`

**למה?** שכבות עליונות הן ה"ספציפיות" ביותר ל-pretrain task. הפשרה מלמעלה למטה מאפשרת ל-head ללמוד בזמן שהמשקולות הטובות מ-pretrain נשמרות.

#### 3.1.2 Stochastic Weight Averaging (SWA)

```
Epochs 0-50:    אימון רגיל (SGD/AdamW)
Epochs 50-100:  SWA — מיצוע משקולות כל epoch
Output:         swa_model = average(epoch_50, epoch_51, ..., epoch_100)
```

**למה?** SWA ידוע בשיפור generalization ב-1-3% AUC בדאטהסטים קטנים. במקום לבחור checkpoint יחיד (שעלול להיות תנודתי), SWA ממצע על פני כל ה-"אגן" של ה-loss landscape.

**BatchNorm recalibration (חובה):** אחרי יצירת `swa_model` יש לבצע מעבר אחד על כל `ft_train` במצב `train()` ללא backward כדי לרענן running mean/var של BatchNorm.  
בלי השלב הזה, ביצועי SWA עלולים לרדת באופן מלאכותי.

```python
swa_model.train()
with torch.no_grad():
    for batch_x, _ in ft_train_loader:
        _ = swa_model(batch_x.to(device))
swa_model.eval()
```

### 3.2 שיפורי אימון

#### 3.2.1 Data Augmentation (חיזוק נתונים)

| Augmentation | פרטים | הסתברות | רציונל |
|-------------|-------|---------|--------|
| **Gaussian Noise** | σ=0.01 על FHR, σ=0.005 על UC | p=0.5 | סימולציית רעש חיישן |
| **Random Scaling** | FHR × uniform(0.95, 1.05) | p=0.3 | עמידות בפני כיול שונה |
| **Temporal Jitter** | הזזת חלון ±50 דגימות (~12s) | p=0.5 | עמידות בפני timing מדויק |
| **Channel Dropout** | אפס UC channel | p=0.1 | עמידות גם כשUC חסר/רועש |
| **Cutout** | אפס 48-96 דגימות רציפות (1-2 פאצ'ים) | p=0.2 | סימולציית signal dropout |
| **Mixup** | λ~Beta(0.2, 0.2) בין חלון normal ל-acidemia | p=0.15 | regularization + מעין oversampling |

**הערה חשובה:** Augmentations חלים **רק על train**, לא על val/test.

#### 3.2.2 Focal Loss במקום Weighted CrossEntropy

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=[1.0, 3.9], gamma=2.0):
        # alpha = class weights (same as now)
        # gamma = focusing parameter
        # When gamma > 0: easy examples contribute less to loss
        # Hard examples (misclassified) get more weight
```

**למה?** הדאטה מאוד לא מאוזן (80/20). Focal loss לא רק נותן משקל למחלקה הקטנה (כמו weighted CE), אלא גם **מתמקד בדוגמאות הקשות** — מה שעוזר במיוחד עם acidemia cases שקשה לזהות.

**נבדוק:** weighted CE (baseline) vs Focal Loss (γ=1) vs Focal Loss (γ=2). נבחר לפי val AUC.

**מדיניות תאימות עם Mixup (חובה):**
- ברירת מחדל בתוכנית: `mixup=off` כאשר `loss=focal`.
- אם מפעילים Mixup יחד עם Focal, חייבים מימוש soft-label focal תקין (אינטרפולציה של שני focal losses), ולא חישוב hard-label נאיבי.

#### 3.2.3 Label Smoothing

```python
# במקום target: 0, 1
# נשתמש ב: 0.05, 0.95
smoothing = 0.05
target_smooth = target * (1 - smoothing) + smoothing / n_classes
```

**למה?** מונע מהמודל להיות "בטוח ב-100%" — מפחית overfitting על דוגמאות בודדות.

#### 3.2.4 שיפור Validation — Recording-Level AUC

**פרוטוקול ראשי:** early stopping נשאר על EMA-smoothed val AUC **ללא TTA**, כדי לשמור על השוואה הוגנת מול ה-baseline.

**אופציונלי (Phase B) — Test-Time Augmentation (TTA) ב-validation:**

```python
def compute_val_auc_with_tta(model, val_recordings, n_tta=5):
    for recording in val_recordings:
        scores_all = []
        # Original inference
        scores_all.append(inference_recording(model, signal, stride=60))
        # TTA: noise variations
        for _ in range(n_tta):
            noisy = signal + np.random.normal(0, 0.005, signal.shape)
            scores_all.append(inference_recording(model, noisy, stride=60))
        # Average scores
        final_score = np.mean([max(s) for s in scores_all])
```

**שימוש:** רק בניתוח משני — **לא בתור פרוטוקול ה-early stopping הראשי**. מדווח בנפרד.

### 3.3 Hyperparameter Selection (לפני CV — על split מקורי)

> **עיקרון מתודולוגי:** בחירת config חייבת להתבצע **לפני ומחוץ ל-CV**, כדי שאומדן ה-CV יישאר approximately unbiased (ראה הערה bias שיורי למטה).

נריץ **3 תצורות מועמדות** על ה-split ההיסטורי 441/56 (train/val) בלבד:

| Config | d_model | layers | lr_backbone | loss | augment |
|--------|---------|--------|-------------|------|--------|
| **A (baseline+)** | 128 | 3 | 5e-5 | Weighted CE | ✅ |
| **B (focal)** | 128 | 3 | 5e-5 | Focal (γ=2) | ✅ |
| **C (bigger)** | 256 | 4 | 3e-5 | Focal (γ=2) | ✅ |

**שלב 1:** אימון כל 3 configs על train=441, val=56 (~30 דקות כל אחד)  
**שלב 2:** בוחר config שנתן **val AUC הכי גבוה על 56 הקלטות val**  
**שלב 3:** נועל config → רץ כל 5 folds של CV עם config קבוע זה

**למה לא על fold 0?** שימוש ב-fold 0 test לבחירה מזהם את אומדן ה-CV. ה-split 441/56 הוא חיצוני ל-CV ולכן בטוח.

> **⚠️ הערה מתודולוגית (bias שיורי):** בחירת config על 441/56 מתוך אותה אוכלוסיית 552 הקלטות יוצרת הטיה אופטימיסטית קלה — האומדן הסופי הוא **approximately unbiased** ולא unbiased מוחלט. יש לציין זאת בדוח הסופי. פתרון נקי יותר: nested CV מלא (יקר זמנית) או config שנקבע מראש לפי ספרות ללא כל טיוב על הנתונים הנוכחיים.

### 3.4 ניטור Finetune

| מטריקה | מטרה | פעולה אם חריגה |
|---------|------|----------------|
| train_loss | יורד בהדרגה | אם plateau → LR גבוה מדי או model capacity לא מספיק |
| val_auc (raw) | עולה עם תנודות | בודקים שהמגמה חיובית |
| val_auc (EMA, β=0.8) | עולה חלק | **Early stopping על EMA** |
| grad_norm (backbone) | > 0 | אם ≈ 0 → backbone קפוא, צריך LR גבוה יותר |
| grad_norm (head) | > 0 | אם >> backbone → head dominates |
| lr_backbone | עוקב schedule | — |
| lr_head | עוקב schedule | — |
| val_loss vs train_loss gap | < 0.3 | אם gap > 0.5 → overfitting → יותר regularization |

**Red flags שדורשים עצירה ושינוי:**
- val_auc < 0.55 אחרי 30 epochs → config B או C
- train_loss לא יורד → LR נמוך מדי
- val_auc יורד בעוד train_loss יורד → overfitting → הוסף dropout/weight decay

---

## 4. אסטרטגיית Alerting — מיצוי מקסימום

### 4.1 Feature Engineering מורחב

במקום 4 features (או 6 עם S14), נרחיב ל-**12 features**:

| # | Feature | מקור | סוג |
|---|---------|------|-----|
| 1 | segment_length | Longest segment | ⏱️ |
| 2 | max_prediction | Longest segment | 📊 |
| 3 | cumulative_sum | Longest segment | 📊 |
| 4 | weighted_integral | Longest segment | 📊 |
| 5 | n_alert_segments | כל ההקלטה | 📊 |
| 6 | alert_fraction | כל ההקלטה | 📊 |
| 7 | **mean_prediction** | **Longest segment** | 📊 ממוצע scores |
| 8 | **std_prediction** | **Longest segment** | 📊 שונות scores |
| 9 | **max_pred_all_segments** | **כל ה-segments** | 📊 max מכל segments |
| 10 | **total_alert_duration** | **כל ההקלטה** | ⏱️ סך כל זמן alert (דקות) |
| 11 | **recording_max_score** | **כל ההקלטה** | 📊 הציון המקסימלי (ללא threshold) |
| 12 | **recording_mean_above_th** | **כל ההקלטה** | 📊 ממוצע scores > threshold |

**חשוב:** עם 12 features ומדגם קטן (~350 train per fold), regularization הכרחי: 
- StandardScaler → PCA(n_components=0.95) → LogisticRegression(C=0.1)
- או: StandardScaler → LogisticRegression(C=0.01, penalty='elasticnet', l1_ratio=0.5)

> **Rollback criterion:** אם 12 features לא משפרים mean ft_val AUC ב-≥0.01 לעומת 6 features, חוזרים ל-6 features כברירת מחדל. הבדיקה מתבצעת על **ft_val של folds 0+1 בממוצע** (לא fold 0 בלבד, לא על test fold!) לפני שממשיכים ל-folds 2-4. תנאי נוסף: השיפור חייב להיות לא-שלילי בשניהם — אם אחד מה-folds מציג ירידה, חוזרים ל-6 features גם אם הממוצע חיובי.

### 4.2 Alert Threshold Sweep

במקום AT=0.40 קבוע, נבדוק:

```python
AT_candidates = [0.30, 0.35, 0.40, 0.45]
# עבור כל fold:
#   1. Extract features עם כל AT על ft_train
#   2. Train LR על ft_train
#   3. threshold ראשי = Sens-max תחת Spec≥0.65 על ft_val  (❗ לא על test fold!)
#      threshold משני = Youden על ft_val  (לדיווח בלבד)
# בחירת AT = הערך שנותן val AUC גבוה ביותר על ft_val
# אחרי בחירה: מריצים AT+threshold_ראשי על test fold (הערכה בלבד).
```

> **No test-time tuning policy:** כל היפר-פרמטרים (AT, decision threshold, LR C, מספר features) נבחרים **רק על ft_train/ft_val**. על test fold מריצים הערכה בלבד, ללא שום בחירה.

### 4.3 מודל Alerting חלופי: LightGBM

אם LR לא מספיק, ננסה **LightGBM** (gradient boosting):

```python
import lightgbm as lgb
model_gb = lgb.LGBMClassifier(
    n_estimators=50,          # מעט עצים - מדגם קטן
    max_depth=3,              # רדוד - נמנע overfitting
    learning_rate=0.05,
    min_child_samples=10,
    class_weight='balanced',
    random_state=42,
)
```

**למה?** LightGBM יכול ללמוד אינטראקציות בין features (למשל max_prediction × segment_length) ש-LR לא יכול.

**בחירה:** LR vs LightGBM — לפי val AUC ממוצע.

### 4.4 Inference Stride

| stride | חלונות/הקלטה | זמן/הקלטה (T4) | דיוק |
|--------|-------------|-----------------|------|
| 1 | ~17,400 | ~8 שניות | מקסימלי |
| 24 | ~725 | ~0.4 שניות | גבוה |
| 60 | ~290 | ~0.15 שניות | בינוני |

**החלטה:** 
- **Feature extraction (train):** stride=24 (איזון מהירות/דיוק — S14 כבר הוריד מ-60 ל-24)
- **Feature extraction (test):** stride=24 (חייב להיות **זהה** ל-train!)
- **No metric comparison on different stride:** stride=1 מותר רק לניתוח איכותני (למשל case studies), לא לדיווח metrics רשמיים ולא להשוואת מודלים.

---

## 5. Ensemble & Post-Processing

### 5.1 Fold Ensemble

אחרי 5-fold CV, יש לנו **5 מודלים מאומנים**. לצורך שימוש עתידי (inference על הקלטה חדשה):

```python
def ensemble_predict(recording, models_5fold, lr_models_5fold):
    scores = []
    for model, lr_model in zip(models_5fold, lr_models_5fold):
        window_scores = inference_recording(model, recording, stride=24)
        features = extract_features(window_scores)
        prob = lr_model.predict_proba(features)[:, 1]
        scores.append(prob)
    return np.mean(scores)  # ממוצע 5 מודלים
```

**למה?** Ensemble של 5 מודלים מ-folds שונים מפחית שונות ומשפר AUC ב-0.5-2%.

### 5.2 Threshold Calibration

לכל fold, ה-Youden threshold נקבע על **ft_val** (לא על train ולא על test):

```
Per-fold:
  1. Train LR on ft_train_fold_k
  2a. threshold_primary_k = argmax Sensitivity s.t. Specificity ≥ 0.65
                                on ft_val_fold_k  ← פרוטוקול ראשי (יעד קליני מוגדר)
  2b. threshold_youden_k  = Youden(LR predictions on ft_val_fold_k)  ← משני
  3. Apply threshold_primary_k to test_fold_k → OOF binary predictions
     (דווח גם Youden בנפרד להשוואה)
OOF evaluation: prediction_i = (score_i ≥ threshold_primary_fold_of_i)
```

> **יציבות threshold:** ~18 חיוביים ב-ft_val יוצר שונות גדולה ב-Youden. לכן הפרוטוקול הראשי מבוסס על יעד Specificity≥0.65 קבוע, שיציב יותר במדגם קטן. יש לדווח CI לסף עצמו (bootstrap על ft_val) ולא רק למדדי הביצוע.

**לא נבצע threshold optimization על ה-OOF predictions** — זה יהיה הוגן מתודולוגית.

---

## 6. Direct Recording-Level Approach (ניסיוני)

### 6.1 Attention Pooling — עקיפת שלב ה-LR

רעיון חדשני: במקום pipeline של 3 שלבים (NN → alerting → LR), ננסה **direct recording-level prediction**:

```python
class RecordingClassifier(nn.Module):
    def __init__(self, patchtst, d_model=128, n_patches=73):
        super().__init__()
        self.patchtst = patchtst  # encoder בלבד
        # Attention pooling over windows
        self.window_attention = nn.Sequential(
            nn.Linear(n_patches * d_model * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),  # attention score per window
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_patches * d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
    
    def forward(self, windows_batch):
        # windows_batch: (n_windows, 2, 1800)
        # 1. Encode each window
        window_reprs = []
        for w in windows_batch:
            repr = self.patchtst.encode(w.unsqueeze(0))  # (1, 18688)
            window_reprs.append(repr)
        W = torch.stack(window_reprs)  # (n_windows, 18688)
        
        # 2. Attention pooling
        attn_scores = self.window_attention(W)  # (n_windows, 1)
        attn_weights = F.softmax(attn_scores, dim=0)  # (n_windows, 1)
        recording_repr = (attn_weights * W).sum(dim=0)  # (18688,)
        
        # 3. Classify
        return self.classifier(recording_repr)  # (2,)
```

**יתרונות:**
- End-to-end diferenceable — אין צורך ב-LR נפרד
- Attention pooling לומד **אילו חלונות חשובים** (לא רק max)
- פוטנציאל לביצועים טובים יותר

**חסרונות:**
- דורש training ברמת הקלטה (batching מורכב — הקלטות באורכים שונים)
- מדגם קטן (~350 הקלטות train per fold)
- לא מתואר במאמר

**החלטה:** ננסה **רק אם** pipeline ה-3-שלבים לא מגיע ליעד. עדיפות ב' — ניסיוני.

---

## 7. תוכנית ביצוע — Colab Sessions

### 7.1 סשן 1: Pretrain + Config Selection (~3 שעות)

> **⏱️ Time Budget קשיח לסשן 1:**
> | שלב | הקצאה מקסימלית | Stop Rule |
> |-----|----------------|----------|
> | Pretrain | 60 דקות | אם > 60 דקות — שמור checkpoint נוכחי וקדם |
> | Config A+B+C | 90 דקות | אם > 90 דקות — בחר A (baseline+) ונעל מייד |
> | Fold 0 pipeline | 60 דקות | אם > 60 דקות — שמור ועבור לסשן 2 |
> | Ablation Gate | 45-60 דקות (בתחילת סשן 2) | אם לא הושלם — לא מתקדמים ל-folds 2-4 |
> | **Plots/Bootstrap** | **בסשן נפרד** | ❌ לא בסשן 1 — עומס I/O מיותר |
>
> **כלל ברזל:** אם נותרו < 90 דקות בסשן — מדלגים ישר ל-folding (MVP), ללא Config Selection.

```
┌─────────────────────────────────────────────────────┐
│  STEP 1: Pretrain (shared) — hard limit 60 דקות     │
│  ├── 687 recordings, cosine annealing, progressive  │
│  ├── max 300 epochs, patience 50                    │
│  ├── Save checkpoints every 25 epochs               │
│  └── Linear probe → verify AUC > 0.60              │
│                                                     │
│  STEP 2: Config Selection on 441/56 — hard limit 90 │
│  ├── Config A: baseline+ (weighted CE + augment)    │
│  ├── Config B: focal loss (γ=2)                     │
│  ├── Config C: d_model=256, 4 layers                │
│  └── Pick best val AUC on 56 validation recordings  │
│  ⚠️ Fallback: אם אין זמן — נעל Config A מייד       │
│                                                     │
│  STEP 3: Fold 0 full pipeline with best config      │
│  ├── Finetune → feature extraction → LR → OOF      │
│  └── Save all artifacts                             │
└─────────────────────────────────────────────────────┘
```

### 7.2 סשן 2: Folds 1-4 (~4-5 שעות)

```
┌─────────────────────────────────────────────────────┐
│  Start with Ablation Gate (mandatory):               │
│  ├── folds 0+1: shared-pretrain vs clean-pretrain   │
│  ├── compute |ΔAUC| and write ablation report        │
│  └── only then continue to folds 1-4                 │
│                                                     │
│  For fold in [1, 2, 3, 4]:                          │
│  ├── Finetune from shared pretrain    ~15-25 min    │
│  ├── SWA collection epochs            ~10 min       │
│  ├── Feature extraction (stride=24)   ~5 min        │
│  ├── LR fit + primary threshold       ~1 min        │
│  ├── OOF predictions on test fold     ~3 min        │
│  └── Save checkpoint + OOF CSV        ~instant      │
│                                                     │
│  After all folds:                                   │
│  ├── Stack 5 OOF CSVs → global AUC                 │
│  ├── Bootstrap CI (N=5,000)                         │
│  ├── Generate plots                                  │
│  └── Final report                                   │
└─────────────────────────────────────────────────────┘
```

### 7.2.1 Ablation Gate (mandatory לפני המשך folds)

ארטיפקט חובה: `results/e2e_cv_v2/ablation_shared_vs_clean.csv` עם:
- `fold_idx`
- `auc_shared`
- `auc_clean`
- `delta_auc = auc_shared - auc_clean`
- `decision` (`pass` / `warn`)

כלל הכרעה:
- `|delta_auc| <= 0.01`  → `pass` (ממשיכים).
- `|delta_auc| > 0.01`   → `warn`; ממשיכים רק עם disclosure מפורש בדוח הסופי ובתרשים התוצאות.

### 7.3 תוכנית גיבוי: Auto-Resume

הסקריפט **חייב** לתמוך ב-resume אוטומטי:

```python
# בתחילת כל fold:
if oof_csv.exists():
    log.info("Fold %d already done — skipping.", fold_idx)
    continue
if best_finetune_ckpt.exists():
    log.info("Fold %d finetune done — skipping to LR.", fold_idx)
    # skip directly to feature extraction
```

**למה?** Colab יכול להיפסק בכל רגע. Resume מאפשר להמשיך מאיפה שנעצרנו.

### 7.4 טיפול ב-Timeout של Colab

| בעיה | פתרון |
|------|-------|
| Session expires (12h) | Auto-resume: הסקריפט מזהה completed folds |
| GPU disconnects | Save checkpoint כל epoch, not just best |
| RAM overflow | `mmap_mode='r'` כבר מיושם, batch_size=32 |
| Disk space | Delete epoch checkpoints after best is saved (keep only best + last 3) |
| Scope creep | אבלציות לא נגמרות | **Stop rule: לא מתחילים אבלציה חדשה אם נותרו < 2 שעות בסשן. MVP קודם תמיד.** |

---

## 8. קוד — שינויים נדרשים

### 8.1 קבצים חדשים

| קובץ | תפקיד |
|------|--------|
| `scripts/run_e2e_cv_v2.py` | סקריפט CV חדש על 552 הקלטות |
| `src/train/augmentations.py` | Data augmentation functions |
| `src/train/focal_loss.py` | Focal Loss implementation |
| `src/train/swa.py` | SWA utilities |
| `notebooks/08_e2e_cv_v2.ipynb` | Colab notebook להרצה |

### 8.2 שינויים בקבצים קיימים

| קובץ | שינוי |
|------|-------|
| `src/train/pretrain.py` | הוספת cosine annealing + progressive masking |
| `src/train/finetune.py` | הוספת progressive unfreezing + SWA + TTA |
| `src/data/dataset.py` | הוספת augmentation pipeline ל-FinetuneDataset |
| `src/inference/alert_extractor.py` | הוספת 6 features חדשים (סה"כ 12) |
| `config/train_config.yaml` | פרמטרים חדשים (pretrain scheduler, augment flags) |

### 8.3 Config החדש (draft)

```yaml
# train_config_v2.yaml — additions/changes from v1

pretrain:
  scheduler: cosine_warm_restarts  # NEW
  T_0: 50                          # NEW — first cosine cycle
  T_mult: 2                        # NEW — cycle length multiplier
  progressive_masking: true        # NEW
  mask_schedule:                   # NEW
    - {until_epoch: 20, ratio: 0.20}
    - {until_epoch: 50, ratio: 0.30}
    - {until_epoch: 999, ratio: 0.40}
  patience: 50                     # CHANGED from 20
  checkpoint_every: 25             # NEW

finetune:
  progressive_unfreeze: true       # NEW
  unfreeze_schedule_mode: top_down_dynamic  # NEW (derived from num_layers)
  unfreeze_schedule_template:               # NEW
    - {epoch: 0,  layers: "head"}
    - {epoch: 5,  layers: "top_1_encoder_layer"}
    - {epoch: 15, layers: "top_2_encoder_layers"}
    - {epoch: 30, layers: "all"}
  lr_head_initial: 1.0e-3          # NEW — high LR when head-only
  swa_start_epoch: 50              # NEW
  swa_lr: 1.0e-5                   # NEW
  swa_bn_recalibration: true       # NEW (mandatory)
  loss: focal                      # NEW (or "ce" for baseline)
  focal_gamma: 2.0                 # NEW
  focal_softlabel_mode: two_term_interp  # NEW (required if mixup on)
  label_smoothing: 0.05            # NEW
  augmentation:                    # NEW
    gaussian_noise_std: 0.01
    random_scale_range: [0.95, 1.05]
    temporal_jitter: 50
    channel_dropout_prob: 0.1
    cutout_prob: 0.2
    cutout_max_len: 96
    mixup_alpha: 0.2
    mixup_prob: 0.15
    mixup_with_focal: false        # NEW default safety policy

alerting:
  threshold_candidates: [0.30, 0.35, 0.40, 0.45]  # NEW — sweep
  n_features: 12                   # NEW (was 4/6)
  lr_model: logistic_regression    # or lightgbm
  lr_C: 0.01                       # NEW — tighter regularization for 12 features
  lr_penalty: elasticnet           # NEW
  lr_l1_ratio: 0.5                 # NEW
  feature_extraction_stride: 24    # CHANGED from 60

cv:
  n_folds: 5
  use_all_552: true                # NEW
  shared_pretrain: true            # NEW
  n_bootstrap: 5000
```

---

## 9. מטריקות יעד ותוצאות צפויות

### 9.1 יעדים

| מטריקה | Baseline (55 הקלטות) | יעד מינימלי (CV 552) | יעד שאפתני |
|---------|---------------------|----------------------|------------|
| AUC | 0.812 | ≥ 0.75 (CV mean) | ≥ 0.82 |
| Sensitivity | 0.091 (AT=0.50) / 0.818 (optimized) | ≥ 0.60 | ≥ 0.75 |
| Specificity | 1.000 / 0.773 | ≥ 0.70 | ≥ 0.80 |
| AUC 95% CI width | 0.323 [0.63-0.95] | < 0.15 | < 0.10 |
| **PR-AUC** | — | ≥ 0.45 | ≥ 0.55 |
| **Brier Score** | — | < 0.20 | < 0.15 |

**מדדים נוספים חובה בנקודת החלטה (threshold ראשי קליני: Sens-max תחת Spec≥0.65):**
- Sensitivity / Specificity / PPV / NPV + 95% CI (bootstrap)
- ECE (Expected Calibration Error) + reliability diagram
- לדיווח משני בנוסף: אותם מדדים בנקודת Youden

**הערה חשובה:** CV AUC צפוי להיות **נמוך יותר** מ-test AUC של 0.812. זה **צפוי ונורמלי** — CV נותן אומדן approximately unbiased בעוד AUC על test set יחיד עשוי להיות אופטימיסטי. CV AUC של 0.75+ הוא תוצאה מצוינת.

### 9.2 קריטריונים להצלחה

| רמה | תנאי | פירוש |
|-----|-------|-------|
| 🟢 **הצלחה מלאה** | CV AUC ≥ 0.80, CI width < 0.12 | עולה על benchmark |
| 🟡 **הצלחה חלקית** | CV AUC ∈ [0.70, 0.80], CI width < 0.15 | תוצאה סבירה, יש מקום לשיפור |
| 🔴 **צריך לחשוב מחדש** | CV AUC < 0.70 | בעיה בארכיטקטורה או בנתונים |

### 9.3 מסלולי דיווח

**מסלול 1 (Reproducibility Track):** ריצת ה-pipeline המשופר על split המקורי 441/56/55 להשוואה ישירה ל-baseline (0.812) ולמאמר (0.826).  
**מסלול 2 (Stability Track):** E2E 5-fold CV על כל 552 ההקלטות — האומדן האמין יותר.

```
┌───────────────────────────────────────────────────────────────┐
│                    SentinelFatal2 Results                      │
├────────────────────┬──────┬─────────────┬────────────────────┤
│ Method             │ n    │ AUC         │ 95% CI             │
├────────────────────┼──────┼─────────────┼────────────────────┤
│ Paper (benchmark)  │ 55   │ 0.826       │ —                  │
│ Baseline (test)    │ 55   │ 0.812       │ [0.630, 0.953]     │
│ ⭐ Repro Track      │ 55   │ ?.???       │ [?.???, ?.???]     │
│ ⭐ E2E CV (new)     │ 552  │ ?.???       │ [?.???, ?.???]     │
│ ⭐ ΔAUC (CV-base)  │ 552  │ ?.???       │ [?.???, ?.???]     │
└────────────────────┴──────┴─────────────┴────────────────────┘
```

> **Repro Track הוא output חובה** — בלי ריצה על 441/56/55 נוקשה השוואה מול המאמר.

---

## 10. סיכום אסטרטגיות — Priority Matrix

| # | אסטרטגיה | השפעה צפויה | מאמץ | עדיפות |
|---|----------|-------------|------|--------|
| 1 | **552-CV** (במקום 55 test) | 🔥🔥🔥 CI צר × 3 | בינוני | **P0 — הכרחי** |
| 2 | **Shared pretrain חזק** (cosine + progressive mask) | 🔥🔥 ייצוגים טובים יותר | נמוך | **P0 — הכרחי** |
| 3 | **Data Augmentation** | 🔥🔥 regularization | נמוך | **P1 — חשוב** |
| 4 | **Progressive Unfreezing** | 🔥 שימור pretrain + adaptation | נמוך | **P1 — חשוב** |
| 5 | **SWA** | 🔥 generalization | נמוך | **P1 — חשוב** |
| 6 | **Focal Loss** | 🔥 handle imbalance better | נמוך | **P1 — חשוב** |
| 7 | **12 features + ElasticNet** | 🔥 richer LR input | בינוני | **P1 — חשוב** |
| 8 | **Config selection** (441/56 split) | 🔥 best hyperparams | בינוני | **P1 — חשוב** |
| 9 | **AT sweep** | 🔥 fine-tuned threshold | נמוך | **P2 — רצוי** |
| 10 | **LightGBM** | 🔥 non-linear relationships | נמוך | **P2 — רצוי** |
| 11 | **Linear probe** (pretrain check) | — quality gate | נמוך | **P2 — רצוי** |
| 12 | **TTA in val** | 🔥 less noisy AUC | בינוני | **P2 — רצוי** |
| 13 | **Fold Ensemble** | 🔥 final prediction boost | נמוך | **P3 — bonus** |
| 14 | **Direct Attention Pooling** | 🔥🔥? unproven | גבוה | **P3 — bonus / ניסיוני** |
| 15 | **Label Smoothing** | 🔥 regularization | נמוך | **P2 — רצוי** |

---

## 11. Decision Gates — שערי PASS/FAIL

לפני שממשיכים לשלב הבא, **חייבים** לעבור את השער:

| שער | תנאי PASS | תנאי FAIL | פעולה ב-FAIL |
|------|-----------|-----------|---------------|
| **G0: Transductive Ablation** | `|ΔAUC(shared-clean)| <= 0.01` על folds 0+1 | `|ΔAUC| > 0.01` | ממשיכים רק עם disclosure מחמיר בדוח (לא מוחקים את התוצאה) |
| **G1: Pretrain** | val MSE < 0.015 **וגם** linear probe AUC > 0.60 | MSE > 0.015 או probe AUC < 0.55 | שנה scheduler / הארך epochs / בדוק data pipeline |
| **G2: Config Selection** | Best val AUC (441/56) ≥ 0.70 | val AUC < 0.65 עבור כל 3 configs | חזור ל-G1, בדוק pretrain checkpoint אחר |
| **G3: Fold 0 Pilot** | **ft_val** AUC fold 0 ≥ 0.65 | ft_val AUC < 0.55 | עצור, אבחן בעיה לפני המשך ל-folds 1-4 |
| **G4: Full CV** | mean CV AUC ≥ 0.70, std < 0.10 | mean AUC < 0.65 או fold variance > 0.12 | בדוק folds חריגים, שקול config אחר |
| **G5: Alerting** | LR AUC > transformer-only AUC | LR AUC < transformer-only | בדוק features, rollback to 6 features |

> שער G3 הוא **checkpoint חובה**: אם fold 0 נכשל, אין טעם להריץ 4 folds נוספים.
> **חשוב:** G3 נבדק על ft_val בלבד (לא על test fold), כדי שה-test של fold 0 יישאר נקי לאומדן CV הסופי.
> **חשוב 2:** G0 הוא disclosure gate ולא hard-stop. אם יש פער גדול בין shared/clean, לא זורקים ריצה — מדווחים זאת באופן בולט ומפרשים בזהירות.

---

## 12. Risk Management

| סיכון | הסתברות | השפעה | מיטיגציה |
|-------|---------|-------|----------|
| Colab timeout | גבוהה | folds לא נגמרים | Auto-resume, save every epoch |
| RAM overflow | נמוכה | crash | mmap_mode='r', batch_size=32 |
| CV AUC << test AUC | בינונית | תוצאה "מאכזבת" | הסבר בדוח: CV = approximately unbiased |
| Config C (d_model=256) OOM | נמוכה | ≈1.6M params vs 413K | fallback to config A/B |
| Augmentation hurts | נמוכה | AUC drops | ablation: with/without augment on fold 0 |
| 12 features overfitting | בינונית | LR overfits | ElasticNet + PCA, **rollback to 6 if ΔAUC < 0.01** |

---

## 13. Multi-Seed Stability Check

**Primary run:** כל 5 folds עם seed=42 (זהה ל-baseline).

**Stability check (Phase B):** אם יש זמן, ריצת folds 0 ו-2 עם seeds נוספים (123, 3407). בדיקה על 2 folds מייצגים (אחד מהתחלה ואחד מהאמצע) נותנת אומדן יציבות טוב יותר מ-fold בודד.

| Seed | Folds | מטרה |
|------|-------|------|
| 42 | 0-4 (full CV) | **Primary — תוצאה ראשית** |
| 123 | 0 + 2 | Stability check |
| 3407 | 0 + 2 | Stability check |

**Criterion:** אם std(AUC) across 3 seeds > 0.03 על אותו fold → training unstable, investigate.
**דיווח:** std על AUC וגם על Sensitivity בנקודת החלטה.

---

## 14. ארטיפקטים סופיים

בסיום ההרצה, נצפה למצוא:

```
checkpoints/
  e2e_cv_v2/
    shared_pretrain/
      best_pretrain.pt             # shared pretrain checkpoint
      pretrain_probe_auc.txt       # linear probe AUC
    fold0/ ... fold4/
      finetune/
        best_finetune.pt           # best finetune per fold
        swa_finetune.pt            # SWA averaged model
      alerting/
        lr_model.pkl               # LR pipeline (scaler + model)

results/
  e2e_cv_v2/
    fold0_oof_scores.csv ... fold4_oof_scores.csv
    global_oof_predictions.csv     # all 552 OOF predictions
    e2e_cv_v2_final_report.csv     # AUC + CI
    e2e_cv_v2_per_fold.csv         # per-fold metrics

logs/
  e2e_cv_v2/
    shared_pretrain_loss.csv
    fold0_finetune.csv ... fold4_finetune.csv
    config_selection_441_56.csv    # Config A/B/C comparison on historical split

docs/images/
  e2e_cv_v2_roc.png               # per-fold ROC curves
  e2e_cv_v2_bootstrap.png         # AUC bootstrap distribution
  e2e_cv_v2_comparison.png        # comparison bar chart
```

---

## 15. Definition of Done

### 15.1 DoD-MVP (חובה לסגירת Phase A)

- [ ] Shared pretrain completes with val MSE < 0.015
- [ ] Linear probe AUC > 0.60
- [ ] Config locked for CV: A/B/C comparison completed **או** fallback רשמי ל-Config A הופעל לפי time-budget rule
- [ ] G0 ablation report נוצר: `results/e2e_cv_v2/ablation_shared_vs_clean.csv`
- [ ] All 5 folds complete with OOF predictions
- [ ] All 552 recordings have exactly 1 OOF prediction
- [ ] Global OOF AUC computed with 95% CI
- [ ] Bootstrap CI width < 0.15
- [ ] Per-fold metrics table generated
- [ ] Comparison table: paper vs our-test-55 vs our-CV-552
- [ ] Sensitivity/Specificity/PPV/NPV at decision threshold + 95% CI
- [ ] Reproducibility Track: pipeline run on 441/56/55 split complete
- [ ] Reproducibility requirements complete:
- [ ] `requirements.txt` עם גרסאות מדויקות של numpy/torch/sklearn/lightgbm
- [ ] Seeds מוגדרים ומתועדים: `random`, `numpy`, `torch`, `torch.cuda` (manual_seed_all)
- [ ] `torch.backends.cudnn.deterministic = True` מוגדר בכל script
- [ ] Split CSVs נשמרים עם SHA-256 hash ב-`logs/e2e_cv_v2/split_hashes.txt`
- [ ] Git commit hash מוטמע בכל output CSV (עמודת `git_commit`)
- [ ] Final report markdown generated

### 15.2 DoD-Extended (Phase B/C בלבד)

- [ ] Visualization: ROC curves + bootstrap + comparison bar chart
- [ ] PR-AUC + 95% CI computed
- [ ] Brier score + ECE + reliability diagram
- [ ] Calibration protocol complete:
- [ ] Brier score + ECE מחושבים על **raw probabilities** (ללא calibration)
- [ ] אם מבוצע Platt scaling / Isotonic regression — נעשה רק על ft_train/ft_val (inner) בכל fold
- [ ] מדווח raw וגם calibrated side-by-side בדוח; לא ניתן להציג רק calibrated
- [ ] All checkpoints saved for future ensemble/inference
- [ ] Failure analysis outputs complete:
- [ ] Top-5 False Negative cases (acidemia missed) — recording IDs + scores
- [ ] Top-5 False Positive cases — recording IDs + scores
- [ ] Per-fold AUC variance table (fold, AUC, n_acidemia, n_total)
- [ ] Subgroup analysis: short vs long recordings, low vs high signal quality (if metadata available)

---

## 16. סיכום — למה התוכנית הזו תעבוד

1. **יותר נתונים לבדיקה:** 552 במקום 55 → CI צר פי 3, ו-113 acidemia cases במקום 11
2. **Pretrain חזק יותר:** cosine annealing + progressive masking → ייצוגים עמוקים
3. **Finetune חכם יותר:** progressive unfreezing + SWA + augmentation + focal loss → generalization טובה
4. **Alerting מחדד:** 12 features + ElasticNet + AT sweep → ניצול מקסימלי של המידע
5. **מתודולוגיה approximately-unbiased:** E2E CV עם shared pretrain (transductive — ראה הערה ב-2.1 ו-ablation חובה), threshold ו-hyperparams מ-inner-val בלבד, config נבחר על split חיצוני ל-CV (עם bias שיורי קל — מצוין בדוח)
6. **עמידות ב-Colab:** auto-resume, save every epoch, shared pretrain
7. **בחירת config מושכלת:** על split 441/56 (חיצוני ל-CV), 3 תצורות מועמדות

> **שורה תחתונה:** גם אם ה-CV AUC יהיה 0.75 (פחות מ-0.812 של ה-test), זו תוצאה **חזקה ואמינה יותר** כי היא מבוססת על 552 הקלטות עם CI צר — וזה מה שחוקרים וגופי רגולציה רוצים לראות.
