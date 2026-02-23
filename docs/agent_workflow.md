# agent_workflow.md — פרומפטים לסוכנים

> **פרויקט:** SentinelFatal2 — Foundation Model לניבוי מצוקה עוברית מ-CTG  
> **גרסה:** 1.0  
> **תאריך:** 22 בפברואר 2026  
> **מסמך SSOT:** `docs/work_plan.md`  
> **מסמך מעקב:** `docs/project_context.md`  
> **מסמך בעיות:** `docs/work_plan_issues_review_he.md`  
> **מסמך סטיות:** `docs/deviation_log.md`

---

## הוראות כלליות לכל סוכן

1. **קרא את כל המסמכים ב-"Read First"** לפני כל פעולה. אם מסמך חסר — עצור ודווח למשתמש.
2. **כל החלטה טכנית חייבת הסמכה** מ-`docs/work_plan.md` או מהמאמר (`docs/2601.06149v1.pdf`). אם אין הסמכה — זוהי **הנחה**: תעד ב-`docs/deviation_log.md` **לפני** שאתה מיישם.
3. **אסור לגעת ב-Test set** עד שלב 7 (סוכן 7). כל גישה מוקדמת לנתוני test היא data leakage.
4. **חובת דיווח:**
   - עדכן את הקטע שלך ב-`docs/project_context.md` (סטטוס, קבצים שנוצרו, בדיקות, בעיות).
   - אם מצאת בעיה — פתח סעיף ב-`docs/work_plan_issues_review_he.md`.
   - אם יצרת הנחה חדשה — תעד ב-`docs/deviation_log.md`.
5. **אל תשנה קבצי מקור** (`CTGDL_norm_metadata.csv`, קבצי `.hea`, PDF) — קרא בלבד.
6. **Handoff:** אל תסמן "הושלם" עד שכל artifact ב-Outputs קיים, וכל Validation מסומן Pass עם הוכחה קצרה.
7. **שאל את המשתמש** כשנדרשת הרשאה (Colab, Google Drive, התקנת חבילות) — אל תנחש.

---

## תרשים תלויות בין סוכנים

```
Agent 1 (שלבים 0+1)
   ↓ data/raw/*, data/processed/*, data/splits/*
Agent 2 (שלב 2)
   ↓ src/model/*, config/train_config.yaml
Agent 3 (שלב 3) ← צריך Agents 1+2
   ↓ checkpoints/pretrain/best_pretrain.pt
Agent 4 (שלב 4) ← צריך Agent 3
   ↓ checkpoints/finetune/best_finetune.pt
Agent 5 (שלב 5) ← צריך Agent 4
   ↓ src/inference/*, checkpoints/alerting/*
Agent 6 (שלב 6) ← צריך Agents 1-5 (קוד מוכן)
   ↓ אימון בפועל על GPU, checkpoints סופיים
Agent 7 (שלב 7) ← צריך Agent 6 (checkpoints מאומנים)
   ↓ results/
Agent 8 (Threshold Optimization + CV)
   ↓ checkpoints/alerting/logistic_regression_at040.pkl, results/final_model_comparison.csv
```

---
---

## סוכן 1 — חילוץ נתונים ועיבוד מקדים (שלבים 0+1)

### תיאור המשימה

אתה סוכן AI שמבצע את השלבים הראשוניים בפרויקט SentinelFatal2 — Foundation Model לניבוי מצוקה עוברית מאותות CTG (Cardiotocography).  
המשימה שלך: לחלץ ארכיונים, לעבד אותות FHR ו-UC, ליצור קבצי splits, ולבנות מטה-דאטה קלינית מלאה.

### Read First (חובה)

| מסמך | מה לחפש |
|------|----------|
| `docs/work_plan.md` — חלק א (מלאי נתונים) | מיקומי קבצים, ספירות, מבנה CSVs |
| `docs/work_plan.md` — חלק ו, שלבים 0+1 | INPUT/ACTIONS/OUTPUT/VALIDATION מלאים |
| `docs/work_plan.md` — חלק ב (SSOT) | מטריצת מקורות סמכות |
| `docs/work_plan.md` — חלק ז (ממשל) | קבועים שאסור לשנות |
| `docs/work_plan.md` — חלק ח | מבנה תיקיות |
| `docs/data_documentation_he.md` | עמודות CSVs, מילון labels |
| `docs/2601.06149v1.pdf` — Section II-B | אלגוריתם preprocessing מקורי |
| `docs/deviation_log.md` | סטיות קיימות S1-S7, במיוחד S7 (נרמול FHR) |

### Pre-Run Gates

- [ ] **G1.1:** קיימים הקבצים `data/CTGDL/CTGDL_ctu_uhb_proc_csv.tar.gz` ו-`data/CTGDL/CTGDL_FHRMA_proc_csv.tar.gz`.
- [ ] **G1.2:** קיים `data/CTGDL/CTGDL_norm_metadata.csv` ומכיל 981 שורות.
- [ ] **G1.3:** קיימים 552 קבצי `.hea` בתיקייה `data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0/`.
- [ ] **G1.4:** תיקייה `data/processed/` לא קיימת (או ריקה) — לא לדרוס נתונים קיימים.

**אם Gate נכשל:** עצור ודווח למשתמש; אל תנחש מיקומים חלופיים.

### Inputs

| קובץ/תיקייה | תיאור |
|-------------|--------|
| `data/CTGDL/CTGDL_ctu_uhb_proc_csv.tar.gz` | ארכיון CTU-UHB processed — 552 CSVs |
| `data/CTGDL/CTGDL_FHRMA_proc_csv.tar.gz` | ארכיון FHRMA processed — 135 CSVs |
| `data/CTGDL/CTGDL_norm_metadata.csv` | מטה-דאטה מרכזי: `pid, id, sig_len, sig_min, fname, stage2_idx, stage2min, dataset, target, test` |
| `data/ctu-chb-…/*.hea` | 552 קבצי header — pH, delivery_type, presentation, NoProgress ועוד |

### Actions

#### A1.0 — חילוץ ארכיונים (שלב 0)
1. צור תיקיות: `data/raw/ctu_uhb/`, `data/raw/fhrma/`, `data/processed/ctu_uhb/`, `data/processed/fhrma/`, `data/splits/`.
2. חלץ `CTGDL_ctu_uhb_proc_csv.tar.gz` → `data/raw/ctu_uhb/`:
   ```bash
   # ⚠ AGW-4 fix: הארכיונים מכילים תיקיית root פנימית — השתמש ב-strip-components
   mkdir -p data/raw/ctu_uhb
   tar -xzf data/CTGDL/CTGDL_ctu_uhb_proc_csv.tar.gz -C data/raw/ctu_uhb/ --strip-components=1
   ```
   שמות צפויים: `ctgdl_ctu_uhb_XXXX.csv`.
3. חלץ `CTGDL_FHRMA_proc_csv.tar.gz` → `data/raw/fhrma/`:
   ```bash
   mkdir -p data/raw/fhrma
   tar -xzf data/CTGDL/CTGDL_FHRMA_proc_csv.tar.gz -C data/raw/fhrma/ --strip-components=1
   ```
4. **בדוק מבנה פוסט-חילוץ:** אם ה-CSVs עדיין בתוך תת-תיקייה (כלומר `data/raw/ctu_uhb/<subfolder>/*.csv`), העבר אותם לרמה העליונה: `mv data/raw/ctu_uhb/*/*.csv data/raw/ctu_uhb/`.
5. בדוק: 552 CSV ב-`data/raw/ctu_uhb/`, 135 CSV ב-`data/raw/fhrma/`.
6. פתח 2-3 CSVs ובדוק שמות עמודות — חפש עמודות FHR ו-UC.

#### A1.1 — עיבוד FHR (מ-Section II-B במאמר)
לכל הקלטה:
```python
fhr = raw_fhr.copy()
fhr[fhr < 50] = np.nan        # הסר outliers נמוכים
fhr[fhr > 220] = np.nan       # הסר outliers גבוהים
fhr = pd.Series(fhr).interpolate(method='linear').values  # אינטרפולציה לינארית
fhr = np.clip(fhr, 50, 210)   # clip לטווח [50, 210]
fhr = (fhr - 50) / 160.0      # נרמול → [0, 1] (סטייה S7: פרשנות (FHR-50)/160)
```
> **שים לב S7:** הנרמול `(fhr-50)/160` הוא הפרשנות שנבחרה. ראה `docs/deviation_log.md` סטייה S7.

#### A1.2 — עיבוד UC (מ-Section II-B)
```python
window = 120  # 30 שניות × 4 Hz
rolling_std = pd.Series(uc_raw).rolling(window=window, center=True).std()
flat_mask = (rolling_std < 1e-5) & (uc_raw < 80) & (~np.isnan(uc_raw))
uc = uc_raw.copy()
uc[flat_mask] = np.nan         # סמן ארטיפקטים (אזורים שטוחים)
uc = np.clip(uc, 0, 100)      # clip לטווח [0, 100]
uc[~np.isnan(uc)] /= 100.0    # נרמול → [0, 1]
uc = np.nan_to_num(uc, nan=0.0)  # NaN → 0
```

#### A1.3 — שמירה בפורמט numpy
```python
# לכל הקלטה: shape (2, T) — [0]=FHR, [1]=UC
np.save(f'data/processed/ctu_uhb/{record_id}.npy', np.stack([fhr, uc]))
np.save(f'data/processed/fhrma/{record_id}.npy', np.stack([fhr, uc]))
```

#### A1.4 — יצירת קבצי Splits
הספליטים **כבר מוגדרים** בעמודת `test` ב-`CTGDL_norm_metadata.csv`. **אסור ליצור ספליטים חדשים.**
```python
meta = pd.read_csv('data/CTGDL/CTGDL_norm_metadata.csv')
ctg = meta[meta['dataset'] == 'ctg']
ctg[ctg['test'] == 0][['id', 'target', 'fname']].to_csv('data/splits/train.csv', index=False)
ctg[ctg['test'] == 1][['id', 'target', 'fname']].to_csv('data/splits/val.csv', index=False)
ctg[ctg['test'] == 2][['id', 'target', 'fname']].to_csv('data/splits/test.csv', index=False)

pretrain = meta[meta['dataset'].isin(['ctg', 'fhrma'])][['id', 'dataset', 'fname']]
pretrain.to_csv('data/splits/pretrain.csv', index=False)
```

#### A1.5 — בניית clinical metadata מ-552 קבצי `.hea` (P1 fix)
> **קריטי:** `metadata_summary.csv` מכיל רק 100 שורות — **אסור להשתמש בו!** בנה מ-`.hea` בלבד.

```python
import re, os, csv
HEA_DIR = 'data/ctu-chb-intrapartum-cardiotocography-database-1.0.0/ctu-chb-intrapartum-cardiotocography-database-1.0.0/'
FIELDS = {
    'pH':            r'#pH\s+([\d.]+)',
    'BDecf':         r'#BDecf\s+([\d.]+)',
    'Apgar1':        r'#Apgar1\s+(\d+)',
    'Apgar5':        r'#Apgar5\s+(\d+)',
    'gest_weeks':    r'#Gest\. weeks\s+(\d+)',
    'weight_g':      r'#Weight\(g\)\s+(\d+)',
    'presentation':  r'#Presentation\s+(\d+)',
    'induced':       r'#Induced\s+(\d+)',
    'stage1_min':    r'#I\.stage\s+(\d+)',
    'NoProgress':    r'#NoProgress\s+(\d+)',
    'stage2_min':    r'#II\.stage\s+(\d+)',
    'delivery_type': r'#Deliv\. type\s+(\d+)',
    'pos_stage2':    r'#Pos\. II\.st\.\s+(\d+)',
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

pd.DataFrame(rows).to_csv('data/processed/ctu_uhb_clinical_full.csv', index=False)
assert len(rows) == 552, f"Expected 552, got {len(rows)}"
```

**שדות קריטיים לשלב 7 (תתי-קבוצות Table 3):**

| עמודה | ערכים | שימוש |
|-------|-------|-------|
| `delivery_type` | 1=ווגינלי, 2=קיסרי | סינון לידות וגינליות |
| `presentation` | 1=ראש (cephalic), 2=עכוז | סינון מצגת ראש |
| `NoProgress` | 0=ללא עצירה, 1=יש עצירת לידה | סינון ללא עצירת לידה |

#### A1.6 — אימות labels מול pH
```python
clinical = pd.read_csv('data/processed/ctu_uhb_clinical_full.csv')
clinical['pH'] = pd.to_numeric(clinical['pH'], errors='coerce')
clinical['record_id'] = clinical['record_id'].astype(str).str.strip()
ctg_meta = meta[meta['dataset'] == 'ctg'].copy()
ctg_meta['id'] = ctg_meta['id'].astype(str).str.strip()

# AGW-5 fix: אכיפת cardinality מלאה לפני בדיקת mismatches
merged = ctg_meta.merge(clinical[['record_id', 'pH']], left_on='id', right_on='record_id')
assert len(merged) == 552, f"Merge returned {len(merged)} rows, expected 552 — check id types!"
assert merged['record_id'].nunique() == 552, "Duplicate record_ids in merge!"
assert merged['pH'].notna().all(), f"Found {merged['pH'].isna().sum()} null pH values!"

mismatches = merged[merged['target'] != (merged['pH'] <= 7.15).astype(int)]
assert len(mismatches) == 0, f"pH-label mismatch in {len(mismatches)} records!"
```

### Outputs (Artifacts חובה)

| # | קובץ | תיאור | בדיקה |
|---|-------|--------|-------|
| O1.1 | `data/raw/ctu_uhb/*.csv` | 552 CSVs | `len(ls) == 552` |
| O1.2 | `data/raw/fhrma/*.csv` | 135 CSVs | `len(ls) == 135` |
| O1.3 | `data/processed/ctu_uhb/*.npy` | 552 numpy (2, T) | `shape[0] == 2`, values in [0,1] |
| O1.4 | `data/processed/fhrma/*.npy` | 135 numpy (2, T) | `shape[0] == 2`, values in [0,1] |
| O1.5 | `data/processed/ctu_uhb_clinical_full.csv` | 552 שורות, כל השדות הקליניים | `len == 552`, `NoProgress` exists |
| O1.6 | `data/splits/train.csv` | 441 שורות | `len == 441` |
| O1.7 | `data/splits/val.csv` | 56 שורות | `len == 56` |
| O1.8 | `data/splits/test.csv` | 55 שורות | `len == 55` |
| O1.9 | `data/splits/pretrain.csv` | 687 שורות | `len == 687` |
| O1.10 | `src/data/preprocessing.py` | סקריפט preprocessing (A1.1-A1.3 כפונקציות) | importable |
| O1.11 | `notebooks/00_data_prep.ipynb` | Notebook לחילוץ + preprocessing + splits (להרצה ב-Colab) | רץ ללא שגיאות |

### Validation (Definition of Done)

- [ ] **V1.1:** 552 קבצי `.npy` ב-`data/processed/ctu_uhb/`, כולם shape `(2, T)`, ערכים [0,1].
- [ ] **V1.2:** 135 קבצי `.npy` ב-`data/processed/fhrma/`, כולם shape `(2, T)`, ערכים [0,1].
- [ ] **V1.3:** FHR: `min >= 0.0`, `max <= 1.0` (מנורמל ל-[0,1]). UC: `min >= 0.0`, `max <= 1.0`.
- [ ] **V1.4:** Acidemia train=90/441 (20.4%), val=12/56 (21.4%), test=11/55 (20.0%) — תואם Table 2 במאמר.
- [ ] **V1.5:** `ctu_uhb_clinical_full.csv` מכיל בדיוק 552 שורות עם עמודת `NoProgress`.
- [ ] **V1.6:** Labels מ-`CTGDL_norm_metadata` תואמים pH <= 7.15 מ-clinical_full — 0 mismatches (S8).
- [ ] **V1.7:** אין חפיפת IDs בין train/val/test: `set(train.id) & set(val.id) == empty`, וכן הלאה.
- [ ] **V1.8:** `pretrain.csv` מכיל 552 (ctg) + 135 (fhrma) = 687 שורות.
- [ ] **V1.9:** `notebooks/00_data_prep.ipynb` קיים, מכיל את שלבי חילוץ + preprocessing + splits, ורץ ללא שגיאות.

### Post-Run Gates

- [ ] **G1.5:** אף קובץ `.npy` אינו מכיל NaN (בדוק `np.isnan(arr).any()` ← False).
- [ ] **G1.6:** V1.1 עד V1.9 כולם Pass.
- [ ] **G1.7:** `docs/project_context.md` > "סוכן 1" עודכן עם כל הפרטים.
- [ ] **G1.8:** אם נמצאו בעיות — נפתח סעיף ב-`docs/work_plan_issues_review_he.md`.

### Handoff Contract → סוכן 2

סוכן 2 צריך:
- `data/processed/ctu_uhb/*.npy` (552) + `data/processed/fhrma/*.npy` (135)
- `data/splits/train.csv`, `val.csv`, `test.csv`, `pretrain.csv`
- `data/processed/ctu_uhb_clinical_full.csv` (552 שורות)
- `src/data/preprocessing.py` (פונקציות `preprocess_fhr`, `preprocess_uc`)

**אל תסמן Complete עד שכל artifact קיים וכל validation עבר.**

### Report Back

בסיום, עדכן את `docs/project_context.md` (קטע סוכן 1) עם:
- סטטוס: הושלם ✅
- תאריך סיום
- רשימת קבצים שנוצרו (O1.1-O1.11)
- רשימת validations שעברו (V1.1-V1.9)
- בעיות שנמצאו (אם יש)
- סטיות שתועדו (אם יש)

---
---

## סוכן 2 — ארכיטקטורת PatchTST (שלב 2)

### תיאור המשימה

אתה סוכן AI שמממש את ארכיטקטורת PatchTST Channel-Independent עם שתי heads (שחזור לפרה-טריינינג, סיווג ל-fine-tuning). עליך לבנות את המודל, קובץ config, ולהריץ sanity check מלא על ה-dimensions.

### Read First (חובה)

| מסמך | מה לחפש |
|------|----------|
| `docs/work_plan.md` — חלק ה (Hyperparameters Reference Card) | כל פרמטרי הארכיטקטורה |
| `docs/work_plan.md` — חלק ו, שלב 2 | INPUT/ACTIONS/OUTPUT/VALIDATION |
| `docs/work_plan.md` — חלק ז (ממשל, ז.1) | קבועים שאסור לשנות |
| `docs/2601.06149v1.pdf` — Section II-C, Equation 1, Figure 3 | מבנה PatchTST, patch embedding, channel independence |
| `docs/2601.06149v1.pdf` — Section II-D | Pretraining head description |
| `docs/2601.06149v1.pdf` — Section II-E | Classification head description |
| `docs/deviation_log.md` | סטייה S2 (d_model, num_layers, n_heads, ffn_dim — הנחות) |

### Pre-Run Gates

- [ ] **G2.1:** סוכן 1 סיים (בדוק `docs/project_context.md` > סוכן 1 > סטטוס: הושלם ✅).
- [ ] **G2.2:** קיים לפחות קובץ `.npy` אחד ב-`data/processed/ctu_uhb/` (לבדיקת shape).
- [ ] **G2.3:** PyTorch מותקן ו-importable.

### Inputs

| קובץ | תיאור |
|-------|--------|
| `docs/2601.06149v1.pdf` | מאמר — ארכיטקטורה |
| `docs/work_plan.md` | SSOT — כל הפרמטרים |
| `data/processed/ctu_uhb/*.npy` | אותות מעובדים (לבדיקת dimensions) |

### Actions

#### A2.1 — יצירת `config/train_config.yaml`

```yaml
# Hyperparameters Reference Card — מקופל מ-work_plan.md חלק ה
data:
  fs: 4                       # Hz — מאמר Section II-A
  window_len: 1800             # דגימות — מאמר Section II-C
  patch_len: 48                # מאמר Equation 1
  patch_stride: 24             # מאמר Equation 1
  n_patches: 73                # מחושב: (1800-48)/24 + 1
  n_channels: 2                # FHR + UC

model:
  d_model: 128                 # ⚠ הנחה S2
  num_layers: 3                # ⚠ הנחה S2
  n_heads: 4                   # ⚠ הנחה S2
  ffn_dim: 256                 # ⚠ הנחה S2
  dropout: 0.2                 # ✓ מאמר Section II-C
  norm_type: batch_norm        # ✓ מאמר Section II-C

pretrain:
  mask_ratio: 0.4              # ✓ מאמר Section II-D
  min_group_size: 2            # ✓ מאמר Section II-D
  max_group_size: 6            # ⚠ הנחה
  optimizer: adam
  lr: 1.0e-4                   # ✓ מאמר Section II-D
  max_epochs: 200              # ⚠ הנחה S5
  patience: 10
  batch_size: 64               # ⚠ הנחה S6
  window_stride: 900           # ⚠ הנחה S4

finetune:
  optimizer: adamw
  lr_backbone: 1.0e-5          # ⚠ הנחה S6 — differential LR
  lr_head: 1.0e-4              # ⚠ הנחה S6
  weight_decay: 1.0e-2         # ⚠ הנחה S6
  max_epochs: 100              # ✓ מאמר Section II-E
  patience: 15
  batch_size: 32               # ⚠ הנחה S6
  gradient_clip: 1.0           # ⚠ הנחה S6
  n_classes: 2

alerting:
  threshold: 0.5               # ✓ מאמר Section II-F
  inference_stride_repro: 1    # ✓ להערכה רשמית
  inference_stride_runtime: 60 # ⚠ הנחה S6 — הדמיה בלבד

seed: 42
```

#### A2.2 — יצירת `src/model/patchtst.py`

מבנה המודל (לפי Section II-C, Equation 1, Figure 3):

1. **PatchEmbedding:** Linear(patch_len=48, d_model=128) + learnable positional embedding (73 positions).
2. **TransformerEncoder:** 3 layers × (MultiHeadAttention(4 heads) + FFN(256) + BatchNorm + Dropout(0.2) + Residual).
3. **Channel-Independent Processing:** FHR ו-UC עוברים **את אותו encoder** (shared weights) בנפרד.
4. **forward() מחזיר:**
   - Per channel: `(batch, 73, 128)` — encoder output.
   - Concatenated+Flattened: `(batch, 18688)` — 73×128×2 (לclassification head).

```python
# Pseudocode structure:
class PatchTST(nn.Module):
    def __init__(self, config):
        self.patch_embed = PatchEmbedding(config)
        self.encoder = TransformerEncoder(config)
        self.head = None  # Set by replace_head()

    def encode_channel(self, x_channel):
        # x_channel: (batch, 1800)
        patches = unfold(x_channel, patch_len=48, stride=24)  # (batch, 73, 48)
        embedded = self.patch_embed(patches)  # (batch, 73, 128)
        encoded = self.encoder(embedded)  # (batch, 73, 128)
        return encoded

    def forward(self, x, mask_indices=None):
        # x: (batch, 2, 1800)
        # mask_indices: required when head is PretrainingHead (AGW-3 fix)
        fhr_enc = self.encode_channel(x[:, 0, :])  # (batch, 73, 128)
        uc_enc  = self.encode_channel(x[:, 1, :])  # (batch, 73, 128)

        if isinstance(self.head, PretrainingHead):
            assert mask_indices is not None, "mask_indices required for PretrainingHead"
            return self.head(fhr_enc, mask_indices)  # reconstruct FHR
        elif isinstance(self.head, ClassificationHead):
            concat = torch.cat([fhr_enc.flatten(1), uc_enc.flatten(1)], dim=1)
            # concat: (batch, 18688)
            return self.head(concat)

    def replace_head(self, new_head):
        self.head = new_head
```

> **⚠ הנחה S2:** `18688 = 73 × 128 × 2` תלוי ב-d_model=128. אם d_model שונה — adjust accordingly.

#### A2.3 — יצירת `src/model/heads.py`

```python
class PretrainingHead(nn.Module):
    """שחזור FHR patches מוסתרים.
    input: (batch, 73, 128) — FHR encoder output
    output: (batch, n_masked, 48) — reconstructed masked patches
    """
    def __init__(self, d_model=128, patch_len=48):
        self.proj = nn.Linear(d_model, patch_len)

    def forward(self, enc_output, mask_indices):
        masked_enc = enc_output[:, mask_indices, :]  # (batch, n_masked, 128)
        return self.proj(masked_enc)  # (batch, n_masked, 48)


class ClassificationHead(nn.Module):
    """Head לסיווג acidemia.
    Section II-E: "a linear layer mapping the flattened encoder output to two classes"
    input: (batch, 18688)
    output: (batch, 2)
    """
    def __init__(self, d_in=18688, n_classes=2, dropout=0.2):
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.linear(self.dropout(x))
```

#### A2.4 — sanity check notebook: `notebooks/01_arch_check.ipynb`

בדיקה מינימלית:
```python
import torch
from src.model.patchtst import PatchTST
from src.model.heads import PretrainingHead, ClassificationHead

config = load_config('config/train_config.yaml')
model = PatchTST(config)

# Test pretraining head
model.replace_head(PretrainingHead(d_model=128, patch_len=48))
dummy = torch.randn(4, 2, 1800)
mask_indices = [1, 2, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 35, 36,
                40, 41, 45, 46, 50, 51, 55, 56, 60, 61, 65, 66, 68]  # 29 indices
# AGW-3 fix: העבר mask_indices מפורשות דרך forward
output = model(dummy, mask_indices=mask_indices)
assert output.shape == (4, 29, 48), f"Pretrain head: {output.shape}"

# Test classification head
model.replace_head(ClassificationHead(d_in=18688, n_classes=2, dropout=0.2))
output = model(dummy)  # mask_indices=None → ברירת מחדל
assert output.shape == (4, 2), f"Classification head: {output.shape}"

print("✓ All dimension checks passed")
```

### Outputs (Artifacts חובה)

| # | קובץ | תיאור |
|---|-------|--------|
| O2.1 | `config/train_config.yaml` | כל hyperparameters — מסומנים ✓ מאמר / ⚠ הנחה |
| O2.2 | `src/model/patchtst.py` | מודל PatchTST מלא |
| O2.3 | `src/model/heads.py` | PretrainingHead + ClassificationHead |
| O2.4 | `src/__init__.py` | ריק (package init) |
| O2.5 | `src/model/__init__.py` | ריק (package init) |
| O2.6 | `notebooks/01_arch_check.ipynb` | sanity check dimensions |

### Validation (Definition of Done)

- [ ] **V2.1:** Input `(batch, 2, 1800)` → 73 patches per channel.
- [ ] **V2.2:** Encoder output shape: `(batch, 73, 128)` per channel.
- [ ] **V2.3:** Pre-training head: reconstructs `(batch, 29, 48)` values (29 = round(0.4×73) masked patches).
- [ ] **V2.4:** Classification head: outputs `(batch, 2)`.
- [ ] **V2.5:** FHR ו-UC encoder משתמשים **באותם משקלים** (shared backbone) — one encoder instance.
- [ ] **V2.6:** BatchNorm (לא LayerNorm) — מאמר Section II-C.
- [ ] **V2.7:** Dropout=0.2 בכל encoder layer ובhead.
- [ ] **V2.8:** Config YAML נטען ומכיל את כל הפרמטרים מחלק ה ב-work_plan.

### Post-Run Gates

- [ ] **G2.4:** `notebooks/01_arch_check.ipynb` רץ ללא שגיאות.
- [ ] **G2.5:** V2.1 עד V2.8 כולם Pass.
- [ ] **G2.6:** `docs/project_context.md` > "סוכן 2" עודכן.

### Handoff Contract → סוכן 3

סוכן 3 צריך:
- `src/model/patchtst.py` — מודל importable.
- `src/model/heads.py` — PretrainingHead importable.
- `config/train_config.yaml` — כל פרמטרי אימון.

### Report Back

עדכן `docs/project_context.md` (קטע סוכן 2) עם סטטוס, קבצים, validations, הערות.

---
---

## סוכן 3 — פייפליין פרה-טריינינג (שלב 3)

### תיאור המשימה

אתה סוכן AI שבונה את כל תשתית הפרה-טריינינג: Dataset/DataLoader, מנגנון masking (P6 fix), loss function, ו-training loop. **אתה כותב קוד בלבד — לא מאמן** (האימון בפועל מתבצע ע"י סוכן 6 ב-Colab).

### Read First (חובה)

| מסמך | מה לחפש |
|------|----------|
| `docs/work_plan.md` — חלק ו, שלב 3 | Masking algorithm (P6 fix v2), loss, training loop מלא |
| `docs/work_plan.md` — חלק ה, ה.4 | כל פרמטרי פרה-טריינינג |
| `docs/work_plan.md` — חלק ז (ממשל) | קבועים: mask_ratio=0.4, groups≥2, boundary preservation |
| `docs/2601.06149v1.pdf` — Section II-D, Equation 2, Figure 4 | Asymmetric masking, MSE loss |
| `docs/deviation_log.md` | סטיות S4 (stride=900), S5 (epochs=200) |
| `config/train_config.yaml` | פרמטרי אימון מעודכנים (מסוכן 2) |

### Pre-Run Gates

- [ ] **G3.1:** סוכן 2 סיים — `src/model/patchtst.py` קיים ו-importable.
- [ ] **G3.2:** סוכן 1 סיים — `data/processed/ctu_uhb/*.npy` (552) + `data/processed/fhrma/*.npy` (135) קיימים.
- [ ] **G3.3:** `data/splits/pretrain.csv` קיים עם 687 שורות.
- [ ] **G3.4:** `config/train_config.yaml` קיים.

### Inputs

| קובץ | תיאור |
|-------|--------|
| `data/processed/ctu_uhb/*.npy` | 552 אותות מעובדים (2, T) |
| `data/processed/fhrma/*.npy` | 135 אותות מעובדים (2, T) |
| `data/splits/pretrain.csv` | 687 IDs |
| `src/model/patchtst.py` | מודל PatchTST |
| `src/model/heads.py` | PretrainingHead |
| `config/train_config.yaml` | hyperparameters |

### Actions

#### A3.1 — Dataset & DataLoader (`src/data/dataset.py`)

```python
class PretrainDataset(torch.utils.data.Dataset):
    """
    טוען הקלטות מ-pretrain.csv, חותך sliding windows.
    stride = 900 דגימות (50% overlap) — הנחה S4.
    כל window: (2, 1800) — FHR + UC.
    """
    def __init__(self, pretrain_csv, processed_dirs, window_len=1800, stride=900):
        # ...load all recordings, compute all valid window start positions
        pass

    def __getitem__(self, idx):
        # return (2, 1800) tensor
        pass
```

#### A3.2 — Channel-Asymmetric Masking (`src/data/masking.py`)

> **P6 fix v2 — אלגוריתם דו-שלבי דטרמיניסטי.**  
> שלב א: פירוק `target_masked` (=29) לסכום קבוצות חוקיות (כל אחת ≥2, ≤6).  
> שלב ב: שיבוץ הקבוצות על אינדקסים חוקיים (1..n-2) ללא חפיפה.  
> אם שיבוץ נכשל — **retry מלא** (לא greedy המשכי).

```python
def _random_partition(total, min_size=2, max_size=6):
    """שלב א: פירוק total לרשימת קבוצות."""
    groups = []
    remaining = total
    while remaining > 0:
        if remaining < min_size:
            groups[-1] += remaining
            remaining = 0
        else:
            g = random.randint(min_size, min(max_size, remaining))
            if remaining - g == 1:
                g = remaining
            groups.append(g)
            remaining -= g
    return groups

def apply_masking(fhr_patches, mask_ratio=0.4, max_retries=100):
    """
    Contiguous group masking with boundary preservation.
    Guarantees: mask[0]=mask[-1]=False, all groups≥2, sum(mask)==target_masked.
    """
    n = len(fhr_patches)  # 73
    target_masked = round(mask_ratio * n)  # 29

    for attempt in range(max_retries):
        groups = _random_partition(target_masked, min_size=2, max_size=6)
        mask = np.zeros(n, dtype=bool)
        success = True
        random.shuffle(groups)
        for g_len in groups:
            valid_starts = [s for s in range(1, n - 1 - g_len + 1)
                          if not any(mask[s:s + g_len])]
            if not valid_starts:
                success = False
                break
            start = random.choice(valid_starts)
            mask[start:start + g_len] = True
        if success and mask.sum() == target_masked:
            break
    else:
        raise RuntimeError(f"Masking failed after {max_retries} retries")

    # Assertions
    assert not mask[0] and not mask[-1], "Boundary violation"
    assert mask.sum() == target_masked
    diffs = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    assert all((e - s) >= 2 for s, e in zip(starts, ends)), "Group < 2"

    fhr_patches[mask] = 0.0  # zero masking
    return fhr_patches, np.where(mask)[0]
```

**בדיקת יציבות חובה (לפני שממשיכים):**
```python
for seed in range(10_000):
    random.seed(seed); np.random.seed(seed)
    dummy = np.random.rand(73, 48)
    apply_masking(dummy.copy())
print("✓ 10,000 seeds passed — masking is stable")
```

#### A3.3 — Loss Function
```python
# Equation 2: MSE על masked FHR patches בלבד
# UC לא מחושב ב-loss כלל
loss = F.mse_loss(pred_fhr[mask_indices], original_fhr[mask_indices])
```

#### A3.4 — Training Script (`src/train/pretrain.py`)
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(max_epochs=200):
    model.train()
    for batch in dataloader:
        fhr_patches = extract_patches(batch[:, 0, :])  # (B, 73, 48)
        uc_patches = extract_patches(batch[:, 1, :])
        original_fhr = fhr_patches.clone()
        masked_fhr, mask_indices = apply_masking(fhr_patches)
        pred = model(batch_with_masked_fhr)
        loss = F.mse_loss(pred, original_fhr[:, mask_indices, :])
        loss.backward()
        optimizer.step(); optimizer.zero_grad()

    # Validation reconstruction loss
    val_loss = evaluate_reconstruction(model, val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, 'checkpoints/pretrain/best_pretrain.pt')
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= 10:
        break  # Early stopping

    save_checkpoint(model, f'checkpoints/pretrain/epoch_{epoch}.pt')
    log(epoch, train_loss, val_loss)
```

> **שים לב:** הסקריפט הזה ייכתב כקוד מוכן להרצה, אך **ההרצה בפועל** מתבצעת ע"י סוכן 6 ב-Colab עם GPU.

#### A3.5 — Notebook: `notebooks/02_pretrain.ipynb`
צור notebook שמריץ את `src/train/pretrain.py` עם:
- Seed setting (SEED=42)
- GPU check
- Drive mount for checkpoints
- Training loop invocation
- Loss curve plotting

### Outputs (Artifacts חובה)

| # | קובץ | תיאור |
|---|-------|--------|
| O3.1 | `src/data/dataset.py` | PretrainDataset + helpers |
| O3.2 | `src/data/masking.py` | `apply_masking()` + `_random_partition()` |
| O3.3 | `src/train/pretrain.py` | Training loop מוכן להרצה |
| O3.4 | `src/train/__init__.py` | package init |
| O3.5 | `notebooks/02_pretrain.ipynb` | Notebook להרצה ב-Colab |

### Validation (Definition of Done)

- [ ] **V3.1:** בדיקת יציבות masking: 10,000 seeds ללא כשל.
- [ ] **V3.2:** Masking: groups≥2, boundary preservation, ratio≈0.4.
- [ ] **V3.3:** Loss מחושב על masked FHR patches בלבד — UC לא נכנס ל-loss.
- [ ] **V3.4:** Training loop: forward → loss → backward → step — ללא שגיאות (על batch אחד, CPU).
- [ ] **V3.5:** Checkpoint נשמר בפורמט `torch.save()`.
- [ ] **V3.6:** val reconstruction loss מחושב ומושווה — early stopping עובד.

### Post-Run Gates

- [ ] **G3.5:** dry-run: 2-3 batches על CPU ללא שגיאות (loss מחזיר מספר סביר).
- [ ] **G3.6:** V3.1 עד V3.6 כולם Pass.
- [ ] **G3.7:** `docs/project_context.md` > "סוכן 3" עודכן.

### Handoff Contract → סוכן 4

סוכן 4 צריך:
- `src/model/patchtst.py`, `src/model/heads.py` (מסוכן 2)
- `src/data/dataset.py` (מסוכן 3 — ישתמש בחלקו ל-FinetuneDataset)
- `config/train_config.yaml`
- **לאחר הרצת סוכן 6:** `checkpoints/pretrain/best_pretrain.pt`

### Report Back

עדכן `docs/project_context.md` (קטע סוכן 3).

---
---

## סוכן 4 — פייפליין Fine-tuning (שלב 4)

### תיאור המשימה

אתה סוכן AI שבונה את פייפליין ה-fine-tuning לסיווג acidemia. עליך לממש: Dataset עם labels, טיפול ב-class imbalance (class_weight), training loop עם differential LR, ו-validation AUC per-recording (P7 fix). **קוד בלבד — הרצה ע"י סוכן 6.**

### Read First (חובה)

| מסמך | מה לחפש |
|------|----------|
| `docs/work_plan.md` — חלק ו, שלב 4 | Training loop, differential LR, class weights, AUC per-recording |
| `docs/work_plan.md` — חלק ה, ה.5 | כל פרמטרי fine-tuning |
| `docs/work_plan.md` — חלק ז, ז.2 | מניעת דליפת מידע — Test אסור! |
| `docs/2601.06149v1.pdf` — Section II-E | Fine-tuning protocol |
| `docs/deviation_log.md` | S6 (פרמטרים חסרים), S6.1 (class_weight נבחר) |
| `config/train_config.yaml` | פרמטרי fine-tuning |

### Pre-Run Gates

- [ ] **G4.1:** סוכנים 1-3 סיימו (בדוק `project_context.md`).
- [ ] **G4.2:** `data/splits/train.csv` (441), `data/splits/val.csv` (56) קיימים.
- [ ] **G4.3:** `src/model/patchtst.py` ו-`src/model/heads.py` importable.
- [ ] **G4.4:** `config/train_config.yaml` קיים.
- [ ] **G4.5:** **אין שימוש ב-test.csv** בשום שלב בקוד זה!

### Inputs

| קובץ | תיאור |
|-------|--------|
| `data/splits/train.csv` | 441 שורות: id, target, fname |
| `data/splits/val.csv` | 56 שורות: id, target, fname |
| `data/processed/ctu_uhb/*.npy` | אותות מעובדים (2, T) |
| `src/model/patchtst.py` | מודל PatchTST |
| `src/model/heads.py` | ClassificationHead |

### Actions

#### A4.1 — FinetuneDataset (`src/data/dataset.py` — הוסף class)

```python
class FinetuneDataset(torch.utils.data.Dataset):
    """
    טוען הקלטות CTU-UHB עם labels.
    כל הקלטה → sliding windows → windows עם label.
    Label: target=1 (acidemia, pH<=7.15), target=0 (normal).
    """
    def __init__(self, split_csv, processed_dir, window_len=1800, stride=900):
        pass  # Load recordings and labels from CSV

    def __getitem__(self, idx):
        # return (2, 1800) tensor, label (int)
        pass
```

#### A4.2 — Class Weight (P8 fix: class_weight נבחר)

```python
# P8: class_weight, לא oversampling!
# חישוב מ-Train בלבד:
train_df = pd.read_csv('data/splits/train.csv')
n_neg = (train_df['target'] == 0).sum()  # ~351
n_pos = (train_df['target'] == 1).sum()  # ~90
class_weights = torch.tensor([1.0, n_neg / n_pos])  # [1.0, ~3.9]
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

#### A4.3 — AUC per-Recording (P7 fix)

> **P7:** יחידת אימון = window. יחידת הערכה (AUC) = **recording**.  
> Aggregation נעולה: `max(window_scores)` per recording.

```python
def compute_recording_auc(model, split_csv, processed_dir, stride=1):
    """P7 fix: AUC per-recording, aggregation=max."""
    df = pd.read_csv(split_csv)
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        signal = np.load(f'{processed_dir}/{row.id}.npy')
        windows = sliding_windows(signal, window_len=1800, stride=stride)
        scores = []
        for win in windows:
            score = model(win.unsqueeze(0)).softmax(-1)[..., 1].item()
            scores.append(score)
        recording_score = max(scores)  # aggregation = max
        y_true.append(row.target)
        y_pred.append(recording_score)
    return roc_auc_score(y_true, y_pred)
```

#### A4.4 — Training Loop (`src/train/finetune.py`)

```python
# Load pretrained backbone
model = PatchTST(config)
model.load_state_dict(torch.load('checkpoints/pretrain/best_pretrain.pt'))
model.replace_head(ClassificationHead(d_in=18688, n_classes=2, dropout=0.2))

# Differential LR
optimizer = torch.optim.AdamW([
    {'params': model.backbone_params(), 'lr': 1e-5},   # backbone: עדין
    {'params': model.head.parameters(), 'lr': 1e-4},   # head: רגיל
], weight_decay=1e-2)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
best_val_auc = 0

for epoch in range(100):
    # Train: per window
    model.train()
    for batch_x, batch_y in train_loader:
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); optimizer.zero_grad()

    # Val: per RECORDING (P7)
    model.eval()
    val_auc = compute_recording_auc(model, 'data/splits/val.csv',
                                     'data/processed/ctu_uhb', stride=1)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'checkpoints/finetune/best_finetune.pt')
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= 15:
        break

    log(epoch, train_loss, val_auc)
```

#### A4.5 — Notebook: `notebooks/03_finetune.ipynb`

### Outputs (Artifacts חובה)

| # | קובץ | תיאור |
|---|-------|--------|
| O4.1 | `src/data/dataset.py` | FinetuneDataset class (הוסף לקובץ קיים) |
| O4.2 | `src/train/finetune.py` | Training loop מוכן |
| O4.3 | `src/train/utils.py` | `compute_recording_auc()` + helpers |
| O4.4 | `notebooks/03_finetune.ipynb` | Notebook להרצה |

### Validation (Definition of Done)

- [ ] **V4.1:** class_weights מחושבים מ-train.csv בלבד: ~[1.0, 3.9].
- [ ] **V4.2:** Training loop: forward → cross-entropy → backward → step — ללא שגיאות על batch אחד.
- [ ] **V4.3:** `compute_recording_auc()` מחזיר ערך בטווח [0,1] על validation set.
- [ ] **V4.4:** AUC מחושב per-recording (aggregation=max), **לא** per-window.
- [ ] **V4.5:** Test set לא מופיע בשום מקום בקוד (אין import של test.csv).
- [ ] **V4.6:** Differential LR: backbone=1e-5, head=1e-4.
- [ ] **V4.7:** gradient clipping max_norm=1.0 מופעל.
- [ ] **V4.8:** Early stopping on val AUC, patience=15.

### Post-Run Gates

- [ ] **G4.6:** dry-run: 1-2 batches → loss מספרי, ללא NaN/Inf.
- [ ] **G4.7:** V4.1 עד V4.8 כולם Pass.
- [ ] **G4.8:** `docs/project_context.md` > "סוכן 4" עודכן.
- [ ] **G4.9:** **לא נגעו בנתוני test** — בדוק grep על "test.csv" בכל הקוד שנוצר.

### Handoff Contract → סוכן 5

סוכן 5 צריך:
- `src/model/patchtst.py`, `src/model/heads.py`
- `src/train/finetune.py` (למבנה המודל + checkpoint loading)
- `src/train/utils.py` — `compute_recording_auc()`
- **לאחר הרצת סוכן 6:** `checkpoints/finetune/best_finetune.pt`

### Report Back

עדכן `docs/project_context.md` (קטע סוכן 4).

---
---

## סוכן 5 — Inference ומנגנון Alerting (שלב 5)

### תיאור המשימה

אתה סוכן AI שבונה את פייפליין ה-inference וה-alerting: sliding window inference, חילוץ קטעי-אזעקה, 4 features, ו-Logistic Regression (Stage 2). **קוד בלבד — הרצה ע"י סוכן 6.**

### Read First (חובה)

| מסמך | מה לחפש |
|------|----------|
| `docs/work_plan.md` — חלק ו, שלב 5 | Sliding window, alert extraction, 4 features, LR |
| `docs/work_plan.md` — חלק ה, ה.6 | כל פרמטרי Alerting |
| `docs/work_plan.md` — חלק ז (ממשל) | alert threshold=0.5, LR features=4 בדיוק |
| `docs/2601.06149v1.pdf` — Section II-F, Figure 5 | Alerting protocol מהמאמר |
| `docs/deviation_log.md` | סטיות רלוונטיות |

### Pre-Run Gates

- [ ] **G5.1:** סוכנים 1-4 סיימו (בדוק `project_context.md`).
- [ ] **G5.2:** `src/model/patchtst.py` ו-`src/model/heads.py` importable.
- [ ] **G5.3:** `config/train_config.yaml` קיים.

### Inputs

| קובץ | תיאור |
|-------|--------|
| `src/model/patchtst.py` | מודל |
| `data/splits/train.csv` | לאימון LR |
| `data/processed/ctu_uhb/*.npy` | אותות |

### Actions

#### A5.1 — Sliding Window Inference (`src/inference/sliding_window.py`)

> **P4 fix:** שני מצבים בלבד.

```python
INFERENCE_STRIDE_REPRO   = 1   # הערכה רשמית (שלב 7) — חובה
INFERENCE_STRIDE_RUNTIME = 60  # הדמיה בלבד

def inference_recording(model, signal, stride=INFERENCE_STRIDE_REPRO):
    """
    signal: (2, T)
    stride: REPRO=1 (רשמי), RUNTIME=60 (הדמיה)
    returns: list of (start_sample, score)
    """
    T = signal.shape[1]
    scores = []
    for start in range(0, T - 1800 + 1, stride):
        window = signal[:, start:start+1800]
        with torch.no_grad():
            score = model(window.unsqueeze(0)).softmax(-1)[..., 1].item()
        scores.append((start, score))
    return scores
```

#### A5.2 — Alert Segment Extraction (`src/inference/alert_extractor.py`)

```python
def extract_alert_segments(scores, threshold=0.5):
    """מחזיר רשימת segments רציפים שבהם score > 0.5."""
    alert_mask = [s > threshold for _, s in scores]
    segments = []
    # חפש רצועות רציפות של True
    # לכל segment: (start_sample, end_sample, segment_scores)
    return segments
```

#### A5.3 — Feature Extraction (4 features בדיוק — Section II-F)

> **P5 fix v2:** כל feature שמכיל זמן/אינטגרל מנורמל ל-stride.

```python
def compute_alert_features(segment_scores, inference_stride=1, fs=4):
    """
    4 features בדיוק מ-Section II-F.
    P5: dt = inference_stride / fs (שניות לצעד).
    """
    p = np.array(segment_scores)
    dt = inference_stride / fs

    return {
        'segment_length':   len(p) * dt / 60,           # דקות
        'max_prediction':   np.max(p),                    # ללא תלות ב-stride
        'cumulative_sum':   np.sum(p) * dt,              # אינטגרל
        'weighted_integral': np.sum((p - 0.5) ** 2) * dt  # אינטגרל
    }
```

#### A5.4 — Logistic Regression (Stage 2)

```python
from sklearn.linear_model import LogisticRegression
import joblib

# אמן LR על Train set בלבד (stride=REPRO)
X_train, y_train = compute_features_for_split('data/splits/train.csv', model, stride=1)
lr = LogisticRegression()
lr.fit(X_train, y_train)

# שמור stride כ-metadata
joblib.dump({'model': lr, 'stride': INFERENCE_STRIDE_REPRO},
            'checkpoints/alerting/logistic_regression.pkl')
```

> **כלל קריטי:** LR training ו-evaluation חייבים להשתמש **באותו stride!** שלב 7 = stride=1.

#### A5.5 — Notebook: `notebooks/04_inference_demo.ipynb`
הדמיה על הקלטה אחת — גרף ציון רציף + סימון קטעי אזעקה.

### Outputs (Artifacts חובה)

| # | קובץ | תיאור |
|---|-------|--------|
| O5.1 | `src/inference/sliding_window.py` | `inference_recording()` |
| O5.2 | `src/inference/__init__.py` | package init |
| O5.3 | `src/inference/alert_extractor.py` | `extract_alert_segments()` + `compute_alert_features()` |
| O5.4 | `src/train/train_lr.py` | Script לאימון LR על Train |
| O5.5 | `notebooks/04_inference_demo.ipynb` | הדמיה |

### Validation (Definition of Done)

- [ ] **V5.1:** `inference_recording()` מחזיר list of (start_sample, score), score ∈ [0,1].
- [ ] **V5.2:** `extract_alert_segments()` מחזיר segments עם threshold=0.5.
- [ ] **V5.3:** `compute_alert_features()` מחזיר dict עם בדיוק 4 keys.
- [ ] **V5.4:** LR אמון על Train בלבד — ללא val/test.
- [ ] **V5.5:** LR checkpoint שמור עם metadata של stride.
- [ ] **V5.6:** stride=REPRO (1) משמש להערכה רשמית.

### Post-Run Gates

- [ ] **G5.4:** dry-run: inference על הקלטה אחת → scores list, segments, features.
- [ ] **G5.5:** V5.1 עד V5.6 כולם Pass.
- [ ] **G5.6:** `docs/project_context.md` > "סוכן 5" עודכן.

### Handoff Contract → סוכן 6

סוכן 6 צריך את כל ה-artifacts מסוכנים 1-5:
- כל הקוד ב-`src/`
- כל ה-notebooks ב-`notebooks/`
- `config/train_config.yaml`
- `data/processed/`, `data/splits/`

### Report Back

עדכן `docs/project_context.md` (קטע סוכן 5).

---
---

## סוכן 6 — סביבת Colab והרצת אימון (שלב 6 + הרצות שלבים 3-5)

### תיאור המשימה

אתה סוכן AI שמכין את סביבת Google Colab, מעלה נתונים ל-Google Drive, ומריץ את כל האימונים בפועל (pretraining, fine-tuning, LR training). **זה השלב היחיד שדורש GPU.**

> **⚠ הרשאות נדרשות:** בתחילת העבודה, **שאל את המשתמש:**
> 1. האם יש גישה ל-Google Colab עם GPU?
> 2. האם תיקיית `SentinelFatal2` כבר קיימת ב-Google Drive?
> 3. האם ה-extension של Colab מותקן ב-VS Code?
> **אל תמשיך עד שתקבל אישור.**

### Read First (חובה)

| מסמך | מה לחפש |
|------|----------|
| `docs/work_plan.md` — חלק ו, שלב 6 | Colab setup, Drive mount, מבנה notebooks |
| `docs/work_plan.md` — חלק ו, שלבים 3+4+5 | סדר הרצה: pretrain → finetune → alerting/LR |
| `docs/colab-vscode-guide-hebrew.md` | חיבור Runtime, Mount Drive, Known Issues, מגבלות Colab |
| `config/train_config.yaml` | כל פרמטרי אימון |
| `docs/deviation_log.md` | סטיות שנפתחו עד כה |

### Pre-Run Gates

- [ ] **G6.1:** סוכנים 1-5 סיימו — כל ה-artifacts קיימים (בדוק `project_context.md`).
- [ ] **G6.2:** **המשתמש אישר** גישה ל-Colab + Google Drive.
- [ ] **G6.3:** `src/`, `config/`, `data/processed/`, `data/splits/`, `notebooks/` — כל התיקיות קיימות ומלאות.

### Inputs

| קובץ/תיקייה | תיאור |
|-------------|--------|
| כל ה-artifacts מסוכנים 1-5 | ראה Handoff Contracts |
| `docs/colab-vscode-guide-hebrew.md` | מדריך Colab+VS Code — חובה לקרוא לפני הגדרת סביבה |

### Actions

#### A6.1 — הגדרת Colab
1. התקן Google Colab extension ב-VS Code (v0.3.0+).
2. Activity Bar → Connect to Colab Runtime.
3. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. בדוק GPU: `torch.cuda.is_available()` → True (T4 לפחות).

#### A6.2 — העתקת קבצים ל-Drive
```
/content/drive/MyDrive/SentinelFatal2/
├── data/processed/
├── data/splits/
├── src/
├── config/
├── notebooks/
└── checkpoints/
    ├── pretrain/
    ├── finetune/
    └── alerting/
```

#### A6.3 — Reproducibility Header (ראש כל notebook)
```python
import torch, random, numpy as np
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
```

#### A6.4 — הרצת אימונים בסדר

| # | Notebook | שלב | מה לבדוק |
|---|----------|------|----------|
| 1 | `notebooks/02_pretrain.ipynb` | 3 | Loss יורד; checkpoint נשמר ל-Drive; early stopping | 
| 2 | `notebooks/03_finetune.ipynb` | 4 | Val AUC עולה; best_finetune.pt נשמר; test לא נגע |
| 3 | `notebooks/04_inference_demo.ipynb` | 5 | Inference עובד; LR אומן על train; checkpoint נשמר |

#### A6.5 — ניטור ובדיקות
- Loss: אם לא יורד ב-5 epochs ראשונים — עצור ובדוק preprocessing.
- Val AUC: צפוי 0.75-0.85 (תלוי ב-backbone quality).
- Checkpoint: ודא שנשמר ל-Drive (לא ל-`/content/` הזמני!).

### Outputs (Artifacts חובה)

| # | קובץ | תיאור |
|---|-------|--------|
| O6.1 | `checkpoints/pretrain/best_pretrain.pt` | Best pretrained backbone |
| O6.2 | `checkpoints/pretrain/epoch_*.pt` | Checkpoint per epoch |
| O6.3 | `checkpoints/finetune/best_finetune.pt` | Best fine-tuned model |
| O6.4 | `checkpoints/alerting/logistic_regression.pkl` | Trained LR model |
| O6.5 | `logs/pretrain_loss.csv` | Train/val loss per epoch |
| O6.6 | `logs/finetune_val_auc.csv` | Val AUC per epoch |

### Validation (Definition of Done)

- [ ] **V6.1:** GPU זמין (T4 לפחות).
- [ ] **V6.2:** Drive mount פעיל — checkpoints נשמרים ל-Drive.
- [ ] **V6.3:** Pretrain loss יורד לאורך epochs.
- [ ] **V6.4:** Fine-tune val AUC עולה.
- [ ] **V6.5:** `best_pretrain.pt` קיים ב-Drive.
- [ ] **V6.6:** `best_finetune.pt` קיים ב-Drive.
- [ ] **V6.7:** `logistic_regression.pkl` קיים.
- [ ] **V6.8:** Test set לא נגע בכל ההרצות.

### Post-Run Gates

- [ ] **G6.4:** כל checkpoints ב-Drive (לא רק ב-`/content/`).
- [ ] **G6.5:** V6.1 עד V6.8 כולם Pass.
- [ ] **G6.6:** `docs/project_context.md` > "סוכן 6" עודכן.
- [ ] **G6.7:** `logs/pretrain_loss.csv` ו-`logs/finetune_val_auc.csv` לא ריקים.

### Handoff Contract → סוכן 7

סוכן 7 צריך:
- `checkpoints/finetune/best_finetune.pt` — מודל מאומן סופי
- `checkpoints/alerting/logistic_regression.pkl` — LR מאומן
- `data/splits/test.csv` — 55 הקלטות test
- `data/processed/ctu_uhb/*.npy` — אותות
- `data/processed/ctu_uhb_clinical_full.csv` — מטה-דאטה קלינית (552 שורות)
- `src/inference/sliding_window.py` — inference function
- `src/inference/alert_extractor.py` — feature extraction + alert segments

### Report Back

עדכן `docs/project_context.md` (קטע סוכן 6) — כולל:
- משך אימון (שעות)
- מספר epochs שרצו (pretrain + finetune)
- best val reconstruction loss (pretrain)
- best val AUC (finetune)
- GPU type

---
---

## סוכן 7 — הערכה סופית (שלב 7)

### תיאור המשימה

אתה סוכן AI שמבצע את ההערכה הסופית על Test set — **פעם אחת בלבד**. עליך לחשב AUC לפי תתי-קבוצות (Table 3), לייצר עקומות ROC, ולכתוב דוח סופי.

> **⚠ כלל קריטי:** שלב זה מורץ **פעם אחת בלבד**, רק לאחר שכל hyperparameters נעולים ו-fine-tuning הסתיים. **אסור לחזור לשנות שום דבר לאחר ריצת שלב 7.**

### Read First (חובה)

| מסמך | מה לחפש |
|------|----------|
| `docs/work_plan.md` — חלק ו, שלב 7 | AUC תתי-קבוצות, Table 3 values, case studies |
| `docs/work_plan.md` — חלק ג.3 | קריטריוני הצלחה |
| `docs/work_plan.md` — חלק ז, ז.1 | קבועים: pH<=7.15, splits, alert threshold |
| `docs/work_plan.md` — חלק ז, ז.2 | מניעת דליפה: LR אומן על train בלבד |
| `docs/2601.06149v1.pdf` — Table 3, Figure 6 | Benchmark values, Case studies |
| `data/processed/ctu_uhb_clinical_full.csv` | שדות לסינון תתי-קבוצות |
| `docs/deviation_log.md` | סטיות שנפתחו — ולדווח עליהן |

### Pre-Run Gates

- [ ] **G7.1:** סוכן 6 סיים — checkpoints מאומנים קיימים.
- [ ] **G7.2:** `checkpoints/finetune/best_finetune.pt` קיים.
- [ ] **G7.3:** `checkpoints/alerting/logistic_regression.pkl` קיים.
- [ ] **G7.4:** `data/splits/test.csv` מכיל 55 שורות.
- [ ] **G7.5:** `data/processed/ctu_uhb_clinical_full.csv` מכיל 552 שורות.
- [ ] **G7.6:** **לא בוצעו שינויים** בקוד/נתונים/hyperparameters לאחר סוכן 6.

### Inputs

| קובץ | תיאור |
|-------|--------|
| `checkpoints/finetune/best_finetune.pt` | מודל מאומן |
| `checkpoints/alerting/logistic_regression.pkl` | LR מאומן (stride metadata בפנים) |
| `data/splits/test.csv` | 55 הקלטות test |
| `data/processed/ctu_uhb/*.npy` | אותות |
| `data/processed/ctu_uhb_clinical_full.csv` | מטה-דאטה קלינית |
| `src/inference/sliding_window.py` | `inference_recording()` |
| `src/inference/alert_extractor.py` | features + alert segments |

### Actions

#### A7.1 — AUC לפי תתי-קבוצות (Table 3)

```python
clinical = pd.read_csv('data/processed/ctu_uhb_clinical_full.csv')
test_ids = pd.read_csv('data/splits/test.csv')

subsets = {
    'All Test':                  test_ids.id.tolist(),
    'Vaginal':                   [id for id in test_ids.id if clinical_lookup(id, 'delivery_type') == 1],
    'Cephalic':                  [id for id in test_ids.id if clinical_lookup(id, 'presentation') == 1],
    'Vaginal+Cephalic':          [id for id in test_ids.id if clinical_lookup(id, 'delivery_type') == 1
                                                           and clinical_lookup(id, 'presentation') == 1],
    'No Labor Arrest':           [id for id in test_ids.id if clinical_lookup(id, 'NoProgress') == 0],
    'Vaginal+Cephalic+NoArrest': [id for id in test_ids.id if clinical_lookup(id, 'delivery_type') == 1
                                                           and clinical_lookup(id, 'presentation') == 1
                                                           and clinical_lookup(id, 'NoProgress') == 0],
}

# חייב להשתמש ב-stride=INFERENCE_STRIDE_REPRO (=1) — זה REPRO MODE!
for name, ids in subsets.items():
    y_true, y_pred = compute_predictions(model, lr_model, ids, stride=1)
    auc = roc_auc_score(y_true, y_pred)
    print(f"{name}: n={len(ids)}, acidemia={sum(y_true)}, AUC={auc:.3f}")
```

**Benchmarks מהמאמר (Table 3):**

| תת-קבוצה | n (מאמר) | AUC (benchmark) | סינון |
|-----------|----------|-----------------|-------|
| כלל Test | 55 | 0.826 | — |
| וגינליות | 50 | 0.850 | `delivery_type == 1` |
| מצגת ראש | 50 | 0.848 | `presentation == 1` |
| וגינלי + ראש | 46 | 0.853 | both |
| ללא עצירת לידה | 47 | 0.837 | `NoProgress == 0` |
| וגינלי+ראש+ללא עצירה | 43 | 0.837 | all three |

> **P3 note:** ספירות בדאטה המקומי עשויות להיות שונות (11 vs 12 acidemia — ידוע). דווח על ספירות בפועל.

#### A7.2 — קריטריוני הצלחה

| קריטריון | ערך | סוג |
|----------|-----|-----|
| AUC Test ≥ 0.75 | כלל test | **מינימלי** (מכה benchmark 0.68-0.75) |
| AUC Test ≥ 0.826 | כלל test | **יעד benchmark** מהמאמר |
| AUC וגינלי+ראש ≥ 0.853 | תת-קבוצה | **reference** (P3: לא pass/fail קשיח) |

> **אזהרה:** עם pretraining על 687 (לא 984), AUC עשוי להיות מעט נמוך. זה צפוי ומתועד (S1).

#### A7.3 — עקומות ROC

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for name, ids in subsets.items():
    fpr, tpr, _ = roc_curve(y_true[name], y_pred[name])
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
ax.legend()
fig.savefig('results/roc_curves.png', dpi=150)
```

#### A7.4 — Case Studies (Figure 6)

הרץ inference על 5 מקרים נבחרים:
- 2 False Positives
- 1 True Positive (pH <= 7.15)
- 1 True Negative
- 1 ניתוח קיסרי

לכל מקרה: גרף ציון רציף + סימון Stage 2 alert.

#### A7.5 — דוח סופי (`results/final_report.md`)

הדוח חייב לכלול:
1. טבלת AUC לפי תתי-קבוצות (עם n, acidemia count, prevalence בפועל)
2. השוואה ל-Table 3 של המאמר
3. עקומות ROC
4. Case studies (תמונות + ניתוח)
5. רשימת סטיות מהמאמר (מ-`deviation_log.md`)
6. מסקנות: האם הושגו קריטריוני ההצלחה

### Outputs (Artifacts חובה)

| # | קובץ | תיאור |
|---|-------|--------|
| O7.1 | `results/evaluation_table3.csv` | AUC per subset |
| O7.2 | `results/roc_curves.png` | גרף ROC |
| O7.3 | `results/case_studies/` | גרפים לכל מקרה |
| O7.4 | `results/final_report.md` | דוח סופי מלא |
| O7.5 | `notebooks/05_evaluation.ipynb` | Notebook ההערכה |

### Validation (Definition of Done)

- [ ] **V7.1:** AUC חושב לכל 6 תתי-קבוצות.
- [ ] **V7.2:** AUC Test כלל ≥ 0.75 (קריטריון מינימלי).
- [ ] **V7.3:** CI/accuracy מדווחים (אם AUC < 0.826 — מתועד עם נימוק S1).
- [ ] **V7.4:** ספירות תתי-קבוצות בפועל מדווחות (לא רק AUC).
- [ ] **V7.5:** Test set נגע **פעם אחת בלבד** — אין checkpoint/parameter שנבחר על בסיס תוצאות test.
- [ ] **V7.6:** ROC curves נשמרו.
- [ ] **V7.7:** Case studies (5 מקרים לפחות) עם גרפים.
- [ ] **V7.8:** `final_report.md` כולל את כל 6 הסעיפים שנדרשו.

### Post-Run Gates

- [ ] **G7.7:** V7.1 עד V7.8 כולם Pass.
- [ ] **G7.8:** `docs/project_context.md` > "סוכן 7" עודכן.
- [ ] **G7.9:** `docs/project_context.md` > "סיכום כללי" — כל 7 הסוכנים מסומנים ✅.
- [ ] **G7.10:** אם נמצאו בעיות חדשות — נפתח סעיף ב-`docs/work_plan_issues_review_he.md`.

### Handoff Contract → אין (סוף Pipeline)

זהו השלב האחרון. אין סוכן הבא.

### Report Back

עדכן `docs/project_context.md` (קטע סוכן 7 + סיכום כללי) עם:
- סטטוס: הושלם ✅
- AUC כלל test
- האם קריטריון הצלחה מינימלי הושג (≥0.75)
- האם benchmark הושג (≥0.826)
- סטיות שתועדו
- מסקנה כללית

---
---

## נספח — סיכום Artifacts לפי סוכן

| סוכן | Outputs |
|-------|---------|
| 1 | `data/raw/*`, `data/processed/*`, `data/splits/*`, `src/data/preprocessing.py`, `notebooks/00_data_prep.ipynb` |
| 2 | `src/model/patchtst.py`, `src/model/heads.py`, `config/train_config.yaml`, `notebooks/01_arch_check.ipynb` |
| 3 | `src/data/dataset.py`, `src/data/masking.py`, `src/train/pretrain.py`, `notebooks/02_pretrain.ipynb` |
| 4 | `src/data/dataset.py` (FinetuneDataset), `src/train/finetune.py`, `src/train/utils.py`, `notebooks/03_finetune.ipynb` |
| 5 | `src/inference/sliding_window.py`, `src/inference/alert_extractor.py`, `src/train/train_lr.py`, `notebooks/04_inference_demo.ipynb` |
| 6 | `checkpoints/pretrain/*.pt`, `checkpoints/finetune/*.pt`, `checkpoints/alerting/*.pkl`, `logs/*.csv` |
| 7 | `results/evaluation_table3.csv`, `results/roc_curves.png`, `results/case_studies/`, `results/final_report.md`, `notebooks/05_evaluation.ipynb` |
| 8 | `results/threshold_optimization_summary.csv`, `results/final_model_comparison.csv`, `results/cv_features_at040.npz`, `checkpoints/alerting/logistic_regression_at040.pkl`, `results/threshold_analysis_stage2.png`, `results/threshold_analysis_stage1.png`, `results/threshold_optimization_comparison.png` |

## נספח — קבועים שאסור לשנות (מחלק ז.1)

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
| Alert threshold | **0.4** | S11: הורד מ-0.5 (Deviation S11) |
| Decision threshold | **0.284** | Youden-optimal (AUC=0.839, Sens=0.818) |
| LR features | 4 בדיוק | Section II-F |
| Adam lr pretrain | 1e-4 | Section II-D |
| Dropout | 0.2 | Section II-E |
