# Azure ML Training Guide — SentinelFatal2

> **מטרת המסמך:** כל agent (אנושי או AI) שיקרא מסמך זה יוכל להריץ אימון על Azure ML מבלי
> לחזור על הבעיות שנתקלנו בהן. המסמך מבוסס על ניסיון ריאלי של עשרות ריצות ותקלות.

---

## תוכן עניינים

1. [ארכיטקטורה וקבצי הפרויקט](#1-ארכיטקטורה-וקבצי-הפרויקט)
2. [תשתית Azure ML — מה קיים](#2-תשתית-azure-ml--מה-קיים)
3. [דרישות מוקדמות](#3-דרישות-מוקדמות)
4. [שלבי הרצה מאפס עד סיום](#4-שלבי-הרצה-מאפס-עד-סיום)
5. [כיצד train_azure.py עובד על הצומת](#5-כיצד-train_azurepy-עובד-על-הצומת)
6. [בעיות שנתקלנו בהן ואיך להימנע מהן](#6-בעיות-שנתקלנו-בהן-ואיך-להימנע-מהן)
7. [הורדת לוגים ותוצאות לאחר האימון](#7-הורדת-לוגים-ותוצאות-לאחר-האימון)
8. [רשימת תיוג לפני הגשת Job](#8-רשימת-תיוג-לפני-הגשת-job)
9. [Troubleshooting — אבחון מהיר](#9-troubleshooting--אבחון-מהיר)

---

## 1. ארכיטקטורה וקבצי הפרויקט

### קבצי Azure ML

```
SentinelFatal2/
├── azure_ml/
│   ├── setup_and_submit.py   ← מריצים מהמחשב המקומי — שולח את ה-Job לענן
│   ├── train_azure.py        ← רץ על הצומת בענן — האימון עצמו
│   └── conda_env.yml         ← הסביבה (PyTorch, CUDA וכו')
├── src/                      ← קוד המודל והאימון (מועלה לצומת)
├── scripts/                  ← עזרי CV (מועלה לצומת)
├── config/                   ← קובצי YAML של המודל (מועלה לצומת)
└── data/splits/              ← CSVs של ה-splits (מועלה לצומת, קטן ~124KB)
```

### הפרדת תפקידים

| קובץ | רץ איפה | תפקיד |
|------|---------|--------|
| `setup_and_submit.py` | מחשב מקומי | אימות, בניית תמונת קוד, הגשת Job |
| `train_azure.py` | צומת ענן | הורדת data, pretrain, 5-fold CV, REPRO_TRACK |
| `conda_env.yml` | Azure ML | בניית סביבת conda על Docker |

---

## 2. תשתית Azure ML — מה קיים

> **כל הרכיבים כבר קיימים** — אין צורך ליצור מחדש.

| רכיב | ערך |
|------|-----|
| Subscription | `02b4b69d-dd14-4e79-b35f-de906edb6b15` (Azure for Students) |
| Tenant | `90373b7d-e0f5-41f4-bf72-c3c39a38bc80` |
| Resource Group | `sentinelfatal2-rg` (francecentral) |
| Workspace | `sentinelfatal2-aml` (francecentral) |
| Compute Cluster | `gpu-t4-cluster` (Standard_NC4as_T4_v3) |
| Environment | `sentinelfatal2-env:2` |
| Data Asset | `ctg-processed:1` (23 MB, data_processed.zip) |
| GPU | NVIDIA Tesla T4, 16 GB VRAM |
| VM Price | ~$0.50/hr (dedicated) |
| Studio URL | https://ml.azure.com |

### למה francecentral?
Azure for Students לא מאשר GPU quota בכל region. France Central הוא region שאושר לנו.

---

## 3. דרישות מוקדמות

### חבילות Python מקומיות
```bash
pip install azure-ai-ml azure-identity azure-mgmt-resource
```

### אימות (Authentication)
**השיטה היחידה שעובדת:** VS Code credential.

1. פתח VS Code
2. `Ctrl+Shift+P` → `Azure: Sign In`
3. התחבר עם חשבון Azure for Students

> **אזהרה:** `az` CLI **אינו** מותקן ולא מחובר. `AzureCliCredential` יכשל — זה בסדר,
> הסקריפט ינסה אוטומטית `VisualStudioCodeCredential` ואחר כך `DeviceCodeCredential`.

### בדיקת אימות מהיר
```python
from azure.identity import VisualStudioCodeCredential
from concurrent.futures import ThreadPoolExecutor
cred = VisualStudioCodeCredential()
with ThreadPoolExecutor(max_workers=1) as ex:
    tok = ex.submit(cred.get_token, "https://management.azure.com/.default").result(timeout=20)
print("Auth OK:", tok.token[:20], "...")
```

---

## 4. שלבי הרצה מאפס עד סיום

### צעד 1: הגשת Job

```bash
# מ-root של הרפו:
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 python -u azure_ml/setup_and_submit.py --dedicated
```

**`--dedicated`** = VMs dedicated (לא spot/low-priority). **חובה להשתמש בזה.**
Spot VMs (low-priority) גורמים לביטול Job ditty באמצע האימון ללא אזהרה.

**פלט צפוי:**
```
[AUTH] Authenticated via VS Code.
[RG]   Resource group 'sentinelfatal2-rg' already exists.
[WORKSPACE] 'sentinelfatal2-aml' already exists in francecentral.
[COMPUTE] 'gpu-t4-cluster' already exists (Standard_NC4as_T4_v3, min=0, max=1).
[ENV]  Environment 'sentinelfatal2-env' version 2 ready.
[DATA] Data asset 'ctg-processed' already registered (version 1) — skipping upload.
[CODE] Building minimal code snapshot ...
[CODE]   + src/
[CODE]   + azure_ml/
[CODE]   + scripts/
[CODE]   + config/
[CODE]   + data/splits/
[CODE] Snapshot ready: 0.6 MB  (was >600 MB with full repo)
[JOB]  Submitting job to Azure ML ...
============================================================
JOB SUBMITTED SUCCESSFULLY
  Job name   : <שם-אוטומטי>
  Status     : Starting
  Studio URL : https://ml.azure.com/runs/<job-name>?...
============================================================
[STREAM] Waiting for job to start ...
```

### צעד 2: המתנה לצומת

לאחר הגשה, Azure ML צריך:
- **~1-5 דקות**: הקצאת VM (אם יש צומת זמין מריצה קודמת)
- **~20-40 דקות**: pull של Docker image (בפעם הראשונה / VM חדש)
- **~1-5 דקות**: הכנת סביבת conda + העלאת קוד

**סה"כ: ≈10-45 דקות עד שה-`[INIT]` הראשון מופיע.**

### צעד 3: מעקב אחר האימון

הסקריפט מזרים logs אוטומטית. סדר השורות הצפוי:

```
[INIT] repo root: /mnt/azureml/...
[INIT] Python : 3.10.x
[DATA] --data not provided; downloading data_processed.zip from GitHub ...
[DATA] Extracting data_processed.zip ...
[DATA] 552 .npy files ready.
[OK] finetune.py signature verified.
[DEVICE] cuda
         Tesla T4
         VRAM: 15935 MB
[PRETRAIN] No checkpoint found — running pretraining ...
...
[FOLD 0] Starting fresh from epoch 0.
...
[FOLD 4] Test  AUC=0.xxxx  ...
GLOBAL OOF RESULTS
  Global OOF AUC : 0.xxxx
...
REPRO_TRACK — Canonical split (train=441 / val=56 / test=55)
...
[DONE] All done. Total runtime: xxx min
```

### צעד 4: הורדת תוצאות

ראה [פרק 7](#7-הורדת-לוגים-ותוצאות-לאחר-האימון).

---

## 5. כיצד train_azure.py עובד על הצומת

### זרימת הנתונים

```
GitHub (data_processed.zip, 23 MB)
    ↓  wget (fallback — כאשר אין --data arg)
/tmp/data_processed.zip
    ↓  unzip
./data/processed/ctu_uhb/*.npy  (552 קבצים)
    ↓
PretrainDataset / FinetuneDataset
    ↓
PatchTST model (413K encoder params)
```

### שיטת קבלת Data

`train_azure.py` תומך בשתי שיטות:

1. **`--data <path>`** — Azure ML Data Asset (zip הורד לפני שה-Python עלה)
   - **לא בשימוש כרגע** (ראה [בעיה #3](#בעיה-3-data-input-modes-גורמים-ל-hang))
2. **ללא `--data`** (ברירת מחדל נוכחית) — מוריד מ-GitHub בזמן ריצה
   - data_processed.zip נגיש ב: `https://raw.githubusercontent.com/ArielShamay/SentinelFatal2/master/data_processed.zip`
   - בסביבת Azure datacenter ההורדה לוקחת ~5-30 שניות

### DataLoader ו-num_workers

```python
# CRITICAL: num_workers חייב להיות 0 בתוך הצומת
# multiprocessing עם fork() על Linux גורם ל-deadlock עם CUDA
DataLoader(dataset, num_workers=0, pin_memory=False)
```

זה מחויב ב-train_azure.py דרך monkey-patch על `DataLoader.__init__`.

### תוצאות שנשמרות לענן (נגישות להורדה)

| קובץ | מיקום בצומת | מה מכיל |
|------|------------|---------|
| `shared_pretrain_loss.csv` | `logs/e2e_cv_v3/` | train/val loss לכל epoch של pretrain |
| `fold{0-4}_finetune_loss.csv` | `logs/e2e_cv_v3/` | train/val AUC לכל epoch לכל fold |
| `repro_track_loss.csv` | `logs/e2e_cv_v3/` | loss curves של REPRO_TRACK |
| `std_log.txt` | `user_logs/` | כל ה-stdout — השורה הראשית לאבחון |

> **חשוב:** `final_cv_report_v3.csv`, `repro_comparison_v3.csv`, ו-`comparison_table_v3.csv`
> נשמרים לדיסק המקומי של הצומת אך **לא מועלים** לאחסון Azure ML (הצומת נמחק בסיום).
> כל המידע הרלוונטי נמצא ב-`std_log.txt`.

---

## 6. בעיות שנתקלנו בהן ואיך להימנע מהן

> זהו הפרק הקריטי ביותר. כל הבעיות מתועדות עם הסיבה המדויקת.

---

### בעיה #1: VisualStudioCodeCredential תוקע לנצח

**תסמין:** `_get_credential()` תקוע ב-`AzureCliCredential` או `VisualStudioCodeCredential` ולא ממשיך.

**סיבה:** שתי קריאות ל-`.get_token()` בלי timeout — אם ה-credential לא זמין, הן חוסמות לנצח.

**פתרון שיושם:**
```python
import concurrent.futures
TIMEOUT = 20  # שניות
cred = VisualStudioCodeCredential()
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
    future = ex.submit(cred.get_token, "https://management.azure.com/.default")
    future.result(timeout=TIMEOUT)  # זורק TimeoutError אם עבר זמן
```

**כלל:** **לעולם אל תקרא ל-`.get_token()` ישירות בלי ThreadPoolExecutor + timeout.**

---

### בעיה #2: conda channels גורמים ל-hang של 75+ דקות בבניית environment

**תסמין:** Job בשלב "Preparing" יותר משעה, אין progress.

**סיבה:** `conda_env.yml` כלל את channels:
```yaml
channels:
  - pytorch      # ← גורם ל-metadata resolution בלתי גמרת
  - nvidia       # ← אותה בעיה
  - conda-forge
  - defaults
```
Azure ML מנסה לסנכרן את כל ה-metadata של channels אלו — מה שלוקח שעות.

**פתרון שיושם:** הסרת channels pytorch ו-nvidia, והוספת torch wheel index ישירות ל-pip:
```yaml
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cu118
    - torch==2.2.0+cu118        # ← גרסה מפורשת עם CUDA
    - torchvision==0.17.0+cu118
    - "numpy>=1.26,<2.0"        # ← חובה <2.0 (ראה בעיה #5)
```

**כלל:** **אל תוסיף `pytorch` או `nvidia` ל-channels בקובץ conda_env.yml.**

---

### בעיה #3: Data Input Modes גורמים ל-HANG לפני שPython עולה

**תסמין:** Job ב-"Running" 10-15 דקות ואין אף שורה ב-std_log.txt. Python לא עלה.

**סיבה:** כאשר Job מוגדר עם `Input(type=URI_FILE)` — בין אם mode='ro_mount' ובין אם mode='download' — Azure ML מנסה לטפל בקובץ **לפני שה-Python process עולה**. ב-ro_mount מנסה לבנות FUSE mount; ב-download מנסה להוריד את הקובץ לדיסק. שניהם תלויים בתהליכי Azure ML agent שנכשלים בשקט.

**שני modes שנבדקו — שניהם נכשלו:**
```python
# ❌ ro_mount — FUSE mount hang
Input(type=AssetTypes.URI_FILE, path=f"azureml:{data_ref}", mode='ro_mount')

# ❌ download — pre-download hang
Input(type=AssetTypes.URI_FILE, path=f"azureml:{data_ref}", mode='download')
```

**פתרון שיושם:** הסרת כל Azure ML data inputs מהגדרת ה-Job. `train_azure.py` מוריד מ-GitHub ישירות:
```python
job = command(
    code=code_dir,
    command="python azure_ml/train_azure.py",  # ← ללא --data arg
    # ← אין inputs= כלל
    environment=env_ref,
    compute=COMPUTE_NAME,
    ...
)
```

**כלל:** **אל תשתמש ב-Azure ML data inputs (URI_FILE/URI_FOLDER) — הם תלויים ב-FUSE agents שאינם אמינים. השתמש ב-GitHub download fallback ב-train_azure.py.**

---

### בעיה #4: URI_FOLDER Outputs גורמים ל-FUSE hang

**תסמין:** Python עולה אבל Job תקוע לפני שמגיע לאימון.

**סיבה:** הגדרת custom outputs עם URI_FOLDER מחייבת Azure ML לבנות FUSE mounts עבור תיקיות ה-output — גם אלה נכשלים בשקט.

**קוד שגורם לבעיה:**
```python
# ❌ אל תעשה זאת
outputs={
    "results":     Output(type=AssetTypes.URI_FOLDER, mode='rw_mount'),
    "checkpoints": Output(type=AssetTypes.URI_FOLDER, mode='rw_mount'),
    "logs":        Output(type=AssetTypes.URI_FOLDER, mode='rw_mount'),
}
```

**פתרון שיושם:** הסרת כל custom outputs. Azure ML יוצר אוטומטית artifacts directory שמועלה בסיום Job:
```python
job = command(
    # ← אין outputs= כלל
    ...
)
```

**כלל:** **אל תגדיר custom outputs (URI_FOLDER) בהגדרת ה-Job. הסתמך על Azure ML default artifacts upload.**

---

### בעיה #5: NumPy >= 2.0 שובר torch.from_numpy()

**תסמין:** `RuntimeError: Numpy is not available` או crash ב-C API level בזמן ריצה.

**סיבה:** NumPy 2.0 שינה את ה-C API. PyTorch 2.2.0+cu118 בנוי כנגד NumPy 1.x.

**פתרון:**
```yaml
- "numpy>=1.26,<2.0"  # ← חובה! אל תוריד את ה-upper bound
```

---

### בעיה #6: Code snapshot גדול גורם ל-hang בחילוץ הארכיב

**תסמין:** Job ב-"Preparing" / "Starting" ו-Python לא עולה גם אחרי 10+ דקות.

**סיבה:** Azure ML מעלה את קוד הפרויקט כ-tar.gz ל-Azure Blob ואז מחלץ אותו על הצומת. עם ה-root של הרפו (932 MB data + 113 MB checkpoints), הארכיב היה 24.5 MB+ וחילוצו על הצומת גרם ל-timeout/hang.

**אבחון:** השווה `placid_yak` (0.6 MB snapshot — Python עלה ב-48 שניות ✅) לעומת `sad_worm` (24.5 MB — Python לא עלה אחרי 9 דקות ❌).

**פתרון שיושם:** `_build_code_snapshot()` בונה temp dir מינימלי:
```python
needed_dirs = ["src", "azure_ml", "scripts", "config"]
for d in needed_dirs:
    shutil.copytree(REPO_ROOT / d, tmp / d)
# + data/splits/ (CSVs, 124 KB)
# תוצאה: ~0.6 MB במקום 600+ MB
```

**כלל:** **אל תוסיף קבצי data לתוך snapshot הקוד** (כולל `data_processed.zip`). **ה-snapshot חייב להיות < 5 MB.**

> **הערה על .amlignore:** קיים קובץ `.amlignore` בשורש הרפו, אבל Azure ML SDK
> לא מכבד אותו כשמשתמשים ב-`code=<temp_dir>` ב-`command()`. הפתרון הנכון הוא **תמיד** `_build_code_snapshot()`.

---

### בעיה #7: torch==2.2.0 מ-PyPI הוא CPU-only

**תסמין:** אימון רץ אבל CUDA לא זמין. `torch.cuda.is_available()` מחזיר `False`.

**סיבה:** `pip install torch==2.2.0` מ-PyPI מביא גרסת CPU-only. גרסת CUDA נמצאת ב-index נפרד.

**פתרון:**
```yaml
- --extra-index-url https://download.pytorch.org/whl/cu118
- torch==2.2.0+cu118   # ← החלק "+cu118" קריטי
```

---

### בעיה #8: num_workers > 0 גורם ל-deadlock עם CUDA

**תסמין:** Job "Running" אבל לא מתקדם לאחר שמגיע ל-DataLoader init.

**סיבה:** `multiprocessing` עם `fork()` על Linux בשילוב CUDA contexts גורם ל-deadlock.

**פתרון שיושם ב-train_azure.py:**
```python
# Monkey-patch DataLoader לפני כל import של קוד training
import torch.utils.data
_orig_init = torch.utils.data.DataLoader.__init__
def _patched_init(self, *args, **kwargs):
    kwargs['num_workers'] = 0
    kwargs.pop('pin_memory', None)
    _orig_init(self, *args, **kwargs)
torch.utils.data.DataLoader.__init__ = _patched_init
```

> **אזהרה לגבי monkey-patch:** הפתרון חייב להחליף את `__init__` על הקלאס עצמו.
> monkey-patch על local reference (`dl_class = DataLoader; dl_class.__init__ = ...`) לא עובד
> כי ה-module כבר ייבא את ה-reference המקורי.

**כלל:** **תמיד `num_workers=0` ו-`pin_memory=False` בסביבת Azure ML.**

---

### בעיה #9: Low Priority (Spot) VMs — ביטול Job באמצע

**תסמין:** Job מבוטל פתאום עם סטטוס "Canceled" בלי שום שגיאה מהקוד.

**סיבה:** Low Priority VMs יכולים להיות מבוטלים על-ידי Azure בכל רגע כשיש ביקוש גבוה.

**פתרון:** השתמש **תמיד** ב-`--dedicated`:
```bash
python azure_ml/setup_and_submit.py --dedicated
```

---

### בעיה #10: PYTHONUNBUFFERED לא מוגדר — stdout לא נראה

**תסמין:** `[STREAM]` מראה שה-Job רץ אבל אין פלט בטרמינל.

**סיבה:** Python מ-buffer את stdout כשהוא לא מחובר ל-TTY.

**פתרון:** הסביבה המשתנה מוגדרת בהגדרת ה-Job:
```python
environment_variables={"PYTHONUNBUFFERED": "1"}
```
וגם בפקודת ההרצה המקומית:
```bash
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 python -u azure_ml/setup_and_submit.py --dedicated
```

---

### בעיה #11: Zombie Nodes לאחר ביטול Job

**תסמין:** Job חדש ב-"Starting" יותר מ-10 דקות, לא מתקדם.

**סיבה:** ביטול Job שמשתמש בצומת פעיל לא תמיד משחרר אותו מיד. הצומת נשאר ב-"Active" state וה-Job החדש ממתין לו.

**פתרון:** לפני הגשת Job חדש, ודא שהצומת ב-"Idle" (Active nodes = 0):
```python
from azure.ai.ml import MLClient
cluster = ml_client.compute.get("gpu-t4-cluster")
print(f"Allocated: {cluster.current_node_count}, Idle: {cluster.idle_node_count}")
```
או בדוק ב-Azure ML Studio → Compute → gpu-t4-cluster.

**המתן עד ש-Active nodes = 0 לפני הגשת Job חדש לאחר ביטול.**

---

## 7. הורדת לוגים ותוצאות לאחר האימון

### הורדת כל ה-artifacts

```python
from azure.ai.ml import MLClient
from azure.identity import VisualStudioCodeCredential
import pathlib

ml = MLClient(
    VisualStudioCodeCredential(),
    "02b4b69d-dd14-4e79-b35f-de906edb6b15",
    "sentinelfatal2-rg",
    "sentinelfatal2-aml"
)

job_name = "<שם-ה-job>"   # למשל: frosty_kite_kdt326xr9h
out_dir = pathlib.Path("logs/e2e_cv_v3/azure_job")
ml.jobs.download(job_name, download_path=str(out_dir), all=True)
```

### מבנה artifacts לאחר הורדה

```
logs/e2e_cv_v3/azure_job/
└── artifacts/
    ├── user_logs/
    │   └── std_log.txt              ← כל ה-stdout (11,000+ שורות)
    ├── logs/
    │   └── e2e_cv_v3/
    │       ├── shared_pretrain_loss.csv
    │       ├── fold0_finetune_loss.csv
    │       ├── fold1_finetune_loss.csv
    │       ├── fold2_finetune_loss.csv
    │       ├── fold3_finetune_loss.csv
    │       ├── fold4_finetune_loss.csv
    │       └── repro_track_loss.csv
    └── system_logs/                 ← לוגי מערכת Azure ML (אבחון בעיות תשתית)
```

### קריאת std_log.txt (Unicode-safe)

```python
log_file = pathlib.Path("logs/e2e_cv_v3/azure_job/artifacts/user_logs/std_log.txt")
lines = log_file.read_text(encoding='utf-8', errors='replace').splitlines()
print(f"Total lines: {len(lines)}")

# שורות מסכם
for i, line in enumerate(lines):
    if any(kw in line for kw in ['[FOLD', 'OOF AUC', 'Per-fold', 'REPRO_TRACK', '[DONE]', 'PASS', 'FAIL']):
        print(f"{i:5d}: {line}")
```

> **אזהרה:** std_log.txt מכיל תווי Unicode (→, ✓, ✗ וכו').
> קרא תמיד עם `encoding='utf-8', errors='replace'`.
> **לעולם אל תדפיס לטרמינל cp1255** — השתמש ב-`sys.stdout.buffer.write(line.encode('utf-8'))`
> או כתוב לקובץ קודם.

### חיפוש תוצאות מרכזיות

```python
# שורות תוצאה ספציפיות:
keywords = {
    'G1 GATE':        'תוצאת pretrain',
    'FOLD 0':         'תוצאת fold 0',
    'Global OOF AUC': 'AUC כולל',
    'G4a':            'gate check',
    'REPRO_TRACK':    'REPRO_TRACK test',
    '[DONE] All done':'זמן ריצה כולל',
}
for kw, desc in keywords.items():
    matches = [f"{i}: {l}" for i, l in enumerate(lines) if kw in l]
    print(f"\n--- {desc} ---")
    for m in matches[-3:]:  # 3 אחרונות
        print(m)
```

---

## 8. רשימת תיוג לפני הגשת Job

לפני כל הגשת Job חדש, עבור על הרשימה:

### קוד ה-Job (`setup_and_submit.py`)

- [ ] `--dedicated` flag — לא `--low-priority`
- [ ] `_build_code_snapshot()` לא כולל `data_processed.zip` או תיקיית data
- [ ] `_build_code_snapshot()` snapshot קטן מ-5 MB (בדוק עם `total_mb` printout)
- [ ] `command="python azure_ml/train_azure.py"` — ללא `--data` arg
- [ ] אין `inputs=` בהגדרת ה-Job (`URI_FILE` / `URI_FOLDER`)
- [ ] אין `outputs=` מותאמות אישית בהגדרת ה-Job
- [ ] `environment_variables={"PYTHONUNBUFFERED": "1"}`

### קוד הסביבה (`conda_env.yml`)

- [ ] אין `pytorch` או `nvidia` ב-channels
- [ ] `torch==2.2.0+cu118` (לא `torch==2.2.0`)
- [ ] `--extra-index-url https://download.pytorch.org/whl/cu118` קודם לשורת torch
- [ ] `"numpy>=1.26,<2.0"` (upper bound חובה)

### קוד האימון (`train_azure.py`)

- [ ] monkey-patch של `DataLoader.__init__` עם `num_workers=0` — מופעל לפני כל import אחר
- [ ] fallback GitHub URL תקין: `https://raw.githubusercontent.com/ArielShamay/SentinelFatal2/master/data_processed.zip`
- [ ] שורה `[INIT]` קיימת (מאמתת שPython עלה)
- [ ] שורה `[DATA] 552 .npy files ready.` צפויה

### תשתית

- [ ] VS Code מחובר לאזור (`Azure: Sign In`)
- [ ] cluster `gpu-t4-cluster` ב-`Idle` state (Active nodes = 0) לפני הגשה
- [ ] Environment `sentinelfatal2-env:2` קיים (לא יצור חדש אלא אם שינית `conda_env.yml`)

---

## 9. Troubleshooting — אבחון מהיר

### Job ב-"Starting" / "Preparing" יותר מ-5 דקות

1. בדוק ב-Studio → Compute → gpu-t4-cluster: כמה Active nodes יש?
2. אם Active > 0 והJob קודם בוטל — המתן לשחרור (5-10 דקות)
3. אם cluster Idle — בדוק system_logs לאחר הורדה

### std_log.txt ריק או 404

הJob עוד לא התחיל להריץ Python. בדוק:
- `system_logs/lifecycler/lifecycler.log` — מה שלב ה-lifecycle?
- `system_logs/snapshot_capability/snapshot-capability.log` — חילוץ snapshot הסתיים?

### Python עלה אבל תקוע לפני `[DATA]`

נדיר — אבל אם קורה, בדוק את ה-system_logs לשגיאות FUSE.

### Python עלה, `[DATA]` הופיע, אבל `[DEVICE]` לא מופיע

בדוק אם `torch.cuda.is_available()` מחזיר False (גרסת CPU-only של torch).
פתרון: ודא `torch==2.2.0+cu118` ב-conda_env.yml.

### Job נכשל עם `DiskSpaceFull`

הצומת הקטן (NC4as_T4_v3) מגיע עם ~80 GB. data_processed.zip מכיל 552 קבצי npy ו-checkpoints — בדוק שאין שמירת checkpoints לכל epoch.

### Job מצליח אבל AUC נמוך מהמצופה

ראה את [ניתוח התוצאות](#e2e-cv-v3-תוצאות-מ-2026-02-27) — זהו אתגר מודל, לא בעיה תשתית.

---

## סיכום — E2E CV v3 תוצאות מ-2026-02-27

**Job:** `frosty_kite_kdt326xr9h` | Runtime: 162.8 דקות | Status: Completed ✅

| מדד | ערך |
|-----|-----|
| Global OOF AUC | 0.6013 (CI: 0.543–0.660) |
| Mean fold AUC | 0.6155 ± 0.027 |
| G4a Gate (mean ≥ 0.70) | **FAIL** |
| G4b Gate (std ≤ 0.10) | PASS |
| REPRO_TRACK test AUC | 0.653 |
| Baseline (Stage2 LR) | 0.812 |
| Paper benchmark | 0.826 |

**הביצועים נמוכים מהבסלייין** — ה-Pipeline האינפרסטרוקטורה עובד. הבעיה היא בגישת המודל/אימון עצמה.
