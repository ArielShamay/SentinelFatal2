# תיעוד דאטה מלא - תיקיית `data`

תאריך סריקה: `2026-02-22`

## 1. תמונת מצב כללית

| נתיב | כמות קבצים | נפח |
|---|---:|---:|
| `data/CTGDL` | 10 | 98.44 MB |
| `data/ctu-chb-intrapartum-cardiotocography-database-1.0.0` | 1110 | 38.06 MB |
| `data/FHRMA-master` | 1318 | 449.65 MB |
| **סה"כ `data`** | **2438** | **586.16 MB** |

---

## 2. תיקיית `data/CTGDL`

### 2.1 מה יש בתיקייה

תיקייה זו מכילה מטא-דאטה מרכזי + ארכיוני CSV מעובדים.

| קובץ | סוג | גודל |
|---|---|---:|
| `CTGDL  -  Data Collection.pdf` | PDF תיעוד | 0.56 MB |
| `CTGDL_ctu_uhb_csv.tar.gz` | ארכיון CSV גולמי (CTU) | 14.03 MB |
| `CTGDL_ctu_uhb_proc_csv.tar.gz` | ארכיון CSV מעובד (CTU) | 25.22 MB |
| `CTGDL_FHRMA_ano_csv.tar.gz` | ארכיון CSV עם אנוטציות מורפולוגיות | 45.98 MB |
| `CTGDL_FHRMA_proc_csv.tar.gz` | ארכיון CSV מעובד (FHRMA) | 6.72 MB |
| `CTGDL_norm_metadata.csv` | מטא-דאטה מאוחד | 67,377 B |
| `CTGDL_FHEMA_metadata.csv` | מטא-דאטה FHRMA | 18,614 B |
| `CTGDL_SPAM_metadata.csv` | מטא-דאטה SPAM | 34,021 B |
| `CTGDL_spam_dataset_read_matlab_signal_and_preprocess.ipynb` | מחברת יצירת נתוני SPAM | 5.80 MB |
| `ctgdl_spam_dataset_read_matlab_signal_and_preprocess.py` | סקריפט Python מקביל למחברת | 14,153 B |

### 2.2 ארכיוני הדאטה והמבנה שלהם

#### `CTGDL_ctu_uhb_csv.tar.gz`
- 552 קבצי CSV.
- מבנה אחיד בכל הקבצים: `fhr,uc`.
- טווח שורות לקובץ: 14,400 עד 21,620.
- סה"כ שורות בארכיון: 9,825,177.
- דוגמת קובץ: `CTGDL_ctu_uhb_csv/ctgdl_ctu_uhb_1001.csv`.

#### `CTGDL_ctu_uhb_proc_csv.tar.gz`
- 552 קבצי CSV.
- מבנה אחיד: `fhr,uc,fhr_is_nan,uc_is_nan`.
- טווח שורות לקובץ: 14,400 עד 21,620.
- סה"כ שורות בארכיון: 9,825,177.
- דוגמת קובץ: `CTGDL_ctu_uhb_proc_csv/ctgdl_ctu_uhb_1001.csv`.

#### `CTGDL_FHRMA_ano_csv.tar.gz`
- 290 קבצי CSV.
- שתי סכמות עמודות:
1. `toco,fhr,baseline,acc,dec` (94 קבצים).
2. `toco,fhr,fhr_nan,uc_nan,uc,baseline,acc,dec` (196 קבצים).
- חלוקה לפי סיומת:
  - `.csv`: 156 קבצים.
  - `.csv.csv`: 134 קבצים.
- טווח שורות לקובץ: 7,010 עד 115,109.
- סה"כ שורות בארכיון: 4,344,195.

#### `CTGDL_FHRMA_proc_csv.tar.gz`
- 135 קבצי CSV.
- סכמה אחידה: `uc,fhr,fhr_is_nan,uc_is_nan,no_value_uc`.
- טווח שורות לקובץ: 7,010 עד 115,109.
- סה"כ שורות בארכיון: 3,370,265.

### 2.3 תיוגים (labels) במטא-דאטה

#### `CTGDL_norm_metadata.csv` (981 שורות)
עמודות:
`pid,id,sig_len,sig_min,fname,stage2_idx,stage2min,dataset,target,test`

התפלגויות חשובות:
- `dataset`:
  - `ctg`: 552
  - `spam`: 294
  - `fhrma`: 135
- `target`:
  - `1`: 116
  - `0`: 865
- `test`:
  - `0`: 760
  - `1`: 56
  - `2`: 165

פירוק `dataset|test|target`:
- `ctg|test=0|target=0`: 351
- `ctg|test=0|target=1`: 90
- `ctg|test=1|target=0`: 44
- `ctg|test=1|target=1`: 12
- `ctg|test=2|target=0`: 44
- `ctg|test=2|target=1`: 11
- `fhrma|test=0|target=0`: 85
- `fhrma|test=2|target=0`: 50
- `spam|test=0|target=0`: 231
- `spam|test=0|target=1`: 3
- `spam|test=2|target=0`: 60

הערות חשובות:
- עבור `dataset='fhrma'` הערכים `stage2_idx=-1` ו-`stage2min=-1` (אין שלב 2 מסומן).
- עבור `dataset='spam'` קיימים גם ערכי `stage2_idx`/`stage2min` חריגים (כולל שליליים).
- בדיקת התאמה מול `CTU .hea` מראה שברשומות `ctg`:
  - `target=1` תואם בפועל ל-`pH <= 7.15`.
  - `target=0` תואם בפועל ל-`pH >= 7.16`.
  זו הסקה מתוך הקבצים המקומיים (לא טקסט רשמי בתוך CSV עצמו).

#### `CTGDL_FHEMA_metadata.csv` (135 שורות)
עמודות:
`pid,test,file_name,sig_length,sig_min,fhr_mean,fhr_min,toco_mean,toco_min,n_fhr_zeros,n_toco_zeros,n_fhr_nan,fhr_per_nan,toco_per_zero,id,org_fname`

תיוגים:
- `test`: `0` (62), `1` (73).

#### `CTGDL_SPAM_metadata.csv` (294 שורות)
עמודות:
`id,fs,fname,stage2_idx,sig_len,n_pad_toco_leading_0,II.stage,sig_min,stage2min,org_file_name,sum_0,uc_not_nan,uc_not_nan_min`

תיוגים/שדות דיסקרטיים:
- `fs=4` לכל הרשומות.
- `n_pad_toco_leading_0`: `0/4/8/12` (269/9/10/6 בהתאמה).

### 2.4 התאמה בין `fname` למידע בפועל

בדיקה מול הארכיונים:
- `ctg` (552 שמות קובץ): כולם קיימים בשני ארכיוני CTU (`raw` ו-`proc`).
- `fhrma` (135 שמות קובץ): כולם קיימים ב-`CTGDL_FHRMA_proc_csv.tar.gz`.
- `spam` (294 שמות קובץ): **לא נמצאו** בארכיונים שבתיקיית `CTGDL` (קיים מטא-דאטה בלבד).

### 2.5 דוגמאות מבנה דאטה (CSV)

דוגמה 1: CTU raw (`fhr,uc`)
```csv
fhr,uc
150.5,7.0
150.5,8.5
151.0,8.5
151.25,7.5
```

דוגמה 2: CTU proc (`fhr,uc,fhr_is_nan,uc_is_nan`)
```csv
fhr,uc,fhr_is_nan,uc_is_nan
150.5,7.0,0,0
150.5,8.5,0,0
151.0,8.5,0,0
151.25,7.5,0,0
```

דוגמה 3: FHRMA ano (סכמה 5 עמודות)
```csv
toco,fhr,baseline,acc,dec
20.0,134.25,128.96627011927552,0,0
19.5,134.75,128.95342255971994,0,0
```

דוגמה 4: FHRMA ano (סכמה 8 עמודות)
```csv
toco,fhr,fhr_nan,uc_nan,uc,baseline,acc,dec
21.5,161.25,161.25,21.5,21.5,162.9431654631044,0,0
21.0,161.75,161.75,21.0,21.0,162.9548868400048,0,0
```

דוגמה 5: FHRMA proc
```csv
uc,fhr,fhr_is_nan,uc_is_nan,no_value_uc
2.5,120.25,0,1,0
2.5,120.0,0,1,0
```

---

## 3. תיקיית `data/ctu-chb-intrapartum-cardiotocography-database-1.0.0`

### 3.1 מבנה תיקיות

- `analysis_results/`
  - `metadata_summary.csv` (100 שורות).
  - `signal_quality_stats.csv` (20 שורות).
- `ctu-chb-intrapartum-cardiotocography-database-1.0.0/`
  - 552 קבצי `.dat` (אות בינארי).
  - 552 קבצי `.hea` (header + מטא-דאטה טקסטואלי).
  - `RECORDS` (552 מזהי רשומה).
  - `SHA256SUMS.txt` (1107 שורות checksum).
  - `wfdbcal` (כיול ערוצים).
  - `ANNOTATIONS` (ריק, 0 בתים).

### 3.2 מבנה ותצורת נתוני `.hea`/`.dat`

שורת כותרת בכל `.hea`:
- `record_id num_signals sampling_freq num_samples`
- בכל הרשומות:
  - `num_signals=2`
  - `sampling_freq=4`
  - `num_samples` בטווח 14,400 עד 21,620

שתי שורות ערוץ טיפוסיות:
- ערוץ `FHR` (יחידות bpm, gain=100, baseline=12).
- ערוץ `UC` (יחידות nd, gain=100, baseline=12).

### 3.3 כל שדות המטא-דאטה (תיוגים) שנמצאו ב-`.hea`

שדות שנמצאו בכל 552 הרשומות:
- `pH`, `BDecf`, `pCO2`, `BE`, `Apgar1`, `Apgar5`
- `NICU days`, `Seizures`, `HIE`, `Intubation`, `Main diag.`, `Other diag.`
- `Gest. weeks`, `Weight(g)`, `Sex`
- `Age`, `Gravidity`, `Parity`, `Diabetes`, `Hypertension`, `Preeclampsia`, `Liq. praecox`, `Pyrexia`, `Meconium`
- `Presentation`, `Induced`, `I.stage`, `NoProgress`, `CK/KP`, `II.stage`, `Deliv. type`
- `dbID`, `Rec. type`, `Pos. II.st.`, `Sig2Birth`

ערכי תיוגים דיסקרטיים בולטים:
- `Deliv. type`: `1` (506), `2` (46)
- `II.stage`: `5/10/15/20/25/30/-1/2`
- `Sex`: `1` (286), `2` (266)
- `Diabetes`: `0` (515), `1` (37)
- `Hypertension`: `0` (525), `1` (27)
- `Preeclampsia`: `0` (530), `1` (22)
- `Meconium`: `0` (488), `1` (64)
- `NoProgress`: `0` (497), `1` (55)
- `Rec. type`: `1/2/12/-1`
- `Sig2Birth`: `0` בכל הרשומות

### 3.4 `analysis_results` (סיכומי עיבוד)

#### `metadata_summary.csv`
- 100 שורות.
- 31 עמודות.
- כולל שדות רפואיים דומים ל-`.hea` אחרי נירמול שמות עמודות:
  `record_id,sampling_freq,num_samples,fhr_gain,fhr_baseline,uc_gain,uc_baseline,num_signals,duration_min,pH,BDecf,pCO2,BE,Apgar1,Apgar5,gest_weeks,weight_g,sex,maternal_age,gravidity,parity,diabetes,hypertension,preeclampsia,meconium,presentation,induced,stage1_min,stage2_min,delivery_type,pos_stage2`

#### `signal_quality_stats.csv`
- 20 שורות.
- 19 עמודות איכות-אות:
  `record_id,duration_min,fhr_missing_pct,fhr_mean,fhr_median,fhr_std,fhr_min,fhr_max,fhr_zeros,fhr_artifacts,uc_missing_pct,uc_mean,uc_median,uc_std,uc_min,uc_max,uc_zeros,uc_artifacts,fhr_spikes`

### 3.5 דוגמאות מבנה דאטה (CTU)

דוגמה 1: `.hea` (טקסט)
```text
1001 2 4 19200
1001.dat 16 100(0)/bpm 12 0 15050 20101 0 FHR
1001.dat 16 100/nd 12 0 700 378 0 UC
#pH           7.14
#II.stage     20
#Deliv. type  1
```

דוגמה 2: `.dat` (בינארי, HEX)
```text
00000000  CA 3A BC 02 CA 3A 52 03 FC 3A 52 03 15 3B EE 02
00000010  15 3B B6 03 B1 3A 52 03 B1 3A 1A 04 B1 3A B0 04
```

דוגמה 3: `wfdbcal` (טקסט)
```text
# file: wfdbcal
FHR  0 - undefined 20 bpm
UC   - - undefined 50 nd
```

---

## 4. תיקיית `data/FHRMA-master`

### 4.1 מה יש בתיקייה

תיקייה זו מכילה גם דאטה וגם קוד/כלים של MATLAB לניתוח FHR.

סיכום סוגי קבצים:
- `.fhrm`: 1076
- `.fhr`: 156
- `.mat`: 25
- `.m`: 42
- `.h5`: 3
- `.ipynb`: 3
- `.py`: 1
- `.zip`: 1
- קבצים נוספים: `.prj`, `.mlapp`, `.fig`, `.png`, `LICENSE`, `.c`, `.mexw32`, `.mexw64`, `README.md`

### 4.2 תתי-תיקיות דאטה וכמויות

#### `Examples/`
- 11 קבצי `.fhrm`.

#### `FHRMAdataset/`
- `traindata/`: 66 קבצי `.fhr`.
- `testdata/`: 90 קבצי `.fhr`.
- `analyses/`: 17 קבצי `.mat`:
  `A_std.mat,C_orig.mat,C_std.mat,expertAnalyses.mat,H_std.mat,J_orig.mat,J_std.mat,L_std.mat,MD_std.mat,MG_std.mat,MT_orig.mat,MT_std.mat,P_std.mat,T_orig.mat,T_std.mat,WMFB_orig.mat,W_std.mat`

#### `FSdataset/`
- `DopMHR/Train`: 591 קבצי `.fhrm`
- `DopMHR/Val`: 142 קבצי `.fhrm`
- `DopMHR/TestDoubleSignals`: 45 קבצי `.fhrm`
- `DopMHR/TestCurrentPractice`: 30 קבצי `.fhrm`
- `ScalpECG/Train`: 188 קבצי `.fhrm`
- `ScalpECG/Val`: 39 קבצי `.fhrm`
- `ScalpECG/Test`: 30 קבצי `.fhrm`

#### `FS training python sources/`
- 8 קבצים: `3 x .h5`, `3 x .ipynb`, `1 x .py`, `1 x dataV8.zip`.
- בתוך `dataV8.zip`:
  - 963 קבצים.
  - `.dop`: 733
  - `.int`: 227
  - `.dat`: 3 (`isval.dat`, `isvalint.dat`, `DiffMF.dat`)

### 4.3 פורמטים בינאריים ותוויות (לפי `fhropen.m`)

לפי `data/FHRMA-master/fhropen.m`:

- `.dat`:
  - נקראים 2 ערוצים `uint16` ומנורמלים /100.
  - פלט: `FHR1`, `TOCO`.
- `.fhr`:
  - `timestamp` (`uint32`) + דגימות `FHR1`, `FHR2` (`uint16/4`) + `TOCO` (`uint8/2`).
- `.fhrm`:
  - `timestamp` (`uint32`) + `FHR1,FHR2,MHR` (`uint16/4`) + `TOCO` (`uint8/2`) + בית איכות `Q`.
  - מבית `Q` מפוענחות תוויות איכות:
    - `infos.Q1`
    - `infos.isECG1`
    - `infos.Q2`
    - `infos.isECG2`
    - `infos.Qm`
    - `infos.isTOCOMHR`
    - `infos.isIUP`

### 4.4 תיוגים נוספים ב-FS (False Signal)

לפי `FalseSigDetectDopMHR.m` ו-`FalseSigDetectScalp.m`:
- פלטי מודל:
  - `PFHR`: הסתברות False Signal ל-FHR.
  - `PMHR`: הסתברות False Signal ל-MHR (בדופלר/MHR).
- כלל סף בקוד: ערכים עם הסתברות `>0.5` מסומנים כ-False ומאופסים.

לפי `1_Prepare_packaged_data.ipynb`:
- בקבצי `dop*.dop` (5 ערוצים אחרי reshape):
  - `isFalse=(data[:,2] < 1)`
  - `docare=(data[:,2] != 1)`
  - `isFalseM=(data[:,3] < 1)`
  - `docareM=(data[:,3] != 1)`
- בקבצי `int*.int` (3 ערוצים אחרי reshape):
  - `isFalse=(data[:,1] < 1)`
  - `docare=(data[:,1] != 1)`

### 4.5 דוגמאות מבנה דאטה (FHRMA)

דוגמה 1: `.fhr` (HEX)
```text
00000000  00 00 00 00 A0 02 00 00 28 00 A0 02 00 00 28 00
00000010  A0 02 00 00 28 00 A8 02 00 00 28 00 A8 02 00 00
```

דוגמה 2: `.fhrm` (HEX)
```text
00000000  00 00 00 00 06 02 00 00 00 00 27 00 06 02 00 00
00000010  00 00 27 00 06 02 00 00 00 00 27 00 06 02 00 00
```

דוגמה 3: `.mat` (חתימת קובץ)
```text
MATLAB 5.0 MAT-file Platform: posix, Created on: ...
```

דוגמה 4: `.h5` (חתימת קובץ)
```text
89 48 44 46 0D 0A 1A 0A ...
```

דוגמה 5: `dataV8.zip`
```text
PK 03 04 ...
```

---

## 5. מילון תיוגים מרוכז (כדי לא לפספס)

### 5.1 תיוגי משימה/ספליט (record-level)
- `target` (ב-`CTGDL_norm_metadata.csv`)
- `test` (ב-`CTGDL_norm_metadata.csv`, `CTGDL_FHEMA_metadata.csv`)
- `dataset` (ב-`CTGDL_norm_metadata.csv`)

### 5.2 תיוגי שלב לידה/מבנה זמן
- `stage2_idx`, `stage2min`, `II.stage`, `Pos. II.st.`, `I.stage`, `stage1_min`, `stage2_min`

### 5.3 תיוגי מורפולוגיה/אנוטציה ברמת דגימה
- `baseline`, `acc`, `dec`
- `fhr_nan`, `uc_nan`
- `fhr_is_nan`, `uc_is_nan`
- `no_value_uc`

### 5.4 תיוגי איכות-אות/חיישן
- ב-CTU/FHRMA:
  - `Q1`, `Q2`, `Qm`
  - `isECG1`, `isECG2`, `isTOCOMHR`, `isIUP`
- ב-FS:
  - `docare`, `isFalse`, `docareM`, `isFalseM`

### 5.5 תיוגים קליניים (CTU `.hea`)
- `pH`, `BDecf`, `pCO2`, `BE`
- `Apgar1`, `Apgar5`
- `Gest. weeks`, `Weight(g)`, `Sex`
- `Age`, `Gravidity`, `Parity`
- `Diabetes`, `Hypertension`, `Preeclampsia`, `Liq. praecox`, `Pyrexia`, `Meconium`
- `Presentation`, `Induced`, `NoProgress`, `CK/KP`, `Deliv. type`
- `NICU days`, `Seizures`, `HIE`, `Intubation`, `Main diag.`, `Other diag.`
- `Rec. type`, `Sig2Birth`, `dbID`

---

## 6. פערים והערות חשובות

- `data/CTGDL/CTGDL_SPAM_metadata.csv` מפנה ל-294 קבצי `ctgdl_spam_*.csv`, אבל הקבצים עצמם לא נמצאים תחת `data/CTGDL` (רק מטא-דאטה וסקריפט יצירה).
- `data/ctu-chb.../ctu-chb.../ANNOTATIONS` קיים אך ריק (0 בתים).
- `analysis_results` בתיקיית CTU הוא סיכום חלקי (100 ו-20 רשומות), לא כיסוי מלא של 552 הרשומות.
- קבצי `.mat` ו-`.h5` זוהו ואומתו כפורמטים תקינים, אך מבנה פנימי מפורט של האובייקטים לא פורק כאן (נדרש כלי ייעודי כמו `scipy`/`h5py` או MATLAB).

