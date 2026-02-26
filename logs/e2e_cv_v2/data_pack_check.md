# Data Pack Check — A1 Report

> Agent: A1 Data Pack & GitHub Transport
> Date: 2026-02-25
> Verified by: A0 (continuation after previous session)

---

## Status: PASS

All checks passed. `data_processed.zip` is accessible, intact, and contains the expected files.

---

## ZIP File Details

| Property | Value |
|----------|-------|
| File path (local) | `data_processed.zip` |
| Size | 23,928,982 bytes (22.82 MB) |
| SHA-256 | `7ca2982cf24b34c8e65f2d20e83dd00a6b27dbe11a6ef2457a171e9095ef6342` |
| ZIP integrity | **PASS** (testzip: no bad files) |

---

## GitHub URL

| Property | Value |
|----------|-------|
| URL | `https://raw.githubusercontent.com/ArielShamay/SentinelFatal2/master/data_processed.zip` |
| HTTP status | **200 OK** |
| Git commit | `883c31958159ccd9e4e947622f1e129761f59417` |

---

## Archive Contents

| Expected | Found | Status |
|----------|-------|--------|
| `processed/ctu_uhb/*.npy` (552 files) | 552 | ✅ PASS |
| `processed/fhrma/*.npy` (135 files) | 135 | ✅ PASS |
| `processed/ctu_uhb_clinical_full.csv` | present | ✅ PASS |
| Total entries in ZIP | 690 | ✅ (552 + 135 + 2 gitkeep + 1 csv) |

---

## Colab Download Command

```python
# Cell: Download data
import os, urllib.request

ZIP_URL = "https://raw.githubusercontent.com/ArielShamay/SentinelFatal2/master/data_processed.zip"
ZIP_PATH = "/content/SentinelFatal2/data_processed.zip"

if not os.path.exists(ZIP_PATH):
    print("Downloading data_processed.zip...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"Downloaded: {os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB")
else:
    print("ZIP already exists — skipping download.")
```

---

## Fallback Plan (if URL becomes unavailable)

1. Create new ZIP locally:
   ```powershell
   Compress-Archive -Path "data\processed" -DestinationPath "data_processed.zip" -Force
   ```
2. Push to GitHub:
   ```bash
   git add data_processed.zip && git commit -m "refresh data zip" && git push
   ```
3. Update URL in this document and in `logs/e2e_cv_v2/run_manifest.md`.
