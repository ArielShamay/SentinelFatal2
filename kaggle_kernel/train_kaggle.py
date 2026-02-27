"""
train_kaggle.py
===============
Kaggle kernel entrypoint for SentinelFatal2 E2E CV v3.

Steps:
  1. Clone the SentinelFatal2 repo from GitHub
  2. Install any missing Python dependencies
  3. Change working directory to the repo root
  4. Execute train_azure.py in-place
     (it handles data download, all 5 folds, outputs to REPO_DIR/results|checkpoints|logs)
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL  = "https://github.com/ArielShamay/SentinelFatal2.git"
REPO_DIR  = Path("/kaggle/working/SentinelFatal2")

# ── 1. Clone repo ──────────────────────────────────────────────────────────
if REPO_DIR.exists():
    print("[REPO] Already cloned — pulling latest changes ...")
    subprocess.check_call(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
else:
    print(f"[REPO] Cloning {REPO_URL} ...")
    subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, str(REPO_DIR)])

print(f"[REPO] Ready at {REPO_DIR}")

# ── 2. Install dependencies ────────────────────────────────────────────────
print("[DEPS] Installing dependencies ...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.2.0",
    "scikit-learn>=1.3",
    "pandas>=2.0",
    "PyYAML>=6.0",
    "scipy>=1.11",
])
print("[DEPS] Done.")

# ── 3. Switch working directory to repo root ───────────────────────────────
os.chdir(str(REPO_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

print(f"[CWD] {os.getcwd()}")

# ── 4. Run the training script ─────────────────────────────────────────────
train_script = REPO_DIR / "azure_ml" / "train_azure.py"
print(f"[RUN] Executing {train_script} ...")
print("=" * 60)

exec(train_script.read_text(), {"__name__": "__main__", "__file__": str(train_script)})
