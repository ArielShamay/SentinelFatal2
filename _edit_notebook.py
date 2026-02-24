"""
Script to edit 07_colab_e2e_cv_launch.ipynb:
1. Update markdown header (cell plan table)
2. Update Cell 5: remove --skip-pretrain, add cleanup
3. Insert new Cell 6: Fresh Pretrain
4. Update old Cell 6 -> Cell 7: add pretrain check, keep --skip-pretrain
5. Delete empty cell (#VSC-8f0931b1)
6. Update morning check -> Cell 8
7. Delete debug cell (#VSC-8125fd6f)
"""
import json, uuid, copy

NB_PATH = r"notebooks\07_colab_e2e_cv_launch.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
print(f"Original cell count: {len(cells)}")
for i, c in enumerate(cells):
    ct = c["cell_type"]
    src_preview = "".join(c["source"][:2]).strip()[:80]
    print(f"  [{i}] {ct:8s} | {src_preview}")

# ── Helper ───────────────────────────────────────────────────────────────────
def make_code_cell(source_str: str, cell_id: str = None) -> dict:
    """Create a new code cell from a multi-line string."""
    if cell_id is None:
        cell_id = str(uuid.uuid4())[:8]
    lines = source_str.split("\n")
    # Convert to list of lines with \n except last
    source = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        source.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 1. UPDATE MARKDOWN HEADER — fix table to show 8 cells
# ══════════════════════════════════════════════════════════════════════════════
md_source = [
    "# Notebook 07 \u2014 Colab E2E CV Launch (VS Code Extension)\n",
    "## SentinelFatal2 \u2014 End-to-End 5-Fold Cross-Validation\n",
    "\n",
    "**\u05d4\u05e8\u05e5 \u05e0\u05d5\u05d8\u05d1\u05d5\u05e7 \u05d6\u05d4 \u05d3\u05e8\u05da \u05ea\u05d5\u05e1\u05e3 Google Colab \u05d1-VS Code \u05e2\u05dd Kernel \u05de\u05e1\u05d5\u05d2 T4 GPU.**\n",
    "\n",
    "### \u05d0\u05d9\u05da \u05dc\u05d7\u05d1\u05e8:\n",
    "1. \u05e4\u05ea\u05d7 \u05e0\u05d5\u05d8\u05d1\u05d5\u05e7 \u05d6\u05d4 \u05d1-VS Code\n",
    "2. \u05dc\u05d7\u05e5 **Select Kernel** (\u05e4\u05d9\u05e0\u05d4 \u05e9\u05de\u05d0\u05dc\u05d9\u05ea \u05e2\u05dc\u05d9\u05d5\u05e0\u05d4) \u2192 **Colab** \u2192 **New Colab Server** \u2192 \u05d1\u05d7\u05e8 T4 GPU\n",
    "3. \u05d4\u05e8\u05e5 \u05ea\u05d0\u05d9\u05dd 1\u20138 \u05dc\u05e4\u05d9 \u05d4\u05e1\u05d3\u05e8\n",
    "\n",
    "| \u05ea\u05d0 | \u05e4\u05e2\u05d5\u05dc\u05d4 | \u05d6\u05de\u05df |\n",
    "|----|--------|-----|\n",
    "| 1 | **GPU CHECK** \u2014 \u05d0\u05e9\u05e8 T4 \u05dc\u05e4\u05e0\u05d9 \u05d4\u05de\u05e9\u05da | \u05e9\u05e0\u05d9\u05d5\u05ea |\n",
    "| 2 | Clone repo \u05dc\u05e9\u05e8\u05ea Colab | ~1 \u05d3\u05e7\u05d4 |\n",
    "| 3 | \u05d4\u05d5\u05e8\u05d3\u05ea \u05e0\u05ea\u05d5\u05e0\u05d9\u05dd \u05de-GitHub \u05d0\u05dd \u05d7\u05e1\u05e8\u05d9\u05dd | ~1\u20133 \u05d3\u05e7\u05d5\u05ea |\n",
    "| 4 | \u05d4\u05ea\u05e7\u05e0\u05ea \u05ea\u05dc\u05d5\u05d9\u05d5\u05ea | ~1 \u05d3\u05e7\u05d4 |\n",
    "| 5 | **Dry-run** \u2014 \u05d1\u05d3\u05d5\u05e7 \u05e9\u05d4\u05db\u05dc \u05e2\u05d5\u05d1\u05d3 \u05e2\u05dc GPU | ~5 \u05d3\u05e7\u05d5\u05ea |\n",
    "| 6 | **Fresh Pretrain** \u2014 MAE \u05e2\u05dc 687 \u05d4\u05e7\u05dc\u05d8\u05d5\u05ea | ~15\u201325 \u05d3\u05e7\u05d5\u05ea |\n",
    "| 7 | **\u05e8\u05d9\u05e6\u05d4 \u05de\u05dc\u05d0\u05d4** \u2014 5-fold CV (\u05e2\u05dd skip-pretrain) | ~60\u201390 \u05d3\u05e7\u05d5\u05ea |\n",
    "| 8 | \u05d1\u05d3\u05d9\u05e7\u05ea \u05ea\u05d5\u05e6\u05d0\u05d5\u05ea \u05d1\u05d1\u05d5\u05e7\u05e8 | \u05e9\u05e0\u05d9\u05d5\u05ea |\n",
    "\n",
    "> \u26a0\ufe0f **\u05d4\u05e9\u05d0\u05e8 VS Code \u05e4\u05ea\u05d5\u05d7 \u05db\u05dc \u05d4\u05dc\u05d9\u05dc\u05d4.**\n",
    "> \u05d3\u05e8\u05da \u05d4\u05ea\u05d5\u05e1\u05e3, \u05d4\u05e8\u05d9\u05e6\u05d4 \u05d7\u05d9\u05d4 \u05db\u05dc \u05e2\u05d5\u05d3 VS Code \u05de\u05d7\u05d5\u05d1\u05e8 \u05dc\u05e9\u05e8\u05ea.\n",
    "> \u05d0\u05dd VS Code \u05e0\u05e1\u05d2\u05e8 \u2014 \u05d4\u05e8\u05d9\u05e6\u05d4 \u05e0\u05e2\u05e6\u05e8\u05ea. \u05d0\u05d9\u05df \u05d1\u05e2\u05d9\u05d4: \u05d4\u05e1\u05e7\u05e8\u05d9\u05e4\u05d8 \u05ea\u05d5\u05de\u05da \u05d1-resume \u05d0\u05d5\u05d8\u05d5\u05de\u05d8\u05d9 \u2014 \u05e4\u05e9\u05d5\u05d8 \u05d4\u05e8\u05e5 \u05e9\u05d5\u05d1 \u05d0\u05ea Cell 7.\n",
]
cells[0]["source"] = md_source
print("\n[1] Updated markdown header")

# ══════════════════════════════════════════════════════════════════════════════
# 2. UPDATE CELL 5: remove --skip-pretrain, keep dry-run as-is
# ══════════════════════════════════════════════════════════════════════════════
cell5_source = r'''# ── Cell 5: DRY-RUN — Verify full pipeline works on GPU (~2 min) ─────────────
# Runs 2 folds with max_batches=2 (StratifiedKFold requires n_splits>=2).
# --skip-pretrain: copies shared best_pretrain.pt instead of retraining.
# Output streams live line-by-line.
# If this cell fails, DO NOT proceed to Cell 6.

import os, sys, subprocess, shutil
from pathlib import Path

REPO_DIR = Path("/content/SentinelFatal2")
os.chdir(REPO_DIR)

# ── Clear stale dry-run artifacts ────────────────────────────────────────────
for stale in [
    REPO_DIR / "checkpoints" / "e2e_cv",
    REPO_DIR / "results"     / "e2e_cv",
    REPO_DIR / "logs"        / "e2e_cv",
]:
    if stale.exists():
        shutil.rmtree(stale)
        print(f"  Cleared: {stale.name}/")

cmd = [
    sys.executable, "scripts/run_e2e_cv.py",
    "--device", DEVICE,
    "--dry-run",
    "--folds", "2",
    "--force-folds",
    "--skip-pretrain",
    "--seed", "42",
]

print("Starting dry-run (2 folds, max_batches=2, --skip-pretrain)...", flush=True)
print("Estimated time: ~2 minutes on T4 GPU", flush=True)
print("-" * 60, flush=True)

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
proc.wait()

print("-" * 60, flush=True)
if proc.returncode == 0:
    print("✅ Dry-run PASSED — safe to run Cell 6.", flush=True)
else:
    print(f"❌ Dry-run FAILED (exit code {proc.returncode}).", flush=True)
    print("   Fix errors above before running Cell 6.", flush=True)
'''

# Find Cell 5 by id
cell5_idx = None
for i, c in enumerate(cells):
    if c.get("id") == "09db5f35" or "Cell 5:" in "".join(c.get("source", [])):
        cell5_idx = i
        break
if cell5_idx is None:
    # Search by content
    for i, c in enumerate(cells):
        if "DRY-RUN" in "".join(c.get("source", [])):
            cell5_idx = i
            break

assert cell5_idx is not None, "Could not find Cell 5!"
print(f"[2] Cell 5 found at index {cell5_idx}")

# Replace source
lines = cell5_source.strip().split("\n")
cells[cell5_idx]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
cells[cell5_idx]["outputs"] = []
cells[cell5_idx]["execution_count"] = None
print(f"     Updated Cell 5 source ({len(lines)} lines)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. INSERT NEW CELL 6: Fresh Pretrain
# ══════════════════════════════════════════════════════════════════════════════
cell6_source = r'''# ── Cell 6: FRESH PRETRAIN — Channel-Asymmetric MAE (~15-25 min) ─────────────
# Trains pretrain from scratch on ALL 687 recordings with improved config:
#   - ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
#   - max_epochs=300 (with early stopping patience=20)
#   - mask_ratio=0.4, window_stride=900
# This is self-supervised (no labels), so using all data causes NO leakage.
# The resulting best_pretrain.pt will be used by Cell 7 (--skip-pretrain).

import os, sys, subprocess, shutil
from pathlib import Path

REPO_DIR = Path("/content/SentinelFatal2")
os.chdir(REPO_DIR)

# ── Clear old pretrain artifacts ─────────────────────────────────────────────
old_ckpt_dir = REPO_DIR / "checkpoints" / "pretrain"
old_log = REPO_DIR / "logs" / "pretrain_loss.csv"

if old_ckpt_dir.exists():
    shutil.rmtree(old_ckpt_dir)
    print("  Cleared old checkpoints/pretrain/")
if old_log.exists():
    old_log.unlink()
    print("  Cleared old logs/pretrain_loss.csv")

# ── Clear dry-run CV artifacts ───────────────────────────────────────────────
for stale in [
    REPO_DIR / "checkpoints" / "e2e_cv",
    REPO_DIR / "results"     / "e2e_cv",
    REPO_DIR / "logs"        / "e2e_cv",
]:
    if stale.exists():
        shutil.rmtree(stale)
        print(f"  Cleared: {stale.name}/")

cmd = [
    sys.executable, "src/train/pretrain.py",
    "--config", "config/train_config.yaml",
    "--device", DEVICE,
]

print()
print("=" * 60)
print("  SentinelFatal2 — Fresh Pretrain (Channel-Asymmetric MAE)")
print(f"  Device          : {DEVICE}")
print("  Recordings      : ALL 687 (self-supervised, no leakage)")
print("  Scheduler       : ReduceLROnPlateau(patience=5)")
print("  Max epochs      : 300 (early stop patience=20)")
print("  Mask ratio      : 0.4")
print("  Expected time   : ~15-25 min on T4 GPU")
print("=" * 60)
print()

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
proc.wait()

print()
print("=" * 60)
if proc.returncode == 0:
    ckpt = REPO_DIR / "checkpoints" / "pretrain" / "best_pretrain.pt"
    if ckpt.exists():
        size_mb = ckpt.stat().st_size / 1e6
        print(f"✅ Fresh pretrain complete!")
        print(f"   Checkpoint: {ckpt}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   → Safe to run Cell 7 (full CV with --skip-pretrain)")
    else:
        print("⚠️  Pretrain finished but best_pretrain.pt not found!")
        print("   Check logs/pretrain_loss.csv for details.")
else:
    print(f"❌ Pretrain FAILED (exit code {proc.returncode}).")
    print("   Fix errors above before running Cell 7.")
print("=" * 60)
'''

new_cell6 = make_code_cell(cell6_source.strip())
# Insert after Cell 5
cells.insert(cell5_idx + 1, new_cell6)
print(f"[3] Inserted new Cell 6 (Fresh Pretrain) at index {cell5_idx + 1}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. UPDATE OLD CELL 6 -> CELL 7 (now at cell5_idx + 2)
# ══════════════════════════════════════════════════════════════════════════════
cell7_idx = cell5_idx + 2  # Original Cell 6 shifted by 1

cell7_source = r'''# ── Cell 7: FULL 5-FOLD CV (~60-90 min with stride=24) ───────────────────────
# S14 Improvements vs previous run (AUC=0.565):
#   --skip-pretrain : use FRESH best_pretrain.pt from Cell 6 (687 recs, ReduceLROnPlateau)
#   lr_backbone=5e-5: 5× higher backbone LR (was 1e-5, backbone nearly frozen)
#   warmup=5 epochs : linear LR warmup to protect pretrained weights
#   EMA val_AUC     : smooth checkpointing (beta=0.8) reduces noisy early stop
#   StandardScaler  : feature normalization before LR (was raw features)
#   C=0.1           : tighter regularization (was 0.5)
#   stride=24       : 1 patch step (was 60), ~3× more windows for feature extraction
#   record-level features: n_alert_segments + alert_fraction (6 features total)
#
# Expected time: ~60-90 min on T4 GPU (stride=24 is ~2.5× slower than stride=60)
# ⚠️  Keep VS Code open. If it disconnects, re-run Cell 7 — auto-resumes.

import os, sys, subprocess, shutil
from pathlib import Path

REPO_DIR = Path("/content/SentinelFatal2")
LOG_FILE = REPO_DIR / "logs" / "e2e_cv_master.log"
(REPO_DIR / "logs").mkdir(parents=True, exist_ok=True)
os.chdir(REPO_DIR)

# ── Verify fresh pretrain checkpoint exists ──────────────────────────────────
pretrain_ckpt = REPO_DIR / "checkpoints" / "pretrain" / "best_pretrain.pt"
if not pretrain_ckpt.exists():
    raise FileNotFoundError(
        f"best_pretrain.pt not found at {pretrain_ckpt}\n"
        "Run Cell 6 (Fresh Pretrain) first!"
    )
size_mb = pretrain_ckpt.stat().st_size / 1e6
print(f"✅ Fresh pretrain checkpoint found ({size_mb:.1f} MB)")

# ── Clear old fold artifacts so S14 improvements actually run ────────────────
for stale in [
    REPO_DIR / "checkpoints" / "e2e_cv",
    REPO_DIR / "results"     / "e2e_cv",
    REPO_DIR / "logs"        / "e2e_cv",
]:
    if stale.exists():
        shutil.rmtree(stale)
        print(f"  Cleared stale artifacts: {stale.name}/")
print()

cmd = [
    sys.executable, "scripts/run_e2e_cv.py",
    "--device", DEVICE,
    "--force-folds",
    "--folds", "5",
    "--stride", "24",
    "--seed", "42",
    "--skip-pretrain",
]

print("=" * 60)
print("  SentinelFatal2 — Full 5-Fold E2E CV (S14 + Fresh Pretrain)")
print(f"  Device        : {DEVICE}")
print("  skip-pretrain : FRESH best_pretrain.pt from Cell 6")
print("  lr_backbone   : 5e-5 (was 1e-5)")
print("  warmup        : 5 epochs linear ramp")
print("  EMA beta      : 0.8 (smooth checkpointing)")
print("  StandardScaler: ON (before LR)")
print("  LR C          : 0.1 (was 0.5)")
print("  stride        : 24 (1 patch step, was 60)")
print("  features      : 6 (4 segment + 2 record-level)")
print("  Expected time : ~60-90 min on T4 GPU")
print("=" * 60)
print()

log_lines = []
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    line_stripped = line.rstrip()
    print(line_stripped, flush=True)
    log_lines.append(line)
proc.wait()

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.writelines(log_lines)

print()
print("=" * 60)
if proc.returncode == 0:
    print("✅ ALL FOLDS COMPLETE — run Cell 8 to see final results.")
else:
    print(f"⚠️  Finished with exit code {proc.returncode}.")
    print("   Some folds may have failed. Run Cell 8 to check partial results.")
print("=" * 60)
'''

lines = cell7_source.strip().split("\n")
cells[cell7_idx]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
cells[cell7_idx]["outputs"] = []
cells[cell7_idx]["execution_count"] = None
print(f"[4] Updated Cell 7 (Full CV) at index {cell7_idx}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. DELETE EMPTY CELL (was right after old Cell 6)
# ══════════════════════════════════════════════════════════════════════════════
# The empty cell is now at cell7_idx + 1
empty_idx = cell7_idx + 1
empty_src = "".join(cells[empty_idx]["source"]).strip()
if len(empty_src) == 0:
    del cells[empty_idx]
    print(f"[5] Deleted empty cell at index {empty_idx}")
else:
    print(f"[5] WARNING: cell at index {empty_idx} is not empty: '{empty_src[:60]}'")
    # Try to find it
    for i, c in enumerate(cells):
        if "".join(c.get("source", [])).strip() == "":
            del cells[i]
            print(f"     → Found and deleted empty cell at index {i}")
            if i < empty_idx:
                empty_idx = i  # adjust
            break

# ══════════════════════════════════════════════════════════════════════════════
# 6. UPDATE MORNING CHECK -> CELL 8
# ══════════════════════════════════════════════════════════════════════════════
# Find the morning check cell
morning_idx = None
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "MORNING CHECK" in src:
        morning_idx = i
        break

assert morning_idx is not None, "Could not find morning check cell!"

cell8_source = r'''# ── Cell 8: MORNING CHECK — Read Results ─────────────────────────────────────
# Run this cell in the morning to check on the run.
# Can be run safely at any time — it only reads files, never modifies anything.

import os, pandas as pd
from pathlib import Path

REPO_DIR = Path("/content/SentinelFatal2")
LOG_FILE = REPO_DIR / "logs" / "e2e_cv_master.log"
PROGRESS = REPO_DIR / "logs" / "e2e_cv_progress.csv"
REPORT   = REPO_DIR / "results" / "e2e_cv_final_report.csv"

print("=" * 60)
print("  SentinelFatal2 — E2E CV Results Check")
print("=" * 60)

# Is the process still running?
pid_check = os.popen("pgrep -f run_e2e_cv.py").read().strip()
if pid_check:
    print(f"\n  Process still running (PID: {pid_check})")
else:
    print("\n  Process has finished (or was interrupted).")

# Last 40 lines of log
print("\n-- Last 40 lines of log " + "-" * 36)
os.system(f"tail -40 {LOG_FILE}")

# Per-fold progress
print("\n-- Per-fold progress " + "-" * 39)
if PROGRESS.exists():
    df_prog = pd.read_csv(PROGRESS)
    print(df_prog.to_string(index=False))
else:
    print("  (no progress file yet — run may still be in fold 0)")

# Final report
print("\n-- Final report " + "-" * 44)
if REPORT.exists():
    df_rep = pd.read_csv(REPORT)
    print(df_rep.to_string(index=False))
    print("\n✅ Run COMPLETE.")
else:
    print("  (no final report yet — run has not completed all 5 folds)")

print("\n-- ROC curve plot " + "-" * 42)
img_path = REPO_DIR / "docs" / "images" / "e2e_cv.png"
if img_path.exists():
    from IPython.display import Image, display
    display(Image(str(img_path)))
else:
    print("  (plot not yet generated)")
print("=" * 60)
'''

lines = cell8_source.strip().split("\n")
cells[morning_idx]["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]
cells[morning_idx]["outputs"] = []
cells[morning_idx]["execution_count"] = None
print(f"[6] Updated Cell 8 (Morning Check) at index {morning_idx}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. DELETE DEBUG CELL (the last cell with "debug" content)
# ══════════════════════════════════════════════════════════════════════════════
debug_idx = None
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    if "Show NEW log and results" in src or "New fold checkpoints" in src:
        debug_idx = i
        break

if debug_idx is not None:
    del cells[debug_idx]
    print(f"[7] Deleted debug cell at index {debug_idx}")
else:
    print("[7] No debug cell found to delete")

# ══════════════════════════════════════════════════════════════════════════════
# Clear ALL outputs from all cells
# ══════════════════════════════════════════════════════════════════════════════
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nFinal cell count: {len(cells)}")
for i, c in enumerate(cells):
    ct = c["cell_type"]
    src_preview = "".join(c["source"][:2]).strip()[:80]
    print(f"  [{i}] {ct:8s} | {src_preview}")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✅ Saved to {NB_PATH}")
