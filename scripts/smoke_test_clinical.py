#!/usr/bin/env python
"""
scripts/smoke_test_clinical.py — Verify clinical rule modules on real recordings
=================================================================================
Loads recordings from data/processed/ctu_uhb/ and runs all 5 clinical rule
modules through the clinical extractor. Checks for plausible output ranges
and ensures no exceptions or NaNs are produced.

Run this BEFORE the full evaluation to confirm the "second brain" works correctly.

Usage:
    python scripts/smoke_test_clinical.py
    python scripts/smoke_test_clinical.py --n 20   # test more recordings
    python scripts/smoke_test_clinical.py --id 1001 1002  # test specific recordings
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.clinical_extractor import (
    extract_clinical_features,
    CLINICAL_FEATURE_NAMES,
    N_CLINICAL_FEATURES,
)
from src.rules.baseline    import calculate_baseline
from src.rules.variability import calculate_variability
from src.rules.decelerations import detect_decelerations
from src.rules.sinusoidal  import detect_sinusoidal_pattern
from src.rules.tachysystole import detect_tachysystole

PROCESSED_DIR = ROOT / "data" / "processed" / "ctu_uhb"

# Expected value ranges (lo, hi) — anything outside → flagged as OOB
RANGES: dict[str, tuple[float, float]] = {
    "baseline_bpm":               (80.0,  200.0),
    "is_tachycardia":             (0.0,   1.0),
    "is_bradycardia":             (0.0,   1.0),
    "variability_amplitude_bpm":  (0.0,   60.0),
    "variability_category":       (0.0,   3.0),
    "n_late_decelerations":       (0.0,   500.0),
    "n_variable_decelerations":   (0.0,   500.0),
    "n_prolonged_decelerations":  (0.0,   100.0),
    "max_deceleration_depth_bpm": (0.0,   130.0),
    "sinusoidal_detected":        (0.0,   1.0),
    "tachysystole_detected":      (0.0,   1.0),
}

# Soft-warning ranges for clinical plausibility (not failure, just notable)
SOFT_RANGES: dict[str, tuple[float, float]] = {
    "baseline_bpm":               (100.0, 180.0),
    "variability_amplitude_bpm":  (1.0,   40.0),
    "max_deceleration_depth_bpm": (0.0,   80.0),
}


def denormalize(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fhr = signal[0] * 160.0 + 50.0
    uc  = signal[1] * 100.0
    return fhr, uc


def check_recording(npy_path: Path, verbose: bool = True) -> bool:
    rec_id = npy_path.stem

    # Load
    try:
        signal = np.load(npy_path, mmap_mode='r')
    except Exception as e:
        print(f"  [{rec_id}] ❌ Load failed: {e}")
        return False

    if signal.ndim != 2 or signal.shape[0] < 2:
        print(f"  [{rec_id}] ❌ Bad shape: {signal.shape}")
        return False

    fhr_bpm, uc_mmhg = denormalize(np.asarray(signal))
    duration_min = signal.shape[1] / (4.0 * 60.0)

    # Run combined extractor
    t0 = time.time()
    try:
        feats = extract_clinical_features(signal, fs=4.0)
    except Exception as e:
        print(f"  [{rec_id}] ❌ extract_clinical_features raised: {e}")
        return False
    elapsed = time.time() - t0

    if len(feats) != N_CLINICAL_FEATURES:
        print(f"  [{rec_id}] ❌ Wrong feature count: {len(feats)} (expected {N_CLINICAL_FEATURES})")
        return False

    passed = True
    issues = []
    rows   = []

    for name, val in zip(CLINICAL_FEATURE_NAMES, feats):
        lo, hi = RANGES[name]
        if np.isnan(val):
            issues.append(f"NaN in {name}")
            flag = "NaN ❌"
            passed = False
        elif not (lo <= val <= hi):
            issues.append(f"{name}={val:.3f} OOB [{lo},{hi}]")
            flag = "OOB ❌"
            passed = False
        elif name in SOFT_RANGES:
            slo, shi = SOFT_RANGES[name]
            flag = "WARN ⚠" if not (slo <= val <= shi) else "ok"
        else:
            flag = "ok"
        rows.append((name, val, flag))

    # Header line
    status = "✅" if passed else "❌"
    fhr_valid = fhr_bpm[~np.isnan(fhr_bpm)]
    fhr_range = f"[{fhr_valid.min():.0f},{fhr_valid.max():.0f}]" if len(fhr_valid) > 0 else "?"
    print(f"  [{rec_id}] {status}  dur={duration_min:.1f}min  FHR={fhr_range}bpm  {elapsed:.1f}s")

    if verbose:
        for name, val, flag in rows:
            print(f"       {name:<35} {val:>8.3f}  {flag}")
        print()

    if issues:
        for iss in issues:
            print(f"       ⚠ {iss}")

    return passed


def run_module_isolation_test(n: int = 3) -> bool:
    """Run each module individually and print results for a few recordings."""
    print("\n── Individual module isolation test ────────────────────────────────")
    npy_files = sorted(PROCESSED_DIR.glob("*.npy"))[:n]
    all_ok = True

    for npy in npy_files:
        rec_id = npy.stem
        try:
            signal = np.asarray(np.load(npy, mmap_mode='r'), dtype=float)
            fhr = signal[0] * 160.0 + 50.0
            uc  = signal[1] * 100.0
            fhr_safe = np.nan_to_num(fhr, nan=130.0)
            uc_safe  = np.nan_to_num(uc,  nan=0.0)

            b = calculate_baseline(fhr_safe, fs=4.0)
            v = calculate_variability(fhr_safe, fs=4.0)
            d = detect_decelerations(fhr_safe, uc_safe, fs=4.0)
            s = detect_sinusoidal_pattern(fhr_safe, fs=4.0)
            t = detect_tachysystole(uc_safe, fs=4.0)

            print(f"  {rec_id}:")
            print(f"    baseline={b.baseline_bpm:.1f}bpm  tachy={b.is_tachycardia:.0f}  brady={b.is_bradycardia:.0f}")
            print(f"    variability={v.amplitude_bpm:.1f}bpm  cat={v.category:.0f}")
            print(f"    late={d.n_late_decelerations:.0f}  var={d.n_variable_decelerations:.0f}  "
                  f"prolong={d.n_prolonged_decelerations:.0f}  depth={d.max_deceleration_depth_bpm:.1f}bpm")
            print(f"    sinusoidal={s.sinusoidal_detected:.0f}  tachysystole={t.tachysystole_detected:.0f}")
        except Exception as e:
            print(f"  {rec_id}: ❌ {e}")
            all_ok = False

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for clinical rule modules")
    parser.add_argument("--n",    type=int,  default=5,
                        help="Number of recordings to test (default 5)")
    parser.add_argument("--id",   nargs="+", default=None,
                        help="Specific recording IDs to test")
    parser.add_argument("--quiet", action="store_true",
                        help="Only show summary, not per-feature details")
    args = parser.parse_args()

    if not PROCESSED_DIR.exists():
        print(f"❌ Processed data directory not found: {PROCESSED_DIR}")
        print("   Make sure data/processed/ctu_uhb/*.npy files are present.")
        sys.exit(1)

    # Collect files to test
    if args.id:
        npy_files = [PROCESSED_DIR / f"{i}.npy" for i in args.id]
        missing = [p for p in npy_files if not p.exists()]
        if missing:
            print(f"❌ Missing: {[p.name for p in missing]}")
            sys.exit(1)
    else:
        all_files = sorted(PROCESSED_DIR.glob("*.npy"))
        if not all_files:
            print(f"❌ No .npy files found in {PROCESSED_DIR}")
            sys.exit(1)

        n = min(args.n, len(all_files))
        # Spread evenly across the FULL dataset for diversity
        rng = np.random.default_rng(42)
        total = len(all_files)
        spread = [int(total * k / max(n - 1, 1)) for k in range(n)]
        random_extra = rng.integers(0, total, size=n).tolist()
        indices = sorted(set(spread + random_extra))[:n]
        npy_files = [all_files[i] for i in indices]

    print(f"Clinical rules smoke test")
    print(f"  Recordings : {len(npy_files)}")
    print(f"  Data dir   : {PROCESSED_DIR.relative_to(ROOT)}")
    print(f"  Features   : {N_CLINICAL_FEATURES} clinical features per recording")
    print()

    verbose = not args.quiet
    n_pass  = 0
    n_fail  = 0
    t_total = time.time()

    for npy_path in npy_files:
        ok = check_recording(npy_path, verbose=verbose)
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    elapsed_total = time.time() - t_total

    # Module isolation test (always on 3 recordings)
    run_module_isolation_test(n=min(3, len(npy_files)))

    print(f"\n{'='*60}")
    print(f"SMOKE TEST RESULT: {n_pass}/{len(npy_files)} passed  "
          f"({n_fail} failed)  total={elapsed_total:.1f}s")
    if n_fail == 0:
        print("✅ All clinical rule modules are working correctly.")
        print("   Ready to run: python scripts/local_eval_cpu.py")
    else:
        print("❌ Some modules returned unexpected values.")
        print("   Investigate before running the full evaluation.")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
