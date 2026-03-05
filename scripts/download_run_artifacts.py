#!/usr/bin/env python
"""
scripts/download_run_artifacts.py — Download Azure ML job artifacts
====================================================================
Downloads checkpoints, results, and logs from a completed Azure ML job.
Uses the same VS Code credential as setup_and_submit.py — no az CLI needed.

Usage:
    python scripts/download_run_artifacts.py --run-id busy_tiger_ds38z13x0g
    python scripts/download_run_artifacts.py          # latest completed job
"""

from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUBSCRIPTION_ID = "02b4b69d-dd14-4e79-b35f-de906edb6b15"
RESOURCE_GROUP  = "sentinelfatal2-rg"
WORKSPACE_NAME  = "sentinelfatal2-aml"


def _get_credential():
    """Try Azure CLI -> VS Code (with timeout) -> DeviceCode fallback."""
    from azure.identity import (
        AzureCliCredential,
        VisualStudioCodeCredential,
        DeviceCodeCredential,
    )
    SCOPE   = "https://management.azure.com/.default"
    TIMEOUT = 20

    for name, cred_cls in [
        ("Azure CLI", AzureCliCredential),
        ("VS Code",   VisualStudioCodeCredential),
    ]:
        try:
            cred = cred_cls()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(cred.get_token, SCOPE).result(timeout=TIMEOUT)
            print(f"[AUTH] Authenticated via {name}.")
            return cred
        except concurrent.futures.TimeoutError:
            print(f"[AUTH] {name} timed out after {TIMEOUT}s.")
        except Exception as e:
            print(f"[AUTH] {name} failed: {e}")

    print("\n[AUTH] === DEVICE CODE LOGIN ===")
    print("[AUTH] Open the URL shown below and enter the code.\n")
    return DeviceCodeCredential()


def _copy_dir(src: Path, dst: Path, label: str) -> int:
    """Copy src -> dst, return count of copied files."""
    if not src.exists():
        print(f"[WARN] {label}: source not found at {src}")
        return 0
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    files = list(dst.rglob("*"))
    files = [f for f in files if f.is_file()]
    print(f"[OK] {label}: {len(files)} files -> {dst.relative_to(ROOT)}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Azure ML job artifacts")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Job name/run-id (e.g. busy_tiger_ds38z13x0g). Default: latest completed.",
    )
    parser.add_argument(
        "--version", type=str, default="v7",
        help="Version tag for local directories (default: v7). "
             "Checkpoints go to checkpoints/e2e_cv_{version}/",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Download base directory. Default: logs/e2e_cv_{version}/azure_job",
    )
    args = parser.parse_args()

    from azure.ai.ml import MLClient

    cred = _get_credential()
    ml   = MLClient(cred, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

    # ── Resolve run ID ─────────────────────────────────────────────────────────
    if args.run_id:
        run_id = args.run_id
    else:
        print("Searching for latest completed job...")
        jobs      = list(ml.jobs.list())
        completed = [j for j in jobs if getattr(j, "status", "") == "Completed"]
        if not completed:
            print("[ERR] No completed jobs found. Specify --run-id explicitly.")
            sys.exit(1)
        completed.sort(
            key=lambda j: getattr(j.creation_context, "created_at", ""), reverse=True
        )
        run_id = completed[0].name
        print(f"Latest completed job: {run_id}")

    version = args.version
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / "logs" / f"e2e_cv_{version}" / "azure_job"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download ───────────────────────────────────────────────────────────────
    print(f"\nDownloading all artifacts for job: {run_id}")
    print(f"  -> {out_dir.relative_to(ROOT)}\n")

    ml.jobs.download(
        name=run_id,
        download_path=str(out_dir),
        all=True,
    )

    print(f"\n[OK] Azure download complete.\n")

    # ── Locate checkpoints (Azure ML path can vary) ────────────────────────────
    # Try canonical path first, then fall back to recursive search
    ckpt_src = out_dir / "artifacts" / "outputs" / "checkpoints"
    if not ckpt_src.exists():
        found = list(out_dir.rglob("checkpoints"))
        ckpt_src = next((p for p in found if p.is_dir()), ckpt_src)
        if ckpt_src.exists():
            print(f"[INFO] Checkpoints found at non-standard path: {ckpt_src.relative_to(out_dir)}")

    ckpt_dst = ROOT / "checkpoints" / f"e2e_cv_{version}"
    n_ckpts = _copy_dir(ckpt_src, ckpt_dst, f"Checkpoints -> checkpoints/e2e_cv_{version}")

    if n_ckpts > 0:
        pts = sorted(ckpt_dst.rglob("*.pt"))
        print(f"   {len(pts)} .pt files found:")
        for p in pts:
            mb = p.stat().st_size / (1024 ** 2)
            print(f"   {p.relative_to(ROOT)}  ({mb:.1f} MB)")

    # ── Locate results (same fallback logic) ───────────────────────────────────
    results_src = out_dir / "artifacts" / "outputs" / "results"
    if not results_src.exists():
        found = list(out_dir.rglob("results"))
        results_src = next((p for p in found if p.is_dir()), results_src)
        if results_src.exists():
            print(f"[INFO] Results found at non-standard path: {results_src.relative_to(out_dir)}")

    results_dst = ROOT / "results" / f"e2e_cv_{version}"
    n_results = _copy_dir(results_src, results_dst, f"Results -> results/e2e_cv_{version}")

    if n_results > 0:
        csvs = sorted(results_dst.glob("*.csv")) + sorted(results_dst.glob("*.json"))
        for f in csvs:
            print(f"   {f.relative_to(ROOT)}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Job:     {run_id}")
    print(f"Ckpts:   checkpoints/e2e_cv_{version}/   ({n_ckpts} files)")
    print(f"Results: results/e2e_cv_{version}/        ({n_results} files)")
    print(f"\nStudio: https://ml.azure.com/runs/{run_id}"
          f"?wsid=/subscriptions/{SUBSCRIPTION_ID}"
          f"/resourcegroups/{RESOURCE_GROUP}/workspaces/{WORKSPACE_NAME}")
    print(f"\nNext step:")
    print(f"  python scripts/smoke_test_clinical.py")
    print(f"  python scripts/local_eval_cpu.py --ckpt checkpoints/e2e_cv_{version}/fold0/best_finetune.pt")


if __name__ == "__main__":
    main()
