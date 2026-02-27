#!/usr/bin/env python
"""
azure_ml/setup_and_submit.py
=============================
Run this script ONCE from VS Code to:
  1. Authenticate with your Azure for Students subscription
  2. Create an Azure ML Workspace (if it doesn't exist)
  3. Create a GPU compute cluster  (Standard_NC4as_T4_v3 — NVIDIA T4, ~$0.50/hr)
  4. Submit the SentinelFatal2 5-Fold E2E CV training job

Prerequisites
-------------
    pip install azure-ai-ml azure-identity azure-mgmt-resource

    Sign into Azure in VS Code first:
      Ctrl+Shift+P → "Azure: Sign In"

Usage
-----
    # From repo root:
    python azure_ml/setup_and_submit.py

    # To monitor the job after submission:
    python azure_ml/setup_and_submit.py --monitor

    # Dry-run (creates workspace + cluster, but does NOT submit job):
    python azure_ml/setup_and_submit.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
SUBSCRIPTION_ID = "02b4b69d-dd14-4e79-b35f-de906edb6b15"  # Azure for Students
RESOURCE_GROUP  = "sentinelfatal2-rg"
WORKSPACE_NAME  = "sentinelfatal2-aml"
LOCATION        = "francecentral"  # Paris — policy-allowed region WITH T4 GPU (NC4as_T4_v3)

# GPU compute settings
COMPUTE_NAME    = "gpu-t4-cluster"
VM_SIZE         = "Standard_NC4as_T4_v3"   # 1× NVIDIA T4, 4 vCPUs, 28 GB RAM (~$0.50/hr spot)
MIN_NODES       = 0    # scale-to-zero when idle (no cost when not training)
MAX_NODES       = 1
IDLE_TIMEOUT    = 300  # seconds before auto-deallocate (5 min)

# Azure ML curated base image with CUDA 11.8 + PyTorch
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run",  action="store_true", help="Create infra, skip job submit")
    p.add_argument("--monitor",  action="store_true", help="Monitor existing job (supply --job-name)")
    p.add_argument("--job-name", type=str, default=None, help="Job name to monitor")
    p.add_argument("--low-priority", action="store_true", default=True,
                   help="Use low-priority (spot) VMs for ~60%% cost reduction (default: True)")
    p.add_argument("--dedicated", action="store_true",
                   help="Use dedicated VMs instead of spot (more expensive, more reliable)")
    return p.parse_args()


def _get_credential():
    """Try Azure CLI → VS Code → browser login, in that order."""
    from azure.identity import (
        AzureCliCredential,
        VisualStudioCodeCredential,
        InteractiveBrowserCredential,
    )
    SCOPE = "https://management.azure.com/.default"

    for name, cred_cls, kwargs in [
        ("Azure CLI",  AzureCliCredential,         {}),
        ("VS Code",    VisualStudioCodeCredential,  {}),
    ]:
        try:
            cred = cred_cls(**kwargs)
            cred.get_token(SCOPE)
            print(f"[AUTH] Authenticated via {name}.")
            return cred
        except Exception as e:
            print(f"[AUTH] {name} credential failed: {e}")

    print("[AUTH] Opening browser login ...")
    return InteractiveBrowserCredential(
        tenant_id="90373b7d-e0f5-41f4-bf72-c3c39a38bc80"
    )


def _ensure_resource_group(credential) -> None:
    """Create resource group if it doesn't exist."""
    from azure.mgmt.resource import ResourceManagementClient

    rg_client = ResourceManagementClient(credential, SUBSCRIPTION_ID)
    if rg_client.resource_groups.check_existence(RESOURCE_GROUP):
        print(f"[RG] Resource group '{RESOURCE_GROUP}' already exists.")
        return
    print(f"[RG] Creating resource group '{RESOURCE_GROUP}' in {LOCATION} ...")
    rg_client.resource_groups.create_or_update(
        RESOURCE_GROUP, {"location": LOCATION}
    )
    print(f"[RG] Created.")


def _ensure_workspace(ml_sub_client, args):
    """Create workspace if it doesn't exist. Returns workspace object."""
    from azure.core.exceptions import ResourceNotFoundError
    from azure.ai.ml.entities import Workspace

    try:
        ws = ml_sub_client.workspaces.get(WORKSPACE_NAME)
        print(f"[WORKSPACE] '{WORKSPACE_NAME}' already exists in {ws.location}.")
        return ws
    except (ResourceNotFoundError, Exception):
        pass

    print(f"[WORKSPACE] Creating '{WORKSPACE_NAME}' in {LOCATION} ...")
    print(f"            This may take 2–5 minutes ...")
    ws = ml_sub_client.workspaces.begin_create(
        Workspace(
            name=WORKSPACE_NAME,
            location=LOCATION,
            resource_group=RESOURCE_GROUP,
            description="SentinelFatal2 — fetal distress detection (PatchTST CTG)",
            tags={
                "project": "SentinelFatal2",
                "owner":   "ArielShamay",
            },
        )
    ).result()
    print(f"[WORKSPACE] Created: {ws.name}  (resource group: {RESOURCE_GROUP})")
    return ws


def _ensure_compute(ml_client, tier: str):
    """Create GPU compute cluster if it doesn't exist."""
    from azure.core.exceptions import ResourceNotFoundError
    from azure.ai.ml.entities import AmlCompute

    try:
        compute = ml_client.compute.get(COMPUTE_NAME)
        print(f"[COMPUTE] '{COMPUTE_NAME}' already exists ({compute.size}, "
              f"min={compute.min_instances}, max={compute.max_instances}).")
        return compute
    except (ResourceNotFoundError, Exception):
        pass

    print(f"[COMPUTE] Creating cluster '{COMPUTE_NAME}' ({VM_SIZE}, tier={tier}) ...")
    print(f"          min_nodes={MIN_NODES}, max_nodes={MAX_NODES}, "
          f"idle_timeout={IDLE_TIMEOUT}s")
    compute = AmlCompute(
        name=COMPUTE_NAME,
        size=VM_SIZE,
        min_instances=MIN_NODES,
        max_instances=MAX_NODES,
        idle_time_before_scale_down=IDLE_TIMEOUT,
        tier=tier,
    )
    try:
        result = ml_client.compute.begin_create_or_update(compute).result()
        print(f"[COMPUTE] Cluster created: {result.name}  state={result.provisioning_state}")
        return result
    except Exception as e:
        err = str(e)
        if "CoreQuota" in err or "quota" in err.lower():
            print("\n" + "=" * 60)
            print("GPU QUOTA INCREASE REQUIRED")
            print("=" * 60)
            print(f"\nYour Azure for Students subscription has 0 vCPU quota")
            print(f"for Azure ML managed compute in '{LOCATION}'.")
            print(f"\nTo request quota increase (~5 min, usually auto-approved):")
            print(f"\n  Option A — Azure ML Studio:")
            print(f"  https://ml.azure.com/quota?wsid=/subscriptions/{SUBSCRIPTION_ID}"
                  f"/resourceGroups/{RESOURCE_GROUP}/providers/"
                  f"Microsoft.MachineLearningServices/workspaces/{WORKSPACE_NAME}")
            print(f"\n  Option B — Azure Portal:")
            print(f"  https://portal.azure.com/#view/Microsoft_Azure_Capacity/"
                  f"QuotaMenuBlade/~/myQuotas")
            print(f"  Filter: Provider=Machine Learning, Region=France Central")
            print(f"  Search: 'NCASv3' → request 4 vCPUs (Low Priority)")
            print(f"\n  After approval: python azure_ml/setup_and_submit.py")
            print("=" * 60)
            raise SystemExit(1)
        raise


def _ensure_environment(ml_client):
    """Register (or update) the conda-based training environment.

    Always calls create_or_update so that changes to conda_env.yml are
    picked up automatically — Azure ML creates a new version only when
    the content hash differs.
    """
    from azure.ai.ml.entities import Environment

    env_name = "sentinelfatal2-env"
    conda_yml = REPO_ROOT / "azure_ml" / "conda_env.yml"

    print(f"[ENV] Registering/updating environment '{env_name}' ...")
    env = Environment(
        name=env_name,
        image=BASE_IMAGE,
        conda_file=str(conda_yml),
        description="SentinelFatal2 training: PyTorch 2.2+cu118 / NumPy<2 / CUDA 11.8",
    )
    registered = ml_client.environments.create_or_update(env)
    print(f"[ENV] Environment '{env_name}' version {registered.version} ready.")
    return f"{env_name}:{registered.version}"


def _ensure_data_asset(ml_client) -> str:
    """Upload data_processed.zip to Azure Blob as a registered Data Asset.

    Returns 'ctg-processed:<version>' reference string.
    Skips upload if the asset is already registered (idempotent).
    """
    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes

    data_name = "ctg-processed"
    zip_path  = REPO_ROOT / "data_processed.zip"

    # Check if already registered
    try:
        existing = ml_client.data.get(data_name, label="latest")
        print(f"[DATA] Data asset '{data_name}' already registered "
              f"(version {existing.version}) — skipping upload.")
        return f"{data_name}:{existing.version}"
    except Exception:
        pass

    size_mb = zip_path.stat().st_size // (1024 * 1024)
    print(f"[DATA] Uploading {zip_path.name} ({size_mb} MB) to Azure Blob ...")
    data = Data(
        path=str(zip_path),
        type=AssetTypes.URI_FILE,
        description="Preprocessed CTG .npy files for SentinelFatal2 training (552 recordings)",
        name=data_name,
        version="1",
    )
    registered = ml_client.data.create_or_update(data)
    print(f"[DATA] Registered: '{data_name}' version {registered.version}")
    return f"{data_name}:{registered.version}"


def _submit_job(ml_client, env_ref: str, data_ref: str, low_priority: bool):
    """Define and submit the training command job."""
    from azure.ai.ml import command, Input, Output
    from azure.ai.ml.constants import AssetTypes

    print("\n[JOB] Building job definition ...")

    job = command(
        code=str(REPO_ROOT),          # uploads repo source code (~50 MB, data excluded by .amlignore)
        command="python azure_ml/train_azure.py --data ${{inputs.data}}",
        inputs={
            "data": Input(
                type=AssetTypes.URI_FILE,
                path=f"azureml:{data_ref}",
                mode="download",      # Azure ML copies zip to local SSD before job starts
            ),
        },
        environment=env_ref,
        compute=COMPUTE_NAME,
        display_name="SentinelFatal2-E2E-CV-v3",
        description=(
            "5-Fold End-to-End Cross-Validation — PatchTST CTG fetal distress detection. "
            "Config A: cross_entropy loss, class_weight=[1.0, 3.9], patience=15, "
            "train_stride=120, val_every_5_epochs, num_workers=0."
        ),
        outputs={
            "results": Output(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml://datastores/workspaceblobstore/paths/sentinelfatal2/results/",
            ),
            "checkpoints": Output(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml://datastores/workspaceblobstore/paths/sentinelfatal2/checkpoints/",
            ),
            "logs": Output(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml://datastores/workspaceblobstore/paths/sentinelfatal2/logs/",
            ),
        },
        environment_variables={
            "PYTHONUNBUFFERED": "1",
        },
        tags={
            "project": "SentinelFatal2",
            "config":  "A",
            "n_folds": "5",
        },
    )

    print("[JOB] Submitting job to Azure ML ...")
    returned_job = ml_client.jobs.create_or_update(job)

    print("\n" + "=" * 60)
    print("JOB SUBMITTED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Job name   : {returned_job.name}")
    print(f"  Status     : {returned_job.status}")
    print(f"  Studio URL : {returned_job.studio_url}")
    print("=" * 60)
    print("\nTo monitor from VS Code terminal:")
    print(f"  python azure_ml/setup_and_submit.py --monitor --job-name {returned_job.name}")
    print("\nOr open the Studio URL in your browser to view logs in real-time.")

    return returned_job


def _stream_job(ml_client, job_name: str):
    """Stream job logs in real-time to the VS Code terminal."""
    print(f"\n[STREAM] Waiting for job '{job_name}' to start ...")
    print(f"[STREAM] Logs will appear here as the job runs. Press Ctrl+C to detach.\n")

    # Wait until job leaves Queued/Starting state before streaming
    for _ in range(60):  # up to 10 min
        job = ml_client.jobs.get(job_name)
        if job.status not in ("Queued", "NotStarted"):
            break
        print(f"  [STATUS] {job.status} — waiting for compute node ...")
        time.sleep(10)

    try:
        # ml_client.jobs.stream() streams stdout/stderr in real-time
        ml_client.jobs.stream(job_name)
    except KeyboardInterrupt:
        print("\n[STREAM] Detached (Ctrl+C). Job continues running in Azure ML.")
        job = ml_client.jobs.get(job_name)
        print(f"[STREAM] Current status: {job.status}")
        print(f"[STREAM] Studio URL: {job.studio_url}")
        print(f"\nTo re-attach:\n  python azure_ml/setup_and_submit.py --monitor --job-name {job_name}")
        return

    job = ml_client.jobs.get(job_name)
    print(f"\n[DONE] Job final status: {job.status}")
    if job.status == "Completed":
        print(f"[DONE] Studio URL: {job.studio_url}")
        print("[DONE] Download results with:")
        print(f"  az ml job download --name {job_name} --output-name results "
              f"--download-path ./results/azure_run/")
    elif job.status == "Failed":
        print(f"[FAILED] Check full logs: {job.studio_url}")


def main():
    args = _parse_args()

    print("SentinelFatal2 — Azure ML Setup & Submit")
    print("=" * 60)
    print(f"  Subscription : {SUBSCRIPTION_ID} (Azure for Students)")
    print(f"  Resource Grp : {RESOURCE_GROUP}")
    print(f"  Workspace    : {WORKSPACE_NAME}")
    print(f"  Location     : {LOCATION}")
    print(f"  Compute      : {COMPUTE_NAME}  ({VM_SIZE})")
    print("=" * 60 + "\n")

    # ── Authenticate ───────────────────────────────────────────────────────
    credential = _get_credential()

    from azure.ai.ml import MLClient

    # Create resource group if needed (must exist before workspace)
    _ensure_resource_group(credential)

    # Connect without workspace to create it if needed
    ml_sub = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP)
    _ensure_workspace(ml_sub, args)

    # Connect with workspace for all further operations
    ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

    # ── Compute ────────────────────────────────────────────────────────────
    tier = "Dedicated" if args.dedicated else "LowPriority"
    _ensure_compute(ml_client, tier=tier)

    # ── Environment ────────────────────────────────────────────────────────
    env_ref = _ensure_environment(ml_client)

    # ── Data Asset (upload data_processed.zip once to Azure Blob) ──────────
    data_ref = _ensure_data_asset(ml_client)

    if args.dry_run:
        print("\n[DRY-RUN] Infrastructure ready. Skipping job submission.")
        return

    if args.monitor:
        if not args.job_name:
            print("[ERROR] --monitor requires --job-name <name>")
            sys.exit(1)
        _stream_job(ml_client, args.job_name)
        return

    # ── Submit + auto-stream ────────────────────────────────────────────────
    job = _submit_job(ml_client, env_ref, data_ref, low_priority=not args.dedicated)
    _stream_job(ml_client, job.name)


if __name__ == "__main__":
    main()
