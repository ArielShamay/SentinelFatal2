"""
pretrain.py — Self-Supervised Pre-training Script for SentinelFatal2
=====================================================================
Source: arXiv:2601.06149v1, Sections II-C, II-D, Equation 2
SSOT:   docs/work_plan.md, Part ה.4 + חלק ו שלב 3 + deviation_log.md S4, S5, S6, S9

Training objective
------------------
Channel-Asymmetric Masked Auto-Encoding:
  • Only FHR patches are masked (zero masking); UC always visible.
  • Encoder processes all 73 FHR patches (masked=0) and all 73 UC patches.
  • PretrainingHead reconstructs masked FHR patches from their encoder tokens.
  • Loss (Eq. 2): MSE on masked FHR patches only.

Masking policy
--------------
  Per-batch: one mask_indices vector shared across all samples in the batch.
  This is regenerated each iteration for stochastic coverage.

Usage
-----
  # Full run (Colab / GPU):
  python src/train/pretrain.py --config config/train_config.yaml

  # Dry-run (CPU, 2 batches, batch_size=4):
  python src/train/pretrain.py --config config/train_config.yaml \\
      --device cpu --batch-size 4 --max-batches 2

Outputs
-------
  checkpoints/pretrain/epoch_NNN.pt    — state dict per epoch
  checkpoints/pretrain/best_pretrain.pt — best val reconstruction loss
  logs/pretrain_loss.csv               — epoch, train_loss, val_loss, lr
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Make sure the project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model.patchtst import PatchTST, load_config
from src.model.heads import PretrainingHead
from src.data.dataset import build_pretrain_loaders
from src.data.masking import apply_masking, _random_partition


# ---------------------------------------------------------------------------
# Mask generation (per-batch)
# ---------------------------------------------------------------------------

def generate_mask_indices(
    n_patches: int = 73,
    mask_ratio: float = 0.4,
    min_group_size: int = 2,
    max_group_size: int = 6,
    max_retries: int = 100,
) -> List[int]:
    """Generate mask_indices for one batch using the P6 fix v2 algorithm.

    Returns a Python list of int patch indices (sorted ascending).
    A dummy (n_patches, 1) array is used as a proxy — only indices matter.
    """
    dummy = np.zeros((n_patches, 1), dtype=np.float32)
    _, idx = apply_masking(dummy, mask_ratio, min_group_size, max_group_size, max_retries)
    return idx.tolist()


# ---------------------------------------------------------------------------
# Pretrain step (single forward pass with masking before embedding)
# ---------------------------------------------------------------------------

def pretrain_step(
    model: PatchTST,
    x: torch.Tensor,
    mask_indices: List[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass with pre-embedding zero masking.

    Bypasses model.forward() to apply masking at patch level before projection,
    matching the MAE convention (masked tokens = zero vectors enter the encoder).

    Args:
        model:        PatchTST with PretrainingHead attached.
        x:            Input tensor (batch, 2, 1800) — raw windows.
        mask_indices: List of patch indices to mask (shared across batch).

    Returns:
        (pred, target) both of shape (batch, n_masked, patch_len).
    """
    mask_t = torch.tensor(mask_indices, dtype=torch.long, device=x.device)

    # 1. Extract FHR patches: (B, 73, 48)
    fhr_patches = model._extract_patches(x[:, 0, :])
    original_fhr = fhr_patches.clone()                     # ground truth for loss

    # 2. Zero masking (in-place on a clone to keep original intact)
    masked_fhr = fhr_patches.clone()
    masked_fhr[:, mask_t, :] = 0.0

    # 3. Embed + encode masked FHR
    fhr_embedded = model.patch_embed(masked_fhr)           # (B, 73, d_model)
    fhr_enc = model.encoder(fhr_embedded)                  # (B, 73, d_model)

    # 4. Encode unmasked UC, then fuse into FHR representation (AGW-19 fix).
    #    Channel-Asymmetric MAE: UC is always fully visible and its encoder
    #    output is added token-wise to FHR's output.  For masked FHR positions
    #    the input was 0.0, so after encoding those tokens hold mostly positional
    #    embedding signal; adding uc_enc provides the cross-channel context that
    #    allows the head to reconstruct masked FHR patches using UC information.
    #    Element-wise addition keeps d_model unchanged (no extra parameters).
    uc_enc  = model.encode_channel(x[:, 1, :])             # (B, 73, d_model)
    fhr_enc = fhr_enc + uc_enc                             # UC context fusion

    # 5. Head reconstructs masked positions
    pred   = model.head(fhr_enc, mask_indices)             # (B, n_masked, 48)
    target = original_fhr[:, mask_t, :]                    # (B, n_masked, 48)

    return pred, target


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(
    model: PatchTST,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: dict,
    clip_norm: float = 1.0,
    training: bool = True,
    max_batches: int = 0,
    verbose: bool = True,
) -> float:
    """Run one training or validation epoch.

    Args:
        model:       PatchTST model.
        loader:      DataLoader for this epoch.
        optimizer:   Adam optimizer (None during validation).
        device:      torch.device.
        cfg:         Full config dict (for masking params).
        clip_norm:   Max gradient norm (default 1.0 — ⚠ S6).
        training:    If True, performs backward + optimizer step.
        max_batches: If > 0, stop after this many batches (dry-run).
        verbose:     Print per-batch loss.

    Returns:
        Mean MSE loss over the epoch.
    """
    ptcfg = cfg["pretrain"]
    mask_ratio     = float(ptcfg["mask_ratio"])
    min_group_size = int(ptcfg["min_group_size"])
    max_group_size = int(ptcfg["max_group_size"])
    n_patches      = int(cfg["data"]["n_patches"])

    model.train(training)
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, x in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        x = x.to(device)

        # Per-batch mask: one shared mask for all samples in the batch
        mask_indices = generate_mask_indices(
            n_patches=n_patches,
            mask_ratio=mask_ratio,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
        )

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred, target = pretrain_step(model, x, mask_indices)
            loss = F.mse_loss(pred, target)           # Equation 2 — masked FHR only

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if verbose:
            split = "train" if training else "val"
            print(f"  [{split} batch {batch_idx + 1}] loss={loss.item():.6f}")

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model: PatchTST, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: PatchTST, path: str | Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

def init_csv_log(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])


def append_csv_log(path: str | Path, epoch: int, train_loss: float,
                   val_loss: float, lr: float) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{lr:.2e}"])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def pretrain(
    config_path: str | Path,
    device_str: str = "cpu",
    batch_size: Optional[int] = None,
    max_batches: int = 0,
    processed_root: Optional[str] = None,
    pretrain_csv: Optional[str] = None,
    checkpoint_dir: str = "checkpoints/pretrain",
    log_path: str = "logs/pretrain_loss.csv",
    quiet: bool = False,
) -> None:
    """Full pre-training loop.

    Args:
        config_path:     Path to config/train_config.yaml.
        device_str:      'cpu', 'cuda', or 'cuda:0' etc.
        batch_size:      Override config batch_size (useful for dry-run).
        max_batches:     If > 0, stop each epoch after this many batches (dry-run).
        processed_root:  Override path to processed .npy root directory.
        pretrain_csv:    Override path to pretrain.csv.
        checkpoint_dir:  Directory to save checkpoints.
        log_path:        Path to loss CSV log.
        quiet:           Suppress per-batch prints.
    """
    # ---- Setup ---------------------------------------------------------------
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str)
    ptcfg  = cfg["pretrain"]
    bs     = batch_size if batch_size is not None else int(ptcfg["batch_size"])

    print(f"[pretrain] device={device}, batch_size={bs}, max_batches={max_batches or 'full'}")

    # ---- Data ----------------------------------------------------------------
    # Derive project root deterministically from config path (AGW-20 fix).
    # config/train_config.yaml -> .parent = config/ -> .parent = project root.
    # This is cwd-independent and works regardless of where the script is run from.
    project_root = Path(config_path).resolve().parent.parent
    if processed_root is None:
        processed_root = project_root / "data" / "processed"
    if pretrain_csv is None:
        pretrain_csv = project_root / "data" / "splits" / "pretrain.csv"

    train_loader, val_loader = build_pretrain_loaders(
        pretrain_csv=pretrain_csv,
        processed_root=processed_root,
        window_len=int(cfg["data"]["window_len"]),
        stride=int(ptcfg["window_stride"]),
        batch_size=bs,
        val_fraction=0.1,
        num_workers=0,
        seed=seed,
    )
    print(f"[pretrain] dataset - train windows={len(train_loader.dataset)}, "
          f"val windows={len(val_loader.dataset)}")

    # Preflight check: ensure dataset loaded enough data
    if len(train_loader.dataset) < 100:
        raise RuntimeError(
            f"[pretrain] FATAL: Only {len(train_loader.dataset)} train windows loaded "
            f"(expected ~12,000+). Check that processed .npy files exist at {processed_root}"
        )

    # ---- Model ---------------------------------------------------------------
    model = PatchTST(cfg)
    model.replace_head(PretrainingHead(
        d_model=int(cfg["model"]["d_model"]),
        patch_len=int(cfg["data"]["patch_len"]),
    ))
    model = model.to(device)
    print(f"[pretrain] model: {model}")
    print(f"[pretrain] encoder params: {model.n_encoder_params:,}")

    # ---- Optimizer + LR Scheduler (S12 improvement) -------------------------
    # Adam lr=1e-4 — ✓ paper II-D.
    # ReduceLROnPlateau: halves LR whenever val_loss stalls for lr_scheduler_patience
    # epochs. Allows model to keep descending past the initial plateau without
    # overfitting (the outer early-stopping patience=20 is the final guard).
    optimizer  = torch.optim.Adam(model.parameters(), lr=float(ptcfg["lr"]))
    lr_min     = float(ptcfg.get("lr_min", 1e-6))
    sched_pat  = int(ptcfg.get("lr_scheduler_patience", 5))
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=sched_pat,
        min_lr=lr_min,
    )
    print(f"[pretrain] scheduler: ReduceLROnPlateau(patience={sched_pat}, min_lr={lr_min:.0e})")

    # ---- Logging -------------------------------------------------------------
    init_csv_log(log_path)

    # ---- Training loop -------------------------------------------------------
    max_epochs    = int(ptcfg["max_epochs"])
    patience      = int(ptcfg["patience"])
    best_val_loss = float("inf")
    patience_ctr  = 0
    verbose       = not quiet

    print(f"[pretrain] Starting training: max_epochs={max_epochs}, patience={patience}")

    for epoch in range(max_epochs):
        t0 = time.time()

        train_loss = run_epoch(
            model, train_loader, optimizer, device, cfg,
            clip_norm=1.0, training=True,
            max_batches=max_batches, verbose=verbose,
        )
        val_loss = run_epoch(
            model, val_loader, None, device, cfg,
            clip_norm=1.0, training=False,
            max_batches=max_batches, verbose=False,
        )

        scheduler.step(val_loss)  # update LR based on val_loss improvement
        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  lr={lr:.2e}  ({elapsed:.1f}s)"
        )
        append_csv_log(log_path, epoch, train_loss, val_loss, lr)

        # Checkpoint every epoch
        save_checkpoint(model, Path(checkpoint_dir) / f"epoch_{epoch:03d}.pt")

        # Best model + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            save_checkpoint(model, Path(checkpoint_dir) / "best_pretrain.pt")
            print(f"  [OK] New best val_loss={best_val_loss:.6f} - saved best_pretrain.pt")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                print(f"[pretrain] Early stopping at epoch {epoch} "
                      f"(patience={patience} exhausted)")
                break

        # Dry-run: stop after first epoch if max_batches is set
        if max_batches > 0:
            print("[pretrain] Dry-run complete - stopping after 1 epoch.")
            break

    print(f"[pretrain] Finished. Best val_loss={best_val_loss:.6f}")
    print(f"[pretrain] Checkpoints: {checkpoint_dir}/")
    print(f"[pretrain] Loss log:     {log_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SentinelFatal2 — Pre-training")
    p.add_argument("--config",         default="config/train_config.yaml",
                   help="Path to train_config.yaml")
    p.add_argument("--device",         default="cpu",
                   help="torch device string, e.g. 'cpu', 'cuda', 'cuda:0'")
    p.add_argument("--batch-size",     type=int, default=None,
                   help="Override batch_size from config (useful for dry-run)")
    p.add_argument("--max-batches",    type=int, default=0,
                   help="If > 0: stop each epoch after this many batches (dry-run)")
    p.add_argument("--processed-root", default=None,
                   help="Override path to processed .npy root directory")
    p.add_argument("--pretrain-csv",   default=None,
                   help="Override path to pretrain.csv")
    p.add_argument("--checkpoint-dir", default="checkpoints/pretrain",
                   help="Directory for checkpoint .pt files")
    p.add_argument("--log-path",       default="logs/pretrain_loss.csv",
                   help="CSV log path")
    p.add_argument("--quiet",          action="store_true",
                   help="Suppress per-batch loss output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pretrain(
        config_path     = args.config,
        device_str      = args.device,
        batch_size      = args.batch_size,
        max_batches     = args.max_batches,
        processed_root  = args.processed_root,
        pretrain_csv    = args.pretrain_csv,
        checkpoint_dir  = args.checkpoint_dir,
        log_path        = args.log_path,
        quiet           = args.quiet,
    )
