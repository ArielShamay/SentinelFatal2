"""
finetune.py — Supervised Fine-tuning Script for SentinelFatal2
===============================================================
Source: arXiv:2601.06149v1, Section II-E
SSOT:   docs/work_plan.md, Part ה.5 + חלק ו שלב 4 + deviation_log.md S6, S6.1

Training objective
------------------
Binary acidemia classification (pH <= 7.15 vs normal) using:
  • Pre-trained PatchTST backbone (frozen LR 1e-5 — differential LR).
  • New ClassificationHead (LR 1e-4).
  • Cross-entropy loss with class_weight (P8 fix: [1.0, ~3.9]).
  • Training unit: window. Validation unit: recording (P7 fix).
  • AUC aggregation: max(window_scores) per recording (LOCKED).

Usage
-----
  # Full run (Colab / GPU):
  python src/train/finetune.py --config config/train_config.yaml

  # Dry-run (CPU, 2 batches):
  python src/train/finetune.py --config config/train_config.yaml \\
      --device cpu --max-batches 2

Outputs
-------
  checkpoints/finetune/epoch_NNN.pt    — state dict per epoch
  checkpoints/finetune/best_finetune.pt — best val AUC
  logs/finetune_loss.csv               — epoch, train_loss, val_auc, lr_backbone, lr_head
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Make sure the project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.model.patchtst import PatchTST, load_config
from src.model.heads import ClassificationHead
from src.data.dataset import build_finetune_loaders
from src.train.utils import compute_recording_auc


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(train_csv: Path) -> torch.Tensor:
    """Compute class weights from train.csv for imbalanced dataset.

    P8 fix: Use class_weight approach (deviation S6.1), not oversampling.

    Args:
        train_csv: Path to train.csv with 'target' column.

    Returns:
        Tensor [w0, w1] where w0=1.0, w1 = n_neg / n_pos (approx 3.9).
    """
    df = pd.read_csv(train_csv, dtype={"target": int})
    n_neg = (df['target'] == 0).sum()  # normal
    n_pos = (df['target'] == 1).sum()  # acidemia

    if n_pos == 0:
        raise ValueError(f"No positive samples (target=1) in {train_csv}")

    weight_pos = n_neg / n_pos
    weights = torch.tensor([1.0, weight_pos], dtype=torch.float32)

    print(f"[class_weights] n_neg={n_neg}, n_pos={n_pos}, weights={weights.tolist()}")
    return weights


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float = 1.0,
    training: bool = True,
    max_batches: int = 0,
    verbose: bool = False,
) -> float:
    """Run one epoch of training or validation.

    Args:
        model:       PatchTST with ClassificationHead.
        loader:      DataLoader yielding (batch_x, batch_y).
        optimizer:   Optimizer (None for validation).
        criterion:   CrossEntropyLoss with class weights.
        device:      torch device.
        clip_norm:   Max gradient norm (1.0 per config).
        training:    If True, backpropagate and update. If False, eval mode.
        max_batches: If > 0, stop after this many batches (dry-run).
        verbose:     Print per-batch progress.

    Returns:
        Average loss over epoch.
    """
    model.train() if training else model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)  # (B, 2, 1800)
        batch_y = batch_y.to(device)  # (B,)

        # Forward
        if training:
            logits = model(batch_x)  # (B, 2)
            loss = criterion(logits, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

        total_loss += loss.item()
        n_batches += 1

        if verbose:
            mode = "train" if training else "val"
            print(f"  [{mode}] batch {batch_idx}/{len(loader)} loss={loss.item():.4f}")

        if max_batches > 0 and n_batches >= max_batches:
            break

    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model state_dict to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_pretrained_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load pre-trained checkpoint and verify.
    
    Only loads backbone weights (patch_embed + encoder), ignoring head weights
    since the pretrained model has PretrainingHead and we need ClassificationHead.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {path}")

    state_dict = torch.load(path, map_location=device)
    
    # Filter out head weights from pretrained checkpoint
    backbone_state = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
    
    # Load with strict=False to allow missing head weights
    model.load_state_dict(backbone_state, strict=False)
    print(f"[load_checkpoint] Loaded pretrained backbone weights from {path}")
    print(f"[load_checkpoint] Loaded {len(backbone_state)} tensors (head will be initialized randomly)")


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def init_csv_log(log_path: Union[str, Path]) -> None:
    """Initialize CSV log with header."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_auc", "lr_backbone", "lr_head"])


def append_csv_log(
    log_path: Union[str, Path],
    epoch: int,
    train_loss: float,
    val_auc: float,
    lr_backbone: float,
    lr_head: float,
) -> None:
    """Append one row to CSV log."""
    log_path = Path(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_auc:.6f}",
                        f"{lr_backbone:.2e}", f"{lr_head:.2e}"])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    config_path: str | Path,
    device_str: str = "cpu",
    max_batches: int = 0,
    processed_root: Optional[str] = None,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    pretrain_checkpoint: Optional[str] = None,
    checkpoint_dir: str = "checkpoints/finetune",
    log_path: str = "logs/finetune_loss.csv",
    quiet: bool = False,
) -> None:
    """Full fine-tuning loop.

    Args:
        config_path:          Path to config/train_config.yaml.
        device_str:           'cpu', 'cuda', or 'cuda:0' etc.
        max_batches:          If > 0, stop each epoch after this many batches (dry-run).
        processed_root:       Override path to processed .npy root directory.
        train_csv:            Override path to train.csv.
        val_csv:              Override path to val.csv.
        pretrain_checkpoint:  Override path to pretrained checkpoint.
        checkpoint_dir:       Directory to save checkpoints.
        log_path:             Path to loss CSV log.
        quiet:                Suppress per-batch prints.
    """
    # ---- Setup ---------------------------------------------------------------
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str)
    ftcfg  = cfg["finetune"]
    bs     = int(ftcfg["batch_size"])

    print(f"[finetune] device={device}, batch_size={bs}, max_batches={max_batches or 'full'}")

    # ---- Data ----------------------------------------------------------------
    # Derive project root deterministically from config path (AGW-20 fix).
    project_root = Path(config_path).resolve().parent.parent
    if processed_root is None:
        processed_root = project_root / "data" / "processed"
    if train_csv is None:
        train_csv = project_root / "data" / "splits" / "train.csv"
    if val_csv is None:
        val_csv = project_root / "data" / "splits" / "val.csv"
    if pretrain_checkpoint is None:
        pretrain_checkpoint = project_root / "checkpoints" / "pretrain" / "best_pretrain.pt"
    else:
        pretrain_checkpoint = Path(pretrain_checkpoint)

    # Class weights (P8 fix: deviation S6.1)
    class_weights = compute_class_weights(train_csv)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # S13: train_stride=60 (dense) produces ~9,200 windows vs ~630 with stride=900.
    # Augmentation (noise + jitter) adds further variety during training only.
    val_stride_ds   = int(ftcfg.get("window_stride", cfg["pretrain"]["window_stride"]))
    train_stride_ds = int(ftcfg.get("train_stride", val_stride_ds))
    do_augment      = (max_batches == 0)  # disable during dry-run for speed
    train_loader, val_loader = build_finetune_loaders(
        train_csv=train_csv,
        val_csv=val_csv,
        processed_root=processed_root,
        window_len=int(cfg["data"]["window_len"]),
        stride=val_stride_ds,
        train_stride=train_stride_ds,
        augment=do_augment,
        batch_size=bs,
        num_workers=0,
    )
    print(f"[finetune] dataset - train windows={len(train_loader.dataset)} "
          f"(stride={train_stride_ds}, augment={do_augment}), "
          f"val windows={len(val_loader.dataset)}")

    # Preflight check: ensure dataset loaded enough data
    if len(train_loader.dataset) < 10:
        raise RuntimeError(
            f"[finetune] FATAL: Only {len(train_loader.dataset)} train windows loaded. "
            f"Check that processed .npy files exist at {processed_root}"
        )

    # ---- Model ---------------------------------------------------------------
    model = PatchTST(cfg)
    load_pretrained_checkpoint(model, pretrain_checkpoint, device)

    # Replace pretrain head with classification head
    d_in = int(cfg["data"]["n_patches"]) * int(cfg["model"]["d_model"]) * int(cfg["data"]["n_channels"])
    model.replace_head(ClassificationHead(
        d_in=d_in,
        n_classes=int(ftcfg["n_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ))
    model = model.to(device)
    print(f"[finetune] model: PatchTST + ClassificationHead")
    print(f"[finetune] encoder params: {model.n_encoder_params:,}")

    # ---- Optimizer (Differential LR) -----------------------------------------
    # Backbone (patch_embed + encoder): lr_backbone = 1e-5
    # Head (classification): lr_head = 1e-4
    backbone_params = list(model.patch_embed.parameters()) + list(model.encoder.parameters())
    head_params = list(model.head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': float(ftcfg["lr_backbone"])},
        {'params': head_params, 'lr': float(ftcfg["lr_head"])},
    ], weight_decay=float(ftcfg["weight_decay"]))

    lr_backbone = optimizer.param_groups[0]['lr']
    lr_head     = optimizer.param_groups[1]['lr']
    print(f"[finetune] optimizer: AdamW, lr_backbone={lr_backbone:.2e}, lr_head={lr_head:.2e}")

    # ReduceLROnPlateau (S12): halves both param-group LRs when val_auc stalls.
    # Outer early-stopping patience=25 is the final guard.
    lr_min    = float(ftcfg.get("lr_min", 1e-7))
    sched_pat = int(ftcfg.get("lr_scheduler_patience", 7))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=sched_pat,
        min_lr=lr_min,
    )
    print(f"[finetune] scheduler: ReduceLROnPlateau(patience={sched_pat}, min_lr={lr_min:.0e})")

    # ---- Logging -------------------------------------------------------------
    init_csv_log(log_path)

    # ---- Training loop -------------------------------------------------------
    max_epochs    = int(ftcfg["max_epochs"])
    patience      = int(ftcfg["patience"])
    best_val_auc  = 0.0
    patience_ctr  = 0
    verbose       = not quiet

    print(f"[finetune] Starting training: max_epochs={max_epochs}, patience={patience}")

    for epoch in range(max_epochs):
        t0 = time.time()

        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_norm=float(ftcfg["gradient_clip"]),
            training=True, max_batches=max_batches, verbose=verbose,
        )

        # Validate: per-recording AUC (P7 fix)
        # val_stride=60 (15-sec step) gives 286 windows/recording vs 14 with stride=900,
        # producing a much more stable AUC signal for model selection (S12 improvement).
        # stride=1 is reserved ONLY for final evaluation (Step 7 / 05_evaluation.ipynb).
        stride_val = int(ftcfg.get("val_stride", 60)) if max_batches == 0 else 60
        val_auc = compute_recording_auc(
            model, val_csv, processed_root, stride=stride_val, device=device_str
        )

        scheduler.step(val_auc)  # update LR based on val_auc improvement
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_head     = optimizer.param_groups[1]['lr']
        elapsed     = time.time() - t0
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f}  "
            f"val_auc={val_auc:.6f}  lr_bb={lr_backbone:.2e}  lr_hd={lr_head:.2e}  ({elapsed:.1f}s)"
        )
        append_csv_log(log_path, epoch, train_loss, val_auc, lr_backbone, lr_head)

        # Checkpoint every epoch
        save_checkpoint(model, Path(checkpoint_dir) / f"epoch_{epoch:03d}.pt")

        # Best model + early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_ctr = 0
            save_checkpoint(model, Path(checkpoint_dir) / "best_finetune.pt")
            print(f"  [OK] New best val_auc={best_val_auc:.6f} - saved best_finetune.pt")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                print(f"[finetune] Early stopping at epoch {epoch} "
                      f"(patience={patience} exhausted)")
                break

        # Dry-run: stop after first epoch if max_batches is set
        if max_batches > 0:
            print("[finetune] Dry-run complete - stopping after 1 epoch.")
            break

    print(f"[finetune] Finished. Best val_auc={best_val_auc:.6f}")
    print(f"[finetune] Checkpoints: {checkpoint_dir}/")
    print(f"[finetune] Loss log:     {log_path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SentinelFatal2 — Fine-tuning")
    p.add_argument("--config",         default="config/train_config.yaml",
                   help="Path to train_config.yaml")
    p.add_argument("--device",         default="cpu",
                   help="torch device string, e.g. 'cpu', 'cuda', 'cuda:0'")
    p.add_argument("--max-batches",    type=int, default=0,
                   help="If > 0: stop each epoch after this many batches (dry-run)")
    p.add_argument("--processed-root", default=None,
                   help="Override path to processed .npy root directory")
    p.add_argument("--train-csv",      default=None,
                   help="Override path to train.csv")
    p.add_argument("--val-csv",        default=None,
                   help="Override path to val.csv")
    p.add_argument("--pretrain-checkpoint", default=None,
                   help="Override path to pretrained checkpoint")
    p.add_argument("--checkpoint-dir", default="checkpoints/finetune",
                   help="Directory to save checkpoints")
    p.add_argument("--log-path",       default="logs/finetune_loss.csv",
                   help="Path to CSV loss log")
    p.add_argument("--quiet",          action="store_true",
                   help="Suppress per-batch prints")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        config_path=args.config,
        device_str=args.device,
        max_batches=args.max_batches,
        processed_root=args.processed_root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        pretrain_checkpoint=args.pretrain_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path,
        quiet=args.quiet,
    )
