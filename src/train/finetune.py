"""
finetune.py — Supervised Fine-tuning Script for SentinelFatal2
===============================================================
Source: arXiv:2601.06149v1, Section II-E
SSOT:   docs/plan_2.md §3

plan_2 improvements (§3)
------------------------
  • Progressive unfreezing (§3.1.1): 4 phases, dynamic by num_layers.
  • SWA epochs 50-100 with mandatory BN recalibration (§3.1.2).
  • Focal Loss γ=2 + Label Smoothing ε=0.05 (§3.2.2-3).
  • Data Augmentation: Gaussian noise, scaling, jitter, ch_dropout, cutout (§3.2.1).
  • Mixup disabled when loss=focal (safety policy, plan_2 §3.2.1).

Usage
-----
  python src/train/finetune.py --config config/train_config.yaml
  python src/train/finetune.py --config config/train_config.yaml \\
      --device cpu --max-batches 2  # dry-run

Outputs
-------
  checkpoints/finetune/epoch_NNN.pt     — per-epoch checkpoint
  checkpoints/finetune/best_finetune.pt — best smooth val AUC
  checkpoints/finetune/swa_model.pt     — SWA-averaged model (if SWA enabled)
  logs/finetune_loss.csv                — epoch, train_loss, val_auc, smooth_auc, lr_bb, lr_hd
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
from src.train.swa import SWAAccumulator


# ---------------------------------------------------------------------------
# Focal Loss (plan_2 §3.2.2) — defined in src/train/focal_loss.py
# ---------------------------------------------------------------------------

from src.train.focal_loss import FocalLoss  # noqa: E402


# ---------------------------------------------------------------------------
# Data Augmentation (plan_2 §3.2.1) — defined in src/train/augmentations.py
# ---------------------------------------------------------------------------

from src.train.augmentations import augment_window  # noqa: E402


# ---------------------------------------------------------------------------
# Progressive Unfreezing (plan_2 §3.1.1)
# ---------------------------------------------------------------------------

def get_unfreeze_phase(epoch: int, phases: list) -> tuple:
    """Return the current unfreeze phase config for a given epoch.

    phases is a list of [start_epoch, n_top_layers, lr_backbone, lr_head].
    n_top_layers = -1 means unfreeze everything (including embed).
    Returns (n_top_layers, lr_backbone, lr_head).
    """
    current = phases[0]
    for phase in phases:
        if epoch >= int(phase[0]):
            current = phase
    return int(current[1]), float(current[2]), float(current[3])


def apply_unfreeze_phase(
    model: PatchTST,
    n_top_layers: int,
    lr_backbone: float,
    lr_head: float,
    optimizer: torch.optim.Optimizer,
    weight_decay: float = 0.01,
) -> None:
    """Freeze/unfreeze backbone layers and update optimizer LRs.

    n_top_layers:
      0   → backbone fully frozen
      k>0 → top-k encoder layers unfrozen (dynamic: uses model.encoder.layers)
      -1  → all params unfrozen (including patch_embed)
    """
    # Infer total number of encoder layers dynamically (plan_2 §3.1.1 constraint)
    encoder_layers = list(model.encoder.parameters())
    try:
        n_total = len(model.encoder.layers)   # list of transformer layers
    except AttributeError:
        n_total = model.encoder.num_layers if hasattr(model.encoder, 'num_layers') else 3

    # Freeze everything first
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False

    if n_top_layers == -1:
        # Full unfreeze
        for param in model.parameters():
            param.requires_grad = True
    elif n_top_layers > 0:
        # Unfreeze top-k layers (layers[-1], layers[-2], ..., layers[-k])
        k = min(n_top_layers, n_total)
        try:
            for layer in model.encoder.layers[-k:]:
                for param in layer.parameters():
                    param.requires_grad = True
        except (AttributeError, TypeError):
            # Fallback: unfreeze all encoder if layers not indexable
            for param in model.encoder.parameters():
                param.requires_grad = True
    # (n_top_layers == 0 → backbone stays frozen — no action needed)

    # Always keep head trainable
    for param in model.head.parameters():
        param.requires_grad = True

    # Rebuild optimizer param groups with updated LRs
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and p not in set(model.head.parameters())]
    head_params = list(model.head.parameters())

    optimizer.param_groups[0]['params'] = backbone_params
    optimizer.param_groups[0]['lr']     = lr_backbone
    optimizer.param_groups[1]['params'] = head_params
    optimizer.param_groups[1]['lr']     = lr_head

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [unfreeze] phase: n_top={n_top_layers}, "
          f"lr_bb={lr_backbone:.1e}, lr_hd={lr_head:.1e}, "
          f"trainable_params={n_trainable:,}")


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
    aug_cfg: Optional[dict] = None,
) -> float:
    """Run one epoch of training or validation.

    Args:
        aug_cfg: augmentation config dict (applied only during training).
    """
    model.train() if training else model.eval()
    rng = np.random.default_rng() if training else None

    total_loss = 0.0
    n_batches = 0

    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Augmentation (training only, plan_2 §3.2.1)
        if training and aug_cfg:
            batch_x = augment_window(batch_x, aug_cfg, rng)

        if training:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
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
            import sys; sys.stdout.flush()

        if max_batches > 0 and n_batches >= max_batches:
            break

    return total_loss / n_batches if n_batches > 0 else 0.0


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
        writer.writerow(["epoch", "train_loss", "val_auc", "smooth_auc", "lr_backbone", "lr_head"])


def append_csv_log(
    log_path: Union[str, Path],
    epoch: int,
    train_loss: float,
    val_auc: float,
    smooth_auc: float,
    lr_backbone: float,
    lr_head: float,
) -> None:
    """Append one row to CSV log."""
    log_path = Path(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_auc:.6f}",
                        f"{smooth_auc:.6f}", f"{lr_backbone:.2e}", f"{lr_head:.2e}"])


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
    config_overrides: Optional[dict] = None,
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
        config_overrides:     Optional flat dict of config values to override.
                              Keys matching 'model' section (d_model, n_layers, etc.)
                              are applied to cfg['model']; all others to cfg['finetune'].
    """
    _MODEL_KEYS = {'d_model', 'n_layers', 'n_heads', 'dropout', 'patch_len', 'stride',
                   'n_patches', 'd_ff', 'activation', 'norm', 'n_channels'}

    # ---- Setup ---------------------------------------------------------------
    cfg = load_config(config_path)
    if config_overrides:
        for k, v in config_overrides.items():
            if k in _MODEL_KEYS:
                cfg.setdefault('model', {})[k] = v
            else:
                cfg.setdefault('finetune', {})[k] = v
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

    # ---- Loss function (plan_2 §3.2.2) ----------------------------------------
    loss_type  = str(ftcfg.get("loss", "cross_entropy"))
    focal_gamma = float(ftcfg.get("focal_gamma", 2.0))
    label_smooth = float(ftcfg.get("label_smoothing", 0.0))
    n_cls = int(ftcfg["n_classes"])
    if loss_type == "focal":
        criterion = FocalLoss(
            alpha=class_weights.tolist(),
            gamma=focal_gamma,
            label_smoothing=label_smooth,
            n_classes=n_cls,
        ).to(device)
        print(f"[finetune] loss: FocalLoss(gamma={focal_gamma}, label_smoothing={label_smooth})")
    else:
        if label_smooth > 0:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(device),
                label_smoothing=label_smooth,
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"[finetune] loss: CrossEntropyLoss(label_smoothing={label_smooth})")

    # Augmentation config (plan_2 §3.2.1) — active during training only
    aug_cfg_raw = ftcfg.get("augmentation", {})
    # Safety: disable Mixup when loss=focal (plan_2 §3.2.1)
    if loss_type == "focal" and aug_cfg_raw.get("mixup_with_focal", False) is False:
        aug_cfg_raw = dict(aug_cfg_raw)
        aug_cfg_raw.pop("mixup", None)
    do_augment = bool(max_batches == 0 and aug_cfg_raw)  # disable in dry-run

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

    d_in = int(cfg["data"]["n_patches"]) * int(cfg["model"]["d_model"]) * int(cfg["data"]["n_channels"])
    model.replace_head(ClassificationHead(
        d_in=d_in,
        n_classes=int(ftcfg["n_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    ))
    model = model.to(device)
    print(f"[finetune] model: PatchTST + ClassificationHead")
    print(f"[finetune] encoder params: {model.n_encoder_params:,}")

    # ---- Optimizer — 2 param groups (backbone / head) ------------------------
    # Phase 1 begins with backbone frozen; phases update via apply_unfreeze_phase().
    backbone_params = list(model.patch_embed.parameters()) + list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 0.0},          # frozen initially
        {'params': head_params,     'lr': float(ftcfg["lr_head"])},
    ], weight_decay=float(ftcfg["weight_decay"]))

    lr_min    = float(ftcfg.get("lr_min", 1e-7))
    sched_pat = int(ftcfg.get("lr_scheduler_patience", 7))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=sched_pat, min_lr=lr_min,
    )

    # Progressive unfreezing phases (plan_2 §3.1.1)
    do_progressive_unfreeze = bool(ftcfg.get("progressive_unfreeze", True))
    unfreeze_phases = ftcfg.get("unfreeze_phases", [
        [0,   0,  0.0,    1.0e-3],
        [5,   1,  1.0e-5, 5.0e-4],
        [15,  2,  3.0e-5, 3.0e-4],
        [30, -1,  5.0e-5, 1.0e-4],
    ])
    last_phase_key = None   # track phase transitions

    # SWA config (plan_2 §3.1.2)
    swa_start = int(ftcfg.get("swa_start", 50))
    swa_end   = int(ftcfg.get("swa_end",   100))
    swa_accum = SWAAccumulator(model)
    swa_active = False

    # ---- Logging -------------------------------------------------------------
    init_csv_log(log_path)

    # ---- Training loop -------------------------------------------------------
    max_epochs    = int(ftcfg["max_epochs"])
    patience      = int(ftcfg["patience"])
    best_smooth_auc = 0.0
    best_val_auc  = 0.0
    smooth_auc    = 0.0
    ema_beta      = 0.8
    patience_ctr  = 0
    verbose       = not quiet

    print(f"[finetune] Starting: max_epochs={max_epochs}, patience={patience}, "
          f"SWA=[{swa_start},{swa_end}], aug={do_augment}")

    for epoch in range(max_epochs):
        t0 = time.time()

        # Progressive unfreezing: check if phase changed
        if do_progressive_unfreeze:
            n_top, lr_bb, lr_hd = get_unfreeze_phase(epoch, unfreeze_phases)
            phase_key = (n_top, lr_bb, lr_hd)
            if phase_key != last_phase_key:
                apply_unfreeze_phase(model, n_top, lr_bb, lr_hd, optimizer,
                                     weight_decay=float(ftcfg["weight_decay"]))
                last_phase_key = phase_key

        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_norm=float(ftcfg["gradient_clip"]),
            training=True, max_batches=max_batches, verbose=verbose,
            aug_cfg=aug_cfg_raw if do_augment else None,
        )

        stride_val = int(ftcfg.get("val_stride", 60)) if max_batches == 0 else 60
        val_auc = compute_recording_auc(
            model, val_csv, processed_root, stride=stride_val, device=device_str
        )

        smooth_auc = val_auc if epoch == 0 else ema_beta * smooth_auc + (1 - ema_beta) * val_auc
        scheduler.step(smooth_auc)

        # SWA accumulation window
        if swa_start <= epoch < swa_end:
            swa_accum.update(model)
            if not swa_active:
                swa_active = True
                print(f"  [SWA] Window started (epochs {swa_start}-{swa_end})")

        lr_backbone = optimizer.param_groups[0]['lr']
        lr_head     = optimizer.param_groups[1]['lr']
        elapsed     = time.time() - t0
        swa_tag     = f" [SWA n={swa_accum.n_collected}]" if swa_active else ""
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f}  "
            f"val_auc={val_auc:.6f}  smooth={smooth_auc:.6f}  "
            f"lr_bb={lr_backbone:.2e}  lr_hd={lr_head:.2e}  "
            f"({elapsed:.1f}s){swa_tag}"
        )
        import sys; sys.stdout.flush()
        append_csv_log(log_path, epoch, train_loss, val_auc, smooth_auc, lr_backbone, lr_head)

        save_checkpoint(model, Path(checkpoint_dir) / f"epoch_{epoch:03d}.pt")

        if smooth_auc > best_smooth_auc:
            best_smooth_auc = smooth_auc
            best_val_auc = max(best_val_auc, val_auc)
            patience_ctr = 0
            save_checkpoint(model, Path(checkpoint_dir) / "best_finetune.pt")
            print(f"  [OK] New best smooth_auc={best_smooth_auc:.6f} "
                  f"(raw={val_auc:.6f}) — saved best_finetune.pt")
        else:
            patience_ctr += 1
            print(f"  No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                print(f"[finetune] Early stopping at epoch {epoch}")
                break

        if max_batches > 0:
            print("[finetune] Dry-run complete — stopping after 1 epoch.")
            break

    # ---- SWA finalization (plan_2 §3.1.2) ------------------------------------
    if swa_accum.n_collected > 0:
        print(f"\n[SWA] Finalizing: averaging {swa_accum.n_collected} checkpoints...")
        swa_model = swa_accum.average(model, device)
        print("[SWA] Running BN recalibration (required — plan_2 §3.1.2)...")
        swa_accum.recalibrate_bn(swa_model, train_loader, device,
                                 max_batches=0 if max_batches == 0 else 2)
        swa_val_auc = compute_recording_auc(
            swa_model, val_csv, processed_root, stride=stride_val, device=device_str
        )
        print(f"[SWA] swa_val_auc={swa_val_auc:.6f}  best_regular={best_smooth_auc:.6f}")
        if swa_val_auc > best_smooth_auc:
            save_checkpoint(swa_model, Path(checkpoint_dir) / "best_finetune.pt")
            print(f"  [SWA] SWA model is BETTER → overwrote best_finetune.pt")
        save_checkpoint(swa_model, Path(checkpoint_dir) / "swa_model.pt")
        print(f"  [SWA] Saved swa_model.pt")

    print(f"\n[finetune] Finished. Best smooth_auc={best_smooth_auc:.6f} "
          f"(best raw={best_val_auc:.6f})")
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
