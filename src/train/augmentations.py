"""
augmentations.py — Data Augmentation for SentinelFatal2
========================================================
Source: docs/plan_2.md §3.2.1
SSOT:   docs/plan_2.md §8.1

Extracted from src/train/finetune.py to satisfy plan_2 §8.1 module structure.
Imported by:
  - src/train/finetune.py
  - scripts/run_e2e_cv_v2.py (via finetune re-export)

Augmentations applied ONLY during training (not val/test).

Supported augmentations (plan_2 §3.2.1):
  - Gaussian Noise:    σ_fhr=0.01, σ_uc=0.005, p=0.5
  - Random Scaling:    FHR × U(0.95, 1.05), p=0.3
  - Temporal Jitter:   roll ±50 samples (~12s), p=0.5
  - Channel Dropout:   zero UC channel, p=0.1
  - Cutout:            zero 48-96 contiguous samples, p=0.2
  Note: Mixup is NOT included here; handled separately in training loop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def augment_window(
    x: torch.Tensor,
    aug_cfg: dict,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Apply stochastic augmentations to a batch of windows.

    Args:
        x:       (B, 2, 1800) float32 tensor — channel 0=FHR, 1=UC.
        aug_cfg: augmentation section from train_config.yaml.
        rng:     numpy random Generator (uses global random if None).

    Returns:
        Augmented tensor of same shape.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = x.clone()
    B, C, T = x.shape

    # Gaussian Noise
    gn = aug_cfg.get("gaussian_noise", {})
    if rng.random() < float(gn.get("p", 0.5)):
        sigma_fhr = float(gn.get("sigma_fhr", 0.01))
        sigma_uc  = float(gn.get("sigma_uc",  0.005))
        x[:, 0, :] += torch.randn(B, T, device=x.device) * sigma_fhr
        x[:, 1, :] += torch.randn(B, T, device=x.device) * sigma_uc

    # Random Scaling (FHR only)
    rs = aug_cfg.get("random_scaling", {})
    if rng.random() < float(rs.get("p", 0.3)):
        lo = float(rs.get("scale_min", 0.95))
        hi = float(rs.get("scale_max", 1.05))
        scale = torch.empty(B, 1, 1, device=x.device).uniform_(lo, hi)
        x[:, 0:1, :] = x[:, 0:1, :] * scale

    # Temporal Jitter (roll each sample independently)
    tj = aug_cfg.get("temporal_jitter", {})
    if rng.random() < float(tj.get("p", 0.5)):
        max_shift = int(tj.get("max_shift", 50))
        shifts = rng.integers(-max_shift, max_shift + 1, size=B)
        for i, s in enumerate(shifts):
            if s != 0:
                x[i] = torch.roll(x[i], int(s), dims=-1)

    # Channel Dropout (zero UC channel)
    cd = aug_cfg.get("channel_dropout", {})
    if rng.random() < float(cd.get("p", 0.1)):
        x[:, 1, :] = 0.0

    # Cutout (zero random contiguous segment)
    co = aug_cfg.get("cutout", {})
    if rng.random() < float(co.get("p", 0.2)):
        min_len = int(co.get("min_len", 48))
        max_len = int(co.get("max_len", 96))
        L = rng.integers(min_len, max_len + 1)
        start = rng.integers(0, max(1, T - L))
        x[:, :, start:start + L] = 0.0

    return x
