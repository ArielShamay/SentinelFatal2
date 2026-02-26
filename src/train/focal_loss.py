"""
focal_loss.py — Focal Loss for SentinelFatal2
==============================================
Source: Lin et al. (2017), "Focal Loss for Dense Object Detection"
SSOT:   docs/plan_2.md §3.2.2

Extracted from src/train/finetune.py to satisfy plan_2 §8.1 module structure.
Imported by:
  - src/train/finetune.py
  - scripts/run_e2e_cv_v2.py (via finetune re-export)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    When gamma > 0, easy examples (high p_t) get down-weighted, forcing the
    model to focus on hard misclassified examples.

    SAFETY: Do NOT use with Mixup unless soft-label interp mode is enabled.
    When mixup=off (default for focal), this class works correctly.
    """

    def __init__(
        self,
        alpha: list,        # per-class weights, e.g. [1.0, 3.9]
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        n_classes: int = 2,
    ):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw logits
            targets: (B,) integer class labels
        """
        # Smooth targets
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / self.n_classes
            targets_oh = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            targets_oh = targets_oh * (1 - self.label_smoothing) + smooth
        else:
            targets_oh = None

        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        if targets_oh is not None:
            # Soft-label focal: interpolate loss with label-smoothed target
            p_t = (probs * targets_oh).sum(dim=1)
        else:
            p_t = probs[torch.arange(len(targets)), targets]

        alpha_t  = self.alpha[targets]
        focal_w  = (1.0 - p_t).pow(self.gamma)

        if targets_oh is not None:
            ce = -(targets_oh * log_probs).sum(dim=1)
        else:
            ce = -log_probs[torch.arange(len(targets)), targets]

        loss = alpha_t * focal_w * ce
        return loss.mean()
