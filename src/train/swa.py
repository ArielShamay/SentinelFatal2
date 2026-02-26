"""
swa.py — Stochastic Weight Averaging (SWA) for SentinelFatal2
==============================================================
SSOT: docs/plan_2.md §3.1.2

SWA averages model weights across epochs [swa_start, swa_end] to improve
generalization on small datasets (expected +1-3% AUC).

BatchNorm Recalibration (CRITICAL)
-----------------------------------
After creating the SWA-averaged model, the BatchNorm running statistics
(running_mean / running_var) are stale because they were accumulated only
for the last checkpoint, not for the weight average.
Without recalibration, SWA model AUC will be artificially degraded.

Usage
-----
    from src.train.swa import SWAAccumulator

    swa = SWAAccumulator(model)

    for epoch in range(max_epochs):
        train_one_epoch(model, ...)
        if swa_start <= epoch < swa_end:
            swa.update(model)

    swa_model = swa.average(base_model, device)
    swa.recalibrate_bn(swa_model, train_loader, device)  # REQUIRED
    val_auc = evaluate(swa_model, ...)
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class SWAAccumulator:
    """Accumulates model weight averages for SWA.

    Maintains a running sum of model state_dicts and a count.
    Call update() at each epoch inside the SWA window, then average() to get
    the averaged model.
    """

    def __init__(self, model: nn.Module) -> None:
        self._sum: Optional[dict] = None
        self._count: int = 0
        # keep model arch reference for constructing the output
        self._model_ref = model

    def update(self, model: nn.Module) -> None:
        """Add current model weights to the running sum."""
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self._sum is None:
            self._sum = state
        else:
            for k in self._sum:
                self._sum[k] += state[k]
        self._count += 1

    @property
    def n_collected(self) -> int:
        return self._count

    def average(
        self,
        base_model: nn.Module,
        device: torch.device,
    ) -> nn.Module:
        """Create a new model with averaged weights.

        Args:
            base_model: Model with the correct architecture (used as template).
            device:     Target device for the averaged model.

        Returns:
            New model with averaged state dict loaded.

        Raises:
            RuntimeError: If no updates have been collected yet.
        """
        if self._count == 0 or self._sum is None:
            raise RuntimeError(
                "[SWA] No checkpoints collected. "
                "Call update() at least once inside the SWA window."
            )

        averaged_state = {k: v / self._count for k, v in self._sum.items()}
        swa_model = copy.deepcopy(base_model)
        swa_model.load_state_dict(averaged_state)
        swa_model = swa_model.to(device)
        print(f"[SWA] Averaged {self._count} checkpoints → swa_model ready")
        return swa_model

    @staticmethod
    def recalibrate_bn(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: int = 0,
    ) -> None:
        """Recalibrate BatchNorm running statistics after weight averaging.

        Without this step, the BN running_mean/running_var reflect only the
        last individual checkpoint, not the averaged weights — causing the
        SWA model to perform poorly.

        Runs a full forward pass through the training set in train() mode
        (no gradients, no backward). Updates running_mean / running_var in-place.

        Args:
            model:       SWA-averaged model (after calling average()).
            train_loader: DataLoader for the training split.
            device:      Device the model lives on.
            max_batches: If > 0, stop after this many batches (for dry-run / speed).
        """
        # Check whether model has any BN layers
        has_bn = any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in model.modules()
        )
        if not has_bn:
            print("[SWA] No BatchNorm layers found — skipping BN recalibration.")
            return

        print("[SWA] Recalibrating BatchNorm running statistics...")
        model.train()  # enables BN running stat accumulation

        # Reset running stats to force fresh accumulation
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()
                module.momentum = None   # cumulative moving average (unbiased)

        n_batches = 0
        with torch.no_grad():
            for batch in train_loader:
                # Support both plain tensors (pretrain) and (x, y) tuples (finetune)
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                model(x)
                n_batches += 1
                if max_batches > 0 and n_batches >= max_batches:
                    break

        model.eval()
        print(f"[SWA] BN recalibration complete ({n_batches} batches processed).")
