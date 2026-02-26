"""
dataset.py — Pre-training & Fine-tuning Dataset & DataLoader
=============================================================
Source: arXiv:2601.06149v1, Section II-C / II-D / II-E
SSOT:   docs/work_plan.md, Part ה.1 (window_len=1800, stride=900)

PretrainDataset
---------------
• Loads all recordings listed in pretrain.csv.
• Generates every valid sliding-window start position for each recording.
• Lazily loads each window on __getitem__ to keep RAM footprint small.
• Returns (2, 1800) float32 tensors — channel 0 = FHR, channel 1 = UC.

FinetuneDataset
---------------
• Loads CTU-UHB recordings with labels from train.csv or val.csv.
• Generates sliding-window positions for each recording.
• Returns ((2, 1800) tensor, label) tuples — label=0 (normal) or 1 (acidemia).

Assumptions / Deviations
--------------------------
• window_stride = 900 samples (50% overlap) — deviation S4
• Both CTU-UHB and FHRMA recordings are pre-processed (2, T) float32 .npy.
• pretrain.csv columns: id, dataset, fname
  - dataset='ctg'   → processed_root / 'ctu_uhb'  / '{id}.npy'
  - dataset='fhrma' → processed_root / 'fhrma'     / '{id}.npy'
• train.csv / val.csv columns: id, target, fname
  - All from CTU-UHB → processed_root / 'ctu_uhb' / '{id}.npy'
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Dataset mapping helper
# ---------------------------------------------------------------------------

_DATASET_SUBDIR = {
    "ctg": "ctu_uhb",
    "fhrma": "fhrma",
}


# ---------------------------------------------------------------------------
# PretrainDataset
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """Sliding-window dataset over pre-training recordings.

    Args:
        pretrain_csv:   Path to `data/splits/pretrain.csv` (cols: id, dataset, fname).
        processed_root: Root directory containing `ctu_uhb/` and `fhrma/` subdirs.
        window_len:     Samples per window (default 1800 — ✓ paper).
        stride:         Sliding stride in samples (default 900 — ⚠ S4).
    """

    def __init__(
        self,
        pretrain_csv: Union[str, Path],
        processed_root: Union[str, Path],
        window_len: int = 1800,
        stride: int = 900,
    ):
        self.window_len = window_len
        self.stride = stride
        self.processed_root = Path(processed_root)

        df = pd.read_csv(pretrain_csv, dtype={"id": str, "dataset": str})

        # Build list of (npy_path, start_index) for every valid window
        self._windows: List[Tuple[Path, int]] = []
        missing: List[str] = []

        for _, row in df.iterrows():
            subdir = _DATASET_SUBDIR.get(str(row["dataset"]).strip().lower())
            if subdir is None:
                raise ValueError(
                    f"Unknown dataset '{row['dataset']}' for id={row['id']}. "
                    f"Expected one of: {list(_DATASET_SUBDIR.keys())}"
                )
            npy_path = self.processed_root / subdir / f"{row['id']}.npy"

            if not npy_path.exists():
                missing.append(str(npy_path))
                continue

            # Load only the shape to compute valid windows (no data yet)
            shape = np.load(npy_path, mmap_mode="r").shape  # (2, T)
            T = shape[1]
            starts = range(0, T - window_len + 1, stride)
            for start in starts:
                self._windows.append((npy_path, start))

        n_loaded = len(df) - len(missing)
        if missing:
            print(
                f"[PretrainDataset] WARNING: {len(missing)}/{len(df)} .npy files not found "
                f"(skipped). First 5: {missing[:5]}"
            )
            if len(missing) > len(df) * 0.5:
                raise RuntimeError(
                    f"[PretrainDataset] FATAL: {len(missing)}/{len(df)} files missing "
                    f"(>{50}%). Check that processed .npy files exist under "
                    f"{self.processed_root}. First missing: {missing[0]}"
                )

        if len(self._windows) == 0:
            raise RuntimeError(
                f"[PretrainDataset] FATAL: 0 windows created from {len(df)} recordings "
                f"({n_loaded} loaded, {len(missing)} missing). "
                f"Check paths and that recordings are long enough (>= {window_len} samples)."
            )

        print(
            f"[PretrainDataset] Loaded {n_loaded}/{len(df)} recordings -> "
            f"{len(self._windows)} windows (stride={stride})"
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return one window as (2, window_len) float32 tensor."""
        npy_path, start = self._windows[idx]
        # mmap_mode='r' avoids loading the whole file into RAM
        signal = np.load(npy_path, mmap_mode="r")[:, start : start + self.window_len]
        return torch.from_numpy(signal.copy())  # copy() to materialise mmap slice

    def __repr__(self) -> str:
        return (
            f"PretrainDataset(n_windows={len(self)}, "
            f"window_len={self.window_len}, stride={self.stride})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_pretrain_loaders(
    pretrain_csv: Union[str, Path],
    processed_root: Union[str, Path],
    window_len: int = 1800,
    stride: int = 900,
    batch_size: int = 64,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for pre-training.

    The split is purely random (no stratification needed for self-supervised
    pre-training).  90% train / 10% val by default.

    Args:
        pretrain_csv:   Path to pretrain.csv.
        processed_root: Root of processed .npy files.
        window_len:     Window size in samples (1800).
        stride:         Sliding stride in samples (900).
        batch_size:     Batch size (64 per config — ⚠ S6).
        val_fraction:   Fraction of windows used for validation (default 0.1).
        num_workers:    DataLoader worker processes (default 0 = main thread).
        seed:           RNG seed for reproducible split.

    Returns:
        (train_loader, val_loader)
    """
    dataset = PretrainDataset(pretrain_csv, processed_root, window_len, stride)

    n_total = len(dataset)
    n_val   = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# FinetuneDataset
# ---------------------------------------------------------------------------

class FinetuneDataset(Dataset):
    """Sliding-window dataset for fine-tuning with labels (acidemia classification).

    Args:
        split_csv:      Path to train.csv or val.csv (cols: id, target, fname).
        processed_root: Root directory containing `ctu_uhb/` subdir.
        window_len:     Samples per window (default 1800 — ✓ paper).
        stride:         Sliding stride in samples (default 900 — ⚠ S4).
        augment:        If True, apply training augmentations (noise + time jitter).
                        Should be False for validation/inference. — ⚠ S13
    """

    def __init__(
        self,
        split_csv: Union[str, Path],
        processed_root: Union[str, Path],
        window_len: int = 1800,
        stride: int = 900,
        augment: bool = False,
    ):
        self.window_len = window_len
        self.stride = stride
        self.augment = augment
        self.processed_root = Path(processed_root)

        df = pd.read_csv(split_csv, dtype={"id": str, "target": int})

        # Build list of (npy_path, start_index, label) for every valid window
        self._windows: List[Tuple[Path, int, int]] = []
        missing: List[str] = []

        for _, row in df.iterrows():
            # All finetune data is CTU-UHB
            npy_path = self.processed_root / "ctu_uhb" / f"{row['id']}.npy"
            label = int(row['target'])  # 0=normal, 1=acidemia

            if not npy_path.exists():
                missing.append(str(npy_path))
                continue

            # Load only the shape to compute valid windows (no data yet)
            shape = np.load(npy_path, mmap_mode="r").shape  # (2, T)
            T = shape[1]
            starts = range(0, T - window_len + 1, stride)
            for start in starts:
                self._windows.append((npy_path, start, label))

        n_loaded = len(df) - len(missing)
        if missing:
            print(
                f"[FinetuneDataset] WARNING: {len(missing)}/{len(df)} .npy files not found "
                f"(skipped). First 5: {missing[:5]}"
            )
            if len(missing) > len(df) * 0.5:
                raise RuntimeError(
                    f"[FinetuneDataset] FATAL: {len(missing)}/{len(df)} files missing "
                    f"(>{50}%). Check that processed .npy files exist under "
                    f"{self.processed_root}/ctu_uhb/. First missing: {missing[0]}"
                )

        if len(self._windows) == 0:
            raise RuntimeError(
                f"[FinetuneDataset] FATAL: 0 windows created from {len(df)} recordings "
                f"({n_loaded} loaded, {len(missing)} missing). "
                f"Check paths and that recordings are long enough (>= {window_len} samples)."
            )

        print(
            f"[FinetuneDataset] Loaded {n_loaded}/{len(df)} recordings -> "
            f"{len(self._windows)} windows (stride={stride})"
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return one window as ((2, window_len) float32 tensor, label).

        If self.augment=True, applies:
          1. Random time jitter ±30 samples (different view of the recording).
          2. Gaussian noise on FHR channel (ch 0), σ=0.02 (signal is normalised).
        Both augmentations are stochastic and only active during training.
        """
        npy_path, start, label = self._windows[idx]
        # mmap_mode='r' avoids loading the whole file into RAM
        sig = np.load(npy_path, mmap_mode="r")

        if self.augment:
            # 1. Random time jitter: shift window start by ±30 samples
            T_total = sig.shape[1]
            jitter  = np.random.randint(-30, 31)
            start   = int(np.clip(start + jitter, 0, T_total - self.window_len))

        window = sig[:, start : start + self.window_len].copy()  # materialise mmap

        if self.augment:
            # 2. Gaussian noise on FHR channel (channel 0 = FHR)
            window[0] = (window[0]
                         + np.random.normal(0, 0.02, self.window_len).astype(np.float32))

        return torch.from_numpy(window), label

    def __repr__(self) -> str:
        return (
            f"FinetuneDataset(n_windows={len(self)}, "
            f"window_len={self.window_len}, stride={self.stride})"
        )


# ---------------------------------------------------------------------------
# Fine-tuning DataLoader factory
# ---------------------------------------------------------------------------

def build_finetune_loaders(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    processed_root: Union[str, Path],
    window_len: int = 1800,
    stride: int = 900,
    train_stride: Optional[int] = None,
    augment: bool = False,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for fine-tuning.

    Args:
        train_csv:      Path to train.csv (441 CTU-UHB recordings).
        val_csv:        Path to val.csv (56 CTU-UHB recordings).
        processed_root: Root of processed .npy files.
        window_len:     Window size in samples (1800).
        stride:         Sliding stride for val DataLoader (default 900).
        train_stride:   Sliding stride for train DataLoader; falls back to stride.
                        Set to 60 (S13) for dense windows and more samples per epoch.
        augment:        Pass True to enable noise+jitter augmentation on train set.
        batch_size:     Batch size (32 per config — ⚠ S6).
        num_workers:    DataLoader worker processes (default 0 = main thread).
        pin_memory:     If True, use pinned (page-locked) memory for faster GPU transfer.

    Returns:
        (train_loader, val_loader)
    """
    eff_train_stride = train_stride if train_stride is not None else stride
    train_ds = FinetuneDataset(train_csv, processed_root, window_len, eff_train_stride,
                               augment=augment)
    val_ds   = FinetuneDataset(val_csv,   processed_root, window_len, stride)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, val_loader
