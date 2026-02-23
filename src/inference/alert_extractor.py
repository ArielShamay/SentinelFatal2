"""
alert_extractor.py — Alert Segment Extraction + Feature Computation
====================================================================
Source: arXiv:2601.06149v1, Section II-F, Figure 5
SSOT:   docs/work_plan.md, Part ה.6 + חלק ו, שלב 5.2-5.3

Stage 2 of the alerting pipeline:
    1. extract_alert_segments()   — find contiguous score > threshold regions.
    2. compute_alert_features()   — compute exactly 4 features from a segment
                                    (P5 fix v2: time-integral features normalized
                                    by dt = stride/fs).

Alert threshold = 0.4 (Deviation S11 — lowered from 0.5; validated 2026-02-23).
Decision threshold = 0.284 (Youden-optimal, test AUC=0.839, Sens=0.818).
Feature count   = 4   (LOCKED — paper Section II-F).

Usage::

    from src.inference.alert_extractor import (
        extract_alert_segments, compute_alert_features,
        ALERT_THRESHOLD, DECISION_THRESHOLD
    )

    segments = extract_alert_segments(scores, threshold=ALERT_THRESHOLD)
    for start_s, end_s, seg_scores in segments:
        feats = compute_alert_features(seg_scores, inference_stride=1, fs=4)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds  (Deviation S11 — validated 2026-02-23)
# ---------------------------------------------------------------------------

ALERT_THRESHOLD: float = 0.4     # S11: lowered from paper 0.5; eliminates zero-segment FNs
DECISION_THRESHOLD: float = 0.284  # Youden-optimal LR decision threshold (test AUC 0.839)


# ---------------------------------------------------------------------------
# extract_alert_segments
# ---------------------------------------------------------------------------

def extract_alert_segments(
    scores: List[Tuple[int, float]],
    threshold: float = ALERT_THRESHOLD,
) -> List[Tuple[int, int, List[float]]]:
    """Identify contiguous windows where score > threshold.

    Source: arXiv:2601.06149v1, Section II-F — "alert segment = contiguous
    sequence of windows with NN score > 0.5".

    Args:
        scores:    List of (start_sample, score) tuples from inference_recording().
        threshold: Score cutoff for alert classification (default 0.5).

    Returns:
        List of (start_sample, end_sample, segment_scores) where:
            start_sample:   int         — first sample of the alert segment.
            end_sample:     int         — first sample of the window AFTER the
                                          last alert window (exclusive endpoint).
            segment_scores: List[float] — per-window P(acidemia) values.

        Empty list if no windows exceed threshold.
    """
    if len(scores) == 0:
        return []

    alert_mask = [s > threshold for _, s in scores]

    segments: List[Tuple[int, int, List[float]]] = []
    in_segment = False
    seg_start_sample: int = 0
    seg_scores: List[float] = []

    for i, (start_sample, score) in enumerate(scores):
        if alert_mask[i]:
            if not in_segment:
                # Begin new segment
                in_segment = True
                seg_start_sample = start_sample
                seg_scores = []
            seg_scores.append(score)
        else:
            if in_segment:
                # Close the segment: end_sample = start of *this* (non-alert) window
                segments.append((seg_start_sample, start_sample, list(seg_scores)))
                in_segment = False
                seg_scores = []

    # Handle segment running to the end of the recording
    if in_segment and seg_scores:
        # Estimate end: last start_sample + 1800 (window length)
        last_start, _ = scores[-1]
        end_sample = last_start + 1800
        segments.append((seg_start_sample, end_sample, list(seg_scores)))

    return segments


# ---------------------------------------------------------------------------
# compute_alert_features
# ---------------------------------------------------------------------------

def compute_alert_features(
    segment_scores: List[float],
    inference_stride: int = 1,
    fs: float = 4.0,
) -> Dict[str, float]:
    """Compute exactly 4 alert features from a contiguous alert segment.

    Source: arXiv:2601.06149v1, Section II-F.
    P5 fix v2: All time-/integral-based features are normalized by
                dt = inference_stride / fs (seconds per step), so the
                feature values are independent of which stride was used.
                LR training and evaluation MUST use the same stride.

    Args:
        segment_scores:   Per-window P(acidemia) scores within the segment.
        inference_stride: Stride used in inference_recording() (samples).
                          MUST match the stride used to train the LR model.
        fs:               Signal sampling frequency in Hz (4 Hz).

    Returns:
        dict with exactly 4 keys:
            'segment_length'    : float — total alert duration in minutes.
            'max_prediction'    : float — max P(acidemia) in the segment.
            'cumulative_sum'    : float — integral of scores (units: score*seconds).
            'weighted_integral' : float — integral of (score-0.5)^2 (score^2*seconds).

    Raises:
        ValueError: If segment_scores is empty.
    """
    if len(segment_scores) == 0:
        raise ValueError("segment_scores must be non-empty.")

    p = np.asarray(segment_scores, dtype=np.float64)
    dt = inference_stride / fs   # seconds per step   (repro: 0.25s, runtime: 15s)

    features: Dict[str, float] = {
        # Duration of alert segment in minutes
        "segment_length":    float(len(p) * dt / 60.0),
        # Maximum NN score (point value — no stride dependency)
        "max_prediction":    float(np.max(p)),
        # Integral of NN score (score * seconds)
        "cumulative_sum":    float(np.sum(p) * dt),
        # Integral of (score - 0.5)^2 (weighted severity)
        "weighted_integral": float(np.sum((p - 0.5) ** 2) * dt),
    }

    assert len(features) == 4, "BUG: must return exactly 4 features"
    return features


# ---------------------------------------------------------------------------
# Convenience: feature vector for recordings with NO alert segments
# ---------------------------------------------------------------------------

ZERO_FEATURES: Dict[str, float] = {
    "segment_length":    0.0,
    "max_prediction":    0.0,
    "cumulative_sum":    0.0,
    "weighted_integral": 0.0,
    "n_alert_segments":  0.0,
    "alert_fraction":    0.0,
}
"""Zero-value feature vector for recordings that produce no alert segments.

Assumption (logged as S10 in deviation_log.md):
    Recordings with zero alert windows are assigned ZERO_FEATURES, representing
    complete absence of alert activity.  Paper does not address this case explicitly.
S14: Added n_alert_segments and alert_fraction (record-level features).
"""
