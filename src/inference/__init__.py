"""
src/inference — Inference & Alerting pipeline for SentinelFatal2
=================================================================
Modules:
    sliding_window   — inference_recording() with REPRO / RUNTIME strides
    alert_extractor  — extract_alert_segments() + compute_alert_features()
"""

from src.inference.sliding_window import (
    INFERENCE_STRIDE_REPRO,
    INFERENCE_STRIDE_RUNTIME,
    inference_recording,
)
from src.inference.alert_extractor import (
    extract_alert_segments,
    compute_alert_features,
)

__all__ = [
    "INFERENCE_STRIDE_REPRO",
    "INFERENCE_STRIDE_RUNTIME",
    "inference_recording",
    "extract_alert_segments",
    "compute_alert_features",
]
