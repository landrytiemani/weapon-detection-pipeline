# Research Experiments

## Overview

| RQ | Focus | Script |
|----|-------|--------|
| RQ1 | Ablation Study | `RQ/run_rq1_ablation.py` |
| RQ2 | Architecture Comparison | `RQ/run_rq2_architecture.py` |
| RQ3 | Temporal Tracking | `RQ/run_rq3_tracking.py` |
| RQ4 | Privacy Preservation | `RQ/run_rq4_privacy.py` |

## RQ1: Ablation Study

Tests contribution of individual components:
- Crop scale: [1.5, 2.0, 2.5, 3.0]
- Overlap threshold: [0.3, 0.5, 0.7]
- TTA: [enabled, disabled]

## RQ2: Architecture Comparison

Compares RT-DETR vs YOLOv8-EfficientViT:
- Accuracy (mAP@0.5, per-class)
- Efficiency (GFLOPs, FPS)
- Trade-offs

## RQ3: Tracking Experiments

Tests ByteTrack integration:
- With/without tracking
- Frame gaps: [1, 2, 3, 5, 10]
- Track buffer sizes

## RQ4: Privacy Analysis

Tests privacy-preserving mechanisms:
- Selective vs blanket blurring
- Pixelate vs Gaussian
- Latency overhead
