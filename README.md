# Lightweight Modular Real-Time Weapon Detection Framework for Edge Deployment Optimization

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

**A Lightweight Modular Real-Time Weapon Detection Framework Using Advanced Computer Vision Techniques for Edge Deployment Optimization**

[Overview](#abstract) •
[Architecture](#architecture) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Results](#results) •
[Citation](#citation)

</div>

---

## Abstract

This repository contains the complete implementation for doctoral dissertation research at **Harrisburg University of Science and Technology**. The framework introduces a novel **three-stage hierarchical detection pipeline** combining person-centric cropping, configurable weapon detection architectures, and privacy-preserving mechanisms for real-world surveillance deployment.

### Key Achievements

| Metric | Result | Description |
|--------|--------|-------------|
| **mAP@0.5** | +46% relative | Person-centric cropping vs full-frame detection |
| **False Positives** | -70.5% | Through hierarchical NMS processing |
| **Efficiency** | 8× reduction | EfficientViT vs RT-DETR computational cost |
| **Privacy** | 2.2% overhead | GDPR-compliant selective face blurring |

---

## Key Contributions

| # | Contribution | Research Question |
|---|--------------|-------------------|
| 1 | **Modular Pipeline Architecture** — Three-stage hierarchical detection with person-centric cropping achieving +21.4pp mAP50 improvement | RQ1 |
| 2 | **Architecture Comparison** — EfficientViT-YOLOv8 outperforms RT-DETR while using 8× less computation | RQ2 |
| 3 | **Temporal Tracking Integration** — ByteTrack enables 30.4% FPS improvement via frame skipping | RQ3 |
| 4 | **Privacy-Preserving Detection** — Selective face blurring with 0pp accuracy loss and <2.5% overhead | RQ4 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODULAR WEAPON DETECTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐     ┌─────────────────────┐     ┌────────────────────────┐   │
│  │  INPUT   │     │      STAGE 2        │     │       STAGE 3          │   │
│  │  FRAME   │────▶│  Person Detection   │────▶│   Weapon Detection     │   │
│  │          │     │  YOLOv8n + ByteTrack│     │  (EfficientViT/RT-DETR)│   │
│  └──────────┘     └─────────────────────┘     └────────────────────────┘   │
│                            │                            │                   │
│                            ▼                            ▼                   │
│                   ┌─────────────────┐         ┌──────────────────┐         │
│                   │ Person Crops    │         │ Post-Processing  │         │
│                   │ • Scale: 1.8×   │         │ • Local NMS      │         │
│                   │ • Square aspect │         │ • Global NMS     │         │
│                   │ • Overlap filter│         │ • Cross-class    │         │
│                   └─────────────────┘         └──────────────────┘         │
│                                                         │                   │
│                                                         ▼                   │
│                                                ┌──────────────────┐         │
│                                                │  Privacy Module  │         │
│                                                │ Selective Blur   │         │
│                                                │ (non-targets)    │         │
│                                                └──────────────────┘         │
│                                                         │                   │
│                                                         ▼                   │
│                                                ┌──────────────────┐         │
│                                                │     OUTPUT       │         │
│                                                │ Weapon Detections│         │
│                                                │ + Track IDs      │         │
│                                                └──────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 2: Person Detection & Tracking

| Component | Model | GFLOPs | Purpose |
|-----------|-------|--------|---------|
| Detector | YOLOv8n | 8.7 | Real-time person detection |
| Alternative | SSD-MobileNetV2 | 3.4 | Ultra-lightweight option |
| Tracker | ByteTrack | ~0.1 | Multi-object tracking & frame skipping |

### Stage 3: Weapon Detection

| Architecture | Type | GFLOPs | mAP@0.5 | Recommendation |
|-------------|------|--------|---------|----------------|
| **YOLOv8-EfficientViT** | CNN-Transformer Hybrid | 6.2 | **0.669** | ✓ **Edge deployment** |
| RT-DETR | Pure Transformer | 81.4 | 0.602 | High-compute scenarios |

> **Key Finding**: EfficientViT achieves 108.5% of RT-DETR's accuracy at only 12.5% of the computational cost — an **8× efficiency advantage**.

---

## Research Questions & Findings

### RQ1: Modular Architecture Ablation

> *How do individual pipeline components contribute to detection accuracy?*

| Hypothesis | Threshold | Result | Status |
|------------|-----------|--------|--------|
| **H1.1** Person-centric cropping improves mAP50 | ≥5pp | **+21.4pp** |  SUPPORTED |
| **H1.2** Hierarchical NMS reduces false positives | ≥20% | **70.5%** |  SUPPORTED |
| **H1.3** Optimal crop scale in [1.0, 1.5] | — | **1.8 optimal** |  NOT SUPPORTED |

**Key Insight**: Larger crop scales (1.8-2.0×) capture weapons at body periphery better than tight crops.

### RQ2: Architecture Comparison

> *How does RT-DETR (Transformer) compare to YOLOv8-EfficientViT (CNN)?*

| Hypothesis | Expected | Result | Status |
|------------|----------|--------|--------|
| **H2.1** RT-DETR achieves parity or superiority | RT-DETR ≥ EfficientViT | EfficientViT **+6.7pp** |  NOT SUPPORTED |
| **H2.2** EfficientViT ≥90% accuracy at ≤50% cost | — | **108.5% @ 12.5%** |  SUPPORTED |
| **H2.3** RT-DETR advantage on small objects (knives) | RT-DETR better | EfficientViT **+14.7pp** |  NOT SUPPORTED |
| **H2.4** Both achieve ≥10 FPS real-time | ≥10 FPS | RT-DETR: 12.2, EffViT: 13.8 |  SUPPORTED |

**Key Insight**: Person-centric cropping may reduce the benefit of transformer global attention, favoring efficient local feature extraction.

### RQ3: Temporal Tracking Integration

> *How does ByteTrack affect computational efficiency?*

| Hypothesis | Threshold | Result | Status |
|------------|-----------|--------|--------|
| **H3.1** Tracking reduces GFLOPs ≥33% | ≥33% | **8.1% max** |  NOT SUPPORTED |
| **H3.2** Frame gap 3-5 achieves ≥30% FPS gain | ≥30% FPS, ≤2pp loss | **30.4% @ ~1.8pp** | SUPPORTED |

**Key Insight**: Weapon detector (Stage 3) dominates pipeline cost; person detection savings are proportionally small.

### RQ4: Privacy-Preserving Detection

> *Can privacy protection be achieved with minimal performance impact?*

| Hypothesis | Threshold | Result | Status |
|------------|-----------|--------|--------|
| **H4.1** Privacy adds ≤5% computational overhead | ≤5% | **2.2%** |  SUPPORTED |
| **H4.2** Privacy causes ≤2pp accuracy loss | ≤2pp | **0pp** |  SUPPORTED |

**Privacy Design**: Selective face blurring applied only to non-weapon-bearing individuals preserves threat actor identifiability while protecting bystander privacy (GDPR compliance).

---

## Dataset

**WeaponSense Dataset** — Custom-curated for weapon detection research

| Split | Images | Purpose |
|-------|--------|---------|
| Train | 2,273 | Model training |
| Validation | 474 | Hyperparameter tuning |
| Test | 276 | Final evaluation |

**Classes**: `handgun`, `knife`

---

## Installation

### Prerequisites

- **OS**: Ubuntu 20.04+ / Windows 10+
- **Python**: 3.10+
- **GPU**: NVIDIA with 8GB+ VRAM (recommended)
- **CUDA**: 11.8+

### Quick Install

```bash
# Clone repository
git clone https://github.com/landrytiemani/weapon-detection-pipeline.git
cd weapon-detection-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Download weights
bash scripts/download_weights.sh
```

### Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

from ultralytics import YOLO
print("✓ Installation successful!")
```

---

## Quick Start

### Run Inference

```bash
# Single experiment with default settings
python main_perclass.py --config config.yaml
```

### Run Research Experiments

```bash
# RQ1: Ablation Study (H1.1, H1.2, H1.3)
python RQ/run_rq1_ablation.py

# RQ2: Architecture Comparison (H2.1-H2.4)
python RQ/run_rq2_architecture.py

# RQ3: Tracking Experiments (H3.1, H3.2)
python RQ/run_rq3_tracking.py

# RQ4: Privacy Analysis (H4.1, H4.2)
python RQ/run_rq4_privacy.py
```

---

## Project Structure

```
weapon-detection-pipeline/
│
├── main_perclass.py             # Main pipeline entry point
├── config.yaml                  # Primary configuration
├── requirements.txt             # Dependencies
│
├── stages/                      # Pipeline stages
│   └── stage_2_persondetection.py    # Person detection + ByteTrack    
│   └── stage_3_weapondetection.py    # Weapon detection
├── tracker/                     # ByteTrack implementation
│   ├── byte_tracker.py          # Main tracker
│   ├── kalman_filter.py         # Motion prediction
│   ├── matching.py              # Hungarian matching
│   └── basetrack.py             # Track base class
│
├── utils/                       # Utilities
│   ├── box_utils.py             # Bounding box operations
│   ├── evaluation.py            # mAP calculation (PipelineEvaluator)
│   ├── privacy.py               # Face blurring module
│   ├── visualization.py         # Debug visualizations
│   ├── flops_utils.py           # GFLOPs computation
│   └── report_utils.py          # Report generation
│
├── RQ/                          # Research experiments
│   ├── run_rq1_ablation.py      # Modular ablation study
│   ├── run_rq2_architecture.py  # RT-DETR vs EfficientViT
│   ├── run_rq3_tracking.py      # ByteTrack frame skipping
│   ├── run_rq4_privacy.py       # Privacy preservation
│   
│
├── weights/                     # Model weights
│   ├── person/                  # Person detection models
│   │   └── yolov8n/
        └── ssd/
│   └── weapon/                  # Weapon detection models
│       ├── efficientvit_yolov8/
│       └── rt_detr/
│
├── data/                  # Weaponsense dataset
│    ├── train/
│    ├── val/  
│    └── test/  
│      
└── Results/                     # Experiment outputs
    ├── rq1_ablation/
    ├── rq2_architecture/
    ├── rq3_tracking/
    └── rq4_privacy/
```

---

## Configuration

### Main Configuration (`config.yaml`)

```yaml
pipeline:
  frames_dir: testing_data/weaponsense/test/images
  labels_dir: testing_data/weaponsense/test/labels

stage_2:
  approach: yolov8_tracker
  crop_scale: 1.8              # Optimal from H1.3
  crop_overlap_threshold: 0.5
  use_tracker: false
  frame_gap: 1
  skip_person_detection: false  # Set true for H1.1 ablation
  
  yolov8_tracker:
    model_path: weights/person/yolov8n/yolov8n.pt
    confidence_threshold: 0.15

stage_3:
  approach: yolov8_efficientvit  # Recommended for edge
  imgsz: 512
  nms_iou_threshold: 0.45
  global_nms_threshold: 0.25
  min_final_confidence: 0.45
  names: ["handgun", "knife"]
  
  yolov8_efficientvit:
    model_path: weights/weapon/efficientvit_yolov8/efficientvit_yolov8.pt
    confidence_threshold: 0.20
  
  rt_detr:
    model_path: weights/weapon/rt_detr/rt_detr.pt
    confidence_threshold: 0.25

privacy:
  enabled: true
  scope: "non_targets"         # Only blur non-weapon-bearing individuals
  face_blur:
    enabled: true
    method: "pixelate"         # Options: pixelate, gaussian
    pixel_block: 15
```

---

## Results Summary

### Computational Profile

| Stage | Model | GFLOPs | Input Size |
|-------|-------|--------|------------|
| Stage 2 | YOLOv8n | 13.59 | 800×800 |
| Stage 3 | EfficientViT-YOLOv8 | 6.2 | 640×640 |
| Stage 3 | RT-DETR | 81.4 | 640×640 |

**Total Pipeline (EfficientViT, 3 crops/frame)**: ~32.2 GFLOPs  
**Total Pipeline (RT-DETR, 3 crops/frame)**: ~257.8 GFLOPs

### Deployment Recommendation

| Scenario | Architecture | Crop Scale | Tracking | Privacy |
|----------|--------------|------------|----------|---------|
| **Edge Device** | EfficientViT | 1.8 | gap=3 | Enabled |
| **High Accuracy** | RT-DETR | 2.0 | gap=1 | Enabled |
| **Ultra-Fast** | EfficientViT | 1.5 | gap=5 | Disabled |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{ytiemani2026weapon,
  title     = {A Lightweight Modular Real-Time Weapon Detection Framework for Edge Deployment Optimization},
  author    = {Yves Tiemani, Yves Cedric},
  year      = {2026},
  school    = {Harrisburg University of Science and Technology},
  type      = {Ph.D. Dissertation},
  department = {Data Sciences}
}
```

### Related Works

```bibtex
@inproceedings{zhang2022bytetrack,
  title     = {ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author    = {Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and others},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}

@inproceedings{lv2024rtdetr,
  title     = {DETRs Beat YOLOs on Real-time Object Detection},
  author    = {Lv, Wenyu and Xu, Shangliang and Zhao, Yian and others},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}

@article{cai2023efficientvit,
  title     = {EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction},
  author    = {Cai, Han and Gan, Chuang and Han, Song},
  journal   = {arXiv preprint arXiv:2205.14756},
  year      = {2023}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 & RT-DETR implementations
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking algorithm
- [EfficientViT](https://github.com/mit-han-lab/efficientvit) — Efficient vision transformer backbone
- **Harrisburg University of Science and Technology** — Doctoral program support

---

<div align="center">

**Developed for Ph.D. Dissertation in Data Sciences**

Harrisburg University of Science and Technology

*Yves Tiemani • Expected May 2026*

</div>
