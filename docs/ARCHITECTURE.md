# Architecture Documentation

## System Overview

The Modular Weapon Detection Pipeline uses a two-stage hierarchical approach for real-time weapon detection with privacy preservation.

## Pipeline Flow

```
Input Frame → Stage 1 (Person Detection) → Stage 2 (Weapon Detection) → Output
                      ↓                            ↓
                 ByteTrack                   Post-Processing
                 Tracking                    + Privacy Module
```

## Stage 1: Person Detection

### Models Supported
| Model | Architecture | Parameters | GFLOPs |
|-------|-------------|------------|--------|
| YOLOv8n | CSPDarknet + PANet | 3.2M | 8.7 |
| SSD-MobileNetV2 | MobileNetV2 + SSD | 3.4M | 3.4 |

### ByteTrack Integration
- Kalman filter for motion prediction
- Hungarian algorithm for matching
- Two-stage association (high/low confidence)

## Stage 2: Weapon Detection

### Models Supported
| Model | Architecture | Parameters | GFLOPs | mAP@0.5 |
|-------|-------------|------------|--------|---------|
| RT-DETR | ResNet50 + Transformer | 32M | 81.4 | 0.721 |
| YOLOv8-EfficientViT | EfficientViT + YOLOv8 | 4.5M | 6.2 | 0.669 |

## Post-Processing

1. **Per-class confidence gating** - Class-specific thresholds
2. **Local NMS** - Per-class duplicate removal
3. **Global NMS** - Cross-crop duplicate removal
4. **Cross-class suppression** - Handle handgun/knife overlaps
5. **Geometry filtering** - Size and aspect ratio constraints

## Privacy Module

- **Selective face blurring** - Only non-targets (people without weapons)
- **Methods**: Pixelate (fast) or Gaussian blur (smooth)
- **Overhead**: <10% latency increase
