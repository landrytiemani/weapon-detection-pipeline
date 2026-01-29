# Model Weights

## Download

### Person Detection
- **YOLOv8n**: [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
  - Place in: `weights/person/yolov8n/`

### Weapon Detection
- **RT-DETR**: Custom trained (contact author)
- **YOLOv8-EfficientViT**: Custom trained (contact author)

## Directory Structure

```
weights/
├── person/
│   └── yolov8n/
│       └── yolov8n.pt
└── weapon/
    ├── rt_detr/
    │   └── rt_detr.pt
    └── efficientvit_yolov8/
        └── efficientvit_yolov8.pt
```

## Training Your Own

See `notebooks/Train.ipynb` for training instructions.

### Dataset Format
YOLO format: `class_id center_x center_y width height`
- 0: handgun
- 1: knife
