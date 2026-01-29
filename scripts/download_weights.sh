#!/bin/bash
echo "Creating directories..."
mkdir -p weights/person/yolov8n
mkdir -p weights/weapon/rt_detr
mkdir -p weights/weapon/efficientvit_yolov8

echo "Downloading YOLOv8n..."
wget -q -O weights/person/yolov8n/yolov8n.pt \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

echo "✓ YOLOv8n downloaded"
echo "For weapon detection weights, see docs/WEIGHTS.md"
