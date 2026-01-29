##

from ultralytics import YOLO
from ptflops import get_model_complexity_info
import torch

class YOLOv8EfficientViT:
    def __init__(self, config):
        from ultralytics import YOLO
        self.model = YOLO(config['model_path'])  # Custom backbone
        self.config = config
        print(f"OBJECT DETECTOR: Loaded YOLOv8-EfficientViT from: {config['model_path']}")

    def detect(self, frame):
        return self.model(frame, conf=self.config['confidence_threshold'])

    
