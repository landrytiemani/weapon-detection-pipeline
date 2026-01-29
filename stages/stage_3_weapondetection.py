"""
Stage 3: Weapon Detection Module
"""

from models.weapon_detectors.rt_detr import RTDETRDetector
from models.weapon_detectors.yolov8_efficientvit import YOLOv8EfficientViT
from utils.analysis import get_model_summary
import cv2
import torch
import torchvision.ops as ops
import numpy as np
import time
from typing import List, Dict, Tuple, Any


class WeaponDetectionStage:
    """
    Stage 3: Weapon Detection with configurable parameters for ablation studies.
    
    Key improvements for RQ1:
    - Configurable crop_scale (H1.3): Sweep [1.0, 1.2, 1.5, 1.8, 2.0] to find optimal
    - Batch inference for GPU efficiency
    - Timing metrics collection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Normalize approach name
        self.model_name = str(self.config.get('approach', '')).lower().replace('-', '_')
        self.draw_enabled = self.config.get("draw", True)
        self.persons_with_weapons = set()
        
        # NEW: Configurable crop scale for H1.3 experiments
        # Original was hardcoded: size = int(max(width, height) * 1.2)
        self.crop_scale = self.config.get('crop_scale', 1.5)
        
        # NEW: Batch size for efficient GPU inference
        self.batch_size = self.config.get('batch_size', 8)
        
        # NEW: Timing metrics for performance analysis
        self.timing_metrics = {
            'crop_time': [],
            'inference_time': [],
            'nms_time': [],
            'total_time': []
        }

        # Resolve per-approach sub-config (accept common aliases)
        if self.model_name in ("rt_detr", "rtdetr"):
            rt_cfg = self.config.get('rt_detr') or self.config.get('rtdetr')
            if not rt_cfg:
                raise KeyError(
                    "Missing Stage-3 block for RT-DETR. Expected 'stage_3.rt_detr:' (or 'rtdetr:')."
                )
            self.detector = RTDETRDetector(rt_cfg)

        elif self.model_name in ("yolov8_efficientvit", "yolov8efficientvit"):
            ycfg = self.config.get('yolov8_efficientvit', {})
            self.detector = YOLOv8EfficientViT(ycfg)

        else:
            raise ValueError(f"Unknown approach for Stage 3: {self.model_name}")

        # Analyze the model for summary logging
        try:
            model_to_analyze = None
            if hasattr(self.detector, 'model') and hasattr(self.detector.model, 'model'):
                model_to_analyze = self.detector.model.model
            elif hasattr(self.detector, 'model'):
                model_to_analyze = self.detector.model

            self.model_stats = get_model_summary(model_to_analyze, (1, 3, 640, 640), self.model_name)
            self.model_stats['Model Name'] = self.model_name
        except Exception as e:
            print(f" Could not get model summary: {e}")
            self.model_stats = {"Model Name": self.model_name, "Total Params": "N/A", "GFLOPs": "N/A"}
        
        print(f"[Stage3] Initialized with crop_scale={self.crop_scale}, batch_size={self.batch_size}")

    def run(self, frame: np.ndarray, person_data: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Run weapon detection on person crops.
        
        Args:
            frame: Input frame (BGR)
            person_data: List of person detections with 'id' and 'bbox'
        
        Returns:
            Tuple of (annotated_frame, weapon_stats)
        """
        total_start = time.time()
        
        if not self.config.get('active', True) or len(person_data) == 0:
            return frame, {"count": 0, "avg_confidence": 0.0, "weapons": []}

        weapon_boxes = []
        confidences = []

        annotated_frame = frame.copy() if self.draw_enabled else frame

        # Phase 1: Collect all crops
        crop_start = time.time()
        crops = []
        crop_metadata = []  # (person_idx, x1, y1, x2, y2)
        
        for idx, person in enumerate(person_data):
            if 'bbox' not in person:
                continue

            px1, py1, px2, py2 = person['bbox']
            width = px2 - px1
            height = py2 - py1
            cx, cy = (px1 + px2) // 2, (py1 + py2) // 2

            # CHANGED: Use configurable crop_scale instead of hardcoded 1.2
            size = int(max(width, height) * self.crop_scale)
            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(frame.shape[1], cx + size // 2)
            y2 = min(frame.shape[0], cy + size // 2)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if self.draw_enabled:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crops.append(crop)
            crop_metadata.append((idx, x1, y1, x2, y2, person.get('id', idx)))
        
        self.timing_metrics['crop_time'].append(time.time() - crop_start)
        
        # Phase 2: Run inference (sequential for now, batch later)
        inference_start = time.time()
        
        for crop_idx, (crop, meta) in enumerate(zip(crops, crop_metadata)):
            person_idx, x1, y1, x2, y2, person_id = meta
            
            results = self.detector.detect(crop)

            if results and results[0].boxes:
                for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), 
                                    results[0].boxes.conf.cpu().numpy()):
                    wx1, wy1, wx2, wy2 = box
                    # Remap to full frame coordinates
                    wx1 += x1
                    wy1 += y1
                    wx2 += x1
                    wy2 += y1

                    weapon_boxes.append([wx1, wy1, wx2, wy2])
                    confidences.append(conf)
                    self.persons_with_weapons.add(person_id)
        
        self.timing_metrics['inference_time'].append(time.time() - inference_start)

        # Phase 3: NMS
        nms_start = time.time()
        if weapon_boxes:
            boxes_tensor = torch.tensor(weapon_boxes, dtype=torch.float32)
            conf_tensor = torch.tensor(confidences, dtype=torch.float32)
            nms_iou = float(self.config.get("nms_iou_threshold", 0.5))
            keep_idxs = ops.nms(boxes_tensor, conf_tensor, iou_threshold=nms_iou)

            weapon_boxes = [weapon_boxes[i] for i in keep_idxs]
            confidences = [confidences[i] for i in keep_idxs]
        
        self.timing_metrics['nms_time'].append(time.time() - nms_start)

        # Draw detections
        if self.draw_enabled:
            for (wx1, wy1, wx2, wy2), conf in zip(weapon_boxes, confidences):
                x1, y1, x2, y2 = map(int, [wx1, wy1, wx2, wy2])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Weapon {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"[DEBUG] Frame: Persons={len(person_data)} | Weapons={len(weapon_boxes)}")

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        
        self.timing_metrics['total_time'].append(time.time() - total_start)
        
        weapon_stats = {
            "count": len(weapon_boxes),
            "avg_confidence": avg_conf,
            "weapons": [
                {
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "confidence": float(c)
                }
                for (x1, y1, x2, y2), c in zip(weapon_boxes, confidences)
            ],
            "bboxes": [
                [float(x1), float(y1), float(x2), float(y2)]
                for (x1, y1, x2, y2) in weapon_boxes
            ],
            "weapon_person_ids": list(self.persons_with_weapons)
        }

        return annotated_frame, weapon_stats

    def get_timing_stats(self) -> Dict[str, float]:
        """Get average timing statistics for performance analysis."""
        stats = {}
        for key, values in self.timing_metrics.items():
            if values:
                stats[f'avg_{key}_ms'] = np.mean(values) * 1000
                stats[f'std_{key}_ms'] = np.std(values) * 1000
        return stats

    def reset_timing(self):
        """Reset timing metrics for new experiment."""
        for key in self.timing_metrics:
            self.timing_metrics[key] = []

    @staticmethod
    def _compute_iou(boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def get_torch_model(self):
        """
        Return the underlying torch.nn.Module for ONNX export.
        Works for RT-DETR and YOLOv8-EfficientViT wrappers.
        """
        m = getattr(self.detector, "model", None)
        if hasattr(m, "model"):
            return m.model
        return m

    @torch.no_grad()
    def export_onnx(self,
                    onnx_path="stage3_weapon.onnx",
                    input_size=(1, 3, 640, 640),
                    opset=17,
                    use_half=False,
                    cpu=True):
        model = self.get_torch_model()
        if model is None:
            raise RuntimeError("Stage-3: no torch model found to export.")

        if cpu:
            model = model.to("cpu")
        model.eval()

        dtype = torch.float16 if use_half else torch.float32
        device = next(model.parameters()).device
        dummy = torch.randn(*input_size, device=device, dtype=dtype)

        dynamic_axes = {
            "images":  {0: "batch"},
            "outputs": {0: "batch", 1: "num_queries"}
        }

        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["images"],
            output_names=["outputs"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
            export_params=True,
        )

        print(f"Stage-3 ONNX exported to {onnx_path}")
