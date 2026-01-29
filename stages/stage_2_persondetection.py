from models.person_detectors.yolov8_tracker import YOLOv8Tracker
from models.person_detectors.ssd_mobilenet_bytetrack import SSDMobileNetByteTrack
from utils.analysis import get_model_summary
from threading import Timer

import cv2
import numpy as np
from collections import defaultdict
import torch

class PersonDetectionStage:
    def __init__(self, config):
        self.config = config
        self.model_name = config['approach']
        self.draw_enabled = config.get("draw", True)
        self.track_history = defaultdict(list)
        self.person_color = (100, 200, 50)
        self.frame_gap = config.get("frame_gap", 1)  # Detect every N frames when tracker enabled
        self.frame_id = 0
        self.last_detections = None  # Store last detections for tracking

        # Load detector + tracker
        if self.model_name == "yolov8_tracker":
            self.detector = YOLOv8Tracker(config['yolov8_tracker'])
        elif self.model_name == "ssd_mobilenet_bytetrack":
            self.detector = SSDMobileNetByteTrack(config.get('ssd_mobilenet_bytetrack', {}))
        else:
            raise ValueError(f"Unknown approach for Stage 2: {self.model_name}")

        self.use_tracker = getattr(self.detector, 'use_tracker', True)

        try:
            model_obj = getattr(self.detector, 'model', None)
            input_size = (1, 3, 300, 300) if self.model_name == "ssd_mobilenet_bytetrack" else (1, 3, 640, 640)
            self.model_stats = get_model_summary(model_obj, input_size, self.model_name)
            self.model_stats['Model Name'] = self.model_name
        except Exception as e:
            print(f"Could not analyze model: {e}")
            self.model_stats = {"Model Name": self.model_name, "Total Params": "N/A", "GFLOPs": "N/A"}

    def run(self, frame, frame_idx=0):
        if not self.config.get('active', True):
            return frame, [], {"count": 0, "avg_confidence": 0.0}

        annotated_frame = frame.copy() if self.draw_enabled else frame

        # Decide whether to run detection or use tracker
        run_detection = (not self.use_tracker) or (frame_idx % self.frame_gap == 0)

        if run_detection:
            results = self.detector.track(frame)
            if isinstance(results, tuple) and len(results) == 2:
                annotated_frame, person_data = results
            else:
                person_data = results if isinstance(results, list) else []
        else:
            person_data = self.detector.predict_only()

        num_detections = len(person_data)
        confidences = [p['conf'] for p in person_data]

        if not self.use_tracker:
            self.track_history.clear()

        # Draw detections with color based on whether detector or tracker was used
        if self.draw_enabled:
            for person in person_data:
                x1, y1, x2, y2 = person['bbox']
                track_id = person['id']
                conf = person['conf']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                box_color = (0, 255, 0) if run_detection else (255, 0, 0)  # Green for detector, Blue for tracker

                if None not in (x1, y1, x2, y2):
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

                label = f"ID: {track_id} ({conf:.2f})"
                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                tx, ty = x1, y1 - 10 if y1 - 10 > th else y1 + th + 10
                cv2.rectangle(
                    annotated_frame,
                    (int(tx), int(ty - th - 5)),
                    (int(tx + tw + 5), int(ty + bl + 5)),
                    box_color,
                    -1
                )
                cv2.putText(annotated_frame, label, (int(tx + 2), int(ty - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1, cv2.LINE_AA)

                # Draw path trail
                if self.use_tracker:
                    self.track_history[track_id].append(center)
                    if len(self.track_history[track_id]) > 90:
                        self.track_history[track_id].pop(0)
                    if len(self.track_history[track_id]) > 1:
                        points_np = np.array(self.track_history[track_id], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points_np], isClosed=False, color=(0, 255, 255), thickness=2)
                        cv2.circle(annotated_frame, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)

        avg_confidence = float(np.mean(confidences)) if num_detections > 0 else 0.0
        person_stats = {"count": num_detections, "avg_confidence": avg_confidence}
        return annotated_frame, person_data, person_stats
    
    
    
    def reset(self):
        """Reset the stage for a new experiment/video sequence."""
        # Clear tracking history
        self.track_history.clear()
        
        # Reset detector if it has a reset method
        if hasattr(self.detector, 'reset'):
            self.detector.reset()
        
        # Reset any frame counters in the detector
        if hasattr(self.detector, 'frame_count'):
            self.detector.frame_count = 0
        if hasattr(self.detector, 'frame_idx'):
            self.detector.frame_idx = 0
        
        # If using ByteTrack, reset the tracker
        if hasattr(self.detector, 'tracker') and self.detector.tracker is not None:
            if hasattr(self.detector.tracker, 'reset'):
                self.detector.tracker.reset()
            # ByteTrack specific resets
            if hasattr(self.detector.tracker, 'frame_id'):
                self.detector.tracker.frame_id = 0
            if hasattr(self.detector.tracker, 'track_id_count'):
                self.detector.tracker.track_id_count = 0
        
        """Reset tracker state for new experiment"""
        self.frame_id = 0
        self.last_detections = None
        if hasattr(self, 'tracker'):
            # Reset ByteTracker state
            self.tracker.frame_id = 0
            self.tracker.tracked_stracks = []
            self.tracker.lost_stracks = []
            self.tracker.removed_stracks = []

    def track(self, frame, force_detect=True):
        """
        Run detection and/or tracking
        Args:
            frame: Input frame
            force_detect: If True, run detection. If False, only track.
        """
        self.frame_id += 1
        timer = Timer()
        timer.tic()
        
        im_height, im_width, _ = frame.shape
        img_info = (im_height, im_width)
        img_size = (im_height, im_width)
        
        if force_detect:
            # Run detection
            print(f"[DEBUG] YOLOv8 Frame {self.frame_id}: Running DETECTION")
            results = self.model.track(
                source=frame,
                persist=True,
                conf=self.config.get('confidence_threshold', 0.25),
                iou=self.config.get('iou_threshold', 0.45),
                classes=0,
                device=self.config.get('device', 0),
                max_det=self.config.get('max_det', 300),
                verbose=False
            )
            
            if results is None or results[0].boxes is None:
                detections = np.zeros((0, 5), dtype=np.float32)
            else:
                detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = box.conf[0].cpu().item()
                    detections.append([x1, y1, x2, y2, score])
                detections = np.array(detections, dtype=np.float32)
            
            self.last_detections = detections
            print(f"  Found {len(detections)} persons")
        else:
            # Only track, no detection
            print(f"[DEBUG] YOLOv8 Frame {self.frame_id}: TRACKING ONLY")
            detections = np.zeros((0, 5), dtype=np.float32)  # No new detections
        
        # Update tracker with detections (empty or not)
        if self.use_tracker:
            online_targets = self.tracker.update(detections, img_info, img_size)
            
            person_data = []
            for t in online_targets:
                x1, y1, w, h = t.tlwh
                x2, y2 = x1 + w, y1 + h
                person_data.append({
                    "id": t.track_id,
                    "bbox": list(map(int, [x1, y1, x2, y2])),
                    "conf": t.score
                })
        else:
            # No tracking, just return detections
            person_data = [{
                "id": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(score)
            } for i, (x1, y1, x2, y2, score) in enumerate(detections)]
        
        timer.toc()
        return frame, person_data

# Onxx creation for visualization ---

    def get_torch_model(self):
        """
        Return the underlying torch.nn.Module for ONNX export.
        Handles wrappers like Ultralytics where the real module may be nested.
        """
        m = getattr(self.detector, "model", None)
        # Ultralytics YOLO usually nests at .model.model
        if hasattr(m, "model"):
            return m.model
        return m  # e.g., SSD MobileNet or Yolov8n

    import torch

    @torch.no_grad()
    def export_onnx(self,
                    onnx_path="stage2_person.onnx",
                    input_size=(1, 3, 640, 640),
                    opset=17,
                    use_half=False):
        model = self.get_torch_model()
        if model is None:
            raise RuntimeError("Stage-2: no torch model found to export.")

        model.eval()

        # Keep dummy input consistent with the model
        device = next(model.parameters()).device
        dtype = torch.float16 if use_half else torch.float32
        dummy = torch.randn(*input_size, device=device, dtype=dtype)

        torch.onnx.export(
            model,
            dummy,
            onnx_path,                                # <-- use the correct var
            input_names=["images"],
            output_names=["outputs"],
            opset_version=opset,                      # <-- use the function arg
            dynamic_axes={"images": {0: "batch"},
                        "outputs": {0: "batch"}},
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,   # be explicit about eval
            verbose=False,
            export_params=True,                      # default True, kept for clarity
        )
        print(f"Stage-2 ONNX exported to {onnx_path}")

# Add this to your stage_2_persondetection.py or create fast variant

class FastPersonDetectionStage:
    def __init__(self, config):
        self.config = config
        # Force smaller model and faster settings
        fast_config = config.copy()
        fast_config['yolov8_tracker']['confidence_threshold'] = 0.25  # Fewer detections
        fast_config['yolov8_tracker']['max_det'] = 3  # Max 3 persons per frame
        fast_config['crop_scale'] = 2.0  # Smaller crops
        
        self.detector = YOLOv8Tracker(fast_config['yolov8_tracker'])
        
    def run(self, frame, frame_idx=0):
        # Ultra-fast path - skip every other frame when tracking
        if hasattr(self, 'use_tracker') and self.use_tracker and frame_idx % 2 == 1:
            return frame, self.last_persons, {"count": len(self.last_persons), "avg_confidence": 0.7}
        
        # Fast detection
        results = self.detector.track(frame)
        self.last_persons = results if isinstance(results, list) else []
        return frame, self.last_persons, {"count": len(self.last_persons), "avg_confidence": 0.7}