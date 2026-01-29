from ultralytics import YOLO
from tracker.byte_tracker import BYTETracker
from tracker.tracking_utils.timer import Timer
from types import SimpleNamespace
import numpy as np
import cv2


class YOLOv8Tracker:
    def __init__(self, config):
        self.config = config
        self.use_tracker = config.get('use_tracker', True)
        self.model = YOLO(config['model_path'])
        self.model.task = "detect"
        print(f"TRACKER: Loaded YOLOv8 from: {config['model_path']}")
        self.frame_id = 0
        self.last_detections = None

        if self.use_tracker:
            default_tracker_config = {
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "frame_rate": 30,
                "mot20": False
            }
             # Allow nested overrides from YAML: stage_2.yolov8_tracker.tracker_config.{...}
            tracker_config_dict = {**default_tracker_config, **config.get("tracker_config", {})}
            tracker_config = SimpleNamespace(**tracker_config_dict)
            self.tracker = BYTETracker(tracker_config) 

        self.frame_id = 0
    
    # add below __init__ (after self.frame_id = 0) in YOLOv8Tracker
    def reset(self):
        """Reset tracker state for a new video."""
        self.frame_id = 0
        self.last_detections = None
        if hasattr(self, "tracker") and self.tracker is not None:
            # Common ByteTrack fields
            if hasattr(self.tracker, "frame_id"):
                self.tracker.frame_id = 0
            if hasattr(self.tracker, "track_id_count"):
                self.tracker.track_id_count = 0
            # Clear track caches if available
            for attr in ("tracked_stracks", "lost_stracks", "removed_stracks"):
                if hasattr(self.tracker, attr):
                    setattr(self.tracker, attr, [])


    def track(self, frame):
        self.frame_id += 1
        timer = Timer()
        timer.tic()

        #results = self.model.track(frame, persist=True, conf=self.config['confidence_threshold'], classes=0)

        # Ultralytics track() with sensible fallbacks
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.config.get('confidence_threshold', 0.25),
            iou=self.config.get('iou_threshold', 0.45),
            classes=0,  # persons
            device=self.config.get('device', 0),
            max_det=self.config.get('max_det', 300),
            verbose=False
        )

        if results is None or results[0].boxes is None:
            return frame, []

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = box.conf[0].cpu().item()
            detections.append([x1, y1, x2, y2, score])

        print(f"[DEBUG] YOLOv8 Frame {self.frame_id}: {len(detections)} detections")
        detections = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 5), dtype=np.float32)

        if not self.use_tracker:
            person_data = [{
                "id": i,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(score)
            } for i, (x1, y1, x2, y2, score) in enumerate(detections)]
            return frame, person_data

        im_height, im_width, _ = frame.shape
        img_info = (im_height, im_width)
        img_size = (im_height, im_width)
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

        timer.toc()
        return frame, person_data

    def predict_only(self):
        if not self.tracker:
            return []
        return [
            {"id": t.track_id, "bbox": t.tlbr.tolist(), "conf": getattr(t, "score", 1.0)}
            for t in self.tracker.tracked_stracks if t.is_activated
        ]


