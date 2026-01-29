from ultralytics import YOLO
from tracker.byte_tracker import BYTETracker
from tracker.tracking_utils.timer import Timer
from types import SimpleNamespace  
import numpy as np
import cv2


class YOLOv8Tracker:
    def __init__(self, config):
        """
        Initializes the YOLOv8 model with optional ByteTrack tracking.
        """
        self.config = config
        self.use_tracker = config.get('use_tracker', True)
        self.model = YOLO(config['model_path'])
        self.model.task = "detect"

        print(f"TRACKER: Loaded YOLOv8 from: {config['model_path']}")

        if self.use_tracker:
            default_tracker_config = {
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "frame_rate": 30
            }
            tracker_config_dict = {**default_tracker_config, **config.get("tracker_config", {})}
            tracker_config = SimpleNamespace(**tracker_config_dict)
            self.tracker = BYTETracker(tracker_config)

        self.frame_id = 0

    def track(self, frame):
        """
        Performs object detection and optional tracking on a frame.
        """
        self.frame_id += 1
        timer = Timer()
        timer.tic()

        results = self.model.track(frame, persist=True, conf=self.config['confidence_threshold'], classes=0)

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

        # Tracker logic
        im_height, im_width, _ = frame.shape
        img_info = (im_height, im_width)
        img_size = (im_height, im_width)

        online_targets = self.tracker.update(detections, img_info, img_size)

        person_data = []
        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            conf = t.score

            person_data.append({
                "id": track_id,
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })

        timer.toc()
        return frame, person_data
