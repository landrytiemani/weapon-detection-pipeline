import cv2
import numpy as np
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from tracker.byte_tracker import BYTETracker
from tracker.tracking_utils.timer import Timer
from types import SimpleNamespace

class SSDMobileNetByteTrack:
    def __init__(self, config):
        self.config = config
        self.use_tracker = config.get('use_tracker', True)

        model_path = config['model_path']
        label_path = config['label_path']
        self.conf_threshold = config.get('confidence_threshold', 0.4)
        self.target_class = config.get('target_class', 'person')

        self.frame_id = 0
        self.last_detections = None

        with open(label_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.net = create_mobilenetv2_ssd_lite(len(self.class_names), is_test=True)
        self.net.load(model_path)
        print(f"TRACKER: Loaded SSD MobilenetV2 from: {model_path}")
        self.predictor = create_mobilenetv2_ssd_lite_predictor(self.net, candidate_size=200, nms_method="soft")
        self.model = self.net
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))

        if self.use_tracker:
            default_tracker_config = {
                "track_thresh":        0.5,
                "track_buffer":        30,
                "match_thresh":        0.8,
                "frame_rate":          30,
                "mot20":               False,
                "aspect_ratio_thresh": 1.6,
                "min_box_area":        10,
            }
            # tracker_config_dict = {**default_tracker_config, **config.get("tracker_config", {})}
            # tracker_config = SimpleNamespace(**tracker_config_dict)
            # self.tracker = BYTETracker(tracker_config)

            nested_overrides = self.config.get("tracker_config", {})
            flat_overrides = {k: self.config[k] for k in default_tracker_config.keys() if k in self.config}
            tracker_cfg = {**default_tracker_config, **nested_overrides, **flat_overrides}
            self.tracker = BYTETracker(SimpleNamespace(**tracker_cfg))

        self.frame_id = 0

    def reset(self):
        """Reset tracker state for new experiment"""
        self.frame_id = 0
        self.last_detections = None
        if hasattr(self, 'tracker'):
            self.tracker.frame_id = 0
            self.tracker.tracked_stracks = []
            self.tracker.lost_stracks = []
            self.tracker.removed_stracks = []


    def track(self, frame):
        self.frame_id += 1
        timer = Timer()
        timer.tic()

        orig_image = frame.copy()
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(image_rgb, top_k=10, prob_threshold=self.conf_threshold)

        detections = []
        for i in range(boxes.size(0)):
            class_name = self.class_names[labels[i]]
            if class_name == self.target_class:
                box = boxes[i].detach().cpu().numpy().astype(np.float32)
                score = np.float32(probs[i].item())
                detections.append(np.concatenate((box, [score])))

        print(f"[DEBUG] SSD Frame {self.frame_id}: {len(detections)} detections")
        detections = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 5), dtype=np.float32)

        if not self.use_tracker:
            person_data = [{
                "id": i,
                "bbox": list(map(int, [x1, y1, x2, y2])),
                "conf": float(score)
            } for i, (x1, y1, x2, y2, score) in enumerate(detections)]
            return orig_image, person_data

        im_height, im_width, _ = frame.shape
        img_info = (im_height, im_width)
        img_size = (im_height, im_width)
        online_targets = self.tracker.update(detections, img_info, img_size)

        # if force_detect:
        #     # Run detection
        #     print(f"[DEBUG] SSD Frame {self.frame_id}: Running DETECTION")
        #     orig_image = frame.copy()
        #     image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        #     boxes, labels, probs = self.predictor.predict(image_rgb, top_k=10, prob_threshold=self.conf_threshold)
            
        #     detections = []
        #     for i in range(boxes.size(0)):
        #         class_name = self.class_names[labels[i]]
        #         if class_name == self.target_class:
        #             box = boxes[i].detach().cpu().numpy().astype(np.float32)
        #             score = np.float32(probs[i].item())
        #             detections.append(np.concatenate((box, [score])))
            
        #     detections = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 5), dtype=np.float32)
        #     self.last_detections = detections
        #     print(f"  Found {len(detections)} persons")
        # else:
        #     # Only track, no detection
        #     print(f"[DEBUG] SSD Frame {self.frame_id}: TRACKING ONLY")
        #     detections = np.zeros((0, 5), dtype=np.float32)
        
        # # Update tracker
        # if self.use_tracker:
        #     online_targets = self.tracker.update(detections, img_info, img_size)


        person_data = []
        for t in online_targets:
            x1, y1, w, h = t.tlwh
            x2, y2 = x1 + w, y1 + h
            person_data.append({
                "id": t.track_id,
                "bbox": list(map(int, [x1, y1, x2, y2])),
                "conf": t.score
            })

            # Optional: Draw tracking info
            color = (0, 255, 0)
            label = f"ID: {t.track_id} ({t.score:.2f})"
            cv2.rectangle(orig_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(orig_image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        timer.toc()
        return orig_image, person_data

    # def predict_only(self):
    #     if not self.tracker:
    #         return []
    #     return [
    #         {"id": t.track_id, "bbox": t.tlbr.tolist(), "conf": getattr(t, "score", 1.0)}
    #         for t in self.tracker.tracked_stracks if t.is_activated
    #     ]
    def predict_only(self):
        if not hasattr(self, "tracker") or self.tracker is None:
            return []
        return [
            {"id": t.track_id, "bbox": t.tlbr.tolist(), "conf": getattr(t, "score", 1.0)}
            for t in self.tracker.tracked_stracks if t.is_activated
        ]



