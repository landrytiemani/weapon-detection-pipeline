"""
main_perclass.py


What's included:
- Stage 2: Person detection/tracking -> square person crops (with overlap filtering)
- Stage 3: Weapon detection on crops with TTA
    * Horizontal flip (config: stage_3.tta_flip)
    * Multi-scale per-crop TTA (config: stage_3.tta_scales, stage_3.imgsz)
    * OPTIONAL crop-tiling pass for tiny/skinny knives (config: stage_3.tile_enable/tile_grid/tile_overlap)
- Privacy Protection:
    * Face blur for people without weapons
    * Silhouette masking option
    * Configurable via config['privacy']
- Post-process:
    * Per-class confidence gating (stage_3.class_specific_conf with final safety floor stage_3.min_final_confidence)
    * Class-wise local NMS (+ optional per-class IoU overrides and knife Soft-NMS if provided)
    * Global NMS across crops (same class)
    * Cross-class suppression (keeps highest-confidence label when handgun/knife overlap)
    * Light geometry filters (min/max size, aspect ratio, edge/tiny-area)
- Evaluation: COCO Evaluator (mAP50, mAP50-95, Precision, Recall, F1, TP/FP/FN)
- Reports/plots: report_utils
- Privacy is applied AFTER weapon detection so we know which people have weapons

NOTE:
- Keep FP16 ONLY on CUDA. On CPU we force FP32 for stability.

"""

import os
import re
import glob
import time
import yaml
import cv2
import pytz
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO  # inference-only

# Project-local imports
from utils.box_utils import (
    load_yolo_labels, square_scale_clip_xyxy, remap_crop_to_frame,
    apply_nms, apply_global_nms,
    filter_overlapping_crops, filter_suspicious_predictions,
    filter_low_confidence_predictions, get_prediction_stats
)
# Soft-NMS is optional; we'll try to import it and fall back gracefully
try:
    from utils.box_utils import apply_soft_nms  # optional
except Exception:
    apply_soft_nms = None  # will fall back to greedy NMS

from utils.flops_utils import (
    get_stage2_module_handle, compute_flops_gflops,
    calculate_pipeline_flops, print_flops_summary
)
from utils.report_utils import (
    ensure_dir, write_detailed_report, write_combined_table,
    write_per_class_table, plot_results, print_summary
)
from utils.evaluation import PipelineEvaluator
from utils.visualization import PipelineVisualizer
from stages.stage_2_persondetection import PersonDetectionStage

# NEW: Privacy protection module
from utils.privacy import PrivacyProtector


# =====================================================================
# Helpers
# =====================================================================
def _pst_stamp():
    tz = pytz.timezone("America/Los_Angeles")
    return datetime.now(tz).strftime("%Y%m%d_%H%M%S")


def _infer_video_id(stem: str) -> str:
    """Group frames by an inferred video id so we can reset trackers at boundaries."""
    m = re.match(r'^(.+?)_(?:frame|img|f)\d+', stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r'^(.+?)[_-]\d{3,}$', stem)
    if m:
        return m.group(1)
    parts = re.split(r'[_-]', stem, maxsplit=1)
    return parts[0] if parts and parts[0] else stem


def _draw_box(img, xyxy, color, thick=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)


def _yolo_to_xyxy(cx, cy, w, h, W, H):
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    return [x1, y1, x2, y2]


def _tile_crop(crop_img, rows: int, cols: int, overlap: float):
    """
    Make a sliding grid of tiles over the person crop, with (rows x cols) nominal grid
    and fractional overlap (0..0.9). Returns (tiles[], tile_xyxy_in_crop[]).
    """
    H, W = crop_img.shape[:2]
    tiles, boxes = [], []
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    overlap = float(max(0.0, min(0.9, overlap)))

    tile_h = max(1, int(round(H / rows)))
    tile_w = max(1, int(round(W / cols)))
    step_y = max(1, int(round(tile_h * (1.0 - overlap))))
    step_x = max(1, int(round(tile_w * (1.0 - overlap))))

    for r in range(0, max(1, H - tile_h + 1), step_y):
        for c in range(0, max(1, W - tile_w + 1), step_x):
            y2 = min(H, r + tile_h)
            x2 = min(W, c + tile_w)
            t = crop_img[r:y2, c:x2]
            if t.size == 0:
                continue
            tiles.append(t)
            boxes.append([c, r, x2, y2])
    return tiles, boxes


def _legend(img, class_names, colors, x=12, y=18, line_h=18):
    """
    Draw a small legend block indicating color coding
    colors: dict with keys 'pred','gt','person','classes' (id->BGR)
    """
    def put(txt, dy, color=(255, 255, 255)):
        cv2.putText(img, txt, (x, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    dy = 0
    put("Legend", dy, (255, 255, 255)); dy += line_h
    put("Pred", dy, colors['pred']); dy += line_h
    put("GT", dy, colors['gt']); dy += line_h
    put("Person", dy, colors['person']); dy += line_h
    for cid, name in enumerate(class_names):
        put(f"class {cid}: {name}", dy, colors['classes'].get(cid, (220, 220, 220))); dy += line_h


def _cross_class_suppression(preds, iou_thresh=0.55):
    """
    Remove cross-class duplicates: if two boxes of different classes overlap
    above iou_thresh, keep the one with higher confidence.
    preds: [[cls, cx, cy, w, h, conf], ...] in [0..1] coords
    """
    if not preds:
        return preds
    boxes = sorted(preds, key=lambda x: x[5], reverse=True)
    keep = []
    for a in boxes:
        suppress = False
        for b in keep:
            if int(a[0]) == int(b[0]):
                continue
            # compute IoU in normalized space
            _, acx, acy, aw, ah, _ = a
            _, bcx, bcy, bw, bh, _ = b
            ax1, ay1 = acx - aw/2, acy - ah/2
            ax2, ay2 = acx + aw/2, acy + ah/2
            bx1, by1 = bcx - bw/2, bcy - bh/2
            bx2, by2 = bcx + bw/2, bcy + bh/2
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ua = aw*ah + bw*bh - inter
            iou = inter/ua if ua > 0 else 0.0
            if iou >= iou_thresh:
                suppress = True
                break
        if not suppress:
            keep.append(a)
    return keep


# =====================================================================
# Single Experiment
# =====================================================================
class SingleExperiment:
    """Runs one experiment (tracker setting + frame gap) across a folder of frames (Custom Evaluator only)."""

    def __init__(self, config_path: str, use_tracker: bool, frame_gap: int,
                 experiment_name: str, output_dir: str):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Apply experiment overrides
        self.cfg["stage_2"]["use_tracker"] = bool(use_tracker)
        self.cfg["stage_2"]["frame_gap"] = int(frame_gap)

        self.experiment_name = experiment_name
        self.base_output_dir = ensure_dir(output_dir)

        # Paths
        p_cfg = self.cfg.get("pipeline", {})
        self.frames_dir = p_cfg.get("frames_dir", "testing_data/gun_action/test/images")
        self.labels_dir = p_cfg.get("labels_dir", "testing_data/gun_action/test/labels")
        self.image_glob = p_cfg.get("image_glob", "*.*")  # allow .jpg/.png mixed

        # Stage-2 (person) - with ablation support
        # H1.1 Ablation: skip_person_detection bypasses person detection entirely
        self.skip_person_detection = bool(self.cfg["stage_2"].get("skip_person_detection", False))
        
        if not self.skip_person_detection:
            self.person_stage = PersonDetectionStage(self.cfg["stage_2"])
        else:
            self.person_stage = None
            print("[ABLATION] Person detection DISABLED - using full frame for weapon detection")
        
        self.use_tracker = bool(self.cfg["stage_2"].get("use_tracker", False))
        self.frame_gap = int(self.cfg["stage_2"].get("frame_gap", 3))
        self.crop_scale = float(self.cfg["stage_2"].get("crop_scale", 1.8))
        self.crop_overlap_threshold = float(self.cfg["stage_2"].get("crop_overlap_threshold", 0.85))

        # Stage-3 (weapon)
        s3 = self.cfg.get("stage_3", {})
        self.class_names = s3.get("names", ["handgun", "knife"])
        self.allowed_cls = set(range(len(self.class_names)))
        self.min_final_conf = float(s3.get("min_final_confidence", 0.50))
        self.class_specific_conf = s3.get("class_specific_conf", {})           # {name: conf}
        self.class_specific_nms_iou = s3.get("class_specific_nms_iou", {})     # {name or id: iou}
        self.nms_iou_threshold = float(s3.get("nms_iou_threshold", 0.40))
        self.global_nms_threshold = float(s3.get("global_nms_threshold", 0.22))
        self.min_box_size = float(s3.get("min_box_size", 0.015))
        self.max_box_size = float(s3.get("max_box_size", 0.30))
        self.min_aspect_ratio = float(s3.get("min_aspect_ratio", 0.22))
        self.max_aspect_ratio = float(s3.get("max_aspect_ratio", 5.0))
        self.use_tta_flip = bool(s3.get("tta_flip", True))
        self.tta_scales = [float(x) for x in s3.get("tta_scales", [1.0, 1.3])]
        self.base_imgsz = int(s3.get("imgsz", 640))

        # OPTIONAL tiling for tiny knives
        self.tile_enable = bool(s3.get("tile_enable", False))
        _tg = s3.get("tile_grid", [1, 1])
        self.tile_rows = int(_tg[0] if isinstance(_tg, (list, tuple)) and len(_tg) > 0 else 1)
        self.tile_cols = int(_tg[1] if isinstance(_tg, (list, tuple)) and len(_tg) > 1 else 1)
        self.tile_overlap = float(s3.get("tile_overlap", 0.15))

        # Cross-class suppression IoU (handgun vs knife)
        self.cross_class_iou = float(s3.get("cross_class_suppression_iou", 0.55))

        # Optional class colors for legend
        self.class_colors = s3.get("class_colors", {
            # default: handgun BLUE, knife RED
            0: [255, 0, 0],
            1: [0, 0, 255],
        })

        method = s3.get("approach", "")
        self.stage3_model_path = s3.get(method, {}).get("model_path", "")
        self.stage3_conf_threshold = float(s3.get(method, {}).get("confidence_threshold", 0.55))

        print(f"[INFO] Loading Stage-3 model: {self.stage3_model_path}")
        self.stage3_model = YOLO(self.stage3_model_path)
        if torch.cuda.is_available() and hasattr(self.stage3_model, "model"):
            try:
                self.stage3_model.model.half()
                print("[SPEED] Stage-3 FP16 enabled (CUDA)")
            except Exception:
                pass
        else:
            if hasattr(self.stage3_model, "model"):
                try:
                    self.stage3_model.model.float()  # CPU: force FP32
                except Exception:
                    pass

        # NEW: Privacy Protection
        privacy_cfg = self.cfg.get("privacy", {})
        self.privacy = PrivacyProtector(privacy_cfg)
        print(f"[PRIVACY] Initialized: enabled={self.privacy.enabled}, scope={self.privacy.scope}")

        # Evaluation / FLOPs
        e_cfg = self.cfg.get("evaluation", {})
        self.iou_eval_thresholds = e_cfg.get("iou_thresholds", [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95])
        self.compute_flops = bool(e_cfg.get("compute_flops", True))
        self.stage2_flops_imgsz = int(self.cfg.get("stage_2", {}).get("flops_imgsz", 800))
        self.stage3_flops_imgsz = int(e_cfg.get("flops_imgsz", 640))

        # Visual debug writer
        self.vdbg_every = int(e_cfg.get("visual_debug_every", 30))
        self.vdbg_limit = int(e_cfg.get("visual_debug_limit", 200))

        self.evaluator = PipelineEvaluator(self.class_names)
        self.visualizer = PipelineVisualizer(self.base_output_dir, self.class_names, num_samples=20)

        # Outputs (per-experiment)
        self.exp_dir = ensure_dir(os.path.join(self.base_output_dir, f"{self.experiment_name}_{_pst_stamp()}"))
        self.pred_dir = ensure_dir(os.path.join(self.exp_dir, "predictions"))
        self.logs_dir = ensure_dir(os.path.join(self.exp_dir, "logs"))
        self.vdbg_dir = ensure_dir(os.path.join(self.exp_dir, "visual_debug"))
        self._vdbg_count = 0

        # Colors for legend/drawing
        self.colors = {
            'pred':   (0, 0, 255),     # red
            'gt':     (0, 200, 0),     # green
            'person': (255, 255, 0),   # cyan
            'classes': {int(k): tuple(map(int, v)) for k, v in self.class_colors.items()}
        }

        # Stats
        self.gt_total_by_cls = defaultdict(int)
        self.gt_covered_by_s2_by_cls = defaultdict(int)
        self.stage2_times_ms, self.stage3_times_ms = [], []
        self.crops_per_frame = []
        self.total_ground_truths = 0
        self.total_predictions = 0
        self.nms_removals = 0
        self.global_nms_removals = 0
        self.total_pre_nms = 0

        # For FLOPs reporting
        s2_method = self.cfg["stage_2"].get("approach", "")
        self.stage2_model_path = self.cfg["stage_2"].get(s2_method, {}).get("model_path", "")
        self.stage2_model_name = os.path.basename(self.stage2_model_path) if self.stage2_model_path else "N/A"
        self.stage3_model_name = os.path.basename(self.stage3_model_path) if self.stage3_model_path else "N/A"

        # Reset stage-2 tracker/detector state
        self._reset_stage2()

    # -------------------------------

    def _reset_stage2(self):
        """Reset any internal counters/trackers inside the Stage-2 wrapper."""
        if self.person_stage is None:
            return  # Skip if person detection is disabled (ablation mode)
        det = getattr(self.person_stage, "detector", None)
        if det is not None and hasattr(det, "reset"):
            try:
                det.reset()
            except Exception:
                pass
        for attr in ("frame_count", "frame_idx"):
            if hasattr(self.person_stage, attr):
                try:
                    setattr(self.person_stage, attr, 0)
                except Exception:
                    pass

    # -------------------------------

    def _predict_stage3_batch(self, imgs):
        """
        Run Stage-3 with multi-scale + optional flip TTA on a list of images.
        Returns: list of lists, where out[i] is a list of (res_orig, res_flip_or_None) tuples for image i,
                 one tuple per scale.
        """
        if not imgs:
            return []
        all_pairs = [[] for _ in range(len(imgs))]

        for scale in self.tta_scales:
            imgsz = int(round(self.base_imgsz * float(scale)))
            # originals
            res_o = self.stage3_model.predict(
                imgs, imgsz=imgsz, conf=self.stage3_conf_threshold,
                verbose=False, batch=len(imgs), half=torch.cuda.is_available(),
                max_det=300, agnostic_nms=False
            )
            # flips (optional)
            if self.use_tta_flip:
                flipped = [cv2.flip(im, 1) for im in imgs]
                res_f = self.stage3_model.predict(
                    flipped, imgsz=imgsz, conf=self.stage3_conf_threshold,
                    verbose=False, batch=len(flipped), half=torch.cuda.is_available(),
                    max_det=300, agnostic_nms=False
                )
            else:
                res_f = [None] * len(res_o)
            for i in range(len(imgs)):
                all_pairs[i].append((res_o[i], res_f[i]))
        return all_pairs

    # -------------------------------

    def _process_frame(self, img_bgr, frame_idx, stem, gt_path, H, W):
        """
        Run the full pipeline on one frame: S2 -> crops -> S3(+TTA and optional tiling) -> post-process -> eval/vis.
        NEW: Apply privacy protection after weapon detection.
        """
        # Ground-truth in YOLO normalized space
        gt = [b for b in load_yolo_labels(gt_path) if int(b[0]) in self.allowed_cls]
        self.total_ground_truths += len(gt)

        # Track per-class GT totals and S2 coverage
        gt_objs = []
        for cls, cx, cy, gw, gh in gt:
            ax, ay = cx * W, cy * H
            aw, ah = gw * W, gh * H
            gt_xyxy = (ax - aw/2.0, ay - ah/2.0, ax + aw/2.0, ay + ah/2.0)
            gt_objs.append({"cls": int(cls), "bbox": gt_xyxy, "covered": False})
            self.gt_total_by_cls[int(cls)] += 1

        # Stage-2: Person detection OR full-frame ablation
        t2 = time.perf_counter()
        
        if self.skip_person_detection:
            # ABLATION MODE: Skip person detection, use full frame as single "crop"
            # This tests H1.1 - the contribution of person-centric cropping
            persons = None
            s2_ms = 0.0  # No Stage-2 latency when skipped
        else:
            # Normal mode: Run person detection
            _, persons, _ = self.person_stage.run(img_bgr.copy(), frame_idx)
            s2_ms = (time.perf_counter() - t2) * 1000.0
        
        self.stage2_times_ms.append(s2_ms)

        # Filter overlapping person boxes
        if persons:
            persons_before = len(persons)
            persons = filter_overlapping_crops(persons, iou_threshold=self.crop_overlap_threshold)
            if frame_idx % 50 == 0 and persons_before != len(persons):
                print(f"  [S2] Overlap filter: {persons_before} -> {len(persons)}")

        # Build crops from person detections OR use full frame for ablation
        crops, metas = [], []
        
        if self.skip_person_detection:
            # ABLATION: Use full frame as single crop (no person-centric processing)
            # All GT objects are considered "covered" since we're processing entire frame
            for g in gt_objs:
                g["covered"] = True
                self.gt_covered_by_s2_by_cls[g["cls"]] += 1
            
            # Create single crop = full frame
            crops.append(img_bgr.copy())
            metas.append({
                "crop_bbox": [0, 0, W, H],  # Full frame coordinates
                "crop_shape": img_bgr.shape
            })
        elif persons:
            for p in persons:
                bbox = p.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                cx1, cy1, cx2, cy2 = square_scale_clip_xyxy(x1, y1, x2, y2, W, H, self.crop_scale)
                ix1, iy1, ix2, iy2 = map(int, [cx1, cy1, cx2, cy2])
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                crop = img_bgr[iy1:iy2, ix1:ix2].copy()
                if crop.size == 0:
                    continue

                # mark GT covered by this crop
                for g in gt_objs:
                    if not g["covered"]:
                        gx1, gy1, gx2, gy2 = g["bbox"]
                        ix1c = max(gx1, cx1); iy1c = max(gy1, cy1)
                        ix2c = min(gx2, cx2); iy2c = min(gy2, cy2)
                        if ix2c > ix1c and iy2c > iy1c:
                            g["covered"] = True

                crops.append(crop)
                metas.append({"crop_bbox": [cx1, cy1, cx2, cy2], "crop_shape": crop.shape})

        self.crops_per_frame.append(len(crops))

        # Stage-3 with TTA on person crops
        frame_preds = []  # normalized YOLO boxes: [cls, cx, cy, w, h, conf] (frame space)
        s3_ms_total = 0.0
        if crops:
            t3 = time.perf_counter()
            all_pairs = self._predict_stage3_batch(crops)
            s3_ms_total = (time.perf_counter() - t3) * 1000.0

            for crop_idx, multi_scale_pairs in enumerate(all_pairs):
                crop_h, crop_w = metas[crop_idx]["crop_shape"][:2]

                def _to_yolo_boxes(result, flipped=False):
                    out = []
                    if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
                        return out
                    for b in result.boxes:
                        cls = int(b.cls[0].item())
                        conf = float(b.conf[0].item())
                        if hasattr(b, "xywhn"):
                            cx_c, cy_c, w_c, h_c = b.xywhn[0].detach().cpu().numpy().tolist()
                        else:
                            x1c, y1c, x2c, y2c = b.xyxy[0].detach().cpu().numpy().tolist()
                            cx_c = (x1c + x2c) / 2.0 / crop_w
                            cy_c = (y1c + y2c) / 2.0 / crop_h
                            w_c = (x2c - x1c) / crop_w
                            h_c = (y2c - y1c) / crop_h
                        if flipped:
                            cx_c = 1.0 - cx_c
                        out.append([cls, cx_c, cy_c, w_c, h_c, conf])
                    return out

                merged_crop = []
                for (res_o, res_f) in multi_scale_pairs:
                    merged_crop += _to_yolo_boxes(res_o, flipped=False)
                    if self.use_tta_flip:
                        merged_crop += _to_yolo_boxes(res_f, flipped=True)

                # Intra-crop fuse (slightly higher IoU for TTA consolidation)
                merged_crop = apply_nms(merged_crop, iou_threshold=0.55, class_specific_thresholds=None)

                # Remap to frame-normalized space and collect
                frame_preds += remap_crop_to_frame(merged_crop, metas[crop_idx]["crop_bbox"], W, H)

        # OPTIONAL: tiling pass for tiny knives (adds another S3 run on tiles)
        # Decide per-crop if we should tile (no confident detections from normal pass)
        need_tile = [True] * len(crops)
        per_crop_keep = [[] for _ in range(len(crops))]
        for p in frame_preds:
            cls, cx, cy, w, h, conf = p
            # map back to crop id via metas
            # quick check: a box belongs to the crop whose bbox contains its center
            fx, fy = cx * W, cy * H
            for ci, m in enumerate(metas):
                x1,y1,x2,y2 = m["crop_bbox"]
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    # If we already have a decent prediction in this crop, skip tiling there
                    if conf >= max(self.min_final_conf, 0.52):
                        need_tile[ci] = False
                    per_crop_keep[ci].append(p)
                    break

        # Run tiling only where needed; and keep ONLY knife from tiles
        knife_id = self.class_names.index("knife") if "knife" in self.class_names else None
        if crops and self.tile_enable and (self.tile_rows * self.tile_cols > 1):
            tile_images, tile_parents = [], []
            tile_owner = []
            for ci, crop in enumerate(crops):
                if not need_tile[ci]:
                    continue
                tiles, tboxes = _tile_crop(crop, self.tile_rows, self.tile_cols, self.tile_overlap)
                tile_images.extend(tiles)
                tile_parents.extend(tboxes)
                tile_owner.extend([ci] * len(tboxes))

            if tile_images:
                t3t = time.perf_counter()
                tile_pairs = self._predict_stage3_batch(tile_images)
                s3_ms_total += (time.perf_counter() - t3t) * 1000.0

                # collect only knife boxes from tiles
                for it, (res_o, res_f) in enumerate([p for pairs in tile_pairs for p in pairs]):
                    ci = tile_owner[it // len(self.tta_scales)]
                    tx1, ty1, tx2, ty2 = tile_parents[it // len(self.tta_scales)]
                    crop_h, crop_w = metas[ci]["crop_shape"][:2]
                    for res, flipped in ((res_o, False), (res_f, True if self.use_tta_flip else False)):
                        if res is None or getattr(res, "boxes", None) is None:
                            continue
                        for b in res.boxes:
                            cls = int(b.cls[0].item())
                            if knife_id is not None and cls != knife_id:
                                continue  # knife-only from tiles
                            conf = float(b.conf[0].item())
                            if hasattr(b, "xywhn"):
                                cx, cy, w, h = b.xywhn[0].detach().cpu().numpy().tolist()
                            else:
                                x1, y1, x2, y2 = b.xyxy[0].detach().cpu().numpy().tolist()
                                tw = max(1.0, (tx2 - tx1)); th = max(1.0, (ty2 - ty1))
                                cx = ((x1 + x2)/2) / tw; cy = ((y1 + y2)/2) / th
                                w = (x2 - x1) / tw;      h = (y2 - y1) / th
                            if flipped: cx = 1.0 - cx
                            # TILE -> CROP -> FRAME
                            cx_c = (tx1 + cx * (tx2 - tx1)) / crop_w
                            cy_c = (ty1 + cy * (ty2 - ty1)) / crop_h
                            w_c  = w * (tx2 - tx1) / crop_w
                            h_c  = h * (ty2 - ty1) / crop_h
                            frame_preds += remap_crop_to_frame([[cls, cx_c, cy_c, w_c, h_c, conf]],
                                                            metas[ci]["crop_bbox"], W, H)

        # track stage-3 avg per-crop latency
        avg_s3_ms = s3_ms_total / max(1, len(crops))
        self.stage3_times_ms.append(avg_s3_ms)

        # Per-class confidence gating
        if frame_preds:
            name_map = {i: n for i, n in enumerate(self.class_names)}
            gated = []
            for cls, cx, cy, w, h, conf in frame_preds:
                if int(cls) not in self.allowed_cls:
                    continue
                cname = name_map[int(cls)]
                minc = float(self.class_specific_conf.get(cname, self.min_final_conf))
                if conf >= minc:
                    gated.append([cls, cx, cy, w, h, conf])
            frame_preds = gated

        # Light geometry trimming
        if frame_preds:
            frame_preds = filter_suspicious_predictions(
                frame_preds,
                min_size=self.min_box_size,
                max_size=self.max_box_size,
                min_aspect=self.min_aspect_ratio,
                max_aspect=self.max_aspect_ratio
            )

        # Local class-wise NMS (use Soft-NMS if provided & a class has negative IoU override)
        if frame_preds:
            self.total_pre_nms += len(frame_preds)

            # normalize per-class IoU overrides (name->id or id pass-through)
            class_iou_overrides = {}
            for k, v in (self.class_specific_nms_iou or {}).items():
                try:
                    if isinstance(k, int) or str(k).isdigit():
                        class_iou_overrides[int(k)] = float(v)
                    elif k in self.class_names:
                        class_iou_overrides[self.class_names.index(k)] = float(v)
                except Exception:
                    continue

            soft_ids = set([cid for cid, thr in class_iou_overrides.items() if thr < 0.0]) if apply_soft_nms else set()
            hard_ids = {cid: thr for cid, thr in class_iou_overrides.items() if thr >= 0.0}

            # Greedy NMS
            after_local = frame_preds
            if hard_ids or not soft_ids:
                after_local = apply_nms(
                    after_local,
                    iou_threshold=self.nms_iou_threshold,
                    class_specific_thresholds=hard_ids
                )

            # Soft-NMS for selected classes
            if apply_soft_nms and soft_ids:
                after_local = apply_soft_nms(
                    after_local,
                    sigma=0.5,
                    score_thresh=self.min_final_conf * 0.5,
                    iou_thresh=self.nms_iou_threshold,
                    cls_filter=soft_ids
                )

            # Global cross-crop NMS (same class)
            after_global = apply_global_nms(after_local, iou_threshold=self.global_nms_threshold)

            # NEW: Cross-class suppression (handgun vs knife duplicates)
            after_cross = _cross_class_suppression(after_global, iou_thresh=self.cross_class_iou)

            # Final safety floor
            final_preds = filter_low_confidence_predictions(after_cross, min_conf=self.min_final_conf)

            # Stats
            local_removed = len(frame_preds) - len(after_local)
            global_removed = len(after_local) - len(after_global)
            # cross-class numbers are not counted separately to keep counters simple
            self.nms_removals += max(0, local_removed)
            self.global_nms_removals += max(0, global_removed)
            frame_preds = final_preds

        # Update GT coverage stats
        for g in gt_objs:
            if g["covered"]:
                self.gt_covered_by_s2_by_cls[g["cls"]] += 1

       
        # NEW: Build weapon_stats for privacy protection (CENTER-DISTANCE)
        # Map crops to persons by finding closest center, then map weapons to crops
        weapon_person_ids = set()
        
        if frame_preds and persons and metas:
            # Build person_id list for each crop
            person_ids_by_crop = []
            for crop_idx, meta in enumerate(metas):
                crop_bbox = meta["crop_bbox"]
                cx1, cy1, cx2, cy2 = crop_bbox
                crop_center_x = (cx1 + cx2) / 2
                crop_center_y = (cy1 + cy2) / 2
                
                # Find person with closest center to crop center
                best_dist = float('inf')
                best_person_id = None
                
                for person in persons:
                    pbbox = person.get("bbox", [])
                    if len(pbbox) >= 4:
                        px1, py1, px2, py2 = pbbox
                        person_center_x = (px1 + px2) / 2
                        person_center_y = (py1 + py2) / 2
                        
                        dist = ((crop_center_x - person_center_x)**2 + (crop_center_y - person_center_y)**2)**0.5
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_person_id = person.get("id")
                
                person_ids_by_crop.append(best_person_id)
            
            # Map weapons to crops, then to persons
            for pred in frame_preds:
                _, pcx, pcy, pw, ph, pconf = pred
                wpx, wpy = pcx * W, pcy * H
                
                for crop_idx, meta in enumerate(metas):
                    cx1, cy1, cx2, cy2 = meta["crop_bbox"]
                    
                    if cx1 <= wpx <= cx2 and cy1 <= wpy <= cy2:
                        person_id = person_ids_by_crop[crop_idx]
                        if person_id is not None:
                            weapon_person_ids.add(person_id)
                        break
        
        weapon_stats = {
            "count": len(frame_preds),
            "weapon_person_ids": list(weapon_person_ids),
            "avg_confidence": float(np.mean([p[5] for p in frame_preds])) if frame_preds else 0.0
        }

        # Debug log every 50 frames
        if frame_idx % 50 == 0:
            print(f"[FRAME {stem}] S2 {s2_ms:.1f} ms | crops={len(crops)} | S3 {avg_s3_ms:.1f} ms/crop | preds={len(frame_preds)}")
            if frame_preds:
                stats = get_prediction_stats(frame_preds, gt)
                print(f"  [STATS] Pred={stats['total_pred']} GT={stats['total_gt']} Ratio={stats['ratio']:.2f}")
            
            # Enhanced privacy debug with detailed mapping info
            if frame_preds and persons:
                print(f"  [PRIVACY-DEBUG] Persons detected: {len(persons)}, Crops: {len(metas)}, Weapons: {len(frame_preds)}")
                
                # Debug: show person IDs
                person_ids = [p.get("id") for p in persons]
                print(f"  [PRIVACY-DEBUG] Person IDs: {person_ids}")
                
                # Debug: show crop-to-person mapping
                if metas:
                    print(f"  [PRIVACY-DEBUG] Crop-to-person mapping:")
                    for crop_idx, meta in enumerate(metas):
                        crop_bbox = meta["crop_bbox"]
                        cx1, cy1, cx2, cy2 = crop_bbox
                        crop_center = ((cx1+cx2)/2, (cy1+cy2)/2)
                        
                        # Find mapped person
                        best_dist = float('inf')
                        best_pid = None
                        for person in persons:
                            pbbox = person.get("bbox", [])
                            if len(pbbox) >= 4:
                                px1, py1, px2, py2 = pbbox
                                pcenter = ((px1+px2)/2, (py1+py2)/2)
                                dist = ((crop_center[0]-pcenter[0])**2 + (crop_center[1]-pcenter[1])**2)**0.5
                                if dist < best_dist:
                                    best_dist = dist
                                    best_pid = person.get("id")
                        print(f"    Crop {crop_idx} → Person {best_pid} (dist={best_dist:.1f}px)")
                
                # Debug: show weapon locations and mappings
                if frame_preds:
                    print(f"  [PRIVACY-DEBUG] Weapon locations:")
                    for i, pred in enumerate(frame_preds):
                        _, pcx, pcy, pw, ph, pconf = pred
                        wpx, wpy = pcx * W, pcy * H
                        print(f"    Weapon {i}: center=({wpx:.0f}, {wpy:.0f})")
                        
                        # Check which crop it's in
                        found = False
                        for crop_idx, meta in enumerate(metas):
                            cx1, cy1, cx2, cy2 = meta["crop_bbox"]
                            if cx1 <= wpx <= cx2 and cy1 <= wpy <= cy2:
                                print(f"      → Found in Crop {crop_idx} bbox=[{cx1:.0f},{cy1:.0f},{cx2:.0f},{cy2:.0f}]")
                                found = True
                                break
                        if not found:
                            print(f"      → NOT IN ANY CROP!")
                
                if weapon_person_ids:
                    print(f"  [PRIVACY] ✓ Weapons mapped to person IDs: {sorted(weapon_person_ids)}")
                else:
                    print(f"  [PRIVACY] ✗ NO MAPPING - Weapons detected but couldn't map to persons")
                    print(f"  [PRIVACY]   Check debug output above for mapping details")
            elif frame_preds and not persons:
                print(f"  [PRIVACY] Weapons detected but no persons found")
            else:
                print(f"  [PRIVACY] No weapons detected")



        # Save predictions (with conf) for debug
        if frame_preds:
            with open(os.path.join(self.pred_dir, f"{stem}.txt"), "w") as f:
                for c, cx, cy, w, h, conf in frame_preds:
                    f.write(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

        # Evaluate & visualize (buffered)
        self.total_predictions += len(frame_preds)
        self.evaluator.evaluate_frame(frame_preds, gt, W, H)
        self.visualizer.store_frame_data(stem, img_bgr, frame_preds, gt, persons or [])

       
        # Lightweight visual debug writer with PRIVACY PROTECTION
       
        if self.vdbg_every and (frame_idx % self.vdbg_every == 0) and (self._vdbg_count < self.vdbg_limit):
            vis = img_bgr.copy()
            
            # NEW: Apply privacy protection BEFORE drawing annotations
            # This ensures debug images also respect privacy settings
            if self.privacy.enabled and persons:
                vis = self.privacy.apply_privacy(vis, persons, weapon_stats)
            
            # draw GT (green)
            for g in gt:
                _, cx, cy, w, h = g
                _draw_box(vis, _yolo_to_xyxy(cx, cy, w, h, W, H), self.colors['gt'], 2)
            # draw predictions (per-class color)
            for pr in frame_preds:
                cls, cx, cy, w, h, conf = pr
                x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h, W, H)
                color = self.colors['classes'].get(int(cls), self.colors['pred'])
                _draw_box(vis, (x1, y1, x2, y2), color, 2)
                label = f"{self.class_names[int(cls)]}:{conf:.2f}"
                cv2.putText(vis, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            # draw person crops (cyan) - after privacy
            for p in (persons or []):
                if "bbox" in p and len(p["bbox"]) >= 4:
                    _draw_box(vis, p["bbox"], self.colors['person'], 1)
            # legend
            _legend(vis, self.class_names, self.colors)
            
            # Add privacy indicator if enabled
            if self.privacy.enabled:
                privacy_text = f"Privacy: {self.privacy.scope}"
                cv2.putText(vis, privacy_text, (W - 250, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Debug: show which people have weapons
                if weapon_person_ids:
                    weapon_ids_text = f"Weapon IDs: {sorted(weapon_person_ids)}"
                    cv2.putText(vis, weapon_ids_text, (W - 250, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                else:
                    no_weapons_text = "Weapon IDs: []"
                    cv2.putText(vis, no_weapons_text, (W - 250, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA)
            
            outp = os.path.join(self.vdbg_dir, f"{stem}.jpg")
            cv2.imwrite(outp, vis)
            self._vdbg_count += 1

    # -------------------------------

    def _run_over_all_frames(self):
        """Run the pipeline across the dataset, grouped by inferred video id for tracker resets."""
        img_paths = sorted(glob.glob(os.path.join(self.frames_dir, self.image_glob)))
        print(f"[INFO] Found {len(img_paths)} frames in {self.frames_dir}")

        # Quick Stage-2 warmup (skip if person detection disabled)
        if img_paths and not self.skip_person_detection:
            test_img = cv2.imread(img_paths[0])
            if test_img is not None:
                print(f"[TEST] Stage-2 test on: {os.path.basename(img_paths[0])}")
                try:
                    _, test_persons, _ = self.person_stage.run(test_img.copy(), 0)
                    print(f"[TEST] Stage-2 persons: {len(test_persons) if test_persons else 0}")
                except Exception as e:
                    print(f"[WARN] Stage-2 warmup failed: {e}")
                self._reset_stage2()
        elif self.skip_person_detection:
            print("[ABLATION] Skipping Stage-2 warmup - person detection disabled")

        # Bucket by video id
        by_video = defaultdict(list)
        for p in img_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            vid = _infer_video_id(stem)
            by_video[vid].append(p)

        # Process each video group
        for vid, frames in sorted(by_video.items()):
            print(f"\n[VIDEO] {vid} ({len(frames)} frames)")
            self._reset_stage2()
            for idx, img_path in enumerate(sorted(frames)):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                H, W = img.shape[:2]
                stem = os.path.splitext(os.path.basename(img_path))[0]
                gt_path = os.path.join(self.labels_dir, f"{stem}.txt")
                self._process_frame(img, idx, stem, gt_path, H, W)

    # -------------------------------

    def run(self):
        """Run this experiment and return a result dict compatible with report_utils."""
        print("\n" + "=" * 90)
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"  Tracker={self.use_tracker}  FrameGap={self.frame_gap}")
        print(f"  SkipPersonDetection={self.skip_person_detection}")  # Ablation flag
        print(f"  S3 conf={self.stage3_conf_threshold}  Final conf={self.min_final_conf}")
        print(f"  CropScale={self.crop_scale}  OverlapThresh={self.crop_overlap_threshold}")
        print(f"  NMS: local={self.nms_iou_threshold}  global={self.global_nms_threshold}")
        print(f"  Tiling: enable={self.tile_enable} grid=({self.tile_rows}x{self.tile_cols}) overlap={self.tile_overlap}")
        print(f"  Cross-class IoU={self.cross_class_iou}")
        print(f"  Privacy: enabled={self.privacy.enabled}, scope={self.privacy.scope}")
        print("=" * 90)

        # FLOPs (safe, optional)
        print("\n[FLOPS] Computing model complexity...")
        if self.compute_flops:
            # Stage-2 FLOPs (0 when person detection is skipped for ablation)
            if self.skip_person_detection:
                print("[ABLATION] Stage-2 skipped - no person detection FLOPs")
                s2_gflops = 0.0
            else:
                try:
                    kind, s2_handle = get_stage2_module_handle(self.person_stage)
                    s2_gflops = compute_flops_gflops(s2_handle, self.stage2_flops_imgsz, 0, "Stage-2") if s2_handle else None
                    if s2_gflops is None:
                        print("[WARN] Stage-2 FLOPs unavailable; using fallback 8.0 GFLOPs")
                        s2_gflops = 8.0
                except Exception as e:
                    print(f"[WARN] Stage-2 FLOPs error: {e}; using fallback 8.0 GFLOPs")
                    s2_gflops = 8.0

            # Stage-3 FLOPs
            try:
                s3_module = getattr(self.stage3_model, "model", None)
                s3_gflops = compute_flops_gflops(s3_module, self.stage3_flops_imgsz, 0, "Stage-3") if s3_module else None
                if s3_gflops is None:
                    name = (self.stage3_model_path or "").lower()
                    if "rtdetr" in name or "rt-detr" in name:
                        s3_gflops = 70.0
                    elif "yolov8n" in name:
                        s3_gflops = 8.1
                    elif "yolov8s" in name:
                        s3_gflops = 28.6
                    elif "yolov8m" in name:
                        s3_gflops = 78.9
                    else:
                        s3_gflops = 50.0
                    print(f"[INFO] Stage-3 FLOPs estimated at {s3_gflops:.2f} GFLOPs")
            except Exception as e:
                print(f"[WARN] Stage-3 FLOPs error: {e}; using 50.0 GFLOPs estimate")
                s3_gflops = 50.0
        else:
            print("[FLOPS] Skipped by config.")
            s2_gflops, s3_gflops = 0.0, 0.0

        # Run pipeline across frames
        t0 = time.perf_counter()
        self._run_over_all_frames()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Metrics from evaluator
        print("\n[EVAL] Computing metrics...")
        results_eval = self.evaluator.compute_metrics()
        self.evaluator.print_results(results_eval)

        # Timing & throughput
        s2_avg_ms = sum(self.stage2_times_ms) / max(1, len(self.stage2_times_ms))
        s3_avg_ms = sum(self.stage3_times_ms) / max(1, len(self.stage3_times_ms))
        avg_crops = sum(self.crops_per_frame) / max(1, len(self.crops_per_frame))
        pipeline_latency_ms = s2_avg_ms + (avg_crops * s3_avg_ms)
        fps = 1000.0 / pipeline_latency_ms if pipeline_latency_ms > 0 else 0.0

        # FLOPs summary (per frame)
        flops_info = calculate_pipeline_flops(s2_gflops, s3_gflops, avg_crops)
        print_flops_summary(flops_info)

        # NMS/pred stats
        total_removed = self.nms_removals + self.global_nms_removals
        nms_rate = (total_removed / self.total_pre_nms * 100.0) if self.total_pre_nms > 0 else 0.0
        local_rate = (self.nms_removals / self.total_pre_nms * 100.0) if self.total_pre_nms > 0 else 0.0
        global_rate = (self.global_nms_removals / self.total_pre_nms * 100.0) if self.total_pre_nms > 0 else 0.0
        pred_gt_ratio = (self.total_predictions / self.total_ground_truths) if self.total_ground_truths > 0 else 0.0

        # Coverage (S2 recall of GT regions)
        coverage_by_class = {}
        for cls_idx, cls_name in enumerate(self.class_names):
            total = self.gt_total_by_cls[cls_idx]
            covered = self.gt_covered_by_s2_by_cls[cls_idx]
            coverage_by_class[cls_name] = (covered / total) if total > 0 else 0.0
        total_coverage = (
            sum(self.gt_covered_by_s2_by_cls.values()) / sum(self.gt_total_by_cls.values())
            if sum(self.gt_total_by_cls.values()) > 0 else 0.0
        )

        # Consolidate evaluator outputs
        overall = results_eval["overall"]
        per_class = results_eval["per_class"]

        per_class_maps = {k: v.get("mAP50", 0.0) for k, v in per_class.items()}
        per_class_rec = {k: v.get("recall", 0.0) for k, v in per_class.items()}
        per_class_prec = {k: v.get("precision", 0.0) for k, v in per_class.items()}
        per_class_f1 = {k: v.get("f1", 0.0) for k, v in per_class.items()}
        per_class_tp = {k: v.get("tp", 0) for k, v in per_class.items()}
        per_class_fp = {k: v.get("fp", 0) for k, v in per_class.items()}
        per_class_fn = {k: v.get("fn", 0) for k, v in per_class.items()}

        results_out = {
            "experiment_name": self.experiment_name,
            "tracker": self.use_tracker,
            "frame_gap": self.frame_gap,
            "skip_person_detection": self.skip_person_detection,  # Ablation flag

            # Coverage (Stage-2)
            "coverage": total_coverage,
            "coverage_by_class": coverage_by_class,

            # Custom evaluator (overall)
            "pipeline_map50": overall.get("mAP50", 0.0),
            "pipeline_recall": overall.get("recall", 0.0),
            "pipeline_precision": overall.get("precision", 0.0),
            "pipeline_f1": overall.get("f1", 0.0),
            "pipeline_tp": overall.get("tp", 0),
            "pipeline_fp": overall.get("fp", 0),
            "pipeline_fn": overall.get("fn", 0),

            # Custom evaluator (per class)
            "pipeline_map50_by_class": per_class_maps,
            "pipeline_recall_by_class": per_class_rec,
            "pipeline_precision_by_class": per_class_prec,
            "pipeline_f1_by_class": per_class_f1,
            "pipeline_tp_by_class": per_class_tp,
            "pipeline_fp_by_class": per_class_fp,
            "pipeline_fn_by_class": per_class_fn,

            # Performance
            "stage2_latency_ms": s2_avg_ms,
            "stage3_latency_ms": s3_avg_ms,
            "latency_ms": pipeline_latency_ms,
            "fps": fps,

            # FLOPs (per frame)
            "stage2_gflops": flops_info["stage2_gflops"],
            "stage3_gflops": flops_info["stage3_per_inference"],
            "stage3_total_gflops": flops_info["stage3_total_per_frame"],
            "gflops": flops_info["total_gflops"],

            # Other stats
            "avg_crops_per_frame": avg_crops,
            "nms_removal_rate": nms_rate,
            "local_nms_removal_rate": local_rate,
            "global_nms_removal_rate": global_rate,
            "pred_gt_ratio": pred_gt_ratio,
            "total_predictions": self.total_predictions,
            "total_ground_truths": self.total_ground_truths,
            
            # NEW: Privacy stats
            "privacy_enabled": self.privacy.enabled,
            "privacy_scope": self.privacy.scope if self.privacy.enabled else None,
        }

        # Console summary
        print("\n" + "=" * 90)
        print(f"[RESULT] mAP50={results_out['pipeline_map50']:.4f} "
              f"P={results_out['pipeline_precision']:.4f} "
              f"R={results_out['pipeline_recall']:.4f} "
              f"F1={results_out['pipeline_f1']:.4f}")
        print(f"[RESULT] FPS={fps:.1f}  latency={pipeline_latency_ms:.1f} ms "
              f"| GFLOPs={results_out['gflops']:.1f}  pred/gt={pred_gt_ratio:.2f}")
        if self.privacy.enabled:
            print(f"[PRIVACY] Enabled - scope: {self.privacy.scope}")
        print("=" * 90 + "\n")

        # Text summary
        with open(os.path.join(self.logs_dir, "summary.txt"), "w") as f:
            f.write(f"mAP50={results_out['pipeline_map50']:.6f}\n")
            f.write(f"precision={results_out['pipeline_precision']:.6f}\n")
            f.write(f"recall={results_out['pipeline_recall']:.6f}\n")
            f.write(f"f1={results_out['pipeline_f1']:.6f}\n")
            f.write(f"fps={fps:.3f}\nlatency_ms={pipeline_latency_ms:.3f}\n")
            if self.privacy.enabled:
                f.write(f"privacy_enabled=true\n")
                f.write(f"privacy_scope={self.privacy.scope}\n")

        return results_out


# =====================================================================
# Multi-Experiment Orchestrator
# =====================================================================
class MultiExperimentPipeline:
    """Runs a battery of experiments and writes clean custom-only reports/plots."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.stamp = _pst_stamp()
        self.output_dir = ensure_dir(os.path.join("processed_videos", f"{self.stamp}_multi_experiments"))

        # Either use a grid in config, or default to: no tracker + tracker gaps 1..10
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)
        grid = cfg.get("grid", {})
        if grid:
            use_tracker_opts = grid.get("use_tracker", [False, True])
            frame_gap_opts = grid.get("frame_gap", [1, 2, 3, 4, 5])
            self.experiments = []
            for ut in use_tracker_opts:
                for fg in frame_gap_opts:
                    name = f"{'tracker' if ut else 'no_tracker'}_gap{fg}"
                    self.experiments.append((bool(ut), int(fg), name))
        else:
            self.experiments = [(False, 1, "no_tracker")]
            self.experiments.extend([(True, g, f"tracker_gap{g}") for g in range(1, 11)])

    def run_all(self):
        results = []
        for ut, fg, name in self.experiments:
            exp = SingleExperiment(self.config_path, ut, fg, name, self.output_dir)
            results.append(exp.run())
        return results

    def run(self):
        print("\n" + "=" * 100)
        print("MULTI-CONFIGURATION PIPELINE EVALUATION")
        print(f"Output: {self.output_dir}")
        print(f"Experiments: {len(self.experiments)}")
        print("=" * 100)

        results = self.run_all()

        # Model info for report headers
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)
        s2_method = cfg["stage_2"].get("approach", "")
        s3_method = cfg["stage_3"].get("approach", "")
        s2_path = cfg["stage_2"].get(s2_method, {}).get("model_path", "")
        s3_path = cfg["stage_3"].get(s3_method, {}).get("model_path", "")
        s2_name = os.path.basename(s2_path).replace(".pt", "") if s2_path else "Unknown"
        s3_name = os.path.basename(s3_path).replace(".pt", "") if s3_path else "Unknown"
        model_name = f"{s2_name} ---> {s3_name}"
        class_names = cfg["stage_3"].get("names", ["handgun", "knife"])

        # Reports (Custom only)
        print("\n[REPORT] Writing reports...")
        write_detailed_report(
            results,
            os.path.join(self.output_dir, "report_detailed.txt"),
            self.stamp, model_name, class_names
        )
        write_combined_table(
            results,
            os.path.join(self.output_dir, "table_combined.txt"),
            model_name
        )
        write_per_class_table(
            results,
            os.path.join(self.output_dir, "table_per_class.txt"),
            model_name, class_names
        )

        # Plots & console summary
        plot_results(results, self.output_dir, class_names)
        print_summary(results, self.output_dir)


# Entry
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    runner = MultiExperimentPipeline(args.config)
    runner.run()
