"""
Box manipulation utilities for YOLO-format detections and post-processing.

Public API (imported elsewhere):
- load_yolo_labels, write_yolo_labels
- square_scale_clip_xyxy
- remap_crop_to_frame
- compute_iou_yolo, compute_iou_xyxy
- apply_nms, apply_soft_nms, apply_global_nms
- filter_overlapping_crops
- filter_suspicious_predictions
- filter_low_confidence_predictions
- get_prediction_stats
"""

from typing import List, Dict, Any, Optional, Iterable
import os
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------
def load_yolo_labels(path_txt: str) -> List[List[float]]:
    """Read YOLO txt -> list[[cls,cx,cy,w,h], ...]; [] if file missing/empty."""
    if not os.path.exists(path_txt):
        return []
    out = []
    with open(path_txt, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                out.append([cls, cx, cy, w, h])
    return out


def write_yolo_labels(path_txt: str, boxes: Iterable[Iterable[float]]) -> None:
    """Write YOLO txt from list[[cls,cx,cy,w,h], ...]."""
    if not boxes:
        open(path_txt, "w").close()
        return
    with open(path_txt, "w") as f:
        for cls, cx, cy, w, h in boxes:
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def square_scale_clip_xyxy(x1: float, y1: float, x2: float, y2: float,
                           img_w: int, img_h: int, scale: float) -> List[float]:
    """
    Square crop centered on bbox center; side = scale * max(w,h); clipped to image.
    Returns [nx1, ny1, nx2, ny2] in absolute pixels.
    """
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    side = max(w, h) * float(scale)

    nx1 = max(0.0, cx - side / 2.0)
    ny1 = max(0.0, cy - side / 2.0)
    nx2 = min(float(img_w - 1), cx + side / 2.0)
    ny2 = min(float(img_h - 1), cy + side / 2.0)

    # ensure valid box
    if nx2 <= nx1 + 1 or ny2 <= ny1 + 1:
        nx1 = max(0.0, min(nx1, img_w - 2))
        ny1 = max(0.0, min(ny1, img_h - 2))
        nx2 = min(float(img_w - 1), nx1 + 2)
        ny2 = min(float(img_h - 1), ny1 + 2)
    return [nx1, ny1, nx2, ny2]


def remap_crop_to_frame(crop_boxes: List[List[float]], crop_xyxy: List[float],
                        img_w: int, img_h: int) -> List[List[float]]:
    """
    Map crop-normalized YOLO boxes back to full-frame normalized [cls,cx,cy,w,h,conf].
    - crop_boxes: [[cls, cx, cy, w, h, conf?], ...] (cx,cy,w,h normalized within crop)
    - crop_xyxy: [x1,y1,x2,y2] crop coords in full-frame pixels
    """
    x1, y1, x2, y2 = map(float, crop_xyxy)
    cw = max(1.0, x2 - x1)
    ch = max(1.0, y2 - y1)
    out = []
    for box in crop_boxes:
        if len(box) < 5:
            continue
        cls, cx, cy, w, h = box[:5]
        conf = float(box[5]) if len(box) > 5 else 1.0

        # crop-normalized center to full-frame pixels
        cx_pix = x1 + cx * cw
        cy_pix = y1 + cy * ch
        w_pix = w * cw
        h_pix = h * ch

        # normalize to full-frame
        ncx = cx_pix / float(img_w)
        ncy = cy_pix / float(img_h)
        nw = w_pix / float(img_w)
        nh = h_pix / float(img_h)

        # clamp
        ncx = min(max(ncx, 0.0), 1.0)
        ncy = min(max(ncy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        out.append([int(cls), float(ncx), float(ncy), float(nw), float(nh), float(conf)])
    return out


# ---------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------
def compute_iou_yolo(boxA: List[float], boxB: List[float]) -> float:
    """IoU between two YOLO boxes [cls,cx,cy,w,h,(conf)] in normalized [0,1] space."""
    _, cx1, cy1, w1, h1 = boxA[:5]
    _, cx2, cy2, w2, h2 = boxB[:5]

    x1_1, y1_1 = cx1 - w1 / 2.0, cy1 - h1 / 2.0
    x2_1, y2_1 = cx1 + w1 / 2.0, cy1 + h1 / 2.0
    x1_2, y1_2 = cx2 - w2 / 2.0, cy2 - h2 / 2.0
    x2_2, y2_2 = cx2 + w2 / 2.0, cy2 + h2 / 2.0

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def compute_iou_xyxy(boxA: List[float], boxB: List[float]) -> float:
    """IoU between two boxes in [x1,y1,x2,y2] pixel coords."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0


# ---------------------------------------------------------------------
# NMS (greedy + soft + global)
# ---------------------------------------------------------------------
def _normalize_class_iou_overrides(class_specific_thresholds: Optional[Dict[Any, float]],
                                   class_names: Optional[List[str]] = None) -> Dict[int, float]:
    """
    Accept thresholds keyed by class id or name and normalize to {class_id: iou}.
    """
    if not class_specific_thresholds:
        return {}
    out: Dict[int, float] = {}
    for k, v in class_specific_thresholds.items():
        try:
            if isinstance(k, int) or str(k).isdigit():
                out[int(k)] = float(v)
            elif class_names and (k in class_names):
                out[int(class_names.index(k))] = float(v)
        except Exception:
            continue
    return out


def apply_nms(
    boxes: List[List[float]],
    iou_threshold: float = 0.5,
    class_specific_thresholds: Optional[Dict[Any, float]] = None,
    class_names: Optional[List[str]] = None,
) -> List[List[float]]:
    """
    Class-wise greedy NMS on YOLO boxes (normalized).
    boxes: [[cls,cx,cy,w,h,conf], ...]
    class_specific_thresholds: optional {class_id or class_name: iou}
    """
    if not boxes:
        return []

    cls_thr = _normalize_class_iou_overrides(class_specific_thresholds, class_names)

    # group by class
    by_cls: Dict[int, List[List[float]]] = defaultdict(list)
    for b in boxes:
        if len(b) < 6:
            continue
        by_cls[int(b[0])].append(b)

    keep_all: List[List[float]] = []
    for cls_id, arr in by_cls.items():
        arr = sorted(arr, key=lambda x: x[5], reverse=True)
        keep: List[List[float]] = []
        while arr:
            best = arr.pop(0)
            keep.append(best)
            thr = float(cls_thr.get(cls_id, iou_threshold))
            rest = []
            for bx in arr:
                if compute_iou_yolo(best, bx) <= thr:
                    rest.append(bx)
            arr = rest
        keep_all.extend(keep)
    return keep_all


def apply_soft_nms(
    boxes: List[List[float]],
    sigma: float = 0.5,
    score_thresh: float = 0.001,
    iou_thresh: float = 0.5,
    cls_filter: Optional[Iterable[int]] = None,
) -> List[List[float]]:
    """
    Soft-NMS (Gaussian) for YOLO boxes in [0..1] space.
    If cls_filter is provided, Soft-NMS is applied ONLY to those class ids; others use greedy NMS logic.

    Args:
        boxes: [[cls,cx,cy,w,h,conf], ...]
        sigma: Gaussian sigma (larger = gentler decay)
        score_thresh: minimum score to keep after decay
        iou_thresh: IoU for the greedy branch
        cls_filter: set/list of class ids to apply Soft-NMS to

    Returns:
        New list of boxes after suppression/decay.
    """
    if not boxes:
        return []

    boxes = [b[:] for b in boxes if len(b) >= 6]  # shallow copy
    boxes.sort(key=lambda x: x[5], reverse=True)
    out: List[List[float]] = []

    while boxes:
        best = boxes.pop(0)
        out.append(best)
        rest: List[List[float]] = []
        for b in boxes:
            if int(b[0]) != int(best[0]):
                # different class: never suppress each other
                rest.append(b)
                continue

            iou = compute_iou_yolo(best, b)

            # If class is in filter set, apply Soft-NMS decay; else use greedy thresholding
            if cls_filter is not None and int(b[0]) in cls_filter:
                # Gaussian decay on score
                decayed = b[:]
                decayed[5] = decayed[5] * np.exp(-(iou * iou) / float(max(1e-6, sigma)))
                if decayed[5] >= score_thresh:
                    rest.append(decayed)
            else:
                if iou <= iou_thresh:
                    rest.append(b)
        boxes = sorted(rest, key=lambda x: x[5], reverse=True)

    return out


def apply_global_nms(predictions: List[List[float]], iou_threshold: float = 0.3) -> List[List[float]]:
    """
    Global NMS across all detections to remove cross-crop duplicates of the SAME class.
    Keep highest-confidence instance per overlapping group.
    """
    if not predictions:
        return []

    arr = sorted(predictions, key=lambda x: x[5] if len(x) > 5 else 0.0, reverse=True)
    keep: List[List[float]] = []
    while arr:
        best = arr.pop(0)
        keep.append(best)
        rest = []
        for bx in arr:
            if int(bx[0]) != int(best[0]):  # only suppress same-class overlaps
                rest.append(bx)
                continue
            if compute_iou_yolo(best, bx) <= iou_threshold:
                rest.append(bx)
        arr = rest
    return keep


# ---------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------
def filter_overlapping_crops(persons: List[dict], iou_threshold: float = 0.7) -> List[dict]:
    """Remove highly overlapping person boxes to reduce duplicate weapon detections."""
    if not persons:
        return persons

    sorted_persons = sorted(persons, key=lambda p: p.get("confidence", 1.0), reverse=True)
    keep: List[dict] = []
    for person in sorted_persons:
        bbox = person.get("bbox")
        if not bbox or len(bbox) < 4:
            continue

        overlaps = False
        for kept in keep:
            kept_bbox = kept.get("bbox")
            if not kept_bbox or len(kept_bbox) < 4:
                continue
            if compute_iou_xyxy(bbox, kept_bbox) > iou_threshold:
                overlaps = True
                break

        if not overlaps:
            keep.append(person)
    return keep


def filter_suspicious_predictions(
    predictions: List[List[float]],
    min_size: float = 0.022,
    max_size: float = 0.28,
    min_aspect: float = 0.32,
    max_aspect: float = 3.2,
) -> List[List[float]]:
    """
    Remove predictions with suspicious dimensions or aspect ratios.
    - min/max size apply to both width and height
    - drop very small area boxes (area < 4e-4)
    - drop boxes too close to edges (3% margin)
    """
    out: List[List[float]] = []
    for p in predictions:
        if len(p) < 6:
            continue
        _, cx, cy, w, h, _ = p[:6]
        if w <= 0 or h <= 0:
            continue
        # size guards
        if w < min_size or h < min_size:
            continue
        if w > max_size or h > max_size:
            continue
        # aspect guards
        ar = w / h if h > 0 else 1e9
        if ar < min_aspect or ar > max_aspect:
            continue
        # area guard
        if (w * h) < 4e-4:
            continue
        # edge guard
        edge = 0.015 #0.03
        if cx < edge or cx > (1.0 - edge) or cy < edge or cy > (1.0 - edge):
            continue
        out.append(p)
    return out




def filter_low_confidence_predictions(predictions: List[List[float]], min_conf: float = 0.50) -> List[List[float]]:
    """Keep only predictions with confidence >= min_conf."""
    return [p for p in predictions if len(p) >= 6 and p[5] >= float(min_conf)]


# ---------------------------------------------------------------------
# Optional stats helper
# ---------------------------------------------------------------------
def get_prediction_stats(predictions: List[List[float]], ground_truths: List[List[float]]) -> Dict[str, Any]:
    """Quick counts for debugging: totals and per-class counts + pred/gt ratio."""
    pred_by_class = defaultdict(int)
    gt_by_class = defaultdict(int)
    for p in predictions:
        if len(p) >= 1:
            pred_by_class[int(p[0])] += 1
    for g in ground_truths:
        if len(g) >= 1:
            gt_by_class[int(g[0])] += 1
    total_pred = len(predictions)
    total_gt = len(ground_truths)
    ratio = total_pred / total_gt if total_gt > 0 else 0.0
    return {
        "total_pred": total_pred,
        "total_gt": total_gt,
        "ratio": ratio,
        "pred_by_class": dict(pred_by_class),
        "gt_by_class": dict(gt_by_class),
    }
