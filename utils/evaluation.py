import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import glob
import os

def compute_iou(boxA, boxB):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    
    return interArea / unionArea if unionArea > 0 else 0.0


def yolo_to_xyxy(yolo_box, img_w, img_h):
    """Convert YOLO format [cls, cx, cy, w, h] to [cls, x1, y1, x2, y2]."""
    cls, cx, cy, w, h = yolo_box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return int(cls), x1, y1, x2, y2


class PipelineEvaluator:
    """End-to-end pipeline evaluator using proper mAP calculation (V7Labs method)."""
    
    def __init__(self, class_names, iou_thresholds=None):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        # Store all predictions and ground truths for proper AP calculation
        self.all_predictions = {cls_idx: [] for cls_idx in range(self.num_classes)}
        self.all_ground_truths = {cls_idx: [] for cls_idx in range(self.num_classes)}
        self.frame_id = 0
        
        # Debug counters
        self.total_frames_evaluated = 0
        self.total_predictions = 0
        self.total_gt_objects = 0
        
    def evaluate_frame(self, pred_boxes, gt_boxes, img_w, img_h, debug=False):
        """Store predictions and ground truths for a single frame."""
        self.total_frames_evaluated += 1
        
        # Convert predictions to xyxy format and store
        for pred in pred_boxes:
            if len(pred) >= 5:
                cls, cx, cy, w, h = pred[:5]
                conf = pred[5] if len(pred) > 5 else 1.0
                cls, x1, y1, x2, y2 = yolo_to_xyxy([cls, cx, cy, w, h], img_w, img_h)
                
                self.all_predictions[cls].append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'frame_id': self.frame_id
                })
                self.total_predictions += 1
        
        # Convert ground truths to xyxy format and store
        for gt in gt_boxes:
            cls, x1, y1, x2, y2 = yolo_to_xyxy(gt, img_w, img_h)
            
            self.all_ground_truths[cls].append({
                'bbox': [x1, y1, x2, y2],
                'frame_id': self.frame_id
            })
            self.total_gt_objects += 1
        
        if self.total_frames_evaluated % 100 == 0:
            print(f"[EVAL] Processed {self.total_frames_evaluated} frames, {self.total_predictions} preds, {self.total_gt_objects} GTs")
        
        self.frame_id += 1
    
    def calculate_ap_per_class(self, cls_idx, iou_thresh):
        """
        Calculate Average Precision for a single class at a given IoU threshold.
        Uses the precision-recall curve method as described in V7Labs blog.
        """
        predictions = self.all_predictions[cls_idx]
        ground_truths = self.all_ground_truths[cls_idx]
        
        num_gt = len(ground_truths)
        
        if num_gt == 0:
            return 0.0, 0, 0, 0
        
        if len(predictions) == 0:
            return 0.0, 0, 0, num_gt
        
        # Sort predictions by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        # Track which GTs have been matched
        gt_matched = [False] * num_gt
        
        # Arrays to store TP and FP for each prediction
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        # Match predictions to ground truths
        for pred_idx, pred in enumerate(predictions):
            # Find GTs from the same frame
            frame_gts = [(i, gt) for i, gt in enumerate(ground_truths) 
                        if gt['frame_id'] == pred['frame_id']]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find the GT with highest IoU
            for gt_idx, gt in frame_gts:
                if gt_matched[gt_idx]:
                    continue
                    
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Assign as TP or FP based on IoU threshold
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                if not gt_matched[best_gt_idx]:
                    tp[pred_idx] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Compute precision and recall at each prediction
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Add sentinel values at the beginning and end
        recalls = np.concatenate(([0.], recalls))
        precisions = np.concatenate(([1.], precisions))
        
        # Ensure precision is monotonically decreasing
        # (for any recall r, precision should be max of all precisions at recall >= r)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Calculate AP using interpolation at recall points
        # Method: Area under precision-recall curve
        ap = 0.0
        for i in range(len(recalls) - 1):
            ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]
        
        # Final counts at IoU threshold
        total_tp = int(np.sum(tp))
        total_fp = int(np.sum(fp))
        total_fn = num_gt - total_tp
        
        return ap, total_tp, total_fp, total_fn
    
    def compute_metrics(self):
        """
        Compute final metrics across all frames.
        Follows COCO-style mAP calculation:
        - Calculate AP for each class at each IoU threshold
        - Average across IoU thresholds to get AP per class
        - Average across classes to get mAP
        """
        results = {'overall': {}, 'per_class': {}}
        
        print(f"\n[EVAL] Computing metrics from {self.total_frames_evaluated} frames...")
        print(f"[EVAL] Total predictions: {self.total_predictions}")
        print(f"[EVAL] Total GT objects: {self.total_gt_objects}")
        
        # Store metrics for averaging
        all_class_ap50 = []
        all_class_ap50_95 = []
        overall_tp_50 = 0
        overall_fp_50 = 0
        overall_fn_50 = 0
        overall_gt = 0
        
        # Compute per-class metrics
        for cls_idx in range(self.num_classes):
            cls_name = self.class_names[cls_idx]
            num_gt = len(self.all_ground_truths[cls_idx])
            
            if num_gt == 0:
                print(f"[EVAL] Class {cls_name}: No ground truth objects, skipping")
                continue
            
            # Calculate AP at each IoU threshold
            ap_scores = []
            tp_50 = fp_50 = fn_50 = 0
            
            for iou_thresh in self.iou_thresholds:
                ap, tp, fp, fn = self.calculate_ap_per_class(cls_idx, iou_thresh)
                ap_scores.append(ap)
                
                # Store metrics at IoU=0.5 for reporting
                if iou_thresh == 0.5:
                    tp_50 = tp
                    fp_50 = fp
                    fn_50 = fn
            
            print(f"[EVAL] Class {cls_name}: TP={tp_50}, FP={fp_50}, FN={fn_50}, GT={num_gt}")
            
            # Calculate metrics at IoU=0.5
            precision_50 = tp_50 / (tp_50 + fp_50) if (tp_50 + fp_50) > 0 else 0.0
            recall_50 = tp_50 / (tp_50 + fn_50) if (tp_50 + fn_50) > 0 else 0.0
            f1_50 = 2 * precision_50 * recall_50 / (precision_50 + recall_50) if (precision_50 + recall_50) > 0 else 0.0
            
            # mAP50: AP at IoU=0.5
            map50 = ap_scores[0] if len(ap_scores) > 0 else 0.0
            
            # mAP50-95: Average AP across all IoU thresholds (0.5 to 0.95)
            map50_95 = np.mean(ap_scores) if ap_scores else 0.0
            
            results['per_class'][cls_name] = {
                'mAP50': map50,
                'mAP50_95': map50_95,
                'precision': precision_50,
                'recall': recall_50,
                'f1': f1_50,
                'tp': int(tp_50),
                'fp': int(fp_50),
                'fn': int(fn_50),
                'gt_count': num_gt
            }
            
            # Accumulate for overall metrics
            all_class_ap50.append(map50)
            all_class_ap50_95.append(map50_95)
            overall_tp_50 += tp_50
            overall_fp_50 += fp_50
            overall_fn_50 += fn_50
            overall_gt += num_gt
        
        # Compute overall metrics
        if overall_gt > 0 and len(all_class_ap50) > 0:
            # Overall mAP: average of per-class AP values
            overall_map50 = np.mean(all_class_ap50)
            overall_map50_95 = np.mean(all_class_ap50_95)
            
            # Overall precision/recall at IoU=0.5
            overall_precision = overall_tp_50 / (overall_tp_50 + overall_fp_50) if (overall_tp_50 + overall_fp_50) > 0 else 0.0
            overall_recall = overall_tp_50 / (overall_tp_50 + overall_fn_50) if (overall_tp_50 + overall_fn_50) > 0 else 0.0
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        else:
            overall_map50 = overall_map50_95 = 0.0
            overall_precision = overall_recall = overall_f1 = 0.0
        
        results['overall'] = {
            'mAP50': overall_map50,
            'mAP50_95': overall_map50_95,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': int(overall_tp_50),
            'fp': int(overall_fp_50),
            'fn': int(overall_fn_50),
            'total_gt': overall_gt
        }
        
        return results
    
    def print_results(self, results):
        """Print evaluation results."""
        print("\n" + "="*60)
        print("PIPELINE EVALUATION RESULTS")
        print("="*60)
        
        overall = results['overall']
        print("\nOverall Metrics:")
        print(f"  mAP50: {overall['mAP50']:.4f}")
        print(f"  mAP50-95: {overall['mAP50_95']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall: {overall['recall']:.4f}")
        print(f"  F1: {overall['f1']:.4f}")
        print(f"  TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']}")
        print(f"  Total GT Objects: {overall['total_gt']}")
        
        print("\nPer-Class Metrics:")
        for cls_name, metrics in results['per_class'].items():
            print(f"\n  {cls_name}:")
            print(f"    mAP50: {metrics['mAP50']:.4f}")
            print(f"    mAP50-95: {metrics['mAP50_95']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1: {metrics['f1']:.4f}")
            print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
            print(f"    GT Count: {metrics['gt_count']}")