"""
Ultralytics validation utilities for pipeline evaluation.
Calculates COCO-style metrics on pre-computed pipeline predictions.
"""

import os
import shutil
import glob
import yaml
import json
import cv2
import torch
import numpy as np
from pathlib import Path


def create_validation_dataset(output_dir, frames_dir, labels_dir, pred_dir, 
                              experiment_name, class_names):
    """Create YOLO-format validation dataset from pipeline outputs."""
    print(f"\n[DATASET] Creating validation dataset for: {experiment_name}")
    
    dataset_root = os.path.join(output_dir, "ultralytics_datasets", experiment_name)
    val_images_dir = os.path.join(dataset_root, "images", "val")
    val_labels_dir = os.path.join(dataset_root, "labels", "val")
    val_preds_dir = os.path.join(dataset_root, "predictions", "val")
    
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(val_preds_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(frames_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(frames_dir, f"*{ext.upper()}")))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {frames_dir}")
    
    print(f"[DATASET] Found {len(image_files)} images")
    
    copied_images = 0
    copied_labels = 0
    copied_preds = 0
    
    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img_name = os.path.basename(img_path)
        
        # Copy image
        dst_img = os.path.join(val_images_dir, img_name)
        shutil.copy2(img_path, dst_img)
        copied_images += 1
        
        # Copy ground truth label
        label_src = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.exists(label_src):
            label_dst = os.path.join(val_labels_dir, f"{stem}.txt")
            shutil.copy2(label_src, label_dst)
            copied_labels += 1
        else:
            label_dst = os.path.join(val_labels_dir, f"{stem}.txt")
            open(label_dst, 'w').close()
        
        # Copy pipeline prediction
        pred_src = os.path.join(pred_dir, f"{stem}.txt")
        if os.path.exists(pred_src):
            pred_dst = os.path.join(val_preds_dir, f"{stem}.txt")
            shutil.copy2(pred_src, pred_dst)
            copied_preds += 1
        else:
            pred_dst = os.path.join(val_preds_dir, f"{stem}.txt")
            open(pred_dst, 'w').close()
    
    print(f"[DATASET] Copied {copied_images} images, {copied_labels} labels, {copied_preds} predictions")
    
    # Create data.yaml
    yaml_content = {
        'path': str(Path(dataset_root).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(dataset_root, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"[DATASET] Created data.yaml at: {yaml_path}")
    
    return yaml_path, dataset_root


def compute_iou_np(box1, box2):
    """
    Compute IoU between two boxes in xyxy format.
    box1, box2: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def xywh_to_xyxy(box):
    """Convert from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def calculate_metrics_from_predictions(pred_dir, labels_dir, image_dir, class_names, iou_threshold=0.5):
    """
    Calculate COCO-style metrics from pre-computed predictions.
    Fixed precision/recall calculation.
    """
    print(f"\n[METRICS] Calculating metrics from predictions...")
    print(f"  Predictions: {pred_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  IoU Threshold: {iou_threshold}")
    
    nc = len(class_names)
    
    # Collect all predictions and ground truths
    all_predictions = {c: [] for c in range(nc)}
    all_ground_truths = {c: [] for c in range(nc)}
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    
    print(f"[METRICS] Processing {len(image_files)} images...")
    
    total_predictions = 0
    total_gt = 0
    
    for img_idx, img_file in enumerate(image_files):
        stem = os.path.splitext(os.path.basename(img_file))[0]
        pred_file = os.path.join(pred_dir, f"{stem}.txt")
        label_file = os.path.join(labels_dir, f"{stem}.txt")
        
        # Read image dimensions
        img = cv2.imread(img_file)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Load predictions (normalized format: class cx cy w h)
        predictions = []
        if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(float(parts[0]))
                        cx, cy, pw, ph = map(float, parts[1:5])
                        # Assume high confidence for pre-filtered predictions
                        conf = 0.99 if len(parts) < 6 else float(parts[5])
                        
                        # Convert to pixel coordinates xyxy
                        box_xyxy = xywh_to_xyxy([cx * w, cy * h, pw * w, ph * h])
                        
                        predictions.append({
                            'class': cls,
                            'box': box_xyxy,
                            'confidence': conf,
                            'matched': False
                        })
        
        # Load ground truth (normalized format: class cx cy w h)
        ground_truths = []
        if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(float(parts[0]))
                        cx, cy, lw, lh = map(float, parts[1:5])
                        
                        # Convert to pixel coordinates xyxy
                        box_xyxy = xywh_to_xyxy([cx * w, cy * h, lw * w, lh * h])
                        
                        ground_truths.append({
                            'class': cls,
                            'box': box_xyxy,
                            'matched': False
                        })
        
        total_predictions += len(predictions)
        total_gt += len(ground_truths)
        
        # Match predictions to ground truths for this image
        # Sort predictions by confidence (highest first)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        for pred in predictions:
            pred_cls = pred['class']
            pred_box = pred['box']
            pred_conf = pred['confidence']
            
            # Find best matching ground truth of same class
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt['class'] != pred_cls:
                    continue
                if gt['matched']:
                    continue
                
                iou = compute_iou_np(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Add prediction with match status
            is_correct = (best_iou >= iou_threshold and best_gt_idx >= 0)
            
            all_predictions[pred_cls].append({
                'confidence': pred_conf,
                'correct': is_correct
            })
            
            if is_correct:
                ground_truths[best_gt_idx]['matched'] = True
                pred['matched'] = True
        
        # Count ground truths per class
        for gt in ground_truths:
            all_ground_truths[gt['class']].append(gt)
    
    print(f"[METRICS] Total predictions: {total_predictions}, Total GT: {total_gt}")
    
    if total_predictions == 0:
        print("[WARN] No predictions found!")
        return get_empty_metrics(class_names)
    
    # Calculate AP per class
    class_aps = []
    class_precisions = []
    class_recalls = []
    class_f1s = []
    
    for cls_idx in range(nc):
        preds = all_predictions[cls_idx]
        n_gt = len(all_ground_truths[cls_idx])
        
        if len(preds) == 0 or n_gt == 0:
            class_aps.append(0.0)
            class_precisions.append(0.0)
            class_recalls.append(0.0)
            class_f1s.append(0.0)
            continue
        
        # Sort predictions by confidence
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall at each prediction
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        f1_scores = []
        
        for pred in preds:
            if pred['correct']:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
            recall = tp_cumsum / n_gt if n_gt > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        # FIXED: Use metrics at maximum F1 score point
        if f1_scores:
            best_f1_idx = np.argmax(f1_scores)
            best_precision = precisions[best_f1_idx]
            best_recall = recalls[best_f1_idx]
            best_f1 = f1_scores[best_f1_idx]
        else:
            best_precision = 0.0
            best_recall = 0.0
            best_f1 = 0.0
        
        # Calculate AP using interpolated precision
        # Add boundary points
        precisions_interp = [1.0] + precisions + [0.0]
        recalls_interp = [0.0] + recalls + [1.0]
        
        # Make precision monotonically decreasing
        for i in range(len(precisions_interp) - 2, -1, -1):
            precisions_interp[i] = max(precisions_interp[i], precisions_interp[i + 1])
        
        # Calculate area under curve (AP)
        ap = 0.0
        for i in range(len(recalls_interp) - 1):
            ap += (recalls_interp[i + 1] - recalls_interp[i]) * precisions_interp[i + 1]
        
        class_aps.append(ap)
        class_precisions.append(best_precision)
        class_recalls.append(best_recall)
        class_f1s.append(best_f1)
        
        print(f"[METRICS] Class {class_names[cls_idx]}: AP={ap:.4f}, P={best_precision:.4f}, R={best_recall:.4f}, F1={best_f1:.4f}")
    
    # Calculate overall metrics (macro-average over classes)
    valid_classes = [i for i in range(nc) if class_aps[i] > 0]
    
    if valid_classes:
        overall_map50 = np.mean([class_aps[i] for i in valid_classes])
        overall_precision = np.mean([class_precisions[i] for i in valid_classes])
        overall_recall = np.mean([class_recalls[i] for i in valid_classes])
        overall_f1 = np.mean([class_f1s[i] for i in valid_classes])
    else:
        overall_map50 = 0.0
        overall_precision = 0.0
        overall_recall = 0.0
        overall_f1 = 0.0
    
    metrics = {
        'mAP50': float(overall_map50),
        'mAP50_95': float(overall_map50),  # Simplified: only using IoU@0.5
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'f1': float(overall_f1),
        'per_class_mAP50': [float(ap) for ap in class_aps],
        'per_class_precision': [float(p) for p in class_precisions],
        'per_class_recall': [float(r) for r in class_recalls],
        'per_class_f1': [float(f) for f in class_f1s]
    }
    
    print(f"\n[METRICS] Overall Results:")
    print(f"  mAP50: {metrics['mAP50']:.4f}")
    print(f"  Precision (at best F1): {metrics['precision']:.4f}")
    print(f"  Recall (at best F1): {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    
    return metrics

def get_empty_metrics(class_names):
    """Return empty metrics structure."""
    nc = len(class_names)
    return {
        'mAP50': 0.0,
        'mAP50_95': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'per_class_mAP50': [0.0] * nc,
        'per_class_precision': [0.0] * nc,
        'per_class_recall': [0.0] * nc
    }


def run_ultralytics_validation(model_path, dataset_yaml, output_dir, experiment_name,
                               conf_threshold=0.001, iou_threshold=0.5, imgsz=640):
    """
    Calculate Ultralytics-style metrics on pipeline predictions.
    """
    print(f"\n{'='*80}")
    print(f"ULTRALYTICS-STYLE VALIDATION - {experiment_name}")
    print('='*80)
    
    try:
        # Load dataset info
        with open(dataset_yaml, 'r') as f:
            data = yaml.safe_load(f)
        
        dataset_root = data['path']
        class_names = list(data['names'].values())
        
        pred_dir = os.path.join(dataset_root, "predictions", "val")
        labels_dir = os.path.join(dataset_root, "labels", "val")
        images_dir = os.path.join(dataset_root, "images", "val")
        
        print(f"[INFO] Dataset root: {dataset_root}")
        print(f"[INFO] Classes: {class_names}")
        
        # Calculate metrics from predictions
        metrics = calculate_metrics_from_predictions(
            pred_dir, labels_dir, images_dir, class_names, iou_threshold
        )
        
        # Save metrics
        val_output = os.path.join(output_dir, "ultralytics_validation", experiment_name)
        os.makedirs(val_output, exist_ok=True)
        
        metrics_file = os.path.join(val_output, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n[ULTRALYTICS] Validation Results:")
        print(f"  mAP50:       {metrics['mAP50']:.4f}")
        print(f"  mAP50-95:    {metrics['mAP50_95']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1:          {metrics['f1']:.4f}")
        
        if metrics['per_class_mAP50']:
            print(f"\n  Per-Class mAP50:")
            for i, (name, ap) in enumerate(zip(class_names, metrics['per_class_mAP50'])):
                print(f"    {name}: {ap:.4f}")
        
        print(f"\n[ULTRALYTICS] Results saved to: {val_output}")
        
        return metrics
        
    except Exception as e:
        print(f"[ERROR] Ultralytics validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def compare_metrics(custom_metrics, ultralytics_metrics, class_names):
    """
    Compare custom evaluator metrics with Ultralytics metrics.
    """
    print(f"\n{'='*80}")
    print("METRICS COMPARISON: Custom Evaluator vs Ultralytics")
    print('='*80)
    
    if 'error' in ultralytics_metrics:
        print(f"[ERROR] Cannot compare - Ultralytics validation failed")
        return None
    
    comparison = {
        'overall': {},
        'per_class': {}
    }
    
    # Overall metrics comparison
    metric_mappings = {
        'mAP50': 'mAP50',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    print("\nOverall Metrics:")
    print(f"{'Metric':<15} {'Custom':>12} {'Ultralytics':>12} {'Diff':>12}")
    print('-' * 55)
    
    for metric, custom_key in metric_mappings.items():
        custom_val = custom_metrics.get(custom_key, 0.0)
        ultra_val = ultralytics_metrics.get(metric, 0.0)
        diff = ultra_val - custom_val
        
        comparison['overall'][metric] = {
            'custom': custom_val,
            'ultralytics': ultra_val,
            'difference': diff
        }
        
        print(f"{metric:<15} {custom_val:>12.4f} {ultra_val:>12.4f} {diff:>+12.4f}")
    
    # Per-class comparison
    if 'per_class_mAP50' in ultralytics_metrics:
        print("\nPer-Class mAP50:")
        print(f"{'Class':<15} {'Custom':>12} {'Ultralytics':>12} {'Diff':>12}")
        print('-' * 55)
        
        for i, class_name in enumerate(class_names):
            # Custom metrics might be in a nested dict
            if isinstance(custom_metrics, dict) and 'per_class' in custom_metrics:
                custom_val = custom_metrics['per_class'].get(class_name, {}).get('mAP50', 0.0)
            else:
                custom_val = 0.0
            
            ultra_val = ultralytics_metrics['per_class_mAP50'][i] if i < len(ultralytics_metrics['per_class_mAP50']) else 0.0
            diff = ultra_val - custom_val
            
            comparison['per_class'][class_name] = {
                'custom_mAP50': custom_val,
                'ultralytics_mAP50': ultra_val,
                'difference': diff
            }
            
            print(f"{class_name:<15} {custom_val:>12.4f} {ultra_val:>12.4f} {diff:>+12.4f}")
    
    print('='*80)
    
    return comparison