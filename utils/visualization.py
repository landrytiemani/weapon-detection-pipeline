import cv2
import numpy as np
import os
import random
from pathlib import Path

class PipelineVisualizer:
    """Visualize pipeline predictions and ground truth for debugging."""
    
    def __init__(self, output_dir, class_names, num_samples=20):
        self.output_dir = output_dir
        self.class_names = class_names
        self.num_samples = num_samples
        self.vis_dir = os.path.join(output_dir, "visual_debug")
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Store frame data for random sampling
        self.frame_data = []
        
        # Colors for visualization
        # FIX: Added 'fn' key that was missing in original
        self.colors = {
            'gt': (0, 255, 0),      # Green for ground truth
            'pred': (0, 0, 255),    # Red for predictions
            'person': (255, 0, 0),  # Blue for person boxes
            'tp': (0, 255, 0),      # Green for true positives
            'fp': (0, 0, 255),      # Red for false positives 
            'fn': (255, 0, 255),    # Magenta for false negatives - THIS WAS MISSING
        }
    
    def yolo_to_xyxy(self, yolo_box, img_w, img_h):
        """Convert YOLO normalized format to pixel coordinates."""
        cx, cy, w, h = yolo_box
        x1 = int((cx - w/2) * img_w)
        y1 = int((cy - h/2) * img_h)
        x2 = int((cx + w/2) * img_w)
        y2 = int((cy + h/2) * img_h)
        return [x1, y1, x2, y2]
    
    def store_frame_data(self, frame_name, img, pred_boxes, gt_boxes, person_boxes):
        """Store frame data for later visualization."""
        self.frame_data.append({
            'frame_name': frame_name,
            'img': img.copy(),
            'pred_boxes': pred_boxes,
            'gt_boxes': gt_boxes,
            'person_boxes': person_boxes
        })
    
    def compute_iou(self, boxA, boxB):
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
    
    def match_predictions_to_gt(self, pred_boxes, gt_boxes, img_w, img_h, iou_thresh=0.5):
        """Match predictions to ground truth and classify as TP/FP/FN."""
        # Convert to pixel coordinates
        pred_xyxy = []
        for pred in pred_boxes:
            if len(pred) >= 5:
                cls, cx, cy, w, h = pred[:5]
                conf = pred[5] if len(pred) > 5 else 1.0
                x1, y1, x2, y2 = self.yolo_to_xyxy([cx, cy, w, h], img_w, img_h)
                pred_xyxy.append({
                    'class': int(cls),
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf,
                    'type': None
                })
        
        gt_xyxy = []
        for gt in gt_boxes:
            cls, cx, cy, w, h = gt[:5]
            x1, y1, x2, y2 = self.yolo_to_xyxy([cx, cy, w, h], img_w, img_h)
            gt_xyxy.append({
                'class': int(cls),
                'bbox': [x1, y1, x2, y2],
                'matched': False
            })
        
        # Match predictions to GT (per class)
        for cls_idx in range(len(self.class_names)):
            cls_preds = [p for p in pred_xyxy if p['class'] == cls_idx]
            cls_gts = [g for g in gt_xyxy if g['class'] == cls_idx]
            
            # Sort by confidence
            cls_preds = sorted(cls_preds, key=lambda x: x['conf'], reverse=True)
            
            matched_gts = set()
            
            for pred in cls_preds:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(cls_gts):
                    if gt_idx in matched_gts:
                        continue
                    iou = self.compute_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thresh and best_gt_idx >= 0:
                    pred['type'] = 'tp'
                    cls_gts[best_gt_idx]['matched'] = True
                    matched_gts.add(best_gt_idx)
                else:
                    pred['type'] = 'fp'
        
        # Mark unmatched GT as FN
        fn_boxes = [g for g in gt_xyxy if not g['matched']]
        
        return pred_xyxy, fn_boxes
    
    def draw_box(self, img, bbox, label, color, thickness=2):
        """Draw a bounding box with label."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + 10)
        cv2.rectangle(img, (x1, label_y - label_size[1] - 10), 
                     (x1 + label_size[0], label_y), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, label_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def visualize_random_frames(self):
        """Visualize random sample of frames."""
        if not self.frame_data:
            print("[WARN] No frame data to visualize")
            return
        
        # Randomly sample frames
        num_to_sample = min(self.num_samples, len(self.frame_data))
        sampled_frames = random.sample(self.frame_data, num_to_sample)
        
        print(f"\n[VIS] Visualizing {num_to_sample} random frames...")
        
        for idx, frame_info in enumerate(sampled_frames):
            img = frame_info['img'].copy()
            h, w = img.shape[:2]
            frame_name = frame_info['frame_name']
            
            # Match predictions to GT
            matched_preds, fn_boxes = self.match_predictions_to_gt(
                frame_info['pred_boxes'],
                frame_info['gt_boxes'],
                w, h
            )
            
            # Draw person boxes (blue)
            for person in frame_info['person_boxes']:
                bbox = person.get('bbox', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    cv2.rectangle(img, (x1, y1), (x2, y2), self.colors['person'], 2)
                    cv2.putText(img, 'Person', (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['person'], 2)
            
            # Draw ground truth boxes (green)
            for gt in frame_info['gt_boxes']:
                cls, cx, cy, gw, gh = gt[:5]
                x1, y1, x2, y2 = self.yolo_to_xyxy([cx, cy, gw, gh], w, h)
                cls_name = self.class_names[int(cls)]
                self.draw_box(img, [x1, y1, x2, y2], f'GT: {cls_name}', 
                            self.colors['gt'], thickness=3)
            
            # Draw predictions with TP/FP marking
            for pred in matched_preds:
                cls_name = self.class_names[pred['class']]
                conf = pred['conf']
                pred_type = pred['type']
                
                if pred_type == 'tp':
                    color = self.colors['tp']
                    label = f'TP: {cls_name} {conf:.2f}'
                else:  # fp
                    color = self.colors['fp']
                    label = f'FP: {cls_name} {conf:.2f}'
                
                self.draw_box(img, pred['bbox'], label, color, thickness=2)
            
            # Draw FN boxes (missed detections) - NOW WORKS WITH FIX
            for fn in fn_boxes:
                cls_name = self.class_names[fn['class']]
                self.draw_box(img, fn['bbox'], f'FN: {cls_name}', 
                            self.colors['fn'], thickness=3)
            
            # Add legend
            legend_y = 30
            cv2.putText(img, 'Legend:', (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            legend_y += 25
            cv2.putText(img, 'Blue=Person, Green=GT/TP, Red=FP, Magenta=FN', 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add stats
            stats_y = h - 80
            num_gt = len(frame_info['gt_boxes'])
            num_pred = len(matched_preds)
            num_tp = sum(1 for p in matched_preds if p['type'] == 'tp')
            num_fp = sum(1 for p in matched_preds if p['type'] == 'fp')
            num_fn = len(fn_boxes)
            
            cv2.putText(img, f'GT: {num_gt} | Pred: {num_pred} | TP: {num_tp} | FP: {num_fp} | FN: {num_fn}', 
                       (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save visualization
            output_path = os.path.join(self.vis_dir, f'{idx+1:03d}_{frame_name}.jpg')
            cv2.imwrite(output_path, img)
        
        print(f"[VIS] Visualization complete. Saved to: {self.vis_dir}")
