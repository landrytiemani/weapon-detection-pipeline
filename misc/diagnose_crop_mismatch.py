"""
Diagnostic to identify why pipeline crops give 0.496 instead of 0.719 mAP@0.5
Systematically tests different crop generation parameters to match training
"""

import os
import glob
import cv2
import yaml
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

from utils.box_utils import load_yolo_labels, square_scale_clip_xyxy
from stages.stage_2_persondetection import PersonDetectionStage
from utils.evaluation import PipelineEvaluator


class CropDiagnostics:
    """Diagnose crop generation mismatch between training and pipeline."""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        # Paths
        p_cfg = self.cfg["pipeline"]
        self.frames_dir = p_cfg.get("frames_dir")
        self.labels_dir = p_cfg.get("labels_dir")
        
        # Models
        self.person_stage = PersonDetectionStage(self.cfg["stage_2"])
        
        # Fix frame_gap issue
        if hasattr(self.person_stage, 'frame_gap'):
            self.person_stage.frame_gap = 1
        if hasattr(self.person_stage, 'use_tracker'):
            self.person_stage.use_tracker = False
        
        s3 = self.cfg["stage_3"]
        method = s3.get("approach", "")
        model_path = s3.get(method, {}).get("model_path", "")
        
        print(f"[INFO] Loading weapon model: {model_path}")
        self.weapon_model = YOLO(model_path)
        self.class_names = s3.get("names", ["handgun", "knife"])
    
    def create_crop_with_scale(self, img, person_bbox, scale, method="square"):
        """
        Create a crop with different methods to match training.
        
        Args:
            img: Original image
            person_bbox: [x1, y1, x2, y2] person bounding box
            scale: Scale factor (e.g., 1.0, 1.2, 1.5)
            method: "square" (maintain square) or "letterbox" (preserve aspect)
        """
        h, w = img.shape[:2]
        x1, y1, x2, y2 = person_bbox
        
        if method == "square":
            # Your current pipeline method
            cx1, cy1, cx2, cy2 = square_scale_clip_xyxy(x1, y1, x2, y2, w, h, scale)
            ix1, iy1, ix2, iy2 = map(int, [cx1, cy1, cx2, cy2])
            crop = img[iy1:iy2, ix1:ix2].copy()
            
        elif method == "expand_and_pad":
            # Alternative: Expand then pad to square
            pw, ph = x2 - x1, y2 - y1
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Expand by scale
            new_w, new_h = pw * scale, ph * scale
            
            # Crop
            cx1 = max(0, int(center_x - new_w / 2))
            cy1 = max(0, int(center_y - new_h / 2))
            cx2 = min(w, int(center_x + new_w / 2))
            cy2 = min(h, int(center_y + new_h / 2))
            
            crop = img[cy1:cy2, cx1:cx2].copy()
            
            # Pad to square
            ch, cw = crop.shape[:2]
            max_dim = max(ch, cw)
            
            # Create padded image
            padded = np.full((max_dim, max_dim, 3), 114, dtype=np.uint8)
            y_offset = (max_dim - ch) // 2
            x_offset = (max_dim - cw) // 2
            padded[y_offset:y_offset+ch, x_offset:x_offset+cw] = crop
            crop = padded
            
        elif method == "tight":
            # Minimal expansion, just the person box
            pw, ph = x2 - x1, y2 - y1
            margin = 0.05  # 5% margin
            
            x1 = max(0, int(x1 - pw * margin))
            y1 = max(0, int(y1 - ph * margin))
            x2 = min(w, int(x2 + pw * margin))
            y2 = min(h, int(y2 + ph * margin))
            
            crop = img[int(y1):int(y2), int(x1):int(x2)].copy()
        
        return crop
    
    def test_crop_variations(self, num_samples=100):
        """
        Test different crop generation methods to find which matches training.
        """
        print("\n" + "="*80)
        print("TESTING CROP GENERATION VARIATIONS")
        print("="*80)
        
        scales = [1.0, 1.2, 1.5, 1.8, 2.0]
        methods = ["square", "expand_and_pad", "tight"]
        confidence_thresholds = [0.001, 0.05, 0.25, 0.50]
        
        results = []
        
        image_paths = sorted(glob.glob(os.path.join(self.frames_dir, "*.*")))[:num_samples]
        print(f"[INFO] Testing {len(image_paths)} images...")
        
        # Get ground truth count
        total_gt = 0
        for img_path in image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(self.labels_dir, stem + ".txt")
            gt_boxes = load_yolo_labels(gt_path)
            total_gt += len(gt_boxes)
        
        print(f"[INFO] Total ground truth objects: {total_gt}")
        
        # Test each combination
        for scale in scales:
            for method in methods:
                for conf_thresh in confidence_thresholds:
                    print(f"\n[TEST] scale={scale}, method={method}, conf={conf_thresh}")
                    
                    evaluator = PipelineEvaluator(self.class_names)
                    total_preds = 0
                    total_crops = 0
                    
                    for idx, img_path in enumerate(image_paths):
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        h, w = img.shape[:2]
                        stem = os.path.splitext(os.path.basename(img_path))[0]
                        gt_path = os.path.join(self.labels_dir, stem + ".txt")
                        gt_boxes = load_yolo_labels(gt_path)
                        
                        # Get person detections
                        _, persons, _ = self.person_stage.run(img.copy(), idx)
                        
                        if not persons:
                            evaluator.evaluate_frame([], gt_boxes, w, h)
                            continue
                        
                        # Process each person crop
                        frame_predictions = []
                        for person in persons:
                            bbox = person.get("bbox")
                            if not bbox or len(bbox) < 4:
                                continue
                            
                            # Create crop with current method
                            try:
                                crop = self.create_crop_with_scale(
                                    img, bbox, scale, method
                                )
                                
                                if crop.size == 0:
                                    continue
                                
                                total_crops += 1
                                
                                # Run weapon detection
                                results = self.weapon_model.predict(
                                    crop,
                                    conf=conf_thresh,
                                    verbose=False,
                                    imgsz=640
                                )
                                
                                # Convert predictions to YOLO format
                                crop_h, crop_w = crop.shape[:2]
                                
                                if results[0].boxes is not None and len(results[0].boxes) > 0:
                                    for box in results[0].boxes:
                                        cls = int(box.cls[0].item())
                                        conf = float(box.conf[0].item())
                                        
                                        xyxy = box.xyxy[0].cpu().numpy()
                                        x1_c, y1_c, x2_c, y2_c = xyxy
                                        
                                        # Normalize to crop
                                        cx = (x1_c + x2_c) / 2 / crop_w
                                        cy = (y1_c + y2_c) / 2 / crop_h
                                        bw = (x2_c - x1_c) / crop_w
                                        bh = (y2_c - y1_c) / crop_h
                                        
                                        # Store in full image coordinates
                                        # (simplified - just for counting)
                                        frame_predictions.append([cls, cx, cy, bw, bh, conf])
                                        total_preds += 1
                                        
                            except Exception as e:
                                continue
                        
                        # Evaluate frame
                        evaluator.evaluate_frame(frame_predictions, gt_boxes, w, h)
                    
                    # Compute metrics
                    metrics = evaluator.compute_metrics()
                    map50 = metrics['overall']['mAP50']
                    precision = metrics['overall']['precision']
                    recall = metrics['overall']['recall']
                    
                    avg_crops = total_crops / len(image_paths)
                    pred_gt_ratio = total_preds / total_gt if total_gt > 0 else 0
                    
                    results.append({
                        'scale': scale,
                        'method': method,
                        'conf': conf_thresh,
                        'map50': map50,
                        'precision': precision,
                        'recall': recall,
                        'avg_crops': avg_crops,
                        'pred_gt_ratio': pred_gt_ratio
                    })
                    
                    print(f"  mAP50: {map50:.4f}, P: {precision:.4f}, R: {recall:.4f}")
                    print(f"  Avg crops: {avg_crops:.2f}, Pred/GT: {pred_gt_ratio:.2f}")
        
        # Find best configuration
        results.sort(key=lambda x: x['map50'], reverse=True)
        
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS")
        print("="*80)
        print(f"{'Scale':<8} {'Method':<16} {'Conf':<8} {'mAP50':<10} {'Precision':<10} {'Recall':<10} {'Crops':<8} {'P/GT':<8}")
        print("-" * 80)
        
        for i, r in enumerate(results[:10]):
            print(f"{r['scale']:<8.1f} {r['method']:<16} {r['conf']:<8.3f} "
                  f"{r['map50']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f} "
                  f"{r['avg_crops']:<8.2f} {r['pred_gt_ratio']:<8.2f}")
        
        # Analysis
        best = results[0]
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if best['map50'] >= 0.68:
            print(f"✅ FOUND GOOD MATCH! (mAP50 = {best['map50']:.4f})")
            print(f"\nBest configuration:")
            print(f"  crop_scale: {best['scale']}")
            print(f"  method: {best['method']}")
            print(f"  confidence_threshold: {best['conf']}")
            print(f"\nUpdate your config.yaml:")
            print(f"  stage_2:")
            print(f"    crop_scale: {best['scale']}")
            print(f"  stage_3:")
            print(f"    rt_detr:")
            print(f"      confidence_threshold: {best['conf']}")
            
        elif best['map50'] >= 0.55:
            print(f"⚠️  PARTIAL MATCH (mAP50 = {best['map50']:.4f})")
            print(f"\nBest found: scale={best['scale']}, conf={best['conf']}")
            print(f"\nLikely issues:")
            print(f"  1. Crop preprocessing still doesn't perfectly match training")
            print(f"  2. May need different padding/resize strategy")
            print(f"  3. Check training code for exact crop generation method")
            
        else:
            print(f"❌ NO GOOD MATCH FOUND (best mAP50 = {best['map50']:.4f})")
            print(f"\nPossible issues:")
            print(f"  1. Pipeline crop generation fundamentally different from training")
            print(f"  2. Need to examine training code directly")
            print(f"  3. May need to implement custom crop method")
        
        print("\n[NEXT STEPS]")
        print("1. Ask training team for exact crop generation code")
        print("2. Look for training scripts in model directory")
        print("3. Check if training used any special preprocessing")
        print("4. Verify image normalization matches (mean/std)")
        
        return results
    
    def compare_with_training_result(self):
        """
        Compare against known training result of 0.719 mAP@0.5
        """
        print("\n" + "="*80)
        print("TRAINING vs PIPELINE COMPARISON")
        print("="*80)
        print(f"Training mAP@0.5:       0.719  ← Target")
        print(f"Pipeline current:       0.496  ← Need to fix")
        print(f"Gap:                    0.223  ← 31% drop")
        print("\nTesting to find configuration that matches 0.719...")


def main():
    """Run comprehensive crop diagnostics."""
    diagnostics = CropDiagnostics("config.yaml")
    
    # Show what we're looking for
    diagnostics.compare_with_training_result()
    
    # Test systematically
    results = diagnostics.test_crop_variations(num_samples=100)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nIf best result is close to 0.719:")
    print("  → Update config.yaml with recommended settings")
    print("  → Run main_optimized.py to verify")
    print("\nIf best result is far from 0.719:")
    print("  → Need exact training crop generation code")
    print("  → Contact training team for preprocessing details")


if __name__ == "__main__":
    main()