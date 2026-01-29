"""
Diagnostic script to understand mAP drop from 0.7 to 0.5
Compares custom evaluation with Ultralytics evaluation
"""

import os
import glob
import cv2
import yaml
from collections import defaultdict
from ultralytics import YOLO
import numpy as np

from utils.box_utils import load_yolo_labels, square_scale_clip_xyxy, remap_crop_to_frame
from stages.stage_2_persondetection import PersonDetectionStage
from utils.evaluation import PipelineEvaluator


class EvaluationDiagnostics:
    """Compare custom evaluation vs Ultralytics evaluation."""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        # Paths
        p_cfg = self.cfg["pipeline"]
        self.frames_dir = p_cfg.get("frames_dir")
        self.labels_dir = p_cfg.get("labels_dir")
        
        # Models
        self.person_stage = PersonDetectionStage(self.cfg["stage_2"])
        
        s3 = self.cfg["stage_3"]
        method = s3.get("approach", "")
        model_path = s3.get(method, {}).get("model_path", "")
        conf_thresh = s3.get(method, {}).get("confidence_threshold", 0.3)
        
        print(f"[INFO] Loading weapon model: {model_path}")
        self.weapon_model = YOLO(model_path)
        self.conf_threshold = conf_thresh
        self.class_names = s3.get("names", ["handgun", "knife"])
        
        # Config
        self.crop_scale = float(self.cfg["stage_2"].get("crop_scale", 1.5))
    
    def test_ultralytics_direct(self, num_samples=50):
        """Test 1: Run Ultralytics directly on full images."""
        print("\n" + "="*80)
        print("TEST 1: ULTRALYTICS DIRECT EVALUATION (NO CROPS)")
        print("="*80)
        
        image_paths = sorted(glob.glob(os.path.join(self.frames_dir, "*.*")))[:num_samples]
        
        # Prepare dataset for Ultralytics
        temp_yaml = "temp_dataset.yaml"
        with open(temp_yaml, "w") as f:
            f.write(f"path: .\n")
            f.write(f"train: {self.frames_dir}\n")
            f.write(f"val: {self.frames_dir}\n")
            f.write(f"test: {self.frames_dir}\n")
            f.write(f"names:\n")
            for i, name in enumerate(self.class_names):
                f.write(f"  {i}: {name}\n")
        
        print(f"[INFO] Testing on {len(image_paths)} images...")
        print(f"[INFO] Confidence threshold: {self.conf_threshold}")
        
        # Run Ultralytics validation
        results = self.weapon_model.val(
            data=temp_yaml,
            split="test",
            conf=self.conf_threshold,
            iou=0.5,
            verbose=True,
            plots=True,
            save_json=True
        )
        
        print(f"\n[ULTRALYTICS] Results:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  Precision: {results.box.p[0]:.4f}")
        print(f"  Recall: {results.box.r[0]:.4f}")
        
        # Clean up
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)
        
        return results.box.map50
    
    def test_custom_evaluation(self, num_samples=50):
        """Test 2: Run custom evaluation on full images (no crops)."""
        print("\n" + "="*80)
        print("TEST 2: CUSTOM EVALUATION ON FULL IMAGES (NO CROPS)")
        print("="*80)
        
        evaluator = PipelineEvaluator(self.class_names)
        
        image_paths = sorted(glob.glob(os.path.join(self.frames_dir, "*.*")))[:num_samples]
        print(f"[INFO] Processing {len(image_paths)} images...")
        
        for idx, img_path in enumerate(image_paths):
            # Load image and GT
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(self.labels_dir, stem + ".txt")
            
            gt_boxes = load_yolo_labels(gt_path)
            
            # Run detection on full image
            results = self.weapon_model.predict(
                img,
                conf=self.conf_threshold,
                verbose=False,
                imgsz=640
            )
            
            # Convert predictions to YOLO format
            predictions = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    # Get normalized xywh
                    if hasattr(box, 'xywhn'):
                        xywhn = box.xywhn[0].cpu().numpy()
                        cx, cy, bw, bh = xywhn
                    else:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        cx = (x1 + x2) / 2 / w
                        cy = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                    
                    predictions.append([cls, cx, cy, bw, bh, conf])
            
            # Evaluate
            evaluator.evaluate_frame(predictions, gt_boxes, w, h)
            
            if idx % 10 == 0:
                print(f"  Processed {idx+1}/{len(image_paths)} images")
        
        # Compute metrics
        results = evaluator.compute_metrics()
        evaluator.print_results(results)
        
        return results['overall']['mAP50']
    
    def test_pipeline_with_crops(self, num_samples=50):
        """Test 3: Run full pipeline with crops."""
        print("\n" + "="*80)
        print("TEST 3: FULL PIPELINE WITH CROPS")
        print("="*80)
        
        evaluator = PipelineEvaluator(self.class_names)
        
        image_paths = sorted(glob.glob(os.path.join(self.frames_dir, "*.*")))[:num_samples]
        print(f"[INFO] Processing {len(image_paths)} images...")
        print(f"[INFO] Crop scale: {self.crop_scale}")
        
        # Reset person stage to avoid frame_gap division by zero
        if hasattr(self.person_stage, 'frame_gap'):
            self.person_stage.frame_gap = 1
        if hasattr(self.person_stage, 'use_tracker'):
            self.person_stage.use_tracker = False
        
        total_crops = 0
        total_predictions = 0
        
        for idx, img_path in enumerate(image_paths):
            # Load image and GT
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            stem = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(self.labels_dir, stem + ".txt")
            
            gt_boxes = load_yolo_labels(gt_path)
            
            # Stage-2: Detect persons
            _, persons, _ = self.person_stage.run(img.copy(), idx)
            
            # Process crops
            frame_predictions = []
            
            if persons:
                total_crops += len(persons)
                
                for person in persons:
                    bbox = person.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue
                    
                    x1, y1, x2, y2 = bbox
                    cx1, cy1, cx2, cy2 = square_scale_clip_xyxy(x1, y1, x2, y2, w, h, self.crop_scale)
                    ix1, iy1, ix2, iy2 = map(int, [cx1, cy1, cx2, cy2])
                    
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue
                    
                    crop = img[iy1:iy2, ix1:ix2].copy()
                    if crop.size == 0:
                        continue
                    
                    # Detect weapons in crop
                    results = self.weapon_model.predict(
                        crop,
                        conf=self.conf_threshold,
                        verbose=False,
                        imgsz=640
                    )
                    
                    # Convert to crop coordinates
                    crop_predictions = []
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        crop_h, crop_w = crop.shape[:2]
                        
                        for box in results[0].boxes:
                            cls = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1_c, y1_c, x2_c, y2_c = xyxy
                            
                            # Normalize to crop
                            cx_crop = (x1_c + x2_c) / 2 / crop_w
                            cy_crop = (y1_c + y2_c) / 2 / crop_h
                            w_crop = (x2_c - x1_c) / crop_w
                            h_crop = (y2_c - y1_c) / crop_h
                            
                            crop_predictions.append([cls, cx_crop, cy_crop, w_crop, h_crop, conf])
                    
                    # Remap to frame
                    remapped = remap_crop_to_frame(
                        crop_predictions,
                        [cx1, cy1, cx2, cy2],
                        w, h
                    )
                    frame_predictions.extend(remapped)
            
            total_predictions += len(frame_predictions)
            
            # Evaluate
            evaluator.evaluate_frame(frame_predictions, gt_boxes, w, h)
            
            if idx % 10 == 0:
                print(f"  Processed {idx+1}/{len(image_paths)} images")
        
        avg_crops = total_crops / len(image_paths)
        avg_preds = total_predictions / len(image_paths)
        
        print(f"\n[STATS] Avg crops/frame: {avg_crops:.2f}")
        print(f"[STATS] Avg predictions/frame: {avg_preds:.2f}")
        
        # Compute metrics
        results = evaluator.compute_metrics()
        evaluator.print_results(results)
        
        return results['overall']['mAP50']
    
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("\n" + "="*80)
        print("EVALUATION DIAGNOSTICS")
        print("="*80)
        
        # Test 1: Ultralytics on full images
        ultralytics_map = self.test_ultralytics_direct(num_samples=100)
        
        # Test 2: Custom eval on full images
        custom_full_map = self.test_custom_evaluation(num_samples=100)
        
        # Test 3: Full pipeline with crops
        pipeline_map = self.test_pipeline_with_crops(num_samples=100)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Test 1 - Ultralytics (full images): mAP50 = {ultralytics_map:.4f}")
        print(f"Test 2 - Custom (full images):      mAP50 = {custom_full_map:.4f}")
        print(f"Test 3 - Pipeline (with crops):     mAP50 = {pipeline_map:.4f}")
        
        print("\n[ANALYSIS]")
        
        eval_diff = abs(ultralytics_map - custom_full_map)
        if eval_diff > 0.05:
            print(f"⚠️  Large difference between Ultralytics and Custom evaluation: {eval_diff:.4f}")
            print("   → Evaluation method mismatch detected!")
        else:
            print(f"✓  Evaluation methods agree (diff: {eval_diff:.4f})")
        
        crop_drop = ultralytics_map - pipeline_map
        if crop_drop > 0.1:
            print(f"⚠️  Large mAP drop with crops: {crop_drop:.4f}")
            print("   → Possible issues:")
            print("     1. Crop scale too small (missing context)")
            print("     2. Coordinate remapping errors")
            print("     3. Excessive NMS removing valid detections")
            print("     4. Confidence threshold too high")
        else:
            print(f"✓  Reasonable performance with crops (drop: {crop_drop:.4f})")
        
        print("\n[RECOMMENDATIONS]")
        if custom_full_map > 0.65:
            print("✓  Base detection model is good (>0.65 mAP)")
            if pipeline_map < 0.55:
                print("→ Focus on pipeline issues:")
                print("  • Increase crop_scale from 1.3 to 1.6-1.8")
                print("  • Lower confidence thresholds (0.25-0.30)")
                print("  • Review NMS settings")
                print("  • Check coordinate conversion in remap_crop_to_frame()")
        else:
            print("⚠️  Base model performance is low (<0.65 mAP)")
            print("→ Model needs retraining or better configuration")


if __name__ == "__main__":
    diagnostics = EvaluationDiagnostics("config.yaml")
    diagnostics.run_all_tests()