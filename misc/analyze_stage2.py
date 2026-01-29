"""
Standalone Stage 2 Coverage Analyzer
Analyzes how well person detection captures weapons without needing imports
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
import yaml

def load_config(config_path):
    """Load pipeline configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def check_weapon_in_person_crop(weapon_box, person_box, crop_scale=1.0):
    """
    Check if weapon is within person crop area
    """
    # Expand person box by crop_scale
    cx = (person_box[0] + person_box[2]) / 2
    cy = (person_box[1] + person_box[3]) / 2
    w = (person_box[2] - person_box[0]) * crop_scale
    h = (person_box[3] - person_box[1]) * crop_scale
    
    expanded_person = [
        max(0, cx - w/2),
        max(0, cy - h/2),
        min(1, cx + w/2),
        min(1, cy + h/2)
    ]
    
    # Check if weapon center is inside expanded person box
    weapon_cx = (weapon_box[0] + weapon_box[2]) / 2
    weapon_cy = (weapon_box[1] + weapon_box[3]) / 2
    
    center_inside = (expanded_person[0] <= weapon_cx <= expanded_person[2] and
                    expanded_person[1] <= weapon_cy <= expanded_person[3])
    
    # Calculate weapon coverage (what % of weapon is inside person crop)
    x1 = max(weapon_box[0], expanded_person[0])
    y1 = max(weapon_box[1], expanded_person[1])
    x2 = min(weapon_box[2], expanded_person[2])
    y2 = min(weapon_box[3], expanded_person[3])
    
    if x2 > x1 and y2 > y1:
        intersection = (x2 - x1) * (y2 - y1)
        weapon_area = (weapon_box[2] - weapon_box[0]) * (weapon_box[3] - weapon_box[1])
        coverage = intersection / weapon_area if weapon_area > 0 else 0
    else:
        coverage = 0
    
    return {
        'center_inside': center_inside,
        'coverage': coverage,
        'fully_inside': coverage >= 0.99
    }

def run_yolov8_detection(img_path, config):
    """Run YOLOv8 person detection"""
    try:
        from ultralytics import YOLO
        
        model_path = config['stage_2']['yolov8_tracker']['model_path']
        conf_thresh = config['stage_2']['yolov8_tracker'].get('confidence_threshold', 0.25)
        
        model = YOLO(model_path)
        results = model(str(img_path), conf=conf_thresh, classes=0, verbose=False)
        
        person_boxes = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                person_boxes.append({
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': float(conf)
                })
        
        return person_boxes
    except Exception as e:
        print(f"  YOLOv8 detection error: {e}")
        return []

def analyze_coverage_simple(config_path, sample_size=None):
    """
    Simplified analysis that runs detection directly
    """
    config = load_config(config_path)
    
    # Get dataset paths
    frames_dir = Path(config['pipeline']['frames_dir'])
    labels_dir = Path(config['pipeline']['labels_dir'])
    
    # Get image files
    image_files = sorted(list(frames_dir.glob('*.jpg')) + list(frames_dir.glob('*.png')))
    if sample_size:
        image_files = image_files[:sample_size]
    
    print(f"\n📊 Analyzing {len(image_files)} images...")
    
    # Statistics
    stats = {
        'total_weapons': 0,
        'weapons_captured': 0,
        'weapons_missed': 0,
        'weapons_partial': 0,
        'capture_by_class': defaultdict(lambda: {'total': 0, 'captured': 0, 'partial': 0}),
        'missed_examples': [],
        'no_person_detected': 0,
        'avg_persons_per_image': []
    }
    
    crop_scale = config['stage_2'].get('crop_scale', 2.0)
    print(f"   Using crop_scale: {crop_scale}")
    print(f"   Confidence threshold: {config['stage_2']['yolov8_tracker'].get('confidence_threshold', 0.25)}\n")
    
    for img_idx, img_path in enumerate(image_files):
        if (img_idx + 1) % 10 == 0:
            print(f"   Processing: {img_idx+1}/{len(image_files)}...")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Get ground truth weapons from labels
        label_path = labels_dir / (img_path.stem + '.txt')
        gt_weapons = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        # Convert to x1y1x2y2
                        x1 = cx - bw/2
                        y1 = cy - bh/2
                        x2 = cx + bw/2
                        y2 = cy + bh/2
                        
                        class_names = config['stage_3'].get('names', ['handgun', 'knife'])
                        class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                        
                        gt_weapons.append({
                            'box': [x1, y1, x2, y2],  # Normalized coords
                            'class': class_name,
                            'size': bw * bh
                        })
        
        # Run person detection
        person_boxes_pixel = run_yolov8_detection(img_path, config)
        
        # Convert to normalized coordinates
        person_boxes = []
        for p in person_boxes_pixel:
            person_boxes.append({
                'box': [
                    p['box'][0] / w,
                    p['box'][1] / h,
                    p['box'][2] / w,
                    p['box'][3] / h
                ],
                'conf': p['conf']
            })
        
        stats['avg_persons_per_image'].append(len(person_boxes))
        
        # Check each weapon
        for weapon in gt_weapons:
            stats['total_weapons'] += 1
            stats['capture_by_class'][weapon['class']]['total'] += 1
            
            if len(person_boxes) == 0:
                stats['no_person_detected'] += 1
                stats['weapons_missed'] += 1
                if len(stats['missed_examples']) < 5:
                    stats['missed_examples'].append({
                        'image': img_path.name,
                        'class': weapon['class'],
                        'reason': 'No persons detected',
                        'weapon_size': weapon['size']
                    })
                continue
            
            # Check coverage with each person
            best_coverage = 0
            best_person_conf = 0
            
            for person in person_boxes:
                coverage_info = check_weapon_in_person_crop(
                    weapon['box'], 
                    person['box'], 
                    crop_scale
                )
                
                if coverage_info['coverage'] > best_coverage:
                    best_coverage = coverage_info['coverage']
                    best_person_conf = person['conf']
            
            # Categorize capture quality
            if best_coverage >= 0.8:  # Good capture
                stats['weapons_captured'] += 1
                stats['capture_by_class'][weapon['class']]['captured'] += 1
            elif best_coverage >= 0.3:  # Partial capture
                stats['weapons_partial'] += 1
                stats['capture_by_class'][weapon['class']]['partial'] += 1
            else:  # Missed
                stats['weapons_missed'] += 1
                if len(stats['missed_examples']) < 10:
                    stats['missed_examples'].append({
                        'image': img_path.name,
                        'class': weapon['class'],
                        'best_coverage': best_coverage,
                        'num_persons': len(person_boxes),
                        'weapon_size': weapon['size']
                    })
    
    return stats

def print_analysis(stats):
    """Print analysis results in a clear format"""
    print("\n" + "="*60)
    print("📊 STAGE 2 WEAPON CAPTURE ANALYSIS RESULTS")
    print("="*60)
    
    if stats['total_weapons'] == 0:
        print("❌ No weapons found in dataset labels!")
        return
    
    # Overall metrics
    capture_rate = stats['weapons_captured'] / stats['total_weapons'] * 100
    partial_rate = stats['weapons_partial'] / stats['total_weapons'] * 100
    missed_rate = stats['weapons_missed'] / stats['total_weapons'] * 100
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   Total weapons: {stats['total_weapons']}")
    print(f"   ✅ Captured: {stats['weapons_captured']} ({capture_rate:.1f}%)")
    print(f"   ⚠️  Partial: {stats['weapons_partial']} ({partial_rate:.1f}%)")
    print(f"   ❌ Missed: {stats['weapons_missed']} ({missed_rate:.1f}%)")
    
    if stats['avg_persons_per_image']:
        avg_persons = np.mean(stats['avg_persons_per_image'])
        print(f"   👥 Avg persons/image: {avg_persons:.1f}")
    
    # Per-class breakdown
    print(f"\n📊 BY WEAPON CLASS:")
    for class_name, class_stats in stats['capture_by_class'].items():
        if class_stats['total'] > 0:
            class_capture = class_stats['captured'] / class_stats['total'] * 100
            class_partial = class_stats['partial'] / class_stats['total'] * 100
            print(f"   {class_name}:")
            print(f"      Captured: {class_stats['captured']}/{class_stats['total']} ({class_capture:.1f}%)")
            print(f"      Partial: {class_stats['partial']} ({class_partial:.1f}%)")
    
    # Missed examples
    if stats['missed_examples']:
        print(f"\n❌ EXAMPLE MISSED WEAPONS:")
        for ex in stats['missed_examples'][:5]:
            print(f"   • {ex['image']}: {ex['class']}")
            if 'reason' in ex:
                print(f"     Reason: {ex['reason']}")
            else:
                print(f"     Coverage: {ex.get('best_coverage', 0):.1%}, Persons detected: {ex.get('num_persons', 0)}")
    
    # Performance assessment
    print(f"\n" + "="*60)
    if capture_rate >= 99:
        print("✅ EXCELLENT! Stage 2 captures 99%+ of weapons!")
    elif capture_rate >= 95:
        print("⚠️  GOOD: Stage 2 captures 95%+ but could be improved")
        print_suggestions(stats)
    else:
        print("❌ NEEDS IMPROVEMENT: Stage 2 misses too many weapons")
        print_suggestions(stats)

def print_suggestions(stats):
    """Print improvement suggestions"""
    print("\n💡 IMPROVEMENT SUGGESTIONS:")
    
    capture_rate = stats['weapons_captured'] / stats['total_weapons'] * 100
    
    if capture_rate < 99:
        suggestions = []
        
        # Check if no persons detected is an issue
        if stats['no_person_detected'] > stats['total_weapons'] * 0.05:
            suggestions.append("1. LOWER confidence threshold to 0.10-0.15 (many weapons with no person detected)")
        
        # Check if partial captures are high
        if stats['weapons_partial'] > stats['weapons_missed']:
            suggestions.append("2. INCREASE crop_scale to 2.5-3.0 (many partial captures)")
        
        # Check class-specific issues
        worst_class = None
        worst_rate = 100
        for class_name, class_stats in stats['capture_by_class'].items():
            if class_stats['total'] > 0:
                class_rate = class_stats['captured'] / class_stats['total'] * 100
                if class_rate < worst_rate:
                    worst_rate = class_rate
                    worst_class = class_name
        
        if worst_class and worst_rate < 90:
            suggestions.append(f"3. {worst_class.upper()} needs attention (only {worst_rate:.0f}% captured)")
        
        suggestions.append("4. REDUCE crop_overlap_threshold to 0.4-0.5")
        suggestions.append("5. Consider multi-scale detection or ensemble methods")
        
        for s in suggestions:
            print(f"   • {s}")

def main():
    """Main function"""
    config_path = "config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("   Please run from your weapon_detection_pipeline directory")
        return
    
    print("\n🔍 STAGE 2 WEAPON CAPTURE ANALYZER")
    print("   This tool checks how many weapons are captured by person detection")
    
    try:
        # Run analysis
        stats = analyze_coverage_simple(config_path, sample_size=50)
        
        # Print results
        print_analysis(stats)
        
        # Save report
        report_path = "stage2_coverage_report.json"
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"\n📄 Detailed report saved to: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()