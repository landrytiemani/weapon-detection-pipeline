"""
Script to analyze Stage 2 person detection coverage for weapon detection pipeline.
Checks how many ground truth weapons are captured within detected person bounding boxes.
"""

import json
import os
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
    Args:
        weapon_box: [x1, y1, x2, y2] normalized
        person_box: [x1, y1, x2, y2] normalized
        crop_scale: scale factor applied to person box
    Returns:
        dict with coverage metrics
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
    
    # Calculate IoU
    iou = calculate_iou(weapon_box, expanded_person)
    
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
        'iou': iou,
        'coverage': coverage,
        'fully_inside': coverage >= 0.99
    }

def analyze_stage2_coverage(config_path, sample_size=None, verbose=True):
    """
    Analyze how well Stage 2 captures weapons
    """
    config = load_config(config_path)
    
    # Import the person detection stage
    import sys
    sys.path.append(os.path.dirname(config_path))
    from stages.stage_2_persondetection import PersonDetectionStage
    
    # Initialize Stage 2
    stage2_config = config['stage_2']
    stage2 = PersonDetectionStage(stage2_config)
    
    # Get dataset paths
    frames_dir = Path(config['pipeline']['frames_dir'])
    labels_dir = Path(config['pipeline']['labels_dir'])
    
    # Get image files
    image_files = sorted(list(frames_dir.glob('*.jpg')) + list(frames_dir.glob('*.png')))
    if sample_size:
        image_files = image_files[:sample_size]
    
    # Statistics
    stats = {
        'total_weapons': 0,
        'weapons_captured': 0,
        'weapons_missed': 0,
        'weapons_partial': 0,
        'capture_by_class': defaultdict(lambda: {'total': 0, 'captured': 0}),
        'missed_weapons_info': [],
        'capture_rates_per_image': [],
        'person_conf_for_missed': [],
        'weapon_sizes_missed': []
    }
    
    crop_scale = stage2_config.get('crop_scale', 1.5)
    
    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Get ground truth weapons
        label_path = labels_dir / (img_path.stem + '.txt')
        gt_weapons = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = cx - bw/2
                        y1 = cy - bh/2
                        x2 = cx + bw/2
                        y2 = cy + bh/2
                        
                        # Class names from config
                        class_names = config['stage_3'].get('names', ['handgun', 'knife'])
                        class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                        
                        gt_weapons.append({
                            'box': [x1, y1, x2, y2],
                            'class': class_name,
                            'size': bw * bh  # normalized area
                        })
        
        # Run Stage 2 detection
        _, person_data, _ = stage2.run(img, frame_idx=0)
        
        # Convert person boxes to normalized coordinates
        person_boxes = []
        for person in person_data:
            x1, y1, x2, y2 = person['bbox']
            person_boxes.append({
                'box': [x1/w, y1/h, x2/w, y2/h],
                'conf': person['conf']
            })
        
        # Check each weapon
        image_captured = 0
        image_total = len(gt_weapons)
        
        for weapon in gt_weapons:
            stats['total_weapons'] += 1
            stats['capture_by_class'][weapon['class']]['total'] += 1
            
            captured = False
            best_coverage = 0
            best_person_conf = 0
            
            # Check against all detected persons
            for person in person_boxes:
                coverage_info = check_weapon_in_person_crop(
                    weapon['box'], 
                    person['box'], 
                    crop_scale
                )
                
                if coverage_info['coverage'] > best_coverage:
                    best_coverage = coverage_info['coverage']
                    best_person_conf = person['conf']
                
                # Consider captured if coverage > 0.8
                if coverage_info['coverage'] >= 0.8:
                    captured = True
                    break
            
            if captured:
                stats['weapons_captured'] += 1
                stats['capture_by_class'][weapon['class']]['captured'] += 1
                image_captured += 1
            elif best_coverage > 0.3:  # Partially captured
                stats['weapons_partial'] += 1
                if verbose:
                    print(f"  Partial capture ({best_coverage:.1%}) for {weapon['class']} in {img_path.name}")
            else:
                stats['weapons_missed'] += 1
                stats['person_conf_for_missed'].append(best_person_conf)
                stats['weapon_sizes_missed'].append(weapon['size'])
                
                if verbose and len(stats['missed_weapons_info']) < 10:  # Limit output
                    stats['missed_weapons_info'].append({
                        'image': img_path.name,
                        'class': weapon['class'],
                        'weapon_box': weapon['box'],
                        'best_coverage': best_coverage,
                        'num_persons_detected': len(person_boxes)
                    })
        
        if image_total > 0:
            stats['capture_rates_per_image'].append(image_captured / image_total)
    
    return stats

def suggest_improvements(stats, config):
    """Suggest parameter improvements based on analysis"""
    suggestions = []
    
    capture_rate = stats['weapons_captured'] / max(1, stats['total_weapons'])
    
    if capture_rate < 0.99:
        suggestions.append("\n=== IMPROVEMENT SUGGESTIONS ===\n")
        
        # 1. Check if confidence threshold is too high
        if stats['person_conf_for_missed'] and np.mean(stats['person_conf_for_missed']) < 0.1:
            suggestions.append("1. LOWER person detection confidence threshold:")
            suggestions.append("   - Current: 0.35 → Suggested: 0.15-0.20")
            suggestions.append("   - Many weapons are in areas with no detected persons")
        
        # 2. Check if crop scale needs adjustment
        if stats['weapons_partial'] > stats['weapons_missed']:
            suggestions.append("2. INCREASE crop_scale:")
            suggestions.append("   - Current: 2.0 → Suggested: 2.5-3.0")
            suggestions.append("   - Many weapons are partially captured")
        
        # 3. Check weapon sizes
        if stats['weapon_sizes_missed']:
            avg_missed_size = np.mean(stats['weapon_sizes_missed'])
            if avg_missed_size < 0.01:  # Very small weapons
                suggestions.append("3. Weapons missed are VERY SMALL:")
                suggestions.append("   - Consider using higher resolution input")
                suggestions.append("   - Or use a more sensitive person detector")
        
        # 4. Class-specific issues
        for class_name, class_stats in stats['capture_by_class'].items():
            class_rate = class_stats['captured'] / max(1, class_stats['total'])
            if class_rate < 0.95:
                suggestions.append(f"4. {class_name.upper()} specific issue:")
                suggestions.append(f"   - Capture rate: {class_rate:.1%}")
                suggestions.append(f"   - May need class-specific tuning")
        
        # 5. NMS/overlap settings
        suggestions.append("5. REDUCE crop_overlap_threshold:")
        suggestions.append("   - Current: 0.8 → Suggested: 0.5-0.6")
        suggestions.append("   - May be filtering out valid person detections")
        
        # 6. Try alternative detector
        current_approach = config['stage_2']['approach']
        alt_approach = 'ssd_mobilenet_bytetrack' if current_approach == 'yolov8_tracker' else 'yolov8_tracker'
        suggestions.append(f"6. Try alternative detector: {alt_approach}")
    
    return suggestions

def print_results(stats, suggestions):
    """Print analysis results"""
    print("\n" + "="*60)
    print("STAGE 2 WEAPON CAPTURE ANALYSIS")
    print("="*60)
    
    capture_rate = stats['weapons_captured'] / max(1, stats['total_weapons'])
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total weapons in dataset: {stats['total_weapons']}")
    print(f"  Weapons captured: {stats['weapons_captured']} ({capture_rate:.1%})")
    print(f"  Weapons partially captured: {stats['weapons_partial']}")
    print(f"  Weapons missed: {stats['weapons_missed']}")
    
    print(f"\nBY CLASS:")
    for class_name, class_stats in stats['capture_by_class'].items():
        class_rate = class_stats['captured'] / max(1, class_stats['total'])
        print(f"  {class_name}: {class_stats['captured']}/{class_stats['total']} ({class_rate:.1%})")
    
    if stats['missed_weapons_info']:
        print(f"\nEXAMPLE MISSED WEAPONS:")
        for info in stats['missed_weapons_info'][:5]:
            print(f"  - {info['image']}: {info['class']} (coverage={info['best_coverage']:.1%}, persons_detected={info['num_persons_detected']})")
    
    if capture_rate >= 0.99:
        print(f"\n✅ EXCELLENT! Stage 2 captures {capture_rate:.1%} of weapons!")
    else:
        print(f"\n⚠️  Stage 2 only captures {capture_rate:.1%} of weapons")
        for suggestion in suggestions:
            print(suggestion)

if __name__ == "__main__":
    # Analyze Stage 2 coverage
    config_path = "config.yaml"
    
    print("Analyzing Stage 2 person detection coverage...")
    stats = analyze_stage2_coverage(config_path, sample_size=50, verbose=False)
    
    # Get improvement suggestions
    config = load_config(config_path)
    suggestions = suggest_improvements(stats, config)
    
    # Print results
    print_results(stats, suggestions)
    
    # Save detailed report
    report_path = "stage2_coverage_report.json"
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_path}")