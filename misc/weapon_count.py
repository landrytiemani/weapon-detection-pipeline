#!/usr/bin/env python3
"""
Quick script to count total weapons in the dataset
"""

import os
import yaml
from pathlib import Path
from collections import defaultdict

def count_weapons_in_dataset(config_path):
    """Count all weapons in the dataset"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths
    labels_dir = Path(config['pipeline']['labels_dir'])
    class_names = config['stage_3'].get('names', ['handgun', 'knife'])
    
    # Count weapons
    total_weapons = 0
    total_files = 0
    files_with_weapons = 0
    weapons_per_class = defaultdict(int)
    weapons_per_file = []
    
    # Get all label files
    label_files = sorted(labels_dir.glob('*.txt'))
    
    print(f"\n📁 Scanning {len(label_files)} label files in: {labels_dir}")
    
    for label_path in label_files:
        total_files += 1
        file_weapon_count = 0
        
        if os.path.getsize(label_path) > 0:  # Skip empty files
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Valid YOLO format line
                        class_id = int(parts[0])
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                            weapons_per_class[class_name] += 1
                        else:
                            weapons_per_class[f'unknown_class_{class_id}'] += 1
                        
                        total_weapons += 1
                        file_weapon_count += 1
        
        if file_weapon_count > 0:
            files_with_weapons += 1
            weapons_per_file.append(file_weapon_count)
    
    # Calculate statistics
    print("\n" + "="*60)
    print("📊 DATASET WEAPON COUNT SUMMARY")
    print("="*60)
    
    print(f"\n📈 TOTALS:")
    print(f"   Total weapon instances: {total_weapons}")
    print(f"   Total label files: {total_files}")
    print(f"   Files with weapons: {files_with_weapons}")
    print(f"   Files without weapons: {total_files - files_with_weapons}")
    
    print(f"\n📊 BY CLASS:")
    for class_name, count in sorted(weapons_per_class.items()):
        percentage = (count / total_weapons * 100) if total_weapons > 0 else 0
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    if weapons_per_file:
        import numpy as np
        print(f"\n📊 DISTRIBUTION:")
        print(f"   Avg weapons per file (with weapons): {np.mean(weapons_per_file):.1f}")
        print(f"   Max weapons in a file: {np.max(weapons_per_file)}")
        print(f"   Min weapons in a file: {np.min(weapons_per_file)}")
    
    print("\n" + "="*60)
    print(f"💡 KEY FINDING: Your dataset contains {total_weapons} weapon instances")
    print(f"   The Stage 2 analyzer should check all {total_weapons} weapons")
    print("="*60)
    
    return total_weapons

if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        print("   Usage: python count_weapons.py [config.yaml]")
    else:
        total = count_weapons_in_dataset(config_path)