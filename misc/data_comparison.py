import os
from collections import Counter
from pathlib import Path

# Define paths
base_dir = Path("dataset_v2/weaponsense_split")
sets = ["train", "val", "test"]
variants = ["", "cropped"]  # normal and cropped

def count_objects(labels_dir):
    """Count total objects per class from YOLO label files."""
    counter = Counter()
    for label_file in Path(labels_dir).glob("*.txt"):
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    cls_id = parts[0]
                    counter[cls_id] += 1
    return counter

def gather_counts(variant):
    """Gather class counts for train/val/test of a given variant."""
    results = {}
    for s in sets:
        labels_path = base_dir / variant / s / "labels" if variant else base_dir / s / "labels"
        if labels_path.exists():
            results[s] = count_objects(labels_path)
        else:
            results[s] = {}
    return results

def print_comparison(normal_counts, cropped_counts):
    """Pretty print side-by-side comparison."""
    print(f"{'Split':<10} | {'Normal Dataset':<30} | {'Cropped Dataset'}")
    print("-"*70)
    all_classes = set()
    for s in sets:
        all_classes.update(normal_counts[s].keys())
        all_classes.update(cropped_counts[s].keys())

    for s in sets:
        print(f"\n== {s.upper()} ==")
        for cls in sorted(all_classes, key=lambda x: int(x)):
            n = normal_counts[s].get(cls, 0)
            c = cropped_counts[s].get(cls, 0)
            diff = c - n
            print(f"Class {cls:<3}: Normal={n:<6} | Cropped={c:<6} | Δ={diff:+}")

if __name__ == "__main__":
    normal_counts = gather_counts("")
    cropped_counts = gather_counts("cropped")

    print("\n📊 OBJECT COUNT COMPARISON (per class)")
    print_comparison(normal_counts, cropped_counts)
