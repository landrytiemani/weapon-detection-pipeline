import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Create figure with larger size for clarity
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Two-Stage Weapon Detection Pipeline Architecture', 
        fontsize=18, fontweight='bold', ha='center')

# Color scheme
color_input = '#E8F4F8'
color_stage1 = '#B8E6B8'
color_stage2 = '#FFD4A3'
color_output = '#FFB6C1'
color_metrics = '#E6E6FA'

# Input Layer
input_box = FancyBboxPatch((0.5, 7.5), 1.5, 1.2, 
                           boxstyle="round,pad=0.05",
                           facecolor=color_input, edgecolor='black', linewidth=2)
ax.add_patch(input_box)
ax.text(1.25, 8.3, 'Input Frames', fontsize=11, fontweight='bold', ha='center')
ax.text(1.25, 7.9, '840 test frames', fontsize=9, ha='center')
ax.text(1.25, 7.7, '+ GT labels', fontsize=9, ha='center')

# Stage 1: Person Detection
stage1_box = FancyBboxPatch((3, 6.5), 4, 2.5,
                            boxstyle="round,pad=0.05",
                            facecolor=color_stage1, edgecolor='black', linewidth=2)
ax.add_patch(stage1_box)
ax.text(5, 8.6, 'STAGE 1: Person Detection & Tracking', fontsize=12, fontweight='bold', ha='center')

# Stage 1 Components
# Detector
detector_box = Rectangle((3.3, 7.5), 1.5, 0.8, facecolor='white', edgecolor='black')
ax.add_patch(detector_box)
ax.text(4.05, 7.9, 'Detector', fontsize=10, fontweight='bold', ha='center')
ax.text(4.05, 7.65, 'YOLOv8 or', fontsize=8, ha='center')
ax.text(4.05, 7.5, 'SSD-MobileNet', fontsize=8, ha='center')

# Tracker
tracker_box = Rectangle((5.2, 7.5), 1.5, 0.8, facecolor='white', edgecolor='black')
ax.add_patch(tracker_box)
ax.text(5.95, 7.9, 'Tracker', fontsize=10, fontweight='bold', ha='center')
ax.text(5.95, 7.65, 'ByteTrack', fontsize=8, ha='center')

# Frame Gap Logic
frame_gap_box = Rectangle((3.3, 6.7), 3.4, 0.6, facecolor='#F0F0F0', edgecolor='black', linestyle='--')
ax.add_patch(frame_gap_box)
ax.text(5, 7.0, 'Frame Gap Control', fontsize=9, fontweight='bold', ha='center')
ax.text(5, 6.8, 'Detect every k frames, track in between', fontsize=8, ha='center', style='italic')

# Cropping Module
crop_box = FancyBboxPatch((3, 4.8), 4, 1.2,
                          boxstyle="round,pad=0.05",
                          facecolor='#E0F7E0', edgecolor='black', linewidth=2)
ax.add_patch(crop_box)
ax.text(5, 5.6, 'Square Crop Generation', fontsize=11, fontweight='bold', ha='center')
ax.text(5, 5.3, 'Scale factor: 1.8×', fontsize=9, ha='center')
ax.text(5, 5.1, 'Avg: 0.53 crops/frame', fontsize=9, ha='center')
ax.text(5, 4.9, '~5929 total crops', fontsize=9, ha='center')

# Stage 2: Weapon Detection
stage2_box = FancyBboxPatch((3, 2.5), 4, 1.8,
                            boxstyle="round,pad=0.05",
                            facecolor=color_stage2, edgecolor='black', linewidth=2)
ax.add_patch(stage2_box)
ax.text(5, 3.9, 'STAGE 2: Weapon Detection', fontsize=12, fontweight='bold', ha='center')
ax.text(5, 3.5, 'RT-DETR Model', fontsize=10, ha='center')
ax.text(5, 3.2, 'Process each crop independently', fontsize=9, ha='center')
ax.text(5, 2.9, 'conf threshold: 0.25', fontsize=8, ha='center')
ax.text(5, 2.65, 'imgsz: 640×640', fontsize=8, ha='center')

# Metrics & Evaluation
metrics_box = FancyBboxPatch((7.8, 4), 2, 3.5,
                             boxstyle="round,pad=0.05",
                             facecolor=color_metrics, edgecolor='black', linewidth=2)
ax.add_patch(metrics_box)
ax.text(8.8, 7.2, 'Pipeline Metrics', fontsize=11, fontweight='bold', ha='center')

# Metrics details
metrics_text = [
    'Coverage Calculation:',
    '  GT covered/GT total',
    '',
    'Pipeline Performance:',
    '  mAP50 = coverage × S3_mAP50',
    '  Recall = coverage × S3_recall',
    '  F1 = 2PR/(P+R)',
    '',
    'Latency Analysis:',
    '  Total = S1_ms + (crops × S2_ms)',
    '  FPS = 1000/total_ms',
    '',
    'FLOPs:',
    '  Total = S1 + (crops × S2)'
]

y_pos = 6.8
for line in metrics_text:
    if line.startswith('  '):
        ax.text(8.8, y_pos, line, fontsize=8, ha='center')
    elif line == '':
        pass
    else:
        ax.text(8.8, y_pos, line, fontsize=9, ha='center', fontweight='bold')
    y_pos -= 0.22

# Output
output_box = FancyBboxPatch((3, 0.5), 4, 1.2,
                            boxstyle="round,pad=0.05",
                            facecolor=color_output, edgecolor='black', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 1.3, 'Final Output', fontsize=11, fontweight='bold', ha='center')
ax.text(5, 1.0, 'Detection Results + Metrics', fontsize=9, ha='center')
ax.text(5, 0.7, 'Reports: table_combined.txt, report.txt', fontsize=8, ha='center')

# Configuration Box
config_box = FancyBboxPatch((0.3, 4), 1.8, 2,
                            boxstyle="round,pad=0.05",
                            facecolor='#FFFACD', edgecolor='black', linewidth=1)
ax.add_patch(config_box)
ax.text(1.2, 5.7, 'Experiment Config', fontsize=10, fontweight='bold', ha='center')
ax.text(1.2, 5.4, 'Tracker: On/Off', fontsize=8, ha='center')
ax.text(1.2, 5.2, 'Frame Gap: 1-10', fontsize=8, ha='center')
ax.text(1.2, 5.0, '11 experiments total', fontsize=8, ha='center')
ax.text(1.2, 4.7, 'Models:', fontsize=9, fontweight='bold', ha='center')
ax.text(1.2, 4.5, 'S1: SSD-MobileNet', fontsize=8, ha='center')
ax.text(1.2, 4.3, 'S2: RT-DETR', fontsize=8, ha='center')

# Arrows
# Input to Stage 1
arrow1 = FancyArrowPatch((2, 8.1), (3, 8.1),
                        connectionstyle="arc3", arrowstyle='->', 
                        lw=2, color='black')
ax.add_patch(arrow1)

# Stage 1 to Cropping
arrow2 = FancyArrowPatch((5, 6.5), (5, 6.0),
                        connectionstyle="arc3", arrowstyle='->', 
                        lw=2, color='black')
ax.add_patch(arrow2)
ax.text(5.3, 6.25, 'Person BBs', fontsize=8, ha='left')

# Cropping to Stage 2
arrow3 = FancyArrowPatch((5, 4.8), (5, 4.3),
                        connectionstyle="arc3", arrowstyle='->', 
                        lw=2, color='black')
ax.add_patch(arrow3)
ax.text(5.3, 4.55, 'Cropped Images', fontsize=8, ha='left')

# Stage 2 to Output
arrow4 = FancyArrowPatch((5, 2.5), (5, 1.7),
                        connectionstyle="arc3", arrowstyle='->', 
                        lw=2, color='black')
ax.add_patch(arrow4)

# Config to Stage 1
arrow5 = FancyArrowPatch((2.1, 5), (3, 7),
                        connectionstyle="arc3,rad=0.3", arrowstyle='->', 
                        lw=1.5, color='gray', linestyle='--')
ax.add_patch(arrow5)

# Stage 1 to Metrics
arrow6 = FancyArrowPatch((6.7, 7.5), (7.8, 6.5),
                        connectionstyle="arc3,rad=-0.3", arrowstyle='->', 
                        lw=1.5, color='blue', alpha=0.7)
ax.add_patch(arrow6)

# Stage 2 to Metrics
arrow7 = FancyArrowPatch((6.7, 3.3), (7.8, 4.5),
                        connectionstyle="arc3,rad=0.3", arrowstyle='->', 
                        lw=1.5, color='orange', alpha=0.7)
ax.add_patch(arrow7)

# Add performance indicators
perf_box = FancyBboxPatch((0.3, 1.5), 1.8, 1.8,
                          boxstyle="round,pad=0.05",
                          facecolor='#F0F8FF', edgecolor='black', linewidth=1)
ax.add_patch(perf_box)
ax.text(1.2, 3.0, 'Key Performance', fontsize=10, fontweight='bold', ha='center')
ax.text(1.2, 2.7, 'Best mAP50: 0.569', fontsize=8, ha='center')
ax.text(1.2, 2.5, '@ gap=1, no track', fontsize=7, ha='center', style='italic')
ax.text(1.2, 2.2, 'Best FPS: 124.6', fontsize=8, ha='center')
ax.text(1.2, 2.0, '@ gap=10, tracker', fontsize=7, ha='center', style='italic')
ax.text(1.2, 1.7, 'Latency: 8-73ms', fontsize=8, ha='center')

# Add legend for data flow
ax.text(8.5, 1.2, 'Data Flow:', fontsize=9, fontweight='bold')
ax.plot([8.2, 8.4], [0.9, 0.9], 'k-', lw=2)
ax.text(8.6, 0.9, 'Main pipeline', fontsize=8)
ax.plot([8.2, 8.4], [0.6, 0.6], 'b-', lw=1.5, alpha=0.7)
ax.text(8.6, 0.6, 'To metrics', fontsize=8)
ax.plot([8.2, 8.4], [0.3, 0.3], 'gray', lw=1.5, linestyle='--')
ax.text(8.6, 0.3, 'Configuration', fontsize=8)

plt.title('Two-Stage Weapon Detection Pipeline\nData Flow & Processing Architecture', 
          fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("Results/plots/architecture_config.png", dpi=300, bbox_inches="tight")
plt.show()