#!/usr/bin/env python3
"""
RQ1: Modular Architecture Ablation Study
=========================================
Evaluates the contribution of each pipeline component to detection accuracy.

Hypotheses tested:
  H1.1 - Person-centric cropping improves mAP50 by >= 5 percentage points
  H1.2 - Hierarchical NMS reduces false positives by >= 20%
  H1.3 - Optimal crop scale falls within [1.0, 1.5] range

"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path for imports (main_perclass.py is in parent dir)
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent  # /home/ubuntu/weapon_detection_pipeline
sys.path.insert(0, str(PIPELINE_DIR))

# Default paths relative to pipeline directory
DEFAULT_CONFIG = PIPELINE_DIR / "config.yaml"
DEFAULT_OUTPUT = PIPELINE_DIR / "Results" / "rq1_ablation"

from main_perclass import SingleExperiment

# =============================================================================
# BENCHMARK GFLOPs VALUES 
# =============================================================================
# Official values at 640px input resolution
# Source: https://github.com/ultralytics/ultralytics (YOLOv8)
#         https://github.com/mit-han-lab/efficientvit (EfficientViT)

GFLOPS_BENCHMARKS = {
    # Person Detection (Stage-2)
    'yolov8n': 8.7,      # YOLOv8n official: 8.7 GFLOPs @ 640px
    'yolov8s': 28.6,     # YOLOv8s
    'yolov8m': 78.9,     # YOLOv8m
    
    # Weapon Detection (Stage-3)
    'efficientvit_yolov8': 6.2,   # EfficientViT-M0 backbone + YOLOv8 head
    'yolov8_efficientvit': 6.2,   # Same model, different naming
    'rt_detr': 81.4,              # RT-DETR-L official
    'rt_detr_l': 110.0,           # RT-DETR-L full
    'rt_detr_x': 232.0,           # RT-DETR-X
    
    # Privacy overhead (negligible)
    'face_blur_pixelate': 0.001,
    'face_blur_gaussian': 0.002,
}

STAGE2_INPUT_SIZE = 800  # Person detection input resolution
STAGE3_INPUT_SIZE = 640  # Weapon detection input resolution


def scale_gflops(base_gflops: float, base_size: int, actual_size: int) -> float:
    """Scale GFLOPs for different input resolutions (quadratic scaling)."""
    scale_factor = (actual_size / base_size) ** 2
    return base_gflops * scale_factor


def compute_pipeline_gflops(stage2_model: str = 'yolov8n',
                            stage3_model: str = 'efficientvit_yolov8',
                            avg_crops_per_frame: float = 3.0,
                            skip_person_detection: bool = False) -> dict:
    """
    Compute total GFLOPs for the pipeline using benchmark values.
    
    This replaces thop-based computation which has hook conflicts.
    Values are from official model documentation.
    """
    # Stage-2: Person detection (0 if skipped for ablation)
    if skip_person_detection:
        stage2_gflops_scaled = 0.0
    else:
        stage2_gflops = GFLOPS_BENCHMARKS.get(stage2_model, 8.7)
        stage2_gflops_scaled = scale_gflops(stage2_gflops, 640, STAGE2_INPUT_SIZE)
    
    # Stage-3: Weapon detection (runs per crop, or once on full frame)
    stage3_per_crop = GFLOPS_BENCHMARKS.get(stage3_model, 6.2)
    stage3_total = stage3_per_crop * avg_crops_per_frame
    
    # Total per frame
    total_per_frame = stage2_gflops_scaled + stage3_total
    
    return {
        'stage2_gflops': round(stage2_gflops_scaled, 2),
        'stage3_per_crop_gflops': round(stage3_per_crop, 2),
        'stage3_total_gflops': round(stage3_total, 2),
        'avg_crops_per_frame': avg_crops_per_frame,
        'total_gflops_per_frame': round(total_per_frame, 2)
    }


def print_gflops_summary():
    """Print GFLOPs summary at experiment start."""
    evit = compute_pipeline_gflops('yolov8n', 'efficientvit_yolov8', 3.0)
    rtdetr = compute_pipeline_gflops('yolov8n', 'rt_detr', 3.0)
    
    print("\n[GFLOPs] Benchmark Values (from compute_flops.py):")
    print(f"  YOLOv8n (Stage-2 @ 800px):      {evit['stage2_gflops']:.2f} GFLOPs")
    print(f"  EfficientViT (Stage-3 @ 640px): {evit['stage3_per_crop_gflops']:.2f} GFLOPs/crop")
    print(f"  RT-DETR (Stage-3 @ 640px):      {rtdetr['stage3_per_crop_gflops']:.2f} GFLOPs/crop")


@dataclass
class ExperimentResult:
    """Stores metrics for a single ablation configuration."""
    config_name: str
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float
    tp_count: int
    fp_count: int
    fn_count: int
    gflops: float
    fps: float
    latency_ms: float
    handgun_mAP50: float = 0.0
    knife_mAP50: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class RQ1AblationExperiment:
    """
    RQ1: Ablation Study for Modular Architecture
    
    Tests three hypotheses about the pipeline design:
      H1.1 - Person-centric cropping provides >= 5pp mAP50 improvement
             (Dissertation result: +21.4pp, 46% relative improvement)
      H1.2 - Hierarchical NMS reduces FP by >= 20%
             (Dissertation result: 70.5% reduction)
      H1.3 - Optimal crop scale in [1.0, 1.5]
             (Dissertation result: 1.8 optimal, hypothesis NOT supported)
    
    The ablation isolates each component by comparing against configurations
    where that component is disabled or modified.
    """
    
    # Crop scales tested per methodology section 3.3.1
    CROP_SCALES = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    
    def __init__(self, config_path: str, output_dir: str = 'Results/rq1_ablation'):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        self.results: Dict[str, ExperimentResult] = {}
        
        if torch.cuda.is_available():
            print(f"[RQ1] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[RQ1] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def _save_config(self, config: dict, name: str) -> str:
        """Write modified config to temp file for experiment run."""
        config_dir = self.output_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        
        temp_path = config_dir / f'{name}_config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return str(temp_path)
    
    def _prepare_config(self, config: dict) -> dict:
        """
        Set safe defaults to prevent runtime errors.
        Key issue: frame_gap=0 causes division by zero in throughput calc.
        """
        if 'stage_2' not in config:
            config['stage_2'] = {}
        
        # Frame gap must be >= 1 to avoid ZeroDivisionError
        frame_gap = config['stage_2'].get('frame_gap', 1)
        config['stage_2']['frame_gap'] = max(1, int(frame_gap))
        
        if 'evaluation' not in config:
            config['evaluation'] = {}
        
        # Disable thop FLOPs - causes hook conflicts with some models
        config['evaluation']['compute_flops'] = False
        
        return config
    
    def _run_single_experiment(self, config: dict, name: str) -> ExperimentResult:
        """Execute one ablation configuration using SingleExperiment."""
        print(f"\n{'='*60}")
        print(f"[RQ1] Running: {name}")
        print(f"{'='*60}")
        
        config = self._prepare_config(config)
        temp_config_path = self._save_config(config, name)
        
        use_tracker = config.get('stage_2', {}).get('use_tracker', False)
        frame_gap = config.get('stage_2', {}).get('frame_gap', 1)
        skip_person = config.get('stage_2', {}).get('skip_person_detection', False)
        
        exp_output_dir = str(self.output_dir / 'runs' / name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        exp = SingleExperiment(
            config_path=temp_config_path,
            use_tracker=use_tracker,
            frame_gap=frame_gap,
            experiment_name=name,
            output_dir=exp_output_dir
        )
        
        results = exp.run()
        
        # Extract per-class metrics for knife vs handgun analysis
        per_class_map = results.get('pipeline_map50_by_class', {})
        handgun_map = per_class_map.get('handgun', 0.0)
        knife_map = per_class_map.get('knife', 0.0)
        
        # Compute GFLOPs using benchmark values (thop disabled due to conflicts)
        avg_crops = results.get('avg_crops_per_frame', 1.0)
        if skip_person:
            avg_crops = 1.0  # Full frame = 1 "crop"
        
        stage3_approach = config.get('stage_3', {}).get('approach', 'yolov8_efficientvit')
        gflops_info = compute_pipeline_gflops(
            stage2_model='yolov8n',
            stage3_model=stage3_approach,
            avg_crops_per_frame=avg_crops,
            skip_person_detection=skip_person
        )
        computed_gflops = gflops_info['total_gflops_per_frame']
        
        # Print GFLOPs breakdown
        print(f"\n[GFLOPs] {name}:")
        print(f"  Stage-2 (YOLOv8n):     {gflops_info['stage2_gflops']:.2f} GFLOPs")
        print(f"  Stage-3 per crop:      {gflops_info['stage3_per_crop_gflops']:.2f} GFLOPs")
        print(f"  Stage-3 total ({avg_crops:.1f}x):  {gflops_info['stage3_total_gflops']:.2f} GFLOPs")
        print(f"  TOTAL per frame:       {computed_gflops:.2f} GFLOPs")
        
        return ExperimentResult(
            config_name=name,
            mAP50=results.get('pipeline_map50', 0.0),
            mAP50_95=results.get('pipeline_map50_95', 0.0),
            precision=results.get('pipeline_precision', 0.0),
            recall=results.get('pipeline_recall', 0.0),
            f1=results.get('pipeline_f1', 0.0),
            tp_count=results.get('pipeline_tp', 0),
            fp_count=results.get('pipeline_fp', 0),
            fn_count=results.get('pipeline_fn', 0),
            gflops=computed_gflops,
            fps=results.get('fps', 0.0),
            latency_ms=results.get('latency_ms', 0.0),
            handgun_mAP50=handgun_map,
            knife_mAP50=knife_map
        )
    
    def run_baseline(self) -> ExperimentResult:
        """
        Baseline: Full pipeline with crop_scale=1.8, hierarchical NMS enabled.
        This is our reference point for all ablation comparisons.
        """
        print("\n[RQ1] Running BASELINE experiment...")
        
        config = deepcopy(self.base_config)
        result = self._run_single_experiment(config, 'baseline')
        self.results['baseline'] = result
        
        self._save_results()
        return result
    
    def run_no_person_detection(self) -> ExperimentResult:
        """
        H1.1 ablation: Bypass person detection entirely.
        """
        print("\n[RQ1/H1.1] Running NO PERSON DETECTION experiment...")
        
        config = deepcopy(self.base_config)
        config['stage_2'] = config.get('stage_2', {})
        config['stage_2']['skip_person_detection'] = True  # Key flag for ablation
        
        result = self._run_single_experiment(config, 'no_person_detection')
        self.results['no_person_detection'] = result
        
        self._save_results()
        return result
    
    def run_crop_scale_sweep(self) -> Dict[str, ExperimentResult]:
        """
        H1.3: Sweep crop scales to find optimal context window.
        
        Tests scales from 1.0 (tight) to 3.0 (wide context).
        Hypothesis predicted [1.0, 1.5] optimal; actual result was 1.8.
        """
        print("\n[RQ1/H1.3] Running CROP SCALE SWEEP...")
        
        results = {}
        
        for scale in self.CROP_SCALES:
            name = f"crop_scale_{scale}"
            print(f"\n[H1.3] Testing crop_scale = {scale}")
            
            config = deepcopy(self.base_config)
            config['stage_2'] = config.get('stage_2', {})
            config['stage_2']['crop_scale'] = scale
            
            result = self._run_single_experiment(config, name)
            results[name] = result
            self.results[name] = result
            
            self._save_results()
            torch.cuda.empty_cache()
        
        return results
    
    def run_no_nms(self) -> ExperimentResult:
        """
        H1.2 ablation: Disable hierarchical NMS.
        
        Sets NMS thresholds to 1.0 (no suppression), which should cause
        massive FP increase due to duplicate detections from overlapping crops.
        """
        print("\n[RQ1/H1.2] Running NO NMS experiment...")
        
        config = deepcopy(self.base_config)
        config['stage_3'] = config.get('stage_3', {})
        config['stage_3']['nms_iou_threshold'] = 1.0      # Local NMS off
        config['stage_3']['global_nms_threshold'] = 1.0   # Global NMS off
        
        result = self._run_single_experiment(config, 'no_nms')
        self.results['no_nms'] = result
        
        self._save_results()
        return result
    
    def run_full_ablation(self) -> Dict[str, ExperimentResult]:
        """Run complete ablation study for all three hypotheses."""
        print("\n" + "=" * 70)
        print("RQ1: FULL ABLATION STUDY")
        print("=" * 70)
        
        self.run_baseline()
        torch.cuda.empty_cache()
        
        self.run_no_person_detection()
        torch.cuda.empty_cache()
        
        self.run_no_nms()
        torch.cuda.empty_cache()
        
        self.run_crop_scale_sweep()
        
        self._save_results()
        return self.results
    
    def validate_hypotheses(self) -> Dict[str, Dict]:
        """
        Evaluate H1.1, H1.2, H1.3 against dissertation thresholds.
        
        Thresholds from Chapter 3:
          H1.1: >= 5 percentage points improvement
          H1.2: >= 20% FP reduction 
          H1.3: Optimal in [1.0, 1.5] 
        """
        validation = {}
        
        baseline = self.results.get('baseline')
        no_person = self.results.get('no_person_detection')
        no_nms = self.results.get('no_nms')
        
        # H1.1: Person-centric cropping improvement
        # Threshold: >= 5 percentage points (0.05 absolute mAP50 difference)
        if baseline and no_person:
            # Calculate absolute improvement in percentage points
            improvement_pp = (baseline.mAP50 - no_person.mAP50) * 100
            
            # Also calculate relative improvement for reference
            if no_person.mAP50 > 0:
                relative_improvement = ((baseline.mAP50 - no_person.mAP50) / no_person.mAP50) * 100
            else:
                relative_improvement = 100.0 if baseline.mAP50 > 0 else 0.0
            
            validation['H1.1'] = {
                'description': 'Person-centric cropping improves mAP50 by >= 5 percentage points',
                'baseline_mAP50': baseline.mAP50,
                'no_person_detection_mAP50': no_person.mAP50,
                'improvement_pp': improvement_pp,
                'relative_improvement_percent': relative_improvement,
                'threshold_pp': 5.0,
                'hypothesis_supported': improvement_pp >= 5.0
            }
        
        # H1.2: Hierarchical NMS FP reduction
        # Threshold: >= 20% reduction in false positives
        if baseline and no_nms:
            if no_nms.fp_count > 0:
                fp_reduction = ((no_nms.fp_count - baseline.fp_count) / no_nms.fp_count) * 100
            else:
                fp_reduction = 0.0
            
            validation['H1.2'] = {
                'description': 'Hierarchical NMS reduces FP by >= 20%',
                'baseline_fp': baseline.fp_count,
                'no_nms_fp': no_nms.fp_count,
                'reduction_percent': fp_reduction,
                'threshold_percent': 20.0,
                'hypothesis_supported': fp_reduction >= 20.0
            }
        
        # H1.3: Optimal crop scale
        # Hypothesis: optimal falls in [1.0, 1.5]
        # Dissertation finding: 1.8 was optimal (hypothesis NOT supported)
        scale_results = {k: v for k, v in self.results.items() if k.startswith('crop_scale')}
        if scale_results:
            best_name = max(scale_results, key=lambda k: scale_results[k].mAP50)
            best_scale = float(best_name.split('_')[-1])
            
            validation['H1.3'] = {
                'description': 'Optimal crop_scale falls within [1.0, 1.5]',
                'optimal_scale': best_scale,
                'optimal_mAP50': scale_results[best_name].mAP50,
                'all_scales': {k: v.mAP50 for k, v in scale_results.items()},
                'expected_range': [1.0, 1.5],
                'hypothesis_supported': 1.0 <= best_scale <= 1.5
            }
        
        return validation
    
    def _save_results(self):
        """Persist results to JSON for later analysis."""
        output_path = self.output_dir / 'ablation_results.json'
        serializable = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[RQ1] Results saved: {output_path}")
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for dissertation Chapter 4."""
        baseline = self.results.get('baseline')
        
        latex = r"""
\begin{table}[htbp]
\centering
\caption{RQ1: Ablation Study Results}
\label{tab:rq1_ablation}
\begin{tabular}{lcccccc}
\toprule
Configuration & mAP50 & Precision & Recall & F1 & FP & $\Delta$mAP \\
\midrule
"""
        for name, result in self.results.items():
            if name.startswith('crop_scale'):
                continue
            
            delta = result.mAP50 - baseline.mAP50 if baseline else 0
            display_name = name.replace('_', ' ').title()
            
            latex += f"{display_name} & {result.mAP50:.3f} & {result.precision:.3f} & "
            latex += f"{result.recall:.3f} & {result.f1:.3f} & {result.fp_count} & "
            latex += f"{delta:+.3f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.output_dir / 'rq1_tables.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[RQ1] LaTeX saved: {output_path}")
        return latex
    
    def generate_figures(self):
        """Generate visualization for H1.3 crop scale sensitivity."""
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        scale_results = {k: v for k, v in self.results.items() if k.startswith('crop_scale')}
        if scale_results:
            scales = []
            mAP50s = []
            
            for name in sorted(scale_results.keys(), key=lambda x: float(x.split('_')[-1])):
                scale = float(name.split('_')[-1])
                scales.append(scale)
                mAP50s.append(scale_results[name].mAP50)
            
            plt.figure(figsize=(8, 5))
            plt.plot(scales, mAP50s, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('Crop Scale', fontsize=12)
            plt.ylabel('mAP50', fontsize=12)
            plt.title('H1.3: Crop Scale Sensitivity Analysis', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # Mark optimal point
            best_idx = np.argmax(mAP50s)
            plt.scatter([scales[best_idx]], [mAP50s[best_idx]], 
                       c='red', s=150, zorder=5, label=f'Optimal: {scales[best_idx]}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(figures_dir / 'crop_scale_sensitivity.png', dpi=300)
            plt.close()
            print(f"[RQ1] Figure saved: {figures_dir / 'crop_scale_sensitivity.png'}")


def main():
    parser = argparse.ArgumentParser(description='RQ1: Ablation Experiment')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG),
                        help='Path to config.yaml')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output directory for results')
    parser.add_argument('--experiment', type=str, default='full',
                        choices=['full', 'baseline', 'crop_scale', 'no_person', 'no_nms'])
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ1: MODULAR ARCHITECTURE ABLATION STUDY")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Experiment: {args.experiment}")
    
    # Print GFLOPs benchmark summary
    print_gflops_summary()
    
    # Verify config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        print(f"[INFO] Looking for config at: {DEFAULT_CONFIG}")
        return
    
    # Load and display key paths from config
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    frames_dir = cfg.get('pipeline', {}).get('frames_dir', 'NOT SET')
    labels_dir = cfg.get('pipeline', {}).get('labels_dir', 'NOT SET')
    print(f"\n[CONFIG] Dataset paths:")
    print(f"  frames_dir: {frames_dir}")
    print(f"  labels_dir: {labels_dir}")
    
    # Check if paths exist
    if not Path(frames_dir).exists():
        print(f"[WARNING] frames_dir does not exist: {frames_dir}")
    else:
        import glob
        frame_count = len(glob.glob(str(Path(frames_dir) / '*')))
        print(f"  Found {frame_count} files in frames_dir")
    
    experiment = RQ1AblationExperiment(config_path=args.config, output_dir=args.output)
    
    if args.experiment == 'full':
        experiment.run_full_ablation()
    elif args.experiment == 'baseline':
        experiment.run_baseline()
    elif args.experiment == 'crop_scale':
        experiment.run_baseline()
        experiment.run_crop_scale_sweep()
    elif args.experiment == 'no_person':
        experiment.run_baseline()
        experiment.run_no_person_detection()
    elif args.experiment == 'no_nms':
        experiment.run_baseline()
        experiment.run_no_nms()
    
    validation = experiment.validate_hypotheses()
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)
    
    for h_id, h_result in validation.items():
        status = "SUPPORTED" if h_result.get('hypothesis_supported') else "NOT SUPPORTED"
        print(f"\n{h_id}: {status}")
        print(f"  {h_result.get('description', '')}")
        for key, value in h_result.items():
            if key not in ['description', 'hypothesis_supported', 'all_scales']:
                print(f"    {key}: {value}")
    
    experiment.generate_latex_table()
    experiment.generate_figures()
    
    validation_path = Path(args.output) / 'hypothesis_validation.json'
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
