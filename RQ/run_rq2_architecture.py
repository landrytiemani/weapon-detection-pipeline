#!/usr/bin/env python3
"""
RQ2: Architecture Comparison - RT-DETR vs YOLOv8-EfficientViT
=============================================================
Compares large transformer-based and lightweight hybrid (CNN-Transformer) weapon detection architectures
within the modular pipeline framework.

Hypotheses tested:
  H2.1 - RT-DETR achieves parity or superiority vs EfficientViT on mAP50
  H2.2 - EfficientViT achieves >= 90% accuracy at <= 50% computational cost
  H2.3 - RT-DETR shows advantage on knife detection (small objects)
  H2.4 - Both architectures achieve >= 10 FPS real-time threshold

"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, asdict
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path for imports 
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent  
sys.path.insert(0, str(PIPELINE_DIR))

# Default paths relative to pipeline directory
DEFAULT_CONFIG = PIPELINE_DIR / "config.yaml"
DEFAULT_OUTPUT = PIPELINE_DIR / "Results" / "rq2_architecture"

from main_perclass import SingleExperiment

# =============================================================================
# BENCHMARK GFLOPs VALUES 
# =============================================================================
# Official values at 640px input resolution

GFLOPS_BENCHMARKS = {
    # Person Detection (Stage-2)
    'yolov8n': 8.7,      # YOLOv8n official: 8.7 GFLOPs @ 640px
    'yolov8s': 28.6,
    'yolov8m': 78.9,
    
    # Weapon Detection (Stage-3)
    'efficientvit_yolov8': 6.2,   # EfficientViT-M0 backbone + YOLOv8 head
    'yolov8_efficientvit': 6.2,   # Same model, different naming
    'rt_detr': 81.4,              # RT-DETR-L official
    'rt_detr_l': 110.0,
    'rt_detr_x': 232.0,
}

STAGE2_INPUT_SIZE = 800
STAGE3_INPUT_SIZE = 640


def scale_gflops(base_gflops: float, base_size: int, actual_size: int) -> float:
    """Scale GFLOPs for different input resolutions."""
    return base_gflops * (actual_size / base_size) ** 2


def compute_pipeline_gflops(stage2_model: str = 'yolov8n',
                            stage3_model: str = 'efficientvit_yolov8',
                            avg_crops_per_frame: float = 3.0) -> dict:
    """Compute total GFLOPs using benchmark values."""
    stage2_gflops = scale_gflops(GFLOPS_BENCHMARKS.get(stage2_model, 8.7), 640, STAGE2_INPUT_SIZE)
    stage3_per_crop = GFLOPS_BENCHMARKS.get(stage3_model, 6.2)
    stage3_total = stage3_per_crop * avg_crops_per_frame
    total = stage2_gflops + stage3_total
    
    return {
        'stage2_gflops': round(stage2_gflops, 2),
        'stage3_per_crop_gflops': round(stage3_per_crop, 2),
        'stage3_total_gflops': round(stage3_total, 2),
        'avg_crops_per_frame': avg_crops_per_frame,
        'total_gflops_per_frame': round(total, 2)
    }


def print_gflops_summary():
    """Print GFLOPs summary for architecture comparison."""
    evit = compute_pipeline_gflops('yolov8n', 'efficientvit_yolov8', 3.0)
    rtdetr = compute_pipeline_gflops('yolov8n', 'rt_detr', 3.0)
    
    print("\n[GFLOPs] Benchmark Values:")
    print(f"  EfficientViT pipeline: {evit['total_gflops_per_frame']:.2f} GFLOPs/frame")
    print(f"  RT-DETR pipeline:      {rtdetr['total_gflops_per_frame']:.2f} GFLOPs/frame")
    print(f"  RT-DETR/EfficientViT:  {rtdetr['total_gflops_per_frame']/evit['total_gflops_per_frame']:.1f}x more compute")


# GFLOPs reference values from compute_flops.py
GFLOPS_RTDETR = 257.79
GFLOPS_EFFICIENTVIT = 32.19


@dataclass
class ArchitectureMetrics:
    """Stores comprehensive metrics for architecture comparison."""
    name: str
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float
    tp_count: int
    fp_count: int
    fn_count: int
    handgun_mAP50: float
    knife_mAP50: float
    handgun_precision: float
    knife_precision: float
    gflops: float
    fps: float
    latency_ms: float
    
    def to_dict(self) -> dict:
        return asdict(self)


class RQ2ArchitectureComparison:
    """
    RQ2: RT-DETR vs YOLOv8-EfficientViT Comparison
    
    This experiment compares two fundamentally different architectures:
    
    RT-DETR (Real-Time DEtection TRansformer):
      - Transformer-based end-to-end detection
      - Global attention mechanisms
      - Higher computational cost 
      - Expected advantage on complex occlusion/small objects
    
    EfficientViT-YOLOv8:
      - CNN backbone with efficient attention
      - Local feature extraction
      - Much lower compute 
      - Expected accuracy trade-off for efficiency
    
    """
    
    ARCHITECTURES = ['rt_detr', 'yolov8_efficientvit']
    
    def __init__(self, config_path: str, output_dir: str = 'Results/rq2_architecture'):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        self.results: Dict[str, ArchitectureMetrics] = {}
        
        if torch.cuda.is_available():
            print(f"[RQ2] GPU: {torch.cuda.get_device_name(0)}")
    
    def _save_config(self, config: dict, name: str) -> str:
        """Write modified config to temp file."""
        config_dir = self.output_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        temp_path = config_dir / f'{name}_config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)
        return str(temp_path)
    
    def _prepare_config(self, config: dict) -> dict:
        """Set safe defaults to prevent runtime errors."""
        if 'stage_2' not in config:
            config['stage_2'] = {}
        
        # Frame gap must be >= 1 for throughput calculation
        frame_gap = config['stage_2'].get('frame_gap', 1)
        config['stage_2']['frame_gap'] = max(1, int(frame_gap))
        
        if 'evaluation' not in config:
            config['evaluation'] = {}
        
        # Disable thop to avoid hook conflicts
        config['evaluation']['compute_flops'] = False
        
        return config
    
    def _run_architecture(self, arch_name: str) -> ArchitectureMetrics:
        """
        Run evaluation for a single architecture configuration.
        
        Architecture is specified in stage_3.approach config field.
        The experiment uses identical preprocessing/postprocessing,
        isolating the weapon detector as the only variable.
        """
        print(f"\n{'='*60}")
        print(f"[RQ2] Testing Architecture: {arch_name}")
        print(f"{'='*60}")
        
        config = deepcopy(self.base_config)
        config['stage_3'] = config.get('stage_3', {})
        config['stage_3']['approach'] = arch_name
        
        config = self._prepare_config(config)
        temp_config_path = self._save_config(config, arch_name)
        
        exp_output_dir = str(self.output_dir / 'runs' / arch_name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        use_tracker = config.get('stage_2', {}).get('use_tracker', False)
        frame_gap = config.get('stage_2', {}).get('frame_gap', 1)
        
        exp = SingleExperiment(
            config_path=temp_config_path,
            use_tracker=use_tracker,
            frame_gap=frame_gap,
            experiment_name=f"rq2_{arch_name}",
            output_dir=exp_output_dir
        )
        
        results = exp.run()
        
        # Extract per-class metrics for knife vs handgun analysis
        per_class_map = results.get('pipeline_map50_by_class', {})
        per_class_prec = results.get('pipeline_precision_by_class', {})
        
        # Compute GFLOPs using benchmark values
        avg_crops = results.get('avg_crops_per_frame', 3.0)
        gflops_info = compute_pipeline_gflops(
            stage2_model='yolov8n',
            stage3_model=arch_name,
            avg_crops_per_frame=avg_crops
        )
        computed_gflops = gflops_info['total_gflops_per_frame']
        
        # Print GFLOPs breakdown
        print(f"\n[GFLOPs] {arch_name}:")
        print(f"  Stage-2 (YOLOv8n):     {gflops_info['stage2_gflops']:.2f} GFLOPs")
        print(f"  Stage-3 per crop:      {gflops_info['stage3_per_crop_gflops']:.2f} GFLOPs")
        print(f"  Stage-3 total ({avg_crops:.1f}x):  {gflops_info['stage3_total_gflops']:.2f} GFLOPs")
        print(f"  TOTAL per frame:       {computed_gflops:.2f} GFLOPs")
        
        return ArchitectureMetrics(
            name=arch_name,
            mAP50=results.get('pipeline_map50', 0.0),
            mAP50_95=results.get('pipeline_map50_95', 0.0),
            precision=results.get('pipeline_precision', 0.0),
            recall=results.get('pipeline_recall', 0.0),
            f1=results.get('pipeline_f1', 0.0),
            tp_count=results.get('pipeline_tp', 0),
            fp_count=results.get('pipeline_fp', 0),
            fn_count=results.get('pipeline_fn', 0),
            handgun_mAP50=per_class_map.get('handgun', 0.0),
            knife_mAP50=per_class_map.get('knife', 0.0),
            handgun_precision=per_class_prec.get('handgun', 0.0),
            knife_precision=per_class_prec.get('knife', 0.0),
            gflops=computed_gflops,
            fps=results.get('fps', 0.0),
            latency_ms=results.get('latency_ms', 0.0)
        )
    
    def run_comparison(self) -> Dict[str, ArchitectureMetrics]:
        """Run both architectures and save results."""
        print("\n" + "=" * 70)
        print("RQ2: ARCHITECTURE COMPARISON")
        print("RT-DETR (Transformer) vs YOLOv8-EfficientViT (CNN)")
        print("=" * 70)
        
        for arch_name in self.ARCHITECTURES:
            try:
                result = self._run_architecture(arch_name)
                self.results[arch_name] = result
                self._save_results()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[RQ2] ERROR with {arch_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def validate_hypotheses(self) -> Dict[str, Dict]:
        """
        Evaluate all four H2.x hypotheses.
        
        Thresholds from Chapter 3:
          H2.1: RT-DETR >= EfficientViT on mAP50
          H2.2: >= 90% accuracy at <= 50% cost 
          H2.3: Larger RT-DETR advantage on knives 
          H2.4: Both >= 10 FPS 
        """
        validation = {}
        
        rt_detr = self.results.get('rt_detr')
        efficientvit = self.results.get('yolov8_efficientvit')
        
        if not rt_detr or not efficientvit:
            print("[RQ2] Missing results for validation")
            return validation
        
        # H2.1: RT-DETR parity or superiority
        # Calculated as (RT-DETR - EfficientViT) in percentage points
        mAP_diff_pp = (rt_detr.mAP50 - efficientvit.mAP50) * 100
        validation['H2.1'] = {
            'description': 'RT-DETR achieves parity or superiority vs EfficientViT',
            'rt_detr_mAP50': rt_detr.mAP50,
            'efficientvit_mAP50': efficientvit.mAP50,
            'difference_pp': mAP_diff_pp,
            'hypothesis_supported': mAP_diff_pp >= 0
        }
        
        # H2.2: Efficiency trade-off
        # EfficientViT should achieve >= 90% accuracy at <= 50% compute cost
        if rt_detr.mAP50 > 0:
            accuracy_ratio = (efficientvit.mAP50 / rt_detr.mAP50) * 100
        else:
            accuracy_ratio = 100.0
        
        if rt_detr.gflops > 0:
            cost_ratio = (efficientvit.gflops / rt_detr.gflops) * 100
        else:
            cost_ratio = 100.0
        
        validation['H2.2'] = {
            'description': 'EfficientViT >= 90% accuracy at <= 50% cost',
            'accuracy_ratio_percent': accuracy_ratio,
            'cost_ratio_percent': cost_ratio,
            'efficientvit_gflops': efficientvit.gflops,
            'rt_detr_gflops': rt_detr.gflops,
            'hypothesis_supported': accuracy_ratio >= 90 and cost_ratio <= 50
        }
        
        # H2.3: Per-class performance
        # Original hypothesis: RT-DETR advantage larger on knives (small objects)
        knife_gap_pp = (rt_detr.knife_mAP50 - efficientvit.knife_mAP50) * 100
        handgun_gap_pp = (rt_detr.handgun_mAP50 - efficientvit.handgun_mAP50) * 100
        
        validation['H2.3'] = {
            'description': 'RT-DETR shows larger advantage on knives than handguns',
            'rt_detr_knife_mAP50': rt_detr.knife_mAP50,
            'efficientvit_knife_mAP50': efficientvit.knife_mAP50,
            'knife_gap_pp': knife_gap_pp,
            'rt_detr_handgun_mAP50': rt_detr.handgun_mAP50,
            'efficientvit_handgun_mAP50': efficientvit.handgun_mAP50,
            'handgun_gap_pp': handgun_gap_pp,
            'hypothesis_supported': knife_gap_pp > handgun_gap_pp and knife_gap_pp > 0
        }
        
        # H2.4: Real-time threshold
        validation['H2.4'] = {
            'description': 'Both architectures achieve >= 10 FPS',
            'rt_detr_fps': rt_detr.fps,
            'efficientvit_fps': efficientvit.fps,
            'threshold_fps': 10.0,
            'hypothesis_supported': rt_detr.fps >= 10 and efficientvit.fps >= 10
        }
        
        return validation
    
    def _save_results(self):
        """Persist results to JSON."""
        output_path = self.output_dir / 'architecture_comparison.json'
        serializable = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[RQ2] Results saved: {output_path}")
    
    def generate_latex_table(self):
        """Generate LaTeX table for dissertation Chapter 4."""
        latex = r"""
\begin{table}[htbp]
\centering
\caption{RQ2: Architecture Comparison Results}
\label{tab:rq2_architecture}
\begin{tabular}{lcccccc}
\toprule
Architecture & mAP50 & Precision & Recall & F1 & GFLOPs & FPS \\
\midrule
"""
        for name, result in self.results.items():
            display_name = name.replace('_', '-').upper()
            if 'efficientvit' in name.lower():
                display_name = 'EfficientViT-YOLOv8'
            latex += f"{display_name} & {result.mAP50:.3f} & {result.precision:.3f} & "
            latex += f"{result.recall:.3f} & {result.f1:.3f} & "
            latex += f"{result.gflops:.1f} & {result.fps:.1f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.output_dir / 'rq2_tables.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[RQ2] LaTeX saved: {output_path}")
    
    def generate_figures(self):
        """Generate visualizations for H2.2 and H2.3."""
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        if len(self.results) < 2:
            return
        
        rt_detr = self.results.get('rt_detr')
        efficientvit = self.results.get('yolov8_efficientvit')
        
        if not rt_detr or not efficientvit:
            return
        
        # Pareto frontier: Accuracy vs Efficiency trade-off
        plt.figure(figsize=(8, 6))
        plt.scatter([rt_detr.gflops], [rt_detr.mAP50], 
                   s=150, c='blue', label='RT-DETR', marker='o')
        plt.scatter([efficientvit.gflops], [efficientvit.mAP50], 
                   s=150, c='red', label='YOLOv8-EfficientViT', marker='s')
        plt.xlabel('GFLOPs (Computational Cost)', fontsize=12)
        plt.ylabel('mAP50 (Detection Accuracy)', fontsize=12)
        plt.title('H2.2: Accuracy vs Efficiency Trade-off', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'pareto_frontier.png', dpi=300)
        plt.close()
        
        # Per-class comparison for H2.3
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(2)
        width = 0.35
        
        rt_vals = [rt_detr.handgun_mAP50, rt_detr.knife_mAP50]
        eff_vals = [efficientvit.handgun_mAP50, efficientvit.knife_mAP50]
        
        ax.bar(x - width/2, rt_vals, width, label='RT-DETR', color='blue')
        ax.bar(x + width/2, eff_vals, width, label='YOLOv8-EfficientViT', color='red')
        ax.set_ylabel('mAP50', fontsize=12)
        ax.set_title('H2.3: Per-Class Detection Performance', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['Handgun', 'Knife'], fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'per_class_comparison.png', dpi=300)
        plt.close()
        
        print(f"[RQ2] Figures saved to: {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description='RQ2: Architecture Comparison')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG),
                        help='Path to config.yaml')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ2: ARCHITECTURE COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    
    # Print GFLOPs benchmark summary
    print_gflops_summary()
    
    # Verify config and display paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return
    
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    frames_dir = cfg.get('pipeline', {}).get('frames_dir', 'NOT SET')
    print(f"\n[CONFIG] frames_dir: {frames_dir}")
    
    if not Path(frames_dir).exists():
        print(f"[WARNING] frames_dir does not exist!")
    else:
        import glob
        frame_count = len(glob.glob(str(Path(frames_dir) / '*')))
        print(f"  Found {frame_count} files")
    
    experiment = RQ2ArchitectureComparison(config_path=args.config, output_dir=args.output)
    experiment.run_comparison()
    
    validation = experiment.validate_hypotheses()
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)
    
    for h_id, h_result in validation.items():
        status = "SUPPORTED" if h_result.get('hypothesis_supported') else "NOT SUPPORTED"
        print(f"\n{h_id}: {status}")
        print(f"  {h_result.get('description', '')}")
    
    experiment.generate_latex_table()
    experiment.generate_figures()
    
    validation_path = Path(args.output) / 'hypothesis_validation.json'
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()