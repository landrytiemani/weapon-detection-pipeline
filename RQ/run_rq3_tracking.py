#!/usr/bin/env python3
"""
RQ3: Temporal Tracking Integration with ByteTrack
=================================================
Evaluates ByteTrack integration for computational efficiency through
selective frame processing (frame skipping with track interpolation).

Hypotheses tested:
  H3.1 - Temporal tracking reduces GFLOPs by >= 33% with <= 2pp accuracy loss
  H3.2 - Frame gap 3-5 achieves >= 30% FPS improvement with <= 2pp accuracy loss
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
DEFAULT_OUTPUT = PIPELINE_DIR / "Results" / "rq3_tracking"

from main_perclass import SingleExperiment

# =============================================================================
# BENCHMARK GFLOPs VALUES (from compute_flops.py)
# =============================================================================
GFLOPS_BENCHMARKS = {
    'yolov8n': 8.7,
    'efficientvit_yolov8': 6.2,
    'yolov8_efficientvit': 6.2,
    'rt_detr': 81.4,
}

STAGE2_INPUT_SIZE = 800
STAGE3_INPUT_SIZE = 640


def scale_gflops(base_gflops: float, base_size: int, actual_size: int) -> float:
    """Scale GFLOPs for different input resolutions."""
    return base_gflops * (actual_size / base_size) ** 2


def compute_pipeline_gflops(stage2_model: str = 'yolov8n',
                            stage3_model: str = 'efficientvit_yolov8',
                            avg_crops_per_frame: float = 3.0,
                            frame_gap: int = 1) -> dict:
    """
    Compute total GFLOPs using benchmark values.
    
    For tracking experiments, Stage-2 (person detection) only runs every
    frame_gap frames, so its effective cost is divided by frame_gap.
    """
    stage2_base = scale_gflops(GFLOPS_BENCHMARKS.get(stage2_model, 8.7), 640, STAGE2_INPUT_SIZE)
    # Effective Stage-2 cost is reduced by frame skipping
    stage2_effective = stage2_base / max(1, frame_gap)
    
    stage3_per_crop = GFLOPS_BENCHMARKS.get(stage3_model, 6.2)
    stage3_total = stage3_per_crop * avg_crops_per_frame
    
    total = stage2_effective + stage3_total
    
    return {
        'stage2_gflops': round(stage2_effective, 2),
        'stage2_base_gflops': round(stage2_base, 2),
        'stage3_per_crop_gflops': round(stage3_per_crop, 2),
        'stage3_total_gflops': round(stage3_total, 2),
        'total_gflops_per_frame': round(total, 2),
        'avg_crops_per_frame': avg_crops_per_frame,
        'frame_gap': frame_gap
    }


def print_gflops_summary():
    """Print GFLOPs summary for tracking experiment."""
    gap1 = compute_pipeline_gflops('yolov8n', 'efficientvit_yolov8', 3.0, frame_gap=1)
    gap3 = compute_pipeline_gflops('yolov8n', 'efficientvit_yolov8', 3.0, frame_gap=3)
    
    print("\n[GFLOPs] Tracking Frame Gap Impact:")
    print(f"  gap=1: {gap1['total_gflops_per_frame']:.2f} GFLOPs/frame (Stage-2: {gap1['stage2_gflops']:.2f})")
    print(f"  gap=3: {gap3['total_gflops_per_frame']:.2f} GFLOPs/frame (Stage-2: {gap3['stage2_gflops']:.2f})")
    print(f"  Savings: {(1 - gap3['total_gflops_per_frame']/gap1['total_gflops_per_frame'])*100:.1f}%")


# Extended range to capture efficiency curve behavior
FRAME_GAPS = [1, 2, 3, 4, 5, 6, 7, 8]


@dataclass
class TrackingResult:
    """Stores metrics for a tracking configuration."""
    config_name: str
    use_tracker: bool
    frame_gap: int
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
    
    def to_dict(self) -> dict:
        return asdict(self)


class RQ3TrackingExperiment:
    """
    RQ3: ByteTrack Temporal Tracking Integration
    
    ByteTrack provides multi-object tracking that enables frame skipping:
    instead of running person detection every frame, we detect on frame N
    and interpolate bounding boxes for frames N+1 to N+gap-1 using tracks.
    
    ByteTrack configuration per methodology:
      - track_thresh: 0.15 (low threshold for consistent tracking)
      - track_buffer: 60 frames (handles brief occlusions)
      - match_thresh: 0.7 (IoU threshold for association)
    """
    
    def __init__(self, config_path: str, output_dir: str = 'Results/rq3_tracking'):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        self.results: Dict[str, TrackingResult] = {}
        
        if torch.cuda.is_available():
            print(f"[RQ3] GPU: {torch.cuda.get_device_name(0)}")
    
    def _save_config(self, config: dict, name: str) -> str:
        """Write modified config to temp file."""
        config_dir = self.output_dir / 'configs'
        config_dir.mkdir(exist_ok=True)
        temp_path = config_dir / f'{name}_config.yaml'
        with open(temp_path, 'w') as f:
            yaml.dump(config, f)
        return str(temp_path)
    
    def _prepare_config(self, config: dict, use_tracker: bool, frame_gap: int) -> dict:
        """
        Configure tracking parameters.
        
        Critical: frame_gap must be >= 1 to avoid division by zero
        in throughput calculations.
        """
        if 'stage_2' not in config:
            config['stage_2'] = {}
        
        config['stage_2']['use_tracker'] = use_tracker
        config['stage_2']['frame_gap'] = max(1, int(frame_gap))
        
        if 'evaluation' not in config:
            config['evaluation'] = {}
        
        # Disable thop to avoid hook conflicts
        config['evaluation']['compute_flops'] = False
        
        return config
    
    def _run_tracking_config(self, use_tracker: bool, frame_gap: int, name: str) -> TrackingResult:
        """Run evaluation for a single tracking configuration."""
        print(f"\n{'='*60}")
        print(f"[RQ3] Running: {name} (tracker={use_tracker}, gap={frame_gap})")
        print(f"{'='*60}")
        
        config = deepcopy(self.base_config)
        config = self._prepare_config(config, use_tracker, frame_gap)
        temp_config_path = self._save_config(config, name)
        
        exp_output_dir = str(self.output_dir / 'runs' / name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        exp = SingleExperiment(
            config_path=temp_config_path,
            use_tracker=use_tracker,
            frame_gap=max(1, frame_gap),
            experiment_name=name,
            output_dir=exp_output_dir
        )
        
        results = exp.run()
        
        # Compute GFLOPs using benchmark values with frame gap consideration
        avg_crops = results.get('avg_crops_per_frame', 3.0)
        stage3_approach = config.get('stage_3', {}).get('approach', 'yolov8_efficientvit')
        effective_gap = frame_gap if use_tracker else 1
        gflops_info = compute_pipeline_gflops(
            stage2_model='yolov8n',
            stage3_model=stage3_approach,
            avg_crops_per_frame=avg_crops,
            frame_gap=effective_gap
        )
        computed_gflops = gflops_info['total_gflops_per_frame']
        
        # Print GFLOPs breakdown
        print(f"\n[GFLOPs] {name} (gap={effective_gap}):")
        print(f"  Stage-2 (effective):   {gflops_info['stage2_gflops']:.2f} GFLOPs")
        print(f"  Stage-3 total ({avg_crops:.1f}x):  {gflops_info['stage3_total_gflops']:.2f} GFLOPs")
        print(f"  TOTAL per frame:       {computed_gflops:.2f} GFLOPs")
        
        return TrackingResult(
            config_name=name,
            use_tracker=use_tracker,
            frame_gap=frame_gap,
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
            latency_ms=results.get('latency_ms', 0.0)
        )
    
    def run_baseline_comparison(self) -> Dict[str, TrackingResult]:
        """
        Compare tracking vs no tracking at gap=1.
        
        This isolates the effect of ByteTrack overhead without frame skipping.
        At gap=1, every frame is processed, so any difference is pure
        tracking overhead (should be minimal).
        """
        print("\n[RQ3] Tracking vs No Tracking baseline...")
        
        result = self._run_tracking_config(False, 1, 'no_tracking')
        self.results['no_tracking'] = result
        torch.cuda.empty_cache()
        
        result = self._run_tracking_config(True, 1, 'with_tracking')
        self.results['with_tracking'] = result
        
        self._save_results()
        return self.results
    
    def run_frame_gap_sweep(self) -> Dict[str, TrackingResult]:
        """
        H3.2: Sweep frame gaps to characterize accuracy-efficiency trade-off.
        
        Tests gaps from 1 (every frame) to 8 (process 1 in 8 frames).
        """
        print("\n[RQ3/H3.2] Frame Gap Sweep...")
        
        for gap in FRAME_GAPS:
            name = f"gap_{gap}"
            result = self._run_tracking_config(True, gap, name)
            self.results[name] = result
            self._save_results()
            torch.cuda.empty_cache()
        
        return self.results
    
    def run_full_experiment(self):
        """Run complete tracking evaluation."""
        self.run_baseline_comparison()
        self.run_frame_gap_sweep()
        return self.results
    
    def validate_hypotheses(self) -> Dict[str, Dict]:
        """
        Evaluate H3.1 and H3.2 against dissertation thresholds.
        
        Thresholds
          H3.1: >= 33% GFLOPs reduction with <= 2pp mAP50 loss
          H3.2: >= 30% FPS improvement with <= 2pp mAP50-95 loss
        """
        validation = {}
        
        no_track = self.results.get('no_tracking')
        with_track = self.results.get('with_tracking')
        gap1 = self.results.get('gap_1')
        gap_results = {k: v for k, v in self.results.items() if k.startswith('gap_')}
        
        # H3.1: GFLOPs reduction with accuracy preservation
        # Threshold: >= 33% reduction, <= 2pp accuracy loss
        if gap1 and gap_results:
            # Find configuration with max GFLOPs reduction
            best_reduction = 0.0
            best_gap = 1
            best_accuracy_loss = 0.0
            
            for name, result in gap_results.items():
                if name == 'gap_1':
                    continue
                gap = int(name.split('_')[-1])
                
                if gap1.gflops > 0:
                    reduction = ((gap1.gflops - result.gflops) / gap1.gflops) * 100
                else:
                    reduction = 0.0
                
                accuracy_loss = (gap1.mAP50 - result.mAP50) * 100
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_gap = gap
                    best_accuracy_loss = accuracy_loss
            
            validation['H3.1'] = {
                'description': 'Tracking reduces GFLOPs by >= 33% with <= 2pp accuracy loss',
                'best_gflops_reduction_percent': best_reduction,
                'best_gap': best_gap,
                'accuracy_loss_pp': best_accuracy_loss,
                'threshold_reduction_percent': 33.0,
                'threshold_accuracy_loss_pp': 2.0,
                'hypothesis_supported': best_reduction >= 33.0 and best_accuracy_loss <= 2.0
            }
        
        # H3.2: FPS improvement with accuracy preservation
        # Threshold: >= 30% FPS improvement, <= 2pp accuracy loss
        if gap1 and gap_results:
            # Check if any gap in [3, 5] range meets criteria
            target_gap = None
            target_fps_gain = 0.0
            target_accuracy_loss = 0.0
            
            for gap_val in [3, 5]:
                gap_name = f'gap_{gap_val}'
                if gap_name in gap_results:
                    result = gap_results[gap_name]
                    
                    if gap1.fps > 0:
                        fps_gain = ((result.fps - gap1.fps) / gap1.fps) * 100
                    else:
                        fps_gain = 0.0
                    
                    # Use mAP50-95 for stricter accuracy metric
                    accuracy_loss = (gap1.mAP50_95 - result.mAP50_95) * 100
                    
                    if fps_gain >= 30.0 and accuracy_loss <= 2.0:
                        target_gap = gap_val
                        target_fps_gain = fps_gain
                        target_accuracy_loss = accuracy_loss
                        break
            
            validation['H3.2'] = {
                'description': 'Frame gap 3-5 achieves >= 30% FPS gain with <= 2pp accuracy loss',
                'target_gap': target_gap,
                'fps_gain_percent': target_fps_gain,
                'accuracy_loss_pp': target_accuracy_loss,
                'threshold_fps_gain_percent': 30.0,
                'threshold_accuracy_loss_pp': 2.0,
                'gap_results': {
                    k: {
                        'mAP50': v.mAP50,
                        'mAP50_95': v.mAP50_95,
                        'fps': v.fps
                    } for k, v in gap_results.items()
                },
                'hypothesis_supported': target_gap is not None
            }
        
        return validation
    
    def _save_results(self):
        """Persist results to JSON."""
        output_path = self.output_dir / 'tracking_results.json'
        serializable = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[RQ3] Results saved: {output_path}")
    
    def generate_latex_table(self):
        """Generate LaTeX table for dissertation Chapter 4."""
        latex = r"""
\begin{table}[htbp]
\centering
\caption{RQ3: Temporal Tracking Results}
\label{tab:rq3_tracking}
\begin{tabular}{lcccccc}
\toprule
Configuration & mAP50 & mAP50-95 & Precision & Recall & FPS & Latency (ms) \\
\midrule
"""
        for name, result in self.results.items():
            display_name = name.replace('_', ' ').title()
            latex += f"{display_name} & {result.mAP50:.3f} & {result.mAP50_95:.3f} & "
            latex += f"{result.precision:.3f} & {result.recall:.3f} & "
            latex += f"{result.fps:.1f} & {result.latency_ms:.1f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.output_dir / 'rq3_tables.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[RQ3] LaTeX saved: {output_path}")
    
    def generate_figures(self):
        """Generate frame gap trade-off visualization."""
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        gap_results = {k: v for k, v in self.results.items() if k.startswith('gap_')}
        if len(gap_results) > 1:
            gaps = []
            mAPs = []
            fpss = []
            
            for name in sorted(gap_results.keys(), key=lambda x: int(x.split('_')[-1])):
                gap = int(name.split('_')[-1])
                gaps.append(gap)
                mAPs.append(gap_results[name].mAP50)
                fpss.append(gap_results[name].fps)
            
            # Dual-axis plot showing accuracy-throughput trade-off
            fig, ax1 = plt.subplots(figsize=(8, 5))
            
            ax1.set_xlabel('Frame Gap', fontsize=12)
            ax1.set_ylabel('mAP50 (Accuracy)', color='blue', fontsize=12)
            ax1.plot(gaps, mAPs, 'b-o', linewidth=2, markersize=8, label='mAP50')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('FPS (Throughput)', color='red', fontsize=12)
            ax2.plot(gaps, fpss, 'r--s', linewidth=2, markersize=8, label='FPS')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Mark recommended operating point
            if 3 in gaps:
                idx = gaps.index(3)
                ax1.axvline(x=3, color='green', linestyle=':', alpha=0.5)
                ax1.annotate('Recommended', xy=(3, mAPs[idx]), 
                            xytext=(3.5, mAPs[idx] + 0.02),
                            fontsize=10, color='green')
            
            plt.title('H3.2: Frame Gap Accuracy-Throughput Trade-off', fontsize=14)
            fig.tight_layout()
            plt.savefig(figures_dir / 'frame_gap_tradeoff.png', dpi=300)
            plt.close()
            
            print(f"[RQ3] Figure saved: {figures_dir / 'frame_gap_tradeoff.png'}")


def main():
    parser = argparse.ArgumentParser(description='RQ3: Tracking Experiment')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG),
                        help='Path to config.yaml')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output directory for results')
    parser.add_argument('--experiment', type=str, default='full',
                        choices=['full', 'baseline', 'frame_gap'])
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ3: TEMPORAL TRACKING EXPERIMENT")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Experiment: {args.experiment}")
    
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
    
    experiment = RQ3TrackingExperiment(config_path=args.config, output_dir=args.output)
    
    if args.experiment == 'full':
        experiment.run_full_experiment()
    elif args.experiment == 'baseline':
        experiment.run_baseline_comparison()
    elif args.experiment == 'frame_gap':
        experiment.run_frame_gap_sweep()
    
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