#!/usr/bin/env python3
"""
RQ4: Privacy-Preserving Weapon Detection
=========================================
Evaluates selective face anonymization for GDPR compliance without
compromising detection accuracy or real-time performance.

Hypotheses tested:
  H4.1 - Privacy processing adds <= 5% computational overhead
  H4.2 - Privacy processing causes <= 2 percentage points accuracy loss

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

# Add parent directory to path for imports (main_perclass.py is in parent dir)
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent  # /home/ubuntu/weapon_detection_pipeline
sys.path.insert(0, str(PIPELINE_DIR))

# Default paths relative to pipeline directory
DEFAULT_CONFIG = PIPELINE_DIR / "config.yaml"
DEFAULT_OUTPUT = PIPELINE_DIR / "Results" / "rq4_privacy"

from main_perclass import SingleExperiment

# =============================================================================
# BENCHMARK GFLOPs VALUES (from compute_flops.py)
# =============================================================================
GFLOPS_BENCHMARKS = {
    'yolov8n': 8.7,
    'efficientvit_yolov8': 6.2,
    'yolov8_efficientvit': 6.2,
    'rt_detr': 81.4,
    'face_blur_pixelate': 0.001,
    'face_blur_gaussian': 0.002,
}

STAGE2_INPUT_SIZE = 800
STAGE3_INPUT_SIZE = 640


def scale_gflops(base_gflops: float, base_size: int, actual_size: int) -> float:
    """Scale GFLOPs for different input resolutions."""
    return base_gflops * (actual_size / base_size) ** 2


def compute_pipeline_gflops(stage2_model: str = 'yolov8n',
                            stage3_model: str = 'efficientvit_yolov8',
                            avg_crops_per_frame: float = 3.0,
                            privacy_enabled: bool = False,
                            privacy_method: str = 'pixelate') -> dict:
    """Compute total GFLOPs using benchmark values, including privacy overhead."""
    stage2_gflops = scale_gflops(GFLOPS_BENCHMARKS.get(stage2_model, 8.7), 640, STAGE2_INPUT_SIZE)
    stage3_per_crop = GFLOPS_BENCHMARKS.get(stage3_model, 6.2)
    stage3_total = stage3_per_crop * avg_crops_per_frame
    
    # Privacy overhead
    privacy_overhead = 0.0
    if privacy_enabled:
        privacy_overhead = GFLOPS_BENCHMARKS.get(f'face_blur_{privacy_method}', 0.001)
    
    total = stage2_gflops + stage3_total + privacy_overhead
    
    return {
        'stage2_gflops': round(stage2_gflops, 2),
        'stage3_per_crop_gflops': round(stage3_per_crop, 2),
        'stage3_total_gflops': round(stage3_total, 2),
        'privacy_overhead_gflops': round(privacy_overhead, 4),
        'avg_crops_per_frame': avg_crops_per_frame,
        'total_gflops_per_frame': round(total, 2)
    }


def print_gflops_summary():
    """Print GFLOPs summary for privacy experiment."""
    no_privacy = compute_pipeline_gflops('yolov8n', 'efficientvit_yolov8', 3.0, privacy_enabled=False)
    with_privacy = compute_pipeline_gflops('yolov8n', 'efficientvit_yolov8', 3.0, privacy_enabled=True)
    
    print("\n[GFLOPs] Privacy Processing Overhead:")
    print(f"  Without privacy: {no_privacy['total_gflops_per_frame']:.2f} GFLOPs/frame")
    print(f"  With privacy:    {with_privacy['total_gflops_per_frame']:.2f} GFLOPs/frame")
    print(f"  Overhead:        {with_privacy['privacy_overhead_gflops']:.4f} GFLOPs ({with_privacy['privacy_overhead_gflops']/no_privacy['total_gflops_per_frame']*100:.3f}%)")


@dataclass
class PrivacyResult:
    """Stores metrics for a privacy configuration."""
    config_name: str
    privacy_enabled: bool
    privacy_scope: str
    privacy_method: str
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


class RQ4PrivacyExperiment:
    """
    RQ4: Privacy-Preserving Weapon Detection
    
    This experiment evaluates selective face anonymization that enables
    GDPR compliance while maintaining detection effectiveness. The key
    innovation is selective scope: only non-target individuals (those
    not associated with weapon detections) are anonymized.
    
    Selective vs blanket anonymization:
      - Selective: preserves threat actor identifiability for response
      - Blanket: anonymizes everyone, may hinder investigation
    
    Implementation per methodology:
      - Face region heuristic: upper 38% height, central 70% width
      - No face detector needed (approximation from person bbox)
      - Pixelation block size: 15 pixels
      - Gaussian kernel: 31x31
    """
    
    # Privacy configurations to test
    # Note: silhouette method was explored but not included in final dissertation
    CONFIGURATIONS = [
        {
            'enabled': False, 
            'scope': 'none', 
            'method': 'none', 
            'name': 'no_privacy'
        },
        {
            'enabled': True, 
            'scope': 'non_targets', 
            'method': 'pixelate', 
            'name': 'selective_pixelate'
        },
        {
            'enabled': True, 
            'scope': 'non_targets', 
            'method': 'gaussian', 
            'name': 'selective_gaussian'
        },
        {
            'enabled': True, 
            'scope': 'everyone', 
            'method': 'pixelate', 
            'name': 'blanket_pixelate'
        },
    ]
    
    def __init__(self, config_path: str, output_dir: str = 'Results/rq4_privacy'):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)
        
        self.results: Dict[str, PrivacyResult] = {}
        
        if torch.cuda.is_available():
            print(f"[RQ4] GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    def _run_privacy_config(self, privacy_cfg: dict) -> PrivacyResult:
        """
        Run evaluation for a single privacy configuration.
        
        Privacy settings per methodology:
          - pixel_block: 15 (balance between anonymization and artifacts)
          - gaussian_ksize: 31x31 (sufficient blur without extreme distortion)
          - detector: 'none' (use bbox heuristic, no separate face detector)
        """
        name = privacy_cfg['name']
        print(f"\n{'='*60}")
        print(f"[RQ4] Running: {name}")
        print(f"{'='*60}")
        
        config = deepcopy(self.base_config)
        
        # Configure privacy settings
        config['privacy'] = {
            'enabled': privacy_cfg['enabled'],
            'scope': privacy_cfg['scope'],
            'face_blur': {
                'enabled': privacy_cfg['enabled'],
                'method': privacy_cfg['method'],
                'pixel_block': 15,        # Block size for pixelation
                'gaussian_ksize': 31,     # Kernel size for Gaussian blur
                'detector': 'none'        # Use bbox heuristic, not face detector
            },
            'silhouette': {'enabled': False}  # Not used in final design
        }
        
        config = self._prepare_config(config)
        temp_config_path = self._save_config(config, name)
        
        exp_output_dir = str(self.output_dir / 'runs' / name)
        os.makedirs(exp_output_dir, exist_ok=True)
        
        use_tracker = config.get('stage_2', {}).get('use_tracker', False)
        frame_gap = config.get('stage_2', {}).get('frame_gap', 1)
        
        exp = SingleExperiment(
            config_path=temp_config_path,
            use_tracker=use_tracker,
            frame_gap=frame_gap,
            experiment_name=name,
            output_dir=exp_output_dir
        )
        
        results = exp.run()
        
        # Compute GFLOPs using benchmark values including privacy overhead
        avg_crops = results.get('avg_crops_per_frame', 3.0)
        stage3_approach = config.get('stage_3', {}).get('approach', 'yolov8_efficientvit')
        gflops_info = compute_pipeline_gflops(
            stage2_model='yolov8n',
            stage3_model=stage3_approach,
            avg_crops_per_frame=avg_crops,
            privacy_enabled=privacy_cfg['enabled'],
            privacy_method=privacy_cfg['method']
        )
        computed_gflops = gflops_info['total_gflops_per_frame']
        
        # Print GFLOPs breakdown
        privacy_str = f"privacy={privacy_cfg['method']}" if privacy_cfg['enabled'] else "no privacy"
        print(f"\n[GFLOPs] {name} ({privacy_str}):")
        print(f"  Stage-2 (YOLOv8n):     {gflops_info['stage2_gflops']:.2f} GFLOPs")
        print(f"  Stage-3 total ({avg_crops:.1f}x):  {gflops_info['stage3_total_gflops']:.2f} GFLOPs")
        if privacy_cfg['enabled']:
            print(f"  Privacy overhead:      {gflops_info['privacy_overhead_gflops']:.4f} GFLOPs")
        print(f"  TOTAL per frame:       {computed_gflops:.2f} GFLOPs")
        
        return PrivacyResult(
            config_name=name,
            privacy_enabled=privacy_cfg['enabled'],
            privacy_scope=privacy_cfg['scope'],
            privacy_method=privacy_cfg['method'],
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
    
    def run_all_configurations(self) -> Dict[str, PrivacyResult]:
        """Run all privacy configurations."""
        print("\n" + "=" * 70)
        print("RQ4: PRIVACY CONFIGURATION COMPARISON")
        print("=" * 70)
        
        for cfg in self.CONFIGURATIONS:
            result = self._run_privacy_config(cfg)
            self.results[cfg['name']] = result
            self._save_results()
            torch.cuda.empty_cache()
        
        return self.results
    
    def validate_hypotheses(self) -> Dict[str, Dict]:
        """
        Evaluate H4.1 and H4.2 against dissertation thresholds.
        
        Thresholds
          H4.1: <= 5% computational overhead 
          H4.2: <= 2pp mAP50 loss 
        """
        validation = {}
        
        baseline = self.results.get('no_privacy')
        selective_pix = self.results.get('selective_pixelate')
        selective_gauss = self.results.get('selective_gaussian')
        blanket = self.results.get('blanket_pixelate')
        
        # H4.1: Computational overhead
        # Threshold: <= 5% latency increase
        if baseline and selective_pix and baseline.latency_ms > 0:
            overhead = ((selective_pix.latency_ms - baseline.latency_ms) / 
                       baseline.latency_ms) * 100
            
            validation['H4.1'] = {
                'description': 'Privacy processing adds <= 5% overhead',
                'baseline_latency_ms': baseline.latency_ms,
                'privacy_latency_ms': selective_pix.latency_ms,
                'overhead_percent': overhead,
                'baseline_fps': baseline.fps,
                'privacy_fps': selective_pix.fps,
                'threshold_percent': 5.0,
                'hypothesis_supported': overhead <= 5.0
            }
        
        # H4.2: Accuracy preservation
        # Threshold: <= 2 percentage points mAP50 loss
        if baseline and selective_pix:
            accuracy_loss_pp = (baseline.mAP50 - selective_pix.mAP50) * 100
            
            validation['H4.2'] = {
                'description': 'Privacy processing causes <= 2pp accuracy loss',
                'baseline_mAP50': baseline.mAP50,
                'privacy_mAP50': selective_pix.mAP50,
                'accuracy_loss_pp': accuracy_loss_pp,
                'threshold_pp': 2.0,
                'hypothesis_supported': accuracy_loss_pp <= 2.0
            }
        
        # Additional analysis: method comparison
        # Not a formal hypothesis, but useful for deployment guidance
        if selective_pix and selective_gauss:
            if selective_gauss.latency_ms > 0:
                pixelate_faster = selective_pix.latency_ms < selective_gauss.latency_ms
                speedup = ((selective_gauss.latency_ms - selective_pix.latency_ms) / 
                          selective_gauss.latency_ms) * 100
            else:
                pixelate_faster = True
                speedup = 0.0
            
            validation['method_comparison'] = {
                'description': 'Pixelation vs Gaussian blur comparison',
                'pixelate_latency_ms': selective_pix.latency_ms,
                'gaussian_latency_ms': selective_gauss.latency_ms,
                'pixelate_mAP50': selective_pix.mAP50,
                'gaussian_mAP50': selective_gauss.mAP50,
                'pixelate_faster': pixelate_faster,
                'speedup_percent': speedup,
                'recommendation': 'pixelate' if pixelate_faster else 'gaussian'
            }
        
        # Additional analysis: selective vs blanket
        if selective_pix and blanket and baseline:
            if baseline.mAP50 > 0:
                selective_utility = selective_pix.mAP50 / baseline.mAP50
                blanket_utility = blanket.mAP50 / baseline.mAP50
            else:
                selective_utility = 1.0
                blanket_utility = 1.0
            
            validation['scope_comparison'] = {
                'description': 'Selective vs blanket anonymization',
                'selective_mAP50': selective_pix.mAP50,
                'blanket_mAP50': blanket.mAP50,
                'selective_utility_ratio': selective_utility,
                'blanket_utility_ratio': blanket_utility,
                'selective_better': selective_utility > blanket_utility,
                'recommendation': 'selective' if selective_utility >= blanket_utility else 'blanket'
            }
        
        return validation
    
    def _save_results(self):
        """Persist results to JSON."""
        output_path = self.output_dir / 'privacy_results.json'
        serializable = {name: result.to_dict() for name, result in self.results.items()}
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[RQ4] Results saved: {output_path}")
    
    def generate_latex_table(self):
        """Generate LaTeX table for dissertation Chapter 4."""
        latex = r"""
\begin{table}[htbp]
\centering
\caption{RQ4: Privacy Configuration Comparison}
\label{tab:rq4_privacy}
\begin{tabular}{llccccc}
\toprule
Scope & Method & mAP50 & Precision & Recall & FPS & Latency (ms) \\
\midrule
"""
        for name, result in self.results.items():
            scope_display = result.privacy_scope if result.privacy_scope != 'none' else 'N/A'
            method_display = result.privacy_method if result.privacy_method != 'none' else 'N/A'
            latex += f"{scope_display} & {method_display} & {result.mAP50:.3f} & "
            latex += f"{result.precision:.3f} & {result.recall:.3f} & "
            latex += f"{result.fps:.1f} & {result.latency_ms:.1f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        output_path = self.output_dir / 'rq4_tables.tex'
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"[RQ4] LaTeX saved: {output_path}")
    
    def generate_figures(self):
        """Generate privacy-utility trade-off visualization."""
        figures_dir = self.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        if len(self.results) < 2:
            return
        
        # Privacy-utility trade-off visualization
        configs = []
        mAPs = []
        latencies = []
        
        for name, result in self.results.items():
            display_name = name.replace('_', '\n')
            configs.append(display_name)
            mAPs.append(result.mAP50)
            latencies.append(result.latency_ms)
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(configs))
        
        # Bar chart for accuracy
        bars = ax1.bar(x, mAPs, color='steelblue', alpha=0.7, label='mAP50')
        ax1.set_ylabel('mAP50 (Detection Accuracy)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs, fontsize=9)
        ax1.set_ylim([0, max(mAPs) * 1.15])
        
        # Line for latency
        ax2 = ax1.twinx()
        ax2.plot(x, latencies, 'ro-', linewidth=2, markersize=8, label='Latency')
        ax2.set_ylabel('Latency (ms)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.title('RQ4: Privacy-Utility Trade-off', fontsize=14)
        fig.tight_layout()
        plt.savefig(figures_dir / 'privacy_utility_tradeoff.png', dpi=300)
        plt.close()
        
        print(f"[RQ4] Figure saved: {figures_dir / 'privacy_utility_tradeoff.png'}")


def main():
    parser = argparse.ArgumentParser(description='RQ4: Privacy Experiment')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG),
                        help='Path to config.yaml')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RQ4: PRIVACY PRESERVATION EXPERIMENT")
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
    
    experiment = RQ4PrivacyExperiment(config_path=args.config, output_dir=args.output)
    experiment.run_all_configurations()
    
    validation = experiment.validate_hypotheses()
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS VALIDATION")
    print("=" * 70)
    
    for h_id, h_result in validation.items():
        if 'hypothesis_supported' in h_result:
            status = "SUPPORTED" if h_result.get('hypothesis_supported') else "NOT SUPPORTED"
            print(f"\n{h_id}: {status}")
        else:
            print(f"\n{h_id}: (Analysis)")
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