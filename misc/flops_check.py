#!/usr/bin/env python3
"""
Comprehensive FLOPs verification script for weapon detection pipeline models.
Tests multiple calculation methods and compares against known benchmarks.
"""

import torch
import os
import sys
from pathlib import Path

print("="*80)
print("FLOPs VERIFICATION SCRIPT")
print("="*80)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("="*80 + "\n")


def check_library(lib_name):
    """Check if a library is available."""
    try:
        __import__(lib_name)
        return True
    except ImportError:
        return False


def verify_flops_single_model(model_path, imgsz=640, name=None):
    """
    Verify GFLOPs calculation for a single model using multiple methods.
    
    Args:
        model_path: Path to model file
        imgsz: Input image size
        name: Optional display name
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}\n")
        return None
    
    display_name = name or os.path.basename(model_path)
    
    print("\n" + "="*80)
    print(f"Model: {display_name}")
    print(f"Path: {model_path}")
    print(f"Input size: {imgsz}×{imgsz}")
    print("="*80)
    
    results = {}
    
    # Load model
    try:
        from ultralytics import YOLO
        print("✓ Loading model...")
        model = YOLO(model_path)
        model_module = model.model
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}\n")
        return None
    
    # Count parameters
    try:
        total_params = sum(p.numel() for p in model_module.parameters())
        trainable_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
        results['params_total'] = total_params / 1e6  # Convert to millions
        results['params_trainable'] = trainable_params / 1e6
        print(f"✓ Total parameters: {results['params_total']:.2f}M")
        print(f"✓ Trainable parameters: {results['params_trainable']:.2f}M")
    except Exception as e:
        print(f"⚠ Failed to count parameters: {e}")
    
    # Method 1: fvcore
    print("\n--- Method 1: fvcore ---")
    if check_library('fvcore'):
        try:
            from fvcore.nn import FlopCountAnalysis
            
            # Ensure model is in eval mode and FP32
            model_fp32 = model_module.float().eval()
            dummy = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)
            
            with torch.no_grad():
                flops_analysis = FlopCountAnalysis(model_fp32, dummy)
                total_flops = flops_analysis.total()
            
            gflops = total_flops / 1e9
            results['fvcore'] = gflops
            print(f"✓ fvcore: {gflops:.2f} GFLOPs")
            
        except Exception as e:
            print(f"❌ fvcore failed: {e}")
    else:
        print("⚠ fvcore not installed (pip install fvcore)")
    
    # Method 2: thop
    print("\n--- Method 2: thop ---")
    if check_library('thop'):
        try:
            from thop import profile
            import copy
            
            # Create a clean copy of the model
            model_copy = copy.deepcopy(model_module.float().eval())
            dummy = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32)
            
            with torch.no_grad():
                macs, params = profile(model_copy, inputs=(dummy,), verbose=False)
            
            # MACs to GFLOPs (multiply by 2)
            gflops = 2.0 * macs / 1e9
            results['thop'] = gflops
            print(f"✓ thop: {gflops:.2f} GFLOPs")
            
            # Clean up
            del model_copy
            
        except Exception as e:
            print(f"❌ thop failed: {e}")
    else:
        print("⚠ thop not installed (pip install thop)")
    
    # Method 3: Ultralytics built-in profiler
    print("\n--- Method 3: Ultralytics Profiler ---")
    try:
        # Try to use Ultralytics' built-in info method
        if hasattr(model, 'info'):
            info = model.info(detailed=False, verbose=False)
            print(f"✓ Model info retrieved")
            # The info might contain GFLOPs, but format varies
    except Exception as e:
        print(f"⚠ Ultralytics profiler unavailable: {e}")
    
    # Method 4: Known benchmark comparison
    print("\n--- Method 4: Known Benchmarks ---")
    model_name_lower = os.path.basename(model_path).lower()
    
    # Official benchmarks at 640×640 from Ultralytics documentation
    known_benchmarks = {
        'yolov8n': {'params': 3.2, 'gflops': 8.7},
        'yolov8s': {'params': 11.2, 'gflops': 28.6},
        'yolov8m': {'params': 25.9, 'gflops': 78.9},
        'yolov8l': {'params': 43.7, 'gflops': 165.2},
        'yolov8x': {'params': 68.2, 'gflops': 257.8},
        'yolov5n': {'params': 1.9, 'gflops': 4.5},
        'yolov5s': {'params': 7.2, 'gflops': 15.9},
        'yolov5m': {'params': 21.2, 'gflops': 48.0},
        'yolov5l': {'params': 46.5, 'gflops': 109.1},
    }
    
    matched_benchmark = None
    for key, bench in known_benchmarks.items():
        if key in model_name_lower:
            matched_benchmark = (key, bench)
            break
    
    if matched_benchmark:
        key, bench = matched_benchmark
        expected_gflops_640 = bench['gflops']
        expected_params = bench['params']
        
        # Scale GFLOPs for different input size (quadratic relationship)
        scale_factor = (imgsz / 640) ** 2
        expected_gflops = expected_gflops_640 * scale_factor
        
        results['benchmark'] = expected_gflops
        results['benchmark_640'] = expected_gflops_640
        
        print(f"✓ Matched benchmark: {key}")
        print(f"  Expected params: {expected_params:.1f}M")
        print(f"  Expected GFLOPs @ 640: {expected_gflops_640:.2f}")
        print(f"  Expected GFLOPs @ {imgsz}: {expected_gflops:.2f}")
        
        # Compare with actual parameters
        if 'params_total' in results:
            param_diff = abs(results['params_total'] - expected_params)
            if param_diff < 0.5:
                print(f"  ✓ Parameter count matches (diff: {param_diff:.2f}M)")
            else:
                print(f"  ⚠ Parameter count differs (diff: {param_diff:.2f}M)")
    else:
        print("⚠ No matching benchmark found")
        # Generic estimation
        if 'params_total' in results:
            # Rule of thumb: ~3 GFLOPs per million parameters for detection models
            estimated = results['params_total'] * 3.0 * (imgsz / 640) ** 2
            results['estimated'] = estimated
            print(f"✓ Generic estimate: {estimated:.2f} GFLOPs")
    
    # Summary
    print("\n" + "-"*80)
    print("SUMMARY:")
    print("-"*80)
    
    if 'fvcore' in results:
        print(f"fvcore:           {results['fvcore']:>8.2f} GFLOPs")
    if 'thop' in results:
        print(f"thop:             {results['thop']:>8.2f} GFLOPs")
    if 'benchmark' in results:
        print(f"Known benchmark:  {results['benchmark']:>8.2f} GFLOPs")
    if 'estimated' in results:
        print(f"Generic estimate: {results['estimated']:>8.2f} GFLOPs")
    
    # Consensus value
    measured_values = [v for k, v in results.items() if k in ['fvcore', 'thop']]
    if measured_values:
        avg_measured = sum(measured_values) / len(measured_values)
        print(f"\nMeasured average: {avg_measured:.2f} GFLOPs")
        results['consensus'] = avg_measured
    elif 'benchmark' in results:
        print(f"\nBest estimate:    {results['benchmark']:.2f} GFLOPs (from benchmark)")
        results['consensus'] = results['benchmark']
    elif 'estimated' in results:
        print(f"\nBest estimate:    {results['estimated']:.2f} GFLOPs (generic)")
        results['consensus'] = results['estimated']
    
    print("-"*80 + "\n")
    
    return results


def main():
    """Main verification routine."""
    
    # Check available libraries
    print("Checking available libraries...")
    libs = {
        'ultralytics': check_library('ultralytics'),
        'fvcore': check_library('fvcore'),
        'thop': check_library('thop'),
    }
    
    for lib, available in libs.items():
        status = "✓" if available else "❌"
        print(f"  {status} {lib}")
    
    if not libs['ultralytics']:
        print("\n❌ ultralytics not installed. Install with: pip install ultralytics")
        return
    
    if not (libs['fvcore'] or libs['thop']):
        print("\n⚠ WARNING: Neither fvcore nor thop installed.")
        print("Install at least one for accurate measurements:")
        print("  pip install fvcore")
        print("  pip install thop")
    
    print("\n")
    
    # Define models to test
    models_to_test = [
        # Stage-2 models
        {
            'path': 'weights/person/yolov8n/yolov8n.pt',
            'sizes': [640, 800],
            'name': 'YOLOv8n (Person Detection)'
        },
        {
            'path': '/home/ubuntu/weapon_detection_pipeline/weights/person/ssd_mobilenetV2_lite/ssdmobilenetv2.pth',
            'sizes': [800],
            'name': 'SSD MobileNetV2 (Person Detection)'
        },
        
        # Stage-3 models
        {
            'path': '/home/ubuntu/weapon_detection_pipeline/weights/weapon/efficientvit_yolov8/efficientvit_yolov8.pt',
            'sizes': [512, 640],
            'name': 'EfficientViT-YOLOv8 (Weapon Detection)'
        },
        {
            'path': '/home/ubuntu/weapon_detection_pipeline/weights/weapon/rt_detr/rt_detr.pt',
            'sizes': [512, 640],
            'name': 'RT-DETR (Weapon Detection)'
        },
    ]
    
    # Test each model
    all_results = {}
    
    for model_config in models_to_test:
        model_path = model_config['path']
        
        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_config['name']} - file not found: {model_path}\n")
            continue
        
        for imgsz in model_config['sizes']:
            key = f"{model_config['name']} @ {imgsz}"
            results = verify_flops_single_model(
                model_path,
                imgsz=imgsz,
                name=key
            )
            
            if results:
                all_results[key] = results
    
    # Final summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL MODELS")
    print("="*80)
    print(f"{'Model':<45} {'Size':<8} {'Params':<10} {'GFLOPs':<10}")
    print("-"*80)
    
    for key, results in all_results.items():
        parts = key.split('@')
        model_name = parts[0].strip()
        size = parts[1].strip() if len(parts) > 1 else 'N/A'
        
        params = f"{results.get('params_total', 0):.1f}M" if 'params_total' in results else 'N/A'
        gflops = f"{results.get('consensus', 0):.2f}" if 'consensus' in results else 'N/A'
        
        print(f"{model_name:<45} {size:<8} {params:<10} {gflops:<10}")
    
    print("="*80)
    print("\nVerification complete!")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if not libs['fvcore'] and not libs['thop']:
        print("⚠ Install measurement libraries for accurate GFLOPs:")
        print("   pip install fvcore thop")
    
    print("\nTo update your pipeline configuration:")
    print("1. Use 'consensus' values from above in your reports")
    print("2. Document which image size you're using for each stage")
    print("3. Clearly separate:")
    print("   - Per-inference GFLOPs (peak cost)")
    print("   - Per-frame GFLOPs (with tracking/frame-skip)")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()