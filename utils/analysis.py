"""
Analysis Utilities for Model Summary and GFLOPs Measurement

"""

import torch
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple


def get_model_summary(model, input_size: Tuple = (1, 3, 640, 640), device: str = 'cuda:0') -> Dict[str, Any]:
    """
    Get comprehensive summary of the model including GFLOPs.
    
    Args:
        model: PyTorch model to analyze
        input_size: Tuple of (batch, channels, height, width)
        device: Device string ('cuda:0' for A100)
    
    Returns:
        dict: Model summary with params, GFLOPs, memory usage
    
    Usage:
        from utils.analysis import get_model_summary
        summary = get_model_summary(model, (1, 3, 640, 640), 'cuda:0')
        print(f"GFLOPs: {summary['gflops']:.2f}")
    """
    summary = {
        'total_params': 0,
        'trainable_params': 0,
        'input_size': input_size,
        'device': device,
        'gflops': None,
        'memory_mb': None
    }
    
    if model is None:
        return summary
    
    try:
        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary['total_params'] = total_params
        summary['trainable_params'] = trainable_params
        summary['params_m'] = total_params / 1e6  # Millions
        
        # GFLOPs measurement using thop
        try:
            from thop import profile, clever_format
            
            # Move model to device
            model_device = next(model.parameters()).device
            dummy_input = torch.randn(*input_size).to(model_device)
            
            with torch.no_grad():
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            
            summary['gflops'] = flops / 1e9
            summary['flops_formatted'], summary['params_formatted'] = clever_format([flops, params], "%.3f")
            
        except ImportError:
            print("[WARN] thop not installed. Install with: pip install thop")
            summary['gflops'] = None
        except Exception as e:
            print(f"[WARN] GFLOPs calculation failed: {e}")
            summary['gflops'] = None
        
        # Memory footprint estimation
        try:
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
            summary['memory_mb'] = (param_memory + buffer_memory) / (1024 ** 2)
        except Exception as e:
            summary['memory_mb'] = None
        
    except Exception as e:
        print(f"[WARN] Could not get model summary: {e}")
    
    return summary


def measure_inference_time(model, input_size: Tuple = (1, 3, 640, 640), 
                           device: str = 'cuda:0', 
                           warmup: int = 10, 
                           iterations: int = 100) -> Dict[str, float]:
    """
    Measure model inference time on A100.
    
    Args:
        model: PyTorch model
        input_size: Input tensor shape
        device: Device to run on ('cuda:0' for A100)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
    
    Returns:
        dict: Timing statistics (mean, std, fps)
    
    Usage:
        timing = measure_inference_time(model, device='cuda:0')
        print(f"FPS: {timing['fps']:.1f}")
    """
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize before timing
    if 'cuda' in device:
        torch.cuda.synchronize()
    
    # Timed iterations
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            if 'cuda' in device:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000.0 / np.mean(times),
        'iterations': iterations
    }


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage (for A100 monitoring).
    
    Returns:
        dict: Memory usage in MB
    
    Usage:
        mem = get_gpu_memory_usage()
        print(f"GPU Memory: {mem['allocated_mb']:.0f} / {mem['total_mb']:.0f} MB")
    """
    if not torch.cuda.is_available():
        return {'allocated_mb': 0, 'reserved_mb': 0, 'total_mb': 0}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
        'reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
        'total_mb': torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
        'device_name': torch.cuda.get_device_name(0)
    }


def print_model_summary(summary: Dict[str, Any]):
    """Print model summary in a readable format."""
    if summary:
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Total Parameters:     {summary.get('total_params', 0):,}")
        print(f"Trainable Parameters: {summary.get('trainable_params', 0):,}")
        print(f"Parameters (M):       {summary.get('params_m', 0):.2f}")
        if summary.get('gflops'):
            print(f"GFLOPs:               {summary['gflops']:.2f}")
        if summary.get('memory_mb'):
            print(f"Model Memory (MB):    {summary['memory_mb']:.2f}")
        print(f"Input Size:           {summary.get('input_size', 'N/A')}")
        print(f"Device:               {summary.get('device', 'N/A')}")
        print("=" * 60 + "\n")


def compare_architectures(models: Dict[str, torch.nn.Module], 
                          input_size: Tuple = (1, 3, 640, 640),
                          device: str = 'cuda:0') -> Dict[str, Dict]:
    """
    Compare multiple architectures (for RQ2: RT-DETR vs EfficientViT).
    
    Args:
        models: Dict mapping architecture name to model
        input_size: Input tensor shape
        device: Device for inference
    
    Returns:
        dict: Comparison results for each architecture
    
    Usage:
        results = compare_architectures({
            'rt_detr': rt_detr_model,
            'efficientvit': efficientvit_model
        })
    """
    results = {}
    
    for name, model in models.items():
        print(f"\n[ANALYSIS] Analyzing {name}...")
        
        # Get summary
        summary = get_model_summary(model, input_size, device)
        
        # Measure timing
        timing = measure_inference_time(model, input_size, device)
        
        results[name] = {
            **summary,
            **timing
        }
        
        print(f"  - Params: {summary['params_m']:.2f}M")
        print(f"  - GFLOPs: {summary.get('gflops', 'N/A')}")
        print(f"  - FPS: {timing['fps']:.1f}")
    
    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    """
    Test analysis utilities on A100.
    
    Run:
        python analysis.py
    """
    print("=" * 60)
    print("ANALYSIS UTILITIES TEST")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        mem = get_gpu_memory_usage()
        print(f"\nGPU: {mem['device_name']}")
        print(f"Total Memory: {mem['total_mb']:.0f} MB")
        print(f"Allocated: {mem['allocated_mb']:.0f} MB")
    else:
        print("\nNo GPU available, using CPU")
    
    # Test with a simple model
    print("\nTesting with simple ConvNet...")
    
    class SimpleConvNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(128, 2)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleConvNet()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    summary = get_model_summary(model, (1, 3, 640, 640), device)
    print_model_summary(summary)
    
    if torch.cuda.is_available():
        timing = measure_inference_time(model, (1, 3, 640, 640), device)
        print(f"Inference Time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
        print(f"FPS: {timing['fps']:.1f}")
    
    print("\n✓ Analysis utilities working correctly")
