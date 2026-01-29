"""
Utility package for weapon detection pipeline.
Contains modular components for box operations, FLOPs calculation, and reporting.
"""

from .box_utils import (
    load_yolo_labels,
    write_yolo_labels,
    square_scale_clip_xyxy,
    remap_crop_to_frame,
    compute_iou_yolo,
    compute_iou_xyxy,
    apply_nms,
    apply_global_nms,
    filter_overlapping_crops,
    filter_suspicious_predictions,
    filter_low_confidence_predictions,
    get_prediction_stats
)

from .flops_utils import (
    get_stage2_module_handle,
    compute_flops_gflops,
    calculate_pipeline_flops,
    print_flops_summary
)

from .report_utils import (
    ensure_dir,
    write_detailed_report,
    write_combined_table,
    write_per_class_table,
    plot_results,
    print_summary
)

__version__ = "1.0.0"
__all__ = [
    # Box utilities
    'load_yolo_labels',
    'write_yolo_labels',
    'square_scale_clip_xyxy',
    'remap_crop_to_frame',
    'compute_iou_yolo',
    'compute_iou_xyxy',
    'apply_nms',
    'apply_global_nms',
    'filter_overlapping_crops',
    'filter_suspicious_predictions',
    'filter_low_confidence_predictions',
    'get_prediction_stats',
    
    # FLOPs utilities
    'get_stage2_module_handle',
    'compute_flops_gflops',
    'calculate_pipeline_flops',
    'print_flops_summary',
    
    # Report utilities
    'ensure_dir',
    'write_detailed_report',
    'write_combined_table',
    'write_per_class_table',
    'plot_results',
    'print_summary',
]