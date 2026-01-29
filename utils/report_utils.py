"""
Reporting and plotting utilities for pipeline evaluation.

Public API:
- ensure_dir
- write_detailed_report
- write_combined_table
- write_per_class_table
- plot_results
- print_summary
- write_ultralytics_comparison (no-op; kept for backward compatibility)
"""

import os
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> str:
    """Create directory if it doesn't exist and return it."""
    os.makedirs(p, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# Detailed Report
# -----------------------------------------------------------------------------
def write_detailed_report(results, output_path, timestamp, model_name, class_names):
    """
    Write a detailed text report with metrics from the Custom Evaluator only.

    Args:
        results: List[dict] result dictionaries from experiments
        output_path: str path to save the report
        timestamp: str timestamp string
        model_name: str pipeline model name
        class_names: List[str] names of classes
    """
    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("MULTI-CONFIGURATION PIPELINE EVALUATION - DETAILED REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Pipeline Model: {model_name}\n")
        f.write(f"Classes: {', '.join(class_names)}\n")
        f.write(f"Total experiments: {len(results)}\n")
        f.write("Evaluation \n\n")

        for i, r in enumerate(results):
            f.write(f"\n{'=' * 100}\n")
            f.write(f"EXPERIMENT {i + 1}: {r.get('experiment_name', 'N/A')}\n")
            f.write(f"{'=' * 100}\n")
            f.write("Configuration:\n")
            f.write(f"  Model: {model_name}\n")
            f.write(f"  Tracker: {'Enabled' if r.get('tracker') else 'Disabled'}\n")
            f.write(f"  Frame Gap: {r.get('frame_gap')}\n")
            f.write(f"  Avg crops/frame: {r.get('avg_crops_per_frame', 0):.2f}\n")
            f.write(f"  NMS removal rate: {r.get('nms_removal_rate', 0):.1f}%\n")
            f.write(f"  Pred/GT ratio: {r.get('pred_gt_ratio', 0):.2f}\n\n")

            # Stage-2
            f.write(f"{'-' * 100}\n")
            f.write("STAGE 2: PERSON DETECTION\n")
            f.write(f"{'-' * 100}\n")
            #f.write(f"  Overall Coverage: {r.get('coverage', 0):.4f}\n")
            f.write(f"  Latency: {r.get('stage2_latency_ms', 0):.2f} ms\n")
            f.write(f"  GFLOPs: {r.get('stage2_gflops', 0):.2f}\n\n")
            # f.write("  Per-Class Coverage:\n")
            # for cls_name in class_names:
            #     cls_cov = r.get('coverage_by_class', {}).get(cls_name, 0.0)
            #     f.write(f"    {cls_name:.<20} {cls_cov:.4f}\n")
            f.write("\n")

            # Stage-3
            f.write(f"{'-' * 100}\n")
            f.write("STAGE 3: WEAPON DETECTION\n")
            f.write(f"{'-' * 100}\n")
            f.write(f"  Latency per crop: {r.get('stage3_latency_ms', 0):.2f} ms\n")
            f.write(f"  GFLOPs per inference: {r.get('stage3_gflops', 0):.2f}\n")
            f.write(f"  Total GFLOPs per frame: {r.get('stage3_total_gflops', 0):.2f}\n\n")

            # Custom evaluator (overall + per-class)
            f.write(f"{'-' * 100}\n")
            f.write("PIPELINE METRICS\n")
            f.write(f"{'-' * 100}\n")
            f.write("  Overall Metrics:\n")
            f.write(f"    mAP50:     {r.get('pipeline_map50', 0):.4f}\n")
            f.write(f"    Recall:    {r.get('pipeline_recall', 0):.4f}\n")
            f.write(f"    Precision: {r.get('pipeline_precision', 0):.4f}\n")
            f.write(f"    F1 Score:  {r.get('pipeline_f1', 0):.4f}\n")
            f.write(f"    TP: {r.get('pipeline_tp', 0)}, FP: {r.get('pipeline_fp', 0)}, FN: {r.get('pipeline_fn', 0)}\n\n")

            f.write("  Per-Class Metrics:\n")
            for cls_name in class_names:
                cls_map50 = r.get('pipeline_map50_by_class', {}).get(cls_name, 0.0)
                cls_rec = r.get('pipeline_recall_by_class', {}).get(cls_name, 0.0)
                cls_prec = r.get('pipeline_precision_by_class', {}).get(cls_name, 0.0)
                cls_f1 = r.get('pipeline_f1_by_class', {}).get(cls_name, 0.0)
                cls_tp = r.get('pipeline_tp_by_class', {}).get(cls_name, 0)
                cls_fp = r.get('pipeline_fp_by_class', {}).get(cls_name, 0)
                cls_fn = r.get('pipeline_fn_by_class', {}).get(cls_name, 0)

                f.write(f"    {cls_name}:\n")
                f.write(f"      mAP50:     {cls_map50:.4f}\n")
                f.write(f"      Recall:    {cls_rec:.4f}\n")
                f.write(f"      Precision: {cls_prec:.4f}\n")
                f.write(f"      F1 Score:  {cls_f1:.4f}\n")
                f.write(f"      TP: {cls_tp}, FP: {cls_fp}, FN: {cls_fn}\n")
            f.write("\n")

            # Performance summary
            f.write(f"{'-' * 100}\n")
            f.write("PERFORMANCE METRICS\n")
            f.write(f"{'-' * 100}\n")
            f.write(f"  Latency (pipeline): {r.get('latency_ms', 0):.2f} ms\n")
            f.write(f"  FPS: {r.get('fps', 0):.1f}\n")
            f.write(f"  Total GFLOPs (per frame): {r.get('gflops', 0):.2f}\n")
            f.write(f"  Stage-2 GFLOPs: {r.get('stage2_gflops', 0):.2f}\n")
            f.write(f"  Stage-3 GFLOPs (per crop): {r.get('stage3_gflops', 0):.2f}\n")
            f.write(f"  Stage-3 GFLOPs (total): {r.get('stage3_total_gflops', 0):.2f}\n\n")


# -----------------------------------------------------------------------------
# Combined Table (Custom Evaluator only)
# -----------------------------------------------------------------------------
def write_combined_table(results, output_path, model_name):
    """
    Write a single combined table of Custom Evaluator metrics.

    Args:
        results: List[dict]
        output_path: str
        model_name: str
    """
    headers = [
        "Exp", "Model", "Track", "Gap",
        "mAP50", "Recall", "Prec", "F1",
        "TP", "FP", "FN",
        "Lat(ms)", "FPS", "GFLOPs", "NMS%", "P/GT"
    ]
    widths = [4, 50, 6, 4, 8, 8, 8, 8, 6, 6, 6, 9, 7, 9, 7]

    with open(output_path, "w") as f:
        f.write("=" * 131 + "\n")
        f.write("METRICS\n")
        f.write("=" * 131 + "\n\n")
        f.write("  ".join(h.ljust(w) for h, w in zip(headers, widths)) + "\n")
        f.write("-" * (sum(widths) + 2 * (len(widths) - 1)) + "\n")

        for i, r in enumerate(results, 1):
            row = [
                str(i),
                (model_name or "")[:50],
                "Yes" if r.get('tracker') else "No",
                str(r.get('frame_gap', "")),
                f"{r.get('pipeline_map50', 0):.4f}",
                f"{r.get('pipeline_recall', 0):.4f}",
                f"{r.get('pipeline_precision', 0):.4f}",
                f"{r.get('pipeline_f1', 0):.4f}",
                str(r.get('pipeline_tp', 0)),
                str(r.get('pipeline_fp', 0)),
                str(r.get('pipeline_fn', 0)),
                f"{r.get('latency_ms', 0):.1f}",
                f"{r.get('fps', 0):.1f}",
                f"{r.get('gflops', 0):.1f}",
                f"{r.get('nms_removal_rate', 0):.1f}",
                #f"{r.get('pred_gt_ratio', 0):.2f}",
            ]
            f.write("  ".join(v.ljust(w) for v, w in zip(row, widths)) + "\n")


# -----------------------------------------------------------------------------
# Per-Class Table (Custom Evaluator only)
# -----------------------------------------------------------------------------
def write_per_class_table(results, output_path, model_name, class_names):
    """
    Write per-class metrics table (Evaluator).
    """
    with open(output_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("PER-CLASS PIPELINE METRICS\n")
        f.write("=" * 100 + "\n\n")

        for cls_name in class_names:
            f.write(f"\n{'-' * 100}\n")
            f.write(f"{cls_name.upper()} CLASS METRICS\n")
            f.write(f"{'-' * 100}\n")

            headers = ["Exp", "Model", "Track", "Gap", "mAP50", "Recall", "Prec", "F1", "TP", "FP", "FN"]
            widths = [4, 14, 6, 4, 8, 8, 8, 8, 6, 6, 6]

            f.write("  ".join(h.ljust(w) for h, w in zip(headers, widths)) + "\n")
            f.write("-" * (sum(widths) + 2 * (len(widths) - 1)) + "\n")

            for i, r in enumerate(results, 1):
                cls_map50 = r.get('pipeline_map50_by_class', {}).get(cls_name, 0.0)
                cls_rec = r.get('pipeline_recall_by_class', {}).get(cls_name, 0.0)
                cls_prec = r.get('pipeline_precision_by_class', {}).get(cls_name, 0.0)
                cls_f1 = r.get('pipeline_f1_by_class', {}).get(cls_name, 0.0)
                cls_tp = r.get('pipeline_tp_by_class', {}).get(cls_name, 0)
                cls_fp = r.get('pipeline_fp_by_class', {}).get(cls_name, 0)
                cls_fn = r.get('pipeline_fn_by_class', {}).get(cls_name, 0)

                row = [
                    str(i),
                    (model_name or "")[:50],
                    "Yes" if r.get('tracker') else "No",
                    str(r.get('frame_gap', "")),
                    f"{cls_map50:.4f}",
                    f"{cls_rec:.4f}",
                    f"{cls_prec:.4f}",
                    f"{cls_f1:.4f}",
                    str(cls_tp),
                    str(cls_fp),
                    str(cls_fn),
                ]
                f.write("  ".join(v.ljust(w) for v, w in zip(row, widths)) + "\n")
            f.write("\n")


# -----------------------------------------------------------------------------
# Plots (Custom Evaluator only)
# -----------------------------------------------------------------------------
def plot_results(results, output_dir, class_names):
    """
    Generate performance plots (Custom Evaluator only).
    """
    plot_dir = os.path.join(output_dir, "plots")
    ensure_dir(plot_dir)

    tracker_results = [r for r in results if r.get('tracker')]
    no_tracker = next((r for r in results if not r.get('tracker')), None)

    if not tracker_results:
        print("[WARN] No tracker results to plot")
        return

    frame_gaps = [r.get('frame_gap') for r in tracker_results]
    custom_maps = [r.get('pipeline_map50', 0) for r in tracker_results]
    fps_values = [r.get('fps', 0) for r in tracker_results]
    gflops_values = [r.get('gflops', 0) for r in tracker_results]
    pred_gt_ratios = [r.get('pred_gt_ratio', 0) for r in tracker_results]

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # 1) Overall mAP50
    ax1 = axes[0, 0]
    ax1.plot(frame_gaps, custom_maps, '.-', markersize=8, linewidth=2, label='Custom')
    if no_tracker:
        ax1.axhline(y=no_tracker.get('pipeline_map50', 0), color='gray', linestyle=':', linewidth=2, alpha=0.7,
                    label='No Tracker')
    ax1.set_xlabel('Frame Gap'); ax1.set_ylabel('Pipeline mAP50'); ax1.set_title('Overall mAP50 vs Frame Gap')
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9)

    # 2) Per-class mAP50
    ax2 = axes[0, 1]
    for i, cls_name in enumerate(class_names):
        cls_vals = [r.get('pipeline_map50_by_class', {}).get(cls_name, 0.0) for r in tracker_results]
        ax2.plot(frame_gaps, cls_vals, '.-', markersize=8, linewidth=2, label=cls_name, color=colors[i % len(colors)])
        if no_tracker:
            ax2.axhline(y=no_tracker.get('pipeline_map50_by_class', {}).get(cls_name, 0.0),
                        color=colors[i % len(colors)], linestyle=':', linewidth=2, alpha=0.6)
    ax2.set_xlabel('Frame Gap'); ax2.set_ylabel('mAP50'); ax2.set_title('Per-Class mAP50 vs Frame Gap')
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=9)

    # 3) FPS
    ax3 = axes[1, 0]
    ax3.plot(frame_gaps, fps_values, '.-', markersize=8, linewidth=2, label='Tracker')
    if no_tracker:
        ax3.axhline(y=no_tracker.get('fps', 0), color='gray', linestyle='--', linewidth=2, label='No Tracker')
    ax3.set_xlabel('Frame Gap'); ax3.set_ylabel('FPS'); ax3.set_title('Speed (FPS) vs Frame Gap')
    ax3.grid(True, alpha=0.3); ax3.legend(fontsize=9)

    # 4) Pred/GT ratio
    ax4 = axes[1, 1]
    ax4.plot(frame_gaps, pred_gt_ratios, '.-', markersize=8, linewidth=2, label='Tracker')
    ax4.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (1.0)')
    ax4.axhline(y=1.2, color='orange', linestyle=':', linewidth=1, alpha=0.6, label='Upper bound (1.2)')
    if no_tracker:
        ax4.axhline(y=no_tracker.get('pred_gt_ratio', 0), color='gray', linestyle='--', linewidth=2,
                    label='No Tracker')
    ax4.set_xlabel('Frame Gap'); ax4.set_ylabel('Pred/GT Ratio'); ax4.set_title('Prediction Ratio vs Frame Gap')
    ax4.grid(True, alpha=0.3); ax4.legend(fontsize=9)

    # 5) Speed–Accuracy
    ax5 = axes[2, 0]
    ax5.plot(fps_values, custom_maps, 'o-', markersize=8, linewidth=2, label='Tracker')
    for i, gap in enumerate(frame_gaps):
        ax5.annotate(f'gap={gap}', (fps_values[i], custom_maps[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    if no_tracker:
        ax5.plot(no_tracker.get('fps', 0), no_tracker.get('pipeline_map50', 0), '*', markersize=14, label='No Tracker')
    ax5.set_xlabel('FPS'); ax5.set_ylabel('Pipeline mAP50'); ax5.set_title('Speed–Accuracy Tradeoff')
    ax5.grid(True, alpha=0.3); ax5.legend(fontsize=9)

    # 6) GFLOPs
    ax6 = axes[2, 1]
    ax6.plot(frame_gaps, gflops_values, '.-', markersize=8, linewidth=2, label='Tracker')
    if no_tracker:
        ax6.axhline(y=no_tracker.get('gflops', 0), color='gray', linestyle='--', linewidth=2, label='No Tracker')
    ax6.set_xlabel('Frame Gap'); ax6.set_ylabel('GFLOPs'); ax6.set_title('Computational Cost vs Frame Gap')
    ax6.grid(True, alpha=0.3); ax6.legend(fontsize=9)

    plt.suptitle('Pipeline Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(plot_dir, 'performance_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved performance analysis to: {out_path}")


# -----------------------------------------------------------------------------
# Console Summary (Custom Evaluator only)
# -----------------------------------------------------------------------------
def print_summary(results, output_dir):
    """
    Print a concise console summary of all experiments (Custom Evaluator only).
    """
    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE - MULTI-CONFIGURATION PIPELINE")
    print("=" * 100)

    best_map = max(results, key=lambda x: x.get('pipeline_map50', 0))
    best_fps = max(results, key=lambda x: x.get('fps', 0))
    best_tradeoff = max(results, key=lambda x: x.get('pipeline_map50', 0) * x.get('fps', 0) / 100.0)

    # Best mAP50
    print(f"\n{'-' * 100}")
    print("BEST mAP50 Configuration:")
    print(f"  mAP50: {best_map.get('pipeline_map50', 0):.4f}")
    print(f"  Tracker: {'Yes' if best_map.get('tracker') else 'No'}, Frame Gap: {best_map.get('frame_gap')}")
    print(f"  Precision: {best_map.get('pipeline_precision', 0):.4f}, Recall: {best_map.get('pipeline_recall', 0):.4f}")
    print(f"  FPS: {best_map.get('fps', 0):.1f}, Latency: {best_map.get('latency_ms', 0):.1f} ms")

    # Best FPS
    print(f"\n{'-' * 100}")
    print("BEST FPS Configuration:")
    print(f"  FPS: {best_fps.get('fps', 0):.1f}, Latency: {best_fps.get('latency_ms', 0):.1f} ms")
    print(f"  Tracker: {'Yes' if best_fps.get('tracker') else 'No'}, Frame Gap: {best_fps.get('frame_gap')}")
    print(f"  mAP50: {best_fps.get('pipeline_map50', 0):.4f}")

    # Best tradeoff
    print(f"\n{'-' * 100}")
    print("BEST Speed–Accuracy Tradeoff:")
    print(f"  mAP50: {best_tradeoff.get('pipeline_map50', 0):.4f}, FPS: {best_tradeoff.get('fps', 0):.1f}")
    print(f"  Tracker: {'Yes' if best_tradeoff.get('tracker') else 'No'}, Frame Gap: {best_tradeoff.get('frame_gap')}")
    print(f"  Precision: {best_tradeoff.get('pipeline_precision', 0):.4f}, Recall: {best_tradeoff.get('pipeline_recall', 0):.4f}")

    # Baseline (no tracker)
    baseline = next((r for r in results if not r.get('tracker')), None)
    if baseline:
        print(f"\n{'-' * 100}")
        print("BASELINE (No Tracker):")
        print(f"  mAP50: {baseline.get('pipeline_map50', 0):.4f}")
        print(f"  FPS: {baseline.get('fps', 0):.1f}")
        print(f"  Pred/GT Ratio: {baseline.get('pred_gt_ratio', 0):.2f}")

    print(f"\n{'-' * 100}")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 100 + "\n")


# -----------------------------------------------------------------------------
# Back-compat: No-op Ultralytics comparison writer
# -----------------------------------------------------------------------------
def write_ultralytics_comparison(results, output_path, model_name, class_names):
    """
    Deprecated. Kept for backward compatibility. Writes a short note and returns.
    """
    with open(output_path, "w") as f:
        f.write("Ultralytics validation & comparison have been removed from the reporting pipeline.\n")
        f.write("This file is generated only to keep legacy calls from failing.\n")
