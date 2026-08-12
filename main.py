"""
Main entry point script for Cell Track pipeline using Trackastra and Napari.
"""

import argparse
from data_loader import load_masks
from model_loader import get_trackastra_model
from tracker import run_cell_tracking, get_frame_cell_counts, print_cell_statistics
from morphology import extract_dataset_morphology, get_morphology_summary_stats
from lineage import detect_cell_events, build_lineage_family_trees
from behavior import compute_cell_kinematics, compute_population_behavior_summary
from visualize import (
    plot_frame_comparison,
    plot_morphology_distributions,
    plot_cell_trajectories,
    plot_motility_distributions,
    launch_napari_viewer
)
from config import DEVICE, DEFAULT_MODEL_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Cell Tracking, Lineage & Behavior Analysis Pipeline")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Pretrained Trackastra model name")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to run inference ('cpu' or 'cuda')")
    parser.add_argument("--subset", type=int, default=None, help="Process only first N time frames for testing")
    parser.add_argument("--save-plot", type=str, default=None, help="Filepath to save frame comparison figure")
    parser.add_argument("--export-csv", action="store_true", help="Export morphology, lineage, and behavior datasets to CSV")
    parser.add_argument("--export-video", action="store_true", help="Export time-lapse animated video GIF ('cell_tracking_video.gif')")
    parser.add_argument("--web-viewer", action="store_true", help="Build interactive HTML5 time-lapse video player dashboard ('cell_tracker_viewer.html')")
    parser.add_argument("--cluster", action="store_true", help="Perform unsupervised cell phenotyping & PCA clustering")
    parser.add_argument("--napari", action="store_true", help="Launch interactive Napari viewer after tracking")
    parser.add_argument("--no-show", action="store_true", help="Do not display Matplotlib pop-up windows")
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print("           Cell Tracking Pipeline Start           ")
    print("==================================================")

    # 1. Load Data
    masks = load_masks()
    if args.subset and args.subset > 0:
        print(f"[Main] Subsetting dataset to first {args.subset} frames...")
        masks = masks[:args.subset]

    total_frames = masks.shape[0]

    # 2. Load Model
    model = get_trackastra_model(model_name=args.model_name, device=args.device)

    # 3. Perform Cell Tracking (Module 2)
    tracked_masks, track_graph = run_cell_tracking(masks=masks, model=model)

    # 4. Extract Cell Morphology Metrics (Module 1)
    df_morphology = extract_dataset_morphology(tracked_masks)
    morph_stats = get_morphology_summary_stats(df_morphology)
    print(f"[Main] Cell Morphology Summary: {morph_stats}")

    # 5. Detect Cell Events & Lineage Trees (Module 3 & 4)
    df_events, event_summary = detect_cell_events(track_graph, total_frames=total_frames)
    family_trees = build_lineage_family_trees(track_graph)
    print(f"[Main] Lineage Summary: {event_summary}")

    # 6. Quantify Cell Behavior & Motility Analytics (Module 4)
    df_kinematics = compute_cell_kinematics(df_morphology)
    behavior_summary = compute_population_behavior_summary(df_kinematics)
    print(f"[Main] Population Behavior Summary: {behavior_summary}")

    # Print Cell Counts
    counts = get_frame_cell_counts(tracked_masks)
    print_cell_statistics(counts)

    # Export CSV datasets if requested
    if args.export_csv:
        df_morphology.to_csv("cell_morphology.csv", index=False)
        df_events.to_csv("cell_events.csv", index=False)
        df_kinematics.to_csv("cell_behavior.csv", index=False)
        print("[Main] Exported 'cell_morphology.csv', 'cell_events.csv', and 'cell_behavior.csv' successfully!")

    # Phenotyping & Clustering
    if args.cluster:
        from phenotyping import perform_cell_phenotyping
        _, pheno_summary = perform_cell_phenotyping(df_kinematics=df_kinematics, df_morphology=df_morphology)
        print(f"[Main] Unsupervised Phenotyping Summary: {pheno_summary}")

    # Video & Web Visualizer Exports
    if args.export_video:
        from visualize import export_animated_video
        export_animated_video(tracked_masks)

    if args.web_viewer:
        from visualize import export_web_visualizer
        export_web_visualizer()

    # 7. Static Plot Visualizations
    plot_frame_comparison(
        masks=masks,
        tracked_masks=tracked_masks,
        frame_idx=0,
        save_path=args.save_plot,
        show=not args.no_show
    )

    if not args.no_show:
        plot_morphology_distributions(df_morphology, show=True)
        plot_cell_trajectories(df_morphology, show=True)
        plot_motility_distributions(df_kinematics, show=True)

    # 8. Interactive Napari Viewer
    if args.napari:
        launch_napari_viewer(masks=masks, tracked_masks=tracked_masks)

    print("==================================================")
    print("==================================================")
    print("          Pipeline Completed Successfully!        ")
    print("==================================================")


if __name__ == "__main__":
    main()


