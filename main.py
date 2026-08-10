"""
Main entry point script for Cell Track pipeline using Trackastra and Napari.
"""

import argparse
from data_loader import load_masks
from model_loader import get_trackastra_model
from tracker import run_cell_tracking, get_frame_cell_counts, print_cell_statistics
from visualize import plot_frame_comparison, launch_napari_viewer
from config import DEVICE, DEFAULT_MODEL_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Cell Tracking with Trackastra & Napari")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Pretrained Trackastra model name")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device to run inference ('cpu' or 'cuda')")
    parser.add_argument("--subset", type=int, default=None, help="Process only first N time frames for testing")
    parser.add_argument("--save-plot", type=str, default=None, help="Filepath to save frame comparison figure")
    parser.add_argument("--napari", action="store_true", help="Launch interactive Napari viewer after tracking")
    parser.add_argument("--no-show", action="store_true", help="Do not display Matplotlib pop-up window")
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

    # 2. Load Model
    model = get_trackastra_model(model_name=args.model_name, device=args.device)

    # 3. Perform Cell Tracking
    tracked_masks, track_graph = run_cell_tracking(masks=masks, model=model)

    # 4. Analyze & Print Statistics
    counts = get_frame_cell_counts(tracked_masks)
    print_cell_statistics(counts)

    # 5. Static Plot Comparison
    plot_frame_comparison(
        masks=masks,
        tracked_masks=tracked_masks,
        frame_idx=0,
        save_path=args.save_plot,
        show=not args.no_show
    )

    # 6. Interactive Napari Viewer
    if args.napari:
        launch_napari_viewer(masks=masks, tracked_masks=tracked_masks)

    print("==================================================")
    print("          Pipeline Completed Successfully!        ")
    print("==================================================")


if __name__ == "__main__":
    main()
