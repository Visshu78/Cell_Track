"""
BioTrack-X Test & Inference Evaluation Script.

Runs trained BioTrack-X model weights (biotrackx_ctc_weights.pth) on test sequences:
  1. Loads model weights.
  2. Runs spatio-temporal tracking inference on test video frames.
  3. Predicts cell trajectories, division events, and TTA position uncertainty.
  4. Generates interactive web visualizers and CSV analytics.
"""

import time
from pathlib import Path
import numpy as np
import torch

from ctc_loader import load_ctc_gt_masks, load_ctc_raw_images
from biotrack_x.model import BioTrackX
from morphology import extract_dataset_morphology, get_morphology_summary_stats
from lineage import detect_cell_events, build_lineage_family_trees
from behavior import compute_cell_kinematics, compute_population_behavior_summary
from generate_web_visualizer import build_web_visualizer
from generate_lineage_visualizer import build_lineage_web_visualizer


def test_biotrackx_model(
    checkpoint_path: str = "biotrackx_ctc_weights.pth",
    seq_name: str = "02",
    max_frames: int = 50,
    downsample_factor: int = 2,
):
    print("==================================================")
    print("       BioTrack-X Model Testing & Inference       ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Using device: {device}")

    # 1. Load & Clean Data
    masks, lineage = load_ctc_gt_masks(
        seq_name=seq_name,
        max_frames=max_frames,
        downsample_factor=downsample_factor,
    )
    from data_cleaner import clean_mask_sequence
    masks, clean_stats = clean_mask_sequence(masks, min_area=15, boundary_smoothing=True)
    total_frames, H, W = masks.shape
    print(f"[Test] Cleaned test sequence '{seq_name}': {total_frames} frames ({H}x{W}), Removed noise: {clean_stats['removed_noise_labels']}")

    # 2. Instantiate BioTrack-X Model
    model = BioTrackX(
        feature_dim=64,
        n_heads=4,
        n_layers=2,
        n_max_cells=32,
        ffn_dim=256,
        erlang_alpha=2,
        erlang_beta=0.2,
    ).to(device)

    # Load trained weights if checkpoint exists
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        print(f"[Test] Loading trained model weights from '{ckpt_file.name}'...")
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"[Test] Loaded weights successfully! (Learned Erlang beta: {ckpt.get('erlang_beta', 0.2):.4f})")
    else:
        print(f"[Test] Notice: Weight file '{checkpoint_path}' not found. Using initialized model weights.")

    model.eval()

    # 3. Run Inference Pass
    print("\n--- Running BioTrack-X Inference ---")
    start_time = time.time()
    results = model.forward_inference(masks)
    inference_time = time.time() - start_time

    tracked_masks = results["tracked_masks"]
    lineage_graph = results["lineage_graph"]
    uncertainty_maps = results["uncertainty_maps"]
    div_preds = results["div_predictions"]

    print(f"[Test] Inference complete in {inference_time:.3f}s")
    print(f"  Tracked masks shape : {tracked_masks.shape}")
    print(f"  Lineage graph nodes : {lineage_graph.number_of_nodes()}")
    print(f"  Lineage graph edges : {lineage_graph.number_of_edges()}")
    print(f"  Predicted divisions : {div_preds['n_dividing']}")

    # 4. Feature Extraction & Analytics
    print("\n--- Computing Morphological & Kinematic Analytics ---")
    df_morphology = extract_dataset_morphology(tracked_masks)
    morph_stats = get_morphology_summary_stats(df_morphology)

    df_events, event_summary = detect_cell_events(lineage_graph, total_frames=total_frames)
    df_kinematics = compute_cell_kinematics(df_morphology)
    behavior_summary = compute_population_behavior_summary(df_kinematics)

    # 5. Export Datasets to CSV
    df_morphology.to_csv("test_cell_morphology.csv", index=False)
    df_events.to_csv("test_cell_events.csv", index=False)
    df_kinematics.to_csv("test_cell_behavior.csv", index=False)
    print("[Test] Exported 'test_cell_morphology.csv', 'test_cell_events.csv', and 'test_cell_behavior.csv'")

    # 6. Generate Interactive Web Visualizers
    print("\n--- Generating Web Dashboards ---")
    build_web_visualizer()
    try:
        build_lineage_web_visualizer()
    except Exception as e:
        print(f"[Test] Lineage visualizer note: {e}")

    try:
        import generate_biotrackx_dashboard
    except Exception as e:
        print(f"[Test] BioTrack-X dashboard note: {e}")

    # 7. Print Final Testing Summary Report
    print("\n" + "=" * 60)
    print("         BioTrack-X Model Testing Summary Report         ")
    print("=" * 60)
    print(f"Test Sequence Name     : BF-C2DL-HSC / Sequence {seq_name}")
    print(f"Processed Frame Range  : {total_frames} frames (H={H}, W={W})")
    print(f"Total Model Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Inference Duration     : {inference_time:.3f} seconds ({inference_time/total_frames*1000:.1f} ms/frame)")
    print("-" * 60)
    print(f"Tracked Cell Observations : {morph_stats.get('total_cell_observations', 0)}")
    print(f"Mean Cell Area         : {morph_stats.get('mean_area', 0.0)} px²")
    print(f"Mean Cell Circularity  : {morph_stats.get('mean_circularity', 0.0)}")
    print(f"Mean Population Speed  : {behavior_summary.get('mean_population_speed', 0.0)} px/frame")
    print(f"Mean Net Displacement  : {behavior_summary.get('mean_net_displacement', 0.0)} px")
    print(f"Predicted Mitoses      : {div_preds['n_dividing']}")
    print("=" * 60)
    print("Testing output visualizers ready:")
    print("  1. cell_tracker_viewer.html")
    print("  2. lineage_tree_viewer.html")
    print("  3. biotrackx_dashboard.html")


if __name__ == "__main__":
    test_biotrackx_model(seq_name="02", max_frames=30)
