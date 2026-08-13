"""
BioTrack-X Module 6: Inference Pipeline Adapter.

Drop-in replacement for tracker.run_cell_tracking() that routes through
the full BioTrack-X architecture instead of Trackastra.

Provides:
  run_biotrackx_inference(): Main entry point, compatible with main.py.
  save_uncertainty_maps():   Saves TTA uncertainty heatmaps as NumPy files.
  print_biotrackx_summary(): Prints structured architecture + inference report.
"""

import os
import time
import numpy as np
import networkx as nx
import torch
from typing import Tuple, Dict, Optional

from biotrack_x.model import BioTrackX


# Global model singleton (lazy-initialized)
_MODEL: Optional[BioTrackX] = None


def get_biotrackx_model(
    feature_dim: int = 128,
    n_heads: int = 8,
    n_layers: int = 3,
    n_max_cells: int = 64,
) -> BioTrackX:
    """
    Lazy-initializes and returns the BioTrackX model singleton.
    Uses the same singleton pattern as model_loader.get_trackastra_model().
    """
    global _MODEL
    if _MODEL is None:
        print("[BioTrackX] Initializing BioTrack-X model...")
        _MODEL = BioTrackX(
            feature_dim=feature_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            n_max_cells=n_max_cells,
            ffn_dim=512,
            erlang_alpha=2,
            erlang_beta=0.2,
            division_threshold=0.5,
            shift_radius=4,
        )
        _MODEL.eval()
    return _MODEL


def run_biotrackx_inference(
    masks: np.ndarray,
    model: Optional[BioTrackX] = None,
) -> Tuple[np.ndarray, nx.DiGraph]:
    """
    Drop-in replacement for tracker.run_cell_tracking().

    Args:
        masks: (T, H, W) integer cell mask array.
        model: Optional pre-initialized BioTrackX model.
    Returns:
        tracked_masks: (T, H, W) np.ndarray with consistent cell IDs.
        lineage_graph: networkx.DiGraph compatible with lineage.py.
    """
    if model is None:
        model = get_biotrackx_model()

    model.eval()

    print("[BioTrackX] Starting BioTrack-X inference pipeline...")
    t_start = time.time()

    with torch.no_grad():
        results = model.forward_inference(masks)

    t_elapsed = time.time() - t_start
    print(f"[BioTrackX] Inference completed in {t_elapsed:.2f}s")

    # Save uncertainty maps as .npy file for visualization
    save_uncertainty_maps(results["uncertainty_maps"])

    # Print summary
    print_biotrackx_summary(results)

    return results["tracked_masks"], results["lineage_graph"]


def save_uncertainty_maps(
    uncertainty_maps: np.ndarray,
    output_path: str = "biotrack_x_uncertainty.npy",
) -> None:
    """
    Saves TTA aleatoric uncertainty heatmaps to disk.

    Args:
        uncertainty_maps: (T, H, W) float32 uncertainty array.
        output_path: Output .npy file path.
    """
    np.save(output_path, uncertainty_maps)
    print(f"[BioTrackX] Saved uncertainty maps -> {output_path} "
          f"(shape: {uncertainty_maps.shape}, "
          f"max var: {uncertainty_maps.max():.4f})")


def print_biotrackx_summary(results: Dict) -> None:
    """Prints a structured BioTrack-X inference summary report."""
    erlang  = results["erlang_stats"]
    div     = results["div_predictions"]
    t_out   = results["transformer_out"]
    graph   = results["lineage_graph"]

    print("\n" + "=" * 60)
    print("         BioTrack-X Inference Summary Report")
    print("=" * 60)

    print("\n[Architecture]")
    print("  Model         : BioTrack-X (Novel Unified ST-GT)")
    print("  Encoder       : ResNet-18-style + TTA Uncertainty")
    print("  Transformer   : Spatio-Temporal Graph Transformer")
    print("  Bio Prior     : Erlang Cell-Cycle Prior")

    print("\n[Spatio-Temporal Transformer Output]")
    print(f"  Active queries: {t_out['n_active']}")
    print(f"  Track preds   : {t_out['track_preds'].shape}  [y_norm, x_norm, conf]")
    print(f"  Div probs     : {t_out['div_probs'].shape}")

    print("\n[Division Predictions (DivisionQueryHead)]")
    print(f"  Dividing cells: {div['n_dividing']}")
    if div['div_probs'].shape[0] > 0:
        print(f"  Mean P(div)   : {div['div_probs'].mean().item():.4f}")
        print(f"  Max  P(div)   : {div['div_probs'].max().item():.4f}")

    print("\n[Erlang Biological Cell-Cycle Prior]")
    print(f"  Tracked cells : {erlang['n_cells']}")
    print(f"  Mean cell age : {erlang['mean_age']:.1f} frames")
    print(f"  Max  cell age : {erlang['max_age']:.1f} frames")
    if "beta" in erlang:
        mean_cycle = 2 / erlang["beta"] if erlang["beta"] > 0 else float("inf")
        print(f"  Erlang beta   : {erlang['beta']:.4f} "
              f"(mean cycle ~= {mean_cycle:.1f} frames)")

    print("\n[Lineage Graph (networkx.DiGraph)]")
    print(f"  Nodes : {graph.number_of_nodes()}")
    print(f"  Edges : {graph.number_of_edges()}")
    roots = [n for n, d in graph.in_degree() if d == 0]
    print(f"  Roots : {len(roots)}")

    print("\n[TTA Aleatoric Uncertainty]")
    unc = results["uncertainty_maps"]
    print(f"  Uncertainty maps shape: {unc.shape}")
    print(f"  Mean var : {unc.mean():.6f}")
    print(f"  Max  var : {unc.max():.6f}")

    print("=" * 60 + "\n")
