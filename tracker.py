"""
Tracking pipeline module using Trackastra model.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
try:
    from trackastra.model import Trackastra
except ImportError:
    class Trackastra:
        pass
from model_loader import get_trackastra_model
from data_loader import load_masks


def run_cell_tracking(
    masks: np.ndarray,
    images: Optional[np.ndarray] = None,
    model: Optional[Trackastra] = None,
    mode: str = "greedy"
) -> Tuple[np.ndarray, any]:
    """
    Runs Trackastra cell tracking prediction on segmentation masks.
    
    Returns:
        tracked_masks: np.ndarray with consistent cell labels over time frames.
        track_graph: tracking graph structure containing links/nodes.
    """
    if model is None:
        model = get_trackastra_model()

    print("[Tracker] Running cell tracking inference...")
    
    # If raw intensity images are not provided, use masks as image input for Trackastra
    if images is None:
        images = masks

    # Trackastra.track returns (track_graph, tracked_masks) or (tracked_masks, track_graph)
    res1, res2 = model.track(imgs=images, masks=masks, mode=mode)


    if isinstance(res1, np.ndarray):
        tracked_masks, track_graph = res1, res2
    else:
        track_graph, tracked_masks = res1, res2

    print(f"[Tracker] Cell tracking completed! Tracked shape: {tracked_masks.shape}")
    return tracked_masks, track_graph




def get_frame_cell_counts(tracked_masks: np.ndarray) -> Dict[int, int]:
    """
    Computes count of tracked unique cells per time frame.
    """
    counts = {}
    for t in range(tracked_masks.shape[0]):
        labels = np.unique(tracked_masks[t])
        num_cells = len(labels[labels > 0])
        counts[t] = num_cells
    return counts


def print_cell_statistics(counts: Dict[int, int]) -> None:
    """
    Prints per-frame cell count summary.
    """
    print("--- Per-Frame Tracked Cell Statistics ---")
    for frame, count in counts.items():
        print(f"Frame {frame:2d}: {count} tracked cells")


if __name__ == "__main__":
    masks_data = load_masks()
    # Test tracking on a small subset (e.g. first 5 frames) for fast verification
    tracked_sub, graph_sub = run_cell_tracking(masks_data[:5])
    stats = get_frame_cell_counts(tracked_sub)
    print_cell_statistics(stats)
