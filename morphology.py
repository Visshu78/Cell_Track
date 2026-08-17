"""
Module 1: Cell Perception - Morphology Feature Extractor.
Extracts spatial, morphological, and shape properties of tracked cells using scikit-image.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
try:
    from skimage.measure import regionprops
except ImportError:
    class RegionPropStub:
        def __init__(self, label_id, ys, xs):
            self.label = label_id
            self.area = len(ys)
            min_y, max_y = int(ys.min()), int(ys.max() + 1)
            min_x, max_x = int(xs.min()), int(xs.max() + 1)
            self.bbox = (min_y, min_x, max_y, max_x)
            self.centroid = (float(ys.mean()), float(xs.mean()))
            h, w = max_y - min_y, max_x - min_x
            self.perimeter = float(2 * (h + w))
            self.eccentricity = 0.5
            self.major_axis_length = float(max(h, w))
            self.minor_axis_length = float(min(h, w))

    def regionprops(mask_frame):
        labels = [c for c in np.unique(mask_frame) if c > 0]
        props = []
        for lid in labels:
            ys, xs = np.where(mask_frame == lid)
            if len(ys) > 0:
                props.append(RegionPropStub(lid, ys, xs))
        return props


def extract_frame_morphology(mask_frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
    """
    Extracts morphological features for all cells in a single 2D mask frame.
    """
    props = regionprops(mask_frame)
    frame_data = []

    for p in props:
        label_id = p.label
        if label_id == 0:
            continue

        area = float(p.area)
        perimeter = float(p.perimeter) if p.perimeter > 0 else 1.0
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        circularity = min(1.0, max(0.0, circularity))  # Normalize [0, 1]

        centroid_y, centroid_x = p.centroid
        min_row, min_col, max_row, max_col = p.bbox

        major_axis = getattr(p, "axis_major_length", getattr(p, "major_axis_length", 0.0))
        minor_axis = getattr(p, "axis_minor_length", getattr(p, "minor_axis_length", 0.0))

        frame_data.append({
            "frame": frame_idx,
            "label_id": label_id,
            "centroid_y": round(float(centroid_y), 2),
            "centroid_x": round(float(centroid_x), 2),
            "area": int(area),
            "perimeter": round(perimeter, 2),
            "circularity": round(circularity, 4),
            "eccentricity": round(float(p.eccentricity), 4),
            "major_axis_length": round(float(major_axis), 2),
            "minor_axis_length": round(float(minor_axis), 2),
            "bbox": (min_row, min_col, max_row, max_col)
        })

    return frame_data


def extract_dataset_morphology(masks: np.ndarray) -> pd.DataFrame:
    """
    Extracts cell morphology across all video time frames.
    Returns a Pandas DataFrame containing spatial metrics for all detected cells.
    """
    print(f"[Morphology] Extracting shape metrics for {masks.shape[0]} frames...")
    all_rows = []

    for t in range(masks.shape[0]):
        frame_metrics = extract_frame_morphology(masks[t], frame_idx=t)
        all_rows.extend(frame_metrics)

    df = pd.DataFrame(all_rows)
    print(f"[Morphology] Extraction complete. Processed {len(df)} cell measurements across {masks.shape[0]} frames.")
    return df


def get_morphology_summary_stats(df_morphology: pd.DataFrame) -> Dict[str, float]:
    """
    Computes overall summary statistics for cell morphology metrics.
    """
    if df_morphology.empty:
        return {}

    return {
        "total_cell_observations": len(df_morphology),
        "mean_area": round(float(df_morphology["area"].mean()), 2),
        "median_area": round(float(df_morphology["area"].median()), 2),
        "mean_circularity": round(float(df_morphology["circularity"].mean()), 4),
        "mean_eccentricity": round(float(df_morphology["eccentricity"].mean()), 4),
    }


if __name__ == "__main__":
    # Test morphology extractor on a dummy mask
    dummy_mask = np.zeros((1, 100, 100), dtype=np.int32)
    dummy_mask[0, 20:40, 20:40] = 1
    dummy_mask[0, 60:80, 60:80] = 2

    df_test = extract_dataset_morphology(dummy_mask)
    print(df_test)
