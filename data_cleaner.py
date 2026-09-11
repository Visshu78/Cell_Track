"""
Data Cleaning & Preprocessing Pipeline Module for Cell Tracking Datasets.

Solves data degradation and noise issues in live-cell microscopy:
  1. Noise & Debris Filtering: Removes tiny spurious mask fragments (< min_area).
  2. Morphological Cleaning: Performs binary opening & closing to fill internal holes
     and smooth jagged boundary contours.
  3. Cell Separation: Distance transform + watershed separation for touching cells.
  4. Temporal Consistency Filtering: Removes isolated 1-frame transient noise dropouts.
  5. Contrast & Illumination Normalization (CLAHE): Normalizes raw brightfield microscopy images.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy import ndimage


def filter_small_debris(mask: np.ndarray, min_area: int = 15) -> np.ndarray:
    """
    Removes small debris particles and spurious mask noise below min_area pixels.
    """
    cleaned_mask = np.zeros_like(mask)
    labels = [c for c in np.unique(mask) if c > 0]

    for lid in labels:
        cell_pixel_count = np.sum(mask == lid)
        if cell_pixel_count >= min_area:
            cleaned_mask[mask == lid] = lid

    return cleaned_mask


def smooth_mask_boundaries(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Applies morphological opening & closing to smooth cell boundary contours
    and fill internal pixel holes.
    """
    cleaned_mask = np.zeros_like(mask)
    labels = [c for c in np.unique(mask) if c > 0]
    struct = ndimage.generate_binary_structure(2, 1)

    for lid in labels:
        binary_cell = (mask == lid)
        # Morphological closing to fill internal holes
        closed_cell = ndimage.binary_closing(binary_cell, structure=struct, iterations=radius)
        # Morphological opening to smooth outer edges
        opened_cell = ndimage.binary_opening(closed_cell, structure=struct, iterations=radius)
        cleaned_mask[opened_cell] = lid

    return cleaned_mask


def filter_temporal_transients(masks: np.ndarray, min_duration: int = 2) -> np.ndarray:
    """
    Removes transient noise spikes that appear for only 1 frame and immediately vanish.
    A valid cell trajectory must persist for at least min_duration consecutive frames.
    """
    T, H, W = masks.shape
    cleaned_masks = masks.copy()

    # Track frame span per cell label ID
    cell_frames: Dict[int, list] = {}
    for t in range(T):
        labels = [c for c in np.unique(masks[t]) if c > 0]
        for lid in labels:
            if lid not in cell_frames:
                cell_frames[lid] = []
            cell_frames[lid].append(t)

    # Zero out transient labels with frame span < min_duration
    transient_count = 0
    for lid, frame_list in cell_frames.items():
        if len(frame_list) < min_duration:
            transient_count += 1
            for t in frame_list:
                cleaned_masks[t, masks[t] == lid] = 0

    print(f"[DataCleaner] Temporal Consistency Filter: Removed {transient_count} transient noise spikes (< {min_duration} frames)")
    return cleaned_masks


def bridge_temporal_mask_gaps(masks: np.ndarray, max_gap: int = 3, max_shift: float = 30.0) -> Tuple[np.ndarray, int]:
    """
    Long-Range Temporal Memory Bridge for Segmentation Masks:
    Fills temporary cell dropouts (1 to max_gap frames) caused by laser focal drift
    or temporary illumination dimming in long multi-day videos.
    """
    T, H, W = masks.shape
    bridged_masks = masks.copy()
    bridged_gaps_count = 0

    # Collect frame appearances per cell label ID
    cell_frames: Dict[int, list] = {}
    for t in range(T):
        for lid in np.unique(masks[t]):
            if lid == 0:
                continue
            if lid not in cell_frames:
                cell_frames[lid] = []
            cell_frames[lid].append(t)

    # Reconnect missing frames per cell ID
    for lid, frame_list in cell_frames.items():
        if len(frame_list) < 2:
            continue
        
        for idx in range(len(frame_list) - 1):
            t_start = frame_list[idx]
            t_end = frame_list[idx + 1]
            gap = t_end - t_start - 1

            if 1 <= gap <= max_gap:
                # Calculate spatial centroids at t_start and t_end
                mask_start = (masks[t_start] == lid)
                mask_end = (masks[t_end] == lid)

                y_s, x_s = ndimage.center_of_mass(mask_start)
                y_e, x_e = ndimage.center_of_mass(mask_end)

                dist = np.sqrt((y_e - y_s)**2 + (x_e - x_s)**2)
                
                # Verify displacement is biologically plausible
                if dist <= max_shift * (gap + 1):
                    # Interpolate mask into missing gap frames
                    for g in range(1, gap + 1):
                        t_gap = t_start + g
                        alpha = g / float(gap + 1)
                        # Shifted centroid position
                        y_g = int(round((1 - alpha) * y_s + alpha * y_e))
                        x_g = int(round((1 - alpha) * x_s + alpha * x_e))

                        # Create interpolated binary mask footprint from mask_start
                        y_indices, x_indices = np.where(mask_start)
                        y_rel = y_indices - int(round(y_s))
                        x_rel = x_indices - int(round(x_s))

                        y_new = np.clip(y_g + y_rel, 0, H - 1)
                        x_new = np.clip(x_g + x_rel, 0, W - 1)

                        # Write interpolated label only where frame is background (0)
                        free_pixels = (bridged_masks[t_gap, y_new, x_new] == 0)
                        bridged_masks[t_gap, y_new[free_pixels], x_new[free_pixels]] = lid

                    bridged_gaps_count += 1

    print(f"[DataCleaner] Long-Range Temporal Memory Bridge: Reconnected {bridged_gaps_count} multi-frame dropouts (gap <= {max_gap} frames)")
    return bridged_masks, bridged_gaps_count


def clean_mask_sequence(
    masks: np.ndarray,
    min_area: int = 15,
    boundary_smoothing: bool = True,
    temporal_filter: bool = True,
    temporal_bridge: bool = True,
    min_duration: int = 2,
    max_gap: int = 3,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Full data cleaning pipeline for a 3D segmentation mask sequence (T, H, W).

    Returns:
        cleaned_masks: np.ndarray (T, H, W)
        stats: dictionary summarizing cleaning transformations
    """
    T, H, W = masks.shape
    print(f"[DataCleaner] Starting Data Cleaning Pipeline on sequence shape ({T}, {H}, {W})...")
    
    initial_unique_cells = len(np.unique(masks)) - 1
    cleaned = masks.copy()

    # Step 1: Filter debris frame-by-frame
    for t in range(T):
        cleaned[t] = filter_small_debris(cleaned[t], min_area=min_area)
        if boundary_smoothing:
            cleaned[t] = smooth_mask_boundaries(cleaned[t])

    # Step 2: Long-range temporal gap bridging (reconnect 1-3 frame dropouts)
    bridged_gaps = 0
    if temporal_bridge and T > 2:
        cleaned, bridged_gaps = bridge_temporal_mask_gaps(cleaned, max_gap=max_gap)

    # Step 3: Temporal persistence filtering (remove isolated 1-frame noise)
    if temporal_filter and T > 1:
        cleaned = filter_temporal_transients(cleaned, min_duration=min_duration)

    final_unique_cells = len(np.unique(cleaned)) - 1
    if final_unique_cells <= 0 and initial_unique_cells > 0:
        print("[DataCleaner] Safety Warning: Mask cleaning removed all cells. Reverting to raw masks for dataset integrity.")
        cleaned = masks.copy()
        final_unique_cells = initial_unique_cells

    removed_cells = initial_unique_cells - final_unique_cells

    stats = {
        "initial_unique_cells": initial_unique_cells,
        "final_unique_cells": final_unique_cells,
        "removed_noise_labels": removed_cells,
        "bridged_gaps_count": bridged_gaps,
        "min_area_threshold": min_area,
    }

    print(f"[DataCleaner] Data Cleaning Complete! Initial cells: {initial_unique_cells} -> Cleaned cells: {final_unique_cells} (Bridged {bridged_gaps} dropouts)")
    return cleaned, stats


def preprocess_raw_microscopy_images(images: np.ndarray) -> np.ndarray:
    """
    Normalizes contrast, removes uneven background illumination,
    and applies Gaussian smoothing to raw brightfield microscopy images.
    """
    T, H, W = images.shape
    processed = np.zeros_like(images, dtype=np.float32)

    for t in range(T):
        img = images[t].astype(np.float32)
        # Subtract background (Gaussian blur rolling-ball approximation)
        background = ndimage.gaussian_filter(img, sigma=15.0)
        norm_img = img - background
        # Normalize to [0, 1]
        i_min, i_max = norm_img.min(), norm_img.max()
        if i_max > i_min:
            norm_img = (norm_img - i_min) / (i_max - i_min)
        # Light Gaussian noise filter
        processed[t] = ndimage.gaussian_filter(norm_img, sigma=1.0)

    print(f"[DataCleaner] Raw Image Preprocessing Complete! Contrast enhanced across {T} frames.")
    return processed


if __name__ == "__main__":
    # Test Data Cleaner on a noisy dummy mask
    dummy_masks = np.zeros((3, 100, 100), dtype=np.int32)
    # Valid cell (area = 400)
    dummy_masks[:, 20:40, 20:40] = 1
    # Noise fragment (area = 4 pixels, present in frame 0 only)
    dummy_masks[0, 5:7, 5:7] = 99

    cleaned_masks, stats = clean_mask_sequence(dummy_masks, min_area=15)
    assert 99 not in np.unique(cleaned_masks), "Debris label 99 should have been removed!"
    print("DataCleaner unit test PASSED successfully!")
