"""
Cell Tracking Challenge (CTC) Dataset Loader for BF-C2DL-HSC.

Handles:
  1. Loading raw time-lapse 16-bit / 8-bit TIF images.
  2. Loading GT tracking masks (man_trackXXXX.tif) and GT segmentation masks.
  3. Parsing man_track.txt into lineage ground-truth records:
     (Cell_ID, Begin_Frame, End_Frame, Parent_ID).
  4. Rescaling / subsetting frames for efficient deep learning training & inference.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

DATASET_ROOT = Path("train-BF-C2DL-HSC/BF-C2DL-HSC")


def parse_man_track_txt(file_path: Path) -> List[Dict[str, int]]:
    """
    Parses man_track.txt CTC lineage file.
    
    File format (space-separated):
      L B E P
      L: Cell label ID
      B: Begin frame (0-indexed)
      E: End frame
      P: Parent cell label ID (0 if root cell)
    
    Returns list of dicts:
      [{'cell_id': L, 'begin_frame': B, 'end_frame': E, 'parent_id': P}, ...]
    """
    records = []
    if not file_path.exists():
        print(f"[CTCLoader] Warning: man_track.txt not found at {file_path}")
        return records

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                records.append({
                    "cell_id": int(parts[0]),
                    "begin_frame": int(parts[1]),
                    "end_frame": int(parts[2]),
                    "parent_id": int(parts[3]),
                })
    return records


def load_ctc_gt_masks(
    seq_name: str = "01",
    dataset_root: Path = DATASET_ROOT,
    max_frames: Optional[int] = None,
    downsample_factor: int = 1,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """
    Loads GT tracking label masks (man_trackXXXX.tif) and lineage text file for a sequence.

    Args:
        seq_name: '01' or '02'
        dataset_root: Path to dataset folder
        max_frames: Limit number of loaded frames (e.g. 50 or 100 for fast training)
        downsample_factor: Downsampling factor for H, W (e.g., 2 for 505x505)

    Returns:
        masks: np.ndarray (T, H, W) integer cell label masks
        lineage_records: parsed list of man_track.txt records
    """
    gt_dir = dataset_root / f"{seq_name}_GT" / "TRA"
    track_txt = gt_dir / "man_track.txt"

    lineage_records = parse_man_track_txt(track_txt)

    tif_files = sorted(glob.glob(os.path.join(gt_dir, "man_track*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No GT track tif files found in {gt_dir}")

    if max_frames and max_frames > 0:
        tif_files = tif_files[:max_frames]

    print(f"[CTCLoader] Loading {len(tif_files)} GT tracking mask frames from {seq_name}_GT/TRA...")
    
    sample_img = np.array(Image.open(tif_files[0]))
    H, W = sample_img.shape

    if downsample_factor > 1:
        H_new, W_new = H // downsample_factor, W // downsample_factor
    else:
        H_new, W_new = H, W

    T = len(tif_files)
    masks = np.zeros((T, H_new, W_new), dtype=np.int32)

    for t, filepath in enumerate(tif_files):
        img = np.array(Image.open(filepath))
        if downsample_factor > 1:
            img = img[::downsample_factor, ::downsample_factor]
        masks[t] = img.astype(np.int32)

    print(f"[CTCLoader] Sequence {seq_name} loaded: masks shape={masks.shape}, "
          f"unique cells={len(np.unique(masks)) - 1}, lineage entries={len(lineage_records)}")

    return masks, lineage_records


def load_ctc_raw_images(
    seq_name: str = "01",
    dataset_root: Path = DATASET_ROOT,
    max_frames: Optional[int] = None,
    downsample_factor: int = 1,
) -> np.ndarray:
    """
    Loads raw microscopy image intensity frames (tXXXX.tif).

    Returns:
        images: np.ndarray (T, H, W) float32 normalized [0, 1] intensity images
    """
    img_dir = dataset_root / seq_name
    tif_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))

    if not tif_files:
        raise FileNotFoundError(f"No raw tif files found in {img_dir}")

    if max_frames and max_frames > 0:
        tif_files = tif_files[:max_frames]

    print(f"[CTCLoader] Loading {len(tif_files)} raw microscopy images from {seq_name}...")

    sample_img = np.array(Image.open(tif_files[0]))
    H, W = sample_img.shape

    if downsample_factor > 1:
        H_new, W_new = H // downsample_factor, W // downsample_factor
    else:
        H_new, W_new = H, W

    T = len(tif_files)
    images = np.zeros((T, H_new, W_new), dtype=np.float32)

    for t, filepath in enumerate(tif_files):
        img = np.array(Image.open(filepath)).astype(np.float32)
        if downsample_factor > 1:
            img = img[::downsample_factor, ::downsample_factor]
        # Normalize to [0, 1]
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        images[t] = img

    print(f"[CTCLoader] Sequence {seq_name} raw images loaded: shape={images.shape}")
    return images


def extract_division_labels_from_lineage(
    lineage_records: List[Dict[str, int]],
    masks: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Extracts GT centroid positions and mitosis (division) labels per frame.

    Returns:
        div_events: dict mapping frame t -> list of dividing cell IDs in that frame
        division_targets: array of shape (T, N_max) binary division targets
    """
    # Parent cell IDs that give birth to daughters
    parent_map = {}  # parent_id -> list of daughter_ids
    division_frames = {}  # parent_id -> frame_of_division (begin_frame of daughter)

    for rec in lineage_records:
        pid = rec["parent_id"]
        cid = rec["cell_id"]
        b_frame = rec["begin_frame"]
        if pid > 0:
            if pid not in parent_map:
                parent_map[pid] = []
            parent_map[pid].append(cid)
            division_frames[pid] = b_frame - 1  # Division occurred at frame prior to daughter birth

    return division_frames, parent_map


if __name__ == "__main__":
    print("Testing CTCLoader on train-BF-C2DL-HSC dataset...")
    masks, lineage = load_ctc_gt_masks(seq_name="01", max_frames=20, downsample_factor=2)
    div_frames, parent_map = extract_division_labels_from_lineage(lineage, masks)
    print(f"Dividing parent cells found: {len(parent_map)}")
    for pid, daughters in parent_map.items():
        print(f"  Parent Cell {pid} divided at frame {div_frames.get(pid)} -> Daughters {daughters}")
