"""
Data loading utilities for cell tracking dataset using Pooch and NumPy.
"""

from pathlib import Path
import numpy as np
import pooch
from config import DATA_DIR, MASKS_DATASET_URL, MASKS_FILE_NAME


def fetch_mask_dataset(data_dir: Path = DATA_DIR, url: str = MASKS_DATASET_URL, filename: str = MASKS_FILE_NAME) -> Path:
    """
    Downloads or retrieves cached cell segmentation mask file via Pooch.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    mask_file_path = pooch.retrieve(
        url=url,
        fname=filename,
        known_hash=None,
        path=data_dir
    )
    print(f"[DataLoader] Mask file retrieved at: {mask_file_path}")
    return Path(mask_file_path)


def load_masks(file_path: Path = None) -> np.ndarray:
    """
    Loads mask array from NPZ file.
    Returns array of shape (time_frames, height, width).
    """
    if file_path is None:
        file_path = fetch_mask_dataset()

    npz_data = np.load(file_path)
    if "masks" not in npz_data.files:
        raise KeyError(f"'masks' key not found in {file_path}. Available keys: {npz_data.files}")

    masks = npz_data["masks"]
    print(f"[DataLoader] Loaded masks shape: {masks.shape}, dtype: {masks.dtype}")
    return masks


if __name__ == "__main__":
    masks_data = load_masks()
    print(f"Data Loader verification complete. Loaded {len(masks_data)} frames.")
