"""
Configuration settings for Cell Tracking pipeline using Trackastra and Napari.
"""

from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "cell_tracking_data"
MODEL_DIR = BASE_DIR / "trackastra_models"

# Dataset settings
MASKS_DATASET_URL = "doi:10.5281/zenodo.15852284/masks_pred.npz"
MASKS_FILE_NAME = "masks_pred.npz"

# Model settings
DEFAULT_MODEL_NAME = "general_2d"
PRETRAINED_MODEL_URL = "https://github.com/weigertlab/trackastra-models/releases/download/v0.3.0/general_2d.zip"

# Device configuration (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
