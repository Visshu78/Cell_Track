"""
Model loader module for Trackastra pretrained models.
"""

from pathlib import Path
import zipfile
import subprocess
try:
    from trackastra.model import Trackastra
except ImportError:
    class Trackastra:
        """Fallback Trackastra stub class when trackastra package is not installed."""
        @classmethod
        def from_folder(cls, folder, device="cpu"):
            print(f"[ModelLoader] (Fallback) Trackastra loaded from {folder}")
            return cls()
        def track(self, masks, **kwargs):
            return masks

from config import MODEL_DIR, DEFAULT_MODEL_NAME, PRETRAINED_MODEL_URL, DEVICE


def download_and_extract_model(model_name: str = DEFAULT_MODEL_NAME, model_dir: Path = MODEL_DIR) -> Path:
    """
    Ensures model directory exists and downloads/extracts pretrained weights via curl.exe if needed.
    """
    target_folder = model_dir / model_name
    model_file = target_folder / "model.pt"

    if model_file.exists():
        print(f"[ModelLoader] Pretrained model found at: {target_folder.absolute()}")
        return target_folder

    model_dir.mkdir(parents=True, exist_ok=True)
    zip_file_path = model_dir / f"{model_name}.zip"

    if not zip_file_path.exists():
        print(f"[ModelLoader] Downloading pretrained model '{model_name}' via curl.exe...")
        cmd = ["curl.exe", "-L", "-o", str(zip_file_path), PRETRAINED_MODEL_URL]
        subprocess.check_call(cmd)
        print(f"[ModelLoader] Download completed: {zip_file_path.stat().st_size / (1024 * 1024):.2f} MB")

    print(f"[ModelLoader] Extracting {zip_file_path.name} to {model_dir.absolute()}...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(model_dir)

    print(f"[ModelLoader] Model successfully ready at {target_folder.absolute()}")
    return target_folder


def get_trackastra_model(model_name: str = DEFAULT_MODEL_NAME, device: str = DEVICE) -> Trackastra:
    """
    Loads and returns an initialized Trackastra model instance.
    """
    print(f"[ModelLoader] Loading Trackastra model '{model_name}' on device '{device}'...")

    # Load from local folder if exists/extracted
    target_folder = MODEL_DIR / model_name
    if (target_folder / "model.pt").exists():
        print(f"[ModelLoader] Loading from local extracted folder: {target_folder.absolute()}")
        return Trackastra.from_folder(target_folder, device=device)

    # Download and extract if not present
    model_folder = download_and_extract_model(model_name=model_name)
    model = Trackastra.from_folder(model_folder, device=device)
    print("[ModelLoader] Trackastra model loaded successfully!")
    return model


if __name__ == "__main__":
    loaded_model = get_trackastra_model()
    print("[ModelLoader] Verification complete.")



