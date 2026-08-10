"""
Visualization module for cell tracking results (Matplotlib & Napari).
"""

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_frame_comparison(
    masks: np.ndarray,
    tracked_masks: np.ndarray,
    frame_idx: int = 0,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots a Matplotlib side-by-side comparison of original vs tracked cell masks for a specific frame.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(masks[frame_idx], cmap="nipy_spectral")
    axes[0].set_title(f"Original Segmentation (Frame {frame_idx})")
    axes[0].axis("off")

    axes[1].imshow(tracked_masks[frame_idx], cmap="nipy_spectral")
    axes[1].set_title(f"Tracked Mask (Frame {frame_idx})")
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Visualize] Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def launch_napari_viewer(
    masks: np.ndarray,
    tracked_masks: np.ndarray,
    images: Optional[np.ndarray] = None
) -> None:
    """
    Launches an interactive Napari multi-dimensional viewer displaying image/mask layers.
    """
    import napari

    print(f"[Visualize] Launching Napari viewer (Napari v{napari.__version__})...")
    viewer = napari.Viewer()

    if images is not None:
        viewer.add_image(
            images,
            name="Raw Microscopy",
            colormap="gray"
        )

    viewer.add_labels(
        masks,
        name="Original Segmentation"
    )

    viewer.add_labels(
        tracked_masks,
        name="Tracked Cells"
    )

    napari.run()


if __name__ == "__main__":
    # Test Matplotlib plot generation with dummy data
    dummy_masks = np.random.randint(0, 10, size=(2, 100, 100))
    dummy_tracked = np.random.randint(0, 10, size=(2, 100, 100))
    plot_frame_comparison(dummy_masks, dummy_tracked, frame_idx=0, show=False)
    print("[Visualize] Matplotlib visualization test passed.")
