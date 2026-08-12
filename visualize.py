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


def plot_morphology_distributions(
    df_morphology: 'pd.DataFrame',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots distributions of cell Area, Circularity, and Eccentricity across frames.
    """
    if df_morphology.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df_morphology["area"], bins=20, color="skyblue", edgecolor="black")
    axes[0].set_title("Cell Area Distribution (px²)")
    axes[0].set_xlabel("Area")
    axes[0].set_ylabel("Count")

    axes[1].hist(df_morphology["circularity"], bins=20, color="lightgreen", edgecolor="black")
    axes[1].set_title("Circularity Distribution (0=Line, 1=Circle)")
    axes[1].set_xlabel("Circularity")

    axes[2].hist(df_morphology["eccentricity"], bins=20, color="salmon", edgecolor="black")
    axes[2].set_title("Eccentricity Distribution")
    axes[2].set_xlabel("Eccentricity")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Visualize] Morphology plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cell_trajectories(
    df_morphology: 'pd.DataFrame',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots 2D spatial cell migration trajectories over time.
    """
    if df_morphology.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    df_sorted = df_morphology.sort_values(by=["label_id", "frame"])

    for label_id, group in df_sorted.groupby("label_id"):
        if len(group) > 1:
            ax.plot(group["centroid_x"], group["centroid_y"], marker="o", markersize=3, label=f"Cell {label_id}")
            ax.scatter(group["centroid_x"].iloc[0], group["centroid_y"].iloc[0], color="green", s=25, zorder=5)  # Start
            ax.scatter(group["centroid_x"].iloc[-1], group["centroid_y"].iloc[-1], color="red", s=25, zorder=5)  # End

    ax.set_title("2D Cell Migration Trajectories (Green=Start, Red=End)")
    ax.set_xlabel("X Position (px)")
    ax.set_ylabel("Y Position (px)")
    ax.invert_yaxis()  # Match image coordinate conventions
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Visualize] Trajectories plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_motility_distributions(
    df_kinematics: 'pd.DataFrame',
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plots histograms of cell Speed, Net Displacement, and Directionality Ratio.
    """
    if df_kinematics.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(df_kinematics["mean_speed"], bins=15, color="coral", edgecolor="black")
    axes[0].set_title("Mean Cell Speed (px/frame)")
    axes[0].set_xlabel("Speed")
    axes[0].set_ylabel("Cell Count")

    axes[1].hist(df_kinematics["net_displacement"], bins=15, color="mediumpurple", edgecolor="black")
    axes[1].set_title("Net Displacement (px)")
    axes[1].set_xlabel("Displacement")

    axes[2].hist(df_kinematics["directionality_ratio"], bins=15, color="gold", edgecolor="black")
    axes[2].set_title("Directionality Ratio (0=Random, 1=Straight)")
    axes[2].set_xlabel("Directionality")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Visualize] Motility plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def export_animated_video(masks: np.ndarray, output_gif: str = "cell_tracking_video.gif", fps: int = 5):
    """
    Exports animated GIF video of time-lapse cell tracking.
    """
    from export_video import create_cell_tracking_video
    return create_cell_tracking_video(masks, output_gif=output_gif, fps=fps)


def export_web_visualizer():
    """
    Builds self-contained HTML5 interactive time-lapse video player dashboard.
    """
    from generate_web_visualizer import build_web_visualizer
    return build_web_visualizer()


if __name__ == "__main__":
    # Test Matplotlib plot generation with dummy data
    dummy_masks = np.random.randint(0, 10, size=(2, 100, 100))
    dummy_tracked = np.random.randint(0, 10, size=(2, 100, 100))
    plot_frame_comparison(dummy_masks, dummy_tracked, frame_idx=0, show=False)
    print("[Visualize] Matplotlib visualization test passed.")



