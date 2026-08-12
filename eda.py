"""
Exploratory Data Analysis (EDA) Module for Cell Tracking Dataset.
Computes comprehensive spatial, morphological, kinematic, and temporal metrics,
generating high-resolution summary figures and tabular metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.spatial.distance import pdist, squareform
from data_loader import load_masks

plt.style.use('ggplot')
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})


def perform_full_eda(masks: np.ndarray = None, output_dir: str = "eda_results") -> dict:
    """
    Performs complete Exploratory Data Analysis on 3D cell segmentation masks array (T, H, W).
    """
    if masks is None:
        masks = load_masks()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_frames, height, width = masks.shape
    total_pixels_per_frame = height * width

    print(f"[EDA] Starting analysis on array shape: {masks.shape} (T={total_frames}, H={height}, W={width})")

    # -------------------------------------------------------------
    # 1. Per-Frame & Per-Cell Extraction
    # -------------------------------------------------------------
    frame_metrics = []
    cell_observations = []

    for t in range(total_frames):
        frame_mask = masks[t]
        unique_labels = np.unique(frame_mask)
        cell_labels = [l for l in unique_labels if l != 0]
        num_cells = len(cell_labels)
        
        # Total foreground pixel area occupied by cells
        fg_pixels = np.count_nonzero(frame_mask)
        fg_percentage = (fg_pixels / total_pixels_per_frame) * 100.0

        frame_metrics.append({
            "frame": t,
            "cell_count": num_cells,
            "fg_pixels": fg_pixels,
            "fg_coverage_pct": fg_percentage
        })

        if num_cells > 0:
            props = regionprops(frame_mask)
            centroids = []
            for p in props:
                c_id = p.label
                area = p.area
                perimeter = p.perimeter
                cy, cx = p.centroid
                centroids.append((cx, cy))

                # Shape metrics
                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
                major = getattr(p, "axis_major_length", getattr(p, "major_axis_length", 0.0))
                minor = getattr(p, "axis_minor_length", getattr(p, "minor_axis_length", 0.0))
                aspect_ratio = (major / minor) if minor > 0 else 1.0
                eccentricity = p.eccentricity
                solidity = p.solidity

                cell_observations.append({
                    "frame": t,
                    "cell_id": c_id,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "area": area,
                    "perimeter": perimeter,
                    "circularity": circularity,
                    "aspect_ratio": aspect_ratio,
                    "eccentricity": eccentricity,
                    "solidity": solidity,
                    "bbox_min_y": p.bbox[0],
                    "bbox_min_x": p.bbox[1],
                    "bbox_max_y": p.bbox[2],
                    "bbox_max_x": p.bbox[3]
                })

            # Calculate nearest neighbor distances in current frame
            if len(centroids) > 1:
                coords = np.array(centroids)
                dist_matrix = squareform(pdist(coords))
                np.fill_diagonal(dist_matrix, np.inf)
                min_dists = dist_matrix.min(axis=1)
                for idx, c_dist in enumerate(min_dists):
                    cell_observations[-len(centroids) + idx]["nn_distance"] = c_dist
            elif len(centroids) == 1:
                cell_observations[-1]["nn_distance"] = np.nan

    df_frames = pd.DataFrame(frame_metrics)
    df_cells = pd.DataFrame(cell_observations)

    # -------------------------------------------------------------
    # 2. Kinematics & Motion Metrics per Cell ID
    # -------------------------------------------------------------
    trajectory_metrics = []

    for cell_id, group in df_cells.groupby("cell_id"):
        group_sorted = group.sort_values(by="frame")
        lifespan = len(group_sorted)
        start_frame = group_sorted["frame"].min()
        end_frame = group_sorted["frame"].max()

        # Displacement and velocities
        xs = group_sorted["centroid_x"].values
        ys = group_sorted["centroid_y"].values
        frames = group_sorted["frame"].values

        dx = np.diff(xs)
        dy = np.diff(ys)
        dt = np.diff(frames)

        step_distances = np.sqrt(dx**2 + dy**2)
        total_distance = step_distances.sum()

        net_displacement = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        directionality = (net_displacement / total_distance) if total_distance > 0 else 1.0
        mean_speed = (step_distances / dt).mean() if len(step_distances) > 0 else 0.0
        max_speed = (step_distances / dt).max() if len(step_distances) > 0 else 0.0

        # Mean Cell Area over lifespan
        mean_area = group_sorted["area"].mean()
        mean_circularity = group_sorted["circularity"].mean()

        trajectory_metrics.append({
            "cell_id": cell_id,
            "lifespan_frames": lifespan,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "total_distance": total_distance,
            "net_displacement": net_displacement,
            "directionality": directionality,
            "mean_speed": mean_speed,
            "max_speed": max_speed,
            "mean_area": mean_area,
            "mean_circularity": mean_circularity
        })

    df_trajectories = pd.DataFrame(trajectory_metrics)

    # Save CSV summaries
    df_frames.to_csv(out_path / "eda_frame_metrics.csv", index=False)
    df_cells.to_csv(out_path / "eda_cell_observations.csv", index=False)
    df_trajectories.to_csv(out_path / "eda_cell_trajectories.csv", index=False)

    print(f"[EDA] Exported tabular datasets to '{out_path.resolve()}'")

    # -------------------------------------------------------------
    # 3. Generate High-Resolution EDA Plot Panels
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Subplot 1: Cell Population over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_frames["frame"], df_frames["cell_count"], marker="o", color="#38bdf8", linewidth=2.5)
    ax1.set_title("Cell Population Count over Time", fontsize=12, fontweight="bold", color="#0f172a")
    ax1.set_xlabel("Time Frame")
    ax1.set_ylabel("Active Cell Count")

    # Subplot 2: Area Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df_cells["area"], color="#10b981", bins=20, edgecolor="black", alpha=0.8)
    ax2.set_title("Cell Area Distribution (px²)", fontsize=12, fontweight="bold", color="#0f172a")
    ax2.set_xlabel("Area (px²)")

    # Subplot 3: Circularity vs Eccentricity
    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(df_cells["circularity"], df_cells["eccentricity"], c=df_cells["area"], cmap="viridis", alpha=0.7, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Area (px²)")
    ax3.set_title("Circularity vs Eccentricity", fontsize=12, fontweight="bold", color="#0f172a")
    ax3.set_xlabel("Circularity (1 = Circle)")
    ax3.set_ylabel("Eccentricity (0 = Circle)")

    # Subplot 4: Spatial Density Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hexbin(df_cells["centroid_x"], df_cells["centroid_y"], gridsize=20, cmap="PuBu", mincnt=1)
    ax4.scatter(df_cells["centroid_x"], df_cells["centroid_y"], s=8, color="#f43f5e", alpha=0.5)
    ax4.set_xlim(0, width)
    ax4.set_ylim(height, 0)  # Invert Y to match image coordinates
    ax4.set_title("Spatial Density Distribution", fontsize=12, fontweight="bold", color="#0f172a")
    ax4.set_xlabel("X Position (px)")
    ax4.set_ylabel("Y Position (px)")

    # Subplot 5: Cell Migration Trajectories
    ax5 = fig.add_subplot(gs[1, 1])
    for c_id, grp in df_cells.groupby("cell_id"):
        grp_s = grp.sort_values(by="frame")
        ax5.plot(grp_s["centroid_x"], grp_s["centroid_y"], alpha=0.7, linewidth=1.5)
        ax5.scatter(grp_s["centroid_x"].iloc[0], grp_s["centroid_y"].iloc[0], color="#22c55e", s=15, zorder=4)  # Start
        ax5.scatter(grp_s["centroid_x"].iloc[-1], grp_s["centroid_y"].iloc[-1], color="#ef4444", s=15, zorder=4)  # End
    ax5.set_xlim(0, width)
    ax5.set_ylim(height, 0)
    ax5.set_title("Cell Migration Trajectories (Green=Start, Red=End)", fontsize=12, fontweight="bold", color="#0f172a")
    ax5.set_xlabel("X Position (px)")
    ax5.set_ylabel("Y Position (px)")

    # Subplot 6: Speed vs Directionality
    ax6 = fig.add_subplot(gs[1, 2])
    sc6 = ax6.scatter(df_trajectories["mean_speed"], df_trajectories["directionality"], c=df_trajectories["lifespan_frames"], cmap="plasma", s=df_trajectories["mean_area"] / 40.0, alpha=0.8, edgecolors="black")
    cbar6 = plt.colorbar(sc6, ax=ax6)
    cbar6.set_label("Lifespan (Frames)")
    ax6.set_title("Cell Speed vs Directionality Ratio", fontsize=12, fontweight="bold", color="#0f172a")
    ax6.set_xlabel("Mean Speed (px/frame)")
    ax6.set_ylabel("Directionality (1 = Straight Line)")

    # Subplot 7: Lifespan Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(df_trajectories["lifespan_frames"], bins=15, color="#8b5cf6", edgecolor="black", alpha=0.8)
    ax7.set_title("Cell Trajectory Lifespan (Frames)", fontsize=12, fontweight="bold", color="#0f172a")
    ax7.set_xlabel("Lifespan (Number of Frames Tracked)")
    ax7.set_ylabel("Cell Count")

    # Subplot 8: Nearest Neighbor Distance Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(df_cells["nn_distance"].dropna(), color="#f59e0b", bins=20, edgecolor="black", alpha=0.8)
    ax8.set_title("Nearest-Neighbor Distance (px)", fontsize=12, fontweight="bold", color="#0f172a")
    ax8.set_xlabel("Distance to Nearest Cell (px)")

    # Subplot 9: Foreground Area Coverage Pct over Time
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(df_frames["frame"], df_frames["fg_coverage_pct"], color="#ec4899", marker="s", linewidth=2)
    ax9.set_title("Total Cell Area Coverage (%)", fontsize=12, fontweight="bold", color="#0f172a")
    ax9.set_xlabel("Time Frame")
    ax9.set_ylabel("Area Coverage (%)")

    plt.suptitle("🔬 Cell Tracking Dataset - Comprehensive Exploratory Data Analysis (EDA)", fontsize=16, fontweight="bold", y=0.98)
    plot_file = out_path / "eda_summary_plots.png"
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Summary plot saved to '{plot_file.resolve()}'")

    # -------------------------------------------------------------
    # 4. Summary Statistics Dictionary
    # -------------------------------------------------------------
    summary_stats = {
        "dataset_shape": list(masks.shape),
        "total_frames": total_frames,
        "spatial_resolution": f"{width}x{height}",
        "total_unique_cells": int(df_trajectories["cell_id"].nunique()),
        "total_cell_observations": int(len(df_cells)),
        "mean_cells_per_frame": round(float(df_frames["cell_count"].mean()), 2),
        "min_cells_per_frame": int(df_frames["cell_count"].min()),
        "max_cells_per_frame": int(df_frames["cell_count"].max()),
        "area_px2": {
            "mean": round(float(df_cells["area"].mean()), 2),
            "median": round(float(df_cells["area"].median()), 2),
            "std": round(float(df_cells["area"].std()), 2),
            "min": int(df_cells["area"].min()),
            "max": int(df_cells["area"].max())
        },
        "circularity": {
            "mean": round(float(df_cells["circularity"].mean()), 4),
            "median": round(float(df_cells["circularity"].median()), 4),
            "std": round(float(df_cells["circularity"].std()), 4)
        },
        "eccentricity": {
            "mean": round(float(df_cells["eccentricity"].mean()), 4),
            "median": round(float(df_cells["eccentricity"].median()), 4)
        },
        "kinematics": {
            "mean_population_speed_px_per_frame": round(float(df_trajectories["mean_speed"].mean()), 2),
            "mean_net_displacement_px": round(float(df_trajectories["net_displacement"].mean()), 2),
            "mean_total_distance_px": round(float(df_trajectories["total_distance"].mean()), 2),
            "mean_directionality_ratio": round(float(df_trajectories["directionality"].mean()), 4)
        },
        "spatial_density": {
            "mean_nearest_neighbor_dist_px": round(float(df_cells["nn_distance"].mean()), 2),
            "min_nearest_neighbor_dist_px": round(float(df_cells["nn_distance"].min()), 2)
        }
    }

    print(f"[EDA] Analysis Complete! Summary: {summary_stats}")
    return summary_stats


if __name__ == "__main__":
    perform_full_eda()
