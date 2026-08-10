"""
Module 4: Behavior Analysis & Cell Motility Analytics.
Quantifies cell migration kinematics, motility dynamics, spatial speed, displacement,
directionality ratio, and population-level proliferative behavior.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def compute_cell_kinematics(df_morphology: pd.DataFrame) -> pd.DataFrame:
    """
    Computes spatial trajectories and kinematic metrics for each tracked cell line.
    """
    if df_morphology.empty:
        return pd.DataFrame()

    df_sorted = df_morphology.sort_values(by=["label_id", "frame"]).copy()
    kinematics = []

    for label_id, group in df_sorted.groupby("label_id"):
        if len(group) < 2:
            # Single frame observation
            kinematics.append({
                "label_id": label_id,
                "start_frame": int(group["frame"].min()),
                "end_frame": int(group["frame"].max()),
                "frames_tracked": 1,
                "total_distance": 0.0,
                "net_displacement": 0.0,
                "mean_speed": 0.0,
                "max_speed": 0.0,
                "directionality_ratio": 1.0,
                "start_x": float(group["centroid_x"].iloc[0]),
                "start_y": float(group["centroid_y"].iloc[0]),
                "end_x": float(group["centroid_x"].iloc[-1]),
                "end_y": float(group["centroid_y"].iloc[-1])
            })
            continue

        xs = group["centroid_x"].values
        ys = group["centroid_y"].values
        frames = group["frame"].values

        # Step displacement calculation
        dx = np.diff(xs)
        dy = np.diff(ys)
        dt = np.diff(frames)

        # Handle potential frame gaps
        dt = np.where(dt == 0, 1, dt)
        step_distances = np.sqrt(dx ** 2 + dy ** 2)
        step_speeds = step_distances / dt

        total_distance = float(np.sum(step_distances))
        net_dx = xs[-1] - xs[0]
        net_dy = ys[-1] - ys[0]
        net_displacement = float(np.sqrt(net_dx ** 2 + net_dy ** 2))

        mean_speed = float(np.mean(step_speeds))
        max_speed = float(np.max(step_speeds))

        # Directionality ratio (Confinement Ratio) = Net Displacement / Total Distance
        directionality = net_displacement / total_distance if total_distance > 0 else 1.0
        directionality = min(1.0, max(0.0, directionality))

        kinematics.append({
            "label_id": label_id,
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "frames_tracked": int(len(group)),
            "total_distance": round(total_distance, 2),
            "net_displacement": round(net_displacement, 2),
            "mean_speed": round(mean_speed, 2),
            "max_speed": round(max_speed, 2),
            "directionality_ratio": round(directionality, 4),
            "start_x": float(xs[0]),
            "start_y": float(ys[0]),
            "end_x": float(xs[-1]),
            "end_y": float(ys[-1])
        })

    df_kinematics = pd.DataFrame(kinematics)
    print(f"[Behavior] Computed kinematics for {len(df_kinematics)} cell trajectories.")
    return df_kinematics


def compute_population_behavior_summary(df_kinematics: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes population-level migratory behavior metrics.
    """
    if df_kinematics.empty:
        return {}

    return {
        "tracked_trajectories": len(df_kinematics),
        "mean_population_speed": round(float(df_kinematics["mean_speed"].mean()), 2),
        "median_population_speed": round(float(df_kinematics["mean_speed"].median()), 2),
        "mean_net_displacement": round(float(df_kinematics["net_displacement"].mean()), 2),
        "mean_total_distance": round(float(df_kinematics["total_distance"].mean()), 2),
        "mean_directionality_ratio": round(float(df_kinematics["directionality_ratio"].mean()), 4),
        "directed_migration_index": round(float((df_kinematics["directionality_ratio"] > 0.5).mean()), 4)
    }


if __name__ == "__main__":
    # Test kinematics on dummy DataFrame
    data = {
        "frame": [0, 1, 2, 0, 1, 2],
        "label_id": [1, 1, 1, 2, 2, 2],
        "centroid_x": [10.0, 15.0, 20.0, 50.0, 52.0, 51.0],
        "centroid_y": [10.0, 15.0, 20.0, 50.0, 51.0, 52.0]
    }
    df_dummy = pd.DataFrame(data)
    df_kin = compute_cell_kinematics(df_dummy)
    print(df_kin)
    print(compute_population_behavior_summary(df_kin))
