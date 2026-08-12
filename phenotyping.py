"""
Unsupervised Cell Phenotyping & Behavioral Clustering Module.
Uses PCA dimensionality reduction and Scipy K-Means clustering to discover
distinct phenotypic states (e.g. Fast Migrating, Quiescent, Rounded) from cell morphometrics and kinematics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
from data_loader import load_masks
from tracker import run_cell_tracking
from model_loader import get_trackastra_model
from morphology import extract_dataset_morphology
from behavior import compute_cell_kinematics

plt.style.use('ggplot')


def standardize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardizes feature matrix to zero mean and unit variance.
    Returns (X_std, mean, std).
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  # Prevent division by zero
    X_std = (X - mean) / std
    return X_std, mean, std


def compute_pca_2d(X_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes top 2 Principal Components using Singular Value Decomposition (SVD).
    Returns (X_pca, explained_variance_ratio).
    """
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    X_pca = X_std @ Vt.T[:, :2]
    
    variances = (S**2) / (len(X_std) - 1)
    total_var = np.sum(variances)
    explained_var_ratio = variances[:2] / total_var if total_var > 0 else np.array([0.5, 0.5])
    
    return X_pca, explained_var_ratio


def auto_annotate_clusters(df_clustered: pd.DataFrame, num_clusters: int) -> dict:
    """
    Assigns intuitive biological state titles based on cluster mean features.
    """
    cluster_annotations = {}
    
    for c_id in range(num_clusters):
        sub = df_clustered[df_clustered["cluster"] == c_id]
        if len(sub) == 0:
            cluster_annotations[c_id] = f"Cluster {c_id}"
            continue
            
        mean_speed = sub["mean_speed"].mean()
        mean_area = sub["mean_area"].mean()
        mean_circ = sub["mean_circularity"].mean()
        mean_dir = sub["directionality"].mean()
        
        # Determine dominant traits
        pop_speed_avg = df_clustered["mean_speed"].mean()
        pop_area_avg = df_clustered["mean_area"].mean()
        pop_circ_avg = df_clustered["mean_circularity"].mean()
        
        if mean_speed > pop_speed_avg * 1.15:
            if mean_dir > 0.15:
                label = "Fast Migrating (Persistent)"
            else:
                label = "Fast Migrating (Exploratory)"
        elif mean_area > pop_area_avg * 1.15:
            label = "Large Spread / Anchored"
        elif mean_circ > pop_circ_avg * 1.05:
            label = "Quiescent / Compact Round"
        else:
            label = "Intermediate Motility State"
            
        cluster_annotations[c_id] = label
        
    return cluster_annotations


def perform_cell_phenotyping(
    df_kinematics: pd.DataFrame = None,
    df_morphology: pd.DataFrame = None,
    num_clusters: int = 3,
    output_dir: str = "phenotyping_results"
) -> tuple[pd.DataFrame, dict]:
    """
    Executes full unsupervised phenotyping pipeline on cell kinematics and morphology data.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if (df_kinematics is None or df_kinematics.empty) and (df_morphology is None or df_morphology.empty):
        print("[Phenotyping] Extracting dataset features for phenotyping...")
        masks = load_masks()
        model = get_trackastra_model()
        tracked_masks, _ = run_cell_tracking(masks=masks, model=model)
        df_morphology = extract_dataset_morphology(tracked_masks)
        df_kinematics = compute_cell_kinematics(df_morphology)
    elif df_kinematics is None or df_kinematics.empty:
        df_kinematics = compute_cell_kinematics(df_morphology)

    # Determine cell ID column name
    if "label_id" in df_kinematics.columns:
        cell_id_col = "label_id"
    else:
        cell_id_col = "cell_id"

    df_feat = df_kinematics.copy()

    # Compute and merge per-cell mean morphometrics if available
    if df_morphology is not None and not df_morphology.empty:
        m_id_col = "label_id" if "label_id" in df_morphology.columns else "cell_id"
        morph_means = df_morphology.groupby(m_id_col)[["area", "circularity", "eccentricity"]].mean().reset_index()
        morph_means.columns = [cell_id_col, "mean_area", "mean_circularity", "mean_eccentricity"]
        
        # Merge if columns not already present
        if "mean_area" not in df_feat.columns:
            df_feat = pd.merge(df_feat, morph_means, on=cell_id_col, how="inner")

    if "directionality_ratio" in df_feat.columns and "directionality" not in df_feat.columns:
        df_feat["directionality"] = df_feat["directionality_ratio"]

    df_kinematics = df_feat

    print(f"[Phenotyping] Performing phenotyping on {len(df_kinematics)} cell trajectories...")

    # Feature Matrix Selection
    feature_cols = ["mean_area", "mean_circularity", "mean_eccentricity", "mean_speed", "net_displacement", "directionality"]
    X_raw = df_kinematics[feature_cols].values

    # 1. Standardize
    X_std, X_mean, X_scale = standardize_features(X_raw)

    # 2. PCA Projection
    X_pca, exp_var = compute_pca_2d(X_std)

    # 3. K-Means Clustering
    np.random.seed(42)
    centroids, labels = vq.kmeans2(X_std, num_clusters, minit="points", iter=30)

    # Build Result DataFrame
    df_res = df_kinematics.copy()
    df_res["cluster"] = labels
    df_res["pca_1"] = X_pca[:, 0]
    df_res["pca_2"] = X_pca[:, 1]

    # Automatic Biological State Annotation
    cluster_labels_map = auto_annotate_clusters(df_res, num_clusters)
    df_res["phenotype_label"] = df_res["cluster"].map(cluster_labels_map)

    # Save CSV dataset
    csv_file = out_path / "cell_phenotypes.csv"
    df_res.to_csv(csv_file, index=False)
    print(f"[Phenotyping] Exported phenotype dataset to '{csv_file.resolve()}'")

    # -------------------------------------------------------------
    # 4. Generate Visualizations (PCA & Cluster Profiles)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Plot 1: 2D PCA Cluster Space
    colors = ["#38bdf8", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
    for c_id in range(num_clusters):
        sub = df_res[df_res["cluster"] == c_id]
        p_name = cluster_labels_map[c_id]
        c_color = colors[c_id % len(colors)]
        axes[0].scatter(
            sub["pca_1"], sub["pca_2"],
            c=c_color, label=f"Cluster {c_id}: {p_name} (n={len(sub)})",
            s=80, alpha=0.85, edgecolors="black"
        )
        # Annotate Cell IDs
        for _, row in sub.iterrows():
            cid = row.get("label_id", row.get("cell_id", 0))
            axes[0].annotate(f"#{int(cid)}", (row["pca_1"] + 0.05, row["pca_2"] + 0.05), fontsize=8, color="#1e293b")

    axes[0].set_title(f"Unsupervised Cell Phenotype PCA Space\n(PC1: {exp_var[0]*100:.1f}%, PC2: {exp_var[1]*100:.1f}% Variance)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Principal Component 1 (PC1)")
    axes[0].set_ylabel("Principal Component 2 (PC2)")
    axes[0].legend(fontsize=9, loc="best")

    # Plot 2: Cluster Feature Comparison Bar Chart
    cluster_means = df_res.groupby("phenotype_label")[["mean_speed", "mean_area", "directionality"]].mean()
    # Normalize for comparison plot
    norm_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-6)
    
    norm_means.plot(kind="bar", ax=axes[1], colormap="Set2", edgecolor="black", width=0.7)
    axes[1].set_title("Normalized Phenotypic Profile Traits by Cluster", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Cell Phenotype Cluster")
    axes[1].set_ylabel("Normalized Score (0 to 1)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right", fontsize=9)
    axes[1].legend(["Mean Speed", "Mean Area", "Directionality"], fontsize=9)

    plt.tight_layout()
    plot_file = out_path / "cell_phenotypes_pca.png"
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()
    print(f"[Phenotyping] Summary figure saved to '{plot_file.resolve()}'")

    summary_info = {
        "num_clusters": num_clusters,
        "cluster_counts": df_res["cluster"].value_counts().to_dict(),
        "cluster_labels": cluster_labels_map,
        "pca_variance_explained": [round(float(exp_var[0]), 4), round(float(exp_var[1]), 4)]
    }

    return df_res, summary_info


if __name__ == "__main__":
    df_pheno, summary = perform_cell_phenotyping()
    print(f"Phenotyping Completed! Clusters Summary: {summary}")
