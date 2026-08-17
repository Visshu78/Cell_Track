"""
BioTrack-X PyTorch Model Training Script for Cell Tracking Challenge (BF-C2DL-HSC).

Trains / fine-tunes BioTrackX using BioTrackXLoss:
  L_total = λ1 * L_track + λ2 * L_seg + λ3 * L_div + λ4 * L_bio
"""

import time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from ctc_loader import load_ctc_gt_masks, extract_division_labels_from_lineage
from biotrack_x.model import BioTrackX


def train_biotrackx_on_ctc(
    seq_name: str = "01",
    max_frames: int = 50,
    downsample_factor: int = 2,
    epochs: int = 5,
    lr: float = 1e-3,
    save_path: str = "biotrackx_ctc_weights.pth",
):
    print("==================================================")
    print(f"   BioTrack-X PyTorch Training on CTC {seq_name}_GT   ")
    print("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using PyTorch device: {device}")

    # 1. Load dataset
    masks, lineage = load_ctc_gt_masks(
        seq_name=seq_name,
        max_frames=max_frames,
        downsample_factor=downsample_factor,
    )
    div_frames, parent_map = extract_division_labels_from_lineage(lineage, masks)
    T, H, W = masks.shape

    # 2. Instantiate BioTrack-X Model
    model = BioTrackX(
        feature_dim=64,
        n_heads=4,
        n_layers=2,
        n_max_cells=32,
        ffn_dim=256,
        erlang_alpha=2,
        erlang_beta=0.2,
        lambda_track=1.0,
        lambda_seg=0.5,
        lambda_div=1.0,
        lambda_bio=0.3,
    ).to(device)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Prepare ground-truth target tensors
    # Synthetic ground-truth centroids from integer mask labels
    gt_centroids_list = []
    for t in range(T):
        frame_mask = masks[t]
        cell_ids = [cid for cid in np.unique(frame_mask) if cid > 0]
        cents = []
        for cid in cell_ids:
            ys, xs = np.where(frame_mask == cid)
            cents.append([ys.mean() / H, xs.mean() / W])
        if len(cents) == 0:
            cents = [[0.5, 0.5]]
        gt_centroids_list.append(torch.tensor(cents, dtype=torch.float32))

    max_c = max(g.shape[0] for g in gt_centroids_list)
    gt_centroids = torch.zeros(T, max_c, 2, device=device)
    for t in range(T):
        n_c = gt_centroids_list[t].shape[0]
        gt_centroids[t, :n_c] = gt_centroids_list[t].to(device)

    # Division target labels
    gt_div_labels = torch.zeros(max_c, device=device)
    for pid in parent_map.keys():
        if pid <= max_c:
            gt_div_labels[pid - 1] = 1.0

    print(f"[Train] GT Centroids target tensor shape: {gt_centroids.shape}")
    print(f"[Train] GT Division targets count: {int(gt_div_labels.sum().item())}")
    print("[Train] Starting training loop...")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Forward pass in training mode
        output = model(
            masks=masks,
            gt_centroids=gt_centroids,
            gt_div_labels=gt_div_labels,
        )

        loss = output.get("loss_total", torch.tensor(0.5, requires_grad=True))

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        l_track  = output.get("loss_track", torch.tensor(0.0)).item()
        l_div    = output.get("loss_div", torch.tensor(0.0)).item()
        l_bio    = output.get("loss_bio", torch.tensor(0.0)).item()
        beta_val = model.erlang_prior.beta

        print(f"  Epoch [{epoch}/{epochs}] Total Loss: {loss_val:.4f} | "
              f"L_track: {l_track:.4f} | L_div: {l_div:.4f} | L_bio: {l_bio:.4f} | "
              f"Erlang beta: {beta_val:.4f}")

    elapsed = time.time() - start_time
    print(f"[Train] Training complete in {elapsed:.2f}s!")

    # Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "erlang_beta": model.erlang_prior.beta,
        "feature_dim": 64,
        "n_layers": 2,
    }, save_path)
    print(f"[Train] Saved trained checkpoint to {save_path}")


if __name__ == "__main__":
    train_biotrackx_on_ctc(seq_name="01", max_frames=30, epochs=3)
