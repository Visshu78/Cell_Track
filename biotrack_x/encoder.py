"""
BioTrack-X Module 1: Spatial Encoder with TTA-based Aleatoric Uncertainty.

Architecture:
  - ResNetCellEncoder: Lightweight ResNet-18-style CNN to extract per-frame
    spatial feature maps F ∈ R^{H/8 × W/8 × d} from binary cell mask inputs.
  - TTAUncertaintyEstimator: Runs 4× spatial shift augmentations and computes
    per-cell centroid mean μ̂ and variance σ̂² (aleatoric position uncertainty),
    adapting the TTA approach from Kaiser et al. (IEEE TMI 2025).

Novel contribution vs. existing work:
  - TrackFormer/MOTR: No per-cell uncertainty estimation at all.
  - Kaiser et al.: TTA uncertainty exists but is decoupled from the transformer.
  - BioTrack-X: TTA uncertainty is propagated directly into cross-frame attention
    as positional confidence weights, allowing the transformer to down-weight
    uncertain detections during association.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Lightweight ResNet-18-style Convolutional Block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Standard residual block: two 3×3 convolutions with skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetCellEncoder(nn.Module):
    """
    Lightweight ResNet-18-inspired encoder for cell mask images.

    Input:  (B, 1, H, W)  — single-channel binary/integer mask frame
    Output: (B, d, H/8, W/8) — dense spatial feature map

    Downsamples 3× with stride-2 convolutions (factor of 8 total),
    matching Deformable DETR / TrackFormer encoder conventions.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        d = feature_dim

        # Stem: stride-2 downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d // 4, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(d // 4),
            nn.ReLU(inplace=True),
        )

        # Stage 1: d/4 → d/2, stride 2
        self.stage1 = nn.Sequential(
            nn.Conv2d(d // 4, d // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            ResBlock(d // 2),
        )

        # Stage 2: d/2 → d, stride 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(d // 2, d, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d),
            ResBlock(d),
        )

        # Projection to final feature dimension
        self.proj = nn.Conv2d(d, d, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.proj(x)


# ---------------------------------------------------------------------------
# TTA Aleatoric Uncertainty Estimator
# ---------------------------------------------------------------------------

class TTAUncertaintyEstimator(nn.Module):
    """
    Test-Time Augmentation (TTA) based aleatoric uncertainty estimator.

    Method (adapted from Kaiser et al., IEEE TMI 2025):
      1. For each frame, apply K spatial shift augmentations (roll operations).
      2. Extract cell centroids from each shifted version.
      3. Compute mean centroid μ̂ and position variance σ̂² across K augmentations.
      4. σ̂² serves as the aleatoric (data-level) uncertainty for each detected cell.

    The uncertainty σ̂² is passed into the ST-GT as a confidence weight:
      - High σ̂² → the transformer assigns lower association confidence.
      - Low σ̂²  → the transformer trusts the detection strongly.
    """

    def __init__(self, shift_radius: int = 4, n_shifts: int = 4) -> None:
        """
        Args:
            shift_radius: Maximum pixel radius for spatial shift augmentation.
            n_shifts: Number of shift augmentation samples (K).
        """
        super().__init__()
        self.shift_radius = shift_radius
        self.n_shifts = n_shifts

        # Precompute fixed shift directions (up, down, left, right)
        r = shift_radius
        self.shifts: List[Tuple[int, int]] = [
            (-r, 0), (r, 0), (0, -r), (0, r)
        ]

    @torch.no_grad()
    def extract_centroids_from_mask(
        self, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract cell centroids and IDs from a 2D integer mask.

        Args:
            mask: (H, W) integer tensor, 0=background, 1..N=cell IDs.
        Returns:
            centroids: (N, 2) float tensor of (y, x) centroid coordinates.
            cell_ids:  (N,) integer tensor of unique cell label IDs.
        """
        cell_ids = mask.unique()
        cell_ids = cell_ids[cell_ids > 0]

        centroids = []
        for cid in cell_ids:
            yx = (mask == cid).nonzero(as_tuple=False).float()
            centroids.append(yx.mean(dim=0))  # (2,)

        if len(centroids) == 0:
            return torch.zeros(0, 2), torch.zeros(0, dtype=torch.long)

        return torch.stack(centroids), cell_ids

    @torch.no_grad()
    def forward(
        self, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate aleatoric centroid uncertainty for all cells in a frame.

        Args:
            mask: (H, W) integer mask tensor.
        Returns:
            mu:       (N, 2) mean centroid positions [y, x].
            sigma_sq: (N, 2) per-axis centroid variance (aleatoric uncertainty).
            cell_ids: (N,) cell label IDs.
        """
        H, W = mask.shape
        all_centroids: List[torch.Tensor] = []

        # Collect centroids under each spatial shift augmentation
        for dy, dx in self.shifts:
            shifted = torch.roll(mask, shifts=(dy, dx), dims=(0, 1))
            centroids, cell_ids = self.extract_centroids_from_mask(shifted)
            if centroids.shape[0] == 0:
                all_centroids.append(torch.zeros(0, 2))
            else:
                # Undo shift to get original-space centroid
                offset = torch.tensor([[-dy, -dx]], dtype=torch.float32)
                all_centroids.append(centroids + offset)

        # Use original (unshifted) mask as reference
        mu, cell_ids = self.extract_centroids_from_mask(mask)
        N = mu.shape[0]

        if N == 0:
            return (
                torch.zeros(0, 2),
                torch.zeros(0, 2),
                torch.zeros(0, dtype=torch.long),
            )

        # Stack shifted centroids → (K, N, 2) and compute variance
        valid = [c for c in all_centroids if c.shape[0] == N]
        if len(valid) >= 2:
            stacked = torch.stack(valid, dim=0)  # (K, N, 2)
            sigma_sq = stacked.var(dim=0)         # (N, 2)
        else:
            sigma_sq = torch.zeros(N, 2)

        return mu, sigma_sq, cell_ids


# ---------------------------------------------------------------------------
# Full-Sequence Encoder: processes all T frames
# ---------------------------------------------------------------------------

class SequenceEncoder(nn.Module):
    """
    Encodes a full T-frame mask sequence using the ResNet encoder + TTA estimator.

    Input:  masks ∈ R^{T, H, W}  (numpy float32 after normalization)
    Output:
      - features:   list of T tensors, each (1, d, H/8, W/8)
      - mu_seq:     list of T centroid tensors, each (N_t, 2)
      - sigma_seq:  list of T uncertainty tensors, each (N_t, 2)
      - ids_seq:    list of T cell_id tensors, each (N_t,)
    """

    def __init__(self, feature_dim: int = 128, shift_radius: int = 4) -> None:
        super().__init__()
        self.cnn     = ResNetCellEncoder(in_channels=1, feature_dim=feature_dim)
        self.tta     = TTAUncertaintyEstimator(shift_radius=shift_radius)

    def forward(self, masks: np.ndarray):
        """
        Args:
            masks: np.ndarray of shape (T, H, W), integer cell labels.
        Returns:
            features, mu_seq, sigma_seq, ids_seq
        """
        T = masks.shape[0]
        features, mu_seq, sigma_seq, ids_seq = [], [], [], []

        for t in range(T):
            frame_np = masks[t].astype(np.float32)

            # Normalize to [0, 1] for CNN input
            max_val = frame_np.max()
            if max_val > 0:
                frame_norm = frame_np / max_val
            else:
                frame_norm = frame_np

            # CNN feature extraction: (1, 1, H, W) → (1, d, H/8, W/8)
            x = torch.from_numpy(frame_norm).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            with torch.no_grad():
                feat = self.cnn(x)

            # TTA uncertainty estimation
            mask_t = torch.from_numpy(masks[t].astype(np.int64))
            mu, sigma_sq, cell_ids = self.tta(mask_t)

            features.append(feat)
            mu_seq.append(mu)
            sigma_seq.append(sigma_sq)
            ids_seq.append(cell_ids)

        return features, mu_seq, sigma_seq, ids_seq
