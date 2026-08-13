"""
BioTrack-X Module 4: Joint Loss Function.

The BioTrack-X loss unifies four objectives in a single differentiable function:

  L_total = λ1 * L_track + λ2 * L_seg + λ3 * L_div + λ4 * L_bio

Where:
  L_track: Hungarian bipartite matching centroid regression loss.
           Assigns predicted track centroids to ground-truth / pseudo-label centroids
           via optimal transport (Hungarian matching), then computes L2 distance.

  L_seg:   Mask reconstruction loss (dice + BCE).
           Optional — active only when per-cell mask predictions are available.

  L_div:   Division binary cross-entropy loss.
           Penalizes incorrect division predictions (false positives / negatives).

  L_bio:   Erlang biological cell-cycle prior loss.
           Penalizes biologically implausible division timings using the
           Erlang cell-cycle lifetime distribution.

Note on Hungarian matching:
  We use the SciPy linear sum assignment algorithm (equivalent to Hungarian)
  since we want to keep the codebase dependency-light (no detectron2/torchvision).
  This is identical in theory to the DETR/TrackFormer matching approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Hungarian Matching Utilities
# ---------------------------------------------------------------------------

def hungarian_match(
    pred_centroids: torch.Tensor,   # (N, 2) predicted [y, x] normalized
    gt_centroids: torch.Tensor,     # (M, 2) ground-truth [y, x] normalized
) -> Tuple[List[int], List[int]]:
    """
    Optimal bipartite matching between N predicted and M GT centroids.

    Uses Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    Returns (pred_indices, gt_indices) of matched pairs.
    """
    if not _SCIPY_AVAILABLE or pred_centroids.shape[0] == 0 or gt_centroids.shape[0] == 0:
        return [], []

    # Compute pairwise L2 cost matrix: (N, M)
    with torch.no_grad():
        cost = torch.cdist(pred_centroids.float(), gt_centroids.float(), p=2)
    cost_np = cost.detach().cpu().numpy()

    # Hungarian matching
    pred_idx, gt_idx = linear_sum_assignment(cost_np)
    return pred_idx.tolist(), gt_idx.tolist()


# ---------------------------------------------------------------------------
# Dice Loss for segmentation masks
# ---------------------------------------------------------------------------

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation masks.
    pred, target: (B, H, W) float tensors ∈ [0, 1].
    """
    pred   = pred.view(-1)
    target = target.view(-1).float()
    intersection = (pred * target).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)


# ---------------------------------------------------------------------------
# BioTrack-X Joint Loss
# ---------------------------------------------------------------------------

class BioTrackXLoss(nn.Module):
    """
    Joint loss function for BioTrack-X.

    Parameters:
        lambda_track:   Weight for centroid regression loss (default 1.0).
        lambda_seg:     Weight for mask segmentation loss (default 0.5).
        lambda_div:     Weight for division BCE loss (default 1.0).
        lambda_bio:     Weight for Erlang biological prior loss (default 0.3).
        centroid_loss:  'l2' or 'smooth_l1' for centroid regression.
    """

    def __init__(
        self,
        lambda_track: float = 1.0,
        lambda_seg:   float = 0.5,
        lambda_div:   float = 1.0,
        lambda_bio:   float = 0.3,
        centroid_loss: str  = "smooth_l1",
    ) -> None:
        super().__init__()
        self.lambda_track  = lambda_track
        self.lambda_seg    = lambda_seg
        self.lambda_div    = lambda_div
        self.lambda_bio    = lambda_bio
        self.centroid_loss = centroid_loss
        self.bce = nn.BCELoss()

    def compute_track_loss(
        self,
        pred_centroids: torch.Tensor,   # (N, 2) in [0, 1]
        gt_centroids: torch.Tensor,     # (M, 2) in [0, 1]
        H: int = 1024,
        W: int = 1024,
    ) -> torch.Tensor:
        """
        Hungarian-matched centroid regression loss.
        Normalizes GT centroids to [0,1] before matching.
        """
        if pred_centroids.shape[0] == 0 or gt_centroids.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True)

        gt_norm = gt_centroids.float().clone()
        gt_norm[:, 0] /= H  # normalize y
        gt_norm[:, 1] /= W  # normalize x

        pred_idx, gt_idx = hungarian_match(pred_centroids.detach(), gt_norm)
        if len(pred_idx) == 0:
            return torch.tensor(0.0, requires_grad=True)

        matched_pred = pred_centroids[pred_idx]          # (K, 2)
        matched_gt   = gt_norm[gt_idx]                   # (K, 2)

        if self.centroid_loss == "smooth_l1":
            return F.smooth_l1_loss(matched_pred, matched_gt.detach())
        else:
            return F.mse_loss(matched_pred, matched_gt.detach())

    def compute_div_loss(
        self,
        div_probs: torch.Tensor,          # (N, 1) predicted division probabilities
        gt_div_labels: torch.Tensor,      # (N,) binary ground-truth division labels
    ) -> torch.Tensor:
        """Binary cross-entropy loss for division detection."""
        if div_probs.shape[0] == 0:
            return torch.tensor(0.0, requires_grad=True)
        gt = gt_div_labels.float().unsqueeze(-1)  # (N, 1)
        return self.bce(div_probs, gt.detach())

    def compute_bio_loss(self, bio_costs: torch.Tensor) -> torch.Tensor:
        """
        Biological Erlang prior loss.
        bio_costs: (N,) from ErlangCellCyclePrior.compute_division_cost().
        We minimize the expected biological cost over all active cells.
        """
        if bio_costs.shape[0] == 0:
            return torch.tensor(0.0)
        return bio_costs.mean()

    def forward(
        self,
        transformer_out: Dict,
        gt_centroids: torch.Tensor,
        gt_div_labels: Optional[torch.Tensor] = None,
        bio_costs: Optional[torch.Tensor] = None,
        pred_masks: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        H: int = 1024,
        W: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the full BioTrack-X joint loss.

        Args:
            transformer_out: Dict from SpatioTemporalGraphTransformer.forward().
            gt_centroids:    (M, 2) ground-truth or pseudo-label centroids.
            gt_div_labels:   (N,) binary division labels (1=dividing, 0=not).
            bio_costs:       (N,) Erlang biological costs per cell.
            pred_masks:      (N, H, W) optional predicted binary masks.
            gt_masks:        (N, H, W) optional ground-truth masks.
            H, W:            Image dimensions (for centroid normalization).
        Returns:
            loss_dict: Dict with individual losses and total loss.
        """
        track_preds = transformer_out["track_preds"]  # (N, 3) [y, x, conf]
        div_probs   = transformer_out["div_probs"]    # (N, 1)
        N = track_preds.shape[0]

        # Extract predicted centroids (y, x normalized components)
        pred_centroids = track_preds[:, :2]  # (N, 2)

        # --- L_track: Hungarian centroid regression ---
        L_track = self.compute_track_loss(pred_centroids, gt_centroids, H=H, W=W)

        # --- L_div: Division BCE ---
        if gt_div_labels is None:
            gt_div_labels = torch.zeros(N, dtype=torch.float32)
        L_div = self.compute_div_loss(div_probs, gt_div_labels)

        # --- L_bio: Erlang biological prior ---
        if bio_costs is not None:
            L_bio = self.compute_bio_loss(bio_costs)
        else:
            L_bio = torch.tensor(0.0)

        # --- L_seg: Dice + BCE mask loss ---
        if pred_masks is not None and gt_masks is not None:
            L_dice = dice_loss(pred_masks, gt_masks)
            L_bce  = F.binary_cross_entropy(pred_masks, gt_masks.float())
            L_seg  = L_dice + L_bce
        else:
            L_seg = torch.tensor(0.0)

        # --- Total loss ---
        L_total = (
            self.lambda_track * L_track
            + self.lambda_seg   * L_seg
            + self.lambda_div   * L_div
            + self.lambda_bio   * L_bio
        )

        return {
            "loss_total":  L_total,
            "loss_track":  L_track,
            "loss_seg":    L_seg,
            "loss_div":    L_div,
            "loss_bio":    L_bio,
        }
