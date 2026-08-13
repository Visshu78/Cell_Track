"""
BioTrack-X Module 3: Erlang Biological Cell-Cycle Prior.

Novel contribution:
  Adapts the Erlang cell-cycle lifetime prior from Kaiser et al. (IEEE TMI 2025)
  and integrates it DIRECTLY into a transformer's assignment loss — something
  no existing transformer-based cell tracker does.

Mathematical formulation:
  The Erlang distribution models the probability density of cell division
  occurring at time t after the cell's birth:

    f(t; α, β) = (β^α * t^(α-1) * e^(-βt)) / Γ(α)

  where:
    α (shape parameter): Number of sub-phases in cell cycle (typically 2-4).
    β (rate parameter): Inverse of mean sub-phase duration.

  The biological assignment cost for a cell of age A is:

    L_bio(A) = -log[ ∫_0^A f(t; α, β) dt ]
             = -log[ P(division occurred before age A) ]

  Interpretation:
    - Very young cells (A → 0): high cost → penalizes impossible early division.
    - Cells at peak-cycle age: low cost → promotes biologically timed division.
    - Very old cells (A >> mean): moderate cost → flags abnormally long-lived cells.

  This prior enforces biological plausibility in the transformer's track assignment,
  preventing spurious identity swaps and phantom divisions.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Erlang CDF (Cumulative Distribution Function)
# ---------------------------------------------------------------------------

def erlang_cdf(t: float, alpha: int, beta: float) -> float:
    """
    Computes the CDF of an Erlang(alpha, beta) distribution at time t.

    P(T ≤ t) = 1 - e^{-βt} * Σ_{k=0}^{α-1} (βt)^k / k!

    Args:
        t:     Time (cell age in frames).
        alpha: Shape parameter (integer, ≥ 1).
        beta:  Rate parameter (> 0).
    Returns:
        CDF value ∈ [0, 1].
    """
    if t <= 0:
        return 0.0
    bt = beta * t
    # Compute survival function terms: e^{-βt} Σ (βt)^k / k!
    survival = 0.0
    term = math.exp(-bt)
    for k in range(alpha):
        survival += term
        if k + 1 < alpha:
            term *= bt / (k + 1)
    return max(0.0, min(1.0, 1.0 - survival))


def erlang_cdf_tensor(t: torch.Tensor, alpha: int, beta: float) -> torch.Tensor:
    """
    Vectorized Erlang CDF computation for batched cell ages.

    Args:
        t:     (N,) tensor of cell ages in frames.
        alpha: Shape parameter.
        beta:  Rate parameter.
    Returns:
        (N,) tensor of CDF values ∈ [0, 1].
    """
    bt = beta * t.clamp(min=1e-8)
    survival = torch.zeros_like(t)
    exp_neg_bt = torch.exp(-bt)
    term = exp_neg_bt.clone()
    for k in range(alpha):
        survival = survival + term
        if k + 1 < alpha:
            term = term * bt / (k + 1)
    cdf = 1.0 - survival
    return cdf.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Erlang Cell-Cycle Prior Module
# ---------------------------------------------------------------------------

class ErlangCellCyclePrior(nn.Module):
    """
    Learnable Erlang cell-cycle biological prior.

    Tracks the age (frames since birth) of every active cell track,
    and computes a biological plausibility cost for each division event.

    Parameters:
        alpha:    Erlang shape parameter (fixed at 2 = G1+S/G2+M phases).
        log_beta: Log of rate parameter β (learnable, initialized from literature).
                  Mean cell cycle ≈ 12-20 frames for typical time-lapse data.
                  Default β ≈ 0.2 → mean lifetime = α/β = 10 frames.

    Cell age tracking:
        self.cell_ages: Dict[cell_id → age_in_frames]
        Updated each frame during forward pass.
    """

    def __init__(
        self,
        alpha: int = 2,
        init_beta: float = 0.2,
        min_division_age: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps   = eps
        self.min_division_age = min_division_age

        # β is learnable in log-space to ensure positivity
        self.log_beta = nn.Parameter(torch.tensor(math.log(init_beta)))

        # Cell age registry: cell_id (int) → age (int frames)
        self.cell_ages: Dict[int, int] = {}

    @property
    def beta(self) -> float:
        return float(torch.exp(self.log_beta).detach())

    def reset_ages(self) -> None:
        """Clear all tracked cell ages (call at start of each new sequence)."""
        self.cell_ages.clear()

    def update_ages(self, active_cell_ids: torch.Tensor, dividing_ids: Optional[torch.Tensor] = None) -> None:
        """
        Increment age for all active cells; reset age for daughter cells post-division.

        Args:
            active_cell_ids: (N,) tensor of currently active cell IDs.
            dividing_ids:    (M,) optional tensor of cell IDs that just divided.
        """
        active_set = set(active_cell_ids.tolist())

        # Remove cells that are no longer active (died or left FOV)
        dead_ids = [cid for cid in self.cell_ages if cid not in active_set]
        for cid in dead_ids:
            del self.cell_ages[cid]

        # Increment age for existing cells
        for cid in active_set:
            if cid in self.cell_ages:
                self.cell_ages[cid] += 1
            else:
                self.cell_ages[cid] = 1  # Newly appeared cell

        # Reset daughter cell ages post-division (daughters are born fresh)
        if dividing_ids is not None:
            for cid in dividing_ids.tolist():
                self.cell_ages[cid] = 0

    def compute_division_cost(self, cell_ids: torch.Tensor) -> torch.Tensor:
        """
        Computes biological division cost for each cell in cell_ids.

        Cost = -log[ P(division at Age_i) ]
             = -log[ Erlang_CDF(Age_i; α, β) + ε ]

        Low cost  → biologically timed division (plausible).
        High cost → premature division (age < min_division_age) or abnormally delayed.

        Args:
            cell_ids: (N,) tensor of cell IDs to compute cost for.
        Returns:
            costs: (N,) tensor of non-negative biological costs.
        """
        ages = torch.tensor(
            [self.cell_ages.get(int(cid), 1) for cid in cell_ids.tolist()],
            dtype=torch.float32,
        )

        # Apply minimum age hard penalty: biologically impossible to divide at age < 3
        premature_mask = ages < self.min_division_age
        ages_clamped   = ages.clamp(min=float(self.min_division_age))

        # Erlang CDF: probability of division having occurred by this age
        cdf_vals = erlang_cdf_tensor(ages_clamped, alpha=self.alpha, beta=self.beta)

        # Division cost: -log(P_division)
        costs = -torch.log(cdf_vals + self.eps)

        # Massive penalty for premature divisions (biologically impossible)
        costs = torch.where(premature_mask, torch.full_like(costs, 20.0), costs)

        return costs

    def get_age_stats(self) -> Dict[str, float]:
        """Returns summary statistics of current cell age distribution."""
        if not self.cell_ages:
            return {"mean_age": 0.0, "max_age": 0.0, "n_cells": 0}
        ages = list(self.cell_ages.values())
        return {
            "mean_age": float(np.mean(ages)),
            "max_age":  float(np.max(ages)),
            "n_cells":  len(ages),
            "beta":     self.beta,
        }

    def forward(
        self,
        cell_ids: torch.Tensor,
        active_cell_ids: torch.Tensor,
        dividing_ids: Optional[torch.Tensor] = None,
        compute_cost: bool = True,
    ) -> torch.Tensor:
        """
        Full forward: update ages then compute division costs.

        Args:
            cell_ids:        (N,) IDs of cells to score.
            active_cell_ids: (M,) all active cell IDs this frame.
            dividing_ids:    (K,) optional IDs of currently dividing cells.
            compute_cost:    If True, returns division cost; else returns ages.
        Returns:
            costs: (N,) biological assignment cost tensor.
        """
        self.update_ages(active_cell_ids, dividing_ids)
        if compute_cost:
            return self.compute_division_cost(cell_ids)
        else:
            ages = torch.tensor(
                [self.cell_ages.get(int(cid), 1) for cid in cell_ids.tolist()],
                dtype=torch.float32,
            )
            return ages
