"""
BioTrack-X: Novel Unified Spatio-Temporal Graph Transformer for Cell Tracking.

Architecture novelties:
  1. End-to-end joint cell tracking + segmentation (no greedy post-processing)
  2. Edge-centric Division Query Head (query splitting for mitosis)
  3. Erlang biological cell-cycle prior integrated into loss
  4. Aleatoric TTA position uncertainty estimation

Reference:
  Inspired by TrackFormer (Meinhardt et al., CVPR 2022),
  HOCT (Higher-Order Cell Tracking, arxiv 2023),
  Kaiser et al. MHT (IEEE TMI 2025),
  and Trackastra (ECCV 2024).
"""

from biotrack_x.model import BioTrackX
from biotrack_x.inference import run_biotrackx_inference

__all__ = ["BioTrackX", "run_biotrackx_inference"]
__version__ = "1.0.0"
