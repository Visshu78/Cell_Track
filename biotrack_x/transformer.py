"""
BioTrack-X Module 2: Spatio-Temporal Graph Transformer (ST-GT).

Architecture:
  Unlike TrackFormer/MOTR (which only model 2 consecutive frames),
  BioTrack-X's ST-GT uses multi-head cross-frame attention across ALL T frames
  simultaneously, giving each cell query a global view of the full sequence.

Key components:

  1. CellObjectQuery:
     Learnable d-dimensional query vector per cell track.
     Propagates cell identity across frames (adapting TrackFormer's track query).

  2. SpatioTemporalAttention:
     Multi-head cross-attention where each cell query Q attends to spatial
     feature keys K from ALL T frames. The attention logit is modulated by:
       - Temporal distance penalty M_temporal (decays with frame distance).
       - Aleatoric uncertainty weight W_sigma (down-weights uncertain detections).

  3. DivisionQueryHead (novel):
     Unlike TrackFormer/MOTR which cannot natively handle cell division,
     the DivisionQueryHead detects when a cell query should split into 2
     child queries. Inspired by HOCT's edge-centric division modeling, but
     integrated directly into the transformer query mechanism.

     Division prediction:
       is_dividing = sigmoid(MLP(q_t)) > threshold
       If True: spawn 2 child queries q_child1, q_child2 from q_t.

  4. TrackHead:
     Linear projection from query representation → (y, x, confidence) triplet.
     Centroid prediction refined by TTA aleatoric uncertainty.
"""

from typing import Tuple, List, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Positional Encoding (2D spatial + 1D temporal)
# ---------------------------------------------------------------------------

class SpatioTemporalPositionalEncoding(nn.Module):
    """
    Combines 2D spatial sinusoidal PE with 1D temporal frame embedding.

    Each feature token (t, y, x) gets a unique positional code:
      PE(t, y, x) = PE_spatial(y, x) + PE_temporal(t)
    """

    def __init__(self, d_model: int, max_frames: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        # Learnable temporal embedding per frame index
        self.temporal_embed = nn.Embedding(max_frames, d_model)

    def get_spatial_pe(self, H: int, W: int) -> torch.Tensor:
        """Returns (H*W, d_model) sinusoidal spatial positional encoding."""
        d = self.d_model // 4
        y_pos = torch.arange(H).float().unsqueeze(1)  # (H, 1)
        x_pos = torch.arange(W).float().unsqueeze(1)  # (W, 1)

        div_term = torch.exp(torch.arange(0, d) * -(math.log(10000.0) / d))

        # Y sinusoidal encoding
        pe_y = torch.zeros(H, 2 * d)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)

        # X sinusoidal encoding
        pe_x = torch.zeros(W, 2 * d)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        # Combine: broadcast y (H,1,2d) + x (1,W,2d) → (H,W,4d) = (H,W,d_model)
        pe_y_full = pe_y.unsqueeze(1).expand(H, W, 2 * d)
        pe_x_full = pe_x.unsqueeze(0).expand(H, W, 2 * d)
        pe = torch.cat([pe_y_full, pe_x_full], dim=-1)  # (H, W, d_model)

        return pe.view(H * W, self.d_model)

    def forward(self, features: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        Args:
            features: (1, d, H, W) spatial feature map from encoder.
            frame_idx: Integer frame index.
        Returns:
            tokens: (H*W, d_model) positionally-encoded feature tokens.
        """
        B, d, H, W = features.shape
        tokens = features.view(d, H * W).T  # (H*W, d)

        # Add spatial PE
        spatial_pe = self.get_spatial_pe(H, W)  # (H*W, d)
        tokens = tokens + spatial_pe

        # Add temporal PE
        t_idx = torch.tensor([frame_idx], dtype=torch.long)
        t_embed = self.temporal_embed(t_idx)  # (1, d)
        tokens = tokens + t_embed

        return tokens  # (H*W, d)


# ---------------------------------------------------------------------------
# Spatio-Temporal Multi-Head Attention
# ---------------------------------------------------------------------------

class SpatioTemporalAttention(nn.Module):
    """
    Cross-frame multi-head attention for BioTrack-X.

    Cell queries Q ∈ R^{N×d} attend to feature tokens from ALL T frames:
      K, V ∈ R^{T×(H*W)×d}  (concatenated across frames)

    Attention logit modulation:
      A(q, k_t) = (Q @ K_t^T / √d) + M_temporal(t) + M_sigma(σ²)

    Where:
      M_temporal(t) = -γ * |t_query - t_key|  (temporal distance decay)
      M_sigma(σ²)   = -δ * σ²_cell            (uncertainty down-weighting)

    This is novel vs. TrackFormer (2-frame only) and MOTR (no uncertainty weighting).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        temporal_decay: float = 0.1,
        uncertainty_weight: float = 0.5,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model           = d_model
        self.n_heads           = n_heads
        self.d_head            = d_model // n_heads
        self.temporal_decay    = temporal_decay
        self.uncertainty_weight = uncertainty_weight

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)

    def _temporal_bias(
        self, query_frame: int, key_frames: List[int], seq_len: int
    ) -> torch.Tensor:
        """
        Compute temporal distance decay bias: (1, n_heads, 1, T*seq_len).
        Tokens from frames further away are penalized.
        """
        biases = []
        for t in key_frames:
            dist = abs(query_frame - t)
            bias = -self.temporal_decay * dist
            biases.extend([bias] * seq_len)
        return torch.tensor(biases, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, TS)

    def forward(
        self,
        queries: torch.Tensor,          # (N, d) — cell object queries
        key_tokens: torch.Tensor,       # (T*S, d) — all frame tokens concatenated
        sigma_sq: Optional[torch.Tensor] = None,  # (N, 2) — TTA uncertainty per query
        query_frame: int = 0,
        key_frames: Optional[List[int]] = None,
        tokens_per_frame: int = 128,
    ) -> torch.Tensor:
        """
        Args:
            queries:   (N, d) cell object queries.
            key_tokens: (T*S, d) concatenated spatial tokens from all T frames,
                        each frame has S = H/8 * W/8 tokens.
            sigma_sq:   (N, 2) per-cell uncertainty from TTA (optional).
            query_frame: current query frame index (for temporal bias).
            key_frames:  list of T frame indices corresponding to key_tokens chunks.
            tokens_per_frame: S = number of tokens per frame.
        Returns:
            out: (N, d) updated query representations.
        """
        N = queries.shape[0]
        TS = key_tokens.shape[0]

        # Project
        Q = self.q_proj(queries)        # (N, d)
        K = self.k_proj(key_tokens)     # (TS, d)
        V = self.v_proj(key_tokens)     # (TS, d)

        # Reshape to multi-head: (n_heads, N or TS, d_head)
        Q = Q.view(N, self.n_heads, self.d_head).permute(1, 0, 2)   # (H, N, d_head)
        K = K.view(TS, self.n_heads, self.d_head).permute(1, 0, 2)  # (H, TS, d_head)
        V = V.view(TS, self.n_heads, self.d_head).permute(1, 0, 2)  # (H, TS, d_head)

        # Scaled dot-product attention
        scale   = math.sqrt(self.d_head)
        attn    = torch.bmm(Q, K.transpose(1, 2)) / scale  # (H, N, TS)

        # Add temporal distance bias
        if key_frames is not None:
            t_bias = self._temporal_bias(query_frame, key_frames, tokens_per_frame)
            t_bias = t_bias.expand(self.n_heads, N, TS)
            attn   = attn + t_bias

        # Add uncertainty down-weighting: cells with high σ² get less attention weight
        if sigma_sq is not None and N > 0:
            # σ² averaged over y/x axes → (N,) scalar uncertainty per cell
            cell_uncert = sigma_sq.mean(dim=-1)  # (N,)
            uncert_bias = -self.uncertainty_weight * cell_uncert  # (N,)
            # Broadcast: (H, N, TS) - expand uncert_bias across TS
            uncert_bias = uncert_bias.unsqueeze(0).unsqueeze(-1).expand(self.n_heads, N, TS)
            attn = attn + uncert_bias

        attn = F.softmax(attn, dim=-1)  # (H, N, TS)
        attn = self.dropout(attn)

        out = torch.bmm(attn, V)        # (H, N, d_head)
        out = out.permute(1, 0, 2).contiguous().view(N, self.d_model)  # (N, d)
        out = self.out_proj(out)

        # Residual + LayerNorm
        return self.norm(out + queries)


# ---------------------------------------------------------------------------
# Division Query Head (novel — edge-centric mitosis detection)
# ---------------------------------------------------------------------------

class DivisionQueryHead(nn.Module):
    """
    Edge-centric cell division detector integrated into the transformer.

    Novel design:
      - Each cell query q_t is scored by a small MLP: P(division) = σ(MLP(q_t)).
      - If P(division) > threshold, the query is flagged as dividing.
      - Two child queries are spawned: q_child = W_child @ q_parent + noise.
      - This integrates division detection into the transformer's computation graph,
        unlike Kaiser MHT (separate MBM step) and DeepKymoTracker (heuristic rules).

    Loss:
      L_div = BCE(P(division), y_division)
      where y_division = 1 if the cell actually divided (from ground-truth or
      Trackastra pseudo-labels), 0 otherwise.
    """

    def __init__(self, d_model: int = 128, division_threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = division_threshold

        # Division scorer MLP: q → P(dividing) ∈ [0, 1]
        self.division_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Child query projection: parent query → 2 daughter queries
        # Each child gets the parent representation + unique learnable offset
        self.child1_proj = nn.Linear(d_model, d_model)
        self.child2_proj = nn.Linear(d_model, d_model)

        # Track prediction head: query → (y, x, confidence)
        self.track_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 3),  # [y_norm, x_norm, confidence]
        )

    def forward(
        self, queries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: (N, d) — cell object query representations.
        Returns:
            div_probs:      (N, 1) division probability per query.
            div_mask:       (N,) boolean mask of dividing cells.
            child_queries:  (M, 2, d) child query pairs for M dividing cells.
            track_preds:    (N, 3) [y_norm, x_norm, confidence] per query.
        """
        # Division probability
        div_probs = self.division_mlp(queries)  # (N, 1)
        div_mask  = (div_probs.squeeze(-1) > self.threshold)  # (N,)

        # Spawn child queries for dividing cells
        dividing_queries = queries[div_mask]         # (M, d)
        if dividing_queries.shape[0] > 0:
            child1 = self.child1_proj(dividing_queries)  # (M, d)
            child2 = self.child2_proj(dividing_queries)  # (M, d)
            child_queries = torch.stack([child1, child2], dim=1)  # (M, 2, d)
        else:
            child_queries = torch.zeros(0, 2, queries.shape[-1])

        # Track centroid predictions
        track_preds = self.track_head(queries)  # (N, 3)
        track_preds = torch.sigmoid(track_preds)  # normalize to [0,1]

        return div_probs, div_mask, child_queries, track_preds


# ---------------------------------------------------------------------------
# Full ST-GT Transformer Layer
# ---------------------------------------------------------------------------

class STGTLayer(nn.Module):
    """
    One layer of the Spatio-Temporal Graph Transformer.

    Composes: SpatioTemporalAttention → FFN → DivisionQueryHead.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = SpatioTemporalAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        key_tokens: torch.Tensor,
        sigma_sq: Optional[torch.Tensor] = None,
        query_frame: int = 0,
        key_frames: Optional[List[int]] = None,
        tokens_per_frame: int = 128,
    ) -> torch.Tensor:
        # Self-attention with residual
        attended = self.attention(
            queries, key_tokens, sigma_sq,
            query_frame, key_frames, tokens_per_frame,
        )

        # FFN with residual
        out = self.norm2(attended + self.dropout(self.ffn(attended)))
        return out


# ---------------------------------------------------------------------------
# Full ST-GT Transformer (multi-layer)
# ---------------------------------------------------------------------------

class SpatioTemporalGraphTransformer(nn.Module):
    """
    Multi-layer Spatio-Temporal Graph Transformer for BioTrack-X.

    Processes all T frames simultaneously:
      1. Concatenate all frame tokens: (T*S, d).
      2. Apply n_layers ST-GT layers to update cell object queries.
      3. Apply DivisionQueryHead to detect and spawn daughter queries.

    Cell object queries Q ∈ R^{N_max × d} are maintained across frames
    and updated per-frame, mimicking TrackFormer's track query propagation
    but with global temporal context instead of local 2-frame context.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        n_max_cells: int = 64,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        division_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.n_max_cells = n_max_cells

        # Learnable cell object queries (N_max × d)
        self.cell_queries = nn.Parameter(torch.randn(n_max_cells, d_model) * 0.02)

        # Positional encoding module
        self.pos_enc = SpatioTemporalPositionalEncoding(d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            STGTLayer(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Division + track prediction head
        self.div_head = DivisionQueryHead(d_model=d_model, division_threshold=division_threshold)

        # Learnable "no-cell" null query (for empty slots)
        self.null_query = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        features: List[torch.Tensor],      # list of T tensors (1, d, H', W')
        mu_seq: List[torch.Tensor],         # list of T centroid tensors (N_t, 2)
        sigma_seq: List[torch.Tensor],      # list of T uncertainty tensors (N_t, 2)
        ids_seq: List[torch.Tensor],        # list of T cell_id tensors (N_t,)
        active_n_cells: int = 32,
    ) -> Dict:
        """
        Args:
            features:    CNN feature maps for each frame.
            mu_seq:      TTA centroid predictions per frame.
            sigma_seq:   TTA uncertainty estimates per frame.
            ids_seq:     Cell IDs detected per frame.
            active_n_cells: Number of active queries to use (≤ n_max_cells).
        Returns:
            dict with keys: track_preds, div_probs, div_masks, child_queries,
                            updated_queries, frame_assignments.
        """
        T = len(features)
        N = min(active_n_cells, self.n_max_cells)

        # Initialize queries from learnable bank
        queries = self.cell_queries[:N].clone()  # (N, d)

        # Build positionally-encoded tokens for ALL frames: list of (S_t, d)
        all_tokens = []
        key_frames = []
        for t, feat in enumerate(features):
            tokens = self.pos_enc(feat, frame_idx=t)  # (S_t, d)
            all_tokens.append(tokens)
            key_frames.append(t)

        # Concatenate all frame tokens: (T*S, d)
        key_tokens = torch.cat(all_tokens, dim=0)      # (T*S, d)
        tokens_per_frame = all_tokens[0].shape[0] if all_tokens else 128

        # Aggregate uncertainty: use mean across all frames
        if any(s.shape[0] > 0 for s in sigma_seq):
            valid_sigmas = [s for s in sigma_seq if s.shape[0] > 0]
            mean_sigma = valid_sigmas[0].mean(dim=0, keepdim=True).expand(N, 2)
        else:
            mean_sigma = None

        # Apply ST-GT layers — each refines queries with global temporal context
        for layer in self.layers:
            queries = layer(
                queries,
                key_tokens,
                sigma_sq=mean_sigma,
                query_frame=T // 2,      # middle frame as reference
                key_frames=key_frames,
                tokens_per_frame=tokens_per_frame,
            )

        # Division + track prediction
        div_probs, div_mask, child_queries, track_preds = self.div_head(queries)

        return {
            "track_preds":     track_preds,     # (N, 3) [y, x, conf]
            "div_probs":       div_probs,        # (N, 1)
            "div_mask":        div_mask,         # (N,) bool
            "child_queries":   child_queries,    # (M, 2, d)
            "updated_queries": queries,          # (N, d)
            "n_active":        N,
        }
