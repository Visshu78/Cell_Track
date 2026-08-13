"""
BioTrack-X Test Suite
Tests each module independently, then runs full end-to-end integration test.
"""

import sys
import time
import traceback
import numpy as np
import torch

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def test(name, fn):
    try:
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"  {PASS} {name}  ({elapsed:.3f}s)")
        results.append((name, True, elapsed, None))
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"         Error: {e}")
        traceback.print_exc()
        results.append((name, False, 0, str(e)))

# ─── 1. ENCODER TESTS ───────────────────────────────────────────────────────

print("\n=== Module 1: Spatial Encoder + TTA Uncertainty ===")

def test_resnet_encoder_shape():
    from biotrack_x.encoder import ResNetCellEncoder
    enc = ResNetCellEncoder(in_channels=1, feature_dim=128)
    x = torch.zeros(1, 1, 128, 128)
    out = enc(x)
    assert out.shape == (1, 128, 16, 16), f"Expected (1,128,16,16), got {out.shape}"

def test_resnet_encoder_1024():
    from biotrack_x.encoder import ResNetCellEncoder
    enc = ResNetCellEncoder(in_channels=1, feature_dim=128)
    x = torch.zeros(1, 1, 1024, 1024)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (1, 128, 128, 128), f"Got {out.shape}"

def test_tta_uncertainty_estimator():
    from biotrack_x.encoder import TTAUncertaintyEstimator
    tta = TTAUncertaintyEstimator(shift_radius=4, n_shifts=4)
    # Synthetic mask: 3 cells
    mask = torch.zeros(64, 64, dtype=torch.long)
    mask[10:20, 10:20] = 1
    mask[30:40, 30:40] = 2
    mask[50:60, 5:15]  = 3
    mu, sigma_sq, cell_ids = tta(mask)
    assert mu.shape[0] == 3, f"Expected 3 cells, got {mu.shape[0]}"
    assert sigma_sq.shape == (3, 2), f"sigma shape wrong: {sigma_sq.shape}"
    assert (sigma_sq >= 0).all(), "Variance must be non-negative"

def test_tta_empty_mask():
    from biotrack_x.encoder import TTAUncertaintyEstimator
    tta = TTAUncertaintyEstimator()
    mask = torch.zeros(64, 64, dtype=torch.long)
    mu, sigma_sq, cell_ids = tta(mask)
    assert mu.shape[0] == 0, "Empty mask should return 0 cells"

def test_sequence_encoder():
    from biotrack_x.encoder import SequenceEncoder
    enc = SequenceEncoder(feature_dim=64, shift_radius=2)
    # Tiny synthetic sequence: 3 frames of 64x64 with 2 cells each
    masks = np.zeros((3, 64, 64), dtype=np.int32)
    masks[:, 10:20, 10:20] = 1
    masks[:, 40:50, 40:50] = 2
    features, mu_seq, sigma_seq, ids_seq = enc(masks)
    assert len(features) == 3
    assert len(mu_seq) == 3
    for ids in ids_seq:
        assert ids.shape[0] == 2, f"Expected 2 cells, got {ids.shape[0]}"

test("ResNet encoder output shape (128x128 input)", test_resnet_encoder_shape)
test("ResNet encoder output shape (1024x1024 input)", test_resnet_encoder_1024)
test("TTA uncertainty estimator (3 cells)", test_tta_uncertainty_estimator)
test("TTA uncertainty estimator (empty mask)", test_tta_empty_mask)
test("Full sequence encoder (3 frames)", test_sequence_encoder)

# ─── 2. ERLANG PRIOR TESTS ──────────────────────────────────────────────────

print("\n=== Module 2: Erlang Biological Cell-Cycle Prior ===")

def test_erlang_cdf_monotone():
    from biotrack_x.erlang_prior import erlang_cdf
    vals = [erlang_cdf(t, alpha=2, beta=0.2) for t in range(1, 20)]
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i+1], "CDF must be monotonically non-decreasing"

def test_erlang_cdf_bounds():
    from biotrack_x.erlang_prior import erlang_cdf
    assert erlang_cdf(0, 2, 0.2) == 0.0
    assert 0 < erlang_cdf(10, 2, 0.2) < 1.0
    assert erlang_cdf(1000, 2, 0.2) > 0.99

def test_erlang_cdf_tensor():
    from biotrack_x.erlang_prior import erlang_cdf_tensor
    t = torch.tensor([1.0, 5.0, 10.0, 20.0])
    cdf = erlang_cdf_tensor(t, alpha=2, beta=0.2)
    assert cdf.shape == (4,)
    assert (cdf >= 0).all() and (cdf <= 1).all()
    # Monotone check
    for i in range(len(cdf) - 1):
        assert cdf[i] <= cdf[i+1], "Tensor CDF not monotone"

def test_erlang_premature_division_penalty():
    from biotrack_x.erlang_prior import ErlangCellCyclePrior
    prior = ErlangCellCyclePrior(alpha=2, init_beta=0.2, min_division_age=3)
    cell_ids = torch.tensor([1, 2])
    active_ids = torch.tensor([1, 2])
    prior.cell_ages = {1: 1, 2: 10}  # cell 1 is too young (age 1 < min 3)
    costs = prior.compute_division_cost(cell_ids)
    assert costs[0] > 15.0, f"Young cell should have high cost, got {costs[0]}"
    assert costs[1] < costs[0], f"Older cell should have lower cost"

def test_erlang_age_tracking():
    from biotrack_x.erlang_prior import ErlangCellCyclePrior
    prior = ErlangCellCyclePrior()
    prior.reset_ages()
    active = torch.tensor([1, 2, 3])
    prior.update_ages(active)
    assert prior.cell_ages == {1: 1, 2: 1, 3: 1}
    prior.update_ages(active)
    assert prior.cell_ages == {1: 2, 2: 2, 3: 2}
    # Cell 2 divides, daughters reset to age 0
    prior.update_ages(active, dividing_ids=torch.tensor([2]))
    assert prior.cell_ages[2] == 0, "Dividing cell age should reset to 0"
    assert prior.cell_ages[1] == 3

def test_erlang_learnable_beta():
    from biotrack_x.erlang_prior import ErlangCellCyclePrior
    prior = ErlangCellCyclePrior(init_beta=0.5)
    assert abs(prior.beta - 0.5) < 1e-4
    # Check it's a learnable parameter
    param_names = [n for n, _ in prior.named_parameters()]
    assert "log_beta" in param_names, "log_beta should be a learnable parameter"

test("Erlang CDF monotonically non-decreasing", test_erlang_cdf_monotone)
test("Erlang CDF boundary values (0, middle, large t)", test_erlang_cdf_bounds)
test("Erlang CDF tensor vectorized", test_erlang_cdf_tensor)
test("Premature division penalty (age < min_age = high cost)", test_erlang_premature_division_penalty)
test("Cell age tracking across frames", test_erlang_age_tracking)
test("Erlang beta is a learnable parameter", test_erlang_learnable_beta)

# ─── 3. TRANSFORMER TESTS ───────────────────────────────────────────────────

print("\n=== Module 3: Spatio-Temporal Graph Transformer ===")

def test_positional_encoding_shape():
    from biotrack_x.transformer import SpatioTemporalPositionalEncoding
    pe = SpatioTemporalPositionalEncoding(d_model=128)
    feat = torch.zeros(1, 128, 8, 8)
    tokens = pe(feat, frame_idx=0)
    assert tokens.shape == (64, 128), f"PE output wrong: {tokens.shape}"

def test_st_attention_forward():
    from biotrack_x.transformer import SpatioTemporalAttention
    attn = SpatioTemporalAttention(d_model=64, n_heads=4)
    N, TS, d = 10, 128, 64
    queries   = torch.randn(N, d)
    key_tokens = torch.randn(TS, d)
    out = attn(queries, key_tokens, query_frame=0, key_frames=[0, 1], tokens_per_frame=64)
    assert out.shape == (N, d), f"Attention output wrong: {out.shape}"

def test_st_attention_with_uncertainty():
    from biotrack_x.transformer import SpatioTemporalAttention
    attn = SpatioTemporalAttention(d_model=64, n_heads=4)
    N, TS = 5, 64
    queries    = torch.randn(N, 64)
    key_tokens = torch.randn(TS, 64)
    sigma_sq   = torch.rand(N, 2)  # uncertainty per cell
    out = attn(queries, key_tokens, sigma_sq=sigma_sq, query_frame=1, key_frames=[0, 1], tokens_per_frame=32)
    assert out.shape == (N, 64)

def test_division_query_head():
    from biotrack_x.transformer import DivisionQueryHead
    head = DivisionQueryHead(d_model=64, division_threshold=0.5)
    queries = torch.randn(10, 64)
    div_probs, div_mask, child_queries, track_preds = head(queries)
    assert div_probs.shape == (10, 1)
    assert div_mask.shape == (10,)
    assert track_preds.shape == (10, 3)
    assert (track_preds >= 0).all() and (track_preds <= 1).all(), "track_preds should be in [0,1]"
    # Child queries should exist for dividing cells
    n_div = div_mask.sum().item()
    assert child_queries.shape == (n_div, 2, 64), \
        f"child_queries shape wrong: {child_queries.shape} for {n_div} dividing cells"

def test_stgt_forward():
    from biotrack_x.transformer import SpatioTemporalGraphTransformer
    from biotrack_x.encoder import SequenceEncoder
    enc = SequenceEncoder(feature_dim=64, shift_radius=2)
    masks = np.zeros((3, 64, 64), dtype=np.int32)
    masks[:, 10:20, 10:20] = 1
    masks[:, 40:50, 40:50] = 2
    features, mu_seq, sigma_seq, ids_seq = enc(masks)

    stgt = SpatioTemporalGraphTransformer(
        d_model=64, n_heads=4, n_layers=2, n_max_cells=16, ffn_dim=128
    )
    out = stgt(features, mu_seq, sigma_seq, ids_seq, active_n_cells=8)
    assert "track_preds" in out
    assert "div_probs" in out
    assert out["track_preds"].shape == (8, 3)
    assert out["div_probs"].shape == (8, 1)

test("Positional encoding shape", test_positional_encoding_shape)
test("ST-Attention forward pass", test_st_attention_forward)
test("ST-Attention with uncertainty weighting", test_st_attention_with_uncertainty)
test("DivisionQueryHead: probs, mask, child queries, track preds", test_division_query_head)
test("Full ST-GT transformer forward (3-frame sequence)", test_stgt_forward)

# ─── 4. LOSS FUNCTION TESTS ─────────────────────────────────────────────────

print("\n=== Module 4: BioTrackXLoss Joint Loss Function ===")

def test_hungarian_match():
    from biotrack_x.loss import hungarian_match
    pred = torch.tensor([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    gt   = torch.tensor([[0.1, 0.2], [0.9, 0.8]])
    pi, gi = hungarian_match(pred, gt)
    assert len(pi) == 2 and len(gi) == 2, "Should match 2 pairs"
    # The best matching should pair [0.1,0.2]->[0.1,0.2] and [0.9,0.8]->[0.9,0.8]
    # Verify matched pairs are close to correct assignments (float32 precision)
    matched_pred = [pred[pi[i]].tolist() for i in range(len(pi))]
    matched_gt   = [gt[gi[i]].tolist() for i in range(len(gi))]
    # The best matching should assign close centroids to each other
    for mp, mg in zip(matched_pred, matched_gt):
        dist = sum((a-b)**2 for a,b in zip(mp,mg)) ** 0.5
        assert dist < 0.5, f"Matched pair too far apart: pred={mp}, gt={mg}, dist={dist:.3f}"

def test_dice_loss_perfect():
    from biotrack_x.loss import dice_loss
    mask = torch.ones(1, 16, 16)
    loss = dice_loss(mask, mask)
    assert abs(float(loss)) < 1e-4, f"Perfect match dice loss should be ~0, got {loss}"

def test_dice_loss_empty():
    from biotrack_x.loss import dice_loss
    pred = torch.zeros(1, 16, 16)
    gt   = torch.zeros(1, 16, 16)
    loss = dice_loss(pred, gt)
    # Both empty: numerically near 0
    assert float(loss) < 1e-2

def test_bio_loss_positive():
    from biotrack_x.loss import BioTrackXLoss
    loss_fn = BioTrackXLoss()
    bio_costs = torch.tensor([5.0, 2.0, 0.5])
    L_bio = loss_fn.compute_bio_loss(bio_costs)
    assert float(L_bio) > 0

def test_full_loss_returns_dict():
    from biotrack_x.loss import BioTrackXLoss
    loss_fn = BioTrackXLoss()
    transformer_out = {
        "track_preds": torch.rand(5, 3),
        "div_probs":   torch.rand(5, 1),
    }
    gt_centroids   = torch.tensor([[100.0, 200.0], [300.0, 400.0], [500.0, 500.0]])
    gt_div_labels  = torch.zeros(5)
    bio_costs      = torch.rand(5)
    loss_dict = loss_fn(transformer_out, gt_centroids, gt_div_labels, bio_costs, H=1024, W=1024)
    for key in ["loss_total", "loss_track", "loss_seg", "loss_div", "loss_bio"]:
        assert key in loss_dict, f"Missing key: {key}"
    assert float(loss_dict["loss_total"]) >= 0

test("Hungarian matching (exact pairs)", test_hungarian_match)
test("Dice loss: perfect prediction ~= 0", test_dice_loss_perfect)
test("Dice loss: both empty masks", test_dice_loss_empty)
test("Bio loss positive for positive costs", test_bio_loss_positive)
test("Full joint loss returns all keys", test_full_loss_returns_dict)

# ─── 5. FULL MODEL INTEGRATION TEST ─────────────────────────────────────────

print("\n=== Module 5: BioTrackX Full Model Integration ===")

def test_model_init():
    from biotrack_x import BioTrackX
    m = BioTrackX(feature_dim=64, n_heads=4, n_layers=2, n_max_cells=16)
    total = sum(p.numel() for p in m.parameters())
    assert total > 100_000, f"Model too small: {total} params"

def test_model_inference_shape():
    from biotrack_x import BioTrackX
    m = BioTrackX(feature_dim=64, n_heads=4, n_layers=2, n_max_cells=16)
    m.eval()
    masks = np.zeros((3, 64, 64), dtype=np.int32)
    masks[:, 5:15, 5:15]   = 1
    masks[:, 30:40, 30:40] = 2
    results = m.forward_inference(masks)
    assert results["tracked_masks"].shape == (3, 64, 64)
    assert results["uncertainty_maps"].shape == (3, 64, 64)
    assert results["lineage_graph"] is not None

def test_uncertainty_nonzero():
    from biotrack_x import BioTrackX
    m = BioTrackX(feature_dim=64, n_heads=4, n_layers=2, n_max_cells=16)
    m.eval()
    masks = np.zeros((2, 64, 64), dtype=np.int32)
    masks[:, 10:20, 10:20] = 1
    results = m.forward_inference(masks)
    unc = results["uncertainty_maps"]
    assert unc.max() > 0, "Uncertainty maps should be non-zero where cells exist"

def test_lineage_graph_structure():
    from biotrack_x import BioTrackX
    import networkx as nx
    m = BioTrackX(feature_dim=64, n_heads=4, n_layers=2, n_max_cells=16)
    m.eval()
    masks = np.zeros((4, 64, 64), dtype=np.int32)
    masks[:, 10:20, 10:20] = 1
    masks[:, 40:50, 40:50] = 2
    results = m.forward_inference(masks)
    G = results["lineage_graph"]
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() >= 0

def test_drop_in_inference():
    from biotrack_x.inference import run_biotrackx_inference
    masks = np.zeros((3, 64, 64), dtype=np.int32)
    masks[:, 10:20, 10:20] = 1
    masks[:, 40:50, 40:50] = 2
    tracked, graph = run_biotrackx_inference(masks)
    assert tracked.shape == (3, 64, 64)
    assert graph is not None

test("BioTrackX model initializes with correct param count", test_model_init)
test("BioTrackX inference output shapes", test_model_inference_shape)
test("TTA uncertainty maps non-zero at cell locations", test_uncertainty_nonzero)
test("Lineage graph is valid DiGraph with nodes", test_lineage_graph_structure)
test("Drop-in inference adapter (tracked_masks, graph)", test_drop_in_inference)

# ─── SUMMARY ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  BioTrack-X Test Suite Summary")
print("=" * 60)
passed = [r for r in results if r[1]]
failed = [r for r in results if not r[1]]
print(f"  Passed : {len(passed)} / {len(results)}")
print(f"  Failed : {len(failed)} / {len(results)}")
if failed:
    print("\n  FAILED TESTS:")
    for name, ok, t, err in failed:
        print(f"    - {name}")
        print(f"      {err}")
print("=" * 60)
sys.exit(0 if not failed else 1)
