"""
quantizer_unit_test.py
======================
Verifies the quantizer works correctly BEFORE training,
using only synthetic (random) data — no audio files needed.

Run from your project root:
    python quantizer_unit_test.py --config config_stage1.json

All checks must PASS before you start training.
"""

import sys
import json
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.simcodec.modules import (
    GroupQuantizer, SingleQuantizer, Quantizer_module
)
from components.simcodec.model import SimCodec, AttrDict

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

passed = []
failed = []

def ok(name, detail=""):
    s = f"  {GREEN}PASS{RESET}  {name}"
    if detail: s += f"  →  {detail}"
    print(s)
    passed.append(name)

def fail(name, reason):
    print(f"  {RED}FAIL{RESET}  {name}")
    print(f"        {RED}→ {reason}{RESET}")
    failed.append((name, reason))

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {YELLOW}{title}{RESET}")
    print(f"{'─'*60}")


# ===========================================================================
# Helpers
# ===========================================================================

def load_cfg(path):
    with open(path) as f:
        return AttrDict(json.load(f))

def make_fake_latent(B=2, vq_dim=256, T=50, device="cpu"):
    """Synthetic encoder output."""
    return torch.randn(B, vq_dim, T, device=device)


# ===========================================================================
# TEST 1 — Quantizer_module basics
# ===========================================================================

def test_quantizer_module(h, device):
    section("1. Quantizer_module — basic correctness")

    n_e   = h.n_codes
    e_dim = h.vq_dim // h.n_code_groups

    try:
        m = Quantizer_module(n_e, e_dim).to(device)
        ok("Quantizer_module constructed",
           f"n_e={n_e}, e_dim={e_dim}")
    except Exception:
        fail("Quantizer_module constructed", traceback.format_exc())
        return

    # -- buffer device check
    try:
        assert m.target.device.type == device.type or device.type == "cpu", \
            f"target buffer on wrong device: {m.target.device}"
        ok("target buffer on correct device", str(m.target.device))
    except Exception as e:
        fail("target buffer device", str(e))

    # -- forward shape
    try:
        x = torch.randn(64, e_dim, device=device)
        z_q, indices, div_loss = m(x, apply_diversity_loss=True)
        assert z_q.shape     == (64, e_dim),  f"z_q shape {z_q.shape}"
        assert indices.shape == (64,),         f"indices shape {indices.shape}"
        assert indices.min() >= 0,             "negative index"
        assert indices.max() <  n_e,           f"index {indices.max()} >= n_e={n_e}"
        assert div_loss.shape == torch.Size([]),f"div_loss not scalar"
        assert not torch.isnan(div_loss),      "div_loss NaN"
        assert not torch.isinf(div_loss),      "div_loss Inf"
        ok("forward output shapes and ranges",
           f"z_q={tuple(z_q.shape)}, indices range [0,{n_e}), "
           f"div_loss={div_loss.item():.4f}")
    except Exception:
        fail("forward output shapes", traceback.format_exc())

    # -- diversity loss disabled
    try:
        x = torch.randn(16, e_dim, device=device)
        _, _, div_loss_off = m(x, apply_diversity_loss=False)
        assert div_loss_off.item() == 0.0, \
            f"Expected 0 when disabled, got {div_loss_off.item()}"
        ok("diversity loss disabled correctly",
           f"div_loss = {div_loss_off.item()}")
    except Exception:
        fail("diversity loss disabled", traceback.format_exc())

    # -- embedding initialisation: no two entries should be identical
    try:
        w = m.embedding.weight.detach()
        dists = torch.cdist(w, w)
        eye_mask = ~torch.eye(n_e, dtype=torch.bool)
        min_dist = dists[eye_mask].min().item()
        assert min_dist > 1e-6, \
            f"Two codebook entries are identical (min dist={min_dist:.2e})"
        ok("Codebook initialisation — no duplicate entries",
           f"min pairwise dist = {min_dist:.4f}")
    except Exception:
        fail("Codebook initialisation", traceback.format_exc())


# ===========================================================================
# TEST 2 — GroupQuantizer: parallel (not residual) quantization
# ===========================================================================

def test_group_quantizer_parallel(h, device):
    section("2. GroupQuantizer — group quantization is parallel, not residual")

    try:
        gq = GroupQuantizer(h).to(device)
        ok("GroupQuantizer constructed",
           f"n_groups={gq.n_code_groups}, chunk_dim={gq.chunk_dim}")
    except Exception:
        fail("GroupQuantizer constructed", traceback.format_exc())
        return

    B, T = 2, 50
    xin = make_fake_latent(B=B, vq_dim=h.vq_dim, T=T, device=device)

    try:
        quantized, loss, indices = gq(xin)

        # Output shape
        assert quantized.shape == xin.shape, \
            f"quantized {quantized.shape} != input {xin.shape}"
        ok("Output shape matches input", str(tuple(quantized.shape)))

        # Loss is scalar and finite
        assert loss.shape == torch.Size([]), "loss not scalar"
        assert not torch.isnan(loss), "loss is NaN"
        assert not torch.isinf(loss), "loss is Inf"
        ok("VQ loss is scalar and finite", f"{loss.item():.6f}")

        # Indices: list of n_code_groups tensors each (B, T)
        assert isinstance(indices, list), \
            f"indices should be list, got {type(indices)}"
        assert len(indices) == h.n_code_groups, \
            f"Expected {h.n_code_groups} index tensors, got {len(indices)}"
        for g, idx in enumerate(indices):
            assert idx.shape == (B, T), \
                f"Group {g} indices shape {idx.shape} != ({B},{T})"
            assert idx.min() >= 0, f"Group {g}: negative index"
            assert idx.max() <  h.n_codes, \
                f"Group {g}: index {idx.max()} >= n_codes={h.n_codes}"
        ok("Indices: correct structure and range",
           f"{len(indices)} groups, each {tuple(indices[0].shape)}, "
           f"range [0, {h.n_codes})")

    except Exception:
        fail("GroupQuantizer forward", traceback.format_exc())
        return

    # KEY TEST: verify group quantization is truly parallel (not residual).
    # In residual VQ: group 1 quantizes the *residual* of group 0.
    # In group VQ: each group quantizes a *different slice* of the vector.
    # We verify this by checking that:
    #   quantized[:, :chunk_dim] depends ONLY on xin[:, :chunk_dim]
    #   quantized[:, chunk_dim:] depends ONLY on xin[:, chunk_dim:]
    try:
        chunk = gq.chunk_dim

        # Perturb only the FIRST half of the input
        xin_perturbed_first = xin.clone()
        xin_perturbed_first[:, :chunk, :] = torch.randn_like(xin[:, :chunk, :]) * 10

        # Perturb only the SECOND half
        xin_perturbed_second = xin.clone()
        xin_perturbed_second[:, chunk:, :] = torch.randn_like(xin[:, chunk:, :]) * 10

        with torch.no_grad():
            q_orig,    _, idx_orig    = gq(xin)
            q_p_first, _, idx_p_first = gq(xin_perturbed_first)
            q_p_second,_, idx_p_second= gq(xin_perturbed_second)

        # Perturbing first half should change group-0 tokens but NOT group-1 tokens
        g0_changed = (idx_p_first[0] != idx_orig[0]).float().mean().item()
        g1_unchanged = (idx_p_first[1] == idx_orig[1]).float().mean().item()

        assert g0_changed > 0.5, \
            f"Group 0 tokens didn't change when first-half input changed " \
            f"(change rate={g0_changed:.2f}) — group split may be wrong"
        assert g1_unchanged > 0.9, \
            f"Group 1 tokens changed when only first-half input changed " \
            f"(unchanged rate={g1_unchanged:.2f}) — quantization is NOT parallel"

        # Perturbing second half should change group-1 tokens but NOT group-0 tokens
        g1_changed   = (idx_p_second[1] != idx_orig[1]).float().mean().item()
        g0_unchanged = (idx_p_second[0] == idx_orig[0]).float().mean().item()

        assert g1_changed > 0.5, \
            f"Group 1 tokens didn't change when second-half input changed " \
            f"(change rate={g1_changed:.2f})"
        assert g0_unchanged > 0.9, \
            f"Group 0 tokens changed when only second-half input changed " \
            f"(unchanged rate={g0_unchanged:.2f}) — quantization is NOT parallel"

        ok("Group quantization is truly parallel (not residual)",
           f"group-0 sensitivity to half-0: {g0_changed:.2f}, "
           f"group-1 isolation from half-0: {g1_unchanged:.2f}")

    except Exception:
        fail("Parallel group quantization test", traceback.format_exc())


# ===========================================================================
# TEST 3 — Straight-through estimator: gradients flow to encoder
# ===========================================================================

def test_straight_through(h, device):
    section("3. Straight-through estimator — gradient flow")

    gq  = GroupQuantizer(h).to(device)
    xin = make_fake_latent(B=2, vq_dim=h.vq_dim, T=50, device=device)
    xin.requires_grad_(True)

    # -- gradients flow from quantized output back to input (STE)
    try:
        quantized, loss, _ = gq(xin)
        grad_out = torch.randn_like(quantized)
        quantized.backward(grad_out)

        assert xin.grad is not None, "xin.grad is None — STE not working"
        grad_norm = xin.grad.norm().item()
        assert grad_norm > 0, f"xin.grad norm = 0 — STE passes zero gradient"
        ok("STE: gradient flows from quantized output back to input",
           f"|grad| = {grad_norm:.4f}")
    except Exception:
        fail("STE gradient flow", traceback.format_exc())
        return

    # -- codebook loss contributes to embedding gradients
    try:
        gq.zero_grad()
        xin2 = make_fake_latent(B=2, vq_dim=h.vq_dim, T=50, device=device)
        quantized2, loss2, _ = gq(xin2)
        loss2.backward()

        emb_grad = gq.quantizer_modules[0].embedding.weight.grad
        assert emb_grad is not None, \
            "Codebook embedding has no gradient — codebook will not update"
        assert emb_grad.norm().item() > 0, \
            "Codebook embedding gradient norm is zero"
        ok("Codebook embeddings receive gradients from VQ loss",
           f"|grad| = {emb_grad.norm().item():.4f}")
    except Exception:
        fail("Codebook embedding gradients", traceback.format_exc())

    # -- STE: quantized output is on correct computation graph
    try:
        xin3 = make_fake_latent(B=2, vq_dim=h.vq_dim, T=50, device=device)
        xin3.requires_grad_(True)
        quantized3, _, _ = gq(xin3)
        # quantized3 should be on the graph (via STE) so autograd can flow
        assert quantized3.requires_grad, \
            "quantized output does not require grad — STE broken"
        ok("Quantized output is on computation graph (requires_grad=True)")
    except Exception:
        fail("Quantized output computation graph", traceback.format_exc())


# ===========================================================================
# TEST 4 — Usage counters work correctly
# ===========================================================================

def test_usage_counters(h, device):
    section("4. Usage counters for codebook reorganization")

    gq  = GroupQuantizer(h).to(device)

    # Initially zero
    try:
        for g in range(h.n_code_groups):
            buf = gq._get_usage(g)
            assert buf.sum().item() == 0, \
                f"Group {g} usage counter not zero at init: {buf.sum().item()}"
        ok("Usage counters initialised to zero")
    except Exception:
        fail("Usage counter initialisation", traceback.format_exc())

    # Counters increment correctly after forward pass
    try:
        xin = make_fake_latent(B=4, vq_dim=h.vq_dim, T=50, device=device)
        with torch.no_grad():
            gq(xin)

        for g in range(h.n_code_groups):
            buf = gq._get_usage(g)
            total_counts = buf.sum().item()
            # Should equal B * T = 4 * 50 = 200
            assert total_counts == 200, \
                f"Group {g}: expected 200 total counts, got {total_counts}"
        ok("Usage counters increment correctly",
           "4 batches × 50 frames = 200 total per group ✓")
    except Exception:
        fail("Usage counter increment", traceback.format_exc())

    # reset_usage_counts works
    try:
        gq.reset_usage_counts()
        for g in range(h.n_code_groups):
            buf = gq._get_usage(g)
            assert buf.sum().item() == 0, \
                f"Group {g} counter not zero after reset"
        ok("reset_usage_counts() works correctly")
    except Exception:
        fail("reset_usage_counts", traceback.format_exc())

    # Counters accumulate across multiple forward passes
    try:
        for _ in range(3):
            xin = make_fake_latent(B=2, vq_dim=h.vq_dim, T=50, device=device)
            with torch.no_grad():
                gq(xin)
        for g in range(h.n_code_groups):
            buf = gq._get_usage(g)
            assert buf.sum().item() == 300, \
                f"Group {g}: expected 300 after 3 passes, got {buf.sum().item()}"
        ok("Usage counters accumulate across multiple passes",
           "3 × 2 × 50 = 300 ✓")
    except Exception:
        fail("Usage counter accumulation", traceback.format_exc())


# ===========================================================================
# TEST 5 — Stage 2 codebook reorganization math
# ===========================================================================

def test_reorganization(h, device):
    section("5. Stage 2 codebook reorganization")

    gq     = GroupQuantizer(h).to(device)
    top_n  = h.top_n
    top_k  = h.top_k

    # Populate usage counts: give each entry a different usage score
    # so topk selects a deterministic subset
    for g in range(h.n_code_groups):
        buf = gq._get_usage(g)
        buf.copy_(torch.arange(h.n_codes, device=device))

    # Run reorganization
    try:
        new_cb = gq.get_reorganized_codebook(top_n, top_k)
        expected = (top_n * top_k, h.vq_dim)
        assert new_cb.shape == expected, \
            f"Shape {new_cb.shape} != expected {expected}"
        ok("Reorganized codebook shape",
           f"{tuple(new_cb.shape)} = {top_n}×{top_k} ✓")
    except Exception:
        fail("Reorganized codebook shape", traceback.format_exc())
        return

    # No zero rows — every concatenated entry should be non-trivial
    try:
        norms = new_cb.norm(dim=-1)
        zero_rows = (norms < 1e-6).sum().item()
        assert zero_rows == 0, \
            f"{zero_rows} zero-norm rows in reorganized codebook"
        ok("No zero-norm rows in reorganized codebook")
    except Exception:
        fail("Reorganized codebook zero rows", traceback.format_exc())

    # First half and second half come from different groups
    try:
        half = h.vq_dim // 2
        # The first half should vary across rows (from group-0 embeddings)
        # The second half should also vary (from group-1 embeddings)
        first_var  = new_cb[:, :half].var(dim=0).mean().item()
        second_var = new_cb[:, half:].var(dim=0).mean().item()
        assert first_var  > 1e-8, \
            f"First half has zero variance — group-0 embeddings all identical"
        assert second_var > 1e-8, \
            f"Second half has zero variance — group-1 embeddings all identical"
        ok("Both halves of reorganized codebook have non-zero variance",
           f"first_var={first_var:.4f}, second_var={second_var:.4f}")
    except Exception:
        fail("Reorganized codebook variance", traceback.format_exc())

    # Verify top-N selection: entries with highest usage should be chosen
    try:
        # Group 0 usage = [0,1,2,...,n_codes-1]
        # topk should select the last top_n entries
        _, expected_idx0 = torch.topk(gq._get_usage(0), top_n)
        _, expected_idx1 = torch.topk(gq._get_usage(1), top_k)

        actual_first_row_first_half  = new_cb[0, :half]
        expected_first_row_first_half = (
            gq.quantizer_modules[0].embedding.weight[expected_idx0[0]]
        )
        assert torch.allclose(
            actual_first_row_first_half,
            expected_first_row_first_half, atol=1e-5
        ), "Top-N selection from group-0 is wrong"
        ok("Top-N/Top-K selection picks correct entries",
           "highest-usage entries selected ✓")
    except Exception:
        fail("Top-N/K selection correctness", traceback.format_exc())

    # SingleQuantizer can be initialized from reorganized codebook
    try:
        stage2_model = SimCodec("config_stage2.json")
        with torch.no_grad():
            stage2_model.quantizer._q.module.embedding.weight.copy_(
                new_cb.cpu()
            )
        # Verify weights were copied
        stored = stage2_model.quantizer._q.module.embedding.weight
        assert torch.allclose(stored, new_cb.cpu(), atol=1e-5), \
            "Codebook weights not copied correctly into Stage 2 model"
        ok("Reorganized codebook loads into Stage 2 SingleQuantizer ✓")
    except Exception:
        fail("Stage 2 codebook injection", traceback.format_exc())


# ===========================================================================
# TEST 6 — Diversity loss is well-formed
# ===========================================================================

def test_diversity_loss(h, device):
    section("6. Diversity loss (gram matrix cross-entropy)")

    gq = GroupQuantizer(h).to(device)

    for g, module in enumerate(gq.quantizer_modules):
        try:
            weights  = module.embedding.weight
            target   = module.target
            gram     = torch.mm(weights, weights.T) * 3
            div_loss = F.cross_entropy(gram, target)

            assert not torch.isnan(div_loss), "diversity loss is NaN"
            assert not torch.isinf(div_loss), "diversity loss is Inf"
            assert div_loss.item() > 0,       "diversity loss is zero or negative"

            # Gram matrix shape check
            n_e = h.n_codes
            assert gram.shape == (n_e, n_e), \
                f"Gram matrix shape {gram.shape} != ({n_e},{n_e})"

            # Target shape and range
            assert target.shape == (n_e,), \
                f"target shape {target.shape} != ({n_e},)"
            assert target.min() == 0 and target.max() == n_e - 1, \
                f"target range [{target.min()},{target.max()}] != [0,{n_e-1}]"

            ok(f"Diversity loss — group {g}",
               f"loss={div_loss.item():.4f}, gram={tuple(gram.shape)}, "
               f"target range=[0,{n_e-1}] ✓")
        except Exception:
            fail(f"Diversity loss — group {g}", traceback.format_exc())


# ===========================================================================
# TEST 7 — Full SimCodec model: encode produces valid token indices
# ===========================================================================

def test_simcodec_encode(h, device):
    section("7. SimCodec encode produces valid token indices")

    try:
        model = SimCodec("config_stage1.json").to(device)
        model.eval()
        ok("SimCodec(stage=1) constructed")
    except Exception:
        fail("SimCodec construction", traceback.format_exc())
        return

    # Synthetic waveform
    sr  = h.sample_rate
    wav = torch.randn(2, 1, sr, device=device)   # 1 second, batch=2

    try:
        with torch.no_grad():
            tokens = model.encode(wav)   # (B, T, n_groups) for stage 1

        assert tokens.dim() == 3, \
            f"Expected 3D tokens (B,T,n_groups), got {tokens.shape}"
        assert tokens.shape[0] == 2, \
            f"Batch dim wrong: {tokens.shape}"
        assert tokens.shape[2] == h.n_code_groups, \
            f"Last dim should be n_groups={h.n_code_groups}, got {tokens.shape}"
        assert tokens.min() >= 0, \
            f"Negative token index: {tokens.min()}"
        assert tokens.max() <  h.n_codes, \
            f"Token {tokens.max()} >= n_codes={h.n_codes}"

        ok("encode() output shape and range",
           f"{tuple(tokens.shape)}  range=[0,{h.n_codes})  ✓")
    except Exception:
        fail("encode() output", traceback.format_exc())

    # Decode from tokens
    try:
        with torch.no_grad():
            recon = model.decode(tokens)
        assert recon.shape[0] == 2,  f"Batch dim wrong: {recon.shape}"
        assert recon.shape[1] == 1,  f"Channel dim wrong: {recon.shape}"
        assert not torch.isnan(recon).any(), "NaN in decoded audio"
        assert not torch.isinf(recon).any(), "Inf in decoded audio"
        ok("decode() produces finite audio",
           f"{tuple(recon.shape)} ✓")
    except Exception:
        fail("decode() output", traceback.format_exc())


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Pre-training quantizer unit tests")
    p.add_argument("--config", type=str, default="config_stage1.json",
                   help="Path to stage 1 config JSON")
    p.add_argument("--gpu",    type=str, default="cpu",
                   help="Device to use: 'cpu' or '0' for GPU 0")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu == "cpu":
        device = torch.device("cpu")
    else:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n[Unit Test] Device : {device}")
    print(f"[Unit Test] Config : {args.config}")

    h = load_cfg(args.config)

    test_quantizer_module(h, device)
    test_group_quantizer_parallel(h, device)
    test_straight_through(h, device)
    test_usage_counters(h, device)
    test_reorganization(h, device)
    test_diversity_loss(h, device)
    test_simcodec_encode(h, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(passed) + len(failed)
    print(f"\n{'='*60}")
    print(f"  {GREEN}{len(passed)} passed{RESET}  |  "
          f"{RED}{len(failed)} failed{RESET}  |  "
          f"{total} total")
    print(f"{'='*60}")

    if failed:
        print(f"\n{RED}Failed checks — fix before training:{RESET}")
        for name, reason in failed:
            print(f"  • {name}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}All checks passed.{RESET}")
        print("Your quantizer implementation is correct. Safe to train.")
        sys.exit(0)


if __name__ == "__main__":
    main()