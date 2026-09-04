"""
sanity_check.py
===============
Runs every check needed to confirm modules.py, model.py, and train.py
will work correctly before you commit to a full training run.

Run from your project root:
    python sanity_check.py

All checks must print PASS. Any FAIL tells you exactly what is wrong
and what to fix before training.
"""

import os
import sys
import json
import random
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Colour helpers (work on any terminal) ───────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

passed = []
failed = []

def ok(name):
    print(f"  {GREEN}PASS{RESET}  {name}")
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
# CHECK 1 — Python packages
# ===========================================================================
section("1. Required packages")

for pkg in ["torch", "soundfile", "librosa", "scipy", "numpy"]:
    try:
        __import__(pkg)
        ok(f"import {pkg}")
    except ImportError as e:
        fail(f"import {pkg}", str(e))


# ===========================================================================
# CHECK 2 — Project structure
# ===========================================================================
section("2. Project structure")

required_files = [
    "config_stage1.json",
    "config_stage2.json",
    "components/__init__.py",
    "components/simcodec/__init__.py",
    "components/simcodec/model.py",
    "components/simcodec/modules.py",
    "dataset",      # the dataset folder
]

for f in required_files:
    p = ROOT / f
    if p.exists():
        ok(f"exists: {f}")
    else:
        fail(f"exists: {f}", f"Not found at {p}")


# ===========================================================================
# CHECK 3 — Config files are valid JSON with all required keys
# ===========================================================================
section("3. Config JSON keys")

REQUIRED_KEYS = [
    "stage", "channel", "en_filters", "de_filters", "vq_dim",
    "upsample_rates", "upsample_kernel_sizes",
    "resblock_kernel_sizes", "resblock_dilation_sizes",
    "n_code_groups", "n_codes",
    "codebook_loss_lambda", "commitment_loss_lambda",
    "top_n", "top_k",
    "sample_rate", "segment_len",
]

configs = {}
for cfg_name in ["config_stage1.json", "config_stage2.json"]:
    try:
        with open(ROOT / cfg_name) as f:
            cfg = json.load(f)
        configs[cfg_name] = cfg
        ok(f"{cfg_name} — valid JSON")
        for key in REQUIRED_KEYS:
            if key in cfg:
                ok(f"  {cfg_name}['{key}'] = {cfg[key]}")
            else:
                fail(f"  {cfg_name}['{key}']", "Key missing from config")
    except Exception as e:
        fail(f"{cfg_name} — parse", str(e))

# Extra: stage values must be 1 and 2
if "config_stage1.json" in configs and configs["config_stage1.json"].get("stage") != 1:
    fail("config_stage1.json stage value", "Must be 1")
if "config_stage2.json" in configs and configs["config_stage2.json"].get("stage") != 2:
    fail("config_stage2.json stage value", "Must be 2")

# Extra: vq_dim must be divisible by n_code_groups
for cfg_name, cfg in configs.items():
    if "vq_dim" in cfg and "n_code_groups" in cfg:
        if cfg["vq_dim"] % cfg["n_code_groups"] == 0:
            ok(f"{cfg_name}: vq_dim ({cfg['vq_dim']}) divisible by n_code_groups ({cfg['n_code_groups']})")
        else:
            fail(f"{cfg_name}: vq_dim % n_code_groups",
                 f"{cfg['vq_dim']} % {cfg['n_code_groups']} != 0 — GroupQuantizer will crash")

# Extra: upsample_rates and upsample_kernel_sizes must have same length
for cfg_name, cfg in configs.items():
    if "upsample_rates" in cfg and "upsample_kernel_sizes" in cfg:
        r, k = cfg["upsample_rates"], cfg["upsample_kernel_sizes"]
        if len(r) == len(k):
            ok(f"{cfg_name}: upsample_rates and upsample_kernel_sizes same length ({len(r)})")
        else:
            fail(f"{cfg_name}: upsample list lengths",
                 f"upsample_rates has {len(r)} items, upsample_kernel_sizes has {len(k)}")


# ===========================================================================
# CHECK 4 — Dataset folder and wav files
# ===========================================================================
section("4. Dataset")

import glob

dataset_dir = ROOT / "dataset"
if dataset_dir.exists():
    wav_files = sorted(set(
        glob.glob(str(dataset_dir / "*.wav"))
        + glob.glob(str(dataset_dir / "**" / "*.wav"), recursive=True)
    ))
    if len(wav_files) > 0:
        ok(f"Found {len(wav_files)} .wav files in dataset/")
    else:
        fail("dataset/ .wav files", "No .wav files found — check dataset folder")

    # Try loading the first wav file
    if wav_files:
        try:
            data, sr = sf.read(wav_files[0], always_2d=True)  # (T, C)
            wav = torch.from_numpy(data.T).float()             # (C, T)
            ok(f"soundfile.read — shape={tuple(wav.shape)}, sr={sr}")
            if sr != 16_000:
                print(f"  {YELLOW}WARN{RESET}  Sample rate is {sr}, not 16000. "
                      "Will be resampled automatically in train.py — OK but slightly slower.")
        except Exception as e:
            fail("soundfile.read first wav file", str(e))
else:
    fail("dataset/ folder exists", "Create a 'dataset' folder beside train.py with your .wav files")


# ===========================================================================
# CHECK 5 — Imports from project modules
# ===========================================================================
section("5. Module imports")

try:
    from components.simcodec.modules import (
        Encoder, Generator, Quantizer,
        GroupQuantizer, SingleQuantizer, Quantizer_module
    )
    ok("from components.simcodec.modules import all classes")
except Exception as e:
    fail("import components.simcodec.modules", traceback.format_exc())

try:
    from components.simcodec.model import SimCodec, AttrDict
    ok("from components.simcodec.model import SimCodec, AttrDict")
except Exception as e:
    fail("import components.simcodec.model", traceback.format_exc())


# ===========================================================================
# CHECK 6 — Model construction (Stage 1 and Stage 2)
# ===========================================================================
section("6. Model construction")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}")

model_s1 = model_s2 = None

try:
    model_s1 = SimCodec("config_stage1.json").to(device)
    n_params = sum(p.numel() for p in model_s1.parameters() if p.requires_grad)
    ok(f"SimCodec(stage=1) constructed — {n_params:,} trainable parameters")
except Exception as e:
    fail("SimCodec(stage=1) construction", traceback.format_exc())

try:
    model_s2 = SimCodec("config_stage2.json").to(device)
    ok("SimCodec(stage=2) constructed")
except Exception as e:
    fail("SimCodec(stage=2) construction", traceback.format_exc())

# Check quantizer types
if model_s1 is not None:
    try:
        assert isinstance(model_s1.quantizer._q, GroupQuantizer), \
            f"Expected GroupQuantizer, got {type(model_s1.quantizer._q)}"
        ok("Stage 1 quantizer is GroupQuantizer")
    except Exception as e:
        fail("Stage 1 quantizer type", str(e))

if model_s2 is not None:
    try:
        assert isinstance(model_s2.quantizer._q, SingleQuantizer), \
            f"Expected SingleQuantizer, got {type(model_s2.quantizer._q)}"
        ok("Stage 2 quantizer is SingleQuantizer")
    except Exception as e:
        fail("Stage 2 quantizer type", str(e))


# ===========================================================================
# CHECK 7 — Forward pass shapes (Stage 1)
# ===========================================================================
section("7. Forward pass shapes — Stage 1")

if model_s1 is not None:
    model_s1.eval()
    cfg1 = configs.get("config_stage1.json", {})
    sr   = cfg1.get("sample_rate", 16_000)
    # 1 second of audio
    dummy = torch.randn(2, 1, sr).to(device)  # (B=2, C=1, T)

    # 7a — encoder output shape
    try:
        with torch.no_grad():
            z = model_s1.encoder(dummy)
        expected_frames = sr // cfg1.get("upsample_rates", [8,5,4,2])[0] \
            if len(cfg1.get("upsample_rates",[])) == 1 else None
        total_stride = 1
        for r in cfg1.get("upsample_rates", []):
            total_stride *= r
        expected_T = sr // total_stride
        assert z.shape[0] == 2,                   f"Batch dim wrong: {z.shape}"
        assert z.shape[1] == cfg1.get("vq_dim"),  f"Channel dim wrong: {z.shape}, expected vq_dim={cfg1.get('vq_dim')}"
        ok(f"Encoder output shape: {tuple(z.shape)}  (expected (2, {cfg1.get('vq_dim')}, {expected_T}))")
    except Exception as e:
        fail("Encoder output shape", traceback.format_exc())

    # 7b — quantizer output shapes
    try:
        with torch.no_grad():
            quantized, vq_loss, indices = model_s1.quantizer(z)
        assert quantized.shape == z.shape, \
            f"Quantized shape {quantized.shape} != encoder output {z.shape}"
        assert isinstance(vq_loss, torch.Tensor) and vq_loss.shape == (), \
            f"vq_loss should be scalar, got shape {vq_loss.shape}"
        assert isinstance(indices, list) and len(indices) == cfg1.get("n_code_groups", 2), \
            f"Expected list of {cfg1.get('n_code_groups')} index tensors, got {type(indices)}"
        ok(f"GroupQuantizer output — quantized: {tuple(quantized.shape)}, "
           f"vq_loss: scalar, indices: list of {len(indices)} tensors each {tuple(indices[0].shape)}")
    except Exception as e:
        fail("GroupQuantizer output shapes", traceback.format_exc())

    # 7c — generator output shape
    try:
        with torch.no_grad():
            recon = model_s1.generator(quantized)
        T_in = dummy.shape[-1]
        recon = recon[..., :T_in]
        if recon.shape[-1] < T_in:
            recon = F.pad(recon, (0, T_in - recon.shape[-1]))
        assert recon.shape == dummy.shape, \
            f"Generator output cannot be aligned to input {dummy.shape}"
        ok(f"Generator output shape: aligned to {tuple(recon.shape)}")
    except Exception as e:
        fail("Generator output shape", traceback.format_exc())

    # 7d — full forward_train
    try:
        model_s1.train()
        recon, vq_loss, idx_stacked = model_s1.forward_train(dummy)
        T_in = dummy.shape[-1]
        recon = recon[..., :T_in]
        if recon.shape[-1] < T_in:
            recon = F.pad(recon, (0, T_in - recon.shape[-1]))
        assert recon.shape == dummy.shape, \
            f"forward_train recon shape {recon.shape} != input {dummy.shape}"
        assert not torch.isnan(vq_loss), "vq_loss is NaN"
        assert not torch.isinf(vq_loss), "vq_loss is Inf"
        assert idx_stacked.shape == (2, z.shape[2], cfg1.get("n_code_groups", 2)), \
            f"Stacked indices shape wrong: {idx_stacked.shape}"
        ok(f"forward_train — recon: {tuple(recon.shape)}, "
           f"vq_loss: {vq_loss.item():.4f}, "
           f"indices: {tuple(idx_stacked.shape)}")
    except Exception as e:
        fail("forward_train (Stage 1)", traceback.format_exc())

    # 7e — encode() returns correct shape
    try:
        model_s1.eval()
        with torch.no_grad():
            tokens = model_s1.encode(dummy)
        assert tokens.shape[0] == 2, f"Batch dim wrong: {tokens.shape}"
        assert tokens.shape[-1] == cfg1.get("n_code_groups", 2), \
            f"Last dim should be n_code_groups={cfg1.get('n_code_groups')}, got {tokens.shape}"
        ok(f"encode() output shape: {tuple(tokens.shape)}  "
           f"— (batch, frames, n_code_groups)")
    except Exception as e:
        fail("encode() Stage 1", traceback.format_exc())

    # 7f — decode() round-trip
    try:
        with torch.no_grad():
            decoded = model_s1.decode(tokens)
        T_in = dummy.shape[-1]
        decoded = decoded[..., :T_in]
        if decoded.shape[-1] < T_in:
            decoded = F.pad(decoded, (0, T_in - decoded.shape[-1]))
        assert decoded.shape == dummy.shape, \
            f"decode() output cannot be aligned to input {dummy.shape}"
        ok(f"decode() round-trip shape: {tuple(decoded.shape)}")
    except Exception as e:
        fail("decode() Stage 1 round-trip", traceback.format_exc())


# ===========================================================================
# CHECK 8 — Loss computation
# ===========================================================================
section("8. Loss computation")

if model_s1 is not None:
    try:
        import librosa

        # Build MelLossModule using librosa mel filterbank — no torchaudio needed
        class MelLossModule(nn.Module):
            FFT_CONFIGS = [
                ( 128,  32,  128), ( 256,  64,  256), ( 512, 128,  512),
                (1024, 256, 1024), (2048, 512, 2048),
            ]
            N_MELS = 80

            def __init__(self, sample_rate):
                super().__init__()
                # Pre-build mel filterbank matrices as buffers (one per FFT size)
                # librosa returns (n_mels, 1 + n_fft//2) float32 numpy arrays
                self.sample_rate = sample_rate
                for idx, (n_fft, hop, win) in enumerate(self.FFT_CONFIGS):
                    fb = librosa.filters.mel(
                        sr=sample_rate, n_fft=n_fft, n_mels=self.N_MELS,
                    )  # (n_mels, 1 + n_fft//2)
                    self.register_buffer(
                        f"fb_{idx}", torch.from_numpy(fb).float()
                    )
                # Store configs for STFT
                self.fft_configs = self.FFT_CONFIGS

            def _mel(self, wav, idx):
                """wav: (B, T), returns log-mel (B, n_mels, frames)"""
                n_fft, hop, win = self.fft_configs[idx]
                window = torch.hann_window(win, device=wav.device)
                stft = torch.stft(
                    wav, n_fft=n_fft, hop_length=hop, win_length=win,
                    window=window, return_complex=True,
                )  # (B, 1+n_fft//2, frames)
                magnitude = stft.abs()                          # amplitude
                fb = getattr(self, f"fb_{idx}").to(wav.device)  # (n_mels, freq)
                mel = torch.matmul(fb, magnitude)               # (B, n_mels, frames)
                return torch.log(mel.clamp(min=1e-5))

            def forward(self, real, fake):
                total = torch.tensor(0.0, device=real.device)
                r, f = real.squeeze(1), fake.squeeze(1)
                for idx in range(len(self.fft_configs)):
                    total = total + F.l1_loss(self._mel(f, idx), self._mel(r, idx))
                return total / len(self.fft_configs)

        mel_fn = MelLossModule(sample_rate=sr).to(device)
        for p in mel_fn.parameters():
            p.requires_grad_(False)

        model_s1.train()
        recon, vq_loss, _ = model_s1.forward_train(dummy)
        T = dummy.shape[-1]
        recon = recon[..., :T]
        if recon.shape[-1] < T:
            recon = F.pad(recon, (0, T - recon.shape[-1]))

        loss_mel = mel_fn(dummy, recon)
        loss     = 45.0 * loss_mel + 1.0 * vq_loss

        assert not torch.isnan(loss), "Total loss is NaN"
        assert not torch.isinf(loss), "Total loss is Inf"
        assert loss.item() > 0,       "Total loss is zero or negative — something is wrong"

        ok(f"mel_loss={loss_mel.item():.4f}  "
           f"vq_loss={vq_loss.item():.4f}  "
           f"total={loss.item():.4f}  (all finite, all > 0)")
    except Exception as e:
        fail("Loss computation", traceback.format_exc())


# ===========================================================================
# CHECK 9 — Backward pass (gradients flow through everything)
# ===========================================================================
section("9. Backward pass and gradients")

if model_s1 is not None:
    try:
        model_s1.train()
        opt = torch.optim.AdamW(model_s1.parameters(), lr=2e-4)
        opt.zero_grad(set_to_none=True)

        recon, vq_loss, _ = model_s1.forward_train(dummy)
        T = dummy.shape[-1]
        recon = recon[..., :T]
        if recon.shape[-1] < T:
            recon = F.pad(recon, (0, T - recon.shape[-1]))

        loss = 45.0 * mel_fn(dummy, recon) + 1.0 * vq_loss
        loss.backward()

        # Check that key parameters actually received gradients
        checks = {
            "encoder.conv_pre":   model_s1.encoder.conv_pre,
            "generator.conv_post": model_s1.generator.conv_post,
        }
        for name, module in checks.items():
            # weight_norm wraps weight into weight_g and weight_v
            for pname, param in module.named_parameters():
                if param.grad is not None:
                    ok(f"Gradient flows to {name}.{pname}")
                    break
            else:
                fail(f"Gradient flows to {name}", "No gradient found — dead path")

        # Check quantizer embedding gradients (straight-through means
        # embedding gets grad via codebook loss, not via the STE path)
        emb_grad = model_s1.quantizer._q.quantizer_modules[0].embedding.weight.grad
        if emb_grad is not None:
            ok("Gradient flows to quantizer codebook embeddings")
        else:
            fail("Gradient flows to quantizer codebook",
                 "Embedding has no gradient — codebook will not learn")

        opt.step()
        ok("opt.step() completed without error")
    except Exception as e:
        fail("Backward pass", traceback.format_exc())


# ===========================================================================
# CHECK 10 — Checkpoint save and load round-trip
# ===========================================================================
section("10. Checkpoint save / load")

import tempfile

if model_s1 is not None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test.pt")

            # Save via save_ckpt
            model_s1.save_ckpt(ckpt_path)
            ok("save_ckpt() wrote file")

            # Load into fresh model
            model_fresh = SimCodec("config_stage1.json").to(device)
            model_fresh.load_ckpt(ckpt_path)
            ok("load_ckpt() loaded file into fresh model")

            # Verify weights are identical
            for (n1, p1), (n2, p2) in zip(
                model_s1.named_parameters(), model_fresh.named_parameters()
            ):
                if not torch.allclose(p1, p2):
                    fail(f"Weight match after load: {n1}",
                         "Weights differ after save/load")
                    break
            else:
                ok("All weights identical after save → load round-trip")
    except Exception as e:
        fail("Checkpoint save/load", traceback.format_exc())


# ===========================================================================
# CHECK 11 — Stage 1 → Stage 2 codebook reorganization
# ===========================================================================
section("11. Stage 1 → Stage 2 codebook reorganization")

if model_s1 is not None and model_s2 is not None and configs:
    try:
        cfg1 = configs["config_stage1.json"]
        cfg2 = configs["config_stage2.json"]
        top_n, top_k = cfg2["top_n"], cfg2["top_k"]

        # Artificially fill usage counters so topk works
        for g in range(cfg1["n_code_groups"]):
            buf = model_s1.quantizer._q._get_usage(g)
            buf.copy_(torch.randint(1, 1000, buf.shape))

        new_cb = model_s1.quantizer.get_reorganized_codebook(top_n, top_k)
        expected_shape = (top_n * top_k, cfg1["vq_dim"])
        assert new_cb.shape == expected_shape, \
            f"Codebook shape {new_cb.shape} != expected {expected_shape}"
        ok(f"get_reorganized_codebook — shape: {tuple(new_cb.shape)}  "
           f"({top_n} × {top_k} = {top_n*top_k} entries)")

        # Full build_stage2_from_stage1
        model_s2_built = SimCodec.build_stage2_from_stage1(
            config_path="config_stage2.json",
            stage1_model=model_s1,
            top_n=top_n,
            top_k=top_k,
        ).to(device)
        ok("build_stage2_from_stage1() completed")

        # Verify Stage 2 encode works end-to-end
        model_s2_built.eval()
        with torch.no_grad():
            tokens_s2 = model_s2_built.encode(dummy)
        assert tokens_s2.shape[0] == 2,  f"Batch dim wrong: {tokens_s2.shape}"
        assert tokens_s2.dim()   == 2,   f"Stage 2 tokens should be 2D, got {tokens_s2.shape}"
        ok(f"Stage 2 encode() shape: {tuple(tokens_s2.shape)}  — (batch, frames)")

        # Token indices must be in valid codebook range
        assert tokens_s2.min() >= 0, "Negative token index"
        assert tokens_s2.max() < top_n * top_k, \
            f"Token index {tokens_s2.max()} >= codebook size {top_n*top_k}"
        ok(f"Token range: [{tokens_s2.min().item()}, {tokens_s2.max().item()}]  "
           f"(valid: 0 – {top_n*top_k - 1})")

    except Exception as e:
        fail("Stage 1 → Stage 2 reorganization", traceback.format_exc())


# ===========================================================================
# CHECK 12 — DataLoader produces correct shapes
# ===========================================================================
section("12. DataLoader")

if dataset_dir.exists() and wav_files:
    try:
        import glob as _glob
        from torch.utils.data import DataLoader

        # Inline minimal AudioDataset (same logic as train.py, loading via soundfile)
        class _DS(torch.utils.data.Dataset):
            def __init__(self, files, sr, seg):
                self.files, self.sr, self.seg = files, sr, seg
            def __len__(self): return len(self.files)
            def __getitem__(self, i):
                data, sr = sf.read(self.files[i], always_2d=True)  # (T, C)
                wav = torch.from_numpy(data.T).float()              # (C, T)
                if sr != self.sr:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g   = gcd(sr, self.sr)
                    arr = resample_poly(data, self.sr // g, sr // g, axis=0)
                    wav = torch.from_numpy(arr.T).float()
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                T = wav.shape[-1]
                seg = self.seg
                if T >= seg:
                    s = random.randint(0, T - seg)
                    wav = wav[:, s:s+seg]
                else:
                    wav = F.pad(wav, (0, seg - T))
                return wav

        sr_  = configs.get("config_stage1.json", {}).get("sample_rate", 16_000)
        seg_ = configs.get("config_stage1.json", {}).get("segment_len", 32_000)

        # Use up to 8 files for the check
        sample_files = wav_files[:min(8, len(wav_files))]
        ds = _DS(sample_files, sr_, seg_)
        dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(dl))

        assert batch.shape == (2, 1, seg_), \
            f"Expected (2, 1, {seg_}), got {tuple(batch.shape)}"
        assert not torch.isnan(batch).any(), "NaN values in batch"
        assert not torch.isinf(batch).any(), "Inf values in batch"
        ok(f"DataLoader batch shape: {tuple(batch.shape)}  (B, 1, segment_len) ✓")
    except Exception as e:
        fail("DataLoader batch shape", traceback.format_exc())


# ===========================================================================
# CHECK 13 — GPU availability
# ===========================================================================
section("13. GPU")

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    ok(f"{n_gpu} GPU(s) visible to PyTorch")
    for i in range(n_gpu):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        try:
            # Quick allocation test to confirm GPU is usable
            t = torch.zeros(1).to(f"cuda:{i}")
            del t
            ok(f"  GPU {i}: {props.name} — {mem_gb:.1f} GB — OK")
        except Exception as e:
            fail(f"  GPU {i}: {props.name}", f"Allocation failed: {e}  (likely the ERR! GPU)")
else:
    print(f"  {YELLOW}WARN{RESET}  No CUDA GPU found — training will run on CPU (very slow)")


# ===========================================================================
# SUMMARY
# ===========================================================================
total = len(passed) + len(failed)
print(f"\n{'='*60}")
print(f"  Results: {GREEN}{len(passed)} passed{RESET}  |  {RED}{len(failed)} failed{RESET}  |  {total} total")
print(f"{'='*60}")

if failed:
    print(f"\n{RED}Failed checks:{RESET}")
    for name, reason in failed:
        print(f"  • {name}")
        print(f"    {reason}")
    print(f"\n{RED}Fix all failed checks before running training.{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}All checks passed. You are ready to train.{RESET}")
    print("\nSanity training run (100 steps, ~2 min):")
    print("  python train.py --stage 1 --config config_stage1.json \\")
    print("    --data_dir dataset --output_dir checkpoints/sanity \\")
    print("    --gpu_ids 0 --batch_size 4 \\")
    print("    --total_steps 100 --log_interval 10 --val_interval 50")
    sys.exit(0)