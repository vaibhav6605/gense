"""
inference.py  —  GenSE: Noisy Speech → Clean Speech
=====================================================
Full pipeline following the architecture in the paper:

  Noisy WAV
      │
      ├──► Semantic Extractor (HuBERT + k-means)
      │         └── noisy_sem  (T_sem,)
      │
      ├──► SimCodec Encoder
      │         └── noisy_ac   (T_ac,)
      │
      ├──► N2S Language Model
      │         noisy_sem → clean_sem  (T_sem,)
      │
      ├──► S2S Language Model
      │         [noisy_sem | clean_sem | BOS | noisy_ac] → clean_ac  (T_ac,)
      │
      └──► SimCodec Decoder
                clean_ac → Clean WAV

Usage (single file):
    python infer.py 
        --input          noisy.wav 
        --output         enhanced.wav 
        --codec_ckpt     s2s/checkpoints/stage2/best.pt 
        --codec_config   s2s/config_stage2.json 
        --n2s_ckpt       n2s/ckpts/n2s/best.pt 
        --s2s_ckpt       s2s/checkpoints/s2s/best.pt 
        --hubert_model   facebook/hubert-base-ls960 
        --km_path        n2s/ckpts/kmeans_hubert_base_l9_c500.pt 
        --semantic_num   500 
        --acoustic_num   1024

    python infer.py 
        --input          noisy.wav 
        --output         enhanced.wav 
        --codec_ckpt     s2s2/ckpts/ckpt_120000.pt 
        --codec_config   s2s2/config.json 
        --n2s_ckpt       n2s/ckpts/n2s/best.pt 
        --s2s_ckpt       s2s2/ckpts/s2s/best.pt 
        --hubert_model   facebook/hubert-base-ls960 
        --km_path        n2s/ckpts/kmeans_hubert_base_l9_c500.pt 
        --semantic_num   500 
        --acoustic_num   1024    

Usage (folder of WAV files):
    python inference.py \
        --input_dir      noisy_wavs/ \
        --output_dir     enhanced_wavs/ \
        --codec_ckpt     checkpoints/stage2/best.pt \
        --codec_config   config_stage2.json \
        --n2s_ckpt       checkpoints/n2s/best.pt \
        --s2s_ckpt       checkpoints/s2s/best.pt \
        --hubert_model   facebook/hubert-base-ls960 \
        --km_path        kmeans_hubert_base_l9_c500.pt \
        --semantic_num   500 \
        --acoustic_num   1024

Notes:
  --semantic_num  must match what you used during N2S and S2S training
  --acoustic_num  must match your SimCodec Stage 2 codebook size (top_n × top_k)
  --km_path       path to the .pt or .pkl k-means model used during training
  --ssl_layer     HuBERT layer used for features (default 9, must match training)
  --temperature   sampling temperature for S2S (default 1.0; lower = more conservative)
"""

import os
import sys
import json
import argparse
import glob
from math import gcd
from pathlib import Path
import joblib

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, HubertModel, Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s.components.simcodec.model import SimCodec, AttrDict


# ===========================================================================
# Audio helpers  (soundfile + scipy, no torchaudio)
# ===========================================================================

def load_audio_mono(path, target_sr=16_000):
    """Load any WAV as float32 mono numpy array at target_sr."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        g    = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data   # (T,)  float32


def save_audio(path, wav_np, sr=16_000):
    """Save float32 numpy array as WAV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, wav_np, sr)


# ===========================================================================
# N2S Model  (must match exactly what you trained)
# ===========================================================================

class N2SModel(nn.Module):
    """
    GPT2 causal LM: noisy semantic tokens → clean semantic tokens.
    Vocabulary:
      0          PAD
      1          BOS
      2          EOS
      3..3+k-1   semantic tokens  (raw_id + 3)
    """
    PAD   = 0
    BOS   = 1
    EOS   = 2
    SHIFT = 3

    def __init__(self, semantic_num, hidden_size=1024,
                 num_hidden_layers=12, num_attention_heads=16,
                 n_positions=2048):
        super().__init__()
        self.semantic_num = semantic_num
        vocab = self.SHIFT + semantic_num
        cfg = GPT2Config(
            vocab_size=vocab, n_embd=hidden_size,
            n_layer=num_hidden_layers, n_head=num_attention_heads,
            activation_function="gelu_new",
            n_positions=n_positions, n_ctx=n_positions,
            bos_token_id=self.BOS, eos_token_id=self.EOS,
        )
        self.lm = GPT2LMHeadModel(cfg)

    @torch.no_grad()
    def generate(self, noisy_sem, temperature=1.0):
        """
        noisy_sem : (B, T_sem)  0-indexed noisy semantic token ids
        returns   : (B, T_sem)  0-indexed clean semantic token ids
        """
        B, T = noisy_sem.shape
        device = noisy_sem.device

        # Prompt: [noisy_sem | BOS]
        bos      = torch.full((B, 1), self.BOS, dtype=torch.long, device=device)
        token_in = torch.cat([noisy_sem + self.SHIFT, bos], dim=1)

        # Position IDs: noisy tokens at 0..T-1, BOS at T
        sem_pos  = torch.arange(T,  device=device)
        bos_pos  = torch.tensor([T], device=device)
        pos_gen  = torch.cat([sem_pos, bos_pos]).unsqueeze(0).expand(B, -1)

        generated = []
        for step in range(T):
            out    = self.lm(input_ids=token_in, position_ids=pos_gen)
            logits = out.logits[:, -1, :].clone()          # (B, vocab)

            # Force output to be a valid semantic token
            logits[:, :self.SHIFT]          = -1e9
            logits[:, self.SHIFT + self.semantic_num:] = -1e9

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)
            probs      = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, 1)        # (B, 1)
            generated.append(next_token)

            next_pos = torch.full((B, 1), step, dtype=torch.long, device=device)
            token_in = torch.cat([token_in, next_token], dim=1)
            pos_gen  = torch.cat([pos_gen,  next_pos],   dim=1)

        return torch.cat(generated, dim=1) - self.SHIFT     # (B, T) 0-indexed


# ===========================================================================
# S2S Model  (copy from s2s_lm_train.py — must match training)
# ===========================================================================

class S2SModel(nn.Module):
    """
    GPT2 causal LM: [noisy_sem | clean_sem | BOS | noisy_ac] → clean_ac
    Vocabulary:
      0                  PAD
      1                  BOS
      2                  EOS
      3..3+sem-1         semantic tokens
      3+sem..vocab-1     acoustic tokens
    """
    PAD       = 0
    BOS       = 1
    EOS       = 2
    SHIFT_SEM = 3

    def __init__(self, semantic_num, acoustic_num,
                 hidden_size=1024, num_hidden_layers=12,
                 num_attention_heads=16, n_positions=4096):
        super().__init__()
        self.semantic_num = semantic_num
        self.acoustic_num = acoustic_num
        self.shift_num    = self.SHIFT_SEM + semantic_num   # acoustic offset

        vocab = self.SHIFT_SEM + semantic_num + acoustic_num
        cfg = GPT2Config(
            vocab_size=vocab, n_embd=hidden_size,
            n_layer=num_hidden_layers, n_head=num_attention_heads,
            activation_function="gelu_new",
            n_positions=n_positions, n_ctx=n_positions,
            bos_token_id=self.BOS, eos_token_id=self.EOS,
        )
        self.lm = GPT2LMHeadModel(cfg)

    def _build_prompt(self, noisy_sem, clean_sem, noisy_ac):
        B     = noisy_sem.shape[0]
        T_sem = noisy_sem.shape[1]
        T_ac  = noisy_ac.shape[1]
        dev   = noisy_sem.device

        ns  = noisy_sem + self.SHIFT_SEM
        cs  = clean_sem + self.SHIFT_SEM
        na  = noisy_ac  + self.shift_num
        bos = torch.full((B, 1), self.BOS, dtype=torch.long, device=dev)

        tokens = torch.cat([ns, cs, bos, na], dim=1)

        sem_pos = torch.arange(T_sem, device=dev)
        bos_pos = torch.tensor([T_sem], device=dev)
        ac_pos  = torch.arange(T_ac,  device=dev)
        pos     = torch.cat([sem_pos, sem_pos, bos_pos, ac_pos])
        pos     = pos.unsqueeze(0).expand(B, -1)

        return tokens, pos

    @torch.no_grad()
    def inference(self, noisy_sem, clean_sem, noisy_ac, temperature=1.0):
        """
        noisy_sem : (B, T_sem)  0-indexed
        clean_sem : (B, T_sem)  0-indexed  (from N2S)
        noisy_ac  : (B, T_ac)   0-indexed  (from SimCodec)
        returns   : (B, T_ac)   0-indexed clean acoustic tokens
        """
        B, T_ac = noisy_ac.shape
        device  = noisy_ac.device

        token_gen, pos_gen = self._build_prompt(noisy_sem, clean_sem, noisy_ac)

        generated = []
        for step in tqdm(range(T_ac), desc="S2S generating", leave=False):
            out         = self.lm(input_ids=token_gen, position_ids=pos_gen)
            last_logits = out.logits[:, -1, :].clone()     # (B, vocab)

            # Only allow acoustic token predictions
            last_logits[:, :self.shift_num] = -1e9

            if temperature != 1.0:
                last_logits = last_logits / max(temperature, 1e-5)
            probs      = last_logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, 1)        # (B, 1)
            generated.append(next_token)

            # Acoustic positions restart at 0 — position for step j is j
            next_pos   = torch.full((B, 1), step, dtype=torch.long, device=device)
            token_gen  = torch.cat([token_gen, next_token], dim=1)
            pos_gen    = torch.cat([pos_gen,   next_pos],   dim=1)

        clean_ac_shifted = torch.cat(generated, dim=1)     # (B, T_ac)
        return clean_ac_shifted - self.shift_num            # back to 0-indexed


# ===========================================================================
# Semantic extractor  (HuBERT + k-means)
# ===========================================================================

class SemanticExtractor:
    """
    Extracts discrete semantic tokens from a waveform using HuBERT + k-means.
    Matches the preprocessing used during training.
    """

    def __init__(self, hubert_model_name, km_path, ssl_layer=9, device="cpu"):
        self.device     = torch.device(device)
        self.ssl_layer  = ssl_layer

        print(f"[SemanticExtractor] Loading HuBERT from {hubert_model_name} …")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        self.hubert    = HubertModel.from_pretrained(
            hubert_model_name, output_hidden_states=True
        ).to(self.device).eval()

        print(f"[SemanticExtractor] Loading k-means from {km_path} …")
        km_data = joblib.load(km_path)

        # Support both sklearn joblib dumps (.pkl) and raw tensor saves (.pt)
        if isinstance(km_data, dict) and "cluster_centers" in km_data:
            # Custom saved format: dict with cluster_centers tensor
            centers = km_data["cluster_centers"].float()   # (K, D)
            self.centers = centers.to(self.device)
            self._use_tensor_km = True
        elif hasattr(km_data, "cluster_centers_"):
            # sklearn KMeans object saved with torch.save
            centers = torch.from_numpy(km_data.cluster_centers_).float()
            self.centers = centers.to(self.device)
            self._use_tensor_km = True
        else:
            raise ValueError(
                f"Unrecognised k-means format in {km_path}. "
                "Expected dict with 'cluster_centers' or sklearn KMeans object."
            )
        print(f"[SemanticExtractor] K-means: {self.centers.shape[0]} clusters, "
              f"dim {self.centers.shape[1]}")

    @torch.no_grad()
    def extract(self, wav_np):
        """
        wav_np : float32 numpy array (T,) at 16 kHz
        returns: int64 numpy array (T_sem,) of 0-indexed cluster ids
        """
        inputs = self.processor(
            wav_np, sampling_rate=16_000,
            return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.hubert(input_values, output_hidden_states=True)
        # hidden_states: tuple of (B, T, D) for each layer
        # Layer indexing: hidden_states[0] = embedding layer,
        #                 hidden_states[1] = transformer layer 1, etc.
        feats = outputs.hidden_states[self.ssl_layer + 1]  # (1, T_sem, D)
        feats = feats.squeeze(0)                           # (T_sem, D)

        # Nearest centroid assignment using L2 distance
        # feats: (T, D), centers: (K, D)
        dists   = torch.cdist(feats.unsqueeze(0),
                              self.centers.unsqueeze(0)).squeeze(0)  # (T, K)
        indices = dists.argmin(dim=-1)                     # (T_sem,)
        return indices.cpu().numpy().astype(np.int64)


# ===========================================================================
# Model loading helpers
# ===========================================================================

def load_simcodec(config_path, ckpt_path, device):
    print(f"[Load] SimCodec  ← {ckpt_path}")
    codec = SimCodec(config_path).to(device)
    codec.load_ckpt(ckpt_path)
    codec.eval()
    return codec


def load_n2s(ckpt_path, semantic_num, hidden_size, num_layers,
             num_heads, n_positions, device):
    print(f"[Load] N2S LM    ← {ckpt_path}")
    model = N2SModel(
        semantic_num=semantic_num,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        n_positions=n_positions,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # Accept both {"model": state_dict} and a bare state_dict
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def load_s2s(ckpt_path, semantic_num, acoustic_num, hidden_size,
             num_layers, num_heads, n_positions, device):
    print(f"[Load] S2S LM    ← {ckpt_path}")
    model = S2SModel(
        semantic_num=semantic_num,
        acoustic_num=acoustic_num,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        n_positions=n_positions,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ===========================================================================
# Single-file inference
# ===========================================================================

def enhance_file(wav_np, sem_extractor, codec, n2s_model,
                 s2s_model, device, temperature=1.0):
    """
    wav_np      : float32 numpy (T,) at 16 kHz — the noisy input waveform
    Returns     : float32 numpy (T',) at 16 kHz — the enhanced waveform
    """

    # ── Step 1: Extract noisy semantic tokens via HuBERT + k-means ───────────
    noisy_sem_np = sem_extractor.extract(wav_np)           # (T_sem,)  int64
    noisy_sem    = torch.from_numpy(noisy_sem_np).long().unsqueeze(0).to(device)  # (1, T_sem)

    # ── Step 2: Extract noisy acoustic tokens via SimCodec encoder ───────────
    wav_t    = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T)
    noisy_ac_raw = codec.encode(wav_t)

    decode_shape = None
    if noisy_ac_raw.dim() == 3:
        decode_shape = noisy_ac_raw.shape  # (B, T, n_groups)
        noisy_ac = noisy_ac_raw.reshape(noisy_ac_raw.shape[0], -1)
    elif noisy_ac_raw.dim() == 2:
        noisy_ac = noisy_ac_raw
    else:
        raise ValueError(f"Unexpected codec token shape: {tuple(noisy_ac_raw.shape)}")


    # ── Step 3: N2S — denoise semantic tokens ────────────────────────────────
    print(f"  N2S: {noisy_sem.shape[1]} semantic tokens → denoising …")
    clean_sem = n2s_model.generate(noisy_sem, temperature=temperature)  # (1, T_sem)

    # ── Step 4: S2S — generate clean acoustic tokens ─────────────────────────
    print(f"  S2S: {noisy_ac.shape[1]} acoustic tokens → generating clean tokens …")
    clean_ac = s2s_model.inference(
        noisy_sem, clean_sem, noisy_ac, temperature=temperature
    )  # (1, T_ac)

    # ── Step 5: SimCodec decoder — reconstruct waveform ──────────────────────
    print("  Decoding clean acoustic tokens → waveform …")
    if decode_shape is not None:
        B, T, G = decode_shape
        expected = T * G
        if clean_ac.shape[1] != expected:
            raise ValueError(
                f"Cannot reshape clean_ac of length {clean_ac.shape[1]} "
                f"back to codec shape {(B, T, G)}; expected {expected} tokens."
            )
        clean_ac_for_decode = clean_ac.reshape(B, T, G)
    else:
        clean_ac_for_decode = clean_ac

    with torch.no_grad():
        enhanced_wav = codec.decode(clean_ac_for_decode)

    enhanced_wav = enhanced_wav.squeeze().cpu().numpy().astype(np.float32)

    return enhanced_wav


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GenSE inference: noisy WAV → clean WAV")

    # Input / output
    p.add_argument("--input",      type=str, default=None,
                   help="Path to a single noisy WAV file")
    p.add_argument("--output",     type=str, default=None,
                   help="Path to write the enhanced WAV file")
    p.add_argument("--input_dir",  type=str, default=None,
                   help="Directory of noisy WAV files (batch mode)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory to write enhanced WAV files (batch mode)")

    # Checkpoints
    p.add_argument("--codec_ckpt",   type=str, required=True,
                   help="SimCodec Stage 2 checkpoint  (best.pt)")
    p.add_argument("--codec_config", type=str, required=True,
                   help="config_stage2.json")
    p.add_argument("--n2s_ckpt",     type=str, required=True,
                   help="N2S language model checkpoint  (best.pt)")
    p.add_argument("--s2s_ckpt",     type=str, required=True,
                   help="S2S language model checkpoint  (best.pt)")

    # Semantic extractor
    p.add_argument("--hubert_model", type=str,
                   default="facebook/hubert-base-ls960",
                   help="HuBERT model name or local path")
    p.add_argument("--km_path",      type=str, required=True,
                   help="Path to k-means model (.pt or .pkl)")
    p.add_argument("--ssl_layer",    type=int, default=9,
                   help="HuBERT layer for feature extraction (must match training)")

    # Vocabulary — must match what you trained with
    p.add_argument("--semantic_num", type=int, default=500,
                   help="Number of semantic clusters (k in k-means)")
    p.add_argument("--acoustic_num", type=int, default=1024,
                   help="SimCodec Stage 2 codebook size (top_n × top_k)")


    # Inference settings
    p.add_argument("--temperature",  type=float, default=1.0,
                   help="Sampling temperature for both N2S and S2S (lower = more conservative)")
    p.add_argument("--gpu",          type=str,   default="0",
                   help="GPU id to use (default 0). Use 'cpu' for CPU.")
    p.add_argument("--sample_rate",  type=int,   default=16_000)

    # Model architecture — must match training config
    p.add_argument("--hidden_size",  type=int, default=None,
                help="Deprecated shared hidden size. Use --n2s_hidden_size and --s2s_hidden_size.")
    p.add_argument("--num_layers",   type=int, default=12,
                help="Shared layer count unless overridden.")
    p.add_argument("--num_heads",    type=int, default=None,
                help="Deprecated shared attention heads. Use --n2s_num_heads and --s2s_num_heads.")

    p.add_argument("--n2s_hidden_size", type=int, default=None)
    p.add_argument("--n2s_num_layers",  type=int, default=None)
    p.add_argument("--n2s_num_heads",   type=int, default=None)

    p.add_argument("--s2s_hidden_size", type=int, default=None)
    p.add_argument("--s2s_num_layers",  type=int, default=None)
    p.add_argument("--s2s_num_heads",   type=int, default=None)

    p.add_argument("--n2s_positions", type=int, default=2048,
                help="n_positions used when training N2S")
    p.add_argument("--s2s_positions", type=int, default=4096,
                help="n_positions used when training S2S")


    return p.parse_args()


def main():
    args = parse_args()

    # Validate input args
    if args.input is None and args.input_dir is None:
        raise ValueError("Provide --input (single file) or --input_dir (batch)")
    if args.input is not None and args.output is None:
        raise ValueError("--input requires --output")
    if args.input_dir is not None and args.output_dir is None:
        raise ValueError("--input_dir requires --output_dir")

    # Device
    if args.gpu.lower() == "cpu":
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    # ── Load all models once ─────────────────────────────────────────────────
    sem_extractor = SemanticExtractor(
        hubert_model_name = args.hubert_model,
        km_path           = args.km_path,
        ssl_layer         = args.ssl_layer,
        device            = str(device),
    )

    codec = load_simcodec(args.codec_config, args.codec_ckpt, device)

    n2s_model = load_n2s(
        ckpt_path    = args.n2s_ckpt,
        semantic_num = args.semantic_num,
        hidden_size  = args.n2s_hidden_size,
        num_layers   = args.n2s_num_layers,
        num_heads    = args.n2s_num_heads,
        n_positions  = args.n2s_positions,
        device       = device,
    )


    s2s_model = load_s2s(
        ckpt_path    = args.s2s_ckpt,
        semantic_num = args.semantic_num,
        acoustic_num = args.acoustic_num,
        hidden_size  = args.s2s_hidden_size,
        num_layers   = args.s2s_num_layers,
        num_heads    = args.s2s_num_heads,
        n_positions  = args.s2s_positions,
        device       = device,
    )


    print("[Inference] All models loaded. Starting enhancement.\n")

    # ── Build file list ───────────────────────────────────────────────────────
    if args.input is not None:
        file_pairs = [(args.input, args.output)]
    else:
        wav_files  = sorted(
            glob.glob(os.path.join(args.input_dir, "*.wav")) +
            glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True)
        )
        if not wav_files:
            raise RuntimeError(f"No .wav files found in {args.input_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        file_pairs = [
            (f, os.path.join(args.output_dir, os.path.basename(f)))
            for f in wav_files
        ]
        print(f"[Inference] Found {len(file_pairs)} files to process.\n")

    # ── Process each file ─────────────────────────────────────────────────────
    for i, (in_path, out_path) in enumerate(file_pairs):
        print(f"[{i+1}/{len(file_pairs)}] {os.path.basename(in_path)}")

        try:
            wav_np = load_audio_mono(in_path, target_sr=args.sample_rate)
            print(f"  Input: {len(wav_np)/args.sample_rate:.2f}s  "
                  f"({len(wav_np)} samples)")

            enhanced = enhance_file(
                wav_np        = wav_np,
                sem_extractor = sem_extractor,
                codec         = codec,
                n2s_model     = n2s_model,
                s2s_model     = s2s_model,
                device        = device,
                temperature   = args.temperature,
            )

            save_audio(out_path, enhanced, sr=args.sample_rate)
            print(f"  Saved → {out_path}\n")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            print()

    print("[Inference] Done.")


if __name__ == "__main__":
    main()