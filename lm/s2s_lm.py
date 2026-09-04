"""
s2s_lm_train.py  —  S2S Language Model Training
=================================================
Trains a GPT2-based causal LM that maps:

    [noisy_sem | clean_sem | BOS | noisy_acoustic]  →  clean acoustic tokens

Following equation (2) from the paper (token chain prompting).

Token vocabulary layout
────────────────────────
  0                       PAD
  1                       BOS
  2                       EOS
  3  ..  3+sem_num-1      semantic tokens   (raw id + 3)
  3+sem_num  ..  vocab-1  acoustic tokens   (raw id + 3 + sem_num)

  Total vocab = 3 + sem_num + acoustic_num
              = 3 + 500    + 8192           = 8695   (your setup)

Input chain (length = 2*T_sem + 1 + 2*T_ac):
  [noisy_sem | clean_sem | BOS | noisy_ac | clean_ac]
   ← prompt (no loss) →               ← loss target →

Position IDs:
  noisy_sem : 0 .. T_sem-1
  clean_sem : 0 .. T_sem-1   (shared semantic positions)
  BOS       : T_sem
  noisy_ac  : 0 .. T_ac-1    (acoustic positions restart at 0)
  clean_ac  : 0 .. T_ac-1    (same restart — target)

Usage
─────
  # Step 1 — extract token cache (run once):
  python3 s2s_lm.py --preprocess 
      --data_dir      ../n2s/data 
      --codec_ckpt    ../s2s/checkpoints/stage2/best.pt 
      --codec_config  ../s2s/config_stage2.json 
      --n2s_ckpt      ../n2s/ckpts/n2ss/ckpt_00092000.pt 
      --n2s_config    ../n2s/config.json 
      --ssl_ckpt      facebook/hubert-base-ls960 
      --ssl_km        ../n2s/ckpts/kmeans_hubert_base_l9_c500.pt 
      --ssl_layer     9 
      --token_dir     tokens/s2s

  # Step 2 — train:
  python3 s2s_lm.py 
      --token_dir   tokens/s2s 
      --output_dir  ../s2s/checkpoints/s2s 
      --semantic_num  500 
      --acoustic_num  1024 

  # Multi-GPU:
  torchrun --nproc_per_node=2 s2s_lm_train.py \
      --token_dir   tokens/s2s \
      --output_dir  ckpts/s2s \
      --semantic_num  500 \
      --acoustic_num  8192 \
      --gpu_ids 0,1
"""

import os
import sys
import glob
import json
import argparse
import random
import time
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

# This gets the 'lm' folder
ROOT = Path(__file__).resolve().parent 
# This adds the folder ABOVE 'lm' (the Project Root) to sys.path
# This allows Python to see the 'n2s' folder as a module
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

ROOT = Path(__file__).resolve().parent

# Add 's2s2' to the path so it finds 'components'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ===========================================================================
# Distributed helpers
# ===========================================================================

def is_dist():        return dist.is_available() and dist.is_initialized()
def is_main():        return (not is_dist()) or dist.get_rank() == 0
def get_rank():       return dist.get_rank()       if is_dist() else 0
def get_world_size(): return dist.get_world_size() if is_dist() else 1

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()

def unwrap(model):
    return model.module if isinstance(model, DDP) else model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Audio helper
# Replaced torchaudio.load + torchaudio.functional.resample with
# soundfile.read + scipy.signal.resample_poly.
# Behaviour is identical: returns float32 mono numpy array at target_sr.
# ===========================================================================

def load_audio_mono(path, target_sr=16_000):
    """
    Load any audio file as float32 mono numpy array resampled to target_sr.
    Uses soundfile for decoding and scipy resample_poly for resampling —
    no torchaudio dependency.
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)

    # Convert to mono: soundfile returns (T,) for mono, (T, C) for multi-channel
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)

    # Resample if needed using scipy — matches torchaudio quality
    if sr != target_sr:
        # resample_poly requires integer ratio; find GCD to reduce fraction
        from math import gcd
        g = gcd(target_sr, sr)
        wav = resample_poly(wav, target_sr // g, sr // g)

    # Ensure float32 after resample (resample_poly returns float64)
    return wav.astype(np.float32)


# ===========================================================================
# S2S Model
# ===========================================================================

class S2SModel(nn.Module):
    """
    GPT2-based causal LM for S2S token generation.

    Vocabulary offsets (applied internally — callers pass 0-indexed ids):
      PAD = 0,  BOS = 1,  EOS = 2
      semantic : raw_id + SHIFT_SEM   where SHIFT_SEM = 3
      acoustic : raw_id + SHIFT_AC    where SHIFT_AC  = 3 + sem_num

    Token chain:
      [noisy_sem | clean_sem | BOS | noisy_ac | clean_ac]
       <------------ no loss ----------->  <--- loss --->

    Position IDs (custom, not sequential):
      noisy_sem[i] -> i          (i in 0..T_sem-1)
      clean_sem[i] -> i          (same — shared semantic space)
      BOS          -> T_sem
      noisy_ac[j]  -> j          (j in 0..T_ac-1, restart at 0)
      clean_ac[j]  -> j          (same restart)
    """

    PAD       = 0
    BOS       = 1
    EOS       = 2
    SHIFT_SEM = 3

    def __init__(
        self,
        semantic_num,
        acoustic_num,
        hidden_size          = 1024,
        num_hidden_layers    = 12,
        num_attention_heads  = 16,
        n_positions          = 4096,
        resid_pdrop          = 0.1,
        embd_pdrop           = 0.1,
        attn_pdrop           = 0.1,
    ):
        super().__init__()
        self.semantic_num = semantic_num
        self.acoustic_num = acoustic_num
        self.SHIFT_AC     = self.SHIFT_SEM + semantic_num

        vocab_size = self.SHIFT_SEM + semantic_num + acoustic_num

        cfg = GPT2Config(
            vocab_size           = vocab_size,
            n_embd               = hidden_size,
            n_layer              = num_hidden_layers,
            n_head               = num_attention_heads,
            activation_function  = "gelu_new",
            n_positions          = n_positions,
            n_ctx                = n_positions,
            resid_pdrop          = resid_pdrop,
            embd_pdrop           = embd_pdrop,
            attn_pdrop           = attn_pdrop,
            layer_norm_epsilon   = 1e-5,
            initializer_range    = 0.02,
            bos_token_id         = self.BOS,
            eos_token_id         = self.EOS,
        )
        self.lm = GPT2LMHeadModel(cfg)

    # -------------------------------------------------------------------------
    def _build_chain(self, noisy_sem, clean_sem, noisy_ac, clean_ac=None):
        """
        Build token ids and position ids.
        All inputs are raw 0-indexed ids; shifts are applied here.

        Returns:
            tokens      : (B, L)
            position_ids: (B, L)
        """
        B     = noisy_sem.shape[0]
        T_sem = noisy_sem.shape[1]
        T_ac  = noisy_ac.shape[1]
        dev   = noisy_sem.device

        ns  = noisy_sem + self.SHIFT_SEM    # (B, T_sem)
        cs  = clean_sem + self.SHIFT_SEM    # (B, T_sem)
        na  = noisy_ac  + self.SHIFT_AC     # (B, T_ac)
        bos = torch.full((B, 1), self.BOS, dtype=torch.long, device=dev)

        sem_pos = torch.arange(T_sem, device=dev)       # 0..T_sem-1
        bos_pos = torch.tensor([T_sem], device=dev)     # T_sem
        ac_pos  = torch.arange(T_ac,  device=dev)       # 0..T_ac-1

        if clean_ac is not None:
            ca     = clean_ac + self.SHIFT_AC
            tokens = torch.cat([ns, cs, bos, na, ca], dim=1)
            pos    = torch.cat([sem_pos, sem_pos, bos_pos, ac_pos, ac_pos])
        else:
            # Inference prompt — no clean_ac yet
            tokens = torch.cat([ns, cs, bos, na], dim=1)
            pos    = torch.cat([sem_pos, sem_pos, bos_pos, ac_pos])

        position_ids = pos.unsqueeze(0).expand(B, -1)
        return tokens, position_ids

    # -------------------------------------------------------------------------
    def forward(self, noisy_sem, clean_sem, noisy_ac, clean_ac):
        """
        Teacher-forced training forward.
        All inputs: (B, T) raw 0-indexed token ids.
        Returns scalar cross-entropy loss over clean_ac positions only.
        """
        tokens, pos = self._build_chain(noisy_sem, clean_sem, noisy_ac, clean_ac)

        outputs = self.lm(input_ids=tokens, position_ids=pos)
        logits  = outputs.logits    # (B, L, vocab)

        T_sem = noisy_sem.shape[1]
        T_ac  = clean_ac.shape[1]

        # Standard LM: logit[i] predicts token[i+1]
        # clean_ac starts at chain position 2*T_sem + T_ac + 1
        # so we use logits at 2*T_sem + T_ac .. 2*T_sem + 2*T_ac
        start       = 2 * T_sem + T_ac
        pred_logits = logits[:, start: start + T_ac, :]   # (B, T_ac, vocab)
        targets     = clean_ac + self.SHIFT_AC             # (B, T_ac)

        # Mask PAD/BOS/EOS/semantic ids — force model to predict acoustic only
        pred_logits[:, :, :self.SHIFT_AC] = -1e9

        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.PAD,
        )
        return loss

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, noisy_sem, clean_sem, noisy_ac, temperature=1.0):
        """
        Autoregressive generation of clean acoustic tokens.

        noisy_sem : (B, T_sem)  0-indexed noisy semantic tokens
        clean_sem : (B, T_sem)  0-indexed clean semantic tokens (from N2S)
        noisy_ac  : (B, T_ac)   0-indexed noisy acoustic tokens

        Returns clean_ac : (B, T_ac)  0-indexed clean acoustic token ids
        """
        B, T_ac = noisy_ac.shape
        dev     = noisy_sem.device

        tokens, pos = self._build_chain(noisy_sem, clean_sem, noisy_ac, clean_ac=None)

        generated = []

        for step in tqdm(range(T_ac), desc="Generating acoustic tokens"):
            outputs    = self.lm(input_ids=tokens, position_ids=pos)
            last_logit = outputs.logits[:, -1, :].clone()   # (B, vocab)

            # Mask everything except acoustic token ids
            mask = torch.ones_like(last_logit, dtype=torch.bool)
            mask[:, self.SHIFT_AC: self.SHIFT_AC + self.acoustic_num] = False
            last_logit[mask] = -1e9

            probs      = (last_logit / max(temperature, 1e-5)).softmax(dim=-1)
            next_token = torch.multinomial(probs, 1)    # (B, 1)
            generated.append(next_token)

            next_pos = torch.full((B, 1), step, dtype=torch.long, device=dev)
            tokens   = torch.cat([tokens, next_token], dim=1)
            pos      = torch.cat([pos,    next_pos],   dim=1)

        clean_ac_shifted = torch.cat(generated, dim=1)   # (B, T_ac)
        return clean_ac_shifted - self.SHIFT_AC           # back to 0-indexed


# ===========================================================================
# Dataset
# ===========================================================================

class S2STokenDataset(Dataset):
    """
    Loads .pkl files. Each contains:
        noisy_sem : (T_sem,)
        clean_sem : (T_sem,)
        noisy_ac  : (T_ac,)
        clean_ac  : (T_ac,)
    All 0-indexed.
    """

    def __init__(self, token_dir, max_ac_len=500, split="train",
                val_fraction=0.02, seed=42, max_files=0):
        self.max_ac_len = max_ac_len
        all_files = sorted(glob.glob(os.path.join(token_dir, "*.pkl")))
        if not all_files:
            raise RuntimeError(
                f"No .pkl files in {token_dir}. Run --preprocess first.")

        rng = random.Random(seed)
        rng.shuffle(all_files)

        if max_files and max_files > 0:
            all_files = all_files[:max_files]

        n_val = max(1, int(len(all_files) * val_fraction))
        self.files = all_files[:n_val] if split == "val" else all_files[n_val:]

        if is_main():
            print(f"[S2SDataset] {split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            s = pickle.load(f)

        noisy_sem = s["noisy_sem"].long()
        clean_sem = s["clean_sem"].long()
        noisy_ac  = s["noisy_ac"].long()
        clean_ac  = s["clean_ac"].long()
        # Stage-1 SimCodec stores grouped acoustic tokens as (T, G).
        # Flatten them into one causal token stream so the LM sees (T*G,).
        if noisy_ac.dim() == 2:
            noisy_ac = noisy_ac.reshape(-1)

        if clean_ac.dim() == 2:
            clean_ac = clean_ac.reshape(-1)


        T, L = noisy_ac.shape[0], self.max_ac_len

        # Crop or pad acoustic tokens
        if T >= L:
            start    = random.randint(0, T - L)
            noisy_ac = noisy_ac[start: start + L]
            clean_ac = clean_ac[start: start + L]
        else:
            noisy_ac = F.pad(noisy_ac, (0, L - T), value=0)
            clean_ac = F.pad(clean_ac, (0, L - T), value=0)

        # Align semantic tokens to acoustic length via nearest-neighbour resample
        T_sem = noisy_sem.shape[0]
        if T_sem != L:
            idx_map   = (torch.arange(L).float() * T_sem / L).long().clamp(0, T_sem - 1)
            noisy_sem = noisy_sem[idx_map]
            clean_sem = clean_sem[idx_map]

        return {
            "noisy_sem": noisy_sem,
            "clean_sem": clean_sem,
            "noisy_ac":  noisy_ac,
            "clean_ac":  clean_ac,
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_dataloaders(args, distributed):
    kw = dict(
        token_dir=args.token_dir,
        max_ac_len=args.max_ac_len,
        seed=args.seed,
        max_files=args.max_train_files,
    )
    train_ds = S2STokenDataset(split="train", **kw)
    val_ds   = S2STokenDataset(split="val",   **kw)

    tr_sam  = DistributedSampler(train_ds, shuffle=True)  if distributed else None
    val_sam = DistributedSampler(val_ds,   shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(tr_sam is None), sampler=tr_sam,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, sampler=val_sam,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader, tr_sam


# ===========================================================================
# Preprocessing — extract all token quadruples and cache as .pkl
# ===========================================================================

def preprocess(args):
    """
    For each (noisy.wav, clean.wav) pair:
      1. noisy acoustic tokens  via SimCodec.encode
      2. clean acoustic tokens  via SimCodec.encode
      3. noisy semantic tokens  via HuBERT + k-means
      4. clean semantic tokens  via N2S model inference on noisy wav
         (or ground-truth clean semantic if N2S not yet available)
    """
    from n2s.components.semantic_extractor import (
        get_ssl_model, extract_semantic_tokens
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── SimCodec ──────────────────────────────────────────────────────────────
    print("[Preprocess] Loading SimCodec …")
    try:
        from s2s.components.simcodec.model import SimCodec
        codec = SimCodec(args.codec_config).to(device)
        codec.load_ckpt(args.codec_ckpt)
        codec.eval()
    except Exception as e:
        raise RuntimeError(f"Could not load SimCodec: {e}")

    # ── SSL semantic extractor ────────────────────────────────────────────────
    print("[Preprocess] Loading HuBERT + k-means …")
    ssl_model, km_model = get_ssl_model(
        ckpt_path = args.ssl_ckpt,
        km_path   = args.ssl_km,
        device    = str(device),
        type      = "hubert_fairseq",
        layer     = args.ssl_layer,
    )

    # ── N2S model for clean semantic tokens ───────────────────────────────────
    n2s_model = None
    if args.n2s_ckpt and os.path.isfile(args.n2s_ckpt):
        print("[Preprocess] Loading N2S model …")
        try:
            from n2s_lm import N2SModel
            with open(args.n2s_config) as f:
                cfg = json.load(f)
            n2s_cfg = cfg["n2s_model"]
            n2s_model = N2SModel(
                semantic_num        = n2s_cfg["semantic_num"],
                hidden_size         = n2s_cfg["hidden_size"],
                num_hidden_layers   = n2s_cfg["num_hidden_layers"],
                num_attention_heads = n2s_cfg["num_attention_heads"],
            ).to(device)
            ckpt = torch.load(args.n2s_ckpt, map_location="cpu")
            n2s_model.load_state_dict(ckpt["model"])
            n2s_model.eval()
            print("[Preprocess] N2S model loaded.")
        except Exception as e:
            print(f"[Preprocess] WARNING: Could not load N2S: {e}")
            print("  -> clean_sem will fall back to ground-truth clean audio tokens.")
            n2s_model = None
    else:
        print("[Preprocess] No N2S ckpt provided."
              " Using ground-truth clean audio tokens as clean_sem.")

    # ── File pairs ────────────────────────────────────────────────────────────
    noisy_dir = os.path.join(args.data_dir, "noisy")
    clean_dir = os.path.join(args.data_dir, "clean")

    all_noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    if not all_noisy_files:
        raise RuntimeError(f"No .wav files in {noisy_dir}")

    # Keep only valid noisy/clean pairs first, then sample randomly.
    paired_files = []
    for noisy_path in all_noisy_files:
        fname = os.path.basename(noisy_path)
        clean_path = os.path.join(clean_dir, fname)
        if os.path.isfile(clean_path):
            paired_files.append(noisy_path)

    if not paired_files:
        raise RuntimeError(f"No matching noisy/clean .wav pairs found in {args.data_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(paired_files)

    if args.max_preprocess_pairs and args.max_preprocess_pairs > 0:
        noisy_files = paired_files[:args.max_preprocess_pairs]
    else:
        noisy_files = paired_files

    os.makedirs(args.token_dir, exist_ok=True)
    skipped = 0

    print(
        f"[Preprocess] Found {len(paired_files)} valid pairs. "
        f"Processing {len(noisy_files)} random pairs.",
        flush=True,
    )

    for i, noisy_path in enumerate(noisy_files):
        fname   = os.path.basename(noisy_path)
        out_pkl = os.path.join(args.token_dir, fname.replace(".wav", ".pkl"))
        if os.path.isfile(out_pkl):
            continue

        clean_path = os.path.join(clean_dir, fname)


        try:
            # load_audio_mono now uses soundfile + scipy instead of torchaudio
            noisy_np = load_audio_mono(noisy_path)
            clean_np = load_audio_mono(clean_path)

            noisy_wav = torch.from_numpy(noisy_np).to(device)
            clean_wav = torch.from_numpy(clean_np).to(device)

            # ── Acoustic tokens (SimCodec expects (B, 1, T)) ──────────────────
            with torch.no_grad():
                noisy_ac = codec.encode(
                    noisy_wav.unsqueeze(0).unsqueeze(0)
                ).cpu().squeeze(0)   # (T_ac,)
                clean_ac = codec.encode(
                    clean_wav.unsqueeze(0).unsqueeze(0)
                ).cpu().squeeze(0)   # (T_ac,)

            # ── Noisy semantic tokens (HuBERT + k-means) ──────────────────────
            noisy_sem = extract_semantic_tokens(
                ssl_model, km_model,
                noisy_wav.unsqueeze(0)
            ).cpu().squeeze(0)   # (T_sem,)

            # ── Clean semantic tokens ─────────────────────────────────────────
            if n2s_model is not None:
                with torch.no_grad():
                    clean_sem = n2s_model.generate(
                        noisy_sem.unsqueeze(0).to(device)
                    ).cpu().squeeze(0)
            else:
                clean_sem = extract_semantic_tokens(
                    ssl_model, km_model,
                    clean_wav.unsqueeze(0)
                ).cpu().squeeze(0)

            sample = {
                "noisy_sem": noisy_sem.long(),
                "clean_sem": clean_sem.long(),
                "noisy_ac":  noisy_ac.long(),
                "clean_ac":  clean_ac.long(),
            }
            with open(out_pkl, "wb") as f:
                pickle.dump(sample, f)

        except Exception as e:
            print(f"  [WARN] {fname}: {e}")
            skipped += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(noisy_files)} done …")

    print(f"[Preprocess] Done. Skipped {skipped}.")


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_ckpt(out_dir, step, model, opt, sched, best_loss, filename=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename or f"ckpt_{step:08d}.pt")
    torch.save({
        "step":          step,
        "model":         unwrap(model).state_dict(),
        "optimizer":     opt.state_dict(),
        "scheduler":     sched.state_dict(),
        "best_val_loss": best_loss,
    }, path)
    return path


def load_ckpt(path, model, opt, sched, device):
    ckpt = torch.load(path, map_location=device)
    unwrap(model).load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    sched.load_state_dict(ckpt["scheduler"])
    step = ckpt["step"]
    best = ckpt.get("best_val_loss", float("inf"))
    if is_main():
        print(f"[Checkpoint] Resumed step={step}, best={best:.6f}")
    return step, best


# ===========================================================================
# Validation
# ===========================================================================

@torch.no_grad()
def validate(model, val_loader, device):
    unwrap(model).eval()
    total, n = 0.0, 0
    for batch in val_loader:
        ns = batch["noisy_sem"].to(device)
        cs = batch["clean_sem"].to(device)
        na = batch["noisy_ac"].to(device)
        ca = batch["clean_ac"].to(device)
        total += unwrap(model)(ns, cs, na, ca).item()
        n += 1
    unwrap(model).train()
    return total / max(n, 1)


# ===========================================================================
# Training loop
# ===========================================================================

def train(args):
    distributed = "LOCAL_RANK" in os.environ
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if distributed:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed + get_rank())

    if is_main():
        vocab = 3 + args.semantic_num + args.acoustic_num
        print(f"[S2S Train] Device      : {device} | World: {get_world_size()}")
        print(f"[S2S Train] semantic_num: {args.semantic_num}")
        print(f"[S2S Train] acoustic_num: {args.acoustic_num}")
        print(f"[S2S Train] vocab_size  : {vocab}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = S2SModel(
        semantic_num        = args.semantic_num,
        acoustic_num        = args.acoustic_num,
        hidden_size         = args.hidden_size,
        num_hidden_layers   = args.num_layers,
        num_attention_heads = args.num_heads,
        n_positions         = args.n_positions,
    ).to(device)

    if is_main():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[S2S Train] Parameters  : {n_params:,}")

    opt   = AdamW(model.parameters(), lr=args.lr,
                  betas=(0.9, 0.98), weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=args.total_steps, eta_min=args.lr * 0.1)

    step, best_val_loss = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        step, best_val_loss = load_ckpt(args.resume, model, opt, sched, device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_loader, val_loader, tr_sam = build_dataloaders(args, distributed)

    model.train()
    running_loss = 0.0
    t0    = time.time()
    epoch = 0

    if is_main():
        print(f"[S2S Train] Starting | steps={args.total_steps} | "
              f"batch={args.batch_size} | lr={args.lr}")

    while step < args.total_steps:
        epoch += 1
        if tr_sam is not None:
            tr_sam.set_epoch(epoch)

        for batch in train_loader:
            if step >= args.total_steps:
                break

            # Linear warmup
            if step < args.warmup_steps:
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * (step + 1) / args.warmup_steps

            ns = batch["noisy_sem"].to(device, non_blocking=True)
            cs = batch["clean_sem"].to(device, non_blocking=True)
            na = batch["noisy_ac"].to(device,  non_blocking=True)
            ca = batch["clean_ac"].to(device,  non_blocking=True)

            loss = unwrap(model)(ns, cs, na, ca)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step >= args.warmup_steps:
                sched.step()

            running_loss += loss.item()
            step += 1

            if is_main() and step % args.log_interval == 0:
                elapsed = time.time() - t0
                print(f"Step {step:7d}/{args.total_steps} | "
                      f"loss={running_loss/args.log_interval:.4f} | "
                      f"lr={opt.param_groups[0]['lr']:.2e} | "
                      f"{elapsed:.1f}s")
                running_loss = 0.0
                t0 = time.time()

            if is_main() and step % args.val_interval == 0:
                val_loss = validate(model, val_loader, device)
                print(f"  [Val] step={step} loss={val_loss:.4f}")
                save_ckpt(args.output_dir, step, model, opt, sched, best_val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_ckpt(args.output_dir, step, model, opt, sched,
                              best_val_loss, filename="best.pt")
                    print(f"  [Val] ✓ New best: {best_val_loss:.6f}")

    if is_main():
        save_ckpt(args.output_dir, step, model, opt, sched,
                  best_val_loss, filename="final.pt")
        print(f"[S2S Train] Done. -> {args.output_dir}/final.pt")

    cleanup_ddp()


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--preprocess",    action="store_true")

    # Preprocessing
    p.add_argument("--data_dir",      type=str, default=None)
    p.add_argument("--codec_ckpt",    type=str, default=None)
    p.add_argument("--codec_config",  type=str, default=None)
    p.add_argument("--n2s_ckpt",      type=str, default=None,
                   help="Trained N2S checkpoint. If absent, uses ground-truth clean tokens.")
    p.add_argument("--n2s_config",    type=str, default="../n2s/config.json")
    p.add_argument("--ssl_ckpt",      type=str, default="../n2s/ckpts/hubert_base_ls960.pt")
    p.add_argument("--ssl_km",        type=str, default="../n2s/ckpts/kmeans_hubert_base_l9_c500.pt")
    p.add_argument("--ssl_layer",     type=int, default=9)

    # Shared
    p.add_argument("--token_dir",     type=str, required=True)
    p.add_argument("--output_dir",    type=str, default="../s2s/checkpoints/s2s")

    # Vocabulary
    p.add_argument("--semantic_num",  type=int, default=500)
    p.add_argument("--acoustic_num",  type=int, default=8192)

    # Model architecture
    p.add_argument("--hidden_size",   type=int, default=1024)
    p.add_argument("--num_layers",    type=int, default=12)
    p.add_argument("--num_heads",     type=int, default=16)
    p.add_argument("--n_positions",   type=int, default=4096,
                   help="Must be > 2*T_sem + 1 + 2*T_ac")
    p.add_argument("--max_ac_len",    type=int, default=500,
                   help="Max acoustic token length. 50 tok/sec -> 500 = 10s")

    # Multi-GPU
    p.add_argument("--gpu_ids",       type=str, default=None)

    # Resume
    p.add_argument("--resume",        type=str, default=None)

    # Training
    p.add_argument("--total_steps",   type=int,   default=200_000)
    p.add_argument("--batch_size",    type=int,   default=4,
                   help="Per-GPU. GPT2-large is VRAM heavy — keep small.")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--warmup_steps",  type=int,   default=4_000)
    p.add_argument("--log_interval",  type=int,   default=100)
    p.add_argument("--val_interval",  type=int,   default=2_000)
    p.add_argument("--seed",          type=int,   default=42)
    # Preprocessing subset
    p.add_argument("--max_preprocess_pairs", type=int, default=1000,
                help="Random number of noisy/clean pairs to preprocess. Use 0 for all.")

    # Training subset
    p.add_argument("--max_train_files", type=int, default=0,
                help="Random number of token .pkl files to train on. Use 0 for all.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.preprocess:
        assert args.data_dir,     "--data_dir required for --preprocess"
        assert args.codec_ckpt,   "--codec_ckpt required for --preprocess"
        assert args.codec_config, "--codec_config required for --preprocess"
        preprocess(args)
    else:
        train(args)



