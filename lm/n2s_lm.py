"""
n2s_lm_train.py  —  N2S Language Model Training
================================================
Trains a GPT2-based causal LM that maps noisy semantic tokens
to clean semantic tokens, following equation (1) from the paper:

    p(x) = prod_n  p(ŝ_n | ŝ_1..ŝ_{n-1},  s̄_1..s̄_n)

Token vocabulary layout
────────────────────────
  0                       PAD
  1                       BOS
  2                       EOS
  3  ..  3+sem_num-1      semantic tokens  (raw id + 3)

  Total vocab_size = 3 + sem_num  (= 503 for sem_num=500)

Input sequence (teacher-forced training):
  [s̄_1 .. s̄_n | BOS | ŝ_1 .. ŝ_{n-1}]  →  predict  ŝ_1 .. ŝ_n

Position IDs:
  s̄ (noisy prefix) : 0 .. T-1
  BOS               : T
  ŝ (clean target)  : 0 .. T-1   (positions restart — shared semantic space)

Dataset folder layout
──────────────────────
  data_dir/
    noisy/  *.wav
    clean/  *.wav    (same filenames as noisy/)

Dependencies
────────────
  pip install soundfile scipy transformers torch

Usage
─────
  # Step 1 — extract token pairs (run once):
  python n2s_lm_train.py --preprocess \
      --data_dir data \
      --ssl_ckpt  ckpts/hubert_base_ls960.pt \
      --ssl_km    ckpts/kmeans_hubert_base_l9_c500.pt \
      --ssl_layer 9 \
      --token_dir tokens/n2s

  # Step 2 — train:
  python n2s_lm_train.py \
      --token_dir  tokens/n2s \
      --output_dir ckpts/n2s \
      --semantic_num 500

  # Multi-GPU:
  torchrun --nproc_per_node=2 n2s_lm_train.py \
      --token_dir  tokens/n2s \
      --output_dir ckpts/n2s \
      --semantic_num 500
"""

import os
import sys
import glob
import json
import argparse
import random
import time
import pickle
from math import gcd
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

ROOT = Path(__file__).resolve().parent 
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

# ===========================================================================
# Distributed helpers
# ===========================================================================

def is_dist():  return dist.is_available() and dist.is_initialized()
def is_main():  return (not is_dist()) or dist.get_rank() == 0
def get_rank(): return dist.get_rank() if is_dist() else 0
def get_world_size(): return dist.get_world_size() if is_dist() else 1

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup_ddp():
    if is_dist(): dist.destroy_process_group()

def unwrap(model):
    return model.module if isinstance(model, DDP) else model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ===========================================================================
# Audio helper  (soundfile + scipy — no torchaudio)
# ===========================================================================

def load_audio_mono(path, target_sr=16_000):
    """
    Load any audio file to a mono float32 numpy array at target_sr.

    soundfile handles: WAV, FLAC, OGG, AIFF, and most PCM formats.
    scipy.signal.resample_poly handles resampling with no quality loss.
    """
    data, sr = sf.read(path, always_2d=True)   # (T, C), float64
    wav = data.mean(axis=1).astype(np.float32)  # mono, float32

    if sr != target_sr:
        g   = gcd(sr, target_sr)
        up  = target_sr // g
        down = sr // g
        wav = resample_poly(wav, up, down).astype(np.float32)

    return wav   # (T,) float32 at target_sr


# ===========================================================================
# N2S Model
# ===========================================================================

class N2SModel(nn.Module):
    """
    GPT2-based causal LM for N2S token denoising.

    Vocabulary:
      0          PAD
      1          BOS
      2          EOS
      3..3+K-1   semantic tokens (raw id + SHIFT)

    Input chain (length = T_noisy + 1 + T_clean):
      [s̄_1..s̄_T | BOS | ŝ_1..ŝ_T]

    Position IDs:
      s̄_i  ->  i-1          (0-indexed, positions 0..T-1)
      BOS  ->  T
      ŝ_i  ->  i-1          (restart at 0 — shared semantic space)

    Loss: cross-entropy only over ŝ positions.
    """

    PAD   = 0
    BOS   = 1
    EOS   = 2
    SHIFT = 3   # semantic tokens start at id 3

    def __init__(
        self,
        semantic_num,
        hidden_size         = 768,
        num_hidden_layers   = 12,
        num_attention_heads = 8,
        n_positions         = 2048,
        resid_pdrop         = 0.1,
        embd_pdrop          = 0.1,
        attn_pdrop          = 0.1,
    ):
        super().__init__()
        self.semantic_num = semantic_num
        vocab_size = self.SHIFT + semantic_num   # 3 + 500 = 503

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
    def _build_chain(self, noisy_sem, clean_sem=None):
        B, T  = noisy_sem.shape
        dev   = noisy_sem.device

        noisy_shifted = noisy_sem + self.SHIFT
        bos           = torch.full((B, 1), self.BOS, dtype=torch.long, device=dev)

        sem_pos = torch.arange(T,   device=dev)
        bos_pos = torch.tensor([T], device=dev)

        if clean_sem is not None:
            clean_shifted = clean_sem + self.SHIFT
            tokens = torch.cat([noisy_shifted, bos, clean_shifted], dim=1)
            pos    = torch.cat([sem_pos, bos_pos, sem_pos])
        else:
            tokens = torch.cat([noisy_shifted, bos], dim=1)
            pos    = torch.cat([sem_pos, bos_pos])

        position_ids = pos.unsqueeze(0).expand(B, -1)
        return tokens, position_ids

    # -------------------------------------------------------------------------
    def forward(self, noisy_sem, clean_sem):
        tokens, pos = self._build_chain(noisy_sem, clean_sem)
        outputs     = self.lm(input_ids=tokens, position_ids=pos)
        logits      = outputs.logits    # (B, 2T+1, vocab)

        T = clean_sem.shape[1]

        # logits[:, T : 2T, :] predict clean_sem[0..T-1]
        pred_logits = logits[:, T: 2 * T, :]
        targets     = clean_sem + self.SHIFT

        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.PAD,
        )
        return loss

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, noisy_sem, temperature: float = 1.0):
        B, T  = noisy_sem.shape
        dev   = noisy_sem.device

        tokens, pos = self._build_chain(noisy_sem, clean_sem=None)
        generated   = []

        for step in range(T):
            outputs    = self.lm(input_ids=tokens, position_ids=pos)
            last_logit = outputs.logits[:, -1, :]

            mask = torch.ones_like(last_logit, dtype=torch.bool)
            mask[:, self.SHIFT: self.SHIFT + self.semantic_num] = False
            last_logit[mask] = -1e9

            probs      = (last_logit / max(temperature, 1e-5)).softmax(dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token)

            next_pos = torch.full((B, 1), step, dtype=torch.long, device=dev)
            tokens   = torch.cat([tokens, next_token], dim=1)
            pos      = torch.cat([pos,    next_pos],   dim=1)

        clean_shifted = torch.cat(generated, dim=1)
        return clean_shifted - self.SHIFT


# ===========================================================================
# Dataset
# ===========================================================================

class N2STokenDataset(Dataset):
    def __init__(self, token_dir, max_len=500, split="train",
                 val_fraction=0.02, seed=42):
        self.max_len = max_len
        all_files    = sorted(glob.glob(os.path.join(token_dir, "*.pkl")))
        if not all_files:
            raise RuntimeError(
                f"No .pkl files in {token_dir}. Run --preprocess first.")

        rng   = random.Random(seed)
        rng.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_fraction))
        self.files = all_files[:n_val] if split == "val" else all_files[n_val:]

        if is_main():
            print(f"[N2SDataset] {split}: {len(self.files)} samples")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            s = pickle.load(f)

        noisy = s["noisy_sem"].long()
        clean = s["clean_sem"].long()
        T, L  = noisy.shape[0], self.max_len

        if T >= L:
            start = random.randint(0, T - L)
            noisy = noisy[start: start + L]
            clean = clean[start: start + L]
        else:
            noisy = F.pad(noisy, (0, L - T), value=0)
            clean = F.pad(clean, (0, L - T), value=0)

        return {"noisy_sem": noisy, "clean_sem": clean}


def collate_fn(batch):
    return {
        "noisy_sem": torch.stack([b["noisy_sem"] for b in batch]),
        "clean_sem": torch.stack([b["clean_sem"] for b in batch]),
    }


def build_dataloaders(args, distributed):
    kw = dict(token_dir=args.token_dir, max_len=args.max_len, seed=args.seed)
    train_ds = N2STokenDataset(split="train", **kw)
    val_ds   = N2STokenDataset(split="val",   **kw)

    tr_sampler  = DistributedSampler(train_ds, shuffle=True)  if distributed else None
    val_sampler = DistributedSampler(val_ds,   shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(tr_sampler is None), sampler=tr_sampler,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader, tr_sampler


# ===========================================================================
# Preprocessing
# ===========================================================================

def preprocess(args):
    from n2s.components.semantic_extractor import (
        get_ssl_model, extract_semantic_tokens
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Preprocess] Loading HuBERT + k-means …")
    ssl_model, km_model = get_ssl_model(
        ckpt_path = args.ssl_ckpt,
        km_path   = args.ssl_km,
        device    = str(device),
        layer     = args.ssl_layer,
    )

    noisy_dir   = os.path.join(args.data_dir, "noisy")
    clean_dir   = os.path.join(args.data_dir, "clean")
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    if not noisy_files:
        raise RuntimeError(f"No .wav files found in {noisy_dir}")

    os.makedirs(args.token_dir, exist_ok=True)
    skipped = 0

    print(f"[Preprocess] Processing {len(noisy_files)} pairs …")

    for i, noisy_path in enumerate(noisy_files):
        fname   = os.path.basename(noisy_path)
        out_pkl = os.path.join(args.token_dir, fname.replace(".wav", ".pkl"))
        if os.path.isfile(out_pkl):
            continue

        clean_path = os.path.join(clean_dir, fname)
        if not os.path.isfile(clean_path):
            print(f"  [SKIP] No clean pair for {fname}")
            skipped += 1
            continue

        try:
            noisy_wav = torch.from_numpy(
                load_audio_mono(noisy_path)).unsqueeze(0).to(device)  # (1, T)
            clean_wav = torch.from_numpy(
                load_audio_mono(clean_path)).unsqueeze(0).to(device)  # (1, T)

            noisy_sem = extract_semantic_tokens(ssl_model, km_model, noisy_wav)
            clean_sem = extract_semantic_tokens(ssl_model, km_model, clean_wav)

            min_len   = min(noisy_sem.shape[1], clean_sem.shape[1])
            noisy_sem = noisy_sem[:, :min_len]
            clean_sem = clean_sem[:, :min_len]

            sample = {
                "noisy_sem": noisy_sem.squeeze(0).cpu().long(),
                "clean_sem": clean_sem.squeeze(0).cpu().long(),
            }
            with open(out_pkl, "wb") as f:
                pickle.dump(sample, f)

        except Exception as e:
            print(f"  [WARN] {fname}: {e}")
            skipped += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(noisy_files)} done …")

    print(f"[Preprocess] Finished. Skipped {skipped} files.")


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
        noisy = batch["noisy_sem"].to(device)
        clean = batch["clean_sem"].to(device)
        total += unwrap(model)(noisy, clean).item()
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
        print(f"[N2S Train] Device      : {device} | World: {get_world_size()}")
        print(f"[N2S Train] semantic_num: {args.semantic_num}")
        print(f"[N2S Train] vocab_size  : {3 + args.semantic_num}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = N2SModel(
        semantic_num        = args.semantic_num,
        hidden_size         = args.hidden_size,
        num_hidden_layers   = args.num_layers,
        num_attention_heads = args.num_heads,
        n_positions         = args.n_positions,
    ).to(device)

    if is_main():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[N2S Train] Parameters  : {n_params:,}")

    opt   = AdamW(model.parameters(), lr=args.lr,
                  betas=(0.9, 0.98), weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=args.total_steps, eta_min=args.lr * 0.1)

    step, best_val_loss = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        step, best_val_loss = load_ckpt(args.resume, model, opt, sched, device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_loader, val_loader, tr_sampler = build_dataloaders(args, distributed)

    model.train()
    running_loss = 0.0
    t0    = time.time()
    epoch = 0

    if is_main():
        print(f"[N2S Train] Starting | steps={args.total_steps} | "
              f"batch={args.batch_size} | lr={args.lr}")

    while step < args.total_steps:
        epoch += 1
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)

        for batch in train_loader:
            if step >= args.total_steps:
                break

            if step < args.warmup_steps:
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * (step + 1) / args.warmup_steps

            noisy = batch["noisy_sem"].to(device, non_blocking=True)
            clean = batch["clean_sem"].to(device, non_blocking=True)

            loss = unwrap(model)(noisy, clean)

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
        print(f"[N2S Train] Done. → {args.output_dir}/final.pt")

    cleanup_ddp()


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--preprocess",   action="store_true")

    p.add_argument("--data_dir",     type=str, default=None)
    p.add_argument("--ssl_ckpt",     type=str, default="ckpts/hubert_base_ls960.pt")
    p.add_argument("--ssl_km",       type=str, default="ckpts/kmeans_hubert_base_l9_c500.pt")
    p.add_argument("--ssl_layer",    type=int, default=9)

    p.add_argument("--token_dir",    type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="ckpts/n2s")

    p.add_argument("--semantic_num", type=int, default=500)

    p.add_argument("--hidden_size",  type=int, default=768)
    p.add_argument("--num_layers",   type=int, default=12)
    p.add_argument("--num_heads",    type=int, default=8)
    p.add_argument("--n_positions",  type=int, default=2048)
    p.add_argument("--max_len",      type=int, default=500)

    p.add_argument("--gpu_ids",      type=str, default=None)
    p.add_argument("--resume",       type=str, default=None)

    p.add_argument("--total_steps",  type=int,   default=200_000)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int,   default=4_000)
    p.add_argument("--log_interval", type=int,   default=100)
    p.add_argument("--val_interval", type=int,   default=2_000)
    p.add_argument("--seed",         type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.preprocess:
        assert args.data_dir, "--data_dir required for --preprocess"
        preprocess(args)
    else:
        train(args)