"""
ac2ac_lm_train.py  —  Noisy Acoustic → Clean Acoustic Token LM
===============================================================
Simplified version of s2s_lm_train.py that maps:

    [noisy_acoustic]  →  clean acoustic tokens

No semantic tokens, no N2S model, no HuBERT.
Only SimCodec Stage 2 is needed.

Token vocabulary:
  0              PAD
  1              BOS
  2              EOS
  3 .. 3+ac-1    acoustic tokens  (raw id + 3)

Input chain:
  [noisy_ac | BOS | clean_ac]
   <no loss>        <loss>

Position IDs:
  noisy_ac[i] -> i          (0 .. T_ac-1)
  BOS         -> T_ac
  clean_ac[i] -> i          (restart at 0)

Usage
─────
  # Step 1 — preprocess (run once):
  python3 s2sh_lm.py --preprocess 
      --data_dir     ../s2s/data 
      --codec_ckpt   ../s2s/checkpoints/stage2/best.pt 
      --codec_config ../s2s/config_stage2.json 
      --token_dir    tokens/s2sh

  # Step 2 — train:
  python3 s2sh_lm.py \
      --token_dir    tokens/ac2ac \
      --output_dir   checkpoints/ac2ac \
      --acoustic_num 8190

  # Multi-GPU:
  torchrun --nproc_per_node=2 ac2ac_lm_train.py \
      --token_dir    tokens/ac2ac \
      --output_dir   checkpoints/ac2ac \
      --acoustic_num 8190 \
      --gpu_ids 0,1
"""

"""
ac2ac_lm_train.py -- Noisy Acoustic -> Clean Acoustic Token LM
==============================================================

This trainer matches the provided SimCodec Stage 2 config:

    top_n = 32
    top_k = 32
    acoustic_num = top_n * top_k = 1024
    segment_len = 32000 samples
    upsample_rates = [8, 5, 4, 2]  -> hop = 320 samples
    max_ac_len = segment_len / hop = 100 tokens

Token vocabulary:
  0                   PAD
  1                   BOS
  2                   EOS
  3 .. 3+acoustic_num acoustic tokens (raw id + 3)

Input chain:
  [noisy_ac | BOS | clean_ac]
   <no loss>        <loss>

Position IDs:
  noisy_ac[i]  -> i
  BOS          -> T_ac
  clean_ac[i]  -> i  (restart at 0)

Usage:
  # Preprocess once:
  python ac2ac_lm_train.py --preprocess ^
      --data_dir dataset ^
      --codec_ckpt checkpoints/stage2/best.pt ^
      --codec_config config_stage2.json ^
      --token_dir tokens/ac2ac

  # Train:
  python ac2ac_lm_train.py ^
      --token_dir tokens/ac2ac ^
      --codec_config config_stage2.json ^
      --output_dir checkpoints/ac2ac

  # Multi-GPU:
  torchrun --nproc_per_node=2 ac2ac_lm_train.py ^
      --token_dir tokens/ac2ac ^
      --codec_config config_stage2.json ^
      --output_dir checkpoints/ac2ac ^
      --gpu_ids 0,1
"""

import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from math import gcd, prod
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

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

def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_main():
    return (not is_dist()) or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if is_dist() else 0


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


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
# Config/audio helpers
# ===========================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def acoustic_num_from_config(config):
    if "acoustic_num" in config:
        return int(config["acoustic_num"])
    if "top_n" in config and "top_k" in config:
        return int(config["top_n"]) * int(config["top_k"])
    raise ValueError("Config must contain either acoustic_num or top_n/top_k.")


def max_ac_len_from_config(config):
    if "segment_len" not in config or "upsample_rates" not in config:
        return None
    hop = prod(int(x) for x in config["upsample_rates"])
    return max(1, int(config["segment_len"]) // hop)


def load_audio_mono(path, target_sr=16_000):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        wav = resample_poly(wav, target_sr // g, sr // g)
    return wav.astype(np.float32)


# ===========================================================================
# AC2AC Model
# ===========================================================================

class AC2ACModel(nn.Module):
    PAD = 0
    BOS = 1
    EOS = 2
    SHIFT = 3

    def __init__(
        self,
        acoustic_num,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        n_positions=2048,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    ):
        super().__init__()
        self.acoustic_num = int(acoustic_num)
        vocab_size = self.SHIFT + self.acoustic_num

        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            activation_function="gelu_new",
            n_positions=n_positions,
            n_ctx=n_positions,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=self.BOS,
            eos_token_id=self.EOS,
            pad_token_id=self.PAD,
        )
        self.lm = GPT2LMHeadModel(cfg)

    def _shift_acoustic(self, acoustic):
        """
        Convert raw ids to LM token ids.

        Raw ids are 0-indexed codec entries. Padding is represented as -1 in
        dataset tensors because raw id 0 is a real codec token.
        """
        tokens = torch.full_like(acoustic, self.PAD)
        valid = acoustic >= 0
        tokens[valid] = acoustic[valid] + self.SHIFT
        return tokens

    def _build_chain(self, noisy_ac, clean_ac=None):
        B, T_ac = noisy_ac.shape
        dev = noisy_ac.device

        na = self._shift_acoustic(noisy_ac)
        bos = torch.full((B, 1), self.BOS, dtype=torch.long, device=dev)

        ac_pos = torch.arange(T_ac, device=dev)
        bos_pos = torch.tensor([T_ac], device=dev)

        if clean_ac is not None:
            ca = self._shift_acoustic(clean_ac)
            tokens = torch.cat([na, bos, ca], dim=1)
            pos = torch.cat([ac_pos, bos_pos, ac_pos])
        else:
            tokens = torch.cat([na, bos], dim=1)
            pos = torch.cat([ac_pos, bos_pos])

        position_ids = pos.unsqueeze(0).expand(B, -1)
        attention_mask = (tokens != self.PAD).long()
        return tokens, position_ids, attention_mask

    def forward(self, noisy_ac, clean_ac):
        tokens, pos, attention_mask = self._build_chain(noisy_ac, clean_ac)
        outputs = self.lm(
            input_ids=tokens,
            position_ids=pos,
            attention_mask=attention_mask,
        )
        logits = outputs.logits
        T_ac = clean_ac.shape[1]

        pred_logits = logits[:, T_ac:T_ac + T_ac, :].clone()
        targets = self._shift_acoustic(clean_ac)

        pred_logits[:, :, :self.SHIFT] = -1e9
        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.PAD,
        )
        return loss

    @torch.no_grad()
    def inference(self, noisy_ac, temperature=1.0):
        B, T_ac = noisy_ac.shape
        dev = noisy_ac.device

        tokens, pos, attention_mask = self._build_chain(noisy_ac, clean_ac=None)
        generated = []

        for step in tqdm(range(T_ac), desc="Generating clean tokens"):
            outputs = self.lm(
                input_ids=tokens,
                position_ids=pos,
                attention_mask=attention_mask,
            )
            last_logit = outputs.logits[:, -1, :].clone()
            mask = torch.ones_like(last_logit, dtype=torch.bool)
            mask[:, self.SHIFT:self.SHIFT + self.acoustic_num] = False
            last_logit[mask] = -1e9

            probs = (last_logit / max(temperature, 1e-5)).softmax(dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token)

            next_pos = torch.full((B, 1), step, dtype=torch.long, device=dev)
            tokens = torch.cat([tokens, next_token], dim=1)
            pos = torch.cat([pos, next_pos], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((B, 1), dtype=torch.long, device=dev)],
                dim=1,
            )

        return torch.cat(generated, dim=1) - self.SHIFT


# ===========================================================================
# Dataset
# ===========================================================================

class AC2ACDataset(Dataset):
    def __init__(
        self,
        token_dir,
        max_ac_len=100,
        split="train",
        val_fraction=0.02,
        seed=42,
        max_files=0,
    ):
        self.max_ac_len = int(max_ac_len)

        all_files = sorted(glob.glob(os.path.join(token_dir, "*.pkl")))
        if not all_files:
            raise RuntimeError(f"No .pkl files in {token_dir}. Run --preprocess first.")

        rng = random.Random(seed)
        rng.shuffle(all_files)
        if max_files and max_files > 0:
            all_files = all_files[:max_files]

        n_val = max(1, int(len(all_files) * val_fraction))
        self.files = all_files[:n_val] if split == "val" else all_files[n_val:]

        if is_main():
            print(f"[AC2ACDataset] {split}: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            sample = pickle.load(f)

        noisy_ac = sample["noisy_ac"].long()
        clean_ac = sample["clean_ac"].long()

        if noisy_ac.dim() == 2:
            noisy_ac = noisy_ac.reshape(-1)
        if clean_ac.dim() == 2:
            clean_ac = clean_ac.reshape(-1)

        T = min(noisy_ac.shape[0], clean_ac.shape[0])
        noisy_ac = noisy_ac[:T]
        clean_ac = clean_ac[:T]

        L = self.max_ac_len
        if T >= L:
            start = random.randint(0, T - L)
            noisy_ac = noisy_ac[start:start + L]
            clean_ac = clean_ac[start:start + L]
        else:
            noisy_ac = F.pad(noisy_ac, (0, L - T), value=-1)
            clean_ac = F.pad(clean_ac, (0, L - T), value=-1)

        return {"noisy_ac": noisy_ac, "clean_ac": clean_ac}


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def build_dataloaders(args, distributed):
    kw = dict(
        token_dir=args.token_dir,
        max_ac_len=args.max_ac_len,
        seed=args.seed,
        max_files=args.max_train_files,
    )
    train_ds = AC2ACDataset(split="train", **kw)
    val_ds = AC2ACDataset(split="val", **kw)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader, train_sampler


# ===========================================================================
# Preprocessing
# ===========================================================================

def preprocess(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_json(args.codec_config)
    sample_rate = int(config.get("sample_rate", 16_000))

    print("[Preprocess] Loading SimCodec Stage 2")
    from s2s.components.simcodec.model import SimCodec

    codec = SimCodec(args.codec_config).to(device)
    codec.load_ckpt(args.codec_ckpt)
    codec.eval()

    noisy_dir = os.path.join(args.data_dir, "noisy")
    clean_dir = os.path.join(args.data_dir, "clean")

    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    if not noisy_files:
        raise RuntimeError(f"No .wav files in {noisy_dir}")

    paired = [
        f for f in noisy_files
        if os.path.isfile(os.path.join(clean_dir, os.path.basename(f)))
    ]
    if not paired:
        raise RuntimeError("No matching noisy/clean pairs found.")

    rng = random.Random(args.seed)
    rng.shuffle(paired)
    if args.max_preprocess_pairs and args.max_preprocess_pairs > 0:
        paired = paired[:args.max_preprocess_pairs]

    os.makedirs(args.token_dir, exist_ok=True)
    skipped = 0
    print(f"[Preprocess] Processing {len(paired)} pairs")

    for i, noisy_path in enumerate(paired):
        fname = os.path.basename(noisy_path)
        out_pkl = os.path.join(args.token_dir, fname.replace(".wav", ".pkl"))
        if os.path.isfile(out_pkl):
            continue

        clean_path = os.path.join(clean_dir, fname)
        try:
            noisy_np = load_audio_mono(noisy_path, sample_rate)
            clean_np = load_audio_mono(clean_path, sample_rate)

            with torch.no_grad():
                noisy_ac = codec.encode(
                    torch.from_numpy(noisy_np).to(device).unsqueeze(0).unsqueeze(0)
                ).cpu().squeeze(0)
                clean_ac = codec.encode(
                    torch.from_numpy(clean_np).to(device).unsqueeze(0).unsqueeze(0)
                ).cpu().squeeze(0)

            with open(out_pkl, "wb") as f:
                pickle.dump(
                    {"noisy_ac": noisy_ac.long(), "clean_ac": clean_ac.long()},
                    f,
                )
        except Exception as exc:
            print(f"  [WARN] {fname}: {exc}")
            skipped += 1

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paired)} done")

    print(f"[Preprocess] Done. Skipped {skipped}.")


# ===========================================================================
# Checkpoints/validation/training
# ===========================================================================

def save_ckpt(out_dir, step, model, opt, sched, best_loss, filename=None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename or f"ckpt_{step:08d}.pt")
    torch.save(
        {
            "step": step,
            "model": unwrap(model).state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "best_val_loss": best_loss,
        },
        path,
    )
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


@torch.no_grad()
def validate(model, val_loader, device):
    unwrap(model).eval()
    total, n = 0.0, 0
    for batch in val_loader:
        noisy_ac = batch["noisy_ac"].to(device)
        clean_ac = batch["clean_ac"].to(device)
        total += unwrap(model)(noisy_ac, clean_ac).item()
        n += 1
    unwrap(model).train()
    return total / max(n, 1)


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

    if args.n_positions < (2 * args.max_ac_len + 1):
        raise ValueError(
            f"--n_positions must be >= {2 * args.max_ac_len + 1} "
            f"for max_ac_len={args.max_ac_len}; got {args.n_positions}."
        )

    if is_main():
        print(f"\n[AC2AC Train] Device      : {device} | World: {get_world_size()}")
        print(f"[AC2AC Train] acoustic_num : {args.acoustic_num}")
        print(f"[AC2AC Train] vocab_size   : {AC2ACModel.SHIFT + args.acoustic_num}")
        print(f"[AC2AC Train] max_ac_len   : {args.max_ac_len}")
        print(f"[AC2AC Train] n_positions  : {args.n_positions}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = AC2ACModel(
        acoustic_num=args.acoustic_num,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        n_positions=args.n_positions,
    ).to(device)

    if is_main():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[AC2AC Train] Parameters  : {n_params:,}")

    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=args.total_steps, eta_min=args.lr * 0.1)

    step, best_val_loss = 0, float("inf")
    if args.resume and os.path.isfile(args.resume):
        step, best_val_loss = load_ckpt(args.resume, model, opt, sched, device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    train_loader, val_loader, train_sampler = build_dataloaders(args, distributed)

    model.train()
    running_loss = 0.0
    t0 = time.time()
    epoch = 0

    if is_main():
        print(
            f"[AC2AC Train] Starting | steps={args.total_steps} | "
            f"batch={args.batch_size} | lr={args.lr} | warmup={args.warmup_steps}\n"
        )

    while step < args.total_steps:
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if step >= args.total_steps:
                break

            if step < args.warmup_steps:
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * (step + 1) / max(1, args.warmup_steps)

            noisy_ac = batch["noisy_ac"].to(device, non_blocking=True)
            clean_ac = batch["clean_ac"].to(device, non_blocking=True)

            loss = unwrap(model)(noisy_ac, clean_ac)

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
                print(
                    f"Step {step:7d}/{args.total_steps} | "
                    f"loss={running_loss / args.log_interval:.4f} | "
                    f"lr={opt.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
                )
                running_loss = 0.0
                t0 = time.time()

            if is_main() and step % args.val_interval == 0:
                val_loss = validate(model, val_loader, device)
                print(f"  [Val] step={step} loss={val_loss:.4f}")
                save_ckpt(args.output_dir, step, model, opt, sched, best_val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_ckpt(
                        args.output_dir,
                        step,
                        model,
                        opt,
                        sched,
                        best_val_loss,
                        filename="best.pt",
                    )
                    print(f"  [Val] New best: {best_val_loss:.6f}")

    if is_main():
        save_ckpt(args.output_dir, step, model, opt, sched, best_val_loss, filename="final.pt")
        print(f"\n[AC2AC Train] Done -> {args.output_dir}/final.pt")
        print(f"[AC2AC Train] Best val loss: {best_val_loss:.6f}")

    cleanup_ddp()


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AC2AC: noisy acoustic tokens -> clean acoustic tokens"
    )
    parser.add_argument("--preprocess", action="store_true")

    parser.add_argument("--data_dir", type=str, default=None, help="Root with noisy/ and clean/")
    parser.add_argument("--codec_ckpt", type=str, default=None, help="SimCodec Stage 2 checkpoint")
    parser.add_argument("--codec_config", type=str, default=None, help="SimCodec Stage 2 JSON config")

    parser.add_argument("--token_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/ac2ac")

    parser.add_argument("--acoustic_num", type=int, default=None)

    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--n_positions", type=int, default=2048)
    parser.add_argument("--max_ac_len", type=int, default=None)

    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4_000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_preprocess_pairs", type=int, default=1000)
    parser.add_argument("--max_train_files", type=int, default=0)

    args = parser.parse_args()

    config = load_json(args.codec_config) if args.codec_config else None
    if config:
        if args.acoustic_num is None:
            args.acoustic_num = acoustic_num_from_config(config)
        if args.max_ac_len is None:
            args.max_ac_len = max_ac_len_from_config(config)

    if args.acoustic_num is None:
        args.acoustic_num = 1024
    if args.max_ac_len is None:
        args.max_ac_len = 100

    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.preprocess:
        assert parsed_args.data_dir, "--data_dir required"
        assert parsed_args.codec_ckpt, "--codec_ckpt required"
        assert parsed_args.codec_config, "--codec_config required"
        preprocess(parsed_args)
    else:
        train(parsed_args)
