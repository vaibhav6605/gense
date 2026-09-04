import os
import sys
import glob
import json
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from scipy.signal import resample_poly
import librosa

from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ── project root on sys.path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.simcodec.model import SimCodec, AttrDict

# ===========================================================================  
# Distributed helpers  
# ===========================================================================

def is_dist(): return dist.is_available() and dist.is_initialized()
def is_main(): return (not is_dist()) or dist.get_rank() == 0
def get_rank(): return dist.get_rank() if is_dist() else 0
def get_world_size(): return dist.get_world_size() if is_dist() else 1

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup_ddp():
    if is_dist(): dist.destroy_process_group()

def unwrap(model): return model.module if isinstance(model, DDP) else model

# ===========================================================================  
# Helpers  
# ===========================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path) as f:
        return AttrDict(json.load(f))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ===========================================================================  
# Mel Loss (NO torchaudio)  
# ===========================================================================

class MelLossModule(nn.Module):
    FFT_CONFIGS = [
        (128, 32, 128),
        (256, 64, 256),
        (512, 128, 512),
        (1024, 256, 1024),
        (2048, 512, 2048),
    ]
    N_MELS = 80

    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_filters = nn.ParameterList()

        for n_fft, _, _ in self.FFT_CONFIGS:
            mel = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=self.N_MELS
            )
            mel = torch.tensor(mel, dtype=torch.float32)
            self.mel_filters.append(nn.Parameter(mel, requires_grad=False))

    def stft_mel(self, x, n_fft, hop, win, mel_filter):
        window = torch.hann_window(win, device=x.device)

        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True
        )

        mag = spec.abs()
        mel = torch.matmul(mel_filter.to(x.device), mag)
        return torch.log(mel.clamp(min=1e-5))

    def forward(self, real, fake):
        total = torch.tensor(0.0, device=real.device)
        r = real.squeeze(1)
        f = fake.squeeze(1)

        for (n_fft, hop, win), mel_filter in zip(self.FFT_CONFIGS, self.mel_filters):
            r_mel = self.stft_mel(r, n_fft, hop, win, mel_filter)
            f_mel = self.stft_mel(f, n_fft, hop, win, mel_filter)
            total += F.l1_loss(f_mel, r_mel)

        return total / len(self.FFT_CONFIGS)

# ===========================================================================  
# Dataset (soundfile instead of torchaudio)  
# ===========================================================================

class AudioDataset(Dataset):

    def __init__(self, root_dir, sample_rate=16000, segment_len=32000,
                 split="train", val_fraction=0.02, seed=42):

        self.sample_rate = sample_rate
        self.segment_len = segment_len

        all_files = sorted(set(
            glob.glob(os.path.join(root_dir, "*.wav")) +
            glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        ))

        if not all_files:
            raise RuntimeError(f"No .wav files found in '{root_dir}'")

        rng = random.Random(seed)
        rng.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_fraction))

        self.files = all_files[:n_val] if split == "val" else all_files[n_val:]

        if is_main():
            print(f"[Dataset] {split}: {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        wav, sr = sf.read(path)

        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)

        if sr != self.sample_rate:
            wav = resample_poly(wav, self.sample_rate, sr)

        return torch.from_numpy(wav).float().unsqueeze(0)

    def __getitem__(self, idx):
        wav = self._load(self.files[idx])
        T = wav.shape[-1]

        if T >= self.segment_len:
            start = random.randint(0, T - self.segment_len)
            wav = wav[:, start:start + self.segment_len]
        else:
            wav = F.pad(wav, (0, self.segment_len - T))

        return wav

# ===========================================================================  
# (Rest of code unchanged — training loop, DDP, etc.)  
# ===========================================================================

# 👉 KEEP EVERYTHING BELOW EXACTLY SAME AS YOUR ORIGINAL FILE
# (optimizer, training loop, validation, checkpointing, args, etc.)
