import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

from components.model import SimCodec


class AudioFolderDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, segment_length=16000, train=True):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.train = train

        exts = ["*.wav"]
        files = []
        for ext in exts:
            files.extend(self.root_dir.rglob(ext))
        self.files = sorted([str(f) for f in files])

        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in: {root_dir}")

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path):
        wav, sr = sf.read(path)  # (T,) or (T, C)

        # Convert to float32
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        # Convert to mono
        if wav.ndim == 2:
            wav = np.mean(wav, axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            wav = resample_poly(wav, self.sample_rate, sr)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(wav)) + 1e-9
        wav = wav / max_val

        # Convert to tensor (1, T)
        wav = torch.from_numpy(wav).unsqueeze(0)

        return wav

    def _crop_or_pad(self, wav):
        T = wav.size(-1)

        if T >= self.segment_length:
            if self.train:
                start = torch.randint(0, T - self.segment_length + 1, (1,)).item()
            else:
                start = (T - self.segment_length) // 2
            wav = wav[:, start:start + self.segment_length]
        else:
            pad = self.segment_length - T
            wav = F.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):
        wav = self._load_audio(self.files[idx])
        wav = self._crop_or_pad(wav)
        return wav


def mrstft_loss(x_hat, x):
    x_hat = x_hat.squeeze(1)
    x = x.squeeze(1)

    configs = [
        (1024, 256, 1024),
        (512, 128, 512),
        (2048, 512, 2048),
    ]

    total_sc = 0.0
    total_mag = 0.0

    for n_fft, hop, win in configs:
        window = torch.hann_window(win, device=x.device)

        X_hat = torch.stft(
            x_hat, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True
        )
        X = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True
        )

        mag_hat = torch.abs(X_hat)
        mag = torch.abs(X)

        sc_loss = torch.norm(mag - mag_hat, p="fro") / (torch.norm(mag, p="fro") + 1e-8)
        mag_loss = F.l1_loss(torch.log(mag_hat + 1e-7), torch.log(mag + 1e-7))

        total_sc += sc_loss
        total_mag += mag_loss

    return (total_sc + total_mag) / len(configs)


def save_checkpoint(model, optimizer, step, epoch, out_path):
    ckpt = {
        "encoder": model.encoder.state_dict(),
        "quantizer": model.quantizer.state_dict(),
        "generator": model.generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    torch.save(ckpt, out_path)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCodec(args.config).to(device)
    model.train()

    dataset = AudioFolderDataset(
        args.data_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        train=True
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=1e-4
    )

    os.makedirs(args.out_dir, exist_ok=True)

    global_step = 0

    for epoch in range(args.epochs):
        for i, x in enumerate(loader):
            x = x.to(device)

            optimizer.zero_grad()

            z = model.encoder(x)
            z_q, q_loss, indices = model.quantizer(z)
            x_hat = model.generator(z_q)

            l1 = F.l1_loss(x_hat, x)
            stft = mrstft_loss(x_hat, x)

            loss = l1 + stft + q_loss

            loss.backward()
            optimizer.step()

            if global_step % args.log_every == 0:
                print(
                    f"step={global_step} "
                    f"loss={loss.item():.4f} "
                    f"l1={l1.item():.4f} "
                    f"stft={stft.item():.4f} "
                    f"vq={q_loss.item():.4f}"
                )

            global_step += 1


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="ckpts")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--segment_length", type=int, default=16000)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--log_every", type=int, default=50)

    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    train(args)