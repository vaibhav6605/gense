import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample_poly
import json

from components.simcodec.modules import Encoder, Quantizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SR = 16000

# ---- config ----
with open("config_stage1.json") as f:
    cfg = json.load(f)

class H:
    def __init__(self, d): self.__dict__ = d

h = H(cfg)

# ---- load model ----
encoder = Encoder(h).to(DEVICE)
quantizer = Quantizer(h).to(DEVICE)

ckpt = torch.load("checkpoints/best1.pt", map_location=DEVICE)

encoder.load_state_dict(ckpt["encoder"])
quantizer.load_state_dict(ckpt["quantizer"])

encoder.eval()
quantizer.eval()

# ---- audio ----
AUDIO_DIR = "audio"   # put 10–20 files here
audio_files = list(Path(AUDIO_DIR).glob("*.wav"))

print(f"Testing {len(audio_files)} files\n")

def extract_tokens(wav):
    wav = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z = encoder(wav)
        _, _, indices_list = quantizer(z)
    tokens = torch.stack(indices_list, dim=-1).squeeze(0).cpu().numpy()  # (T, 2)
    combined = tokens[:, 0] * h.n_codes + tokens[:, 1]
    return tokens, combined


def analyze(tokens, combined):
    # ---- run-length ----
    runs = []
    count = 1
    for i in range(1, len(combined)):
        if combined[i] == combined[i-1]:
            count += 1
        else:
            runs.append(count)
            count = 1
    runs.append(count)
    avg_run = np.mean(runs)

    # ---- unique ratio ----
    unique_ratio = len(np.unique(combined)) / len(combined)

    return avg_run, unique_ratio


# ---- run ----
for audio_path in audio_files:
    print(f"\n--- {audio_path.name} ---")

    wav, sr = sf.read(audio_path)

    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)

    if sr != TARGET_SR:
        wav = resample_poly(wav, TARGET_SR, sr)

    tokens1, combined1 = extract_tokens(wav)
    tokens2, combined2 = extract_tokens(wav)

    avg_run, uniq_ratio = analyze(tokens1, combined1)

    # ---- stability ----
    stability = np.mean(combined1 == combined2)

    # ---- print sample ----
    print("Sample tokens:", combined1[:20])
    print(f"Avg run length : {avg_run:.2f}")
    print(f"Unique ratio   : {uniq_ratio:.3f}")
    print(f"Stability      : {stability:.3f}")

    # ---- verdict ----
    if avg_run < 2:
        print("❌ Too random")
    elif uniq_ratio > 0.5:
        print("❌ Too random (high diversity)")
    elif uniq_ratio < 0.02:
        print("❌ Collapse")
    elif stability < 0.9:
        print("❌ Unstable")
    else:
        print("✅ GOOD TOKENS")

print("\nDone!")