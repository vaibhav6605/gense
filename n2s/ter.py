import os
import json
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path
from tqdm import tqdm

from components.semantic_extractor import get_ssl_model, extract_semantic_tokens

# ── Config ────────────────────────────────────────────────────────────────
with open("config.json") as f:
    cfg = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KM_PATH = cfg["ssl_model"]["km_path"]
MODEL_NAME = cfg["ssl_model"]["ckpt_path"]
TARGET_SR = 16000
AUDIO_DIR = "audio"

# ── Load models ───────────────────────────────────────────────────────────
print("Loading models...")
ssl_model, km_model = get_ssl_model(MODEL_NAME, KM_PATH, device=DEVICE, type="hubert")

# ── Utils ────────────────────────────────────────────────────────────────

def add_noise(signal, snr_db=20):
    """Add white Gaussian noise at given SNR (dB)."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.randn(len(signal))
    noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))

    return signal + noise


def preprocess(audio_path):
    wav, sr = sf.read(str(audio_path))

    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)

    if sr != TARGET_SR:
        wav = resample_poly(wav, TARGET_SR, sr)

    return wav


def extract_tokens(wav):
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        tokens = extract_semantic_tokens(ssl_model, km_model, wav_tensor)
    return tokens.squeeze(0).cpu().numpy()


def edit_distance(ref, hyp):
    """Levenshtein distance"""
    n, m = len(ref), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1]  # substitution
                )
    return dp[n][m]


def compute_ter(ref, hyp):
    dist = edit_distance(ref, hyp)
    return dist / max(len(ref), 1)


# ── Main ─────────────────────────────────────────────────────────────────
audio_files = list(Path(AUDIO_DIR).glob("*.wav"))

results = []

print(f"Processing {len(audio_files)} files...\n")

for path in tqdm(audio_files):
    clean = preprocess(path)
    noisy = add_noise(clean, snr_db=20)

    clean_tokens = extract_tokens(clean)
    noisy_tokens = extract_tokens(noisy)

    ter = compute_ter(clean_tokens, noisy_tokens)

    results.append((path.name, ter))

# ── Print table ───────────────────────────────────────────────────────────
print("\n=== TER Results ===")
print(f"{'File':30s} | TER")
print("-" * 45)

for name, ter in results:
    print(f"{name:30s} | {ter:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────
all_ter = [x[1] for x in results]
print("\nAverage TER:", np.mean(all_ter))