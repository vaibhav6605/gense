import os
import json
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pathlib import Path
from components.semantic_extractor import get_ssl_model, extract_semantic_tokens

# ── Config ──────────────────────────────────────────────────────────────────
with open("config.json") as f:
    cfg = json.load(f)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
KM_PATH     = cfg["ssl_model"]["km_path"]
MODEL_NAME  = cfg["ssl_model"]["ckpt_path"]
AUDIO_DIR   = "audio"
OUT_DIR     = "tokens"
TARGET_SR   = 16000

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load models ─────────────────────────────────────────────────────────────
print("Loading models...")
ssl_model, km_model = get_ssl_model(MODEL_NAME, KM_PATH, device=DEVICE, type="hubert")

# ── Process audio files ─────────────────────────────────────────────────────
audio_files = list(Path(AUDIO_DIR).glob("**/*.wav"))
print(f"Found {len(audio_files)} audio file(s)")

for audio_path in audio_files:
    print(f"Processing {audio_path.name} ...")

    # Load audio using soundfile
    waveform, sr = sf.read(str(audio_path))  # shape: (T,) or (T, C)

    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        waveform = resample_poly(waveform, TARGET_SR, sr)

    # Convert to torch tensor → shape (1, T)
    waveform = torch.from_numpy(waveform).float().unsqueeze(0).to(DEVICE)

    # Extract tokens
    tokens = extract_semantic_tokens(ssl_model, km_model, waveform)

    # Save tokens
    out_path = Path(OUT_DIR) / (audio_path.stem + "_tokens.npy")
    np.save(str(out_path), tokens.cpu().numpy())

    print(f"  → saved to {out_path}  |  shape: {tokens.shape}")

print("Done!")