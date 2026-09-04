import os
import random
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import pandas as pd

LIBRISPEECH_ROOT = "LibriSpeech/test-clean"
NUM_SAMPLES = 1000
TARGET_SR = 16000
SNR_RANGE = (0, 20)

CLEAN_DIR = "clean_audio"
NOISY_DIR = "noisy_audio"
OUTPUT_DIR = "data"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_flac_files(root):
    flac_files = []
    for root_dir, _, files in os.walk(root):
        for f in files:
            if f.endswith(".flac"):
                flac_files.append(os.path.join(root_dir, f))
    return flac_files

def resample_audio(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    return resample_poly(waveform, target_sr, orig_sr)

def convert_to_wav(src_path, dst_path):
    waveform, sr = sf.read(src_path)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    waveform = resample_audio(waveform, sr, TARGET_SR)
    sf.write(dst_path, waveform, TARGET_SR)

def add_gaussian_noise(waveform, snr_db):
    signal_power = np.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(waveform)) * np.sqrt(noise_power)
    return waveform + noise

def create_clean_and_noisy():
    print("Collecting LibriSpeech files...")
    all_files = collect_flac_files(LIBRISPEECH_ROOT)
    print(f"Total available files: {len(all_files)}")

    selected = random.sample(all_files, min(len(all_files), NUM_SAMPLES))

    for i, filepath in enumerate(selected):
        filename = os.path.basename(filepath).replace(".flac", ".wav")
        clean_path = os.path.join(CLEAN_DIR, filename)
        noisy_path = os.path.join(NOISY_DIR, filename)

        convert_to_wav(filepath, clean_path)
        waveform, sr = sf.read(clean_path)

        snr_db = random.uniform(*SNR_RANGE)
        noisy_waveform = add_gaussian_noise(waveform, snr_db)

        sf.write(noisy_path, noisy_waveform, TARGET_SR)

        if i % 100 == 0:
            print(f"Processed {i}/{len(selected)}")

    print("Clean & Noisy dataset created.")

def create_noise_pairs():
    rows = []
    clean_files = sorted(os.listdir(CLEAN_DIR))
    for fname in clean_files:
        clean_path = os.path.join(CLEAN_DIR, fname)
        noisy_path = os.path.join(NOISY_DIR, fname)
        if os.path.exists(noisy_path):
            rows.append({"clean_path": clean_path, "noisy_path": noisy_path})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "noise_pairs.csv"), index=False)
    print("Saved noise_pairs.csv")

def create_speaker_labels():
    rows = []
    for fname in os.listdir(CLEAN_DIR):
        if not fname.endswith(".wav"):
            continue
        speaker_id = fname.split("-")[0]
        rows.append({"audio_path": os.path.join(CLEAN_DIR, fname), "label": speaker_id})

    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("category").cat.codes
    df.to_csv(os.path.join(OUTPUT_DIR, "speaker_labels.csv"), index=False)
    print("Saved speaker_labels.csv")

if __name__ == "__main__":
    create_clean_and_noisy()
    create_noise_pairs()
    create_speaker_labels()
    print("\nFull dataset preparation complete.")