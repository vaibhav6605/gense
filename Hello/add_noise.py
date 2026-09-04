import numpy as np
import soundfile as sf

INPUT_FILE = "clean_audio/61-70968-0000.wav"
OUTPUT_FILE = "noisy.wav"
SNR_DB = 40

def add_noise_snr(clean, snr_db):
    signal_power = np.mean(clean ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), clean.shape)
    return clean + noise

audio, sr = sf.read(INPUT_FILE)

if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float32)

noisy_audio = add_noise_snr(audio, SNR_DB)
noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

sf.write(OUTPUT_FILE, noisy_audio, sr)

print(f"SNR: {SNR_DB} dB")