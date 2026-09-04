import os
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000

SSL_MODEL = os.environ.get("SSL_MODEL", "microsoft/wavlm-base")
SSL_LAYER = int(os.environ.get("SSL_LAYER", "12"))
NOISE_SNR_DB = float(os.environ.get("NOISE_SNR_DB", "20"))


print(f"\nLoading SSL model: {SSL_MODEL}")
feature_extractor = AutoFeatureExtractor.from_pretrained(SSL_MODEL)
model = AutoModel.from_pretrained(SSL_MODEL, output_hidden_states=True).to(DEVICE)
model.eval()


def load_audio_mono_16k(path):
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.astype(np.float32)
    wav = wav.mean(axis=1)
    if sr != SR:
        wav = resample_linear(wav, sr, SR)
    return wav


def resample_linear(wav, sr, target_sr):
    ratio = target_sr / sr
    new_len = int(round(wav.shape[0] * ratio))
    x_old = np.linspace(0.0, 1.0, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def add_noise_snr(clean, snr_db=20.0):
    if clean.size == 0:
        return clean
    clean_power = np.mean(clean ** 2)
    if clean_power == 0:
        return clean
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = clean_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=clean.shape).astype(np.float32)
    noisy = clean + noise
    return np.clip(noisy, -1.0, 1.0)


def extract_ssl_hidden(wav):
    inputs = feature_extractor(
        wav,
        sampling_rate=SR,
        return_tensors="pt"
    )

    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_values)

    hidden_states = outputs.hidden_states
    layer_features = hidden_states[SSL_LAYER]  # (1, T, D)

    return layer_features.squeeze(0).cpu().numpy()  # (T, D)


def frame_level_similarity(clean_hidden, noisy_hidden):
    min_len = min(len(clean_hidden), len(noisy_hidden))

    clean_trim = clean_hidden[:min_len]
    noisy_trim = noisy_hidden[:min_len]

    sims = []

    for c, n in zip(clean_trim, noisy_trim):
        sim = cosine_similarity([c], [n])[0, 0]
        sims.append(sim)

    return np.mean(sims)


def pooled_similarity(clean_hidden, noisy_hidden):
    clean_mean = clean_hidden.mean(axis=0)
    noisy_mean = noisy_hidden.mean(axis=0)
    return cosine_similarity([clean_mean], [noisy_mean])[0, 0]


def main():
    input_dir = "input"

    results = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            print(f"Processing: {filename}")

            file_path = os.path.join(input_dir, filename)

            clean_audio = load_audio_mono_16k(file_path)
            noisy_audio = add_noise_snr(clean_audio, snr_db=NOISE_SNR_DB)

            clean_hidden = extract_ssl_hidden(clean_audio)
            noisy_hidden = extract_ssl_hidden(noisy_audio)

            frame_sim = frame_level_similarity(clean_hidden, noisy_hidden)
            pool_sim = pooled_similarity(clean_hidden, noisy_hidden)

            results.append({
                "file": filename,
                "frame_similarity": frame_sim,
                "pooled_similarity": pool_sim
            })

    if results:
        print("\nResults:")
        for r in results:
            print(
                f"{r['file']} -> "
                f"Frame Cosine: {r['frame_similarity']:.4f} | "
                f"Pooled Cosine: {r['pooled_similarity']:.4f}"
            )

        mean_frame = np.mean([r["frame_similarity"] for r in results])
        mean_pool = np.mean([r["pooled_similarity"] for r in results])

        print("\nOverall:")
        print(f"Average Frame Similarity:  {mean_frame:.4f}")
        print(f"Average Pooled Similarity: {mean_pool:.4f}")


if __name__ == "__main__":
    main()
