'''import os
import numpy as np
import soundfile as sf
from src.local_tokenizer import LocalTokenizer
from src.llm_denoiser import GroqDenoiser

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
# --- CONFIGURATION ---
# Make sure you set this in your terminal or here temporarily
# os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxx" 


def main():
    input_dir = "input"
    output_dir = "outputs"
    noisy_audio_dir = os.path.join(output_dir, "noisy_audio")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(noisy_audio_dir, exist_ok=True)

    snr_db = float(os.environ.get("NOISE_SNR_DB", "10"))

    try:
        # Initialize Local Tokenizer
        tokenizer = LocalTokenizer()
        denoiser = GroqDenoiser()
    except Exception as e:
        print(f"Setup Failed: {e}")
        return

    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            print(f"Processing: {filename}...")
            file_path = os.path.join(input_dir, filename)
            
            try:
                clean_audio = load_audio_mono_16k(file_path)
                noisy_audio = add_noise_snr(clean_audio, snr_db=snr_db)

                noisy_path = os.path.join(noisy_audio_dir, filename.replace(".wav", "_noisy.wav"))
                sf.write(noisy_path, noisy_audio, 16000)

                clean_tokens = tokenizer.extract_tokens(file_path)
                noisy_tokens = tokenizer.extract_tokens(noisy_path)
                denoised_tokens = denoiser.denoise_tokens(noisy_tokens.tolist())
                
                # Save
                clean_save = filename.replace(".wav", "_clean.npy")
                noisy_save = filename.replace(".wav", "_noisy.npy")
                denoised_save = filename.replace(".wav", "_denoised.npy")
                np.save(os.path.join(output_dir, clean_save), clean_tokens)
                np.save(os.path.join(output_dir, noisy_save), noisy_tokens)
                np.save(os.path.join(output_dir, denoised_save), np.array(denoised_tokens, dtype=np.int64))
                
                min_len = min(len(clean_tokens), len(denoised_tokens), len(noisy_tokens))
                clean_trim = clean_tokens[:min_len]
                noisy_trim = noisy_tokens[:min_len]
                denoised_trim = np.array(denoised_tokens[:min_len], dtype=np.int64)
                baseline_acc = float(np.mean(noisy_trim == clean_trim))
                denoise_acc = float(np.mean(denoised_trim == clean_trim))
                baseline_ed = edit_distance(noisy_trim.tolist(), clean_trim.tolist())
                denoise_ed = edit_distance(denoised_trim.tolist(), clean_trim.tolist())
                clean_len = max(len(clean_trim), 1)
                baseline_ter = baseline_ed / clean_len
                denoise_ter = denoise_ed / clean_len

                results.append({
                    "file": filename,
                    "len": len(clean_tokens),
                    "baseline_ter": baseline_ter,
                    "denoised_ter": denoise_ter,
                    "ter_gain": baseline_ter - denoise_ter,
                })
                
            except Exception as e:
                print(f"  -> Error on {filename}: {e}")

    if results:
        print_results_table(results)
        print_summary_stats(results)

def load_audio_mono_16k(path):
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.astype(np.float32)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_linear(wav, sr, 16000)
    return wav


def resample_linear(wav, sr, target_sr):
    ratio = target_sr / sr
    new_len = int(round(wav.shape[0] * ratio))
    x_old = np.linspace(0.0, 1.0, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def add_noise_snr(clean, snr_db=10.0):
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


def edit_distance(a, b):
    # Levenshtein distance
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost     # substitution
            )
            prev = temp
    return dp[m]


def print_results_table(results):
    headers = [
        "file", "len",
        "TER(noisy)", "TER(denoise)", "TER_gain"
    ]
    rows = []
    for r in results:
        rows.append([
            r["file"],
            str(r["len"]),
            f"{r['baseline_ter']:.4f}",
            f"{r['denoised_ter']:.4f}",
            f"{r['ter_gain']:.4f}",
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells):
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    print("\nPer-file comparison:")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def print_summary_stats(results):
    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    def variance(xs):
        m = mean(xs)
        return sum((x - m) ** 2 for x in xs) / max(len(xs), 1)

    metrics = {
        "TER(noisy)": [r["baseline_ter"] for r in results],
        "TER(denoise)": [r["denoised_ter"] for r in results],
        "TER_gain": [r["ter_gain"] for r in results],
    }

    print("\nOverall stats (mean ± variance):")
    for name, values in metrics.items():
        print(f"  {name}: {mean(values):.4f} ± {variance(values):.6f}")


if __name__ == "__main__":
    main()

import os
import numpy as np
import soundfile as sf
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from src.local_tokenizer import LocalTokenizer
from src.llm_denoiser import GroqDenoiser

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def load_kmeans_centroids(kmeans_path):
    try:
        kmeans_data = joblib.load(kmeans_path)
        if hasattr(kmeans_data, 'cluster_centers_'):
            return kmeans_data.cluster_centers_
        elif isinstance(kmeans_data, dict) and 'centroids' in kmeans_data:
            import torch
            centroids = kmeans_data['centroids']
            return centroids.numpy() if hasattr(centroids, 'numpy') else centroids
        else:
            raise ValueError("Cannot extract centroids from K-means model")
    except Exception as e:
        print(f"Warning: Could not load centroids: {e}")
        return None


def embedding_similarity_score(predicted, reference, centroids):
    if centroids is None:
        return 0.0
    
    min_len = min(len(predicted), len(reference))
    pred = predicted[:min_len]
    ref = reference[:min_len]
    
    pred_embeddings = centroids[pred]
    ref_embeddings = centroids[ref]
    
    similarities = []
    for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
        sim = cosine_similarity([pred_emb], [ref_emb])[0, 0]
        similarities.append(sim)
    
    return np.mean(similarities)


def main():
    input_dir = "input"
    output_dir = "outputs"
    noisy_audio_dir = os.path.join(output_dir, "noisy_audio")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(noisy_audio_dir, exist_ok=True)

    snr_db = float(os.environ.get("NOISE_SNR_DB", "20"))
    kmeans_path = "models/kmeans_hubert_base_l9_c500.pt"

    try:
        tokenizer = LocalTokenizer(kmeans_path=kmeans_path)
        denoiser = GroqDenoiser()
        
        centroids = load_kmeans_centroids(kmeans_path)
        if centroids is not None:
            print(f"✓ Loaded K-means centroids: {centroids.shape}\n")
        else:
            print("⚠ Could not load centroids - cosine similarity unavailable\n")
        
    except Exception as e:
        print(f"Setup Failed: {e}")
        return

    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            print(f"Processing: {filename}...")
            file_path = os.path.join(input_dir, filename)
            
            try:
                clean_audio = load_audio_mono_16k(file_path)
                noisy_audio = add_noise_snr(clean_audio, snr_db=snr_db)

                noisy_path = os.path.join(noisy_audio_dir, filename.replace(".wav", "_noisy.wav"))
                sf.write(noisy_path, noisy_audio, 16000)

                clean_tokens = tokenizer.extract_tokens(file_path)
                noisy_tokens = tokenizer.extract_tokens(noisy_path)
                denoised_tokens = denoiser.denoise_tokens(noisy_tokens.tolist())
                
                clean_save = filename.replace(".wav", "_clean.npy")
                noisy_save = filename.replace(".wav", "_noisy.npy")
                denoised_save = filename.replace(".wav", "_denoised.npy")
                np.save(os.path.join(output_dir, clean_save), clean_tokens)
                np.save(os.path.join(output_dir, noisy_save), noisy_tokens)
                np.save(os.path.join(output_dir, denoised_save), np.array(denoised_tokens, dtype=np.int64))
                
                min_len = min(len(clean_tokens), len(denoised_tokens), len(noisy_tokens))
                clean_trim = clean_tokens[:min_len]
                noisy_trim = noisy_tokens[:min_len]
                denoised_trim = np.array(denoised_tokens[:min_len], dtype=np.int64)
                
                baseline_sim = embedding_similarity_score(noisy_trim, clean_trim, centroids)
                denoised_sim = embedding_similarity_score(denoised_trim, clean_trim, centroids)
                
                baseline_error = 1 - baseline_sim
                denoised_error = 1 - denoised_sim
                error_reduction = baseline_error - denoised_error

                results.append({
                    "file": filename,
                    "len": len(clean_tokens),
                    "baseline_sim": baseline_sim,
                    "denoised_sim": denoised_sim,
                    "baseline_error": baseline_error,
                    "denoised_error": denoised_error,
                    "error_reduction": error_reduction,
                })
                
            except Exception as e:
                print(f"  -> Error on {filename}: {e}")

    if results:
        print_results_table(results)
        print_summary_stats(results)


def load_audio_mono_16k(path):
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.astype(np.float32)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_linear(wav, sr, 16000)
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


def print_results_table(results):
    headers = [
        "file", "len",
        "Sim(noisy)", "Sim(denoise)", "Err(noisy)", "Err(denoise)", "Err_reduction"
    ]
    rows = []
    for r in results:
        rows.append([
            r["file"],
            str(r["len"]),
            f"{r['baseline_sim']:.4f}",
            f"{r['denoised_sim']:.4f}",
            f"{r['baseline_error']:.4f}",
            f"{r['denoised_error']:.4f}",
            f"{r['error_reduction']:.4f}",
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells):
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    print("\nPer-file comparison:")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))


def print_summary_stats(results):
    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    def variance(xs):
        m = mean(xs)
        return sum((x - m) ** 2 for x in xs) / max(len(xs), 1)

    metrics = {
        "Similarity(noisy)": [r["baseline_sim"] for r in results],
        "Similarity(denoise)": [r["denoised_sim"] for r in results],
        "Error(noisy)": [r["baseline_error"] for r in results],
        "Error(denoise)": [r["denoised_error"] for r in results],
        "Error_reduction": [r["error_reduction"] for r in results],
    }

    print("\nOverall stats (mean ± variance):")
    for name, values in metrics.items():
        print(f"  {name}: {mean(values):.4f} ± {variance(values):.6f}")


if __name__ == "__main__":
    main()'''

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

SSL_MODEL = os.environ.get("SSL_MODEL", "facebook/wav2vec2-base-960h")
SSL_LAYER = int(os.environ.get("SSL_LAYER", "11"))


print(f"\nLoading SSL model: {SSL_MODEL}")
feature_extractor = AutoFeatureExtractor.from_pretrained(SSL_MODEL)
model = AutoModel.from_pretrained(SSL_MODEL, output_hidden_states=True).to(DEVICE)
model.eval()


def load_audio_mono_16k(path):
    wav, sr = sf.read(path, always_2d=True)
    wav = wav.astype(np.float32)
    wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_linear(wav, sr, 16000)
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


def extract_ssl_embedding(wav):
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

    pooled = layer_features.mean(dim=1)  # (1, D)
    embedding = pooled.squeeze(0).cpu().numpy()

    return embedding


def cosine_sim(a, b):
    return cosine_similarity([a], [b])[0, 0]


def main():
    input_dir = "input"
    snr_db = float(os.environ.get("NOISE_SNR_DB", "20"))

    results = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            print(f"Processing: {filename}")

            file_path = os.path.join(input_dir, filename)

            clean_audio = load_audio_mono_16k(file_path)
            noisy_audio = add_noise_snr(clean_audio, snr_db=snr_db)

            clean_emb = extract_ssl_embedding(clean_audio)
            noisy_emb = extract_ssl_embedding(noisy_audio)

            sim = cosine_sim(clean_emb, noisy_emb)

            results.append({
                "file": filename,
                "similarity": sim
            })

    if results:
        print("\nResults:")
        for r in results:
            print(f"{r['file']} -> Cosine Similarity: {r['similarity']:.4f}")

        mean_sim = np.mean([r["similarity"] for r in results])
        print(f"\nAverage Similarity: {mean_sim:.4f}")


if __name__ == "__main__":
    main()