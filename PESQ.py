from pesq import pesq
import soundfile as sf
import numpy as np

def load_audio(path):
    wav, sr = sf.read(path)

    # Convert to mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    return wav, sr


def compute_pesq(ref_path, deg_path):
    ref, sr1 = load_audio(ref_path)
    deg, sr2 = load_audio(deg_path)

    assert sr1 == sr2, "Sample rates must match"
    assert sr1 in [8000, 16000], "PESQ supports only 8k or 16k"

    # Match lengths
    min_len = min(len(ref), len(deg))
    ref = ref[:min_len]
    deg = deg[:min_len]

    score = pesq(sr1, ref, deg, mode='wb')  # 'wb' = wideband (16kHz)

    return score


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    ref_path = "clean.wav"
    deg_path = "enhanced2.wav"

    score = compute_pesq(ref_path, deg_path)

    print(f"PESQ Score: {score:.4f}")