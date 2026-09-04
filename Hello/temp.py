'''import os
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000

SSL_MODEL = "microsoft/wavlm-base"
SSL_LAYER = 12

print(f"\nLoading SSL model: {SSL_MODEL}")
feature_extractor = AutoFeatureExtractor.from_pretrained(SSL_MODEL)
model = AutoModel.from_pretrained(
    SSL_MODEL,
    output_hidden_states=True
).to(DEVICE)

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


# 🔥 NEW: Frame-Level Cosine Instead of Mean Pooling
def frame_cosine(wav1, wav2):
    inputs1 = feature_extractor(wav1, sampling_rate=SR, return_tensors="pt")
    inputs2 = feature_extractor(wav2, sampling_rate=SR, return_tensors="pt")

    with torch.no_grad():
        out1 = model(inputs1.input_values.to(DEVICE))
        out2 = model(inputs2.input_values.to(DEVICE))

    h1 = out1.hidden_states[SSL_LAYER].squeeze(0).cpu().numpy()
    h2 = out2.hidden_states[SSL_LAYER].squeeze(0).cpu().numpy()

    min_len = min(h1.shape[0], h2.shape[0])

    sims = []
    for t in range(min_len):
        sims.append(
            cosine_similarity([h1[t]], [h2[t]])[0, 0]
        )

    return np.mean(sims)


def main():
    # ⚠ FIXED PATHS (use raw string or double backslash)
    file1 = r"clean_audio\61-70968-0000.wav"
    file2 = r"noisy_audio\61-70968-0000.wav"

    print(f"\nProcessing:\n  {file1}\n  {file2}")

    wav1 = load_audio_mono_16k(file1)
    wav2 = load_audio_mono_16k(file2)

    sim = frame_cosine(wav1, wav2)

    print(f"\nFrame-Level Cosine Similarity: {sim:.4f}")


if __name__ == "__main__":
    main()'''

'''import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly
from encodec import EncodecModel

# -----------------------
# CONFIG
# -----------------------
FILE1 = r"clean_audio\61-70968-0000.wav"
FILE2 = r"noisy.wav"

SR = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load EnCodec
# -----------------------
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model = model.to(DEVICE)
model.eval()

# -----------------------
# Load Audio
# -----------------------
def load_audio(path):
    wav, sr = sf.read(path)

    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)

    if sr != SR:
        wav = resample_poly(wav, SR, sr)

    return wav.astype(np.float32)

# -----------------------
# Extract EnCodec Tokens
# -----------------------
def extract_tokens(wav):

    wav = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoded = model.encode(wav)

    codes = torch.cat([e[0] for e in encoded], dim=-1)

    return codes.view(-1).cpu().numpy().tolist()

# -----------------------
# Edit Distance
# -----------------------
def edit_distance(seq1, seq2):

    m, n = len(seq1), len(seq2)

    dp = np.zeros((m+1, n+1), dtype=np.int32)

    for i in range(m+1):
        dp[i][0] = i

    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):

            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]

            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1],
                )

    return dp[m][n]

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    wav1 = load_audio(FILE1)
    wav2 = load_audio(FILE2)

    tokens1 = extract_tokens(wav1)
    tokens2 = extract_tokens(wav2)

    edits = edit_distance(tokens1, tokens2)

    ter = edits / len(tokens1)

    print("\n===== TER RESULT =====")
    print(f"File 1 tokens : {len(tokens1)}")
    print(f"File 2 tokens : {len(tokens2)}")
    print(f"Edit distance : {edits}")
    print(f"TER           : {ter:.4f}")'''

import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from encodec import EncodecModel
from sklearn.metrics.pairwise import cosine_similarity

FILE1 = r"clean_audio\61-70968-0000.wav"
FILE2 = r"noisy_audio\61-70968-0000.wav"

SR = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = EncodecModel.encodec_model_24khz().to(DEVICE)
model.set_target_bandwidth(6.0)
model.eval()

def load_audio(path):
    wav, sr = sf.read(path)
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
    if sr != SR:
        wav = resample_poly(wav, SR, sr)
    return wav.astype(np.float32)

def extract_embeddings(wav):
    wav = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encoder(wav)
    emb = emb.squeeze(0).transpose(0,1).cpu().numpy()
    return emb

def pooled_cosine(e1, e2):
    v1 = e1.mean(axis=0)
    v2 = e2.mean(axis=0)
    return cosine_similarity([v1],[v2])[0,0]

def frame_cosine(e1, e2):
    T = min(len(e1), len(e2))
    sims = []
    for t in range(T):
        sims.append(cosine_similarity([e1[t]],[e2[t]])[0,0])
    return np.mean(sims)

wav1 = load_audio(FILE1)
wav2 = load_audio(FILE2)

emb1 = extract_embeddings(wav1)
emb2 = extract_embeddings(wav2)

pooled_sim = pooled_cosine(emb1, emb2)
frame_sim = frame_cosine(emb1, emb2)

print(f"Pooled cosine similarity : {pooled_sim:.4f}")
print(f"Frame-level cosine similarity : {frame_sim:.4f}")