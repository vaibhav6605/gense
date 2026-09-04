import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from transformers import AutoModel, AutoFeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODEL_NAME = "facebook/hubert-base-ls960"
NOISE_PAIRS_CSV = "data/noise_pairs.csv"
SPEAKER_LABELS_CSV = "data/speaker_labels.csv"

ALPHA = 1.0
GAMMA = 1.0
SR = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SSL model...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
model.eval()

def load_audio(path):
    waveform, sr = sf.read(path)

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    if sr != SR:
        waveform = resample_poly(waveform, SR, sr)

    waveform = torch.tensor(waveform, dtype=torch.float32)
    return waveform

def extract_layer_features(waveform):
    inputs = feature_extractor(waveform, sampling_rate=SR, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    hidden_states = outputs.hidden_states
    layer_feats = []

    for layer in hidden_states:
        pooled = layer.mean(dim=1)
        layer_feats.append(pooled.squeeze(0).cpu())

    return layer_feats

def compute_noise_invariance():
    df = pd.read_csv(NOISE_PAIRS_CSV)
    all_sims = []

    print("\nComputing Noise Invariance...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        clean_wave = load_audio(row["clean_path"])
        noisy_wave = load_audio(row["noisy_path"])

        clean_feats = extract_layer_features(clean_wave)
        noisy_feats = extract_layer_features(noisy_wave)

        sims = []
        for cf, nf in zip(clean_feats, noisy_feats):
            sim = F.cosine_similarity(cf, nf, dim=0)
            sims.append(sim.item())

        all_sims.append(sims)

    return np.mean(all_sims, axis=0)

def compute_speaker_leakage():
    df = pd.read_csv(SPEAKER_LABELS_CSV)

    features_per_layer = None
    labels = []

    print("\nExtracting features for Speaker Probe...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        waveform = load_audio(row["audio_path"])
        feats = extract_layer_features(waveform)

        if features_per_layer is None:
            features_per_layer = [[] for _ in range(len(feats))]

        for i, f in enumerate(feats):
            features_per_layer[i].append(f.numpy())

        labels.append(row["label"])

    labels = np.array(labels)

    print("\nTraining Linear Speaker Probes...")
    speaker_acc = []

    for layer_feats in features_per_layer:
        X = np.vstack(layer_feats)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        speaker_acc.append(acc)

    return np.array(speaker_acc)

def main():
    noise_inv = compute_noise_invariance()
    speaker_leak = compute_speaker_leakage()

    semantic_score = ALPHA * noise_inv - GAMMA * speaker_leak

    print("\n========== RESULTS ==========")

    for i in range(len(semantic_score)):
        print(f"Layer {i}:")
        print(f"  Noise Invariance : {noise_inv[i]:.4f}")
        print(f"  Speaker Leakage  : {speaker_leak[i]:.4f}")
        print(f"  Semantic Score   : {semantic_score[i]:.4f}")
        print()

    best_layer = np.argmax(semantic_score)

    print("================================")
    print(f"Best Semantic Layer: {best_layer}")
    print("================================")

if __name__ == "__main__":
    main()