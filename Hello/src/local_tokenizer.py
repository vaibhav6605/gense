import torch
import numpy as np
import soundfile as sf

from transformers import HubertModel, Wav2Vec2FeatureExtractor

class LocalTokenizer:
    def __init__(self, kmeans_path="models/kmeans_hubert_base_l9_c500.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading HuBERT Locally on {self.device}...")
        
        # 1. Load the Model components from Hugging Face
        # (This will download about ~360MB the first time you run it)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device)
        self.model.eval() # Set to evaluation mode
        
        # 2. Load your K-Means (Dummy or Real)
        print(f"Loading K-Means from {kmeans_path}...")
        kmeans_data = self._load_kmeans(kmeans_path)
        if isinstance(kmeans_data, dict) and "centroids" in kmeans_data:
            self.centroids = kmeans_data["centroids"].to(self.device)
        elif hasattr(kmeans_data, "cluster_centers_"):
            self.centroids = torch.from_numpy(kmeans_data.cluster_centers_).to(self.device)
        else:
            raise ValueError("Unsupported K-Means format. Expected dict with 'centroids' or sklearn KMeans.")

    def extract_tokens(self, audio_path):
        # --- A. Load & Preprocess Audio ---
        wav, sr = sf.read(audio_path, always_2d=True)
        # soundfile returns shape (num_samples, num_channels)
        wav = wav.astype(np.float32)
        wav = wav.mean(axis=1)  # mix to mono
        wav = self._resample_if_needed(wav, sr, 16000)
        wav = torch.from_numpy(wav).unsqueeze(0)
        
        # Normalize inputs (Standardization)
        # Note: We use .squeeze() because feature_extractor expects 1D array
        inputs = self.feature_extractor(
            wav.squeeze().numpy(), 
            return_tensors="pt", 
            sampling_rate=16000
        )
        input_values = inputs.input_values.to(self.device)

        # --- B. Extract Vectors (Layer 9) ---
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
            
            # We specifically want Layer 9 features for Semantic Tasks
            # hidden_states is a tuple: (embeddings, layer1, layer2 ... layer12)
            # So index 9 gives us the output of the 9th Transformer Layer
            features = outputs.hidden_states[9] # Shape: (1, Time, 768)

        # --- C. Quantize (Vector -> Token ID) ---
        # Find nearest cluster center for each time step
        x = features.squeeze(0) # (Time, 768)
        y = self.centroids      # (500, 768)
        
        # Compute distances: |x-y|^2 = x^2 - 2xy + y^2
        dists = (
            (x**2).sum(1, keepdim=True) 
            - 2 * torch.matmul(x, y.t()) 
            + (y**2).sum(1).unsqueeze(0)
        )
        
        # The index of the smallest distance is our Token ID
        tokens = dists.argmin(dim=1)
        
        return tokens.cpu().numpy()

    @staticmethod
    def _load_kmeans(kmeans_path):
        try:
            return torch.load(kmeans_path)
        except Exception:
            # Fallbacks for sklearn pickled KMeans (trust only known sources)
            try:
                import joblib
                return joblib.load(kmeans_path)
            except Exception:
                from sklearn.cluster import MiniBatchKMeans
                from torch.serialization import safe_globals

                with safe_globals([MiniBatchKMeans]):
                    return torch.load(kmeans_path, weights_only=False)

    @staticmethod
    def _resample_if_needed(wav, sr, target_sr):
        if sr == target_sr:
            return wav
        # Simple linear resampling (no external deps)
        ratio = target_sr / sr
        new_len = int(round(wav.shape[0] * ratio))
        x_old = np.linspace(0.0, 1.0, num=wav.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        return np.interp(x_new, x_old, wav).astype(np.float32)
