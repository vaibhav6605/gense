import os
import torch
import numpy as np
from huggingface_hub import InferenceClient

class RemoteSemanticTokenizer:
    def __init__(self, kmeans_path="models/kmeans_hubert_base_l9_c500.pt"):
        # 1. Setup the API Client
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("Please set your HF_TOKEN environment variable!")
            
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=token
        )
        self.model_id = "facebook/hubert-base-ls960"

        # 2. Load Local K-Means (The Quantizer)
        print(f"Loading local K-Means model from {kmeans_path}...")
        # Load the PyTorch model
        kmeans_data = torch.load(kmeans_path)
        # We only need the centroids (cluster centers) for distance calculation
        self.cluster_centers = kmeans_data["centroids"] # Shape: (500, 768)
        
    def extract_tokens(self, audio_path):
        """
        1. Uploads audio to HF API -> Gets Embeddings
        2. Local K-Means -> Gets Tokens
        """
        
        # --- Step 1: Remote Feature Extraction ---
        # The API expects the file path directly for audio tasks
        response = self.client.feature_extraction(
            audio_path, 
            model=self.model_id
        )
        
        # Response is usually a list of lists (Time, 768). Convert to Tensor.
        # Note: API might return shape (1, Time, 768) or (Time, 768). We ensure it's (Time, 768).
        embeddings = torch.tensor(response)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
            
        # --- Step 2: Local Quantization ---
        # Find nearest cluster center for each time step
        # x: (Time, 768), y: (500, 768)
        x = embeddings
        y = self.cluster_centers
        
        # Euclidean distance trick: |x-y|^2 = x^2 - 2xy + y^2
        dists = (
            (x**2).sum(1, keepdim=True) 
            - 2 * torch.matmul(x, y.t()) 
            + (y**2).sum(1).unsqueeze(0)
        )
        
        # Get index of closest center (The Token!)
        tokens = dists.argmin(dim=1)
        
        return tokens.numpy()