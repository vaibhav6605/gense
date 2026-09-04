import os
import torch

def create_dummy_model():
    print("Generating dummy K-Means model...")
    
    # 1. Define the correct path
    target_dir = "models"
    filename = "kmeans_hubert_base_l9_c500.pt"
    save_path = os.path.join(target_dir, filename)
    
    os.makedirs(target_dir, exist_ok=True)
    
    # 2. Create random centroids (500 clusters, 768 dimensions)
    # This matches the exact shape of the real HuBERT K-Means model
    dummy_centroids = torch.randn(500, 768)
    
    # 3. Save it in the format your tokenizer expects
    torch.save(
        {"centroids": dummy_centroids}, 
        save_path
    )
    
    print(f"\n[SUCCESS] Dummy model created at: {save_path}")
    print("You can now run main.py!")

if __name__ == "__main__":
    create_dummy_model()