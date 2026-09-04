import torch
# This should NOT throw an error if the file is healthy
data = torch.load(r"n2s\ckpts\kmeans_hubert_base_l9_c500.pt", map_location="cpu", weights_only=False)
print("File is healthy!")