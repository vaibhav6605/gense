import torch
from components.model import SimCodec

# -----------------------------
# 1. Initialize model
# -----------------------------
config_path = "config.json"   # path to your config
model = SimCodec(config_path)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# -----------------------------
# 2. Dummy input (2 samples, 1 channel, 16000 samples)
# -----------------------------
x = torch.randn(2, 1, 16000).to(device)

# -----------------------------
# 3. Forward pass (encode → tokens)
# -----------------------------
with torch.no_grad():
    tokens = model(x)

print("Token shape:", tokens.shape)

# -----------------------------
# 4. Decode (tokens → waveform)
# -----------------------------
with torch.no_grad():
    x_hat = model.decode(tokens)

print("Reconstructed shape:", x_hat.shape)

# -----------------------------
# 5. Sanity checks
# -----------------------------
print("\nSanity Checks:")
print("Input min/max:", x.min().item(), x.max().item())
print("Output min/max:", x_hat.min().item(), x_hat.max().item())

# -----------------------------
# 6. Optional: check token stats
# -----------------------------
print("\nToken stats:")
print("Unique tokens:", torch.unique(tokens).numel())
print("Token dtype:", tokens.dtype)