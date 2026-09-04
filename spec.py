import matplotlib.pyplot as plt
import numpy as np

# Models
models = ["RVQ", "SimCodec"]

# PESQ
pesq = [1.24, 1.63]

# DNSMOS (SIG, BAK, OVRL)
sig = [1.03, 2.74]
bak = [1.92, 2.50]
ovrl = [1.31, 2.09]

x = np.arange(len(models))

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(10,4))

# ---------------- PESQ ----------------
axes[0].bar(models, pesq)
axes[0].set_title("PESQ Comparison")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 3)
axes[0].grid(axis='y')

# ---------------- DNSMOS ----------------
width = 0.25

axes[1].bar(x - width, sig, width, label="SIG")
axes[1].bar(x, bak, width, label="BAK")
axes[1].bar(x + width, ovrl, width, label="OVRL")

axes[1].set_xticks(x)
axes[1].set_xticklabels(models)
axes[1].set_title("DNSMOS Comparison")
axes[1].set_ylim(0, 3)
axes[1].legend()
axes[1].grid(axis='y')

# Layout
plt.tight_layout()
plt.savefig("metrics_bar.png", dpi=300)
plt.show()