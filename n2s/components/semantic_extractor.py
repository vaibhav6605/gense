import torch
import torch.nn as nn
import joblib
from transformers import HubertModel


# ======================================================================
# KMeans Module
# ======================================================================

class ApplyKmeans(nn.Module):
    def __init__(self, km_path, device='cuda'):
        super().__init__()
        print(f'Loading k-means model from {km_path}')

        self.km_model = joblib.load(km_path)

        n_clusters, feat_dim = self.km_model.cluster_centers_.shape
        print(f'  -> {n_clusters} clusters, feature dim {feat_dim}')

        # Precompute for fast distance calculation
        self.C_np     = self.km_model.cluster_centers_.transpose()   # (D, K)
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)       # (1, K)

        self.C     = torch.from_numpy(self.C_np).float().to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).float().to(device)

        # Optional embedding (not used but kept for compatibility)
        self.emb = nn.Embedding(n_clusters, feat_dim)
        self.emb.weight.data = self.C.transpose(0, 1)
        self.emb.weight.requires_grad = False

    def forward(self, x, b, t):
        # x: (B*T, D)
        if self.C.device != x.device:
            self.C     = self.C.to(x.device)
            self.Cnorm = self.Cnorm.to(x.device)

        # Efficient L2 distance
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        tokens = dist.argmin(dim=-1).reshape(b, t)
        return tokens


# ======================================================================
# HuBERT Feature Extractor (HuggingFace ONLY)
# ======================================================================

class HuBERTSemanticExtractor(nn.Module):
    """
    HuggingFace HuBERT feature extractor.

    Example:
        ckpt_path = "facebook/hubert-base-ls960"

    Output:
        (B, T', D)
    """

    def __init__(self, ckpt_path: str, layer: int = 9, device: str = 'cuda'):
        super().__init__()

        print(f'Loading HuBERT from HuggingFace: {ckpt_path}')
        self.model = HubertModel.from_pretrained(ckpt_path)

        self.model.eval()
        self.model.requires_grad_(False)

        self.layer = layer
        self.device = device
        self.model.to(device)

        print(f'  -> extracting features from layer {layer}')

    @torch.no_grad()
    def forward(self, source: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Args:
            source: (B, T) float32 waveform @ 16kHz
            padding_mask: optional (B, T)

        Returns:
            features: (B, T', D)
        """

        # Ensure float32
        source = source.float()

        attention_mask = None
        if padding_mask is not None:
            attention_mask = (~padding_mask).long()

        outputs = self.model(
            source,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states[0] = CNN output
        # hidden_states[1..12] = transformer layers
        features = outputs.hidden_states[self.layer]

        return features


# ======================================================================
# Factory
# ======================================================================

def get_ssl_model(ckpt_path: str, km_path: str, device: str = 'cuda',
                  type: str = 'hubert', **kwargs):
    """
    Returns:
        ssl_model, km_model

    Config example:
        ckpt_path = "facebook/hubert-base-ls960"
        km_path   = "ckpts/kmeans_hubert_base_l9_c500.pt"
    """

    layer = kwargs.get('layer', 9)

    ssl_model = HuBERTSemanticExtractor(
        ckpt_path=ckpt_path,
        layer=layer,
        device=device
    )

    km_model = ApplyKmeans(km_path, device)

    return ssl_model, km_model


# ======================================================================
# Token Extraction Pipeline
# ======================================================================

@torch.no_grad()
def extract_semantic_tokens(
    ssl_model,
    km_model: ApplyKmeans,
    waveforms: torch.Tensor,
    padding_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Full pipeline:
        waveform -> HuBERT -> k-means -> tokens

    Args:
        waveforms: (B, T)
    Returns:
        tokens: (B, T')
    """

    features = ssl_model(waveforms, padding_mask)  # (B, T', D)

    B, T_prime, D = features.shape
    flat = features.reshape(B * T_prime, D)

    tokens = km_model(flat, B, T_prime)

    return tokens