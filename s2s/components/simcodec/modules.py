import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import Conv1d, ConvTranspose1d

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------

class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                               dilation=1, padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)
        num_layers = len(self.convs1) + len(self.convs2)
        self.activations = nn.ModuleList(
            [nn.LeakyReLU(LRELU_SLOPE) for _ in range(num_layers)]
        )

    def forward(self, x):
        acts1 = self.activations[::2]
        acts2 = self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x); xt = c1(xt); xt = a2(xt); xt = c2(xt)
            x  = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1: remove_weight_norm(l)
        for l in self.convs2: remove_weight_norm(l)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.n_filters     = h.en_filters
        self.vq_dim        = h.vq_dim
        self.num_kernels   = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = weight_norm(Conv1d(h.channel, self.n_filters, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            reversed(list(zip(h.upsample_rates, h.upsample_kernel_sizes)))
        ):
            self.ups.append(weight_norm(
                Conv1d(self.n_filters * (2**i), self.n_filters * (2**(i+1)),
                       k, u, padding=(k - u) // 2)
            ))

        self.resblocks = nn.ModuleList()
        self.normalize  = nn.ModuleList()
        ch = self.n_filters
        for i in range(len(self.ups)):
            ch = self.n_filters * (2 ** (i + 1))
            for k, d in zip(list(reversed(h.resblock_kernel_sizes)),
                            list(reversed(h.resblock_dilation_sizes))):
                self.resblocks.append(ResBlock1(h, ch, k, d))
                self.normalize.append(
                    nn.LayerNorm([ch], eps=1e-6, elementwise_affine=True)
                )

        self.activation_post = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_post       = Conv1d(ch, self.vq_dim, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            stride     = self.ups[i].stride[0]
            pad_needed = (stride - x.shape[-1] % stride) % stride
            if pad_needed:
                x = F.pad(x, (0, pad_needed))
            x  = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                bo = self.resblocks[i * self.num_kernels + j](x)
                bo = self.normalize[i * self.num_kernels + j](
                    bo.transpose(1, 2)
                ).transpose(1, 2)
                xs = bo if xs is None else xs + bo
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        for l in self.ups:       remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)


# ---------------------------------------------------------------------------
# Quantizer Module  (single codebook)
#
# Why these specific values:
#
# n_codes = 500  (set in config)
#   With batch=4, segment=2s, stride=320: tokens/step = 400.
#   400/500 = 80% coverage per step by random chance alone.
#   With 500 codes, the EMA needs time to distinguish genuinely popular
#   entries from randomly touched ones. 91 codes would be trivially full.
#
# ema_decay = 0.999
#   Half-life ≈ 693 steps. An entry needs ~700 consecutive zero-use steps
#   before its EMA count drops to dead_threshold=0.05.
#   (Previously 0.99 → half-life 69 steps → far too aggressive)
#
# dead_threshold = 0.05
#   An entry is "dead" only if it has been essentially unused for ~700 steps.
#   (Previously 0.5 → entries marked dead after only ~70 steps)
#
# reset_delay = 5000
#   No resets for the first 5000 steps. Before this, the encoder is random
#   and reinitialising to encoder outputs is meaningless — it just causes
#   all entries to look uniformly used without any real learning.
#
# Unit-sphere initialisation
#   Embeddings initialised as random unit vectors. Since encoder outputs are
#   also L2-normalised (norm=1), distances are in [0,4] from step 1.
#   Previously uniform(±0.002) vs encoder norm~1 → all distances ≈1.0.
#
# No diversity loss
#   The 500×500 gram-matrix cross-entropy is expensive (125k elements per
#   forward pass). Unit-sphere init already spreads embeddings uniformly.
#   Removed to save compute without hurting codebook quality.
# ---------------------------------------------------------------------------

class Quantizer_module(nn.Module):

    def __init__(self, n_e, e_dim,
                 ema_decay=0.999, dead_threshold=0.05, reset_delay=5000):
        super().__init__()
        self.n_e            = n_e
        self.e_dim          = e_dim
        self.ema_decay      = ema_decay
        self.dead_threshold = dead_threshold
        self.reset_delay    = reset_delay

        self.embedding = nn.Embedding(n_e, e_dim)
        with torch.no_grad():
            # Unit-sphere init: all entries start at norm=1,
            # matching L2-normalised encoder outputs
            w = torch.randn(n_e, e_dim)
            self.embedding.weight.data.copy_(F.normalize(w, p=2, dim=-1))

        self.register_buffer('target',      torch.arange(n_e))
        self.register_buffer('ema_count',   torch.ones(n_e))
        self.register_buffer('_step_count', torch.zeros(1, dtype=torch.long))

    def forward(self, x, apply_diversity_loss=False):
        """
        x : (N, e_dim)
        returns z_q (N, e_dim), min_indices (N,), zero_loss (scalar)
        """
        d = (
            torch.sum(x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(x, self.embedding.weight.T)
        )
        min_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_indices)

        frozen = getattr(self, "frozen", False)

        if self.training and not frozen:
            with torch.no_grad():
                self._step_count.add_(1)
                step = int(self._step_count.item())

                one_hot = torch.zeros(self.n_e, device=x.device)
                one_hot.scatter_add_(
                    0,
                    min_indices,
                    torch.ones(min_indices.shape[0], device=x.device)
                )
                self.ema_count.mul_(self.ema_decay).add_(
                    one_hot * (1.0 - self.ema_decay)
                )

                if step >= self.reset_delay:
                    dead_mask = self.ema_count < self.dead_threshold
                    n_dead = int(dead_mask.sum().item())

                    if n_dead > 0:
                        n_avail = x.shape[0]
                        if n_avail >= n_dead:
                            idx = torch.randperm(n_avail, device=x.device)[:n_dead]
                            self.embedding.weight[dead_mask] = x[idx].detach()
                        else:
                            reps = (n_dead + n_avail - 1) // n_avail
                            pool = x.detach().repeat(reps, 1)[:n_dead]
                            self.embedding.weight[dead_mask] = pool

                        self.ema_count[dead_mask] = 1.0

        return z_q, min_indices, torch.tensor(0.0, device=x.device)



# ---------------------------------------------------------------------------
# Stage 1 — Group Quantizer
#
# Splits vq_dim into n_code_groups equal chunks.
# Each chunk quantised independently (group VQ, not residual).
# Both groups equally informative → fair top-N/K selection for Stage 2.
#
# n_codes=500 per group:
#   Stage 1 capacity : 2 × 500 = 1000 entries
#   After training   : top-91 from G0, top-90 from G1 by usage
#   Stage 2 codebook : 91 × 90 = 8190 entries
# ---------------------------------------------------------------------------

class GroupQuantizer(nn.Module):

    def __init__(self, h):
        super().__init__()
        assert h.vq_dim % h.n_code_groups == 0
        self.vq_dim = h.vq_dim
        self.n_code_groups = h.n_code_groups
        self.chunk_dim = h.vq_dim // h.n_code_groups
        self.n_codes = h.n_codes

        self.codebook_loss_lambda = h.codebook_loss_lambda
        self.commitment_loss_lambda = h.commitment_loss_lambda

        self.quantizer_modules = nn.ModuleList([
            Quantizer_module(h.n_codes, self.chunk_dim)
            for _ in range(self.n_code_groups)
        ])

        for g in range(self.n_code_groups):
            self.register_buffer(
                f'usage_count_{g}',
                torch.zeros(h.n_codes, dtype=torch.long)
            )

    def _get_usage(self, g):
        return getattr(self, f'usage_count_{g}')

    def _update_usage(self, g, indices):
        self._get_usage(g).scatter_add_(
            0,
            indices.view(-1),
            torch.ones_like(indices.view(-1))
        )

    def reset_usage_counts(self):
        for g in range(self.n_code_groups):
            self._get_usage(g).zero_()

    def set_frozen(self, frozen=True):
        self.frozen = frozen
        for module in self.quantizer_modules:
            module.frozen = frozen

    def forward(self, xin):
        """
        xin : (B, vq_dim, T)
        returns quantized (B, vq_dim, T), loss scalar,
                indices list[n_groups x (B, T)]
        """
        frozen = getattr(self, "frozen", False)

        x = xin.transpose(1, 2)
        chunks = torch.split(x, self.chunk_dim, dim=-1)

        z_q_parts = []
        all_indices = []
        total_loss = torch.tensor(0.0, device=xin.device)

        for g, (chunk, module) in enumerate(zip(chunks, self.quantizer_modules)):
            B, T, C = chunk.shape
            flat = chunk.reshape(-1, C)

            flat_norm = F.normalize(flat, p=2, dim=-1)
            flat_norm = flat_norm * 0.5

            z_q_flat, indices, _ = module(flat_norm, apply_diversity_loss=False)

            if not frozen:
                self._update_usage(g, indices)

            z_q_norm = z_q_flat.reshape(B, T, C)
            flat_n2d = flat_norm.reshape(B, T, C)

            codebook_loss = F.mse_loss(z_q_norm, flat_n2d.detach())
            commitment_loss = F.mse_loss(z_q_norm.detach(), flat_n2d)

            quant_error = (z_q_norm - flat_n2d).detach()
            z_q_st = chunk + quant_error

            total_loss = total_loss + (
                self.codebook_loss_lambda * codebook_loss
                + self.commitment_loss_lambda * commitment_loss
            )

            z_q_parts.append(z_q_st)
            all_indices.append(indices.reshape(B, T))

        quantized = torch.cat(z_q_parts, dim=-1).transpose(1, 2)
        loss = total_loss / self.n_code_groups

        if frozen:
            return quantized.detach(), loss.detach(), all_indices

        return quantized, loss, all_indices

    def forward_frozen(self, xin):
        self.set_frozen(True)
        return self.forward(xin)

    def embed(self, indices_list):
        parts = [
            m.embedding(idx)
            for m, idx in zip(self.quantizer_modules, indices_list)
        ]
        return torch.cat(parts, dim=-1).transpose(1, 2)

    @torch.no_grad()
    def get_reorganized_codebook(self, top_n, top_k):
        assert self.n_code_groups == 2

        _, idx0 = torch.topk(self._get_usage(0).float(), top_n)
        _, idx1 = torch.topk(self._get_usage(1).float(), top_k)

        emb0 = self.quantizer_modules[0].embedding.weight[idx0]
        emb1 = self.quantizer_modules[1].embedding.weight[idx1]

        emb0_rep = emb0.unsqueeze(1).expand(-1, top_k, -1)
        emb1_rep = emb1.unsqueeze(0).expand(top_n, -1, -1)

        new_cb = torch.cat([emb0_rep, emb1_rep], dim=-1)
        return new_cb.reshape(top_n * top_k, self.vq_dim)



# ---------------------------------------------------------------------------
# Stage 2 — Single Large Quantizer (8190 entries)
# ---------------------------------------------------------------------------

class SingleQuantizer(nn.Module):

    def __init__(self, h, init_codebook=None):
        super().__init__()
        self.vq_dim = h.vq_dim
        n_s2        = h.top_n * h.top_k

        self.codebook_loss_lambda   = h.codebook_loss_lambda
        self.commitment_loss_lambda = h.commitment_loss_lambda

        self.module = Quantizer_module(n_s2, self.vq_dim)

        if init_codebook is not None:
            assert init_codebook.shape == (n_s2, self.vq_dim)
            with torch.no_grad():
                # Normalise injected codebook onto unit sphere
                self.module.embedding.weight.copy_(
                    F.normalize(init_codebook, p=2, dim=-1)
                )

    def forward(self, xin):
        x = xin.transpose(1, 2)
        B, T, C = x.shape
        flat = x.reshape(-1, C)

        flat_norm = F.normalize(flat, p=2, dim=-1)
        z_q_flat, indices, _ = self.module(flat_norm, apply_diversity_loss=False)

        z_q_norm = z_q_flat.reshape(B, T, C)
        flat_n2d = flat_norm.reshape(B, T, C)

        codebook_loss = F.mse_loss(z_q_norm, flat_n2d.detach())
        commitment_loss = F.mse_loss(z_q_norm.detach(), flat_n2d)

        loss = (
            self.codebook_loss_lambda * codebook_loss
            + self.commitment_loss_lambda * commitment_loss
        )

        quant_error = (z_q_norm - flat_n2d).detach()
        quantized = (x + quant_error).transpose(1, 2)
        indices_2d = indices.reshape(B, T)

        if getattr(self, "frozen", False):
            return quantized.detach(), loss.detach(), indices_2d

        return quantized, loss, indices_2d

    def forward_frozen(self, xin):
        self.frozen = True
        self.module.frozen = True
        return self.forward(xin)


    def embed(self, indices):
        return self.module.embedding(indices).transpose(1, 2)


# ---------------------------------------------------------------------------
# Unified Quantizer wrapper
# ---------------------------------------------------------------------------

class Quantizer(nn.Module):

    def __init__(self, h, init_codebook=None):
        super().__init__()
        stage = getattr(h, 'stage', 1)

        if stage == 1:
            self._q = GroupQuantizer(h)
        else:
            self._q = SingleQuantizer(h, init_codebook=init_codebook)

        self.stage = stage
        self.frozen = False

    def set_frozen(self, frozen=True):
        self.frozen = frozen

        if hasattr(self._q, "set_frozen"):
            self._q.set_frozen(frozen)
        else:
            self._q.frozen = frozen
            if hasattr(self._q, "module"):
                self._q.module.frozen = frozen

    def forward(self, xin):
        if getattr(self, "frozen", False):
            return self._q.forward_frozen(xin)

        return self._q(xin)

    def embed(self, x):
        return self._q.embed(x)

    @property
    def group_quantizer(self):
        assert self.stage == 1
        return self._q

    def get_reorganized_codebook(self, top_n, top_k):
        assert self.stage == 1
        return self._q.get_reorganized_codebook(top_n, top_k)

    def reset_usage_counts(self):
        assert self.stage == 1
        self._q.reset_usage_counts()



# ---------------------------------------------------------------------------
# Generator  (Decoder)
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h             = h
        self.n_filters     = h.de_filters
        self.vq_dim        = h.vq_dim
        self.num_kernels   = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upsample_initial_channel = self.n_filters * (2 ** self.num_upsamples)

        self.conv_pre = weight_norm(
            Conv1d(self.vq_dim, self.upsample_initial_channel, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    self.upsample_initial_channel // (2**i),
                    self.upsample_initial_channel // (2**(i+1)),
                    k, u, padding=(k - u) // 2,
                )
            ))

        self.resblocks = nn.ModuleList()
        ch = self.upsample_initial_channel
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(h, ch, k, d))

        self.activation_post = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_post       = weight_norm(Conv1d(ch, h.channel, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x  = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                b  = self.resblocks[i * self.num_kernels + j](x)
                xs = b if xs is None else xs + b
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for l in self.ups:       remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)