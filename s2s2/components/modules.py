import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import Conv1d, ConvTranspose1d

LRELU_SLOPE = 0.1
alpha = 1.0

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers
        self.activations = nn.ModuleList([nn.LeakyReLU(LRELU_SLOPE) for _ in range(self.num_layers)])


    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2,a1,a2 in zip(self.convs1, self.convs2,acts1,acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Encoder(torch.nn.Module):
    def __init__(self, h):
        super(Encoder, self).__init__()
        self.n_filters = h.en_filters
        self.vq_dim = h.vq_dim
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upsample_initial_channel = self.n_filters * ( 2**self.num_upsamples )
        self.conv_pre = weight_norm(Conv1d(h.channel, self.n_filters, 7, 1, padding=3))
        self.normalize = nn.ModuleList()
        resblock = ResBlock1 

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(list(reversed(list(zip(h.upsample_rates, h.upsample_kernel_sizes))))):
            self.ups.append(weight_norm(
                Conv1d(self.n_filters*(2**i), self.n_filters*(2**(i+1)),
                       k, u,
                       padding=((k-u)//2)
                )))
        self.resblocks = nn.ModuleList()
        ch = 1
        for i in range(len(self.ups)):
            ch = self.n_filters*(2**(i+1))
            for j, (k, d) in enumerate(
                    zip(
                        list(reversed(h.resblock_kernel_sizes)),
                        list(reversed(h.resblock_dilation_sizes))
                    )
            ):
                self.resblocks.append(resblock(h, ch, k, d))
                self.normalize.append(torch.nn.LayerNorm([ch],eps=1e-6,elementwise_affine=True))
        
        self.activation_post = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_post = Conv1d(ch, self.vq_dim, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                    xs = self.normalize[i*self.num_kernels+j](xs.transpose(1,2)).transpose(1,2)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
                    xs = self.normalize[i*self.num_kernels+j](xs.transpose(1,2)).transpose(1,2)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)

class Quantizer_module(nn.Module):
    def __init__(self, n_e, e_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        # x: (N, D)

        # Compute L2 distance
        d = torch.sum(x ** 2, dim=1, keepdim=True) \
            + torch.sum(self.embedding.weight ** 2, dim=1) \
            - 2 * torch.matmul(x, self.embedding.weight.T)

        indices = torch.argmin(d, dim=1)
        z_q = self.embedding(indices)

        # VQ losses
        codebook_loss = torch.mean((z_q - x.detach()) ** 2)
        commitment_loss = torch.mean((z_q.detach() - x) ** 2)

        loss = codebook_loss + 0.25 * commitment_loss

        return z_q, indices, loss

class Quantizer(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.vq_dim = h.vq_dim
        self.n_q = h.n_q

        # ONE quantizer per RVQ layer
        self.quantizers = nn.ModuleList([
            Quantizer_module(h.n_codes, self.vq_dim)
            for _ in range(self.n_q)
        ])

    def forward(self, x):
        # x: (B, C, T)

        residual = x
        quantized_out = 0.0

        all_losses = []
        all_indices = []

        for i in range(self.n_q):
            # flatten
            B, C, T = residual.shape
            flat = residual.permute(0, 2, 1).reshape(-1, C)

            z_q, indices, loss = self.quantizers[i](flat)

            # reshape back
            z_q = z_q.reshape(B, T, C).permute(0, 2, 1)

            # straight-through estimator
            z_q_st = residual + (z_q - residual).detach()

            # residual update
            residual = residual - z_q.detach()

            quantized_out = quantized_out + z_q_st

            all_losses.append(loss)
            all_indices.append(indices)

        total_loss = torch.stack(all_losses).mean()

        return quantized_out, total_loss, all_indices

    def embed(self, x):
        # x: (B, T, n_q)
        # reconstruct from indices

        quantized_out = 0.0

        for i in range(self.n_q):
            indices = x[:, :, i]  # (B, T)
            embed = self.quantizers[i].embedding(indices)
            embed = embed.permute(0, 2, 1)

            quantized_out = quantized_out + embed

        return quantized_out


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.n_filters = h.de_filters
        self.vq_dim = h.vq_dim
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upsample_initial_channel = self.n_filters * ( 2**self.num_upsamples )
        self.conv_pre = weight_norm(Conv1d(self.vq_dim, self.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1
        

        self.norm = nn.Identity()

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    self.upsample_initial_channel//(2**i), self.upsample_initial_channel//(2**(i+1)),
                                k, u,
                                padding=(k - u )//2,
                )
            ))
        ch = 1
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = self.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        
        
        self.activation_post = nn.LeakyReLU(LRELU_SLOPE)
        self.conv_post = weight_norm(Conv1d(ch, h.channel, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv_pre(x)
        
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)