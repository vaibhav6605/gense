import json
import torch
import torch.nn as nn
from s2s2.components.modules import Encoder, Quantizer, Generator

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class SimCodec(nn.Module):
    def __init__(self, config_path):
        super(SimCodec, self).__init__()
        self.config_path = config_path
        with open(self.config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.encoder = Encoder(self.h)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
    
    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path,map_location='cpu')
        self.encoder.load_state_dict(ckpt['encoder'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        self.generator.load_state_dict(ckpt['generator'])

    def forward(self, x):
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        z = self.encoder(x)

        _, _, indices = self.quantizer(z)

        B, C, T = z.shape

        # reshape indices properly
        tokens = []
        for idx in indices:
            idx = idx.view(B, T)   # (B*T,) → (B, T)
            tokens.append(idx)

        tokens = torch.stack(tokens, dim=-1)  # (B, T, n_q)

        return tokens

    def encode(self, x):
        return self.forward(x)

    def decode(self, x):
        return self.generator(self.quantizer.embed(x))