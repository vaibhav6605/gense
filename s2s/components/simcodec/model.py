import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from s2s.components.simcodec.modules import Encoder, Quantizer, Generator


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

        self.encoder  = Encoder(self.h)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)

    # -------------------- checkpoint --------------------

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.encoder.load_state_dict(ckpt['encoder'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        self.generator.load_state_dict(ckpt['generator'])

    def save_ckpt(self, ckpt_path):
        torch.save({
            'encoder':   self.encoder.state_dict(),
            'quantizer': self.quantizer.state_dict(),
            'generator': self.generator.state_dict(),
        }, ckpt_path)

    # -------------------- input fix --------------------

    @staticmethod
    def _fix_input(x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
        return x

    # -------------------- forward (tokens only) --------------------

    def forward(self, x):
        x = self._fix_input(x)
        z = self.encoder(x)
        z = F.avg_pool1d(z, kernel_size=8, stride=4, padding=2)
        _, _, indices = self.quantizer(z)

        if isinstance(indices, list):
            return torch.stack(indices, dim=-1)
        else:
            return indices

    def encode(self, x):
        return self.forward(x)

    # -------------------- decode --------------------

    def decode(self, x):
        if self.quantizer.stage == 1:
            if isinstance(x, torch.Tensor) and x.dim() == 3:
                indices_list = [x[..., g] for g in range(x.shape[-1])]
            elif isinstance(x, list):
                indices_list = x
            else:
                raise ValueError("Invalid Stage 1 input")

            quantized = self.quantizer.embed(indices_list)

        else:
            if not (isinstance(x, torch.Tensor) and x.dim() == 2):
                raise ValueError("Invalid Stage 2 input")

            quantized = self.quantizer.embed(x)

        return self.generator(quantized)

    # -------------------- training forward --------------------

    def forward_train(self, x):
        x = self._fix_input(x)

        # original encoder output (full resolution)
        z = self.encoder(x)

        # 🔥 REAL FIX: stronger temporal reduction
        z_q = F.avg_pool1d(z, kernel_size=8, stride=4, padding=2)

        quantized, vq_loss, indices = self.quantizer(z_q)

        # -------- 🔥 CRITICAL FIX --------
        # upsample back to original length
        quantized = F.interpolate(
            quantized,
            size=z.shape[-1],
            mode='nearest'
        )

        # generator uses correct resolution
        recon = self.generator(quantized)

        if isinstance(indices, list):
            indices = torch.stack(indices, dim=-1)

        return recon, vq_loss, indices

    # -------------------- Stage1 → Stage2 --------------------

    @classmethod
    def build_stage2_from_stage1(cls, config_path, stage1_model, top_n, top_k):
        from components.simcodec.modules import SingleQuantizer

        assert stage1_model.quantizer.stage == 1

        new_codebook = stage1_model.quantizer.get_reorganized_codebook(
            top_n, top_k
        ).cpu()

        stage2_model = cls(config_path)
        assert stage2_model.quantizer.stage == 2

        with torch.no_grad():
            stage2_model.quantizer._q.module.embedding.weight.copy_(new_codebook)

        stage2_model.encoder.load_state_dict(stage1_model.encoder.state_dict())
        stage2_model.generator.load_state_dict(stage1_model.generator.state_dict())

        print(f"[SimCodec] Stage 2 ready: {top_n * top_k} codes")
        return stage2_model