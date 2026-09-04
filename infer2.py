"""
inference.py -- Direct acoustic-token enhancement
=================================================

Pipeline:

    Noisy WAV
        -> SimCodec Encoder
        -> noisy_ac
        -> AC2AC/S2S acoustic LM: [noisy_ac | BOS] -> clean_ac
        -> SimCodec Decoder
        -> Clean WAV

There is no N2S model, no HuBERT, and no k-means in this inference path.

Usage (single file):
    python infer2.py 
        --input noisy.wav 
        --output enhanced1.wav 
        --codec_ckpt s2s/checkpoints/stage2/best.pt 
        --codec_config s2s/config_stage2.json 
        --s2s_ckpt s2s/checkpoints/s2sh/best.pt

Usage (folder of WAV files):
    python inference.py ^
        --input_dir noisy_wavs ^
        --output_dir enhanced_wavs ^
        --codec_ckpt checkpoints/stage2/best.pt ^
        --codec_config config_stage2.json ^
        --s2s_ckpt checkpoints/ac2ac/best.pt

Notes:
  --s2s_ckpt      is the direct acoustic LM checkpoint trained by
                  ac2ac_lm_train.py. --ac2ac_ckpt is accepted as an alias.
  --acoustic_num  defaults to top_n * top_k from codec_config.
  --max_ac_len    defaults to segment_len / product(upsample_rates) from
                  codec_config. Long files are processed in chunks of this
                  many acoustic tokens to match training.
"""

import argparse
import glob
import json
import os
import sys
from math import gcd, prod
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from s2s.components.simcodec.model import SimCodec


# ===========================================================================
# Audio/config helpers
# ===========================================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def acoustic_num_from_config(config):
    if "acoustic_num" in config:
        return int(config["acoustic_num"])
    if "top_n" in config and "top_k" in config:
        return int(config["top_n"]) * int(config["top_k"])
    raise ValueError("Config must contain either acoustic_num or top_n/top_k.")


def max_ac_len_from_config(config):
    if "segment_len" not in config or "upsample_rates" not in config:
        return None
    hop = prod(int(x) for x in config["upsample_rates"])
    return max(1, int(config["segment_len"]) // hop)


def load_audio_mono(path, target_sr=16_000):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data.astype(np.float32)


def save_audio(path, wav_np, sr=16_000):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, wav_np.astype(np.float32), sr)


# ===========================================================================
# Direct acoustic LM
# ===========================================================================

class AC2ACModel(nn.Module):
    """
    GPT2 causal LM matching ac2ac_lm_train.py:

      prompt: [noisy_ac | BOS]
      output: clean_ac
    """

    PAD = 0
    BOS = 1
    EOS = 2
    SHIFT = 3

    def __init__(
        self,
        acoustic_num,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        n_positions=2048,
    ):
        super().__init__()
        self.acoustic_num = int(acoustic_num)
        vocab_size = self.SHIFT + self.acoustic_num
        cfg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            activation_function="gelu_new",
            n_positions=n_positions,
            n_ctx=n_positions,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=self.BOS,
            eos_token_id=self.EOS,
            pad_token_id=self.PAD,
        )
        self.lm = GPT2LMHeadModel(cfg)

    def _build_prompt(self, noisy_ac):
        B, T_ac = noisy_ac.shape
        dev = noisy_ac.device

        tokens = noisy_ac + self.SHIFT
        bos = torch.full((B, 1), self.BOS, dtype=torch.long, device=dev)
        tokens = torch.cat([tokens, bos], dim=1)

        ac_pos = torch.arange(T_ac, device=dev)
        bos_pos = torch.tensor([T_ac], device=dev)
        pos = torch.cat([ac_pos, bos_pos]).unsqueeze(0).expand(B, -1)
        return tokens, pos

    @torch.no_grad()
    def inference(self, noisy_ac, temperature=1.0, show_progress=True):
        B, T_ac = noisy_ac.shape
        dev = noisy_ac.device

        token_gen, pos_gen = self._build_prompt(noisy_ac)
        generated = []

        iterator = range(T_ac)
        if show_progress:
            iterator = tqdm(iterator, desc="AC2AC generating", leave=False)

        for step in iterator:
            out = self.lm(input_ids=token_gen, position_ids=pos_gen)
            last_logits = out.logits[:, -1, :].clone()

            last_logits[:, :self.SHIFT] = -1e9
            last_logits[:, self.SHIFT + self.acoustic_num:] = -1e9

            if temperature != 1.0:
                last_logits = last_logits / max(temperature, 1e-5)

            probs = last_logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token)

            next_pos = torch.full((B, 1), step, dtype=torch.long, device=dev)
            token_gen = torch.cat([token_gen, next_token], dim=1)
            pos_gen = torch.cat([pos_gen, next_pos], dim=1)

        return torch.cat(generated, dim=1) - self.SHIFT


# ===========================================================================
# Model loading
# ===========================================================================

def load_simcodec(config_path, ckpt_path, device):
    print(f"[Load] SimCodec  <- {ckpt_path}")
    codec = SimCodec(config_path).to(device)
    codec.load_ckpt(ckpt_path)
    codec.eval()
    return codec


def load_ac2ac(
    ckpt_path,
    acoustic_num,
    hidden_size,
    num_layers,
    num_heads,
    n_positions,
    device,
):
    print(f"[Load] AC2AC LM  <- {ckpt_path}")
    model = AC2ACModel(
        acoustic_num=acoustic_num,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        n_positions=n_positions,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


# ===========================================================================
# Enhancement
# ===========================================================================

def normalize_acoustic_tokens(tokens):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    elif tokens.dim() == 3:
        tokens = tokens.reshape(tokens.shape[0], -1)
    if tokens.dim() != 2:
        raise ValueError(f"Expected codec tokens with shape (B,T), got {tuple(tokens.shape)}")
    return tokens.long()


def validate_token_range(tokens, acoustic_num):
    if tokens.numel() == 0:
        raise ValueError("SimCodec encoder returned zero acoustic tokens.")
    min_id = int(tokens.min().item())
    max_id = int(tokens.max().item())
    if min_id < 0 or max_id >= acoustic_num:
        raise ValueError(
            f"Acoustic token ids out of range for acoustic_num={acoustic_num}: "
            f"min={min_id}, max={max_id}"
        )


@torch.no_grad()
def generate_clean_ac(ac2ac_model, noisy_ac, max_ac_len, temperature):
    chunks = []
    total = noisy_ac.shape[1]

    for start in tqdm(range(0, total, max_ac_len), desc="Token chunks", leave=False):
        chunk = noisy_ac[:, start:start + max_ac_len]
        clean_chunk = ac2ac_model.inference(
            chunk,
            temperature=temperature,
            show_progress=False,
        )
        chunks.append(clean_chunk)

    return torch.cat(chunks, dim=1)


def enhance_file(wav_np, codec, ac2ac_model, args, device):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0).to(device)

    print("  Encoding noisy waveform -> noisy_ac")
    with torch.no_grad():
        noisy_ac = codec.encode(wav_t)
    noisy_ac = normalize_acoustic_tokens(noisy_ac).to(device)
    validate_token_range(noisy_ac, args.acoustic_num)

    print(f"  AC2AC: {noisy_ac.shape[1]} acoustic tokens -> clean_ac")
    clean_ac = generate_clean_ac(
        ac2ac_model=ac2ac_model,
        noisy_ac=noisy_ac,
        max_ac_len=args.max_ac_len,
        temperature=args.temperature,
    )

    print("  Decoding clean_ac -> waveform")
    with torch.no_grad():
        enhanced = codec.decode(clean_ac)

    return enhanced.squeeze().detach().cpu().numpy().astype(np.float32)


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Direct AC2AC inference: noisy WAV -> clean WAV"
    )

    parser.add_argument("--input", type=str, default=None, help="Single noisy WAV")
    parser.add_argument("--output", type=str, default=None, help="Output WAV path")
    parser.add_argument("--input_dir", type=str, default=None, help="Folder of noisy WAV files")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder for enhanced WAV files")

    parser.add_argument("--codec_ckpt", type=str, required=True)
    parser.add_argument("--codec_config", type=str, required=True)
    parser.add_argument(
        "--s2s_ckpt",
        "--ac2ac_ckpt",
        dest="ac2ac_ckpt",
        type=str,
        required=True,
        help="Direct acoustic LM checkpoint from ac2ac_lm_train.py",
    )

    parser.add_argument("--acoustic_num", type=int, default=None)
    parser.add_argument("--max_ac_len", type=int, default=None)

    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--n_positions", type=int, default=2048)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu", type=str, default="0", help="GPU id, or 'cpu'")
    parser.add_argument("--sample_rate", type=int, default=None)

    args = parser.parse_args()

    config = load_json(args.codec_config)
    if args.acoustic_num is None:
        args.acoustic_num = acoustic_num_from_config(config)
    if args.max_ac_len is None:
        args.max_ac_len = max_ac_len_from_config(config)
    if args.max_ac_len is None:
        args.max_ac_len = 100
    if args.sample_rate is None:
        args.sample_rate = int(config.get("sample_rate", 16_000))

    if args.n_positions < (2 * args.max_ac_len + 1):
        raise ValueError(
            f"--n_positions must be >= {2 * args.max_ac_len + 1} "
            f"for max_ac_len={args.max_ac_len}; got {args.n_positions}."
        )

    return args


def build_file_pairs(args):
    if args.input is None and args.input_dir is None:
        raise ValueError("Provide --input or --input_dir")
    if args.input is not None and args.output is None:
        raise ValueError("--input requires --output")
    if args.input_dir is not None and args.output_dir is None:
        raise ValueError("--input_dir requires --output_dir")

    if args.input is not None:
        return [(args.input, args.output)]

    wav_files = sorted(
        set(
            glob.glob(os.path.join(args.input_dir, "*.wav"))
            + glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True)
        )
    )
    if not wav_files:
        raise RuntimeError(f"No .wav files found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    return [
        (path, os.path.join(args.output_dir, os.path.basename(path)))
        for path in wav_files
    ]


def main():
    args = parse_args()

    if args.gpu.lower() == "cpu":
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[Inference] Device      : {device}")
    print(f"[Inference] acoustic_num: {args.acoustic_num}")
    print(f"[Inference] max_ac_len  : {args.max_ac_len}")
    print(f"[Inference] sample_rate : {args.sample_rate}")

    codec = load_simcodec(args.codec_config, args.codec_ckpt, device)
    ac2ac_model = load_ac2ac(
        ckpt_path=args.ac2ac_ckpt,
        acoustic_num=args.acoustic_num,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        n_positions=args.n_positions,
        device=device,
    )

    file_pairs = build_file_pairs(args)
    print(f"[Inference] Processing {len(file_pairs)} file(s).\n")

    for index, (in_path, out_path) in enumerate(file_pairs, start=1):
        print(f"[{index}/{len(file_pairs)}] {os.path.basename(in_path)}")
        try:
            wav_np = load_audio_mono(in_path, target_sr=args.sample_rate)
            print(f"  Input: {len(wav_np) / args.sample_rate:.2f}s ({len(wav_np)} samples)")

            enhanced = enhance_file(
                wav_np=wav_np,
                codec=codec,
                ac2ac_model=ac2ac_model,
                args=args,
                device=device,
            )

            save_audio(out_path, enhanced, sr=args.sample_rate)
            print(f"  Saved -> {out_path}\n")
        except Exception as exc:
            import traceback

            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            print()

    print("[Inference] Done.")


if __name__ == "__main__":
    main()
