import os
import numpy as np
from torch.utils.data import Dataset


def find_pairs(data_dir):
    pairs = []
    for name in os.listdir(data_dir):
        if name.endswith("_noisy.npy"):
            clean_name = name.replace("_noisy.npy", "_clean.npy")
            noisy_path = os.path.join(data_dir, name)
            clean_path = os.path.join(data_dir, clean_name)
            if os.path.exists(clean_path):
                pairs.append((noisy_path, clean_path))
    return sorted(pairs)


class N2SDataset(Dataset):
    def __init__(self, data_dir, n_units=500, max_seq_len=2048, shift_num=3):
        self.data_dir = data_dir
        self.n_units = n_units
        self.max_seq_len = max_seq_len
        self.shift_num = shift_num
        self.bos = 1
        self.eos = 2
        self.pad = 0

        self.pairs = find_pairs(data_dir)
        if not self.pairs:
            raise ValueError("No *_noisy.npy / *_clean.npy pairs found in data_dir.")

        self.index = []
        for noisy_path, clean_path in self.pairs:
            noisy = np.load(noisy_path)
            clean = np.load(clean_path)
            n = min(len(noisy), len(clean))
            if n == 0:
                continue
            noisy = noisy[:n]
            clean = clean[:n]

            # chunk into max_seq_len (we need room for BOS and labels)
            # input length = noisy_len + 1 + (clean_len - 1)
            # we keep noisy_len == clean_len
            chunk_len = max_seq_len // 2
            if chunk_len < 16:
                chunk_len = 16

            for start in range(0, n, chunk_len):
                end = min(start + chunk_len, n)
                self.index.append((noisy_path, clean_path, start, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        noisy_path, clean_path, start, end = self.index[idx]
        noisy = np.load(noisy_path)[start:end]
        clean = np.load(clean_path)[start:end]

        n = min(len(noisy), len(clean))
        noisy = noisy[:n]
        clean = clean[:n]

        noisy = noisy.astype(np.int64) + self.shift_num
        clean = clean.astype(np.int64) + self.shift_num

        # Build input and labels
        # input: noisy + BOS + clean[:-1]
        # labels: ignore noisy + BOS, predict clean
        if n == 0:
            return {
                "input_ids": np.array([self.pad], dtype=np.int64),
                "labels": np.array([-100], dtype=np.int64),
            }

        input_ids = np.concatenate([
            noisy,
            np.array([self.bos], dtype=np.int64),
            clean[:-1] if n > 1 else np.array([], dtype=np.int64)
        ])

        labels = np.concatenate([
            np.full(len(noisy) + 1, -100, dtype=np.int64),
            clean
        ])

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_batch(batch, pad_id=0):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for x in batch:
        ids = x["input_ids"]
        lab = x["labels"]
        pad_len = max_len - len(ids)
        input_ids.append(np.pad(ids, (0, pad_len), constant_values=pad_id))
        labels.append(np.pad(lab, (0, pad_len), constant_values=-100))
        attention_mask.append(np.pad(np.ones(len(ids), dtype=np.int64), (0, pad_len), constant_values=0))

    return {
        "input_ids": np.stack(input_ids),
        "labels": np.stack(labels),
        "attention_mask": np.stack(attention_mask),
    }
