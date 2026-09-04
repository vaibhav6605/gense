# N2S (Noisy-to-Semantic) GPT-2 Training

This folder contains a minimal, self-contained training pipeline for a GPT-2 style denoiser that maps
noisy semantic tokens to clean semantic tokens.

## Expected data format
Put your paired token files in `n2s_train/DATA` (default) or pass another folder via `--data_dir`.

Files must be named like:
- `example_noisy.npy`
- `example_clean.npy`

The loader will match `*_noisy.npy` with `*_clean.npy` of the same base name.

Each `.npy` file should be a 1D array of integer tokens in `[0, n_units-1]`.

## Model behavior
We train an autoregressive GPT-2 style model:
- Input: noisy tokens + `BOS` + (clean tokens shifted by 1)
- Labels: only the clean tokens (no loss on noisy or BOS positions)

Special tokens:
- PAD = 0
- BOS = 1
- EOS = 2
- shift_num = 3 (all semantic tokens are offset by +3)

## Example run
```
python train.py --out_dir .\checkpoints --n_units 500 --max_seq_len 2048
```

## Notes
- Use `--val_ratio` to hold out a small validation split.
- Use `--max_seq_len` to limit memory usage. Long token streams are chunked.
- The model checkpoints are saved in `--out_dir`.
