import argparse
import math
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW

from dataset import N2SDataset, collate_batch
from model import build_gpt2
from utils import set_seed, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--n_units", type=int, default=500)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.data_dir = os.path.join(script_dir, "DATA")

    dataset = N2SDataset(
        data_dir=args.data_dir,
        n_units=args.n_units,
        max_seq_len=args.max_seq_len,
    )

    if args.val_ratio > 0:
        val_len = int(len(dataset) * args.val_ratio)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
    else:
        train_set, val_set = dataset, None

    def collate_fn(batch):
        return collate_batch(batch, pad_id=0)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_set is not None and len(val_set) > 0:
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

    model = build_gpt2(
        n_units=args.n_units,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_positions=args.max_seq_len,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device == "cuda")

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16 and device == "cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.grad_accum
            step += 1

            if args.save_every > 0 and step % args.save_every == 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step_{step}")
                model.save_pretrained(ckpt_path)

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(len(train_loader), 1)
        ppl = math.exp(min(avg_loss, 20))
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, ppl={ppl:.2f}, time={elapsed:.1f}s")

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device, args.fp16)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"  Val: loss={val_loss:.4f}, ppl={val_ppl:.2f}")

        ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch}")
        model.save_pretrained(ckpt_path)

    final_path = os.path.join(args.out_dir, "final")
    model.save_pretrained(final_path)
    print(f"Saved final model to: {final_path}")


def evaluate(model, loader, device, fp16):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)

            with torch.cuda.amp.autocast(enabled=fp16 and device == "cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


if __name__ == "__main__":
    main()
