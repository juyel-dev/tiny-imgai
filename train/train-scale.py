from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from losses import PrintCleanLoss
from model_scalable import ScaledTinyUNet
from train import build_cache, dataset_signature, load_manifest, load_page_tensor

INPUT_SIZE_DEFAULT = 512

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tiny-imgai scalable 512px CPU trainer.")
    train_dir = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--input-size", type=int, default=INPUT_SIZE_DEFAULT)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-pages", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--max-minutes", type=float, default=120)
    p.add_argument("--checkpoint-every-pages", type=int, default=10)
    p.add_argument("--min-free-ram-mib", type=int, default=768)
    p.add_argument("--max-rss-mib", type=int, default=3072)
    p.add_argument("--threads", type=int, default=6)
    return p.parse_args()

def arch_id(input_size: int, base: int, ckpt: bool) -> str:
    suffix = "ckpt" if ckpt else "eager"
    return f"{input_size}-b{base}-{suffix}"

def memory_state(process: psutil.Process, args: argparse.Namespace) -> dict:
    rss = process.memory_info().rss / 1048576
    available = psutil.virtual_memory().available / 1048576
    return {"rss_mib": round(rss, 1), "available_mib": round(available, 1),
            "hard_stop": rss > args.max_rss_mib or available < args.min_free_ram_mib}

def checkpoint_path_for(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).resolve()
    suffix = arch_id(args.input_size, args.base_channels, args.activation_checkpointing)
    return Path(__file__).resolve().parent / "checkpoints" / f"scaled-{suffix}"

def save_checkpoint(path, model, optimizer, epoch, next_page, global_step, last_loss, signature, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch, "next_page_index": next_page, "global_step": global_step,
        "last_loss": last_loss, "dataset_signature": signature,
        "input_size": args.input_size, "base_channels": args.base_channels,
        "activation_checkpointing": args.activation_checkpointing,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "loss_profile": "print-clean-v1",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
    }, tmp)
    tmp.replace(path)

def load_checkpoint(path, model, optimizer, signature, args):
    if not path.exists():
        return 0, 0, 0, float("nan")
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "dataset_signature": signature, "input_size": args.input_size,
        "base_channels": args.base_channels, "activation_checkpointing": args.activation_checkpointing,
        "batch_size": args.batch_size, "loss_profile": "print-clean-v1"
    }
    for key, value in expected.items():
        if state.get(key) != value:
            print(f"Existing checkpoint mismatch on {key}; starting fresh.")
            return 0, 0, 0, float("nan")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return (int(state.get("epoch", 0)), int(state.get("next_page_index", 0)),
            int(state.get("global_step", 0)), float(state.get("last_loss", float("nan"))))

def main() -> int:
    args = parse_args()
    if args.input_size != 512:
        raise ValueError("The scalable trainer currently targets exactly 512x512.")
    if args.batch_size != 1:
        raise ValueError("Scalable laptop training currently uses batch size 1.")
    if args.base_channels not in (16, 24, 32, 40, 48):
        raise ValueError("base_channels must be one of 16, 24, 32, 40, 48.")
    if args.epochs <= 0 or args.max_pages < 0:
        raise ValueError("Invalid epochs/max-pages.")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data_dir = Path(args.data_dir).resolve()
    cache_root = data_dir / "cache512"
    checkpoint_dir = checkpoint_path_for(args)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    loss_log = checkpoint_dir / "loss.csv"
    process = psutil.Process(os.getpid())

    manifest, all_records = load_manifest(data_dir)
    records = all_records[:args.max_pages] if args.max_pages else all_records
    if not records:
        raise ValueError("No PDF pages found.")

    print("tiny-imgai scalable 512px trainer")
    print(f"PyTorch: {torch.__version__}")
    print(f"Input: {args.input_size}x{args.input_size}")
    print(f"U-Net width: {args.base_channels}->{args.base_channels*2}->{args.base_channels*4}->{args.base_channels*8}")
    print(f"Checkpointing: {args.activation_checkpointing}")
    print(f"Threads: {args.threads} | Batch: {args.batch_size}")
    print(f"Pages: {len(records)} | Epochs: {args.epochs}")
    print(f"RAM guard: free >= {args.min_free_ram_mib} MiB, RSS <= {args.max_rss_mib} MiB")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    build_cache(records, cache_root, args.input_size)
    signature = dataset_signature(manifest, records, args.input_size) + hashlib.sha256(
        json.dumps({"base": args.base_channels, "ckpt": args.activation_checkpointing}, sort_keys=True).encode()
    ).hexdigest()

    model = ScaledTinyUNet(args.input_size, args.base_channels, args.activation_checkpointing)
    model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-7)
    criterion = PrintCleanLoss()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")

    epoch, page_index, global_step, last_loss = load_checkpoint(
        checkpoint_path, model, optimizer, signature, args
    )
    if not loss_log.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        loss_log.write_text("epoch,page,global_step,loss,seconds_per_page,timestamp\n", encoding="utf-8")

    deadline = time.monotonic() + max(0.1, args.max_minutes * 60)
    while epoch < args.epochs:
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_started = time.monotonic()
        print(f"\n== epoch {epoch + 1}/{args.epochs} ==")

        while page_index < len(records):
            mem = memory_state(process, args)
            if mem["hard_stop"]:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, page_index, global_step, last_loss, signature, args)
                print(f"RAM guard stopped run: {mem}")
                return 0
            if time.monotonic() >= deadline:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, page_index, global_step, last_loss, signature, args)
                print("Time budget reached. Checkpoint saved; run again to resume.")
                return 0

            record = records[page_index]
            op, yp = record.cache_paths(cache_root)
            started = time.monotonic()
            x = load_page_tensor(op, args.input_size)
            y = load_page_tensor(yp, args.input_size)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu().item())
            elapsed = max(1e-6, time.monotonic() - started)
            last_loss = value
            epoch_loss += value
            epoch_steps += 1
            global_step += 1
            page_index += 1
            with loss_log.open("a", encoding="utf-8") as handle:
                handle.write(f"{epoch},{record.page_number},{global_step},{value:.8f},{elapsed:.4f},{time.strftime("%Y-%m-%dT%H:%M:%S")}\n")
            print(f"page {record.page_number}/{len(records)} loss {value:.5f} step {global_step} {elapsed:.2f}s/page", flush=True)
            del pred, loss, x, y

            if args.checkpoint_every_pages > 0 and page_index < len(records) and page_index % args.checkpoint_every_pages == 0:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, page_index, global_step, last_loss, signature, args)
                print(f"  checkpoint saved at page {page_index}")

        avg_loss = epoch_loss / max(1, epoch_steps)
        print(f"== epoch {epoch + 1} complete · avg loss {avg_loss:.5f} · {time.monotonic() - epoch_started:.1f}s ==")
        epoch += 1
        page_index = 0
        save_checkpoint(checkpoint_path, model, optimizer, epoch, page_index, global_step, last_loss, signature, args)
        print(f"checkpoint saved: {checkpoint_path}")

    print("\nTraining target reached.")
    print(f"Final checkpoint: {checkpoint_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())