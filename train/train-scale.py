from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from losses import PrintCleanLoss
from model_scalable import ScaledTinyUNet
from scale_common import (
    DEFAULT_BASE_CHANNELS,
    INPUT_SIZE,
    calibrate_batchnorm,
    scalable_dataset_signature,
    validate_scale_config,
)
from train import build_cache, load_manifest, load_page_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="tiny-imgai production 512px CPU trainer."
    )
    train_dir = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--input-size", type=int, default=INPUT_SIZE)
    p.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-pages", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--max-minutes", type=float, default=120)
    p.add_argument("--checkpoint-every-pages", type=int, default=10)
    p.add_argument("--min-free-ram-mib", type=int, default=768)
    p.add_argument("--max-rss-mib", type=int, default=2048)
    p.add_argument("--threads", type=int, default=6)
    return p.parse_args()


def arch_id(input_size: int, base: int, ckpt: bool) -> str:
    return f"{input_size}-b{base}-{'ckpt' if ckpt else 'eager'}"


def memory_state(process: psutil.Process, args: argparse.Namespace) -> dict:
    rss = process.memory_info().rss / 1048576
    available = psutil.virtual_memory().available / 1048576
    return {
        "rss_mib": round(rss, 1),
        "available_mib": round(available, 1),
        "hard_stop": rss > args.max_rss_mib or available < args.min_free_ram_mib,
    }


def checkpoint_path_for(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).resolve()
    return (
        Path(__file__).resolve().parent
        / "checkpoints"
        / f"scaled-{arch_id(args.input_size, args.base_channels, args.activation_checkpointing)}"
    )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch,
    next_page,
    global_step,
    last_loss,
    signature,
    args,
    bn_calibrated: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "next_page_index": next_page,
            "global_step": global_step,
            "last_loss": last_loss,
            "dataset_signature": signature,
            "input_size": args.input_size,
            "base_channels": args.base_channels,
            "activation_checkpointing": args.activation_checkpointing,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "loss_profile": "print-clean-v1",
            "bn_calibrated": bool(bn_calibrated),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, signature, args):
    if not path.exists():
        return 0, 0, 0, float("nan"), False

    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "dataset_signature": signature,
        "input_size": args.input_size,
        "base_channels": args.base_channels,
        "activation_checkpointing": args.activation_checkpointing,
        "batch_size": args.batch_size,
        "loss_profile": "print-clean-v1",
    }
    for key, value in expected.items():
        if state.get(key) != value:
            print(f"Existing checkpoint mismatch on {key}; starting fresh.")
            return 0, 0, 0, float("nan"), False

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return (
        int(state.get("epoch", 0)),
        int(state.get("next_page_index", 0)),
        int(state.get("global_step", 0)),
        float(state.get("last_loss", float("nan"))),
        bool(state.get("bn_calibrated", False)),
    )


def main() -> int:
    args = parse_args()
    validate_scale_config(
        args.input_size,
        args.base_channels,
        args.activation_checkpointing,
    )

    if args.batch_size != 1:
        raise ValueError("Production laptop training uses batch size 1.")
    if args.epochs <= 0 or args.max_pages < 0:
        raise ValueError("Invalid epochs/max-pages.")
    if args.max_minutes <= 0:
        raise ValueError("--max-minutes must be positive.")
    if args.threads <= 0:
        raise ValueError("--threads must be positive.")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(20260922)
    np.random.seed(20260922)

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

    print("tiny-imgai production 512px trainer")
    print(f"PyTorch: {torch.__version__}")
    print("Device: cpu")
    print(f"Input: {args.input_size}x{args.input_size}")
    print(
        f"U-Net width: {args.base_channels}->{args.base_channels * 2}"
        f"->{args.base_channels * 4}->{args.base_channels * 8}"
    )
    print("Loss profile: print-clean-v1")
    print(f"Checkpointing: {args.activation_checkpointing}")
    print(f"Threads: {args.threads} | Batch: {args.batch_size}")
    print(f"Pages: {len(records)} | Epoch target: {args.epochs}")
    print(
        f"RAM guard: free >= {args.min_free_ram_mib} MiB, "
        f"RSS <= {args.max_rss_mib} MiB"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print()

    build_cache(records, cache_root, args.input_size)
    signature = scalable_dataset_signature(
        manifest,
        records,
        args.input_size,
        args.base_channels,
        args.activation_checkpointing,
    )

    model = ScaledTinyUNet(
        args.input_size,
        args.base_channels,
        args.activation_checkpointing,
    ).to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
    )
    criterion = PrintCleanLoss()

    print(
        f"Trainable params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(
        "State values (including BatchNorm buffers): "
        f"{sum(t.numel() for t in model.state_dict().values()):,}"
    )

    epoch, page_index, global_step, last_loss, bn_calibrated = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        signature,
        args,
    )
    start_epoch = epoch

    if epoch >= args.epochs and bn_calibrated:
        print("\nCheckpoint already reaches the requested target and is finalized.")
        print(f"Final checkpoint: {checkpoint_path}")
        return 0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not loss_log.exists():
        loss_log.write_text(
            "epoch,page,global_step,loss,seconds_per_page,timestamp\n",
            encoding="utf-8",
        )

    deadline = time.monotonic() + args.max_minutes * 60

    while epoch < args.epochs:
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_started = time.monotonic()
        current_page = page_index if epoch == start_epoch else 0

        print(f"\n== epoch {epoch + 1}/{args.epochs} ==")

        while current_page < len(records):
            mem = memory_state(process, args)
            if mem["hard_stop"]:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    current_page,
                    global_step,
                    last_loss,
                    signature,
                    args,
                    False,
                )
                print(f"RAM guard stopped run: {mem}")
                return 0

            if time.monotonic() >= deadline:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    current_page,
                    global_step,
                    last_loss,
                    signature,
                    args,
                    False,
                )
                print("Time budget reached. Checkpoint saved; run again to resume.")
                return 0

            record = records[current_page]
            original_cache, processed_cache = record.cache_paths(cache_root)
            started = time.monotonic()

            x = load_page_tensor(original_cache, args.input_size)
            y = load_page_tensor(processed_cache, args.input_size)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            value = float(loss.detach().cpu().item())
            elapsed = max(1e-6, time.monotonic() - started)
            last_loss = value
            epoch_loss += value
            epoch_steps += 1
            global_step += 1
            current_page += 1

            with loss_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch},{record.page_number},{global_step},"
                    f"{value:.8f},{elapsed:.4f},"
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                )

            print(
                f"page {record.page_number}/{len(records)} "
                f"loss {value:.5f} step {global_step} "
                f"{elapsed:.2f}s/page",
                flush=True,
            )

            del prediction, loss, x, y
            gc.collect()

            if (
                args.checkpoint_every_pages > 0
                and current_page < len(records)
                and current_page % args.checkpoint_every_pages == 0
            ):
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    current_page,
                    global_step,
                    last_loss,
                    signature,
                    args,
                    False,
                )
                print(f"  checkpoint saved at page {current_page}")

        avg_loss = epoch_loss / max(1, epoch_steps)
        epoch += 1
        page_index = 0
        bn_calibrated = False

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            page_index,
            global_step,
            last_loss,
            signature,
            args,
            False,
        )
        print(
            f"== epoch {epoch} complete · avg loss {avg_loss:.5f} · "
            f"{time.monotonic() - epoch_started:.1f}s =="
        )
        print(f"checkpoint saved: {checkpoint_path}")

    print("\nTarget epoch reached. Finalizing BatchNorm over all selected pages...")
    model.eval()
    bn_count = calibrate_batchnorm(
        model,
        records,
        cache_root,
        args.input_size,
    )

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch,
        0,
        global_step,
        last_loss,
        signature,
        args,
        True,
    )

    final_report = checkpoint_dir / "finalization.json"
    final_report.write_text(
        json.dumps(
            {
                "input_size": args.input_size,
                "base_channels": args.base_channels,
                "pages": len(records),
                "batch_norm_layers": bn_count,
                "method": "cumulative_moving_average",
                "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"BatchNorm layers finalized: {bn_count}")
    print(f"Final checkpoint: {checkpoint_path}")
    print(f"Finalization report: {final_report}")
    print("\nTraining target reached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
