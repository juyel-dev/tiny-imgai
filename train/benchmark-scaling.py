from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

from losses import PrintCleanLoss
from model_scalable import ScaledTinyUNet, describe
from train import build_cache, load_manifest, load_page_tensor

WIDTHS = [16, 24, 32, 40, 48]
CHECKPOINTING = [False, True]
INPUT_SIZE = 512

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safe 512px U-Net scaling benchmark.")
    train_dir = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument("--pages", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--min-free-ram-mib", type=int, default=768)
    p.add_argument("--max-rss-mib", type=int, default=3072)
    return p.parse_args()

def check_memory(process: psutil.Process, args: argparse.Namespace) -> dict:
    rss = process.memory_info().rss / 1048576
    available = psutil.virtual_memory().available / 1048576
    return {"rss_mib": round(rss, 1), "available_mib": round(available, 1),
            "hard_stop": available < args.min_free_ram_mib or rss > args.max_rss_mib}

def load_pair(record, cache_root):
    op, yp = record.cache_paths(cache_root)
    return load_page_tensor(op, INPUT_SIZE), load_page_tensor(yp, INPUT_SIZE)

def run_candidate(records, cache_root, base_channels, checkpointing, args, process, index):
    torch.set_num_threads(args.threads)
    torch.manual_seed(1000 + index)
    model = ScaledTinyUNet(INPUT_SIZE, base_channels, checkpointing)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    criterion = PrintCleanLoss()
    spec = describe(model)
    peak_rss = process.memory_info().rss
    min_available = psutil.virtual_memory().available
    times = []
    x = y = pred = loss = None
    try:
        x, y = load_pair(records[index % len(records)], cache_root)
        memory = check_memory(process, args)
        if memory["hard_stop"]:
            raise MemoryError(f"Memory guard before warmup: {memory}")
        for _ in range(args.warmup):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            del pred, loss
            pred = loss = None
            memory = check_memory(process, args)
            if memory["hard_stop"]:
                raise MemoryError(f"Memory guard during warmup: {memory}")
        for step in range(args.steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            del pred, loss
            pred = loss = None
            peak_rss = max(peak_rss, process.memory_info().rss)
            min_available = min(min_available, psutil.virtual_memory().available)
            memory = check_memory(process, args)
            if memory["hard_stop"]:
                raise MemoryError(f"Memory guard at step {step + 1}: {memory}")
    finally:
        del x, y, pred, loss, model, optimizer, criterion
        gc.collect()
    return {**spec, "status": "ok",
            "avg_step_s": round(float(np.mean(times)), 4),
            "page_s": round(float(np.mean(times)), 4),
            "p95_step_s": round(float(np.percentile(times, 95)), 4),
            "min_step_s": round(float(np.min(times)), 4),
            "max_step_s": round(float(np.max(times)), 4),
            "peak_rss_mib": round(peak_rss / 1048576, 1),
            "min_available_ram_mib": round(min_available / 1048576, 1)}

def main() -> int:
    args = parse_args()
    if args.pages <= 0 or args.steps <= 0 or args.warmup < 0:
        raise ValueError("pages/steps must be positive and warmup non-negative")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    data_dir = Path(args.data_dir).resolve()
    cache_root = data_dir / "cache512"
    manifest, all_records = load_manifest(data_dir)
    records = all_records[:min(args.pages, len(all_records))]
    if not records:
        raise ValueError("No PDF pages found")
    print("tiny-imgai 512px safe scaling benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU threads: {args.threads}")
    print(f"Pages: {len(records)}")
    print(f"Widths: {WIDTHS}")
    print("Checkpointing: off/on")
    print(f"RAM hard-stop: < {args.min_free_ram_mib} MiB available or > {args.max_rss_mib} MiB process RSS")
    print()
    build_cache(records, cache_root, INPUT_SIZE)
    process = psutil.Process(os.getpid())
    results = []
    candidate_index = 0
    total = len(WIDTHS) * len(CHECKPOINTING)
    for width in WIDTHS:
        for ckpt in CHECKPOINTING:
            candidate_index += 1
            name = f"512-b{width}-ckpt{'on' if ckpt else 'off'}"
            print(f"[{candidate_index}/{total}] {name}", flush=True)
            started = time.perf_counter()
            try:
                row = run_candidate(records, cache_root, width, ckpt, args, process, candidate_index)
                row["candidate"] = name
                row["seconds"] = round(time.perf_counter() - started, 2)
                results.append(row)
                print(f"  {row['page_s']}s/page | params {row['trainable_params']:,} | RSS {row['peak_rss_mib']} MiB | free {row['min_available_ram_mib']} MiB", flush=True)
            except Exception as exc:
                results.append({"candidate": name, "input_size": INPUT_SIZE, "base_channels": width,
                    "channels": [width, width*2, width*4, width*8], "activation_checkpointing": ckpt,
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.perf_counter() - started, 2)})
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            gc.collect()
    ok = [r for r in results if r["status"] == "ok"]
    print("\n=== feasible candidates ===")
    for r in sorted(ok, key=lambda x: x["page_s"]):
        print(f"{r['candidate']}: {r['page_s']}s/page | p95 {r['p95_step_s']}s | RSS {r['peak_rss_mib']} MiB | free {r['min_available_ram_mib']} MiB")
    out = Path(__file__).resolve().parent / "benchmarks"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out / f"scale512-{stamp}.json"
    csv_path = out / f"scale512-{stamp}.csv"
    json_path.write_text(json.dumps({"torch": torch.__version__, "input_size": INPUT_SIZE, "args": vars(args), "results": results}, indent=2), encoding="utf-8")
    fields = sorted({k for r in results for k in r})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nJSON: {json_path}")
    print(f"CSV: {csv_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())