from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from train import TinyUNet, PageRecord, load_manifest


CONFIGS = [
    (6, 1),
    (7, 1),
    (8, 1),
    (6, 2),
    (7, 2),
    (8, 2),
    (6, 4),
    (7, 4),
    (8, 4),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sustained tiny-imgai CPU stability benchmark.")
    train_dir = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument("--pages", type=int, default=10)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260922)
    return p.parse_args()


def load_page(record: PageRecord, cache_root: Path, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    op, yp = record.cache_paths(cache_root)
    x = np.load(op, mmap_mode="r")
    y = np.load(yp, mmap_mode="r")
    expected = (size, size, 3)
    if x.shape != expected or y.shape != expected or x.dtype != np.uint8 or y.dtype != np.uint8:
        raise ValueError(f"Invalid cache for page {record.page_number}")

    xt = torch.from_numpy(np.array(x, copy=True)).permute(2, 0, 1).float().div_(255.0)
    yt = torch.from_numpy(np.array(y, copy=True)).permute(2, 0, 1).float().div_(255.0)
    return xt, yt


def make_batch(records, cache_root, size, batch_size):
    chosen = [records[i % len(records)] for i in range(batch_size)]
    xs, ys = zip(*(load_page(r, cache_root, size) for r in chosen))
    return torch.stack(xs).contiguous(memory_format=torch.channels_last), torch.stack(ys).contiguous(memory_format=torch.channels_last)


def run_config(records, cache_root, threads, batch_size, warmup, steps, size, process):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = TinyUNet(size).to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    model.train()

    x, y = make_batch(records, cache_root, size, batch_size)
    times = []
    rss_max = 0
    cpu_samples = []
    last_loss = float("nan")

    def sample():
        nonlocal rss_max
        rss = process.memory_info().rss
        rss_max = max(rss_max, rss)
        cpu_samples.append(process.cpu_percent(None))

    try:
        process.cpu_percent(None)

        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            del pred, loss
        sample()

        for _ in range(steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            elapsed = time.perf_counter() - t0

            last_loss = float(loss.detach().item())
            times.append(elapsed)
            sample()
            del pred, loss
            optimizer.zero_grad(set_to_none=True)

        avg_step = sum(times) / len(times)
        p95 = float(np.percentile(times, 95))
        return {
            "threads": threads,
            "batch": batch_size,
            "status": "OK",
            "page_s": round(avg_step / batch_size, 4),
            "step_s": round(avg_step, 4),
            "p95_step_s": round(p95, 4),
            "min_step_s": round(min(times), 4),
            "max_step_s": round(max(times), 4),
            "peak_rss_mib": round(rss_max / 1048576, 1),
            "avg_cpu_percent": round(float(np.mean(cpu_samples)), 1),
            "loss": round(last_loss, 6),
        }
    except Exception as exc:
        return {
            "threads": threads,
            "batch": batch_size,
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        del x, y, model, optimizer, criterion
        import gc
        gc.collect()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    cache_root = data_dir / "cache256"
    _, records = load_manifest(data_dir)
    records = records[: args.pages]
    if not records:
        raise ValueError("No cached pages available.")

    for r in records:
        o, y = r.cache_paths(cache_root)
        if not o.exists() or not y.exists():
            raise FileNotFoundError(f"Missing cache for page {r.page_number}")

    configs = CONFIGS[:]
    random.shuffle(configs)

    process = psutil.Process(os.getpid())
    print("tiny-imgai sustained CPU stability benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU: {os.cpu_count()} logical processors")
    print(f"Pages: {len(records)} · warmup: {args.warmup} · measured steps: {args.steps}")
    print("Configurations:", ", ".join(f"{t}t/b{b}" for t, b in configs))
    print()

    rows = []
    for index, (threads, batch) in enumerate(configs, 1):
        print(f"[{index}/{len(configs)}] threads={threads} batch={batch}", flush=True)
        row = run_config(records, cache_root, threads, batch, args.warmup, args.steps, 256, process)
        rows.append(row)
        if row["status"] == "OK":
            print(
                f"  {row['page_s']}s/page · p95 {row['p95_step_s']}s/step · "
                f"RSS {row['peak_rss_mib']} MiB · CPU {row['avg_cpu_percent']}%",
                flush=True,
            )
        else:
            print(f"  FAIL: {row['error']}", flush=True)

    ok = [r for r in rows if r["status"] == "OK"]
    ok.sort(key=lambda r: float(r["page_s"]))

    print("
=== sustained ranking ===")
    for r in ok:
        print(
            f"{r['page_s']}s/page | {r['threads']} threads | batch {r['batch']} | "
            f"RSS {r['peak_rss_mib']} MiB | p95 {r['p95_step_s']}s"
        )

    out_dir = Path(__file__).resolve().parent / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"cpu-stability-{stamp}.json"
    csv_path = out_dir / f"cpu-stability-{stamp}.csv"

    payload = {
        "torch": torch.__version__,
        "input_size": 256,
        "pages": len(records),
        "warmup": args.warmup,
        "steps": args.steps,
        "configs": configs,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"
Saved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
