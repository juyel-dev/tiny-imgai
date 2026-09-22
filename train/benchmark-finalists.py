from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from train import TinyUNet, PageRecord, load_manifest


CONFIGS = [(8, 1), (8, 4)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalist sustained benchmark for tiny-imgai CPU training.")
    train_dir = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument("--pages", type=int, default=10)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--steps-per-round", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def load_batch(
    records: list[PageRecord],
    cache_root: Path,
    size: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i in range(batch_size):
        record = records[i % len(records)]
        op, yp = record.cache_paths(cache_root)
        x = np.load(op, mmap_mode="r")
        y = np.load(yp, mmap_mode="r")
        expected = (size, size, 3)
        if x.shape != expected or y.shape != expected or x.dtype != np.uint8 or y.dtype != np.uint8:
            raise ValueError(f"Invalid cache for page {record.page_number}")
        xs.append(torch.from_numpy(np.array(x, copy=True)).permute(2, 0, 1).float().div_(255.0))
        ys.append(torch.from_numpy(np.array(y, copy=True)).permute(2, 0, 1).float().div_(255.0))

    xb = torch.stack(xs).contiguous(memory_format=torch.channels_last)
    yb = torch.stack(ys).contiguous(memory_format=torch.channels_last)
    return xb, yb


def run_segment(
    records: list[PageRecord],
    cache_root: Path,
    threads: int,
    batch_size: int,
    warmup: int,
    steps: int,
    process: psutil.Process,
) -> dict:
    torch.set_num_threads(threads)
    model = TinyUNet(256).to(memory_format=torch.channels_last)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    x, y = load_batch(records, cache_root, 256, batch_size)
    times = []
    rss_peak = process.memory_info().rss
    available_min = psutil.virtual_memory().available
    cpu_samples = []

    try:
        process.cpu_percent(None)

        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            del pred, loss

        for _ in range(steps):
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            elapsed = time.perf_counter() - t0

            times.append(elapsed)
            rss_peak = max(rss_peak, process.memory_info().rss)
            available_min = min(available_min, psutil.virtual_memory().available)
            cpu_samples.append(process.cpu_percent(None))
            del pred, loss
            optimizer.zero_grad(set_to_none=True)

    finally:
        del x, y, model, optimizer, criterion
        import gc
        gc.collect()

    if not times:
        raise RuntimeError("No measured steps completed.")

    return {
        "threads": threads,
        "batch": batch_size,
        "avg_step_s": round(float(np.mean(times)), 4),
        "page_s": round(float(np.mean(times)) / batch_size, 4),
        "p95_step_s": round(float(np.percentile(times, 95)), 4),
        "min_step_s": round(float(np.min(times)), 4),
        "max_step_s": round(float(np.max(times)), 4),
        "first10_page_s": round(float(np.mean(times[:10])) / batch_size, 4),
        "last10_page_s": round(float(np.mean(times[-10:])) / batch_size, 4),
        "peak_rss_mib": round(rss_peak / 1048576, 1),
        "minimum_system_available_ram_mib": round(available_min / 1048576, 1),
        "avg_process_cpu_percent": round(float(np.mean(cpu_samples)), 1),
    }


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    cache_root = data_dir / "cache256"

    _, all_records = load_manifest(data_dir)
    records = all_records[: args.pages]
    if not records:
        raise ValueError("No cached pages available.")

    for record in records:
        op, yp = record.cache_paths(cache_root)
        if not op.exists() or not yp.exists():
            raise FileNotFoundError(f"Missing cache for page {record.page_number}")

    process = psutil.Process(os.getpid())
    print("tiny-imgai finalist CPU benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"Pages: {len(records)} · rounds: {args.rounds} · steps/round: {args.steps_per_round}")
    print("Alternating configs: 8 threads/batch 1 <-> 8 threads/batch 4")
    print()

    results = []
    for round_index in range(1, args.rounds + 1):
        for threads, batch in CONFIGS:
            print(
                f"[round {round_index}/{args.rounds}] "
                f"threads={threads} batch={batch}",
                flush=True,
            )
            row = run_segment(
                records,
                cache_root,
                threads,
                batch,
                args.warmup,
                args.steps_per_round,
                process,
            )
            row["round"] = round_index
            results.append(row)
            print(
                f"  {row['page_s']}s/page · p95 {row['p95_step_s']}s · "
                f"RSS {row['peak_rss_mib']} MiB · "
                f"free RAM min {row['minimum_system_available_ram_mib']} MiB · "
                f"first→last {row['first10_page_s']}→{row['last10_page_s']}s/page",
                flush=True,
            )

    out_dir = Path(__file__).resolve().parent / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    summary = {}
    for threads, batch in CONFIGS:
        rows = [r for r in results if r["threads"] == threads and r["batch"] == batch]
        summary[f"{threads}t/b{batch}"] = {
            "avg_page_s": round(float(np.mean([r["page_s"] for r in rows])), 4),
            "round_spread_page_s": round(float(max(r["page_s"] for r in rows) - min(r["page_s"] for r in rows)), 4),
            "max_p95_step_s": round(float(max(r["p95_step_s"] for r in rows)), 4),
            "max_rss_mib": round(float(max(r["peak_rss_mib"] for r in rows)), 1),
            "min_system_available_ram_mib": round(float(min(r["minimum_system_available_ram_mib"] for r in rows)), 1),
            "max_first_to_last_slowdown": round(
                float(max(
                    (r["last10_page_s"] / r["first10_page_s"])
                    for r in rows
                    if r["first10_page_s"] > 0
                )),
                3,
            ),
        }

    print("\n=== finalist summary ===")
    for name, s in summary.items():
        print(
            f"{name}: {s['avg_page_s']}s/page | "
            f"spread {s['round_spread_page_s']}s | "
            f"max p95 {s['max_p95_step_s']}s | "
            f"RSS {s['max_rss_mib']} MiB | "
            f"free RAM min {s['min_system_available_ram_mib']} MiB | "
            f"slowdown {s['max_first_to_last_slowdown']}x"
        )

    payload = {
        "torch": torch.__version__,
        "input_size": 256,
        "pages": len(records),
        "rounds": args.rounds,
        "steps_per_round": args.steps_per_round,
        "warmup": args.warmup,
        "results": results,
        "summary": summary,
    }
    json_path = out_dir / f"cpu-finalists-{stamp}.json"
    csv_path = out_dir / f"cpu-finalists-{stamp}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in results for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
