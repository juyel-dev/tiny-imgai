from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from train import TinyUNet, load_manifest, PageRecord


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark tiny-imgai local CPU training setups.")
    train_dir = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument("--pages", type=int, default=4, help="Number of cached pages to use.")
    p.add_argument("--steps", type=int, default=3, help="Measured training steps per configuration.")
    p.add_argument("--warmup", type=int, default=1, help="Warmup steps per configuration.")
    p.add_argument("--threads", default="1,2,4,6,8")
    p.add_argument("--batches", default="1,2")
    p.add_argument("--formats", default="contiguous,channels_last")
    return p.parse_args()


def load_cached_page(record: PageRecord, cache_root: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    original_path, processed_path = record.cache_paths(cache_root)
    original = np.load(original_path, mmap_mode="r")
    processed = np.load(processed_path, mmap_mode="r")

    expected = (size, size, 3)
    if original.shape != expected or original.dtype != np.uint8:
        raise ValueError(f"Invalid original cache: {original_path}")
    if processed.shape != expected or processed.dtype != np.uint8:
        raise ValueError(f"Invalid processed cache: {processed_path}")

    return np.array(original, copy=True), np.array(processed, copy=True)


def to_batch(records: list[PageRecord], cache_root: Path, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for record in records:
        x, y = load_cached_page(record, cache_root, size)
        xs.append(torch.from_numpy(x).permute(2, 0, 1).float().div_(255.0))
        ys.append(torch.from_numpy(y).permute(2, 0, 1).float().div_(255.0))

    x_batch = torch.stack(xs)
    y_batch = torch.stack(ys)
    return x_batch, y_batch


def configure_threads(n: int) -> None:
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def run_config(
    records: list[PageRecord],
    cache_root: Path,
    size: int,
    threads: int,
    batch_size: int,
    memory_format: str,
    warmup: int,
    steps: int,
) -> dict:
    configure_threads(threads)

    model = TinyUNet(size)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    fmt = torch.contiguous_format if memory_format == "contiguous" else torch.channels_last
    model = model.to(memory_format=fmt)

    batch_records = [records[i % len(records)] for i in range(batch_size)]
    x, y = to_batch(batch_records, cache_root, size)
    if memory_format == "channels_last":
        x = x.contiguous(memory_format=torch.channels_last)
        y = y.contiguous(memory_format=torch.channels_last)

    times = []
    last_loss = float("nan")
    error = None

    try:
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            del prediction, loss
            optimizer.zero_grad(set_to_none=True)

        # CPU synchronization is unnecessary; operations are synchronous.
        for _ in range(steps):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            elapsed = time.perf_counter() - start

            last_loss = float(loss.detach().item())
            times.append(elapsed)
            del prediction, loss
            optimizer.zero_grad(set_to_none=True)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    finally:
        del x, y, model, optimizer, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if error:
        return {
            "threads": threads,
            "batch": batch_size,
            "format": memory_format,
            "status": "FAIL",
            "step_s": "",
            "page_s": "",
            "loss": "",
            "error": error,
        }

    avg_step = sum(times) / len(times)
    return {
        "threads": threads,
        "batch": batch_size,
        "format": memory_format,
        "status": "OK",
        "step_s": round(avg_step, 4),
        "page_s": round(avg_step / batch_size, 4),
        "loss": round(last_loss, 6),
        "error": "",
    }


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    cache_root = data_dir / "cache256"
    size = 256

    torch_version = torch.__version__
    print("tiny-imgai CPU setup benchmark")
    print(f"PyTorch: {torch_version}")
    print(f"CPU: {os.cpu_count()} logical processors reported")
    print(f"Pages: {args.pages} · warmup: {args.warmup} · measured steps: {args.steps}")
    print()

    _, all_records = load_manifest(data_dir)
    records = all_records[: args.pages]
    if not records:
        raise ValueError("No pages found in manifest.")

    missing = []
    for record in records:
        o, p = record.cache_paths(cache_root)
        if not o.exists() or not p.exists():
            missing.append(record.page_number)
    if missing:
        raise FileNotFoundError(
            "Missing 256x256 page cache. Run the local trainer once first; "
            f"missing pages: {missing[:10]}"
        )

    threads = [int(v.strip()) for v in args.threads.split(",") if v.strip()]
    batches = [int(v.strip()) for v in args.batches.split(",") if v.strip()]
    formats = [v.strip() for v in args.formats.split(",") if v.strip()]

    rows = []
    total = len(threads) * len(batches) * len(formats)
    current = 0

    for fmt in formats:
        for batch in batches:
            for thread_count in threads:
                current += 1
                print(
                    f"[{current}/{total}] threads={thread_count} "
                    f"batch={batch} format={fmt}",
                    flush=True,
                )
                result = run_config(
                    records,
                    cache_root,
                    size,
                    thread_count,
                    batch,
                    fmt,
                    args.warmup,
                    args.steps,
                )
                rows.append(result)
                if result["status"] == "OK":
                    print(
                        f"  {result['step_s']}s/step · "
                        f"{result['page_s']}s/page · loss {result['loss']}",
                        flush=True,
                    )
                else:
                    print(f"  FAIL: {result['error']}", flush=True)

    ok = [r for r in rows if r["status"] == "OK"]
    ok.sort(key=lambda r: float(r["page_s"]))

    print("\n=== fastest configurations ===")
    for row in ok[:10]:
        print(
            f"threads={row['threads']} batch={row['batch']} "
            f"format={row['format']} -> {row['page_s']}s/page"
        )

    out_dir = Path(__file__).resolve().parent / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"cpu-setup-{stamp}.json"
    csv_path = out_dir / f"cpu-setup-{stamp}.csv"

    payload = {
        "torch": torch_version,
        "input_size": size,
        "pages": args.pages,
        "warmup": args.warmup,
        "steps": args.steps,
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
