from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

from train import TinyUNet, PageRecord, build_cache, load_manifest


THREADS = [5, 6, 7, 8]
BATCHES = [1, 2, 4]
MODES = ["eager_fp32", "eager_bf16", "compile_fp32", "compile_bf16"]


def parse_args() -> argparse.Namespace:
    train_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Exhaustive tiny-imgai CPU training setup benchmark."
    )
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument("--pages", type=int, default=100)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--steps-per-config", type=int, default=25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip torch.compile variants.",
    )
    p.add_argument(
        "--skip-bf16",
        action="store_true",
        help="Skip CPU BF16 autocast variants.",
    )
    p.add_argument(
        "--endurance-steps",
        type=int,
        default=300,
        help="Long run for top throughput candidates.",
    )
    p.add_argument(
        "--endurance-candidates",
        type=int,
        default=4,
        help="Number of fastest non-failed configs to endurance-test.",
    )
    return p.parse_args()


def make_mode_plan(args: argparse.Namespace) -> list[str]:
    modes = []
    for mode in MODES:
        if args.skip_compile and mode.startswith("compile_"):
            continue
        if args.skip_bf16 and mode.endswith("_bf16"):
            continue
        modes.append(mode)
    return modes


def autocast_context(mode: str):
    if mode.endswith("_bf16"):
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def load_batch(
    records: list[PageRecord],
    cache_root: Path,
    batch_size: int,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    for i in range(batch_size):
        record = records[(offset + i) % len(records)]
        original_path, processed_path = record.cache_paths(cache_root)

        x_np = np.load(original_path, mmap_mode="r")
        y_np = np.load(processed_path, mmap_mode="r")

        expected = (256, 256, 3)
        if (
            x_np.shape != expected
            or y_np.shape != expected
            or x_np.dtype != np.uint8
            or y_np.dtype != np.uint8
        ):
            raise ValueError(f"Invalid cache for page {record.page_number}")

        x = torch.from_numpy(np.array(x_np, copy=True)).permute(2, 0, 1)
        y = torch.from_numpy(np.array(y_np, copy=True)).permute(2, 0, 1)

        xs.append(x.float().div_(255.0))
        ys.append(y.float().div_(255.0))

    xb = torch.stack(xs).contiguous(memory_format=torch.channels_last)
    yb = torch.stack(ys).contiguous(memory_format=torch.channels_last)
    return xb, yb


def reset_compile_state() -> None:
    try:
        torch._dynamo.reset()
    except Exception:
        pass


def build_model(mode: str) -> tuple[nn.Module, nn.Module, nn.Module]:
    model: nn.Module = TinyUNet(256).to(memory_format=torch.channels_last)
    model.train()

    if mode.startswith("compile_"):
        model = torch.compile(model, backend="inductor")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
) -> float:
    optimizer.zero_grad(set_to_none=True)

    with autocast_context(mode):
        prediction = model(x)
        loss = criterion(prediction, y)

    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def run_segment(
    records: list[PageRecord],
    cache_root: Path,
    threads: int,
    batch_size: int,
    mode: str,
    warmup: int,
    steps: int,
    process: psutil.Process,
    segment_index: int,
    end_to_end: bool = False,
) -> dict:
    torch.set_num_threads(threads)
    torch.manual_seed(1234 + segment_index)

    model = None
    optimizer = None
    criterion = None
    x = None
    y = None

    rss_peak = process.memory_info().rss
    available_min = psutil.virtual_memory().available
    cpu_samples: list[float] = []
    times: list[float] = []
    load_times: list[float] = []

    try:
        model, optimizer, criterion = build_model(mode)

        if not end_to_end:
            x, y = load_batch(
                records,
                cache_root,
                batch_size,
                (segment_index * batch_size) % len(records),
            )

        process.cpu_percent(None)

        for warm_index in range(warmup):
            if end_to_end:
                load_started = time.perf_counter()
                x, y = load_batch(
                    records,
                    cache_root,
                    batch_size,
                    ((warm_index + segment_index) * batch_size) % len(records),
                )
                load_times.append(time.perf_counter() - load_started)

            training_step(model, optimizer, criterion, x, y, mode)

            if end_to_end:
                del x, y
                x, y = None, None

        for step_index in range(steps):
            if end_to_end:
                load_started = time.perf_counter()
                x, y = load_batch(
                    records,
                    cache_root,
                    batch_size,
                    ((step_index + segment_index + warmup) * batch_size) % len(records),
                )
                load_elapsed = time.perf_counter() - load_started
            else:
                load_elapsed = 0.0

            t0 = time.perf_counter()
            training_step(model, optimizer, criterion, x, y, mode)
            elapsed = time.perf_counter() - t0

            if end_to_end:
                elapsed += load_elapsed
                load_times.append(load_elapsed)
                del x, y
                x, y = None, None

            times.append(elapsed)
            rss_peak = max(rss_peak, process.memory_info().rss)
            available_min = min(available_min, psutil.virtual_memory().available)
            cpu_samples.append(process.cpu_percent(None))

    finally:
        del x, y, model, optimizer, criterion
        gc.collect()
        reset_compile_state()

    if not times:
        raise RuntimeError("No measured steps completed.")

    first_count = min(10, len(times))
    first_mean = float(np.mean(times[:first_count]))
    last_mean = float(np.mean(times[-first_count:]))

    result = {
        "threads": threads,
        "batch": batch_size,
        "mode": mode,
        "end_to_end": end_to_end,
        "avg_step_s": round(float(np.mean(times)), 4),
        "page_s": round(float(np.mean(times)) / batch_size, 4),
        "p95_step_s": round(float(np.percentile(times, 95)), 4),
        "min_step_s": round(float(np.min(times)), 4),
        "max_step_s": round(float(np.max(times)), 4),
        "first_page_s": round(first_mean / batch_size, 4),
        "last_page_s": round(last_mean / batch_size, 4),
        "slowdown_x": round(last_mean / max(first_mean, 1e-9), 3),
        "peak_rss_mib": round(rss_peak / 1048576, 1),
        "minimum_system_available_ram_mib": round(available_min / 1048576, 1),
        "avg_process_cpu_percent": round(
            float(np.mean(cpu_samples)), 1
        ),
    }

    if load_times:
        result["avg_load_s"] = round(float(np.mean(load_times)), 4)

    return result


def safe_run(*args, **kwargs) -> dict:
    started = time.perf_counter()
    try:
        row = run_segment(*args, **kwargs)
        row["status"] = "ok"
        row["segment_seconds"] = round(time.perf_counter() - started, 2)
        return row
    except Exception as exc:
        return {
            "threads": kwargs["threads"],
            "batch": kwargs["batch_size"],
            "mode": kwargs["mode"],
            "end_to_end": kwargs["end_to_end"],
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "segment_seconds": round(time.perf_counter() - started, 2),
        }


def summarize_matrix(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for row in results:
        if row.get("status") != "ok" or row.get("end_to_end"):
            continue
        key = (row["mode"], row["threads"], row["batch"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (mode, threads, batch), rows in groups.items():
        summary.append(
            {
                "mode": mode,
                "threads": threads,
                "batch": batch,
                "avg_page_s": round(
                    float(np.mean([r["page_s"] for r in rows])), 4
                ),
                "round_spread_page_s": round(
                    float(max(r["page_s"] for r in rows) - min(r["page_s"] for r in rows)),
                    4,
                ),
                "max_p95_step_s": round(
                    float(max(r["p95_step_s"] for r in rows)), 4
                ),
                "max_rss_mib": round(
                    float(max(r["peak_rss_mib"] for r in rows)), 1
                ),
                "min_free_ram_mib": round(
                    float(min(r["minimum_system_available_ram_mib"] for r in rows)),
                    1,
                ),
                "max_slowdown_x": round(
                    float(max(r["slowdown_x"] for r in rows)), 3
                ),
            }
        )

    return sorted(summary, key=lambda r: r["avg_page_s"])


def pareto_frontier(summary: list[dict]) -> list[dict]:
    frontier = []
    for candidate in summary:
        dominated = False
        for other in summary:
            if other is candidate:
                continue
            no_worse = (
                other["avg_page_s"] <= candidate["avg_page_s"]
                and other["max_rss_mib"] <= candidate["max_rss_mib"]
                and other["max_p95_step_s"] <= candidate["max_p95_step_s"]
            )
            strictly_better = (
                other["avg_page_s"] < candidate["avg_page_s"]
                or other["max_rss_mib"] < candidate["max_rss_mib"]
                or other["max_p95_step_s"] < candidate["max_p95_step_s"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def thermal_info() -> dict:
    temps = {}
    try:
        raw = psutil.sensors_temperatures()
        for group, entries in raw.items():
            temps[group] = [
                {
                    "label": e.label,
                    "current": e.current,
                    "high": e.high,
                    "critical": e.critical,
                }
                for e in entries
            ]
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    return {"available": bool(temps), "temperatures": temps}


def main() -> int:
    args = parse_args()

    if args.pages <= 0:
        raise ValueError("--pages must be positive.")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if args.steps_per_config <= 0:
        raise ValueError("--steps-per-config must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.endurance_steps <= 0:
        raise ValueError("--endurance-steps must be positive.")

    modes = make_mode_plan(args)
    data_dir = Path(args.data_dir).resolve()
    cache_root = data_dir / "cache256"

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    manifest, all_records = load_manifest(data_dir)
    records = all_records[: min(args.pages, len(all_records))]
    if not records:
        raise ValueError("No PDF pages found.")

    print("tiny-imgai exhaustive CPU benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"Available CPU threads: {os.cpu_count() or 1}")
    print(f"Selected pages: {len(records)} / requested {args.pages}")
    print(f"Rounds/config: {args.rounds}")
    print(f"Measured steps/config/round: {args.steps_per_config}")
    print(f"Warmup/config/round: {args.warmup}")
    print(f"Threads: {THREADS}")
    print(f"Batches: {BATCHES}")
    print(f"Modes: {', '.join(modes)}")
    print()

    build_cache(records, cache_root, 256)

    process = psutil.Process(os.getpid())
    results: list[dict] = []
    segment_index = 0

    total_configs = len(THREADS) * len(BATCHES) * len(modes)
    print(f"Matrix configs: {total_configs}")
    print()

    for round_index in range(1, args.rounds + 1):
        order = [
            (mode, threads, batch)
            for mode in modes
            for threads in THREADS
            for batch in BATCHES
        ]
        if round_index % 2 == 0:
            order.reverse()

        for mode, threads, batch in order:
            segment_index += 1
            print(
                f"[matrix {round_index}/{args.rounds}] "
                f"{mode} · {threads}t/b{batch}",
                flush=True,
            )

            row = safe_run(
                records=records,
                cache_root=cache_root,
                threads=threads,
                batch_size=batch,
                mode=mode,
                warmup=args.warmup,
                steps=args.steps_per_config,
                process=process,
                segment_index=segment_index,
                end_to_end=False,
            )
            row["round"] = round_index
            results.append(row)

            if row["status"] == "ok":
                print(
                    f"  {row['page_s']}s/page · "
                    f"p95 {row['p95_step_s']}s · "
                    f"RSS {row['peak_rss_mib']} MiB · "
                    f"free RAM min {row['minimum_system_available_ram_mib']} MiB · "
                    f"slowdown {row['slowdown_x']}x",
                    flush=True,
                )
            else:
                print(f"  FAILED: {row['error']}", flush=True)

    summary = summarize_matrix(results)
    frontier = pareto_frontier(summary)

    print("\n=== matrix summary: fastest configs ===")
    for row in summary[:12]:
        print(
            f"{row['mode']} {row['threads']}t/b{row['batch']}: "
            f"{row['avg_page_s']}s/page | "
            f"p95 {row['max_p95_step_s']}s | "
            f"RSS {row['max_rss_mib']} MiB | "
            f"free RAM min {row['min_free_ram_mib']} MiB | "
            f"spread {row['round_spread_page_s']}s | "
            f"slowdown {row['max_slowdown_x']}x"
        )

    print("\n=== Pareto frontier (speed / RSS / p95) ===")
    for row in sorted(frontier, key=lambda r: r["avg_page_s"]):
        print(
            f"{row['mode']} {row['threads']}t/b{row['batch']}: "
            f"{row['avg_page_s']}s/page | "
            f"p95 {row['max_p95_step_s']}s | "
            f"RSS {row['max_rss_mib']} MiB"
        )

    throughput_candidates = summary[: max(1, args.endurance_candidates)]
    endurance_results = []

    print("\n=== endurance / thermal stage ===")
    for candidate_index, candidate in enumerate(throughput_candidates, start=1):
        print(
            f"[endurance {candidate_index}/{len(throughput_candidates)}] "
            f"{candidate['mode']} · {candidate['threads']}t/b{candidate['batch']} · "
            f"{args.endurance_steps} steps",
            flush=True,
        )

        row = safe_run(
            records=records,
            cache_root=cache_root,
            threads=candidate["threads"],
            batch_size=candidate["batch"],
            mode=candidate["mode"],
            warmup=max(5, args.warmup),
            steps=args.endurance_steps,
            process=process,
            segment_index=segment_index + candidate_index,
            end_to_end=True,
        )
        row["stage"] = "endurance"
        endurance_results.append(row)

        if row["status"] == "ok":
            print(
                f"  {row['page_s']}s/page · "
                f"p95 {row['p95_step_s']}s · "
                f"RSS {row['peak_rss_mib']} MiB · "
                f"free RAM min {row['minimum_system_available_ram_mib']} MiB · "
                f"slowdown {row['slowdown_x']}x · "
                f"load {row.get('avg_load_s', 0)}s/batch",
                flush=True,
            )
        else:
            print(f"  FAILED: {row['error']}", flush=True)

    temperature = thermal_info()
    system_memory = psutil.virtual_memory()

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch": torch.__version__,
        "input_size": 256,
        "manifest_pair_count": manifest.get("pairCount"),
        "available_pages": len(all_records),
        "selected_pages": len(records),
        "threads": THREADS,
        "batches": BATCHES,
        "modes": modes,
        "rounds": args.rounds,
        "steps_per_config": args.steps_per_config,
        "warmup": args.warmup,
        "endurance_steps": args.endurance_steps,
        "matrix_results": results,
        "matrix_summary": summary,
        "pareto_frontier": frontier,
        "endurance_results": endurance_results,
        "thermal": temperature,
        "final_system_memory": {
            "available_mib": round(system_memory.available / 1048576, 1),
            "percent_used": system_memory.percent,
        },
    }

    out_dir = Path(__file__).resolve().parent / "benchmarks"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    json_path = out_dir / f"cpu-exhaustive-{stamp}.json"
    csv_path = out_dir / f"cpu-exhaustive-{stamp}.csv"
    endurance_csv_path = out_dir / f"cpu-exhaustive-endurance-{stamp}.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    matrix_fieldnames = sorted(
        {key for row in results for key in row}
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=matrix_fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if endurance_results:
        fields = sorted({key for row in endurance_results for key in row})
        with endurance_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(endurance_results)

    print("\n=== machine-readable outputs ===")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    if endurance_results:
        print(f"Endurance CSV: {endurance_csv_path}")

    print("\n=== raw environment ===")
    print(f"RAM used: {system_memory.percent}%")
    if temperature.get("available"):
        print("CPU temperature sensors: available")
    else:
        print("CPU temperature sensors: unavailable through psutil on this system")

    print("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
