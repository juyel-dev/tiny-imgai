from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image, ImageDraw

from losses import PrintCleanLoss
from model_scalable import ScaledTinyUNet, describe
from train import build_cache, load_manifest, load_page_tensor

INPUT_SIZE = 512
WIDTHS = [16, 24, 32, 48, 64]


def parse_args():
    p = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    p.add_argument("--data-dir", default=str(base / "data"))
    p.add_argument("--train-pages", type=int, default=16)
    p.add_argument("--val-pages", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--min-free-ram-mib", type=int, default=768)
    p.add_argument("--max-rss-mib", type=int, default=1536)
    p.add_argument("--preview-pages", type=int, default=2)
    return p.parse_args()


def split_records(records, train_count, val_count):
    total = train_count + val_count
    if total > len(records):
        raise ValueError("Not enough pages for the requested split.")
    idx = np.linspace(0, len(records) - 1, total, dtype=int)
    chosen = [records[int(i)] for i in idx]
    return chosen[:train_count], chosen[train_count:]


def load_pair(record, cache_root):
    op, yp = record.cache_paths(cache_root)
    return load_page_tensor(op, INPUT_SIZE), load_page_tensor(yp, INPUT_SIZE)


def memory_guard(process, args):
    rss = process.memory_info().rss / 1048576
    free = psutil.virtual_memory().available / 1048576
    if rss > args.max_rss_mib or free < args.min_free_ram_mib:
        raise MemoryError(
            "memory guard: rss={:.1f} MiB free={:.1f} MiB".format(rss, free)
        )
    return rss, free


def calibrate_bn(model, records, cache_root):
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    old = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    with torch.no_grad():
        for record in records:
            x, _ = load_pair(record, cache_root)
            model(x)
            del x
    for m, value in zip(bns, old):
        m.momentum = value
    model.eval()


def metrics(pred, target):
    p = pred.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    y = target.detach().cpu().squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    mse = float(np.mean((p - y) ** 2))
    mae = float(np.mean(np.abs(p - y)))
    pl = (0.2126 * p[..., 0] + 0.7152 * p[..., 1] + 0.0722 * p[..., 2]) < 0.75
    yl = (0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]) < 0.75
    tp = np.count_nonzero(pl & yl)
    fp = np.count_nonzero(pl & ~yl)
    fn = np.count_nonzero(~pl & yl)
    f1 = 1.0 if (2 * tp + fp + fn) == 0 else float(2 * tp / max(1, 2 * tp + fp + fn))
    return mae, mse, f1


def make_preview(model, record, cache_root, path):
    model.eval()
    with torch.no_grad():
        x, y = load_pair(record, cache_root)
        pred = model(x)

    def array(t):
        return (
            t.detach()
            .cpu()
            .squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .numpy()
        )

    canvas = Image.new("RGB", (1536, 512), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ["Original", "Target", "Prediction"]
    for i, img_array in enumerate((array(x), array(y), array(pred))):
        image = Image.fromarray(img_array).resize((512, 512), Image.Resampling.NEAREST)
        canvas.paste(image, (i * 512, 0))
        draw.text((i * 512 + 8, 8), labels[i] + " page " + str(record.page_number), fill="red")
    canvas.save(path, "PNG", optimize=True)
    del x, y, pred


def run_one(width, train_records, val_records, cache_root, args, process, out_dir):
    torch.set_num_threads(args.threads)
    torch.manual_seed(10000 + width)
    model = ScaledTinyUNet(INPUT_SIZE, width, False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    criterion = PrintCleanLoss()
    spec = describe(model)
    started = time.perf_counter()
    peak_rss = process.memory_info().rss
    min_free = psutil.virtual_memory().available
    train_losses = []

    for epoch in range(args.epochs):
        for record in train_records:
            memory_guard(process, args)
            x, y = load_pair(record, cache_root)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            del pred, loss, x, y
            peak_rss = max(peak_rss, process.memory_info().rss)
            min_free = min(min_free, psutil.virtual_memory().available)

    calibrate_bn(model, train_records, cache_root)

    val_rows = []
    with torch.no_grad():
        for record in val_records:
            memory_guard(process, args)
            x, y = load_pair(record, cache_root)
            pred = model(x)
            mae, mse, f1 = metrics(pred, y)
            val_rows.append({"page": record.page_number, "mae": mae, "mse": mse, "dark_f1": f1})
            del pred, x, y
            peak_rss = max(peak_rss, process.memory_info().rss)
            min_free = min(min_free, psutil.virtual_memory().available)

    mean_mae = float(np.mean([r["mae"] for r in val_rows]))
    mean_mse = float(np.mean([r["mse"] for r in val_rows]))
    mean_f1 = float(np.mean([r["dark_f1"] for r in val_rows]))
    psnr = 10.0 * math.log10(1.0 / mean_mse) if mean_mse > 0 else float("inf")

    summary = dict(
        spec,
        status="ok",
        mean_val_mae=round(mean_mae, 6),
        mean_val_psnr_db=round(psnr, 3),
        mean_val_dark_f1=round(mean_f1, 6),
        final_train_loss=round(train_losses[-1], 6),
        elapsed_seconds=round(time.perf_counter() - started, 2),
        peak_rss_mib=round(peak_rss / 1048576, 1),
        min_free_ram_mib=round(min_free / 1048576, 1),
        validation_pages=val_rows,
    )
    torch.save(
        {"model": model.state_dict(), "width": width, "input_size": INPUT_SIZE},
        out_dir / ("candidate-b{}.pt".format(width)),
    )
    return summary, model


def main():
    args = parse_args()
    if args.train_pages <= 0 or args.val_pages <= 0:
        raise ValueError("train-pages and val-pages must be positive")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data_dir = Path(args.data_dir).resolve()
    cache_root = data_dir / "cache512"
    _, records = load_manifest(data_dir)
    build_cache(records, cache_root, INPUT_SIZE)
    train_records, val_records = split_records(records, args.train_pages, args.val_pages)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(__file__).resolve().parent / "evaluation" / ("quality-screen-" + stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()

    print("tiny-imgai 512px quality screen")
    print("Dataset: {} pages".format(len(records)))
    print("Train: {} | Validation: {}".format(len(train_records), len(val_records)))
    print("Widths: {}".format(WIDTHS))
    print("RAM stop: free < {} MiB OR RSS > {} MiB".format(args.min_free_ram_mib, args.max_rss_mib))
    print()

    results = []
    for index, width in enumerate(WIDTHS, start=1):
        print("[{}/{}] 512-b{}".format(index, len(WIDTHS), width), flush=True)
        model = None
        try:
            result, model = run_one(width, train_records, val_records, cache_root, args, process, out_dir)
            results.append(result)
            print(
                "  val F1 {} | MAE {} | PSNR {} dB | {}s | RSS {} MiB".format(
                    result["mean_val_dark_f1"],
                    result["mean_val_mae"],
                    result["mean_val_psnr_db"],
                    result["elapsed_seconds"],
                    result["peak_rss_mib"],
                ),
                flush=True,
            )
            for record in val_records[:args.preview_pages]:
                preview = out_dir / ("b{}-page-{:04d}.png".format(width, record.page_number))
                make_preview(model, record, cache_root, preview)
        except Exception as exc:
            results.append({
                "base_channels": width,
                "status": "failed",
                "error": "{}: {}".format(type(exc).__name__, exc),
            })
            print("  FAILED: {}: {}".format(type(exc).__name__, exc), flush=True)
        finally:
            del model
            gc.collect()

    print("\n=== quality screen summary ===")
    for result in sorted(
        [r for r in results if r.get("status") == "ok"],
        key=lambda r: r["mean_val_dark_f1"],
        reverse=True,
    ):
        print(
            "b{}: F1 {} | MAE {} | PSNR {} dB | {}s".format(
                result["base_channels"],
                result["mean_val_dark_f1"],
                result["mean_val_mae"],
                result["mean_val_psnr_db"],
                result["elapsed_seconds"],
            )
        )

    report = {
        "timestamp": stamp,
        "input_size": INPUT_SIZE,
        "train_pages": [r.page_number for r in train_records],
        "validation_pages": [r.page_number for r in val_records],
        "results": results,
    }
    report_path = out_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nReport: {}".format(report_path))
    print("Models/previews: {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
