from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import psutil
import pymupdf as fitz
from PIL import Image, ImageDraw
import torch

from model_scalable import ScaledTinyUNet
from scale_common import (
    DEFAULT_BASE_CHANNELS,
    INPUT_SIZE,
    scalable_dataset_signature,
    validate_scale_config,
)
from train import build_cache, load_manifest, load_page_tensor


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Evaluate a scalable tiny-imgai checkpoint at 512px."
    )
    p.add_argument("--data-dir", default=str(base / "data"))
    p.add_argument(
        "--checkpoint-dir",
        default=str(base / "checkpoints" / "scaled-512-b48-eager"),
    )
    p.add_argument("--input-size", type=int, default=INPUT_SIZE)
    p.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    p.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
    p.add_argument(
        "--preview-pages",
        type=int,
        default=12,
        help="Evenly sampled pages included in the comparison PDF.",
    )
    p.add_argument("--max-rss-mib", type=int, default=2048)
    p.add_argument("--min-free-ram-mib", type=int, default=768)
    return p.parse_args()


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = (
        tensor.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .numpy()
    )
    return np.ascontiguousarray(array)


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def luminance(array: np.ndarray) -> np.ndarray:
    return (
        0.2126 * array[..., 0]
        + 0.7152 * array[..., 1]
        + 0.0722 * array[..., 2]
    ) / 255.0


def binary_f1(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[float, int, int, int]:
    tp = int(np.count_nonzero(pred_mask & target_mask))
    fp = int(np.count_nonzero(pred_mask & ~target_mask))
    fn = int(np.count_nonzero(~pred_mask & target_mask))
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, tp, fp, fn
    return float((2.0 * tp) / max(1, 2 * tp + fp + fn)), tp, fp, fn


def make_comparison(
    original: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    page_number: int,
) -> Image.Image:
    label_h = 40
    canvas = Image.new(
        "RGB",
        (INPUT_SIZE * 3, INPUT_SIZE + label_h),
        "white",
    )

    for index, array in enumerate((original, target, prediction)):
        canvas.paste(Image.fromarray(array, mode="RGB"), (index * INPUT_SIZE, 0))

    draw = ImageDraw.Draw(canvas)
    labels = (
        f"Original · page {page_number}",
        "Target processed",
        "Model prediction",
    )
    for index, label in enumerate(labels):
        x = index * INPUT_SIZE
        draw.rectangle(
            (x, INPUT_SIZE, x + INPUT_SIZE, INPUT_SIZE + label_h),
            fill="white",
        )
        draw.text((x + 8, INPUT_SIZE + 12), label, fill="black")

    return canvas


def memory_state(max_rss_mib: int, min_free_ram_mib: int) -> dict:
    process = psutil.Process()
    rss = process.memory_info().rss / 1048576
    available = psutil.virtual_memory().available / 1048576
    return {
        "rss_mib": round(rss, 1),
        "available_mib": round(available, 1),
        "hard_stop": rss > max_rss_mib or available < min_free_ram_mib,
    }


def main() -> int:
    args = parse_args()
    if args.max_pages < 0:
        raise ValueError("--max-pages cannot be negative.")
    if args.preview_pages <= 0:
        raise ValueError("--preview-pages must be positive.")
    if args.max_rss_mib <= 0 or args.min_free_ram_mib <= 0:
        raise ValueError("RAM limits must be positive.")

    validate_scale_config(args.input_size, args.base_channels)

    train_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    manifest, all_records = load_manifest(data_dir)
    records = all_records[:args.max_pages] if args.max_pages else all_records
    if not records:
        raise ValueError("No PDF page pairs found.")

    cache_root = data_dir / "cache512"
    build_cache(records, cache_root, args.input_size)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_input = int(checkpoint.get("input_size", -1))
    checkpoint_base = int(checkpoint.get("base_channels", -1))
    checkpoint_ckpt = bool(checkpoint.get("activation_checkpointing", False))

    if checkpoint_input != args.input_size or checkpoint_base != args.base_channels:
        raise ValueError(
            f"Checkpoint architecture is {checkpoint_input}px/b{checkpoint_base}, "
            f"but evaluation requested {args.input_size}px/b{args.base_channels}."
        )

    expected_signature = scalable_dataset_signature(
        manifest,
        records,
        args.input_size,
        args.base_channels,
        checkpoint_ckpt,
    )
    if checkpoint.get("dataset_signature") != expected_signature:
        raise ValueError(
            "Checkpoint dataset signature does not match the selected "
            "pages/configuration. Refusing to evaluate the wrong dataset."
        )

    model = ScaledTinyUNet(
        args.input_size,
        args.base_channels,
        checkpoint_ckpt,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(memory_format=torch.channels_last)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = train_dir / "evaluation" / f"scale512-{stamp}"
    prediction_dir = out_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    preview_set = {
        int(round(value))
        for value in np.linspace(
            1,
            len(records),
            min(args.preview_pages, len(records)),
        )
    }

    prediction_pdf = fitz.open()
    comparison_pdf = fitz.open()
    rows: list[dict] = []
    total_started = time.perf_counter()
    peak_rss = 0.0
    min_free = float("inf")

    print("tiny-imgai scalable 512px checkpoint evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Architecture: {args.input_size}px · b{args.base_channels}")
    print(f"Pages: {len(records)}")
    print(f"Epoch: {checkpoint.get('epoch')}")
    print(f"Global step: {checkpoint.get('global_step')}")
    print(f"BN calibrated: {bool(checkpoint.get('bn_calibrated', False))}")
    print("Dataset signature: MATCH")
    print()

    try:
        with torch.inference_mode():
            for index, record in enumerate(records, start=1):
                memory = memory_state(args.max_rss_mib, args.min_free_ram_mib)
                if memory["hard_stop"]:
                    raise MemoryError(
                        f"RAM guard before page {record.page_number}: {memory}"
                    )

                original_cache, processed_cache = record.cache_paths(cache_root)
                x = load_page_tensor(original_cache, args.input_size)
                y = load_page_tensor(processed_cache, args.input_size)
                prediction = model(x)

                pred_u8 = to_uint8(prediction)
                target_u8 = to_uint8(y)
                original_u8 = np.load(original_cache).astype(np.uint8, copy=False)

                pred_float = pred_u8.astype(np.float32) / 255.0
                target_float = target_u8.astype(np.float32) / 255.0
                pred_luma = luminance(pred_u8)
                target_luma = luminance(target_u8)

                diff = np.abs(pred_float - target_float)
                mse = float(np.mean((pred_float - target_float) ** 2))
                mae = float(np.mean(diff))

                target_foreground = target_luma < 0.95
                target_background = ~target_foreground
                pred_dark = pred_luma < 0.75
                target_dark = target_luma < 0.75

                f1, tp, fp, fn = binary_f1(pred_dark, target_dark)
                if np.any(target_foreground):
                    fg_mask = np.repeat(target_foreground[..., None], 3, axis=2)
                    foreground_mae = float(np.mean(diff[fg_mask]))
                else:
                    foreground_mae = 0.0

                if np.any(target_background):
                    background_whiteness = float(
                        np.mean(np.abs(1.0 - pred_luma[target_background]))
                    )
                    white_background_rate = float(
                        np.mean(pred_luma[target_background] >= 0.95)
                    )
                else:
                    background_whiteness = 0.0
                    white_background_rate = 1.0

                color_residual = float(
                    np.mean(
                        np.abs(
                            pred_u8.astype(np.float32)
                            - pred_luma[..., None] * 255.0
                        )
                    )
                    / 255.0
                )

                page_png = prediction_dir / f"page_{record.page_number:05d}.png"
                Image.fromarray(pred_u8, mode="RGB").save(
                    page_png,
                    format="PNG",
                    optimize=True,
                )
                page = prediction_pdf.new_page(
                    width=args.input_size,
                    height=args.input_size,
                )
                page.insert_image(
                    page.rect,
                    filename=str(page_png),
                    keep_proportion=False,
                )

                if record.page_number in preview_set:
                    comparison = make_comparison(
                        original_u8,
                        target_u8,
                        pred_u8,
                        record.page_number,
                    )
                    comparison_path = (
                        out_root / f"_comparison_{record.page_number:05d}.png"
                    )
                    comparison.save(
                        comparison_path,
                        format="PNG",
                        optimize=True,
                    )
                    comparison_page = comparison_pdf.new_page(
                        width=args.input_size * 3,
                        height=args.input_size + 40,
                    )
                    comparison_page.insert_image(
                        comparison_page.rect,
                        filename=str(comparison_path),
                        keep_proportion=False,
                    )

                rows.append(
                    {
                        "page": record.page_number,
                        "pair_id": record.pair_id,
                        "mse": round(mse, 8),
                        "mae": round(mae, 8),
                        "psnr_db": round(psnr_from_mse(mse), 3),
                        "dark_pixel_f1": round(f1, 6),
                        "dark_tp": tp,
                        "dark_fp": fp,
                        "dark_fn": fn,
                        "foreground_mae": round(foreground_mae, 8),
                        "background_whiteness_error": round(
                            background_whiteness, 8
                        ),
                        "predicted_white_background_rate": round(
                            white_background_rate, 6
                        ),
                        "color_residual": round(color_residual, 8),
                        "prediction_png": str(page_png),
                    }
                )

                rss = psutil.Process().memory_info().rss / 1048576
                available = psutil.virtual_memory().available / 1048576
                peak_rss = max(peak_rss, rss)
                min_free = min(min_free, available)

                if index == 1 or index % 10 == 0 or index == len(records):
                    print(
                        f"page {index}/{len(records)} "
                        f"MAE {mae:.5f} "
                        f"PSNR {psnr_from_mse(mse):.2f} dB "
                        f"dark-F1 {f1:.4f} "
                        f"RSS {rss:.1f} MiB",
                        flush=True,
                    )

                del prediction, x, y
                gc.collect()
    finally:
        prediction_pdf_path = out_root / "predictions-512.pdf"
        comparison_pdf_path = out_root / "comparison-previews-512.pdf"
        try:
            prediction_pdf.save(prediction_pdf_path, garbage=4, deflate=True)
        finally:
            prediction_pdf.close()
        try:
            comparison_pdf.save(comparison_pdf_path, garbage=4, deflate=True)
        finally:
            comparison_pdf.close()

    if not rows:
        raise RuntimeError("No pages were evaluated.")

    metrics_csv_path = out_root / "metrics.csv"
    metrics_json_path = out_root / "metrics.json"

    mean_mae = float(np.mean([row["mae"] for row in rows]))
    median_mae = float(np.median([row["mae"] for row in rows]))
    mean_mse = float(np.mean([row["mse"] for row in rows]))
    mean_f1 = float(np.mean([row["dark_pixel_f1"] for row in rows]))
    mean_fg_mae = float(np.mean([row["foreground_mae"] for row in rows]))
    mean_bg = float(np.mean([row["background_whiteness_error"] for row in rows]))
    mean_white_rate = float(
        np.mean([row["predicted_white_background_rate"] for row in rows])
    )
    mean_color = float(np.mean([row["color_residual"] for row in rows]))

    summary = {
        "pages_evaluated": len(rows),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "input_size": args.input_size,
        "base_channels": args.base_channels,
        "channels": [args.base_channels, args.base_channels * 2, args.base_channels * 4, args.base_channels * 8],
        "trainable_params": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "bn_calibrated": bool(checkpoint.get("bn_calibrated", False)),
        "mean_mae": round(mean_mae, 8),
        "median_mae": round(median_mae, 8),
        "mean_mse": round(mean_mse, 8),
        "mean_psnr_db": round(psnr_from_mse(mean_mse), 3),
        "mean_dark_pixel_f1": round(mean_f1, 6),
        "mean_foreground_mae": round(mean_fg_mae, 8),
        "mean_background_whiteness_error": round(mean_bg, 8),
        "mean_predicted_white_background_rate": round(mean_white_rate, 6),
        "mean_color_residual": round(mean_color, 8),
        "peak_rss_mib": round(peak_rss, 1),
        "min_free_ram_mib": round(min_free, 1) if min_free != float("inf") else None,
        "elapsed_seconds": round(time.perf_counter() - total_started, 2),
        "prediction_pdf": str(prediction_pdf_path),
        "comparison_pdf": str(comparison_pdf_path),
    }

    metrics_json_path.write_text(
        json.dumps({"summary": summary, "pages": rows}, indent=2),
        encoding="utf-8",
    )

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== scalable evaluation summary ===")
    print(f"Pages evaluated: {summary['pages_evaluated']}")
    print(f"Mean MAE: {summary['mean_mae']}")
    print(f"Median MAE: {summary['median_mae']}")
    print(f"Mean PSNR: {summary['mean_psnr_db']} dB")
    print(f"Mean dark-pixel F1: {summary['mean_dark_pixel_f1']}")
    print(f"Mean foreground MAE: {summary['mean_foreground_mae']}")
    print(f"Background whiteness error: {summary['mean_background_whiteness_error']}")
    print(f"White background rate: {summary['mean_predicted_white_background_rate']}")
    print(f"Color residual: {summary['mean_color_residual']}")
    print(f"Peak RSS: {summary['peak_rss_mib']} MiB")
    print(f"Elapsed: {summary['elapsed_seconds']}s")
    print()
    print(f"Predictions PDF: {prediction_pdf_path}")
    print(f"Comparison PDF: {comparison_pdf_path}")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Metrics JSON: {metrics_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
