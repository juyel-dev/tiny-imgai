from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image

import torch

from train import (
    TinyUNet,
    build_cache,
    dataset_signature,
    load_manifest,
    load_page_tensor,
)


def parse_args() -> argparse.Namespace:
    train_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Evaluate the saved tiny-imgai checkpoint on the full PDF pair."
    )
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument(
        "--checkpoint-dir",
        default=str(train_dir / "checkpoints" / "python-model"),
    )
    p.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
    p.add_argument(
        "--preview-pages",
        type=int,
        default=12,
        help="Evenly sampled pages for the comparison PDF.",
    )
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


def dark_mask(array: np.ndarray, threshold: float = 0.75) -> np.ndarray:
    luminance = (
        0.2126 * array[..., 0]
        + 0.7152 * array[..., 1]
        + 0.0722 * array[..., 2]
    ) / 255.0
    return luminance < threshold


def binary_f1(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    tp = np.count_nonzero(pred_mask & target_mask)
    fp = np.count_nonzero(pred_mask & ~target_mask)
    fn = np.count_nonzero(~pred_mask & target_mask)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    return float((2 * tp) / (2 * tp + fp + fn))


def save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array, mode="RGB").save(path, format="PNG", optimize=True)


def add_png_page(
    doc: fitz.Document,
    png_path: Path,
    width: float,
    height: float,
) -> None:
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, filename=str(png_path), keep_proportion=False)


def make_comparison_image(
    original: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    page_number: int,
) -> Image.Image:
    canvas = Image.new("RGB", (768, 288), "white")
    for index, array in enumerate((original, target, prediction)):
        image = Image.fromarray(array, mode="RGB")
        image = image.resize((256, 256), Image.Resampling.NEAREST)
        canvas.paste(image, (index * 256, 0))

    # Small labels keep the comparison self-describing.
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(canvas)
    labels = (
        f"Original · page {page_number}",
        "Target processed",
        "Model prediction",
    )
    for index, label in enumerate(labels):
        draw.rectangle((index * 256, 256, (index + 1) * 256, 288), fill="white")
        draw.text((index * 256 + 6, 264), label, fill="black")

    return canvas


def main() -> int:
    args = parse_args()
    if args.max_pages < 0:
        raise ValueError("--max-pages cannot be negative")
    if args.preview_pages <= 0:
        raise ValueError("--preview-pages must be positive")

    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    train_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    manifest, all_records = load_manifest(data_dir)
    records = all_records[: args.max_pages] if args.max_pages else all_records
    if not records:
        raise ValueError("No PDF page pairs found.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cache_root = data_dir / "cache256"
    build_cache(records, cache_root, 256)

    expected_signature = dataset_signature(manifest, records, 256)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_signature = checkpoint.get("dataset_signature")
    signature_match = checkpoint_signature == expected_signature
    if not signature_match:
        raise ValueError(
            "Checkpoint dataset signature does not match the selected evaluation pages. "
            "This prevents accidentally evaluating the wrong dataset."
        )

    model = TinyUNet(256)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(memory_format=torch.channels_last)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_root = train_dir / "evaluation" / f"full-{stamp}"
    prediction_dir = out_root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    prediction_pdf = fitz.open()
    comparison_pdf = fitz.open()
    preview_set = {
        round(i)
        for i in np.linspace(
            1, len(records), min(args.preview_pages, len(records))
        )
    }

    rows: list[dict] = []
    total_started = time.perf_counter()

    print("tiny-imgai full checkpoint evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Pages: {len(records)}")
    print(f"Epoch in checkpoint: {checkpoint.get('epoch')}")
    print(f"Global step: {checkpoint.get('global_step')}")
    print("Dataset signature: MATCH")
    print()

    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            original_cache, processed_cache = record.cache_paths(cache_root)

            x = load_page_tensor(original_cache, 256)
            y = load_page_tensor(processed_cache, 256)

            prediction = model(x)

            pred_u8 = to_uint8(prediction)
            target_u8 = to_uint8(y)
            original_u8 = np.load(original_cache).astype(np.uint8, copy=False)

            pred_float = pred_u8.astype(np.float32) / 255.0
            target_float = target_u8.astype(np.float32) / 255.0

            diff = np.abs(pred_float - target_float)
            mse = float(np.mean((pred_float - target_float) ** 2))
            mae = float(np.mean(diff))
            f1 = binary_f1(
                dark_mask(pred_u8),
                dark_mask(target_u8),
            )

            page_png = prediction_dir / f"page_{record.page_number:05d}.png"
            save_png(pred_u8, page_png)

            prediction_pdf_page = prediction_pdf.new_page(
                width=256,
                height=256,
            )
            prediction_pdf_page.insert_image(
                prediction_pdf_page.rect,
                filename=str(page_png),
                keep_proportion=False,
            )

            if record.page_number in preview_set:
                comparison = make_comparison_image(
                    original_u8,
                    target_u8,
                    pred_u8,
                    record.page_number,
                )
                comparison_path = out_root / f"_comparison_{record.page_number:05d}.png"
                comparison.save(comparison_path, format="PNG", optimize=True)
                add_png_page(comparison_pdf, comparison_path, 768, 288)

            rows.append(
                {
                    "page": record.page_number,
                    "pair_id": record.pair_id,
                    "mse": round(mse, 8),
                    "mae": round(mae, 8),
                    "psnr_db": round(psnr_from_mse(mse), 3),
                    "dark_pixel_f1": round(f1, 6),
                    "prediction_png": str(page_png),
                }
            )

            print(
                f"page {index}/{len(records)} "
                f"MAE {mae:.5f} "
                f"PSNR {psnr_from_mse(mse):.2f} dB "
                f"dark-F1 {f1:.4f}",
                flush=True,
            )

            del prediction, x, y
            gc.collect()

    prediction_pdf_path = out_root / "predictions-256.pdf"
    comparison_pdf_path = out_root / "comparison-previews.pdf"
    metrics_csv_path = out_root / "metrics.csv"
    metrics_json_path = out_root / "metrics.json"

    prediction_pdf.save(prediction_pdf_path, garbage=4, deflate=True)
    prediction_pdf.close()

    comparison_pdf.save(comparison_pdf_path, garbage=4, deflate=True)
    comparison_pdf.close()

    mean_mae = float(np.mean([r["mae"] for r in rows]))
    mean_mse = float(np.mean([r["mse"] for r in rows]))
    mean_f1 = float(np.mean([r["dark_pixel_f1"] for r in rows]))
    median_mae = float(np.median([r["mae"] for r in rows]))

    summary = {
        "pages_evaluated": len(rows),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "mean_mae": round(mean_mae, 8),
        "median_mae": round(median_mae, 8),
        "mean_mse": round(mean_mse, 8),
        "mean_psnr_db": round(psnr_from_mse(mean_mse), 3),
        "mean_dark_pixel_f1": round(mean_f1, 6),
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
    print("=== evaluation summary ===")
    print(f"Pages evaluated: {summary['pages_evaluated']}")
    print(f"Mean MAE: {summary['mean_mae']}")
    print(f"Median MAE: {summary['median_mae']}")
    print(f"Mean PSNR: {summary['mean_psnr_db']} dB")
    print(f"Mean dark-pixel F1: {summary['mean_dark_pixel_f1']}")
    print(f"Elapsed: {summary['elapsed_seconds']}s")
    print()
    print(f"Predictions PDF: {prediction_pdf_path}")
    print(f"Comparison PDF: {comparison_pdf_path}")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Metrics JSON: {metrics_json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
