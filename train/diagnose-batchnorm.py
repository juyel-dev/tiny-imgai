from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from train import build_cache, load_manifest, load_page_tensor, TinyUNet


def parse_args() -> argparse.Namespace:
    train_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Diagnose BatchNorm train/eval behavior on the saved checkpoint."
    )
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument(
        "--checkpoint-dir",
        default=str(train_dir / "checkpoints" / "python-model"),
    )
    p.add_argument("--max-pages", type=int, default=0, help="0 = all")
    return p.parse_args()


def metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    p = (
        pred.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .numpy()
    )
    y = (
        target.detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .clamp(0, 1)
        .numpy()
    )
    diff = np.abs(p - y)
    mse = float(np.mean((p - y) ** 2))
    mae = float(np.mean(diff))

    p_l = (
        0.2126 * p[..., 0] + 0.7152 * p[..., 1] + 0.0722 * p[..., 2]
    ) < 0.75
    y_l = (
        0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]
    ) < 0.75
    tp = np.count_nonzero(p_l & y_l)
    fp = np.count_nonzero(p_l & ~y_l)
    fn = np.count_nonzero(~p_l & y_l)
    f1 = 1.0 if (tp + fp + fn) == 0 else float(
        2 * tp / max(1, 2 * tp + fp + fn)
    )
    return mae, mse, f1


def evaluate(
    model: TinyUNet,
    records,
    cache_root: Path,
    training_mode: bool,
) -> tuple[dict, list[dict]]:
    if training_mode:
        model.train()
    else:
        model.eval()

    page_rows = []
    started = time.perf_counter()

    with torch.no_grad():
        for i, record in enumerate(records, start=1):
            op, yp = record.cache_paths(cache_root)
            x = load_page_tensor(op, 256)
            y = load_page_tensor(yp, 256)

            pred = model(x)
            mae, mse, f1 = metrics(pred, y)

            page_rows.append(
                {
                    "page": record.page_number,
                    "mae": mae,
                    "mse": mse,
                    "dark_f1": f1,
                }
            )

            del pred, x, y

    gc.collect()

    maes = [r["mae"] for r in page_rows]
    mses = [r["mse"] for r in page_rows]
    f1s = [r["dark_f1"] for r in page_rows]

    return (
        {
            "mode": "train" if training_mode else "eval",
            "mean_mae": float(np.mean(maes)),
            "median_mae": float(np.median(maes)),
            "mean_mse": float(np.mean(mses)),
            "mean_psnr_db": (
                10 * math.log10(1 / float(np.mean(mses)))
                if float(np.mean(mses)) > 0
                else float("inf")
            ),
            "mean_dark_f1": float(np.mean(f1s)),
            "elapsed_seconds": time.perf_counter() - started,
        },
        page_rows,
    )


def calibrate_bn(model: TinyUNet, records, cache_root: Path) -> None:
    # Only BatchNorm running statistics are updated. We never backpropagate
    # and never change learnable convolution weights.
    model.train()
    with torch.no_grad():
        for record in records:
            op, _ = record.cache_paths(cache_root)
            x = load_page_tensor(op, 256)
            _ = model(x)
            del x
    gc.collect()


def load_model(checkpoint_path: Path) -> TinyUNet:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = TinyUNet(256)
    model.load_state_dict(checkpoint["model"])
    model = model.to(memory_format=torch.channels_last)
    return model


def main() -> int:
    args = parse_args()
    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data_dir = Path(args.data_dir).resolve()
    checkpoint_path = (
        Path(args.checkpoint_dir).resolve() / "checkpoint.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    manifest, all_records = load_manifest(data_dir)
    records = all_records[: args.max_pages] if args.max_pages else all_records
    cache_root = data_dir / "cache256"
    build_cache(records, cache_root, 256)

    print("tiny-imgai BatchNorm inference diagnosis")
    print(f"Pages: {len(records)}")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # 1) True inference behavior.
    eval_model = load_model(checkpoint_path)
    eval_summary, eval_rows = evaluate(
        eval_model, records, cache_root, training_mode=False
    )
    del eval_model
    gc.collect()

    # 2) Batch-stat behavior used during training, without gradients.
    train_model = load_model(checkpoint_path)
    train_summary, train_rows = evaluate(
        train_model, records, cache_root, training_mode=True
    )
    del train_model
    gc.collect()

    # 3) Same weights after a BN-only calibration pass, then true eval.
    calibrated_model = load_model(checkpoint_path)
    calibrate_bn(calibrated_model, records, cache_root)
    calibrated_summary, calibrated_rows = evaluate(
        calibrated_model, records, cache_root, training_mode=False
    )
    del calibrated_model
    gc.collect()

    print("=== results ===")
    for summary in (eval_summary, train_summary, calibrated_summary):
        print(
            f"{summary['mode']}: "
            f"MAE {summary['mean_mae']:.6f} | "
            f"PSNR {summary['mean_psnr_db']:.3f} dB | "
            f"dark-F1 {summary['mean_dark_f1']:.6f} | "
            f"time {summary['elapsed_seconds']:.1f}s"
        )

    print()
    print("=== deltas vs true eval ===")
    print(
        "train-mode MAE delta: "
        f"{eval_summary['mean_mae'] - train_summary['mean_mae']:+.6f}"
    )
    print(
        "calibrated-eval MAE delta: "
        f"{eval_summary['mean_mae'] - calibrated_summary['mean_mae']:+.6f}"
    )
    print(
        "train-mode F1 delta: "
        f"{train_summary['mean_dark_f1'] - eval_summary['mean_dark_f1']:+.6f}"
    )
    print(
        "calibrated-eval F1 delta: "
        f"{calibrated_summary['mean_dark_f1'] - eval_summary['mean_dark_f1']:+.6f}"
    )

    # Show the pages with the largest true-eval error. This is useful for
    # spotting whether a small number of pathological pages dominate the mean.
    worst = sorted(eval_rows, key=lambda r: r["mae"], reverse=True)[:15]
    print()
    print("=== worst 15 pages (true eval) ===")
    for row in worst:
        print(
            f"page {row['page']}: "
            f"MAE {row['mae']:.6f} | "
            f"dark-F1 {row['dark_f1']:.4f}"
        )

    out_dir = Path(__file__).resolve().parent / "evaluation"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"batchnorm-diagnostic-{stamp}.json"

    payload = {
        "pages": len(records),
        "checkpoint": str(checkpoint_path),
        "eval": eval_summary,
        "train_mode": train_summary,
        "calibrated_eval": calibrated_summary,
        "worst_eval_pages": worst,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
