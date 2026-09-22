from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch import nn

from train import build_cache, dataset_signature, load_manifest, load_page_tensor, TinyUNet


def parse_args() -> argparse.Namespace:
    train_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Finalize a trained tiny-imgai checkpoint by recalibrating BatchNorm."
    )
    p.add_argument("--data-dir", default=str(train_dir / "data"))
    p.add_argument(
        "--checkpoint-dir",
        default=str(train_dir / "checkpoints" / "python-model"),
    )
    p.add_argument("--max-pages", type=int, default=0, help="0 = all pages")
    return p.parse_args()


def calibrate_batchnorm(
    model: nn.Module,
    records,
    cache_root: Path,
) -> int:
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norms:
        return 0

    # Use a cumulative moving average across the complete page set.
    # This is much less order-sensitive than the normal momentum=0.01
    # behavior when training with batch size 1.
    original_momentum = [module.momentum for module in batch_norms]
    original_training = model.training

    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None

    model.train()
    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            original_cache, _ = record.cache_paths(cache_root)
            x = load_page_tensor(original_cache, 256)
            _ = model(x)
            del x

            if index % 25 == 0 or index == len(records):
                print(f"  BN calibration: {index}/{len(records)}", flush=True)

    for module, momentum in zip(batch_norms, original_momentum):
        module.momentum = momentum

    model.train(original_training)
    gc.collect()
    return len(batch_norms)


def main() -> int:
    args = parse_args()
    if args.max_pages < 0:
        raise ValueError("--max-pages cannot be negative")

    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data_dir = Path(args.data_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    manifest, all_records = load_manifest(data_dir)
    records = all_records[: args.max_pages] if args.max_pages else all_records
    if not records:
        raise ValueError("No PDF pages found.")

    cache_root = data_dir / "cache256"
    build_cache(records, cache_root, 256)

    signature = dataset_signature(manifest, records, 256)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("dataset_signature") != signature:
        raise ValueError(
            "Checkpoint dataset signature does not match the selected full dataset. "
            "Refusing to modify the checkpoint."
        )

    model = TinyUNet(256)
    model.load_state_dict(checkpoint["model"])
    model = model.to(memory_format=torch.channels_last)

    print("tiny-imgai checkpoint finalization")
    print(f"Pages: {len(records)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Existing epoch: {checkpoint.get('epoch')} · "
        f"global step: {checkpoint.get('global_step')}"
    )
    print()
    print("Recalibrating BatchNorm running statistics (weights are unchanged)...")

    started = time.perf_counter()
    bn_count = calibrate_batchnorm(model, records, cache_root)

    temporary = checkpoint_path.with_suffix(".calibrating.tmp")
    checkpoint["model"] = model.state_dict()
    checkpoint["bn_calibrated"] = True
    checkpoint["bn_calibration_pages"] = len(records)
    checkpoint["bn_calibration_method"] = "cumulative_moving_average"
    checkpoint["bn_calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)

    summary = {
        "pages": len(records),
        "batch_norm_layers": bn_count,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "checkpoint": str(checkpoint_path),
        "bn_calibrated": True,
    }

    report_path = (
        Path(__file__).resolve().parent
        / "evaluation"
        / f"checkpoint-finalization-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=== finalization complete ===")
    print(f"BatchNorm layers: {bn_count}")
    print(f"Elapsed: {summary['elapsed_seconds']}s")
    print(f"Checkpoint updated: {checkpoint_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
