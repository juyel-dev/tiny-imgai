from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from model_scalable import ScaledTinyUNet
from scale_common import INPUT_SIZE, DEFAULT_BASE_CHANNELS, calibrate_batchnorm, scalable_dataset_signature, validate_scale_config
from train import build_cache, load_manifest


def main() -> int:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Recompute BatchNorm statistics for a 512px checkpoint.")
    p.add_argument("--data-dir", default=str(base / "data"))
    p.add_argument("--checkpoint-dir", default=str(base / "checkpoints" / "scaled-512-b48-eager"))
    p.add_argument("--input-size", type=int, default=INPUT_SIZE)
    p.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    p.add_argument("--max-pages", type=int, default=0)
    args = p.parse_args()
    if args.max_pages < 0:
        raise ValueError("--max-pages cannot be negative")
    validate_scale_config(args.input_size, args.base_channels)
    torch.set_num_threads(6)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    data_dir = Path(args.data_dir).resolve()
    ckpt_dir = Path(args.checkpoint_dir).resolve()
    ckpt_path = ckpt_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))
    manifest, all_records = load_manifest(data_dir)
    records = all_records[:args.max_pages] if args.max_pages else all_records
    if not records:
        raise ValueError("No PDF page pairs found")
    cache_root = data_dir / "cache512"
    build_cache(records, cache_root, args.input_size)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_input = int(state.get("input_size", -1))
    ckpt_base = int(state.get("base_channels", -1))
    ckpt_mode = bool(state.get("activation_checkpointing", False))
    if (ckpt_input, ckpt_base) != (args.input_size, args.base_channels):
        raise ValueError(f"Checkpoint is {ckpt_input}px/b{ckpt_base}; requested {args.input_size}px/b{args.base_channels}")
    expected = scalable_dataset_signature(manifest, records, args.input_size, args.base_channels, ckpt_mode)
    if state.get("dataset_signature") != expected:
        raise ValueError("Checkpoint dataset signature does not match the selected pages/configuration")
    model = ScaledTinyUNet(args.input_size, args.base_channels, ckpt_mode)
    model.load_state_dict(state["model"])
    model = model.to(memory_format=torch.channels_last)
    print("tiny-imgai 512px BatchNorm finalization")
    print(f"Architecture: b{args.base_channels} | pages: {len(records)}")
    started = time.perf_counter()
    count = calibrate_batchnorm(model, records, cache_root, args.input_size)
    state["model"] = model.state_dict()
    state["bn_calibrated"] = True
    state["bn_calibration_pages"] = len(records)
    state["bn_calibration_method"] = "cumulative_moving_average"
    state["bn_calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    temp = ckpt_path.with_suffix(".finalizing.tmp")
    torch.save(state, temp)
    temp.replace(ckpt_path)
    report = {"checkpoint": str(ckpt_path), "input_size": args.input_size, "base_channels": args.base_channels, "pages": len(records), "batch_norm_layers": count, "method": "cumulative_moving_average", "elapsed_seconds": round(time.perf_counter() - started, 2)}
    report_path = base / "evaluation" / f"scale-bn-{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== complete ===")
    print(f"BatchNorm layers: {count}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
