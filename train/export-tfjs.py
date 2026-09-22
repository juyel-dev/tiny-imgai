from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_scalable import ScaledTinyUNet
from scale_common import DEFAULT_BASE_CHANNELS, INPUT_SIZE, validate_scale_config


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Prepare a PyTorch checkpoint for tf.js conversion.")
    p.add_argument("--checkpoint-dir", default=str(base / "checkpoints" / "scaled-512-b48-eager"))
    p.add_argument("--staging-dir", default=str(base / "evaluation" / "tfjs-export-staging"))
    p.add_argument("--input-size", type=int, default=INPUT_SIZE)
    p.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CHANNELS)
    return p.parse_args()


def conv1_kernel(state: dict, prefix: str) -> np.ndarray:
    value = state[f"{prefix}.layers.0.weight"].detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).transpose(2, 3, 1, 0)


def conv2_kernel(state: dict, prefix: str) -> np.ndarray:
    value = state[f"{prefix}.layers.3.weight"].detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).transpose(2, 3, 1, 0)


def tensor(state: dict, key: str) -> np.ndarray:
    return np.asarray(state[key].detach().cpu().numpy(), dtype=np.float32)


def block_tensors(state: dict, prefix: str) -> list[tuple[str, np.ndarray]]:
    return [
        (f"{prefix}_conv1/kernel", conv1_kernel(state, prefix)),
        (f"{prefix}_conv1/bias", tensor(state, f"{prefix}.layers.0.bias")),
        (f"{prefix}_bn1/gamma", tensor(state, f"{prefix}.layers.1.weight")),
        (f"{prefix}_bn1/beta", tensor(state, f"{prefix}.layers.1.bias")),
        (f"{prefix}_bn1/moving_mean", tensor(state, f"{prefix}.layers.1.running_mean")),
        (f"{prefix}_bn1/moving_variance", tensor(state, f"{prefix}.layers.1.running_var")),
        (f"{prefix}_conv2/kernel", conv2_kernel(state, prefix)),
        (f"{prefix}_conv2/bias", tensor(state, f"{prefix}.layers.3.bias")),
        (f"{prefix}_bn2/gamma", tensor(state, f"{prefix}.layers.4.weight")),
        (f"{prefix}_bn2/beta", tensor(state, f"{prefix}.layers.4.bias")),
        (f"{prefix}_bn2/moving_mean", tensor(state, f"{prefix}.layers.4.running_mean")),
        (f"{prefix}_bn2/moving_variance", tensor(state, f"{prefix}.layers.4.running_var")),
    ]


def output_tensors(state: dict) -> list[tuple[str, np.ndarray]]:
    value = state["output.weight"].detach().cpu().numpy()
    kernel = np.asarray(value, dtype=np.float32).transpose(2, 3, 1, 0)
    return [
        ("output/kernel", kernel),
        ("output/bias", tensor(state, "output.bias")),
    ]


def main() -> int:
    args = parse_args()
    validate_scale_config(args.input_size, args.base_channels)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    staging_dir = Path(args.staging_dir).resolve()
    checkpoint_path = checkpoint_dir / "checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("input_size", -1)) != args.input_size:
        raise ValueError("Checkpoint input size does not match export configuration.")
    if int(checkpoint.get("base_channels", -1)) != args.base_channels:
        raise ValueError("Checkpoint base width does not match export configuration.")
    if not checkpoint.get("bn_calibrated", False):
        raise ValueError("Checkpoint is not BatchNorm-calibrated. Run finalize-scale.ps1 first.")

    model = ScaledTinyUNet(
        args.input_size,
        args.base_channels,
        bool(checkpoint.get("activation_checkpointing", False)),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    state = model.state_dict()
    tensors: list[tuple[str, np.ndarray]] = []
    for prefix in ("enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1"):
        tensors.extend(block_tensors(state, prefix))
    tensors.extend(output_tensors(state))

    expected_state_values = sum(value.numel() for value in state.values())
    exported_values = sum(int(np.prod(array.shape)) for _, array in tensors)
    non_exported_buffers = expected_state_values - exported_values

    if non_exported_buffers != 14:
        raise ValueError(
            f"Expected 14 PyTorch num_batches_tracked buffers, found {non_exported_buffers}."
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    weights_path = staging_dir / "weights.bin"
    metadata_path = staging_dir / "metadata.json"
    probe_input_path = staging_dir / "probe-input.bin"
    probe_output_path = staging_dir / "probe-output.bin"

    offset = 0
    specs = []
    with weights_path.open("wb") as handle:
        for name, array in tensors:
            array = np.ascontiguousarray(array, dtype=np.float32)
            raw = array.tobytes(order="C")
            handle.write(raw)
            specs.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": "float32",
                    "byteOffset": offset,
                    "byteLength": len(raw),
                }
            )
            offset += len(raw)

    # Deterministic cross-runtime probe. The tf.js exporter consumes these
    # raw float32 buffers and verifies numerical parity after binding weights.
    probe_h = args.input_size
    probe_w = args.input_size
    probe_channels = 3
    probe_values = np.linspace(
        0.0,
        1.0,
        num=probe_h * probe_w * probe_channels,
        dtype=np.float32,
    ).reshape(1, probe_h, probe_w, probe_channels)
    with torch.inference_mode():
        probe_tensor = torch.from_numpy(
            probe_values.transpose(0, 3, 1, 2).copy()
        )
        probe_output = (
            model(probe_tensor)
            .detach()
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
            .astype(np.float32)
        )
    np.ascontiguousarray(probe_values).tofile(probe_input_path)
    np.ascontiguousarray(probe_output).tofile(probe_output_path)

    metadata = {
        "format": "tiny-imgai-torch-to-tfjs-staging-v1",
        "input_size": args.input_size,
        "base_channels": args.base_channels,
        "channels": list(model.channels),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "state_values": expected_state_values,
        "exported_values": exported_values,
        "weight_count": len(specs),
        "weight_bytes": offset,
        "weights_file": weights_path.name,
        "weight_specs": specs,
        "tfjs_parameter_count": exported_values,
        "probe": {
            "input_file": probe_input_path.name,
            "output_file": probe_output_path.name,
            "shape": [1, probe_h, probe_w, probe_channels],
            "dtype": "float32",
        },
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "bn_calibrated": bool(checkpoint.get("bn_calibrated", False)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("tiny-imgai tf.js export staging")
    print(f"Architecture: 512px / b{args.base_channels}")
    print(f"Trainable params: {metadata['trainable_params']:,}")
    print(f"Weight tensors: {len(specs)}")
    print(f"Weight bytes: {offset:,}")
    print(f"Staging: {staging_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
