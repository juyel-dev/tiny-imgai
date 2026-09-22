from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn

from train import PageRecord, build_cache, dataset_signature, load_page_tensor

INPUT_SIZE = 512
DEFAULT_BASE_CHANNELS = 48
SUPPORTED_BASE_CHANNELS = (16, 24, 32, 40, 48)


def validate_scale_config(
    input_size: int,
    base_channels: int,
    activation_checkpointing: bool = False,
) -> None:
    if input_size != INPUT_SIZE:
        raise ValueError(
            f"Scalable production path is fixed at {INPUT_SIZE}x{INPUT_SIZE}; got {input_size}."
        )
    if base_channels not in SUPPORTED_BASE_CHANNELS:
        raise ValueError(
            "base_channels must be one of "
            + ", ".join(str(x) for x in SUPPORTED_BASE_CHANNELS)
            + "."
        )


def scalable_dataset_signature(
    manifest: dict,
    records: list[PageRecord],
    input_size: int,
    base_channels: int,
    activation_checkpointing: bool,
) -> str:
    validate_scale_config(input_size, base_channels, activation_checkpointing)
    base = dataset_signature(manifest, records, input_size)
    architecture = {
        "input_size": input_size,
        "base_channels": base_channels,
        "activation_checkpointing": bool(activation_checkpointing),
        "loss_profile": "print-clean-v1",
    }
    payload = {
        "dataset_signature": base,
        "architecture": architecture,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def calibrate_batchnorm(
    model: nn.Module,
    records: list[PageRecord],
    cache_root: Path,
    input_size: int = INPUT_SIZE,
) -> int:
    """Recompute BatchNorm running statistics over the selected dataset."""
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not batch_norms:
        return 0

    original_momentum = [module.momentum for module in batch_norms]
    original_training = model.training

    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None

    model.train()
    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            original_cache, _ = record.cache_paths(cache_root)
            x = load_page_tensor(original_cache, input_size)
            model(x)
            del x

            if index == 1 or index % 25 == 0 or index == len(records):
                print(f"  BN calibration: {index}/{len(records)}", flush=True)

    for module, momentum in zip(batch_norms, original_momentum):
        module.momentum = momentum

    model.train(original_training)
    return len(batch_norms)


def prepare_cache_and_signature(
    manifest: dict,
    records: list[PageRecord],
    data_dir: Path,
    input_size: int,
    base_channels: int,
    activation_checkpointing: bool,
) -> tuple[Path, str]:
    validate_scale_config(input_size, base_channels, activation_checkpointing)
    cache_root = data_dir / "cache512"
    build_cache(records, cache_root, input_size)
    signature = scalable_dataset_signature(
        manifest,
        records,
        input_size,
        base_channels,
        activation_checkpointing,
    )
    return cache_root, signature


def mark_bn_metadata(checkpoint: dict, pages: int, method: str) -> None:
    checkpoint["bn_calibrated"] = True
    checkpoint["bn_calibration_pages"] = int(pages)
    checkpoint["bn_calibration_method"] = method
    checkpoint["bn_calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
