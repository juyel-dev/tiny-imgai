from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
import torch
from torch import nn

from losses import PrintCleanLoss


INPUT_SIZE = 256
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_MINUTES = 30
DEFAULT_MAX_PAGES = 0
DEFAULT_CHECKPOINT_EVERY_PAGES = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tiny-imgai local Python trainer")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--checkpoint-dir", default=str(Path(__file__).resolve().parent / "checkpoints" / "python-model"))
    parser.add_argument("--loss-profile", choices=("mse", "print-clean-v1"), default=os.getenv("LOSS_PROFILE", "mse"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", DEFAULT_EPOCHS)))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", DEFAULT_LEARNING_RATE)))
    parser.add_argument("--max-minutes", type=float, default=float(os.getenv("MAX_MINUTES", DEFAULT_MAX_MINUTES)))
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("MAX_PAGES", DEFAULT_MAX_PAGES)))
    parser.add_argument(
        "--checkpoint-every-pages",
        type=int,
        default=int(os.getenv("CHECKPOINT_EVERY_PAGES", DEFAULT_CHECKPOINT_EVERY_PAGES)),
    )
    parser.add_argument("--input-size", type=int, default=int(os.getenv("INPUT_SIZE", INPUT_SIZE)))
    return parser.parse_args()


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            # Match Keras BatchNormalization defaults used by the original tf.js model:
            # epsilon=0.001 and momentum=0.99 -> PyTorch momentum=0.01.
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TinyUNet(nn.Module):
    """PyTorch translation of /core/model.js. Architecture intentionally unchanged."""

    def __init__(self, input_size: int = INPUT_SIZE) -> None:
        super().__init__()
        self.input_size = input_size

        self.enc1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = ConvBlock(128 + 64, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ConvBlock(64 + 32, 32)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ConvBlock(32 + 16, 16)

        self.output = nn.Conv2d(16, 3, kernel_size=1, bias=True)

        self._initialize_like_tfjs()

    @staticmethod
    def _initialize_conv(layer: nn.Conv2d) -> None:
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _initialize_like_tfjs(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                self._initialize_conv(module)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(self._cat(u3, e3))

        u2 = self.up2(d3)
        d2 = self.dec2(self._cat(u2, e2))

        u1 = self.up1(d2)
        d1 = self.dec1(self._cat(u1, e1))

        return torch.sigmoid(self.output(d1))


@dataclass(frozen=True)
class PageRecord:
    pair_id: str
    page_number: int
    original_page: int
    processed_page: int
    original_pdf: Path
    processed_pdf: Path

    @property
    def cache_dir_name(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", self.pair_id).strip("._-")
        return safe or hashlib.sha1(self.pair_id.encode("utf-8")).hexdigest()[:12]

    def cache_paths(self, cache_root: Path) -> tuple[Path, Path]:
        folder = cache_root / self.cache_dir_name
        return (
            folder / f"page_{self.page_number:05d}_original.npy",
            folder / f"page_{self.page_number:05d}_processed.npy",
        )


def load_manifest(data_dir: Path) -> tuple[dict, list[PageRecord]]:
    originals_dir = data_dir / "originals"
    processed_dir = data_dir / "processed"

    if not originals_dir.exists() or not processed_dir.exists():
        raise FileNotFoundError(
            f"Expected PDF folders: {originals_dir} and {processed_dir}"
        )

    original_files = {
        p.stem: p for p in originals_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    }
    processed_files = {
        p.stem: p for p in processed_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    }

    matched_ids = sorted(set(original_files) & set(processed_files))
    if not matched_ids:
        raise ValueError(
            f"No matching PDF pairs found in {originals_dir} and {processed_dir}."
        )

    missing_processed = sorted(set(original_files) - set(processed_files))
    missing_original = sorted(set(processed_files) - set(original_files))

    for pair_id in missing_processed[:10]:
        print(f"Missing processed PDF for: {pair_id}")
    for pair_id in missing_original[:10]:
        print(f"Missing original PDF for: {pair_id}")

    page_records: list[PageRecord] = []
    manifest_pairs = []
    page_number = 0

    for pair_id in matched_ids:
        original_pdf = original_files[pair_id]
        processed_pdf = processed_files[pair_id]

        with fitz.open(original_pdf) as original_doc, fitz.open(processed_pdf) as processed_doc:
            original_count = original_doc.page_count
            processed_count = processed_doc.page_count

        count = min(original_count, processed_count)
        if count <= 0:
            continue

        if original_count != processed_count:
            print(
                f"[{pair_id}] page count mismatch ({original_count} vs {processed_count}) "
                f"— using first {count} page(s)"
            )

        manifest_pairs.append({
            "id": pair_id,
            "originalPdf": f"originals/{original_pdf.name}",
            "processedPdf": f"processed/{processed_pdf.name}",
            "pageCount": count,
        })

        for page_index in range(1, count + 1):
            page_number += 1
            page_records.append(
                PageRecord(
                    pair_id=pair_id,
                    page_number=page_number,
                    original_page=page_index,
                    processed_page=page_index,
                    original_pdf=original_pdf,
                    processed_pdf=processed_pdf,
                )
            )

    manifest = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pairCount": len(manifest_pairs),
        "pairs": manifest_pairs,
    }

    return manifest, page_records


def render_page(pdf: fitz.Document, page_number: int, size: int) -> np.ndarray:
    page = pdf.load_page(page_number - 1)
    rect = page.rect
    scale = min(size / rect.width, size / rect.height)
    matrix = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)

    page_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)

    x = (size - pix.width) // 2
    y = (size - pix.height) // 2
    canvas[y : y + pix.height, x : x + pix.width] = page_rgb
    return np.ascontiguousarray(canvas)



def validate_cache_shape(path: Path, input_size: int) -> bool:
    try:
        array = np.load(path, mmap_mode="r")
        return array.dtype == np.uint8 and array.shape == (input_size, input_size, 3)
    except (OSError, ValueError):
        return False


def build_cache(records: list[PageRecord], cache_root: Path, input_size: int) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[Path, Path], list[PageRecord]] = {}
    for record in records:
        grouped.setdefault((record.original_pdf, record.processed_pdf), []).append(record)

    created = 0
    total = len(records)

    for group_index, ((original_path, processed_path), group) in enumerate(grouped.items(), start=1):
        print(f"Cache PDF pair {group_index}/{len(grouped)}: {group[0].pair_id}")

        with fitz.open(original_path) as original_doc, fitz.open(processed_path) as processed_doc:
            for done, record in enumerate(group, start=1):
                original_cache, processed_cache = record.cache_paths(cache_root)
                if (
                    original_cache.exists()
                    and processed_cache.exists()
                    and validate_cache_shape(original_cache, input_size)
                    and validate_cache_shape(processed_cache, input_size)
                ):
                    continue

                original_cache.parent.mkdir(parents=True, exist_ok=True)

                original = render_page(original_doc, record.original_page, input_size)
                processed = render_page(processed_doc, record.processed_page, input_size)

                np.save(original_cache, original, allow_pickle=False)
                np.save(processed_cache, processed, allow_pickle=False)
                created += 1

                print(
                    f"  cached {record.page_number}/{total} "
                    f"(pair page {record.original_page})",
                    flush=True,
                )

    print(f"Cache ready: {created} new page pair(s), {total} selected page pair(s).")


def dataset_signature(manifest: dict, records: list[PageRecord], input_size: int) -> str:
    payload = {
        "pair_count": manifest.get("pairCount"),
        "pairs": [
            {
                "id": r.pair_id,
                "originalPdf": str(r.original_pdf),
                "processedPdf": str(r.processed_pdf),
                "originalPage": r.original_page,
                "processedPage": r.processed_page,
            }
            for r in records
        ],
        "input_size": input_size,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_signature: str,
    expected_loss_profile: str,
) -> tuple[int, int, int, float]:
    if not checkpoint_path.exists():
        return 0, 0, 0, float("nan")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("dataset_signature") != expected_signature:
        print("Existing checkpoint belongs to a different dataset/input size; starting fresh.")
        return 0, 0, 0, float("nan")

    if checkpoint.get("loss_profile", "mse") != expected_loss_profile:
        print("Existing checkpoint uses a different loss profile; starting fresh.")
        return 0, 0, 0, float("nan")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = int(checkpoint.get("epoch", 0))
    next_page_index = int(checkpoint.get("next_page_index", 0))
    global_step = int(checkpoint.get("global_step", 0))
    last_loss = float(checkpoint.get("last_loss", float("nan")))

    print(
        f"Resuming checkpoint: epoch {epoch + 1}, "
        f"page {next_page_index + 1}, global step {global_step}"
    )
    return epoch, next_page_index, global_step, last_loss


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_page_index: int,
    global_step: int,
    last_loss: float,
    signature: str,
    loss_profile: str,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "next_page_index": next_page_index,
            "global_step": global_step,
            "last_loss": last_loss,
            "dataset_signature": signature,
            "loss_profile": loss_profile,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(checkpoint_path)


def load_page_tensor(path: Path, input_size: int) -> torch.Tensor:
    array = np.load(path, mmap_mode="r")
    if array.shape != (input_size, input_size, 3) or array.dtype != np.uint8:
        raise ValueError(f"Invalid cache page {path}: expected uint8 [{input_size},{input_size},3]")

    tensor = torch.from_numpy(np.array(array, copy=True))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    return tensor.contiguous(memory_format=torch.channels_last)


def main() -> int:
    args = parse_args()

    if args.input_size != INPUT_SIZE:
        raise ValueError(f"INPUT_SIZE must remain {INPUT_SIZE}; got {args.input_size}.")
    if args.batch_size != 1:
        raise ValueError("This laptop trainer intentionally uses batch size 1.")
    if args.max_pages < 0:
        raise ValueError("--max-pages cannot be negative.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    data_dir = Path(args.data_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    cache_root = data_dir / "cache256"
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    loss_log = checkpoint_dir / "loss.csv"

    torch.set_num_threads(min(6, max(1, os.cpu_count() or 1)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device("cpu")
    print("tiny-imgai Python local trainer")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Input: {args.input_size}x{args.input_size}")
    print(f"Batch: {args.batch_size}")
    print(f"Loss profile: {args.loss_profile}")
    print(f"Epoch target: {args.epochs}")
    print(f"Max pages: {'all' if args.max_pages == 0 else args.max_pages}")
    print()

    manifest, all_records = load_manifest(data_dir)
    records = all_records[: args.max_pages] if args.max_pages else all_records
    if not records:
        raise ValueError("No PDF pages found in the manifest.")

    print(f"Manifest document pairs: {len(manifest.get('pairs', []))}")
    print(f"Selected page pairs: {len(records)}")

    build_cache(records, cache_root, args.input_size)

    signature = dataset_signature(manifest, records, args.input_size)

    model = TinyUNet(args.input_size).to(device)
    model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-7,
    )
    criterion = PrintCleanLoss() if args.loss_profile == "print-clean-v1" else nn.MSELoss()

    total_state_params = sum(t.numel() for t in model.state_dict().values())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params}")
    print(f"Model state values (including BatchNorm running stats): {total_state_params}")

    start_epoch, next_page_index, global_step, last_loss = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        signature,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not loss_log.exists():
        loss_log.write_text("epoch,page,global_step,loss,seconds_per_page,timestamp\n", encoding="utf-8")

    deadline = time.monotonic() + max(0.1, args.max_minutes * 60)
    epoch = start_epoch

    while epoch < args.epochs:
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.monotonic()

        page_index = next_page_index if epoch == start_epoch else 0
        print(f"\n== epoch {epoch + 1}/{args.epochs} ==")

        while page_index < len(records):
            if time.monotonic() >= deadline:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    page_index,
                    global_step,
                    last_loss,
                    signature,
                    args.loss_profile,
                )
                print("Time budget reached. Checkpoint saved; run again to resume.")
                return 0

            record = records[page_index]
            original_cache, processed_cache = record.cache_paths(cache_root)

            start_time = time.monotonic()
            x = load_page_tensor(original_cache, args.input_size).to(device)
            y = load_page_tensor(processed_cache, args.input_size).to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            elapsed = max(1e-6, time.monotonic() - start_time)
            last_loss = loss_value
            epoch_loss += loss_value
            epoch_steps += 1
            global_step += 1

            page_index += 1

            with loss_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{epoch},{record.page_number},{global_step},{loss_value:.8f},"
                    f"{elapsed:.4f},{time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                )

            print(
                f"page {record.page_number}/{len(records)} "
                f"loss {loss_value:.5f} "
                f"step {global_step} "
                f"{elapsed:.2f}s/page",
                flush=True,
            )

            del prediction, loss, x, y

            if (
                args.checkpoint_every_pages > 0
                and page_index < len(records)
                and page_index % args.checkpoint_every_pages == 0
            ):
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    page_index,
                    global_step,
                    last_loss,
                    signature,
                    args.loss_profile,
                )
                print(f"  checkpoint saved at page {page_index}")

        average_loss = epoch_loss / max(1, epoch_steps)
        epoch_seconds = time.monotonic() - epoch_start

        print(
            f"== epoch {epoch + 1} complete · avg loss {average_loss:.5f} · "
            f"{epoch_seconds:.1f}s =="
        )

        epoch += 1
        next_page_index = 0

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            next_page_index,
            global_step,
            last_loss,
            signature,
            args.loss_profile,
        )
        print(f"checkpoint saved: {checkpoint_path}")

    print("\nTraining target reached.")
    print(f"Final checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
