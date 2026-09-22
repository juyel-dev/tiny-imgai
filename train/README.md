# tiny-imgai training pipeline

The validated production path is **local Python + PyTorch CPU**. The current
production target is:

- 512×512 input/output
- RGB input and RGB output
- same 3-level U-Net topology as the original model
- base width **48** → **48/96/192/384** channels
- batch size 1
- `print-clean-v1` loss
- CPU, normally 6 threads on the tested laptop
- BatchNorm recalibration before deployment

## Dataset

Put matching PDF pairs here:

```
train/data/originals/<id>.pdf
train/data/processed/<id>.pdf
```

Pages are paired by page number inside each PDF. The development dataset has
335 paired pages.

The Python trainer reads the PDFs directly and stores rendered uint8 RGB
pages in `train/data/cache512/`.

## Production training

From PowerShell:

```powershell
cd "C:\Users\JUYEL\Documents\tiny-imgai\train"
.\train-scale.ps1
```

Defaults:

- b48
- 512×512
- 3 epochs
- 120 minute wall-clock budget
- batch 1
- 6 CPU threads
- minimum available RAM 768 MiB
- maximum process RSS 2048 MiB
- checkpoint every 10 pages

The run is resumable. A time-budget or RAM-guard stop writes the checkpoint
and the next run resumes from the recorded epoch/page.

When the target epoch is reached, the trainer recalculates BatchNorm running
statistics across all selected pages and marks the checkpoint deployable.

## Evaluate

```powershell
.\evaluate-scale.ps1
```

The 512px evaluator verifies architecture and dataset signatures before
loading the checkpoint. It writes per-page predictions, a 512px prediction
PDF, comparison previews, CSV/JSON metrics, and memory statistics.

Reported metrics include MAE, PSNR, dark-pixel F1, foreground MAE, background
whiteness error, white-background rate, and color residual.

## BatchNorm finalization

For an already-completed checkpoint:

```powershell
.\finalize-scale.ps1
```

This changes only BatchNorm running statistics. It does not retrain the
learned weights.

## Export to browser / tf.js

After training and BatchNorm finalization:

```powershell
.\export-tfjs.ps1
```

The exporter has two strict stages. Python reads the PyTorch checkpoint and
writes an ordered float32 staging bundle. Node/tf.js builds the exact shared
U-Net from `/core/model.js`, checks every weight name and shape, loads the
weights, verifies the 512×512×3 output and b48 parameter count, then writes:

```
train/checkpoints/tfjs-512-b48/model.json
train/checkpoints/tfjs-512-b48/weights.bin
train/checkpoints/tfjs-512-b48/metadata.json
```

The generated model remains ignored by git through `train/checkpoints/`.

## Older paths

`train.py` and `train.js` are retained for compatibility with the earlier
256px workflow. They are not the validated production training path.

The browser training worker is also retained, but the production workflow is
now local Python training followed by tf.js export for browser inference.
