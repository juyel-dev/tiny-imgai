# tiny-imgai

Browser-side image cleanup model for converting dark/colored handwritten
note pages into print-friendly pages with a white background and dark
handwriting/diagrams. Training data is supplied as matched original/processed
PDF pairs.

## Current production path

The validated development target is:

- **512×512** input/output
- RGB input/output
- 3-level U-Net
- **base width 48** → **48/96/192/384**
- batch size 1
- `print-clean-v1` loss
- local **PyTorch CPU** training
- 6 CPU threads on the tested laptop
- BatchNorm recalibration before deployment

The width selection was made with a process-isolated holdout screen on
64 training pages and 16 validation pages. The selected b48 run reached:

- dark-pixel F1: **0.707689**
- MAE: **0.107083**
- PSNR: **18.572 dB**
- peak RSS: **1045.9 MiB**

b64 used more memory and was slower in the same screen while producing lower
validation metrics, so no further width expansion is planned at this stage.

## Train locally

Put the matching PDFs here:

```
train/data/originals/<id>.pdf
train/data/processed/<id>.pdf
```

The Python pipeline matches PDF filenames, pairs pages by page number, and
renders a local `cache512/` for training.

Install/run from PowerShell:

```powershell
cd "C:\Users\JUYEL\Documents\tiny-imgai"
git pull
cd ".\train"
.\run-production.ps1
```

That one command runs the resumable b48/512 training, finalizes BatchNorm
when the requested epoch is reached, evaluates the complete selected set,
and prepares a strict tf.js export.

The default target is 3 epochs over all available pages, with a 120-minute
wall-clock budget per invocation. A timeout or RAM guard saves progress; run
the same command again to resume.

## Outputs

After a completed run, the local `train/` tree contains:

```
checkpoints/
  scaled-512-b48-eager/checkpoint.pt
  tfjs-512-b48/model.json
  tfjs-512-b48/weights.bin
  tfjs-512-b48/metadata.json
evaluation/
  scale512-*/predictions/
  scale512-*/predictions-512.pdf
  scale512-*/comparison-previews-512.pdf
  scale512-*/metrics.csv
  scale512-*/metrics.json
```

The generated checkpoints/evaluation artifacts are intentionally ignored by
git. The tf.js exporter verifies every weight name and shape against the
shared U-Net before saving the browser model.

## Repository layout

```
/web/       Browser training/inference UI
/core/      Shared tf.js U-Net architecture and PDF rendering helpers
/train/     Local Python production trainer + legacy Node/tf.js trainer
```

The older 256px Node/browser training path remains in the repository for
compatibility, but it is not the validated production training path.
