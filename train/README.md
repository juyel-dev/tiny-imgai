# tiny-imgai training pipeline

Trains the U-Net model in `/core/model.js` on original/processed PDF
page pairs. Runs identically on a laptop or on GitHub Actions — no
native GPU drivers, no cloud account, no vendor lock-in. Every run is
checkpointed and resumable.

## 1. Add the dataset (5k pages)

Put matched PDF pairs here, one PDF per document (each can have many
pages — pages are paired by page number within a document):

```
train/data/originals/<id>.pdf
train/data/processed/<id>.pdf
```

`<id>` can be anything, as long as the same id exists on both sides
(e.g. `chapter-04.pdf` in both `originals/` and `processed/`). You
don't have to split into one-PDF-per-page — a single PDF can hold
hundreds of pages.

PDFs are tracked with **Git LFS** (see `.gitattributes`) so the repo
itself stays fast to clone even with thousands of pages:

```bash
git lfs install        # once, per machine
git add train/data/originals/*.pdf train/data/processed/*.pdf
git commit -m "Add dataset batch"
git push
```

## 2. Build the manifest

This is the "help me manage this many pages" part — it scans both
folders, matches pairs by filename, and tells you about anything
mismatched instead of silently skipping it:

```bash
cd train
npm ci
node build-manifest.js
```

Writes `train/data/manifest.json`. Re-run it any time you add more
PDFs — safe to run repeatedly.

## 3. Train

```bash
node train.js
```

Env vars (all optional):

| Var | Default | Meaning |
|---|---|---|
| `MAX_MINUTES` | 300 | Wall-clock budget for this run. Exits cleanly and checkpoints when reached — safe to re-run to continue. |
| `EPOCHS` | 1000 | Target epoch (training resumes toward this across runs). |
| `BATCH_SIZE` | 4 | Pages per gradient step. |
| `INPUT_SIZE` | 256 | Page render resolution (square, letterboxed). |
| `LEARNING_RATE` | 0.001 | Adam learning rate. |

Checkpoints (`model.json`, `weights.bin`, `state.json`, `loss.log`)
are written to `train/checkpoints/model/`. Running `train.js` again
with that folder present **resumes from it** rather than starting
over.

### Backend

`train.js` uses the fastest backend it can find, and never hard-fails
if the fast one isn't available:

1. `@tensorflow/tfjs-node` (native, fast) — install separately if you
   want it: `npm install @tensorflow/tfjs-node`. It's an
   `optionalDependency`, so a plain `npm ci` never fails because of it.
2. Plain-JS `cpu` backend — always available, zero native/network
   dependency, just slower. This is what runs on a fresh GitHub
   Actions checkout unless you add the native package.

(The WASM backend is deliberately not used for training — as of
tf.js 4.x it has no `Conv2DBackpropFilter` kernel, i.e. it can't run
backprop through conv layers at all. Inference-only.)

## 4. GitHub Actions

`.github/workflows/train.yml` — trigger manually from the Actions
tab (`workflow_dispatch`). It checks out the repo (LFS included),
installs deps, builds the manifest, trains for `max_minutes`, then
commits the updated checkpoint back to `main`. Public repo = free,
unlimited-minute runners. Trigger it again any time to keep training
where the last run left off.

## Windows / PowerShell laptop mode

For local training on Windows, put matching PDFs in:

```
train/data/originals/<id>.pdf
train/data/processed/<id>.pdf
```

Then from PowerShell:

```powershell
cd .\\train
.\\train-local.ps1
````

The launcher uses **256×256**, **batch 1**, and a **30-minute run budget** by default. The run resumes from `train/checkpoints/model/` on the next start. You can change the budget without editing code:

```powershell
.\\train-local.ps1 -MaxMinutes 60 -BatchSize 1
````

The browser and Node.js are not involved in training. The local Node trainer uses the native TensorFlow.js Node backend when it is available, otherwise it falls back to the pure JavaScript CPU backend.


Python scans `data/originals` and `data/processed` directly and matches PDFs by filename, so the local training workflow does not require the Node.js trainer or `manifest.json`.
