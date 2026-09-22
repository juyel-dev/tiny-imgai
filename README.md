# tiny-imgai

Browser-native tool for turning plain handwritten notes into
recolored, annotated versions — trained on your own before/after PDF
pairs, no server, no cloud AI API.

## Repo layout

```
/web/      Static browser app (Vercel). Zero build step.
/core/     Shared model architecture (U-Net) — imported unmodified
           by both /web (browser inference) and /train (Node training).
/train/    Node.js training pipeline — runs on a laptop or GitHub
           Actions, see train/README.md.
.github/workflows/train.yml   Manually-triggered, resumable training runs.
```

## ⚠️ Deployment note

`index.html` moved from the repo root into `/web/`. **Update the
Vercel project's Root Directory setting to `web`** (Project Settings →
General → Root Directory) or the live deploy will 404 on the next
build.

## Status

Migrating from a hand-written WebGPU-shader model (3×3 conv, 443
params — too small to learn the real transform, which is
context-dependent recoloring + added annotation, not local pixel
correction) to a proper U-Net trained via `/train`, with the browser
app doing inference-only once weights are trained. See `train/README.md`
to add your dataset and start training.
