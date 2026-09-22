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

## Deployment (Vercel)

**Root Directory must stay at the repo default (blank) — do NOT set it to
`web`.** `web/model.js` and `web/pdf.js` import from `/core` (shared with
`/train`), which sits *outside* `web/`. Vercel's Root Directory setting
excludes everything outside it from the deployment, so setting it to `web`
breaks those imports and the app fails to load.

`vercel.json` at the repo root handles routing instead: it rewrites `/` to
`/web/index.html` while the whole repo (including `/core`) stays deployed
and reachable at its normal path.

## Status

Migrating from a hand-written WebGPU-shader model (3×3 conv, 443
params — too small to learn the real transform, which is
context-dependent recoloring + added annotation, not local pixel
correction) to a proper U-Net trained via `/train`, with the browser
app doing inference-only once weights are trained. See `train/README.md`
to add your dataset and start training.
