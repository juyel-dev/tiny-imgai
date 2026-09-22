// Training loop for tiny-imgai. Runs on plain Node (laptop or GitHub
// Actions). Backend auto-selects: native @tensorflow/tfjs-node if
// installed (fast), else falls back to the pure-JS cpu backend (slower,
// zero native/network dependency) — see the backend-selection block in
// main() for why. Either way, no vendor/platform lock-in: same code
// runs anywhere Node runs.
//
// Resumable by design: every epoch, weights + progress are written to
// CHECKPOINT_DIR. A GH Actions run that hits its time limit exits
// cleanly mid-training, commits the checkpoint, and the next run
// picks up where it left off.
//
// Env vars:
//   DATA_DIR        default ./data      (must contain manifest.json — run build-manifest.js first)
//   CHECKPOINT_DIR   default ./checkpoints/model
//   BATCH_SIZE       default 4   (pages per gradient step)
//   MAX_MINUTES      default 300 (wall-clock budget for this run)
//   EPOCHS           default 1000 (target epoch; resumes toward this)
//   LEARNING_RATE    default 0.001

import fs from "node:fs";
import path from "node:path";
import * as tf from "@tensorflow/tfjs";
import { createCanvas } from "@napi-rs/canvas";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { fileURLToPath } from "node:url";
import { buildModel, INPUT_SIZE } from "../core/model.js";
import { renderPdfPageToCanvas } from "../core/render.js";

// pdfjs-dist ships the standard 14 font metrics locally; without pointing
// at them explicitly it tries to fetch them over the network and can hang
// indefinitely in a sandboxed/offline CI environment instead of failing fast.
const STANDARD_FONTS_URL = `file://${path.join(path.dirname(fileURLToPath(import.meta.url)), "node_modules", "pdfjs-dist", "standard_fonts")}/`;

const DATA_DIR = process.env.DATA_DIR || path.join(process.cwd(), "data");
const CHECKPOINT_DIR = process.env.CHECKPOINT_DIR || path.join(process.cwd(), "checkpoints", "model");
const BATCH_SIZE = Number(process.env.BATCH_SIZE || 4);
const MAX_MINUTES = Number(process.env.MAX_MINUTES || 300);
const EPOCHS = Number(process.env.EPOCHS || 1000);
const TRAIN_INPUT_SIZE = Number(process.env.INPUT_SIZE) || INPUT_SIZE;
const LEARNING_RATE = Number(process.env.LEARNING_RATE || 1e-3);

function canvasToTensor(canvas) {
  const ctx = canvas.getContext("2d");
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const buf = new Float32Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4) {
    buf[j++] = data[i] / 255;
    buf[j++] = data[i + 1] / 255;
    buf[j++] = data[i + 2] / 255;
  }
  return tf.tensor3d(buf, [height, width, 3]);
}

async function saveModelToDisk(model, dir) {
  fs.mkdirSync(dir, { recursive: true });
  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      const weightsManifest = [{ paths: ["weights.bin"], weights: artifacts.weightSpecs }];
      const modelJson = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest,
      };
      fs.writeFileSync(path.join(dir, "model.json"), JSON.stringify(modelJson));
      fs.writeFileSync(path.join(dir, "weights.bin"), Buffer.from(artifacts.weightData));
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
    })
  );
}

async function loadModelFromDisk(dir) {
  const modelJson = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
  const weightData = fs.readFileSync(path.join(dir, "weights.bin"));
  const weightSpecs = modelJson.weightsManifest[0].weights;
  return tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs,
      weightData: weightData.buffer.slice(weightData.byteOffset, weightData.byteOffset + weightData.byteLength),
    })
  );
}

async function loadOrCreateModel() {
  const modelJsonPath = path.join(CHECKPOINT_DIR, "model.json");
  if (fs.existsSync(modelJsonPath)) {
    const model = await loadModelFromDisk(CHECKPOINT_DIR);
    const statePath = path.join(CHECKPOINT_DIR, "state.json");
    const startEpoch = fs.existsSync(statePath) ? JSON.parse(fs.readFileSync(statePath, "utf8")).epoch || 0 : 0;
    console.log(`Resumed checkpoint from ${CHECKPOINT_DIR}, epoch ${startEpoch}`);
    return { model, startEpoch };
  }
  const model = buildModel(tf, TRAIN_INPUT_SIZE);
  console.log(`New model. Trainable params: ${model.countParams()}`);
  return { model, startEpoch: 0 };
}

async function main() {
  // Backend selection, fastest available first, always falls back to
  // something that works with zero native/network dependencies:
  //   1. @tensorflow/tfjs-node (native, fastest) — optional. Only used if
  //      already installed (`npm install @tensorflow/tfjs-node` separately
  //      in this folder). Not a hard dependency: its native binary download
  //      can fail in sandboxed/offline environments, and this script must
  //      never hard-fail because of that.
  //   2. plain-JS 'cpu' backend (@tensorflow/tfjs core) — always available,
  //      pure JS, no native/wasm binary, works identically on any Node,
  //      any OS, any CI runner. Slower, but it is the guaranteed fallback.
  //   (WASM backend is deliberately not used for training: as of tfjs 4.x
  //   it has no Conv2DBackpropFilter kernel, i.e. it cannot run backprop
  //   through conv layers at all — inference-only.)
  try {
    await import("@tensorflow/tfjs-node");
    await tf.setBackend("tensorflow");
    console.log("Using native @tensorflow/tfjs-node backend (fast path).");
  } catch (err) {
    console.log("@tensorflow/tfjs-node not available, falling back to plain-JS cpu backend (slower, but zero native deps).");
    await tf.setBackend("cpu");
  }
  await tf.ready();
  console.log("tf backend:", tf.getBackend());

  const manifestPath = path.join(DATA_DIR, "manifest.json");
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`No manifest.json in ${DATA_DIR}. Run "node build-manifest.js" first.`);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (!manifest.pairs?.length) throw new Error("manifest.json has zero pairs.");
  console.log(`Loaded manifest: ${manifest.pairs.length} PDF pairs`);

  fs.mkdirSync(CHECKPOINT_DIR, { recursive: true });
  const { model, startEpoch } = await loadOrCreateModel();
  model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: "meanSquaredError" });

  const canvasFactory = (w, h) => createCanvas(w, h);
  const deadline = Date.now() + MAX_MINUTES * 60 * 1000;
  const lossLogPath = path.join(CHECKPOINT_DIR, "loss.log");

  epochLoop: for (let epoch = startEpoch; epoch < EPOCHS; epoch++) {
    let epochLoss = 0, steps = 0;

    for (const docPair of manifest.pairs) {
      if (Date.now() > deadline) break epochLoop;

      const origPath = path.join(DATA_DIR, docPair.originalPdf);
      const procPath = path.join(DATA_DIR, docPair.processedPdf);
      const pdfOpts = { standardFontDataUrl: STANDARD_FONTS_URL, disableFontFace: true };
      const [origDoc, procDoc] = await Promise.all([
        getDocument({ url: origPath, ...pdfOpts }).promise,
        getDocument({ url: procPath, ...pdfOpts }).promise,
      ]);
      const pageCount = Math.min(origDoc.numPages, procDoc.numPages);
      if (origDoc.numPages !== procDoc.numPages) {
        console.warn(`[${docPair.id}] page count mismatch (${origDoc.numPages} vs ${procDoc.numPages}) — using first ${pageCount}`);
      }

      for (let start = 1; start <= pageCount; start += BATCH_SIZE) {
        const pageNums = [];
        for (let p = start; p < Math.min(start + BATCH_SIZE, pageCount + 1); p++) pageNums.push(p);

        const inputs = [], targets = [];
        for (const p of pageNums) {
          const oCanvas = await renderPdfPageToCanvas(origDoc, p, TRAIN_INPUT_SIZE, canvasFactory);
          const tCanvas = await renderPdfPageToCanvas(procDoc, p, TRAIN_INPUT_SIZE, canvasFactory);
          inputs.push(canvasToTensor(oCanvas));
          targets.push(canvasToTensor(tCanvas));
        }

        const xBatch = tf.stack(inputs);
        const yBatch = tf.stack(targets);
        const history = await model.fit(xBatch, yBatch, { epochs: 1, batchSize: inputs.length, verbose: 0 });
        const loss = history.history.loss[0];
        epochLoss += loss; steps++;
        tf.dispose([xBatch, yBatch, ...inputs, ...targets]);

        console.log(`epoch ${epoch} [${docPair.id} p${start}] loss ${loss.toFixed(5)}`);
        if (Date.now() > deadline) { await origDoc.destroy(); await procDoc.destroy(); break epochLoop; }
      }
      await origDoc.destroy();
      await procDoc.destroy();
    }

    const avg = epochLoss / Math.max(1, steps);
    console.log(`== epoch ${epoch} avg loss ${avg.toFixed(5)} ==`);
    fs.appendFileSync(lossLogPath, `${epoch},${avg},${new Date().toISOString()}\n`);

    await saveModelToDisk(model, CHECKPOINT_DIR);
    fs.writeFileSync(path.join(CHECKPOINT_DIR, "state.json"), JSON.stringify({ epoch: epoch + 1, avgLoss: avg, updatedAt: new Date().toISOString() }, null, 2));
  }

  console.log("Run finished (time budget reached or target epoch hit). Checkpoint saved.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
