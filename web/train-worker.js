import { TinyImageModel, INPUT_SIZE, canvasToRGB8 } from "./model.js";
import { getDocument, getCachedPage, cachePage } from "./dataset.js";
import { renderPdfPageToCanvas } from "../core/render.js";

const PDFJS_MODULE = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.3.289/build/pdf.mjs";
let pdfjsPromise = null;

async function loadPdfJs() {
  if (!pdfjsPromise) pdfjsPromise = import(PDFJS_MODULE);
  return pdfjsPromise;
}

async function openPdfInWorker(blob) {
  const lib = await loadPdfJs();
  const data = new Uint8Array(await blob.arrayBuffer());
  return lib.getDocument({ data, disableWorker: true }).promise;
}

function releaseCanvas(canvas) {
  if (!canvas) return;
  canvas.width = 0;
  canvas.height = 0;
}

async function renderPage(pdf, pageNumber, size) {
  if (typeof OffscreenCanvas !== "function") {
    throw new Error("OffscreenCanvas is unavailable in this browser.");
  }

  return renderPdfPageToCanvas(
    pdf,
    pageNumber,
    size,
    (w, h) => new OffscreenCanvas(w, h)
  );
}

function groupedPairs(pairs) {
  const groups = new Map();
  for (const p of pairs) {
    if (!p.originalDocId || !p.processedDocId) continue;
    const key = p.originalDocId + "|" + p.processedDocId;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  }
  return groups;
}

async function prepareDataset(pairs) {
  const groups = groupedPairs(pairs);
  if (!groups.size) throw new Error("No PDF-backed training pairs found.");

  const allPairs = [...groups.values()].flat();
  const trainingData = new Map();
  let created = 0;

  self.postMessage({ type: "phase", phase: "prepare", total: allPairs.length });

  for (const group of groups.values()) {
    const first = group[0];
    const [od, pd] = await Promise.all([
      getDocument(first.originalDocId),
      getDocument(first.processedDocId),
    ]);

    const [opdf, ppdf] = await Promise.all([
      openPdfInWorker(od.blob),
      openPdfInWorker(pd.blob),
    ]);

    try {
      for (const pair of group) {
        let cached = await getCachedPage(pair.uuid);

        if (!(cached?.size === INPUT_SIZE && cached.original && cached.target)) {
          const originalCanvas = await renderPage(opdf, pair.originalPage, INPUT_SIZE);
          const original = canvasToRGB8(originalCanvas);
          releaseCanvas(originalCanvas);

          const targetCanvas = await renderPage(ppdf, pair.processedPage, INPUT_SIZE);
          const target = canvasToRGB8(targetCanvas);
          releaseCanvas(targetCanvas);

          cached = await cachePage(pair.uuid, {
            size: INPUT_SIZE,
            original,
            target,
          });
          created++;
        }

        trainingData.set(pair.uuid, {
          original: new Uint8Array(cached.original),
          target: new Uint8Array(cached.target),
        });

        const completed = trainingData.size;
        if (completed === 1 || completed % 10 === 0 || completed === allPairs.length) {
          self.postMessage({
            type: "progress",
            phase: "prepare",
            completed,
            total: allPairs.length,
            created,
          });
        }
      }
    } finally {
      await Promise.allSettled([
        typeof opdf.destroy === "function" ? opdf.destroy() : opdf.cleanup?.(),
        typeof ppdf.destroy === "function" ? ppdf.destroy() : ppdf.cleanup?.(),
      ]);
    }
  }

  return { allPairs, trainingData, created };
}

async function setupTensorFlow() {
  const tf = await import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/+esm");
  globalThis.tf = tf;
  if (typeof tf.enableProdMode === "function") tf.enableProdMode();

  try {
    await tf.setBackend("webgl");
  } catch (error) {
    console.warn("[tiny-imgai worker] WebGL unavailable; falling back to CPU.", error);
    await tf.setBackend("cpu");
  }

  await tf.ready();
  return { tf, backend: tf.getBackend() };
}

function memorySnapshot(tf, stage) {
  if (!tf || typeof tf.memory !== "function") return null;
  const m = tf.memory();
  return {
    stage,
    tensors: m.numTensors,
    bytes: m.numBytes,
    gpuBytes: typeof m.numBytesInGPU === "number" ? m.numBytesInGPU : null,
  };
}

async function runTraining(pairs) {
  const { tf, backend } = await setupTensorFlow();
  const model = new TinyImageModel({
    version: 0,
  });

  try {
    try {
      await model.loadFromBrowserStorage();
    } catch {
      await model.init();
    }

    self.postMessage({
      type: "ready",
      backend,
      parameterCount: model.parameterCount,
      architecture: model.architecture,
      memory: memorySnapshot(tf, "model-ready"),
    });

    const prepared = await prepareDataset(pairs);
    const allPairs = prepared.allPairs;
    const trainingData = prepared.trainingData;

    const epochs = 20;
    const batchSize = 1;

    self.postMessage({
      type: "phase",
      phase: "training",
      total: allPairs.length,
      created: prepared.created,
      batchSize,
      memoryBytes: [...trainingData.values()].reduce(
        (sum, item) => sum + item.original.byteLength + item.target.byteLength,
        0
      ),
    });

    for (let epoch = 1; epoch <= epochs; epoch++) {
      let epochLoss = 0;
      let processed = 0;

      for (let start = 0; start < allPairs.length; start += batchSize) {
        const pair = allPairs[start];
        const item = trainingData.get(pair.uuid);

        try {
          const loss = await model.trainBatch(
            [item.original],
            [item.target],
            INPUT_SIZE
          );
          epochLoss += loss;
          processed++;
          
          const completed = start + 1;
          if (completed === 1 || completed % 10 === 0 || completed === allPairs.length) {
            self.postMessage({
              type: "progress",
              phase: "training",
              epoch,
              epochs,
              completed,
              total: allPairs.length,
              loss,
              memory: memorySnapshot(tf, `train-e${epoch}-p${completed}`),
            });
          }
        } catch (error) {
          error.message = `Training failed at epoch ${epoch}, page ${start + 1}: ${error.message}`;
          throw error;
        }
      }

      const avgLoss = epochLoss / Math.max(1, processed);
      model.version = epoch;

      self.postMessage({
        type: "epoch",
        epoch,
        epochs,
        loss: avgLoss,
      });

      await model.saveToBrowserStorage();

      self.postMessage({
        type: "checkpoint",
        epoch,
        epochs,
        loss: avgLoss,
        memory: memorySnapshot(tf, `checkpoint-${epoch}`),
      });
    }

    const finalVersion = model.version;
    model.dispose();
    self.postMessage({
      type: "done",
      version: finalVersion,
      backend,
      memory: memorySnapshot(tf, "done"),
    });
  }

self.addEventListener("message", async (event) => {
  if (!event.data || event.data.type !== "start") return;

  try {
    await runTraining(event.data.pairs || []);
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error?.message || String(error),
      stack: error?.stack || "",
    });
  }
});
