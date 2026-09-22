import { TinyImageModel, INPUT_SIZE } from "./model.js";
import { getCachedPage } from "./dataset.js";

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

async function loadCachedDataset(pairs) {
  const trainingData = new Map();
  const allPairs = pairs.filter((p) => p.originalDocId && p.processedDocId);

  if (!allPairs.length) throw new Error("No PDF-backed training pairs found.");

  self.postMessage({ type: "phase", phase: "cache", total: allPairs.length });

  let prepared = 0;
  let bytes = 0;

  for (const pair of allPairs) {
    const cached = await getCachedPage(pair.uuid);

    if (!(cached?.size === INPUT_SIZE && cached.original && cached.target)) {
      throw new Error("Missing cached page " + pair.pageNumber + ". Reload and create the page cache first.");
    }

    const original = new Uint8Array(cached.original);
    const target = new Uint8Array(cached.target);

    trainingData.set(pair.uuid, { original, target });
    bytes += original.byteLength + target.byteLength;
    prepared++;

    if (prepared === 1 || prepared % 10 === 0 || prepared === allPairs.length) {
      self.postMessage({
        type: "progress",
        phase: "cache",
        completed: prepared,
        total: allPairs.length,
      });
    }
  }

  return { allPairs, trainingData, bytes };
}

async function runTraining(pairs) {
  const { tf, backend } = await setupTensorFlow();
  const model = new TinyImageModel({ version: 0 });

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

    const prepared = await loadCachedDataset(pairs);
    const allPairs = prepared.allPairs;
    const trainingData = prepared.trainingData;

    const epochs = 20;
    const batchSize = 1;

    self.postMessage({
      type: "phase",
      phase: "training",
      total: allPairs.length,
      batchSize,
      memoryBytes: prepared.bytes,
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
              memory: memorySnapshot(tf, "train-e" + epoch + "-p" + completed),
            });
          }
        } catch (error) {
          const message = error?.message || String(error);
          throw new Error(
            "Training failed at epoch " + epoch + ", page " + (start + 1) + ": " + message,
            { cause: error }
          );
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
        memory: memorySnapshot(tf, "checkpoint-" + epoch),
      });
    }

    const finalVersion = model.version;
    self.postMessage({
      type: "done",
      version: finalVersion,
      backend,
      memory: memorySnapshot(tf, "done"),
    });
  } finally {
    model.dispose();
  }
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
