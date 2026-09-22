import { buildModel } from "../core/model.js";

const DEFAULT_MODEL_URL = "/web/models/tfjs-512-b48/model.json";
const DEFAULT_METADATA_URL = "/web/models/tfjs-512-b48/metadata.json";
const OUTPUT_SIZE = 512;
const TILE_SIZE = 320;
const TILE_POSITIONS = [0, OUTPUT_SIZE - TILE_SIZE];
const EXPECTED_PARAMS = 4387299;
const CHANNELS = [48, 96, 192, 384];

let tf = null;
let model = null;
let ready = false;
let busy = false;

function fail(message, error) {
  self.postMessage({
    type: "error",
    message,
    stack: error?.stack || "",
  });
}

async function init(modelUrl = DEFAULT_MODEL_URL, metadataUrl = DEFAULT_METADATA_URL) {
  try {
    const tfModule = await import(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/+esm"
    );
    tf = tfModule;

    if (typeof tf.enableProdMode === "function") {
      tf.enableProdMode();
    }

    // The 512px b48 model is intentionally executed on the CPU backend.
    // Its full-resolution WebGL activation textures can exhaust Chrome's
    // GPU process memory and crash the browser.
    await tf.setBackend("cpu");
    await tf.ready();

    const response = await fetch(metadataUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Production model metadata HTTP " + response.status);
    }

    const metadata = await response.json();
    if (
      Number(metadata.input_size) !== OUTPUT_SIZE ||
      Number(metadata.base_channels) !== 48 ||
      !metadata.bn_calibrated
    ) {
      throw new Error(
        "Invalid production model metadata: expected calibrated 512px b48."
      );
    }

    const sourceModel = await tf.loadLayersModel(modelUrl);
    const actualParams = sourceModel.countParams();
    if (actualParams !== EXPECTED_PARAMS) {
      sourceModel.dispose();
      throw new Error(
        "Production model parameter mismatch: expected " +
          EXPECTED_PARAMS +
          ", got " +
          actualParams
      );
    }

    // Rebuild the same fully-convolutional network at 320px and copy the
    // exact trained weights. This keeps the trained 512px checkpoint but
    // bounds activation memory during browser inference.
    model = buildModel(tf, TILE_SIZE, CHANNELS);
    const sourceWeights = sourceModel.getWeights();
    const copiedWeights = sourceWeights.map((weight) => weight.clone());
    model.setWeights(copiedWeights);

    copiedWeights.forEach((weight) => weight.dispose());
    sourceWeights.forEach((weight) => weight.dispose());
    sourceModel.dispose();

    if (model.countParams() !== EXPECTED_PARAMS) {
      model.dispose();
      model = null;
      throw new Error("Safe inference model parameter count changed.");
    }

    ready = true;
    self.postMessage({
      type: "ready",
      backend: tf.getBackend(),
      parameterCount: model.countParams(),
      architecture:
        "U-Net 48→96→192→384→192→96→48 · 512px tiled CPU inference",
      tileSize: TILE_SIZE,
      tilesPerImage: TILE_POSITIONS.length * TILE_POSITIONS.length,
      memory: tf.memory(),
    });
  } catch (error) {
    fail(error?.message || String(error), error);
  }
}

async function predictTile(rgb, x0, y0, output, counts) {
  const packed = new Float32Array(TILE_SIZE * TILE_SIZE * 3);

  for (let ty = 0; ty < TILE_SIZE; ty++) {
    const srcRow = (y0 + ty) * OUTPUT_SIZE;
    const dstRow = ty * TILE_SIZE;

    for (let tx = 0; tx < TILE_SIZE; tx++) {
      const srcIndex = (srcRow + x0 + tx) * 3;
      const dstIndex = (dstRow + tx) * 3;
      packed[dstIndex] = rgb[srcIndex] / 255;
      packed[dstIndex + 1] = rgb[srcIndex + 1] / 255;
      packed[dstIndex + 2] = rgb[srcIndex + 2] / 255;
    }
  }

  let x = null;
  let y = null;

  try {
    x = tf.tensor4d(
      packed,
      [1, TILE_SIZE, TILE_SIZE, 3],
      "float32"
    );
    y = model.predict(x);

    const values = await y.data();

    for (let ty = 0; ty < TILE_SIZE; ty++) {
      const outRow = (y0 + ty) * OUTPUT_SIZE;
      const tileRow = ty * TILE_SIZE;

      for (let tx = 0; tx < TILE_SIZE; tx++) {
        const outPixel = outRow + x0 + tx;
        const tilePixel = tileRow + tx;
        const outIndex = outPixel * 3;
        const tileIndex = tilePixel * 3;

        output[outIndex] += values[tileIndex];
        output[outIndex + 1] += values[tileIndex + 1];
        output[outIndex + 2] += values[tileIndex + 2];
        counts[outPixel] += 1;
      }
    }
  } finally {
    if (y) y.dispose();
    if (x) x.dispose();
    if (typeof tf.nextFrame === "function") {
      await tf.nextFrame();
    }
  }
}

async function infer(id, buffer) {
  if (!ready || !model) {
    throw new Error("Production model is not ready.");
  }
  if (busy) {
    throw new Error("Inference worker is busy.");
  }

  busy = true;

  try {
    const rgb = new Uint8Array(buffer);
    const expectedBytes = OUTPUT_SIZE * OUTPUT_SIZE * 3;

    if (rgb.byteLength !== expectedBytes) {
      throw new Error(
        "Invalid input size: expected " + expectedBytes + " RGB bytes."
      );
    }

    const output = new Float32Array(expectedBytes);
    const counts = new Uint8Array(OUTPUT_SIZE * OUTPUT_SIZE);

    for (const y0 of TILE_POSITIONS) {
      for (const x0 of TILE_POSITIONS) {
        await predictTile(rgb, x0, y0, output, counts);
      }
    }

    const rgba = new Uint8ClampedArray(OUTPUT_SIZE * OUTPUT_SIZE * 4);

    for (let pixel = 0, rgbIndex = 0, rgbaIndex = 0; pixel < counts.length; pixel++) {
      const count = Math.max(1, counts[pixel]);
      rgba[rgbaIndex] = Math.max(
        0,
        Math.min(255, Math.round((output[rgbIndex] / count) * 255))
      );
      rgba[rgbaIndex + 1] = Math.max(
        0,
        Math.min(255, Math.round((output[rgbIndex + 1] / count) * 255))
      );
      rgba[rgbaIndex + 2] = Math.max(
        0,
        Math.min(255, Math.round((output[rgbIndex + 2] / count) * 255))
      );
      rgba[rgbaIndex + 3] = 255;
      rgbIndex += 3;
      rgbaIndex += 4;
    }

    self.postMessage(
      {
        type: "result",
        id,
        width: OUTPUT_SIZE,
        height: OUTPUT_SIZE,
        buffer: rgba.buffer,
      },
      [rgba.buffer]
    );
  } catch (error) {
    self.postMessage({
      type: "result-error",
      id,
      message: error?.message || String(error),
      stack: error?.stack || "",
    });
  } finally {
    busy = false;
    if (typeof tf?.nextFrame === "function") {
      await tf.nextFrame();
    }
  }
}

self.addEventListener("message", async (event) => {
  const message = event.data || {};

  if (message.type === "init") {
    if (!ready) {
      await init(message.modelUrl, message.metadataUrl);
    }
    return;
  }

  if (message.type === "infer") {
    await infer(message.id, message.buffer);
  }
});
