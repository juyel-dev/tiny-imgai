const MODEL_URL = "/web/models/tfjs-512-b48/model.json";
const METADATA_URL = "/web/models/tfjs-512-b48/metadata.json";
const INPUT_SIZE = 512;
const EXPECTED_PARAMS = 4387299;

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

async function init() {
  try {
    const tfModule = await import(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/+esm"
    );
    tf = tfModule;

    if (typeof tf.enableProdMode === "function") {
      tf.enableProdMode();
    }

    // Never use WebGL for the 512px b48 production model. WebGL texture
    // allocation for this U-Net can exhaust Chrome's GPU process memory.
    await tf.setBackend("cpu");
    await tf.ready();

    const response = await fetch(METADATA_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error("Production model metadata HTTP " + response.status);
    }

    const metadata = await response.json();
    if (
      Number(metadata.input_size) !== INPUT_SIZE ||
      Number(metadata.base_channels) !== 48 ||
      !metadata.bn_calibrated
    ) {
      throw new Error(
        "Invalid production model metadata: expected calibrated 512px b48."
      );
    }

    model = await tf.loadLayersModel(MODEL_URL);

    if (model.countParams() !== EXPECTED_PARAMS) {
      model.dispose();
      model = null;
      throw new Error(
        "Production model parameter mismatch: expected " +
          EXPECTED_PARAMS +
          ", got " +
          model.countParams()
      );
    }

    ready = true;
    self.postMessage({
      type: "ready",
      backend: tf.getBackend(),
      parameterCount: model.countParams(),
      architecture: "U-Net 48→96→192→384→192→96→48 · 512px",
      memory: tf.memory(),
    });
  } catch (error) {
    fail(error?.message || String(error), error);
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

  let x = null;
  let y = null;

  try {
    const rgb = new Uint8Array(buffer);
    const pixels = INPUT_SIZE * INPUT_SIZE * 3;
    if (rgb.byteLength !== pixels) {
      throw new Error(
        "Invalid input size: expected " + pixels + " RGB bytes."
      );
    }

    const packed = new Float32Array(pixels);
    for (let i = 0; i < pixels; i++) {
      packed[i] = rgb[i] / 255;
    }

    x = tf.tensor4d(packed, [1, INPUT_SIZE, INPUT_SIZE, 3], "float32");
    y = model.predict(x);

    const values = await y.data();
    const rgba = new Uint8ClampedArray(INPUT_SIZE * INPUT_SIZE * 4);

    for (let i = 0, p = 0; i < values.length; i += 3, p += 4) {
      rgba[p] = Math.max(0, Math.min(255, Math.round(values[i] * 255)));
      rgba[p + 1] = Math.max(0, Math.min(255, Math.round(values[i + 1] * 255)));
      rgba[p + 2] = Math.max(0, Math.min(255, Math.round(values[i + 2] * 255)));
      rgba[p + 3] = 255;
    }

    self.postMessage(
      {
        type: "result",
        id,
        width: INPUT_SIZE,
        height: INPUT_SIZE,
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
    if (y) y.dispose();
    if (x) x.dispose();
    busy = false;
    if (typeof tf.nextFrame === "function") {
      await tf.nextFrame();
    }
  }
}

self.addEventListener("message", async (event) => {
  const message = event.data || {};

  if (message.type === "init") {
    if (!ready) await init();
    return;
  }

  if (message.type === "infer") {
    await infer(message.id, message.buffer);
  }
});
