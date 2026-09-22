import { buildModel, INPUT_SIZE } from "../core/model.js";

export { INPUT_SIZE };

// Raw RGB bytes (no alpha) — same shape the IndexedDB page cache stores.
export function canvasToRGB8(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4) {
    out[j++] = data[i];
    out[j++] = data[i + 1];
    out[j++] = data[i + 2];
  }
  return out;
}

export function imageToCanvas(source, size = INPUT_SIZE) {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  c.getContext("2d", { willReadFrequently: true }).drawImage(source, 0, 0, size, size);
  return c;
}

function batchBytesToTensor(bytesArray, size) {
  const count = bytesArray.length;
  const pixels = size * size * 3;
  const packed = new Float32Array(count * pixels);

  for (let b = 0; b < count; b++) {
    const src = bytesArray[b];
    if (!src || src.length !== pixels) {
      throw new Error(`Invalid RGB page tensor: expected ${pixels} bytes, got ${src?.length ?? "none"}.`);
    }
    const offset = b * pixels;
    for (let i = 0; i < pixels; i++) packed[offset + i] = src[i] / 255;
  }

  return globalThis.tf.tensor4d(packed, [count, size, size, 3], "float32");
}

const OPTIMIZER_LR = 1e-3;

export class TinyImageModel {
  constructor({ version = 0 } = {}) {
    this.version = version;
    this.model = null;
  }

  get parameterCount() {
    return this.model ? this.model.countParams() : 0;
  }

  get architecture() {
    return "U-Net (enc 16→32→64 → bottleneck 128 → dec 64→32→16), tf.js";
  }

  compile() {
    if (!this.model) throw new Error("Model is not initialized.");
    this.model.compile({
      optimizer: globalThis.tf.train.adam(OPTIMIZER_LR),
      loss: "meanSquaredError",
    });
  }

  dispose() {
    if (!this.model) return;
    const optimizer = this.model.optimizer;
    try {
      this.model.dispose();
    } finally {
      if (optimizer && typeof optimizer.dispose === "function") optimizer.dispose();
      this.model = null;
    }
  }

  async init() {
    const tf = globalThis.tf;
    if (!tf) throw new Error("tf.js did not load (check network / CDN block)");
    this.dispose();
    this.model = buildModel(tf, INPUT_SIZE);
    this.compile();
  }

  async loadFromBrowserStorage() {
    const tf = globalThis.tf;
    if (!tf) throw new Error("tf.js did not load (check network / CDN block)");
    const nextModel = await tf.loadLayersModel("indexeddb://tiny-imgai-model");
    this.dispose();
    this.model = nextModel;
    this.compile();
  }

  async saveToBrowserStorage() {
    if (!this.model) throw new Error("Model is not initialized.");
    await this.model.save("indexeddb://tiny-imgai-model");
  }

  async trainBatch(inputBytesArray, targetBytesArray, size = INPUT_SIZE) {
    if (!this.model) throw new Error("Model is not initialized.");
    if (inputBytesArray.length !== targetBytesArray.length || !inputBytesArray.length) {
      throw new Error("Input/target batch mismatch.");
    }

    const tf = globalThis.tf;
    let x = null;
    let y = null;

    try {
      x = batchBytesToTensor(inputBytesArray, size);
      y = batchBytesToTensor(targetBytesArray, size);
      // One explicit gradient update for this batch. This avoids
      // creating a full fit() history/callback cycle for every batch.
      const result = await this.model.trainOnBatch(x, y);
      const loss = Array.isArray(result) ? result[0] : result;
      return Number(loss);
    } finally {
      tf.dispose([x, y]);
    }
  }

  async predict(canvas) {
    if (!this.model) throw new Error("Model is not initialized.");
    const tf = globalThis.tf;
    const size = canvas.width;
    const x = tf.tidy(() =>
      tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0)
    );

    let y = null;
    let out = null;
    try {
      y = this.model.predict(x);
      out = tf.tidy(() => y.squeeze([0]).clipByValue(0, 1));

      const outCanvas = document.createElement("canvas");
      outCanvas.width = size;
      outCanvas.height = size;
      await tf.browser.toPixels(out, outCanvas);
      return outCanvas;
    } finally {
      tf.dispose([x, y, out]);
    }
  }

  async exportWeights() {
    if (!this.model) throw new Error("Model is not initialized.");
    await this.model.save(`downloads://tiny-imgai-model-v${this.version}`);
  }

  async loadWeights(fileList) {
    const tf = globalThis.tf;
    if (!fileList?.length) throw new Error("No model files selected.");
    const nextModel = await tf.loadLayersModel(
      tf.io.browserFiles(Array.from(fileList))
    );
    this.dispose();
    this.model = nextModel;
    this.compile();
  }
}
