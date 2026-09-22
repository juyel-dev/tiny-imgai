import { buildModel, INPUT_SIZE } from "../core/model.js";

export { INPUT_SIZE };

// Raw RGB bytes (no alpha) — same shape the IndexedDB page cache already
// stores, so cached pages can go straight to a tensor with no canvas
// round-trip.
export function canvasToRGB8(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const out = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4) {
    out[j++] = data[i]; out[j++] = data[i + 1]; out[j++] = data[i + 2];
  }
  return out;
}

export function imageToCanvas(source, size = INPUT_SIZE) {
  const c = document.createElement("canvas");
  c.width = size; c.height = size;
  c.getContext("2d", { willReadFrequently: true }).drawImage(source, 0, 0, size, size);
  return c;
}

function rgb8ToTensor(bytes, size) {
  return window.tf.tidy(() => window.tf.tensor3d(bytes, [size, size, 3], "float32").div(255));
}

const OPTIMIZER_LR = 1e-3;

export class TinyImageModel {
  constructor({ version = 0 } = {}) {
    this.version = version;
    this.model = null;
  }

  get parameterCount() { return this.model ? this.model.countParams() : 0; }
  get architecture() { return "U-Net (enc 16\u219232\u219264 \u2192 bottleneck 128 \u2192 dec 64\u219232\u219216), tf.js"; }

  compile() {
    this.model.compile({ optimizer: window.tf.train.adam(OPTIMIZER_LR), loss: "meanSquaredError" });
  }

  async init() {
    const tf = window.tf;
    if (!tf) throw new Error("tf.js did not load (check network / CDN block)");
    this.model = buildModel(tf, INPUT_SIZE);
    this.compile();
  }

  // Best-effort resume of whatever was last saved in this browser.
  // Throws if nothing has been saved yet — caller falls back to init().
  async loadFromBrowserStorage() {
    this.model = await window.tf.loadLayersModel("indexeddb://tiny-imgai-model");
    this.compile();
  }

  async saveToBrowserStorage() {
    await this.model.save("indexeddb://tiny-imgai-model");
  }

  async trainBatch(inputBytesArray, targetBytesArray, size = INPUT_SIZE) {
    const tf = window.tf;
    const x = tf.stack(inputBytesArray.map((b) => rgb8ToTensor(b, size)));
    const y = tf.stack(targetBytesArray.map((b) => rgb8ToTensor(b, size)));
    const history = await this.model.fit(x, y, { epochs: 1, batchSize: inputBytesArray.length, verbose: 0 });
    tf.dispose([x, y]);
    return history.history.loss[0];
  }

  async predict(canvas) {
    const tf = window.tf;
    const size = canvas.width;
    const x = tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0));
    const y = this.model.predict(x);
    const out = tf.tidy(() => y.squeeze([0]).clipByValue(0, 1));
    const outCanvas = document.createElement("canvas");
    outCanvas.width = size; outCanvas.height = size;
    await tf.browser.toPixels(out, outCanvas);
    tf.dispose([x, y, out]);
    return outCanvas;
  }

  // Downloads model.json + weights.bin — the exact same format
  // train.js writes on the Node/GH Actions side, so a checkpoint trained
  // there can be imported here, and vice versa.
  async exportWeights() {
    await this.model.save(`downloads://tiny-imgai-model-v${this.version}`);
  }

  // fileList must contain both the .json and its .bin file(s), selected
  // together (the Import button's file input has `multiple`).
  async loadWeights(fileList) {
    this.model = await window.tf.loadLayersModel(window.tf.io.browserFiles(Array.from(fileList)));
    this.compile();
  }
}
