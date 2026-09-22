import {
  buildModel,
  INPUT_SIZE,
  DEFAULT_CHANNELS,
  PRODUCTION_INPUT_SIZE,
  PRODUCTION_CHANNELS,
} from "../core/model.js";

export { INPUT_SIZE, PRODUCTION_INPUT_SIZE, PRODUCTION_CHANNELS };

export const PRODUCTION_MODEL_URL = "/web/models/tfjs-512-b48/model.json";
export const PRODUCTION_METADATA_URL = "/web/models/tfjs-512-b48/metadata.json";
export const PRODUCTION_TFJS_PARAMETER_COUNT = 4387299;

export function canvasToRGB8(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const { data, width, height } = ctx.getImageData(
    0,
    0,
    canvas.width,
    canvas.height
  );
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
  c.getContext("2d", { willReadFrequently: true }).drawImage(
    source,
    0,
    0,
    size,
    size
  );
  return c;
}

function batchBytesToTensor(bytesArray, size) {
  const count = bytesArray.length;
  const pixels = size * size * 3;
  const packed = new Float32Array(count * pixels);

  for (let b = 0; b < count; b++) {
    const src = bytesArray[b];
    if (!src || src.length !== pixels) {
      throw new Error(
        `Invalid RGB page tensor: expected ${pixels} bytes, got ${src?.length ?? "none"}.`
      );
    }
    const offset = b * pixels;
    for (let i = 0; i < pixels; i++) {
      packed[offset + i] = src[i] / 255;
    }
  }

  return globalThis.tf.tensor4d(
    packed,
    [count, size, size, 3],
    "float32"
  );
}

const OPTIMIZER_LR = 1e-3;

function assertProductionModel(model, metadata) {
  const inputShape = model.inputs?.[0]?.shape || [];
  const expectedShape = [null, PRODUCTION_INPUT_SIZE, PRODUCTION_INPUT_SIZE, 3];
  if (
    inputShape.length !== expectedShape.length ||
    inputShape[1] !== expectedShape[1] ||
    inputShape[2] !== expectedShape[2] ||
    inputShape[3] !== expectedShape[3]
  ) {
    throw new Error(
      "Production model input mismatch: expected 512×512 RGB."
    );
  }

  const expectedParams =
    Number(metadata?.tfjs_parameter_count) ||
    Number(metadata?.trainable_params) ||
    PRODUCTION_TFJS_PARAMETER_COUNT;

  if (model.countParams() !== expectedParams) {
    throw new Error(
      `Production model parameter mismatch: expected ${expectedParams}, got ${model.countParams()}.`
    );
  }

  const expectedChannels = JSON.stringify(PRODUCTION_CHANNELS);
  const actualChannels = JSON.stringify(metadata?.channels || []);
  if (actualChannels && actualChannels !== expectedChannels) {
    throw new Error(
      `Production model channel mismatch: expected ${expectedChannels}, got ${actualChannels}.`
    );
  }
}

export class TinyImageModel {
  constructor({ version = 0, mode = "production" } = {}) {
    this.version = version;
    this.mode = mode;
    this.model = null;
    this.trainable = false;
  }

  get parameterCount() {
    return this.model ? this.model.countParams() : 0;
  }

  get architecture() {
    if (this.mode === "production") {
      return "U-Net 48→96→192→384→192→96→48 · 512px · tf.js";
    }
    return "U-Net 16→32→64→128→64→32→16 · 256px · tf.js";
  }

  compile() {
    if (!this.model) throw new Error("Model is not initialized.");
    this.model.compile({
      optimizer: globalThis.tf.train.adam(OPTIMIZER_LR),
      loss: "meanSquaredError",
    });
    this.trainable = true;
  }

  dispose() {
    if (!this.model) return;
    const optimizer = this.model.optimizer;
    try {
      this.model.dispose();
    } finally {
      if (optimizer && typeof optimizer.dispose === "function") {
        optimizer.dispose();
      }
      this.model = null;
    }
    this.trainable = false;
  }

  async init() {
    const tf = globalThis.tf;
    if (!tf) {
      throw new Error("tf.js did not load (check network / CDN block)");
    }
    this.dispose();
    this.model = buildModel(tf, INPUT_SIZE, DEFAULT_CHANNELS);
    this.mode = "legacy";
    this.trainable = false;
    this.compile();
  }

  async loadProductionModel({
    modelUrl = PRODUCTION_MODEL_URL,
    metadataUrl = PRODUCTION_METADATA_URL,
  } = {}) {
    const tf = globalThis.tf;
    if (!tf) {
      throw new Error("tf.js did not load (check network / CDN block)");
    }

    let metadata = null;
    try {
      const response = await fetch(metadataUrl, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`metadata HTTP ${response.status}`);
      }
      metadata = await response.json();
    } catch (error) {
      throw new Error(
        "Production model metadata is not deployed at " +
          metadataUrl +
          ". Run train\\install-browser-model.ps1 and deploy web/models."
      );
    }

    if (
      Number(metadata.input_size) !== PRODUCTION_INPUT_SIZE ||
      Number(metadata.base_channels) !== 48 ||
      !metadata.bn_calibrated
    ) {
      throw new Error(
        "Production model metadata is invalid: expected 512px b48 with calibrated BatchNorm."
      );
    }

    const nextModel = await tf.loadLayersModel(modelUrl);
    try {
      assertProductionModel(nextModel, metadata);
    } catch (error) {
      nextModel.dispose();
      throw error;
    }

    this.dispose();
    this.model = nextModel;
    this.mode = "production";
    this.trainable = false;
    this.version = Number(metadata.checkpoint_epoch || 3);
    return metadata;
  }

  async loadFromBrowserStorage() {
    const tf = globalThis.tf;
    if (!tf) {
      throw new Error("tf.js did not load (check network / CDN block)");
    }
    const nextModel = await tf.loadLayersModel("indexeddb://tiny-imgai-model");
    this.dispose();
    this.model = nextModel;
    this.mode = "legacy";
    this.compile();
  }

  async saveToBrowserStorage() {
    if (!this.model) throw new Error("Model is not initialized.");
    await this.model.save("indexeddb://tiny-imgai-model");
  }

  async trainBatch(inputBytesArray, targetBytesArray, size = INPUT_SIZE) {
    if (!this.model) throw new Error("Model is not initialized.");
    if (!this.trainable) {
      throw new Error("The production model is inference-only. Train locally with PyTorch.");
    }
    if (
      inputBytesArray.length !== targetBytesArray.length ||
      !inputBytesArray.length
    ) {
      throw new Error("Input/target batch mismatch.");
    }

    const tf = globalThis.tf;
    let x = null;
    let y = null;

    try {
      x = batchBytesToTensor(inputBytesArray, size);
      y = batchBytesToTensor(targetBytesArray, size);
      const result = await this.model.trainOnBatch(x, y);
      const loss = Array.isArray(result) ? result[0] : result;
      return Number(loss);
    } finally {
      tf.dispose([x, y]);
    }
  }

  async predict(canvas, outputSize = canvas.width) {
    if (!this.model) throw new Error("Model is not initialized.");
    if (canvas.width !== PRODUCTION_INPUT_SIZE || canvas.height !== PRODUCTION_INPUT_SIZE) {
      throw new Error("Production inference requires a 512×512 canvas.");
    }

    const tf = globalThis.tf;
    const x = tf.tidy(() =>
      tf.browser
        .fromPixels(canvas)
        .toFloat()
        .div(255)
        .expandDims(0)
    );

    let y = null;
    let out = null;
    try {
      y = this.model.predict(x);
      out = tf.tidy(() => y.squeeze([0]).clipByValue(0, 1));

      const outCanvas = document.createElement("canvas");
      outCanvas.width = outputSize;
      outCanvas.height = outputSize;
      await tf.browser.toPixels(out, outCanvas);
      return outCanvas;
    } finally {
      tf.dispose([x, y, out]);
    }
  }

  async exportWeights() {
    if (!this.model) throw new Error("Model is not initialized.");
    await this.model.save(
      `downloads://tiny-imgai-model-v${this.version}`
    );
  }

  async loadWeights(fileList) {
    const tf = globalThis.tf;
    if (!fileList?.length) {
      throw new Error("No model files selected.");
    }
    const nextModel = await tf.loadLayersModel(
      tf.io.browserFiles(Array.from(fileList))
    );
    try {
      if (
        nextModel.inputs?.[0]?.shape?.[1] !== PRODUCTION_INPUT_SIZE ||
        nextModel.inputs?.[0]?.shape?.[2] !== PRODUCTION_INPUT_SIZE
      ) {
        throw new Error("Imported model is not the 512×512 production model.");
      }
    } catch (error) {
      nextModel.dispose();
      throw error;
    }
    this.dispose();
    this.model = nextModel;
    this.mode = "production";
    this.trainable = false;
  }
}
