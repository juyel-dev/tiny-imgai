// Shared model architecture for tiny-imgai.
//
// Platform-neutral tf.js Layers model used by browser inference and
// Node training. The input size and channel widths are configurable,
// while the default configuration remains the original 256px U-Net.

export const INPUT_SIZE = 256;
export const DEFAULT_CHANNELS = [16, 32, 64, 128];

function resolveChannels(channels) {
  if (!Array.isArray(channels) || channels.length !== 4) {
    throw new Error("channels must be [c1, c2, c3, bottleneck]");
  }
  if (channels.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error("channels must contain positive integers");
  }
  return channels;
}

function convBlock(tf, x, filters, name) {
  x = tf.layers.conv2d({
    filters,
    kernelSize: 3,
    padding: "same",
    name: name + "_conv1",
  }).apply(x);
  x = tf.layers.batchNormalization({ name: name + "_bn1" }).apply(x);
  x = tf.layers.reLU({ name: name + "_relu1" }).apply(x);
  x = tf.layers.conv2d({
    filters,
    kernelSize: 3,
    padding: "same",
    name: name + "_conv2",
  }).apply(x);
  x = tf.layers.batchNormalization({ name: name + "_bn2" }).apply(x);
  x = tf.layers.reLU({ name: name + "_relu2" }).apply(x);
  return x;
}

export function buildModel(
  tf,
  inputSize = INPUT_SIZE,
  channels = DEFAULT_CHANNELS,
) {
  const [c1, c2, c3, c4] = resolveChannels(channels);
  const input = tf.input({ shape: [inputSize, inputSize, 3] });

  const e1 = convBlock(tf, input, c1, "enc1");
  const p1 = tf.layers.maxPooling2d({ poolSize: 2, name: "pool1" }).apply(e1);

  const e2 = convBlock(tf, p1, c2, "enc2");
  const p2 = tf.layers.maxPooling2d({ poolSize: 2, name: "pool2" }).apply(e2);

  const e3 = convBlock(tf, p2, c3, "enc3");
  const p3 = tf.layers.maxPooling2d({ poolSize: 2, name: "pool3" }).apply(e3);

  const b = convBlock(tf, p3, c4, "bottleneck");

  const u3 = tf.layers.upSampling2d({ size: [2, 2], name: "up3" }).apply(b);
  const cat3 = tf.layers.concatenate({ name: "concat3" }).apply([u3, e3]);
  const d3 = convBlock(tf, cat3, c3, "dec3");

  const u2 = tf.layers.upSampling2d({ size: [2, 2], name: "up2" }).apply(d3);
  const cat2 = tf.layers.concatenate({ name: "concat2" }).apply([u2, e2]);
  const d2 = convBlock(tf, cat2, c2, "dec2");

  const u1 = tf.layers.upSampling2d({ size: [2, 2], name: "up1" }).apply(d2);
  const cat1 = tf.layers.concatenate({ name: "concat1" }).apply([u1, e1]);
  const d1 = convBlock(tf, cat1, c1, "dec1");

  const output = tf.layers.conv2d({
    filters: 3,
    kernelSize: 1,
    activation: "sigmoid",
    name: "output",
  }).apply(d1);

  return tf.model({
    inputs: input,
    outputs: output,
    name: "tiny-imgai-unet",
  });
}
