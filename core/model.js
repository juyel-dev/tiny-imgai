// Shared model architecture for tiny-imgai.
//
// This file has ZERO platform-specific code. It only uses the tf.js
// Layers API (tf.input / tf.layers.* / tf.model), which is identical
// across every tf.js build — browser (@tensorflow/tfjs, WebGPU/WebGL/WASM
// backend) and Node (@tensorflow/tfjs + a Node backend). The caller
// injects `tf`, so this same file is imported unmodified from:
//   - /web/  (browser inference)
//   - /train/train.js (Node/GitHub Actions training)
//
// Architecture: small U-Net style encoder-decoder.
// Local 3x3-only convs (the old model) cannot learn context-dependent
// recoloring (heading vs. formula vs. diagram get different treatment).
// Downsample/upsample with skip connections gives the network enough
// receptive field to see page-level structure while skip connections
// keep fine handwriting detail from being lost.

export const INPUT_SIZE = 256; // page render size (square, letterboxed)

function convBlock(tf, x, filters, name) {
  x = tf.layers.conv2d({ filters, kernelSize: 3, padding: "same", name: name + "_conv1" }).apply(x);
  x = tf.layers.batchNormalization({ name: name + "_bn1" }).apply(x);
  x = tf.layers.reLU({ name: name + "_relu1" }).apply(x);
  x = tf.layers.conv2d({ filters, kernelSize: 3, padding: "same", name: name + "_conv2" }).apply(x);
  x = tf.layers.batchNormalization({ name: name + "_bn2" }).apply(x);
  x = tf.layers.reLU({ name: name + "_relu2" }).apply(x);
  return x;
}

export function buildModel(tf, inputSize = INPUT_SIZE) {
  const input = tf.input({ shape: [inputSize, inputSize, 3] });

  const e1 = convBlock(tf, input, 16, "enc1");
  const p1 = tf.layers.maxPooling2d({ poolSize: 2, name: "pool1" }).apply(e1);

  const e2 = convBlock(tf, p1, 32, "enc2");
  const p2 = tf.layers.maxPooling2d({ poolSize: 2, name: "pool2" }).apply(e2);

  const e3 = convBlock(tf, p2, 64, "enc3");
  const p3 = tf.layers.maxPooling2d({ poolSize: 2, name: "pool3" }).apply(e3);

  const b = convBlock(tf, p3, 128, "bottleneck");

  const u3 = tf.layers.upSampling2d({ size: [2, 2], name: "up3" }).apply(b);
  const c3 = tf.layers.concatenate({ name: "concat3" }).apply([u3, e3]);
  const d3 = convBlock(tf, c3, 64, "dec3");

  const u2 = tf.layers.upSampling2d({ size: [2, 2], name: "up2" }).apply(d3);
  const c2 = tf.layers.concatenate({ name: "concat2" }).apply([u2, e2]);
  const d2 = convBlock(tf, c2, 32, "dec2");

  const u1 = tf.layers.upSampling2d({ size: [2, 2], name: "up1" }).apply(d2);
  const c1 = tf.layers.concatenate({ name: "concat1" }).apply([u1, e1]);
  const d1 = convBlock(tf, c1, 16, "dec1");

  const output = tf.layers.conv2d({ filters: 3, kernelSize: 1, activation: "sigmoid", name: "output" }).apply(d1);

  return tf.model({ inputs: input, outputs: output, name: "tiny-imgai-unet" });
}
