import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as tf from "@tensorflow/tfjs";
import { buildModel } from "../core/model.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function arg(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

function readStaging(stagingDir) {
  const metadata = JSON.parse(
    fs.readFileSync(path.join(stagingDir, "metadata.json"), "utf8")
  );
  const weightData = fs.readFileSync(
    path.join(stagingDir, metadata.weights_file)
  );
  return { metadata, weightData };
}

function makeTensor(weightData, spec) {
  const bytes = weightData.subarray(
    spec.byteOffset,
    spec.byteOffset + spec.byteLength
  );
  if (bytes.byteLength !== spec.byteLength) {
    throw new Error("Truncated weight data for " + spec.name);
  }
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength
  );
  return tf.tensor(new Float32Array(buffer), spec.shape, "float32");
}

async function main() {
  const stagingDir = path.resolve(
    arg("--staging-dir", path.join(__dirname, "evaluation", "tfjs-export-staging"))
  );
  const outputDir = path.resolve(
    arg("--output-dir", path.join(__dirname, "checkpoints", "tfjs-512-b48"))
  );

  const { metadata, weightData } = readStaging(stagingDir);
  if (metadata.input_size !== 512 || metadata.base_channels !== 48) {
    throw new Error("This exporter is locked to 512px / b48.");
  }

  await tf.ready();
  const model = buildModel(tf, metadata.input_size, metadata.channels);
  const specs = metadata.weight_specs;
  const modelWeights = model.weights;

  if (model.countParams() !== metadata.exported_values) {
    throw new Error(
      "tf.js/exported weight count mismatch: model=" +
      model.countParams() +
      " staging=" +
      metadata.exported_values
    );
  }
  if (modelWeights.length !== specs.length) {
    throw new Error(
      "Weight count mismatch: tf.js=" + modelWeights.length +
      " staging=" + specs.length
    );
  }

  const tensors = [];
  try {
    for (let i = 0; i < specs.length; i++) {
      const expected = modelWeights[i];
      const spec = specs[i];
      if (expected.name !== spec.name) {
        throw new Error(
          "Weight name mismatch at " + i + ": expected " +
          expected.name + ", got " + spec.name
        );
      }
      if (expected.shape.join(",") !== spec.shape.join(",")) {
        throw new Error(
          "Weight shape mismatch for " + spec.name +
          ": expected [" + expected.shape.join(",") +
          "], got [" + spec.shape.join(",") + "]"
        );
      }
      tensors.push(makeTensor(weightData, spec));
    }

    model.setWeights(tensors);

    const probe = tf.zeros([1, 512, 512, 3]);
    const output = model.predict(probe);
    if (Array.isArray(output) || output.shape.join(",") !== "1,512,512,3") {
      throw new Error("Unexpected model output shape.");
    }
    output.dispose();
    probe.dispose();

    fs.mkdirSync(outputDir, { recursive: true });
    await model.save(
      tf.io.withSaveHandler(async (artifacts) => {
        const modelJson = {
          modelTopology: artifacts.modelTopology,
          format: artifacts.format,
          generatedBy: artifacts.generatedBy,
          convertedBy: artifacts.convertedBy,
          weightsManifest: [
            {
              paths: ["weights.bin"],
              weights: artifacts.weightSpecs,
            },
          ],
        };
        fs.writeFileSync(
          path.join(outputDir, "model.json"),
          JSON.stringify(modelJson, null, 2)
        );
        fs.writeFileSync(
          path.join(outputDir, "weights.bin"),
          Buffer.from(artifacts.weightData)
        );
        return {
          modelArtifactsInfo: {
            dateSaved: new Date(),
            modelTopologyType: "JSON",
          },
        };
      })
    );

    fs.writeFileSync(
      path.join(outputDir, "metadata.json"),
      JSON.stringify(
        {
          format: "tiny-imgai-tfjs-production-v1",
          input_size: metadata.input_size,
          base_channels: metadata.base_channels,
          channels: metadata.channels,
          trainable_params: model.countParams(),
          checkpoint_epoch: metadata.checkpoint_epoch,
          checkpoint_global_step: metadata.checkpoint_global_step,
          bn_calibrated: metadata.bn_calibrated,
          exported_at: new Date().toISOString(),
        },
        null,
        2
      )
    );

    console.log("tiny-imgai tf.js export complete");
    console.log("Output:", outputDir);
    console.log("Params:", model.countParams());
  } finally {
    for (const tensor of tensors) tensor.dispose();
    model.dispose();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
