import { addPdfDataset, listPairs, getDocument, getCachedPage, cachePage } from "./dataset.js";
import { loadState, saveState } from "./storage.js";
import { TinyImageModel, PRODUCTION_INPUT_SIZE, canvasToRGB8 } from "./model.js";
import { pdfPageCount, openPdf, renderPdfPage, disposePdf } from "./pdf.js";

const state = { pairs: await listPairs(), training: false, losses: [], ...loadState() };
const $ = (s) => document.querySelector(s);
const pairCount = $("#pairCount");
const trainDataset = $("#trainDataset");
const datasetList = $("#datasetList");
const trainBtn = $("#trainBtn");
const testBtn = $("#testBtn");

let model = null;
let trainingWorker = null;
let originalFile = null;
let processedFile = null;
let originalPages = 0;
let processedPages = 0;

function render() {
  pairCount.textContent = state.pairs.length;
  trainDataset.textContent = state.pairs.length + " pair" + (state.pairs.length === 1 ? "" : "s");
  trainBtn.disabled = state.pairs.length === 0 || state.training;
  testBtn.disabled = !testImageCanvas || state.training;
  datasetList.innerHTML = state.pairs.length
    ? state.pairs.slice(-8).reverse().map((p) =>
        `<div class="pair"><span class="pair-id">#${p.pageNumber} · page ${p.originalPage} ↔ page ${p.processedPage}</span><span class="pair-status">ready</span></div>`
      ).join("")
    : '<div class="empty">No PDF page pairs yet.</div>';
}

function escapeText(name) {
  return name || "Untitled.pdf";
}

function setPdfSlot(kind, file, pages) {
  const prefix = kind === "original" ? "original" : "processed";
  const zone = $("#" + prefix + "Dropzone");
  $("#" + prefix + "File").textContent = escapeText(file?.name);
  $("#" + prefix + "Pages").textContent = pages + " page" + (pages === 1 ? "" : "s");
  $("#" + prefix + "Hint").textContent = pages + " pages ready";
  zone.classList.toggle("is-ready", !!file);
}

function updatePairButton() {
  const ready =
    !!originalFile &&
    !!processedFile &&
    originalPages > 0 &&
    processedPages > 0 &&
    originalPages === processedPages;

  trainBtn.disabled = state.pairs.length === 0 || state.training;
  $("#pairPdfBtn").disabled = !ready || state.training;
  $("#pdfSummary").classList.toggle("ready", ready);
}

async function choosePdf(kind, file) {
  if (!file) return;

  try {
    const pages = await pdfPageCount(file);
    if (kind === "original") {
      originalFile = file;
      originalPages = pages;
    } else {
      processedFile = file;
      processedPages = pages;
    }

    setPdfSlot(kind, file, pages);

    if (originalFile && processedFile && originalPages !== processedPages) {
      $("#pairPdfBtn").textContent = "Page counts do not match";
      $("#pairPdfBtn").disabled = true;
      $("#processedHint").textContent = processedPages + " pages — need " + originalPages;
    } else {
      $("#pairPdfBtn").textContent = "Create page pairs";
    }
  } catch (error) {
    const prefix = kind === "original" ? "original" : "processed";
    $("#" + prefix + "Hint").textContent = error.message;
  }

  updatePairButton();
}

function bindDropZone(zoneId, inputId, kind) {
  const zone = $("#" + zoneId);
  const input = $("#" + inputId);

  zone.ondragover = (e) => {
    e.preventDefault();
    zone.style.borderColor = "#66717d";
  };
  zone.ondragleave = () => {
    zone.style.borderColor = "";
  };
  zone.ondrop = (e) => {
    e.preventDefault();
    zone.style.borderColor = "";
    choosePdf(kind, e.dataTransfer.files[0]);
  };
  input.onchange = (e) => choosePdf(kind, e.target.files[0]);
}

bindDropZone("originalDropzone", "originalInput", "original");
bindDropZone("processedDropzone", "processedInput", "processed");
$("#originalChoose").onclick = () => $("#originalInput").click();
$("#processedChoose").onclick = () => $("#processedInput").click();

$("#pairPdfBtn").onclick = async () => {
  if (!originalFile || !processedFile || originalPages !== processedPages) return;

  const btn = $("#pairPdfBtn");
  btn.disabled = true;
  btn.textContent = "Creating page pairs…";

  try {
    const result = await addPdfDataset({
      originalFile,
      processedFile,
      originalPages,
      processedPages,
    });

    state.pairs.push(...result.pairs);
    saveState({ pairCount: state.pairs.length });

    btn.textContent = result.pairs.length + " pairs created";
    originalFile = null;
    processedFile = null;
    originalPages = 0;
    processedPages = 0;

    $("#originalFile").textContent = "—";
    $("#originalPages").textContent = "0 pages";
    $("#originalHint").textContent = "Drop the original notes PDF";
    $("#originalDropzone").classList.remove("is-ready");
    $("#processedFile").textContent = "—";
    $("#processedPages").textContent = "0 pages";
    $("#processedHint").textContent = "Drop the optimized / processed PDF";
    $("#processedDropzone").classList.remove("is-ready");

    render();
  } catch (error) {
    btn.disabled = false;
    btn.textContent = "Create page pairs";
    $("#processedHint").textContent = error.message;
    return;
  }

  updatePairButton();
};

function drawLoss() {
  if (!state.losses.length) return;
  const max = Math.max(...state.losses);
  const min = Math.min(...state.losses);
  const range = max - min || 1;
  const pts = state.losses
    .map((v, i) => `${i / (state.losses.length - 1 || 1) * 600},${145 - ((v - min) / range) * 120}`)
    .join(" ");
  $("#lossLine").setAttribute("points", pts);
}

function releaseCanvas(canvas) {
  if (!canvas) return;
  canvas.width = 0;
  canvas.height = 0;
}


async function ensureModel() {
  if (!model) {
    const tf = window.tf;
    if (!tf) {
      throw new Error("tf.js did not load (check network / ad-blocker / CDN access)");
    }

    const candidate = new TinyImageModel({
      version: state.modelVersion ?? 3,
      mode: "production",
    });

    try {
      setTrainingStatus("Loading 512×512 production model…");
      const metadata = await candidate.loadProductionModel();

      model = candidate;
      $("#engineStatus").textContent = "tf.js backend: " + tf.getBackend();
      $("#modelRuntime").textContent = tf.getBackend() + " · production";
      $("#version").textContent = "v" + model.version;
      $("#modelBadge").textContent = "PRODUCTION v" + model.version;
      $("#modelParams").textContent = model.parameterCount.toLocaleString();
      $("#modelSize").textContent =
        (model.parameterCount * 4 / 1048576).toFixed(2) + " MiB";
      $("#modelArch").textContent = model.architecture;
      $("#trainStatus").textContent =
        "Production model loaded · 512px inference ready";
      return model;
    } catch (error) {
      candidate.dispose();
      throw error;
    }
  }

  return model;
}

function setTrainingStatus(text) {
  $("#trainStatus").textContent = text;
}

function updateModelUi(info) {
  if (!info) return;
  if (info.backend) {
    $("#engineStatus").textContent = "tf.js worker: " + info.backend;
    $("#modelRuntime").textContent = info.backend + " · worker";
  }
  if (info.parameterCount) $("#modelParams").textContent = info.parameterCount;
  if (info.architecture) $("#modelArch").textContent = info.architecture;
  if (info.parameterCount) $("#modelSize").textContent = (info.parameterCount * 4) + " B";
}

function disposeMainModel() {
  if (!model) return;
  model.dispose();
  model = null;
}

function finishTrainingWorker() {
  if (!trainingWorker) return;
  trainingWorker.terminate();
  trainingWorker = null;
}

function yieldToBrowser() {
  return new Promise((resolve) => {
    if (typeof requestIdleCallback === "function") {
      requestIdleCallback(() => resolve(), { timeout: 16 });
    } else {
      setTimeout(resolve, 0);
    }
  });
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

async function prepareTrainingCache(pairs) {
  const groups = groupedPairs(pairs);
  if (!groups.size) throw new Error("No PDF-backed training pairs found.");

  const allPairs = [...groups.values()].flat();
  let created = 0;
  let prepared = 0;

  setTrainingStatus("Preparing pages for background training");
  $("#lossHint").textContent = "PDF rendering is kept separate from TensorFlow training";

  for (const group of groups.values()) {
    const first = group[0];
    const [od, pd] = await Promise.all([
      getDocument(first.originalDocId),
      getDocument(first.processedDocId),
    ]);

    const [opdf, ppdf] = await Promise.all([
      openPdf(od.blob),
      openPdf(pd.blob),
    ]);

    try {
      for (const pair of group) {
        const cached = await getCachedPage(pair.uuid);

        if (!(cached?.size === PRODUCTION_INPUT_SIZE && cached.original && cached.target)) {
          const originalCanvas = await renderPdfPage(opdf, pair.originalPage, PRODUCTION_INPUT_SIZE);
          const original = canvasToRGB8(originalCanvas);
          releaseCanvas(originalCanvas);

          await yieldToBrowser();

          const targetCanvas = await renderPdfPage(ppdf, pair.processedPage, PRODUCTION_INPUT_SIZE);
          const target = canvasToRGB8(targetCanvas);
          releaseCanvas(targetCanvas);

          await cachePage(pair.uuid, {
            size: PRODUCTION_INPUT_SIZE,
            original,
            target,
          });
          created++;
        }

        prepared++;
        if (prepared === 1 || prepared % 5 === 0 || prepared === allPairs.length) {
          const percent = Math.min(35, (prepared / allPairs.length) * 35);
          $("#progress").style.width = percent + "%";
          setTrainingStatus("Preparing page " + prepared + "/" + allPairs.length);
        }

        await yieldToBrowser();
      }
    } finally {
      await Promise.allSettled([
        disposePdf(opdf),
        disposePdf(ppdf),
      ]);
    }
  }

  return { total: allPairs.length, created };
}


trainBtn.onclick = () => {
  if (state.training || !state.pairs.length) return;

  disposeMainModel();
  state.training = true;
  state.losses = [];
  render();

  setTrainingStatus("Starting background trainer");
  $("#lossHint").textContent = "PDF preparation + training are running outside the UI thread";
  $("#progress").style.width = "0%";

  setTrainingStatus("Preparing training cache");
  prepareTrainingCache(state.pairs).then(({ total, created }) => {
    setTrainingStatus("Starting background trainer");
    $("#lossHint").textContent =
      created
        ? "Cached " + created + " new pages; TensorFlow training is now in a worker"
        : "Using existing 256×256 page cache; TensorFlow training is now in a worker";
    $("#progress").style.width = "0%";

    trainingWorker = new Worker(new URL("./train-worker.js", import.meta.url), { type: "module" });

    trainingWorker.onmessage = (event) => {
    const msg = event.data || {};

    if (msg.type === "ready") {
      updateModelUi(msg);
      setTrainingStatus("Worker ready · " + msg.backend);
      return;
    }

    if (msg.type === "phase") {
      if (msg.phase === "prepare") {
        setTrainingStatus("Preparing " + msg.total + " pages in background");
      } else if (msg.phase === "training") {
        setTrainingStatus("Training " + msg.total + " pages · 256×256 · batch 1");
        if (msg.memoryBytes) {
          $("#lossHint").textContent =
            "Worker dataset RAM: " + (msg.memoryBytes / 1048576).toFixed(1) + " MiB";
        }
      }
      return;
    }

    if (msg.type === "progress") {
      const total = msg.total || 1;
      const percent = Math.min(100, (msg.completed / total) * 100);
      $("#progress").style.width = percent + "%";

      if (msg.phase === "prepare") {
        setTrainingStatus("Preparing page " + msg.completed + "/" + msg.total);
      } else if (msg.phase === "training") {
        $("#step").textContent = msg.completed;
        $("#epoch").textContent = msg.epoch + " / " + msg.epochs;
        $("#loss").textContent = Number(msg.loss).toFixed(5);
        setTrainingStatus(
          "Training epoch " + msg.epoch + " · page " + msg.completed + "/" + msg.total
        );
      }
      return;
    }

    if (msg.type === "epoch") {
      state.losses.push(msg.loss);
      drawLoss();
      $("#loss").textContent = Number(msg.loss).toFixed(5);
      setTrainingStatus("Epoch " + msg.epoch + "/" + msg.epochs + " complete");
      return;
    }

    if (msg.type === "checkpoint") {
      state.modelVersion = msg.epoch;
      saveState({
        pairCount: state.pairs.length,
        modelVersion: state.modelVersion,
        losses: state.losses,
      });
      $("#version").textContent = "v" + msg.epoch;
      $("#modelBadge").textContent = "MODEL v" + msg.epoch;
      return;
    }

    if (msg.type === "done") {
      state.training = false;
      state.modelVersion = msg.version;
      saveState({
        pairCount: state.pairs.length,
        modelVersion: state.modelVersion,
        losses: state.losses,
      });
      $("#progress").style.width = "100%";
      $("#epoch").textContent = "20 / 20";
      setTrainingStatus("Complete · " + msg.backend);
      finishTrainingWorker();
      render();
      updatePairButton();
      return;
    }

    if (msg.type === "error") {
      console.error("[tiny-imgai worker]", msg.message, msg.stack);
      state.training = false;
      setTrainingStatus("Training stopped");
      $("#lossHint").textContent = msg.message;
      finishTrainingWorker();
      render();
      updatePairButton();
    }
  };

  trainingWorker.onerror = (error) => {
    console.error("[tiny-imgai worker error]", error);
    state.training = false;
    setTrainingStatus("Training worker crashed");
    $("#lossHint").textContent =
      error?.message || "The background trainer stopped unexpectedly.";
    finishTrainingWorker();
    render();
    updatePairButton();
  };

    trainingWorker.postMessage({
      type: "start",
      pairs: state.pairs,
    });
  }).catch((error) => {
    console.error("[tiny-imgai prepare error]", error);
    state.training = false;
    setTrainingStatus("Training stopped");
    $("#lossHint").textContent = error?.message || String(error);
    render();
    updatePairButton();
  });
};

let testImageCanvas = null;

async function loadTestImage(file) {
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    throw new Error("Please select a PNG, JPG, WEBP, or other image file.");
  }

  const bitmap = await createImageBitmap(file);
  try {
    const canvas = imageToCanvasFromBitmap(bitmap, PRODUCTION_INPUT_SIZE);
    releaseCanvas(testImageCanvas);
    testImageCanvas = canvas;

    $("#inputPreview").replaceChildren(canvas);
    $("#outputPreview").textContent = "Ready — click Run model";
    $("#targetPreview").textContent = file.name;
    $("#testBtn").disabled = false;
    $("#testHint").textContent =
      "Input resized to 512×512 for production inference.";
  } finally {
    bitmap.close();
  }
}

function imageToCanvasFromBitmap(bitmap, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  canvas.getContext("2d", { willReadFrequently: true }).drawImage(
    bitmap,
    0,
    0,
    size,
    size
  );
  return canvas;
}

const testInput = $("#testImageInput");
const testDropzone = $("#testImageDropzone");

testDropzone.ondragover = (event) => {
  event.preventDefault();
  testDropzone.classList.add("is-ready");
};

testDropzone.ondragleave = () => {
  testDropzone.classList.remove("is-ready");
};

testDropzone.ondrop = async (event) => {
  event.preventDefault();
  testDropzone.classList.remove("is-ready");
  try {
    await loadTestImage(event.dataTransfer.files[0]);
  } catch (error) {
    $("#testHint").textContent = error.message;
  }
};

testInput.onchange = async (event) => {
  try {
    await loadTestImage(event.target.files[0]);
  } catch (error) {
    $("#testHint").textContent = error.message;
  } finally {
    event.target.value = "";
  }
};

$("#testChoose").onclick = () => testInput.click();

testBtn.onclick = async () => {
  if (!testImageCanvas) return;

  testBtn.disabled = true;
  testBtn.textContent = "Running…";

  try {
    const m = await ensureModel();
    const output = await m.predict(testImageCanvas, PRODUCTION_INPUT_SIZE);
    $("#outputPreview").replaceChildren(output);
    setTrainingStatus("Inference complete");
    $("#testHint").textContent =
      "512×512 production model · output generated locally in this browser.";
  } catch (error) {
    $("#outputPreview").textContent = error.message;
    $("#testHint").textContent = error.message;
  } finally {
    testBtn.disabled = false;
    testBtn.textContent = "Run model";
  }
};

$("#exportBtn").onclick = async () => {
  const btn = $("#exportBtn");
  try {
    const m = await ensureModel();
    await m.exportWeights();
  } catch (error) {
    btn.textContent = "Export failed";
    setTimeout(() => btn.textContent = "Export model", 1500);
    console.error(error);
  }
};

$("#importBtn").onclick = () => $("#importInput").click();

$("#importInput").onchange = async (e) => {
  const files = e.target.files;
  e.target.value = "";
  if (!files || !files.length) return;

  const btn = $("#importBtn");

  try {
    const m = new TinyImageModel({ version: 3, mode: "production" });
    await m.loadWeights(files);
    model?.dispose();
    model = m;
    state.modelVersion = m.version;
    saveState({
      pairCount: state.pairs.length,
      modelVersion: m.version,
      losses: state.losses,
    });

    $("#version").textContent = "v" + m.version;
    $("#modelBadge").textContent = "MODEL v" + m.version;
    $("#modelParams").textContent = m.parameterCount.toLocaleString();
    $("#modelSize").textContent =
      (m.parameterCount * 4 / 1048576).toFixed(2) + " MiB";
    $("#modelArch").textContent = m.architecture;

    btn.textContent = "Imported v" + m.version;
    setTimeout(() => btn.textContent = "Import model", 1500);
  } catch (error) {
    btn.textContent = error.message.slice(0, 40);
    setTimeout(() => btn.textContent = "Import model", 2000);
    console.error(error);
  }
};

(() => {
  render();
  updatePairButton();
})();
