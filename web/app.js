import { addPdfDataset, listPairs } from "./dataset.js";
import { loadState, saveState } from "./storage.js";
import { TinyImageModel, INPUT_SIZE } from "./model.js";
import { pdfPageCount } from "./pdf.js";

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
  testBtn.disabled = state.pairs.length === 0 || state.training;
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
    if (!tf) throw new Error("tf.js did not load (check network / ad-blocker / CDN access)");

    model = new TinyImageModel({ version: state.modelVersion ?? 0 });

    try {
      await model.loadFromBrowserStorage();
      $("#trainStatus").textContent = "Resumed saved model from this browser";
    } catch {
      await model.init();
    }

    $("#engineStatus").textContent = "tf.js backend: " + tf.getBackend();
    $("#modelRuntime").textContent = tf.getBackend();
    $("#version").textContent = "v" + model.version;
    $("#modelBadge").textContent = "MODEL v" + model.version;
    $("#modelParams").textContent = model.parameterCount;
    $("#modelSize").textContent = (model.parameterCount * 4) + " B";
    $("#modelArch").textContent = model.architecture;
    reportMemory("model-ready");
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

trainBtn.onclick = () => {
  if (state.training || !state.pairs.length) return;

  disposeMainModel();
  state.training = true;
  state.losses = [];
  render();

  setTrainingStatus("Starting background trainer");
  $("#lossHint").textContent = "PDF preparation + training are running outside the UI thread";
  $("#progress").style.width = "0%";

  trainingWorker = new Worker("./train-worker.js", { type: "module" });

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
    $("#lossHint").textContent = "The background trainer stopped unexpectedly.";
    finishTrainingWorker();
    render();
    updatePairButton();
  };

  trainingWorker.postMessage({
    type: "start",
    pairs: state.pairs,
  });
};

testBtn.onclick = async () => {
  if (!state.pairs.length) return;

  try {
    const m = await ensureModel();
    const p = state.pairs.find((x) => x.originalDocId && x.processedDocId);
    if (!p) throw new Error("No PDF-backed pair available.");

    const dbModule = await import("./dataset.js");
    const pdfModule = await import("./pdf.js");
    const [od, pd] = await Promise.all([
      dbModule.getDocument(p.originalDocId),
      dbModule.getDocument(p.processedDocId),
    ]);
    const [opdf, ppdf] = await Promise.all([
      pdfModule.openPdf(od.blob),
      pdfModule.openPdf(pd.blob),
    ]);

    try {
      const input = await pdfModule.renderPdfPage(opdf, p.originalPage, INPUT_SIZE);
      const target = await pdfModule.renderPdfPage(ppdf, p.processedPage, INPUT_SIZE);

      try {
        const output = await m.predict(input);
        $("#inputPreview").replaceChildren(input);
        $("#targetPreview").replaceChildren(target);
        $("#outputPreview").replaceChildren(output);
      } catch (error) {
        releaseCanvas(input);
        releaseCanvas(target);
        throw error;
      }
    } finally {
      await pdfModule.disposePdf(opdf);
      await pdfModule.disposePdf(ppdf);
    }
  } catch (error) {
    $("#outputPreview").textContent = error.message;
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
    const m = await ensureModel();
    await m.loadWeights(files);
    m.version++;
    state.modelVersion = m.version;

    await m.saveToBrowserStorage();
    saveState({
      pairCount: state.pairs.length,
      modelVersion: m.version,
      losses: state.losses,
    });

    $("#version").textContent = "v" + m.version;
    $("#modelBadge").textContent = "MODEL v" + m.version;
    $("#modelParams").textContent = m.parameterCount;

    btn.textContent = "Imported v" + m.version;
    setTimeout(() => btn.textContent = "Import model", 1500);
    reportMemory("model-imported");
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
