import { addPdfDataset, getCachedPage, cachePage, getDocument, listPairs } from "./dataset.js";
import { loadState, saveState } from "./storage.js";
import { TinyImageModel, INPUT_SIZE, canvasToRGB8 } from "./model.js";
import { openPdf, pdfPageCount, renderPdfPage, disposePdf } from "./pdf.js";

const state = { pairs: await listPairs(), training: false, losses: [], ...loadState() };
const $ = (s) => document.querySelector(s);
const pairCount = $("#pairCount");
const trainDataset = $("#trainDataset");
const datasetList = $("#datasetList");
const trainBtn = $("#trainBtn");
const testBtn = $("#testBtn");

let model = null;
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

function reportMemory(stage) {
  const tf = window.tf;
  if (!tf || typeof tf.memory !== "function") return;
  const m = tf.memory();
  const details = [
    `stage=${stage}`,
    `tensors=${m.numTensors}`,
    `bytes=${m.numBytes}`,
  ];
  if (typeof m.numBytesInGPU === "number") details.push(`gpuBytes=${m.numBytesInGPU}`);
  console.debug("[tiny-imgai memory]", details.join(" "));
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

function groupedPdfResources() {
  const groups = new Map();
  for (const p of state.pairs.filter((p) => p.originalDocId && p.processedDocId)) {
    const key = p.originalDocId + "|" + p.processedDocId;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(p);
  }
  return groups;
}

async function cacheTrainingDataset(groups) {
  const allPairs = [...groups.values()].flat();
  let ready = 0;
  let created = 0;
  const cacheSize = INPUT_SIZE;

  $("#trainStatus").textContent = "Checking page cache";

  for (const groupPairs of groups.values()) {
    const first = groupPairs[0];
    const [od, pd] = await Promise.all([
      getDocument(first.originalDocId),
      getDocument(first.processedDocId),
    ]);

    const [opdf, ppdf] = await Promise.all([
      openPdf(od.blob),
      openPdf(pd.blob),
    ]);

    try {
      for (const p of groupPairs) {
        const cached = await getCachedPage(p.uuid);

        if (cached?.size === cacheSize && cached.original && cached.target) {
          ready++;
        } else {
          const a = await renderPdfPage(opdf, p.originalPage, cacheSize);
          const original = canvasToRGB8(a);
          releaseCanvas(a);

          const b = await renderPdfPage(ppdf, p.processedPage, cacheSize);
          const target = canvasToRGB8(b);
          releaseCanvas(b);

          await cachePage(p.uuid, {
            size: cacheSize,
            original,
            target,
          });

          ready++;
          created++;
        }

        $("#trainStatus").textContent = `Caching pages ${ready}/${allPairs.length}`;
        $("#progress").style.width = (ready / allPairs.length * 100) + "%";
        reportMemory("cache-" + ready);
        await new Promise(requestAnimationFrame);
      }
    } finally {
      await disposePdf(opdf);
      await disposePdf(ppdf);
    }
  }

  return { total: allPairs.length, created };
}

async function readCachedBatch(batchPairs) {
  return Promise.all(batchPairs.map(async (p) => {
    const cached = await getCachedPage(p.uuid);
    if (!cached?.original || !cached?.target) {
      throw new Error(`Missing cached tensors for pair ${p.uuid}`);
    }

    return {
      original: new Uint8Array(cached.original),
      target: new Uint8Array(cached.target),
    };
  }));
}

trainBtn.onclick = async () => {
  if (state.training || !state.pairs.length) return;

  state.training = true;
  state.losses = [];
  render();

  $("#trainStatus").textContent = "Starting CNN";
  $("#lossHint").textContent = "tf.js training loss";

  try {
    const m = await ensureModel();
    const groups = groupedPdfResources();
    if (!groups.size) throw new Error("No PDF-backed training pairs found.");

    const cache = await cacheTrainingDataset(groups);
    $("#lossHint").textContent = cache.created
      ? cache.created + " pages cached"
      : "cache hit — no PDF rendering needed";

    const epochs = 20;
    // Keep the full 256×256 experiment. Batch 2 is the first speed step;
    // trainOnBatch() now gives us deterministic one-update-per-batch cleanup.
    const batchSize = 1;
    const totalBatches = [...groups.values()]
      .reduce((n, g) => n + Math.ceil(g.length / batchSize), 0);

    for (let epoch = 1; epoch <= epochs; epoch++) {
      let epochLoss = 0;
      let processedPairs = 0;
      let completedBatches = 0;

      for (const groupPairs of groups.values()) {
        for (let start = 0; start < groupPairs.length; start += batchSize) {
          const batchPairs = groupPairs.slice(start, start + batchSize);
          const batch = await readCachedBatch(batchPairs);

          try {
            const loss = await m.trainBatch(
              batch.map((x) => x.original),
              batch.map((x) => x.target),
              INPUT_SIZE
            );

            epochLoss += loss * batchPairs.length;
            processedPairs += batchPairs.length;
            completedBatches++;

            $("#step").textContent = (epoch - 1) * totalBatches + completedBatches;
            $("#epoch").textContent = `${epoch} / ${epochs}`;
            $("#loss").textContent = loss.toFixed(5);
            $("#progress").style.width =
              (completedBatches / totalBatches * 100) + "%";
            $("#trainStatus").textContent =
              `Training batch ${completedBatches}/${totalBatches}`;

            if (
              completedBatches === 1 ||
              completedBatches % 10 === 0 ||
              completedBatches === totalBatches
            ) {
              reportMemory(`train-e${epoch}-b${completedBatches}`);
            }
          } finally {
            batch.length = 0;
          }

          // Yield often enough for UI updates, but don't force a
          // full animation frame wait after every training batch.
          if (completedBatches % 4 === 0 || completedBatches === totalBatches) {
            await new Promise(requestAnimationFrame);
          }
        }
      }

      const avg = epochLoss / Math.max(1, processedPairs);
      state.losses.push(avg);
      drawLoss();
      $("#loss").textContent = avg.toFixed(5);

      $("#trainStatus").textContent = `Saving checkpoint after epoch ${epoch}`;
      await new Promise(requestAnimationFrame);
      await m.saveToBrowserStorage();

      m.version++;
      state.modelVersion = m.version;
      saveState({
        pairCount: state.pairs.length,
        modelVersion: m.version,
        losses: state.losses,
      });

      $("#version").textContent = "v" + m.version;
      $("#modelBadge").textContent = "MODEL v" + m.version;
      reportMemory(`epoch-${epoch}-checkpointed`);
    }

    state.training = false;
    $("#trainStatus").textContent = "Complete";
    saveState({
      pairCount: state.pairs.length,
      modelVersion: m.version,
      losses: state.losses,
    });
  } catch (error) {
    console.error(error);
    state.training = false;
    $("#trainStatus").textContent = "Error";
    $("#lossHint").textContent = error.message;
  }

  render();
  updatePairButton();
};

testBtn.onclick = async () => {
  if (!state.pairs.length) return;

  try {
    const m = await ensureModel();
    const p = state.pairs.find((x) => x.originalDocId && x.processedDocId);
    if (!p) throw new Error("No PDF-backed pair available.");

    const [od, pd] = await Promise.all([
      getDocument(p.originalDocId),
      getDocument(p.processedDocId),
    ]);
    const [opdf, ppdf] = await Promise.all([
      openPdf(od.blob),
      openPdf(pd.blob),
    ]);

    try {
      const input = await renderPdfPage(opdf, p.originalPage, INPUT_SIZE);
      const target = await renderPdfPage(ppdf, p.processedPage, INPUT_SIZE);

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
      await disposePdf(opdf);
      await disposePdf(ppdf);
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

(async () => {
  try {
    await ensureModel();
  } catch (error) {
    $("#engineStatus").textContent = "Model engine unavailable: " + error.message;
  }

  render();
  updatePairButton();
})();
