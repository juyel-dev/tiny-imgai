import { addPdfDataset, getDocument, listPairs } from "./dataset.js";
import { loadState, saveState } from "./storage.js";
import { TinyImageModel, imageToCanvas, canvasToRGB8 } from "./model.js";
import { openPdf, pdfPageCount, renderPdfPage } from "./pdf.js";

const state={pairs:await listPairs(),training:false,losses:[],...loadState()};
const $=s=>document.querySelector(s);
const pairCount=$("#pairCount"),trainDataset=$("#trainDataset"),datasetList=$("#datasetList"),trainBtn=$("#trainBtn"),testBtn=$("#testBtn");
let model=null;
let originalFile=null,processedFile=null;
let originalPages=0,processedPages=0;

function render(){
  pairCount.textContent=state.pairs.length;
  trainDataset.textContent=state.pairs.length+" pair"+(state.pairs.length===1?"":"s");
  trainBtn.disabled=state.pairs.length===0||state.training;
  testBtn.disabled=state.pairs.length===0||state.training;
  datasetList.innerHTML=state.pairs.length
    ? state.pairs.slice(-8).reverse().map(p=>`<div class="pair"><span class="pair-id">#${p.pageNumber} · page ${p.originalPage} ↔ page ${p.processedPage}</span><span class="pair-status">ready</span></div>`).join("")
    : '<div class="empty">No PDF page pairs yet.</div>';
}
function escapeText(name){return name||"Untitled.pdf"}
function setPdfSlot(kind,file,pages){
  const prefix=kind==="original"?"original":"processed";
  const zone=$("#"+prefix+"Dropzone");
  $("#"+prefix+"File").textContent=escapeText(file?.name);
  $("#"+prefix+"Pages").textContent=pages+" page"+(pages===1?"":"s");
  $("#"+prefix+"Hint").textContent=pages+" pages ready";
  zone.classList.toggle("is-ready",!!file);
}
function updatePairButton(){
  const ready=!!originalFile&&!!processedFile&&originalPages>0&&processedPages>0&&originalPages===processedPages;
  trainBtn.disabled=state.pairs.length===0||state.training;
  $("#pairPdfBtn").disabled=!ready||state.training;
  const summary=$("#pdfSummary");
  summary.classList.toggle("ready",ready);
}
async function choosePdf(kind,file){
  if(!file)return;
  try{
    const pages=await pdfPageCount(file);
    if(kind==="original"){originalFile=file;originalPages=pages}
    else{processedFile=file;processedPages=pages}
    setPdfSlot(kind,file,pages);
    if(originalFile&&processedFile&&originalPages!==processedPages){
      $("#pairPdfBtn").textContent="Page counts do not match";
      $("#pairPdfBtn").disabled=true;
      $("#processedHint").textContent=processedPages+" pages — need "+originalPages;
    }else{
      $("#pairPdfBtn").textContent="Create page pairs";
    }
  }catch(error){
    const prefix=kind==="original"?"original":"processed";
    $("#"+prefix+"Hint").textContent=error.message;
  }
  updatePairButton();
}
function bindDropZone(zoneId,inputId,kind){
  const zone=$("#"+zoneId),input=$("#"+inputId);
  zone.ondragover=e=>{e.preventDefault();zone.style.borderColor="#66717d"};
  zone.ondragleave=()=>zone.style.borderColor="";
  zone.ondrop=e=>{e.preventDefault();zone.style.borderColor="";choosePdf(kind,e.dataTransfer.files[0])};
  input.onchange=e=>choosePdf(kind,e.target.files[0]);
}
bindDropZone("originalDropzone","originalInput","original");
bindDropZone("processedDropzone","processedInput","processed");
$("#originalChoose").onclick=()=>$("#originalInput").click();
$("#processedChoose").onclick=()=>$("#processedInput").click();

$("#pairPdfBtn").onclick=async()=>{
  if(!originalFile||!processedFile||originalPages!==processedPages)return;
  const btn=$("#pairPdfBtn");
  btn.disabled=true;btn.textContent="Creating page pairs…";
  try{
    const result=await addPdfDataset({originalFile,processedFile,originalPages,processedPages});
    state.pairs.push(...result.pairs);
    saveState({pairCount:state.pairs.length});
    btn.textContent=result.pairs.length+" pairs created";
    render();
  }catch(error){
    btn.disabled=false;
    btn.textContent="Create page pairs";
    $("#processedHint").textContent=error.message;
    return;
  }
  updatePairButton();
};

function drawLoss(){
  if(!state.losses.length)return;
  const max=Math.max(...state.losses),min=Math.min(...state.losses),range=max-min||1;
  const pts=state.losses.map((v,i)=>`${i/(state.losses.length-1||1)*600},${145-((v-min)/range)*120}`).join(" ");
  $("#lossLine").setAttribute("points",pts);
}

async function ensureModel(){
  if(!model){
    model=new TinyImageModel({version:state.modelVersion??0,weights:state.weights});
    await model.init();
    $("#engineStatus").textContent="WebGPU active";
    $("#version").textContent="v"+model.version;
    $("#modelBadge").textContent="MODEL v"+model.version;
    $("#modelParams").textContent=model.parameterCount;
    $("#modelSize").textContent=(model.parameterCount*4)+" B";
  }
  return model;
}

async function prepareBatch(pdfOriginal,pdfProcessed,pairs){
  const inputs=[],targets=[];
  for(const p of pairs){
    const [a,b]=await Promise.all([
      renderPdfPage(pdfOriginal,p.originalPage,32),
      renderPdfPage(pdfProcessed,p.processedPage,32)
    ]);
    inputs.push(canvasToRGB8(a));targets.push(canvasToRGB8(b));
  }
  return {inputs,targets};
}

trainBtn.onclick=async()=>{
  if(state.training||!state.pairs.length)return;
  state.training=true;state.losses=[];render();
  $("#trainStatus").textContent="Opening PDFs";
  $("#lossHint").textContent="real WebGPU loss";
  try{
    const m=await ensureModel();
    const first=state.pairs.find(p=>p.originalDocId&&p.processedDocId);
    if(!first)throw new Error("No PDF-backed training pairs found.");
    const [od,pd]=await Promise.all([getDocument(first.originalDocId),getDocument(first.processedDocId)]);
    const [opdf,ppdf]=await Promise.all([openPdf(od.blob),openPdf(pd.blob)]);
    const epochs=20,batchSize=8;
    const batches=Math.ceil(state.pairs.length/batchSize);
    for(let epoch=1;epoch<=epochs;epoch++){
      let epochLoss=0;
      for(let start=0,batchIndex=0;start<state.pairs.length;start+=batchSize,batchIndex++){
        const batchPairs=state.pairs.slice(start,start+batchSize);
        const {inputs,targets}=await prepareBatch(opdf,ppdf,batchPairs);
        const result=await m.trainBatch(inputs,targets);
        m.applyGradient(result.grad);
        epochLoss+=result.loss*batchPairs.length;
        $("#step").textContent=(epoch-1)*batches+batchIndex+1;
        $("#epoch").textContent=`${epoch} / ${epochs}`;
        $("#loss").textContent=result.loss.toFixed(5);
        $("#progress").style.width=((batchIndex+1)/batches*100)+"%";
        $("#trainStatus").textContent=`Training batch ${batchIndex+1}/${batches}`;
        await new Promise(requestAnimationFrame);
      }
      const avg=epochLoss/state.pairs.length;
      state.losses.push(avg);drawLoss();$("#loss").textContent=avg.toFixed(5);
      saveState({pairCount:state.pairs.length,modelVersion:m.version,weights:[...m.weights],losses:state.losses});
    }
    await opdf.destroy();await ppdf.destroy();
    m.version++;
    state.modelVersion=m.version;state.weights=[...m.weights];
    state.training=false;
    $("#trainStatus").textContent="Complete";
    $("#modelBadge").textContent="MODEL v"+m.version;
    $("#version").textContent="v"+m.version;
    saveState({pairCount:state.pairs.length,modelVersion:m.version,weights:[...m.weights],losses:state.losses});
  }catch(error){
    console.error(error);
    state.training=false;
    $("#trainStatus").textContent="Error";
    $("#lossHint").textContent=error.message;
  }
  render();updatePairButton();
};

testBtn.onclick=async()=>{
  if(!state.pairs.length)return;
  try{
    const m=await ensureModel(),p=state.pairs.find(x=>x.originalDocId&&x.processedDocId);
    if(!p)throw new Error("No PDF-backed pair available.");
    const [od,pd]=await Promise.all([getDocument(p.originalDocId),getDocument(p.processedDocId)]);
    const [opdf,ppdf]=await Promise.all([openPdf(od.blob),openPdf(pd.blob)]);
    const [input,target]=await Promise.all([renderPdfPage(opdf,p.originalPage,64),renderPdfPage(ppdf,p.processedPage,64)]);
    const output=await m.predict(input);
    $("#inputPreview").replaceChildren(input);
    $("#targetPreview").replaceChildren(target);
    $("#outputPreview").replaceChildren(output);
    await opdf.destroy();await ppdf.destroy();
  }catch(error){$("#outputPreview").textContent=error.message}
};

(async()=>{
  try{await ensureModel()}catch(error){$("#engineStatus").textContent="WebGPU unavailable"}
  render();updatePairButton();
})();
