import { addPair, listPairs } from "./dataset.js";
import { loadState, saveState } from "./storage.js";
import { TinyImageModel, imageToCanvas } from "./model.js";

const state={pairs:await listPairs(),training:false,losses:[],...loadState()};
const $=s=>document.querySelector(s);
const pairCount=$("#pairCount"),trainDataset=$("#trainDataset"),datasetList=$("#datasetList"),trainBtn=$("#trainBtn"),testBtn=$("#testBtn");
let model=null;

function render(){
  pairCount.textContent=state.pairs.length;
  trainDataset.textContent=state.pairs.length+" pair"+(state.pairs.length===1?"":"s");
  trainBtn.disabled=state.pairs.length===0||state.training;
  testBtn.disabled=state.pairs.length===0||state.training;
  datasetList.innerHTML=state.pairs.length?state.pairs.slice(-8).reverse().map(p=>`<div class="pair"><span class="pair-id">#${p.pageNumber} · ${escapeHtml(p.name)}</span><span class="pair-status">ready</span></div>`).join(""):'<div class="empty">No training pairs yet.</div>';
}
function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function fileToDataURL(file){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(file)})}
function dataURLToImage(src){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=reject;img.src=src})}
async function addFiles(files){
  const imgs=[...files].filter(f=>f.type.startsWith("image/"));
  for(let i=0;i<imgs.length;i+=2){
    if(!imgs[i+1]) break;
    const pair=await addPair({original:await fileToDataURL(imgs[i]),target:await fileToDataURL(imgs[i+1]),name:imgs[i].name+" + "+imgs[i+1].name});
    state.pairs.push(pair);
  }
  saveState({pairCount:state.pairs.length});
  render();
}
$("#addPair").onclick=()=>$("#fileInput").click();
$("#fileInput").onchange=e=>addFiles(e.target.files);
$("#dropzone").ondragover=e=>{e.preventDefault();$("#dropzone").style.borderColor="#66717d"};
$("#dropzone").ondragleave=()=>$("#dropzone").style.borderColor="";
$("#dropzone").ondrop=e=>{e.preventDefault();$("#dropzone").style.borderColor="";addFiles(e.dataTransfer.files)};

function drawLoss(){
  if(!state.losses.length)return;
  const max=Math.max(...state.losses),min=Math.min(...state.losses),range=max-min||1;
  const pts=state.losses.map((v,i)=>`${i/(state.losses.length-1||1)*600},${145-((v-min)/range)*120}`).join(" ");
  $("#lossLine").setAttribute("points",pts);
}

async function ensureModel(){
  if(!model){
    model=new TinyImageModel({version:state.modelVersion ?? 0,weights:state.weights});
    await model.init();
    $("#engineStatus").textContent="WebGPU active";
    $("#version").textContent="v"+model.version;
    $("#modelBadge").textContent="MODEL v"+model.version;
    $("#modelParams").textContent=model.parameterCount;
    $("#modelSize").textContent=(model.parameterCount*4)+" B";
  }
  return model;
}

trainBtn.onclick=async()=>{
  if(state.training||!state.pairs.length)return;
  state.training=true;state.losses=[];render();
  $("#trainStatus").textContent="Starting";
  $("#lossHint").textContent="real loss";
  try{
    const m=await ensureModel();
    const epochs=20;
    for(let epoch=1;epoch<=epochs;epoch++){
      let epochLoss=0;
      for(let i=0;i<state.pairs.length;i++){
        const p=state.pairs[i];
        const input=await dataURLToImage(p.original);
        const target=await dataURLToImage(p.target);
        const a=imageToCanvas(input,64), b=imageToCanvas(target,64);
        const result=await m.trainPair(a,b);
        m.applyGradient(result.grad,0.8);
        epochLoss+=result.loss;
        $("#step").textContent=((epoch-1)*state.pairs.length+i+1);
        $("#epoch").textContent=`${epoch} / ${epochs}`;
        $("#loss").textContent=result.loss.toFixed(5);
        $("#progress").style.width=(i+1)/state.pairs.length*100+"%";
        $("#trainStatus").textContent=`Training ${i+1}/${state.pairs.length}`;
        await new Promise(requestAnimationFrame);
      }
      const avg=epochLoss/state.pairs.length;
      state.losses.push(avg); drawLoss();
      $("#loss").textContent=avg.toFixed(5);
      $("#progress").style.width=(epoch/epochs*100)+"%";
    }
    m.version++;
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
  render();
};

testBtn.onclick=async()=>{
  if(!state.pairs.length)return;
  try{
    const m=await ensureModel();
    const p=state.pairs[0];
    const [input,target]=await Promise.all([dataURLToImage(p.original),dataURLToImage(p.target)]);
    const c=imageToCanvas(input,64), t=imageToCanvas(target,64), out=await m.predict(c);
    $("#inputPreview").replaceChildren(c); $("#targetPreview").replaceChildren(t); $("#outputPreview").replaceChildren(out);
  }catch(error){
    $("#outputPreview").textContent=error.message;
  }
};

(async()=>{
  try{await ensureModel();}catch(error){$("#engineStatus").textContent="WebGPU unavailable"}
  render();
})();
