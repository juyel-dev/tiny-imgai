import { addPair, listPairs } from "./dataset.js";
import { saveState } from "./storage.js";
import { TinyImageModel } from "./model.js";

const state={pairs:await listPairs(),training:false,losses:[]};
const $=s=>document.querySelector(s);
const pairCount=$("#pairCount"),trainDataset=$("#trainDataset"),datasetList=$("#datasetList"),trainBtn=$("#trainBtn"),testBtn=$("#testBtn");

function render(){
  pairCount.textContent=state.pairs.length;
  trainDataset.textContent=state.pairs.length+" pair"+(state.pairs.length===1?"":"s");
  trainBtn.disabled=state.pairs.length===0||state.training;
  testBtn.disabled=state.pairs.length===0;
  datasetList.innerHTML=state.pairs.length?state.pairs.slice(-8).reverse().map(p=>`<div class="pair"><span class="pair-id">${p.pageNumber} · ${p.name}</span><span class="pair-status">ready</span></div>`).join(""):'<div class="empty">No training pairs yet.</div>';
}
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
function fileToDataURL(file){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(file)})}
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
trainBtn.onclick=()=>{
  if(state.training)return;
  state.training=true;state.losses=[];render();$("#trainStatus").textContent="Training";
  const worker=new Worker("./worker.js",{type:"module"});
  worker.onmessage=e=>{
    const d=e.data;
    if(d.type==="progress"){
      state.losses.push(d.loss);$("#step").textContent=d.step*100;$("#epoch").textContent=`${d.epoch} / ${d.epochs}`;$("#loss").textContent=d.loss.toFixed(3);$("#progress").style.width=(d.epoch/d.epochs*100)+"%";$("#lossHint").textContent="live";drawLoss();
    } else { state.training=false;$("#trainStatus").textContent="Complete";$("#modelBadge").textContent="MODEL v1";$("#version").textContent="v1";render();worker.terminate(); }
  };
  worker.onerror=()=>{state.training=false;$("#trainStatus").textContent="Error";render();worker.terminate()};
  worker.postMessage({type:"train",epochs:20,version:1});
};
testBtn.onclick=async()=>{
  try{await new TinyImageModel().init();$("#previewOutput").textContent="WebGPU ready";}catch(e){$("#previewOutput").textContent=e.message}
};
render();