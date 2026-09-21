const state={pairs:[],training:false,losses:[]};
const $=s=>document.querySelector(s);
const pairCount=$("#pairCount"),trainDataset=$("#trainDataset"),datasetList=$("#datasetList"),trainBtn=$("#trainBtn"),testBtn=$("#testBtn");

function render(){
  pairCount.textContent=state.pairs.length;
  trainDataset.textContent=state.pairs.length+" pair"+(state.pairs.length===1?"":"s");
  trainBtn.disabled=state.pairs.length===0||state.training;
  testBtn.disabled=state.pairs.length===0;
  datasetList.innerHTML=state.pairs.length?state.pairs.slice(-8).reverse().map((p,i)=>`<div class="pair"><span class="pair-id">${p.id} · ${p.name}</span><span class="pair-status">ready</span></div>`).join(""):'<div class="empty">No training pairs yet.</div>';
}
function addFiles(files){
  const imgs=[...files].filter(f=>f.type.startsWith("image/"));
  for(let i=0;i<imgs.length;i+=2){
    if(!imgs[i+1]) break;
    state.pairs.push({id:"pair_"+String(state.pairs.length+1).padStart(4,"0"),name:imgs[i].name+" + "+imgs[i+1].name,original:imgs[i],target:imgs[i+1]});
  }
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
trainBtn.onclick=()=>{
  if(state.training)return;
  state.training=true;render();
  let step=0,total=20;state.losses=[];
  $("#trainStatus").textContent="Training";
  $("#lossHint").textContent="live";
  const timer=setInterval(()=>{
    step++;
    const loss=2.6*Math.exp(-step/5)+0.12+Math.random()*.08;
    state.losses.push(loss);
    $("#step").textContent=step*100;
    $("#epoch").textContent=`${step} / ${total}`;
    $("#loss").textContent=loss.toFixed(3);
    $("#progress").style.width=(step/total*100)+"%";
    drawLoss();
    if(step>=total){
      clearInterval(timer);state.training=false;
      $("#trainStatus").textContent="Complete";
      $("#modelBadge").textContent="MODEL v1";
      $("#version").textContent="v1";
      render();
    }
  },180);
};

render();