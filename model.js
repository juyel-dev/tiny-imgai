const CONV_CONFIG={inChannels:3,hiddenChannels:8,outChannels:3,kernel:3};
const L1=CONV_CONFIG.hiddenChannels*(CONV_CONFIG.inChannels*9+1);
const L2=CONV_CONFIG.hiddenChannels*(CONV_CONFIG.hiddenChannels*9+1);
const L3=CONV_CONFIG.outChannels*(CONV_CONFIG.hiddenChannels*9+1);
const PARAMS=L1+L2+L3;
const ACCS=PARAMS+1;
const SCALE=1000.0;
const HIDDEN=8;

const HIDDEN1_WGSL=`
struct Params{pixels:u32;width:u32;height:u32};
@group(0) @binding(0)var<storage,read>x:array<f32>;
@group(0) @binding(1)var<storage,read>w:array<f32>;
@group(0) @binding(2)var<storage,read_write>h:array<f32>;
@group(0) @binding(3)var<uniform>p:Params;
fn inputAt(base:u32,px:i32,py:i32,c:u32)->f32{
 if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
 return x[base+(u32(py)*p.width+u32(px))*3u+c];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
 let i=id.x;if(i>=p.pixels){return;}
 let hw=p.width*p.height,s=i/hw,local=i%hw,px=i32(local%p.width),py=i32(local/p.width),base=s*hw*3u;
 for(var q:u32=0u;q<8u;q++){
  var z=w[q*28u+27u];
  for(var c:u32=0u;c<3u;c++){for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
   z+=w[q*28u+c*9u+ky*3u+kx]*inputAt(base,px+i32(kx)-1,py+i32(ky)-1,c);
  }}}
  h[i*8u+q]=max(z,0.0);
 }
}`;

const HIDDEN2_WGSL=`
struct Params{pixels:u32};
@group(0) @binding(0)var<storage,read>h:array<f32>;
@group(0) @binding(1)var<storage,read>w:array<f32>;
@group(0) @binding(2)var<storage,read_write>z:array<f32>;
@group(0) @binding(3)var<uniform>p:Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
 let i=id.x;if(i>=p.pixels){return;}
 for(var o:u32=0u;o<8u;o++){
  var v=w[224u+o*73u+72u];
  v+=w[224u+o*73u+0u]*h[i*8u+o];
  z[i*8u+o]=max(v,0.0);
 }
}`;

const OUTPUT_WGSL=`
struct Params{pixels:u32;width:u32;height:u32};
@group(0) @binding(0)var<storage,read>z:array<f32>;
@group(0) @binding(1)var<storage,read>w:array<f32>;
@group(0) @binding(2)var<storage,read>x:array<f32>;
@group(0) @binding(3)var<storage,read_write>y:array<f32>;
@group(0) @binding(4)var<uniform>p:Params;
fn inputAt(base:u32,i:u32)->f32{return x[base+i];}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
 let i=id.x;if(i>=p.pixels){return;}
 let base=i*3u;
 for(var o:u32=0u;o<3u;o++){
  var d=w[808u+o*73u+72u];
  for(var h:u32=0u;h<8u;h++){d+=w[808u+o*73u+h*9u]*z[i*8u+h];}
  y[base+o]=clamp(x[base+o]+d,0.0,1.0);
 }
}`;

const TRAIN_WGSL=`
struct Params{pixels:u32;width:u32;height:u32};
@group(0) @binding(0)var<storage,read>x:array<f32>;
@group(0) @binding(1)var<storage,read>t:array<f32>;
@group(0) @binding(2)var<storage,read>h1:array<f32>;
@group(0) @binding(3)var<storage,read>h2:array<f32>;
@group(0) @binding(4)var<storage,read>pred:array<f32>;
@group(0) @binding(5)var<storage,read>w:array<f32>;
@group(0) @binding(6)var<storage,read_write>acc:array<atomic<i32>>;
@group(0) @binding(7)var<storage,read_write>gradH2:array<f32>;
@group(0) @binding(8)var<uniform>p:Params;
const SCALE:f32=1000.0;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
 let i=id.x;if(i>=p.pixels){return;}
 let base=i*3u;
 for(var o:u32=0u;o<3u;o++){
  let e=pred[base+o]-t[base+o];
  atomicAdd(&acc[PARAMS],i32(e*e*SCALE));
  let d=2.0*e;
  for(var hh:u32=0u;hh<8u;hh++){atomicAdd(&acc[808u+o*73u+hh*9u],i32(d*h2[i*8u+hh]*SCALE));}
  atomicAdd(&acc[808u+o*73u+72u],i32(d*SCALE));
  gradH2[i*8u+o]=d*w[808u+o*73u+o*9u];
 }
}`;

const H2BACK_WGSL=`
struct Params{pixels:u32;width:u32;height:u32};
@group(0) @binding(0)var<storage,read>x:array<f32>;
@group(0) @binding(1)var<storage,read>h1:array<f32>;
@group(0) @binding(2)var<storage,read>h2:array<f32>;
@group(0) @binding(3)var<storage,read>w:array<f32>;
@group(0) @binding(4)var<storage,read>g2:array<f32>;
@group(0) @binding(5)var<storage,read_write>acc:array<atomic<i32>>;
@group(0) @binding(6)var<storage,read_write>g1:array<f32>;
@group(0) @binding(7)var<uniform>p:Params;
const SCALE:f32=1000.0;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
 let i=id.x;if(i>=p.pixels){return;}
 for(var o:u32=0u;o<8u;o++){
  if(h2[i*8u+o]<=0.0){continue;}
  var d=0.0;
  for(var out:u32=0u;out<8u;out++){d+=g2[i*8u+out]*w[224u+out*73u+o*9u];}
  g1[i*8u+o]=d;
  for(var h:u32=0u;h<8u;h++){atomicAdd(&acc[224u+o*73u+h*9u],i32(d*h1[i*8u+h]*SCALE));}
  atomicAdd(&acc[224u+o*73u+72u],i32(d*SCALE));
 }
}`;

const H1BACK_WGSL=`
struct Params{pixels:u32;width:u32;height:u32};
@group(0) @binding(0)var<storage,read>x:array<f32>;
@group(0) @binding(1)var<storage,read>h1:array<f32>;
@group(0) @binding(2)var<storage,read>w:array<f32>;
@group(0) @binding(3)var<storage,read>g1:array<f32>;
@group(0) @binding(4)var<storage,read_write>acc:array<atomic<i32>>;
@group(0) @binding(5)var<uniform>p:Params;
const SCALE:f32=1000.0;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
 let i=id.x;if(i>=p.pixels*8u){return;}
 let h=i%8u,px=i32((i/8u)%p.width),py=i32((i/8u)/p.width),sample=(i/8u)/(p.width*p.height);
 let base=sample*p.width*p.height*3u;
 if(h1[i]<=0.0){return;}
 let d=g1[i];
 for(var c:u32=0u;c<8u;c++){
   for(var k:u32=0u;k<9u;k++){
     var dx=i32(k%3u)-1,dy=i32(k/3u)-1;
     let ix=px+dx,iy=py+dy;
     if(ix>=0&&iy>=0&&ix<i32(p.width)&&iy<i32(p.height)){
       let idx=(sample*p.width*p.height+u32(iy)*p.width+u32(ix))*8u+c;
       atomicAdd(&acc[224u+c*73u+k*8u+h],i32(d*h1[idx]*SCALE));
     }
   }
 }
}`;

function size4(n){return Math.max(4,((n+3)>>2)<<2)}
export function canvasToRGB8(canvas){
 const d=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,canvas.width,canvas.height).data;
 const out=new Uint8Array(canvas.width*canvas.height*3);for(let i=0,j=0;i<d.length;i+=4){out[j++]=d[i];out[j++]=d[i+1];out[j++]=d[i+2]}return out;
}
function toF32(canvas){const u=canvasToRGB8(canvas),f=new Float32Array(u.length);for(let i=0;i<u.length;i++)f[i]=u[i]/255;return f}
function pack(arrays){const n=arrays.reduce((s,a)=>s+a.length,0),f=new Float32Array(n);let o=0;for(const a of arrays){for(let i=0;i<a.length;i++)f[o+i]=a[i]/255;o+=a.length}return f}
export function imageToCanvas(source,size=64){const c=document.createElement("canvas");c.width=size;c.height=size;c.getContext("2d",{willReadFrequently:true}).drawImage(source,0,0,size,size);return c}

export class TinyImageModel{
 constructor(saved){
  const valid=Array.isArray(saved?.weights)&&saved.weights.length===PARAMS;
  this.version=valid?(saved.version??0):0;this.weights=new Float32Array(valid?saved.weights:PARAMS);
  if(!valid)this.initWeights();
  this.m=new Float32Array(PARAMS);this.v=new Float32Array(PARAMS);this.optimizerStep=0;
  this.parameterCount=PARAMS;this.modelBytes=PARAMS*4;
  this.architecture="3×3 Conv 3→8 → 3×3 Conv 8→8 → 3×3 Conv 8→3 + residual skip";
 }
 initWeights(){for(let i=0;i<PARAMS;i++)this.weights[i]=(Math.random()-0.5)*0.02}
 async init(){
  if(!navigator.gpu)throw new Error("WebGPU is not available in this browser.");
  this.adapter=await navigator.gpu.requestAdapter();if(!this.adapter)throw new Error("No WebGPU adapter found.");
  this.device=await this.adapter.requestDevice();const mk=c=>this.device.createShaderModule({code:c});
  this.p1=this.device.createComputePipeline({layout:"auto",compute:{module:mk(HIDDEN1_WGSL),entryPoint:"main"}});
  this.p2=this.device.createComputePipeline({layout:"auto",compute:{module:mk(HIDDEN2_WGSL),entryPoint:"main"}});
  this.po=this.device.createComputePipeline({layout:"auto",compute:{module:mk(OUTPUT_WGSL),entryPoint:"main"}});
  this.pt=this.device.createComputePipeline({layout:"auto",compute:{module:mk(TRAIN_WGSL),entryPoint:"main"}});
  this.pb2=this.device.createComputePipeline({layout:"auto",compute:{module:mk(H2BACK_WGSL),entryPoint:"main"}});
  this.pb1=this.device.createComputePipeline({layout:"auto",compute:{module:mk(H1BACK_WGSL),entryPoint:"main"}});
  return this;
 }
 async trainBatch(inputs,targets){
  const x=inputs[0] instanceof Uint8Array?pack(inputs):inputs.reduce((o,c)=>{const f=toF32(c),n=new Float32Array(o.length+f.length);n.set(o);n.set(f,o.length);return n},new Float32Array());
  const t=targets[0] instanceof Uint8Array?pack(targets):targets.reduce((o,c)=>{const f=toF32(c),n=new Float32Array(o.length+f.length);n.set(o);n.set(f,o.length);return n},new Float32Array());
  if(x.length!==t.length)throw new Error("Input and target batch sizes do not match.");
  const pixels=x.length/3,perImage=pixels/inputs.length,wid=Math.sqrt(perImage);if(!Number.isInteger(wid))throw new Error("Training tensors must be square.");
  const device=this.device,hgt=wid;
  const xb=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const tb=device.createBuffer({size:size4(t.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const wb=device.createBuffer({size:size4(this.weights.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const h1=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE});
  const h2=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE});
  const pred=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE});
  const g2=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE});
  const g1=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE});
  const acc=device.createBuffer({size:ACCS*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const p=device.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(xb,0,x);device.queue.writeBuffer(tb,0,t);device.queue.writeBuffer(wb,0,this.weights);device.queue.writeBuffer(acc,0,new Int32Array(ACCS));device.queue.writeBuffer(p,0,new Uint32Array([pixels,wid,hgt]));
  const bg=(pipe,entries)=>device.createBindGroup({layout:pipe.getBindGroupLayout(0),entries});
  const b1=bg(this.p1,[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:h1}},{binding:3,resource:{buffer:p}}]);
  const b2=bg(this.p2,[{binding:0,resource:{buffer:h1}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:h2}},{binding:3,resource:{buffer:{size:12}}}]);
  const bo=bg(this.po,[{binding:0,resource:{buffer:h2}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:xb}},{binding:3,resource:{buffer:pred}},{binding:4,resource:{buffer:p}}]);
  const bt=bg(this.pt,[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:tb}},{binding:2,resource:{buffer:h1}},{binding:3,resource:{buffer:h2}},{binding:4,resource:{buffer:pred}},{binding:5,resource:{buffer:wb}},{binding:6,resource:{buffer:acc}},{binding:7,resource:{buffer:g2}},{binding:8,resource:{buffer:p}}]);
  const bb2=bg(this.pb2,[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:h1}},{binding:2,resource:{buffer:h2}},{binding:3,resource:{buffer:wb}},{binding:4,resource:{buffer:g2}},{binding:5,resource:{buffer:acc}},{binding:6,resource:{buffer:g1}},{binding:7,resource:{buffer:p}}]);
  const bb1=bg(this.pb1,[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:h1}},{binding:2,resource:{buffer:wb}},{binding:3,resource:{buffer:g1}},{binding:4,resource:{buffer:acc}},{binding:5,resource:{buffer:p}}]);
  const enc=device.createCommandEncoder();let pass=enc.beginComputePass();
  pass.setPipeline(this.p1);pass.setBindGroup(0,b1);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
  pass=enc.beginComputePass();pass.setPipeline(this.p2);pass.setBindGroup(0,b2);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
  pass=enc.beginComputePass();pass.setPipeline(this.po);pass.setBindGroup(0,bo);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
  pass=enc.beginComputePass();pass.setPipeline(this.pt);pass.setBindGroup(0,bt);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
  pass=enc.beginComputePass();pass.setPipeline(this.pb2);pass.setBindGroup(0,bb2);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
  pass=enc.beginComputePass();pass.setPipeline(this.pb1);pass.setBindGroup(0,bb1);pass.dispatchWorkgroups(Math.ceil(pixels*8/64));pass.end();
  const read=device.createBuffer({size:ACCS*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});enc.copyBufferToBuffer(acc,0,read,0,ACCS*4);device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);const a=new Int32Array(read.getMappedRange().slice());read.unmap();
  const denom=pixels*3,grad=new Float32Array(PARAMS);for(let i=0;i<PARAMS;i++)grad[i]=a[i]/SCALE/denom;
  const loss=a[PARAMS]/SCALE/denom;
  for(const b of [xb,tb,wb,h1,h2,pred,g2,g1,acc,p,read])b.destroy();
  return {loss,grad};
 }
 applyGradient(grad,lr=.005){
  this.optimizerStep++;const b1=.9,b2=.999,eps=1e-8;
  for(let i=0;i<PARAMS;i++){this.m[i]=b1*this.m[i]+(1-b1)*grad[i];this.v[i]=b2*this.v[i]+(1-b2)*grad[i]*grad[i];
   const mh=this.m[i]/(1-Math.pow(b1,this.optimizerStep)),vh=this.v[i]/(1-Math.pow(b2,this.optimizerStep));
   this.weights[i]-=lr*mh/(Math.sqrt(vh)+eps);}
 }
 async predict(canvas){
  const x=toF32(canvas),pixels=x.length/3,device=this.device,wid=canvas.width,hgt=canvas.height;
  const xb=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),wb=device.createBuffer({size:size4(this.weights.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const h1=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE}),h2=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE});
  const out=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),p=device.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(xb,0,x);device.queue.writeBuffer(wb,0,this.weights);device.queue.writeBuffer(p,0,new Uint32Array([pixels,wid,hgt]));
  const bh1=device.createBindGroup({layout:this.p1.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:h1}},{binding:3,resource:{buffer:p}}]});
  const bh2p=device.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(bh2p,0,new Uint32Array([pixels]));
  const bh2=device.createBindGroup({layout:this.p2.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h1}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:h2}},{binding:3,resource:{buffer:bh2p}}]});
  const bo=device.createBindGroup({layout:this.po.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h2}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:xb}},{binding:3,resource:{buffer:out}},{binding:4,resource:{buffer:p}}]});
  const enc=device.createCommandEncoder();let pass=enc.beginComputePass();pass.setPipeline(this.p1);pass.setBindGroup(0,bh1);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
  pass=enc.beginComputePass();pass.setPipeline(this.p2);pass.setBindGroup(0,bh2);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
  pass=enc.beginComputePass();pass.setPipeline(this.po);pass.setBindGroup(0,bo);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
  const read=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});enc.copyBufferToBuffer(out,0,read,0,size4(x.byteLength));device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);const o=new Float32Array(read.getMappedRange().slice());read.unmap();
  const c=document.createElement("canvas");c.width=wid;c.height=hgt;const image=c.getContext("2d").createImageData(wid,hgt);
  for(let i=0,j=0;i<o.length;i+=3){image.data[j++]=Math.max(0,Math.min(255,o[i]*255));image.data[j++]=Math.max(0,Math.min(255,o[i+1]*255));image.data[j++]=Math.max(0,Math.min(255,o[i+2]*255));image.data[j++]=255}
  c.getContext("2d").putImageData(image,0,0);for(const b of [xb,wb,h1,h2,out,p,bh2p,read])b.destroy();return c;
 }
}
export const MODEL_INFO={parameterCount:PARAMS,modelBytes:PARAMS*4,architecture:"3×3 Conv 3→8 → 3×3 Conv 8→8 → 3×3 Conv 8→3 + residual skip"};
