const CONV_CONFIG={
  inChannels:3,
  hiddenChannels:4,
  outChannels:3,
  kernel:3
};
const L1=CONV_CONFIG.hiddenChannels*(CONV_CONFIG.inChannels*9+1);
const L2=CONV_CONFIG.outChannels*(CONV_CONFIG.hiddenChannels*9+1);
const PARAMS=L1+L2;
const ACCS=PARAMS+1;
const SCALE=1000.0;
const HIDDEN=CONV_CONFIG.hiddenChannels;
const OUT=CONV_CONFIG.outChannels;
const K=CONV_CONFIG.kernel;
const L2_BASE=L1;

const HIDDEN_WGSL=`
struct Params { pixels:u32, width:u32, height:u32 };
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read_write> hidden:array<f32>;
@group(0) @binding(3) var<uniform> p:Params;

fn l1(h:u32,c:u32,k:u32)->u32{return h*28u+c*9u+k;}
fn inputAt(base:u32,px:i32,py:i32,c:u32)->f32{
  if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
  return x[base+(u32(py)*p.width+u32(px))*3u+c];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=p.pixels){return;}
  let hw=p.width*p.height, sample=i/hw, local=i%hw;
  let px=i32(local%p.width),py=i32(local/p.width);
  let base=sample*hw*3u, hiddenBase=i*4u;
  for(var h:u32=0u;h<4u;h++){
    var sum=w[l1(h,0u,0u)+27u];
    for(var c:u32=0u;c<3u;c++){
      for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
        sum+=w[l1(h,c,ky*3u+kx)]*inputAt(base,px+i32(kx)-1,py+i32(ky)-1,c);
      }}
    }
    hidden[hiddenBase+h]=max(sum,0.0);
  }
}`;

const OUTPUT_WGSL=`
struct Params { pixels:u32, width:u32, height:u32 };
@group(0) @binding(0) var<storage,read> hidden:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
@group(0) @binding(3) var<uniform> p:Params;

fn l2(o:u32,h:u32,k:u32)->u32{return 112u+o*37u+h*9u+k;}
fn hiddenAt(sample:u32,px:i32,py:i32,h:u32)->f32{
  if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
  let idx=(sample*p.width*p.height+u32(py)*p.width+u32(px))*4u+h;
  return hidden[idx];
}
fn sigmoid(v:f32)->f32{return 1.0/(1.0+exp(-v));}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=p.pixels){return;}
  let hw=p.width*p.height,sample=i/hw,local=i%hw;
  let px=i32(local%p.width),py=i32(local/p.width),base=local*3u+i/hw*hw*3u;
  for(var o:u32=0u;o<3u;o++){
    var z=w[112u+o*37u+36u];
    for(var h:u32=0u;h<4u;h++){
      for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
        z+=w[l2(o,h,ky*3u+kx)]*hiddenAt(sample,px+i32(kx)-1,py+i32(ky)-1,h);
      }}
    }
    out[base+o]=sigmoid(z);
  }
}`;

const GRAD2_WGSL=`
struct Params { pixels:u32, width:u32, height:u32 };
@group(0) @binding(0) var<storage,read> hidden:array<f32>;
@group(0) @binding(1) var<storage,read> pred:array<f32>;
@group(0) @binding(2) var<storage,read> target:array<f32>;
@group(0) @binding(3) var<storage,read> w:array<f32>;
@group(0) @binding(4) var<storage,read_write> acc:array<atomic<i32>>;
@group(0) @binding(5) var<storage,read_write> dy:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
const SCALE:f32=1000.0;
fn hiddenAt(sample:u32,px:i32,py:i32,h:u32)->f32{
  if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
  return hidden[(sample*p.width*p.height+u32(py)*p.width+u32(px))*4u+h];
}
fn l2(o:u32,h:u32,k:u32)->u32{return 112u+o*37u+h*9u+k;}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=p.pixels){return;}
  let hw=p.width*p.height,sample=i/hw,local=i%hw;
  let px=i32(local%p.width),py=i32(local/p.width),base=sample*hw*3u,idx=base+local*3u;
  for(var o:u32=0u;o<3u;o++){
    let y=pred[idx+o],tv=target[idx+o],e=y-tv,ddy=2.0*e*y*(1.0-y);
    dy[idx+o]=ddy;
    atomicAdd(&acc[223u],i32(e*e*SCALE));
    for(var h:u32=0u;h<4u;h++){
      for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
        let hx=px+i32(kx)-1,hy=py+i32(ky)-1;
        if(hx>=0&&hy>=0&&hx<i32(p.width)&&hy<i32(p.height)){
          atomicAdd(&acc[l2(o,h,ky*3u+kx)],i32(ddy*hiddenAt(sample,hx,hy,h)*SCALE));
        }
      }}
    }
    atomicAdd(&acc[112u+o*37u+36u],i32(ddy*SCALE));
  }
}`;

const GRAD1_WGSL=`
struct Params { pixels:u32, width:u32, height:u32 };
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> hidden:array<f32>;
@group(0) @binding(2) var<storage,read> w:array<f32>;
@group(0) @binding(3) var<storage,read> dy:array<f32>;
@group(0) @binding(4) var<storage,read_write> acc:array<atomic<i32>>;
@group(0) @binding(5) var<uniform> p:Params;
const SCALE:f32=1000.0;
fn l1(h:u32,c:u32,k:u32)->u32{return h*28u+c*9u+k;}
fn l2(o:u32,h:u32,k:u32)->u32{return 112u+o*37u+h*9u+k;}
fn inputAt(base:u32,px:i32,py:i32,c:u32)->f32{
  if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
  return x[base+(u32(py)*p.width+u32(px))*3u+c];
}
fn hiddenAt(sample:u32,px:i32,py:i32,h:u32)->f32{
  if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
  return hidden[(sample*p.width*p.height+u32(py)*p.width+u32(px))*4u+h];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
  let linear=id.x;
  if(linear>=p.pixels*4u){return;}
  let h=linear%4u,i=linear/4u;
  let hw=p.width*p.height,sample=i/hw,local=i%hw;
  let px=i32(local%p.width),py=i32(local/p.width),base=sample*hw*3u;
  if(hidden[linear]<=0.0){return;}

  var dh=0.0;
  for(var o:u32=0u;o<3u;o++){
    for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
      let ox=px-i32(kx)+1,oy=py-i32(ky)+1;
      if(ox>=0&&oy>=0&&ox<i32(p.width)&&oy<i32(p.height)){
        let dyIdx=(sample*hw+u32(oy)*p.width+u32(ox))*3u+o;
        dh+=dy[dyIdx]*w[l2(o,h,ky*3u+kx)];
      }
    }}
  }

  for(var c:u32=0u;c<3u;c++){
    for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
      atomicAdd(&acc[l1(h,c,ky*3u+kx)],i32(dh*inputAt(base,px+i32(kx)-1,py+i32(ky)-1,c)*SCALE));
    }}
  }
  atomicAdd(&acc[l1(h,0u,0u)+27u],i32(dh*SCALE));
}`;

function size4(bytes){return Math.max(4,((bytes+3)>>2)<<2);}
export function canvasToRGB8(canvas){
  const d=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,canvas.width,canvas.height).data;
  const out=new Uint8Array(canvas.width*canvas.height*3);
  for(let i=0,j=0;i<d.length;i+=4){out[j++]=d[i];out[j++]=d[i+1];out[j++]=d[i+2]}
  return out;
}
function toF32(canvas){
  const u=canvasToRGB8(canvas),f=new Float32Array(u.length);
  for(let i=0;i<u.length;i++)f[i]=u[i]/255;return f;
}
function pack(arrays){
  const total=arrays.reduce((n,a)=>n+a.length,0),f=new Float32Array(total);let off=0;
  for(const a of arrays){for(let i=0;i<a.length;i++)f[off+i]=a[i]/255;off+=a.length}
  return f;
}
export function imageToCanvas(source,size=64){
  const c=document.createElement("canvas");c.width=size;c.height=size;
  c.getContext("2d",{willReadFrequently:true}).drawImage(source,0,0,size,size);return c;
}

export class TinyImageModel{
  constructor(saved){
    const valid=Array.isArray(saved?.weights)&&saved.weights.length===PARAMS;
    this.version=valid?(saved.version??0):0;
    this.weights=new Float32Array(valid?saved.weights:PARAMS);
    this.m=new Float32Array(PARAMS);this.v=new Float32Array(PARAMS);this.optimizerStep=0;
    if(!valid)this.initWeights();
    this.parameterCount=PARAMS;this.modelBytes=PARAMS*4;
    this.architecture="3×3 Conv 3→4 + ReLU + 3×3 Conv 4→3";
  }
  initWeights(){
    for(let i=0;i<PARAMS;i++)this.weights[i]=(Math.random()-0.5)*0.04;
  }
  async init(){
    if(!navigator.gpu)throw new Error("WebGPU is not available in this browser.");
    this.adapter=await navigator.gpu.requestAdapter();if(!this.adapter)throw new Error("No WebGPU adapter found.");
    this.device=await this.adapter.requestDevice();const mk=code=>this.device.createShaderModule({code});
    this.hiddenPipeline=this.device.createComputePipeline({layout:"auto",compute:{module:mk(HIDDEN_WGSL),entryPoint:"main"}});
    this.outputPipeline=this.device.createComputePipeline({layout:"auto",compute:{module:mk(OUTPUT_WGSL),entryPoint:"main"}});
    this.grad2Pipeline=this.device.createComputePipeline({layout:"auto",compute:{module:mk(GRAD2_WGSL),entryPoint:"main"}});
    this.grad1Pipeline=this.device.createComputePipeline({layout:"auto",compute:{module:mk(GRAD1_WGSL),entryPoint:"main"}});
    return this;
  }
  async trainBatch(inputs,targets){
    const x=inputs[0] instanceof Uint8Array?pack(inputs):inputs.reduce((out,c)=>{const f=toF32(c),n=new Float32Array(out.length+f.length);n.set(out);n.set(f,out.length);return n},new Float32Array());
    const t=targets[0] instanceof Uint8Array?pack(targets):targets.reduce((out,c)=>{const f=toF32(c),n=new Float32Array(out.length+f.length);n.set(out);n.set(f,out.length);return n},new Float32Array());
    if(x.length!==t.length)throw new Error("Input and target batch sizes do not match.");
    const pixels=x.length/3,perImage=pixels/inputs.length,width=Math.sqrt(perImage);
    if(!Number.isInteger(width))throw new Error("Training tensors must be square.");
    const height=width,device=this.device;
    const xBuf=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const tBuf=device.createBuffer({size:size4(t.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const wBuf=device.createBuffer({size:size4(this.weights.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const hiddenBuf=device.createBuffer({size:size4(pixels*HIDDEN*4),usage:GPUBufferUsage.STORAGE});
    const predBuf=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const dyBuf=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE});
    const accBuf=device.createBuffer({size:ACCS*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const pBuf=device.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(xBuf,0,x);device.queue.writeBuffer(tBuf,0,t);device.queue.writeBuffer(wBuf,0,this.weights);device.queue.writeBuffer(accBuf,0,new Int32Array(ACCS));device.queue.writeBuffer(pBuf,0,new Uint32Array([pixels,width,height]));

    const hbg=device.createBindGroup({layout:this.hiddenPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:xBuf}},{binding:1,resource:{buffer:wBuf}},{binding:2,resource:{buffer:hiddenBuf}},{binding:3,resource:{buffer:pBuf}}
    ]});
    const obg=device.createBindGroup({layout:this.outputPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:hiddenBuf}},{binding:1,resource:{buffer:wBuf}},{binding:2,resource:{buffer:predBuf}},{binding:3,resource:{buffer:pBuf}}
    ]});
    const g2bg=device.createBindGroup({layout:this.grad2Pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:hiddenBuf}},{binding:1,resource:{buffer:predBuf}},{binding:2,resource:{buffer:tBuf}},{binding:3,resource:{buffer:wBuf}},
      {binding:4,resource:{buffer:accBuf}},{binding:5,resource:{buffer:dyBuf}},{binding:6,resource:{buffer:pBuf}}
    ]});
    const g1bg=device.createBindGroup({layout:this.grad1Pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:xBuf}},{binding:1,resource:{buffer:hiddenBuf}},{binding:2,resource:{buffer:wBuf}},{binding:3,resource:{buffer:dyBuf}},
      {binding:4,resource:{buffer:accBuf}},{binding:5,resource:{buffer:pBuf}}
    ]});
    const enc=device.createCommandEncoder();
    let pass=enc.beginComputePass();pass.setPipeline(this.hiddenPipeline);pass.setBindGroup(0,hbg);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    pass=enc.beginComputePass();pass.setPipeline(this.outputPipeline);pass.setBindGroup(0,obg);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    pass=enc.beginComputePass();pass.setPipeline(this.grad2Pipeline);pass.setBindGroup(0,g2bg);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    pass=enc.beginComputePass();pass.setPipeline(this.grad1Pipeline);pass.setBindGroup(0,g1bg);pass.dispatchWorkgroups(Math.ceil(pixels*HIDDEN/64));pass.end();

    const read=device.createBuffer({size:ACCS*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    enc.copyBufferToBuffer(accBuf,0,read,0,ACCS*4);device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);const acc=new Int32Array(read.getMappedRange().slice());read.unmap();
    const denom=pixels*3,grad=new Float32Array(PARAMS);
    for(let i=0;i<PARAMS;i++)grad[i]=acc[i]/SCALE/denom;
    const loss=acc[PARAMS]/SCALE/denom;
    for(const b of [xBuf,tBuf,wBuf,hiddenBuf,predBuf,dyBuf,accBuf,pBuf,read])b.destroy();
    return {loss,grad};
  }
  applyGradient(grad,lr=.01){
    this.optimizerStep++;const b1=.9,b2=.999,eps=1e-8;
    for(let i=0;i<PARAMS;i++){
      this.m[i]=b1*this.m[i]+(1-b1)*grad[i];
      this.v[i]=b2*this.v[i]+(1-b2)*grad[i]*grad[i];
      const mh=this.m[i]/(1-Math.pow(b1,this.optimizerStep));
      const vh=this.v[i]/(1-Math.pow(b2,this.optimizerStep));
      this.weights[i]-=lr*mh/(Math.sqrt(vh)+eps);
    }
  }
  async predict(canvas){
    const x=toF32(canvas),pixels=x.length/3,width=canvas.width,height=canvas.height,device=this.device;
    const xBuf=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const wBuf=device.createBuffer({size:size4(this.weights.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const hiddenBuf=device.createBuffer({size:size4(pixels*HIDDEN*4),usage:GPUBufferUsage.STORAGE});
    const outBuf=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const pBuf=device.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(xBuf,0,x);device.queue.writeBuffer(wBuf,0,this.weights);device.queue.writeBuffer(pBuf,0,new Uint32Array([pixels,width,height]));
    const hbg=device.createBindGroup({layout:this.hiddenPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:xBuf}},{binding:1,resource:{buffer:wBuf}},{binding:2,resource:{buffer:hiddenBuf}},{binding:3,resource:{buffer:pBuf}}
    ]});
    const obg=device.createBindGroup({layout:this.outputPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:hiddenBuf}},{binding:1,resource:{buffer:wBuf}},{binding:2,resource:{buffer:outBuf}},{binding:3,resource:{buffer:pBuf}}
    ]});
    const enc=device.createCommandEncoder();
    let pass=enc.beginComputePass();pass.setPipeline(this.hiddenPipeline);pass.setBindGroup(0,hbg);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    pass=enc.beginComputePass();pass.setPipeline(this.outputPipeline);pass.setBindGroup(0,obg);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    const read=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    enc.copyBufferToBuffer(outBuf,0,read,0,size4(x.byteLength));device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);const out=new Float32Array(read.getMappedRange().slice());read.unmap();
    const c=document.createElement("canvas");c.width=width;c.height=height;
    const image=c.getContext("2d").createImageData(width,height);
    for(let i=0,j=0;i<out.length;i+=3){image.data[j++]=Math.max(0,Math.min(255,out[i]*255));image.data[j++]=Math.max(0,Math.min(255,out[i+1]*255));image.data[j++]=Math.max(0,Math.min(255,out[i+2]*255));image.data[j++]=255}
    c.getContext("2d").putImageData(image,0,0);
    for(const b of [xBuf,wBuf,hiddenBuf,outBuf,pBuf,read])b.destroy();
    return c;
  }
}
export const MODEL_INFO={parameterCount:PARAMS,modelBytes:PARAMS*4,architecture:"3×3 Conv 3→4 + ReLU + 3×3 Conv 4→3"};
