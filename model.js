const TRAIN_WGSL=`
struct Params { n: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> t: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(4) var<uniform> p: Params;

const SCALE:f32 = 10000.0;
fn sigmoid(v:f32)->f32 { return 1.0/(1.0+exp(-v)); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
  let i=id.x; if(i>=p.n){return;}
  let base=i*3u;
  let r=x[base]; let g=x[base+1u]; let b=x[base+2u];
  for(var o:u32=0u;o<3u;o++){
    let z=w[o*4u]*r+w[o*4u+1u]*g+w[o*4u+2u]*b+w[o*4u+3u];
    let y=sigmoid(z);
    let e=y-t[base+o];
    let dz=2.0*e*y*(1.0-y);
    atomicAdd(&acc[o*4u],i32(dz*r*SCALE));
    atomicAdd(&acc[o*4u+1u],i32(dz*g*SCALE));
    atomicAdd(&acc[o*4u+2u],i32(dz*b*SCALE));
    atomicAdd(&acc[o*4u+3u],i32(dz*SCALE));
    atomicAdd(&acc[12u],i32(e*e*SCALE));
  }
}`;

const PREDICT_WGSL=`
struct Params { n: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
fn sigmoid(v:f32)->f32 { return 1.0/(1.0+exp(-v)); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
  let i=id.x; if(i>=p.n){return;}
  let base=i*3u; let r=x[base];let g=x[base+1u];let b=x[base+2u];
  for(var o:u32=0u;o<3u;o++){
    let z=w[o*4u]*r+w[o*4u+1u]*g+w[o*4u+2u]*b+w[o*4u+3u];
    y[base+o]=sigmoid(z);
  }
}`;

const SCALE=10000;
export function canvasToRGB8(canvas){
  const ctx=canvas.getContext("2d",{willReadFrequently:true}),d=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  const out=new Uint8Array(canvas.width*canvas.height*3);
  for(let i=0,j=0;i<d.length;i+=4){out[j++]=d[i];out[j++]=d[i+1];out[j++]=d[i+2]}
  return out;
}
function toRGBFloats(canvas){const u8=canvasToRGB8(canvas),out=new Float32Array(u8.length);for(let i=0;i<u8.length;i++)out[i]=u8[i]/255;return out}
function packRGB8Arrays(arrays){
  const total=arrays.reduce((n,a)=>n+a.length,0),out=new Float32Array(total);let off=0;
  for(const a of arrays){for(let i=0;i<a.length;i++)out[off+i]=a[i]/255;off+=a.length}
  return out;
}
export function imageToCanvas(source,size=64){
  const c=document.createElement("canvas");c.width=size;c.height=size;c.getContext("2d",{willReadFrequently:true}).drawImage(source,0,0,size,size);return c;
}
function bufferSize(bytes){return Math.max(4,((bytes+3)>>2)<<2)}
export class TinyImageModel{
  constructor(saved){
    this.version=saved?.version??0;this.runtime="WebGPU";this.parameterCount=12;
    this.weights=new Float32Array(saved?.weights?.length===12?saved.weights:new Float32Array(12));
    this.m=new Float32Array(12); this.v=new Float32Array(12); this.optimizerStep=0;
  }
  async init(){
    if(!navigator.gpu)throw new Error("WebGPU is not available in this browser.");
    this.adapter=await navigator.gpu.requestAdapter();if(!this.adapter)throw new Error("No WebGPU adapter found.");
    this.device=await this.adapter.requestDevice();
    this.predictPipeline=this.device.createComputePipeline({layout:"auto",compute:{module:this.device.createShaderModule({code:PREDICT_WGSL}),entryPoint:"main"}});
    this.trainPipeline=this.device.createComputePipeline({layout:"auto",compute:{module:this.device.createShaderModule({code:TRAIN_WGSL}),entryPoint:"main"}});
    return this;
  }
  async trainBatch(inputs,targets){
    const x=inputs[0] instanceof Uint8Array?packRGB8Arrays(inputs):inputs.reduce((out,a)=>{const f=toRGBFloats(a);const next=new Float32Array(out.length+f.length);next.set(out);next.set(f,out.length);return next},new Float32Array());
    const t=targets[0] instanceof Uint8Array?packRGB8Arrays(targets):targets.reduce((out,a)=>{const f=toRGBFloats(a);const next=new Float32Array(out.length+f.length);next.set(out);next.set(f,out.length);return next},new Float32Array());
    if(x.length!==t.length)throw new Error("Input and target batch sizes do not match.");
    const xLength=x.length,n=xLength/3,device=this.device;
    const xBuf=device.createBuffer({size:bufferSize(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const tBuf=device.createBuffer({size:bufferSize(t.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const wBuf=device.createBuffer({size:this.weights.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const accBuf=device.createBuffer({size:13*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const pBuf=device.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(xBuf,0,x);device.queue.writeBuffer(tBuf,0,t);device.queue.writeBuffer(wBuf,0,this.weights);device.queue.writeBuffer(accBuf,0,new Int32Array(13));device.queue.writeBuffer(pBuf,0,new Uint32Array([n]));
    const bg=device.createBindGroup({layout:this.trainPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:xBuf}},{binding:1,resource:{buffer:tBuf}},{binding:2,resource:{buffer:wBuf}},{binding:3,resource:{buffer:accBuf}},{binding:4,resource:{buffer:pBuf}}]});
    const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(this.trainPipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(Math.ceil(n/64));pass.end();
    const read=device.createBuffer({size:13*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});enc.copyBufferToBuffer(accBuf,0,read,0,13*4);device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);const acc=new Int32Array(read.getMappedRange().slice());read.unmap();
    const grad=new Float32Array(12),denom=n*3;for(let i=0;i<12;i++)grad[i]=acc[i]/SCALE/denom;
    const loss=acc[12]/SCALE/denom;
    for(const b of [xBuf,tBuf,wBuf,accBuf,pBuf,read])b.destroy();
    return {loss,grad};
  }
  applyGradient(grad,lr=.03){
    this.optimizerStep++;
    const b1=.9,b2=.999,eps=1e-8;
    for(let i=0;i<12;i++){
      this.m[i]=b1*this.m[i]+(1-b1)*grad[i];
      this.v[i]=b2*this.v[i]+(1-b2)*grad[i]*grad[i];
      const mh=this.m[i]/(1-Math.pow(b1,this.optimizerStep));
      const vh=this.v[i]/(1-Math.pow(b2,this.optimizerStep));
      this.weights[i]-=lr*mh/(Math.sqrt(vh)+eps);
    }
  }
  async predict(canvas){
    const x=toRGBFloats(canvas),n=canvas.width*canvas.height,device=this.device;
    const xBuf=device.createBuffer({size:bufferSize(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const wBuf=device.createBuffer({size:this.weights.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const yBuf=device.createBuffer({size:bufferSize(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const pBuf=device.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(xBuf,0,x);device.queue.writeBuffer(wBuf,0,this.weights);device.queue.writeBuffer(pBuf,0,new Uint32Array([n]));
    const bg=device.createBindGroup({layout:this.predictPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:xBuf}},{binding:1,resource:{buffer:wBuf}},{binding:2,resource:{buffer:yBuf}},{binding:3,resource:{buffer:pBuf}}]});
    const enc=device.createCommandEncoder(),pass=enc.beginComputePass();pass.setPipeline(this.predictPipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(Math.ceil(n/64));pass.end();
    const read=device.createBuffer({size:bufferSize(x.byteLength),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});enc.copyBufferToBuffer(yBuf,0,read,0,bufferSize(x.byteLength));device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);const output=new Float32Array(read.getMappedRange().slice());read.unmap();
    const c=document.createElement("canvas");c.width=canvas.width;c.height=canvas.height;const image=c.getContext("2d").createImageData(c.width,c.height);
    for(let i=0,j=0;i<output.length;i+=3){image.data[j++]=output[i]*255;image.data[j++]=output[i+1]*255;image.data[j++]=output[i+2]*255;image.data[j++]=255}
    c.getContext("2d").putImageData(image,0,0);
    for(const b of [xBuf,wBuf,yBuf,pBuf,read])b.destroy();return c;
  }
}
