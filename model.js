const WGSL=`
struct Params { n: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

fn sigmoid(v:f32)->f32 { return 1.0/(1.0+exp(-v)); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i=id.x;
  if(i>=p.n){return;}
  let base=i*3u;
  let r=x[base]; let g=x[base+1u]; let b=x[base+2u];
  for(var o:u32=0u;o<3u;o++){
    let z=w[o*4u]*r+w[o*4u+1u]*g+w[o*4u+2u]*b+w[o*4u+3u];
    y[base+o]=sigmoid(z);
  }
}`;
const F32_SIZE=4;

function toRGBFloats(canvas){
  const ctx=canvas.getContext("2d",{willReadFrequently:true});
  const d=ctx.getImageData(0,0,canvas.width,canvas.height).data;
  const out=new Float32Array(canvas.width*canvas.height*3);
  for(let i=0,j=0;i<d.length;i+=4){out[j++]=d[i]/255;out[j++]=d[i+1]/255;out[j++]=d[i+2]/255}
  return out;
}
export function imageToCanvas(source,size=64){
  const c=document.createElement("canvas"); c.width=size;c.height=size;
  const ctx=c.getContext("2d",{willReadFrequently:true});
  ctx.drawImage(source,0,0,size,size); return c;
}

export class TinyImageModel {
  constructor(saved){
    this.version=saved?.version ?? 0; this.runtime="WebGPU"; this.parameterCount=12;
    this.weights=new Float32Array(saved?.weights?.length===12?saved.weights:[8,8,8,-12,8,8,8,-12,8,8,8,-12]);
  }
  async init(){
    if(!navigator.gpu) throw new Error("WebGPU is not available in this browser.");
    this.adapter=await navigator.gpu.requestAdapter();
    if(!this.adapter) throw new Error("No WebGPU adapter found.");
    this.device=await this.adapter.requestDevice();
    this.pipeline=this.device.createComputePipeline({
      layout:"auto",compute:{module:this.device.createShaderModule({code:WGSL}),entryPoint:"main"}
    });
    return this;
  }
  async predict(canvas){
    const x=toRGBFloats(canvas), n=canvas.width*canvas.height;
    const output=await this.forward(x,n);
    const outCanvas=document.createElement("canvas");outCanvas.width=canvas.width;outCanvas.height=canvas.height;
    const ctx=outCanvas.getContext("2d"), image=ctx.createImageData(canvas.width,canvas.height);
    for(let i=0,j=0;i<output.length;i+=3){image.data[j++]=output[i]*255;image.data[j++]=output[i+1]*255;image.data[j++]=output[i+2]*255;image.data[j++]=255}
    ctx.putImageData(image,0,0); return outCanvas;
  }
  async trainPair(inputCanvas,targetCanvas){
    const x=toRGBFloats(inputCanvas), t=toRGBFloats(targetCanvas), n=inputCanvas.width*inputCanvas.height;
    const y=await this.forward(x,n);
    const grad=new Float32Array(12); let loss=0;
    for(let i=0,p=0;i<n;i++,p+=3){
      for(let o=0;o<3;o++){
        const yp=y[p+o], tp=t[p+o], e=yp-tp, dz=2*e*yp*(1-yp);
        loss += e*e;
        grad[o*4]+=dz*x[p]; grad[o*4+1]+=dz*x[p+1]; grad[o*4+2]+=dz*x[p+2]; grad[o*4+3]+=dz;
      }
    }
    const denom=n*3; for(let i=0;i<12;i++)grad[i]/=denom;
    return {loss:loss/denom,grad};
  }
  applyGradient(grad,lr=0.8){
    for(let i=0;i<12;i++) this.weights[i]-=lr*grad[i];
  }
  async forward(x,n){
    const device=this.device;
    const xBuf=device.createBuffer({size:x.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const wBuf=device.createBuffer({size:this.weights.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const yBuf=device.createBuffer({size:x.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const pBuf=device.createBuffer({size:4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(xBuf,0,x); device.queue.writeBuffer(wBuf,0,this.weights); device.queue.writeBuffer(pBuf,0,new Uint32Array([n]));
    const bg=device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:xBuf}},{binding:1,resource:{buffer:wBuf}},{binding:2,resource:{buffer:yBuf}},{binding:3,resource:{buffer:pBuf}}
    ]});
    const enc=device.createCommandEncoder(); const pass=enc.beginComputePass();
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(Math.ceil(n/64));pass.end();
    const read=device.createBuffer({size:x.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    enc.copyBufferToBuffer(yBuf,0,read,0,x.byteLength);device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ); const result=new Float32Array(read.getMappedRange().slice());read.unmap();
    for(const b of [xBuf,wBuf,yBuf,pBuf,read])b.destroy(); return result;
  }
}
