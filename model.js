const HIDDEN=8;
const L1=HIDDEN*(3*9+1);          // 224
const L2=3*(HIDDEN*9+1);          // 219
const PARAMS=L1+L2;              // 443
const ACCS=PARAMS+1;
const SCALE=1000.0;

const HIDDEN_WGSL=`
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
  let hw=p.width*p.height,s=i/hw,local=i%hw;
  let px=i32(local%p.width),py=i32(local/p.width),base=s*hw*3u;
  for(var q:u32=0u;q<8u;q++){
    var z=w[q*28u+27u];
    for(var c:u32=0u;c<3u;c++){for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
      z+=w[q*28u+c*9u+ky*3u+kx]*inputAt(base,px+i32(kx)-1,py+i32(ky)-1,c);
    }}}
    h[i*8u+q]=max(z,0.0);
  }
}`;

const OUTPUT_WGSL=`
struct Params{pixels:u32;width:u32;height:u32};
@group(0) @binding(0)var<storage,read>h:array<f32>;
@group(0) @binding(1)var<storage,read>w:array<f32>;
@group(0) @binding(2)var<storage,read>x:array<f32>;
@group(0) @binding(3)var<storage,read_write>y:array<f32>;
@group(0) @binding(4)var<uniform>p:Params;

fn hiddenAt(sample:u32,px:i32,py:i32,c:u32)->f32{
  if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
  return h[(sample*p.width*p.height+u32(py)*p.width+u32(px))*8u+c];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
  let i=id.x;if(i>=p.pixels){return;}
  let hw=p.width*p.height,s=i/hw,local=i%hw;
  let px=i32(local%p.width),py=i32(local/p.width),base=s*hw*3u+local*3u;
  for(var o:u32=0u;o<3u;o++){
    var d=w[224u+o*73u+72u];
    for(var q:u32=0u;q<8u;q++){for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
      d+=w[224u+o*73u+q*9u+ky*3u+kx]*hiddenAt(s,px+i32(kx)-1,py+i32(ky)-1,q);
    }}}
    y[base+o]=clamp(x[base+o]+d,0.0,1.0);
  }
}`;

const GRAD_OUT_WGSL=`
struct Params{pixels:u32;width:u32;height:u32};
@group(0) @binding(0)var<storage,read>h:array<f32>;
@group(0) @binding(1)var<storage,read>pred:array<f32>;
@group(0) @binding(2)var<storage,read>target:array<f32>;
@group(0) @binding(3)var<storage,read_write>acc:array<atomic<i32>>;
@group(0) @binding(4)var<storage,read_write>dh:array<f32>;
@group(0) @binding(5)var<uniform>p:Params;

const SCALE:f32=1000.0;
fn hiddenAt(sample:u32,px:i32,py:i32,c:u32)->f32{
  if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
  return h[(sample*p.width*p.height+u32(py)*p.width+u32(px))*8u+c];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
  let i=id.x;if(i>=p.pixels){return;}
  let hw=p.width*p.height,s=i/hw,local=i%hw;
  let px=i32(local%p.width),py=i32(local/p.width),base=s*hw*3u+local*3u;

  for(var o:u32=0u;o<3u;o++){
    let y=pred[base+o],t=target[base+o],e=y-t;
    let active=select(0.0,1.0,y>0.0001 && y<0.9999);
    let d=2.0*e*active;
    atomicAdd(&acc[PARAMS],i32(e*e*SCALE));
    for(var q:u32=0u;q<8u;q++){for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
      let hx=px+i32(kx)-1,hy=py+i32(ky)-1;
      if(hx>=0&&hy>=0&&hx<i32(p.width)&&hy<i32(p.height)){
        let hi=(s*hw+u32(hy)*p.width+u32(hx))*8u+q;
        atomicAdd(&acc[224u+o*73u+q*9u+ky*3u+kx],i32(d*hiddenAt(s,hx,hy,q)*SCALE));
      }
    }}}
  }

  for(var q:u32=0u;q<8u;q++){
    var total=0.0;
    for(var o:u32=0u;o<3u;o++){
      for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
        let ox=px-i32(kx)+1,oy=py-i32(ky)+1;
        if(ox>=0&&oy>=0&&ox<i32(p.width)&&oy<i32(p.height)){
          let outBase=(s*hw+u32(oy)*p.width+u32(ox))*3u;
          total += 2.0*(pred[outBase+o]-target[outBase+o])*w[224u+o*73u+q*9u+ky*3u+kx];
        }
      }}}
    }
    dh[i*8u+q]=total;
  }
}`;

const GRAD_IN_WGSL=`
struct Params{pixels:u32;width:u32;height:u32};
@group(0) @binding(0)var<storage,read>x:array<f32>;
@group(0) @binding(1)var<storage,read>h:array<f32>;
@group(0) @binding(2)var<storage,read>dh:array<f32>;
@group(0) @binding(3)var<storage,read_write>acc:array<atomic<i32>>;
@group(0) @binding(4)var<uniform>p:Params;

const SCALE:f32=1000.0;
fn inputAt(base:u32,px:i32,py:i32,c:u32)->f32{
  if(px<0||py<0||px>=i32(p.width)||py>=i32(p.height)){return 0.0;}
  return x[base+(u32(py)*p.width+u32(px))*3u+c];
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3<u32>){
  let i=id.x;if(i>=p.pixels*8u){return;}
  let q=i%8u,px=i32((i/8u)%p.width),py=i32((i/8u)/p.width);
  let hw=p.width*p.height,s=(i/8u)/hw,local=u32((i/8u)%hw),base=s*hw*3u;
  let hval=h[i],d=dh[i];
  if(hval<=0.0){return;}
  for(var c:u32=0u;c<3u;c++){for(var ky:u32=0u;ky<3u;ky++){for(var kx:u32=0u;kx<3u;kx++){
    atomicAdd(&acc[q*28u+c*9u+ky*3u+kx],i32(d*inputAt(base,px+i32(kx)-1,py+i32(ky)-1,c)*SCALE));
  }}}
  atomicAdd(&acc[q*28u+27u],i32(d*SCALE));
}`;

function size4(n){return Math.max(4,((n+3)>>2)<<2)}
export function canvasToRGB8(canvas){
  const d=canvas.getContext("2d",{willReadFrequently:true}).getImageData(0,0,canvas.width,canvas.height).data;
  const out=new Uint8Array(canvas.width*canvas.height*3);
  for(let i=0,j=0;i<d.length;i+=4){out[j++]=d[i];out[j++]=d[i+1];out[j++]=d[i+2]}return out;
}
function toF32(canvas){const u=canvasToRGB8(canvas),f=new Float32Array(u.length);for(let i=0;i<u.length;i++)f[i]=u[i]/255;return f}
function pack(arrays){const n=arrays.reduce((s,a)=>s+a.length,0),f=new Float32Array(n);let o=0;for(const a of arrays){for(let i=0;i<a.length;i++)f[o+i]=a[i]/255;o+=a.length}return f}
export function imageToCanvas(source,size=64){const c=document.createElement("canvas");c.width=size;c.height=size;c.getContext("2d",{willReadFrequently:true}).drawImage(source,0,0,size,size);return c}

export class TinyImageModel{
  constructor(saved){
    const valid=Array.isArray(saved?.weights)&&saved.weights.length===PARAMS;
    this.version=valid?(saved.version??0):0;
    this.weights=new Float32Array(valid?saved.weights:PARAMS);
    if(!valid)this.initWeights();
    this.m=new Float32Array(PARAMS);this.v=new Float32Array(PARAMS);this.optimizerStep=0;
    this.parameterCount=PARAMS;this.modelBytes=PARAMS*4;
    this.architecture="3×3 Conv 3→8 → ReLU → 3×3 Conv 8→3 + residual skip";
  }
  initWeights(){for(let i=0;i<PARAMS;i++)this.weights[i]=(Math.random()-0.5)*0.02}
  async init(){
    if(!navigator.gpu)throw new Error("WebGPU is not available in this browser.");
    this.adapter=await navigator.gpu.requestAdapter();if(!this.adapter)throw new Error("No WebGPU adapter found.");
    this.device=await this.adapter.requestDevice();const mk=c=>this.device.createShaderModule({code:c});
    this.pHidden=this.device.createComputePipeline({layout:"auto",compute:{module:mk(HIDDEN_WGSL),entryPoint:"main"}});
    this.pOutput=this.device.createComputePipeline({layout:"auto",compute:{module:mk(OUTPUT_WGSL),entryPoint:"main"}});
    this.pGradOut=this.device.createComputePipeline({layout:"auto",compute:{module:mk(GRAD_OUT_WGSL),entryPoint:"main"}});
    this.pGradIn=this.device.createComputePipeline({layout:"auto",compute:{module:mk(GRAD_IN_WGSL),entryPoint:"main"}});
    return this;
  }
  _pack(vs){return vs[0] instanceof Uint8Array?pack(vs):vs.reduce((o,c)=>{const f=toF32(c),n=new Float32Array(o.length+f.length);n.set(o);n.set(f,o.length);return n},new Float32Array())}
  async trainBatch(inputs,targets){
    const x=this._pack(inputs),t=this._pack(targets);
    if(x.length!==t.length)throw new Error("Input and target batch sizes do not match.");
    const pixels=x.length/3,perImage=pixels/inputs.length,width=Math.sqrt(perImage);
    if(!Number.isInteger(width))throw new Error("Training tensors must be square.");
    const device=this.device,height=width;
    const xb=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const tb=device.createBuffer({size:size4(t.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const wb=device.createBuffer({size:size4(this.weights.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const hb=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE});
    const pb=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const dhb=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE});
    const acc=device.createBuffer({size:ACCS*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    const par=device.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(xb,0,x);device.queue.writeBuffer(tb,0,t);device.queue.writeBuffer(wb,0,this.weights);device.queue.writeBuffer(acc,0,new Int32Array(ACCS));device.queue.writeBuffer(par,0,new Uint32Array([pixels,width,height]));
    const bg=(pipe,entries)=>device.createBindGroup({layout:pipe.getBindGroupLayout(0),entries});
    const bh=bg(this.pHidden,[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:hb}},{binding:3,resource:{buffer:par}}]);
    const bo=bg(this.pOutput,[{binding:0,resource:{buffer:hb}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:xb}},{binding:3,resource:{buffer:pb}},{binding:4,resource:{buffer:par}}]);
    const bgout=bg(this.pGradOut,[{binding:0,resource:{buffer:hb}},{binding:1,resource:{buffer:pb}},{binding:2,resource:{buffer:tb}},{binding:3,resource:{buffer:acc}},{binding:4,resource:{buffer:dhb}},{binding:5,resource:{buffer:par}}]);
    const bgin=bg(this.pGradIn,[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:hb}},{binding:2,resource:{buffer:dhb}},{binding:3,resource:{buffer:acc}},{binding:4,resource:{buffer:par}}]);
    const enc=device.createCommandEncoder();let pass=enc.beginComputePass();
    pass.setPipeline(this.pHidden);pass.setBindGroup(0,bh);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    pass=enc.beginComputePass();pass.setPipeline(this.pOutput);pass.setBindGroup(0,bo);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    pass=enc.beginComputePass();pass.setPipeline(this.pGradOut);pass.setBindGroup(0,bgout);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    pass=enc.beginComputePass();pass.setPipeline(this.pGradIn);pass.setBindGroup(0,bgin);pass.dispatchWorkgroups(Math.ceil(pixels*8/64));pass.end();
    const read=device.createBuffer({size:ACCS*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    enc.copyBufferToBuffer(acc,0,read,0,ACCS*4);device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);const a=new Int32Array(read.getMappedRange().slice());read.unmap();
    const denom=pixels*3,grad=new Float32Array(PARAMS);for(let i=0;i<PARAMS;i++)grad[i]=a[i]/SCALE/denom;
    const loss=a[PARAMS]/SCALE/denom;
    for(const b of [xb,tb,wb,hb,pb,dhb,acc,par,read])b.destroy();
    return {loss,grad};
  }
  applyGradient(g,lr=.005){
    this.optimizerStep++;const b1=.9,b2=.999,eps=1e-8;
    for(let i=0;i<PARAMS;i++){this.m[i]=b1*this.m[i]+(1-b1)*g[i];this.v[i]=b2*this.v[i]+(1-b2)*g[i]*g[i];
      const mh=this.m[i]/(1-Math.pow(b1,this.optimizerStep)),vh=this.v[i]/(1-Math.pow(b2,this.optimizerStep));
      this.weights[i]-=lr*mh/(Math.sqrt(vh)+eps);}
  }
  async predict(canvas){
    const x=toF32(canvas),pixels=x.length/3,device=this.device,wid=canvas.width,hgt=canvas.height;
    const xb=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const wb=device.createBuffer({size:size4(this.weights.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    const hb=device.createBuffer({size:size4(pixels*8*4),usage:GPUBufferUsage.STORAGE});
    const out=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const par=device.createBuffer({size:12,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(xb,0,x);device.queue.writeBuffer(wb,0,this.weights);device.queue.writeBuffer(par,0,new Uint32Array([pixels,wid,hgt]));
    const bh= device.createBindGroup({layout:this.pHidden.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:xb}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:hb}},{binding:3,resource:{buffer:par}}]});
    const bo=device.createBindGroup({layout:this.pOutput.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:hb}},{binding:1,resource:{buffer:wb}},{binding:2,resource:{buffer:xb}},{binding:3,resource:{buffer:out}},{binding:4,resource:{buffer:par}}]});
    const enc=device.createCommandEncoder();let pass=enc.beginComputePass();
    pass.setPipeline(this.pHidden);pass.setBindGroup(0,bh);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    pass=enc.beginComputePass();pass.setPipeline(this.pOutput);pass.setBindGroup(0,bo);pass.dispatchWorkgroups(Math.ceil(pixels/64));pass.end();
    const read=device.createBuffer({size:size4(x.byteLength),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    enc.copyBufferToBuffer(out,0,read,0,size4(x.byteLength));device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);const o=new Float32Array(read.getMappedRange().slice());read.unmap();
    const c=document.createElement("canvas");c.width=wid;c.height=hgt;const image=c.getContext("2d").createImageData(wid,hgt);
    for(let i=0,j=0;i<o.length;i+=3){image.data[j++]=Math.max(0,Math.min(255,o[i]*255));image.data[j++]=Math.max(0,Math.min(255,o[i+1]*255));image.data[j++]=Math.max(0,Math.min(255,o[i+2]*255));image.data[j++]=255}
    c.getContext("2d").putImageData(image,0,0);for(const b of [xb,wb,hb,out,par,read])b.destroy();return c;
  }
}
export const MODEL_INFO={parameterCount:PARAMS,modelBytes:PARAMS*4,architecture:"3×3 Conv 3→8 → ReLU → 3×3 Conv 8→3 + residual skip"};
