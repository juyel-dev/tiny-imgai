const PDFJS_MODULE="https://cdn.jsdelivr.net/npm/pdfjs-dist@6.3.289/build/pdf.mjs";
const PDFJS_WORKER="https://cdn.jsdelivr.net/npm/pdfjs-dist@6.3.289/build/pdf.worker.mjs";
let libraryPromise=null;

async function pdfjs(){
  if(!libraryPromise){
    libraryPromise=import(PDFJS_MODULE).then(lib=>{
      lib.GlobalWorkerOptions.workerSrc=PDFJS_WORKER;
      return lib;
    });
  }
  return libraryPromise;
}

export async function openPdf(blob){
  const lib=await pdfjs();
  const data=new Uint8Array(await blob.arrayBuffer());
  const task=lib.getDocument({data});
  return task.promise;
}

export async function pdfPageCount(blob){
  const pdf=await openPdf(blob);
  const count=pdf.numPages;
  await disposePdf(pdf);
  return count;
}

export async function disposePdf(pdf){
  if(!pdf)return;
  if(typeof pdf.destroy==="function"){await pdf.destroy();return;}
  if(typeof pdf.cleanup==="function"){await pdf.cleanup();}
}

export async function renderPdfPage(pdf,pageNumber,size=32){
  const page=await pdf.getPage(pageNumber);
  const base=page.getViewport({scale:1});
  const scale=Math.min(size/base.width,size/base.height);
  const viewport=page.getViewport({scale});
  const canvas=document.createElement("canvas");
  canvas.width=size;canvas.height=size;
  const ctx=canvas.getContext("2d",{willReadFrequently:true});
  ctx.fillStyle="#fff";ctx.fillRect(0,0,size,size);
  const x=(size-viewport.width)/2,y=(size-viewport.height)/2;
  await page.render({canvasContext:ctx,viewport,transform:[1,0,0,1,x,y]}).promise;
  return canvas;
}
