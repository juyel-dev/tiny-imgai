const DB_NAME="tiny-imgai";
const DB_VERSION=3;
const PAIRS="pairs";
const DOCS="documents";
const META="meta";

export function openDataset(){
  return new Promise((resolve,reject)=>{
    const req=indexedDB.open(DB_NAME,DB_VERSION);
    req.onupgradeneeded=()=>{
      const db=req.result;
      if(!db.objectStoreNames.contains(PAIRS))db.createObjectStore(PAIRS,{keyPath:"uuid"});
      if(!db.objectStoreNames.contains(DOCS))db.createObjectStore(DOCS,{keyPath:"uuid"});
      if(!db.objectStoreNames.contains(META))db.createObjectStore(META,{keyPath:"key"});
    };
    req.onsuccess=()=>resolve(req.result);
    req.onerror=()=>reject(req.error);
  });
}

export async function addPdfDataset({originalFile,processedFile,originalPages,processedPages}){
  if(originalPages!==processedPages)throw new Error("The PDFs have different page counts. Pairing requires matching page counts.");
  const existing=await listPairs();
  const nextStart=existing.reduce((m,p)=>Math.max(m,Number(p.pageNumber)||0),0)+1;
  const db=await openDataset();
  const originalDocId=crypto.randomUUID(),processedDocId=crypto.randomUUID(),now=new Date().toISOString();
  const pairs=[];
  return new Promise((resolve,reject)=>{
    const tx=db.transaction([PAIRS,DOCS,META],"readwrite");
    const pairsStore=tx.objectStore(PAIRS),docs=tx.objectStore(DOCS),meta=tx.objectStore(META);
    docs.put({uuid:originalDocId,role:"original",name:originalFile.name,blob:originalFile,pageCount:originalPages,createdAt:now});
    docs.put({uuid:processedDocId,role:"processed",name:processedFile.name,blob:processedFile,pageCount:processedPages,createdAt:now});
    for(let page=1;page<=originalPages;page++){
      pairs.push({uuid:crypto.randomUUID(),pageNumber:nextStart+page-1,originalDocId,processedDocId,originalPage:page,processedPage:page,createdAt:now});
      pairsStore.put(pairs[pairs.length-1]);
    }
    meta.put({key:"pairCounter",value:nextStart+originalPages-1});
    tx.oncomplete=()=>resolve({originalDocId,processedDocId,pairs});
    tx.onerror=()=>reject(tx.error);
    tx.onabort=()=>reject(tx.error||new Error("Could not save PDF dataset."));
  });
}

export async function getDocument(uuid){
  const db=await openDataset();
  return new Promise((resolve,reject)=>{
    const req=db.transaction(DOCS).objectStore(DOCS).get(uuid);
    req.onsuccess=()=>resolve(req.result);
    req.onerror=()=>reject(req.error);
  });
}

export async function listPairs(){
  const db=await openDataset();
  return new Promise((resolve,reject)=>{
    const req=db.transaction(PAIRS).objectStore(PAIRS).getAll();
    req.onsuccess=()=>resolve(req.result.sort((a,b)=>(Number(a.pageNumber)||0)-(Number(b.pageNumber)||0)));
    req.onerror=()=>reject(req.error);
  });
}