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
    req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);
  });
}

export async function addPdfDataset({originalFile,processedFile,originalPages,processedPages}){
  if(originalPages!==processedPages)throw new Error("The PDFs have different page counts. Page pairing needs matching page counts.");
  const db=await openDataset();
  const originalDocId=crypto.randomUUID(),processedDocId=crypto.randomUUID();
  const now=new Date().toISOString();
  const pairs=[];
  const tx=db.transaction([PAIRS,DOCS,META],"readwrite");
  const meta=tx.objectStore(META),pairsStore=tx.objectStore(PAIRS),docs=tx.objectStore(DOCS);
  const counter=await requestValue(meta,"pairCounter");
  let next=Math.max(Number(counter?.value)||0,await maxPairNumber(pairsStore))+1;
  docs.put({uuid:originalDocId,role:"original",name:originalFile.name,blob:originalFile,pageCount:originalPages,createdAt:now});
  docs.put({uuid:processedDocId,role:"processed",name:processedFile.name,blob:processedFile,pageCount:processedPages,createdAt:now});
  for(let page=1;page<=originalPages;page++){
    const pair={uuid:crypto.randomUUID(),pageNumber:next++,originalDocId,processedDocId,originalPage:page,processedPage:page,createdAt:now};
    pairsStore.put(pair);pairs.push(pair);
  }
  meta.put({key:"pairCounter",value:next-1});
  return new Promise((resolve,reject)=>{tx.oncomplete=()=>resolve({originalDocId,processedDocId,pairs});tx.onerror=()=>reject(tx.error)});
}

function requestValue(store,key){return new Promise((resolve,reject)=>{const r=store.get(key);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)})}
function maxPairNumber(store){return new Promise((resolve,reject)=>{const r=store.getAll();r.onsuccess=()=>resolve(r.result.reduce((m,p)=>Math.max(m,Number(p.pageNumber)||0),0));r.onerror=()=>reject(r.error)})}

export async function getDocument(uuid){
  const db=await openDataset();
  return new Promise((resolve,reject)=>{const r=db.transaction(DOCS).objectStore(DOCS).get(uuid);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error)});
}
export async function listPairs(){
  const db=await openDataset();
  return new Promise((resolve,reject)=>{
    const req=db.transaction(PAIRS).objectStore(PAIRS).getAll();
    req.onsuccess=()=>resolve(req.result.sort((a,b)=>(Number(a.pageNumber)||0)-(Number(b.pageNumber)||0)));
    req.onerror=()=>reject(req.error);
  });
}