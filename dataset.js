const DB_NAME="tiny-imgai";
const DB_VERSION=2;
const STORE="pairs";
const META="meta";

export function openDataset(){
  return new Promise((resolve,reject)=>{
    const req=indexedDB.open(DB_NAME,DB_VERSION);
    req.onupgradeneeded=()=>{
      const db=req.result;
      if(!db.objectStoreNames.contains(STORE))db.createObjectStore(STORE,{keyPath:"uuid"});
      if(!db.objectStoreNames.contains(META))db.createObjectStore(META,{keyPath:"key"});
    };
    req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);
  });
}
export async function addPair({original,target,name="Untitled"}){
  const db=await openDataset();
  const pairs=await listPairs();
  const pageNumber=pairs.reduce((m,p)=>Math.max(m,Number(p.pageNumber)||0),0)+1;
  const pair={uuid:crypto.randomUUID(),pageNumber,name,original,target,createdAt:new Date().toISOString()};
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE,"readwrite");tx.objectStore(STORE).add(pair);
    tx.oncomplete=()=>resolve(pair);tx.onerror=()=>reject(tx.error);
  });
}
export async function listPairs(){
  const db=await openDataset();
  return new Promise((resolve,reject)=>{
    const req=db.transaction(STORE).objectStore(STORE).getAll();
    req.onsuccess=()=>resolve(req.result.sort((a,b)=>(Number(a.pageNumber)||0)-(Number(b.pageNumber)||0)));
    req.onerror=()=>reject(req.error);
  });
}