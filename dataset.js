const DB_NAME = "tiny-imgai";
const DB_VERSION = 1;
const STORE = "pairs";

export function openDataset() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "uuid" });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function addPair({ original, target, name = "Untitled" }) {
  const db = await openDataset();
  const pair = { uuid: crypto.randomUUID(), pageNumber: Date.now(), name, original, target, createdAt: new Date().toISOString() };
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).add(pair);
    tx.oncomplete = () => resolve(pair);
    tx.onerror = () => reject(tx.error);
  });
}

export async function listPairs() {
  const db = await openDataset();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE).objectStore(STORE).getAll();
    req.onsuccess = () => resolve(req.result.sort((a,b) => a.createdAt.localeCompare(b.createdAt)));
    req.onerror = () => reject(req.error);
  });
}
