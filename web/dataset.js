const DB_NAME = "tiny-imgai";
const DB_VERSION = 4;
const PAIRS = "pairs";
const DOCS = "documents";
const META = "meta";
const PAGE_CACHE = "pageCache";

let dbPromise = null;

export function openDataset() {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(PAIRS)) db.createObjectStore(PAIRS, { keyPath: "uuid" });
      if (!db.objectStoreNames.contains(DOCS)) db.createObjectStore(DOCS, { keyPath: "uuid" });
      if (!db.objectStoreNames.contains(META)) db.createObjectStore(META, { keyPath: "key" });
      if (!db.objectStoreNames.contains(PAGE_CACHE)) db.createObjectStore(PAGE_CACHE, { keyPath: "uuid" });
    };

    req.onsuccess = () => {
      const db = req.result;
      db.onversionchange = () => {
        db.close();
        dbPromise = null;
      };
      resolve(db);
    };

    req.onerror = () => {
      dbPromise = null;
      reject(req.error);
    };
  });

  return dbPromise;
}

export async function addPdfDataset({ originalFile, processedFile, originalPages, processedPages }) {
  if (originalPages !== processedPages) {
    throw new Error("The PDFs have different page counts. Pairing requires matching page counts.");
  }

  const existing = await listPairs();
  const nextStart = existing.reduce(
    (m, p) => Math.max(m, Number(p.pageNumber) || 0),
    0
  ) + 1;

  const db = await openDataset();
  const originalDocId = crypto.randomUUID();
  const processedDocId = crypto.randomUUID();
  const now = new Date().toISOString();
  const pairs = [];

  return new Promise((resolve, reject) => {
    const tx = db.transaction([PAIRS, DOCS, META], "readwrite");
    const pairsStore = tx.objectStore(PAIRS);
    const docs = tx.objectStore(DOCS);
    const meta = tx.objectStore(META);

    docs.put({
      uuid: originalDocId,
      role: "original",
      name: originalFile.name,
      blob: originalFile,
      pageCount: originalPages,
      createdAt: now,
    });

    docs.put({
      uuid: processedDocId,
      role: "processed",
      name: processedFile.name,
      blob: processedFile,
      pageCount: processedPages,
      createdAt: now,
    });

    for (let page = 1; page <= originalPages; page++) {
      pairs.push({
        uuid: crypto.randomUUID(),
        pageNumber: nextStart + page - 1,
        originalDocId,
        processedDocId,
        originalPage: page,
        processedPage: page,
        createdAt: now,
      });
      pairsStore.put(pairs[pairs.length - 1]);
    }

    meta.put({
      key: "pairCounter",
      value: nextStart + originalPages - 1,
    });

    tx.oncomplete = () => resolve({
      originalDocId,
      processedDocId,
      pairs,
    });
    tx.onerror = () => reject(tx.error);
    tx.onabort = () =>
      reject(tx.error || new Error("Could not save PDF dataset."));
  });
}

export async function getDocument(uuid) {
  const db = await openDataset();

  return new Promise((resolve, reject) => {
    const req = db.transaction(DOCS).objectStore(DOCS).get(uuid);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function listPairs() {
  const db = await openDataset();

  return new Promise((resolve, reject) => {
    const req = db.transaction(PAIRS).objectStore(PAIRS).getAll();
    req.onsuccess = () =>
      resolve(
        req.result.sort(
          (a, b) =>
            (Number(a.pageNumber) || 0) - (Number(b.pageNumber) || 0)
        )
      );
    req.onerror = () => reject(req.error);
  });
}

export async function getCachedPage(uuid) {
  const db = await openDataset();

  return new Promise((resolve, reject) => {
    const req = db.transaction(PAGE_CACHE).objectStore(PAGE_CACHE).get(uuid);
    req.onsuccess = () => resolve(req.result || null);
    req.onerror = () => reject(req.error);
  });
}

export async function cachePage(uuid, { size, original, target }) {
  const db = await openDataset();
  const record = {
    uuid,
    size,
    original,
    target,
    bytes: original.byteLength + target.byteLength,
    cachedAt: new Date().toISOString(),
  };

  return new Promise((resolve, reject) => {
    const tx = db.transaction(PAGE_CACHE, "readwrite");
    tx.objectStore(PAGE_CACHE).put(record);
    tx.oncomplete = () => resolve(record);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error("Could not cache page."));
  });
}

export async function clearPageCache() {
  const db = await openDataset();

  return new Promise((resolve, reject) => {
    const tx = db.transaction(PAGE_CACHE, "readwrite");
    tx.objectStore(PAGE_CACHE).clear();
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error || new Error("Could not clear page cache."));
  });
}
