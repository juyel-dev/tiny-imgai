// Scans train/data/originals/*.pdf and train/data/processed/*.pdf,
// matches pairs by filename (minus extension), and writes
// train/data/manifest.json — the single file train.js reads.
//
// Expected layout:
//   train/data/originals/<id>.pdf
//   train/data/processed/<id>.pdf
// <id> can be anything (chapter-01, page-0423, whatever) as long as
// the same id exists on both sides. Mismatches are reported, not
// silently skipped, so a bad batch of scans doesn't corrupt training.
//
// Usage: node build-manifest.js  (from inside /train)
//        DATA_DIR=/path/to/data node build-manifest.js

import { readdirSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const DATA_DIR = process.env.DATA_DIR || join(process.cwd(), "data");
const ORIGINALS_DIR = join(DATA_DIR, "originals");
const PROCESSED_DIR = join(DATA_DIR, "processed");

function pdfIds(dir) {
  if (!existsSync(dir)) return new Set();
  return new Set(
    readdirSync(dir)
      .filter((f) => f.toLowerCase().endsWith(".pdf"))
      .map((f) => f.replace(/\.pdf$/i, ""))
  );
}

const origIds = pdfIds(ORIGINALS_DIR);
const procIds = pdfIds(PROCESSED_DIR);

if (!origIds.size && !procIds.size) {
  console.error(`No PDFs found. Expected:\n  ${ORIGINALS_DIR}/*.pdf\n  ${PROCESSED_DIR}/*.pdf`);
  process.exit(1);
}

const matched = [...origIds].filter((id) => procIds.has(id)).sort();
const missingProcessed = [...origIds].filter((id) => !procIds.has(id)).sort();
const missingOriginal = [...procIds].filter((id) => !origIds.has(id)).sort();

const manifest = {
  generatedAt: new Date().toISOString(),
  pairCount: matched.length,
  pairs: matched.map((id) => ({
    id,
    originalPdf: `originals/${id}.pdf`,
    processedPdf: `processed/${id}.pdf`,
  })),
};

const outPath = join(DATA_DIR, "manifest.json");
writeFileSync(outPath, JSON.stringify(manifest, null, 2));

console.log(`Matched pairs: ${matched.length}`);
if (missingProcessed.length) {
  console.log(`Missing processed PDF for ${missingProcessed.length} id(s): ${missingProcessed.slice(0, 10).join(", ")}${missingProcessed.length > 10 ? " ..." : ""}`);
}
if (missingOriginal.length) {
  console.log(`Missing original PDF for ${missingOriginal.length} id(s): ${missingOriginal.slice(0, 10).join(", ")}${missingOriginal.length > 10 ? " ..." : ""}`);
}
console.log(`Wrote ${outPath}`);
