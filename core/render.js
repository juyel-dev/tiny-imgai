// Shared PDF page -> fixed-size square canvas renderer.
// Platform-agnostic: pass a canvasFactory(width, height) that returns
// anything with a 2D-context-compatible getContext('2d').
//   - browser: (w, h) => { const c = document.createElement('canvas'); c.width=w; c.height=h; return c; }
//   - Node:    (w, h) => require('canvas').createCanvas(w, h)
export async function renderPdfPageToCanvas(pdf, pageNumber, size, canvasFactory) {
  const page = await pdf.getPage(pageNumber);
  const base = page.getViewport({ scale: 1 });
  const scale = Math.min(size / base.width, size / base.height);
  const viewport = page.getViewport({ scale });
  const canvas = canvasFactory(size, size);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, size, size);
  const x = (size - viewport.width) / 2;
  const y = (size - viewport.height) / 2;
  await page.render({ canvasContext: ctx, viewport, transform: [1, 0, 0, 1, x, y] }).promise;
  return canvas;
}
