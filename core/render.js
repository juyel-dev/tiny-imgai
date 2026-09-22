// Shared PDF page -> fixed-size square canvas renderer.
// Platform-agnostic: pass a canvasFactory(width, height) that returns
// anything with a 2D-context-compatible getContext("2d").
export async function renderPdfPageToCanvas(pdf, pageNumber, size, canvasFactory) {
  const page = await pdf.getPage(pageNumber);
  try {
    const base = page.getViewport({ scale: 1 });
    const scale = Math.min(size / base.width, size / base.height);
    const viewport = page.getViewport({ scale });

    const canvas = canvasFactory(size, size);
    try {
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, size, size);

      const x = (size - viewport.width) / 2;
      const y = (size - viewport.height) / 2;

      await page.render({
        canvasContext: ctx,
        viewport,
        transform: [1, 0, 0, 1, x, y],
      }).promise;

      return canvas;
    } catch (error) {
      if (canvas && typeof canvas.width === "number") {
        canvas.width = 0;
        canvas.height = 0;
      }
      throw error;
    } finally {
      if (typeof page.cleanup === "function") page.cleanup();
    }
  } catch (error) {
    throw error;
  }
}
