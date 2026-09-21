self.onmessage = async ({ data }) => {
  if (data.type !== "train") return;
  const epochs = data.epochs || 20;
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const loss = 0.92 * Math.exp(-epoch / 7) + 0.06;
    self.postMessage({ type: "progress", epoch, epochs, loss, step: epoch });
    await new Promise(r => setTimeout(r, 80));
  }
  self.postMessage({ type: "done", version: data.version || 1 });
};
