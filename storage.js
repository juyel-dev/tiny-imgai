const KEY = "tiny-imgai-state";
export function loadState() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } }
export function saveState(state) { localStorage.setItem(KEY, JSON.stringify(state)); }
