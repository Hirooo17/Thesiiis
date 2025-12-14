export function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

export function formatPct01(x, digits = 1) {
  if (typeof x !== 'number' || Number.isNaN(x)) return '--';
  return `${(x * 100).toFixed(digits)}%`;
}

export function toDataImage(base64) {
  if (!base64) return null;
  return `data:image/png;base64,${base64}`;
}
