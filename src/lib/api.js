function joinUrl(base, path) {
  if (!base) return path;
  return `${String(base).replace(/\/$/, '')}${path.startsWith('/') ? '' : '/'}${path}`;
}

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export async function apiGet(path) {
  const res = await fetch(joinUrl(API_BASE_URL, path), { credentials: 'include' });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}

export async function apiPostJson(path, payload) {
  const res = await fetch(joinUrl(API_BASE_URL, path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    credentials: 'include'
  });
  const data = await res.json().catch(() => null);
  if (!res.ok || (data && data.success === false)) {
    throw new Error((data && data.error) || `Request failed (${res.status})`);
  }
  return data;
}

export async function apiPostForm(path, formData) {
  const res = await fetch(joinUrl(API_BASE_URL, path), {
    method: 'POST',
    body: formData,
    credentials: 'include'
  });
  const data = await res.json().catch(() => null);
  if (!res.ok || (data && data.success === false)) {
    throw new Error((data && data.error) || `Request failed (${res.status})`);
  }
  return data;
}
