// URL hash: #model=…&chat=<base64url JSON of the conversation>
export interface HashState {
  model?: string;
  chat?: string;
}

export function readHash(): HashState {
  let params: URLSearchParams;
  try {
    params = new URLSearchParams(location.hash.slice(1));
  } catch {
    params = new URLSearchParams();
  }
  const get = (k: string) => params.get(k) ?? undefined;
  return { model: get('model'), chat: get('chat') };
}

export function writeHash(s: HashState) {
  const params = new URLSearchParams();
  if (s.model) params.set('model', s.model);
  if (s.chat) params.set('chat', s.chat);
  window.history.replaceState(null, '', '#' + params.toString());
}

// UTF-8 safe base64url, so the conversation survives a reload.
export function encodeJson(v: unknown): string {
  const bytes = new TextEncoder().encode(JSON.stringify(v));
  let bin = '';
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

export function decodeJson<T>(s: string): T | null {
  try {
    const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(new TextDecoder().decode(Uint8Array.from(bin, c => c.charCodeAt(0)))) as T;
  } catch {
    return null;
  }
}
