// Visible representation of whitespace / control chars
export function display(s: string): string {
  return s.replace(/\r\n|\n/g, '↵').replace(/ /g, '·').replace(/\t/g, '→');
}

export function fmtPct(p: number): string {
  const pct = p * 100;
  if (pct >= 10) return pct.toFixed(0) + '%';
  if (pct >= 0.1) return pct.toFixed(1) + '%';
  return '<0.1%';
}

export const isSpecialToken = (s: string) => /^<[^<>\s]+>$/.test(s.trim());

// Little-endian IEEE half float → number.
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/** A float16 tensor's data as float32: transformers.js hands it over as raw
 *  Uint16 bits or, where the browser has it, as a Float16Array. */
export function toFloat32(data: ArrayLike<number | bigint>): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  if (data instanceof Uint16Array) {
    for (let i = 0; i < data.length; i++) out[i] = halfToFloat(data[i]);
  } else {
    for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
  }
  return out;
}

// Bring an element into view inside the page, centered vertically.
export function scrollIntoViewCentered(el: Element | null | undefined) {
  el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
}
