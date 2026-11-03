import { defineStore } from 'pinia';
import { computed, ref, shallowRef, watch } from 'vue';
import { modelLabel } from '../lib/config';
import { decode } from './model';
import { useChatStore, type AttnRun, type Segment } from './chat';
import { useTraceStore } from './trace';

// Where a marked stretch of text reads from: the attention of its tokens,
// summed by earlier message, per layer and head, and drawn as lines back to
// the tokens it reads. Mark a sentence of the reply to see whether it came
// from the user's request, the system prompt or an instruction injected into
// a tool result. A result can be pinned and compared with the next run (the
// same instruction placed elsewhere, reworded, …).
//
// The tokens read are filtered as in the trace (other messages only, no first
// token or template tokens, by default). Shares are raw attention, so they add
// up; the lines use the trace's weights (above each key's usual share, by
// default), so resting spots don't drown them.

/** Per layer: each head's share of the span's attention on the target message. */
export interface SpanLayer {
  layer: number;
  share: Float32Array; // [H]
}

export interface SpanResult {
  model: string;
  label: string;       // the span's text
  role: string;        // the span's message
  target: string;      // the message the grid measures
  queries: number;
  layers: SpanLayer[];
}

/** Span token → earlier token links, averaged over the heads in view, for drawing. */
export interface SpanFlow {
  links: { from: number; to: number; w: number }[]; // from: a span token, to: a token it reads
  received: Map<number, number>; // token read → attention from the span, mean over its tokens
  given: Map<number, number>;    // span token → attention it puts on what the filters allow
  heads: string;                 // which heads, as a phrase: "averaged over all 448 heads"
}

/** Share of the span's attention per earlier message. */
export interface MessageShare {
  seg: number;  // index into run.segs
  label: string;
  role: Segment['role'];
  share: number;    // of the span's attention, averaged over its tokens and every head
  tokens: number;   // tokens of the message the filters let the span read
  perToken: number; // share / tokens
}

const ROLE_LABEL: Record<Segment['role'], string> = {
  system: 'System', user: 'User', assistant: 'Assistant', tool: 'Tool result', prompt: 'Generation prompt',
};

/** Message labels as the conversation view numbers them (User 1, Assistant 2, …). */
export function segLabels(r: AttnRun): string[] {
  const n: Record<string, number> = {};
  return r.segs.map(s => (s.role === 'system' || s.role === 'prompt'
    ? ROLE_LABEL[s.role] : `${ROLE_LABEL[s.role]} ${n[s.role] = (n[s.role] ?? 0) + 1}`));
}

/** Mean over every head of every layer. */
export function overall(r: SpanResult): number {
  let sum = 0, n = 0;
  for (const L of r.layers) for (const v of L.share) { sum += v; n++; }
  return n ? sum / n : 0;
}

export const useSpanStore = defineStore('span', () => {
  const chat = useChatStore();
  const trace = useTraceStore();

  const start = ref<number | null>(null); // inclusive token positions
  const end = ref<number | null>(null);
  let anchor: number | null = null;       // first end of a shift-click pair
  const target = ref<number | null>(null); // segment the head grid measures; null: the most-read one
  const pinned = shallowRef<SpanResult | null>(null);
  // The conversation shows lines from the span (true) or the clicked token's trace.
  const showLines = ref(true);

  // Positions change with the conversation; a new run starts without a span.
  watch(() => chat.run, () => { start.value = end.value = null; anchor = null; target.value = null; });

  // Leading and trailing whitespace and template tokens are left out (a drag
  // to the end of a line picks up its newline).
  function mark(a: number, b: number) {
    const r = chat.run;
    if (!r) return;
    let s0 = Math.min(a, b), s1 = Math.max(a, b);
    const blank = (p: number) => !!trace.special[p] || !decode(r.ids[p]).trim();
    while (s0 < s1 && blank(s0)) s0++;
    while (s1 > s0 && blank(s1)) s1--;
    start.value = s0;
    end.value = s1;
    target.value = null;
    showLines.value = true;
  }

  /** Shift-click: the first click starts a span, the second ends it. */
  function shiftClick(p: number) {
    if (anchor == null) {
      anchor = p;
      mark(p, p);
    } else {
      mark(anchor, p);
      anchor = null;
    }
  }

  function clear() {
    start.value = end.value = null;
    anchor = null;
    target.value = null;
  }

  // The span's tokens of text, and where their attention is read: one
  // position earlier when tracing what produced a token, as in the trace.
  const measured = computed(() => {
    const r = chat.run;
    if (!r || start.value == null || end.value == null || end.value >= r.S) return null;
    const s0 = start.value, s1 = end.value;
    const toks: number[] = [];
    for (let p = s0; p <= s1; p++) if (r.content[p] && !trace.special[p]) toks.push(p);
    const pairs = toks.map(p => ({ tok: p, q: trace.produced ? p - 1 : p })).filter(x => x.q >= 0);
    // what the span reads: allowed by the trace's filters, and outside the span itself
    const reads = (q: number, key: number) => (key < s0 || key > s1) && trace.allowed(q, key);
    return pairs.length ? { r, s0, s1, pairs, reads } : null;
  });

  // Share of the span's attention per message, every head weighted equally,
  // and how much of the span's attention the recorded keys hold at all: only
  // the K strongest keys of each head are kept, so the shares are low by
  // whatever fell below them.
  const shares = computed(() => {
    const m = measured.value;
    if (!m) return null;
    const { r, pairs, reads } = m;
    const share = new Float64Array(r.segs.length);
    let heads = 0, kept = 0;
    for (const { H, K, vals, idx } of r.layers) {
      heads += H;
      for (const { q } of pairs) {
        for (let h = 0; h < H; h++) {
          const base = (q * H + h) * K;
          for (let k = 0; k < K; k++) {
            const w = vals[base + k];
            if (w < 1e-4) break;
            kept += w;
            const key = idx[base + k];
            if (reads(q, key)) share[r.segOf[key]] += w;
          }
        }
      }
    }
    const n = heads * pairs.length;
    const labels = segLabels(r);
    const messages: MessageShare[] = [];
    share.forEach((v, si) => {
      if (v <= 0) return;
      const s = r.segs[si];
      let tokens = 0;
      for (let p = s.start; p < s.end; p++) if (pairs.some(({ q }) => reads(q, p))) tokens++;
      messages.push({ seg: si, label: labels[si], role: s.role, share: v / n, tokens, perToken: tokens ? v / n / tokens : 0 });
    });
    return { messages, captured: kept / n, k: r.layers[0]?.K ?? 0 };
  });
  const messages = computed<MessageShare[]>(() => shares.value?.messages ?? []);
  /** How much of the span's attention the recorded top-K keys hold, and K. */
  const mass = computed(() => ({ captured: shares.value?.captured ?? 0, k: shares.value?.k ?? 0 }));

  const targetSeg = computed(() => {
    if (target.value != null && messages.value.some(x => x.seg === target.value)) return target.value;
    return [...messages.value].sort((a, b) => b.share - a.share)[0]?.seg ?? null;
  });

  const result = computed<SpanResult | null>(() => {
    const m = measured.value;
    const t = targetSeg.value;
    if (!m || t == null) return null;
    const { r, s0, s1, pairs, reads } = m;
    const tSeg = r.segs[t];
    const layers = r.layers.map(({ layer, H, K, vals, idx }) => {
      const share = new Float32Array(H);
      for (const { q } of pairs) {
        for (let h = 0; h < H; h++) {
          const base = (q * H + h) * K;
          for (let k = 0; k < K; k++) {
            const w = vals[base + k];
            if (w < 1e-4) break;
            const key = idx[base + k];
            if (key >= tSeg.start && key < tSeg.end && reads(q, key)) share[h] += w;
          }
        }
      }
      for (let h = 0; h < H; h++) share[h] /= pairs.length;
      return { layer, share };
    });
    const text = r.ids.slice(s0, s1 + 1).map(decode).join('').replace(/\s+/g, ' ').trim();
    const labels = segLabels(r);
    return {
      model: r.model,
      label: text.length > 80 ? `${text.slice(0, 77)}…` : text,
      role: labels[r.segOf[s0]],
      target: labels[t],
      queries: pairs.length,
      layers,
    };
  });

  // Links for the conversation view, over the heads the trace has in view:
  // one layer's visible heads when a layer is selected (clicking a head in a
  // grid selects it alone), otherwise every head of every layer.
  const flow = computed<SpanFlow | null>(() => {
    const m = measured.value;
    if (!m) return null;
    const { r, pairs, reads } = m;
    const acc = new Map<number, number>(); // pair index * S + key → summed weight
    let heads = 0;
    r.layers.forEach(({ H, K, vals, idx }, li) => {
      if (trace.layerSel >= 0 && li !== trace.layerSel) return;
      for (let h = 0; h < H; h++) {
        if (trace.layerSel >= 0 && trace.hidden.has(h)) continue;
        heads++;
        pairs.forEach(({ q }, pi) => {
          const base = (q * H + h) * K;
          for (let k = 0; k < K; k++) {
            if (vals[base + k] < 1e-4) break;
            const key = idx[base + k];
            if (!reads(q, key)) continue;
            const w = trace.weight(li, h, q, key, vals[base + k]);
            if (w > 0) acc.set(pi * r.S + key, (acc.get(pi * r.S + key) ?? 0) + w);
          }
        });
      }
    });
    if (!heads) return null;
    const links: SpanFlow['links'] = [];
    const received = new Map<number, number>(), given = new Map<number, number>();
    for (const [code, sum] of acc) {
      const pi = Math.floor(code / r.S), key = code % r.S, w = sum / heads;
      const from = pairs[pi].tok;
      links.push({ from, to: key, w });
      received.set(key, (received.get(key) ?? 0) + w / pairs.length);
      given.set(from, (given.get(from) ?? 0) + w);
    }
    const L = trace.layerSel >= 0 ? r.layers[trace.layerSel] : null;
    const shown = L ? L.H - [...trace.hidden].filter(h => h < L.H).length : 0;
    return {
      links, received, given,
      heads: !L ? `averaged over all ${heads} heads`
        : shown === 1 ? `for head L${L.layer} H${[...Array(L.H).keys()].find(h => !trace.hidden.has(h))}`
          : `averaged over the ${shown} heads of L${L.layer} shown`,
    };
  });

  // Why there is no result, for the panel.
  const problem = computed(() => {
    if (start.value == null) return '';
    if (!measured.value) return 'The span has no text tokens.';
    return messages.value.length ? '' : 'The span reads nothing the filters allow. Widen "Look at" in the trace settings.';
  });

  // A pinned result can be compared with the current one when the heads match.
  const comparable = computed(() => {
    const a = pinned.value, b = result.value;
    return !!a && !!b && a.model === b.model && a.layers.length === b.layers.length
      && a.layers.every((L, i) => L.share.length === b.layers[i].share.length);
  });

  function pin() {
    pinned.value = result.value;
  }

  function unpin() {
    pinned.value = null;
  }

  const describe = (x: SpanResult) => `“${x.label}” (${x.role}) reading ${x.target}`;
  const pinnedLabel = computed(() => (pinned.value ? `${modelLabel(pinned.value.model)}: ${describe(pinned.value)}` : ''));
  const currentLabel = computed(() => (result.value ? describe(result.value) : ''));

  return {
    start, end, target, targetSeg, pinned, showLines, messages, mass, result, flow, problem, comparable, pinnedLabel, currentLabel,
    mark, shiftClick, clear, pin, unpin,
  };
});
