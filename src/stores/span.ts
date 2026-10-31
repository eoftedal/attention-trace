import { defineStore } from 'pinia';
import { computed, ref, shallowRef, watch } from 'vue';
import { modelLabel } from '../lib/config';
import { decode } from './model';
import { useChatStore, type AttnRun } from './chat';
import { useTraceStore } from './trace';

// Attention from a reply to a marked span, per layer and head: how much of
// what the reply reads comes from, say, an instruction injected into a tool
// result, compared with the rest of that message. This is the per-head
// "probe attention" of attention studies of prompt injection, for one
// conversation. A run can be pinned, and compared with the next one (the same
// instruction placed in the data or in the instructions, before and after an
// edit, …).

export type SpanMetric = 'span' | 'rest' | 'ratio';

/** Per layer: mean attention share of each head, over the reply's queries. */
export interface SpanLayer {
  layer: number;
  span: Float32Array; // [H] share of attention on the span
  rest: Float32Array; // [H] share on the rest of the span's message
}

/** Reply token → span token links, averaged over the heads in view, for drawing. */
export interface SpanFlow {
  links: { from: number; to: number; w: number }[]; // from: the reply token produced, to: a span token
  received: Map<number, number>;  // span token → attention it gets, mean over the reply tokens
  given: Map<number, number>;     // reply token → its attention on the whole span
  heads: string;                  // which heads, as a phrase: "averaged over all 448 heads"
}

export interface SpanResult {
  model: string;
  label: string;      // the span's text
  role: string;       // the span's message
  spanLen: number;
  restLen: number;
  queries: number;
  layers: SpanLayer[];
}

// Attention per token of the span over attention per token of the rest, as log2.
const EPS = 1e-6;
export function densityRatio(r: SpanResult, li: number, h: number): number {
  const L = r.layers[li];
  if (!r.restLen) return 0;
  return Math.log2((L.span[h] / r.spanLen + EPS) / (L.rest[h] / r.restLen + EPS));
}

export function metricAt(r: SpanResult, metric: SpanMetric, li: number, h: number): number {
  if (metric === 'span') return r.layers[li].span[h];
  if (metric === 'rest') return r.layers[li].rest[h];
  return densityRatio(r, li, h);
}

/** Mean over every head of every layer. */
export function overall(r: SpanResult, metric: SpanMetric): number {
  let sum = 0, n = 0;
  r.layers.forEach((L, li) => {
    for (let h = 0; h < L.span.length; h++) { sum += metricAt(r, metric, li, h); n++; }
  });
  return n ? sum / n : 0;
}

export const useSpanStore = defineStore('span', () => {
  const chat = useChatStore();
  const trace = useTraceStore();

  const start = ref<number | null>(null); // inclusive token positions
  const end = ref<number | null>(null);
  let anchor: number | null = null;       // first end of a shift-click pair
  const replyTokens = ref(32);            // the reply's first N tokens are the queries
  const metric = ref<SpanMetric>('span');
  const pinned = shallowRef<SpanResult | null>(null);
  // The conversation shows lines to the span (true) or the clicked token's trace.
  const showLines = ref(true);

  // Positions change with the conversation; a new run starts without a span.
  watch(() => chat.run, () => { start.value = end.value = null; anchor = null; });

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
  }

  // The reply whose attention is measured: the last assistant message.
  const reply = computed(() => {
    const r = chat.run;
    if (!r) return null;
    for (let i = r.segs.length - 1; i >= 0; i--) if (r.segs[i].role === 'assistant') return r.segs[i];
    return null;
  });

  // Query positions: the attention that produced each of the reply's first N
  // tokens of text is read one position earlier.
  function queries(r: AttnRun, after: number): number[] {
    const seg = reply.value;
    if (!seg) return [];
    const out: number[] = [];
    for (let p = seg.start; p < seg.end && out.length < replyTokens.value; p++) {
      if (r.content[p] && !trace.special[p] && p - 1 > after) out.push(p - 1);
    }
    return out;
  }

  // The span, the rest of its message, and the queries measured.
  const measured = computed(() => {
    const r = chat.run;
    if (!r || start.value == null || end.value == null || end.value >= r.S) return null;
    const s0 = start.value, s1 = end.value;
    const seg = r.segs[r.segOf[s0]];
    // the rest of the span's message: its text outside the span
    const inRest = (p: number) => p >= seg.start && p < seg.end && (p < s0 || p > s1) && !!r.content[p] && !trace.special[p];
    let restLen = 0;
    for (let p = seg.start; p < seg.end; p++) if (inRest(p)) restLen++;
    const qs = queries(r, Math.max(s1, seg.end - 1));
    return qs.length ? { r, s0, s1, seg, inRest, restLen, qs } : null;
  });

  const result = computed<SpanResult | null>(() => {
    const m = measured.value;
    if (!m) return null;
    const { r, s0, s1, seg, inRest, restLen, qs } = m;
    const layers = r.layers.map(({ layer, H, K, vals, idx }) => {
      const span = new Float32Array(H), rest = new Float32Array(H);
      for (const q of qs) {
        for (let h = 0; h < H; h++) {
          const base = (q * H + h) * K;
          for (let k = 0; k < K; k++) {
            const w = vals[base + k];
            if (w < 1e-4) break;
            const key = idx[base + k];
            if (key >= s0 && key <= s1) span[h] += w;
            else if (inRest(key)) rest[h] += w;
          }
        }
      }
      for (let h = 0; h < H; h++) { span[h] /= qs.length; rest[h] /= qs.length; }
      return { layer, span, rest };
    });
    const text = r.ids.slice(s0, s1 + 1).map(decode).join('').replace(/\s+/g, ' ').trim();
    return {
      model: r.model,
      label: text.length > 80 ? `${text.slice(0, 77)}…` : text,
      role: seg.role === 'tool' ? 'tool result' : seg.role,
      spanLen: s1 - s0 + 1,
      restLen,
      queries: qs.length,
      layers,
    };
  });

  // Links for the conversation view, over the heads the trace has in view:
  // one layer's visible heads when a layer is selected (clicking a head in a
  // grid selects it alone), otherwise every head of every layer.
  const flow = computed<SpanFlow | null>(() => {
    const m = measured.value;
    if (!m) return null;
    const { r, s0, s1, qs } = m;
    const n = s1 - s0 + 1;
    const acc = new Float64Array(qs.length * n);
    let heads = 0;
    r.layers.forEach(({ H, K, vals, idx }, li) => {
      if (trace.layerSel >= 0 && li !== trace.layerSel) return;
      for (let h = 0; h < H; h++) {
        if (trace.layerSel >= 0 && trace.hidden.has(h)) continue;
        heads++;
        qs.forEach((q, qi) => {
          const base = (q * H + h) * K;
          for (let k = 0; k < K; k++) {
            const w = vals[base + k];
            if (w < 1e-4) break;
            const key = idx[base + k];
            if (key >= s0 && key <= s1) acc[qi * n + key - s0] += w;
          }
        });
      }
    });
    if (!heads) return null;
    const links: SpanFlow['links'] = [];
    const received = new Map<number, number>(), given = new Map<number, number>();
    qs.forEach((q, qi) => {
      let total = 0;
      for (let j = 0; j < n; j++) {
        const w = acc[qi * n + j] / heads;
        if (w <= 0) continue;
        total += w;
        links.push({ from: q + 1, to: s0 + j, w });
        received.set(s0 + j, (received.get(s0 + j) ?? 0) + w / qs.length);
      }
      given.set(q + 1, total);
    });
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
    const r = chat.run;
    if (!r || start.value == null || end.value == null) return '';
    if (!reply.value) return 'The conversation has no reply to measure.';
    if (r.segOf[start.value] !== r.segOf[end.value]) return 'Mark a span within one message.';
    if (end.value >= reply.value.start) return 'Mark a span before the last reply: its attention to the span is what is measured.';
    return result.value ? '' : 'The reply has no tokens after the span.';
  });

  // A pinned result can be compared with the current one when the heads match.
  const comparable = computed(() => {
    const a = pinned.value, b = result.value;
    return !!a && !!b && a.model === b.model && a.layers.length === b.layers.length
      && a.layers.every((L, i) => L.span.length === b.layers[i].span.length);
  });

  function pin() {
    pinned.value = result.value;
  }

  function unpin() {
    pinned.value = null;
  }

  const pinnedLabel = computed(() => (pinned.value ? `${modelLabel(pinned.value.model)}: “${pinned.value.label}” (${pinned.value.role})` : ''));

  return {
    start, end, replyTokens, metric, pinned, showLines, result, flow, problem, comparable, pinnedLabel,
    mark, shiftClick, clear, pin, unpin,
  };
});
