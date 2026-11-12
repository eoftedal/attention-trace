import { defineStore } from 'pinia';
import { computed, ref, watch } from 'vue';
import { isSpecialToken } from '../lib/format';
import { decode } from './model';
import { useChatStore, type AttnRun } from './chat';

// Which keys count when tracing back from a token:
//  messages: only tokens in other messages (what travels between turns)
//  far:      only tokens at least minDist positions back
//  all:      every earlier token
export type Reach = 'messages' | 'far' | 'all';
export type Combine = 'max' | 'mean';

/** One attention link: query `from` attends to key `to`. */
export interface Arc {
  from: number;
  to: number;
  series: number; // head index (one layer shown) or layer index (all layers)
  w: number;      // attention weight of this link
  path: number;   // product of the weights along the trace up to `to`
  hop: number;    // 1 = directly from the selected token
}

/** A token reached by the trace. */
export interface Hit {
  pos: number;
  path: number;   // strongest path weight reaching it
  hop: number;    // earliest hop reaching it
  w: number;      // strongest link weight on that hop
  series: number; // series of that link
  links: { series: number; w: number; hop: number; from: number }[];
}

export interface Trace {
  arcs: Arc[];
  hits: Map<number, Hit>;
}

/** How opaque each hop is drawn: later hops fade. */
export function hopFade(hop: number): number {
  return [1, 0.55, 0.32, 0.2][hop - 1] ?? 0.15;
}

/** Golden-angle hues: neighbouring series get clearly different colors. */
export function seriesColor(i: number): string {
  return `oklch(0.66 0.19 ${(i * 137.508 + 20) % 360})`;
}

export const useTraceStore = defineStore('trace', () => {
  const chat = useChatStore();

  const selPos = ref<number | null>(null);
  const layerSel = ref(-1); // index into run.layers; -1: all layers
  const combine = ref<Combine>('max');
  const reach = ref<Reach>('messages');
  const minDist = ref(16);
  const hops = ref(3);
  const branch = ref(3);     // keys followed from each source on the next hop
  const arcsPerSource = ref(12); // from the selected token; later hops draw `branch` each
  const minWeight = ref(0.03);
  const skipFirst = ref(true);   // the first token soaks up attention ("attention sink")
  const skipSpecial = ref(true); // template tokens: <|im_start|>, role names, newlines between messages
  // Heads with nothing to do park their attention on a few early tokens (the
  // first token, newlines and punctuation of the system prompt). Subtracting
  // what a key gets on average from the other messages leaves the links that
  // are specific to the query.
  const discount = ref(true);
  const hidden = ref<Set<number>>(new Set()); // series switched off
  const mapScope = ref<'token' | 'message'>('token');
  // The attention at a token is what the model reads while predicting the
  // token after it. What produced the selected token is read one position
  // earlier, so by default that is the query.
  const produced = ref(true);

  const run = computed(() => chat.run);

  // A new run keeps the settings, but the selection and layer must fit it.
  watch(run, (r, old) => {
    if (!r) { selPos.value = null; return; }
    if (selPos.value != null && (selPos.value >= r.S || r.model !== old?.model)) selPos.value = null;
    if (layerSel.value >= r.layers.length || r.model !== old?.model) layerSel.value = -1;
    if (selPos.value == null) selPos.value = defaultSelection(r);
  }, { immediate: true });

  // The last token of the last message: the reply's final position.
  function defaultSelection(r: AttnRun): number {
    const last = r.segs[r.segs.length - 1];
    for (let p = last.end - 1; p >= last.start; p--) if (!isSpecialToken(decode(r.ids[p])) && decode(r.ids[p]).trim()) return p;
    return r.S - 1;
  }

  const queryPos = computed(() => {
    const p = selPos.value;
    if (p == null) return null;
    return produced.value ? Math.max(0, p - 1) : p;
  });

  const special = computed(() => {
    const r = run.value;
    if (!r) return new Uint8Array(0);
    return Uint8Array.from(r.ids, id => (isSpecialToken(decode(id)) ? 1 : 0));
  });

  // Tokens outside the message text: special tokens and the template's role names and newlines.
  const template = computed(() => {
    const r = run.value;
    if (!r) return new Uint8Array(0);
    return special.value.map((sp, p) => (sp || !r.content[p] ? 1 : 0));
  });

  // Per layer, [H, S]: the mean attention each key gets from the tokens of
  // later messages, leaving out the queries of one message. The message being
  // traced is left out, so that the reading under study does not count as the
  // key's usual share (a reply reading the password in every token would
  // otherwise make that reading "usual"). The key's own message is left out
  // too: its tokens read the key at short range (and the key reads itself),
  // which says nothing about what a later message usually puts there.
  // A mean over few queries is unreliable, so it is shrunk toward zero as if
  // FEW more queries had read nothing; a key no other message can see (in the
  // message right before the one traced) counts with its raw weight.
  // Computed per message as needed.
  const FEW = 8;
  const baselines = new Map<number, Float32Array[]>(); // message left out → per layer
  let baselinesOf: AttnRun | null = null;

  function baselineFor(seg: number): Float32Array[] {
    const r = run.value!;
    if (baselinesOf !== r) { baselines.clear(); baselinesOf = r; }
    let b = baselines.get(seg);
    if (b) return b;
    const s = r.segs[seg];
    b = r.layers.map(({ H, K, vals, idx }) => {
      const out = new Float32Array(H * r.S);
      for (let q = 0; q < r.S; q++) {
        if (q >= s.start && q < s.end) continue;
        const own = r.segOf[q];
        for (let h = 0; h < H; h++) {
          const base = (q * H + h) * K;
          for (let k = 0; k < K; k++) {
            const w = vals[base + k];
            if (w < 1e-4) break;
            const key = idx[base + k];
            if (r.segOf[key] !== own) out[h * r.S + key] += w;
          }
        }
      }
      for (let key = 0; key < r.S; key++) {
        // queries after the key's message, without those of the message left out
        const after = r.segs[r.segOf[key]].end;
        const n = r.S - after - Math.max(0, s.end - Math.max(s.start, after));
        const f = 1 / (n + FEW);
        for (let h = 0; h < H; h++) out[h * r.S + key] *= f;
      }
      return out;
    });
    baselines.set(seg, b);
    return b;
  }

  // The weight a link from query q counts with: raw, or above the key's usual
  // share from the messages other than q's.
  function weight(li: number, h: number, q: number, key: number, w: number): number {
    const r = run.value!;
    return discount.value ? w - baselineFor(r.segOf[q])[li][h * r.S + key] : w;
  }

  const seriesCount = computed(() => {
    const r = run.value;
    if (!r) return 0;
    return layerSel.value < 0 ? r.layers.length : r.layers[layerSel.value].H;
  });

  const seriesLabel = (i: number) => {
    const r = run.value;
    if (!r) return '';
    return layerSel.value < 0 ? `L${r.layers[i].layer}` : `H${i}`;
  };

  function allowed(q: number, key: number): boolean {
    const r = run.value!;
    if (key >= q || key < 0) return false;
    if (skipFirst.value && key === 0) return false;
    if (skipSpecial.value && template.value[key]) return false;
    if (reach.value === 'messages') return r.segOf[key] !== r.segOf[q];
    if (reach.value === 'far') return q - key >= minDist.value;
    return true;
  }

  // Links from query q that pass the filters, per series.
  function contributions(q: number): { key: number; series: number; w: number }[] {
    const r = run.value!;
    const out: { key: number; series: number; w: number }[] = [];
    const addLayer = (li: number, perHead: boolean) => {
      const { H, K, vals, idx } = r.layers[li];
      const acc = perHead ? null : new Map<number, number>();
      for (let h = 0; h < H; h++) {
        if (perHead && hidden.value.has(h)) continue;
        const base = (q * H + h) * K;
        for (let k = 0; k < K; k++) {
          if (vals[base + k] < 1e-4) break; // sorted descending
          const key = idx[base + k];
          if (!allowed(q, key)) continue;
          const w = weight(li, h, q, key, vals[base + k]);
          if (w <= 0) continue;
          if (acc) {
            const prev = acc.get(key) ?? 0;
            acc.set(key, combine.value === 'max' ? Math.max(prev, w) : prev + w / H);
          } else {
            out.push({ key, series: h, w });
          }
        }
      }
      if (acc) for (const [key, w] of acc) out.push({ key, series: li, w });
    };
    if (layerSel.value >= 0) addLayer(layerSel.value, true);
    else for (let li = 0; li < r.layers.length; li++) if (!hidden.value.has(li)) addLayer(li, false);
    return out;
  }

  // Later hops follow links in lower layers than the one that reached their
  // source, so with a single layer in view there is nothing to follow.
  const maxHops = computed(() => (layerSel.value < 0 ? hops.value : 1));

  const trace = computed<Trace>(() => {
    const r = run.value;
    const q = queryPos.value;
    const arcs: Arc[] = [];
    const hits = new Map<number, Hit>();
    if (!r || q == null) return { arcs, hits };

    // What a link in layer ℓ carries from its key was gathered there by the
    // layers below ℓ. So a source keeps the links that reached it (layer and
    // path weight), and is followed on only through lower layers than one of
    // them, with the strongest path such a link continues.
    type Reached = { li: number; path: number };
    let frontier = new Map<number, Reached[]>([[q, [{ li: Infinity, path: 1 }]]]);
    const expanded = new Set<number>();
    for (let hop = 1; hop <= maxHops.value; hop++) {
      const next = new Map<number, Reached[]>();
      for (const [src, reached] of frontier) {
        expanded.add(src);
        const pathInto = (li: number) => reached.reduce((p, x) => (x.li > li && x.path > p ? x.path : p), 0);
        const links = contributions(src)
          .map(c => ({ ...c, path: pathInto(c.series) * c.w })) // series: the layer, past the first hop
          .filter(c => c.path > 0 && c.w >= minWeight.value).sort((a, b) => b.w - a.w);
        links.slice(0, hop === 1 ? arcsPerSource.value : branch.value).forEach(c =>
          arcs.push({ from: src, to: c.key, series: c.series, w: c.w, path: c.path, hop }));
        const best = new Map<number, number>(); // key → strongest link
        for (const c of links) {
          let hit = hits.get(c.key);
          if (!hit) hits.set(c.key, hit = { pos: c.key, path: c.path, hop, w: c.w, series: c.series, links: [] });
          hit.path = Math.max(hit.path, c.path);
          if (hop < hit.hop || (hop === hit.hop && c.w > hit.w)) Object.assign(hit, { hop, w: c.w, series: c.series });
          hit.links.push({ series: c.series, w: c.w, hop, from: src });
          best.set(c.key, Math.max(best.get(c.key) ?? 0, c.w));
        }
        const follow = new Set([...best].sort((a, b) => b[1] - a[1]).slice(0, branch.value).map(([key]) => key));
        for (const c of links) {
          if (!follow.has(c.key) || expanded.has(c.key)) continue;
          let into = next.get(c.key);
          if (!into) next.set(c.key, into = []);
          into.push({ li: c.series, path: c.path });
        }
      }
      frontier = next;
      if (!frontier.size) break;
    }
    return { arcs, hits };
  });

  // Mass each (layer, head) puts on the keys the filters allow: for the
  // selected token, or averaged over every token of its message.
  const headMap = computed(() => {
    const r = run.value;
    const q = queryPos.value;
    if (!r || q == null) return null;
    let queries = [q];
    if (mapScope.value === 'message') {
      const s = r.segs[r.segOf[q]];
      queries = [];
      for (let p = s.start; p < s.end; p++) queries.push(p);
    }
    let max = 0;
    const rows = r.layers.map((L, li) => {
      const cells = new Float32Array(L.H);
      for (const qq of queries) {
        for (let h = 0; h < L.H; h++) {
          const base = (qq * L.H + h) * L.K;
          for (let k = 0; k < L.K; k++) {
            if (L.vals[base + k] < 1e-4) break;
            const key = L.idx[base + k];
            if (allowed(qq, key)) cells[h] += Math.max(0, weight(li, h, qq, key, L.vals[base + k]));
          }
        }
      }
      for (let h = 0; h < L.H; h++) {
        cells[h] /= queries.length;
        max = Math.max(max, cells[h]);
      }
      return { layer: L.layer, cells };
    });
    return { rows, max, queries: queries.length };
  });

  function select(p: number) {
    selPos.value = p;
  }

  function toggleSeries(i: number) {
    const s = new Set(hidden.value);
    if (s.has(i)) s.delete(i); else s.add(i);
    hidden.value = s;
  }

  function soloSeries(i: number | null) {
    const s = new Set<number>();
    if (i != null) for (let j = 0; j < seriesCount.value; j++) if (j !== i) s.add(j);
    hidden.value = s;
  }

  function setLayer(li: number) {
    if (li === layerSel.value) return;
    layerSel.value = li;
    hidden.value = new Set();
  }

  /** Focus one head of one layer (from the head map). */
  function focusHead(li: number, h: number) {
    layerSel.value = li;
    soloSeries(h);
  }

  return {
    selPos, layerSel, combine, reach, minDist, hops, branch, arcsPerSource, minWeight, skipFirst, skipSpecial, discount,
    hidden, mapScope, produced, queryPos, run, special, template, seriesCount, maxHops, trace, headMap,
    seriesLabel, allowed, weight, select, toggleSeries, soloSeries, setLayer, focusHead,
  };
});
