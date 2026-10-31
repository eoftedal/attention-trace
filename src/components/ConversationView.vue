<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import { display, fmtPct } from '../lib/format';
import { decode } from '../stores/model';
import { hopFade, seriesColor, useTraceStore } from '../stores/trace';
import { useSpanStore } from '../stores/span';
import type { Segment } from '../stores/chat';

// The traced conversation as running text, one block per message. Tokens the
// trace reaches are shaded in the color of their strongest link (head or
// layer), more opaque for stronger attention; arcs join each query to the
// keys it attends to most. With a span marked, it shows instead where the
// span reads from: lines from each of its tokens back to the tokens it
// attends to, more opaque for more attention.
const trace = useTraceStore();
const span = useSpanStore();
const inSpan = (p: number) => span.start != null && span.end != null && p >= span.start && p <= span.end;
const spanMode = computed(() => span.showLines && !!span.flow);
const SPAN_COLOR = 'var(--heat-neg)';
const MAX_SPAN_LINES = 100;

const ROLE_LABEL: Record<Segment['role'], string> = {
  system: 'System', user: 'User', assistant: 'Assistant', tool: 'Tool result', prompt: 'Generation prompt',
};

const blocks = computed(() => {
  const r = trace.run!;
  const n = { user: 0, assistant: 0, tool: 0 };
  return r.segs.map((s, si) => {
    const label = s.role === 'user' || s.role === 'assistant' || s.role === 'tool'
      ? `${ROLE_LABEL[s.role]} ${++n[s.role]}` : ROLE_LABEL[s.role];
    const toks = [];
    for (let p = s.start; p < s.end; p++) {
      const text = decode(r.ids[p]);
      toks.push({ p, text, special: !!trace.special[p] });
    }
    return { si, role: s.role, label, toks };
  });
});

// Shading and tooltips; only the tokens the trace reaches change. Opacity
// follows the attention weight of the link that reached the token, faded by
// how many hops back it is.
const shade = computed(() => {
  const out = new Map<number, { style: string; title: string }>();
  if (spanMode.value) {
    // tokens read by the attention the span gives them, span tokens by how much they read
    const { received, given } = span.flow!;
    const tok = (p: number) => `#${p} ${JSON.stringify(decode(trace.run!.ids[p]))}`;
    const put = (m: Map<number, number>, what: string) => {
      const max = Math.max(...m.values(), 1e-9);
      for (const [p, w] of m) {
        out.set(p, {
          style: `--hl:color-mix(in srgb, ${SPAN_COLOR} ${Math.round(Math.sqrt(w / max) * 70)}%, transparent)`,
          title: `${tok(p)}\n${what} ${fmtPct(w)}`,
        });
      }
    };
    put(received, 'Read by the span (mean over its tokens):');
    put(given, 'Attention this token puts on what it reads:');
    return out;
  }
  const { hits } = trace.trace;
  for (const hit of hits.values()) {
    const a = Math.round(Math.min(1, Math.sqrt(hit.w)) * 85 * hopFade(hit.hop));
    const links = [...hit.links].sort((x, y) => y.w - x.w).slice(0, 6)
      .map(l => `${trace.seriesLabel(l.series)} ${fmtPct(l.w)}${l.hop > 1 ? ` (hop ${l.hop}, from #${l.from})` : ''}`);
    out.set(hit.pos, {
      style: `--hl:color-mix(in srgb, ${seriesColor(hit.series)} ${a}%, transparent)`,
      title: `#${hit.pos} ${JSON.stringify(decode(trace.run!.ids[hit.pos]))}\n${links.join('\n')}`,
    });
  }
  return out;
});

// Sources the trace expanded from (hop 2+), outlined.
const sources = computed(() => (spanMode.value ? new Set<number>() : new Set(trace.trace.arcs.filter(a => a.hop > 1).map(a => a.from))));

// ── Arcs ──────────────────────────────────────────────────────────
const wrap = ref<HTMLElement>();
const paths = shallowRef<{ d: string; color: string; width: number; opacity: number; dash: string; x: number; y: number }[]>([]);

function layout() {
  const el = wrap.value;
  if (!el) return;
  const box = el.getBoundingClientRect();
  const rect = (p: number) => {
    const t = el.querySelector(`[data-pos="${p}"]`);
    if (!t) return null;
    const r = t.getBoundingClientRect();
    return { cx: r.left - box.left + r.width / 2, top: r.top - box.top, bottom: r.bottom - box.top };
  };
  const cache = new Map<number, ReturnType<typeof rect>>();
  const at = (p: number) => {
    if (!cache.has(p)) cache.set(p, rect(p));
    return cache.get(p)!;
  };
  const curve = (s: NonNullable<ReturnType<typeof rect>>, t: NonNullable<ReturnType<typeof rect>>) => {
    let d: string;
    if (Math.abs(s.top - t.top) < 4) {
      // same line: a hump over the text
      const h = Math.min(42, 10 + Math.abs(s.cx - t.cx) * 0.15);
      d = `M${s.cx},${s.top} C${s.cx},${s.top - h} ${t.cx},${t.top - h} ${t.cx},${t.top}`;
    } else {
      // from the top of the query up to the bottom of the key
      const k = Math.min(140, (s.top - t.bottom) * 0.45);
      d = `M${s.cx},${s.top} C${s.cx},${s.top - k} ${t.cx},${t.bottom + k} ${t.cx},${t.bottom}`;
    }
    return { d, x: t.cx, y: Math.abs(s.top - t.top) < 4 ? t.top : t.bottom };
  };

  if (spanMode.value) {
    // The strongest links, scaled to the strongest one; weakest drawn first.
    const links = [...span.flow!.links].sort((a, b) => b.w - a.w).slice(0, MAX_SPAN_LINES).reverse();
    const max = links.length ? links[links.length - 1].w : 1;
    paths.value = links.flatMap(l => {
      const s = at(l.from), t = at(l.to);
      if (!s || !t) return [];
      const strength = Math.sqrt(l.w / max);
      return [{ ...curve(s, t), color: SPAN_COLOR, width: 0.75 + 3 * strength, opacity: 0.06 + 0.8 * strength, dash: '' }];
    });
    return;
  }

  // Faintest first (later hops, then weaker links), so the strongest arcs are drawn on top.
  const arcs = [...trace.trace.arcs].sort((a, b) => b.hop - a.hop || a.w - b.w);
  paths.value = arcs.flatMap(a => {
    // Direct links start at the clicked token, even when the attention was
    // read one position earlier (what produced it); that position is outlined.
    const s = at(a.hop === 1 && trace.selPos != null ? trace.selPos : a.from), t = at(a.to);
    if (!s || !t) return [];
    const strength = Math.sqrt(a.w);
    return [{
      ...curve(s, t), color: seriesColor(a.series),
      width: 1 + 3.5 * strength,
      opacity: (0.3 + 0.65 * strength) * hopFade(a.hop),
      dash: a.hop > 1 ? '5 4' : '',
    }];
  });
}

const relayout = () => nextTick(layout);
watch(() => [trace.trace, blocks.value, spanMode.value, span.flow], relayout);
let ro: ResizeObserver | null = null;
onMounted(() => {
  ro = new ResizeObserver(relayout);
  if (wrap.value) ro.observe(wrap.value);
  relayout();
});
onBeforeUnmount(() => ro?.disconnect());

// Click traces a token; shift-click marks a span (first and last token), as
// does dragging across text.
function onClick(e: MouseEvent) {
  const el = (e.target as Element).closest('[data-pos]') as HTMLElement | null;
  if (!el) return;
  if (e.shiftKey) {
    window.getSelection()?.removeAllRanges(); // shift-click also extends the browser's text selection
    span.shiftClick(Number(el.dataset.pos));
  }
  else if (window.getSelection()?.isCollapsed !== false) {
    span.showLines = false; // back to the clicked token's trace
    trace.select(Number(el.dataset.pos));
  }
}

const posOf = (n: Node | null) => {
  const el = (n instanceof Element ? n : n?.parentElement)?.closest('.conv [data-pos]') as HTMLElement | null;
  return el ? Number(el.dataset.pos) : null;
};

function onMouseUp(e: MouseEvent) {
  if (e.shiftKey) return;
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;
  const a = posOf(sel.anchorNode), b = posOf(sel.focusNode);
  if (a == null || b == null || a === b && sel.toString().length < 2) return;
  span.mark(a, b);
  sel.removeAllRanges();
}

const tokText = (t: { text: string; special: boolean }) => (t.special ? display(t.text) : t.text);
</script>

<template>
  <div ref="wrap" class="conv" @click="onClick" @mouseup="onMouseUp">
    <div v-for="b in blocks" :key="b.si" class="msg" :class="`role-${b.role}`">
      <div class="msg-label">{{ b.label }}</div>
      <div class="msg-body"><span
        v-for="t in b.toks" :key="t.p" class="tok" :data-pos="t.p"
        :class="{
          special: t.special, hit: shade.has(t.p), src: sources.has(t.p), span: inSpan(t.p),
          sel: !spanMode && t.p === trace.selPos, query: !spanMode && t.p === trace.queryPos && t.p !== trace.selPos,
        }"
        :style="shade.get(t.p)?.style" :title="shade.get(t.p)?.title ?? `#${t.p} ${JSON.stringify(t.text)}`"
      >{{ tokText(t) }}</span></div>
    </div>
    <svg class="arcs" aria-hidden="true">
      <g v-for="(p, i) in paths" :key="i">
        <path :d="p.d" :style="{ stroke: p.color }" :stroke-width="p.width" :stroke-opacity="p.opacity"
              :stroke-dasharray="p.dash" fill="none" stroke-linecap="round" />
        <circle :cx="p.x" :cy="p.y" :r="1.5 + p.width / 2" :style="{ fill: p.color }" :fill-opacity="p.opacity" />
      </g>
    </svg>
  </div>
</template>
