<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue';
import { display, fmtPct } from '../lib/format';
import { decode } from '../stores/model';
import { hopFade, seriesColor, useTraceStore } from '../stores/trace';
import type { Segment } from '../stores/chat';

// The traced conversation as running text, one block per message. Tokens the
// trace reaches are shaded in the color of their strongest link (head or
// layer), more opaque for stronger attention; arcs join each query to the
// keys it attends to most.
const trace = useTraceStore();

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
const sources = computed(() => new Set(trace.trace.arcs.filter(a => a.hop > 1).map(a => a.from)));

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
  // Faintest first (later hops, then weaker links), so the strongest arcs are drawn on top.
  const arcs = [...trace.trace.arcs].sort((a, b) => b.hop - a.hop || a.w - b.w);
  paths.value = arcs.flatMap(a => {
    // Direct links start at the clicked token, even when the attention was
    // read one position earlier (what produced it); that position is outlined.
    const s = at(a.hop === 1 && trace.selPos != null ? trace.selPos : a.from), t = at(a.to);
    if (!s || !t) return [];
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
    const strength = Math.sqrt(a.w);
    return [{
      d, color: seriesColor(a.series),
      width: 1 + 3.5 * strength,
      opacity: (0.3 + 0.65 * strength) * hopFade(a.hop),
      dash: a.hop > 1 ? '5 4' : '',
      x: t.cx, y: Math.abs(s.top - t.top) < 4 ? t.top : t.bottom,
    }];
  });
}

const relayout = () => nextTick(layout);
watch(() => [trace.trace, blocks.value], relayout);
let ro: ResizeObserver | null = null;
onMounted(() => {
  ro = new ResizeObserver(relayout);
  if (wrap.value) ro.observe(wrap.value);
  relayout();
});
onBeforeUnmount(() => ro?.disconnect());

function onClick(e: MouseEvent) {
  const el = (e.target as Element).closest('[data-pos]') as HTMLElement | null;
  if (el) trace.select(Number(el.dataset.pos));
}

const tokText = (t: { text: string; special: boolean }) => (t.special ? display(t.text) : t.text);
</script>

<template>
  <div ref="wrap" class="conv" @click="onClick">
    <div v-for="b in blocks" :key="b.si" class="msg" :class="`role-${b.role}`">
      <div class="msg-label">{{ b.label }}</div>
      <div class="msg-body"><span
        v-for="t in b.toks" :key="t.p" class="tok" :data-pos="t.p"
        :class="{ special: t.special, sel: t.p === trace.selPos, query: t.p === trace.queryPos && t.p !== trace.selPos, hit: shade.has(t.p), src: sources.has(t.p) }"
        :style="shade.get(t.p)?.style" :title="shade.get(t.p)?.title ?? `#${t.p} ${JSON.stringify(t.text)}`"
      >{{ tokText(t) }}</span></div>
    </div>
    <svg class="arcs" aria-hidden="true">
      <g v-for="(p, i) in paths" :key="i">
        <path :d="p.d" :stroke="p.color" :stroke-width="p.width" :stroke-opacity="p.opacity"
              :stroke-dasharray="p.dash" fill="none" stroke-linecap="round" />
        <circle :cx="p.x" :cy="p.y" :r="1.5 + p.width / 2" :fill="p.color" :fill-opacity="p.opacity" />
      </g>
    </svg>
  </div>
</template>
