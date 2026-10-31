<script setup lang="ts">
import { computed } from 'vue';
import { display, fmtPct } from '../lib/format';
import { decode } from '../stores/model';
import { seriesColor, useTraceStore } from '../stores/trace';
import StageSection from './StageSection.vue';

// The tokens the trace reaches, strongest first, with a little context.
const trace = useTraceStore();
const LIMIT = 25;

const rows = computed(() => {
  const r = trace.run!;
  const q = trace.queryPos;
  if (q == null) return [];
  const ctx = (a: number, b: number) => display(r.ids.slice(Math.max(0, a), Math.max(0, b)).map(decode).join(''));
  return [...trace.trace.hits.values()]
    .sort((a, b) => b.path - a.path)
    .slice(0, LIMIT)
    .map(h => {
      const seg = r.segs[r.segOf[h.pos]];
      const links = [...h.links].sort((a, b) => b.w - a.w);
      const seen = new Set<number>();
      const top = links.filter(l => !seen.has(l.series * 8 + l.hop) && seen.add(l.series * 8 + l.hop)).slice(0, 5);
      return {
        pos: h.pos,
        before: ctx(h.pos - 4, h.pos),
        tok: display(decode(r.ids[h.pos])),
        after: ctx(h.pos + 1, h.pos + 4),
        msg: seg.role === 'prompt' ? 'gen. prompt' : seg.role,
        dist: q - h.pos,
        hop: h.hop,
        path: h.path,
        top: top.map(l => ({ label: trace.seriesLabel(l.series) + (l.hop > 1 ? ` hop ${l.hop}` : ''), color: seriesColor(l.series), w: l.w })),
      };
    });
});

function locate(p: number) {
  document.querySelector(`.conv [data-pos="${p}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
}
</script>

<template>
  <StageSection v-if="trace.selPos != null" id="sources" title="Strongest sources" collapsed>
    <p class="hint explain">
      The tokens the trace reaches, by path weight (the product of the attention weights along the way).
      Click a row to find the token; use the button to trace from it instead.
    </p>
    <div v-if="rows.length" class="table-wrap">
      <table class="sources">
        <thead><tr><th>Token in context</th><th>Message</th><th>Back</th><th>Hop</th><th>Path weight</th><th>Links (attention weight)</th><th></th></tr></thead>
        <tbody>
          <tr v-for="row in rows" :key="row.pos" @click="locate(row.pos)">
            <td class="ctx"><span class="dim">{{ row.before }}</span><mark>{{ row.tok }}</mark><span class="dim">{{ row.after }}</span></td>
            <td>{{ row.msg }}</td>
            <td class="num">{{ row.dist }}</td>
            <td class="num">{{ row.hop }}</td>
            <td class="num">{{ fmtPct(row.path) }}</td>
            <td><span v-for="l in row.top" :key="l.label" class="link-chip" :style="{ '--sc': l.color }">{{ l.label }} {{ fmtPct(l.w) }}</span></td>
            <td><button type="button" class="ghost small" @click.stop="trace.select(row.pos)">Trace</button></td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-else class="hint">No tokens pass the filters.</p>
  </StageSection>
</template>
