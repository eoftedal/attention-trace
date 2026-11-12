<script setup lang="ts">
import { computed } from 'vue';
import type { Segment } from '../stores/chat';

// A layers × heads grid where each cell is colored by the message the head
// reads most, in the color of that message's role, more opaque the more of
// the head's attention goes there. A cell with no message stays blank.
const props = defineProps<{
  title?: string;
  rows: { layer: number; top: ArrayLike<number>; share: ArrayLike<number> }[];
  sources: { label: string; role: Segment['role'] }[];
  format: (v: number) => string;
}>();
const emit = defineEmits<{ cell: [li: number, h: number] }>();

type Row = { layer: number; top: ArrayLike<number>; share: ArrayLike<number> };

const ROLE_COLOR: Record<Segment['role'], string> = {
  system: 'var(--role-system)', user: 'var(--role-user)', assistant: 'var(--role-assistant)',
  tool: 'var(--role-tool)', prompt: 'var(--muted)',
};

const maxHeads = computed(() => Math.max(...props.rows.map(r => r.top.length)));
const max = computed(() => {
  let m = 0;
  for (const r of props.rows) for (let h = 0; h < r.top.length; h++) if (r.top[h] >= 0) m = Math.max(m, r.share[h]);
  return m;
});
// The messages some head reads most, for the legend.
const legend = computed(() => {
  const used = new Set<number>();
  for (const r of props.rows) for (let h = 0; h < r.top.length; h++) if (r.top[h] >= 0) used.add(r.top[h]);
  return [...used].sort((a, b) => a - b).map(si => ({ ...props.sources[si], color: ROLE_COLOR[props.sources[si].role] }));
});

function cellStyle(row: Row, h: number) {
  const si = row.top[h];
  if (si < 0) return { '--p': '0%' };
  return { '--p': `${Math.round((max.value > 0 ? row.share[h] / max.value : 0) * 100)}%`, '--hc': ROLE_COLOR[props.sources[si].role] };
}
function cellTitle(row: Row, h: number) {
  const si = row.top[h];
  return `Layer ${row.layer}, head ${h}: ${si < 0 ? 'no message' : `${props.sources[si].label}, ${props.format(row.share[h])}`}`;
}
</script>

<template>
  <figure class="head-grid">
    <figcaption v-if="title">{{ title }}</figcaption>
    <div class="table-wrap headmap-wrap">
      <table class="headmap">
        <thead>
          <tr><th class="layer-label"></th><th v-for="h in maxHeads" :key="h" class="hm-head">{{ h - 1 }}</th></tr>
        </thead>
        <tbody>
          <tr v-for="(row, li) in rows" :key="row.layer">
            <th class="layer-label">L{{ row.layer }}</th>
            <td v-for="h in row.top.length" :key="h">
              <button type="button" class="hm-cell" :style="cellStyle(row, h - 1)" :title="cellTitle(row, h - 1)"
                      @click="emit('cell', li, h - 1)"></button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="grid-scale">
      <span v-for="s in legend" :key="s.label" class="legend-item"><span class="swatch role" :style="{ '--sc': s.color }"></span> {{ s.label }}</span>
      <span v-if="legend.length">· strongest {{ format(max) }}</span>
      <span v-else>no head reads a message the filters allow</span>
    </p>
  </figure>
</template>
