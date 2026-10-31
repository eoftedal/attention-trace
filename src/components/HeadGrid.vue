<script setup lang="ts">
import { computed } from 'vue';

// A layers × heads grid of values. Sequential values shade from nothing to
// the strongest; diverging values (differences, log ratios) shade one color
// above zero and another below, scaled to the largest magnitude.
const props = defineProps<{
  title?: string;
  rows: { layer: number; cells: ArrayLike<number> }[];
  diverging?: boolean;
  format: (v: number) => string;
}>();
const emit = defineEmits<{ cell: [li: number, h: number] }>();

const maxHeads = computed(() => Math.max(...props.rows.map(r => r.cells.length)));
const max = computed(() => {
  let m = 0;
  for (const r of props.rows) for (let h = 0; h < r.cells.length; h++) m = Math.max(m, Math.abs(r.cells[h]));
  return m;
});

function cellStyle(v: number) {
  const p = `${Math.round((max.value > 0 ? Math.abs(v) / max.value : 0) * 100)}%`;
  return props.diverging && v < 0 ? { '--p': p, '--hc': 'var(--heat-neg)' } : { '--p': p };
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
            <td v-for="h in row.cells.length" :key="h">
              <button type="button" class="hm-cell" :style="cellStyle(row.cells[h - 1])"
                      :title="`Layer ${row.layer}, head ${h - 1}: ${format(row.cells[h - 1])}`"
                      @click="emit('cell', li, h - 1)"></button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="grid-scale">
      <template v-if="diverging">
        <span class="swatch neg"></span> −{{ format(max).replace(/^[+-]/, '') }}
        <span class="swatch pos"></span> +{{ format(max).replace(/^[+-]/, '') }}
      </template>
      <template v-else><span class="swatch pos"></span> {{ format(max) }} at most</template>
    </p>
  </figure>
</template>
