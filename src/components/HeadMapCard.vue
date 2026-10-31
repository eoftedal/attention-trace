<script setup lang="ts">
import { computed } from 'vue';
import { fmtPct } from '../lib/format';
import { useTraceStore } from '../stores/trace';
import StageSection from './StageSection.vue';

// Layers × heads: how much attention each head puts on the tokens the filters
// allow (other messages, far back, …). Finds the heads that carry
// information between turns; click one to trace with it alone.
const trace = useTraceStore();

const map = computed(() => trace.headMap);
const maxHeads = computed(() => Math.max(...trace.run!.layers.map(l => l.H)));
const what = computed(() => ({
  messages: 'tokens in other messages',
  far: `tokens at least ${trace.minDist} back`,
  all: 'earlier tokens',
}[trace.reach]));

const cellStyle = (v: number, max: number) => ({ '--p': `${Math.round((max > 0 ? v / max : 0) * 100)}%` });
const isOn = (li: number, h: number) => trace.layerSel === li && !trace.hidden.has(h);
</script>

<template>
  <StageSection v-if="trace.queryPos != null" id="headmap" title="Which heads look there" collapsed>
    <template #head="{ open }">
      <div v-if="open" class="seg-toggle" role="group" aria-label="Scope">
        <button type="button" :class="{ on: trace.mapScope === 'token' }" @click="trace.mapScope = 'token'">Selected token</button>
        <button type="button" :class="{ on: trace.mapScope === 'message' }" @click="trace.mapScope = 'message'">Its whole message</button>
      </div>
    </template>
    <template v-if="map">
    <p class="hint explain">
      Share of each head's attention that goes to {{ what }}{{ trace.mapScope === 'message' ? `, averaged over the ${map.queries} tokens of the selected token's message` : '' }}.
      Darker is more (scaled to the strongest head, {{ fmtPct(map.max) }}). Click a head to trace with it alone.
    </p>
    <div class="table-wrap headmap-wrap">
      <table class="headmap">
        <thead>
          <tr><th class="layer-label"></th><th v-for="h in maxHeads" :key="h" class="hm-head">{{ h - 1 }}</th></tr>
        </thead>
        <tbody>
          <tr v-for="(row, li) in map.rows" :key="row.layer" :class="{ cur: trace.layerSel === li }">
            <th class="layer-label">L{{ row.layer }}</th>
            <td v-for="(v, h) in row.cells" :key="h">
              <button type="button" class="hm-cell" :class="{ on: isOn(li, h) }" :style="cellStyle(v, map.max)"
                      :title="`Layer ${row.layer}, head ${h}: ${fmtPct(v)}`" @click="trace.focusHead(li, h)"></button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    </template>
  </StageSection>
</template>
