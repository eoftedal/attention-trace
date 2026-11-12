<script setup lang="ts">
import { computed } from 'vue';
import { modelLabel } from '../lib/config';
import { display } from '../lib/format';
import { useChatStore } from '../stores/chat';
import { decode } from '../stores/model';
import { seriesColor, useTraceStore } from '../stores/trace';
import { useSpanStore } from '../stores/span';
import { fmtPct } from '../lib/format';
import ConversationView from './ConversationView.vue';
import StageSection from './StageSection.vue';

const chat = useChatStore();
const trace = useTraceStore();
const span = useSpanStore();
const spanLines = computed(() => {
  const f = span.showLines ? span.flow : null;
  return f && { heads: f.heads, max: Math.max(...f.links.map(l => l.w), 0) };
});

const facts = computed(() => {
  const r = trace.run!;
  const heads = r.layers.reduce((n, l) => n + l.H, 0);
  return [modelLabel(r.model), `${r.S} tokens`, `${r.layers.length} attention layers`, `${heads} heads`,
    `${(r.ms / 1000).toFixed(1)} s`];
});

const tokAt = (p: number | null) => (p == null ? '' : display(decode(trace.run!.ids[p])));

const series = computed(() => Array.from({ length: trace.seriesCount }, (_, i) => ({
  i, label: trace.seriesLabel(i), color: seriesColor(i), off: trace.hidden.has(i),
})));

const layerModel = computed({
  get: () => trace.layerSel,
  set: (v: number) => trace.setLayer(v),
});

const nArcs = computed(() => trace.trace.arcs.length);
const nHits = computed(() => trace.trace.hits.size);
</script>

<template>
  <StageSection id="trace" title="Attention trace">
    <template #head>
      <ul class="facts"><li v-for="f in facts" :key="f">{{ f }}</li></ul>
    </template>
    <p class="hint explain">
      Click a token to trace where it reads from. Shading and arcs show the keys it attends to,
      colored by {{ trace.layerSel < 0 ? 'layer' : 'head' }}; stronger attention is more opaque.
      Hover a shaded token for the weights.
      <strong v-if="chat.stale" class="warn">The conversation changed since this trace — trace it again.</strong>
    </p>

    <div class="controls">
      <div class="ctl">
        <label for="layer-sel">Layer</label>
        <select id="layer-sel" v-model.number="layerModel">
          <option :value="-1">All layers (color = layer)</option>
          <option v-for="(l, li) in trace.run!.layers" :key="l.layer" :value="li">
            Layer {{ l.layer }} ({{ l.H }} heads)
          </option>
        </select>
      </div>
      <div v-if="trace.layerSel < 0" class="ctl">
        <label for="combine-sel">Heads in a layer</label>
        <select id="combine-sel" v-model="trace.combine">
          <option value="max">Strongest head</option>
          <option value="mean">Average of heads</option>
        </select>
      </div>
      <div class="ctl">
        <label for="reach-sel">Look at</label>
        <div class="ctl-row">
          <select id="reach-sel" v-model="trace.reach">
            <option value="messages">Other messages only</option>
            <option value="far">Tokens far back</option>
            <option value="all">Every earlier token</option>
          </select>
          <label v-if="trace.reach === 'far'" class="inline">≥
            <input v-model.number="trace.minDist" type="number" min="1" class="text-input num"> back</label>
        </div>
      </div>
      <div class="ctl">
        <label for="hops-sel">Hops</label>
        <select v-if="trace.layerSel < 0" id="hops-sel" v-model.number="trace.hops">
          <option :value="1">1 — direct only</option>
          <option :value="2">2</option>
          <option :value="3">3</option>
          <option :value="4">4</option>
        </select>
        <select v-else id="hops-sel" disabled
                title="A later hop follows what a token gathered in lower layers, so it needs all layers in view">
          <option>1 — needs all layers</option>
        </select>
      </div>
      <div class="ctl">
        <label for="minw">Min weight {{ Math.round(trace.minWeight * 100) }}%</label>
        <input id="minw" v-model.number="trace.minWeight" type="range" min="0.005" max="0.3" step="0.005">
      </div>
    </div>
    <div class="toggles">
      <label class="check" title="The attention at a token is what the model reads while predicting the next one">
        <input v-model="trace.produced" type="checkbox"> Trace what produced the token (from the position before it)</label>
      <label class="check"><input v-model="trace.skipFirst" type="checkbox"> Ignore the first token (attention sink)</label>
      <label class="check" title="Role names, special tokens and the newlines between messages">
        <input v-model="trace.skipSpecial" type="checkbox"> Ignore template tokens</label>
      <label class="check" title="Idle heads park attention on a few early tokens. This subtracts the attention each key gets on average from the tokens of later messages (other than the one traced), so only links specific to this token remain.">
        <input v-model="trace.discount" type="checkbox"> Only count attention above each key's usual share</label>
      <label v-if="trace.maxHops > 1" class="check inline">Follow
        <input v-model.number="trace.branch" type="number" min="1" max="10" class="text-input num"> strongest per hop</label>
    </div>

    <div class="legend">
      <span class="legend-label">{{ trace.layerSel < 0 ? 'Layers' : 'Heads' }}</span>
      <button
        v-for="s in series" :key="s.i" type="button" class="series" :class="{ off: s.off }"
        :style="{ '--sc': s.color }" :title="`${s.label}: click to toggle, double-click to show only this one`"
        @click="trace.toggleSeries(s.i)" @dblclick="trace.soloSeries(s.i)"
      >{{ s.label }}</button>
      <button type="button" class="link-btn" @click="trace.soloSeries(null)">all</button>
    </div>

    <p class="trace-status hint">
      <template v-if="spanLines">
        Where the <span class="swatch-span">marked span</span> reads from, {{ spanLines.heads }}: lines from its tokens
        back to the tokens they read, the more opaque the more attention (the strongest line is {{ fmtPct(spanLines.max) }}).
        What it reads is shaded by how much. The trace settings apply (what to look at, ignored tokens).
        Click a token to see its own trace again.
      </template>
      <template v-else-if="trace.selPos != null">
        <template v-if="trace.queryPos !== trace.selPos">
          What produced <span class="swatch-sel">{{ tokAt(trace.selPos) }}</span>: the attention at
          <span class="swatch-query">{{ tokAt(trace.queryPos) }}</span> (#{{ trace.queryPos }}) while predicting it —
        </template>
        <template v-else>From <span class="swatch-sel">{{ tokAt(trace.selPos) }}</span> (#{{ trace.selPos }}):</template>
        {{ nHits }} tokens reached, {{ nArcs }} arcs.
        <template v-if="!nHits"> Nothing passes the filters — lower the min weight or widen what to look at.</template>
      </template>
      <template v-else>Click a token.</template>
    </p>

    <ConversationView />
  </StageSection>
</template>
