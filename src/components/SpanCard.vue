<script setup lang="ts">
import { computed } from 'vue';
import { fmtPct } from '../lib/format';
import { overall, useSpanStore, type SpanResult } from '../stores/span';
import { useTraceStore } from '../stores/trace';
import HeadGrid from './HeadGrid.vue';
import StageSection from './StageSection.vue';

const span = useSpanStore();
const trace = useTraceStore();

const fmtDiff = (v: number) => `${v >= 0 ? '+' : '−'}${(Math.abs(v) * 100).toFixed(1)} points`;

const rowsOf = (r: SpanResult) => r.layers.map(L => ({ layer: L.layer, cells: L.share }));
const current = computed(() => (span.result ? rowsOf(span.result) : null));
const pinnedRows = computed(() => (span.pinned ? rowsOf(span.pinned) : null));
const diffRows = computed(() => {
  if (!span.comparable || !current.value || !pinnedRows.value) return null;
  const a = pinnedRows.value;
  return current.value.map((row, li) => ({ layer: row.layer, cells: row.cells.map((v, h) => v - a[li].cells[h]) }));
});

// Messages by share; what the filters leave out is the rest.
const bars = computed(() => {
  const max = Math.max(...span.messages.map(m => m.share), 1e-9);
  return span.messages.map(m => ({ ...m, width: `${(m.share / max) * 100}%` }));
});
const counted = computed(() => span.messages.reduce((s, m) => s + m.share, 0));
const leftOut = computed(() => [
  trace.reach === 'messages' ? 'its own message' : trace.reach === 'far' ? `the ${trace.minDist} tokens before each token` : '',
  trace.skipFirst ? 'the first token' : '',
  trace.skipSpecial ? 'template tokens' : '',
].filter(Boolean).join(', '));

// The heads that read the target message most.
const topHeads = computed(() => {
  const r = span.result;
  if (!r) return [];
  const all: { li: number; h: number; v: number }[] = [];
  r.layers.forEach((L, li) => L.share.forEach((v, h) => all.push({ li, h, v })));
  return all.sort((a, b) => b.v - a.v).slice(0, 8).map(x => ({ ...x, label: `L${r.layers[x.li].layer} H${x.h}` }));
});

function showLines() {
  span.showLines = true;
  document.querySelector('.conv')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
</script>

<template>
  <StageSection id="span" title="Where a span reads from">
    <p class="hint explain">
      Mark some text, such as a sentence of the reply, to see where it came from: which earlier messages its tokens
      read, and which heads do the reading.
      <template v-if="span.start == null">
        <strong>Drag across text in the conversation</strong> (or shift-click its first and last token).
      </template>
      <template v-else>
        The conversation above draws lines from the span back to what it reads, more opaque for more attention.
      </template>
    </p>

    <p v-if="span.problem" class="hint warn">{{ span.problem }}</p>

    <template v-if="span.result">
      <ul class="span-facts">
        <li>Span <mark>{{ span.result.label }}</mark> in {{ span.result.role }}, {{ span.result.queries }} tokens</li>
      </ul>

      <h3 class="span-sub">Which messages it reads</h3>
      <p class="hint">
        Share of the span's attention, averaged over its tokens and every head. Not counted: {{ leftOut || 'nothing' }}
        (see the trace settings above) — {{ fmtPct(Math.max(0, 1 - counted)) }} of the attention.
        Click a message to see which heads read it.
      </p>
      <div class="msg-bars">
        <button v-for="b in bars" :key="b.seg" type="button" class="msg-bar" :class="[`role-${b.role}`, { on: b.seg === span.targetSeg }]"
                @click="span.target = b.seg">
          <span class="msg-bar-label">{{ b.label }}</span>
          <span class="msg-bar-track"><span class="msg-bar-fill" :style="{ width: b.width }"></span></span>
          <span class="msg-bar-value">{{ fmtPct(b.share) }}</span>
        </button>
      </div>

      <h3 class="span-sub">Which heads read {{ span.result.target }}</h3>
      <p class="span-summary">
        Averaged over every head, the span puts <strong>{{ fmtPct(overall(span.result)) }}</strong> of its attention on
        {{ span.result.target }}.
      </p>
      <div class="span-heads">
        <span class="legend-label">Strongest:</span>
        <button v-for="t in topHeads" :key="t.label" type="button" class="series" style="--sc: var(--heat)"
                :title="`Show the lines of ${t.label} alone`" @click="trace.focusHead(t.li, t.h); showLines()">{{ t.label }} {{ fmtPct(t.v) }}</button>
      </div>
      <div class="span-actions">
        <button type="button" class="ghost small" @click="span.pin()"
                title="Keep these numbers, then change the conversation (move the instruction, edit it) and compare">
          {{ span.pinned ? 'Pin this instead' : 'Pin for comparison' }}</button>
        <button v-if="span.pinned" type="button" class="ghost small" @click="span.unpin()">Remove pin</button>
        <button v-if="!span.showLines" type="button" class="ghost small" @click="showLines">Show lines in the conversation</button>
        <button type="button" class="ghost small" @click="span.clear()">Clear span</button>
      </div>

      <div class="head-grids">
        <HeadGrid v-if="pinnedRows && span.comparable" :rows="pinnedRows" :format="fmtPct"
                  :title="`Pinned — ${span.pinnedLabel}`" @cell="(li, h) => { trace.focusHead(li, h); showLines(); }" />
        <HeadGrid v-if="current" :rows="current" :format="fmtPct"
                  :title="span.pinned && span.comparable ? `Now — ${span.currentLabel}` : undefined"
                  @cell="(li, h) => { trace.focusHead(li, h); showLines(); }" />
        <HeadGrid v-if="diffRows" :rows="diffRows" :format="fmtDiff" diverging
                  title="Now minus pinned" @cell="(li, h) => { trace.focusHead(li, h); showLines(); }" />
      </div>
      <p v-if="span.pinned && !span.comparable" class="hint">
        Pinned: {{ span.pinnedLabel }}. It can only be compared with a run of the same model.
      </p>
    </template>
    <p v-else-if="span.pinned" class="hint">Pinned: {{ span.pinnedLabel }}. Mark a span to compare with it.</p>
  </StageSection>
</template>
