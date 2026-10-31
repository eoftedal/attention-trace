<script setup lang="ts">
import { computed } from 'vue';
import { fmtPct } from '../lib/format';
import { metricAt, overall, useSpanStore, type SpanMetric, type SpanResult } from '../stores/span';
import { useTraceStore } from '../stores/trace';
import HeadGrid from './HeadGrid.vue';

const span = useSpanStore();
const trace = useTraceStore();

const METRICS: { key: SpanMetric; label: string }[] = [
  { key: 'span', label: 'On the span' },
  { key: 'rest', label: 'On the rest of its message' },
  { key: 'ratio', label: 'Span vs rest, per token' },
];

// Shares as percentages; the per-token ratio is a log2, shown as a factor.
const factor = (log2: number) => {
  const f = 2 ** Math.abs(log2);
  return `${f >= 10 ? f.toFixed(0) : f.toFixed(1)}×${log2 < 0 ? ' less' : ''}`;
};
const fmt = (v: number) => (span.metric === 'ratio' ? factor(v) : fmtPct(v));
const fmtDiff = (v: number) => (span.metric === 'ratio'
  ? (v >= 0 ? `${factor(v)} more` : factor(v))
  : `${v >= 0 ? '+' : '−'}${(Math.abs(v) * 100).toFixed(1)} points`);

const rowsOf = (r: SpanResult) => r.layers.map((L, li) => ({
  layer: L.layer,
  cells: Float32Array.from(L.span, (_, h) => metricAt(r, span.metric, li, h)),
}));

const current = computed(() => (span.result ? rowsOf(span.result) : null));
const pinnedRows = computed(() => (span.pinned ? rowsOf(span.pinned) : null));
const diffRows = computed(() => {
  if (!span.comparable || !current.value || !pinnedRows.value) return null;
  const a = pinnedRows.value;
  return current.value.map((row, li) => ({ layer: row.layer, cells: row.cells.map((v, h) => v - a[li].cells[h]) }));
});

// The heads that put the most attention on the span.
const topHeads = computed(() => {
  const r = span.result;
  if (!r) return [];
  const all: { li: number; h: number; v: number }[] = [];
  r.layers.forEach((L, li) => L.span.forEach((v, h) => all.push({ li, h, v })));
  return all.sort((a, b) => b.v - a.v).slice(0, 8).map(x => ({ ...x, label: `L${r.layers[x.li].layer} H${x.h}` }));
});

const summary = computed(() => {
  const r = span.result;
  if (!r) return null;
  return { span: overall(r, 'span'), rest: overall(r, 'rest'), ratio: overall(r, 'ratio') };
});

function showLines() {
  span.showLines = true;
  document.querySelector('.conv')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

const replyTokens = computed({
  get: () => span.replyTokens,
  set: (v: number) => { span.replyTokens = Math.max(1, Math.min(512, Math.round(v) || 1)); },
});
</script>

<template>
  <section class="stage-section">
    <header class="section-head">
      <h2>Attention on a span</h2>
      <div v-if="span.result" class="seg-toggle" role="group" aria-label="What to show">
        <button v-for="m in METRICS" :key="m.key" type="button" :class="{ on: span.metric === m.key }"
                @click="span.metric = m.key">{{ m.label }}</button>
      </div>
    </header>
    <p class="hint explain">
      How much the reply reads from a stretch of text, such as an instruction injected into a tool result,
      for every layer and head: the share of the attention of its first tokens that lands on the span,
      compared with the rest of the span's message.
      <template v-if="span.start == null">
        <strong>Drag across text in the conversation</strong> (or shift-click its first and last token) to mark a span.
      </template>
      <template v-else>
        The conversation above draws lines from the reply's tokens to the span, more opaque for more attention;
        pick a head below to see its lines alone.
      </template>
    </p>

    <p v-if="span.problem" class="hint warn">{{ span.problem }}</p>

    <template v-if="span.result && summary">
      <ul class="span-facts">
        <li>Span <mark>{{ span.result.label }}</mark> in the {{ span.result.role }}, {{ span.result.spanLen }} tokens</li>
        <li>Rest of the message: {{ span.result.restLen }} tokens</li>
        <li><label>Reply tokens measured
          <input v-model.number.lazy="replyTokens" type="number" min="1" max="512" class="text-input"></label>
          <template v-if="span.result.queries < span.replyTokens"> (the reply has {{ span.result.queries }})</template>
        </li>
      </ul>
      <p class="span-summary">
        Averaged over every head, the reply puts <strong>{{ fmtPct(summary.span) }}</strong> of its attention on the span
        and <strong>{{ fmtPct(summary.rest) }}</strong> on the rest of the message:
        <strong>{{ factor(summary.ratio) }}</strong> per token.
      </p>
      <div class="span-heads">
        <span class="legend-label">Strongest on the span:</span>
        <button v-for="t in topHeads" :key="t.label" type="button" class="series" style="--sc: var(--heat)"
                :title="`Trace with ${t.label} alone`" @click="trace.focusHead(t.li, t.h)">{{ t.label }} {{ fmtPct(t.v) }}</button>
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
        <HeadGrid v-if="pinnedRows && span.comparable" :rows="pinnedRows" :format="fmt" :diverging="span.metric === 'ratio'"
                  :title="`Pinned — ${span.pinnedLabel}`" @cell="trace.focusHead" />
        <HeadGrid v-if="current" :rows="current" :format="fmt" :diverging="span.metric === 'ratio'"
                  :title="span.pinned && span.comparable ? `Now — “${span.result.label}” (${span.result.role})` : undefined"
                  @cell="trace.focusHead" />
        <HeadGrid v-if="diffRows" :rows="diffRows" :format="fmtDiff" diverging
                  title="Now minus pinned" @cell="trace.focusHead" />
      </div>
      <p v-if="span.pinned && !span.comparable" class="hint">
        The pinned span is from {{ span.pinnedLabel }}; it can only be compared with a run of the same model.
      </p>
    </template>
    <p v-else-if="span.pinned" class="hint">Pinned: {{ span.pinnedLabel }}. Mark a span to compare with it.</p>
  </section>
</template>
