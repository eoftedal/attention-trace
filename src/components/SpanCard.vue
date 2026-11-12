<script setup lang="ts">
import { computed, ref } from 'vue';
import { modelLabel } from '../lib/config';
import { fmtPct } from '../lib/format';
import { overall, useSpanStore, type MessageShare, type SpanResult } from '../stores/span';
import { useTraceStore } from '../stores/trace';
import HeadGrid from './HeadGrid.vue';
import SourceGrid from './SourceGrid.vue';
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

// Messages by share, in total or per readable token of the message.
const by = ref<'total' | 'token'>('total');
const bars = computed(() => {
  const of = (m: MessageShare) => (by.value === 'token' ? m.perToken : m.share);
  const max = Math.max(...span.messages.map(of), 1e-9);
  return span.messages.map(m => ({ ...m, width: `${(of(m) / max) * 100}%` }));
});
// Of the span's attention: on tokens the filters leave out, and below the recorded top-K keys.
const counted = computed(() => span.messages.reduce((s, m) => s + m.share, 0));
const filtered = computed(() => Math.max(0, span.mass.captured - counted.value));
const lost = computed(() => Math.max(0, 1 - span.mass.captured));

// Inside the target message: its sentences and lines.
const targetRole = computed(() => span.messages.find(m => m.seg === span.targetSeg)?.role ?? 'user');
const pieceBars = computed(() => {
  const of = (p: { share: number; perToken: number }) => (by.value === 'token' ? p.perToken : p.share);
  const max = Math.max(...span.pieces.map(of), 1e-9);
  return span.pieces.map(p => ({ ...p, width: `${(of(p) / max) * 100}%` }));
});
function locate(p: number) {
  document.querySelector(`.conv [data-pos="${p}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

// The message each head reads most, and the heads reading a different one
// than in the pinned run (among heads that read some message at all).
const MIN_SOURCE = 0.02; // a head reads a message from this share of its attention
const sourceRows = (r: SpanResult) => r.layers.map(L => ({ layer: L.layer, top: L.top, share: L.topShare }));
const spanLabel = (x: SpanResult) => `“${x.label}” (${x.role})`;
const switched = computed(() => {
  const a = span.pinned, b = span.result;
  if (!a || !b || !span.comparable) return null;
  let n = 0, of = 0;
  const rows = b.layers.map((L, li) => {
    const P = a.layers[li];
    const top = new Int16Array(L.top.length).fill(-1);
    for (let h = 0; h < L.top.length; h++) {
      if (L.topShare[h] < MIN_SOURCE && P.topShare[h] < MIN_SOURCE) continue;
      of++;
      const was = P.top[h] >= 0 ? a.sources[P.top[h]].label : null;
      const now = L.top[h] >= 0 ? b.sources[L.top[h]].label : null;
      if (was !== now) { n++; top[h] = L.top[h]; }
    }
    return { layer: L.layer, top, share: L.topShare };
  });
  return { rows, n, of };
});
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
        Share of the span's attention, averaged over its tokens and every head. Per token divides a message's
        share by the tokens the span can read in it, so a long message does not win by size.
        Not counted: {{ leftOut || 'nothing' }} (see the trace settings above) — {{ fmtPct(filtered) }} of the attention.
        Only the {{ span.mass.k }} strongest keys of each head are recorded; they hold {{ fmtPct(span.mass.captured) }}
        of the span's attention, so the shares are up to {{ fmtPct(lost) }} low.
        Click a message to see which heads read it.
      </p>
      <div class="seg-toggle span-by" role="group" aria-label="Bars show">
        <button type="button" :class="{ on: by === 'total' }" @click="by = 'total'">Total</button>
        <button type="button" :class="{ on: by === 'token' }" @click="by = 'token'">Per token</button>
      </div>
      <div class="msg-bars">
        <button v-for="b in bars" :key="b.seg" type="button" class="msg-bar" :class="[`role-${b.role}`, { on: b.seg === span.targetSeg }]"
                :title="`${b.label}: ${fmtPct(b.share)} of the span's attention over ${b.tokens} readable tokens, ${fmtPct(b.perToken)} per token`"
                @click="span.target = b.seg">
          <span class="msg-bar-label">{{ b.label }}</span>
          <span class="msg-bar-track"><span class="msg-bar-fill" :style="{ width: b.width }"></span></span>
          <span class="msg-bar-value">{{ by === 'token' ? `${fmtPct(b.perToken)} / token` : fmtPct(b.share) }}</span>
        </button>
      </div>
      <template v-if="span.pieces.length > 1">
        <h4 class="span-sub sub2">Inside {{ span.result.target }}, by sentence or line</h4>
        <div class="msg-bars pieces">
          <button v-for="pc in pieceBars" :key="pc.start" type="button" class="msg-bar" :class="`role-${targetRole}`"
                  :title="`${fmtPct(pc.share)} of the span's attention over ${pc.tokens} readable tokens, ${fmtPct(pc.perToken)} per token. Click to find it.`"
                  @click="locate(pc.start)">
            <span class="msg-bar-label piece-label">{{ pc.label }}</span>
            <span class="msg-bar-track"><span class="msg-bar-fill" :style="{ width: pc.width }"></span></span>
            <span class="msg-bar-value">{{ by === 'token' ? `${fmtPct(pc.perToken)} / token` : fmtPct(pc.share) }}</span>
          </button>
        </div>
      </template>

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

      <h3 class="span-sub">Which message each head reads most</h3>
      <p class="hint">
        Each head is colored by the message it puts most of the span's attention on, more opaque the more.
        Heads that read instructions normally read the system prompt or the user's request; under an injection
        they read the tool result instead. Pin a run and change the conversation: the third grid shows the heads
        that read a different message now.
        <template v-if="switched">
          <strong>{{ switched.n }} of {{ switched.of }}</strong> heads read a different message than in the pinned run
          (heads that put at least {{ fmtPct(MIN_SOURCE) }} of their attention on some message).
        </template>
      </p>
      <div class="head-grids">
        <SourceGrid v-if="span.pinned && span.comparable" :rows="sourceRows(span.pinned)" :sources="span.pinned.sources" :format="fmtPct"
                    :title="`Pinned — ${modelLabel(span.pinned.model)}: ${spanLabel(span.pinned)}`"
                    @cell="(li, h) => { trace.focusHead(li, h); showLines(); }" />
        <SourceGrid :rows="sourceRows(span.result)" :sources="span.result.sources" :format="fmtPct"
                    :title="span.pinned && span.comparable ? `Now — ${spanLabel(span.result)}` : undefined"
                    @cell="(li, h) => { trace.focusHead(li, h); showLines(); }" />
        <SourceGrid v-if="switched" :rows="switched.rows" :sources="span.result.sources" :format="fmtPct"
                    title="Heads that switched, colored by what they read now"
                    @cell="(li, h) => { trace.focusHead(li, h); showLines(); }" />
      </div>
      <p v-if="span.pinned && !span.comparable" class="hint">
        Pinned: {{ span.pinnedLabel }}. It can only be compared with a run of the same model.
      </p>
    </template>
    <p v-else-if="span.pinned" class="hint">Pinned: {{ span.pinnedLabel }}. Mark a span to compare with it.</p>
  </StageSection>
</template>
