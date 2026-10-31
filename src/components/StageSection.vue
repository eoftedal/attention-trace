<script setup lang="ts">
import { ref, watch } from 'vue';

// A stage section whose title folds it away. Whether it is open is remembered
// per section in this browser; a folded section does not render (or compute)
// its body. The `head` slot sits beside the title and gets `open`.
const props = defineProps<{ id: string; title: string; collapsed?: boolean }>();

const key = `attention-trace:section:${props.id}`;
function remembered(): boolean | null {
  try {
    const v = localStorage.getItem(key);
    return v == null ? null : v === 'open';
  } catch {
    return null;
  }
}

const open = ref(remembered() ?? !props.collapsed);
watch(open, v => {
  try {
    localStorage.setItem(key, v ? 'open' : 'closed');
  } catch {
    // not remembered; the default applies next time
  }
});
</script>

<template>
  <section class="stage-section" :class="{ folded: !open }">
    <header class="section-head">
      <h2>
        <button type="button" class="section-toggle" :aria-expanded="open" @click="open = !open">
          <svg class="chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5 7.5 6 4 9.5" /></svg>{{ title }}
        </button>
      </h2>
      <slot name="head" :open="open" />
    </header>
    <div v-if="open" class="section-body"><slot /></div>
  </section>
</template>
