<script setup lang="ts">
import { computed } from 'vue';

// Letter weights rise from thin to heavy, the way a readout sharpens from the
// embedding to the output layer.
const props = defineProps<{ text: string }>();

const letters = computed(() => {
  const chars = [...props.text];
  const n = Math.max(1, chars.length - 1);
  return chars.map((c, i) => ({
    c,
    style: `font-weight:${Math.round(300 + (i / n) * 600)};--k:${Math.round(35 + (i / n) * 65)}%`,
  }));
});
</script>

<template>
  <h1 class="title" :aria-label="text">
    <span v-for="(l, i) in letters" :key="i" :style="l.style" aria-hidden="true">{{ l.c }}</span>
  </h1>
</template>
