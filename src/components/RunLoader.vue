<script setup lang="ts">
import { useChatStore } from '../stores/chat';

const chat = useChatStore();
</script>

<template>
  <section class="run-loader" role="status" aria-live="polite">
    <h2>{{ chat.generating ? 'Generating the reply' : 'Recording attention' }}</h2>
    <p class="hint">
      {{ chat.generating
        ? 'The reply streams into the conversation on the left. Attention is recorded when it is done.'
        : 'One forward pass over the whole conversation, keeping the strongest keys of every head.' }}
    </p>
    <div class="ghost-arcs" aria-hidden="true">
      <span v-for="i in 14" :key="i" :style="{ '--i': i }"></span>
    </div>
    <p class="hint">The first run after loading a model takes longer while the GPU compiles its shaders.</p>
  </section>
</template>
