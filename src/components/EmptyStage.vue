<script setup lang="ts">
import { computed } from 'vue';
import { useChatStore } from '../stores/chat';
import { useModelStore } from '../stores/model';

const model = useModelStore();
const chat = useChatStore();

const step = computed(() => (chat.busy ? 3 : model.ready ? 2 : 1));
const cls = (n: number) => (n < step.value ? 'done' : n === step.value ? 'now' : '');
</script>

<template>
  <section class="empty-stage">
    <h2>What travels between the messages of a chat</h2>
    <p>
      Every token a language model writes is built from what it reads back through attention. Each
      layer has several attention heads, and each head decides how much every earlier token matters
      for the current one. Most of that attention goes to the token just before, or to a sink
      token at the start. The interesting part is the long reach: the reply looking back at a name
      in the user's message, or at a rule in the system prompt.
    </p>
    <p>
      Hold a conversation with a small model in your browser, then click any token to see which
      earlier tokens it reads from, which heads do the reading, and how far back the trail goes.
    </p>
    <ol class="steps">
      <li :class="cls(1)">
        <div><strong>Load a model</strong>
          Weights download from Hugging Face once and are cached in the browser.</div>
      </li>
      <li :class="cls(2)">
        <div><strong>Have a conversation</strong>
          Write a system prompt and a message, or start from a sample, then generate the reply.</div>
      </li>
      <li :class="cls(3)">
        <div><strong>Trace the attention</strong>
          Click a token. By default only tokens in other messages count, so the links that carry
          information between turns stand out.</div>
      </li>
    </ol>
    <p class="hint">Everything runs locally on WebGPU; no text leaves your machine.</p>
  </section>
</template>
