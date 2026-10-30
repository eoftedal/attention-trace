<script setup lang="ts">
import { ref, watch } from 'vue';
import { writeHash } from './lib/hash';
import { useChatStore } from './stores/chat';
import { useModelStore } from './stores/model';
import ThemeToggle from './components/ThemeToggle.vue';
import AppTitle from './components/AppTitle.vue';
import HowItWorks from './components/HowItWorks.vue';
import SetupCard from './components/SetupCard.vue';
import ChatCard from './components/ChatCard.vue';
import EmptyStage from './components/EmptyStage.vue';
import RunLoader from './components/RunLoader.vue';
import TraceCard from './components/TraceCard.vue';
import SourcesCard from './components/SourcesCard.vue';
import HeadMapCard from './components/HeadMapCard.vue';

const model = useModelStore();
const chat = useChatStore();

watch(() => ({ model: model.modelId, chat: chat.hashChat }), writeHash, { immediate: true });

// Collapsing the rail gives the trace the full width; remembered per browser.
const readCollapsed = () => {
  try {
    return localStorage.getItem('railCollapsed') === '1';
  } catch {
    return false;
  }
};
const collapsed = ref(readCollapsed());
watch(collapsed, v => {
  try {
    localStorage.setItem('railCollapsed', v ? '1' : '0');
  } catch {
    // storage unavailable: the choice just isn't remembered
  }
});
</script>

<template>
  <div class="shell" :class="{ collapsed }">
    <aside class="rail">
      <div v-if="collapsed" class="rail-strip">
        <button
          type="button" class="icon-btn" aria-controls="rail-body" aria-expanded="false"
          title="Show controls" aria-label="Show controls" @click="collapsed = false"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6" /></svg>
        </button>
        <span class="strip-title" aria-hidden="true">Attention trace</span>
        <ThemeToggle />
      </div>
      <div v-show="!collapsed" id="rail-body">
        <header class="masthead">
          <AppTitle text="Attention trace" />
          <div class="masthead-tools">
            <ThemeToggle />
            <button
              type="button" class="icon-btn" aria-controls="rail-body" aria-expanded="true"
              title="Hide controls" aria-label="Hide controls" @click="collapsed = true"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                   stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 6l-6 6 6 6" /></svg>
            </button>
          </div>
        </header>
        <p class="intro">
          Follow a token's attention back through a conversation, head by head, and see what
          travels between the system prompt, the user and the reply.
          <a href="https://eoftedal.github.io/logprobs/">Token probability explorer</a>
        </p>

        <div v-if="!model.webgpu" class="alert">
          <strong>WebGPU is not available in this browser.</strong>
          Use Chrome or Edge 113 or newer. On a supported browser, turn on
          <code>chrome://flags/#enable-unsafe-webgpu</code>.
        </div>

        <SetupCard />
        <ChatCard />
        <HowItWorks />
      </div>
    </aside>

    <main class="stage">
      <template v-if="chat.run">
        <TraceCard />
        <HeadMapCard />
        <SourcesCard />
      </template>
      <RunLoader v-else-if="chat.busy" />
      <EmptyStage v-else />
    </main>
  </div>
</template>
