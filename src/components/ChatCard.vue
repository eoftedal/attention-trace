<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { samples, systemPrompts, tools } from '../lib/config';
import type { ToolCall, Turn } from '../lib/chatformat';
import { hasReasoning, useChatStore } from '../stores/chat';
import { useModelStore } from '../stores/model';

const model = useModelStore();
const chat = useChatStore();

const systemEl = ref<HTMLTextAreaElement>();
const draftEl = ref<HTMLTextAreaElement>();
const list = ref<HTMLElement>();

// The preset matching the current system prompt, or '' for the user's own text.
const ownSystem = ref(false);
const systemPreset = computed({
  get: () => {
    const i = systemPrompts.findIndex(p => p.text === chat.system);
    return ownSystem.value || i < 0 ? '' : String(i);
  },
  set: v => {
    ownSystem.value = v === '';
    if (v !== '') chat.system = systemPrompts[Number(v)].text;
    else nextTick(() => systemEl.value?.focus());
  },
});

const lastRole = computed(() => chat.turns.at(-1)?.role);

// Templates keep only the answer of a reply once the user has written again.
const lastUser = computed(() => chat.turns.findLastIndex(t => t.role === 'user'));
const reasoningDropped = (i: number) =>
  chat.turns[i].role === 'assistant' && i < lastUser.value && hasReasoning(chat.turns[i].content);
// A reply answers the user, or follows the results of a tool call.
const awaitsReply = computed(() => !!lastRole.value && lastRole.value !== 'assistant');
const canSend = computed(() =>
  model.ready && !chat.busy && (!!chat.draft.trim() || awaitsReply.value));
const sendLabel = computed(() => (chat.draft.trim() || !awaitsReply.value ? 'Send' : 'Generate reply'));

const roleLabel = (t: Turn) =>
  t.role === 'user' ? 'User' : t.role === 'tool' ? 'Tool result' : t.toolCalls?.length ? 'Assistant · tool call' : 'Assistant';

// A tool call is edited as the tool plus one field per parameter of its schema.
const toolParams = (name: string): string[] =>
  Object.keys((tools.find(d => d.function.name === name)?.function.parameters.properties ?? {}) as object);
const callParams = (c: ToolCall) => [...new Set([...toolParams(c.name), ...Object.keys(c.arguments)])];
const argText = (v: unknown) => (typeof v === 'string' ? v : JSON.stringify(v));

function setCallName(i: number, j: number, name: string) {
  const c = chat.turns[i].toolCalls![j];
  chat.setToolCall(i, j, { name, arguments: Object.fromEntries(toolParams(name).map(k => [k, c.arguments[k] ?? ''])) });
}
function setCallArg(i: number, j: number, key: string, value: string) {
  const c = chat.turns[i].toolCalls![j];
  chat.setToolCall(i, j, { ...c, arguments: { ...c.arguments, [key]: value } });
}

const hint = computed(() => {
  if (chat.runHint) return chat.runHint;
  if (!model.ready) return 'Load a model first.';
  if (chat.generating) return 'Generating…';
  if (chat.running) return 'Recording attention…';
  return 'Ctrl/⌘ + Enter sends. Every reply is traced when it is done.';
});

// Grow (never shrink) a textarea to fit its content.
function autoGrow(el?: Element | null) {
  if (!(el instanceof HTMLTextAreaElement) || el.offsetParent === null) return;
  if (el.scrollHeight > el.clientHeight) {
    const cs = getComputedStyle(el);
    const borderY = parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth);
    el.style.height = (el.scrollHeight + borderY) + 'px';
  }
}
const growAll = () => nextTick(() => {
  autoGrow(systemEl.value);
  autoGrow(draftEl.value);
  list.value?.querySelectorAll('textarea').forEach(autoGrow);
});

watch(() => [chat.system, chat.turns.length, chat.draft, chat.sample], growAll);
onMounted(growAll);
</script>

<template>
  <section class="rail-section">
    <h2>Conversation</h2>
    <div class="field">
      <label for="sample-sel">Start from</label>
      <select id="sample-sel" :value="chat.sample" :disabled="chat.busy"
              @change="chat.pickSample(($event.target as HTMLSelectElement).value)">
        <option v-for="(s, key) in samples" :key="key" :value="key">{{ s.label }}</option>
        <option value="">Your own conversation</option>
      </select>
    </div>
    <div class="field">
      <label for="system-sel">System prompt</label>
      <select id="system-sel" v-model="systemPreset" class="stacked" :disabled="chat.busy">
        <option v-for="(p, i) in systemPrompts" :key="i" :value="String(i)">{{ p.label }}</option>
        <option value="">Your own text</option>
      </select>
      <textarea
        v-show="chat.system || systemPreset === ''" ref="systemEl" v-model="chat.system"
        aria-label="System prompt text" class="msg-text role-system" spellcheck="false" :disabled="chat.busy"
      ></textarea>
    </div>

    <ol ref="list" class="turns">
      <li v-for="(t, i) in chat.turns" :key="i" class="turn" :class="`role-${t.role}`">
        <div class="turn-head">
          <span class="role-tag">{{ roleLabel(t) }}</span>
          <button type="button" class="link-btn" :disabled="chat.busy"
                  :title="t.toolCalls?.length ? 'Remove this tool call and its result' : 'Remove this message'"
                  @click="chat.removeTurn(i)">Remove</button>
        </div>
        <div v-for="(c, j) in t.toolCalls ?? []" :key="j" class="tool-call msg-text">
          <select :value="c.name" :aria-label="`Tool of call ${j + 1}`" :disabled="chat.busy"
                  @change="setCallName(i, j, ($event.target as HTMLSelectElement).value)">
            <option v-for="d in tools" :key="d.function.name" :value="d.function.name">{{ d.function.name }}</option>
          </select>
          <label v-for="k in callParams(c)" :key="k" class="tool-arg">{{ k }}
            <input :value="argText(c.arguments[k])" class="text-input" spellcheck="false" :disabled="chat.busy"
                   @input="setCallArg(i, j, k, ($event.target as HTMLInputElement).value)">
          </label>
        </div>
        <textarea
          v-if="!t.toolCalls?.length"
          :value="t.content" :aria-label="`${roleLabel(t)} ${i + 1}`" :placeholder="t.role === 'tool' ? 'What the tool returned' : undefined" class="msg-text" spellcheck="false"
          :disabled="chat.busy" @input="chat.setTurn(i, ($event.target as HTMLTextAreaElement).value); autoGrow($event.target as Element)"
        ></textarea>
        <p v-if="reasoningDropped(i)" class="hint">
          The chat template leaves this reply's reasoning out of later turns, so the model — and the trace — only see its answer.
        </p>
      </li>
      <li v-if="!chat.generating" class="turn-add">
        <button type="button" class="link-btn" :disabled="chat.busy"
                title="Add a call to a tool and what it returned; nothing is really called"
                @click="chat.addToolCall()">+ Tool call and result</button>
      </li>
      <li v-if="chat.generating" class="turn role-assistant pending">
        <div class="turn-head"><span class="role-tag">Assistant</span>
          <button type="button" class="link-btn" @click="chat.stop()">Stop</button></div>
        <div class="msg-text streaming">{{ chat.streamText || '…' }}</div>
      </li>
    </ol>

    <div class="field">
      <label for="draft">Next user message</label>
      <textarea
        id="draft" ref="draftEl" v-model="chat.draft" class="msg-text draft" spellcheck="false"
        :placeholder="awaitsReply ? 'Leave empty to answer the last message' : 'Ask something'"
        @keydown.enter.exact.meta.prevent="chat.send()" @keydown.enter.exact.ctrl.prevent="chat.send()"
      ></textarea>
    </div>
    <div class="run-row">
      <label class="check" title="Qwen3, Qwen3.5 and Gemma 4: let the model reason before answering">
        <input v-model="chat.thinking" type="checkbox" :disabled="chat.busy"> Thinking
      </label>
      <label class="check max-tokens">{{ chat.thinking ? 'Answer' : 'Max' }}
        <input v-model.number="chat.maxNewTokens" type="number" min="1" max="4096" class="text-input" :disabled="chat.busy">
        tokens</label>
    </div>
    <div v-if="chat.thinking" class="run-row">
      <label class="check max-tokens" title="Reasoning longer than this is cut short, and the model is told to answer from what it has">
        Reasoning budget
        <input v-model.number="chat.thinkingBudget" type="number" min="16" max="4096" step="64" class="text-input" :disabled="chat.busy">
        tokens</label>
    </div>
    <div class="run-row">
      <label class="check" title="Off: always take the likeliest token, so the same conversation gives the same reply">
        <input v-model="chat.sampling" type="checkbox" :disabled="chat.busy"> Sample</label>
      <label v-if="chat.sampling" class="check max-tokens">Temperature
        <input v-model.number="chat.temperature" type="number" min="0.05" max="2" step="0.05" class="text-input" :disabled="chat.busy">
      </label>
    </div>
    <div class="run-row">
      <button class="ghost" :disabled="!model.ready || chat.busy || !chat.turns.length" @click="chat.analyze()"
              title="Record attention for the conversation as it is, without generating">
        Trace without generating
      </button>
      <button :disabled="!canSend" @click="chat.send()">{{ sendLabel }}</button>
    </div>
    <p class="hint" :class="{ error: hint.startsWith('Error') }">{{ hint }}</p>
  </section>
</template>
