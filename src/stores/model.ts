import { defineStore } from 'pinia';
import { computed, reactive, ref, shallowRef } from 'vue';
import { env } from '@huggingface/transformers';
import { defaultModel, defaultTraceLength, models, topKFor, traceLengths } from '../lib/config';
import { readHash } from '../lib/hash';
import { installAttnHooks, type AttnPatchInfo, type BaseCache } from '../lib/onnxattn';
import { loadModel, loadTokenizer, type CausalLM, type ProgressInfo, type Tokenizer } from '../lib/transformers';
import { useChatStore } from './chat';

export type StatusLevel = 'info' | 'warn' | 'error';

export interface ProgressRow {
  key: string;
  label: string;
  pct: number;
}

// The tokenizer and model are large, non-reactive objects; they live outside
// the store's state.
let tok: Tokenizer | null = null;
let mdl: CausalLM | null = null;
let patchInfo: AttnPatchInfo | null = null; // set by onPatched while from_pretrained runs
const decodeCache = new Map<number, string>();

const lastPatch = (): AttnPatchInfo | null => patchInfo;

export const getTokenizer = () => tok;
export const getModel = () => mdl;

export function decode(id: number): string {
  let s = decodeCache.get(id);
  if (s === undefined) {
    s = tok ? tok.decode([id], { skip_special_tokens: false }) : '';
    decodeCache.set(id, s);
  }
  return s;
}

// ── transformers.js hooks ─────────────────────────────────────────
// The unpatched graph is cached in the same `transformers-cache` the logit
// lens and the token probability explorer use, so their downloads are reused.
let hooks: Promise<void> | null = null;
let onModelProgress: (url: string, loaded: number, total: number) => void = () => {};
let traceLengthForLoad = defaultTraceLength;

function ensureHooks() {
  hooks ??= (async () => {
    installAttnHooks(env, {
      baseCache: await openBaseCache(),
      attn: () => ({ maxTokens: traceLengthForLoad, topK: topKFor(traceLengthForLoad) }),
      onProgress: (url, loaded, total) => onModelProgress(url, loaded, total),
      onPatched: (_url, info) => { patchInfo = info; },
    });
  })();
  return hooks;
}

async function openBaseCache(): Promise<BaseCache> {
  try {
    if (typeof caches !== 'undefined') return await caches.open(env.cacheKey);
  } catch (e) {
    console.warn('Cache API unavailable, models will not be cached:', e);
  }
  const mem = new Map<string, Response>();
  return {
    match: async k => mem.get(k)?.clone(),
    put: async (k, r) => { mem.set(k, r); },
  };
}

export const useModelStore = defineStore('model', () => {
  const h = readHash();
  const webgpu = typeof navigator !== 'undefined' && 'gpu' in navigator && !!navigator.gpu;

  const modelId = ref(h.model && models.some(m => m.id === h.model) ? h.model : defaultModel);
  const traceLength = ref(defaultTraceLength);
  const loading = ref(false);
  const loadedModel = ref<string | null>(null);
  const numLayers = ref(0);
  const patch = shallowRef<AttnPatchInfo | null>(null);

  const status = ref('');
  const statusLevel = ref<StatusLevel>('info');
  const progressVisible = ref(false);
  const progressRows = ref<ProgressRow[]>([]);

  const ready = computed(() => loadedModel.value != null);
  // Longest conversation the loaded model can trace.
  const maxTokens = computed(() => patch.value?.maxTokens ?? traceLength.value);
  const canLoad = computed(() => webgpu && !loading.value &&
    (modelId.value !== loadedModel.value || traceLength.value !== patch.value?.maxTokens));

  function setStatus(text: string, level: StatusLevel = 'info') {
    status.value = text;
    statusLevel.value = level;
  }

  function fileRow(key: string, file: string): ProgressRow {
    let row = progressRows.value.find(r => r.key === key);
    if (!row) {
      row = reactive({ key, label: `${file}: …`, pct: 0 });
      progressRows.value.push(row);
    }
    return row;
  }

  function makeProgress(label: string) {
    return (info: ProgressInfo) => {
      if (!info || !info.file) return;
      const row = fileRow(`${label}|${info.file}`, info.file);
      if (info.status === 'done') {
        row.pct = 100;
        row.label = `${info.file}: done`;
      } else if (info.progress != null) {
        row.pct = info.progress;
        row.label = `${info.file}: ${Math.round(info.progress)}%`;
      } else if (info.status) {
        row.label = `${info.file}: ${info.status}`;
      }
    };
  }

  onModelProgress = (url, loaded, total) => {
    const file = url.split('/resolve/main/')[1] ?? url;
    const row = fileRow(`Model|${file}`, file);
    if (total) {
      row.pct = loaded / total * 100;
      row.label = `${file}: ${Math.round(row.pct)}%`;
    } else {
      row.label = `${file}: ${(loaded / 1e6).toFixed(0)} MB`;
    }
  };

  async function load() {
    const id = modelId.value;
    const chat = useChatStore();
    loading.value = true;
    progressVisible.value = true;
    progressRows.value = [];
    setStatus('');

    try {
      await ensureHooks();
      if (mdl) {
        setStatus('Releasing previous model…');
        await mdl.dispose();
        mdl = null;
        loadedModel.value = null;
      }
      decodeCache.clear();
      patchInfo = null;
      traceLengthForLoad = traceLengths.includes(traceLength.value) ? traceLength.value : defaultTraceLength;
      chat.clearRun();

      setStatus('Loading tokenizer…');
      tok = await loadTokenizer(id, makeProgress('Tokenizer'));

      setStatus('Downloading model weights (cached after first load)…');
      mdl = await loadModel(id, makeProgress('Model'));
      const info = lastPatch();
      if (!info) throw new Error('This model\'s graph has no attention layers that could be captured.');

      loadedModel.value = id;
      numLayers.value = info.numLayers;
      patch.value = info;
      progressRows.value = [];
      const n = info.sites.length;
      const note = n < info.numLayers ? ` (${info.numLayers - n} layers have no attention)` : '';
      if (info.skipped.length) {
        console.warn('Attention not captured:', info.skipped);
        setStatus(`Ready — attention captured in ${n} of ${info.numLayers} layers${note}; ` +
          `${info.skipped.length} could not be captured (see console).`, 'warn');
      } else {
        setStatus(`Ready — attention captured in ${n} of ${info.numLayers} layers${note}.`);
      }
    } catch (err) {
      setStatus(`Error: ${(err as Error).message}`, 'error');
      console.error(err);
      mdl = null;
      loadedModel.value = null;
      patch.value = null;
    } finally {
      loading.value = false;
    }
  }

  return {
    webgpu, modelId, traceLength, maxTokens, loading, loadedModel, numLayers, patch,
    status, statusLevel, progressVisible, progressRows, ready, canLoad, load,
  };
});
