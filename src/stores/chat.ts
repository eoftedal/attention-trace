import { defineStore } from 'pinia';
import { computed, markRaw, ref, shallowRef } from 'vue';
import { samples, tools, type Sample } from '../lib/config';
import {
  buildMessages, detectToolStyle, hasToolCalls, turnPieces, type Role, type ToolCall, type ToolStyle, type Turn,
} from '../lib/chatformat';
import { decodeJson, encodeJson, readHash } from '../lib/hash';
import { ATTN_IDX, ATTN_VALS } from '../lib/onnxattn';
import {
  generate, inputsFromIds, InterruptableStoppingCriteria, ThinkingBudget, type Tokenizer,
} from '../lib/transformers';
import { isSpecialToken, toFloat32 } from '../lib/format';
import { decode, getModel, getTokenizer, useModelStore } from './model';

export type { Role, Turn };

/** A stretch of the token sequence belonging to one message. */
export interface Segment {
  role: 'system' | Role | 'prompt';
  turn: number; // index into turns, -1 for the system prompt and the generation prompt
  start: number;
  end: number;  // exclusive
}

/** One attention layer of an analysis pass: top-K keys per query and head. */
export interface AttnLayer {
  layer: number;
  H: number;
  K: number;
  vals: Float32Array; // [S, H, K]
  idx: Int16Array;    // [S, H, K]; positions stay below the 4096-token trace limit
}

export interface AttnRun {
  model: string;
  key: string;        // conversation signature at analysis time
  ids: number[];
  S: number;
  layers: AttnLayer[];
  segs: Segment[];
  segOf: Int16Array;  // token → index into segs
  content: Uint8Array; // 1 for tokens of message text, 0 for the template around it
  ms: number;
}

interface Saved {
  system: string;
  turns: Turn[];
}

// Reasoning markers: Qwen3 <think>…</think>, Gemma 4 <|channel>…<channel|>.
const THINK_MARKER = /<\/?think>|<\|channel>|<channel\|>/;
const THINK_MARKERS = new RegExp(THINK_MARKER.source, 'g');

const THINK_OPEN = /^(<think>|<\|channel>)$/;
const THINK_CLOSE = /<\/think>|<channel\|>/;
// Ends reasoning that ran over its budget (Qwen's recipe for a thinking budget).
const CUT_SHORT = '\n\nConsidering the limited time, I have to give the answer based on the thinking so far.\n';
// The generation prompt may already open the reasoning (Qwen3.5: "<think>\n").
const openedBy = (prompt: string) => prompt.match(/(<think>|<\|channel>)\s*$/)?.[1];

/** Whether a reply has a finished reasoning block (templates drop those from later turns). */
export const hasReasoning = (content: string) => THINK_CLOSE.test(content);

// Generation is cut short through this; one generation runs at a time.
let stopper: InterruptableStoppingCriteria | null = null;

// The last reply as the model produced it. The trace uses these ids rather
// than re-tokenizing the reply's text, which can split it differently.
interface Generated {
  model: string;
  key: string;         // conversation signature with the reply added
  prompt: string;      // the generation prompt it was generated from
  promptIds: number[];
  ids: number[];       // the generated tokens, end-of-turn token included
}
let generated: Generated | null = null;

export const useChatStore = defineStore('chat', () => {
  const saved = (() => {
    const h = readHash();
    return h.chat ? decodeJson<Saved>(h.chat) : null;
  })();
  const first = samples.name;
  const system = ref(saved?.system ?? first.system);
  const turns = ref<Turn[]>(saved?.turns ?? first.turns.map(t => ({ ...t })));
  const draft = ref('');
  const sample = ref(saved ? '' : 'name');
  const thinking = ref(false);
  const maxNewTokens = ref(200);    // the answer (after any reasoning)
  const thinkingBudget = ref(1024); // reasoning tokens before it is cut short
  const sampling = ref(false);       // sample instead of always taking the likeliest token
  const temperature = ref(0.7);

  const generating = ref(false);
  const streamText = ref('');
  const running = ref(false);
  const runHint = ref('');
  const run = shallowRef<AttnRun | null>(null);

  const convKey = computed(() => JSON.stringify([system.value, turns.value]));
  const stale = computed(() => !!run.value && run.value.key !== convKey.value);
  const hashChat = computed(() => encodeJson({ system: system.value, turns: turns.value }));
  const busy = computed(() => generating.value || running.value);

  function pickSample(key: string) {
    sample.value = key;
    const s: Sample | undefined = samples[key];
    if (!s) return;
    system.value = s.system;
    turns.value = structuredClone(s.turns);
  }

  function setTurn(i: number, content: string) {
    turns.value[i].content = content;
    sample.value = '';
  }

  function removeTurn(i: number) {
    // a tool call takes its results with it
    let n = 1;
    if (turns.value[i].toolCalls?.length) while (turns.value[i + n]?.role === 'tool') n++;
    turns.value.splice(i, n);
    sample.value = '';
  }

  /** Adds a (fake) tool call by the assistant and its result. */
  function addToolCall() {
    turns.value.push(
      { role: 'assistant', content: '', toolCalls: [{ name: tools[0].function.name, arguments: { url: 'https://' } }] },
      { role: 'tool', content: '' },
    );
    sample.value = '';
  }

  function setToolCall(i: number, j: number, call: ToolCall) {
    turns.value[i].toolCalls![j] = call;
    sample.value = '';
  }

  function clearRun() {
    run.value = null;
  }

  // ── Templates ─────────────────────────────────────────────────────
  // Per tokenizer: whether the system prompt must be folded into the first
  // user turn (the template rejects a system role), how the template takes
  // tool calls (see chatformat.ts), and its end-of-turn tokens.
  let formatFor: Tokenizer | null = null;
  let foldSystem = false;
  let toolStyle: ToolStyle = 'native';
  let endOfTurn = new Set<number>();

  function template(tok: Tokenizer, ts: Turn[], addGen: boolean): string {
    if (formatFor !== tok) {
      formatFor = tok;
      foldSystem = !acceptsSystem(tok);
      toolStyle = detectToolStyle(tok, tools);
      endOfTurn = endOfTurnIds(tok);
    }
    // Tools are offered whenever the conversation uses them, also in renderings of a prefix.
    const withTools = hasToolCalls(turns.value);
    const opts = {
      add_generation_prompt: addGen, tokenize: false, enable_thinking: thinking.value,
      ...(withTools && toolStyle !== 'prompt' ? { tools } : {}),
    };
    const messages = buildMessages(system.value, ts, { fold: foldSystem, style: toolStyle, tools: withTools ? tools : null });
    return tok.apply_chat_template(messages, opts);
  }

  // Probed once per tokenizer, so that a template refusing some other
  // rendering (Qwen3.5 refuses a conversation without a user message) does
  // not move the system prompt.
  function acceptsSystem(tok: Tokenizer): boolean {
    try {
      tok.apply_chat_template([{ role: 'system', content: 'S' }, { role: 'user', content: 'U' }], {
        add_generation_prompt: true, tokenize: false,
      });
      return true;
    } catch {
      return false;
    }
  }

  const encode = (tok: Tokenizer, text: string) =>
    Array.from(tok(text, { add_special_tokens: false }).input_ids.data, Number);

  // The special tokens a template puts right after a message (<|im_end|>, <end_of_turn>, <turn|>).
  function endOfTurnIds(tok: Tokenizer): Set<number> {
    const out = new Set<number>();
    try {
      const text = tok.apply_chat_template([{ role: 'user', content: 'Hello' }, { role: 'assistant', content: 'World' }], {
        add_generation_prompt: false, tokenize: false,
      });
      const ids = encode(tok, text);
      for (const word of ['Hello', 'World']) {
        const n = encode(tok, text.slice(0, text.indexOf(word) + word.length)).length;
        if (n < ids.length && isSpecialToken(decode(ids[n]))) out.add(ids[n]);
      }
    } catch {
      // no probe, no end-of-turn tokens: boundaries fall back to template prefixes
    }
    return out;
  }

  // Message boundaries: the template rendered up to and including turn i is
  // normally a prefix of the full rendering (up to tokenization at the seam),
  // so its common prefix with the full ids ends turn i. Some templates render
  // the latest message differently (Qwen3 adds an empty <think> block to the
  // last reply, also when it is a tool call), so a turn also ends no earlier
  // than its own text in the full rendering, and runs on to the end-of-turn
  // token after it (and the newlines after that) if there is one before the
  // next message's text.
  // Also marks which tokens are message text rather than template (role
  // names, headers, newlines between messages).
  function segment(tok: Tokenizer, ids: number[], text: string, addGen: boolean): { segs: Segment[]; content: Uint8Array } {
    const common = (a: number[]) => {
      let n = 0;
      while (n < a.length && n < ids.length && a[n] === ids[n]) n++;
      return n;
    };
    const ends: { role: Segment['role']; turn: number; end: number }[] = [];
    const ts = turns.value;
    // A message is matched piece by piece: its text split around reasoning
    // markers (templates rewrite those, or drop the reasoning of earlier
    // replies altogether), then the name and arguments of each tool call.
    let cursor = 0; // in `text`
    const content = new Uint8Array(ids.length);
    // [start, end) of a message's own text in tokens, or null if not found.
    const span = (pieces: string[]): [number, number] | null => {
      let start = -1, end = 0;
      for (const piece of pieces.flatMap(p => p.split(THINK_MARKERS))) {
        const c = piece.trim();
        const at = c ? text.indexOf(c, cursor) : -1;
        if (at < 0) continue;
        cursor = at + c.length;
        const from = common(encode(tok, text.slice(0, at)));
        end = common(encode(tok, text.slice(0, cursor)));
        content.fill(1, from, end);
        if (start < 0) start = from;
      }
      return start < 0 ? null : [start, end];
    };
    const sysSpan = system.value.trim() ? span([system.value]) : null;
    const spans = ts.map(t => span(turnPieces(t)));
    // Past a message's own text, through the end-of-turn token after it and
    // the newlines after that, if that token comes before the next message's text.
    const throughEndOfTurn = (own: [number, number], next: number): number => {
      let p = own[1];
      while (p < next && !endOfTurn.has(ids[p])) p++;
      if (p >= next) return own[1];
      p++;
      while (p < ids.length && !decode(ids[p]).trim()) p++;
      return p;
    };
    const nextStart = (i: number) => spans.slice(i + 1).find(Boolean)?.[0] ?? ids.length;

    if (system.value.trim() && !foldSystem && ts.length) {
      // The system message alone; some templates refuse that (Qwen3.5), or
      // fold it into the first turn themselves (Gemma 3).
      let end = 0;
      try {
        end = common(encode(tok, template(tok, [], false)));
      } catch {
        if (sysSpan) end = throughEndOfTurn(sysSpan, nextStart(-1));
      }
      ends.push({ role: 'system', turn: -1, end });
    }
    for (let i = 0; i < ts.length; i++) {
      let end: number;
      if (i === ts.length - 1 && !addGen) {
        end = ids.length;
      } else {
        end = common(encode(tok, template(tok, ts.slice(0, i + 1), false)));
        if (spans[i]) end = Math.max(end, throughEndOfTurn(spans[i]!, nextStart(i)));
      }
      ends.push({ role: ts[i].role, turn: i, end });
    }
    if (addGen) ends.push({ role: 'prompt', turn: -1, end: ids.length });

    // A stretch of template tokens only (Gemma folds the system prompt into
    // the first user turn, leaving just <bos>) joins the next message.
    const textless = (a: number, b: number) => {
      for (let p = a; p < b; p++) if (!isSpecialToken(decode(ids[p]))) return false;
      return true;
    };
    const segs: Segment[] = [];
    let start = 0, from = 0; // `from`: where the next segment begins, including a carried stretch
    for (const e of ends) {
      const end = Math.max(start, Math.min(e.end, ids.length));
      if (end > start && !(e.role === 'system' && textless(start, end))) {
        segs.push({ role: e.role, turn: e.turn, start: from, end });
        from = end;
      }
      start = end;
    }
    if (start < ids.length) {
      if (segs.length) segs[segs.length - 1].end = ids.length;
      else segs.push({ role: 'user', turn: 0, start: 0, end: ids.length });
    }
    return { segs, content };
  }

  // The text of the whole conversation as the model reads it. A final reply
  // is appended to the generation prompt (returned as `prompt`) as text
  // rather than re-rendered: templates rewrite or drop the reasoning of a
  // final reply. A reply this page generated is traced from its ids instead
  // (see `analyze`).
  function conversationText(tok: Tokenizer): { text: string; addGen: boolean; prompt: string | null } {
    const ts = turns.value;
    const last = ts[ts.length - 1];
    if (last.role !== 'assistant') return { text: template(tok, ts, true), addGen: true, prompt: null };
    if (last.toolCalls?.length) return { text: template(tok, ts, false), addGen: false, prompt: null };
    const prompt = template(tok, ts.slice(0, -1), true);
    let reply = last.content.trim();
    const open = openedBy(prompt);
    if (open && reply.startsWith(open)) reply = reply.slice(open.length).replace(/^\n/, '');
    return { text: prompt + reply, addGen: false, prompt };
  }

  // Marks the generated tokens that are message text: runs of ordinary tokens
  // between special tokens, without blank tokens at either end (as `segment`
  // marks the pieces of a message's text).
  function markGenerated(content: Uint8Array, ids: number[], from: number) {
    content.fill(0, from);
    const blank = (p: number) => !decode(ids[p]).trim();
    for (let p = from; p < ids.length;) {
      if (isSpecialToken(decode(ids[p]))) { p++; continue; }
      let q = p;
      while (q < ids.length && !isSpecialToken(decode(ids[q]))) q++;
      let a = p, b = q - 1;
      while (a <= b && blank(a)) a++;
      while (b >= a && blank(b)) b--;
      if (a <= b) content.fill(1, a, b + 1);
      p = q;
    }
  }

  // ── Analysis pass ─────────────────────────────────────────────────
  async function analyze() {
    const tok = getTokenizer();
    const mdl = getModel();
    const model = useModelStore();
    if (!tok || !mdl || busy.value || !turns.value.length) return;
    running.value = true;
    runHint.value = '';
    try {
      const conv = conversationText(tok);
      // The reply as generated, while the conversation still ends with it.
      const gen = generated && generated.model === model.loadedModel && generated.key === convKey.value
        && generated.prompt === conv.prompt ? generated : null;
      const text = gen ? gen.prompt + tok.decode(gen.ids, { skip_special_tokens: false }) : conv.text;
      const inputs = gen
        ? inputsFromIds(gen.promptIds.concat(gen.ids))
        : tok(text, { return_tensors: 'pt', add_special_tokens: false });
      const ids = Array.from(inputs.input_ids.data, Number);
      if (ids.length > model.maxTokens) {
        throw new Error(`The conversation has ${ids.length} tokens; the model was loaded to trace up to ${model.maxTokens}. ` +
          'Pick a longer traceable conversation and reload it.');
      }
      const { segs, content } = segment(tok, ids, text, conv.addGen);
      if (gen) markGenerated(content, ids, gen.promptIds.length);
      // Let the loader paint before the forward pass starts.
      await new Promise(r => requestAnimationFrame(() => setTimeout(r)));

      const t0 = performance.now();
      const out = await mdl(inputs);
      const ms = performance.now() - t0;

      const layers: AttnLayer[] = [];
      for (const site of model.patch?.sites ?? []) {
        const v = out[ATTN_VALS + site.layer], ix = out[ATTN_IDX + site.layer];
        if (!v || !ix) continue;
        const [, , H, K] = v.dims;
        layers.push({ layer: site.layer, H, K, vals: toFloat32(v.data), idx: Int16Array.from(ix.data, Number) });
      }
      for (const [key, t] of Object.entries(out)) {
        if (key.startsWith('present')) t.dispose?.();
      }
      if (!layers.length) throw new Error('The model did not return attention outputs.');

      const segOf = new Int16Array(ids.length);
      segs.forEach((s, i) => segOf.fill(i, s.start, s.end));
      run.value = markRaw({ model: model.loadedModel!, key: convKey.value, ids, S: ids.length, layers, segs, segOf, content, ms });
    } catch (err) {
      runHint.value = `Error: ${(err as Error).message}`;
      console.error(err);
    } finally {
      running.value = false;
    }
  }

  // ── Generation ────────────────────────────────────────────────────
  /** Adds the draft as a user turn (if any), generates the reply, then traces. */
  async function send() {
    const tok = getTokenizer();
    const mdl = getModel();
    if (!tok || !mdl || busy.value) return;
    if (draft.value.trim()) {
      turns.value.push({ role: 'user', content: draft.value.trim() });
      draft.value = '';
      sample.value = '';
    }
    // a reply answers the user, or follows the results of a tool call
    if (!turns.value.length || turns.value[turns.value.length - 1].role === 'assistant') return;

    // The old trace no longer matches the conversation; the stage shows the loader instead.
    run.value = null;
    generating.value = true;
    streamText.value = '';
    runHint.value = '';
    stopper = new InterruptableStoppingCriteria();
    const cap = useModelStore().maxTokens;
    let unfinished = false;
    try {
      const text = template(tok, turns.value, true);
      const inputs = tok(text, { return_tensors: 'pt', add_special_tokens: false });
      const promptLen = inputs.input_ids.dims[1];
      let promptIds = Array.from(inputs.input_ids.data, Number);
      // The patched graph cannot even read a prompt longer than the trace limit.
      if (promptLen > cap) {
        throw new Error(`The conversation has ${promptLen} tokens; the model was loaded for up to ${cap}. ` +
          'Pick a longer traceable conversation and reload it.');
      }
      const reasoning = thinking.value;
      if (promptLen + maxNewTokens.value + (reasoning ? thinkingBudget.value : 0) > cap) {
        runHint.value = `Note: past ${cap} tokens the conversation can no longer be traced.`;
      }
      const opts = {
        temperature: sampling.value ? temperature.value : 0,
        onText: (t: string) => { streamText.value += t; },
      };
      const budget = reasoning
        ? new ThinkingBudget(promptLen, thinkingBudget.value, maxNewTokens.value, !!openedBy(text), id => {
          const s = decode(id);
          return THINK_OPEN.test(s) ? 'open' : THINK_CLOSE.test(s) ? 'close' : null;
        })
        : null;
      let newIds = await generate(tok, mdl, inputs, {
        ...opts,
        maxNewTokens: maxNewTokens.value + (reasoning ? thinkingBudget.value : 0),
        stoppers: budget ? [stopper, budget] : [stopper],
      });
      if (budget?.overBudget && !stopper.interrupted) {
        // Close the reasoning and let the model answer from what it has.
        const opened = openedBy(text) ?? newIds.map(decode).find(s => THINK_OPEN.test(s));
        const cut = `${CUT_SHORT}${opened === '<|channel>' ? '<channel|>' : '</think>'}\n\n`;
        streamText.value += cut;
        const soFar = text + tok.decode(newIds, { skip_special_tokens: false }) + cut;
        const more = tok(soFar, { return_tensors: 'pt', add_special_tokens: false });
        const answer = await generate(tok, mdl, more, { ...opts, maxNewTokens: maxNewTokens.value, stoppers: [stopper] });
        const moreIds = Array.from(more.input_ids.data, Number);
        promptIds = moreIds.slice(0, promptLen); // the answer was generated from this rendering
        newIds = moreIds.slice(promptLen).concat(answer);
      }
      const reply = replyText(tok, newIds, text);
      turns.value.push({ role: 'assistant', content: reply });
      generated = { model: useModelStore().loadedModel!, key: convKey.value, prompt: text, promptIds, ids: newIds };
      unfinished = THINK_MARKER.test(reply) && !THINK_CLOSE.test(reply);
    } catch (err) {
      runHint.value = `Error: ${(err as Error).message}`;
      console.error(err);
    } finally {
      generating.value = false;
      streamText.value = '';
      stopper = null;
    }
    await analyze();
    if (unfinished && !runHint.value) {
      runHint.value = 'The reply stopped while still reasoning, so it has no answer.';
    }
  }

  // The reply as text: special tokens dropped (end of turn), reasoning markers
  // kept so the template can tell reasoning from answer. When the generation
  // prompt opened the reasoning (Qwen3.5), the reply gets the opening marker.
  function replyText(tok: Tokenizer, newIds: number[], prompt: string): string {
    const kept = newIds.filter(id => {
      const s = decode(id);
      return !isSpecialToken(s) || THINK_MARKER.test(s);
    });
    const reply = tok.decode(kept, { skip_special_tokens: false }).trim();
    const open = openedBy(prompt);
    return open && !reply.includes(open) ? `${open}\n${reply}` : reply;
  }

  function stop() {
    stopper?.interrupt();
  }

  return {
    system, turns, draft, sample, thinking, maxNewTokens, thinkingBudget, sampling, temperature, generating, streamText, running, runHint, run,
    stale, hashChat, busy,
    pickSample, setTurn, removeTurn, addToolCall, setToolCall, clearRun, analyze, send, stop,
  };
});
