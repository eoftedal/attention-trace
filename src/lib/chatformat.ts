// Turns of a conversation → chat-template messages, including (fake) tool
// calls. Each model family gets tool calls in its own native format, so to
// the model it looks like it really called the tool:
//  native: the template renders `tool_calls` and `tool` messages itself (Qwen3, Qwen3.5, Gemma 4)
//  lfm2:   the template lists tools and wraps results, but the call is part of
//          the assistant text: <|tool_call_start|>[fetch_url(url="…")]<|tool_call_end|>
//  prompt: no tool support in the template (Gemma 3): tools are described in
//          the system prompt, calls are ```tool_code blocks and results
//          ```tool_output blocks in a user turn, as Google documents for Gemma 3.

export type Role = 'user' | 'assistant' | 'tool';

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface Turn {
  role: Role;
  content: string;
  toolCalls?: ToolCall[]; // assistant turns only
}

export interface ToolDef {
  type: 'function';
  function: { name: string; description: string; parameters: Record<string, unknown> };
}

export interface ChatMessage {
  role: 'system' | Role;
  content: string;
  tool_calls?: { id: string; type: 'function'; function: { name: string; arguments: Record<string, unknown> } }[];
  name?: string;
  tool_call_id?: string;
}

export type ToolStyle = 'native' | 'lfm2' | 'prompt';

export interface BuildOptions {
  fold: boolean;            // fold the system prompt into the first user turn
  style: ToolStyle;
  tools: ToolDef[] | null;  // null when the conversation has no tool calls
}

export const hasToolCalls = (turns: Turn[]) => turns.some(t => t.toolCalls?.length || t.role === 'tool');

// fetch_url(url="https://…"): Python-style call, as LFM2 and Gemma 3 write them.
const pyCall = (c: ToolCall) =>
  `${c.name}(${Object.entries(c.arguments).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ')})`;

function promptToolsText(tools: ToolDef[]): string {
  return 'You have access to the following functions. To call a function, write it as Python in a ```tool_code block, '
    + 'and its result will be given to you in a ```tool_output block.\n'
    + tools.map(t => JSON.stringify(t.function)).join('\n');
}

export function buildMessages(system: string, turns: Turn[], { fold, style, tools }: BuildOptions): ChatMessage[] {
  let sys = system.trim() ? system : '';
  if (tools && style === 'prompt') sys = sys ? `${sys}\n\n${promptToolsText(tools)}` : promptToolsText(tools);

  const out: ChatMessage[] = [];
  // The last assistant turn's calls, matched to the results that follow it in order.
  let callIds: string[] = [], callNames: string[] = [], nextResult = 0, nCalls = 0;
  for (const t of turns) {
    const calls = t.role === 'assistant' ? t.toolCalls ?? [] : [];
    if (calls.length) {
      callIds = calls.map(() => `call_${++nCalls}`);
      callNames = calls.map(c => c.name);
      nextResult = 0;
      if (style === 'native') {
        out.push({
          role: 'assistant', content: t.content,
          tool_calls: calls.map((c, i) => ({ id: callIds[i], type: 'function', function: { name: c.name, arguments: c.arguments } })),
        });
      } else if (style === 'lfm2') {
        out.push({ role: 'assistant', content: `${t.content}<|tool_call_start|>[${calls.map(pyCall).join(', ')}]<|tool_call_end|>` });
      } else {
        const code = '```tool_code\n' + calls.map(pyCall).join('\n') + '\n```';
        out.push({ role: 'assistant', content: t.content ? `${t.content}\n${code}` : code });
      }
      continue;
    }
    if (t.role === 'tool') {
      if (style === 'prompt') {
        out.push({ role: 'user', content: '```tool_output\n' + t.content + '\n```' });
      } else {
        out.push({
          role: 'tool', content: t.content,
          name: callNames[nextResult] ?? callNames[0],
          tool_call_id: callIds[nextResult] ?? callIds[0],
        });
        nextResult++;
      }
      continue;
    }
    out.push({ role: t.role, content: t.content });
  }

  // Templates without tool support insist on alternating roles: merge neighbours.
  const merged: ChatMessage[] = [];
  for (const m of out) {
    const last = merged[merged.length - 1];
    if (style === 'prompt' && last && last.role === m.role) last.content = `${last.content}\n\n${m.content}`;
    else merged.push(m);
  }

  if (!sys) return merged;
  if (!fold) return [{ role: 'system', content: sys }, ...merged];
  const i = merged.findIndex(m => m.role === 'user');
  if (i >= 0) merged[i] = { ...merged[i], content: `${sys}\n\n${merged[i].content}` };
  return merged;
}

/** Text pieces of a turn as they appear in a rendering, in order (to find the turn in it). */
export function turnPieces(t: Turn): string[] {
  const out = [t.content];
  for (const c of t.toolCalls ?? []) {
    out.push(c.name);
    for (const v of Object.values(c.arguments)) out.push(typeof v === 'string' ? v : JSON.stringify(v));
  }
  return out;
}

interface TemplateTokenizer {
  (text: string, opts: { add_special_tokens?: boolean }): { input_ids: { data: ArrayLike<number | bigint> } };
  apply_chat_template(messages: ChatMessage[], opts: Record<string, unknown>): string;
}

/** How a tokenizer's chat template takes tool calls (see the top of this file). */
export function detectToolStyle(tok: TemplateTokenizer, tools: ToolDef[]): ToolStyle {
  if (tok('<|tool_call_start|>', { add_special_tokens: false }).input_ids.data.length === 1) return 'lfm2';
  const probe: Turn[] = [
    { role: 'user', content: 'u' },
    { role: 'assistant', content: '', toolCalls: [{ name: 'probe_tool_name', arguments: { a: 'probe_argument' } }] },
    { role: 'tool', content: 'probe_result' },
  ];
  try {
    const text = tok.apply_chat_template(buildMessages('', probe, { fold: false, style: 'native', tools }), {
      tools, add_generation_prompt: true, tokenize: false,
    });
    return text.includes('probe_tool_name') && text.includes('probe_argument') && text.includes('probe_result') ? 'native' : 'prompt';
  } catch {
    return 'prompt';
  }
}
