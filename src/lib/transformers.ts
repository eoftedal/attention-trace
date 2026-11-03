// Minimal typings for the parts of transformers.js this page uses; the
// library's own types do not describe the callable tokenizer/model objects.
import type { ChatMessage } from './chatformat';
import {
  AutoModelForCausalLM, AutoTokenizer, InterruptableStoppingCriteria, StoppingCriteria, Tensor as HfTensor, TextStreamer,
} from '@huggingface/transformers';

export interface Tensor {
  dims: number[];
  type?: string;
  data: ArrayLike<number | bigint>;
  dispose?: () => void;
}


export interface TemplateOptions {
  add_generation_prompt?: boolean;
  tokenize?: boolean;
  enable_thinking?: boolean;
  tools?: unknown[];
}

export interface Tokenizer {
  (text: string, opts: { return_tensors?: 'pt'; add_special_tokens?: boolean }): Record<string, Tensor> & { input_ids: Tensor };
  decode(ids: number[], opts?: { skip_special_tokens?: boolean }): string;
  apply_chat_template(messages: ChatMessage[], opts: TemplateOptions): string;
}

/** Model inputs for a token sequence, as the tokenizer would return them. */
export function inputsFromIds(ids: number[]): Record<string, Tensor> & { input_ids: Tensor } {
  const n = ids.length;
  const input_ids = new HfTensor('int64', BigInt64Array.from(ids, v => BigInt(v)), [1, n]);
  const attention_mask = new HfTensor('int64', new BigInt64Array(n).fill(1n), [1, n]);
  return { input_ids, attention_mask } as unknown as Record<string, Tensor> & { input_ids: Tensor };
}

export interface CausalLM {
  (inputs: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  generate(opts: Record<string, unknown>): Promise<Tensor>;
  dispose(): Promise<unknown>;
}

export interface ProgressInfo {
  status?: string;
  file?: string;
  progress?: number;
}

type ProgressCallback = (info: ProgressInfo) => void;

export async function loadTokenizer(id: string, progress_callback: ProgressCallback): Promise<Tokenizer> {
  return (await AutoTokenizer.from_pretrained(id, { progress_callback } as never)) as unknown as Tokenizer;
}

export async function loadModel(id: string, progress_callback: ProgressCallback): Promise<CausalLM> {
  return (await AutoModelForCausalLM.from_pretrained(id, {
    dtype: 'q4f16',
    device: 'webgpu',
    progress_callback,
  } as never)) as unknown as CausalLM;
}

export interface GenerateOptions {
  maxNewTokens: number;
  temperature: number; // 0: greedy
  onText: (text: string) => void;
  stoppers: StoppingCriteria[];
}

/**
 * Caps reasoning at `budget` new tokens and the answer after it at `answer`.
 * `reasoning`: whether the reply starts inside a reasoning block (the prompt
 * opened it); otherwise it does when the first new token opens one.
 */
export class ThinkingBudget extends StoppingCriteria {
  overBudget = false;
  private closedAt = -1;
  private open: boolean | null;

  constructor(
    private promptLen: number,
    private budget: number,
    private answer: number,
    reasoning: boolean,
    private marker: (id: number) => 'open' | 'close' | null,
  ) {
    super();
    this.open = reasoning ? true : null;
  }

  _call(inputIds: number[][]): boolean[] {
    return inputIds.map(ids => {
      const n = ids.length - this.promptLen;
      if (n <= 0) return false;
      const m = this.marker(ids[ids.length - 1]);
      if (this.open === null) this.open = m === 'open';
      if (this.open && this.closedAt < 0) {
        if (m === 'close') this.closedAt = n;
        else if (n >= this.budget) return (this.overBudget = true);
        return false;
      }
      return n - Math.max(0, this.closedAt) >= this.answer;
    });
  }
}

/** Greedy or sampled generation; returns the new token ids. */
export async function generate(
  tok: Tokenizer, mdl: CausalLM, inputs: Record<string, Tensor>, { maxNewTokens, temperature, onText, stoppers }: GenerateOptions,
): Promise<number[]> {
  const streamer = new TextStreamer(tok as never, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: onText,
  } as never);
  const out = await mdl.generate({
    ...inputs,
    max_new_tokens: maxNewTokens,
    ...(temperature > 0 ? { do_sample: true, temperature } : { do_sample: false }),
    streamer,
    stopping_criteria: stoppers,
  });
  const n = inputs.input_ids.dims[1];
  return Array.from(out.data, Number).slice(n);
}

export { InterruptableStoppingCriteria };
