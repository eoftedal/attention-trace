# Attention Trace

Hold a conversation with a small language model in the browser, then click any token to trace its
attention back through the conversation: which earlier tokens it reads, through which heads, and over
how many hops. Built to surface what travels *between* messages: from the system prompt and earlier
turns into the reply. Sibling of [logitlens-web](../logitlens-web), same stack (TypeScript, Vue 3,
Pinia, Vite, transformers.js on WebGPU).

Requires WebGPU (Chrome or Edge 113+).

## Development

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production build into dist/
```

## How attention is captured

The onnx-community models run attention inside fused `GroupQueryAttention` nodes, which never
expose their weights. `src/lib/onnxattn.ts` patches the ONNX graph in memory when it loads:

- Each GQA node is cloned with its value input replaced by one-hot rows (blocks of an identity
  matrix, one clone per `head_dim` keys). The clone's output `softmax(QKᵀ)·V` is then the attention
  matrix itself, with the model's own masking, sliding windows, softcapping and rotary embeddings.
- Layers that compute attention explicitly (Gemma 4's KV-shared layers) are read at their `Softmax`.
- Per layer, a `TopK` keeps the k strongest keys of every query and head
  (`attn_vals.<layer>` float16, `attn_idx.<layer>` int32, `[1, S, heads, k]`; k = 32, or 16 for
  4096-token traces). Any weight above 1/k is guaranteed to be kept.
- `logits` is cut to the last position (generation only needs that one).

The patched graph also generates normally. During cached decoding the attention outputs are
meaningless and ignored; after each reply one forward pass over the whole conversation records the
attention. The longest traceable conversation (1024, 2048 or 4096 tokens) is picked when the model
is loaded. The attention clones also run while generating, so a longer limit makes every generated
token slower (on Qwen3-0.6B, CPU: 2×, 3.4× and 6.7× the unpatched time).

The unpatched graphs share the `transformers-cache` with logitlens-web and the token probability
explorer, so models are only downloaded once.

## Attention on a span

Mark a stretch of text (drag across it, or shift-click its first and last token), such as an
instruction injected into a tool result. For every layer and head, the panel shows the share of the
attention of the reply's first tokens (32 by default) that lands on the span, on the rest of its
message, and the two compared per token. This is the per-head "probe attention" used in attention
studies of prompt injection, for one conversation. Pin a result, change the conversation (move the
instruction from the tool result into the user or system message, reword it) and trace again to
see both side by side with their difference. The shares are slightly low: only the strongest keys of
each head are kept (see above).

## Tool calls

A conversation can contain fake tool calls (`fetch_url`) and their results; nothing is ever called.
They are rendered in each model's own tool-calling format, so to the model it looks like it really
made the call: Qwen3, Qwen3.5 and Gemma 4 templates render `tool_calls` and `tool` messages
themselves, LFM2 gets its `<|tool_call_start|>[fetch_url(url="…")]<|tool_call_end|>` call text, and
Gemma 3, whose template has no tools, gets Google's prompt format (tools described in the system
prompt, `tool_code` and `tool_output` blocks). See `src/lib/chatformat.ts`.

## Layout

- `src/lib/onnxattn.ts`: graph surgery and the transformers.js fetch/cache hooks.
- `src/lib/chatformat.ts`: turns (including tool calls) to chat-template messages, per model family.
- `src/stores/model.ts`: loading. `src/stores/chat.ts`: conversation, generation, the analysis pass
  and splitting tokens into messages. `src/stores/trace.ts`: filters, multi-hop tracing, head map.
- `src/components/`: the rail (model, conversation) and the stage (trace view with arcs, head map,
  strongest sources).

The URL hash keeps the model and the conversation.
