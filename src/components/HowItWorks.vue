<template>
  <details class="how">
    <summary>How the trace works</summary>
    <p>
      The models run their attention in a fused GroupQueryAttention kernel that never exposes the
      weights. When a model loads, its graph is patched in memory: each attention node is cloned
      with the values replaced by one-hot vectors, so the clone's output
      <span class="mono">softmax(QKᵀ)·V</span> is the attention matrix itself, with the model's own
      masking, sliding windows and rotary embeddings. Layers that compute attention explicitly are
      read at their softmax.
    </p>
    <p>
      Only the 32 strongest keys of every query and head are kept (16 for traces up to 4096 tokens),
      which keeps the readback small and never drops a weight above 1/32 ≈ 3% (1/16 ≈ 6%).
    </p>
    <p>
      One hop shows what the selected token reads directly. With more hops, the strongest keys are
      traced in turn — a reply that reads an earlier reply is followed on to the user message that
      reply read — and each hop is drawn fainter, with dashed arcs. The path weight is the product of
      the weights along the way. Each hop uses
      the same layer setting, so this is an approximation: information that reached a token in layer
      ℓ was itself gathered by earlier layers.
    </p>
    <p>
      Heads that have nothing to read park their attention on a few early tokens: the first token,
      and newlines and punctuation near the start. By default every link counts only with what it
      gets above the key's usual share: its mean attention, in that head, from the later tokens of
      every message but the one being traced, so the reading under study does not count as usual.
      A system prompt then shows up only when a token really reads it.
    </p>
    <p>
      Attention says where a head reads from, not how much the result matters to the output. Treat
      the trace as a map of where to look, not as an attribution.
    </p>
  </details>
</template>
