// Attention-capture graph surgery for onnxruntime-genai style ONNX exports.
//
// The onnx-community q4f16 models compute attention inside fused
// GroupQueryAttention (GQA) nodes, which never expose the attention weights.
// This module edits the ONNX protobuf in memory and adds, for every attention
// layer l, two graph outputs with the strongest keys of every query and head:
//
//   attn_vals.<l>  float16 [1, positions, heads, K]   attention weight
//   attn_idx.<l>   int32   [1, positions, heads, K]   key position
//
// GQA layers: the node is cloned with its value input replaced by one-hot
// rows, V[j] = e_(j − c·d) for the keys of block c (d = head dim). The
// attention output softmax(QKᵀ)·V of each clone is then the attention matrix
// itself, for keys c·d … c·d + d − 1. With ⌈T/d⌉ clones per layer the whole
// matrix is recovered, with everything the model's own attention does (the
// causal mask, sliding windows, softcapping, attention bias, rotary inside the
// node) and nothing reimplemented. The clones share the query and key inputs
// and write their present_key/present_value to unused tensors.
//
// Explicit attention (Gemma 4's KV-shared layers compute
// MatMul → Where → Softmax → MatMul): the Softmax output [1, heads, S, T] is
// read directly.
//
// Either way the [S, heads, T] weights are cut down on the GPU to the top K
// keys per query and head (TopK), so the readback stays small. Every weight
// above 1/K is guaranteed to be kept, since at most K keys can exceed it.
//
// The `logits` output is also cut down to the last position, as in
// logitlens-web: generation only reads that one, and the analysis pass does
// not read logits at all.
//
// The graph keeps working for generation with a KV cache. The clones then see
// the real past values next to one-hot current values, so the attention
// outputs of those steps are meaningless, and they are ignored.

export const ATTN_VALS = 'attn_vals.';
export const ATTN_IDX = 'attn_idx.';

const te = new TextEncoder();
const td = new TextDecoder();

type Bytes = Uint8Array;
type Chunk = Bytes | number[];

// ── Protobuf wire format ─────────────────────────────────────────────

interface Field {
  field: number;
  wire: number;
  tagStart: number;
  valStart: number;
  end: number;
  value: number;
}

function readVarint(b: Bytes, p: number): [number, number] {
  let v = 0, m = 1, x: number;
  do {
    x = b[p++];
    v += (x & 0x7f) * m;
    m *= 128;
  } while (x & 0x80);
  return [v, p];
}

function* fields(b: Bytes, start: number, end: number): Generator<Field> {
  let p = start;
  while (p < end) {
    const tagStart = p;
    let tag: number;
    [tag, p] = readVarint(b, p);
    const field = Math.floor(tag / 8), wire = tag % 8;
    let value = 0, valStart = p;
    if (wire === 0) [value, p] = readVarint(b, p);
    else if (wire === 1) p += 8;
    else if (wire === 5) p += 4;
    else if (wire === 2) {
      let len: number;
      [len, p] = readVarint(b, p);
      valStart = p;
      p += len;
    } else throw new Error(`Unsupported protobuf wire type ${wire}`);
    yield { field, wire, tagStart, valStart, end: p, value };
  }
}

const str = (b: Bytes, f: Field) => td.decode(b.subarray(f.valStart, f.end));

function varint(v: number | bigint): number[] {
  const out: number[] = [];
  if (typeof v === 'bigint' || v < 0) {
    let x = BigInt.asUintN(64, BigInt(v));
    while (x >= 0x80n) { out.push(Number(x & 0x7fn) | 0x80); x >>= 7n; }
    out.push(Number(x));
    return out;
  }
  while (v >= 128) { out.push((v % 128) | 0x80); v = Math.floor(v / 128); }
  out.push(v);
  return out;
}

function cat(chunks: Chunk[]): Bytes {
  let n = 0;
  for (const c of chunks) n += c.length;
  const out = new Uint8Array(n);
  let p = 0;
  for (const c of chunks) { out.set(c, p); p += c.length; }
  return out;
}

const fVarint = (field: number, v: number | bigint) => cat([varint(field * 8), varint(v)]);
const fBytes = (field: number, bytes: Bytes) => cat([varint(field * 8 + 2), varint(bytes.length), bytes]);
const fStr = (field: number, s: string) => fBytes(field, te.encode(s));

// ── ONNX messages ────────────────────────────────────────────────────

interface OnnxNode {
  name: string;
  op_type: string;
  domain?: string;
  input: string[];
  output: string[];
  attrs?: Bytes[];
}

function parseNode(b: Bytes, start: number, end: number): Required<OnnxNode> {
  const n: Required<OnnxNode> = { input: [], output: [], name: '', op_type: '', domain: '', attrs: [] };
  for (const f of fields(b, start, end)) {
    if (f.field === 1) n.input.push(str(b, f));
    else if (f.field === 2) n.output.push(str(b, f));
    else if (f.field === 3) n.name = str(b, f);
    else if (f.field === 4) n.op_type = str(b, f);
    else if (f.field === 7) n.domain = str(b, f);
    else if (f.field === 5) n.attrs.push(b.subarray(f.valStart, f.end));
  }
  return n;
}

function encodeNode({ name, op_type, domain = '', input, output, attrs = [] }: OnnxNode): Bytes {
  return cat([
    ...input.map(s => fStr(1, s)),
    ...output.map(s => fStr(2, s)),
    fStr(3, name),
    fStr(4, op_type),
    ...attrs.map(a => fBytes(5, a)),
    ...(domain ? [fStr(7, domain)] : []),
  ]);
}

// The integer value of attribute `name` (AttributeProto: name=1, i=3), or null.
function intAttr(node: Required<OnnxNode>, name: string): number | null {
  for (const a of node.attrs) {
    let n = '', v: number | null = null;
    for (const f of fields(a, 0, a.length)) {
      if (f.field === 1) n = str(a, f);
      else if (f.field === 3) v = f.value;
    }
    if (n === name) return v;
  }
  return null;
}

// AttributeProto: name=1, i=3, ints=8, type=20 (INT=2, INTS=7)
const attrInt = (name: string, v: number) => cat([fStr(1, name), fVarint(3, v), fVarint(20, 2)]);
const attrInts = (name: string, vs: number[]) => cat([fStr(1, name), ...vs.map(v => fVarint(8, v)), fVarint(20, 7)]);

// TensorProto: dims=1, data_type=2, name=8, raw_data=9
const FLOAT = 1, INT32 = 6, FLOAT16 = 10;

function int64Tensor(name: string, values: (number | bigint)[]): Bytes {
  const raw = new Uint8Array(new BigInt64Array(values.map(v => BigInt(v))).buffer);
  return cat([fVarint(1, values.length), fVarint(2, 7), fStr(8, name), fBytes(9, raw)]);
}

function rawTensor(name: string, dims: number[], dataType: number, raw: Bytes): Bytes {
  return cat([...dims.map(d => fVarint(1, d)), fVarint(2, dataType), fStr(8, name), fBytes(9, raw)]);
}

// Identity matrix [n, n] as little-endian float16 (1.0 = 0x3c00) or float32.
function identity(n: number, dataType: number): Bytes {
  const size = dataType === FLOAT16 ? 2 : 4;
  const out = new Uint8Array(n * n * size);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < n; i++) {
    if (dataType === FLOAT16) dv.setUint16((i * n + i) * 2, 0x3c00, true);
    else dv.setFloat32((i * n + i) * 4, 1, true);
  }
  return out;
}

// ValueInfoProto: name=1, type=2 → TypeProto.tensor_type=1 → elem_type=1, shape=2
// TensorShapeProto.dim=1 → Dimension: dim_value=1 | dim_param=2
const dim = (d: number | string) => fBytes(1, typeof d === 'string' ? fStr(2, d) : fVarint(1, d));
const valueInfo = (name: string, elemType: number, shape: (number | string)[]) =>
  cat([fStr(1, name), fBytes(2, fBytes(1, cat([fVarint(1, elemType), fBytes(2, cat(shape.map(dim)))])))]);

const layerOf = (n: OnnxNode) => {
  const m = /^\/model\/layers\.(\d+)\//.exec(n.name);
  return m ? +m[1] : -1;
};

// ── Patch ────────────────────────────────────────────────────────────

/** One attention layer with captured weights. */
export interface AttnSite {
  layer: number;
  kind: 'gqa' | 'softmax';
  heads: number | null; // null: only known from the output's shape
}

export interface AttnPatchInfo {
  bytes: Uint8Array;
  numLayers: number;
  sites: AttnSite[];
  topK: number;
  maxTokens: number;
  skipped: string[]; // attention nodes that could not be captured, and why
}

export interface AttnOptions {
  topK?: number;
  /** Longest sequence whose full attention is recovered; a multiple of every head dim. */
  maxTokens?: number;
}

/**
 * Adds attention outputs to a serialized ONNX decoder. Returns the patched
 * bytes and the captured layers, or null if the graph has no `logits` output
 * (an embedding or vision graph) or is already patched.
 */
export function addAttentionOutputs(
  bytes: Uint8Array,
  { topK = 32, maxTokens = 1024 }: AttnOptions = {},
): AttnPatchInfo | null {
  const b = bytes;
  let graph: Field | null = null;
  let opset = 0; // of the default domain
  for (const f of fields(b, 0, b.length)) {
    if (f.field === 7 && f.wire === 2) graph = f;
    else if (f.field === 8 && f.wire === 2) {
      // OperatorSetIdProto: domain=1, version=2
      let domain = '', version = 0;
      for (const g of fields(b, f.valStart, f.end)) {
        if (g.field === 1) domain = str(b, g);
        else if (g.field === 2) version = g.value;
      }
      if (domain === '' || domain === 'ai.onnx') opset = Math.max(opset, version);
    }
  }
  if (!graph) return null;

  const nodes: Required<OnnxNode>[] = [];
  const nodeRanges: Field[] = [];
  const graphOutputs: string[] = [];
  for (const f of fields(b, graph.valStart, graph.end)) {
    if (f.field === 1) {
      nodes.push(parseNode(b, f.valStart, f.end));
      nodeRanges.push(f);
    } else if (f.field === 12) {
      for (const g of fields(b, f.valStart, f.end)) if (g.field === 1) graphOutputs.push(str(b, g));
    }
  }
  if (!graphOutputs.includes('logits') || graphOutputs.some(o => o.startsWith(ATTN_VALS))) return null;

  const producer = new Map<string, Required<OnnxNode>>();
  for (const n of nodes) for (const o of n.output) producer.set(o, n);
  const numLayers = nodes.reduce((m, n) => Math.max(m, layerOf(n) + 1), 0);


  const add: Bytes[] = [];
  const inits: Bytes[] = [];
  const outs: Bytes[] = [];
  const sites: AttnSite[] = [];
  const skipped: string[] = [];
  const initNames = new Set<string>();
  const P = '/attn/';
  const init = (name: string, make: () => Bytes) => {
    if (!initNames.has(name)) { initNames.add(name); inits.push(make()); }
    return name;
  };
  const i64 = (name: string, values: (number | bigint)[]) => init(P + name, () => int64Tensor(P + name, values));

  // Top-K of x [1, S, heads, T] along the keys, as the layer's two outputs.
  // `pad`: T may be smaller than K (short prompts), so K zero columns are added.
  function tail(q: string, x: string, l: number, heads: number | string, pad: boolean) {
    add.push(encodeNode({ name: q + 'Cast32', op_type: 'Cast', input: [x], output: [q + 'f'], attrs: [attrInt('to', FLOAT)] }));
    let scores = q + 'f';
    if (pad) {
      add.push(
        encodeNode({ name: q + 'Shape', op_type: 'Shape', input: [q + 'f'], output: [q + 'shape'] }),
        encodeNode({
          name: q + 'Lead', op_type: 'Slice', input: [q + 'shape', i64('zero', [0]), i64('three', [3]), i64('zero', [0])],
          output: [q + 'lead'],
        }),
        encodeNode({
          name: q + 'PadShape', op_type: 'Concat', input: [q + 'lead', i64('k', [topK])], output: [q + 'padshape'],
          attrs: [attrInt('axis', 0)],
        }),
        encodeNode({
          name: q + 'Zeros', op_type: 'Expand',
          input: [init(P + 'zero_f', () => rawTensor(P + 'zero_f', [], FLOAT, new Uint8Array(4))), q + 'padshape'],
          output: [q + 'zeros'],
        }),
        encodeNode({
          name: q + 'Pad', op_type: 'Concat', input: [q + 'f', q + 'zeros'], output: [q + 'padded'], attrs: [attrInt('axis', 3)],
        }),
      );
      scores = q + 'padded';
    }
    const vals = ATTN_VALS + l, idx = ATTN_IDX + l;
    add.push(
      encodeNode({
        name: q + 'TopK', op_type: 'TopK', input: [scores, i64('k', [topK])], output: [q + 'vals', q + 'idx'],
        attrs: [attrInt('axis', -1), attrInt('largest', 1), attrInt('sorted', 1)],
      }),
      encodeNode({ name: q + 'Vals16', op_type: 'Cast', input: [q + 'vals'], output: [vals], attrs: [attrInt('to', FLOAT16)] }),
      encodeNode({ name: q + 'Idx32', op_type: 'Cast', input: [q + 'idx'], output: [idx], attrs: [attrInt('to', INT32)] }),
    );
    outs.push(
      valueInfo(vals, FLOAT16, [1, 'attn_positions', heads, topK]),
      valueInfo(idx, INT32, [1, 'attn_positions', heads, topK]),
    );
  }

  // Head dim of a GQA node: the value projection's width over the KV heads.
  // The value input may pass through a norm or reshape after the projection.
  function headDim(gqa: Required<OnnxNode>, kvHeads: number): number | null {
    let t = gqa.input[2];
    for (let i = 0; i < 8; i++) {
      const p = producer.get(t);
      if (!p) return null;
      if (p.op_type === 'MatMulNBits') {
        const n = intAttr(p, 'N');
        return n && n % kvHeads === 0 ? n / kvHeads : null;
      }
      t = p.input[0];
    }
    return null;
  }

  const done = new Set<number>();
  for (const n of nodes) {
    if (n.op_type !== 'GroupQueryAttention') continue;
    const l = layerOf(n);
    const heads = intAttr(n, 'num_heads'), kv = intAttr(n, 'kv_num_heads');
    if (l < 0 || done.has(l)) continue;
    if (!heads || !kv || !n.input[2]) {
      skipped.push(`${n.name}: packed QKV or missing head counts`);
      continue;
    }
    const d = headDim(n, kv);
    if (!d || maxTokens % d !== 0) {
      skipped.push(`${n.name}: head dim ${d ?? 'unknown'}`);
      continue;
    }
    done.add(l);
    const q = `${P}L${l}/`;
    const eye = init(P + 'eye', () => rawTensor(P + 'eye', [maxTokens, maxTokens], FLOAT16, identity(maxTokens, FLOAT16)));

    // rows = eye[:S], S = the value input's sequence length, in the value's
    // dtype. q4f16 graphs run attention in float16 (even where logits are
    // float32); CastLike makes sure where the opset has it (15+).
    add.push(
      encodeNode({ name: q + 'VShape', op_type: 'Shape', input: [n.input[2]], output: [q + 'vshape'] }),
      encodeNode({
        name: q + 'S', op_type: 'Slice', input: [q + 'vshape', i64('one', [1]), i64('two', [2]), i64('zero', [0])],
        output: [q + 'S'],
      }),
      encodeNode({
        name: q + 'Rows', op_type: 'Slice', input: [eye, i64('zero', [0]), q + 'S', i64('zero', [0])],
        output: [opset >= 15 ? q + 'rows16' : q + 'rows'],
      }),
    );
    if (opset >= 15) {
      add.push(encodeNode({ name: q + 'RowsLike', op_type: 'CastLike', input: [q + 'rows16', n.input[2]], output: [q + 'rows'] }));
    }
    const blocks = maxTokens / d;
    const parts: string[] = [];
    for (let c = 0; c < blocks; c++) {
      const v = `${q}v${c}`, o = `${q}o${c}`;
      add.push(
        encodeNode({
          name: `${q}Block${c}`, op_type: 'Slice',
          input: [q + 'rows', i64(`b${c * d}`, [c * d]), i64(`b${(c + 1) * d}`, [(c + 1) * d]), i64('one', [1])],
          output: [`${q}blk${c}`],
        }),
        encodeNode({ name: `${q}Tile${c}`, op_type: 'Tile', input: [`${q}blk${c}`, i64(`tile${kv}`, [1, kv])], output: [`${q}tile${c}`] }),
        encodeNode({
          name: `${q}V${c}`, op_type: 'Reshape', input: [`${q}tile${c}`, i64(`vdims${kv * d}`, [1, -1, kv * d])], output: [v],
        }),
        encodeNode({
          ...n, name: `${q}GQA${c}`,
          input: n.input.map((x, i) => (i === 2 ? v : x)),
          output: n.output.map((_, i) => (i === 0 ? o : `${q}present${c}_${i}`)),
        }),
        encodeNode({
          name: `${q}Heads${c}`, op_type: 'Reshape', input: [o, i64(`hdims${heads}x${d}`, [1, -1, heads, d])],
          output: [`${q}h${c}`],
        }),
      );
      parts.push(`${q}h${c}`);
    }
    let probs = parts[0];
    if (parts.length > 1) {
      probs = q + 'probs';
      add.push(encodeNode({ name: q + 'Keys', op_type: 'Concat', input: parts, output: [probs], attrs: [attrInt('axis', 3)] }));
    }
    tail(q, probs, l, heads, false);
    sites.push({ layer: l, kind: 'gqa', heads });
  }

  for (const n of nodes) {
    if (n.op_type !== 'Softmax' || !/\/attn\/.*softmax/i.test(n.name)) continue;
    const l = layerOf(n);
    if (l < 0 || done.has(l)) continue;
    done.add(l);
    const q = `${P}L${l}/`;
    add.push(encodeNode({
      name: q + 'ByQuery', op_type: 'Transpose', input: [n.output[0]], output: [q + 'byq'], attrs: [attrInts('perm', [0, 2, 1, 3])],
    }));
    tail(q, q + 'byq', l, `attn_heads_${l}`, true);
    sites.push({ layer: l, kind: 'softmax', heads: null });
  }
  if (!sites.length) return null;
  sites.sort((a, c) => a.layer - c.layer);

  // Rename the output of the node producing `logits` and re-create `logits`
  // as a slice of its last position.
  const logitsIdx = nodes.findIndex(n => n.output.includes('logits'));
  const logitsNode = nodes[logitsIdx];
  const fullLogits = P + 'logits_full';
  const renamed = fBytes(1, encodeNode({
    ...logitsNode,
    output: logitsNode.output.map(o => (o === 'logits' ? fullLogits : o)),
  }));
  add.push(encodeNode({
    name: P + 'LastLogits', op_type: 'Slice',
    input: [fullLogits, i64('last', [-1]), i64('ends', [2n ** 63n - 1n]), i64('one', [1])], output: ['logits'],
  }));

  // Repeated fields may appear in any order, so new nodes, initializers and
  // outputs go to the end of the GraphProto; the new nodes are in dependency order.
  const extra = cat([
    ...add.map(x => fBytes(1, x)),
    ...inits.map(t => fBytes(5, t)),
    ...outs.map(o => fBytes(12, o)),
  ]);
  const range = nodeRanges[logitsIdx];
  const graphParts = [
    b.subarray(graph.valStart, range.tagStart),
    renamed,
    b.subarray(range.end, graph.end),
    extra,
  ];
  const graphLen = graphParts.reduce((s, part) => s + part.length, 0);
  const header = cat([varint(7 * 8 + 2), varint(graphLen)]);
  const bytesOut = cat([b.subarray(0, graph.tagStart), header, ...graphParts, b.subarray(graph.end)]);
  return { bytes: bytesOut, numLayers, sites, topK, maxTokens, skipped };
}

// ── transformers.js integration ──────────────────────────────────────

/** The parts of transformers.js' `env` the hooks touch. */
export interface HookEnv {
  fetch: (input: string | URL, init?: any) => Promise<any>;
  useBrowserCache: boolean;
  useFSCache: boolean;
  useCustomCache: boolean;
  customCache: unknown;
}

/** The subset of the Web Cache API the hooks need. */
export interface BaseCache {
  match(key: string): Promise<Response | undefined>;
  put(key: string, resp: Response): Promise<void>;
}

export interface HookOptions {
  baseCache: BaseCache;
  attn: () => AttnOptions; // read for every graph, so it can change between loads
  onProgress?: (url: string, loaded: number, total: number) => void;
  onPatched?: (url: string, info: AttnPatchInfo) => void;
}

const isGraphFile = (url: string) => /\/onnx\/[^/]+\.onnx$/.test(new URL(url).pathname);

/**
 * Hooks transformers.js so every downloaded `onnx/*.onnx` graph gets the
 * attention outputs. The unmodified file is stored in `baseCache` (normally
 * the shared `transformers-cache`), so models downloaded by the logit lens or
 * the token probability explorer are reused, and the other way round.
 */
export function installAttnHooks(env: HookEnv, { baseCache, attn, onProgress, onPatched }: HookOptions) {
  const origFetch = env.fetch;
  env.useBrowserCache = false;
  env.useFSCache = false;
  env.useCustomCache = true;
  env.customCache = {
    match: async (key: string) => (isGraphFile(key) ? undefined : baseCache.match(key)),
    put: async (key: string, resp: Response) => (isGraphFile(key) ? undefined : baseCache.put(key, resp)),
  };
  env.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : (input as Request).url;
    if (!isGraphFile(url) || new Headers(init?.headers).has('Range')) return origFetch(input, init);

    let bytes: Uint8Array;
    const cached = await baseCache.match(url);
    if (cached) {
      bytes = new Uint8Array(await cached.arrayBuffer());
    } else {
      const resp: Response = await origFetch(input, init);
      if (!resp.ok) return resp;
      bytes = await readBody(resp, onProgress && ((loaded, total) => onProgress(url, loaded, total)));
      // Caching is best effort: a full storage quota only means downloading again next time.
      await baseCache.put(url, new Response(bytes as BodyInit, {
        headers: { 'content-type': 'application/octet-stream', 'content-length': String(bytes.length) },
      })).catch(err => console.warn(`Could not cache ${url}:`, err));
    }
    const patched = addAttentionOutputs(bytes, attn());
    if (patched) onPatched?.(url, patched);
    const body = patched ? patched.bytes : bytes;
    return new Response(body as BodyInit, { status: 200, headers: { 'content-length': String(body.length) } });
  };
}

/** Reads a response body, reporting progress as chunks arrive. */
async function readBody(resp: Response, onProgress?: (loaded: number, total: number) => void): Promise<Uint8Array> {
  const total = parseInt(resp.headers.get('content-length') ?? '0', 10);
  if (!resp.body || !onProgress) return new Uint8Array(await resp.arrayBuffer());
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  return cat(chunks);
}
