const HEADER_SIZE = 128;
const MODEL_BASE = 128;
const WASM_PAGE = 65536;
const MAX_MEMORY = 2 ** 31;
const MAX_PROMPT_BYTES = 8192;
const FIB_MULT = 0x9E3779B97F4A7C15n;
const NO_MERGE = 0xFFFFFFFF;
const CHAT_TURN_END = [151645, 198];

const align64 = (n) => Math.ceil(n / 64) * 64;

function requireModel(ok, detail) {
  if (!ok) throw new Error("无效的 ASMBIN 模型（" + detail + "）");
}

function fits(start, length, lower, upper) {
  return start >= lower && start <= upper && length <= upper - start;
}

function readHeader(header) {
  const view = new DataView(header.buffer);
  const u32 = (offset) => view.getUint32(offset, true);
  const f32 = (offset) => view.getFloat32(offset, true);
  const u64 = (offset) => Number(view.getBigUint64(offset, true));
  const version = u32(4);
  const flags = u32(60);
  requireModel(u32(0) === 0x4C4D5341 && (version === 1 || version === 2), "文件标记或版本");
  requireModel((flags & 3) === 3, "需要共享词嵌入和输出权重的 Qwen3 QK-Norm 模型");

  const dim = u32(8);
  const hiddenDim = u32(12);
  const layerCount = u32(16);
  const heads = u32(20);
  const kvHeads = u32(24);
  const headDim = u32(28);
  const vocabSize = u32(32);
  const maxContext = u32(36);
  const eps = f32(40);
  const ropeTheta = f32(44);
  const eos = [u32(48), u32(52)];
  const hashSlots = u32(56) + 1;
  const scaleBytes = version === 2 || (flags & 4) ? 2 : 4;
  const tokenizer = u64(64);
  const embedding = u64(72);
  const layers = u64(80);
  const layerStride = u64(88);
  const finalNorm = u64(96);
  const fileSize = u64(104);
  const qDim = heads * headDim;
  const kvDim = kvHeads * headDim;

  requireModel(fileSize >= HEADER_SIZE && fileSize <= MAX_MEMORY - MODEL_BASE, "文件大小");
  requireModel([dim, hiddenDim, layerCount, kvHeads, headDim, qDim, vocabSize, maxContext]
    .every((n) => n > 0 && n <= fileSize), "模型维度");
  requireModel(dim % 32 === 0 && hiddenDim % 32 === 0 && qDim % 32 === 0 &&
    heads % kvHeads === 0 && headDim % 4 === 0, "矩阵或 GQA 维度");
  requireModel(eos.every((id) => id < vocabSize) &&
    Number.isFinite(eps) && eps > 0 && Number.isFinite(ropeTheta) && ropeTheta > 0, "结束标记或归一化参数");

  const matrixBytes = (rows, cols) => rows * (cols + cols / 32 * scaleBytes);
  requireModel([tokenizer, embedding, layers, layerStride, finalNorm].every((n) => n % 64 === 0) &&
    fits(tokenizer, 64, HEADER_SIZE, embedding) &&
    fits(embedding, matrixBytes(vocabSize, dim), tokenizer, layers) &&
    fits(layers, layerStride * layerCount, embedding, finalNorm) &&
    fits(finalNorm, dim * 4, layers, fileSize), "权重区段");

  // 层内顺序与 scripts/export_weights.py 写入的顺序一致。
  const layerOffsets = [0, dim * 4, (dim + headDim) * 4, (dim + 2 * headDim) * 4];
  let layerEnd = (2 * dim + 2 * headDim) * 4;
  for (const [rows, cols] of [
    [qDim, dim], [kvDim, dim], [kvDim, dim], [dim, qDim],
    [hiddenDim, dim], [hiddenDim, dim], [dim, hiddenDim],
  ]) {
    const offset = align64(layerEnd);
    layerOffsets.push(offset);
    layerEnd = offset + matrixBytes(rows, cols);
  }
  requireModel(layerEnd <= layerStride, "层内权重大小");

  return {
    dim, hiddenDim, layerCount, heads, kvHeads, headDim, vocabSize, maxContext, eps,
    ropeTheta, eos, hashSlots, scaleBytes, tokenizer, embedding, layers, layerStride,
    finalNorm, fileSize, qDim, kvDim, layerOffsets,
  };
}

function planMemory(model, contextLength) {
  const {
    dim, hiddenDim, layerCount, heads, kvHeads, headDim, vocabSize, scaleBytes,
    embedding, finalNorm, fileSize, qDim, kvDim, layerOffsets,
  } = model;
  let end = MODEL_BASE + fileSize;
  function allocate(bytes) {
    const ptr = align64(end);
    end = ptr + bytes;
    return ptr;
  }

  const layerTable = allocate(layerCount * layerOffsets.length * 4);
  const x = allocate(dim * 4);
  const h = allocate(dim * 4);
  const q = allocate(qDim * 4);
  const k = allocate(kvDim * 4);
  const v = allocate(kvDim * 4);
  const attention = allocate(qDim * 4);
  const scores = allocate(contextLength * 4);
  const gate = allocate(hiddenDim * 4);
  const up = allocate(hiddenDim * 4);
  const logits = allocate(vocabSize * 4);
  const frequencies = allocate(headDim / 2 * 8);
  const cosines = allocate(headDim / 2 * 4);
  const sines = allocate(headDim / 2 * 4);
  const kvBytes = layerCount * contextLength * kvDim * 4;
  const keys = allocate(kvBytes);
  const values = allocate(kvBytes);

  // 与 qwen3.wat 的 forward 入口使用相同的字段顺序。
  const state = [
    dim, hiddenDim, layerCount, heads, kvHeads, headDim, vocabSize, contextLength,
    0, scaleBytes, MODEL_BASE + embedding, layerTable, MODEL_BASE + finalNorm,
    x, h, q, k, v, attention, scores, gate, up, logits, frequencies, cosines, sines, keys, values,
  ];
  if (end > MAX_MEMORY) throw new RangeError("权重和 KV 缓存超出此 Wasm 模块的 2 GiB 上限");
  return { state, end, layerTable, frequencies, logits };
}

async function loadWeights(memory, source, requestedContext) {
  if (source instanceof ArrayBuffer) source = new Uint8Array(source);
  if (source instanceof Uint8Array) source = [source];

  const header = new Uint8Array(HEADER_SIZE);
  let written = 0;
  let model, contextLength, layout, bytes;
  for await (let chunk of source) {
    if (!(chunk instanceof Uint8Array)) throw new TypeError("模型数据块必须为 Uint8Array");
    if (!model) {
      const count = Math.min(chunk.length, HEADER_SIZE - written);
      header.set(chunk.subarray(0, count), written);
      written += count;
      if (written < HEADER_SIZE) continue;
      model = readHeader(header);
      contextLength = requestedContext ?? Math.min(512, model.maxContext);
      if (!Number.isInteger(contextLength) || contextLength < 1 || contextLength > model.maxContext) {
        throw new RangeError("上下文长度必须为 1 到 " + model.maxContext + " 的整数");
      }
      layout = planMemory(model, contextLength);
      memory.grow(Math.ceil(layout.end / WASM_PAGE) - memory.buffer.byteLength / WASM_PAGE);
      bytes = new Uint8Array(memory.buffer, MODEL_BASE, model.fileSize);
      bytes.set(header);
      chunk = chunk.subarray(count);
    }
    requireModel(chunk.length <= model.fileSize - written, "数据超过文件头声明的大小");
    bytes.set(chunk, written);
    written += chunk.length;
  }
  requireModel(model && written === model.fileSize, "文件不完整");
  return { model, contextLength, layout, bytes };
}

function createTokenizer(bytes, model) {
  const { tokenizer, embedding, hashSlots, vocabSize } = model;
  const view = new DataView(bytes.buffer, bytes.byteOffset, embedding);
  const sectionOffset = (offset) => tokenizer + Number(view.getBigUint64(tokenizer + offset, true));
  const b2t = sectionOffset(0);
  const offsets = sectionOffset(8);
  const pieces = sectionOffset(16);
  const table = sectionOffset(24);
  requireModel((hashSlots & (hashSlots - 1)) === 0 &&
    fits(b2t, 256 * 4, tokenizer + 64, offsets) &&
    fits(offsets, (vocabSize + 1) * 4, b2t, pieces) &&
    fits(pieces, 0, offsets, table) &&
    fits(table, hashSlots * 16, pieces, embedding), "分词器区段");

  const offsetOf = (id) => view.getUint32(offsets + id * 4, true);
  let prev = 0;
  for (let id = 0; id <= vocabSize; id++) {
    const offset = offsetOf(id);
    requireModel(offset >= prev && offset <= table - pieces, "词元字节偏移");
    prev = offset;
  }
  const byteTokens = Array.from({ length: 256 }, (_, b) => view.getUint32(b2t + b * 4, true));
  requireModel(byteTokens.every((id) => id < vocabSize), "单字节词元 ID");

  const mask = hashSlots - 1;
  const shift = BigInt(64 - Math.log2(hashSlots));
  const encoder = new TextEncoder();

  function encode(text, { raw = false, think = false } = {}) {
    if (typeof text !== "string") throw new TypeError("提示词必须为字符串");
    if (text.length > MAX_PROMPT_BYTES) throw new RangeError("提示词最多支持 8192 个 UTF-8 字节");
    const input = encoder.encode(text);
    if (input.length > MAX_PROMPT_BYTES) throw new RangeError("提示词最多支持 8192 个 UTF-8 字节");
    const ids = Array.from(input, (byte) => byteTokens[byte]);
    // 使用与原版相同的字节级 BPE，同排名时先合并左边的词元对。
    while (ids.length > 1) {
      let bestRank = NO_MERGE;
      let bestIndex = -1;
      let merged = 0;
      for (let index = 0; index < ids.length - 1; index++) {
        const key = (BigInt(ids[index]) << 32n) | BigInt(ids[index + 1]);
        let slot = Number(BigInt.asUintN(64, key * FIB_MULT) >> shift);
        for (let probes = 0; probes < hashSlots; probes++) {
          const entry = table + slot * 16;
          const rank = view.getUint32(entry + 12, true);
          if (rank === NO_MERGE) break;
          if (view.getBigUint64(entry, true) === key) {
            if (rank < bestRank) {
              bestRank = rank;
              bestIndex = index;
              merged = view.getUint32(entry + 8, true);
            }
            break;
          }
          slot = (slot + 1) & mask;
        }
      }
      if (bestIndex === -1) break;
      requireModel(merged < vocabSize, "BPE 合并词元 ID");
      ids.splice(bestIndex, 2, merged);
    }
    if (raw) return ids;
    requireModel(vocabSize > 151668, "缺少 Qwen3 对话模板词元");
    const prompt = [151644, 872, 198, ...ids, ...CHAT_TURN_END, 151644, 77091, 198];
    if (!think) prompt.push(151667, 271, 151668, 271);
    return prompt;
  }

  function createDecoder() {
    const decoder = new TextDecoder("utf-8", { ignoreBOM: true });
    return {
      push(id) {
        return decoder.decode(bytes.subarray(pieces + offsetOf(id), pieces + offsetOf(id + 1)), { stream: true });
      },
      finish() {
        return decoder.decode();
      },
    };
  }

  return { encode, createDecoder };
}

export async function loadQwen3(wasmBytes, modelSource, { contextLength: requestedContext } = {}) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, { math: Math });
  const { memory, forward } = instance.exports;
  const { model, contextLength, layout, bytes } = await loadWeights(memory, modelSource, requestedContext);
  const { encode, createDecoder } = createTokenizer(bytes, model);
  const config = new DataView(memory.buffer);
  layout.state.forEach((value, i) => config.setUint32(i * 4, value, true));
  config.setFloat32(32, model.eps, true);
  let entry = layout.layerTable;
  for (let i = 0; i < model.layerCount; i++) {
    const base = MODEL_BASE + model.layers + i * model.layerStride;
    for (const offset of model.layerOffsets) {
      config.setUint32(entry, base + offset, true);
      entry += 4;
    }
  }
  for (let i = 0; i < model.headDim / 2; i++) {
    config.setFloat64(layout.frequencies + i * 8, model.ropeTheta ** (-2 * i / model.headDim), true);
  }
  const values = new Float32Array(memory.buffer, layout.logits, model.vocabSize);
  const cachedIds = [];
  let generating = false;

  function checkedIds(ids) {
    if (!Array.isArray(ids) && !(ids instanceof Uint32Array)) throw new TypeError("需要词元 ID 数组");
    if (ids.length === 0 || ids.length > contextLength) {
      throw new RangeError("提示词元数量必须为 1 到 " + contextLength);
    }
    for (const id of ids) {
      if (!Number.isInteger(id) || id < 0 || id >= model.vocabSize) throw new RangeError("词元 ID 越界");
    }
    return Array.from(ids);
  }

  function* prefill(ids) {
    const last = ids.length - 1;
    let pos = 0;
    // 复用匹配的 KV 前缀，最后一项重新计算以取得这一轮的 logits。
    while (pos < last && pos < cachedIds.length && cachedIds[pos] === ids[pos]) pos++;
    cachedIds.length = pos;
    for (; pos <= last; pos++) {
      forward(ids[pos], pos, Number(pos === last));
      cachedIds.push(ids[pos]);
      yield pos + 1;
    }
  }

  function logits(ids) {
    if (generating) throw new Error("模型正在生成回复");
    for (const _ of prefill(checkedIds(ids))) {
      // 消耗全部预填充步骤。
    }
    return values.slice();
  }

  function* generateSteps(ids, maxNewTokens = 32) {
    if (generating) throw new Error("模型正在生成回复");
    if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1) {
      throw new RangeError("生成词元数量必须为正整数");
    }
    const prompt = checkedIds(ids);
    if (prompt.length >= contextLength) throw new RangeError("上下文没有可用于生成的空间");
    const limit = Math.min(maxNewTokens, contextLength - prompt.length);
    generating = true;
    try {
      for (const position of prefill(prompt)) yield { prefill: position };
      const decoder = createDecoder();
      for (let step = 0; step < limit; step++) {
        let next = 0;
        for (let i = 1; i < values.length; i++) {
          if (values[i] > values[next]) next = i;
        }
        if (!Number.isFinite(values[next])) throw new Error("推理得到非有限的 logits");
        if (model.eos.includes(next)) break;
        yield { tokenId: next, text: decoder.push(next) };
        if (step + 1 < limit) {
          forward(next, prompt.length + step, 1);
          cachedIds.push(next);
        }
      }
      const tail = decoder.finish();
      if (tail) yield { text: tail };
    } finally {
      generating = false;
    }
  }

  function generate(ids, maxNewTokens) {
    const tokenIds = [];
    let text = "";
    for (const step of generateSteps(ids, maxNewTokens)) {
      if (step.tokenId !== undefined) tokenIds.push(step.tokenId);
      if (step.text) text += step.text;
    }
    return { tokenIds, text };
  }

  return { contextLength, encode, logits, generate, generateSteps };
}
