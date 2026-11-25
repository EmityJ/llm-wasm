#!/usr/bin/env python3
"""
Export Qwen3-0.6B (BF16 safetensors + tokenizer.json) into a compact,
64-byte aligned, directly mmap-able .asmbin file for pure x86-64 asm-llm.

Weight format: Per-row split Q8_0 (block size = 32, symmetric int8 in [-127, 127]):
  Each row of K elements (K multiple of 32) is stored as:
    [K bytes of int8 quantized weights] [K/32 * 4 bytes of fp32 block scales]
  For Qwen3-0.6B (K in {1024, 2048, 3072}), every row stride is an exact
  multiple of 64 bytes (1152, 2304, 3456), ensuring 32-byte aligned AVX2 loads.
"""

import argparse
import json
import os
import struct
import time
import numpy as np

MAGIC = b"ASML"
VERSION = 1
HEADER_SIZE = 128
TOK_HEADER_SIZE = 64
HT_BITS = 19
HT_SIZE = 1 << HT_BITS  # 524,288 slots for 151,387 merges (load factor ~0.289)
HT_MASK = HT_SIZE - 1
FIB_MULT = 0x9E3779B97F4A7C15


def align64(n: int) -> int:
    return (n + 63) & ~63


def pad64(f) -> int:
    pos = f.tell()
    aligned = align64(pos)
    if aligned > pos:
        f.write(b"\x00" * (aligned - pos))
    return aligned


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_safetensors_header(path: str):
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    data_offset = 8 + hdr_len
    return hdr, data_offset


def read_tensor_f32(f, hdr, data_offset: int, name: str) -> np.ndarray:
    meta = hdr[name]
    dtype = meta["dtype"]
    shape = tuple(meta["shape"])
    start, end = meta["data_offsets"]
    f.seek(data_offset + start)
    raw = f.read(end - start)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape)
    elif dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    elif dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    else:
        raise ValueError(f"Unsupported dtype {dtype} for tensor {name}")


def quantize_matrix_q8_0_row_split(w: np.ndarray) -> bytes:
    """Quantize [M, K] float32 matrix into per-row split Q8_0 bytes."""
    assert w.ndim == 2
    M, K = w.shape
    assert K % 32 == 0, f"K={K} must be a multiple of 32"
    nb = K // 32
    blocks = w.reshape(M, nb, 32).astype(np.float32, copy=False)
    amax = np.max(np.abs(blocks), axis=2)  # [M, nb]
    scale = np.where(amax > 0.0, amax / np.float32(127.0), np.float32(0.0)).astype(np.float32)
    inv_scale = np.where(scale > 0.0, np.float32(1.0) / scale, np.float32(0.0)).astype(np.float32)
    q = np.clip(np.rint(blocks * inv_scale[:, :, None]), -127, 127).astype(np.int8)
    q_flat = q.reshape(M, K)  # [M, K] int8
    s_bytes = scale.view(np.uint8).reshape(M, nb * 4)  # [M, nb*4] uint8
    # Pack each row contiguously: [K int8][nb*4 scale bytes]
    row_packed = np.concatenate([q_flat.view(np.uint8), s_bytes], axis=1)
    assert row_packed.shape == (M, K + nb * 4)
    return row_packed.tobytes()


def build_tokenizer_blob(tok_json_path: str, vocab_size: int) -> bytes:
    with open(tok_json_path, "r", encoding="utf-8") as f:
        tj = json.load(f)

    vocab = tj["model"]["vocab"]
    merges = tj["model"]["merges"]
    added_tokens = tj.get("added_tokens", [])

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    # 1. byte_to_tok[256]
    byte_to_tok = np.array([vocab[b2u[b]] for b in range(256)], dtype=np.uint32)

    # 2. Raw decoded UTF-8 byte string for every token ID 0..vocab_size-1
    tok_raw = [b""] * vocab_size
    for piece, tid in vocab.items():
        if tid < vocab_size:
            tok_raw[tid] = bytes(u2b[ch] for ch in piece)
    for at in added_tokens:
        tid = at["id"]
        if tid < vocab_size:
            tok_raw[tid] = at["content"].encode("utf-8")

    tok_offsets = np.zeros(vocab_size + 1, dtype=np.uint32)
    chunks = []
    cur = 0
    for tid in range(vocab_size):
        b = tok_raw[tid]
        tok_offsets[tid] = cur
        chunks.append(b)
        cur += len(b)
    tok_offsets[vocab_size] = cur
    tok_bytes = b"".join(chunks)

    # 3. Pre-built Fibonacci-hashed open-addressing merge table
    # Entry layout (16 bytes): <Q key, <I merged_id, <I rank (0xFFFFFFFF = empty)
    ht_keys = np.zeros(HT_SIZE, dtype=np.uint64)
    ht_merged = np.zeros(HT_SIZE, dtype=np.uint32)
    ht_rank = np.full(HT_SIZE, 0xFFFFFFFF, dtype=np.uint32)

    max_probe = 0
    total_probes = 0
    for rank, m in enumerate(merges):
        if isinstance(m, list):
            a, b = m[0], m[1]
        else:
            a, b = m.split(" ")
        ia = vocab[a]
        ib = vocab[b]
        ic = vocab[a + b]
        key = (ia << 32) | ib
        slot = ((key * FIB_MULT) & 0xFFFFFFFFFFFFFFFF) >> (64 - HT_BITS)
        probes = 1
        while ht_rank[slot] != 0xFFFFFFFF:
            slot = (slot + 1) & HT_MASK
            probes += 1
        ht_keys[slot] = key
        ht_merged[slot] = ic
        ht_rank[slot] = rank
        total_probes += probes
        if probes > max_probe:
            max_probe = probes

    print(
        f"Tokenizer: {len(merges)} merges in {HT_SIZE} slots "
        f"(load={len(merges)/HT_SIZE:.3f}, avg_probe={total_probes/len(merges):.2f}, max_probe={max_probe}), "
        f"tok_bytes={len(tok_bytes):,} B"
    )

    # Pack structured array for merge table
    ht_dtype = np.dtype([("key", "<u8"), ("merged_id", "<u4"), ("rank", "<u4")])
    ht_arr = np.empty(HT_SIZE, dtype=ht_dtype)
    ht_arr["key"] = ht_keys
    ht_arr["merged_id"] = ht_merged
    ht_arr["rank"] = ht_rank
    ht_bytes = ht_arr.tobytes()

    # Lay out sub-sections inside tokenizer blob (all 64-byte aligned relative to blob start)
    off_byte_to_tok = TOK_HEADER_SIZE
    off_tok_offsets = align64(off_byte_to_tok + byte_to_tok.nbytes)
    off_tok_bytes = align64(off_tok_offsets + tok_offsets.nbytes)
    off_merge_ht = align64(off_tok_bytes + len(tok_bytes))
    total_tok_size = align64(off_merge_ht + len(ht_bytes))

    buf = bytearray(total_tok_size)
    struct.pack_into(
        "<QQQQ",
        buf,
        0,
        off_byte_to_tok,
        off_tok_offsets,
        off_tok_bytes,
        off_merge_ht,
    )
    buf[off_byte_to_tok : off_byte_to_tok + byte_to_tok.nbytes] = byte_to_tok.tobytes()
    buf[off_tok_offsets : off_tok_offsets + tok_offsets.nbytes] = tok_offsets.tobytes()
    buf[off_tok_bytes : off_tok_bytes + len(tok_bytes)] = tok_bytes
    buf[off_merge_ht : off_merge_ht + len(ht_bytes)] = ht_bytes
    return bytes(buf)


def export_model(model_dir: str, out_path: str, max_seq_len: int = 4096):
    t0 = time.time()
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    dim = int(cfg["hidden_size"])
    hidden_dim = int(cfg["intermediate_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"])
    vocab_size = int(cfg["vocab_size"])
    rms_norm_eps = float(cfg["rms_norm_eps"])
    rope_theta = float(cfg["rope_theta"])
    tie_emb = bool(cfg.get("tie_word_embeddings", True))

    st_path = os.path.join(model_dir, "model.safetensors")
    tok_path = os.path.join(model_dir, "tokenizer.json")
    hdr, data_offset = load_safetensors_header(st_path)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tok_blob = build_tokenizer_blob(tok_path, vocab_size)

    with open(st_path, "rb") as sf, open(out_path, "wb") as out:
        # Reserve header
        out.write(b"\x00" * HEADER_SIZE)

        # 1. Tokenizer section
        off_tokenizer = pad64(out)
        out.write(tok_blob)

        # 2. Embeddings (tied with lm_head)
        off_embed = pad64(out)
        emb = read_tensor_f32(sf, hdr, data_offset, "model.embed_tokens.weight")
        assert emb.shape == (vocab_size, dim), f"Unexpected embed shape {emb.shape}"
        if "lm_head.weight" in hdr and tie_emb:
            lmh = read_tensor_f32(sf, hdr, data_offset, "lm_head.weight")
            max_diff = float(np.max(np.abs(emb - lmh)))
            assert max_diff == 0.0, f"tie_word_embeddings=True but diff={max_diff}"
            print("Verified embed_tokens == lm_head exact (max_diff=0.0)")
            del lmh
        print(f"Quantizing embed_tokens {emb.shape} -> Q8_0...")
        out.write(quantize_matrix_q8_0_row_split(emb))
        del emb

        # 3. Transformer layers (fixed 64-byte aligned stride)
        off_layers = pad64(out)
        layer_stride = 0
        for l in range(n_layers):
            layer_start = pad64(out)
            prefix = f"model.layers.{l}"

            # 4 FP32 norm vectors
            in_norm = read_tensor_f32(sf, hdr, data_offset, f"{prefix}.input_layernorm.weight")
            q_norm = read_tensor_f32(sf, hdr, data_offset, f"{prefix}.self_attn.q_norm.weight")
            k_norm = read_tensor_f32(sf, hdr, data_offset, f"{prefix}.self_attn.k_norm.weight")
            post_norm = read_tensor_f32(sf, hdr, data_offset, f"{prefix}.post_attention_layernorm.weight")

            assert in_norm.shape == (dim,)
            assert q_norm.shape == (head_dim,)
            assert k_norm.shape == (head_dim,)
            assert post_norm.shape == (dim,)

            out.write(in_norm.astype("<f4").tobytes())
            out.write(q_norm.astype("<f4").tobytes())
            out.write(k_norm.astype("<f4").tobytes())
            out.write(post_norm.astype("<f4").tobytes())

            # 7 Q8_0 projection matrices in fixed order
            proj_specs = [
                (f"{prefix}.self_attn.q_proj.weight", (n_heads * head_dim, dim)),
                (f"{prefix}.self_attn.k_proj.weight", (n_kv_heads * head_dim, dim)),
                (f"{prefix}.self_attn.v_proj.weight", (n_kv_heads * head_dim, dim)),
                (f"{prefix}.self_attn.o_proj.weight", (dim, n_heads * head_dim)),
                (f"{prefix}.mlp.gate_proj.weight", (hidden_dim, dim)),
                (f"{prefix}.mlp.up_proj.weight", (hidden_dim, dim)),
                (f"{prefix}.mlp.down_proj.weight", (dim, hidden_dim)),
            ]
            for name, expected_shape in proj_specs:
                pad64(out)
                mat = read_tensor_f32(sf, hdr, data_offset, name)
                assert mat.shape == expected_shape, f"{name}: got {mat.shape}, expected {expected_shape}"
                out.write(quantize_matrix_q8_0_row_split(mat))

            layer_end = pad64(out)
            this_stride = layer_end - layer_start
            if l == 0:
                layer_stride = this_stride
            else:
                assert this_stride == layer_stride, f"Layer {l} stride {this_stride} != {layer_stride}"

            if (l + 1) % 7 == 0 or l == n_layers - 1:
                print(f"  Exported layer {l + 1}/{n_layers} (stride={layer_stride:,} B)")

        # 4. Final RMSNorm
        off_final_norm = pad64(out)
        final_norm = read_tensor_f32(sf, hdr, data_offset, "model.norm.weight")
        assert final_norm.shape == (dim,)
        out.write(final_norm.astype("<f4").tobytes())
        file_size = pad64(out)

        # Write 128-byte header
        flags = (1 if tie_emb else 0) | 2  # bit0=tied_emb, bit1=qk_norm
        eos_1 = 151645  # <|im_end|>
        eos_2 = 151643  # <|endoftext|>
        header = struct.pack(
            "<4sIIIIIIIIIffIIIIQQQQQQ16s",
            MAGIC,
            VERSION,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            rms_norm_eps,
            rope_theta,
            eos_1,
            eos_2,
            HT_MASK,
            flags,
            off_tokenizer,
            off_embed,
            off_layers,
            layer_stride,
            off_final_norm,
            file_size,
            b"\x00" * 16,
        )
        assert len(header) == HEADER_SIZE
        out.seek(0)
        out.write(header)

    dt = time.time() - t0
    print(f"\nExport complete: {out_path}")
    print(f"  Total size   : {file_size:,} bytes ({file_size / (1024**2):.2f} MiB)")
    print(f"  off_tokenizer: {off_tokenizer:,}")
    print(f"  off_embed    : {off_embed:,}")
    print(f"  off_layers   : {off_layers:,} (stride={layer_stride:,} x {n_layers})")
    print(f"  off_final_nrm: {off_final_norm:,}")
    print(f"  Elapsed      : {dt:.2f} s")


def main():
    ap = argparse.ArgumentParser(description="Export Qwen3-0.6B safetensors to .asmbin Q8_0")
    ap.add_argument("--model-dir", default="models/qwen3-0.6b-bf16", help="Directory containing Qwen3-0.6B config.json, model.safetensors, tokenizer.json")
    ap.add_argument("--out", default="models/qwen3-0.6b-q8_0.asmbin", help="Output .asmbin path")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    args = ap.parse_args()
    export_model(args.model_dir, args.out, args.max_seq_len)


if __name__ == "__main__":
    main()
