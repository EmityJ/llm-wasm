# Phase 1: Qwen3-0.6B Inference in a 13 KB Pure x86-64 ELF

A complete inference engine for **Qwen/Qwen3-0.6B** written in **100% pure x86-64 NASM assembly** for Linux.

- **13,528 bytes** stripped static ELF executable (`.text` = 10,990 B, `.rodata` = 1,943 B).
- **Zero runtime dependencies:** no `gcc`, no `libc`, no `libm`, no dynamic linker, no `.so` files.
- **Full Qwen3 architecture:** GQA (`16Q / 8KV`, `head_dim = 128`), per-head QK-Norm (`RMSNorm` on Q and K before RoPE), split-half NeoX RoPE (`rope_theta = 1e6`), SwiGLU MLP, tied embeddings / LM head, persistent multi-turn KV-cache (`--chat`), greedy and temperature/top-p sampling.
- **Built-in BPE tokenizer in assembly:** Fibonacci-hashed open-addressing merge table (`524,288` slots, 100% token-exact match with HuggingFace `AutoTokenizer`), UTF-8 prompt encoding (English, Cyrillic, code, numbers), Qwen3 chat template (default no-think and `--think` reasoning mode), and raw `write()` streaming to stdout.
- **AVX2 + FMA Q8_0 x FP32 matmul + zero-libc thread pool:** 4-row unrolled AVX2/FMA kernel, fused QKV and gate+up projections, and a raw Linux `clone()` + `futex()` thread pool with cache-line-isolated completion slots. **105.4 tok/s** peak decode and **154.3 tok/s** peak prefill on AMD EPYC 7642, bit-exact across `1T..47T`.

---

## Prerequisites

- Linux x86-64 CPU with **AVX2 + FMA**.
- Build tools:
  ```bash
  sudo apt install nasm binutils
  ```
- Offline weight export (one-time, Python only; never needed at runtime):
  ```bash
  pip install numpy huggingface_hub
  ```

---

## Step-by-Step Guide

### 1. Download `Qwen/Qwen3-0.6B`

Using the Hugging Face CLI (`hf` or `huggingface-cli`):

```bash
mkdir -p models
hf download Qwen/Qwen3-0.6B \
  config.json model.safetensors tokenizer.json \
  --local-dir models/qwen3-0.6b-bf16
```

*(If you have the older CLI installed, replace `hf download` with `huggingface-cli download`.)*

### 2. Export Weights to `.asmbin` (Q8_0)

Convert the BF16 safetensors and `tokenizer.json` into a single 64-byte aligned, directly `mmap()`-able `.asmbin` file (`649.18 MiB`):

```bash
python3 scripts/export_weights.py \
  --model-dir models/qwen3-0.6b-bf16 \
  --out models/qwen3-0.6b-q8_0.asmbin
```

### 3. Build the 13 KB Pure x86-64 Binary

From the repository root:

```bash
make inference test-math
make check
```

Expected ELF inspection output:

```text
-rwxr-xr-x 1 user user 13528 build/asm-llm
build/asm-llm: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked, stripped
There is no dynamic section in this file.
```

### 4. Run Inference

```bash
# Default Qwen3 chat (direct no-think answer, greedy, 16 threads)
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin \
  -p "Explain how a CPU cache works in 3 concise sentences." -n 128 -t 16

# Interactive multi-turn chat preserving KV-cache across turns (/clear, /think, /nothink, /quit)
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin --chat -t 16

# Enable Qwen3 <think> reasoning mode
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin \
  -p "What is 17 * 23?" -n 512 -t 16 --think

# Temperature + top-p sampling
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin \
  -p "Write a one-sentence motto for a systems programmer." \
  -n 64 -t 16 --temp 0.6 --topp 0.95 --seed 1337

# Russian / raw UTF-8 prompt and output
./build/asm-llm -m models/qwen3-0.6b-q8_0.asmbin \
  -p "Объясни в двух предложениях, как работает стек вызовов в процессоре." -n 128 -t 16
```

---

## CLI Reference

| Flag | Default | Description |
|---|---:|---|
| `-m <path>` | *(required)* | Path to `.asmbin` weight file |
| `-p <text>` | built-in | User prompt (UTF-8) |
| `-n <int>` | `128` | Maximum generated tokens |
| `-t <int>` | `32` | Worker thread count (`clone()` + `futex()` pool) |
| `-c <int>` | `2048` | KV-cache context length |
| `--chat` | off | Interactive multi-turn chat via raw Linux `read(0)`/`write(1)` syscalls; preserves KV-cache across turns, supports `/clear` (`/new`, `/reset`), `/think`, `/nothink`, `/quit` (`/exit`, Ctrl-D), and prints `ctx=X/2048` |
| `--think` | off | Enable Qwen3 `<think>...</think>` reasoning mode |
| `--raw` | off | Skip Qwen3 chat template, encode prompt bytes directly |
| `--tokens <csv>` | none | Feed comma-separated token IDs directly |
| `--temp <float>` | `0.0` | Sampling temperature (`0.0` = greedy argmax) |
| `--topp <float>` | `0.95` | Nucleus (top-p) sampling threshold |
| `--seed <uint>` | `42` | PRNG seed for sampling |
| `--dump-logits <path>` | none | Dump final prefill logits (`[vocab_size]` FP32) to file |
| `--print-tokens` | off | Print `PROMPT_TOKENS:` and `GEN_TOKENS:` CSV lines to stderr |
| `-h`, `--help` | - | Print usage and exit |

---

## `.asmbin` Binary Format (Version 1)

Designed for zero-copy `mmap()` with **64-byte alignment** for every section, matrix, and row:

- **Header (128 bytes):** magic `"ASML"`, version `1`, model dimensions (`dim=1024`, `hidden_dim=3072`, `n_layers=28`, `n_heads=16`, `n_kv_heads=8`, `head_dim=128`, `vocab_size=151936`, `rms_norm_eps=1e-6`, `rope_theta=1e6`, EOS tokens `151645`/`151643`), hash table mask, flags, and 64-byte aligned section offsets.
- **Tokenizer section (~9.5 MiB):** `byte_to_tok[256]`, `tok_offsets[vocab_size+1]`, raw UTF-8 `tok_bytes` pool (`976,309` B), and a pre-built `524,288`-slot open-addressing merge table (`151,387` merges, load factor `0.289`, avg probe `1.20`).
- **Tied Embeddings / LM Head (`[151936, 1024]` Q8_0):** verified bit-identical in BF16 (`max_diff = 0.0`), stored once (`166.92 MiB`).
- **28 Transformer Layers (`stride = 17,703,936 B` each):** 4 FP32 norm vectors (`input_layernorm`, `q_norm`, `k_norm`, `post_attention_layernorm`) followed by 7 per-row split Q8_0 matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Contiguous layout allows fused QKV (`[4096, 1024]`) and gate+up (`[6144, 1024]`) matmuls in a single thread-pool barrier.
- **Per-row split Q8_0:** each row of `K` weights stores `[K int8 in -127..127][K/32 fp32 scales]`. For `K in {1024, 2048, 3072}`, row strides are `{1152, 2304, 3456}` bytes, all exact multiples of 64 bytes.
- **Total size:** `680,718,528` bytes (`649.18 MiB`) vs `1.40 GiB` BF16 safetensors.

---

## Verification

### 1. Pure-Assembly Math & SIMD Unit Tests

```bash
python3 scripts/test_math.py build/test_math
```

| Kernel | Metric | Measured Error | Target |
|---|---|---:|---:|
| `fast_expf_avx2` | Max relative error (`[-120, +80]`) | `1.394e-07` | `< 1e-6` |
| `fast_sincosf` (sin) | Max abs error (`[-4096, +12345]`) | `2.940e-08` | `< 1e-6` |
| `fast_sincosf` (cos) | Max abs error (`[-4096, +12345]`) | `2.928e-08` | `< 1e-6` |
| `init_rope_freqs` (FP64) | Max abs error (`theta=1e6`, 64 freqs) | `4.366e-14` | `< 1e-12` |
| `rmsnorm_avx2` | Max abs error (`n=128`) | `2.056e-07` | `< 1e-6` |
| `silu_mul_avx2` | Max rel error (`[-150, +150]`, overflow-free) | `6.774e-08` | `< 1e-6` |
| `matmul_q8_0_range` | Max abs error (`8x64` Q8_0 x FP32) | `9.537e-07` | `< 1e-5` |

### 2. End-to-End Logits & 64-Token Greedy Parity

```bash
python3 scripts/verify_ref.py \
  --model models/qwen3-0.6b-q8_0.asmbin \
  --ref-dir models/qwen3-0.6b-bf16 \
  --compare-asm build/asm-llm \
  --threads 16
```

Prompt: `"Explain the difference between L1, L2, and L3 CPU caches and why cache locality matters."` (`34` prompt tokens, `64` generated tokens):

| Comparison | Tokenizer Match | `max_abs_diff` | `mean_abs_diff` | `cosine_similarity` | Greedy Match |
|---|---|---:|---:|---:|---|
| **`build/asm-llm` vs Python `.asmbin` Q8_0 ref** | **Exact (34/34)** | **`0.0000167`** | **`0.0000030`** | **`1.000000000`** | **64/64 tokens (100%)** |
| Python `.asmbin` Q8_0 vs PyTorch FP32 `Qwen3-0.6B` | **Exact (34/34)** | `0.427719` | `0.075629` | `0.9998471` | **64/64 tokens (100%)** |

All three implementations (PyTorch FP32, Python Q8_0 reference, and the 12 KB assembly ELF) produce the exact same 64 greedy tokens.

---

## Benchmarks (AMD EPYC 7642, 47C Online, 256 MiB L3, DDR4)

37 prompt tokens, 64 generated tokens, greedy decoding (`-n 64`), verified bit-for-bit identical across all thread counts:

| Threads | Prefill (ms) | Prefill (`tok/s`) | Decode (ms) | Decode (`tok/s`) | Speedup vs 1T (Decode) | Determinism |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 3133.09 | 11.81 | 7910.50 | 7.96 | 1.00x | Baseline (64/64) |
| 4 | 933.82 | 39.62 | 2330.10 | 27.04 | 3.40x | Exact match |
| 8 | 461.59 | 80.16 | 1134.25 | 55.54 | 6.98x | Exact match |
| 12 | 370.08 | 99.98 | 914.84 | 68.86 | 8.65x | Exact match |
| 16 | 318.88 | 116.03 | 796.55 | 79.09 | 9.94x | Exact match |
| 24 | 271.67 | 136.20 | 673.23 | 93.58 | 11.76x | Exact match |
| 32 | 251.00 | 147.41 | 636.52 | 98.98 | 12.43x | Exact match |
| **40** | **243.91** | **151.70** | **628.88** | **100.18** | **12.59x** | **Exact match** |
| 47 | 265.49 | 139.36 | 730.02 | 86.30 | 10.84x | Exact match |

Peak throughput observed on shorter live prompts at `-t 32`: **154.28 tok/s prefill** and **105.39 tok/s decode**.

---

## Live Generation Samples (`build/asm-llm`, `-t 32`)

**1. Technical explanation:**
> *Prompt:* `Explain how a CPU cache works in 3 concise sentences.`  
> *Output (154.28 tok/s prefill, 103.40 tok/s decode):*  
> A CPU cache is a small amount of memory used to store frequently accessed data and instructions. It reduces the need for more memory by storing data that is accessed again quickly. This improves performance by minimizing the number of times data needs to be retrieved from main memory.

**2. Logic / arithmetic (direct mode):**
> *Prompt:* `If Alice has 3 apples and Bob gives her 4 more, then she eats 2, how many apples does Alice have?`  
> *Output (147.98 tok/s prefill, 90.06 tok/s decode):*  
> Let's break it down step by step:  
> - Alice starts with **3 apples**.  
> - Bob gives her **4 more apples**, so she now has: `3 + 4 = 7 apples`  
> - Then she eats **2 apples**, so the final number of apples she has is: `7 - 2 = 5`  
> **Answer:** Alice has **5 apples**.

**3. Russian UTF-8 (native UTF-8 BPE encode and stdout decode in assembly):**
> *Prompt:* `Объясни в двух предложениях, как работает стек вызовов в процессоре.`  
> *Output (149.49 tok/s prefill, 96.09 tok/s decode):*  
> Стек вызовов — это линейная последовательность вызовов, в которой каждый вызов ссылается на предыдущий, создавая цепочку выполнения. Он позволяет процессору эффективно обрабатывать задачи, обеспечивая стабильность и оптимизацию выполнения.

**4. Reasoning mode (`--think`):**
> *Prompt:* `What is 17 * 23?`  
> Breaks down `(10 + 7) * 23 = 230 + 161 = 391` and cross-checks `17 * 20 + 17 * 3 = 340 + 51 = 391` inside `<think>...</think>`.

**5. Sampling (`--temp 0.6 --topp 0.95 --seed 1337`):**
> *Prompt:* `Write a one-sentence motto for a systems programmer.`  
> *Output (146.74 tok/s prefill, 102.09 tok/s decode):*  
> A systems programmer thrives in the intersection of complexity and precision, building the foundation of innovation.
