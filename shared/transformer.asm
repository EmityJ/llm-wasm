; ============================================================================
; transformer.asm — Full Qwen3-0.6B Transformer Forward Pass & Model Loader
;
; Implements exact Qwen3 architecture:
;   - Q8_0 tied embeddings + LM head
;   - 28 layers: RMSNorm -> Q/K/V (shared Q8_0 activation quant) -> per-head
;     QK-Norm -> split-half NeoX RoPE -> KV-cache -> GQA Attention (16Q/8KV)
;     -> O-proj + residual -> RMSNorm -> SwiGLU MLP + residual
;   - Final RMSNorm -> tied LM head logits
; ============================================================================

bits 64
default rel

%include "constants.inc"

%ifndef BARE_METAL
extern sys_openat_ro
extern sys_close
extern sys_fstat
extern sys_mmap_file
%endif
extern sys_mmap_anon
extern sys_munmap

extern init_rope_freqs
extern compute_rope_cs
extern rope_apply_avx2
extern rmsnorm_avx2
extern softmax_avx2
extern silu_mul_avx2
extern vec_add_avx2

extern dequant_row_q8_0
extern dot_f32_avx2
extern saxpy_f32_avx2

extern tp_matmul_q8_0

section .text

global g_ctx
global model_init
global model_free
global transformer_forward

; Helper macro: align reg up to 64 bytes
%macro ALIGN64 1
    add     %1, 63
    and     %1, -64
%endmacro

; ----------------------------------------------------------------------------
; Linux:      int64 model_init(const char *path, uint64 max_seq_override)
; Bare-Metal: int64 model_init(const void *base, uint64 size, uint64 max_seq_override)
; Returns 0 on success, negative error code on failure.
; ----------------------------------------------------------------------------
model_init:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, STAT_SIZE + 8          ; 6 pushes (48) + 152 + 8(ret) = 208 (16B aligned)

    lea     rbx, [rel g_ctx]
%ifdef BARE_METAL
    mov     r15, rdi                    ; base ptr in physical RAM
    mov     r14, rsi                    ; size
    mov     r12, rdx                    ; max_seq_override
%else
    mov     r12, rsi                    ; max_seq_override

    ; Open .asmbin read-only
    call    sys_openat_ro
    test    rax, rax
    js      .err
    mov     r13, rax                    ; fd

    ; fstat to get file size
    mov     rdi, r13
    mov     rsi, rsp
    call    sys_fstat
    test    rax, rax
    js      .err_close
    mov     r14, [rsp + STAT_ST_SIZE]   ; file size

    ; mmap entire .asmbin read-only
    mov     rdi, r14
    mov     rsi, r13
    call    sys_mmap_file
    mov     r15, rax                    ; mmap_base

    ; Close fd (mmap keeps mapping alive)
    mov     rdi, r13
    call    sys_close

    test    r15, r15
    js      .err_mmap
%endif

    ; Verify header magic and version
    cmp     dword [r15 + HDR_MAGIC], ASMBIN_MAGIC
    jne     .err_bad_hdr
    cmp     dword [r15 + HDR_VERSION], ASMBIN_VERSION
    jne     .err_bad_hdr

    mov     [rbx + CTX_MMAP_BASE], r15
    mov     [rbx + CTX_MMAP_SIZE], r14

    ; Load scalar dimensions from header
    mov     eax, [r15 + HDR_DIM]
    mov     [rbx + CTX_DIM], rax
    mov     eax, [r15 + HDR_HIDDEN_DIM]
    mov     [rbx + CTX_HIDDEN_DIM], rax
    mov     eax, [r15 + HDR_N_LAYERS]
    mov     [rbx + CTX_N_LAYERS], rax
    mov     eax, [r15 + HDR_N_HEADS]
    mov     [rbx + CTX_N_HEADS], rax
    mov     eax, [r15 + HDR_N_KV_HEADS]
    mov     [rbx + CTX_N_KV_HEADS], rax
    mov     eax, [r15 + HDR_HEAD_DIM]
    mov     [rbx + CTX_HEAD_DIM], rax

    ; q_dim = n_heads * head_dim; kv_dim = n_kv_heads * head_dim
    mov     rax, [rbx + CTX_N_HEADS]
    imul    rax, [rbx + CTX_HEAD_DIM]
    mov     [rbx + CTX_Q_DIM], rax
    mov     rax, [rbx + CTX_N_KV_HEADS]
    imul    rax, [rbx + CTX_HEAD_DIM]
    mov     [rbx + CTX_KV_DIM], rax

    mov     eax, [r15 + HDR_VOCAB_SIZE]
    mov     [rbx + CTX_VOCAB_SIZE], rax
    mov     eax, [r15 + HDR_MAX_SEQ_LEN]
    test    r12, r12
    jz      .use_hdr_seq
    cmp     r12, rax
    ja      .use_hdr_seq
    mov     rax, r12
.use_hdr_seq:
    mov     [rbx + CTX_MAX_SEQ], rax

    mov     rax, [r15 + HDR_LAYER_STRIDE]
    mov     [rbx + CTX_LAYER_STRIDE], rax
    mov     eax, [r15 + HDR_HT_MASK]
    mov     [rbx + CTX_HT_MASK], rax
    mov     eax, [r15 + HDR_EOS_1]
    mov     [rbx + CTX_EOS_1], rax
    mov     eax, [r15 + HDR_EOS_2]
    mov     [rbx + CTX_EOS_2], rax
    mov     eax, [r15 + HDR_RMS_EPS]
    mov     [rbx + CTX_RMS_EPS], eax
    mov     eax, [r15 + HDR_ROPE_THETA]
    mov     [rbx + CTX_ROPE_THETA], eax

    ; Resolve pointers into mmap'd .asmbin
    mov     rax, [r15 + HDR_OFF_TOKENIZER]
    add     rax, r15                    ; tok_base
    mov     rcx, [rax + TOKH_OFF_B2T]
    add     rcx, rax
    mov     [rbx + CTX_B2T_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_OFFSETS]
    add     rcx, rax
    mov     [rbx + CTX_TOK_OFFS_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_BYTES]
    add     rcx, rax
    mov     [rbx + CTX_TOK_BYTES_PTR], rcx
    mov     rcx, [rax + TOKH_OFF_HT]
    add     rcx, rax
    mov     [rbx + CTX_HT_PTR], rcx

    mov     rax, [r15 + HDR_OFF_EMBED]
    add     rax, r15
    mov     [rbx + CTX_EMB_PTR], rax
    mov     rax, [r15 + HDR_OFF_LAYERS]
    add     rax, r15
    mov     [rbx + CTX_LAYERS_PTR], rax
    mov     rax, [r15 + HDR_OFF_FINAL_NORM]
    add     rax, r15
    mov     [rbx + CTX_FINAL_NORM_PTR], rax

    ; Compute single anonymous mmap arena size for all activation buffers + KV cache
    ; First pass: compute exact aligned offsets (stored temporarily as relative offsets!)
    xor     rbp, rbp                    ; running offset = 0

    %macro ALLOC_BUF 2
        ALIGN64 rbp
        mov     [rbx + %1], rbp
        add     rbp, %2
    %endmacro

    mov     rax, [rbx + CTX_DIM]
    shl     rax, 2                      ; dim * 4
    ALLOC_BUF CTX_BUF_X, rax
    ALLOC_BUF CTX_BUF_XB, rax

    ; Place buf_q, buf_k, buf_v CONTIGUOUSLY so Q+K+V execute in ONE fused matmul!
    mov     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2                      ; q_dim * 4
    ALLOC_BUF CTX_BUF_Q, rax
    mov     rax, [rbx + CTX_KV_DIM]
    shl     rax, 2                      ; kv_dim * 4
    ALLOC_BUF CTX_BUF_K, rax
    ALLOC_BUF CTX_BUF_V, rax

    mov     rax, [rbx + CTX_Q_DIM]
    shl     rax, 2                      ; q_dim * 4
    ALLOC_BUF CTX_BUF_ATT_OUT, rax

    mov     rax, [rbx + CTX_MAX_SEQ]
    shl     rax, 2                      ; max_seq * 4
    ALLOC_BUF CTX_BUF_ATT_SCR, rax

    ; Place buf_gate and buf_up CONTIGUOUSLY so gate+up execute in ONE fused matmul!
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    shl     rax, 2                      ; hidden_dim * 4
    ALLOC_BUF CTX_BUF_GATE, rax
    ALLOC_BUF CTX_BUF_UP, rax

    mov     rax, [rbx + CTX_VOCAB_SIZE]
    shl     rax, 2                      ; vocab_size * 4
    ALLOC_BUF CTX_BUF_LOGITS, rax
    ALLOC_BUF CTX_BUF_PROBS, rax

    ; Activation Q8_0 buffers (max K = max(dim, q_dim, hidden_dim) <= hidden_dim + 4096)
    mov     rax, [rbx + CTX_HIDDEN_DIM]
    cmp     rax, [rbx + CTX_Q_DIM]
    jae     .kmax_ok
    mov     rax, [rbx + CTX_Q_DIM]
.kmax_ok:
    ALLOC_BUF CTX_BUF_XQ, rax           ; max_K int8
    mov     rcx, rax
    shr     rcx, 3                      ; (max_K / 32) * 4
    ALLOC_BUF CTX_BUF_XS, rcx

    ; RoPE buffers: inv_freq (head_dim/2 * 8), cos (head_dim/2 * 4), sin (head_dim/2 * 4)
    mov     rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2                      ; (head_dim/2) * 8 = head_dim * 4
    ALLOC_BUF CTX_BUF_INV_FREQ, rax
    shr     rax, 1                      ; (head_dim/2) * 4
    ALLOC_BUF CTX_BUF_COS, rax
    ALLOC_BUF CTX_BUF_SIN, rax

    ; Prompt & gen token buffers
    ALLOC_BUF CTX_BUF_PROMPT, MAX_PROMPT_TOKENS * 4
    ALLOC_BUF CTX_BUF_GEN,    MAX_GEN_TOKENS * 4

    ; KV-cache: each is n_layers * max_seq * kv_dim * 4 bytes
    mov     rax, [rbx + CTX_N_LAYERS]
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, 2
    ALLOC_BUF CTX_BUF_K_CACHE, rax
    ALLOC_BUF CTX_BUF_V_CACHE, rax

    ALIGN64 rbp
    mov     [rbx + CTX_ARENA_SIZE], rbp

    ; Allocate arena via anonymous mmap
    mov     rdi, rbp
    call    sys_mmap_anon
    test    rax, rax
    js      .err_arena
    mov     [rbx + CTX_ARENA_BASE], rax

    ; Relocate all 20 CTX_BUF_* offsets -> absolute pointers
    lea     rcx, [rbx + CTX_BUF_X]
    mov     edx, 20
.reloc_lp:
    add     [rcx], rax
    add     rcx, 8
    dec     edx
    jnz     .reloc_lp

    ; Precompute exact FP64 RoPE inverse frequencies
    mov     rdi, [rbx + CTX_BUF_INV_FREQ]
    mov     rsi, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_ROPE_THETA]
    call    init_rope_freqs

    xor     eax, eax                    ; success!
    jmp     .ret

.err_arena:
    mov     rdi, r15
    mov     rsi, r14
    call    sys_munmap
    mov     rax, -4
    jmp     .ret
.err_bad_hdr:
    mov     rdi, r15
    mov     rsi, r14
    call    sys_munmap
    mov     rax, -3
    jmp     .ret
%ifndef BARE_METAL
.err_mmap:
    mov     rax, -2
    jmp     .ret
.err_close:
    mov     rdi, r13
    call    sys_close
.err:
    mov     rax, -1
%endif
.ret:
    add     rsp, STAT_SIZE + 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void model_free(void)
; Unmaps arena and .asmbin file cleanly.
; ----------------------------------------------------------------------------
model_free:
    push    rbx
    lea     rbx, [rel g_ctx]
    mov     rdi, [rbx + CTX_ARENA_BASE]
    test    rdi, rdi
    jz      .unmap_model
    mov     rsi, [rbx + CTX_ARENA_SIZE]
    call    sys_munmap
.unmap_model:
    mov     rdi, [rbx + CTX_MMAP_BASE]
    test    rdi, rdi
    jz      .done
    mov     rsi, [rbx + CTX_MMAP_SIZE]
    call    sys_munmap
.done:
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void transformer_forward(uint32 token_id, uint64 pos, int compute_logits)
; rdi = token_id, rsi = pos, rdx = compute_logits (0 during non-final prefill)
; ----------------------------------------------------------------------------
transformer_forward:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    ; Reserve 128 bytes of stack locals (6 pushes=48 + 128 + 8(ret)=184 -> use 136 for 16B align!)
    sub     rsp, 136
    ; Stack locals:
    ;   [rsp + 0]  = pos
    ;   [rsp + 8]  = compute_logits
    ;   [rsp + 16] = layer_idx l
    ;   [rsp + 24] = p_in_norm
    ;   [rsp + 32] = p_q_norm
    ;   [rsp + 40] = p_k_norm
    ;   [rsp + 48] = p_post_norm
    ;   [rsp + 56] = p_q_proj
    ;   [rsp + 64] = p_k_proj
    ;   [rsp + 72] = p_v_proj
    ;   [rsp + 80] = p_o_proj
    ;   [rsp + 88] = p_gate_proj
    ;   [rsp + 96] = p_up_proj
    ;   [rsp + 104]= p_down_proj
    ;   [rsp + 112]= att_scale (float32)
    ;   [rsp + 116]= gqa_ratio (uint32)
    ;   [rsp + 120]= layer_kv_base_bytes (l * max_seq * kv_dim * 4)

    mov     [rsp + 0], rsi              ; pos
    mov     [rsp + 8], rdx              ; compute_logits
    lea     rbx, [rel g_ctx]

    ; Precompute att_scale = 1.0f / sqrt(head_dim) and gqa_ratio = n_heads / n_kv_heads
    mov     rax, [rbx + CTX_HEAD_DIM]
    vcvtsi2ss xmm0, xmm0, rax
    vsqrtss xmm0, xmm0, xmm0
    vmovss  xmm1, [rel t_one]
    vdivss  xmm1, xmm1, xmm0
    vmovss  [rsp + 112], xmm1

    mov     rax, [rbx + CTX_N_HEADS]
    xor     edx, edx
    div     qword [rbx + CTX_N_KV_HEADS]
    mov     [rsp + 116], eax

    ; 1. Embedding lookup: dequantize row token_id of embed_tokens into buf_x
    mov     rax, [rbx + CTX_DIM]
    mov     rcx, rax
    shr     rcx, 3
    add     rcx, rax                    ; emb_row_stride = dim + dim/8
    imul    rdi, rcx
    add     rdi, [rbx + CTX_EMB_PTR]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, rax                    ; dim
    call    dequant_row_q8_0

    ; 2. Compute RoPE cos/sin tables ONCE for this position (reused by all 28 layers!)
    mov     rdi, [rbx + CTX_BUF_COS]
    mov     rsi, [rbx + CTX_BUF_SIN]
    mov     rdx, [rbx + CTX_BUF_INV_FREQ]
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shr     rcx, 1                      ; half_dim
    mov     r8,  [rsp + 0]              ; pos
    call    compute_rope_cs

    ; 3. Transformer Layer Loop (l = 0 .. n_layers - 1)
    mov     qword [rsp + 16], 0

.layer_lp:
    mov     rax, [rsp + 16]
    cmp     rax, [rbx + CTX_N_LAYERS]
    jae     .layers_done

    ; Compute all 11 tensor pointers for layer l (matching export_weights.py layout)
    imul    rax, [rbx + CTX_LAYER_STRIDE]
    add     rax, [rbx + CTX_LAYERS_PTR]

    mov     [rsp + 24], rax             ; in_norm
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + 32], rax             ; q_norm
    mov     rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    add     rax, rcx
    mov     [rsp + 40], rax             ; k_norm
    add     rax, rcx
    mov     [rsp + 48], rax             ; post_norm
    mov     rcx, [rbx + CTX_DIM]
    shl     rcx, 2
    add     rax, rcx

    ; Helper: row_stride(K) = K + K/8
    ; dim_stride = dim + dim/8
    mov     r12, [rbx + CTX_DIM]
    mov     r13, r12
    shr     r13, 3
    add     r12, r13                    ; r12 = dim_stride (1152)

    ALIGN64 rax
    mov     [rsp + 56], rax             ; q_proj [q_dim, dim]
    mov     rcx, [rbx + CTX_Q_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 64], rax             ; k_proj [kv_dim, dim]
    mov     rcx, [rbx + CTX_KV_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 72], rax             ; v_proj [kv_dim, dim]
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 80], rax             ; o_proj [dim, q_dim]
    mov     rcx, [rbx + CTX_Q_DIM]
    mov     rdx, rcx
    shr     rdx, 3
    add     rcx, rdx                    ; q_dim_stride (2304)
    imul    rcx, [rbx + CTX_DIM]
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 88], rax             ; gate_proj [hidden_dim, dim]
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    imul    rcx, r12
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 96], rax             ; up_proj [hidden_dim, dim]
    add     rax, rcx

    ALIGN64 rax
    mov     [rsp + 104], rax            ; down_proj [dim, hidden_dim]

    ; --- 3a. Input RMSNorm: buf_xb = rmsnorm(buf_x, in_norm) ---
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rsp + 24]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

    ; --- 3b. Fused Q + K + V projection in ONE parallel matmul (zero padding between q/k/v!) ---
    mov     rdi, [rsp + 56]             ; q_proj (contiguous with k_proj, v_proj)
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_Q]      ; buf_q (contiguous with buf_k, buf_v)
    mov     rcx, [rbx + CTX_Q_DIM]
    add     rcx, [rbx + CTX_KV_DIM]
    add     rcx, [rbx + CTX_KV_DIM]     ; M = q_dim + 2*kv_dim (4096)
    mov     r8,  [rbx + CTX_DIM]        ; K = dim (1024)
    call    tp_matmul_q8_0

    ; --- 3c. Qwen3 QK-Norm (RMSNorm per head over head_dim, in-place) ---
    mov     r12, [rbx + CTX_BUF_Q]
    mov     r13, [rbx + CTX_N_HEADS]
    mov     r14, [rbx + CTX_HEAD_DIM]
    shl     r14, 2                      ; head_bytes
.qnorm_lp:
    mov     rdi, r12
    mov     rsi, r12
    mov     rdx, [rsp + 32]             ; q_norm
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    add     r12, r14
    dec     r13
    jnz     .qnorm_lp

    mov     r12, [rbx + CTX_BUF_K]
    mov     r13, [rbx + CTX_N_KV_HEADS]
.knorm_lp:
    mov     rdi, r12
    mov     rsi, r12
    mov     rdx, [rsp + 40]             ; k_norm
    mov     rcx, [rbx + CTX_HEAD_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2
    add     r12, r14
    dec     r13
    jnz     .knorm_lp

    ; --- 3d. Split-half RoPE on Q and K ---
    mov     rdi, [rbx + CTX_BUF_Q]
    mov     rsi, [rbx + CTX_BUF_COS]
    mov     rdx, [rbx + CTX_BUF_SIN]
    mov     rcx, [rbx + CTX_N_HEADS]
    mov     r8,  [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2

    mov     rdi, [rbx + CTX_BUF_K]
    mov     rsi, [rbx + CTX_BUF_COS]
    mov     rdx, [rbx + CTX_BUF_SIN]
    mov     rcx, [rbx + CTX_N_KV_HEADS]
    mov     r8,  [rbx + CTX_HEAD_DIM]
    call    rope_apply_avx2

    ; --- 3e. Store K, V at cache[l, pos] ---
    ; layer_kv_base_bytes = l * max_seq * kv_dim * 4
    mov     rax, [rsp + 16]             ; l
    imul    rax, [rbx + CTX_MAX_SEQ]
    imul    rax, [rbx + CTX_KV_DIM]
    shl     rax, 2
    mov     [rsp + 120], rax

    ; pos_offset = pos * kv_dim * 4
    mov     rcx, [rsp + 0]              ; pos
    imul    rcx, [rbx + CTX_KV_DIM]
    shl     rcx, 2
    add     rax, rcx                    ; total byte offset into k_cache / v_cache

    mov     rdi, [rbx + CTX_BUF_K_CACHE]
    add     rdi, rax
    mov     rsi, [rbx + CTX_BUF_K]
    mov     rdx, [rbx + CTX_BUF_V_CACHE]
    add     rdx, rax
    mov     r8,  [rbx + CTX_BUF_V]
    mov     rcx, [rbx + CTX_KV_DIM]
    shl     rcx, 2                      ; kv_bytes
    xor     r9d, r9d
.kv_copy_lp:
    cmp     r9, rcx
    jae     .kv_copy_done
    vmovups ymm0, [rsi + r9]
    vmovups ymm1, [r8 + r9]
    vmovups [rdi + r9], ymm0
    vmovups [rdx + r9], ymm1
    add     r9, 32
    jmp     .kv_copy_lp
.kv_copy_done:

    ; --- 3f. GQA Attention over all heads h = 0 .. n_heads - 1 ---
    xor     r12d, r12d                  ; h = 0
.head_lp:
    cmp     r12, [rbx + CTX_N_HEADS]
    jae     .att_done

    ; kv_h = h / gqa_ratio
    mov     eax, r12d
    xor     edx, edx
    div     dword [rsp + 116]           ; eax = kv_h
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2                      ; kv_h_bytes = kv_h * head_dim * 4
    mov     r13, rax                    ; r13 = kv_h_bytes

    ; q_head = buf_q + h * head_dim * 4
    mov     rax, r12
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    mov     r14, [rbx + CTX_BUF_Q]
    add     r14, rax                    ; r14 = q_head

    ; Compute attention scores[0..pos]: dot(q_head, k_cache[l, t, kv_h]) * att_scale
    mov     r15, [rbx + CTX_BUF_K_CACHE]
    add     r15, [rsp + 120]            ; layer_k_base
    add     r15, r13                    ; k_ptr for t = 0
    mov     rbp, [rbx + CTX_KV_DIM]
    shl     rbp, 2                      ; step per timestep t = kv_dim * 4

    xor     ecx, ecx                    ; t = 0
    mov     r11, [rbx + CTX_BUF_ATT_SCR]
.score_lp:
    cmp     rcx, [rsp + 0]
    ja      .scores_done
    push    rcx
    push    r11                         ; 2 pushes -> 16B aligned
    mov     rdi, r14                    ; q_head
    mov     rsi, r15                    ; k_vec
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    dot_f32_avx2                ; xmm0 = dot
    pop     r11
    pop     rcx
    vmulss  xmm0, xmm0, [rsp + 112]     ; * att_scale
    vmovss  [r11 + rcx*4], xmm0
    add     r15, rbp
    inc     rcx
    jmp     .score_lp

.scores_done:
    ; Softmax scores[0..pos] (length = pos + 1)
    mov     rdi, [rbx + CTX_BUF_ATT_SCR]
    mov     rsi, [rsp + 0]
    inc     rsi
    call    softmax_avx2

    ; Zero att_head = buf_att_out + h * head_dim * 4
    mov     rax, r12
    imul    rax, [rbx + CTX_HEAD_DIM]
    shl     rax, 2
    mov     r14, [rbx + CTX_BUF_ATT_OUT]
    add     r14, rax                    ; r14 = att_head

    mov     rcx, [rbx + CTX_HEAD_DIM]
    shl     rcx, 2
    vxorps  ymm0, ymm0, ymm0
    xor     eax, eax
.zero_att:
    cmp     rax, rcx
    jae     .val_accum
    vmovups [r14 + rax], ymm0
    add     rax, 32
    jmp     .zero_att

.val_accum:
    ; Weighted sum of V vectors: att_head += probs[t] * v_cache[l, t, kv_h]
    mov     r15, [rbx + CTX_BUF_V_CACHE]
    add     r15, [rsp + 120]
    add     r15, r13                    ; v_ptr for t = 0
    xor     ecx, ecx
    mov     r11, [rbx + CTX_BUF_ATT_SCR]
.val_lp:
    cmp     rcx, [rsp + 0]
    ja      .head_next
    vmovss  xmm0, [r11 + rcx*4]         ; prob[t]
    push    rcx
    push    r11
    mov     rdi, r14                    ; att_head
    mov     rsi, r15                    ; v_vec
    mov     rdx, [rbx + CTX_HEAD_DIM]
    call    saxpy_f32_avx2
    pop     r11
    pop     rcx
    add     r15, rbp
    inc     rcx
    jmp     .val_lp

.head_next:
    inc     r12
    jmp     .head_lp

.att_done:
    ; --- 3g. Output projection + residual: buf_x += o_proj @ buf_att_out ---
    mov     rdi, [rsp + 80]             ; o_proj
    mov     rsi, [rbx + CTX_BUF_ATT_OUT]
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_Q_DIM]
    call    tp_matmul_q8_0

    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2

    ; --- 3h. Post-attention RMSNorm + SwiGLU MLP + residual ---
    mov     rdi, [rbx + CTX_BUF_XB]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rsp + 48]             ; post_norm
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

    ; Fused gate_proj + up_proj in ONE parallel matmul (contiguous in weights & arena!)
    mov     rdi, [rsp + 88]             ; gate_proj (contiguous with up_proj)
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_BUF_GATE]   ; buf_gate (contiguous with buf_up)
    mov     rcx, [rbx + CTX_HIDDEN_DIM]
    add     rcx, rcx                    ; M = 2 * hidden_dim (6144)
    mov     r8,  [rbx + CTX_DIM]        ; K = dim (1024)
    call    tp_matmul_q8_0

    ; SwiGLU: buf_gate = silu(buf_gate) * buf_up
    mov     rdi, [rbx + CTX_BUF_GATE]
    mov     rsi, [rbx + CTX_BUF_UP]
    mov     rdx, [rbx + CTX_HIDDEN_DIM]
    call    silu_mul_avx2

    ; down_proj + residual
    mov     rdi, [rsp + 104]            ; down_proj
    mov     rsi, [rbx + CTX_BUF_GATE]
    mov     rdx, [rbx + CTX_BUF_XB]
    mov     rcx, [rbx + CTX_DIM]
    mov     r8,  [rbx + CTX_HIDDEN_DIM]
    call    tp_matmul_q8_0

    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_XB]
    mov     rdx, [rbx + CTX_DIM]
    call    vec_add_avx2

    inc     qword [rsp + 16]
    jmp     .layer_lp

.layers_done:
    cmp     qword [rsp + 8], 0
    je      .done

    ; 4. Final RMSNorm + tied LM-head projection -> buf_logits
    mov     rdi, [rbx + CTX_BUF_X]
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_FINAL_NORM_PTR]
    mov     rcx, [rbx + CTX_DIM]
    vmovss  xmm0, [rbx + CTX_RMS_EPS]
    call    rmsnorm_avx2

    mov     rdi, [rbx + CTX_EMB_PTR]    ; tied embeddings!
    mov     rsi, [rbx + CTX_BUF_X]
    mov     rdx, [rbx + CTX_BUF_LOGITS]
    mov     rcx, [rbx + CTX_VOCAB_SIZE]
    mov     r8,  [rbx + CTX_DIM]
    call    tp_matmul_q8_0

.done:
    add     rsp, 136
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

section .rodata
align 4
t_one: dd 1.0

section .bss
alignb 64
g_ctx: resb CTX_SIZE
