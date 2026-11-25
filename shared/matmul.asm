; ============================================================================
; matmul.asm — High-throughput AVX2 + FMA Q8_0 x F32 kernels (zero libc)
;
; Weight layout (per-row split Q8_0, block size = 32, int8 in [-127, 127]):
;   row_stride = K + (K / 32) * 4 = K + (K >> 3) bytes
;   [row + 0 .. row + K - 1]         : K int8 weights (32-byte aligned!)
;   [row + K .. row + row_stride - 1]: K/32 fp32 block scales (4-byte aligned!)
;
; Functions:
;   dequant_row_q8_0  : (const void *row_ptr, float *out, uint64 K)
;   matmul_q8_0_range : (const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
;   dot_f32_avx2      : (const float *a, const float *b, uint64 n) -> xmm0
;   saxpy_f32_avx2    : (float *dst, const float *src, uint64 n, float alpha)
; ============================================================================

bits 64
default rel

section .text

global dequant_row_q8_0
global matmul_q8_0_range
global dot_f32_avx2
global saxpy_f32_avx2

; ----------------------------------------------------------------------------
; void dequant_row_q8_0(const void *row_ptr, float *out, uint64 K)
; rdi = row_ptr ([K int8][K/32 fp32]), rsi = out, rdx = K (multiple of 32)
; ----------------------------------------------------------------------------
dequant_row_q8_0:
    lea     r8, [rdi + rdx]             ; scales_ptr = row_ptr + K
    mov     rcx, rdx
    shr     rcx, 5                      ; nb = K / 32
    xor     eax, eax                    ; b = 0
.lp:
    cmp     rax, rcx
    jae     .done
    vbroadcastss ymm4, [r8 + rax*4]     ; scale[b]
    mov     r9, rax
    shl     r9, 5                       ; b * 32 (q offset)
    mov     r10, rax
    shl     r10, 7                      ; b * 128 (out offset)

    vpmovsxbd ymm0, qword [rdi + r9]
    vpmovsxbd ymm1, qword [rdi + r9 + 8]
    vpmovsxbd ymm2, qword [rdi + r9 + 16]
    vpmovsxbd ymm3, qword [rdi + r9 + 24]
    vcvtdq2ps ymm0, ymm0
    vcvtdq2ps ymm1, ymm1
    vcvtdq2ps ymm2, ymm2
    vcvtdq2ps ymm3, ymm3
    vmulps  ymm0, ymm0, ymm4
    vmulps  ymm1, ymm1, ymm4
    vmulps  ymm2, ymm2, ymm4
    vmulps  ymm3, ymm3, ymm4
    vmovups [rsi + r10], ymm0
    vmovups [rsi + r10 + 32], ymm1
    vmovups [rsi + r10 + 64], ymm2
    vmovups [rsi + r10 + 96], ymm3

    inc     rax
    jmp     .lp
.done:
    ret

; ----------------------------------------------------------------------------
; void matmul_q8_0_range(const void *W, const float *x, float *out,
;                        uint64 K, uint64 r0, uint64 r1)
; rdi = W, rsi = x (f32), rdx = out (f32), rcx = K, r8 = r0, r9 = r1
;
; 4-row unrolled AVX2 + FMA Q8_0 x FP32 matrix-vector multiplication:
;   - Full FP32 activations (zero activation quantization noise or rounding cliffs!)
;   - Loads x[b*32 .. b*32+31] ONCE per block and reuses across 4 rows
;   - Interleaves 2 rows at a time across ymm8..ymm15 to hide convert+FMA latency
;   - ZERO stack accesses inside inner block loop
;   - Bit-for-bit identical across any thread count (1T..64T)
; ----------------------------------------------------------------------------
matmul_q8_0_range:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 40
    ; Stack locals (touched only once per 4 rows):
    ;   [rsp + 0]  = out_ptr (advances by 16 per 4 rows)
    ;   [rsp + 8]  = rows_left (r1 - r0)
    ;   [rsp + 16] = row_stride = K + (K >> 3)
    ;   [rsp + 24] = stride4 = 4 * row_stride
    ;   [rsp + 32] = base_w_cur (advances by stride4)

    mov     rax, r9
    sub     rax, r8                     ; rows_left = r1 - r0
    jle     .exit
    mov     [rsp + 8], rax

    ; row_stride = K + (K >> 3)
    mov     rbx, rcx
    shr     rbx, 3
    add     rbx, rcx
    mov     [rsp + 16], rbx             ; row_stride
    lea     rbp, [rbx * 4]
    mov     [rsp + 24], rbp             ; 4 * row_stride

    ; out_ptr = out + r0 * 4
    lea     rdx, [rdx + r8 * 4]
    mov     [rsp + 0], rdx

    ; base_w_cur = W + r0 * row_stride
    mov     rax, r8
    imul    rax, rbx
    add     rdi, rax
    mov     [rsp + 32], rdi

    ; Persistent registers across all row groups
    mov     r11, rsi                    ; x_ptr (float*)
    mov     r15, rcx                    ; K (byte limit for b_q)

.row4_lp:
    cmp     qword [rsp + 8], 4
    jb      .row1_check

    mov     rax, [rsp + 32]             ; w0
    mov     rbx, [rsp + 16]             ; row_stride
    lea     rcx, [rax + rbx]            ; w1
    lea     rdx, [rcx + rbx]            ; w2
    lea     rsi, [rdx + rbx]            ; w3

    lea     rdi, [rax + r15]            ; s0 = w0 + K
    lea     r8,  [rcx + r15]            ; s1 = w1 + K
    lea     r9,  [rdx + r15]            ; s2 = w2 + K
    lea     r10, [rsi + r15]            ; s3 = w3 + K

    vxorps  ymm0, ymm0, ymm0            ; acc0
    vxorps  ymm1, ymm1, ymm1            ; acc1
    vxorps  ymm2, ymm2, ymm2            ; acc2
    vxorps  ymm3, ymm3, ymm3            ; acc3

    xor     r12d, r12d                  ; b_x = 0, 128, ..., K*4 - 128
    xor     r13d, r13d                  ; b_q = 0, 32,  ..., K - 32
    xor     r14d, r14d                  ; b_s = 0, 4,   ..., K/8 - 4

.blk4_lp:
    ; Load 32 FP32 activations ONCE (reused across all 4 rows!)
    vmovups ymm4, [r11 + r12]
    vmovups ymm5, [r11 + r12 + 32]
    vmovups ymm6, [r11 + r12 + 64]
    vmovups ymm7, [r11 + r12 + 96]

    ; --- Interleaved Rows 0 & 1 ---
    vpmovsxbd ymm8,  qword [rax + r13]
    vpmovsxbd ymm12, qword [rcx + r13]
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm12, ymm12
    vmulps    ymm8,  ymm8,  ymm4
    vmulps    ymm12, ymm12, ymm4

    vpmovsxbd ymm9,  qword [rax + r13 + 8]
    vpmovsxbd ymm13, qword [rcx + r13 + 8]
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm13, ymm13
    vfmadd231ps ymm8,  ymm9,  ymm5
    vfmadd231ps ymm12, ymm13, ymm5

    vpmovsxbd ymm10, qword [rax + r13 + 16]
    vpmovsxbd ymm14, qword [rcx + r13 + 16]
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm14, ymm14
    vfmadd231ps ymm8,  ymm10, ymm6
    vfmadd231ps ymm12, ymm14, ymm6

    vpmovsxbd ymm11, qword [rax + r13 + 24]
    vpmovsxbd ymm15, qword [rcx + r13 + 24]
    vcvtdq2ps ymm11, ymm11
    vcvtdq2ps ymm15, ymm15
    vfmadd231ps ymm8,  ymm11, ymm7
    vfmadd231ps ymm12, ymm15, ymm7

    vbroadcastss ymm9,  dword [rdi + r14]
    vbroadcastss ymm13, dword [r8 + r14]
    vfmadd231ps ymm0, ymm8,  ymm9
    vfmadd231ps ymm1, ymm12, ymm13

    ; --- Interleaved Rows 2 & 3 ---
    vpmovsxbd ymm8,  qword [rdx + r13]
    vpmovsxbd ymm12, qword [rsi + r13]
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm12, ymm12
    vmulps    ymm8,  ymm8,  ymm4
    vmulps    ymm12, ymm12, ymm4

    vpmovsxbd ymm9,  qword [rdx + r13 + 8]
    vpmovsxbd ymm13, qword [rsi + r13 + 8]
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm13, ymm13
    vfmadd231ps ymm8,  ymm9,  ymm5
    vfmadd231ps ymm12, ymm13, ymm5

    vpmovsxbd ymm10, qword [rdx + r13 + 16]
    vpmovsxbd ymm14, qword [rsi + r13 + 16]
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm14, ymm14
    vfmadd231ps ymm8,  ymm10, ymm6
    vfmadd231ps ymm12, ymm14, ymm6

    vpmovsxbd ymm11, qword [rdx + r13 + 24]
    vpmovsxbd ymm15, qword [rsi + r13 + 24]
    vcvtdq2ps ymm11, ymm11
    vcvtdq2ps ymm15, ymm15
    vfmadd231ps ymm8,  ymm11, ymm7
    vfmadd231ps ymm12, ymm15, ymm7

    vbroadcastss ymm9,  dword [r9 + r14]
    vbroadcastss ymm13, dword [r10 + r14]
    vfmadd231ps ymm2, ymm8,  ymm9
    vfmadd231ps ymm3, ymm12, ymm13

    add     r12, 128
    add     r13, 32
    add     r14, 4
    cmp     r13, r15
    jb      .blk4_lp

    ; Horizontal sum of 4 accumulators (ymm0..ymm3 -> 4 floats in xmm0)
    vhaddps ymm0, ymm0, ymm1
    vhaddps ymm2, ymm2, ymm3
    vhaddps ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1

    mov     rax, [rsp + 0]
    vmovups [rax], xmm0
    add     rax, 16
    mov     [rsp + 0], rax

    mov     rax, [rsp + 24]             ; stride4
    add     [rsp + 32], rax
    sub     qword [rsp + 8], 4
    jmp     .row4_lp

    ; Tail loop for 1..3 remaining rows
.row1_check:
    cmp     qword [rsp + 8], 0
    je      .exit
.row1_lp:
    mov     rax, [rsp + 32]             ; w0
    lea     rdi, [rax + r15]            ; s0
    vxorps  ymm0, ymm0, ymm0
    xor     r12d, r12d
    xor     r13d, r13d
    xor     r14d, r14d
.blk1_lp:
    vpmovsxbd ymm8,  qword [rax + r13]
    vpmovsxbd ymm9,  qword [rax + r13 + 8]
    vpmovsxbd ymm10, qword [rax + r13 + 16]
    vpmovsxbd ymm11, qword [rax + r13 + 24]
    vcvtdq2ps ymm8,  ymm8
    vcvtdq2ps ymm9,  ymm9
    vcvtdq2ps ymm10, ymm10
    vcvtdq2ps ymm11, ymm11
    vmulps    ymm8,  ymm8,  [r11 + r12]
    vfmadd231ps ymm8, ymm9,  [r11 + r12 + 32]
    vfmadd231ps ymm8, ymm10, [r11 + r12 + 64]
    vfmadd231ps ymm8, ymm11, [r11 + r12 + 96]
    vbroadcastss ymm12, dword [rdi + r14]
    vfmadd231ps ymm0, ymm8, ymm12

    add     r12, 128
    add     r13, 32
    add     r14, 4
    cmp     r13, r15
    jb      .blk1_lp

    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    mov     rax, [rsp + 0]
    vmovss  [rax], xmm0
    add     rax, 4
    mov     [rsp + 0], rax
    mov     rax, [rsp + 16]
    add     [rsp + 32], rax
    dec     qword [rsp + 8]
    jnz     .row1_lp

.exit:
    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; float dot_f32_avx2(const float *a, const float *b, uint64 n) -> xmm0
; rdi = a, rsi = b, rdx = n (multiple of 32, e.g. head_dim = 128)
; ----------------------------------------------------------------------------
dot_f32_avx2:
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .red
    vmovups ymm4, [rdi + rax]
    vmovups ymm5, [rdi + rax + 32]
    vmovups ymm6, [rdi + rax + 64]
    vmovups ymm7, [rdi + rax + 96]
    vfmadd231ps ymm0, ymm4, [rsi + rax]
    vfmadd231ps ymm1, ymm5, [rsi + rax + 32]
    vfmadd231ps ymm2, ymm6, [rsi + rax + 64]
    vfmadd231ps ymm3, ymm7, [rsi + rax + 96]
    add     rax, 128
    jmp     .lp
.red:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    ret

; ----------------------------------------------------------------------------
; void saxpy_f32_avx2(float *dst, const float *src, uint64 n, float alpha)
; rdi = dst, rsi = src, rdx = n (multiple of 32, e.g. head_dim = 128), xmm0 = alpha
; dst[i] += alpha * src[i]
; ----------------------------------------------------------------------------
saxpy_f32_avx2:
    vbroadcastss ymm0, xmm0
    shl     rdx, 2
    xor     eax, eax
.lp:
    cmp     rax, rdx
    jae     .done
    vmovups ymm1, [rdi + rax]
    vmovups ymm2, [rdi + rax + 32]
    vmovups ymm3, [rdi + rax + 64]
    vmovups ymm4, [rdi + rax + 96]
    vfmadd231ps ymm1, ymm0, [rsi + rax]
    vfmadd231ps ymm2, ymm0, [rsi + rax + 32]
    vfmadd231ps ymm3, ymm0, [rsi + rax + 64]
    vfmadd231ps ymm4, ymm0, [rsi + rax + 96]
    vmovups [rdi + rax], ymm1
    vmovups [rdi + rax + 32], ymm2
    vmovups [rdi + rax + 64], ymm3
    vmovups [rdi + rax + 96], ymm4
    add     rax, 128
    jmp     .lp
.done:
    ret
