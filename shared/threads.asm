; ============================================================================
; threads.asm — Zero-libc Linux clone() + futex() thread pool for matmul_q8_0
;
; Architecture:
;   - Spawns (n_threads - 1) workers via raw SYS_clone (CLONE_VM|CLONE_FS|
;     CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD) with private mmap stacks.
;   - Lock-free cache-line-isolated completion array (tp_done_arr[tid * 64]):
;     each worker signals completion with a plain TSO release store to its own
;     64-byte cache line — eliminating cross-CCX atomic contention on EPYC!
;   - Hybrid spin-then-futex wakeup with Dekker fence: zero kernel syscalls
;     during active generation, zero idle CPU burn when paused.
;   - Deterministic 4-row-aligned static partitioning: outputs are BIT-FOR-BIT
;     identical across 1T..64T.
; ============================================================================

bits 64
default rel

%include "constants.inc"

%define SPIN_LIMIT  200000

extern sys_mmap_anon
extern matmul_q8_0_range

section .text

global tp_init
global tp_shutdown
global tp_matmul_q8_0

; ----------------------------------------------------------------------------
; void tp_init(uint64 n_threads)
; rdi = requested total thread count (clamped to [1, MAX_THREADS])
; ----------------------------------------------------------------------------
tp_init:
    push    rbx
    push    r12
    push    r13

    cmp     rdi, 1
    jge     .chk_max
    mov     rdi, 1
.chk_max:
    cmp     rdi, MAX_THREADS
    jle     .store_n
    mov     rdi, MAX_THREADS
.store_n:
    mov     [rel tp_n_threads], rdi
    mov     dword [rel tp_gen], 0
    mov     dword [rel tp_stop], 0
    mov     dword [rel tp_waiters], 0

    ; Clear per-worker completion slots
    lea     rax, [rel tp_done_arr]
    xor     ecx, ecx
.clr_lp:
    cmp     rcx, MAX_THREADS
    jae     .clr_done
    mov     rdx, rcx
    shl     rdx, 6                      ; tid * 64 bytes
    mov     dword [rax + rdx], 0
    inc     rcx
    jmp     .clr_lp
.clr_done:

    cmp     rdi, 1
    jle     .done

    mov     r12, rdi                    ; total threads
    mov     rbx, 1                      ; worker tid = 1 .. n_threads - 1
.spawn_lp:
    cmp     rbx, r12
    jae     .done

    mov     rdi, WORKER_STACK_SIZE
    call    sys_mmap_anon
    test    rax, rax
    js      .spawn_fail

    lea     rsi, [rax + WORKER_STACK_SIZE - 16]
    mov     [rsi], rbx                  ; pass tid on worker's own stack

    mov     edi, CLONE_THREAD_FLAGS
    xor     edx, edx
    xor     r10d, r10d
    xor     r8d, r8d
    mov     eax, SYS_clone
    syscall

    test    rax, rax
    jz      .worker_entry
    js      .spawn_fail

    inc     rbx
    jmp     .spawn_lp

.spawn_fail:
    mov     [rel tp_n_threads], rbx
.done:
    pop     r13
    pop     r12
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; Worker thread entrypoint (runs on its own private mmap stack!)
; ----------------------------------------------------------------------------
.worker_entry:
    pop     r12                         ; r12 = worker tid (1 .. n_threads-1)
    and     rsp, -16
    xor     r13d, r13d                  ; r13d = my_gen = 0

.worker_wait:
    mov     ecx, SPIN_LIMIT
.spin_lp:
    mov     eax, [rel tp_gen]
    cmp     eax, r13d
    jne     .got_job
    pause
    dec     ecx
    jnz     .spin_lp

    lock inc dword [rel tp_waiters]
    mov     eax, [rel tp_gen]
    cmp     eax, r13d
    jne     .unwait

    lea     rdi, [rel tp_gen]
    mov     esi, FUTEX_WAIT_PRIVATE
    mov     edx, r13d
    xor     r10d, r10d
    xor     r8d, r8d
    xor     r9d, r9d
    mov     eax, SYS_futex
    syscall

.unwait:
    lock dec dword [rel tp_waiters]
    jmp     .worker_wait

.got_job:
    cmp     dword [rel tp_stop], 0
    jne     .worker_exit
    mov     r13d, eax                   ; latch new generation

    ; Deterministic 4-row-aligned slice [r0, r1) for tid = r12
    mov     rcx, [rel tp_M]
    mov     r8, [rel tp_n_threads]
    mov     r9, rcx
    shr     r9, 2                       ; blocks4 = M / 4

    mov     rax, r9
    imul    rax, r12
    xor     edx, edx
    div     r8
    shl     rax, 2
    mov     r14, rax                    ; r0

    lea     rax, [r12 + 1]
    cmp     rax, r8
    je      .last_slice
    imul    rax, r9
    xor     edx, edx
    div     r8
    shl     rax, 2
    mov     r15, rax                    ; r1
    jmp     .run_slice
.last_slice:
    mov     r15, rcx                    ; r1 = M

.run_slice:
    cmp     r15, r14
    jle     .job_done

    mov     rdi, [rel tp_W]
    mov     rsi, [rel tp_x]
    mov     rdx, [rel tp_out]
    mov     rcx, [rel tp_K]
    mov     r8,  r14                    ; r0
    mov     r9,  r15                    ; r1
    call    matmul_q8_0_range

.job_done:
    ; Lock-free TSO release store to worker's private 64B cache line!
    lea     rax, [rel tp_done_arr]
    mov     rcx, r12
    shl     rcx, 6                      ; tid * 64
    mov     [rax + rcx], r13d
    jmp     .worker_wait

.worker_exit:
    xor     edi, edi
    mov     eax, SYS_exit
    syscall
    hlt

; ----------------------------------------------------------------------------
; void tp_shutdown(void)
; ----------------------------------------------------------------------------
tp_shutdown:
    cmp     qword [rel tp_n_threads], 1
    jle     .ret
    mov     dword [rel tp_stop], 1
    lock inc dword [rel tp_gen]
    lea     rdi, [rel tp_gen]
    mov     esi, FUTEX_WAKE_PRIVATE
    mov     edx, FUTEX_WAKE_ALL
    xor     r10d, r10d
    xor     r8d, r8d
    xor     r9d, r9d
    mov     eax, SYS_futex
    syscall
.ret:
    ret

; ----------------------------------------------------------------------------
; void tp_matmul_q8_0(const void *W, const float *x, float *out, uint64 M, uint64 K)
; rdi = W, rsi = x (f32), rdx = out (f32), rcx = M, r8 = K
; ----------------------------------------------------------------------------
tp_matmul_q8_0:
    mov     rax, [rel tp_n_threads]
    cmp     rax, 1
    jle     .single_thread

    push    rbx
    push    r12
    sub     rsp, 8                      ; 16B aligned
    mov     rbx, rax                    ; n_threads

    mov     [rel tp_W], rdi
    mov     [rel tp_x], rsi
    mov     [rel tp_out], rdx
    mov     [rel tp_M], rcx
    mov     [rel tp_K], r8

    ; Release barrier: atomically increment generation and capture new gen in r12d
    mov     r12d, 1
    lock xadd dword [rel tp_gen], r12d
    inc     r12d                        ; r12d = new_gen

    cmp     dword [rel tp_waiters], 0
    je      .run_main_slice
    lea     rdi, [rel tp_gen]
    mov     esi, FUTEX_WAKE_PRIVATE
    mov     edx, FUTEX_WAKE_ALL
    xor     r10d, r10d
    xor     r8d, r8d
    xor     r9d, r9d
    mov     eax, SYS_futex
    syscall

.run_main_slice:
    mov     rax, [rel tp_M]
    shr     rax, 2                      ; blocks4
    xor     edx, edx
    div     rbx
    shl     rax, 2                      ; r1 for tid = 0
    test    rax, rax
    jz      .wait_workers

    mov     rdi, [rel tp_W]
    mov     rsi, [rel tp_x]
    mov     rdx, [rel tp_out]
    mov     rcx, [rel tp_K]
    xor     r8d, r8d                    ; r0 = 0
    mov     r9,  rax                    ; r1
    call    matmul_q8_0_range

.wait_workers:
    ; Wait for each worker tid = 1 .. n_threads-1 to store new_gen in its cache line
    lea     rdi, [rel tp_done_arr]
    mov     rcx, 1
.wait_tid:
    cmp     rcx, rbx
    jae     .all_done
    mov     rdx, rcx
    shl     rdx, 6                      ; tid * 64
.spin_tid:
    cmp     dword [rdi + rdx], r12d
    je      .next_tid
    pause
    jmp     .spin_tid
.next_tid:
    inc     rcx
    jmp     .wait_tid

.all_done:
    lfence
    add     rsp, 8
    pop     r12
    pop     rbx
    ret

.single_thread:
    mov     r9,  rcx                    ; r1 = M
    mov     rcx, r8                     ; K
    xor     r8d, r8d                    ; r0 = 0
    jmp     matmul_q8_0_range

section .bss
alignb 64
tp_gen:       resd 1
tp_stop:      resd 1
tp_n_threads: resq 1
tp_W:         resq 1
tp_x:         resq 1
tp_out:       resq 1
tp_M:         resq 1
tp_K:         resq 1

alignb 64
tp_waiters:   resd 1
resb 60

alignb 64
; 64-byte isolated completion slot per worker (MAX_THREADS * 64 = 4096 bytes in .bss)
tp_done_arr:  resb MAX_THREADS * 64
