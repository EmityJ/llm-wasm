#!/usr/bin/env python3
"""Unit-test pure x86-64 assembly math & Q8_0 matmul kernels against Python/NumPy."""

import math
import subprocess
import sys
import numpy as np


def main():
    bin_path = sys.argv[1] if len(sys.argv) > 1 else "build/test_math"
    rng = np.random.default_rng(12345)

    # 1. expf inputs: cover [-100, +20] including edge cases, negatives, zeros, large neg
    in_exp = np.concatenate([
        np.array([-120.0, -88.0, -87.0, -50.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 80.0], dtype=np.float32),
        rng.uniform(-20.0, 15.0, size=48).astype(np.float32),
    ])

    # 2. sincosf inputs: cover [-5000, +5000] including RoPE-scale large angles
    in_sc = np.concatenate([
        np.array([-4096.0, -1000.0, -math.pi, -math.pi/2, -1.0, -0.1, 0.0, 0.1, 1.0, math.pi/2, math.pi, 10.0, 100.0, 1000.0, 4096.0, 12345.67], dtype=np.float32),
        rng.uniform(-500.0, 500.0, size=48).astype(np.float32),
    ])

    # 3. rmsnorm (n=128)
    in_rms_x = rng.normal(0.0, 2.0, size=128).astype(np.float32)
    in_rms_w = rng.normal(1.0, 0.3, size=128).astype(np.float32)

    # 4. silu_mul (n=64), including huge negative & positive gate values!
    in_gate = np.concatenate([
        np.array([-150.0, -90.0, -20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 90.0, 150.0, -0.001, 0.001, -50.0, 50.0, -100.0], dtype=np.float32),
        rng.normal(0.0, 5.0, size=48).astype(np.float32),
    ])
    in_up = rng.normal(0.0, 2.0, size=64).astype(np.float32)

    # 5. matmul Q8_0 x Q8_0 (8 rows x 64 cols)
    in_mm_x = rng.normal(0.0, 1.5, size=64).astype(np.float32)
    w_f32 = rng.normal(0.0, 0.5, size=(8, 64)).astype(np.float32)
    wb = w_f32.reshape(8, 2, 32)
    ws = (np.max(np.abs(wb), axis=2) / np.float32(127.0)).astype(np.float32)
    wq = np.clip(np.rint(wb / ws[:, :, None]), -127, 127).astype(np.int8)
    in_mm_w = np.concatenate([wq.reshape(8, 64).view(np.uint8), ws.view(np.uint8).reshape(8, 8)], axis=1).tobytes()

    payload = (
        in_exp.tobytes()
        + in_sc.tobytes()
        + in_rms_x.tobytes()
        + in_rms_w.tobytes()
        + in_gate.tobytes()
        + in_up.tobytes()
        + in_mm_x.tobytes()
        + in_mm_w
    )

    proc = subprocess.run([bin_path], input=payload, capture_output=True, check=True)
    raw = proc.stdout

    off = 0
    def take(dtype, count):
        nonlocal off
        dt = np.dtype(dtype)
        arr = np.frombuffer(raw, dtype=dt, count=count, offset=off)
        off += count * dt.itemsize
        return arr

    out_exp = take("<f4", 64)
    out_sin = take("<f4", 64)
    out_cos = take("<f4", 64)
    out_rope = take("<f8", 64)
    out_rms = take("<f4", 128)
    out_silu = take("<f4", 64)
    out_mm = take("<f4", 8)
    assert off == len(raw), f"Extra output bytes: {len(raw) - off}"

    # References
    ref_exp = np.array([0.0 if x < -87.33654 else math.exp(float(x)) for x in in_exp], dtype=np.float64)
    rel_exp = np.max(np.abs(out_exp.astype(np.float64) - ref_exp) / np.maximum(ref_exp, 1e-30))

    ref_sin = np.array([math.sin(float(x)) for x in in_sc], dtype=np.float64)
    ref_cos = np.array([math.cos(float(x)) for x in in_sc], dtype=np.float64)
    err_sin = np.max(np.abs(out_sin.astype(np.float64) - ref_sin))
    err_cos = np.max(np.abs(out_cos.astype(np.float64) - ref_cos))

    ref_rope = 1.0 / (1000000.0 ** (np.arange(0, 128, 2, dtype=np.float64) / 128.0))
    err_rope = np.max(np.abs(out_rope - ref_rope))

    ss = np.mean(in_rms_x.astype(np.float64) ** 2)
    ref_rms = (in_rms_x.astype(np.float64) / math.sqrt(ss + 1e-6)) * in_rms_w.astype(np.float64)
    err_rms = np.max(np.abs(out_rms.astype(np.float64) - ref_rms))

    ref_silu = np.array([
        (float(g) / (1.0 + math.exp(-float(g))) if g >= 0 else float(g) * math.exp(float(g)) / (1.0 + math.exp(float(g)))) * float(u)
        for g, u in zip(in_gate, in_up)
    ], dtype=np.float64)
    rel_silu = np.max(np.abs(out_silu.astype(np.float64) - ref_silu) / np.maximum(np.abs(ref_silu), 1.0))

    xb = in_mm_x.reshape(2, 32).astype(np.float32)
    ref_mm = np.sum(np.einsum("mnb,nb->mn", wq.astype(np.float32), xb) * ws, axis=1).astype(np.float32)
    err_mm = np.max(np.abs(out_mm - ref_mm))

    print(f"fast_expf_avx2   max_rel_err = {rel_exp:.3e}  (target < 1e-6)")
    print(f"fast_sincosf sin max_abs_err = {err_sin:.3e}  (target < 1e-6)")
    print(f"fast_sincosf cos max_abs_err = {err_cos:.3e}  (target < 1e-6)")
    print(f"init_rope_freqs  max_abs_err = {err_rope:.3e}  (target < 1e-12)")
    print(f"rmsnorm_avx2     max_abs_err = {err_rms:.3e}  (target < 1e-6)")
    print(f"silu_mul_avx2    max_rel_err = {rel_silu:.3e}  (target < 1e-6)")
    print(f"matmul_q8_0      max_abs_err = {err_mm:.3e}  (target < 1e-5)")

    assert rel_exp < 1e-6, f"expf error too high: {rel_exp}"
    assert err_sin < 1e-6, f"sinf error too high: {err_sin}"
    assert err_cos < 1e-6, f"cosf error too high: {err_cos}"
    assert err_rope < 1e-12, f"rope freq error too high: {err_rope}"
    assert err_rms < 1e-6, f"rmsnorm error too high: {err_rms}"
    assert rel_silu < 1e-6, f"silu error too high: {rel_silu}"
    assert err_mm < 1e-5, f"matmul error too high: {err_mm}"
    print("\nALL PURE-ASM MATH & SIMD KERNEL UNIT TESTS PASSED!")


if __name__ == "__main__":
    main()
