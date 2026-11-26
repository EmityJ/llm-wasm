(module
  (import "math" "exp" (func $exp (param f64) (result f64)))
  (import "math" "sin" (func $sin (param f64) (result f64)))
  (import "math" "cos" (func $cos (param f64) (result f64)))
  (memory (export "memory") 1 32768)

  ;; 运行时配置位于线性内存起始处，由 qwen3.mjs 初始化。
  (func $at (param $base i32) (param $index i32) (result i32)
    (i32.add (local.get $base) (i32.shl (local.get $index) (i32.const 2))))

  (func $sum_lanes (param $x v128) (result f32)
    (f32.add
      (f32.add (f32x4.extract_lane 0 (local.get $x)) (f32x4.extract_lane 1 (local.get $x)))
      (f32.add (f32x4.extract_lane 2 (local.get $x)) (f32x4.extract_lane 3 (local.get $x)))))

  ;; ASMBIN v1 使用 FP32 缩放，v2 使用 IEEE 754 binary16。
  (func $scale (param $p i32) (param $bytes i32) (result f32)
    (local $half i32) (local $sign i32) (local $magnitude i32) (local $exponent i32)
    (if (i32.eq (local.get $bytes) (i32.const 4))
      (then (return (f32.load (local.get $p)))))
    (local.set $half (i32.load16_u (local.get $p)))
    (local.set $sign (i32.shl (i32.and (local.get $half) (i32.const 0x8000)) (i32.const 16)))
    (local.set $magnitude (i32.and (local.get $half) (i32.const 0x7fff)))
    (local.set $exponent (i32.and (local.get $half) (i32.const 0x7c00)))
    (if (i32.eqz (local.get $exponent))
      (then
        (return
          (f32.copysign
            (f32.mul (f32.convert_i32_u (local.get $magnitude)) (f32.const 0x1p-24))
            (f32.reinterpret_i32 (local.get $sign))))))
    (if (i32.eq (local.get $exponent) (i32.const 0x7c00))
      (then
        (return
          (f32.reinterpret_i32
            (i32.or (local.get $sign)
              (i32.or (i32.const 0x7f800000)
                (i32.shl (i32.and (local.get $magnitude) (i32.const 0x3ff)) (i32.const 13))))))))
    (f32.reinterpret_i32
      (i32.or (local.get $sign)
        (i32.add (i32.shl (local.get $magnitude) (i32.const 13)) (i32.const 0x38000000)))))

  (func $dequant_row (param $dst i32) (param $w i32) (param $cols i32) (param $scale_bytes i32)
    (local $scale_ptr i32) (local $end i32) (local $block_end i32)
    (local $factor f32)
    (local.set $end (i32.add (local.get $w) (local.get $cols)))
    (local.set $scale_ptr (local.get $end))
    (loop $block
      (local.set $factor (call $scale (local.get $scale_ptr) (local.get $scale_bytes)))
      (local.set $block_end (i32.add (local.get $w) (i32.const 32)))
      (loop $element
        (f32.store (local.get $dst)
          (f32.mul (local.get $factor) (f32.convert_i32_s (i32.load8_s (local.get $w)))))
        (local.set $dst (i32.add (local.get $dst) (i32.const 4)))
        (local.set $w (i32.add (local.get $w) (i32.const 1)))
        (br_if $element (i32.lt_u (local.get $w) (local.get $block_end))))
      (local.set $scale_ptr (i32.add (local.get $scale_ptr) (local.get $scale_bytes)))
      (br_if $block (i32.lt_u (local.get $w) (local.get $end)))))

  (func $dot_q8_8 (param $w i32) (param $x i32) (result v128)
    (local $q v128)
    (local.set $q (v128.load8x8_s (local.get $w)))
    (f32x4.add
      (f32x4.mul
        (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $q)))
        (v128.load (local.get $x)))
      (f32x4.mul
        (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $q)))
        (v128.load offset=16 (local.get $x)))))

  ;; 每行以 32 个有符号字节为一块，所有块的缩放因子紧随权重字节。
  (func $matvec
    (param $dst i32) (param $weights i32) (param $x i32)
    (param $rows i32) (param $cols i32) (param $scale_bytes i32)
    (local $dst_end i32) (local $w i32) (local $input i32) (local $scale_ptr i32) (local $w_end i32)
    (local $sum v128) (local $dot v128)
    (local.set $dst_end (call $at (local.get $dst) (local.get $rows)))
    (loop $row
      (local.set $w (local.get $weights))
      (local.set $w_end (i32.add (local.get $weights) (local.get $cols)))
      (local.set $scale_ptr (local.get $w_end))
      (local.set $input (local.get $x))
      (local.set $sum (v128.const f32x4 0 0 0 0))
      (loop $block
        (local.set $dot
          (f32x4.add
            (f32x4.add
              (call $dot_q8_8 (local.get $w) (local.get $input))
              (call $dot_q8_8
                (i32.add (local.get $w) (i32.const 8))
                (i32.add (local.get $input) (i32.const 32))))
            (f32x4.add
              (call $dot_q8_8
                (i32.add (local.get $w) (i32.const 16))
                (i32.add (local.get $input) (i32.const 64)))
              (call $dot_q8_8
                (i32.add (local.get $w) (i32.const 24))
                (i32.add (local.get $input) (i32.const 96))))))
        (local.set $sum
          (f32x4.add (local.get $sum)
            (f32x4.mul (local.get $dot)
              (f32x4.splat (call $scale (local.get $scale_ptr) (local.get $scale_bytes))))))
        (local.set $w (i32.add (local.get $w) (i32.const 32)))
        (local.set $input (i32.add (local.get $input) (i32.const 128)))
        (local.set $scale_ptr (i32.add (local.get $scale_ptr) (local.get $scale_bytes)))
        (br_if $block (i32.lt_u (local.get $w) (local.get $w_end))))
      (f32.store (local.get $dst) (call $sum_lanes (local.get $sum)))
      (local.set $weights (local.get $scale_ptr))
      (local.set $dst (i32.add (local.get $dst) (i32.const 4)))
      (br_if $row (i32.lt_u (local.get $dst) (local.get $dst_end)))))

  (func $dot_f32 (param $a i32) (param $b i32) (param $n i32) (result f32)
    (local $end i32)
    (local $sum v128)
    (local.set $end (call $at (local.get $a) (local.get $n)))
    (loop $chunk
      (local.set $sum
        (f32x4.add (local.get $sum)
          (f32x4.mul (v128.load (local.get $a)) (v128.load (local.get $b)))))
      (local.set $a (i32.add (local.get $a) (i32.const 16)))
      (local.set $b (i32.add (local.get $b) (i32.const 16)))
      (br_if $chunk (i32.lt_u (local.get $a) (local.get $end))))
    (call $sum_lanes (local.get $sum)))

  (func $rmsnorm (param $dst i32) (param $src i32) (param $weights i32) (param $n i32) (param $eps f32)
    (local $end i32)
    (local $inv v128)
    (local.set $inv
      (f32x4.splat
        (f32.div (f32.const 1)
          (f32.sqrt
            (f32.add (local.get $eps)
              (f32.div (call $dot_f32 (local.get $src) (local.get $src) (local.get $n))
                (f32.convert_i32_u (local.get $n))))))))
    (local.set $end (call $at (local.get $dst) (local.get $n)))
    (loop $chunk
      (v128.store (local.get $dst)
        (f32x4.mul (v128.load (local.get $weights))
          (f32x4.mul (v128.load (local.get $src)) (local.get $inv))))
      (local.set $dst (i32.add (local.get $dst) (i32.const 16)))
      (local.set $src (i32.add (local.get $src) (i32.const 16)))
      (local.set $weights (i32.add (local.get $weights) (i32.const 16)))
      (br_if $chunk (i32.lt_u (local.get $dst) (local.get $end)))))

  (func $axpy (param $dst i32) (param $src i32) (param $n i32) (param $factor f32)
    (local $end i32)
    (local $v v128)
    (local.set $end (call $at (local.get $dst) (local.get $n)))
    (local.set $v (f32x4.splat (local.get $factor)))
    (loop $chunk
      (v128.store (local.get $dst)
        (f32x4.add (v128.load (local.get $dst)) (f32x4.mul (v128.load (local.get $src)) (local.get $v))))
      (local.set $dst (i32.add (local.get $dst) (i32.const 16)))
      (local.set $src (i32.add (local.get $src) (i32.const 16)))
      (br_if $chunk (i32.lt_u (local.get $dst) (local.get $end)))))

  (func $exp_f32 (param $x f32) (result f32)
    (f32.demote_f64 (call $exp (f64.promote_f32 (local.get $x)))))

  (func $silu_mul (param $gate i32) (param $up i32) (param $n i32)
    (local $end i32)
    (local $x f32) (local $e f32) (local $numerator f32)
    (local.set $end (call $at (local.get $gate) (local.get $n)))
    (loop $element
      (local.set $x (f32.load (local.get $gate)))
      (local.set $e (call $exp_f32 (f32.neg (f32.abs (local.get $x)))))
      (local.set $numerator
        (f32.mul (local.get $x)
          (select (local.get $e) (f32.const 1) (f32.lt (local.get $x) (f32.const 0)))))
      (f32.store (local.get $gate)
        (f32.mul (f32.load (local.get $up))
          (f32.div (local.get $numerator) (f32.add (f32.const 1) (local.get $e)))))
      (local.set $gate (i32.add (local.get $gate) (i32.const 4)))
      (local.set $up (i32.add (local.get $up) (i32.const 4)))
      (br_if $element (i32.lt_u (local.get $gate) (local.get $end)))))

  ;; 每个位置的 RoPE 系数在所有层和头之间共享。
  (func $rope_angles (param $pos i32) (param $half i32) (param $freq i32) (param $cosines i32) (param $sines i32)
    (local $index i32)
    (local $angle f64)
    (loop $pair
      (local.set $angle
        (f64.mul (f64.convert_i32_u (local.get $pos))
          (f64.load (i32.add (local.get $freq) (i32.shl (local.get $index) (i32.const 3))))))
      (f32.store (call $at (local.get $cosines) (local.get $index))
        (f32.demote_f64 (call $cos (local.get $angle))))
      (f32.store (call $at (local.get $sines) (local.get $index))
        (f32.demote_f64 (call $sin (local.get $angle))))
      (local.set $index (i32.add (local.get $index) (i32.const 1)))
      (br_if $pair (i32.lt_u (local.get $index) (local.get $half)))))

  (func $norm_rope
    (param $x i32) (param $weights i32) (param $heads i32)
    (param $width i32) (param $cosines i32) (param $sines i32)
    (param $eps f32)
    (local $half i32) (local $end i32) (local $first i32) (local $second i32) (local $index i32)
    (local $a f32) (local $b f32) (local $c f32) (local $s f32)
    (local.set $half (i32.shr_u (local.get $width) (i32.const 1)))
    (local.set $end (call $at (local.get $x) (i32.mul (local.get $heads) (local.get $width))))
    (loop $head
      (call $rmsnorm (local.get $x) (local.get $x) (local.get $weights) (local.get $width) (local.get $eps))
      (local.set $index (i32.const 0))
      (loop $pair
        (local.set $first (call $at (local.get $x) (local.get $index)))
        (local.set $second (call $at (local.get $first) (local.get $half)))
        (local.set $a (f32.load (local.get $first)))
        (local.set $b (f32.load (local.get $second)))
        (local.set $c (f32.load (call $at (local.get $cosines) (local.get $index))))
        (local.set $s (f32.load (call $at (local.get $sines) (local.get $index))))
        (f32.store (local.get $first)
          (f32.sub (f32.mul (local.get $a) (local.get $c)) (f32.mul (local.get $b) (local.get $s))))
        (f32.store (local.get $second)
          (f32.add (f32.mul (local.get $b) (local.get $c)) (f32.mul (local.get $a) (local.get $s))))
        (local.set $index (i32.add (local.get $index) (i32.const 1)))
        (br_if $pair (i32.lt_u (local.get $index) (local.get $half))))
      (local.set $x (call $at (local.get $x) (local.get $width)))
      (br_if $head (i32.lt_u (local.get $x) (local.get $end)))))

  ;; KV 缓存按层、位置、KV 头排列。每组查询头共享同一个 KV 头。
  (func $attention
    (param $dst i32) (param $q i32) (param $keys i32)
    (param $values i32) (param $scores i32) (param $heads i32)
    (param $kv_heads i32) (param $width i32) (param $pos i32)
    (local $head i32) (local $group i32) (local $stride i32)
    (local $kv_offset i32) (local $k i32) (local $v i32)
    (local $s i32) (local $end i32)
    (local $factor f32) (local $max f32) (local $value f32) (local $sum f32)
    (local.set $group (i32.div_u (local.get $heads) (local.get $kv_heads)))
    (local.set $stride (call $at (i32.const 0) (i32.mul (local.get $kv_heads) (local.get $width))))
    (local.set $end (call $at (local.get $scores) (i32.add (local.get $pos) (i32.const 1))))
    (local.set $factor (f32.div (f32.const 1) (f32.sqrt (f32.convert_i32_u (local.get $width)))))
    (loop $head_loop
      (local.set $kv_offset (i32.mul (i32.div_u (local.get $head) (local.get $group)) (local.get $width)))
      (local.set $k (call $at (local.get $keys) (local.get $kv_offset)))
      (local.set $s (local.get $scores))
      (local.set $max (f32.const -inf))
      (loop $score
        (local.set $value
          (f32.mul (local.get $factor) (call $dot_f32 (local.get $q) (local.get $k) (local.get $width))))
        (f32.store (local.get $s) (local.get $value))
        (local.set $max (f32.max (local.get $max) (local.get $value)))
        (local.set $k (i32.add (local.get $k) (local.get $stride)))
        (local.set $s (i32.add (local.get $s) (i32.const 4)))
        (br_if $score (i32.lt_u (local.get $s) (local.get $end))))
      (local.set $s (local.get $scores))
      (local.set $sum (f32.const 0))
      (loop $softmax
        (local.set $value (call $exp_f32 (f32.sub (f32.load (local.get $s)) (local.get $max))))
        (f32.store (local.get $s) (local.get $value))
        (local.set $sum (f32.add (local.get $sum) (local.get $value)))
        (local.set $s (i32.add (local.get $s) (i32.const 4)))
        (br_if $softmax (i32.lt_u (local.get $s) (local.get $end))))
      (memory.fill (local.get $dst) (i32.const 0) (call $at (i32.const 0) (local.get $width)))
      (local.set $s (local.get $scores))
      (local.set $v (call $at (local.get $values) (local.get $kv_offset)))
      (loop $weighted_sum
        (call $axpy (local.get $dst) (local.get $v) (local.get $width)
          (f32.div (f32.load (local.get $s)) (local.get $sum)))
        (local.set $v (i32.add (local.get $v) (local.get $stride)))
        (local.set $s (i32.add (local.get $s) (i32.const 4)))
        (br_if $weighted_sum (i32.lt_u (local.get $s) (local.get $end))))
      (local.set $dst (call $at (local.get $dst) (local.get $width)))
      (local.set $q (call $at (local.get $q) (local.get $width)))
      (local.set $head (i32.add (local.get $head) (i32.const 1)))
      (br_if $head_loop (i32.lt_u (local.get $head) (local.get $heads)))))

  (func (export "forward") (param $token i32) (param $pos i32) (param $compute_logits i32)
    (local $dim i32) (local $hidden i32) (local $layer_end i32)
    (local $heads i32) (local $kv_heads i32) (local $width i32)
    (local $q_dim i32) (local $kv_dim i32) (local $scale_bytes i32)
    (local $embed i32) (local $layer i32) (local $x i32)
    (local $h i32) (local $q i32) (local $k i32)
    (local $v i32) (local $att i32) (local $scores i32)
    (local $gate i32) (local $up i32) (local $cosines i32)
    (local $sines i32)
    (local $keys i32) (local $values i32) (local $kv_bytes i32) (local $kv_pos i32) (local $kv_layer_bytes i32)
    (local $eps f32)

    (local.set $dim (i32.load (i32.const 0)))
    (local.set $hidden (i32.load (i32.const 4)))
    (local.set $heads (i32.load (i32.const 12)))
    (local.set $kv_heads (i32.load (i32.const 16)))
    (local.set $width (i32.load (i32.const 20)))
    (local.set $eps (f32.load (i32.const 32)))
    (local.set $scale_bytes (i32.load (i32.const 36)))
    (local.set $embed (i32.load (i32.const 40)))
    (local.set $layer (i32.load (i32.const 44)))
    (local.set $x (i32.load (i32.const 52)))
    (local.set $h (i32.load (i32.const 56)))
    (local.set $q (i32.load (i32.const 60)))
    (local.set $k (i32.load (i32.const 64)))
    (local.set $v (i32.load (i32.const 68)))
    (local.set $att (i32.load (i32.const 72)))
    (local.set $scores (i32.load (i32.const 76)))
    (local.set $gate (i32.load (i32.const 80)))
    (local.set $up (i32.load (i32.const 84)))
    (local.set $cosines (i32.load (i32.const 96)))
    (local.set $sines (i32.load (i32.const 100)))
    (local.set $keys (i32.load (i32.const 104)))
    (local.set $values (i32.load (i32.const 108)))

    (local.set $layer_end
      (i32.add (local.get $layer) (i32.mul (i32.load (i32.const 8)) (i32.const 44))))
    (local.set $q_dim (i32.mul (local.get $heads) (local.get $width)))
    (local.set $kv_dim (i32.mul (local.get $kv_heads) (local.get $width)))
    (local.set $kv_bytes (call $at (i32.const 0) (local.get $kv_dim)))
    (local.set $kv_layer_bytes (i32.mul (local.get $kv_bytes) (i32.load (i32.const 28))))
    (local.set $kv_pos (i32.mul (local.get $kv_bytes) (local.get $pos)))

    (call $dequant_row (local.get $x)
      (i32.add (local.get $embed)
        (i32.mul (local.get $token)
          (i32.add (local.get $dim)
            (i32.mul (i32.shr_u (local.get $dim) (i32.const 5)) (local.get $scale_bytes)))))
      (local.get $dim) (local.get $scale_bytes))
    (call $rope_angles (local.get $pos) (i32.shr_u (local.get $width) (i32.const 1))
      (i32.load (i32.const 92)) (local.get $cosines) (local.get $sines))

    (loop $layers
      (call $rmsnorm (local.get $h) (local.get $x) (i32.load (local.get $layer)) (local.get $dim) (local.get $eps))
      (call $matvec (local.get $q) (i32.load offset=16 (local.get $layer)) (local.get $h)
        (local.get $q_dim) (local.get $dim) (local.get $scale_bytes))
      (call $matvec (local.get $k) (i32.load offset=20 (local.get $layer)) (local.get $h)
        (local.get $kv_dim) (local.get $dim) (local.get $scale_bytes))
      (call $matvec (local.get $v) (i32.load offset=24 (local.get $layer)) (local.get $h)
        (local.get $kv_dim) (local.get $dim) (local.get $scale_bytes))
      (call $norm_rope (local.get $q) (i32.load offset=4 (local.get $layer)) (local.get $heads)
        (local.get $width) (local.get $cosines) (local.get $sines) (local.get $eps))
      (call $norm_rope (local.get $k) (i32.load offset=8 (local.get $layer)) (local.get $kv_heads)
        (local.get $width) (local.get $cosines) (local.get $sines) (local.get $eps))
      (memory.copy (i32.add (local.get $keys) (local.get $kv_pos)) (local.get $k) (local.get $kv_bytes))
      (memory.copy (i32.add (local.get $values) (local.get $kv_pos)) (local.get $v) (local.get $kv_bytes))
      (call $attention (local.get $att) (local.get $q) (local.get $keys) (local.get $values)
        (local.get $scores) (local.get $heads) (local.get $kv_heads) (local.get $width) (local.get $pos))
      (call $matvec (local.get $h) (i32.load offset=28 (local.get $layer)) (local.get $att)
        (local.get $dim) (local.get $q_dim) (local.get $scale_bytes))
      (call $axpy (local.get $x) (local.get $h) (local.get $dim) (f32.const 1))

      (call $rmsnorm (local.get $h) (local.get $x) (i32.load offset=12 (local.get $layer))
        (local.get $dim) (local.get $eps))
      (call $matvec (local.get $gate) (i32.load offset=32 (local.get $layer)) (local.get $h)
        (local.get $hidden) (local.get $dim) (local.get $scale_bytes))
      (call $matvec (local.get $up) (i32.load offset=36 (local.get $layer)) (local.get $h)
        (local.get $hidden) (local.get $dim) (local.get $scale_bytes))
      (call $silu_mul (local.get $gate) (local.get $up) (local.get $hidden))
      (call $matvec (local.get $h) (i32.load offset=40 (local.get $layer)) (local.get $gate)
        (local.get $dim) (local.get $hidden) (local.get $scale_bytes))
      (call $axpy (local.get $x) (local.get $h) (local.get $dim) (f32.const 1))

      (local.set $keys (i32.add (local.get $keys) (local.get $kv_layer_bytes)))
      (local.set $values (i32.add (local.get $values) (local.get $kv_layer_bytes)))
      (local.set $layer (i32.add (local.get $layer) (i32.const 44)))
      (br_if $layers (i32.lt_u (local.get $layer) (local.get $layer_end))))

    (if (local.get $compute_logits)
      (then
        (call $rmsnorm (local.get $h) (local.get $x) (i32.load (i32.const 48)) (local.get $dim) (local.get $eps))
        (call $matvec (i32.load (i32.const 88)) (local.get $embed) (local.get $h)
          (i32.load (i32.const 24)) (local.get $dim) (local.get $scale_bytes)))))
)
