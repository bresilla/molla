// Stage 24c — Stockham radix-2 FFT (forward + inverse).
//
// === Algorithm choice ===
//
// We use the **Stockham auto-sort** radix-2 FFT (Cooley/Lewis/Welch
// 1970) instead of the textbook Cooley-Tukey decimation-in-time. Both
// have the same O(N log N) arithmetic cost; Stockham trades a bit of
// extra memory (a second buffer for ping-pong) for two big wins on a
// GPU:
//
//   1. **No bit-reversal permutation pass.** Cooley-Tukey requires
//      the input (or output) to be bit-reversed; on a GPU that's
//      either a separate scatter pass or a strided load that hurts
//      coalescing. Stockham permutes implicitly: at every pass the
//      "input layout" and "output layout" interpret the flat array
//      differently, and after `log2(N)` passes the flat output array
//      is in natural index order.
//
//   2. **Coalesced reads + writes within each pass.** Adjacent
//      threads read adjacent positions in the input buffer and write
//      adjacent positions in the output buffer (within one
//      butterfly group), which is the access pattern the GPU's
//      memory subsystem prefers.
//
// Per-pass butterfly (for thread `i` in `[0, N/2)`, pass `s` with
// half-size `m = 1 << s` and full-size `M = 2 * m`):
//
//   j = i / m                        # group index (0 .. N/M)
//   k = i % m                        # twiddle index within group (0 .. m)
//   a = in[j*m + k]
//   b = in[j*m + k + N/2]            # paired element is N/2 away
//   w = exp(±2πi * k / M)            # − for forward, + for inverse
//   out[j*M + k]     = a + w*b
//   out[j*M + k + m] = a - w*b
//
// We verified this end-to-end against `rustfft` (see
// `tests/fft_gpu.rs`). For x = [1,2,3,4], DFT = [10, −2+2i, −2, −2−2i],
// matching to f32 precision after two passes.
//
// === Inverse FFT ===
//
// Identical algorithm, twiddle phase sign flipped (`+2πi` instead of
// `-2πi`). Normalisation is **not** applied inside the kernel — the
// host must divide by N if it wants `ifft(fft(x)) == x`. We picked
// the "forward unnormalised, inverse unnormalised" convention
// (so forward * inverse = N, matching `rustfft::FftPlanner::plan_fft_*`
// and most physics-engine FFT users).
//
// === Ping-pong buffers ===
//
// The host allocates two storage buffers of N complex elements each
// and alternates which one is `input` vs `output` per pass. After
// `log2(N)` passes the host knows whether the result is in buffer A
// or B (parity of `L = log2(N)`); see the host-side dispatcher.
//
// === Per-dispatch shape ===
//
// One thread per butterfly. For an N-point transform, one pass spawns
// `N/2` threads, organised into workgroups of `WORKGROUP_SIZE = 64`.
// 1D, 2D, and 3D are all driven by the same kernel: 2D = N₂ rows of
// 1D-of-N₁ then N₁ columns of 1D-of-N₂; 3D = 2D-on-each-slice then 1D
// along the third axis. The `Uniforms.row_count` field lets one
// dispatch process many independent rows in parallel (each
// workgroup-row handles one independent FFT, indexed by the y
// dispatch dimension).
//
// === Constraints ===
//
// * Lengths must be powers of two (cap documented in the host
//   wrapper's docstring; v1 has no Bluestein / mixed-radix support).
// * Element type: `vec2<f32>` (real, imag).
// * Twiddles are computed inline per-thread via `cos`/`sin`. We tried
//   workgroup-shared twiddle tables and the gain was lost in the
//   barrier overhead at small N — the inline path is simpler and
//   matches CPU oracle to 1e−5 across all tested sizes (N=1..32k).

const WORKGROUP_SIZE: u32 = 64u;
const TWO_PI:         f32 = 6.28318530717958647692;

struct Uniforms {
    n:           u32,   // FFT length (power of two)
    half_n:      u32,   // n / 2 — number of butterflies per row per pass
    pass_idx:    u32,   // current pass index s in [0, log2(n))
    row_count:   u32,   // number of independent FFTs in this dispatch
    row_stride:  u32,   // distance (in vec2 elements) between adjacent rows
    inverse:     u32,   // 0 = forward, 1 = inverse (sign flip on twiddle)
    _pad0:       u32,
    _pad1:       u32,
};

@group(0) @binding(0) var<storage, read>       input_buf:  array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output_buf: array<vec2<f32>>;
@group(1) @binding(0) var<uniform>             uniforms:   Uniforms;

// Complex multiply: (a.x + i a.y) * (b.x + i b.y).
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

// One Stockham butterfly. Each (gx, gy) is (butterfly index, row index).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn fft_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i   = gid.x;
    let row = gid.y;
    if (i >= uniforms.half_n) { return; }
    if (row >= uniforms.row_count) { return; }

    let s   = uniforms.pass_idx;
    let m   = 1u << s;             // current butterfly half-size
    let big = m << 1u;             // full butterfly size = 2 * m
    let j   = i / m;                // butterfly group index
    let k   = i % m;                // twiddle index within group

    let row_off = row * uniforms.row_stride;

    // Read paired elements: positions are N/2 apart in the input layout.
    let a = input_buf[row_off + j * m + k];
    let b = input_buf[row_off + j * m + k + uniforms.half_n];

    // Twiddle: exp(±2πi * k / big). Sign: − for forward, + for inverse.
    let phase_mag = TWO_PI * f32(k) / f32(big);
    let phase     = select(-phase_mag, phase_mag, uniforms.inverse == 1u);
    let w         = vec2<f32>(cos(phase), sin(phase));

    let wb        = cmul(w, b);
    output_buf[row_off + j * big + k]     = a + wb;
    output_buf[row_off + j * big + k + m] = a - wb;
}

// Transpose kernel for 2D / 3D dispatchers. Reads `[height][width]`
// from `input_buf` and writes `[width][height]` to `output_buf`. For
// a 2D FFT we run rows-FFT → transpose → rows-FFT (= columns-FFT) →
// transpose back. The transpose lets us reuse the same row-major
// kernel for the column pass.
//
// `Uniforms.row_count` is reused as `width`, `Uniforms.row_stride`
// as `height`, and the dispatch shape is `(width, height, 1)` rounded
// up to workgroup tiles.

@compute @workgroup_size(8, 8, 1)
fn transpose_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;        // column in input = row in output
    let y = gid.y;        // row in input = column in output
    let width  = uniforms.row_count;
    let height = uniforms.row_stride;
    if (x >= width || y >= height) { return; }
    output_buf[x * height + y] = input_buf[y * width + x];
}
