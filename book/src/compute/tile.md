# Tile compute primitives

Stage 37.8 bundled together a small suite of reusable compute
primitives — workgroup-tiled GEMM, separable convolution, named
filters, block-Cholesky factorisation, stream compaction, and a
Monte-Carlo gradient estimator. They live in
`crates/molla/src/compute/tile_*.rs` and share two conventions:

- A **CPU reference** function (`*_cpu`) that's the precision oracle
  and the no-feature fallback.
- A **GPU dispatcher** (`Tile*Kernel`) gated behind the `gpu` feature.

This page covers each tile primitive at a glance; deep details live
in the source modules and the WGSL shader headers.

## `tile_matmul` — workgroup-tiled GEMM

```rust
pub fn matmul_cpu(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, n: usize, k: usize,
) -> Result<()>;

#[cfg(feature = "gpu")]
pub struct TileMatmulKernel { /* … */ }
```

Row-major `C = A · B` with `A: m × k`, `B: k × n`, `C: m × n`. The
WGSL kernel uses 16×16 workgroup tiles loaded into `workgroup` memory
once per tile and reused across `k`-axis traversals — the classic
shared-memory matmul. The host wrapper `TileMatmulKernel::dispatch_blocking`
allocates buffers, uploads `A` and `B`, dispatches, reads `C` back.

Shape `(M, K, N)` → grid `(ceil(N/16), ceil(M/16), 1)`. The
implementation is a textbook starting point, not optimised for any
specific aspect ratio.

## `tile_convolution` — separable 2D convolution

```rust
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Kernel5 {
    pub coeffs: [f32; 5],
}

impl Kernel5 {
    pub const fn box5() -> Self;
    pub const fn gaussian5() -> Self;    // (1, 4, 6, 4, 1) / 16
    pub const fn identity() -> Self;     // δ-function
    pub const fn laplacian1d() -> Self;  // (0, 1, -2, 1, 0)
}

pub fn convolve_separable_cpu(
    input: &[f32], output: &mut [f32],
    width: usize, height: usize,
    kernel: &Kernel5,
) -> Result<()>;
```

Two-pass separable 5-tap convolution with clamp-to-edge boundary.
The GPU side wires two pipelines (`conv_horiz`, `conv_vert`) and
chains them through a scratch buffer. Used by viewer post-processing,
preview filters, and the Stage 37.x wave-equation experiments.

## `tile_filtering` — named filter wrappers

```rust
pub fn box_filter_cpu(input, output, width, height) -> Result<()>;
pub fn gaussian_filter_cpu(input, output, width, height) -> Result<()>;
pub fn laplacian_filter_cpu(input, output, width, height) -> Result<()>;
pub fn identity_filter_cpu(input, output, width, height) -> Result<()>;

// GPU variants take a TileConvolutionKernel + device + queue.
```

Thin wrappers over `convolve_separable_cpu` with named filter
coefficients (box, Gaussian, Laplacian, identity). The math is
identical — convolution is convolution — but exposing the named
filters as their own API keeps call-sites readable. Identity is the
smoke-test path: it should return the input unchanged.

## `tile_block_cholesky` — single-workgroup SPD factorisation

```rust
pub fn cholesky_cpu(a: &mut [f32], n: usize) -> Result<()>;
pub fn cholesky_cpu_lenient(a: &mut [f32], n: usize) -> Result<()>;
pub fn frobenius_recon_error(orig: &[f32], factor: &[f32], n: usize) -> f32;

#[cfg(feature = "gpu")]
pub struct TileBlockCholeskyKernel { /* … */ }
```

In-place row-major Cholesky. On exit the lower triangle of `a` holds
`L` with `L · Lᵀ = A`; the strict upper triangle is zeroed.
`cholesky_cpu` returns `Error::Internal` if any pivot is `< 1e-30`
(non-SPD); `cholesky_cpu_lenient` clamps negative pivots to zero and
proceeds — only use it when you genuinely want the clamped
pseudo-factor (incomplete-Cholesky preconditioning of nearly-SPD
systems).

The GPU path is a single-workgroup factor with 64 threads, capped at
`N ≤ 1024`. For larger `N` it returns garbage (no protection in the
shader; the host wrapper rejects explicitly). Multi-workgroup tiling
is a v2 — the current consumers are all small-N FEM Schur
complements where a single workgroup suffices.

## `tile_stream_compaction` — gather-via-scan

```rust
pub const MAX_N: usize = 1024;

pub fn compact_cpu(input: &[f32], mask: &[bool]) -> Result<Vec<f32>>;
pub fn mask_to_u32(mask: &[bool]) -> Vec<u32>;

#[cfg(feature = "gpu")]
pub struct TileStreamCompactionKernel { /* … */ }
```

Single-workgroup Hillis-Steele scan + scatter. Walks `input`/`mask`,
emits a dense vector of every entry where `mask[i] == true`. The GPU
path is capped at `n ≤ MAX_N = 1024` so a single workgroup covers
the input; multi-workgroup decoupled-look-back is a v2.

For the cases that need `n ≫ 1024`, use the
[`ScanKernel`](./../math/scan_sort.md) directly — it ships the
production-grade scan path.

## `tile_mcgp` — SPSA gradient estimator

```rust
#[derive(Copy, Clone, Debug)]
pub struct McgpConfig {
    pub n_samples: usize,
    pub epsilon: Real,
    pub seed: u64,
}

pub fn mcgp_estimate<F>(f: &F, params: &[Real], config: &McgpConfig) -> Vec<Real>
where F: Fn(&[Real]) -> Real;

pub fn finite_diff_central<F>(f: &F, params: &[Real], h: Real) -> Vec<Real>
where F: Fn(&[Real]) -> Real;
```

Stochastic gradient estimator for losses that aren't tape-recordable
(or where building the tape is too expensive). Given a black-box
`f: &[Real] -> Real` and a parameter vector, runs `n_samples`
perturbations of the form `params ± ε · noise[k]` (Rademacher
`±1` noise) and accumulates the per-component gradient estimate:

```text
g_i ≈ (1/N) · Σₖ (f(p + ε·dₖ) − f(p − ε·dₖ)) / (2 ε) · dₖ,i
```

This is the SPSA recipe of Spall (1992) — one of the cheapest
stochastic gradient approximations and a useful complement to molla's
tape autograd (`crate::autograd::tape`) when the loss isn't
tape-recordable.

The estimate is **noisy** by design. An `n_samples = 64` estimator on
a smooth quadratic typically lands within ~5 % of the true gradient.
Tighten the tolerance by raising `n_samples` (variance ∝ 1/N) or by
using a Richardson-style averaged perturbation pattern (a v2
nice-to-have).

The internal RNG is a tiny xorshift64 keyed off the config seed, so
two runs with the same seed produce the same gradient — important
for reproducible parameter-ID experiments.

`finite_diff_central` is the deterministic O(2n) reference used as
the oracle in tests; exposed publicly for callers who want a
gold-standard FD check alongside the noisy SPSA estimate.

## Cross-references

- [Kernel infrastructure](./kernels.md) — the dispatcher pattern
  every `Tile*Kernel` follows.
- [Scan & 64-bit radix sort](./../math/scan_sort.md) — for stream
  compaction beyond the 1024 cap.
- [Tape-based autograd](./../autograd/tape.md) — the deterministic
  alternative to SPSA when the loss is tape-recordable.
