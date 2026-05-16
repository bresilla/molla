# `BsrGpu` (GPU)

`crates/molla/src/math/sparse_gpu.rs` is the on-device mirror of
[`BsrMatrix`](./bsr_cpu.md). It carries the three CSR buffers and
gates compute through a single `BsrAlgebraKernel` that owns all the
compiled pipelines.

The module is gated behind `#![cfg(feature = "gpu")]`. The host
counterpart works without the `gpu` feature; everything in this page
disappears in CPU-only builds.

## `BsrGpu` — the buffer triple

```rust
pub struct BsrGpu {
    pub row_offsets: Arc<wgpu::Buffer>,
    pub col_indices: Arc<wgpu::Buffer>,
    pub values: Arc<wgpu::Buffer>,
    pub nrows: u32,
    pub ncols: u32,
    pub nnz: u32,
}
```

`row_offsets` and `col_indices` are `u32` buffers; `values` is `f32`.
Shape metadata stays on the host so dispatchers can size workgroup
grids without round-tripping. `Arc<wgpu::Buffer>` lets the same
buffers be borrowed across multiple bind groups (the SpGEMM path
binds the same `row_offsets` into both its symbolic and its numeric
pipelines).

### Precision boundary

The CSR layout values are `f32` on the device, `f64` on the host
(matches [`BsrMatrix`](./bsr_cpu.md)). Upload converts `f64 → f32`,
readback the reverse:

```rust
let values_f32: Vec<f32> = m.values.iter().map(|&v| v as f32).collect();
```

This is the same tradeoff `fem::gpu_data` documents — WGSL has no
`f64`, and a Vulkan-only path would fork the codebase.

### Upload / readback

```rust
impl BsrGpu {
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        m: &BsrMatrix,
    ) -> Result<Self>;

    pub fn read_back(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<BsrMatrix>;
}
```

`upload` rejects `B != 1` and validates the CSR shape invariants
(`row_offsets[last] == values.len() == col_idx.len()`,
`row_offsets.len() == block_rows + 1`). Without those checks an
inconsistent host matrix would silently produce out-of-bounds reads
inside the WGSL kernels. `read_back` rebuilds a host `BsrMatrix` and
promotes `f32 → Real`, slicing each buffer down to its logical length
(wgpu may pad allocations up to 16 bytes).

## `BsrAlgebraKernel` — pipeline owner

```rust
pub struct BsrAlgebraKernel { /* … */ }

impl BsrAlgebraKernel {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self;

    pub fn matvec(
        &self,
        a: &BsrGpu,
        x_buf: &wgpu::Buffer,
        y_buf: &wgpu::Buffer,
        alpha: f32,
        beta: f32,
    ) -> Result<()>;

    pub fn axpy_same_pattern_unchecked(
        &self,
        a: &BsrGpu,
        b: &mut BsrGpu,
        alpha: f32,
        beta: f32,
    ) -> Result<()>;

    pub fn transpose(&self, a: &BsrGpu) -> Result<BsrGpu>;
    pub fn spgemm(&self, a: &BsrGpu, b: &BsrGpu) -> Result<BsrGpu>;
}
```

Build once per device, reuse across many dispatches. The constructor
compiles four WGSL modules (`bsr_mv_gpu.wgsl`, `bsr_axpy.wgsl`,
`bsr_transpose.wgsl`, `bsr_spgemm.wgsl`) plus a fifth
(`bsr_spgemm_symbolic.wgsl`) for the GPU-symbolic SpGEMM path, and
stamps out the bind-group layouts up front.

### `matvec` — generalised SpMV

`matvec(a, x_buf, y_buf, alpha, beta)` computes `y = α · A · x + β ·
y`. Buffers must already be allocated as `STORAGE | COPY_SRC |
COPY_DST`. With `β = 0` the previous contents of `y_buf` are read but
masked out — so a freshly-allocated `y_buf` is fine, you don't have
to zero it.

The dispatch grid is `(min(nrows, 65535), ceil(nrows / 65535), 1)` —
2D so that very tall matrices stay under the per-axis dispatch limit.

### Prepared matvec — Stage 41 v5

The FEM CG inner loop dispatches the same matvec thousands of times
per solve. To avoid per-call uniform allocation and bind-group
creation, the kernel exposes a "prepared" path:

```rust
pub struct PreparedMatvec { /* uniform_buf, bg_io, bg_u, dim_x, dim_y */ }

impl BsrAlgebraKernel {
    pub fn prepare_matvec(
        &self,
        a: &BsrGpu,
        x_buf: &wgpu::Buffer,
        y_buf: &wgpu::Buffer,
        alpha: f32,
        beta: f32,
    ) -> Result<PreparedMatvec>;

    pub fn matvec_into_pass(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        prepared: &PreparedMatvec,
    );
}
```

`prepare_matvec` caches the uniform buffer and bind groups for a
fixed `(A, x, y, α, β)`. `matvec_into_pass` records one extra
dispatch into an existing compute pass — no extra encoder / submit.
This is what lets the FEM solver fold its CG batch into a single
compute pass.

### `axpy_same_pattern_unchecked` — sparse axpy with a safety contract

`B = α · A + β · B`. Documented contract: caller asserts that `A` and
`B` share **identical `row_offsets` AND `col_indices`** — only
`values` differ. The function verifies only the cheap scalar shape
`(nnz, nrows, ncols)`; a true pattern check would need reading
`row_offsets + col_indices` back to the host (two mappable readbacks),
which makes the call unusable inside any per-iter solver loop.
Mismatched patterns produce **numerical garbage silently** — build
the pattern once at upload time and keep it across all axpy calls.
The pattern-union case is intentionally not in v1.

### `transpose(a) -> BsrGpu`

Host computes the transposed pattern (counts per column, prefix-sum,
scatter map) and uploads three small `u32` buffers; GPU runs a single
`bsr_transpose_scatter` dispatch that scatters values into their
transposed slots. The host-symbolic step is fast enough that moving
it on-device hasn't shown up in a profile, but the same bitmap-count
approach used for SpGEMM (next page) would work.

### `spgemm(a, b) -> BsrGpu`

Dispatcher that picks between **host-symbolic + GPU-numeric** and the
**full-GPU symbolic** variant based on `max(A.nrows, B.ncols)`:

```rust
pub const SPGEMM_GPU_SYMBOLIC_CUTOVER: u32 = 64;
```

For matrices smaller than 64 the host-symbolic v1 path wins (kernel
launch overhead dominates); for larger problems the GPU symbolic path
takes over. See [SpGEMM design](./spgemm.md) for the full
explanation of why bitmaps beat hashtables on FEM-shaped inputs.

## Parity testing

Every kernel here has an oracle: run the matching `BsrMatrix` op on
the host and compare with `BsrGpu::read_back`. The parity tests in
`crates/molla-kernels/tests/bsr_gpu_parity.rs` cover matvec, axpy, transpose,
and SpGEMM at multiple sizes — the contract is **bit-identical CSR
structure** plus an `f32`-tolerance comparison on values.
