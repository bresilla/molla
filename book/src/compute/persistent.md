# Persistent-buffer pattern

`Buffer<T>` from [Storage abstractions](./storage.md) is the right
tool when a host array is the source of truth and the device copy
just mirrors it. For long-lived solver state — buffers that the GPU
mutates across thousands of dispatches without the host ever needing
to look — the upload/download semantics of `Buffer<T>` are the wrong
abstraction. Use **persistent GPU buffers** instead.

## When to use persistent buffers

The pattern: **upload once, dispatch many**. The buffer's contents are
authoritative on the device for the lifetime of the solver. The host
side keeps the `wgpu::Buffer` handle (typically in an `Arc<wgpu::Buffer>`
field of the kernel/solver struct) and never copies back unless the
caller asks for a snapshot.

Three current users:

### 1. MPM resident state

The MPM grid (velocities, masses, force accumulators) is the largest
buffer in the project and gets read+write every substep. Copying it
host↔device per frame would cost more than the actual physics.

`crate::compute::mpm` and `crate::compute::implicit_mpm` allocate the
grid buffers at scene-build time, zero them on the device, and the
host never sees them again until the caller asks for a snapshot via
an explicit `MpmState::snapshot_*` method. The state lives entirely
on the device for the duration of `n_substeps`.

### 2. FEM CG bind-group caching

The CG inner loop dispatches the same `(A, x, y)` matvec thousands of
times per solve. The "prepared matvec" path in `BsrAlgebraKernel`
(Stage 41 v5) caches the uniform buffer + IO bind group up front so
the inner loop becomes a single `dispatch_workgroups` with zero
per-iter allocation:

```rust
pub struct PreparedMatvec {
    uniform_buf: wgpu::Buffer,
    bg_io: wgpu::BindGroup,
    bg_u: wgpu::BindGroup,
    dim_x: u32,
    dim_y: u32,
}

impl BsrAlgebraKernel {
    pub fn prepare_matvec(&self, a, x_buf, y_buf, alpha, beta) -> Result<PreparedMatvec>;
    pub fn matvec_into_pass(&self, pass: &mut wgpu::ComputePass<'_>, prepared: &PreparedMatvec);
}
```

`prepare_matvec` is called once per CG solve; `matvec_into_pass` is
called every CG iteration into a long-lived compute pass. The whole
CG batch ends up as one encoder submission, which is what unlocks
the latency reductions the FEM perf gates measure.

See [`BsrGpu`](./../math/bsr_gpu.md#prepared-matvec--stage-41-v5)
for the prepared-matvec API in detail.

### 3. GPU FEM steppers

The FEM solvers (`fem::solvers`, `fem::elasticity_gpu`,
`fem::stokes_ns`, `fem::magneto`) keep their assembled mass + stiffness
matrices, vector workspaces, and boundary-condition encodings on the
device for the lifetime of the simulation. Reassembly per step is
done in-place — there's an `axpy_same_pattern_unchecked` rebuild path
that lets viscoelastic mass matrices update without rebuilding the
sparsity pattern.

The host side keeps a `MeshGpu` / `FemMatricesGpu` struct of
`Arc<wgpu::Buffer>` handles. CPU never reads back unless a test
explicitly asks (parity tests, post-mortem diagnostics).

## The pattern in code

The typical solver struct shape:

```rust
pub struct FemElasticityGpu {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,

    // Persistent solver state — written by kernels, never round-tripped.
    k_matrix: BsrGpu,            // stiffness matrix
    m_matrix: BsrGpu,            // mass matrix
    u: Arc<wgpu::Buffer>,        // displacements
    v: Arc<wgpu::Buffer>,        // velocities
    f_ext: Arc<wgpu::Buffer>,    // external forces

    // Pipeline owners — built once per device.
    algebra: BsrAlgebraKernel,
    scan: ScanKernel,
    // …
}

impl FemElasticityGpu {
    pub fn new(/* … */) -> Self { /* allocate + upload once */ }

    pub fn step(&mut self, dt: f64) -> Result<()> {
        // 100% on-device — no upload, no download.
    }

    pub fn snapshot_displacements(&self) -> Result<Vec<f64>> {
        // Explicit readback — only call from test or diagnostic code.
    }
}
```

Compare with `Buffer<T>`'s sync-state machine: persistent buffers
don't need one because there's no ambiguous "which side is fresh?"
question — the device is always fresh, and the host only reads
through a named snapshot method.

## When NOT to use persistent buffers

- **Hot reload of host-side data.** When the caller mutates a `Model`
  field and expects it to take effect next step, `Buffer<T>` with
  `host_mut` + `upload` is the right primitive. Persistent buffers
  force the caller to call a rebuild method explicitly.
- **One-shot kernels.** `TileMatmulKernel::dispatch_blocking` allocates
  fresh buffers per call. Caching them across calls would save ~50 µs
  of allocation but add ownership complexity. Don't bother unless a
  profile says you should.
- **Tests.** Test code reads results back to compare against a CPU
  oracle. `Buffer<T>`'s `download` is exactly that — the persistent
  pattern means tests need to call snapshot methods that themselves
  exist only for tests, which couples test infrastructure into
  production paths.

## Lifetime + concurrency

`Arc<wgpu::Buffer>` is the standard handle. wgpu's runtime reference-counts
GPU resources internally, so dropping the `Arc` releases the buffer once
all outstanding command-encoder references resolve. The same buffer can
be bound into multiple bind groups simultaneously — wgpu's tracker
prevents read-write hazards across passes.

Two solvers sharing a buffer (e.g. an FEM displacement field consumed
by both the elasticity stepper and a rendering pipeline) just hand
each other `Arc<wgpu::Buffer>` clones. No additional synchronisation
is needed inside a single command encoder — wgpu inserts the barriers.

## Cross-references

- [GPU runtime](./runtime.md) — `Engine::device()` /
  `Engine::queue()` is what the persistent buffers get wired against.
- [Storage abstractions](./storage.md) — the round-tripping
  `Buffer<T>` for the cases where you do want the host as the source
  of truth.
- [`BsrGpu`](./../math/bsr_gpu.md#prepared-matvec--stage-41-v5) — the
  prepared-matvec implementation that's the canonical example of the
  pattern.
