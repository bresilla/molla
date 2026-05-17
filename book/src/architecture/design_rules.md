# Design rules

The conventions below are load-bearing. Every contributor — human
or AI — has to respect them; reviewers reject PRs that violate them
without an explicit deviation note. These are not style guidelines —
they are architectural invariants the rest of the engine assumes.

## 1. Physics must be switchable CPU↔GPU at all times

> Every solver, sensor, actuator, FEM stepper, and primitive kernel
> can step on either CPU or GPU, and the choice is a runtime
> `Backend` toggle — not a compile-time feature.

Concrete consequences:

- Every kernel ships with **both** a Rust CPU reference implementation
  and a WGSL device implementation. They produce the same output
  modulo float-op ordering nondeterminism.
- Every solver exposes `new_cpu()` and `new_gpu(&engine)` constructors.
  The viewer can swap backends at runtime by dropping the old solver
  and constructing the new one.
- Every kernel has a `tests/<name>_gpu.rs` parity test that runs
  both backends on identical inputs and asserts trajectory agreement
  within a documented tolerance (typically 1e-4 for f32-narrowing).
- The CPU path is the **oracle**, not the fallback. When a GPU
  parity test fails, the CPU path is the source of truth — you
  debug by reading the CPU code, not the WGSL.

This rule predates the `gpu` Cargo feature flag. The Cargo feature
gates *availability* (it pulls in `wgpu`). The CPU↔GPU rule is about
*runtime* selection within a build that has the feature on.

See user memory note: **"GPU-first rule"** — implement physics in
WGSL first, copy from Newton / Warp, never start with a Rust host
loop. CPU is the oracle, not the starting point.

## 2. BSR `B = 1` everywhere

molla's sparse matrix type is `math::BsrMatrix` — block-sparse-row
with a runtime block dimension `B`. Every actual user of the type
sets `B = 1`, which recovers a plain CSR matrix.

Why we ship a parameterised type and then only ever use `B = 1`:

- The data layout (`row_offsets` + `col_idx` + `values`) is the
  same as CSR, so all the iterative solvers (`cg`, `bicgstab`,
  `gmres`, `pcg`) work bit-identical on either.
- The room to grow to `B > 1` exists. FEM elasticity has natural
  3×3 block structure; if the per-iter `BSR matvec` becomes the
  bottleneck we can lift `B` without rewriting the solver layer.
- WGSL's lack of dynamic-size shared memory and pointer-into-storage
  restrictions make a templated `B` more painful on the GPU than
  on the CPU. Keeping `B = 1` lets the GPU mirror (`BsrGpu`) be a
  trivial three-buffer (`row_offsets`, `col_idx`, `values`) layout
  with one matvec shader.

The rule is *not* "BSR with `B > 1` is forbidden". The rule is
"don't introduce `B > 1` without a measured win that justifies the
GPU complexity". As of Stage 51 no such win has been measured, so
every sparse matrix in the engine is effectively CSR.

## 3. Persistent buffers, single-encoder dispatch

The naive way to drive a GPU compute kernel is:

```text
loop {
    create_buffer();
    create_bind_group();
    create_command_encoder();
    dispatch();
    submit();
    readback();
}
```

This was the Stage 24d v1 MPM pattern and it ran at 20 Hz on 10k
particles. The Stage 24d v2 dispatcher hoisted the per-iter
uniform buffer + bind group + readback staging into
`MpmSimulation::new_with_device()`, then introduced a `step_n` API
that encodes N substeps into a **single** `wgpu::CommandEncoder`
and submits **once**. Measured speedup: 6.9× (20 Hz → 75 Hz).

The convention going forward:

- **Allocate persistent buffers at solver construction time.** Not
  per-step.
- **Cache bind groups + pipeline handles once.** Look up by buffer
  identity, not by re-creating per step.
- **Batch substeps into one encoder + one submit.** Use `step_n` /
  `step_n_with` patterns; the user-facing `step` is a back-compat
  wrapper around `step_n(_, _, 1)`.
- **Read back exactly once per visible frame.** Per-substep readback
  is the most common perf bug; the GPU is happy to crunch 100
  substeps in 13 ms, but the wgpu submit + readback overhead per
  call dominates if you call it 100 times.

The MPM dispatcher (`compute::MpmSimulation::step_n`) and the FEM
GPU-resident CG (`fem::CgStepperGpu`) are the worked examples.

## 4. Atomic-CAS pattern for `f32` reductions

WGSL has `atomic<u32>` and `atomic<i32>`. It does **not** have
`atomic<f32>`. Floating-point reductions need a workaround.

The molla convention is a compare-and-swap loop:

```wgsl
// Conceptually: *target += addend;
fn atomic_add_f32(target: ptr<storage, atomic<u32>, read_write>, addend: f32) {
    loop {
        let old_bits = atomicLoad(target);
        let old_f = bitcast<f32>(old_bits);
        let new_f = old_f + addend;
        let new_bits = bitcast<u32>(new_f);
        let res = atomicCompareExchangeWeak(target, old_bits, new_bits);
        if (res.exchanged) { break; }
    }
}
```

Naga (the WGSL→SPIR-V translator wgpu uses) **rejects pointer-to-storage
as a function argument from one storage binding to another**. The
consequence: every atomic-add target lives in its own bound buffer
and you ship one helper per target (e.g. `atomic_add_mass` and
`atomic_add_mom` for the MPM grid). You can't have a generic
`atomic_add_f32` helper that takes a pointer to any storage buffer.

This is annoying. It is also unavoidable until WGSL adds either
`atomic<f32>` or generic pointers — neither is on the spec roadmap.

Where the pattern shows up:

- MPM grid (`shaders/mpm.wgsl`) — mass + momentum scatter from
  particles to grid.
- FEM assembly (`shaders/fem_assembly.wgsl`) — element-matrix
  scatter into the global CSR.
- 3DGS backward (`shaders/splat_raytrace_backward.wgsl`) — 5
  per-splat gradient channels.
- Sparse grid mass + position scatter.

## 5. No `cooperative_matrix`, no subgroup intrinsics

wgpu has experimental `subgroup_*` operations behind a feature flag;
the WGSL spec does **not** stabilise them. NVIDIA's
`VK_KHR_cooperative_matrix` extension is not exposed portably
through wgpu either. molla therefore commits to:

- **No `subgroup_*` calls in shipped WGSL.** If a shader needs a
  reduction it spells the reduction out (workgroup shared memory
  + barrier + log₂N tree).
- **No `cooperative_matrix` tile matmul.** The molla
  `compute::tile_matmul` kernel is a plain WGSL implementation
  (`var<workgroup>` tile loads + per-thread inner products). It
  is not competitive with cuBLAS or Metal Performance Shaders at
  large M / N / K, and that's a deliberate trade for portability.

When a workload genuinely needs TensorCore-class throughput, the
right answer in molla is **don't use molla** — call cuBLAS or
Metal Performance Shaders from the host side and pass the result
back as a `Buffer`. The engine doesn't pretend to be a generic
matrix library.

## 6. Host `f64`, device `f32`, conversions at the buffer boundary

The math layer is `f64` end-to-end (`math::Real = f64`). The WGSL
device side is `f32` end-to-end (WGSL has no `f64`). The conversion
happens *exactly once*, when a `Buffer<T>` is uploaded or downloaded:

- `Buffer::host() -> &[T_host]` — `f64` view.
- `Buffer::upload(&device, &queue)` — narrows to `f32`, writes to
  `wgpu::Buffer`.
- `Buffer::download_from_device(&device, &queue)` — widens back
  to `f64`.

The conversion is bytemuck-cast through a packed `f32` representation
in the wgpu buffer; the `f64` host vector is the source of truth
between substeps.

Solvers never mix the two precisions in the same arithmetic
operation. If a solver needs intermediate device-side state, that
state lives in a separate `Buffer` and never crosses back to the
host until the substep is done.

## 7. Errors are typed, no `anyhow` in the engine

`molla::core::Error` is a `thiserror`-derived enum. Library code
never uses `anyhow::Result` — that's reserved for the viewer and
applications. The reason is the standard library-hygiene one:
applications want to add context; libraries want to be exhaustively
matchable.

The few places `anyhow` shows up — like the former `usd_molla`
crate that briefly existed — were application-level code, not
engine.

## 8. No `unsafe` outside Pod-compatible data

The engine has zero `unsafe` blocks outside of:

- `bytemuck::Pod` / `Zeroable` derives.

Workspace lint `unsafe_op_in_unsafe_fn = "deny"` enforces that any
new `unsafe` is flagged and reviewed.

## 9. Tests are `#[ignore]`'d when they need a wgpu adapter

The default `cargo test` lane should pass on a CPU-only CI runner.
Tests that require a real `wgpu::Adapter` are marked `#[ignore]` so
they only run when the developer explicitly opts in via the
`make test-*-gpu` Makefile targets or `cargo test -- --ignored`.

The exception is `crates/molla/tests/cpu_smoke.rs` — pure-CPU
smoke tests that run in the default lane.

## 10. Reference Newton + Warp

Per the user memory note: when implementing a new physics
primitive, consult **both** `xtra/warp` (compute primitives) and
`xtra/newton` (algorithms). molla copies Newton's structure
expressed via Warp-style spatial operations in Rust + WGSL. The
two trees together are the reference; consulting only one is the
common mistake.
