# Sparse linear algebra

molla's sparse stack is built on a **block-sparse-row (BSR)** matrix.
With block size `B = 1` it degenerates to plain CSR, which is the
configuration almost every consumer actually wants — FEM assemblies,
implicit-MPM grid systems, SpGEMM products, BSR-GPU mirrors. Keeping
the single `BsrMatrix` type for both cases avoids the maintenance
overhead of carrying parallel CSR and BSR codebases.

Three layers stack on top of each other:

1. **CPU BSR matrix** — `crates/molla/src/math/sparse.rs`. Construction
   from triplets, matvec, axpy, transpose, basic SpGEMM, plus three
   iterative solvers (CG, BiCGSTAB, restartable GMRES) implemented
   against a [`LinearOperator`] trait so callers can plug in a closure
   or a custom matrix-free operator. See [BsrMatrix (CPU)](./bsr_cpu.md).
2. **GPU BSR mirror** — `crates/molla/src/math/sparse_gpu.rs`. Three
   storage buffers (`row_offsets`, `col_indices`, `values`) holding the
   CSR layout with `f32` values; a single `BsrAlgebraKernel` owns the
   compiled pipelines for matvec, axpy, transpose, SpGEMM, and SpGEMM's
   on-device symbolic phase. See [BsrGpu (GPU)](./bsr_gpu.md).
3. **Closure-driven iterative solvers** —
   `crates/molla/src/math/iterative.rs`. A second copy of CG, PCG, and
   BiCGSTAB with a matrix-free API (`F: Fn(&[Real]) -> Vec<Real>`),
   plus a built-in Jacobi preconditioner. See
   [Iterative solvers](./iterative.md).

## When CSR vs blocked

The crate defaults to **B = 1 / CSR**. The few places `B > 1` still
make sense are:

- Featherstone CRBA outputs a `dof × dof` mass matrix where every entry
  is a dense `B × B` joint block. The current code path materialises
  it as a dense `Vec<Real>` and uses `dense_chol`; if we ever needed a
  banded BSR layout this is where it would land.
- Hypothetical multi-component rigid-body Jacobians (`J ∈ R^(6 × dof)`
  in blocks of 6×1). Not used today — the spatial-Jacobian path
  materialises columns directly into a flat `Vec`.

All FEM, MPM, and GPU code uses `B = 1`. `BsrGpu::upload` errors on
`B != 1` to make the constraint explicit:

```rust
if m.b != 1 {
    return Err(Error::Build(format!(
        "BsrGpu::upload: only B = 1 supported in v1, got B = {}",
        m.b
    )));
}
```

## Design tradeoffs

The BSR format keeps two invariants that the GPU side relies on:

1. **Column indices within a row are sorted ascending.** Constructors
   (`from_pattern`, `from_triplets`, `from_scalar_triplets`) sort the
   pattern up front. SpGEMM relies on this — its numeric kernel
   binary-searches `c_col_indices[lo..hi]` to find the output slot for
   a contribution.
2. **Repeated `(row, col)` triplets accumulate into one slot.**
   `from_scalar_triplets` and `from_triplets` both coalesce duplicates;
   FEM assembly is the dominant consumer and relies on it heavily.

The format does **not** support pattern unions natively. `axpy` on the
GPU is documented as "same pattern, unchecked" — pattern-changing
operations route through SpGEMM or rebuild on the host. See
[BsrGpu (GPU)](./bsr_gpu.md) for the gory details.

## Iterative solvers — closure-driven vs trait-driven

molla ships two near-duplicate sets of CG/BiCGSTAB:

- The **trait-driven** path
  (`crates/molla/src/math/sparse.rs::{cg, bicgstab, gmres}`) takes a
  `&dyn LinearOperator` plus `&mut x` and writes in place. This is
  what the FEM solvers, MPM implicit pass, and any code that already
  has a custom `LinearOperator` impl call.
- The **closure-driven** path
  (`crates/molla/src/math/iterative.rs::{cg_solve, pcg_solve,
  bicgstab_solve}`) takes `F: Fn(&[Real]) -> Vec<Real>` and returns a
  fresh `CgResult` struct. Ad-hoc consumers and tests use this.

Both paths converge to the same iterates given the same matrix; the
choice is API ergonomics, not numerics. Style3D, Stam-fluid, and the
parameter-ID experiments use the closure-driven version because they
build their `A·x` callable inline from a few flat host buffers.

Cross-references:

- [BsrMatrix (CPU)](./bsr_cpu.md) — full CPU API.
- [BsrGpu (GPU)](./bsr_gpu.md) — GPU mirror + kernels.
- [SpGEMM design](./spgemm.md) — full-GPU symbolic phase (Stage 39F).
- [Iterative solvers](./iterative.md) — closure-driven CG/PCG/BiCGSTAB.
