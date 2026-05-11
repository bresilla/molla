# `BsrMatrix` (CPU)

`crates/molla/src/math/sparse.rs` defines the host-side block-sparse-row
matrix and its iterative solvers.

## Layout

```rust
#[derive(Debug, Clone)]
pub struct BsrMatrix {
    pub b: usize,             // block dimension; B = 1 ⇒ CSR
    pub block_rows: usize,
    pub block_cols: usize,
    pub row_offsets: Vec<usize>,  // length block_rows + 1
    pub col_idx: Vec<usize>,      // length = total nonzero blocks
    pub values: Vec<Real>,        // b² · nnz_blocks values, row-major per block
}
```

The conceptual size is `(block_rows · B) × (block_cols · B)`. Within
each block the values are stored row-major. Across blocks within a
row the `col_idx` is sorted ascending; constructors enforce this.

## Construction

```rust
impl BsrMatrix {
    pub fn from_pattern(
        b: usize,
        block_rows: usize,
        block_cols: usize,
        pattern: Vec<(usize, usize)>,
    ) -> Result<Self>;

    pub fn from_triplets(
        b: usize,
        block_rows: usize,
        block_cols: usize,
        triplets: &[(usize, usize, Vec<Real>)],
    ) -> Result<Self>;

    pub fn from_scalar_triplets(
        nrows: usize,
        ncols: usize,
        triplets: &[(usize, usize, Real)],
    ) -> Result<Self>;
}
```

`from_pattern` allocates a zero-valued matrix with the given block
positions; the pattern is sorted+deduped internally. `from_triplets`
takes a list of `(block_row, block_col, B² values)` and accumulates
duplicates into a single slot. `from_scalar_triplets` is the fast
path for `B = 1` matrices — takes flat `(row, col, value)` and avoids
the per-triplet `Vec<Real>` allocation that dominated FEM assembly
time on 16³+ meshes.

A contract test pins the equivalence between the two triplet builders
so callers can rely on byte-identical CSR layout regardless of which
path they use:

> `from_scalar_triplets` must produce byte-identical CSR
> (`row_offsets`, `col_idx`, `values`) to `from_triplets` with
> `vec![val]` wrappers. Locks the equivalence the FEM assembly path
> now relies on.

Empty rows produce equal consecutive `row_offsets[]`; this is
exercised by the `from_scalar_triplets_handles_empty_rows_and_interior_cols`
test in the source.

## Algebra

```rust
impl BsrMatrix {
    pub fn nrows(&self) -> usize;          // block_rows * b
    pub fn ncols(&self) -> usize;          // block_cols * b
    pub fn diagonal(&self) -> Vec<Real>;   // B = 1 only
    pub fn mv(&self, x: &[Real], y: &mut [Real]) -> Result<()>;
    pub fn get_diag(&self) -> Vec<Real>;   // BSR-aware diagonal
    pub fn scale(&mut self, s: Real);
}
```

`mv` is the matvec `y = A · x`. `get_diag` returns the per-row-block
diagonal as a flat `(block_rows · b)` vector — collects the diagonal
of each diagonal block. `diagonal` is a `B = 1`-only convenience that
returns scalars; it's the cheaper path for the Jacobi preconditioner
when you know the matrix is CSR.

## `LinearOperator` trait + `ClosureOp`

```rust
pub trait LinearOperator {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn mv(&self, x: &[Real], y: &mut [Real]) -> Result<()>;
}

pub struct ClosureOp<F>
where F: Fn(&[Real], &mut [Real]) -> Result<()> {
    pub rows: usize,
    pub cols: usize,
    pub f: F,
}
```

`BsrMatrix` implements `LinearOperator`. `ClosureOp` is the adapter
that lets a closure stand in for a matrix — handy when the operator is
formed implicitly (`(J · Jᵀ + λI) · x` for IK normal equations, the
implicit-MPM grid solve, FEM Schur complements).

## Iterative solvers

```rust
#[derive(Copy, Clone, Debug)]
pub struct IterConfig {
    pub max_iter: usize,
    pub tol: Real,
    pub restart: usize,  // GMRES only
}

#[derive(Copy, Clone, Debug)]
pub struct IterStats {
    pub iters: usize,
    pub residual: Real,
    pub converged: bool,
}

pub fn cg(
    op: &dyn LinearOperator,
    b: &[Real],
    x: &mut [Real],
    m_inv_diag: Option<&[Real]>,
    cfg: &IterConfig,
) -> Result<IterStats>;

pub fn bicgstab(
    op: &dyn LinearOperator,
    b: &[Real],
    x: &mut [Real],
    m_inv_diag: Option<&[Real]>,
    cfg: &IterConfig,
) -> Result<IterStats>;

pub fn gmres(
    op: &dyn LinearOperator,
    b: &[Real],
    x: &mut [Real],
    m_inv_diag: Option<&[Real]>,
    cfg: &IterConfig,
) -> Result<IterStats>;
```

All three solvers write into `x` in place — pass an initial guess if
you have one (warm-start from a previous solve, for example). The
`m_inv_diag` argument is the Jacobi preconditioner: pass `None` for
unpreconditioned, or `Some(&inv_diag)` where `inv_diag[i] == 1.0 /
diag(A)[i]` (zero diagonals become zero — pass-through). The
trait-driven path is what the FEM solvers, MPM implicit pass, and
multigrid V-cycles call.

GMRES restarts at `cfg.restart` iterations (default 30); the implementation
uses Givens rotations to maintain the Hessenberg factorisation
in-place and falls back to upper-triangular back-sub at restart time.

A perf gate enforces that 1000×1000 tridiagonal CG converges in under
100 ms in release mode — see the `cg_meets_dod_1000x1000_in_under_100ms`
test.

## Cross-references

- [Sparse linear algebra index](./sparse.md) — when to pick which
  layer.
- [Iterative solvers](./iterative.md) — closure-driven variants of the
  same algorithms.
- [BsrGpu (GPU)](./bsr_gpu.md) — GPU mirror of this type.
