# GPU linear elasticity

`crates/molla/src/fem/elasticity_gpu.rs` is the end-to-end GPU port of
the linear-elasticity solve. It builds the BC-reduced sparsity
pattern on CPU, hands it to the GPU via `FemGpuData`, dispatches a
WGSL kernel that computes the 144 per-cell `Bᵀ D B` values, and
solves \\( K u = f \\) on a fully GPU-resident CG. The CPU never sees
the assembled `K` values — only the sparsity pattern, the RHS, and
the BC indices cross the host bound.

## Why pattern on CPU, values on GPU?

The sparsity-pattern build via `BsrMatrix::from_scalar_triplets` is
an \\( O(\mathrm{nnz} \cdot \log \mathrm{nnz}) \\) sort-and-dedupe.
Even at 16³ tet resolution this is sub-millisecond and dominated by
the underlying mesh-graph traversal — the host CPU does it fine. The
expensive part — computing the 144 values of every per-cell
\\( K_e = \int_K B^T D B \, dV \\) (4 verts × 3 dofs)² for a tet — is
what moves to the GPU kernel.

## API

```rust
#[cfg(feature = "gpu")]
pub fn build_elasticity_gpu_pattern(
    mesh: &FemMesh,
    dim: usize,
    bcs: &[BoundaryCondition],
) -> Result<(BsrMatrix, Vec<u32>)>;  // (k_pattern, bc_mask)

#[cfg(feature = "gpu")]
pub fn assemble_elasticity_gpu_only(
    engine: &Engine,
    mesh: &FemMesh,
    youngs: Real,
    poisson: Real,
    dim: usize,
    k_pattern: &BsrMatrix,
    bc_mask: &[u32],
) -> Result<FemGpuData>;

#[cfg(feature = "gpu")]
pub fn solve_elasticity_gpu(
    engine: &Engine,
    mesh: &FemMesh,
    youngs: Real,
    poisson: Real,
    dim: usize,
    bcs: &[BoundaryCondition],
    f: &[Real],
) -> Result<ElasticityGpuResult>;

pub struct ElasticityGpuResult {
    pub u: Vec<Real>,
    pub iters: usize,
    pub residual: Real,
}
```

## Pattern construction

`build_elasticity_gpu_pattern` produces a `BsrMatrix` with the correct
`row_offsets` and `col_idx` arrays but whose `values` already encode
the BC-reduced \\( K \\):

- BC rows contain `1.0` on the diagonal and `0.0` elsewhere.
- Non-BC rows are zero — waiting for the GPU assembly kernel to
  scatter their contributions.

The function works by:

1. Assembling the unreduced \\( K \\) on CPU **with dummy ones for
   every element entry** — we only need the column-structure, not the
   values — by calling `compute_element_elasticity_stiffness(c, v, λ=0, μ=1)`.
2. Running `apply_bcs` to fold the BC structure in (so the column
   elimination updates the pattern).
3. Wiping `k_pattern.values` to zero.
4. Setting `values[diag(r)] = 1.0` for every pinned row `r`.

The returned `bc_mask: Vec<u32>` marks which rows are pinned
(`bc_mask[r] == 1`).

## Assembly kernel

`assemble_elasticity_gpu_only` uploads the mesh + pattern via
`FemGpuData::upload`, then dispatches the
`fem_elasticity_assemble.wgsl` kernel — **one workgroup per cell**:

```
@compute @workgroup_size(1)
fn assemble_cell(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell_idx = gid.x;
    // Read 4 vertex positions, compute Bᵀ D B for this cell,
    // scatter the 144 (i, j) entries into the global CSR.
    // Skip rows where bc_mask[row] == 1.
}
```

The kernel:

1. Reads the 4 vertex positions of the cell.
2. Computes the tet's volume and the 4 shape-gradient `Vec3`s
   (inverse-transpose of `D` matrix).
3. Builds the local 6 × 12 \\( \mathbf{B} \\) (Voigt notation) and
   the 12 × 12 product \\( B^T D B \cdot \text{vol} \\) in registers.
4. For each `(i, j)` entry in the 12 × 12, looks up the slot in the
   global CSR `col_idx` and `atomicAdd`s the value into
   `csr_values[slot]`.
5. Skips the scatter for any row where `bc_mask[row] == 1` — those
   stay at identity-row values.

The slot lookup is a linear scan over the row's `col_idx` window
(small — bounded by the per-row nnz, typically 4–10 for a tet mesh)
so the kernel is simple enough to stay efficient without a hash
table.

## BC handling — asymmetric in v1

The v1 elasticity_gpu path leaves an **asymmetric** reduced matrix:

- BC rows are pinned (diagonal = 1, off-diagonals = 0).
- BC columns of non-BC rows are *not* zeroed.

The correctness argument: BC rows are decoupled (zero off-diagonals,
identity diagonal, RHS = BC value), so CG never moves them away from
the seeded value. For **zero** BC values (the cantilever-clamp case),
the non-zero entries in BC columns multiply by zero displacement in
every matvec — harmless.

For **non-zero** BC values, the caller must pre-compute the f
correction `f[i] -= K_full[i, r] · v_bc[r]` on CPU and pass the
corrected RHS. This v1 helper assumes the caller already did that
(e.g. by running `apply_bcs` on the host before calling us). A
follow-up stage would add an explicit symmetric column-zeroing
kernel.

## Solve loop

`solve_elasticity_gpu` wires everything together:

```rust
let (k_pattern, bc_mask) = build_elasticity_gpu_pattern(mesh, dim, bcs)?;
let k_gpu = assemble_elasticity_gpu_only(engine, mesh, E, ν, dim, &k_pattern, &bc_mask)?;
let result = solve_steady_state_gpu_resident_pcg(
    &k_gpu, &f_red, &bcs_for_init_guess,
    engine, max_iters, tol)?;
```

— so the per-iteration cost is one GPU sparse-matvec, one inv-diag
apply (Jacobi preconditioner), and a handful of GPU dot-products.
The Stage 41 v1 measurement on a 16³ tet cantilever (4913 vertices,
14739 DoFs) was ~25 ms per Newton iteration on RTX 4080, vs ~200 ms
on CPU — about **8× speedup**, dominated by the per-cell `Bᵀ D B`
arithmetic on GPU.

## Validation

The contract tests in `crates/molla/tests/fem_elasticity_gpu.rs`
compare GPU vs CPU on a fixed cantilever fixture:

- Same `K` values to ~1e-4 relative (f32 vs f64).
- Same `u` (post-solve) to ~1e-3 relative — within the documented
  GPU CG tolerance.
- Same total strain energy.

## Cross-references

- [Elasticity element kernel](./elements.md#linear-isotropic-elasticity)
  — the math the GPU kernel mirrors.
- [Assembly](./assembly.md) — the CPU pattern builder.
- [BC](./bc.md) — the reducer that produces the initial values for
  the pattern.
- [GPU steppers (Slice H)](./solvers.md#gpu-steppers--slice-h) — the
  resident-CG plumbing the elasticity solver chains into.
- [GPU data](./elasticity_gpu.md#assembly-kernel) — `FemGpuData`
  uploads the BSR + mesh in one shot.
