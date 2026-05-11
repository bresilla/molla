# Stokes and Navier-Stokes

Incompressible Stokes / Navier–Stokes lives in
`crates/molla/src/fem/stokes.rs` (~1600 lines) and uses
**Taylor–Hood mixed elements** — P2 vector velocity + P1 scalar
pressure — solved with a SIMPLE-style projection iteration on the
pressure-Schur complement. A FFT-based alternative for periodic 2D
grids lives in `fft_ns.rs`. The GPU port `stokes_gpu.rs` keeps the
matrices resident on the device.

## The Taylor-Hood mixed space

```rust
pub struct MixedTaylorHood {
    pub dim: usize,           // 2 (plane-strain) or 3
    pub map: P2DofMap,        // velocity addressing
    pub n_velocity: usize,    // (vert + edge) · dim
    pub n_pressure: usize,    // vert_count
}
```

- **Velocity** lives in `P2Vector { dim }` — corner DoFs + edge-
  midpoint DoFs, each carrying `dim` components, interleaved as
  `[ux₀, uy₀, uz₀, ux₁, ...]`.
- **Pressure** lives in `P1Nodal` — one scalar per mesh vertex.

The Taylor–Hood pair is **LBB-stable** (inf-sup stable) for
incompressible Stokes — the canonical safe choice for mixed
velocity-pressure formulations. Equal-order P1/P1 *is not stable* and
would need extra stabilisation; Taylor–Hood doesn't.

## The Stokes saddle-point system

\\[
\begin{pmatrix} A & B^T \\ B & 0 \end{pmatrix}
\begin{pmatrix} u \\ p \end{pmatrix} =
\begin{pmatrix} f \\ 0 \end{pmatrix}
\\]

with:

- **A** = velocity block = component-decoupled `μ · ∇u : ∇v` Laplacian.
  Each component is a P2 stiffness matrix scaled by viscosity `μ`.
- **B** = divergence block = `−∫_Ω q · ∇·u dx` evaluated against the P1
  pressure test function `q`.

The system is **indefinite** (it has the saddle-point structure with
zero in the (2,2) block), so the molla CG cannot solve it directly.
Instead we use a **SIMPLE-style projection** that splits the velocity
update from the pressure correction:

## SIMPLE iteration

1. **Predictor:** \\( A \cdot u^* = f - B^T \cdot p_{\text{old}} \\). Solve
   via CG on the velocity-block matrix (component-decoupled SPD).
2. **Pressure correction:** \\( (B A_{\text{diag}}^{-1} B^T) \cdot \delta p = -B \cdot u^* \\).
   The Schur complement on the diagonal of `A`. `A_diag` is the lumped
   diagonal of `A` (cheap; gives a diagonal SPD operator). Solve via
   CG on the matrix-free closure
   \\( q \mapsto B \cdot (A_{\text{diag}}^{-1} \cdot B^T \cdot q) \\).
3. **Update:**
   \\( u = u^* - A_{\text{diag}}^{-1} B^T \cdot \delta p \\),
   \\( p = p_{\text{old}} + \alpha \delta p \\).
4. Repeat until \\( \|B u\| < \text{tol} \\) (incompressibility
   residual).

`α` is a relaxation factor (typical 0.7–1.0 for Stokes; lower for
stiff Navier-Stokes). The whole inner loop is built from `cg_solve`
+ matvec closures — no specialised saddle-point solver needed.

## Public API

```rust
pub fn solve_stokes_steady(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    config: &StokesSimpleConfig,
    bcs: &[StokesBc],
    viscosity: Real,
) -> Result<StokesResult>;

pub fn solve_stokes_with_matrices(...);
```

`StokesBc` is its own enum (separate from
[`BoundaryCondition`](./bc.md) because velocity needs vector-valued
pinning and pressure needs a single-vertex anchor):

```rust
pub enum StokesBc {
    VelocityDirichlet { node: P2NodeRef, values: Vec<Real> },
    PressureAnchor    { vertex: u32, value: Real },
    NeumannTraction   { face: [u32; 3], traction: [Real; 3] },
}
```

The pressure anchor breaks the constant-pressure nullspace (the
system is otherwise rank-deficient by one). Without it, CG drifts
along the nullspace direction and `δp` grows unboundedly.

## Per-element kernels

```rust
pub fn compute_element_stokes_velocity_block(cell, vertices, dim, mu)
    -> Vec<Vec<Real>>;    // (P2-vector × P2-vector) — the A^e block

pub fn compute_element_stokes_div_block(cell, vertices, dim, vert_count, n_pressure)
    -> Vec<Vec<Real>>;    // (P1 × P2-vector) — the B^e block

pub fn compute_element_mass_p2_vector(cell, vertices, dim)
    -> Vec<Vec<Real>>;    // (P2-vector × P2-vector) — for the transient case
```

The velocity block is **block-diagonal in components**: with the
Laplacian form \\( \mu \int \nabla u : \nabla v \\) (rather than the
symmetric stress form), the per-component blocks decouple and the
assembler builds `dim` copies of the same P2 scalar stiffness, scaled
by `μ`. Approximating the symmetric stress with the Laplacian is
exact for fully-Dirichlet boundaries (the boundary integral cancels),
which is what the lid-driven cavity test uses.

The divergence block uses the gradient identities
\\( \int_K q \nabla \cdot v \, dx = -\int_K \nabla q \cdot v \, dx + \text{(boundary)} \\)
and integrates via the 6-pt Dunavant rule on triangles or the 11-pt
Keast rule on tets.

## Time-stepping (Navier-Stokes)

```rust
pub struct NavierStokesConfig {
    pub dt: Real,
    pub viscosity: Real,
    pub max_simple_iters: usize,
    pub simple_tol: Real,
    pub max_advection_iters: usize,  // outer Picard / Newton on (u·∇)u
    pub advection_tol: Real,
}

pub fn solve_navier_stokes_step(mesh, th, config, bcs, u_old, p_old) -> Result<StokesResult>;
pub fn solve_navier_stokes_transient(...);
```

The momentum update is **IMEX**: implicit for the viscous + pressure
parts, explicit for the convection `(u·∇)u` term. This is the
classic Chorin–Marchuk fractional-step recipe.

For high Reynolds numbers, the explicit convection becomes
oscillatory; the kernel-level
[`compute_element_supg_convection`](./elements.md#convection--supg-stabilisation)
adds Hughes' streamline-upwind stabilisation, exposed at the
assembly level via:

```rust
pub fn assemble_convection_block_with_supg(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    velocity_field: &[Real],
    alpha: Real,
    h_per_cell: &[Real],
) -> Result<BsrMatrix>;
```

## Lid-driven cavity test

```rust
pub fn build_cavity_bcs_2d(mesh: &FemMesh, th: &MixedTaylorHood, lid_velocity: Real) -> Vec<StokesBc>;
```

builds the canonical Stokes / NS test: zero velocity on three walls,
\\( (u_{\text{lid}}, 0) \\) on the top wall, pressure anchored at the
bottom-left corner. The viewer scene `StokesCavity2D` runs this and
the resulting streamlines match Ghia-Ghia-Shin's 1982 reference at
Re = 100.

## FFT-Poisson fractional-step (`fft_ns.rs`)

```rust
#[cfg(feature = "gpu")]
pub fn solve_navier_stokes_fractional_step(
    device, queue, kernel,
    n_grid: usize,         // power of two
    num_steps: usize,
    dt: f32, nu: f32,
) -> Result<Vec<[f32; 2]>>;
```

A **complementary** path to the Taylor-Hood Stokes solver. Same
pipeline (predictor → pressure-Poisson → corrector) but discretised
on a **uniform 2D grid** with **periodic boundaries**, and the
pressure-Poisson solve uses the GPU FFT kernel instead of CG on a
Schur complement:

\\[
\hat p(\mathbf{k}) = -\frac{1}{|\mathbf{k}|^2} \cdot \frac{1}{\Delta t} (i \mathbf{k} \cdot \hat u^*(\mathbf{k})),
\quad \hat p(\mathbf{0}) = 0
\\]

with the constant nullspace gauge-fixed at \\( \mathbf{k} = 0 \\). The
`PoissonStencil` enum picks between the tight 5-point stencil (for
staggered-grid gradients) and the wide stencil (for collocated
central-difference gradients) — passing the wrong one produces
checkerboard divergence on the odd-even mode.

This is the path the `WarpNavierStokesFftDemo` viewer scene uses
because it gets to a working 2D NS in two screens of code.

## GPU port — `stokes_gpu.rs`

```rust
#[cfg(feature = "gpu")]
pub fn solve_stokes_gpu(...) -> Result<StokesGpuResult>;

#[cfg(feature = "gpu")]
pub fn solve_navier_stokes_gpu(...) -> Result<NavierStokesGpuResult>;
```

Uploads `A`, `B`, the mass matrix, BC metadata once; runs the SIMPLE
loop with GPU sparse-matvec + resident CG inside each predictor /
corrector. Stage 41 v5 added async-dot fusion for the inner CG so
each iteration is a single compute pass.

The shader pair is:

- `crates/molla/shaders/fem_stokes_assemble.wgsl` — per-cell assembly
  of the velocity and divergence blocks (one workgroup per cell).
- `crates/molla/shaders/fem_ns_advect.wgsl` /
  `fem_ns_rhs_assemble.wgsl` — Navier-Stokes per-step convection +
  RHS construction.

## Cross-references

- [P2 elements](./p2.md) — the velocity basis and DoF map.
- [Element kernels](./elements.md#convection--supg-stabilisation) —
  per-cell SUPG.
- [Solvers](./solvers.md) — `cg_solve` driving the SIMPLE inner loops.
- [APIC](./apic.md) — the FEM-MPM hybrid uses the Stokes solver's P2
  velocity block as its mesh-side velocity update (planned for Stage
  37.6.1; v1 ships viscous smoothing instead).
