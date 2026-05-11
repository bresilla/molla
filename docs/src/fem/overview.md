# Finite-Element Method — overview

Molla ships a substantial, layered Finite-Element Method (FEM) framework
under `crates/molla/src/fem/`. It is the single largest physical-modelling
subsystem in the engine and is where molla materially **leads** its
ancestor projects: Nvidia's `newton` has *no* FEM module at all (it is a
purely particle / soft-rigid solver), and `warp` ships only the bare
infrastructure needed for the `newton` examples. Molla's FEM stack
covers heat / Poisson, linear elasticity, hyperelasticity (St Venant–
Kirchhoff and symmetric-Dirichlet distortion), wave propagation,
incompressible Stokes / Navier–Stokes (Taylor–Hood mixed elements),
Discontinuous Galerkin advection / shallow-water, eddy currents
(time-harmonic Maxwell), magnetostatics (H(curl) Nedelec elements),
adaptive mesh refinement with hanging-node constraints,
cross-mesh L²-projection (mortar), Signorini contact via Lagrange
multipliers, multigrid and AMS preconditioning, and an APIC FEM-MPM
hybrid coupling — most of which have a matching GPU port behind the
`gpu` Cargo feature.

This chapter set documents that machinery. Read it top-to-bottom for a
walking tour, or jump to a specific topic via the cross-links.

## Layered architecture

The framework is split into ~25 modules that compose along the
classical FEM data-flow:

```
mesh  ──►  function_space  ──►  quadrature  ──►  element  ──►  assembly
                                                                  │
                                                                  ▼
                                                            BsrMatrix
                                                                  │
                                  ┌───────────────────────────────┤
                                  ▼                               ▼
                                 bc                          solvers
                          (Dirichlet / Neumann /              (steady, transient,
                           Nitsche / Signorini)               Newmark, Newton,
                                                              adjoint, GPU,
                                                              russell direct)
```

The five "core" chapters mirror this layering:

- [mesh](./mesh.md) — `FemMesh` + `FemCell::{Tri, Tet}`, grid builders,
  Gmsh MSH 4.1 import, `compute_edges`.
- [quadrature](./quadrature.md) — Gauss rules: 1 / 3 / 6-pt on
  triangles, 1 / 4 / 5 / 11-pt on tets, plus 3D-tri surface rules used
  by the Stage 47 v3 nonconforming projection.
- [elements](./elements.md) — per-cell `B^T D B` kernels: mass,
  stiffness (Laplacian / wave / `α∇²`), linear elasticity, St Venant–
  Kirchhoff stress, convection, SUPG stabilisation, symmetric-Dirichlet
  distortion energy.
- [assembly](./assembly.md) — closure-driven `assemble_global` that
  scatters local matrices into a `BsrMatrix` (`B = 1`, CSR-shaped).
- [function spaces](./mesh.md#function-spaces) — exposed via the
  `FunctionSpace` enum: `P1Nodal`, `P1Vector { dim }`, `P2Nodal`,
  `P2Vector { dim }`, `DGNodal { degree }`, `NedelecEdge { order }`.

The five "physics" chapter clusters layer on top:

- **[Boundary conditions](./bc.md)** — strong Dirichlet (symmetric
  reduction), `NeumannTraction` integration, DG Nitsche
  weak-Dirichlet.
- **[Solvers](./solvers.md)** — steady-state (CG + PCG), transient
  (backward Euler), wave (Newmark-β), Newton–Raphson, GPU-resident
  steppers, distortion-minimisation, optional `russell` direct LU.
- **[P2 nodal elements](./p2.md)** and the **[DG cluster](./dg.md)** —
  higher-order continuous and discontinuous bases.
- **[Stokes / Navier–Stokes](./stokes_ns.md)** — Taylor–Hood mixed
  elements + Uzawa-CG and the FFT-Poisson alternative.
- **[APIC coupling](./apic.md)**, **[adjoint](./adjoint.md)**,
  **[AMR](./amr.md)**, **[elasticity-GPU](./elasticity_gpu.md)**,
  **[magnetostatics](./magneto.md)**, **[eddy currents](./eddy.md)**,
  **[multigrid / AMG](./multigrid.md)**,
  **[nonconforming](./nonconforming.md)**,
  **[Signorini contact](./signorini.md)**,
  **[friction](./friction.md)** — each its own self-contained chapter.

## Storage convention

Every assembled global matrix lives in `crate::math::sparse::BsrMatrix`
with block size `B = 1` — i.e. a plain CSR. The choice is deliberate:
CG / GMRES / PCG only need `matvec`, the `BsrMatrix` builder
`from_scalar_triplets` already dedupes coincident `(row, col)` entries,
and the GPU port (`FemGpuData`) is a direct mirror of the same flat
arrays. Vector-valued fields are stored **interleaved** — DoF `v · d + c`
for vertex `v`, component `c`, dim `d` — so the scalar CSR row count
equals `n_vertices · d` for vector spaces.

The element visitor returns a dense `Vec<Vec<Real>>` of size
`(n_v · d) × (n_v · d)` per cell. This is intentionally simple at the
expense of allocator churn — Stage 36.5 moved the hot per-cell payload
to a packed `[Real; 144]` style buffer on the GPU side, but the CPU
visitor API kept the dense form for readability.

## Layered correctness guarantees

Every contract test in `crates/molla/tests/fem_*.rs` exercises one
layer in isolation:

| Layer            | Test                                            | Property checked                                                              |
|------------------|-------------------------------------------------|-------------------------------------------------------------------------------|
| quadrature       | `bary_coords_partition_unity`                   | `∑ λᵢ = 1` at every quad point.                                              |
| element          | `unit_tet_stiffness_kills_constant_field`       | `K · 1 = 0` per element ⇒ Galerkin invariant.                                |
| assembly         | `mass_matrix_row_sum_equals_lumped_mass`        | `∑ M = |Ω|` exactly.                                                          |
| BC               | `dirichlet_symmetric_reduction_preserves_symmetry` | Reduced system stays SPD after symmetric column elimination.              |
| solver           | `steady_state_constant_dirichlet_yields_constant` | Laplace with all-boundary Dirichlet `= 5` ⇒ `u ≡ 5` inside.                |
| P2               | `p2_tet_mass_total_equals_volume`               | `∑∑ Mᵢⱼ = vol(K)` (partition-of-unity squared).                              |
| DG               | `ssp_rk3_is_3rd_order`                          | Empirical 3rd-order time accuracy with SSP-RK3.                              |

When you change something deep in the stack — say the quadrature
weights or the BC reducer — these tests typically catch it at the
right layer, so you don't have to debug the wrong one.

## Why molla leads here

`xtra/newton/` (the Nvidia reference under `xtra/`) has **no** FEM
solver — it is a particle / rigid-body / cloth engine that delegates
all field problems to a hand-rolled BVH + Position-Based-Dynamics
loop. `xtra/warp/` ships compute primitives (sparse matvec, FFT, BVH,
mesh queries) but no element kernels and no assembly machinery.

The molla FEM stack is therefore **architecturally novel** within the
project family. Where the reference codebases overlap with molla
(particles, rigid bodies, contact graphs, MPM) we copy the Newton / Warp
algorithms verbatim. Where they don't — *every chapter in this
section* — we wrote the algorithm from textbook references (Hughes,
Zienkiewicz, Wriggers for contact, Cockburn-Karniadakis-Shu for DG,
Toro for SWE, Hiptmair-Xu for AMS, Solin for Nedelec) and validated
against analytic / convergence-rate tests. Every chapter that follows
calls out the textbook reference and the test that locks the
implementation in.

## File map

| Chapter file                                  | Source file(s)                          |
|----------------------------------------------|-----------------------------------------|
| [mesh.md](./mesh.md)                          | `mesh.rs`                              |
| [quadrature.md](./quadrature.md)              | `quadrature.rs`                        |
| [elements.md](./elements.md)                  | `element.rs`                           |
| [assembly.md](./assembly.md)                  | `assembly.rs`                          |
| [bc.md](./bc.md)                              | `bc.rs`                                |
| [solvers.md](./solvers.md)                    | `solver.rs`, `russell.rs`              |
| [p2.md](./p2.md)                              | `p2_element.rs`                        |
| [dg.md](./dg.md) and sub-chapters             | `dg_element.rs`, `dg_time_integrator.rs` |
| [stokes_ns.md](./stokes_ns.md)                | `stokes.rs`, `stokes_gpu.rs`, `fft_ns.rs` |
| [apic.md](./apic.md)                          | `apic_coupling.rs`, `apic_coupling_gpu.rs` |
| [adjoint.md](./adjoint.md)                    | `adjoint.rs`                           |
| [amr.md](./amr.md)                            | `amr_mesh.rs`                          |
| [elasticity_gpu.md](./elasticity_gpu.md)      | `elasticity_gpu.rs`                    |
| [magneto.md](./magneto.md)                    | `magneto.rs`, `nedelec_element.rs`     |
| [eddy.md](./eddy.md)                          | `eddy_current.rs`                      |
| [multigrid.md](./multigrid.md)                | `multigrid.rs`                         |
| [nonconforming.md](./nonconforming.md)        | `nonconforming.rs`                     |
| [signorini.md](./signorini.md)                | `signorini.rs`                         |
| [friction.md](./friction.md)                  | (planned, Stage 50)                    |

Cross-references throughout each chapter point you back to the
matching source file — `crates/molla/src/fem/<file>.rs` — so you can
read the doc alongside the actual implementation.
