# molla — what's left to do

> **Comprehensive documentation lives in [`book/`](./book/).** This file is
> a minimal pointer to what's *still open* — the historical roadmap and
> gap analyses have been folded into `book/src/roadmap/`.

For everything that's been built, start with `book/src/introduction.md`
or the table of contents in `book/src/SUMMARY.md`. The full open-items
list with effort estimates + dependencies is in
[`book/src/roadmap/open.md`](./book/src/roadmap/open.md); the explicit
out-of-scope catalogue is in
[`book/src/roadmap/oos.md`](./book/src/roadmap/oos.md).

---

## Snapshot — what's done

Stages **0 through 51** are on `main`, including:

- Stage 38 Tier-1 (GPU sparse BSR / adj_svd3 / Adam+SGD / graph coloring / scan+radix-64) + diff-MPM (38.1) + diff-FEM (38.2)
- Stage 39A-F Tier-2 (implicit MPM, DG, Stam fluid, hydroelastic anchor, CABLE joint, full-GPU SpGEMM)
- Stage 40-49 cluster (implicit MPM v4, GPU FEM elasticity / Stokes / NS / APIC / hydroelastic, DG higher-order, 3DGS forward+adjoint+SH, AMR with hanging nodes + coarsening, magnetostatics + eddy currents with AMS/MG/AMG, NonconformingField + Signorini + Coulomb friction)

~400+ tests green; viewer scenes #1–#101.

---

## What's still open

Three tiers, summarized — see `book/src/roadmap/open.md` for full
detail (problem statement, dependencies, effort, value).

### Tier-A — small v5 follow-ups (1-3 days each)

1. Kinetic vs static friction split (Stage 50 v2)
2. Variable per-muscle stiffness + pennation angle GPU paths (Stage 42 v5)
3. SH GPU forward for diffuse/view-dep split (Stage 44 v5)
4. DG limiter GPU port (Stage 43 v5)
5. DG-AMR coarsening + level-gap recursion (Stage 45 v5)
6. Source-side surface quadrature + TetBvh acceleration (Stage 47 v5)
7. Active-set anti-cycling hardening (Stage 51 v2)

### Tier-B — algorithmic upgrades (multi-day each)

1. Pi-projection branch of Hiptmair-Xu AMS (Stage 48 v5)
2. Block-coupled preconditioning at high ωσ (Stage 48 v6)
3. Frequency sweep + Coulomb-gauge enforcement (Stage 48 misc)
4. Full-GPU NS time integrator with on-device SSP-RK (Stage 41 v7)
5. Coupled active-set + friction (Stage 49 v2 + Stage 50 integration)
6. Self-contact (47 / 49 / 50 currently cross-mesh only)

### Tier-C — new physics axes (weeks each)

1. Multi-phase flow with surface tension
2. Compressible / Euler equations (DG infrastructure ready)
3. Phase-field fracture (Allen-Cahn / Cahn-Hilliard)
4. Mooney-Rivlin / Neo-Hookean hyperelasticity tangents
5. Rate-and-state friction (Dieterich-Ruina)
6. GPU port of muscle SEE for non-default stiffness ratios
7. Krylov restart in transient FEM steppers

### Cross-cutting

- Critic round on Stages 39–51
- Viewer scenes for v3-v6 work (only scenes #94-101 exist for the recent wave)

---

## Explicitly out of scope

Per design — see `book/src/roadmap/oos.md` for full rationale:

- Python bindings (`pyo3`) — Newton serves Python, molla serves Rust.
- Style3D cloth — Stage 20 picked VBD; not a gap.
- Kamino PA-ADMM — overlaps with MuJoCo wrapper.
- TensorCore tile matmul — `cooperative_matrix` not portable in wgpu.
- CUDA Graphs — no wgpu equivalent capture API.
- Multi-GPU peer access — wgpu is single-device.

---

## How to contribute

Pick any item from Tier-A or Tier-B; consult the relevant chapter
under `book/src/` for the existing implementation, then open a branch.
Every existing stage in `main` is paired with at least one parity /
FD-gradient / convergence-rate test in `crates/molla/tests/` — new
work is expected to do the same.
