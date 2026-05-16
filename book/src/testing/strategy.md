# Test strategy

molla treats tests as part of the engine, not as documentation. The
quality bar from `PLAN.md` §1.3 is:

> Every shipped feature must have:
>
> 1. A trajectory or unit test against the Newton/Warp references (where
>    applicable).
> 2. A doctest or example demonstrating idiomatic use.
> 3. A bench in `criterion` measuring cost per step.
> 4. CPU-fallback parity: the same algorithm runs on CPU
>    (single-threaded) for debugging, and produces bit-identical
>    results modulo floating-point ordering nondeterminism on GPU.

Integration tests and Criterion benches live with the crate that owns the
functionality under test: for example FEM tests in `crates/molla-fem/tests/`,
solver and dynamics tests in `crates/molla-solvers/tests/`, kernel parity tests
in `crates/molla-kernels/tests/`, and facade/product compatibility tests in
`crates/molla/tests/`. Most GPU tests are `#[ignore]`-gated so the default
`cargo test` lane runs CPU-only on machines that don't have a wgpu adapter.

## The testing pyramid

```
                  ┌──────────────────────────┐
                  │  reference-port tests    │   algorithm fixtures
                  │  (Newton + Warp parity)  │
                  ├──────────────────────────┤
                  │  CPU/GPU parity tests    │   #[ignore]'d, GPU-only
                  │  (same scene both paths) │
                  ├──────────────────────────┤
                  │  FD gradient tests       │   compare AD vs finite-diff
                  │  (adjoint correctness)   │
                  ├──────────────────────────┤
                  │  convergence rate tests  │   manufactured solutions
                  │  (FEM, h refinement)     │
                  ├──────────────────────────┤
                  │  smoke / scene tests     │   "does it build + step?"
                  ├──────────────────────────┤
                  │  unit tests              │   per-module algorithm checks
                  └──────────────────────────┘
```

## Reference-port tests

The read-only `xtra/newton/` and `xtra/warp/` trees are treated as algorithm
references, not as runtime dependencies. The pattern:

1. Author a scene in molla using `ModelBuilder`.
2. Port the upstream fixture or formula into Rust.
3. Step molla for N substeps with the same `dt` and solver intent.
4. Compare body poses, velocities, contacts, or scalar diagnostics to within a
   documented tolerance.

The facade-level ported assertions live at `crates/molla/tests/oracle_newton.rs`
and do not shell out to external runtimes.

Conventions documented in the file header:

- Newton's narrow-phase returns a *signed distance* (positive =
  separated, negative = penetrating). molla returns
  `Option<ContactManifold>` where `depth` is positive penetration.
  The oracle adapter translates between the two.
- Newton's box "size" is the half-extent vector; molla calls it
  `half_extents`. Interchangeable.
- Both follow "normal points from geom 0 toward geom 1" (`a → b`).

## CPU/GPU parity (the everyday check)

The most-run test family. Every `_gpu` test follows the same recipe, regardless
of whether it lives in `crates/molla/tests/` or in an owning functionality crate:

1. Build a `Model` with fixed-seed inputs.
2. Step on CPU for N steps; collect trajectory.
3. Reupload `Model` to the GPU, step on GPU for N steps; collect
   trajectory.
4. Compare per-step: assert `‖pos_cpu - pos_gpu‖ < 1e-4` (the
   f32-narrowing tolerance), `‖quat_cpu - quat_gpu‖ < 1e-4`, etc.

The CPU and GPU paths intentionally produce slightly different
results because:

- WGSL is `f32` and the host is `f64` — narrowing introduces
  ULPs of error per arithmetic op.
- Floating-point reductions on GPU are order-dependent
  (atomic-add via CAS races); CPU reductions are sequential. Two
  GPU runs of the same kernel are not bit-identical to each other
  either.

The tolerance budget (1e-4 over hundreds of substeps) is what the
parity tests assert.

## Manufactured solutions for FEM

For FEM, the trajectory-oracle pattern doesn't apply (PDE solvers
don't have analytic trajectories). Instead we use the **method of
manufactured solutions**:

1. Pick an exact solution `u_ex(x, y, z, t)` that's smooth.
2. Compute the residual `r = ∂u_ex/∂t − α∇²u_ex` symbolically.
3. Use `r` as the RHS source term in the FEM solve.
4. Run the solve. The numerical solution should approach `u_ex`
   at the documented convergence rate (`O(h²)` for P1, `O(h³)`
   for P2).
5. Run on a mesh sequence (8³, 16³, 32³, ...) and assert the
   error ratio matches the theoretical rate.

These tests live alongside the regular FEM tests in `fem_*.rs`. The
ones that explicitly call out manufactured solutions:

- `fem_v1.rs` — P1 heat equation, manufactured `sin(πx)sin(πy)`.
- `fem_p2.rs` — P2 heat, same manufactured solution, asserts the
  better convergence rate.
- `fem_stokes.rs` — Taylor-Hood Stokes vs Ghia & Ghia reference.

## Convergence rate tests

For iterative solvers (CG / PCG / BiCGStab / GMRES), tests assert
that the iteration count grows as expected with problem size and
shrinks with preconditioning:

- `fem_pcg_cpu.rs` — CG vs PCG iteration sweep over 8³, 16³, 24³,
  32³, 40³, 48³ meshes. PCG cuts iters by 25–31 %.
- `iterative.rs` — CG on a 1D Laplacian, asserts iter count
  bounded by `O(√n)`.

## Performance gates

Some tests are perf gates rather than correctness checks. They
assert that a measured ratio (CPU vs GPU, naive vs batched, ...)
beats a threshold:

- `mpm_perf.rs::mpm_step_n_beats_individual_steps` — asserts
  `step_n(60)` is at least 2× faster than 60× `step()`. (Measured:
  9.1×.)
- `fem_assembly_perf.rs` — asserts GPU assembly beats CPU at 16³+.
- `mesh_bvh_perf.rs` — asserts BVH query throughput at scale.

When a perf gate fails it's the most actionable kind of failure —
you have a measurement, a threshold, and a target to chase.

## What's not covered by tests

Documenting honest gaps:

- **No fuzz testing.** Tested inputs are deterministic; no
  randomized stress harness.
- **No long-running drift tests.** Tests run hundreds of substeps,
  not millions. Long-term energy drift is observed but not
  asserted.
- **No multi-platform CI.** The Makefile assumes nix + Vulkan; tests
  run only on Linux. Metal / DX12 paths compile but aren't
  CI-exercised.
- **No external-runtime diff CI.** Reference trees are not built or executed by
  molla; tests should port the relevant invariant into Rust.

These are roadmap items. Until they materialise, the everyday
contract is "the test suite passes and the parity tolerances
hold".

## See also

- [Parity testing](./parity.md) — the CPU/GPU parity pattern in
  detail.
- [FD gradient checks](./fd_gradients.md) — central-difference vs
  analytic adjoint pattern.
- [Benchmarks](./benchmarks.md) — what each criterion bench
  measures.
- [GPU port status matrix](./gpu_status.md) — which solvers have
  GPU paths, which don't, which have parity tests.
