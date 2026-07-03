# CPU/GPU parity testing

The CPU/GPU parity test is molla's everyday correctness check.
Every kernel has one. The pattern is the same regardless of what
kernel is being tested.

## The recipe

`crates/molla-solvers/tests/cpu_gpu_parity.rs` is the canonical example.
Five steps:

1. **Gate the test**. `#![cfg(feature = "gpu")]` at file top.
2. **Build the same scene twice.** Once for CPU, once for GPU (or,
   more often, build once and use two state slots — the model is
   shared).
3. **Step the CPU path**, collecting per-step trajectory.
4. **Step the GPU path**, reading back state each step.
5. **Assert agreement** within tolerance.

Skeleton:

```rust
#![cfg(feature = "gpu")]

use molla_compute::{Backend, Engine};
use molla_math::{SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Model, ModelBuilder};

fn build_scene(n: usize) -> Model {
    let mut b = ModelBuilder::new();
    for i in 0..n {
        let body = b.add_body(BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(i as f64 * 1.5, 10.0, 0.0)));
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    }
    b.build().unwrap()
}

#[test] #[ignore]
fn cpu_gpu_parity_semi_implicit() {
    let engine = Engine::new(Backend::Gpu).unwrap();
    let model = build_scene(8);
    let traj_cpu = run_cpu(&model, 60, 1.0 / 60.0);
    let traj_gpu = run_gpu(&engine, &model, 60, 1.0 / 60.0);
    for (i, (a, b)) in traj_cpu.iter().zip(traj_gpu.iter()).enumerate() {
        for (j, (xa, xb)) in a.iter().zip(b.iter()).enumerate() {
            let err = (xa.position - xb.position).length();
            assert!(err < 1e-4, "step {i} body {j}: pos err {err}");
        }
    }
}
```

## Why `#[ignore]`

The default `cargo test` lane has to pass on CI runners without a
GPU. `wgpu::Adapter` initialisation fails (or worse, picks a
software adapter and runs glacially) when no real device is
available. Gating the GPU lane behind `#[ignore]` lets the same
test file:

- Build successfully (the test is compiled, just not run).
- Run on demand via `cargo test -- --ignored` when a real adapter
  is around.

The Makefile has one target per parity test family
(`make test-gpu`, `make test-gpu P=molla-solvers T=featherstone_gpu_parity`, etc.) — each is just
a `cargo test --features gpu --test <name> -- --ignored` invocation.

## Fixed seeds, deterministic inputs

Parity tests use **fixed-seed inputs** to make failures
reproducible. No `rand::thread_rng()`. The patterns are:

- Hand-laid positions (`Vec3::new(i as f64 * 1.5, 10.0, 0.0)`).
- Deterministic-hash inputs for "random" perturbations
  (`(i * 0x9E3779B9) as f32 / u32::MAX as f32`).
- Reusable fixtures beside the owning crate's tests, or in the facade
  test tree when the coverage is explicitly facade/product-level.

When a parity test fails, the developer can rerun it bit-for-bit
identical to reproduce.

## Tolerances

The tolerance budget depends on the operation count and the
narrowing path:

| Operation                        | Tolerance     | Rationale                                     |
|----------------------------------|---------------|-----------------------------------------------|
| Single-step physics (per body)   | 1e-4          | f32 narrowing introduces ~1e-6 per arithmetic op, accumulates over a step |
| Sensor readback (per ray)        | 1e-4          | Same — single dispatch                        |
| 60-substep trajectory            | 1e-3          | Tolerance grows with step count               |
| Adjoint vs FD (per gradient)     | 1e-3 (rel)    | FD step size 1e-4–1e-6; finite-diff truncation is the wall |
| SH coefficient gradient          | 1e-7          | Closed-form, no integrator state              |
| FEM solve (per-DOF)              | depends       | CG tol + f32 narrowing combine                |

The 1e-3 relative tolerance for adjoint-vs-FD is the standard
molla budget. Tighter than that is the FD truncation wall; looser
than that signals a real adjoint bug.

## What if the CPU and GPU disagree

When a parity test fails:

1. **The CPU path is the oracle.** Read it. Make sure the algorithm
   is what you think it is.
2. **Check the WGSL kernel.** Compare the math op-by-op against
   the CPU code. The most common bugs are:
   - Index typos (`v[i+1]` vs `v[i-1]`).
   - Mat3 column-major vs row-major mix-up (glam is column-major;
     WGSL `mat3x3` is column-major; the layout matches but it's
     easy to flip while debugging).
   - Atomic-CAS misuse (forgetting to bitcast, forgetting the
     loop, forgetting to break on `exchanged`).
   - Single-precision overflow in cumulative reductions
     (manifests as one body's position blowing up after thousands
     of substeps).
3. **Bisect the substep.** If step 100 disagrees but step 1
   agrees, write a smaller test that only steps once and inspects
   the divergence at the buffer level.

The WGSL has no `printf`. Debug by writing to a dedicated debug
buffer and reading it back — the parity test framework can be
extended to dump the contents of any buffer at the disagreement
step.

## Tests that don't use the parity pattern

Some test families are intentionally CPU-only:

- **Unit tests** in module `#[cfg(test)] mod tests {}` blocks.
  No `#[ignore]`. Run in the default lane.
- **Smoke tests** (`scene_smoke.rs`, `cpu_smoke.rs`) — assert that
  a build + a few steps succeed without crashing. CPU-only.
- **Reference-port tests** (`oracle_newton.rs`) — compare molla CPU
  output against Rust translations of Newton/Warp fixtures.

Some test families are intentionally GPU-only because the CPU path
doesn't exist or isn't competitive:

- **Stage 24e sparse-grid** mutable insert — only a GPU
  implementation ships.
- **LBVH build / refit** — GPU-only.
- **Tile FFT** — GPU-only.

These tests assert a GPU result against a CPU oracle hand-rolled
in the test itself (e.g. a `rustfft`-based reference for FFT, a
flat `O(n²)` reference for spatial-hash queries).

## See also

- [GPU port status matrix](./gpu_status.md) — exhaustive list of
  what has parity tests and what doesn't.
- [Test strategy](./strategy.md) — the higher-level testing
  philosophy.
