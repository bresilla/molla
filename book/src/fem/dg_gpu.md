# DG GPU port — Stage 43 v4/v5 status

## Current state

Stage 43 v4 ports the SWE-specific Riemann solvers (`HLL`, `HLLC`,
`RoeSwe`) to per-face WGSL. Stage 43 v5 adds the post-flux limiters so
DG-P1 shallow-water steps can keep the flux + limiter stages resident on
the GPU instead of downloading state between SSP-RK stages. The relevant
pieces in the repository are:

- The CPU SWE flux and limiter implementations in
  `crates/molla-fem/src/dg_element.rs`.
- The per-face GPU flux dispatcher:
  `crates/molla-fem/src/gpu_kernels/dg_swe_flux.rs` +
  `crates/molla-fem/shaders/dg_swe_flux.wgsl`.
- The GPU limiter dispatcher:
  `crates/molla-fem/src/gpu_kernels/dg_swe_limiter.rs` +
  `crates/molla-fem/shaders/dg_swe_limiter.wgsl`.
- The `FemGpuData` type ([gpu_data.rs](./elasticity_gpu.md)) which
  uploads a `FemMesh` + `BsrMatrix` to the device. The DG variant
  would extend this with per-cell DoF offsets and the
  per-face match arrays.

## v4 flux path

The flux shader has one entry point per system flux:

1. `compute_swe_hll_flux_per_face`
2. `compute_swe_hllc_flux_per_face`
3. `compute_swe_roe_flux_per_face`

Each work item processes one interior face, reads the two cells' P1
DoFs, evaluates a two-point edge quadrature, and **atomically** scatters
the per-face contribution into the global residual buffer.

The per-cell mass / stiffness / interior-volume terms can reuse the
existing `fem_elasticity_assemble.wgsl` infrastructure with minor
edits (the kernel is one workgroup per cell with no atomic conflict).
The face kernel is where the v4 GPU-port complexity lives — face
quadrature, two-sided shape evaluation, and atomic scatter.

## v5 limiter path

`DgSweLimiterKernel` exposes two resident limiter passes over the same
flat DG-P1 SWE state layout (`cell * 9 + node * 3 + component`):

1. `dispatch_scalar` — component-wise scalar Barth-Jespersen limiting,
   with a hard floor for `h` and unclamped momenta.
2. `dispatch_characteristic` — Krivodonova-style characteristic limiting
   using a prepacked principal normal and the same Roe-style
   shallow-water eigenbasis as the CPU implementation.

The limiter reads from `state_in` and writes `state_out` rather than
mutating in place. This preserves the CPU algorithm's snapshot contract:
all cell averages and neighbour bounds are computed from the pre-limit
state, while the limited reconstruction is written to a separate buffer
that can be ping-ponged into the next GPU flux sweep.

## Expected speedup

Per the precedents in the FEM stack (elasticity assembly, transient
stepper) the GPU port should deliver:

- ~10-15× per-step throughput on the SWE dam-break viewer scene at
  64×64 grid (target hardware: RTX 4080).
- Lower memory bandwidth than the CPU version because the per-face
  flux only reads 6–20 DoFs (two cells × 3–10 DoFs each) and writes
  6–20 — the kernel is heavily memory-bound but the access pattern
  is regular.

## Integration with the SSP-RK stepper

With the v4/v5 GPU kernels, the host-side stepper from
[`dg_time`](./dg_time.md) needs minimal change:

```rust
ssp_rk3_step(&mut u_gpu, t, dt,
    |u, _t| compute_dg_swe_rhs_gpu(&engine, &mesh, u),  // GPU dispatch
    |u| limit_solution_gpu(&limiter, u, scratch));      // GPU limiter
```

`pack_dg_swe_limiter_cells` prepares fixed-width neighbour + principal
normal metadata once on the host. The timestep loop can then keep state
buffers resident and ping-pong `state_in` / `state_out`.

## Verification

`crates/molla-fem/tests/fem_dg_swe_gpu.rs` contains ignored adapter
tests for real-GPU parity:

- `dg_swe_flux_gpu_matches_cpu_hll_hllc_roe`
- `dg_swe_limiters_gpu_match_cpu_scalar_and_characteristic`

Both are compile-checked by the GPU test target; running them requires a
working `wgpu` adapter.

## Cross-references

- [DG flux](./dg_flux.md) — the CPU implementations that v4 mirrors.
- [DG basis](./dg_basis.md) — the per-cell DoF layout the GPU kernel
  must respect.
- [Elasticity GPU](./elasticity_gpu.md) — the closest precedent for a
  per-cell FEM assembly kernel.
- [Transient GPU stepper](./solvers.md#gpu-steppers--slice-h) — the
  closest precedent for keeping FEM state resident on the device
  across many timesteps.
