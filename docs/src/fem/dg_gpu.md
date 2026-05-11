# DG GPU port — Stage 43 v4 status

## Current state

The Stage 43 v4 spec calls for a GPU port of the SWE-specific Riemann
solvers (`HLL`, `HLLC`, `RoeSwe`) in per-face WGSL — the natural next
step once the CPU implementation has stabilised. As of the most
recent commit, this stage is **queued but not yet landed**. The
relevant pieces in the repository are:

- The CPU SWE flux implementations in `dg_element.rs` lines ~1796 →
  ~2160 — three closed-form Riemann solvers that the GPU kernel must
  match bit-identically (up to f32 round-off).
- The non-DG GPU FEM machinery in `crates/molla/shaders/fem_*.wgsl`
  and `crates/molla/src/kernels/` — sparse matvec, CG, elasticity
  assembly. These provide the GPU-data flow scaffolding that the DG
  kernel will plug into.
- The `FemGpuData` type ([gpu_data.rs](./elasticity_gpu.md)) which
  uploads a `FemMesh` + `BsrMatrix` to the device. The DG variant
  would extend this with per-cell DoF offsets and the
  per-face match arrays.

## What v4 needs

Three WGSL kernels (one per system flux):

1. **`fem_dg_swe_hll.wgsl`** — one workgroup per interior face, reads
   the two cells' DoFs, computes \\( (h, hu, hv) \\) on each side of
   the face, evaluates the HLL flux, and **atomically** scatters the
   per-face contribution into the global residual buffer.
2. **`fem_dg_swe_hllc.wgsl`** — same shape, HLLC math.
3. **`fem_dg_swe_roe.wgsl`** — same shape, Roe math (incl. the
   Harten-Hyman entropy fix).

The per-cell mass / stiffness / interior-volume terms can reuse the
existing `fem_elasticity_assemble.wgsl` infrastructure with minor
edits (the kernel is one workgroup per cell with no atomic conflict).
The face kernel is where the GPU-port complexity lives — face
quadrature, two-sided shape evaluation, atomic scatter — and is what
v4 ships.

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

Once the v4 GPU kernel lands, the host-side stepper from
[`dg_time`](./dg_time.md) needs minimal change:

```rust
ssp_rk3_step(&mut u_gpu, t, dt,
    |u, _t| compute_dg_swe_rhs_gpu(&engine, &mesh, u),  // GPU dispatch
    |u| limit_solution_gpu(&mesh, u, degree));          // CPU-fallback initially
```

The limiter (`limit_solution_characteristic`) can stay on the CPU
initially — characteristic projection is harder to GPU-port cleanly
because the Roe eigendecomposition per cell needs branchy logic — and
move to GPU in a v5 follow-up if profiling shows it dominates.

## Why this stage is queued

Practical reasons rather than fundamental ones:

1. **The CPU implementation just stabilised** — Stage 43 v3 added
   HLLC and the entropy-fixed Roe solver, plus the characteristic
   limiter; the test fixtures (`DgShallowWaterDamBreak`) are still
   maturing. Stable CPU math first, GPU port second.
2. **Higher-priority items elsewhere** — the autograd / adjoint work
   (Stage 36.4 derivatives), AMS multigrid (Stage 48 v3/v4), and the
   active-set Signorini follow-up (Stage 49 v2 / Stage 51) all
   compete for the GPU-port budget.
3. **The performance ceiling for the test problems is fine** — the
   `DgShallowWaterDamBreak` scene runs at interactive frame rates on
   CPU at the resolutions a 2D viewer cares about. The GPU port is
   the right answer for production-scale SWE (city-scale flood
   simulation) but molla's test suite doesn't push that scale yet.

## Cross-references

- [DG flux](./dg_flux.md) — the CPU implementations that v4 mirrors.
- [DG basis](./dg_basis.md) — the per-cell DoF layout the GPU kernel
  must respect.
- [Elasticity GPU](./elasticity_gpu.md) — the closest precedent for a
  per-cell FEM assembly kernel.
- [Transient GPU stepper](./solvers.md#gpu-steppers--slice-h) — the
  closest precedent for keeping FEM state resident on the device
  across many timesteps.
