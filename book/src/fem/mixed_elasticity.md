# Mixed (u, p) Taylor-Hood elasticity

Nearly-incompressible linear elasticity locks volumetrically when the
`λ div·div` term sits inside the displacement operator: as ν → ½, λ → ∞ and
P1 displacement elements can barely move. The mixed formulation moves that
term into a pressure field on Taylor-Hood elements (P2 displacement / P1
pressure):

```text
∫ 2μ ε(u):ε(v) + ∫ p div v = ∫ f·v      (momentum)
∫ q div u − (1/λ) ∫ p q    = 0          (p = λ div u)
```

λ never enters the displacement operator, so the formulation cannot lock;
λ → ∞ recovers exact incompressibility (Stokes).

## API

- CPU: `molla_fem::solve_mixed_elasticity_steady(mesh, th, mu, lambda, f, bcs, cfg)`
  — f64, Uzawa-CG on the pressure Schur complement `B A⁻¹ Bᵀ + (1/λ)·M_p`.
- GPU: `molla_fem::solve_mixed_elasticity_gpu(...)` — same saddle machinery as
  `solve_stokes_gpu` (device-resident inner+outer CG, all matvecs on device)
  with the compliance matvec added inside the Schur closure. The elastic
  `ε:ε` block is host-assembled (the GPU assembly shader is the *viscous*
  full-gradient operator, which is wrong for elasticity); the solve is where
  the GPU time goes. v1 shares the Stokes-GPU 2D-tri restriction.
- Building blocks: `compute_element_mixed_elastic_block` (the coupled
  `μ(δ_ab ∇φᵢ·∇φⱼ + ∂_b φᵢ ∂_a φⱼ)` element), `assemble_mixed_elastic_block`,
  `assemble_pressure_mass` (consistent P1 mass).

## Two solver rules the compliance term changes

Both were caught by the patch test and apply to any compliance-carrying
saddle you add later:

1. **No dead-pressure-row pinning.** With `+ (1/λ)M_p` every Schur row is
   definite; the Stokes-era LBB guard would wrongly force boundary pressures
   to zero.
2. **No default pressure pin.** Stokes silently pins vertex 0 when the
   caller supplies no `PressureZero` BC (needed for its constant nullspace).
   Under compliance that pin corrupts the solution — the exact pressure is a
   nonzero constant in the patch test, and the corruption rippled 8e-5 into
   every displacement before the gate was added.

## Validation

- **Constant-strain patch test** (`fem_mixed_elasticity.rs`): the linear
  field `u = (ε·x, −ν·ε·y, −ν·ε·z)` imposed on every boundary P2 node is
  reproduced in the interior to 2.8e-14 (CPU, f64) / 6.1e-9 (GPU, f32), with
  `p = λ·div u` constant to machine-level.
- **Anti-locking cantilever**: tip deflection at ν = 0.4999 stays within 20%
  of the ν = 0.3 value (a locking element collapses it 10–100×); GPU matches
  the f64 CPU tip to 0.11%.
