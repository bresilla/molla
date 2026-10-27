# Phase-field fracture

`molla-fem::phase_field` contains the first CPU oracle for brittle
phase-field fracture. The current model is the standard frozen-history
AT2 scalar damage update on the same P1 triangle/tetrahedron meshes used
by the rest of FEM.

The degradation law is:

\\[
g(d) = (1-d)^2 + k,\qquad 0 \le d \le 1
\\]

with history updated irreversibly as:

\\[
H_{n+1} = \max(H_n, \psi^+).
\\]

For a frozen tensile-energy history `H`, the assembled weak form is:

\\[
\int_\Omega G_c \ell \nabla d \cdot \nabla \eta
  + (G_c/\ell + 2H) d\eta\,dx
=
\int_\Omega 2H\eta\,dx.
\\]

## Public API

```rust
pub struct PhaseFieldFractureParams {
    pub critical_energy_release_rate: Real,
    pub length_scale: Real,
    pub residual_stiffness: Real,
}

pub fn phase_field_degradation(damage, residual_stiffness) -> Real;
pub fn update_phase_field_history(previous, tensile_energy_density) -> Real;
pub fn phase_field_fracture_energy_density(d, grad_d_norm_sq, history, params) -> Real;

pub fn assemble_phase_field_fracture_system(mesh, history, params) -> Result<(BsrMatrix, Vec<Real>)>;
pub fn solve_phase_field_fracture_jacobi(mesh, damage_old, history, params, max_iters, tol)
    -> Result<PhaseFieldFractureSolveResult>;
```

`assemble_phase_field_fracture_system` reuses the existing element mass
and stiffness kernels, averaging nodal history per element. The Jacobi
solver is deliberately conservative: every iterate is clamped to
`damage_old <= damage <= 1`, making it a simple CPU correctness oracle
for later Newton/AMG/GPU paths.

