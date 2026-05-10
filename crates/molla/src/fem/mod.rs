//! Finite-element method (FEM) framework — v1, P1 nodal, CPU.
//!
//! Stage 36 v1 ships the foundational machinery for solving scalar
//! continuum-PDE problems on unstructured triangle / tetrahedron
//! meshes. The architecture is deliberately layered so follow-up
//! stages (36.1 elasticity, 36.2 Stokes, 36.3 higher-order, 36.4
//! adjoint, 36.5 GPU port) can extend it without rewriting the v1
//! surface.
//!
//! # Modules
//!
//! - [`mesh`]        — `FemMesh` + `FemCell` (Tri / Tet) + grid builders.
//! - [`function_space`] — `FunctionSpace` enum (P1Nodal in v1).
//! - [`quadrature`]  — Gauss rules: 1- / 3-point on tri, 1- / 4-point
//!   on tet. Weights are pre-normalised to sum to 1; assembly multiplies
//!   by the cell measure (area / volume).
//! - [`element`]     — local element matrices: `compute_element_mass`
//!   and `compute_element_stiffness` (Laplacian operator).
//! - [`assembly`]    — `assemble_global` using a visitor over the
//!   per-cell local matrices, scattering into a CSR (`BsrMatrix` with
//!   `B = 1`).
//! - [`bc`]          — `BoundaryCondition::{Dirichlet, Neumann,
//!   DirichletVector, NeumannTraction}` plus the `apply_dirichlet`
//!   (scalar) and `apply_bcs` (vector-aware) reducers.
//! - [`solver`]      — `solve_steady_state` + `solve_transient`
//!   (backward Euler), built on top of `cg_solve`.
//! - [`stokes`]      — Stage 36.2 mixed Taylor-Hood (P2 velocity, P1
//!   pressure) Stokes / Navier-Stokes solver via Uzawa-CG on the
//!   pressure-Schur complement.
//!
//! # End-to-end demo
//!
//! `crates/molla/tests/fem_v1.rs` ships the contract tests: a unit-
//! triangle mass-matrix check, a tet stiffness "linear-field-in-kernel"
//! check, a 2D Laplacian Dirichlet solve, a 3D heat-equation steady
//! state, and a backward-Euler convergence check.
//!
//! `Scene::FemHeat` (#47) in `molla_viewer` wires the same heat-
//! equation solve to a 16×16×16 tet cube with live colour rendering.
//!
//! # Caveats (v1)
//!
//! - CPU only. Stage 36.5 ports `assemble_global` + sparse-matvec to
//!   WGSL; the v1 closure-driven solver API survives unchanged.
//! - P1 nodal only. Stage 36.3 adds P2 / P3 by extending
//!   [`FunctionSpace`] + adding richer quadrature rules.
//! - Scalar fields only. Stage 36.1 (elasticity) adds vector-valued
//!   DoFs by introducing a per-vertex DoF count in the assembly path.
//! - No mesh adaptivity (uniform refinement only).

pub mod adjoint;
pub mod apic_coupling;
pub mod assembly;
pub mod bc;
pub mod element;
#[cfg(feature = "gpu")]
pub mod fft_ns;
pub mod function_space;
#[cfg(feature = "gpu")]
pub mod gpu_data;
pub mod mesh;
pub mod p2_element;
pub mod quadrature;
pub mod solver;
pub mod stokes;

pub use adjoint::{compute_grad_load, compute_grad_youngs, solve_adjoint, solve_adjoint_with};
pub use apic_coupling::{locate_particle_tet, validate_hybrid, ApicHybrid, ApicParticle};
pub use assembly::assemble_global;
pub use bc::{apply_bcs, apply_dirichlet, BoundaryCondition};
#[cfg(feature = "gpu")]
pub use fft_ns::{
    advance_navier_stokes_fractional_step, solve_navier_stokes_fractional_step,
    solve_pressure_poisson_fft, FractionalStepConfig,
};
pub use element::{
    compute_element_elasticity_stiffness, compute_element_mass,
    compute_element_st_venant_kirchhoff_stress, compute_element_stiffness,
    compute_element_wave_stiffness,
};
pub use function_space::{compute_edges, FunctionSpace};
#[cfg(feature = "gpu")]
pub use gpu_data::FemGpuData;
pub use mesh::{FemCell, FemMesh};
pub use p2_element::{
    assemble_global_p2, compute_element_p2_mass, compute_element_p2_stiffness,
    p2_tet_shape_functions, p2_tet_shape_gradients, p2_tri_shape_functions,
    p2_tri_shape_gradients, P2DofMap, TET_EDGES, TRI_EDGES,
};
pub use solver::{
    solve_nonlinear_static, solve_steady_state, solve_steady_state_vector,
    solve_transient, solve_wave_transient, WaveStepResult,
    NEWMARK_BETA_DEFAULT, NEWMARK_GAMMA_DEFAULT,
};
pub use stokes::{
    apply_stokes_velocity_bcs, assemble_convection_block, assemble_div_block,
    assemble_velocity_block, assemble_velocity_mass, build_cavity_bcs_2d,
    compute_element_mass_p2_vector, compute_element_stokes_div_block,
    compute_element_stokes_velocity_block, solve_navier_stokes_step,
    solve_navier_stokes_transient, solve_stokes_steady,
    solve_stokes_with_matrices, MixedTaylorHood, NavierStokesConfig, StokesBc,
    StokesResult, StokesSimpleConfig,
};
