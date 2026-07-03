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

#[cfg(feature = "gpu")]
pub use molla_compute as compute;
pub use molla_core as core;
pub use molla_geometry as geometry;
#[cfg(feature = "gpu")]
pub use molla_kernels as kernels;
pub use molla_math as math;
pub use molla_storage as storage;

/// Compatibility namespace so the extracted crate can keep historical
/// `crate::fem::...` intra-module paths while the facade preserves
/// `molla::fem::...`.
pub mod fem {
    pub use crate::*;
}

pub mod adjoint;
pub mod amg;
pub mod amr_mesh;
pub mod apic_coupling;
#[cfg(feature = "gpu")]
pub mod apic_coupling_gpu;
pub mod assembly;
pub mod bc;
pub mod dg_element;
pub mod dg_time_integrator;
pub mod eddy_current;
#[cfg(feature = "gpu")]
pub mod elasticity_gpu;
pub mod element;
#[cfg(feature = "gpu")]
pub mod fft_ns;
pub mod function_space;
#[cfg(feature = "gpu")]
pub mod gpu_data;
#[cfg(feature = "gpu")]
pub mod gpu_kernels;
pub mod magneto;
pub mod mesh;
pub mod multigrid;
pub mod multiphase;
pub mod nedelec_element;
pub mod nonconforming;
pub mod p2_element;
pub mod phase_field;
pub mod quadrature;
#[cfg(feature = "russell")]
pub mod russell;
pub mod signorini;
pub mod solver;
pub mod stam_fluid;
pub mod stokes;
#[cfg(feature = "gpu")]
pub mod stokes_gpu;
pub mod surface;

pub use adjoint::{compute_grad_load, compute_grad_youngs, solve_adjoint, solve_adjoint_with};
pub use amg::{AmgHierarchy, AmgLevel};
pub use amr_mesh::{
    AmrMesh, DgAmrFaceMatch, HangingNode, RefinedCell, RefinedChildren, RefinementCriterion,
    assemble_global_amr, compute_ttkt, near_point, project_dg_p1_tri_children_to_parent_l2,
    tt_times_vector,
};
pub use apic_coupling::{
    ApicHybrid, ApicParticle, fluid_divergence_norm, fluid_weak_divergence_norm,
    locate_particle_tet, validate_hybrid,
};
#[cfg(feature = "gpu")]
pub use apic_coupling_gpu::ApicGpuState;
pub use assembly::assemble_global;
pub use bc::{BoundaryCondition, apply_bcs, apply_dg_bcs, apply_dirichlet, default_nitsche_sigma};
pub use dg_element::{
    InteriorFace, NumericalFlux, assemble_dg_global, build_cell_neighbours,
    compute_dg_flux_jump_block, compute_dg_numerical_flux_block,
    compute_dg_numerical_flux_block_vector, compute_element_dg_mass, compute_element_dg_stiffness,
    compute_euler_numerical_flux_hll_2d, compute_euler_numerical_flux_rusanov_2d,
    compute_interior_faces, compute_nitsche_boundary_block, compute_swe_numerical_flux_hll,
    compute_swe_numerical_flux_hllc, compute_swe_numerical_flux_roe, dg_nodal_positions,
    euler_flux_dot_n_2d, euler_pressure_2d, limit_solution, limit_solution_characteristic,
    periodic_face_pairs,
};
pub use dg_time_integrator::{ssp_rk2_step, ssp_rk3_step};
pub use eddy_current::{
    AmsPreconditioner, EddyBlockOp, EddyCurrentConfig, EddyCurrentResult, EddyCurrentSolveMode,
    EddyFrequencySweepSample, apply_block_coupled_ams_preconditioner, assemble_eddy_system,
    build_discrete_gradient, build_p1_vector_interpolation, build_p1_vector_laplacian,
    coulomb_gauge_residual, gmres_with_precond, project_nedelec_coulomb_gauge,
    solve_eddy_current_frequency_sweep, solve_eddy_current_steady, solve_eddy_current_steady_ams,
    solve_eddy_current_steady_ams_amg, solve_eddy_current_steady_ams_block,
    solve_eddy_current_steady_ams_mg,
};
#[cfg(feature = "gpu")]
pub use elasticity_gpu::{
    ElasticityGpuResult, assemble_elasticity_gpu_only, build_elasticity_gpu_pattern,
    solve_elasticity_gpu,
};
pub use element::{
    compute_element_convection, compute_element_distortion_energy,
    compute_element_elasticity_stiffness, compute_element_mass,
    compute_element_mooney_rivlin_energy, compute_element_mooney_rivlin_force,
    compute_element_mooney_rivlin_tangent, compute_element_neo_hookean_energy,
    compute_element_neo_hookean_force, compute_element_neo_hookean_tangent,
    compute_element_st_venant_kirchhoff_stress, compute_element_stiffness,
    compute_element_supg_convection, compute_element_wave_stiffness,
};
#[cfg(feature = "gpu")]
pub use fft_ns::{
    FractionalStepConfig, PoissonStencil, advance_navier_stokes_fractional_step,
    solve_navier_stokes_fractional_step, solve_pressure_poisson_fft,
    solve_pressure_poisson_fft_with_stencil,
};
pub use function_space::{FunctionSpace, compute_edges, dg_ndof_per_cell};
#[cfg(feature = "gpu")]
pub use gpu_data::FemGpuData;
#[cfg(feature = "gpu")]
pub use gpu_kernels::{
    CgGpuResult, DG_SWE_COMPONENTS, DG_SWE_FACE_STRIDE, DG_SWE_NODES_PER_CELL, DgSweFace,
    DgSweFluxKernel, DgSweFluxVariant, DgSweLimiterCell, DgSweLimiterKernel, FemCgGpuKernel,
    FemElasticityGpuKernel, FemGpuKernel, FemNsAdvectKernel, FemNsRhsAssembleKernel,
    FemStokesGpuKernel, NsRhsMode, StokesEntry, TransientStepOpsKernel, WaveStepOpsKernel,
    pack_dg_swe_limiter_cells,
};
pub use magneto::{
    MagnetostaticsResult, compute_curl_per_cell, integrate_field_along_edges,
    project_field_on_edges, solve_magnetostatics_steady,
};
pub use mesh::{FemCell, FemMesh};
pub use multigrid::MeshHierarchy;
pub use multiphase::{
    csf_surface_tension_force_2d, level_set_curvature_2d, level_set_smooth_delta,
};
pub use nedelec_element::{
    NedelecDofMap, assemble_global_nedelec, compute_element_curl_curl, compute_element_nedelec_mass,
};
pub use nonconforming::{
    NonconformingField, NonconformingPenaltyContact, solve_coupled_elasticity,
};
pub use p2_element::{
    P2DofMap, TET_EDGES, TRI_EDGES, assemble_global_p2, compute_element_p2_mass,
    compute_element_p2_stiffness, p2_tet_shape_functions, p2_tet_shape_gradients,
    p2_tri_shape_functions, p2_tri_shape_gradients,
};
pub use phase_field::{
    PhaseFieldFractureParams, PhaseFieldFractureSolveResult, assemble_phase_field_fracture_system,
    phase_field_degradation, phase_field_fracture_energy_density,
    solve_phase_field_fracture_jacobi, update_phase_field_history,
};
#[cfg(feature = "russell")]
pub use russell::{DirectSolver, MAX_DIRECT_DIM};
pub use signorini::{
    ActiveSetFrictionResult, ActiveSetIteration, ActiveSetResult, ActiveSetTrace, ContactPair,
    DEFAULT_ACTIVE_SET_GAP_TOL, DEFAULT_ACTIVE_SET_INNER_MAX, DEFAULT_ACTIVE_SET_INNER_TOL,
    DEFAULT_ACTIVE_SET_MAX_OUTER, DEFAULT_INNER_CG_MAX_ITERS, DEFAULT_INNER_CG_TOL, FrictionState,
    FrictionUzawaResult, RateStateFrictionParams, RateStateFrictionState, SignoriniContact,
    SignoriniContactCandidates, UzawaResult, advance_rate_state_friction,
    assemble_constraint_matrix, assemble_full_constraint_matrix, estimate_dual_step,
    rate_state_friction_coeff, signorini_candidates_from_self_contacts, solve_active_set_signorini,
    solve_active_set_signorini_with_friction, solve_augmented_system, solve_augmented_system_with,
    solve_augmented_system_with_friction, solve_augmented_system_with_friction_full,
    target_gap_vector,
};
#[cfg(feature = "russell")]
pub use solver::solve_steady_state_direct;
pub use solver::{
    KrylovRestartCache, NEWMARK_BETA_DEFAULT, NEWMARK_GAMMA_DEFAULT, WaveStepResult,
    solve_distortion_minimization, solve_nonlinear_static, solve_steady_state,
    solve_steady_state_pcg, solve_steady_state_vector, solve_transient, solve_transient_pcg,
    solve_transient_pcg_restarted, solve_wave_transient,
};
#[cfg(feature = "gpu")]
pub use solver::{TransientStepStats, TransientStepperGpu, WaveStepStats, WaveStepperGpu};
pub use stam_fluid::{StamFluidConfig, StamFluidState};
pub use stokes::{
    MixedTaylorHood, NavierStokesConfig, StokesBc, StokesResult, StokesSimpleConfig,
    apply_stokes_velocity_bcs, assemble_convection_block, assemble_convection_block_with_supg,
    assemble_div_block, assemble_velocity_block, assemble_velocity_mass, build_cavity_bcs_2d,
    collect_velocity_pins_pub, compute_element_mass_p2_vector, compute_element_stokes_div_block,
    compute_element_stokes_velocity_block, solve_navier_stokes_step, solve_navier_stokes_transient,
    solve_stokes_steady, solve_stokes_with_matrices,
};
#[cfg(feature = "gpu")]
pub use stokes_gpu::{
    NavierStokesGpuConfig, NavierStokesGpuResult, NavierStokesGpuTimeIntegrator, StokesGpuConfig,
    StokesGpuResult, assemble_stokes_gpu_only, solve_navier_stokes_gpu, solve_stokes_gpu,
};
pub use surface::boundary_facets;
