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
//! - [`bc`]          — `BoundaryCondition::{Dirichlet, Neumann}` plus
//!   the `apply_dirichlet` reducer.
//! - [`solver`]      — `solve_steady_state` + `solve_transient`
//!   (backward Euler), built on top of `cg_solve`.
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

pub mod assembly;
pub mod bc;
pub mod element;
pub mod function_space;
pub mod mesh;
pub mod quadrature;
pub mod solver;

pub use assembly::assemble_global;
pub use bc::{apply_dirichlet, BoundaryCondition};
pub use element::{compute_element_mass, compute_element_stiffness};
pub use function_space::FunctionSpace;
pub use mesh::{FemCell, FemMesh};
pub use solver::{solve_steady_state, solve_transient};
