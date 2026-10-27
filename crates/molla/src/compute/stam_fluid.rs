//! Facade compatibility for the 2-D Stam stable-fluid solver.
//!
//! The implementation lives in `molla-fem` because its GPU projection path
//! reuses the FEM FFT-Poisson machinery.

pub use molla_fem::stam_fluid::*;
