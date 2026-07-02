//! Solver and dynamics utilities for molla.
//!
//! This crate owns solver functionality that is independent of the public
//! facade. The first extracted slice contains rigid-body dynamics helpers that
//! depend only on the core/math/sim data crates.

pub use molla_actuators as actuators;
#[cfg(feature = "autograd")]
pub use molla_autograd as autograd;
#[cfg(feature = "gpu")]
pub use molla_compute as compute;
pub use molla_core as core;
pub use molla_geometry as geometry;
#[cfg(feature = "gpu")]
pub use molla_kernels as kernels;
pub use molla_math as math;
pub use molla_sim as sim;
pub use molla_storage as storage;

pub mod crba;
mod featherstone;
pub mod ik;
pub mod implicit_mpm;
pub mod jacobian;
#[cfg(feature = "gpu")]
pub mod mpm;
pub mod mpm_oracle;
pub mod rnea;
pub mod semi_implicit;
#[cfg(feature = "gpu")]
pub mod sph;
pub mod sph_diff;
pub mod sph_oracle;
mod vbd;
pub mod vbd_soft_contact;
mod xpbd;

pub use crba::mass_matrix;
pub use featherstone::FeatherstoneSolver;
pub use ik::{IkConfig, IkObjective, IkResult, IkSolver, ik_solver_health};
pub use jacobian::jacobian;
pub use math::robust_loss::RobustLoss;
#[cfg(feature = "gpu")]
pub use mpm::{
    MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep, MpmTireFeedbackParams, MpmTireTerrainRaster,
};
pub use mpm_oracle::{
    MpmRigidBodySample, MpmRigidCellSample, ReducedMpmMaterial, ReducedMpmMaterialState,
    ReducedMpmMaterialStep, RigidCellWrench, SandProjectStep, SnowClampStep, ViscousDampingStep,
    aggregate_rigid_cell_wrench, drucker_prager_alpha, reduced_material_update,
    rigid_body_cell_sample, rigid_body_velocity_at_cell, rigid_cell_imprint_wrench,
    sand_project_singular_values, snow_clamp_singular_values, viscous_damping_step,
};
pub use rnea::inverse_dynamics;
pub use semi_implicit::SemiImplicitSolver;
#[cfg(feature = "gpu")]
pub use sph::{SphConfig, SphRigidBody, SphSimulation, box_tank_walls};
pub use sph_oracle::{SphFsiOracle, SphOracle, SphOracleBody};
pub use vbd::{VbdMode, VbdSolver};
pub use vbd_soft_contact::{
    EeContact, SoftContact, VtContact, gather_ee_contacts, gather_soft_contacts,
    gather_vt_contacts, project_soft_contacts,
};
pub use xpbd::XpbdSolver;

use molla_core::Result;
use molla_math::Real;
use molla_sim::{Contacts, Control, Model, State};

/// Common interface for every time-integration scheme.
///
/// A solver advances a [`State`] by one timestep `dt`, possibly using the
/// current [`Control`] inputs and [`Contacts`] from the collision pipeline.
pub trait Solver: Send + Sync {
    /// Advance `state_in` by `dt` and write the result into `state_out`.
    ///
    /// `state_in` and `state_out` must be distinct buffers — solvers may
    /// hold borrows of both for the duration of the call.
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()>;
}
