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

mod articulated_response;
pub mod barrier_contact;
pub mod barrier_solver;
mod body_wrenches;
mod cone_constraints;
mod constraint_allocation;
#[cfg(feature = "gpu")]
pub mod constraint_rows_gpu;
mod contact_friction;
#[cfg(feature = "gpu")]
pub mod contact_islands_gpu;
#[cfg(feature = "gpu")]
pub mod contact_publication_gpu;
#[cfg(feature = "gpu")]
mod contact_reaction_gpu;
#[cfg(feature = "gpu")]
pub mod contact_response_gpu;
#[cfg(feature = "gpu")]
pub mod coulomb_disk_gpu;
#[cfg(feature = "gpu")]
pub mod coupled_constraint_gpu;
pub mod coupled_constraints;
pub mod crba;
pub mod devices;
mod distance_constraints;
#[cfg(feature = "gpu")]
mod distance_gpu;
mod drive_response;
mod featherstone;
pub mod fem_rigid;
#[cfg(feature = "gpu")]
pub mod fem_rigid_gpu;
pub mod ik;
pub mod impact_constraints;
#[cfg(feature = "gpu")]
pub mod impact_constraints_gpu;
#[cfg(feature = "gpu")]
pub mod impact_episodes_gpu;
#[cfg(feature = "gpu")]
pub mod impact_merge_gpu;
pub mod implicit_mpm;
pub mod jacobian;
mod joint_angles;
mod joint_compliance;
mod joint_drives;
mod joint_limits;
mod joint_mass;
mod joint_reactions;
mod kinematic_motion;
mod loop_forces;
#[cfg(feature = "gpu")]
mod loop_forces_gpu;
mod mixed_constraints;
#[cfg(feature = "gpu")]
pub mod mpm;
pub mod mpm_oracle;
#[cfg(feature = "gpu")]
pub mod normal_contact_gpu;
pub mod penalty_contact;
pub mod pressure_tire;
#[cfg(feature = "gpu")]
mod reaction_gpu;
#[cfg(feature = "gpu")]
mod rigid_contact_activation_gpu;
mod rigid_impact;
pub mod rigid_impulse;
pub mod rigid_islands;
mod rigid_loop_constraints;
mod rigid_momentum;
pub mod rigid_world;
pub mod rnea;
pub mod semi_implicit;
pub mod soft_rigid_contact;
#[cfg(feature = "gpu")]
pub mod sph;
pub mod sph_diff;
pub mod sph_oracle;
pub mod track_forces;
mod vbd;
pub mod vbd_soft_contact;
pub mod wheel_forces;
mod xpbd;

pub use barrier_solver::BarrierSolver;
pub use crba::{mass_matrix, response_mass_matrix};
pub use featherstone::{FeatherstoneSolver, JointEffort};
pub use ik::{IkConfig, IkObjective, IkResult, IkSolver, ik_solver_health};
pub use jacobian::jacobian;
pub use loop_forces::LoopReaction;
pub use math::robust_loss::RobustLoss;
#[cfg(feature = "gpu")]
pub use mpm::{
    MAX_MPM_MATERIALS, MpmConfig, MpmRheology, MpmSimulation, MpmTimeStep, MpmTireFeedbackParams,
    MpmTireTerrainRaster,
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
