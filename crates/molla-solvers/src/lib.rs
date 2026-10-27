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
#[cfg(feature = "gpu")]
pub mod generalized_prediction_gpu;
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
pub mod native_prediction_gpu;
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
pub mod soil_admm_geometry_gpu;
#[cfg(feature = "gpu")]
pub mod soil_admm_gpu;
#[cfg(feature = "gpu")]
pub mod soil_candidates_gpu;
#[cfg(feature = "gpu")]
pub mod soil_contact_geometry_gpu;
#[cfg(feature = "gpu")]
pub mod soil_contact_motion_gpu;
#[cfg(feature = "gpu")]
pub mod soil_contact_projection_gpu;
#[cfg(feature = "gpu")]
pub mod soil_floor_projection_gpu;
#[cfg(feature = "gpu")]
pub mod soil_friction_disk_gpu;
#[cfg(feature = "gpu")]
pub mod soil_friction_geometry_gpu;
#[cfg(feature = "gpu")]
pub mod soil_friction_projection_gpu;
#[cfg(all(test, feature = "protected-newton", not(target_arch = "wasm32")))]
mod soil_newton_delta_cache_tests;
#[cfg(feature = "gpu")]
mod soil_newton_failure;
#[cfg(feature = "gpu")]
pub mod soil_newton_geometry_gpu;
#[cfg(feature = "gpu")]
pub mod soil_newton_gpu;
#[cfg(feature = "gpu")]
pub(crate) mod soil_newton_history_gpu;
#[cfg(feature = "gpu")]
pub mod soil_newton_krylov_gpu;
#[cfg(feature = "gpu")]
pub mod soil_newton_operator_gpu;
#[cfg(all(
    any(test, feature = "parallel-newton"),
    feature = "protected-newton",
    not(target_arch = "wasm32")
))]
mod soil_newton_parallel_release;
#[cfg(all(test, feature = "gpu"))]
mod soil_newton_precise_tests;
#[cfg(feature = "gpu")]
pub mod soil_newton_prepare_gpu;
#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
mod soil_newton_protected;
#[cfg(feature = "gpu")]
pub mod soil_newton_publish_gpu;
#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
mod soil_newton_width;
#[cfg(feature = "gpu")]
pub mod soil_normal_projection_gpu;
#[cfg(feature = "gpu")]
mod soil_rate_encoding;
#[cfg(all(test, feature = "gpu"))]
mod soil_setup_trace;
#[cfg(feature = "gpu")]
pub mod soil_shared_wall_friction_gpu;
#[cfg(feature = "gpu")]
pub mod soil_shared_wall_geometry_gpu;
#[cfg(feature = "gpu")]
pub mod soil_tangential_projection_gpu;
#[cfg(feature = "gpu")]
pub mod soil_tractor;
#[cfg(feature = "gpu")]
pub mod soil_wall_friction_gpu;
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
    MAX_MPM_MATERIALS, MPM_COUPLING_BOUNDARY_FAULT, MPM_COUPLING_CAPACITY_FAULT,
    MpmActiveGridStats, MpmCommandBatch, MpmConfig, MpmMassCouplingInputs, MpmParticleRenderSource,
    MpmParticleRenderView, MpmPrescribedSolid, MpmProfileStage, MpmRheology, MpmRigidDeviceBinding,
    MpmSimulation, MpmSoilCanonicalSelection, MpmSoilCanonicalStore, MpmSoilCanonicalStoreRead,
    MpmSoilCanonicalStoreReadCompletion, MpmSoilCanonicalStoreReadTask,
    MpmSoilCanonicalStoreWritebackCompletion, MpmSoilCanonicalStoreWritebackTask,
    MpmSoilCanonicalTileArchive, MpmSoilCheckpoint, MpmSoilCheckpointCompletion,
    MpmSoilCheckpointReadback, MpmSoilCheckpointTicket, MpmSoilEnergy, MpmSoilFrame,
    MpmSoilFrameCheckpoint, MpmSoilFrameSelection, MpmSoilFrameStore, MpmSoilFrameStoreRead,
    MpmSoilFrameStoreReadCompletion, MpmSoilFrameStoreReadTask,
    MpmSoilFrameStoreWritebackCompletion, MpmSoilFrameStoreWritebackTask, MpmSoilFrameTileArchive,
    MpmSoilIdentityBuffers, MpmSoilIdentitySnapshot, MpmSoilMomenta, MpmSoilPinConfig,
    MpmSoilQuiescence, MpmSoilQuiescenceSnapshot, MpmSoilReadbackStats, MpmSoilRegionEnvelope,
    MpmSoilRetainedSupportParticle, MpmSoilRetainedSupportSnapshot, MpmSoilSelection,
    MpmSoilStencilSupport, MpmSoilStore, MpmSoilStoreLimits, MpmSoilStoreStats,
    MpmSoilStoreStorageStats, MpmSoilSupportClosureParticle, MpmSoilSupportClosureSnapshot,
    MpmSoilSupportSnapshot, MpmSoilTileArchive, MpmSoilTransferPath, MpmSoilTransferStats,
    MpmSoilWorld, MpmSoilWorldStatus, MpmSolidShape, MpmStageTiming, MpmStepProfile, MpmTimeStep,
    MpmTireFeedbackParams, MpmTireTerrainRaster, PreparedMpmMassCoupling,
    PreparedMpmRigidDeviceBridge, PreparedMpmStep,
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
