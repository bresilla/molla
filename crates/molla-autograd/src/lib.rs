//! CPU autograd and adjoint utilities for molla.
//!
//! This crate owns the engine-agnostic tape and closed-form CPU adjoints. The
//! first extracted slice intentionally avoids facade-only compute modules so it
//! can depend only on `molla-math`.

pub use molla_math as math;

pub mod diff_arm;
pub mod featherstone_adjoint;
pub mod mpm_adjoint;
pub mod mpm_cpu_diff;
pub mod stam_diff;
pub mod tape;
pub mod xpbd_adjoint;

pub use diff_arm::DiffArm;
pub use featherstone_adjoint::{
    AutogradError, FeatherstoneRevoluteChain, FeatherstoneStepTrace, FeatherstoneTrace,
    JacobianScratch, JacobianStrategy, JointKind, JointProjection, RevoluteLink,
    record_featherstone_step,
};
pub use mpm_adjoint::{
    StateAdjoint, backward, centroid_loss_and_grad, forward_backward_e_centroid_loss,
};
pub use tape::{Tape, TapeCheckpoint, TapeId, TapeOp};
pub use xpbd_adjoint::{
    DistanceConstraintAdjoint, distance_chain_grad_initial_pos, distance_chain_grad_stiffness,
    smooth_plane_contact_step_with_grad, xpbd_distance_step_with_grad,
};
