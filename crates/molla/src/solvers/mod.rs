//! Time-integration solvers.
//!
//! Every solver implements the [`Solver`] trait so callers can swap
//! integrators without rewiring the host loop. Available backends:
//!
//! - [`SemiImplicitSolver`] — explicit symplectic Euler for free
//!   rigid bodies (CPU + GPU).
//! - [`XpbdSolver`] — extended-PBD rigid + cloth + particles
//!   (CPU + GPU; substepping + iterations are configurable).
//! - [`FeatherstoneSolver`] — articulated-body algorithm for
//!   reduced-coordinate articulations (CPU + GPU; multi-DOF joints
//!   via Stage 17.1).
//! - [`VbdSolver`] — vertex block descent for soft bodies +
//!   tetrahedral solids (CPU + GPU; AVBD hard mode).
//!
//! Standalone primitives (not full integrators) include the
//! [`crba::mass_matrix`] CRBA, the recursive Newton-Euler
//! [`rnea::inverse_dynamics`], the geometric [`jacobian::jacobian`],
//! and the differentiable [`ik::IkSolver`].

pub mod crba;
mod featherstone;
pub mod ik;
pub mod jacobian;
pub mod rnea;
mod semi_implicit;
mod vbd;
pub mod vbd_soft_contact;
mod xpbd;

pub use crate::math::robust_loss::RobustLoss;
pub use crba::mass_matrix;
pub use featherstone::FeatherstoneSolver;
pub use ik::{IkConfig, IkObjective, IkResult, IkSolver, ik_solver_health};
pub use jacobian::jacobian;
pub use molla_solvers::Solver;
pub use rnea::inverse_dynamics;
pub use semi_implicit::SemiImplicitSolver;
pub use vbd::{VbdMode, VbdSolver};
pub use vbd_soft_contact::{
    EeContact, SoftContact, VtContact, gather_ee_contacts, gather_soft_contacts,
    gather_vt_contacts, project_soft_contacts,
};
pub use xpbd::XpbdSolver;
