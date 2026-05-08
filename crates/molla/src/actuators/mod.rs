//! Actuator pipeline — joint controllers that read state + targets
//! and write effort to `Control.joint_force`.
//!
//! **GPU-first** by construction: each controller's effort math runs
//! in WGSL (`shaders/actuator_*.wgsl`); these orchestrators own the
//! per-actuator parameter buffers and the dispatch wrapper.
//!
//! Stage 9 deliverables:
//! - [`ControllerPd`] — stateless proportional-derivative controller
//!   with effort cap. Mirrors Newton's `ControllerPD` minus the
//!   feed-forward / const-effort plumbing (added back when an
//!   example needs them).
//! - [`ControllerPid`] — adds an integral term with anti-windup.
//!   Stateful: owns its own `integral` buffer.
//! - **TODO**: clamping (max-effort lives in PD/PID for now;
//!   velocity-saturation + DC-motor curves arrive when an example
//!   needs them), **delay** (ring buffer for sim-to-real transport
//!   lag), composable `Actuator` (delay → controller → clamping)
//!   matching Newton's full pipeline.
//!
//! # Single-DOF assumption
//!
//! These controllers read `state.joint_q[dof]` / `state.joint_qd[dof]`
//! and write `control.joint_force[dof]` directly — i.e. they assume
//! `q` and `qd` use the **same** index. This is true for revolute /
//! prismatic joints; ball / free joints carry quaternion components
//! in `q` and angular vel in `qd`, where indexes diverge. Position-
//! tracking PD on a quaternion isn't well-defined anyway; for those
//! cases use an external orientation-error controller.

#[cfg(feature = "gpu")]
pub mod controller_pd;
#[cfg(feature = "gpu")]
pub mod controller_pid;

#[cfg(feature = "gpu")]
pub use controller_pd::{ControllerPd, ControllerPdConfig};
#[cfg(feature = "gpu")]
pub use controller_pid::{ControllerPid, ControllerPidConfig};
