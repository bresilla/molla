//! Actuator pipeline — joint controllers that read state + targets
//! and write effort to `Control.joint_force`.
//!
//! **GPU-first** by construction: each controller's effort math runs
//! in WGSL (`shaders/actuator_*.wgsl`); these orchestrators own the
//! per-actuator parameter buffers and the dispatch wrapper.
//!
//! ## Standalone controllers
//!
//! - [`ControllerPd`] — stateless proportional-derivative controller
//!   with effort cap baked in. Mirrors Newton's `ControllerPD` minus
//!   the feed-forward / const-effort plumbing.
//! - [`ControllerPid`] — adds an integral term with anti-windup.
//!
//! ## Composable pipeline
//!
//! - [`Delay`] — per-DOF circular-buffer command-input lag.
//! - [`ClampingMaxEffort`], [`ClampingDcMotor`],
//!   [`ClampingPositionBased`] — actuator-physics clamps.
//! - [`Actuator`] — orchestrator that wires
//!   `Delay → Controller → Clamping` per substep.
//!
//! ## Neural controllers
//!
//! - [`ControllerNeuralMlp`] — feed-forward MLP plugged in via
//!   `ControllerStage::Mlp`.
//! - [`ControllerNeuralLstm`] — single-cell LSTM forward pass + linear
//!   projection. Hidden + cell state survive substepping. Plugs into
//!   the orchestrator via `ControllerStage::Lstm`.
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

// Stage 42 — Hill-type muscle actuator. Always compiled (no GPU
// kernel yet; consumed host-side by the CPU Featherstone solver and
// stored as parallel buffers on `Model` so the GPU path can pick it
// up in a future stage).
pub mod muscle;

pub use muscle::{
    force_length, force_velocity, tendon_force, update_muscle_activation, Muscle,
    MuscleConfig, DEFAULT_TAU_ACTIVATION, DEFAULT_TENDON_STIFFNESS_RATIO,
};

#[cfg(feature = "gpu")]
pub mod actuator;
#[cfg(feature = "gpu")]
pub mod clamping;
#[cfg(feature = "gpu")]
pub mod controller_pd;
#[cfg(feature = "gpu")]
pub mod controller_pid;
#[cfg(feature = "gpu")]
pub mod controller_neural_lstm;
#[cfg(feature = "gpu")]
pub mod controller_neural_mlp;
#[cfg(feature = "gpu")]
pub mod controller_pipeline_pd;
#[cfg(feature = "gpu")]
pub mod controller_pipeline_pid;
#[cfg(feature = "gpu")]
pub mod delay;

#[cfg(feature = "gpu")]
pub use actuator::{Actuator, ControllerStage};
#[cfg(feature = "gpu")]
pub use clamping::{
    Clamping, ClampingDcMotor, ClampingDcMotorConfig, ClampingMaxEffort,
    ClampingMaxEffortConfig, ClampingPositionBased, ClampingStage,
};
#[cfg(feature = "gpu")]
pub use controller_pd::{ControllerPd, ControllerPdConfig};
#[cfg(feature = "gpu")]
pub use controller_pid::{ControllerPid, ControllerPidConfig};
#[cfg(feature = "gpu")]
pub use controller_neural_lstm::{
    ControllerNeuralLstm, ControllerNeuralLstmConfig, LstmWeightDims, LstmWeights,
};
#[cfg(feature = "gpu")]
pub use controller_neural_mlp::{
    ControllerNeuralMlp, ControllerNeuralMlpConfig, MlpWeights,
};
#[cfg(feature = "gpu")]
pub use controller_pipeline_pd::{ControllerPipelinePd, ControllerPipelinePdConfig};
#[cfg(feature = "gpu")]
pub use controller_pipeline_pid::{ControllerPipelinePid, ControllerPipelinePidConfig};
#[cfg(feature = "gpu")]
pub use delay::{Delay, DelayConfig};
