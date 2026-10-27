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

pub use molla_actuators::{
    DEFAULT_TAU_ACTIVATION, DEFAULT_TENDON_STIFFNESS_RATIO, Muscle, MuscleConfig, force_length,
    force_velocity, tendon_force, update_muscle_activation,
};

#[cfg(feature = "gpu")]
pub mod actuator;
#[cfg(feature = "gpu")]
pub mod clamping;
#[cfg(feature = "gpu")]
pub mod controller_neural_lstm;
#[cfg(feature = "gpu")]
pub mod controller_neural_mlp;
#[cfg(feature = "gpu")]
pub mod controller_pd;
#[cfg(feature = "gpu")]
pub mod controller_pid;
#[cfg(feature = "gpu")]
pub mod controller_pipeline_pd;
#[cfg(feature = "gpu")]
pub mod controller_pipeline_pid;
#[cfg(feature = "gpu")]
pub mod delay;

#[cfg(feature = "gpu")]
pub use molla_actuators::{
    Actuator, Clamping, ClampingDcMotor, ClampingDcMotorConfig, ClampingMaxEffort,
    ClampingMaxEffortConfig, ClampingPositionBased, ClampingStage, ControllerNeuralLstm,
    ControllerNeuralLstmConfig, ControllerNeuralMlp, ControllerNeuralMlpConfig, ControllerPd,
    ControllerPdConfig, ControllerPid, ControllerPidConfig, ControllerPipelinePd,
    ControllerPipelinePdConfig, ControllerPipelinePid, ControllerPipelinePidConfig,
    ControllerStage, Delay, DelayConfig, LstmWeightDims, LstmWeights, MlpWeights,
};
