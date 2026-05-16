//! Actuator models and utilities.

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
pub mod muscle;

#[cfg(feature = "gpu")]
pub use actuator::{Actuator, ControllerStage};
#[cfg(feature = "gpu")]
pub use clamping::{
    Clamping, ClampingDcMotor, ClampingDcMotorConfig, ClampingMaxEffort, ClampingMaxEffortConfig,
    ClampingPositionBased, ClampingStage,
};
#[cfg(feature = "gpu")]
pub use controller_neural_lstm::{
    ControllerNeuralLstm, ControllerNeuralLstmConfig, LstmWeightDims, LstmWeights,
};
#[cfg(feature = "gpu")]
pub use controller_neural_mlp::{ControllerNeuralMlp, ControllerNeuralMlpConfig, MlpWeights};
#[cfg(feature = "gpu")]
pub use controller_pd::{ControllerPd, ControllerPdConfig};
#[cfg(feature = "gpu")]
pub use controller_pid::{ControllerPid, ControllerPidConfig};
#[cfg(feature = "gpu")]
pub use controller_pipeline_pd::{ControllerPipelinePd, ControllerPipelinePdConfig};
#[cfg(feature = "gpu")]
pub use controller_pipeline_pid::{ControllerPipelinePid, ControllerPipelinePidConfig};
#[cfg(feature = "gpu")]
pub use delay::{Delay, DelayConfig};
pub use muscle::{
    DEFAULT_TAU_ACTIVATION, DEFAULT_TENDON_STIFFNESS_RATIO, Muscle, MuscleConfig, force_length,
    force_velocity, tendon_force, update_muscle_activation,
};
