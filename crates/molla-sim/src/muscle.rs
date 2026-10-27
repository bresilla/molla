//! Muscle configuration stored on the simulation model.

use molla_math::Real;

/// Default activation time constant `tau_act` (s).
pub const DEFAULT_TAU_ACTIVATION: Real = 0.020;

/// Tendon stiffness scaling, `E * A / f_max`.
pub const DEFAULT_TENDON_STIFFNESS_RATIO: Real = 30.0;

/// Configuration for a single Hill-type muscle attached to one cable joint.
///
/// This lives in `molla-sim` because the builder copies these parameters into
/// `Model::muscle_*` buffers. The actuator implementation consumes the model
/// buffers and re-exports this type for the old `molla::actuators` path.
#[derive(Debug, Clone, Copy)]
pub struct MuscleConfig {
    pub f_max: Real,
    pub l_opt: Real,
    pub v_max: Real,
    pub activation: Real,
    pub excitation: Real,
    pub tau_activation: Real,
    pub l_tendon_slack: Real,
    pub tendon_stiffness_ratio: Real,
    pub pennation_angle: Real,
    pub polarity: Real,
}

impl Default for MuscleConfig {
    fn default() -> Self {
        Self {
            f_max: 100.0,
            l_opt: 0.05,
            v_max: 0.5,
            activation: 0.0,
            excitation: 0.0,
            tau_activation: DEFAULT_TAU_ACTIVATION,
            l_tendon_slack: 0.0,
            tendon_stiffness_ratio: DEFAULT_TENDON_STIFFNESS_RATIO,
            pennation_angle: 0.0,
            polarity: 1.0,
        }
    }
}
