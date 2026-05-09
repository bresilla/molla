//! Clamping stages — modify `Control.joint_force[dof]` in-place
//! after the controller has written the raw effort there.
//!
//! Mirrors Newton's `Clamping` family
//! (xtra/newton/_src/actuators/clamping/). Each clamp reads the
//! current effort plus whatever simulation state it needs (joint
//! position for [`ClampingPositionBased`], joint velocity for
//! [`ClampingDcMotor`]) and writes the clipped value back into the
//! same slot.

#[cfg(feature = "gpu")]
pub mod dc_motor;
#[cfg(feature = "gpu")]
pub mod max_effort;
#[cfg(feature = "gpu")]
pub mod position_based;

#[cfg(feature = "gpu")]
pub use dc_motor::{ClampingDcMotor, ClampingDcMotorConfig};
#[cfg(feature = "gpu")]
pub use max_effort::{ClampingMaxEffort, ClampingMaxEffortConfig};
#[cfg(feature = "gpu")]
pub use position_based::ClampingPositionBased;

#[cfg(feature = "gpu")]
use crate::compute::Engine;
#[cfg(feature = "gpu")]
use crate::core::Result;
#[cfg(feature = "gpu")]
use crate::sim::{Control, State};

/// Common interface for an actuator clamp stage.
///
/// Implementations modify `Control.joint_force[dof]` in-place,
/// reading whatever simulation state they need from `State`. The
/// orchestrator calls `step` once per substep, after the controller
/// stage.
#[cfg(feature = "gpu")]
pub trait Clamping {
    fn step(
        &mut self,
        engine: &Engine,
        state: &State,
        control: &mut Control,
    ) -> Result<()>;
}

/// Concrete enum of the built-in clamp variants. Lets the
/// [`crate::actuators::Actuator`] orchestrator hold a non-`Box`
/// clamp without imposing dynamic-dispatch costs at the per-substep
/// hot path. New clamps add a variant here.
#[cfg(feature = "gpu")]
pub enum ClampingStage {
    None,
    MaxEffort(ClampingMaxEffort),
    DcMotor(ClampingDcMotor),
    PositionBased(ClampingPositionBased),
}

#[cfg(feature = "gpu")]
impl ClampingStage {
    pub fn step(
        &mut self,
        engine: &Engine,
        state: &State,
        control: &mut Control,
    ) -> Result<()> {
        match self {
            ClampingStage::None => Ok(()),
            ClampingStage::MaxEffort(c) => c.step(engine, state, control),
            ClampingStage::DcMotor(c) => c.step(engine, state, control),
            ClampingStage::PositionBased(c) => c.step(engine, state, control),
        }
    }
}
