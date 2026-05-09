//! Composable actuator pipeline `Delay → Controller → Clamping`,
//! mirroring Newton's `Actuator` orchestrator
//! (xtra/newton/_src/actuators/actuator.py).
//!
//! Each per-step `step()` call runs the three stages in order:
//! 1. **Delay** pushes the current command targets into a circular
//!    buffer and reads the per-DOF lagged sample. When constructed
//!    with `delay_steps == 0` per actuator, the kernel passes the
//!    current target through unchanged.
//! 2. **Controller** consumes the (possibly lagged) targets and
//!    writes raw effort into `Control.joint_force[dof]`.
//! 3. **Clamping** modifies `joint_force[dof]` in place to enforce
//!    actuator-physics limits (effort cap, DC-motor envelope,
//!    position-based gating). [`ClampingStage::None`] is a no-op.
//!
//! With `(delay_steps=0, ctrl=PD, clamp=MaxEffort)` this matches the
//! behaviour of the standalone [`super::ControllerPd`] one-for-one.

use crate::actuators::clamping::ClampingStage;
use crate::actuators::controller_pipeline_pd::ControllerPipelinePd;
use crate::actuators::controller_pipeline_pid::ControllerPipelinePid;
use crate::actuators::delay::Delay;
use crate::compute::Engine;
use crate::core::Result;
use crate::math::Real;
use crate::sim::{Control, State};

/// Concrete enum for the controller stage of the pipeline. Other
/// controller flavours (neural MLP, neural LSTM) plug in here.
pub enum ControllerStage {
    Pd(ControllerPipelinePd),
    Pid(ControllerPipelinePid),
    /// Stage 18.1 fixed-architecture 3-layer MLP forward pass.
    /// The controller writes its output directly into
    /// `Control.joint_force[dof]`, one MLP output per registered DOF.
    /// Inputs are gathered host-side per substep via
    /// [`crate::actuators::ControllerNeuralMlp::set_inputs`].
    NeuralMlp(crate::actuators::ControllerNeuralMlp),
    /// Stage 31 single-cell LSTM forward pass + linear projection.
    /// Same input / output contract as `NeuralMlp`, but the cell
    /// carries hidden + cell state across substeps so the controller
    /// can express stateful (recurrent) policies. Reset both states
    /// via [`crate::actuators::ControllerNeuralLstm::reset_hidden_state`]
    /// on episode boundaries.
    Lstm(crate::actuators::ControllerNeuralLstm),
}

/// Composable actuator pipeline.
///
/// Owns its own [`Delay`], a [`ControllerStage`], and a
/// [`ClampingStage`]. All three stages are addressed against the
/// same per-actuator DOF list (the orchestrator does not enforce
/// matching arity at the API level — caller is responsible for
/// providing equal-length configs across stages).
pub struct Actuator {
    delay: Delay,
    controller: ControllerStage,
    clamping: ClampingStage,
}

impl Actuator {
    pub fn new(delay: Delay, controller: ControllerStage, clamping: ClampingStage) -> Self {
        Self {
            delay,
            controller,
            clamping,
        }
    }

    /// Mutable access to the embedded delay stage.
    pub fn delay_mut(&mut self) -> &mut Delay {
        &mut self.delay
    }

    /// Mutable access to the embedded clamping stage.
    pub fn clamping_mut(&mut self) -> &mut ClampingStage {
        &mut self.clamping
    }

    /// Mutable access to the embedded controller stage.
    pub fn controller_mut(&mut self) -> &mut ControllerStage {
        &mut self.controller
    }

    /// Number of actuators in this pipeline (= delay's actuator
    /// count; the controller is required to use the same count).
    pub fn actuator_count(&self) -> u32 {
        self.delay.actuator_count()
    }

    /// Run one substep of the pipeline.
    ///
    /// `dt` is the substep duration; it's consumed by the integral
    /// term of [`ControllerStage::Pid`] and ignored by the PD / MLP
    /// variants. Caller-supplied so the pipeline doesn't have to
    /// pin its own opinion about the host's timestep.
    pub fn step(
        &mut self,
        engine: &Engine,
        state: &State,
        control: &mut Control,
        dt: Real,
    ) -> Result<()> {
        // Stage 1 — push & read delayed targets.
        self.delay.step(engine, control)?;

        // Stage 2 — controller consumes the delayed buffers.
        match &mut self.controller {
            ControllerStage::Pd(pd) => {
                pd.step(
                    engine,
                    state,
                    control,
                    self.delay.out_pos(),
                    self.delay.out_vel(),
                )?;
            }
            ControllerStage::Pid(pid) => {
                pid.step(
                    engine,
                    state,
                    control,
                    self.delay.out_pos(),
                    self.delay.out_vel(),
                    dt,
                )?;
            }
            ControllerStage::NeuralMlp(mlp) => {
                // The MLP doesn't read the delay outputs (its inputs
                // are pushed by the host via `set_inputs`); but for
                // future-proofing we still tick the delay so a
                // downstream stage can rely on `out_pos / out_vel`
                // being fresh.
                mlp.step(engine, control)?;
            }
            ControllerStage::Lstm(lstm) => {
                // Same convention as NeuralMlp — observations are
                // pushed by the host, the kernel handles the gate
                // math + scatter. Hidden + cell state survive
                // across substeps inside the controller.
                lstm.step(engine, control)?;
            }
        }

        // Stage 3 — clamping in-place.
        self.clamping.step(engine, state, control)?;
        Ok(())
    }
}
