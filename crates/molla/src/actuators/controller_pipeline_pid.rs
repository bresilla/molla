//! Pipeline-PID controller: drives `Control.joint_force[dof]` from
//! per-actuator `(delayed_target_pos, delayed_target_vel)` produced
//! by an upstream [`crate::actuators::Delay`] stage.
//!
//! Stateful: each `step` updates an integral accumulator from the
//! previous frame's value via a host-side ping-pong, so the integral
//! survives across substep state-buffer swaps. Effort cap is
//! intentionally **not** applied here — clamping is its own stage in
//! the Newton-style `Delay → Controller → Clamping` pipeline.

use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::kernels::ActuatorPipelinePidKernel;
use crate::math::Real;
use crate::sim::{Control, State};
use crate::storage::{Buffer, BufferUsage};

/// Per-DOF PID configuration. Mirrors [`crate::actuators::ControllerPidConfig`]
/// minus the `max_effort` field (clamping lives downstream in the
/// pipeline).
#[derive(Debug, Clone, Copy)]
pub struct ControllerPipelinePidConfig {
    /// Index into `state.joint_q / state.joint_qd / control.joint_force`.
    pub dof: u32,
    /// Proportional gain.
    pub kp: Real,
    /// Integral gain.
    pub ki: Real,
    /// Derivative gain.
    pub kd: Real,
    /// Anti-windup clamp on the integral term. `0.0` disables.
    pub integral_max: Real,
}

pub struct ControllerPipelinePid {
    kernel: ActuatorPipelinePidKernel,
    actuator_count: u32,
    dof_indices: Buffer<u32>,
    kp: Buffer<Real>,
    ki: Buffer<Real>,
    kd: Buffer<Real>,
    integral_max: Buffer<Real>,
    integral: Buffer<Real>,
    /// Scratch: kernel writes the new integral here, then the host
    /// ping-pongs `integral ↔ next_integral`. The integral buffer
    /// survives substep state-buffer swaps since the controller owns
    /// it (mirrors how Stage 17 propagates `state.body_f`).
    next_integral: Buffer<Real>,
}

impl ControllerPipelinePid {
    pub fn new(
        engine: &Engine,
        configs: &[ControllerPipelinePidConfig],
    ) -> Result<Self> {
        if configs.is_empty() {
            return Err(Error::Solver(
                "ControllerPipelinePid::new: empty config list".into(),
            ));
        }
        // Duplicate-DOF check: each actuator writes joint_force[dof],
        // and concurrent writes to the same slot would race.
        for i in 0..configs.len() {
            for j in (i + 1)..configs.len() {
                if configs[i].dof == configs[j].dof {
                    return Err(Error::Solver(format!(
                        "ControllerPipelinePid::new: duplicate dof {} at actuators {i} and {j}",
                        configs[i].dof
                    )));
                }
            }
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePid: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePid: no GPU queue".into()))?
            .clone();

        let n = configs.len();
        let dof: Vec<u32> = configs.iter().map(|c| c.dof).collect();
        let kp_v: Vec<Real> = configs.iter().map(|c| c.kp).collect();
        let ki_v: Vec<Real> = configs.iter().map(|c| c.ki).collect();
        let kd_v: Vec<Real> = configs.iter().map(|c| c.kd).collect();
        let imax_v: Vec<Real> = configs.iter().map(|c| c.integral_max).collect();

        let mut dof_indices = Buffer::from_vec(dof, BufferUsage::Storage);
        let mut kp = Buffer::from_vec(kp_v, BufferUsage::Storage);
        let mut ki = Buffer::from_vec(ki_v, BufferUsage::Storage);
        let mut kd = Buffer::from_vec(kd_v, BufferUsage::Storage);
        let mut integral_max = Buffer::from_vec(imax_v, BufferUsage::Storage);
        let mut integral =
            Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);
        let mut next_integral =
            Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);

        dof_indices.ensure_device(&device)?;
        kp.ensure_device(&device)?;
        ki.ensure_device(&device)?;
        kd.ensure_device(&device)?;
        integral_max.ensure_device(&device)?;
        integral.ensure_device(&device)?;
        next_integral.ensure_device(&device)?;

        dof_indices.upload(&queue)?;
        kp.upload(&queue)?;
        ki.upload(&queue)?;
        kd.upload(&queue)?;
        integral_max.upload(&queue)?;
        integral.upload(&queue)?;
        next_integral.upload(&queue)?;

        Ok(Self {
            kernel: ActuatorPipelinePidKernel::new(&device),
            actuator_count: n as u32,
            dof_indices,
            kp,
            ki,
            kd,
            integral_max,
            integral,
            next_integral,
        })
    }

    pub fn actuator_count(&self) -> u32 {
        self.actuator_count
    }

    /// Update the gain table. Lengths must equal `actuator_count`.
    pub fn set_gains(
        &mut self,
        engine: &Engine,
        kp: &[Real],
        ki: &[Real],
        kd: &[Real],
    ) -> Result<()> {
        let n = self.actuator_count as usize;
        if kp.len() != n || ki.len() != n || kd.len() != n {
            return Err(Error::Solver(format!(
                "ControllerPipelinePid::set_gains: expected {n} entries each"
            )));
        }
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("set_gains: no GPU queue".into()))?
            .clone();
        self.kp.host_mut()?.copy_from_slice(kp);
        self.ki.host_mut()?.copy_from_slice(ki);
        self.kd.host_mut()?.copy_from_slice(kd);
        self.kp.upload(&queue)?;
        self.ki.upload(&queue)?;
        self.kd.upload(&queue)?;
        Ok(())
    }

    /// Update the per-actuator anti-windup limit.
    pub fn set_integral_max(&mut self, engine: &Engine, values: &[Real]) -> Result<()> {
        let n = self.actuator_count as usize;
        if values.len() != n {
            return Err(Error::Solver(format!(
                "ControllerPipelinePid::set_integral_max: expected {n} entries"
            )));
        }
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("set_integral_max: no GPU queue".into()))?
            .clone();
        self.integral_max.host_mut()?.copy_from_slice(values);
        self.integral_max.upload(&queue)?;
        Ok(())
    }

    /// Reset every actuator's integral accumulator to zero.
    pub fn reset(&mut self, engine: &Engine) -> Result<()> {
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePid::reset: no GPU queue".into()))?
            .clone();
        for v in self.integral.host_mut()? {
            *v = 0.0;
        }
        for v in self.next_integral.host_mut()? {
            *v = 0.0;
        }
        self.integral.upload(&queue)?;
        self.next_integral.upload(&queue)?;
        Ok(())
    }

    pub fn step(
        &mut self,
        engine: &Engine,
        state: &State,
        control: &mut Control,
        delayed_target_pos: &Buffer<Real>,
        delayed_target_vel: &Buffer<Real>,
        dt: Real,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePid::step: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePid::step: no GPU queue".into()))?
            .clone();
        self.kernel.dispatch(
            &device,
            &queue,
            state,
            control,
            delayed_target_pos,
            delayed_target_vel,
            &self.dof_indices,
            &self.kp,
            &self.ki,
            &self.kd,
            &self.integral_max,
            &self.integral,
            &mut self.next_integral,
            self.actuator_count,
            dt as f32,
        )?;
        // Ping-pong: integral ← next_integral.
        std::mem::swap(&mut self.integral, &mut self.next_integral);
        Ok(())
    }
}
