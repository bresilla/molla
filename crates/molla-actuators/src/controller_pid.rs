//! Public PID controller — adds an integral term + anti-windup
//! clamp on top of the PD effort. Stateful: each `step` updates the
//! integral accumulator from the previous frame's value.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::ActuatorPidKernel;
use molla_math::Real;
use molla_sim::{Control, Model, State};
use molla_storage::{Buffer, BufferUsage};

/// Configuration for one PID-controlled DOF.
#[derive(Debug, Clone, Copy)]
pub struct ControllerPidConfig {
    /// Index into `state.joint_q` / `state.joint_qd` /
    /// `control.joint_target_pos` / `control.joint_target_vel`.
    pub dof: u32,
    /// Proportional gain.
    pub kp: Real,
    /// Integral gain.
    pub ki: Real,
    /// Derivative gain.
    pub kd: Real,
    /// Anti-windup clamp on the integral term. `0.0` disables.
    pub integral_max: Real,
    /// Effort cap. `0.0` disables.
    pub max_effort: Real,
}

/// GPU PID controller batch with internal integral state.
pub struct ControllerPid {
    kernel: ActuatorPidKernel,
    actuator_count: u32,
    dof_indices: Buffer<u32>,
    kp: Buffer<Real>,
    ki: Buffer<Real>,
    kd: Buffer<Real>,
    integral_max: Buffer<Real>,
    max_effort: Buffer<Real>,
    integral: Buffer<Real>,
    /// Scratch: kernel writes the new integral here, then we ping-pong.
    next_integral: Buffer<Real>,
}

impl ControllerPid {
    pub fn new(engine: &Engine, model: &Model, configs: &[ControllerPidConfig]) -> Result<Self> {
        if configs.is_empty() {
            return Err(Error::Solver(
                "ControllerPid::new: empty config list".into(),
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("controller_pid: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("controller_pid: no GPU queue".into()))?
            .clone();

        for (i, c) in configs.iter().enumerate() {
            if c.dof as usize >= model.joint_dof_total {
                return Err(Error::Solver(format!(
                    "ControllerPid::new: actuator {i} dof {} out of range (joint_dof_total={})",
                    c.dof, model.joint_dof_total
                )));
            }
        }
        for i in 0..configs.len() {
            for j in (i + 1)..configs.len() {
                if configs[i].dof == configs[j].dof {
                    return Err(Error::Solver(format!(
                        "ControllerPid::new: duplicate dof {} at actuators {i} and {j}",
                        configs[i].dof
                    )));
                }
            }
        }

        let n = configs.len();
        let dof: Vec<u32> = configs.iter().map(|c| c.dof).collect();
        let kp: Vec<Real> = configs.iter().map(|c| c.kp).collect();
        let ki: Vec<Real> = configs.iter().map(|c| c.ki).collect();
        let kd: Vec<Real> = configs.iter().map(|c| c.kd).collect();
        let imax: Vec<Real> = configs.iter().map(|c| c.integral_max).collect();
        let me: Vec<Real> = configs.iter().map(|c| c.max_effort).collect();

        let mut dof_indices = Buffer::from_vec(dof, BufferUsage::Storage);
        let mut kp_b = Buffer::from_vec(kp, BufferUsage::Storage);
        let mut ki_b = Buffer::from_vec(ki, BufferUsage::Storage);
        let mut kd_b = Buffer::from_vec(kd, BufferUsage::Storage);
        let mut integral_max = Buffer::from_vec(imax, BufferUsage::Storage);
        let mut max_effort = Buffer::from_vec(me, BufferUsage::Storage);
        let mut integral = Buffer::from_vec(vec![0.0; n], BufferUsage::StorageReadWrite);
        let mut next_integral = Buffer::from_vec(vec![0.0; n], BufferUsage::StorageReadWrite);

        for buf in [
            dof_indices.device_buffer().is_some(),
            kp_b.device_buffer().is_some(),
        ]
        .iter()
        {
            // No-op; the next loop calls ensure_device on every buffer.
            let _ = buf;
        }
        dof_indices.ensure_device(&device)?;
        kp_b.ensure_device(&device)?;
        ki_b.ensure_device(&device)?;
        kd_b.ensure_device(&device)?;
        integral_max.ensure_device(&device)?;
        max_effort.ensure_device(&device)?;
        integral.ensure_device(&device)?;
        next_integral.ensure_device(&device)?;

        dof_indices.upload(&queue)?;
        kp_b.upload(&queue)?;
        ki_b.upload(&queue)?;
        kd_b.upload(&queue)?;
        integral_max.upload(&queue)?;
        max_effort.upload(&queue)?;
        integral.upload(&queue)?;
        next_integral.upload(&queue)?;

        Ok(Self {
            kernel: ActuatorPidKernel::new(&device),
            actuator_count: n as u32,
            dof_indices,
            kp: kp_b,
            ki: ki_b,
            kd: kd_b,
            integral_max,
            max_effort,
            integral,
            next_integral,
        })
    }

    /// Reset every actuator's integral accumulator to zero.
    pub fn reset(&mut self, engine: &Engine) -> Result<()> {
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("controller_pid::reset: no GPU queue".into()))?
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

    /// Run the kernel and ping-pong the integral buffer.
    pub fn step(
        &mut self,
        engine: &Engine,
        state: &State,
        control: &mut Control,
        dt: Real,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("controller_pid: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("controller_pid: no GPU queue".into()))?
            .clone();

        self.kernel.dispatch(
            &device,
            &queue,
            state,
            control,
            &self.dof_indices,
            &self.kp,
            &self.ki,
            &self.kd,
            &self.integral_max,
            &self.max_effort,
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
