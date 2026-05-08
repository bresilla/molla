//! Public PD controller — orchestrates an [`ActuatorPdKernel`] dispatch.
//!
//! Effort law:
//!
//! ```text
//! effort = kp · (target_pos - q) + kd · (target_vel - qd)
//! effort = clamp(effort, -max_effort, +max_effort)   if max_effort > 0
//! ```
//!
//! Each controller owns N actuators (one per joint DOF being driven).
//! The kernel writes the result **directly** to `control.joint_force[dof]`
//! — caller is responsible for zeroing `joint_force` before the
//! step if multiple controllers share the buffer (each actuator
//! still has to target a unique DOF; multiple actuators on the same
//! DOF would race).

use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::kernels::ActuatorPdKernel;
use crate::math::Real;
use crate::sim::{Control, Model, State};
use crate::storage::{Buffer, BufferUsage};

/// Configuration for one PD-controlled DOF.
#[derive(Debug, Clone, Copy)]
pub struct ControllerPdConfig {
    /// Index into `state.joint_q` / `state.joint_qd` /
    /// `control.joint_target_pos` / `control.joint_target_vel`.
    pub dof: u32,
    /// Proportional gain [N/m or N·m/rad].
    pub kp: Real,
    /// Derivative gain [N·s/m or N·m·s/rad].
    pub kd: Real,
    /// Effort cap [N or N·m]. `0.0` disables the cap.
    pub max_effort: Real,
}

/// Stateless GPU PD controller batch.
pub struct ControllerPd {
    kernel: ActuatorPdKernel,
    actuator_count: u32,
    dof_indices: Buffer<u32>,
    kp: Buffer<Real>,
    kd: Buffer<Real>,
    max_effort: Buffer<Real>,
}

impl ControllerPd {
    pub fn new(
        engine: &Engine,
        model: &Model,
        configs: &[ControllerPdConfig],
    ) -> Result<Self> {
        if configs.is_empty() {
            return Err(Error::Solver(
                "ControllerPd::new: empty config list".into(),
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("controller_pd: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("controller_pd: no GPU queue".into()))?
            .clone();

        // Validate DOF bounds.
        for (i, c) in configs.iter().enumerate() {
            if c.dof as usize >= model.joint_dof_total {
                return Err(Error::Solver(format!(
                    "ControllerPd::new: actuator {i} dof {} out of range (joint_dof_total={})",
                    c.dof, model.joint_dof_total
                )));
            }
        }
        // Check unique DOFs (else the kernel writes race on same slot).
        for i in 0..configs.len() {
            for j in (i + 1)..configs.len() {
                if configs[i].dof == configs[j].dof {
                    return Err(Error::Solver(format!(
                        "ControllerPd::new: duplicate dof {} at actuators {i} and {j}",
                        configs[i].dof
                    )));
                }
            }
        }

        let dof_idx: Vec<u32> = configs.iter().map(|c| c.dof).collect();
        let kp_v: Vec<Real> = configs.iter().map(|c| c.kp).collect();
        let kd_v: Vec<Real> = configs.iter().map(|c| c.kd).collect();
        let me_v: Vec<Real> = configs.iter().map(|c| c.max_effort).collect();

        let mut dof_indices = Buffer::from_vec(dof_idx, BufferUsage::Storage);
        let mut kp = Buffer::from_vec(kp_v, BufferUsage::Storage);
        let mut kd = Buffer::from_vec(kd_v, BufferUsage::Storage);
        let mut max_effort = Buffer::from_vec(me_v, BufferUsage::Storage);

        dof_indices.ensure_device(&device)?;
        kp.ensure_device(&device)?;
        kd.ensure_device(&device)?;
        max_effort.ensure_device(&device)?;
        dof_indices.upload(&queue)?;
        kp.upload(&queue)?;
        kd.upload(&queue)?;
        max_effort.upload(&queue)?;

        Ok(Self {
            kernel: ActuatorPdKernel::new(&device),
            actuator_count: configs.len() as u32,
            dof_indices,
            kp,
            kd,
            max_effort,
        })
    }

    /// Number of actuators in this controller.
    pub fn actuator_count(&self) -> u32 {
        self.actuator_count
    }

    /// Run the kernel against the current `state` + `control`. Writes
    /// effort to `control.joint_force[dof]` for each configured DOF.
    pub fn step(
        &mut self,
        engine: &Engine,
        state: &State,
        control: &mut Control,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("controller_pd: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("controller_pd: no GPU queue".into()))?
            .clone();

        self.kernel.dispatch(
            &device,
            &queue,
            state,
            control,
            &self.dof_indices,
            &self.kp,
            &self.kd,
            &self.max_effort,
            self.actuator_count,
        )
    }
}
