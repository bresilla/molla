//! Symmetric per-actuator effort clamp stage.

use super::Clamping;
use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::kernels::ClampingMaxEffortKernel;
use crate::math::Real;
use crate::sim::{Control, State};
use crate::storage::{Buffer, BufferUsage};

#[derive(Debug, Clone, Copy)]
pub struct ClampingMaxEffortConfig {
    /// Index into `Control.joint_force`.
    pub dof: u32,
    /// Effort cap [N or N·m]. `0` ⇒ disabled (passthrough).
    pub max_effort: Real,
}

pub struct ClampingMaxEffort {
    kernel: ClampingMaxEffortKernel,
    actuator_count: u32,
    dof_indices: Buffer<u32>,
    max_effort: Buffer<Real>,
}

impl ClampingMaxEffort {
    pub fn new(engine: &Engine, configs: &[ClampingMaxEffortConfig]) -> Result<Self> {
        if configs.is_empty() {
            return Err(Error::Solver(
                "ClampingMaxEffort::new: empty config list".into(),
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ClampingMaxEffort: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ClampingMaxEffort: no GPU queue".into()))?
            .clone();

        let dof: Vec<u32> = configs.iter().map(|c| c.dof).collect();
        let me: Vec<Real> = configs.iter().map(|c| c.max_effort).collect();

        let mut dof_indices = Buffer::from_vec(dof, BufferUsage::Storage);
        let mut max_effort = Buffer::from_vec(me, BufferUsage::Storage);
        dof_indices.ensure_device(&device)?;
        max_effort.ensure_device(&device)?;
        dof_indices.upload(&queue)?;
        max_effort.upload(&queue)?;

        Ok(Self {
            kernel: ClampingMaxEffortKernel::new(&device),
            actuator_count: configs.len() as u32,
            dof_indices,
            max_effort,
        })
    }

    /// Update the per-actuator cap from host-side data.
    pub fn set_max_effort(&mut self, engine: &Engine, values: &[Real]) -> Result<()> {
        if values.len() != self.actuator_count as usize {
            return Err(Error::Solver(format!(
                "ClampingMaxEffort::set_max_effort: expected {} entries, got {}",
                self.actuator_count,
                values.len()
            )));
        }
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("set_max_effort: no GPU queue".into()))?
            .clone();
        let dst = self.max_effort.host_mut()?;
        dst.copy_from_slice(values);
        self.max_effort.upload(&queue)?;
        Ok(())
    }
}

impl Clamping for ClampingMaxEffort {
    fn step(
        &mut self,
        engine: &Engine,
        _state: &State,
        control: &mut Control,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ClampingMaxEffort::step: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ClampingMaxEffort::step: no GPU queue".into()))?
            .clone();
        self.kernel.dispatch(
            &device,
            &queue,
            control,
            &self.dof_indices,
            &self.max_effort,
            self.actuator_count,
        )
    }
}
