//! Pipeline-PD controller: drives `Control.joint_force[dof]` from
//! per-actuator `(delayed_target_pos, delayed_target_vel)` produced
//! by an upstream [`crate::Delay`] stage. Effort cap is
//! intentionally not applied here — clamping is its own stage in
//! the Newton-style `Delay → Controller → Clamping` pipeline.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::ActuatorPipelinePdKernel;
use molla_math::Real;
use molla_sim::{Control, State};
use molla_storage::{Buffer, BufferUsage};

#[derive(Debug, Clone, Copy)]
pub struct ControllerPipelinePdConfig {
    pub dof: u32,
    pub kp: Real,
    pub kd: Real,
}

pub struct ControllerPipelinePd {
    kernel: ActuatorPipelinePdKernel,
    actuator_count: u32,
    dof_indices: Buffer<u32>,
    kp: Buffer<Real>,
    kd: Buffer<Real>,
}

impl ControllerPipelinePd {
    pub fn new(engine: &Engine, configs: &[ControllerPipelinePdConfig]) -> Result<Self> {
        if configs.is_empty() {
            return Err(Error::Solver(
                "ControllerPipelinePd::new: empty config list".into(),
            ));
        }
        // Duplicate-DOF check: each actuator writes joint_force[dof],
        // and concurrent writes to the same slot would race.
        for i in 0..configs.len() {
            for j in (i + 1)..configs.len() {
                if configs[i].dof == configs[j].dof {
                    return Err(Error::Solver(format!(
                        "ControllerPipelinePd::new: duplicate dof {} at actuators {i} and {j}",
                        configs[i].dof
                    )));
                }
            }
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePd: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePd: no GPU queue".into()))?
            .clone();

        let dof: Vec<u32> = configs.iter().map(|c| c.dof).collect();
        let kp_v: Vec<Real> = configs.iter().map(|c| c.kp).collect();
        let kd_v: Vec<Real> = configs.iter().map(|c| c.kd).collect();
        let mut dof_indices = Buffer::from_vec(dof, BufferUsage::Storage);
        let mut kp = Buffer::from_vec(kp_v, BufferUsage::Storage);
        let mut kd = Buffer::from_vec(kd_v, BufferUsage::Storage);
        dof_indices.ensure_device(&device)?;
        kp.ensure_device(&device)?;
        kd.ensure_device(&device)?;
        dof_indices.upload(&queue)?;
        kp.upload(&queue)?;
        kd.upload(&queue)?;

        Ok(Self {
            kernel: ActuatorPipelinePdKernel::new(&device),
            actuator_count: configs.len() as u32,
            dof_indices,
            kp,
            kd,
        })
    }

    pub fn actuator_count(&self) -> u32 {
        self.actuator_count
    }

    /// Update the gain table.
    pub fn set_gains(&mut self, engine: &Engine, kp: &[Real], kd: &[Real]) -> Result<()> {
        let n = self.actuator_count as usize;
        if kp.len() != n || kd.len() != n {
            return Err(Error::Solver(format!(
                "ControllerPipelinePd::set_gains: expected {n} entries each"
            )));
        }
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("set_gains: no GPU queue".into()))?
            .clone();
        self.kp.host_mut()?.copy_from_slice(kp);
        self.kd.host_mut()?.copy_from_slice(kd);
        self.kp.upload(&queue)?;
        self.kd.upload(&queue)?;
        Ok(())
    }

    pub fn step(
        &mut self,
        engine: &Engine,
        state: &State,
        control: &mut Control,
        delayed_target_pos: &Buffer<Real>,
        delayed_target_vel: &Buffer<Real>,
    ) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePd::step: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ControllerPipelinePd::step: no GPU queue".into()))?
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
            &self.kd,
            self.actuator_count,
        )
    }
}
