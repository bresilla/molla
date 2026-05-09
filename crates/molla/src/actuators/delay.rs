//! Per-DOF command-input delay stage.
//!
//! Mirrors Newton's `Delay` (xtra/newton/_src/actuators/delay.py).
//! Owns a circular buffer of depth `max_delay` per actuator, plus the
//! per-DOF lag values. Each step pushes the current targets into the
//! buffer, advances the write pointer, and reads the lagged sample
//! per actuator into the controller's input arrays.
//!
//! When `max_delay == 0` (or every actuator has `delay_steps == 0`)
//! the kernel still runs but degenerates into a passthrough — the
//! caller's controller sees the current targets unchanged.

use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::kernels::ActuatorDelayKernel;
use crate::math::Real;
use crate::sim::Control;
use crate::storage::{Buffer, BufferUsage};

/// Per-DOF delay configuration.
#[derive(Debug, Clone, Copy)]
pub struct DelayConfig {
    /// Index into `Control.joint_target_pos / joint_target_vel`.
    pub dof: u32,
    /// Number of substeps of lag for this actuator. `0` ⇒ passthrough.
    pub delay_steps: u32,
}

/// GPU-resident circular delay buffer + per-DOF lag values.
pub struct Delay {
    kernel: ActuatorDelayKernel,
    actuator_count: u32,
    buf_depth: u32,
    dof_indices: Buffer<u32>,
    delay_steps: Buffer<u32>,

    // Buffer state — kept on the device, never read by host except in
    // tests that explicitly download.
    buffer_pos: Buffer<Real>,
    buffer_vel: Buffer<Real>,
    num_pushes: Buffer<u32>,
    write_idx: Buffer<u32>,

    // Outputs of the read pass — consumed by the controller stage.
    out_pos: Buffer<Real>,
    out_vel: Buffer<Real>,
}

impl Delay {
    /// Allocate a new Delay stage. `max_delay` is the per-actuator
    /// circular-buffer depth, which must be ≥ 1; per-DOF
    /// `delay_steps` are clamped to `max_delay` at read time.
    pub fn new(engine: &Engine, configs: &[DelayConfig], max_delay: u32) -> Result<Self> {
        if configs.is_empty() {
            return Err(Error::Solver("Delay::new: empty config list".into()));
        }
        if max_delay == 0 {
            return Err(Error::Solver(
                "Delay::new: max_delay must be >= 1".into(),
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("Delay: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("Delay: no GPU queue".into()))?
            .clone();

        let n = configs.len();
        let dof: Vec<u32> = configs.iter().map(|c| c.dof).collect();
        let delays: Vec<u32> = configs
            .iter()
            .map(|c| c.delay_steps.min(max_delay))
            .collect();

        let mut dof_indices = Buffer::from_vec(dof, BufferUsage::Storage);
        let mut delay_steps = Buffer::from_vec(delays, BufferUsage::Storage);
        let buf_size = (max_delay as usize) * n;
        let mut buffer_pos =
            Buffer::from_vec(vec![0.0 as Real; buf_size], BufferUsage::StorageReadWrite);
        let mut buffer_vel =
            Buffer::from_vec(vec![0.0 as Real; buf_size], BufferUsage::StorageReadWrite);
        let mut num_pushes = Buffer::from_vec(vec![0u32; n], BufferUsage::StorageReadWrite);
        // Start at `max_delay - 1` so the first push lands at slot 0
        // (matching Newton's convention).
        let mut write_idx =
            Buffer::from_vec(vec![max_delay - 1], BufferUsage::StorageReadWrite);
        let mut out_pos = Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);
        let mut out_vel = Buffer::from_vec(vec![0.0 as Real; n], BufferUsage::StorageReadWrite);

        for buf in [
            &mut dof_indices as &mut dyn DeviceUploadable,
            &mut delay_steps,
        ] {
            buf.ensure_and_upload(&device, &queue)?;
        }
        for buf in [
            &mut buffer_pos as &mut dyn DeviceUploadableReal,
            &mut buffer_vel,
            &mut out_pos,
            &mut out_vel,
        ] {
            buf.ensure_and_upload(&device, &queue)?;
        }
        for buf in [
            &mut num_pushes as &mut dyn DeviceUploadable,
            &mut write_idx,
        ] {
            buf.ensure_and_upload(&device, &queue)?;
        }

        Ok(Self {
            kernel: ActuatorDelayKernel::new(&device),
            actuator_count: n as u32,
            buf_depth: max_delay,
            dof_indices,
            delay_steps,
            buffer_pos,
            buffer_vel,
            num_pushes,
            write_idx,
            out_pos,
            out_vel,
        })
    }

    pub fn actuator_count(&self) -> u32 {
        self.actuator_count
    }

    pub fn buf_depth(&self) -> u32 {
        self.buf_depth
    }

    /// Read-only access to the delayed-output position array (one
    /// entry per actuator). The controller stage reads from this.
    pub fn out_pos(&self) -> &Buffer<Real> {
        &self.out_pos
    }

    /// Read-only access to the delayed-output velocity array.
    pub fn out_vel(&self) -> &Buffer<Real> {
        &self.out_vel
    }

    /// Mutable access to the delayed-output position array, used by
    /// tests / introspection that need to call `download` on it.
    pub fn out_pos_mut(&mut self) -> &mut Buffer<Real> {
        &mut self.out_pos
    }

    /// Mutable access to the delayed-output velocity array, used by
    /// tests / introspection that need to call `download` on it.
    pub fn out_vel_mut(&mut self) -> &mut Buffer<Real> {
        &mut self.out_vel
    }

    /// Update per-DOF delay values from host-side data (number of
    /// substeps; clamped to `max_delay`). Useful for live-tuning lag
    /// from a UI without rebuilding the Delay.
    pub fn set_delay_steps(&mut self, engine: &Engine, values: &[u32]) -> Result<()> {
        if values.len() != self.actuator_count as usize {
            return Err(Error::Solver(format!(
                "Delay::set_delay_steps: expected {} entries, got {}",
                self.actuator_count,
                values.len()
            )));
        }
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("Delay::set_delay_steps: no GPU queue".into()))?
            .clone();
        let dst = self.delay_steps.host_mut()?;
        for (d, v) in dst.iter_mut().zip(values.iter()) {
            *d = (*v).min(self.buf_depth);
        }
        self.delay_steps.upload(&queue)?;
        Ok(())
    }

    /// Reset every per-DOF buffer + counter to its initial state.
    /// Subsequent reads behave as if no targets had ever been pushed.
    pub fn reset(&mut self, engine: &Engine) -> Result<()> {
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("Delay::reset: no GPU queue".into()))?
            .clone();
        for v in self.buffer_pos.host_mut()? {
            *v = 0.0;
        }
        for v in self.buffer_vel.host_mut()? {
            *v = 0.0;
        }
        for v in self.num_pushes.host_mut()? {
            *v = 0;
        }
        for v in self.write_idx.host_mut()? {
            *v = self.buf_depth - 1;
        }
        for v in self.out_pos.host_mut()? {
            *v = 0.0;
        }
        for v in self.out_vel.host_mut()? {
            *v = 0.0;
        }
        self.buffer_pos.upload(&queue)?;
        self.buffer_vel.upload(&queue)?;
        self.num_pushes.upload(&queue)?;
        self.write_idx.upload(&queue)?;
        self.out_pos.upload(&queue)?;
        self.out_vel.upload(&queue)?;
        Ok(())
    }

    /// Push the current targets through the buffer and read the
    /// lagged sample. Should be called once per substep, before the
    /// controller stage that will consume `out_pos / out_vel`.
    pub fn step(&mut self, engine: &Engine, control: &Control) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("Delay::step: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("Delay::step: no GPU queue".into()))?
            .clone();

        self.kernel.dispatch(
            &device,
            &queue,
            &control.joint_target_pos,
            &control.joint_target_vel,
            &self.dof_indices,
            &self.delay_steps,
            &mut self.buffer_pos,
            &mut self.buffer_vel,
            &mut self.num_pushes,
            &mut self.write_idx,
            &mut self.out_pos,
            &mut self.out_vel,
            self.actuator_count,
            self.buf_depth,
        )
    }
}

// Tiny ad-hoc trait so the constructor can fan out the
// ensure_device + upload pair across heterogeneous Real / u32 buffers
// without 12 lines of repetition.
trait DeviceUploadable {
    fn ensure_and_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<()>;
}

impl DeviceUploadable for Buffer<u32> {
    fn ensure_and_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        self.ensure_device(device)?;
        self.upload(queue)
    }
}

trait DeviceUploadableReal {
    fn ensure_and_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<()>;
}

impl DeviceUploadableReal for Buffer<Real> {
    fn ensure_and_upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        self.ensure_device(device)?;
        self.upload(queue)
    }
}
