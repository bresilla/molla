//! DC-motor four-quadrant clamp stage. Models torque-vs-speed
//! saturation directly (back-EMF + winding-resistance behaviour
//! linearised against velocity).

use super::Clamping;
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::ClampingDcMotorKernel;
use molla_math::Real;
use molla_sim::{Control, State};
use molla_storage::{Buffer, BufferUsage};

/// Per-actuator DC-motor parameters. Equivalent (Newton-style) form
/// of the same numbers a roboticist already has on a motor sheet:
///
/// - `stall_torque`        — peak torque at zero velocity
///   [N·m or N], `≈ Kt · V / R`
/// - `no_load_speed`       — velocity at which the linear envelope
///   crosses zero torque [rad/s or m/s],
///   `≈ V / Kt`
/// - `max_motor_effort`    — current-limited torque ceiling
///   [N·m or N]; the `min(line, cap)` rail.
#[derive(Debug, Clone, Copy)]
pub struct ClampingDcMotorConfig {
    pub dof: u32,
    pub stall_torque: Real,
    pub no_load_speed: Real,
    pub max_motor_effort: Real,
}

pub struct ClampingDcMotor {
    kernel: ClampingDcMotorKernel,
    actuator_count: u32,
    dof_indices: Buffer<u32>,
    stall_torque: Buffer<Real>,
    no_load_speed: Buffer<Real>,
    max_motor_effort: Buffer<Real>,
    corner_velocity: Buffer<Real>,
}

impl ClampingDcMotor {
    pub fn new(engine: &Engine, configs: &[ClampingDcMotorConfig]) -> Result<Self> {
        if configs.is_empty() {
            return Err(Error::Solver(
                "ClampingDcMotor::new: empty config list".into(),
            ));
        }
        for (i, c) in configs.iter().enumerate() {
            if c.no_load_speed <= 0.0 {
                return Err(Error::Solver(format!(
                    "ClampingDcMotor::new: actuator {i} no_load_speed must be > 0, got {}",
                    c.no_load_speed
                )));
            }
            if c.stall_torque < 0.0 {
                return Err(Error::Solver(format!(
                    "ClampingDcMotor::new: actuator {i} stall_torque must be ≥ 0, got {}",
                    c.stall_torque
                )));
            }
            if c.max_motor_effort < 0.0 {
                return Err(Error::Solver(format!(
                    "ClampingDcMotor::new: actuator {i} max_motor_effort must be ≥ 0, got {}",
                    c.max_motor_effort
                )));
            }
        }

        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ClampingDcMotor: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ClampingDcMotor: no GPU queue".into()))?
            .clone();

        let dof: Vec<u32> = configs.iter().map(|c| c.dof).collect();
        let st: Vec<Real> = configs.iter().map(|c| c.stall_torque).collect();
        let nl: Vec<Real> = configs.iter().map(|c| c.no_load_speed).collect();
        let me: Vec<Real> = configs.iter().map(|c| c.max_motor_effort).collect();
        let corner: Vec<Real> = configs
            .iter()
            .map(|c| Self::compute_corner(c.stall_torque, c.no_load_speed, c.max_motor_effort))
            .collect();

        let mut dof_indices = Buffer::from_vec(dof, BufferUsage::Storage);
        let mut stall_torque = Buffer::from_vec(st, BufferUsage::Storage);
        let mut no_load_speed = Buffer::from_vec(nl, BufferUsage::Storage);
        let mut max_motor_effort = Buffer::from_vec(me, BufferUsage::Storage);
        let mut corner_velocity = Buffer::from_vec(corner, BufferUsage::Storage);
        {
            let buf = &mut dof_indices as &mut dyn UploadU32;
            buf.ensure_and_upload(&device, &queue)?;
        }
        for buf in [
            &mut stall_torque as &mut dyn UploadReal,
            &mut no_load_speed,
            &mut max_motor_effort,
            &mut corner_velocity,
        ] {
            buf.ensure_and_upload(&device, &queue)?;
        }

        Ok(Self {
            kernel: ClampingDcMotorKernel::new(&device),
            actuator_count: configs.len() as u32,
            dof_indices,
            stall_torque,
            no_load_speed,
            max_motor_effort,
            corner_velocity,
        })
    }

    /// Closed-form intersection of `stall · (1 - v / no_load)` with
    /// `max_motor_effort`. Beyond this velocity the linear envelope
    /// is below the cap and the cap doesn't bite, so we clamp the
    /// velocity at `corner` before evaluating the line — without
    /// this, very-high-velocity inputs would flip the clamp into the
    /// negative quadrant.
    fn compute_corner(stall: Real, no_load: Real, max_motor: Real) -> Real {
        if stall <= 0.0 || max_motor >= stall {
            return no_load;
        }
        no_load * (1.0 + max_motor / stall)
    }

    /// Update the per-actuator parameters in place; recomputes
    /// `corner_velocity` accordingly.
    pub fn set_params(
        &mut self,
        engine: &Engine,
        stall: &[Real],
        no_load: &[Real],
        max_motor: &[Real],
    ) -> Result<()> {
        let n = self.actuator_count as usize;
        if stall.len() != n || no_load.len() != n || max_motor.len() != n {
            return Err(Error::Solver(format!(
                "ClampingDcMotor::set_params: expected {n} entries each"
            )));
        }
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("set_params: no GPU queue".into()))?
            .clone();
        self.stall_torque.host_mut()?.copy_from_slice(stall);
        self.no_load_speed.host_mut()?.copy_from_slice(no_load);
        self.max_motor_effort.host_mut()?.copy_from_slice(max_motor);
        let corner_dst = self.corner_velocity.host_mut()?;
        for i in 0..n {
            corner_dst[i] = Self::compute_corner(stall[i], no_load[i], max_motor[i]);
        }
        self.stall_torque.upload(&queue)?;
        self.no_load_speed.upload(&queue)?;
        self.max_motor_effort.upload(&queue)?;
        self.corner_velocity.upload(&queue)?;
        Ok(())
    }
}

impl Clamping for ClampingDcMotor {
    fn step(&mut self, engine: &Engine, state: &State, control: &mut Control) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ClampingDcMotor::step: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ClampingDcMotor::step: no GPU queue".into()))?
            .clone();
        self.kernel.dispatch(
            &device,
            &queue,
            state,
            control,
            &self.dof_indices,
            &self.stall_torque,
            &self.no_load_speed,
            &self.max_motor_effort,
            &self.corner_velocity,
            self.actuator_count,
        )
    }
}

trait UploadU32 {
    fn ensure_and_upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()>;
}
impl UploadU32 for Buffer<u32> {
    fn ensure_and_upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        self.ensure_device(device)?;
        self.upload(queue)
    }
}
trait UploadReal {
    fn ensure_and_upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()>;
}
impl UploadReal for Buffer<Real> {
    fn ensure_and_upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        self.ensure_device(device)?;
        self.upload(queue)
    }
}
