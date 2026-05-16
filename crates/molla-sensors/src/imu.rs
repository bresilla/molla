//! Public IMU sensor — orchestrates an [`ImuKernel`] dispatch.
//!
//! GPU-first: the per-sensor accelerometer + gyroscope math runs in
//! `shaders/imu.wgsl`. This module owns the sensor configuration
//! buffers, a previous-frame `body_qd` snapshot (for finite-difference
//! acceleration), and the output buffers.
//!
//! # Acceleration via finite difference
//!
//! Newton's IMU reads `body_qdd` (linear + angular accelerations) from
//! the solver. molla doesn't carry that field, so [`ImuSensor::update`]
//! finite-differences successive `body_qd` snapshots:
//!
//! ```text
//! a_lin = (qd.linear  - qd_prev.linear)  / dt
//! a_ang = (qd.angular - qd_prev.angular) / dt
//! ```
//!
//! On the very first call `qd_prev == qd`, so the first reading's
//! linear acceleration is just `-gravity` (i.e. accelerometer reads
//! gravity at rest, as expected). Discard the first frame's reading
//! if you need a clean signal.

use molla_compute::Engine;
use molla_core::{BodyId, Error, Result};
use molla_kernels::ImuKernel;
use molla_math::{Quat, Real, SpatialVector, Transform, Vec3};
use molla_sim::{Model, State};
use molla_storage::{Buffer, BufferUsage};

/// Single per-frame IMU reading.
#[derive(Debug, Clone, Copy)]
pub struct ImuReading {
    /// Linear acceleration (specific force) in **sensor frame** [m/s²].
    /// A body in free fall reads ≈ 0; at rest on the ground reads
    /// `-g` along whichever local axis points "up" through the sensor.
    pub accelerometer: Vec3,
    /// Angular velocity in **sensor frame** [rad/s].
    pub gyroscope: Vec3,
}

/// One IMU mount point.
#[derive(Debug, Clone, Copy)]
pub struct ImuMount {
    /// Body the sensor is bolted to.
    pub body: BodyId,
    /// Sensor pose in body frame — origin = mount position, rotation
    /// = sensor axis convention (e.g. +Y up for many automotive IMUs).
    pub local_pose: Transform,
}

/// GPU IMU sensor batch. Owns all per-sensor buffers + the
/// previous-frame `body_qd` snapshot.
pub struct ImuSensor {
    kernel: ImuKernel,
    sensor_count: u32,

    sensor_body: Buffer<u32>,
    sensor_local_pos: Buffer<Vec3>,
    sensor_local_rot: Buffer<Quat>,

    /// Previous-frame `state.body_qd` snapshot, sized to the model's
    /// `body_count`. We swap with `state.body_qd` after each dispatch
    /// to keep the finite-difference rolling.
    prev_body_qd: Buffer<SpatialVector>,
    /// `true` once we've populated `prev_body_qd` with a real frame.
    /// Until then, `update` skips the kernel and returns `None` — the
    /// first dispatch would compare against zeroes and produce a
    /// nonsense impulse.
    primed: bool,

    accel: Buffer<Vec3>,
    gyro: Buffer<Vec3>,
}

impl ImuSensor {
    /// Build a sensor batch with one IMU per entry in `mounts`. Cheap
    /// to construct; reusable across `update` calls.
    pub fn new(engine: &Engine, model: &Model, mounts: &[ImuMount]) -> Result<Self> {
        if mounts.is_empty() {
            return Err(Error::Solver("ImuSensor::new: empty mount list".into()));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("imu: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("imu: engine has no GPU queue".into()))?
            .clone();

        // Validate body indices.
        for (i, m) in mounts.iter().enumerate() {
            let idx = m.body.index();
            if idx >= model.body_count {
                return Err(Error::Solver(format!(
                    "ImuSensor::new: mount {i} body {idx} out of range (count={})",
                    model.body_count
                )));
            }
        }

        let n = mounts.len();
        let body_idx: Vec<u32> = mounts.iter().map(|m| m.body.index() as u32).collect();
        let local_pos: Vec<Vec3> = mounts.iter().map(|m| m.local_pose.position).collect();
        let local_rot: Vec<Quat> = mounts.iter().map(|m| m.local_pose.rotation).collect();

        let mut sensor_body = Buffer::from_vec(body_idx, BufferUsage::Storage);
        let mut sensor_local_pos = Buffer::from_vec(local_pos, BufferUsage::Storage);
        let mut sensor_local_rot = Buffer::from_vec(local_rot, BufferUsage::Storage);
        let mut prev_body_qd = Buffer::from_vec(
            vec![SpatialVector::ZERO; model.body_count],
            BufferUsage::StorageReadWrite,
        );
        let mut accel = Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);
        let mut gyro = Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);

        sensor_body.ensure_device(&device)?;
        sensor_local_pos.ensure_device(&device)?;
        sensor_local_rot.ensure_device(&device)?;
        prev_body_qd.ensure_device(&device)?;
        accel.ensure_device(&device)?;
        gyro.ensure_device(&device)?;
        sensor_body.upload(&queue)?;
        sensor_local_pos.upload(&queue)?;
        sensor_local_rot.upload(&queue)?;
        prev_body_qd.upload(&queue)?;
        accel.upload(&queue)?;
        gyro.upload(&queue)?;

        Ok(Self {
            kernel: ImuKernel::new(&device),
            sensor_count: n as u32,
            sensor_body,
            sensor_local_pos,
            sensor_local_rot,
            prev_body_qd,
            primed: false,
            accel,
            gyro,
        })
    }

    /// Number of IMU mounts in the batch.
    pub fn sensor_count(&self) -> u32 {
        self.sensor_count
    }

    /// Run the kernel against the current `state` with timestep `dt`.
    /// Returns the per-sensor readings (length = `sensor_count`), or
    /// an empty vec on the first call (sensor isn't primed yet —
    /// finite-difference would compare against zeroes).
    pub fn update(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        dt: Real,
    ) -> Result<Vec<ImuReading>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("imu: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("imu: engine has no GPU queue".into()))?
            .clone();

        // First call: just snapshot the current body_qd into prev_body_qd
        // and bail out — we have nothing to finite-difference against yet.
        if !self.primed {
            // Pull fresh values from the device, then upload to prev.
            let host_now = state.body_qd.host()?.to_vec();
            self.prev_body_qd.host_mut()?.copy_from_slice(&host_now);
            self.prev_body_qd.upload(&queue)?;
            self.primed = true;
            return Ok(Vec::new());
        }

        self.kernel.dispatch(
            &device,
            &queue,
            model,
            state,
            &self.prev_body_qd,
            &self.sensor_body,
            &self.sensor_local_pos,
            &self.sensor_local_rot,
            &mut self.accel,
            &mut self.gyro,
            self.sensor_count,
            dt as f32,
        )?;

        // Roll prev_body_qd ← state.body_qd for the next call.
        let host_now = state.body_qd.host()?.to_vec();
        self.prev_body_qd.host_mut()?.copy_from_slice(&host_now);
        self.prev_body_qd.upload(&queue)?;

        // Pull outputs.
        self.accel.download(&device, &queue)?;
        self.gyro.download(&device, &queue)?;
        let accel = self.accel.host()?;
        let gyro = self.gyro.host()?;
        let mut out = Vec::with_capacity(self.sensor_count as usize);
        for i in 0..self.sensor_count as usize {
            out.push(ImuReading {
                accelerometer: accel[i],
                gyroscope: gyro[i],
            });
        }
        Ok(out)
    }

    /// Reset the finite-difference filter — next [`Self::update`] will
    /// re-prime by snapshotting `state.body_qd` and return an empty
    /// vec. Use after teleporting a body or on scene reset.
    pub fn reset(&mut self) {
        self.primed = false;
    }
}
