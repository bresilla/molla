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
//! The first call snapshots device velocity and returns no readings.

use crate::device_results::{
    EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
    SensorFramePoolConfig, SubmitAction,
};
use crate::dispatch::{BindingsCache, SensorDispatchStats};
use crate::readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp, ReadbackStats};
use crate::{SensorArrayLimits, SensorArrayStorage};
use molla_compute::Engine;
use molla_core::{BodyId, Error, Result};
use molla_kernels::{ImuBindings, ImuKernel};
use molla_math::{Quat, Real, SpatialVector, Transform, Vec3};
use molla_sim::{Model, State};
use molla_storage::{Buffer, BufferUsage};

/// Borrowed IMU outputs; enqueue GPU consumers or readback before the next dispatch.
pub struct ImuDeviceReadings<'a> {
    pub accelerometer: &'a DeviceBuffer<Vec3>,
    pub gyroscope: &'a DeviceBuffer<Vec3>,
}

/// A priming frame has empty active channels; measured frames contain all mounts.
pub struct EncodedImuSample {
    pub frame: EncodedSensorFrame,
    pub priming: bool,
}

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
    bindings: BindingsCache<ImuBindings>,
    leased_bindings: BindingsCache<ImuBindings>,
    dispatch_stats: SensorDispatchStats,
    sensor_count: u32,
    storage_usage: SensorArrayStorage,

    sensor_body: Buffer<u32>,
    sensor_local_pos: Buffer<Vec3>,
    sensor_local_rot: Buffer<Quat>,

    /// Previous sampled device velocity, sized to the model's body count.
    prev_body_qd: Buffer<SpatialVector>,
    /// Whether velocity history contains a previous sample.
    primed: bool,
    model_buffer: Option<wgpu::Buffer>,
    pending_model: Option<wgpu::Buffer>,

    accel: DeviceBuffer<Vec3>,
    gyro: DeviceBuffer<Vec3>,
    readback: Option<ReadbackRing>,
}

impl ImuSensor {
    /// Build a sensor batch with one IMU per entry in `mounts`. Cheap
    /// to construct; reusable across `update` calls.
    pub fn new(engine: &Engine, model: &Model, mounts: &[ImuMount]) -> Result<Self> {
        Self::new_with_limits(engine, model, mounts, SensorArrayLimits::default())
    }

    /// Construct with a combined retained host/device array budget.
    pub fn new_with_limits(
        engine: &Engine,
        model: &Model,
        mounts: &[ImuMount],
        limits: SensorArrayLimits,
    ) -> Result<Self> {
        engine.check_device()?;
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

        let n = mounts.len();
        let storage_usage = SensorArrayStorage::for_imu(n, model.body_count)?;
        let sensor_count =
            u32::try_from(n).map_err(|_| Error::Solver("imu: too many mounts".into()))?;
        limits.check(storage_usage, sensor_count, 0, &device.limits())?;

        for (i, m) in mounts.iter().enumerate() {
            let idx = m.body.index();
            if idx >= model.body_count {
                return Err(Error::Solver(format!(
                    "ImuSensor::new: mount {i} body {idx} out of range (count={})",
                    model.body_count
                )));
            }
            if !m.local_pose.position.is_finite()
                || !m.local_pose.position.as_vec3().is_finite()
                || !m.local_pose.rotation.is_finite()
                || (m.local_pose.rotation.length_squared() - 1.0).abs() > 1e-6
            {
                return Err(Error::Solver(format!("imu: invalid mount {i} pose")));
            }
        }

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
        let accel =
            DeviceBuffer::from_host(engine, vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite)?;
        let gyro =
            DeviceBuffer::from_host(engine, vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite)?;

        sensor_body.ensure_device(&device)?;
        sensor_local_pos.ensure_device(&device)?;
        sensor_local_rot.ensure_device(&device)?;
        prev_body_qd.ensure_device(&device)?;
        sensor_body.upload(&queue)?;
        sensor_local_pos.upload(&queue)?;
        sensor_local_rot.upload(&queue)?;
        prev_body_qd.upload(&queue)?;

        Ok(Self {
            kernel: ImuKernel::new(&device),
            bindings: BindingsCache::new(),
            leased_bindings: BindingsCache::with_capacity(128),
            dispatch_stats: SensorDispatchStats::default(),
            sensor_count,
            storage_usage,
            sensor_body,
            sensor_local_pos,
            sensor_local_rot,
            prev_body_qd,
            primed: false,
            model_buffer: None,
            pending_model: None,
            accel,
            gyro,
            readback: None,
        })
    }

    /// Number of IMU mounts in the batch.
    pub fn sensor_count(&self) -> u32 {
        self.sensor_count
    }

    /// Combined retained host/device array payload, excluding collection and output pools.
    pub fn storage_usage(&self) -> SensorArrayStorage {
        self.storage_usage
    }

    /// Run the kernel against the current `state` with timestep `dt`.
    /// Returns the per-sensor readings (length = `sensor_count`), or
    /// an empty vec on the first call (sensor isn't primed yet —
    /// finite-difference would compare against zeroes).
    /// Inputs must be uploaded; `dt` must convert to a positive normal f32.
    /// Reset before replacing a same-sized model; recreate for a new body count.
    pub fn update(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        dt: Real,
    ) -> Result<Vec<ImuReading>> {
        if self.dispatch(engine, model, state, dt)?.is_none() {
            return Ok(Vec::new());
        }
        if self.readback.is_none() {
            let bytes = self.sensor_count as u64 * 32;
            self.readback = Some(ReadbackRing::new(
                engine,
                ReadbackConfig {
                    slots: 1,
                    bytes_per_slot: bytes,
                    max_total_bytes: bytes,
                },
            )?);
        }
        let ring = self.readback.as_mut().expect("initialized readback");
        let sources = [
            self.accel.readback_source(0..self.sensor_count as usize)?,
            self.gyro.readback_source(0..self.sensor_count as usize)?,
        ];
        let ticket = ring
            .try_enqueue(&sources, ReadbackStamp::default())?
            .ok_or_else(|| Error::Storage("imu: synchronous readback slot busy".into()))?;
        let frame = match ring.wait(&ticket, std::time::Duration::from_secs(30)) {
            Ok(frame) => frame,
            Err(error) => {
                ring.reset();
                return Err(error);
            }
        };
        Ok(frame
            .channel_iter::<Vec3>(0)?
            .zip(frame.channel_iter::<Vec3>(1)?)
            .map(|(accelerometer, gyroscope)| ImuReading {
                accelerometer,
                gyroscope,
            })
            .collect())
    }

    /// Submit evaluation and history update without host readback or blocking waits.
    /// Returns `None` while priming. Borrowed outputs belong to this engine's queue.
    pub fn dispatch(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        dt: Real,
    ) -> Result<Option<ImuDeviceReadings<'_>>> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("imu: no GPU device".into()))?;
        if !std::sync::Arc::ptr_eq(device, &self.accel.device) {
            return Err(Error::Gpu("imu belongs to a different engine".into()));
        }
        let gpu_dt = dt as f32;
        if !gpu_dt.is_normal() || gpu_dt <= 0.0 {
            return Err(Error::Solver(
                "imu: dt must be positive and normal in f32".into(),
            ));
        }
        if model.body_count != self.prev_body_qd.len()
            || state.body_qd.len() != model.body_count
            || state.body_q.len() != model.body_count
        {
            return Err(Error::Solver(
                "imu: body count changed; recreate the sensor".into(),
            ));
        }
        let velocity = state.body_qd.device_buffer_current()?;
        state.body_q.device_buffer_current()?;
        let model_buffer = model.body_com.device_buffer_current()?;
        model.body_world.device_buffer_current()?;
        model.gravity.device_buffer_current()?;
        if self
            .model_buffer
            .as_ref()
            .is_some_and(|old| old != model_buffer)
        {
            return Err(Error::Solver(
                "imu: model changed; reset the sensor before sampling".into(),
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("imu: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("imu: engine has no GPU queue".into()))?
            .clone();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::imu sample and history"),
        });
        let was_primed = self.primed;
        if was_primed {
            let buffers = [
                state.body_q.device_buffer_current()?,
                velocity,
                self.prev_body_qd.device_buffer_current()?,
                model_buffer,
                model.body_world.device_buffer_current()?,
                model.gravity.device_buffer_current()?,
                self.sensor_body.device_buffer_current()?,
                self.sensor_local_pos.device_buffer_current()?,
                self.sensor_local_rot.device_buffer_current()?,
                self.accel.device_buffer()?,
                self.gyro.device_buffer()?,
            ];
            let (bindings, prepared) = self.bindings.get_or_prepare(
                |binding| binding.matches(buffers),
                || self.kernel.prepare(&device, buffers),
            );
            self.dispatch_stats.binding_preparations += u64::from(prepared);
            self.kernel.encode_prepared(
                &queue,
                &mut encoder,
                bindings,
                self.sensor_count,
                gpu_dt,
            )?;
        }

        let history = self
            .prev_body_qd
            .device_buffer()
            .ok_or_else(|| Error::Storage("imu: missing history buffer".into()))?;
        encoder.copy_buffer_to_buffer(velocity, 0, history, 0, history.size());
        queue.submit(std::iter::once(encoder.finish()));
        self.dispatch_stats.submissions += 1;
        self.dispatch_stats.history_copies += 1;
        if was_primed {
            self.dispatch_stats.kernel_dispatches += 1;
            self.accel.mark_device_dirty();
            self.gyro.mark_device_dirty();
        }
        self.prev_body_qd.mark_device_dirty();
        self.primed = true;
        self.model_buffer = Some(model_buffer.clone());
        if !was_primed {
            return Ok(None);
        }

        Ok(Some(ImuDeviceReadings {
            accelerometer: &self.accel,
            gyroscope: &self.gyro,
        }))
    }

    fn output_specs(&self) -> [SensorChannelSpec; 2] {
        [SensorChannelSpec::of::<Vec3>(self.sensor_count as usize); 2]
    }

    /// Fixed-capacity channels: accelerometer and gyroscope.
    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        engine.check_device()?;
        if !std::sync::Arc::ptr_eq(engine.device().expect("checked engine"), &self.accel.device) {
            return Err(Error::Gpu("imu: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Encode a sample and its history copy into leased output slots.
    /// `None` is backpressure; `priming` frames have no measured elements.
    /// Model/state must be allocated with the engine's canonical device Arc.
    /// History advances only on submission; drop or failed submission cancels the sample.
    /// `dt` measures time since the last submitted sample, not a backpressured attempt.
    ///
    /// ```compile_fail
    /// use molla_sensors::{ImuSensor, SensorCommandBatch, SensorFramePool};
    /// use molla_compute::Engine;
    /// use molla_sim::{Model, State};
    /// fn reset_pending(imu: &mut ImuSensor, engine: &Engine, model: &Model, state: &State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     imu.encode(&mut batch, engine, model, state, 0.1, pool, Default::default()).unwrap();
    ///     imu.reset();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_sensors::{ImuSensor, SensorCommandBatch, SensorFramePool};
    /// use molla_compute::Engine;
    /// use molla_sim::{Model, State};
    /// fn mutate_model(imu: &mut ImuSensor, engine: &Engine, model: &mut Model, state: &State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     imu.encode(&mut batch, engine, model, state, 0.1, pool, Default::default()).unwrap();
    ///     model.gravity.host_mut().unwrap();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn encode<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        engine: &Engine,
        model: &'a Model,
        state: &'a State,
        dt: Real,
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedImuSample>> {
        batch.check_engine(engine)?;
        batch.check_commit_capacity()?;
        let device = engine.device().expect("checked engine");
        if !std::sync::Arc::ptr_eq(device, &self.accel.device) {
            return Err(Error::Gpu("imu: foreign engine".into()));
        }
        let gpu_dt = dt as f32;
        if !gpu_dt.is_normal() || gpu_dt <= 0.0 {
            return Err(Error::Solver(
                "imu: dt must be positive and normal in f32".into(),
            ));
        }
        if model.body_count != self.prev_body_qd.len()
            || state.body_qd.len() != model.body_count
            || state.body_q.len() != model.body_count
            || model.body_com.len() != model.body_count
            || model.body_world.len() != model.body_count
            || model.world_count == 0
            || model.gravity.len() != model.world_count
        {
            return Err(Error::Solver(
                "imu: inconsistent model/state dimensions; recreate sensor after body remapping"
                    .into(),
            ));
        }
        let pose = state.body_q.device_buffer_for(device)?;
        let velocity = state.body_qd.device_buffer_for(device)?;
        let model_buffer = model.body_com.device_buffer_for(device)?;
        let world = model.body_world.device_buffer_for(device)?;
        let gravity = model.gravity.device_buffer_for(device)?;
        if self
            .model_buffer
            .as_ref()
            .is_some_and(|old| old != model_buffer)
        {
            return Err(Error::Solver(
                "imu: model changed; reset the sensor before sampling".into(),
            ));
        }
        let history = self.prev_body_qd.device_buffer_for(device)?;
        let mounts = [
            self.sensor_body.device_buffer_for(device)?,
            self.sensor_local_pos.device_buffer_for(device)?,
            self.sensor_local_rot.device_buffer_for(device)?,
        ];
        let specs = self.output_specs();
        let priming = !self.primed;
        let active = if priming {
            0
        } else {
            self.sensor_count as usize
        };
        let encoded = batch.encode_output(
            pool,
            &specs,
            &[0..active, 0..active],
            stamp,
            |device, queue, encoder, outputs| {
                if !priming {
                    let buffers = [
                        pose,
                        velocity,
                        history,
                        model_buffer,
                        world,
                        gravity,
                        mounts[0],
                        mounts[1],
                        mounts[2],
                        &outputs[0],
                        &outputs[1],
                    ];
                    let (bindings, prepared) = self.leased_bindings.get_or_prepare(
                        |binding| binding.matches(buffers),
                        || self.kernel.prepare(device, buffers),
                    );
                    self.dispatch_stats.binding_preparations += u64::from(prepared);
                    self.kernel.encode_prepared(
                        queue,
                        encoder,
                        bindings,
                        self.sensor_count,
                        gpu_dt,
                    )?;
                }
                encoder.copy_buffer_to_buffer(velocity, 0, history, 0, history.size());
                Ok(u64::from(!priming))
            },
        )?;
        let Some(frame) = encoded else {
            return Ok(None);
        };
        self.pending_model = Some(model_buffer.clone());
        batch.record_history_copy();
        batch.on_submit(self)?;
        Ok(Some(EncodedImuSample { frame, priming }))
    }

    /// Legacy submitted work plus preparations from both paths. Direct work is counted on its batch.
    pub fn dispatch_stats(&self) -> SensorDispatchStats {
        self.dispatch_stats
    }

    /// Internal synchronous collection counters; device-only dispatch leaves them zero.
    pub fn readback_stats(&self) -> ReadbackStats {
        self.readback
            .as_ref()
            .map(ReadbackRing::stats)
            .unwrap_or_default()
    }

    /// Reset the finite-difference filter — next [`Self::update`] will
    /// re-prime by snapshotting `state.body_qd` and return an empty
    /// vec. Use after teleporting a body or on scene reset.
    pub fn reset(&mut self) {
        if let Some(readback) = &mut self.readback {
            readback.reset();
        }
        self.primed = false;
        self.model_buffer = None;
        self.pending_model = None;
    }
}

impl SubmitAction for ImuSensor {
    fn commit(&mut self) {
        self.model_buffer = self.pending_model.take();
        self.prev_body_qd.mark_device_dirty();
        self.primed = true;
    }

    fn cancel(&mut self) {
        self.pending_model = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor_storage::tests::{assert_payload, buffer_bytes};

    #[test]
    #[ignore = "requires GPU"]
    fn imu_array_estimate_matches_allocated_payload() {
        let engine = Engine::new_gpu().unwrap();
        let mut builder = molla_sim::ModelBuilder::new();
        let body = builder.add_body(Default::default());
        builder.add_body(Default::default());
        let model = builder.build().unwrap();
        let sensor = ImuSensor::new(
            &engine,
            &model,
            &[ImuMount {
                body,
                local_pose: Transform::IDENTITY,
            }; 3],
        )
        .unwrap();
        assert_payload(
            sensor.storage_usage(),
            &[
                buffer_bytes(&sensor.sensor_body),
                buffer_bytes(&sensor.sensor_local_pos),
                buffer_bytes(&sensor.sensor_local_rot),
                buffer_bytes(&sensor.prev_body_qd),
                buffer_bytes(&sensor.accel.buffer),
                buffer_bytes(&sensor.gyro.buffer),
            ],
        );
    }
}
