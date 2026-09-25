//! Public frame-transform sensor — orchestrates a
//! [`FrameTransformKernel`] dispatch.
//!
//! Tracks N (body, local-pose) frames per step, returning each
//! frame's world pose, and (optionally) its pose **relative to a
//! reference frame** (also a body+local-pose pair). Useful for
//! end-effector tracking, chase-cam targets, trajectory recording.
//!
//! GPU-first: the per-frame world / relative pose math runs in
//! `shaders/frame_transform.wgsl` (one thread per target). When a
//! reference frame is configured, the caller passes it as **target
//! index 0** and the kernel computes everyone else's pose relative
//! to it.

use crate::device_results::{
    EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
    SensorFramePoolConfig,
};
use crate::dispatch::{BindingsCache, SensorDispatchStats};
use crate::readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp, ReadbackStats};
use crate::{SensorArrayLimits, SensorArrayStorage};
use molla_compute::Engine;
use molla_core::{BodyId, Error, Result};
use molla_kernels::{FrameTransformBindings, FrameTransformKernel};
use molla_math::{Quat, Transform, Vec3};
use molla_sim::{Model, State};
use molla_storage::{Buffer, BufferUsage};

/// Device channels with an active target range excluding the internal reference.
/// Submit consumers before the next dispatch; unsubmitted commands do not lease outputs.
pub struct FrameDeviceReadings<'a> {
    pub world_position: &'a DeviceBuffer<Vec3>,
    pub world_rotation: &'a DeviceBuffer<Quat>,
    pub relative_position: &'a DeviceBuffer<Vec3>,
    pub relative_rotation: &'a DeviceBuffer<Quat>,
    pub target_range: std::ops::Range<usize>,
}

/// One frame to track.
#[derive(Debug, Clone, Copy)]
pub struct FrameTarget {
    pub body: BodyId,
    pub local_pose: Transform,
}

/// Per-frame reading.
#[derive(Debug, Clone, Copy)]
pub struct FrameReading {
    /// World-space pose of the target frame.
    pub world: Transform,
    /// Pose of the target frame **expressed in the reference frame**.
    /// Equal to `world` when the sensor was created without a
    /// reference (`reference = None`).
    pub relative: Transform,
}

/// GPU frame-transform sensor batch.
pub struct FrameTransformSensor {
    kernel: FrameTransformKernel,
    bindings: BindingsCache<FrameTransformBindings>,
    leased_bindings: BindingsCache<FrameTransformBindings>,
    dispatch_stats: SensorDispatchStats,
    has_reference: bool,
    target_count: u32,
    body_count: usize,
    storage_usage: SensorArrayStorage,

    target_body: Buffer<u32>,
    target_local_pos: Buffer<Vec3>,
    target_local_rot: Buffer<Quat>,

    world_pos: DeviceBuffer<Vec3>,
    world_rot: DeviceBuffer<Quat>,
    relative_pos: DeviceBuffer<Vec3>,
    relative_rot: DeviceBuffer<Quat>,
    readback: Option<ReadbackRing>,
}

impl FrameTransformSensor {
    /// Build the sensor. If `reference` is `Some`, every reading's
    /// `relative` field is expressed in the reference frame; if
    /// `None`, `relative == world`.
    pub fn new(
        engine: &Engine,
        model: &Model,
        reference: Option<FrameTarget>,
        targets: &[FrameTarget],
    ) -> Result<Self> {
        Self::new_with_limits(
            engine,
            model,
            reference,
            targets,
            SensorArrayLimits::default(),
        )
    }

    /// Construct with a combined retained host/device array budget.
    pub fn new_with_limits(
        engine: &Engine,
        model: &Model,
        reference: Option<FrameTarget>,
        targets: &[FrameTarget],
        limits: SensorArrayLimits,
    ) -> Result<Self> {
        engine.check_device()?;
        if targets.is_empty() {
            return Err(Error::Solver(
                "FrameTransformSensor::new: empty target list".into(),
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("frame_transform: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("frame_transform: engine has no GPU queue".into()))?
            .clone();

        let has_ref = reference.is_some();
        let n = targets
            .len()
            .checked_add(usize::from(has_ref))
            .ok_or_else(|| Error::Solver("frame_transform: target count overflow".into()))?;
        let target_count = u32::try_from(n)
            .map_err(|_| Error::Solver("frame_transform: too many targets".into()))?;
        let bodies = u32::try_from(model.body_count)
            .map_err(|_| Error::Solver("frame_transform: too many bodies".into()))?;
        let storage_usage = SensorArrayStorage::for_frames(targets.len(), has_ref)?;
        limits.check(
            storage_usage,
            target_count,
            u64::from(bodies) * 32,
            &device.limits(),
        )?;
        let all = || reference.iter().chain(targets.iter());
        for (i, t) in all().enumerate() {
            if t.body.index() >= model.body_count {
                return Err(Error::Solver(format!(
                    "FrameTransformSensor::new: target {i} body {} out of range (count={})",
                    t.body.index(),
                    model.body_count
                )));
            }
            if !t.local_pose.position.is_finite()
                || !t.local_pose.position.as_vec3().is_finite()
                || !t.local_pose.rotation.is_finite()
                || (t.local_pose.rotation.length_squared() - 1.0).abs() > 1e-6
            {
                return Err(Error::Solver(format!(
                    "frame_transform: invalid target {i} pose"
                )));
            }
        }

        let target = |i: usize| {
            if let Some(reference) = reference.as_ref() {
                if i == 0 {
                    return reference;
                }
                &targets[i - 1]
            } else {
                &targets[i]
            }
        };
        let body_idx = crate::host_arrays::from_fn(n, |i| target(i).body.index() as u32)?;
        let local_pos = crate::host_arrays::from_fn(n, |i| target(i).local_pose.position)?;
        let local_rot = crate::host_arrays::from_fn(n, |i| target(i).local_pose.rotation)?;

        let mut target_body = Buffer::from_vec(body_idx, BufferUsage::Storage);
        let mut target_local_pos = Buffer::from_vec(local_pos, BufferUsage::Storage);
        let mut target_local_rot = Buffer::from_vec(local_rot, BufferUsage::Storage);
        let usage = BufferUsage::StorageReadWrite;
        let world_pos =
            DeviceBuffer::from_host(engine, crate::host_arrays::filled(n, Vec3::ZERO)?, usage)?;
        let world_rot = DeviceBuffer::from_host(
            engine,
            crate::host_arrays::filled(n, Quat::IDENTITY)?,
            usage,
        )?;
        let relative_pos =
            DeviceBuffer::from_host(engine, crate::host_arrays::filled(n, Vec3::ZERO)?, usage)?;
        let relative_rot = DeviceBuffer::from_host(
            engine,
            crate::host_arrays::filled(n, Quat::IDENTITY)?,
            usage,
        )?;

        target_body.ensure_device(&device)?;
        target_local_pos.ensure_device(&device)?;
        target_local_rot.ensure_device(&device)?;
        target_body.upload(&queue)?;
        target_local_pos.upload(&queue)?;
        target_local_rot.upload(&queue)?;

        Ok(Self {
            kernel: FrameTransformKernel::new(&device),
            bindings: BindingsCache::new(),
            leased_bindings: BindingsCache::with_capacity(128),
            dispatch_stats: SensorDispatchStats::default(),
            has_reference: has_ref,
            target_count,
            body_count: model.body_count,
            storage_usage,
            target_body,
            target_local_pos,
            target_local_rot,
            world_pos,
            world_rot,
            relative_pos,
            relative_rot,
            readback: None,
        })
    }

    /// Combined retained host/device array payload, excluding collection and output pools.
    pub fn storage_usage(&self) -> SensorArrayStorage {
        self.storage_usage
    }

    /// Run the kernel against `state` and return one reading per
    /// caller-supplied target (the reference, if any, is **not**
    /// returned — it's an implementation slot).
    pub fn update(&mut self, engine: &Engine, state: &State) -> Result<Vec<FrameReading>> {
        self.dispatch(engine, state)?;
        let range = usize::from(self.has_reference)..self.target_count as usize;
        if self.readback.is_none() {
            let bytes = range.len() as u64 * 64;
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
            self.world_pos.readback_source(range.clone())?,
            self.world_rot.readback_source(range.clone())?,
            self.relative_pos.readback_source(range.clone())?,
            self.relative_rot.readback_source(range)?,
        ];
        let ticket = ring
            .try_enqueue(&sources, ReadbackStamp::default())?
            .ok_or_else(|| Error::Storage("frame_transform: readback slot busy".into()))?;
        let frame = match ring.wait(&ticket, std::time::Duration::from_secs(30)) {
            Ok(frame) => frame,
            Err(error) => {
                ring.reset();
                return Err(error);
            }
        };
        Ok(frame
            .channel_iter::<Vec3>(0)?
            .zip(frame.channel_iter::<Quat>(1)?)
            .zip(frame.channel_iter::<Vec3>(2)?)
            .zip(frame.channel_iter::<Quat>(3)?)
            .map(|(((wp, wr), rp), rr)| FrameReading {
                world: Transform::new(wp, wr),
                relative: Transform::new(rp, rr),
            })
            .collect())
    }

    /// Submit world/relative transforms without collection. State must contain
    /// current poses on this engine; recreate after body remapping or replacement.
    pub fn dispatch(&mut self, engine: &Engine, state: &State) -> Result<FrameDeviceReadings<'_>> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("frame_transform: no GPU device".into()))?
            .clone();
        if !std::sync::Arc::ptr_eq(&device, &self.world_pos.device) {
            return Err(Error::Gpu(
                "frame_transform belongs to a different engine".into(),
            ));
        }
        if state.body_q.len() != self.body_count {
            return Err(Error::Solver(
                "frame_transform: body count changed; recreate sensor".into(),
            ));
        }
        state.body_q.device_buffer_current()?;
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("frame_transform: no GPU queue".into()))?
            .clone();

        let buffers = [
            state.body_q.device_buffer_current()?,
            self.target_body.device_buffer_current()?,
            self.target_local_pos.device_buffer_current()?,
            self.target_local_rot.device_buffer_current()?,
            self.world_pos.device_buffer()?,
            self.world_rot.device_buffer()?,
            self.relative_pos.device_buffer()?,
            self.relative_rot.device_buffer()?,
        ];
        let (bindings, prepared) = self.bindings.get_or_prepare(
            |binding| binding.matches(buffers),
            || self.kernel.prepare(&device, buffers),
        );
        self.dispatch_stats.binding_preparations += u64::from(prepared);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::frame_transform sensor"),
        });
        self.kernel.encode_prepared(
            &queue,
            &mut encoder,
            bindings,
            self.target_count,
            self.has_reference,
        );
        queue.submit([encoder.finish()]);
        self.dispatch_stats.submissions += 1;
        self.dispatch_stats.kernel_dispatches += 1;
        self.world_pos.mark_device_dirty();
        self.world_rot.mark_device_dirty();
        self.relative_pos.mark_device_dirty();
        self.relative_rot.mark_device_dirty();
        Ok(FrameDeviceReadings {
            world_position: &self.world_pos,
            world_rotation: &self.world_rot,
            relative_position: &self.relative_pos,
            relative_rotation: &self.relative_rot,
            target_range: usize::from(self.has_reference)..self.target_count as usize,
        })
    }

    fn output_specs(&self) -> [SensorChannelSpec; 4] {
        let count = self.target_count as usize;
        [
            SensorChannelSpec::of::<Vec3>(count),
            SensorChannelSpec::of::<Quat>(count),
            SensorChannelSpec::of::<Vec3>(count),
            SensorChannelSpec::of::<Quat>(count),
        ]
    }

    /// Create direct-output slots. Channels are world position/rotation, then relative position/rotation.
    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        engine.check_device()?;
        if !std::sync::Arc::ptr_eq(
            engine.device().expect("checked engine"),
            &self.world_pos.device,
        ) {
            return Err(Error::Gpu("frame_transform: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Encode directly into leased slots without submitting or collecting.
    /// `None` signals backpressure without encoding commands. Active ranges exclude the reference.
    /// The producer and input state stay borrowed until batch submission or drop.
    /// Upload state using `engine.device()`'s canonical Arc, not a raw device reference.
    ///
    /// ```compile_fail
    /// use molla_sensors::{FrameTransformSensor, SensorCommandBatch, SensorFramePool};
    /// use molla_sensors::readback::ReadbackStamp;
    /// use molla_compute::Engine;
    /// use molla_sim::State;
    /// fn overwrite(sensor: &mut FrameTransformSensor, engine: &Engine, state: &State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     sensor.encode(&mut batch, engine, state, pool, ReadbackStamp::default()).unwrap();
    ///     sensor.dispatch(engine, state).unwrap();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_sensors::{FrameTransformSensor, SensorCommandBatch, SensorFramePool};
    /// use molla_sensors::readback::ReadbackStamp;
    /// use molla_compute::Engine;
    /// use molla_sim::State;
    /// fn mutate_state(sensor: &mut FrameTransformSensor, engine: &Engine, state: &mut State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     sensor.encode(&mut batch, engine, state, pool, ReadbackStamp::default()).unwrap();
    ///     state.body_q.host_mut().unwrap();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    pub fn encode<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        engine: &Engine,
        state: &'a State,
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedSensorFrame>> {
        batch.check_engine(engine)?;
        let device = engine.device().expect("checked engine");
        if !std::sync::Arc::ptr_eq(device, &self.world_pos.device) {
            return Err(Error::Gpu("frame_transform: foreign engine".into()));
        }
        if state.body_q.len() != self.body_count {
            return Err(Error::Solver(
                "frame_transform: body count changed; recreate sensor".into(),
            ));
        }
        let poses = state.body_q.device_buffer_for(device)?;
        let inputs = [
            poses,
            self.target_body.device_buffer_for(device)?,
            self.target_local_pos.device_buffer_for(device)?,
            self.target_local_rot.device_buffer_for(device)?,
        ];
        let specs = self.output_specs();
        let ranges = std::array::from_fn::<_, 4, _>(|_| {
            usize::from(self.has_reference)..self.target_count as usize
        });
        batch.encode_output(
            pool,
            &specs,
            &ranges,
            stamp,
            |device, queue, encoder, outputs| {
                let buffers = [
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    &outputs[0],
                    &outputs[1],
                    &outputs[2],
                    &outputs[3],
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
                    self.target_count,
                    self.has_reference,
                );
                Ok(1)
            },
        )
    }

    /// Legacy submitted work plus preparations from both paths. Direct encode work is counted on its batch.
    /// Legacy bindings cache two variants; direct-output bindings cache 128 variants.
    pub fn dispatch_stats(&self) -> SensorDispatchStats {
        self.dispatch_stats
    }

    /// Collection counters; device-only dispatch leaves these zero.
    pub fn readback_stats(&self) -> ReadbackStats {
        self.readback
            .as_ref()
            .map(ReadbackRing::stats)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor_storage::tests::{assert_payload, buffer_bytes};

    #[test]
    #[ignore = "requires GPU"]
    fn frame_array_estimate_matches_allocated_payload_with_optional_reference() {
        let engine = Engine::new_gpu().unwrap();
        let mut builder = molla_sim::ModelBuilder::new();
        let body = builder.add_body(Default::default());
        let model = builder.build().unwrap();
        let target = FrameTarget {
            body,
            local_pose: Transform::IDENTITY,
        };
        for reference in [None, Some(target)] {
            let sensor =
                FrameTransformSensor::new(&engine, &model, reference, &[target; 3]).unwrap();
            assert_payload(
                sensor.storage_usage(),
                &[
                    buffer_bytes(&sensor.target_body),
                    buffer_bytes(&sensor.target_local_pos),
                    buffer_bytes(&sensor.target_local_rot),
                    buffer_bytes(&sensor.world_pos.buffer),
                    buffer_bytes(&sensor.world_rot.buffer),
                    buffer_bytes(&sensor.relative_pos.buffer),
                    buffer_bytes(&sensor.relative_rot.buffer),
                ],
            );
        }
    }
}
