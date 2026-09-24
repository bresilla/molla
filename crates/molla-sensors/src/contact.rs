//! Public contact sensor — orchestrates a [`ContactAggregateKernel`]
//! dispatch and returns one [`BodyContactReading`] per body.
//!
//! GPU-first: the per-body aggregation runs in
//! `shaders/contact_aggregate.wgsl` (one thread per body, walks the
//! contact CSR). This module owns the four output GPU buffers
//! and optionally collects a host-side `Vec<BodyContactReading>`.
//!
//! # What it measures
//!
//! For each body, the sensor reports
//! - `count` — number of contacts the body participates in.
//! - `max_depth` — largest penetration depth across the body's
//!   contacts.
//! - `normal_sum` — unweighted signed normals, negative on the a-side
//!   and positive on the b-side of each contact.
//! - `centroid` — mean contact point in world frame.
//!
//! This GPU sensor reports geometry, not forces. CPU Featherstone exposes
//! penalty forces through `FeatherstoneSolver::contact_forces`; `RigidWorld`
//! groups those readings by stable collider pair.

use crate::device_results::{
    EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
    SensorFramePoolConfig,
};
use crate::dispatch::{BindingsCache, SensorDispatchStats};
use crate::readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp, ReadbackStats};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{ContactAggregateBindings, ContactAggregateKernel};
use molla_math::{Real, Vec3};
use molla_sim::{Contacts, Model};
use molla_storage::BufferUsage;

/// Per-body device outputs; submit consumers before the next dispatch.
pub struct ContactDeviceReadings<'a> {
    pub count: &'a DeviceBuffer<u32>,
    pub max_depth: &'a DeviceBuffer<Real>,
    pub normal_sum: &'a DeviceBuffer<Vec3>,
    pub centroid: &'a DeviceBuffer<Vec3>,
}

/// Per-body contact summary returned by [`ContactSensor::update`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BodyContactReading {
    /// Number of contacts this body participates in this frame.
    pub count: u32,
    /// Largest penetration depth across the body's contacts (m).
    /// Zero when the body has no contacts.
    pub max_depth: Real,
    /// Unweighted signed normals, negative on the a-side and positive
    /// on the b-side. Zero when the body has no contacts; not a force sum.
    pub normal_sum: Vec3,
    /// Mean contact point in world frame. Zero when no contacts.
    pub centroid: Vec3,
}

/// Fixed-size GPU contact sensor; recreate after body remapping or count changes.
pub struct ContactSensor {
    kernel: ContactAggregateKernel,
    bindings: BindingsCache<ContactAggregateBindings>,
    leased_bindings: BindingsCache<ContactAggregateBindings>,
    dispatch_stats: SensorDispatchStats,
    body_count: u32,
    out_count: DeviceBuffer<u32>,
    out_max_depth: DeviceBuffer<Real>,
    out_normal_sum: DeviceBuffer<Vec3>,
    out_centroid: DeviceBuffer<Vec3>,
    readback: Option<ReadbackRing>,
}

impl ContactSensor {
    pub fn new(engine: &Engine, model: &Model) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("contact sensor: engine has no GPU device".into()))?
            .clone();
        let n = model.body_count;
        let body_count = u32::try_from(n)
            .map_err(|_| Error::Solver("contact sensor: too many bodies".into()))?;
        let limits = device.limits();
        if n as u64 * 16 > limits.max_storage_buffer_binding_size as u64
            || n as u64 * 16 > limits.max_buffer_size
            || body_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(Error::Solver(
                "contact sensor: capacity exceeds GPU limits".into(),
            ));
        }
        let usage = BufferUsage::StorageReadWrite;
        let out_count = DeviceBuffer::from_host(engine, vec![0u32; n], usage)?;
        let out_max_depth = DeviceBuffer::from_host(engine, vec![0.0; n], usage)?;
        let out_normal_sum = DeviceBuffer::from_host(engine, vec![Vec3::ZERO; n], usage)?;
        let out_centroid = DeviceBuffer::from_host(engine, vec![Vec3::ZERO; n], usage)?;

        Ok(Self {
            kernel: ContactAggregateKernel::new(&device),
            bindings: BindingsCache::new(),
            leased_bindings: BindingsCache::with_capacity(128),
            dispatch_stats: SensorDispatchStats::default(),
            body_count,
            out_count,
            out_max_depth,
            out_normal_sum,
            out_centroid,
            readback: None,
        })
    }

    /// Number of bodies covered by the sensor (= `model.body_count`
    /// at construction time).
    pub fn body_count(&self) -> u32 {
        self.body_count
    }

    /// Run the kernel against the current `contacts` and return one
    /// reading per body (length = `body_count`).
    pub fn update(
        &mut self,
        engine: &Engine,
        contacts: &Contacts,
    ) -> Result<Vec<BodyContactReading>> {
        self.dispatch(engine, contacts)?;
        let n = self.body_count as usize;
        if n == 0 {
            return Ok(Vec::new());
        }
        if self.readback.is_none() {
            let bytes = n as u64 * 40;
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
            self.out_count.readback_source(0..n)?,
            self.out_max_depth.readback_source(0..n)?,
            self.out_normal_sum.readback_source(0..n)?,
            self.out_centroid.readback_source(0..n)?,
        ];
        let ticket = ring
            .try_enqueue(&sources, ReadbackStamp::default())?
            .ok_or_else(|| Error::Storage("contact sensor: readback slot busy".into()))?;
        let frame = match ring.wait(&ticket, std::time::Duration::from_secs(30)) {
            Ok(frame) => frame,
            Err(error) => {
                ring.reset();
                return Err(error);
            }
        };
        Ok(frame
            .channel_iter::<u32>(0)?
            .zip(frame.channel_iter::<Real>(1)?)
            .zip(frame.channel_iter::<Vec3>(2)?)
            .zip(frame.channel_iter::<Vec3>(3)?)
            .map(
                |(((count, max_depth), normal_sum), centroid)| BodyContactReading {
                    count,
                    max_depth,
                    normal_sum,
                    centroid,
                },
            )
            .collect())
    }

    /// Submit aggregation without readback. Contacts and their CSR must be current
    /// on this engine; recreate the sensor after changing body indexing/count.
    pub fn dispatch(
        &mut self,
        engine: &Engine,
        contacts: &Contacts,
    ) -> Result<ContactDeviceReadings<'_>> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("contact sensor: engine has no GPU device".into()))?
            .clone();
        if !std::sync::Arc::ptr_eq(&device, &self.out_count.device) {
            return Err(Error::Gpu(
                "contact sensor belongs to a different engine".into(),
            ));
        }
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("contact sensor: engine has no GPU queue".into()))?
            .clone();

        if self.body_count == 0 {
            if contacts.count != 0 {
                return Err(Error::Solver(
                    "contact sensor: contacts without bodies".into(),
                ));
            }
        } else if contacts.count == 0 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::contact sensor clear"),
            });
            for buffer in [
                self.out_count.device_buffer()?,
                self.out_max_depth.device_buffer()?,
                self.out_normal_sum.device_buffer()?,
                self.out_centroid.device_buffer()?,
            ] {
                encoder.clear_buffer(buffer, 0, None);
            }
            queue.submit([encoder.finish()]);
            self.dispatch_stats.submissions += 1;
            self.dispatch_stats.output_clears += 1;
            self.out_count.mark_device_dirty();
            self.out_max_depth.mark_device_dirty();
            self.out_normal_sum.mark_device_dirty();
            self.out_centroid.mark_device_dirty();
        } else {
            let count = contacts.count as usize;
            if count > contacts.capacity
                || count > contacts.a_index.len()
                || count > contacts.b_index.len()
                || count > contacts.normal.len()
                || count > contacts.depth.len()
                || count > contacts.point.len()
                || contacts.body_contact_offsets.len() != self.body_count as usize + 1
                || contacts.body_contact_indices.is_empty()
            {
                return Err(Error::Solver(
                    "contact sensor: inconsistent contact arrays/CSR".into(),
                ));
            }
            let buffers = [
                contacts.a_index.device_buffer_current()?,
                contacts.b_index.device_buffer_current()?,
                contacts.normal.device_buffer_current()?,
                contacts.depth.device_buffer_current()?,
                contacts.point.device_buffer_current()?,
                contacts.body_contact_offsets.device_buffer_current()?,
                contacts.body_contact_indices.device_buffer_current()?,
                self.out_count.device_buffer()?,
                self.out_max_depth.device_buffer()?,
                self.out_normal_sum.device_buffer()?,
                self.out_centroid.device_buffer()?,
            ];
            let (bindings, prepared) = self.bindings.get_or_prepare(
                |binding| binding.matches(buffers),
                || self.kernel.prepare(&device, buffers),
            );
            self.dispatch_stats.binding_preparations += u64::from(prepared);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::contact sensor aggregate"),
            });
            self.kernel
                .encode_prepared(&queue, &mut encoder, bindings, self.body_count);
            queue.submit([encoder.finish()]);
            self.dispatch_stats.submissions += 1;
            self.dispatch_stats.kernel_dispatches += 1;
            self.out_count.mark_device_dirty();
            self.out_max_depth.mark_device_dirty();
            self.out_normal_sum.mark_device_dirty();
            self.out_centroid.mark_device_dirty();
        }
        Ok(ContactDeviceReadings {
            count: &self.out_count,
            max_depth: &self.out_max_depth,
            normal_sum: &self.out_normal_sum,
            centroid: &self.out_centroid,
        })
    }

    fn output_specs(&self) -> [SensorChannelSpec; 4] {
        let count = self.body_count as usize;
        [
            SensorChannelSpec::of::<u32>(count),
            SensorChannelSpec::of::<Real>(count),
            SensorChannelSpec::of::<Vec3>(count),
            SensorChannelSpec::of::<Vec3>(count),
        ]
    }

    /// Fixed-capacity channels: count, maximum depth, signed normals, centroid.
    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        engine.check_device()?;
        if !std::sync::Arc::ptr_eq(
            engine.device().expect("checked engine"),
            &self.out_count.device,
        ) {
            return Err(Error::Gpu("contact sensor: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Encode aggregation into leased slots. Empty contacts clear all active bodies.
    /// Contacts must be allocated/uploaded with the engine's canonical device Arc.
    /// `None` reports backpressure without encoding; input/producer borrows last until submit/drop.
    ///
    /// ```compile_fail
    /// use molla_sensors::{ContactSensor, SensorCommandBatch, SensorFramePool};
    /// use molla_compute::Engine;
    /// use molla_sim::Contacts;
    /// fn change_contacts(sensor: &mut ContactSensor, engine: &Engine, contacts: &mut Contacts, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     sensor.encode(&mut batch, engine, contacts, pool, Default::default()).unwrap();
    ///     contacts.clear();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    pub fn encode<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        engine: &Engine,
        contacts: &'a Contacts,
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedSensorFrame>> {
        batch.check_engine(engine)?;
        let device = engine.device().expect("checked engine");
        if !std::sync::Arc::ptr_eq(device, &self.out_count.device) {
            return Err(Error::Gpu("contact sensor: foreign engine".into()));
        }
        let count = contacts.count as usize;
        let inputs = if count == 0 {
            None
        } else {
            if self.body_count == 0
                || count > contacts.capacity
                || count > contacts.a_index.len()
                || count > contacts.b_index.len()
                || count > contacts.normal.len()
                || count > contacts.depth.len()
                || count > contacts.point.len()
                || contacts.body_contact_offsets.len() != self.body_count as usize + 1
                || contacts.body_contact_indices.is_empty()
            {
                return Err(Error::Solver(
                    "contact sensor: inconsistent contact arrays/CSR".into(),
                ));
            }
            Some([
                contacts.a_index.device_buffer_for(device)?,
                contacts.b_index.device_buffer_for(device)?,
                contacts.normal.device_buffer_for(device)?,
                contacts.depth.device_buffer_for(device)?,
                contacts.point.device_buffer_for(device)?,
                contacts.body_contact_offsets.device_buffer_for(device)?,
                contacts.body_contact_indices.device_buffer_for(device)?,
            ])
        };
        let specs = self.output_specs();
        let ranges = std::array::from_fn::<_, 4, _>(|_| 0..self.body_count as usize);
        batch.encode_output(
            pool,
            &specs,
            &ranges,
            stamp,
            |device, queue, encoder, outputs| {
                let Some(inputs) = inputs else {
                    return Ok(0);
                };
                let buffers = [
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4],
                    inputs[5],
                    inputs[6],
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
                self.kernel
                    .encode_prepared(queue, encoder, bindings, self.body_count);
                Ok(1)
            },
        )
    }

    /// Legacy submitted work plus preparations from both paths. Direct work is counted on its batch.
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
