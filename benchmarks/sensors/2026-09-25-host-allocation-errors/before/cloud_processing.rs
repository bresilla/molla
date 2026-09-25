//! Device-resident cloud construction, rigid transforms, filters and stable compaction.

use crate::processor_storage::filled;
use crate::{
    Compare, EncodedLidarBatch, LidarDeviceBatch, LidarPattern,
    device_results::{
        EncodedSensorFrame, PendingSensorInputs, SensorChannelSpec, SensorCommandBatch,
        SensorFramePool, SensorFramePoolConfig,
    },
    dispatch::BindingsCache,
    readback::{BufferLease, DeviceBuffer, ReadbackSource, ReadbackStamp},
};
use crate::{ProcessorStorageLimits, ProcessorStorageUsage};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{CloudDescriptor, CloudProcessBindings, CloudProcessKernel};
use molla_math::{Real, Transform, Vec3};
use molla_storage::{BufferElement, BufferUsage};
use std::{any::TypeId, ops::Range, sync::Arc};

/// 112-byte GPU/host record. Flags: valid=1, color=2, scalar=4, angles=8.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudPoint {
    /// XYZ and intensity.
    pub position: [f32; 4],
    /// Normal XYZ and original ray distance.
    pub normal: [f32; 4],
    /// Actual ray direction XYZ and optional scalar.
    pub direction: [f32; 4],
    /// Sensor ID low/high, sample ID low/high.
    pub ids: [u32; 4],
    /// Row, column, sub-ray and original source index.
    pub grid: [u32; 4],
    /// Shape ID, packed RGBA8, flags and reserved word.
    pub extra: [u32; 4],
    /// Theta, phi and two reserved floats.
    pub angles: [f32; 4],
}

impl BufferElement for CloudPoint {
    type Gpu = Self;
    fn to_gpu(&self) -> Self {
        *self
    }
    fn from_gpu(value: &Self) -> Self {
        *value
    }
}

impl CloudPoint {
    pub fn new(position: Vec3, sensor_id: u64, sample_id: u64) -> Self {
        Self {
            position: [position.x as f32, position.y as f32, position.z as f32, 1.0],
            normal: [0.0, 1.0, 0.0, 0.0],
            ids: [
                sensor_id as u32,
                (sensor_id >> 32) as u32,
                sample_id as u32,
                (sample_id >> 32) as u32,
            ],
            extra: [u32::MAX, u32::MAX, 1, 0],
            ..Default::default()
        }
    }
    pub fn sensor_id(&self) -> u64 {
        self.ids[0] as u64 | ((self.ids[1] as u64) << 32)
    }
    pub fn sample_id(&self) -> u64 {
        self.ids[2] as u64 | ((self.ids[3] as u64) << 32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CloudRaster {
    pub sensor_id: u64,
    pub rows: u32,
    pub columns: u32,
    pub subrays: u32,
    pub source_offset: u32,
    pub angles: Option<[(Real, Real); 2]>,
}

pub enum CloudInput<'a> {
    Points(ReadbackSource<'a>),
    /// Channels: point, normal, actual direction, distance, signed shape ID.
    Rays {
        channels: [ReadbackSource<'a>; 5],
        raster: CloudRaster,
    },
}

impl<'a> CloudInput<'a> {
    pub fn from_lidar(batch: &LidarDeviceBatch<'a>, scan: usize) -> Result<Self> {
        let range = batch
            .scans
            .get(scan)
            .ok_or_else(|| Error::Solver("cloud: unknown LiDAR scan".into()))?;
        let (rows, columns, angles) = match range.scan.pattern {
            LidarPattern::Terrestrial(meta) => (
                meta.n_theta,
                meta.n_phi,
                Some([meta.theta_range, meta.phi_range]),
            ),
            LidarPattern::Aerial(_) => (1, range.pulse_count, None),
        };
        let start = range.ray_offset as usize;
        let end = start
            .checked_add(range.ray_count as usize)
            .ok_or_else(|| Error::Solver("cloud: ray range overflow".into()))?;
        if end > batch.hits.ray_count {
            return Err(Error::Solver("cloud: ray range exceeds active hits".into()));
        }
        Ok(Self::Rays {
            channels: [
                batch.hits.point.readback_source(start..end)?,
                batch.hits.normal.readback_source(start..end)?,
                batch.hits.directions.readback_source(start..end)?,
                batch.hits.distance.readback_source(start..end)?,
                batch.hits.shape.readback_source(start..end)?,
            ],
            raster: CloudRaster {
                sensor_id: range.scan.sensor_id,
                rows,
                columns,
                subrays: range.subray_count,
                source_offset: range.ray_offset,
                angles,
            },
        })
    }

    fn sources(&self) -> &[ReadbackSource<'a>] {
        match self {
            Self::Points(source) => std::slice::from_ref(source),
            Self::Rays { channels, .. } => channels,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CloudFilter {
    pub distance: Option<(Vec3, Real)>,
    pub bounds: Option<(Vec3, Vec3, bool)>,
    pub min_intensity: Option<f32>,
    pub scalar: Option<(Compare, f32)>,
}

pub struct CloudJob<'a> {
    pub input: CloudInput<'a>,
    pub transform: Transform,
    pub filter: CloudFilter,
    pub stamp: ReadbackStamp,
}

pub struct PendingCloudSource<'a> {
    pub frame: &'a EncodedSensorFrame,
    pub channel: usize,
    /// Relative to the source channel's active range.
    pub elements: Range<usize>,
}

pub enum PendingCloudInput<'a> {
    Points(PendingCloudSource<'a>),
    /// Channels: point, normal, actual direction, distance, signed shape ID.
    Rays {
        channels: [PendingCloudSource<'a>; 5],
        raster: CloudRaster,
    },
}

impl<'a> PendingCloudInput<'a> {
    pub fn from_lidar(batch: &'a EncodedLidarBatch, scan: usize) -> Result<Self> {
        let range = batch
            .scans
            .get(scan)
            .ok_or_else(|| Error::Solver("cloud: unknown LiDAR scan".into()))?;
        let start = range.ray_offset as usize;
        let end = start
            .checked_add(range.ray_count as usize)
            .filter(|end| *end <= batch.ray_count)
            .ok_or_else(|| Error::Solver("cloud: ray range exceeds active hits".into()))?;
        let (rows, columns, angles) = match range.scan.pattern {
            LidarPattern::Terrestrial(meta) => (
                meta.n_theta,
                meta.n_phi,
                Some([meta.theta_range, meta.phi_range]),
            ),
            LidarPattern::Aerial(_) => (1, range.pulse_count, None),
        };
        Ok(Self::Rays {
            channels: [4, 5, 1, 2, 3].map(|channel| PendingCloudSource {
                frame: &batch.frame,
                channel,
                elements: start..end,
            }),
            raster: CloudRaster {
                sensor_id: range.scan.sensor_id,
                rows,
                columns,
                subrays: range.subray_count,
                source_offset: range.ray_offset,
                angles,
            },
        })
    }

    fn sources(&self) -> &[PendingCloudSource<'_>] {
        match self {
            Self::Points(source) => std::slice::from_ref(source),
            Self::Rays { channels, .. } => channels,
        }
    }
}

pub struct PendingCloudJob<'a> {
    pub input: PendingCloudInput<'a>,
    pub transform: Transform,
    pub filter: CloudFilter,
    pub stamp: ReadbackStamp,
}

/// Channels: compacted CloudPoint records and u32 count/summary words.
pub struct EncodedCloud {
    pub frame: EncodedSensorFrame,
    pub layouts: Vec<CloudLayout>,
    /// Upper bound; the valid prefix length remains on device in count word zero.
    pub input_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct CloudProcessorConfig {
    pub max_points: u32,
    pub max_jobs: u32,
    pub max_buffer_bytes: u64,
}

impl Default for CloudProcessorConfig {
    fn default() -> Self {
        Self {
            max_points: 65_536,
            max_jobs: 64,
            max_buffer_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CloudProcessingStats {
    pub frames: u64,
    pub encoded_frames: u64,
    pub submissions: u64,
    pub copied_bytes: u64,
    pub kernel_dispatches: u64,
    pub buffer_allocations: u64,
    pub buffer_bytes: u64,
    pub binding_preparations: u64,
}

#[derive(Debug, Clone)]
pub struct CloudLayout {
    pub stamp: ReadbackStamp,
    pub input_range: Range<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloudSummary {
    pub start: u32,
    pub count: u32,
    pub misses: u32,
    pub invalid: u32,
    pub filtered: u32,
}

impl CloudSummary {
    pub fn from_words(words: &[u32]) -> Result<Self> {
        let [start, count, misses, invalid, filtered] = words else {
            return Err(Error::Solver("cloud: summary needs five words".into()));
        };
        Ok(Self {
            start: *start,
            count: *count,
            misses: *misses,
            invalid: *invalid,
            filtered: *filtered,
        })
    }
}

pub struct CloudFrame<'a> {
    processor: &'a CloudProcessor,
    input_count: usize,
}

impl<'a> CloudFrame<'a> {
    pub fn layouts(&self) -> &[CloudLayout] {
        &self.processor.layouts
    }
    pub fn input_count(&self) -> usize {
        self.input_count
    }
    pub fn count_source(&self) -> Result<ReadbackSource<'a>> {
        self.processor.counts.readback_source(0..1)
    }
    pub fn summary_source(&self, job: usize) -> Result<ReadbackSource<'a>> {
        if job >= self.processor.layouts.len() {
            return Err(Error::Solver("cloud: unknown job".into()));
        }
        self.processor
            .counts
            .readback_source(1 + job * 5..1 + (job + 1) * 5)
    }
    /// Only the GPU-counted prefix is valid; the remaining active input extent is zeroed.
    pub fn points_source(&self, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        if range.start > range.end || range.end > self.input_count {
            return Err(Error::Solver(
                "cloud: output range exceeds active input".into(),
            ));
        }
        self.processor.output.readback_source(range)
    }
}

pub struct CloudProcessor {
    storage_usage: ProcessorStorageUsage,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: CloudProcessorConfig,
    raw: wgpu::Buffer,
    work: DeviceBuffer<CloudPoint>,
    output: DeviceBuffer<CloudPoint>,
    descriptors: wgpu::Buffer,
    counts: DeviceBuffer<u32>,
    kernel: CloudProcessKernel,
    direct: BindingsCache<CloudProcessBindings>,
    layouts: Vec<CloudLayout>,
    stats: CloudProcessingStats,
}

impl CloudProcessor {
    /// Requested construction payload, excluding driver staging, metadata and output pools.
    pub fn storage_usage(&self) -> ProcessorStorageUsage {
        self.storage_usage
    }

    pub fn new(engine: &Engine, config: CloudProcessorConfig) -> Result<Self> {
        Self::new_with_storage_limits(engine, config, ProcessorStorageLimits::default())
    }

    /// Validate device budgets and combined construction payload before allocating arrays.
    pub fn new_with_storage_limits(
        engine: &Engine,
        config: CloudProcessorConfig,
        limits: ProcessorStorageLimits,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("cloud requires GPU".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("cloud requires queue".into()))?
            .clone();
        let storage_usage = ProcessorStorageUsage::for_cloud(config, &device.limits())?;
        limits.check(storage_usage)?;
        let bytes = storage_usage.device_bytes;
        let work = DeviceBuffer::from_host(
            engine,
            filled(config.max_points as usize, CloudPoint::default())?,
            BufferUsage::StorageReadWrite,
        )?;
        let output = DeviceBuffer::from_host(
            engine,
            filled(config.max_points as usize, CloudPoint::default())?,
            BufferUsage::StorageReadWrite,
        )?;
        let counts = DeviceBuffer::from_host(
            engine,
            filled(1 + config.max_jobs as usize * 5, 0u32)?,
            BufferUsage::StorageReadWrite,
        )?;
        let buffer = |size, label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let raw = buffer(config.max_points as u64 * 56, "molla::cloud ray channels");
        let descriptors = buffer(config.max_jobs as u64 * 160, "molla::cloud jobs");
        let kernel = CloudProcessKernel::new(
            &device,
            [
                &raw,
                work.device_buffer()?,
                output.device_buffer()?,
                &descriptors,
                counts.device_buffer()?,
            ],
            config.max_points,
            config.max_jobs,
        )?;
        Ok(Self {
            storage_usage,
            device,
            queue,
            config,
            raw,
            work,
            output,
            descriptors,
            counts,
            kernel,
            direct: BindingsCache::with_capacity(128),
            layouts: Vec::new(),
            stats: CloudProcessingStats {
                buffer_allocations: 8,
                buffer_bytes: bytes,
                binding_preparations: 1,
                ..Default::default()
            },
        })
    }

    pub fn stats(&self) -> CloudProcessingStats {
        self.stats
    }

    fn output_specs(&self) -> [SensorChannelSpec; 2] {
        [
            SensorChannelSpec::of::<CloudPoint>(self.config.max_points as usize),
            SensorChannelSpec::of::<u32>(1 + self.config.max_jobs as usize * 5),
        ]
    }

    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        self.output.check_device()?;
        engine.check_device()?;
        if !Arc::ptr_eq(&self.device, engine.device().expect("checked engine")) {
            return Err(Error::Gpu("cloud: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Encode external inputs into leased compacted points and device counts.
    /// Sources share the command batch's 64-external-input bound.
    ///
    /// ```compile_fail
    /// use molla_sensors::{CloudJob, CloudProcessor, SensorCommandBatch, SensorFramePool};
    /// fn reuse<'a>(p: &'a mut CloudProcessor, b: &mut SensorCommandBatch<'a>, jobs: &'a [CloudJob<'a>], pool: &mut SensorFramePool) {
    ///     p.encode(b, jobs, pool, Default::default()).unwrap();
    ///     p.process(&[]).unwrap();
    ///     b.stats();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_math::Transform;
    /// use molla_sensors::{CloudInput, CloudJob, CloudPoint, CloudProcessor, SensorCommandBatch, SensorFramePool};
    /// use molla_sensors::readback::DeviceBuffer;
    /// fn mutate(engine: &Engine, p: &mut CloudProcessor, input: &mut DeviceBuffer<CloudPoint>, pool: &mut SensorFramePool) {
    ///     let jobs = [CloudJob { input: CloudInput::Points(input.readback_source(0..1).unwrap()), transform: Transform::IDENTITY, filter: Default::default(), stamp: Default::default() }];
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     p.encode(&mut batch, &jobs, pool, Default::default()).unwrap();
    ///     input.host_mut().unwrap()[0] = CloudPoint::default();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    pub fn encode<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        jobs: &'a [CloudJob<'a>],
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedCloud>> {
        self.output.check_device()?;
        batch.check_device(&self.device)?;
        check_stamp(stamp)?;
        if jobs.len() > self.config.max_jobs as usize {
            return Err(Error::Solver("cloud: job budget exceeded".into()));
        }
        batch.check_source_capacity(jobs.iter().map(|job| job.input.sources().len()).sum())?;
        let (descriptors, layouts, count) = self.plan(jobs)?;
        let specs = self.output_specs();
        let ranges = [0..count as usize, 0..1 + jobs.len() * 5];
        let mut copied = 0;
        let frame =
            batch.encode_output(pool, &specs, &ranges, stamp, |_, _, encoder, outputs| {
                for (job, layout) in jobs.iter().zip(&layouts) {
                    let mut offset = layout.input_range.start as u64 * 56;
                    for source in job.input.sources() {
                        if source.channel.bytes != 0 {
                            let (target, target_offset) = match job.input {
                                CloudInput::Points(_) => (
                                    self.work.device_buffer()?,
                                    layout.input_range.start as u64 * 112,
                                ),
                                CloudInput::Rays { .. } => (&self.raw, offset),
                            };
                            encoder.copy_buffer_to_buffer(
                                source.buffer.expect("validated cloud source"),
                                source.offset,
                                target,
                                target_offset,
                                source.channel.bytes as u64,
                            );
                            copied += source.channel.bytes as u64;
                        }
                        offset += source.channel.bytes as u64;
                    }
                }
                self.encode_direct(encoder, outputs, &descriptors, count)
            })?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        for job in jobs {
            for source in job.input.sources() {
                batch.retain_source(source)?;
            }
        }
        self.stats.encoded_frames += 1;
        self.stats.copied_bytes += copied;
        Ok(Some(EncodedCloud {
            frame,
            layouts,
            input_count: count as usize,
        }))
    }

    /// Encode up to 1024 jobs from earlier frames, deduplicating their dependencies.
    ///
    /// ```compile_fail
    /// use molla_sensors::{CloudProcessor, PendingCloudJob, SensorCommandBatch, SensorFramePool};
    /// fn reuse<'a>(p: &'a mut CloudProcessor, b: &mut SensorCommandBatch<'a>, jobs: &[PendingCloudJob<'_>], pool: &mut SensorFramePool) {
    ///     p.encode_pending(b, jobs, pool, Default::default()).unwrap();
    ///     p.process(&[]).unwrap();
    ///     b.stats();
    /// }
    /// ```
    pub fn encode_pending<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        jobs: &[PendingCloudJob<'_>],
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedCloud>> {
        self.output.check_device()?;
        batch.check_device(&self.device)?;
        check_stamp(stamp)?;
        if jobs.len() > self.config.max_jobs as usize {
            return Err(Error::Solver("cloud: job budget exceeded".into()));
        }
        let mut receipts: Vec<&EncodedSensorFrame> = Vec::with_capacity(64);
        let mut indices = Vec::with_capacity(jobs.len() * 5);
        let mut metadata = Vec::with_capacity(jobs.len());
        for job in jobs {
            let mut lengths = [0u32; 5];
            for (i, source) in job.input.sources().iter().enumerate() {
                let range = match (&job.input, i) {
                    (PendingCloudInput::Points(_), _) => batch.pending_range::<CloudPoint>(
                        source.frame,
                        source.channel,
                        source.elements.clone(),
                    )?,
                    (_, 0..=2) => batch.pending_range::<Vec3>(
                        source.frame,
                        source.channel,
                        source.elements.clone(),
                    )?,
                    (_, 3) => batch.pending_range::<Real>(
                        source.frame,
                        source.channel,
                        source.elements.clone(),
                    )?,
                    _ => batch.pending_range::<i32>(
                        source.frame,
                        source.channel,
                        source.elements.clone(),
                    )?,
                };
                lengths[i] = u32::try_from(range.len())
                    .map_err(|_| Error::Solver("cloud: channel count exceeds u32".into()))?;
                let index = if let Some(index) = receipts
                    .iter()
                    .position(|frame| frame.same_receipt(source.frame))
                {
                    index
                } else {
                    if receipts.len() == 64 {
                        return Err(Error::Solver(
                            "cloud: pending dependency budget exceeded".into(),
                        ));
                    }
                    receipts.push(source.frame);
                    receipts.len() - 1
                };
                indices.push(index);
            }
            let raster = match job.input {
                PendingCloudInput::Points(_) => None,
                PendingCloudInput::Rays { raster, .. } => {
                    if lengths.iter().any(|n| *n != lengths[0]) {
                        return Err(Error::Solver("cloud: inconsistent ray channels".into()));
                    }
                    Some(raster)
                }
            };
            metadata.push((lengths[0], raster, job.transform, job.filter, job.stamp));
        }
        let (descriptors, layouts, count) = self.plan_metadata(metadata.into_iter().map(Ok))?;
        let specs = self.output_specs();
        let ranges = [0..count as usize, 0..1 + jobs.len() * 5];
        let mut copied = 0;
        let frame = if jobs.is_empty() {
            batch.encode_output(pool, &specs, &ranges, stamp, |_, _, _, _| Ok(0))?
        } else {
            batch.encode_from_pending(
                &receipts,
                pool,
                &specs,
                &ranges,
                stamp,
                |encoder, inputs, outputs| {
                    let mut indices = indices.iter();
                    for (job, layout) in jobs.iter().zip(&layouts) {
                        let mut offset = layout.input_range.start as u64 * 56;
                        for (i, source) in job.input.sources().iter().enumerate() {
                            let index = *indices.next().expect("validated cloud dependency");
                            let bytes = match (&job.input, i) {
                                (PendingCloudInput::Points(_), _) => copy_pending::<CloudPoint>(
                                    encoder,
                                    &inputs,
                                    index,
                                    source,
                                    self.work.device_buffer()?,
                                    layout.input_range.start as u64 * 112,
                                )?,
                                (_, 0..=2) => copy_pending::<Vec3>(
                                    encoder, &inputs, index, source, &self.raw, offset,
                                )?,
                                (_, 3) => copy_pending::<Real>(
                                    encoder, &inputs, index, source, &self.raw, offset,
                                )?,
                                _ => copy_pending::<i32>(
                                    encoder, &inputs, index, source, &self.raw, offset,
                                )?,
                            };
                            offset += bytes;
                            copied += bytes;
                        }
                    }
                    self.encode_direct(encoder, outputs, &descriptors, count)
                },
            )?
        };
        let Some(frame) = frame else {
            return Ok(None);
        };
        self.stats.encoded_frames += 1;
        self.stats.copied_bytes += copied;
        Ok(Some(EncodedCloud {
            frame,
            layouts,
            input_count: count as usize,
        }))
    }

    fn encode_direct(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        outputs: &[wgpu::Buffer],
        descriptors: &[CloudDescriptor],
        count: u32,
    ) -> Result<u64> {
        if count == 0 {
            return Ok(0);
        }
        self.queue
            .write_buffer(&self.descriptors, 0, bytemuck::cast_slice(descriptors));
        let buffers = [
            &self.raw,
            self.work.device_buffer()?,
            &outputs[0],
            &self.descriptors,
            &outputs[1],
        ];
        let (bindings, prepared) = self.direct.try_get_or_prepare(
            |bindings| bindings.matches(buffers),
            || self.kernel.prepare(&self.device, buffers),
        )?;
        self.stats.binding_preparations += u64::from(prepared);
        self.kernel.encode_prepared(
            &self.queue,
            encoder,
            bindings,
            count,
            descriptors.len() as u32,
        )?;
        self.stats.kernel_dispatches += 3;
        Ok(3)
    }

    pub fn process(&mut self, jobs: &[CloudJob<'_>]) -> Result<CloudFrame<'_>> {
        self.output.check_device()?;
        let (descriptors, layouts, count) = self.plan(jobs)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::cloud pipeline"),
            });
        encoder.clear_buffer(self.counts.device_buffer()?, 0, None);
        let mut bytes = 0;
        if count != 0 {
            encoder.clear_buffer(self.output.device_buffer()?, 0, Some(count as u64 * 112));
            self.queue
                .write_buffer(&self.descriptors, 0, bytemuck::cast_slice(&descriptors));
            for (job, layout) in jobs.iter().zip(&layouts) {
                let mut offset = layout.input_range.start as u64 * 56;
                for source in job.input.sources() {
                    if let Some(buffer) = source.buffer {
                        let (target, target_offset) = match job.input {
                            CloudInput::Points(_) => (
                                self.work.device_buffer()?,
                                layout.input_range.start as u64 * 112,
                            ),
                            CloudInput::Rays { .. } => (&self.raw, offset),
                        };
                        encoder.copy_buffer_to_buffer(
                            buffer,
                            source.offset,
                            target,
                            target_offset,
                            source.channel.bytes as u64,
                        );
                        bytes += source.channel.bytes as u64;
                    }
                    offset += source.channel.bytes as u64;
                }
            }
            self.kernel
                .encode(&self.queue, &mut encoder, count, jobs.len() as u32)?;
        }
        self.queue.submit([encoder.finish()]);
        let leases: Vec<BufferLease> = jobs
            .iter()
            .flat_map(|job| job.input.sources())
            .filter_map(|source| source.lease.clone())
            .collect();
        if !leases.is_empty() {
            self.queue.on_submitted_work_done(move || drop(leases));
        }
        self.counts.mark_device_dirty();
        if count != 0 {
            self.work.mark_device_dirty();
            self.output.mark_device_dirty();
        }
        self.layouts = layouts;
        self.stats.frames += 1;
        self.stats.submissions += 1;
        self.stats.copied_bytes += bytes;
        self.stats.kernel_dispatches += if count == 0 { 0 } else { 3 };
        Ok(CloudFrame {
            processor: self,
            input_count: count as usize,
        })
    }

    fn plan(&self, jobs: &[CloudJob<'_>]) -> Result<(Vec<CloudDescriptor>, Vec<CloudLayout>, u32)> {
        if jobs.len() > self.config.max_jobs as usize {
            return Err(Error::Solver("cloud: job budget exceeded".into()));
        }
        self.plan_metadata(jobs.iter().map(|job| {
            let (count, raster) = match &job.input {
                CloudInput::Points(source) => (self.check_source::<CloudPoint>(source)?, None),
                CloudInput::Rays { channels, raster } => {
                    let counts = [
                        self.check_source::<Vec3>(&channels[0])?,
                        self.check_source::<Vec3>(&channels[1])?,
                        self.check_source::<Vec3>(&channels[2])?,
                        self.check_source::<Real>(&channels[3])?,
                        self.check_source::<i32>(&channels[4])?,
                    ];
                    if counts.iter().any(|n| *n != counts[0]) {
                        return Err(Error::Solver("cloud: inconsistent ray channels".into()));
                    }
                    (counts[0], Some(*raster))
                }
            };
            Ok((count, raster, job.transform, job.filter, job.stamp))
        }))
    }

    fn plan_metadata(
        &self,
        jobs: impl ExactSizeIterator<
            Item = Result<(
                u32,
                Option<CloudRaster>,
                Transform,
                CloudFilter,
                ReadbackStamp,
            )>,
        >,
    ) -> Result<(Vec<CloudDescriptor>, Vec<CloudLayout>, u32)> {
        if jobs.len() > self.config.max_jobs as usize {
            return Err(Error::Solver("cloud: job budget exceeded".into()));
        }
        let mut descriptors = Vec::with_capacity(jobs.len());
        let mut layouts = Vec::with_capacity(jobs.len());
        let mut total = 0u32;
        for metadata in jobs {
            let (count, raster, transform, filter, stamp) = metadata?;
            let mut descriptor = describe_metadata(transform, filter, stamp)?;
            if let Some(raster) = raster {
                let expected = raster
                    .rows
                    .checked_mul(raster.columns)
                    .and_then(|n| n.checked_mul(raster.subrays))
                    .ok_or_else(|| Error::Solver("cloud: raster count overflow".into()))?;
                if raster.rows == 0
                    || raster.columns == 0
                    || raster.subrays == 0
                    || count != expected
                    || raster.sensor_id != stamp.sensor_id
                    || raster.source_offset.checked_add(count).is_none()
                {
                    return Err(Error::Solver(
                        "cloud: inconsistent ray channels or raster identity".into(),
                    ));
                }
                descriptor.counts[2] = if raster.angles.is_some() { 1 } else { 2 };
                descriptor.grid = [
                    raster.columns,
                    raster.rows,
                    raster.subrays,
                    raster.source_offset,
                ];
                if let Some([(t0, t1), (p0, p1)]) = raster.angles {
                    descriptor.angles = [
                        checked_real(t0)?,
                        checked_real(t1)?,
                        checked_real(p0)?,
                        checked_real(p1)?,
                    ];
                    if !(descriptor.angles[1] - descriptor.angles[0]).is_finite()
                        || !(descriptor.angles[3] - descriptor.angles[2]).is_finite()
                    {
                        return Err(Error::Solver("cloud: angular span exceeds f32".into()));
                    }
                }
            }
            let end = total
                .checked_add(count)
                .ok_or_else(|| Error::Solver("cloud: point count overflow".into()))?;
            if end > self.config.max_points {
                return Err(Error::Solver("cloud: point budget exceeded".into()));
            }
            descriptor.counts[0] = total;
            descriptor.counts[1] = count;
            descriptors.push(descriptor);
            layouts.push(CloudLayout {
                stamp,
                input_range: total as usize..end as usize,
            });
            total = end;
        }
        Ok((descriptors, layouts, total))
    }

    fn check_source<T: BufferElement>(&self, source: &ReadbackSource<'_>) -> Result<u32> {
        if !Arc::ptr_eq(&self.device, &source.device) {
            return Err(Error::Gpu("cloud: foreign source".into()));
        }
        if source.channel.kind != TypeId::of::<T>()
            || source.channel.element_size != std::mem::size_of::<T::Gpu>()
            || !source
                .channel
                .bytes
                .is_multiple_of(std::mem::size_of::<T::Gpu>())
            || (source.channel.bytes != 0 && source.buffer.is_none())
        {
            return Err(Error::Solver("cloud: wrong channel type".into()));
        }
        u32::try_from(source.channel.bytes / source.channel.element_size)
            .map_err(|_| Error::Solver("cloud: channel count exceeds u32".into()))
    }
}

fn copy_pending<T: BufferElement>(
    encoder: &mut wgpu::CommandEncoder,
    inputs: &PendingSensorInputs<'_>,
    index: usize,
    source: &PendingCloudSource<'_>,
    target: &wgpu::Buffer,
    offset: u64,
) -> Result<u64> {
    let channel = inputs.channel::<T>(index, source.channel)?;
    let view = channel.slice(source.elements.clone())?;
    let stride = std::mem::size_of::<T::Gpu>() as u64;
    let bytes = view.len() as u64 * stride;
    if bytes != 0 {
        encoder.copy_buffer_to_buffer(
            view.binding()?.buffer,
            view.active_range().start as u64 * stride,
            target,
            offset,
            bytes,
        );
    }
    Ok(bytes)
}

fn checked_real(value: Real) -> Result<f32> {
    if value.is_finite() && (value as f32).is_finite() {
        Ok(value as f32)
    } else {
        Err(Error::Solver(
            "cloud: nonfinite or unrepresentable value".into(),
        ))
    }
}
fn checked_vec(value: Vec3) -> Result<[f32; 4]> {
    Ok([
        checked_real(value.x)?,
        checked_real(value.y)?,
        checked_real(value.z)?,
        0.0,
    ])
}

fn describe_metadata(
    transform: Transform,
    filter: CloudFilter,
    stamp: ReadbackStamp,
) -> Result<CloudDescriptor> {
    check_stamp(stamp)?;
    let q = transform.rotation;
    if !q.is_finite() || (q.length_squared() - 1.0).abs() > 1e-6 {
        return Err(Error::Solver(
            "cloud: invalid rotation or sample time".into(),
        ));
    }
    let mut d = CloudDescriptor {
        translation: checked_vec(transform.position)?,
        rotation: q.as_quat().to_array(),
        ids: [
            stamp.sensor_id as u32,
            (stamp.sensor_id >> 32) as u32,
            stamp.sample_id as u32,
            (stamp.sample_id >> 32) as u32,
        ],
        ..Default::default()
    };
    if let Some((origin, radius)) = filter.distance {
        d.distance = checked_vec(origin)?;
        d.distance[3] = checked_real(radius)?;
        if radius < 0.0 {
            return Err(Error::Solver("cloud: negative distance limit".into()));
        }
        d.counts[3] |= 1;
    }
    if let Some((min, max, inside)) = filter.bounds {
        d.bounds_min = checked_vec(min)?;
        d.bounds_max = checked_vec(max)?;
        if min.cmpgt(max).any() {
            return Err(Error::Solver("cloud: reversed bounds".into()));
        }
        d.counts[3] |= 2 | if inside { 4 } else { 0 };
    }
    if let Some(min) = filter.min_intensity {
        d.translation[3] = checked_real(min as Real)?;
        d.counts[3] |= 8;
    }
    if let Some((op, threshold)) = filter.scalar {
        d.bounds_min[3] = checked_real(threshold as Real)?;
        d.misc[0] = match op {
            Compare::Lt => 0,
            Compare::Le => 1,
            Compare::Gt => 2,
            Compare::Ge => 3,
            Compare::Eq => 4,
        };
        d.counts[3] |= 16;
    }
    Ok(d)
}

fn check_stamp(stamp: ReadbackStamp) -> Result<()> {
    if stamp
        .simulation_time
        .is_some_and(|time| !time.is_finite() || time < 0.0)
    {
        return Err(Error::Solver("cloud: invalid sample time".into()));
    }
    Ok(())
}

pub(crate) fn checked_budget(config: CloudProcessorConfig, limits: &wgpu::Limits) -> Result<u64> {
    if config.max_points == 0
        || config.max_jobs == 0
        || config.max_jobs > 1024
        || config.max_points.max(config.max_jobs).div_ceil(256)
            > limits.max_compute_workgroups_per_dimension
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_storage_buffers_per_shader_stage < 7
        || limits.max_compute_workgroup_storage_size < 1024
    {
        return Err(Error::Solver(
            "cloud: invalid configuration or device capacity".into(),
        ));
    }
    let n = config.max_points as u64;
    let buffers = [
        n * 56,
        n * 112,
        n * 112,
        n * 8,
        n.div_ceil(256) * 8,
        config.max_jobs as u64 * 160,
        (1 + config.max_jobs as u64 * 5) * 4,
        16,
    ];
    let bytes: u64 = buffers.iter().sum();
    if bytes > config.max_buffer_bytes
        || buffers.into_iter().any(|size| {
            size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size as u64
        })
    {
        return Err(Error::Solver(
            "cloud: device or buffer byte budget exceeded".into(),
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires GPU"]
    fn construction_budget_matches_host_arrays_and_upload_payload() {
        use crate::sensor_storage::tests::buffer_bytes;
        let engine = Engine::new_gpu().unwrap();
        let config = CloudProcessorConfig {
            max_points: 257,
            max_jobs: 3,
            max_buffer_bytes: u64::MAX,
        };
        let usage =
            ProcessorStorageUsage::for_cloud(config, &engine.device().unwrap().limits()).unwrap();
        assert!(
            CloudProcessor::new_with_storage_limits(
                &engine,
                config,
                ProcessorStorageLimits {
                    max_construction_bytes: usage.construction_bytes - 1,
                }
            )
            .is_err()
        );
        let processor = CloudProcessor::new_with_storage_limits(
            &engine,
            config,
            ProcessorStorageLimits {
                max_construction_bytes: usage.construction_bytes,
            },
        )
        .unwrap();
        assert_eq!(processor.storage_usage(), usage);
        assert_eq!(processor.stats().buffer_bytes, usage.device_bytes);
        crate::processor_storage::tests::check_live(
            usage,
            &[
                buffer_bytes(&processor.work.buffer),
                buffer_bytes(&processor.output.buffer),
                buffer_bytes(&processor.counts.buffer),
            ],
        );
    }

    #[test]
    fn layouts_and_exact_buffer_budget_match_shader_strides() {
        assert_eq!(std::mem::size_of::<CloudPoint>(), 112);
        assert_eq!(std::mem::size_of::<CloudDescriptor>(), 160);
        let point = CloudPoint::new(Vec3::ONE, u64::MAX, 0x123456789abcdef0);
        assert_eq!(point.sensor_id(), u64::MAX);
        assert_eq!(point.sample_id(), 0x123456789abcdef0);
        assert_eq!(CloudPoint::from_gpu(&point.to_gpu()), point);
        let config = CloudProcessorConfig {
            max_points: 257,
            max_jobs: 3,
            max_buffer_bytes: u64::MAX,
        };
        let limits = wgpu::Limits::default();
        let bytes = 257 * 288 + 2 * 8 + 3 * 160 + 16 * 4 + 16;
        assert_eq!(checked_budget(config, &limits).unwrap(), bytes);
        assert!(
            checked_budget(
                CloudProcessorConfig {
                    max_buffer_bytes: bytes - 1,
                    ..config
                },
                &limits
            )
            .is_err()
        );
        assert!(
            checked_budget(
                CloudProcessorConfig {
                    max_buffer_bytes: bytes,
                    ..config
                },
                &limits
            )
            .is_ok()
        );
        assert!(CloudSummary::from_words(&[1, 2]).is_err());
        assert_eq!(
            CloudSummary::from_words(&[1, 2, 3, 4, 5]).unwrap(),
            CloudSummary {
                start: 1,
                count: 2,
                misses: 3,
                invalid: 4,
                filtered: 5
            }
        );
    }

    #[test]
    fn invalid_device_and_application_budgets_are_rejected_before_allocation() {
        let limits = wgpu::Limits::default();
        let base = CloudProcessorConfig::default();
        for config in [
            CloudProcessorConfig {
                max_points: 0,
                ..base
            },
            CloudProcessorConfig {
                max_points: u32::MAX,
                ..base
            },
            CloudProcessorConfig {
                max_jobs: 0,
                ..base
            },
            CloudProcessorConfig {
                max_jobs: 1025,
                ..base
            },
        ] {
            assert!(checked_budget(config, &limits).is_err());
        }
        for limits in [
            wgpu::Limits {
                max_compute_workgroups_per_dimension: 1,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_invocations_per_workgroup: 128,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_workgroup_size_x: 128,
                ..limits.clone()
            },
            wgpu::Limits {
                max_compute_workgroup_storage_size: 512,
                ..limits.clone()
            },
            wgpu::Limits {
                max_storage_buffers_per_shader_stage: 6,
                ..limits.clone()
            },
            wgpu::Limits {
                max_buffer_size: 16,
                ..limits.clone()
            },
            wgpu::Limits {
                max_storage_buffer_binding_size: 16,
                ..limits.clone()
            },
        ] {
            assert!(checked_budget(base, &limits).is_err());
        }
        let small = CloudProcessorConfig {
            max_points: 1,
            max_jobs: 1024,
            ..base
        };
        assert!(checked_budget(small, &limits).is_ok());
        assert!(
            checked_budget(
                small,
                &wgpu::Limits {
                    max_compute_workgroups_per_dimension: 1,
                    ..limits
                }
            )
            .is_err()
        );
    }
}
