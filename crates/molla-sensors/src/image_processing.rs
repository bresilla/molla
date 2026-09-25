//! Bounded device-only packed-image processing with independent per-image pipelines.

use crate::device_results::{
    EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
    SensorFramePoolConfig,
};
use crate::dispatch::BindingsCache;
use crate::host_arrays::filled;
use crate::readback::{BufferLease, DeviceBuffer, ReadbackSource, ReadbackStamp};
use crate::{ProcessorStorageLimits, ProcessorStorageUsage};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{ImageProcessBindings, ImageProcessKernel};
use molla_math::Vec3;
use molla_storage::BufferUsage;
use std::{any::TypeId, collections::HashSet, ops::Range, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageOp {
    Exposure(f32),
    Gamma(f32),
    LinearToSrgb,
    SrgbToLinear,
    Reinhard,
    ReinhardWhitePoint(f32),
    BrightnessContrast { brightness: f32, contrast: f32 },
    Saturation(f32),
    GrayWorld,
    WhitePatch,
}

impl ImageOp {
    fn parameters(self) -> Result<(u32, f32, f32)> {
        let (op, a, b) = match self {
            Self::Exposure(v) => (0, v, 0.0),
            Self::Gamma(v) => {
                finite(v)?;
                (1, 1.0 / v.max(1e-6), 0.0)
            }
            Self::LinearToSrgb => (2, 0.0, 0.0),
            Self::SrgbToLinear => (3, 0.0, 0.0),
            Self::Reinhard => (4, 0.0, 0.0),
            Self::ReinhardWhitePoint(v) => {
                finite(v)?;
                (5, 1.0 / v.max(1e-6).powi(2), 0.0)
            }
            Self::BrightnessContrast {
                brightness,
                contrast,
            } => {
                if brightness.abs() as f64 + 0.5 * contrast.abs() as f64 + 0.5 > f32::MAX as f64 {
                    return Err(Error::Solver(
                        "image processing: brightness/contrast can overflow".into(),
                    ));
                }
                (6, brightness, contrast)
            }
            Self::Saturation(v) => (7, v, 0.0),
            Self::GrayWorld => (8, 0.0, 0.0),
            Self::WhitePatch => (9, 0.0, 0.0),
        };
        finite(a)?;
        finite(b)?;
        Ok((op, a, b))
    }

    fn is_white_balance(self) -> bool {
        matches!(self, Self::GrayWorld | Self::WhitePatch)
    }
}

fn finite(v: f32) -> Result<()> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(Error::Solver(
            "image processing: nonfinite parameter".into(),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImageProcessorConfig {
    pub max_pixels: u32,
    pub max_images: u32,
    pub max_operations_per_image: u32,
    /// Application buffer bytes; excludes host metadata, pipelines and driver staging.
    pub max_buffer_bytes: u64,
}

impl Default for ImageProcessorConfig {
    fn default() -> Self {
        Self {
            max_pixels: 4_000_000,
            max_images: 16,
            max_operations_per_image: 16,
            max_buffer_bytes: 32 * 1024 * 1024,
        }
    }
}

/// One packed RGBA8 image. Every operation quantizes back to RGBA8; alpha is preserved.
pub struct ImageJob<'a> {
    pub source: ReadbackSource<'a>,
    pub operations: &'a [ImageOp],
    pub stamp: ReadbackStamp,
}

pub struct PendingImageJob<'a> {
    pub frame: &'a EncodedSensorFrame,
    pub channel: usize,
    /// Pixel indices relative to the pending channel's active range.
    pub pixels: Range<usize>,
    pub operations: &'a [ImageOp],
    pub stamp: ReadbackStamp,
}

pub struct EncodedProcessedImages {
    pub frame: EncodedSensorFrame,
    pub layouts: Vec<ProcessedImageLayout>,
}

#[derive(Debug, Clone)]
pub struct ProcessedImageLayout {
    pub stamp: ReadbackStamp,
    pub pixels: Range<usize>,
    pub stages: Range<usize>,
    pub white_balance_stages: Vec<usize>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ImageProcessingStats {
    pub frames: u64,
    pub submissions: u64,
    pub copied_bytes: u64,
    pub kernel_dispatches: u64,
    pub buffer_allocations: u64,
    pub buffer_bytes: u64,
    pub binding_preparations: u64,
    /// Direct frames encoded, including batches subsequently cancelled by callers.
    pub encoded_frames: u64,
}

pub struct ImageProcessor {
    storage_usage: ProcessorStorageUsage,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: ImageProcessorConfig,
    pixels: DeviceBuffer<u32>,
    gains: DeviceBuffer<Vec3>,
    kernel: ImageProcessKernel,
    direct: BindingsCache<ImageProcessBindings>,
    images: Vec<ProcessedImageLayout>,
    /// Sensor identities of the frame being planned; capacity is retained.
    plan_ids: HashSet<u64>,
    stats: ImageProcessingStats,
}

/// Borrowed outputs; queue collection or capture snapshots before mutable processor reuse.
pub struct ProcessedImages<'a> {
    processor: &'a ImageProcessor,
}

impl<'a> ProcessedImages<'a> {
    pub fn layouts(&self) -> &[ProcessedImageLayout] {
        &self.processor.images
    }
    pub fn image(&self, sensor_id: u64) -> Option<ProcessedImage<'a>> {
        self.processor
            .images
            .iter()
            .find(|image| image.stamp.sensor_id == sensor_id)
            .map(|layout| ProcessedImage {
                processor: self.processor,
                layout,
            })
    }
}

pub struct ProcessedImage<'a> {
    processor: &'a ImageProcessor,
    layout: &'a ProcessedImageLayout,
}

impl<'a> ProcessedImage<'a> {
    pub fn layout(&self) -> &ProcessedImageLayout {
        self.layout
    }
    pub fn readback_source(&self) -> Result<ReadbackSource<'a>> {
        self.processor
            .pixels
            .readback_source(self.layout.pixels.clone())
    }
    /// The index is local to this image's operation list, not the batch-wide stage range.
    pub fn white_balance_source(&self, stage: usize) -> Result<ReadbackSource<'a>> {
        if !self.layout.white_balance_stages.contains(&stage) {
            return Err(Error::Solver(
                "image processing: stage has no white-balance gains".into(),
            ));
        }
        let index = self.layout.stages.start + stage;
        self.processor.gains.readback_source(index..index + 1)
    }
}

impl ImageProcessor {
    /// Requested construction payload, excluding driver staging, metadata and output pools.
    pub fn storage_usage(&self) -> ProcessorStorageUsage {
        self.storage_usage
    }

    pub fn new(engine: &Engine, config: ImageProcessorConfig) -> Result<Self> {
        Self::new_with_storage_limits(engine, config, ProcessorStorageLimits::default())
    }

    /// Validate device budgets and combined construction payload before allocating arrays.
    pub fn new_with_storage_limits(
        engine: &Engine,
        config: ImageProcessorConfig,
        limits: ProcessorStorageLimits,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("image processing requires GPU".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("image processing requires queue".into()))?
            .clone();
        let storage_usage = ProcessorStorageUsage::for_image(config, &device.limits())?;
        limits.check(storage_usage)?;
        let bytes = storage_usage.device_bytes;
        let pixels = DeviceBuffer::from_host(
            engine,
            filled(config.max_pixels as usize, 0u32)?,
            BufferUsage::StorageReadWrite,
        )?;
        let slots = config.max_images * config.max_operations_per_image;
        let gains = DeviceBuffer::from_host(
            engine,
            filled(slots as usize, Vec3::ZERO)?,
            BufferUsage::StorageReadWrite,
        )?;
        let kernel = ImageProcessKernel::new(
            &device,
            pixels.device_buffer()?,
            gains.device_buffer()?,
            config.max_pixels,
            slots,
        )?;
        Ok(Self {
            storage_usage,
            device,
            queue,
            config,
            pixels,
            gains,
            kernel,
            direct: BindingsCache::with_capacity(128),
            images: Vec::new(),
            plan_ids: HashSet::new(),
            stats: ImageProcessingStats {
                buffer_allocations: 4,
                buffer_bytes: bytes,
                binding_preparations: 1,
                ..Default::default()
            },
        })
    }

    pub fn stats(&self) -> ImageProcessingStats {
        self.stats
    }

    fn output_specs(&self) -> [SensorChannelSpec; 2] {
        [
            SensorChannelSpec::of::<u32>(self.config.max_pixels as usize),
            SensorChannelSpec::of::<Vec3>(
                (self.config.max_images * self.config.max_operations_per_image) as usize,
            ),
        ]
    }

    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        self.pixels.check_device()?;
        engine.check_device()?;
        if !Arc::ptr_eq(&self.device, engine.device().expect("checked engine")) {
            return Err(Error::Gpu("image processing: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Encode device-source jobs into leased pixel/gain channels; submit through the batch.
    /// The processor and source jobs remain borrowed until submission or cancellation.
    ///
    /// ```compile_fail
    /// use molla_sensors::{ImageJob, ImageProcessor, SensorCommandBatch, SensorFramePool};
    /// fn reuse<'a>(p: &'a mut ImageProcessor, b: &mut SensorCommandBatch<'a>, jobs: &'a [ImageJob<'a>], pool: &mut SensorFramePool) {
    ///     p.encode(b, jobs, pool, Default::default()).unwrap();
    ///     p.process(jobs).unwrap();
    ///     b.stats();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sensors::{ImageJob, ImageProcessor, SensorCommandBatch, SensorFramePool};
    /// use molla_sensors::readback::DeviceBuffer;
    /// fn mutate(engine: &Engine, p: &mut ImageProcessor, input: &mut DeviceBuffer<u32>, pool: &mut SensorFramePool) {
    ///     let jobs = [ImageJob { source: input.readback_source(0..1).unwrap(), operations: &[], stamp: Default::default() }];
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     p.encode(&mut batch, &jobs, pool, Default::default()).unwrap();
    ///     input.host_mut().unwrap()[0] = 0;
    ///     batch.submit().unwrap();
    /// }
    /// ```
    pub fn encode<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        jobs: &'a [ImageJob<'a>],
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedProcessedImages>> {
        self.pixels.check_device()?;
        batch.check_device(&self.device)?;
        batch.check_source_capacity(jobs.len())?;
        let images = self.plan(jobs)?;
        let (pixels, stages) = active_sizes(&images);
        let specs = self.output_specs();
        let frame = batch.encode_output(
            pool,
            &specs,
            &[0..pixels, 0..stages],
            stamp,
            |_, _, encoder, outputs| {
                for (job, image) in jobs.iter().zip(&images) {
                    if let Some(source) = job.source.buffer {
                        encoder.copy_buffer_to_buffer(
                            source,
                            job.source.offset,
                            &outputs[0],
                            image.pixels.start as u64 * 4,
                            job.source.channel.bytes as u64,
                        );
                    }
                }
                self.encode_stages(
                    encoder,
                    outputs,
                    jobs.iter()
                        .zip(&images)
                        .map(|(job, image)| (job.operations, image)),
                )
            },
        )?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        for job in jobs {
            batch.retain_source(&job.source)?;
        }
        self.stats.encoded_frames += 1;
        self.stats.copied_bytes += pixels as u64 * 4;
        Ok(Some(EncodedProcessedImages {
            frame,
            layouts: images,
        }))
    }

    /// Encode jobs that read earlier results from this same batch.
    ///
    /// ```compile_fail
    /// use molla_sensors::{ImageProcessor, PendingImageJob, SensorCommandBatch, SensorFramePool};
    /// fn reuse<'a>(p: &'a mut ImageProcessor, b: &mut SensorCommandBatch<'a>, jobs: &[PendingImageJob<'_>], pool: &mut SensorFramePool) {
    ///     p.encode_pending(b, jobs, pool, Default::default()).unwrap();
    ///     p.process(&[]).unwrap();
    ///     b.stats();
    /// }
    /// ```
    pub fn encode_pending<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        jobs: &[PendingImageJob<'_>],
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedProcessedImages>> {
        self.pixels.check_device()?;
        batch.check_device(&self.device)?;
        if jobs.len() > self.config.max_images as usize {
            return Err(Error::Solver(
                "image processing: image budget exceeded".into(),
            ));
        }
        for job in jobs {
            batch.pending_range::<u32>(job.frame, job.channel, job.pixels.clone())?;
        }
        let mut ids = std::mem::take(&mut self.plan_ids);
        let planned = self.plan_metadata(
            jobs.iter()
                .map(|job| (job.pixels.len(), job.operations, job.stamp)),
            &mut ids,
        );
        self.plan_ids = ids;
        let images = planned?;
        let (pixels, stages) = active_sizes(&images);
        let specs = self.output_specs();
        let ranges = [0..pixels, 0..stages];
        let frame = if jobs.is_empty() {
            batch.encode_output(pool, &specs, &ranges, stamp, |_, _, _, _| Ok(0))?
        } else {
            let receipts: [_; 64] = std::array::from_fn(|i| jobs.get(i).unwrap_or(&jobs[0]).frame);
            batch.encode_from_pending(
                &receipts[..jobs.len()],
                pool,
                &specs,
                &ranges,
                stamp,
                |encoder, inputs, outputs| {
                    for (i, (job, image)) in jobs.iter().zip(&images).enumerate() {
                        let channel = inputs.channel::<u32>(i, job.channel)?;
                        let view = channel.slice(job.pixels.clone())?;
                        if !view.is_empty() {
                            encoder.copy_buffer_to_buffer(
                                view.binding()?.buffer,
                                view.active_range().start as u64 * 4,
                                &outputs[0],
                                image.pixels.start as u64 * 4,
                                view.len() as u64 * 4,
                            );
                        }
                    }
                    self.encode_stages(
                        encoder,
                        outputs,
                        jobs.iter()
                            .zip(&images)
                            .map(|(job, image)| (job.operations, image)),
                    )
                },
            )?
        };
        let Some(frame) = frame else {
            return Ok(None);
        };
        self.stats.encoded_frames += 1;
        self.stats.copied_bytes += pixels as u64 * 4;
        Ok(Some(EncodedProcessedImages {
            frame,
            layouts: images,
        }))
    }

    fn encode_stages<'b>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        outputs: &[wgpu::Buffer],
        jobs: impl Iterator<Item = (&'b [ImageOp], &'b ProcessedImageLayout)> + Clone,
    ) -> Result<u64> {
        if !jobs.clone().any(|(ops, image)| {
            ops.iter()
                .any(|op| !image.pixels.is_empty() || op.is_white_balance())
        }) {
            return Ok(0);
        }
        let (bindings, prepared) = self.direct.try_get_or_prepare(
            |binding| binding.matches(&outputs[0], &outputs[1]),
            || self.kernel.prepare(&self.device, &outputs[0], &outputs[1]),
        )?;
        self.stats.binding_preparations += u64::from(prepared);
        let mut dispatches = 0;
        for (operations, image) in jobs {
            for (index, op) in operations.iter().copied().enumerate() {
                if image.pixels.is_empty() && !op.is_white_balance() {
                    continue;
                }
                let (op, a, b) = op.parameters()?;
                dispatches += u64::from(self.kernel.encode_prepared(
                    &self.queue,
                    encoder,
                    bindings,
                    (image.stages.start + index) as u32,
                    image.pixels.start as u32,
                    image.pixels.len() as u32,
                    op,
                    a,
                    b,
                )?);
            }
        }
        self.stats.kernel_dispatches += dispatches;
        Ok(dispatches)
    }

    /// Validate the whole batch, copy active input ranges and encode all stages in one submission.
    pub fn process(&mut self, jobs: &[ImageJob<'_>]) -> Result<ProcessedImages<'_>> {
        self.pixels.check_device()?;
        let images = self.plan(jobs)?;
        let has_work = jobs.iter().any(|job| {
            job.source.channel.bytes != 0 || job.operations.iter().any(|op| op.is_white_balance())
        });
        if has_work {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("molla::image processing"),
                });
            let mut dispatches = 0;
            let mut bytes = 0;
            let mut written_gains = false;
            for (job, image) in jobs.iter().zip(&images) {
                if let Some(source) = job.source.buffer {
                    encoder.copy_buffer_to_buffer(
                        source,
                        job.source.offset,
                        self.pixels.device_buffer()?,
                        image.pixels.start as u64 * 4,
                        job.source.channel.bytes as u64,
                    );
                    bytes += job.source.channel.bytes as u64;
                }
                for (index, operation) in job.operations.iter().copied().enumerate() {
                    if image.pixels.is_empty() && !operation.is_white_balance() {
                        continue;
                    }
                    let (op, a, b) = operation.parameters()?;
                    dispatches += self.kernel.encode(
                        &self.queue,
                        &mut encoder,
                        (image.stages.start + index) as u32,
                        image.pixels.start as u32,
                        image.pixels.len() as u32,
                        op,
                        a,
                        b,
                    )? as u64;
                    written_gains |= operation.is_white_balance();
                }
            }
            self.queue.submit([encoder.finish()]);
            let leases: Vec<BufferLease> = jobs
                .iter()
                .filter_map(|job| job.source.lease.clone())
                .collect();
            if !leases.is_empty() {
                self.queue.on_submitted_work_done(move || drop(leases));
            }
            if bytes != 0 {
                self.pixels.mark_device_dirty();
            }
            if written_gains {
                self.gains.mark_device_dirty();
            }
            self.stats.submissions += 1;
            self.stats.copied_bytes += bytes;
            self.stats.kernel_dispatches += dispatches;
        }
        self.images = images;
        self.stats.frames += 1;
        Ok(ProcessedImages { processor: self })
    }

    fn plan(&mut self, jobs: &[ImageJob<'_>]) -> Result<Vec<ProcessedImageLayout>> {
        if jobs.len() > self.config.max_images as usize {
            return Err(Error::Solver(
                "image processing: image budget exceeded".into(),
            ));
        }
        for job in jobs {
            if !Arc::ptr_eq(&self.device, &job.source.device) {
                return Err(Error::Gpu("image processing: foreign source".into()));
            }
            if job.source.channel.kind != TypeId::of::<u32>()
                || job.source.channel.element_size != 4
            {
                return Err(Error::Solver(
                    "image processing: source must contain packed u32 pixels".into(),
                ));
            }
        }
        let mut ids = std::mem::take(&mut self.plan_ids);
        let planned = self.plan_metadata(
            jobs.iter()
                .map(|job| (job.source.channel.bytes / 4, job.operations, job.stamp)),
            &mut ids,
        );
        self.plan_ids = ids;
        planned
    }

    fn plan_metadata<'a>(
        &self,
        jobs: impl ExactSizeIterator<Item = (usize, &'a [ImageOp], ReadbackStamp)>,
        ids: &mut HashSet<u64>,
    ) -> Result<Vec<ProcessedImageLayout>> {
        if jobs.len() > self.config.max_images as usize {
            return Err(Error::Solver(
                "image processing: image budget exceeded".into(),
            ));
        }
        ids.clear();
        let mut images = Vec::with_capacity(jobs.len());
        let mut pixels = 0usize;
        let mut stages = 0usize;
        for (count, operations, stamp) in jobs {
            if !ids.insert(stamp.sensor_id)
                || operations.len() > self.config.max_operations_per_image as usize
                || stamp
                    .simulation_time
                    .is_some_and(|time| !time.is_finite() || time < 0.0)
            {
                return Err(Error::Solver(
                    "image processing: invalid image type, ID, stage budget or time".into(),
                ));
            }
            let end = pixels
                .checked_add(count)
                .ok_or_else(|| Error::Solver("image processing: pixel count overflow".into()))?;
            if end > self.config.max_pixels as usize {
                return Err(Error::Solver(
                    "image processing: pixel budget exceeded".into(),
                ));
            }
            for op in operations {
                op.parameters()?;
            }
            let white_balance_stages = operations
                .iter()
                .enumerate()
                .filter_map(|(i, op)| op.is_white_balance().then_some(i))
                .collect();
            images.push(ProcessedImageLayout {
                stamp,
                pixels: pixels..end,
                stages: stages..stages + operations.len(),
                white_balance_stages,
            });
            pixels = end;
            stages += operations.len();
        }
        Ok(images)
    }
}

fn active_sizes(images: &[ProcessedImageLayout]) -> (usize, usize) {
    images
        .last()
        .map_or((0, 0), |image| (image.pixels.end, image.stages.end))
}

pub(crate) fn checked_budget(config: ImageProcessorConfig, limits: &wgpu::Limits) -> Result<u64> {
    if config.max_images == 0
        || config.max_images > 64
        || config.max_operations_per_image == 0
        || config.max_operations_per_image > 64
        || config.max_pixels == 0
        || config.max_pixels.div_ceil(256) > limits.max_compute_workgroups_per_dimension
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_storage_size < 4096
    {
        return Err(Error::Solver(
            "image processing: invalid capacity or dispatch budget".into(),
        ));
    }
    let slots = config.max_images as u64 * config.max_operations_per_image as u64;
    let alignment = limits.min_uniform_buffer_offset_alignment as u64;
    if !alignment.is_power_of_two() {
        return Err(Error::Solver(
            "image processing: invalid uniform alignment".into(),
        ));
    }
    let uniform = slots * 32u64.div_ceil(alignment) * alignment;
    let storage = [
        config.max_pixels as u64 * 4,
        config.max_pixels.div_ceil(256) as u64 * 16,
        slots * 16,
    ];
    let bytes = storage.iter().sum::<u64>() + uniform;
    if bytes > config.max_buffer_bytes
        || uniform > limits.max_buffer_size
        || uniform > u32::MAX as u64
        || storage.into_iter().any(|size| {
            size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size as u64
        })
    {
        return Err(Error::Solver(
            "image processing: device or buffer byte budget exceeded".into(),
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
        let config = ImageProcessorConfig {
            max_pixels: 257,
            max_images: 3,
            max_operations_per_image: 2,
            max_buffer_bytes: u64::MAX,
        };
        let usage =
            ProcessorStorageUsage::for_image(config, &engine.device().unwrap().limits()).unwrap();
        assert!(
            ImageProcessor::new_with_storage_limits(
                &engine,
                config,
                ProcessorStorageLimits {
                    max_construction_bytes: usage.construction_bytes - 1,
                }
            )
            .is_err()
        );
        let processor = ImageProcessor::new_with_storage_limits(
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
                buffer_bytes(&processor.pixels.buffer),
                buffer_bytes(&processor.gains.buffer),
            ],
        );
    }

    #[test]
    fn buffer_budget_covers_pixels_reduction_gains_and_aligned_parameters() {
        let limits = wgpu::Limits::default();
        let config = ImageProcessorConfig {
            max_pixels: 257,
            max_images: 2,
            max_operations_per_image: 3,
            max_buffer_bytes: u64::MAX,
        };
        let expected = 257 * 4 + 2 * 16 + 6 * 16 + 6 * 256;
        assert_eq!(checked_budget(config, &limits).unwrap(), expected);
        assert!(
            checked_budget(
                ImageProcessorConfig {
                    max_buffer_bytes: expected - 1,
                    ..config
                },
                &limits
            )
            .is_err()
        );
        assert!(
            checked_budget(
                ImageProcessorConfig {
                    max_buffer_bytes: expected,
                    ..config
                },
                &limits
            )
            .is_ok()
        );
        let limits = wgpu::Limits {
            min_uniform_buffer_offset_alignment: 512,
            ..limits
        };
        assert_eq!(checked_budget(config, &limits).unwrap(), expected + 6 * 256);
    }

    #[test]
    fn capacities_reject_storage_dispatch_stage_and_workgroup_overflow() {
        let limits = wgpu::Limits::default();
        let base = ImageProcessorConfig::default();
        for config in [
            ImageProcessorConfig {
                max_pixels: 0,
                ..base
            },
            ImageProcessorConfig {
                max_pixels: u32::MAX,
                ..base
            },
            ImageProcessorConfig {
                max_images: 0,
                ..base
            },
            ImageProcessorConfig {
                max_images: 65,
                ..base
            },
            ImageProcessorConfig {
                max_operations_per_image: 0,
                ..base
            },
            ImageProcessorConfig {
                max_operations_per_image: 65,
                ..base
            },
        ] {
            assert!(checked_budget(config, &limits).is_err());
        }
        for limits in [
            wgpu::Limits {
                max_storage_buffer_binding_size: 16,
                ..limits.clone()
            },
            wgpu::Limits {
                max_buffer_size: 16,
                ..limits.clone()
            },
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
                max_compute_workgroup_storage_size: 2048,
                ..limits.clone()
            },
        ] {
            assert!(checked_budget(base, &limits).is_err());
        }
        assert!(
            checked_budget(
                ImageProcessorConfig {
                    max_pixels: 1,
                    max_images: 64,
                    max_operations_per_image: 64,
                    max_buffer_bytes: u64::MAX
                },
                &limits
            )
            .is_ok()
        );
    }

    #[test]
    fn operation_validation_preserves_clamped_cpu_parameters() {
        assert_eq!(ImageOp::Gamma(-1.0).parameters().unwrap(), (1, 1e6, 0.0));
        assert_eq!(
            ImageOp::ReinhardWhitePoint(f32::MAX).parameters().unwrap(),
            (5, 0.0, 0.0)
        );
        assert!(
            ImageOp::ReinhardWhitePoint(0.0)
                .parameters()
                .unwrap()
                .1
                .is_finite()
        );
        assert!(ImageOp::Exposure(f32::MAX).parameters().is_ok());
        assert!(ImageOp::Saturation(-2.0).parameters().is_ok());
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for op in [
                ImageOp::Exposure(v),
                ImageOp::Gamma(v),
                ImageOp::ReinhardWhitePoint(v),
                ImageOp::Saturation(v),
                ImageOp::BrightnessContrast {
                    brightness: v,
                    contrast: 1.0,
                },
                ImageOp::BrightnessContrast {
                    brightness: 0.0,
                    contrast: v,
                },
            ] {
                assert!(op.parameters().is_err());
            }
        }
        assert!(
            ImageOp::BrightnessContrast {
                brightness: f32::MAX,
                contrast: f32::MAX
            }
            .parameters()
            .is_err()
        );
    }
}
