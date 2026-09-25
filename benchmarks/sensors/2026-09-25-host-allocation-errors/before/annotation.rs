//! Device-resident shape-instance masks and per-camera semantic bounds.

use crate::processor_storage::filled;
use crate::{
    BoundingBox2D, CameraChannels, CameraImage, MISS_SHAPE_ID,
    device_results::{
        EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
        SensorFramePoolConfig,
    },
    dispatch::BindingsCache,
    readback::{BufferLease, DeviceBuffer, ReadbackSource, ReadbackStamp},
};
use crate::{ProcessorStorageLimits, ProcessorStorageUsage};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{AnnotationBindings, AnnotationKernel};
use molla_storage::{BufferElement, BufferUsage};
use std::{any::TypeId, collections::HashSet, ops::Range, sync::Arc};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShapeLabel {
    pub shape_id: u32,
    pub class_id: u32,
}

/// Packed 32-byte record; zero pixel count denotes an unused compacted tail slot.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeviceBoundingBox {
    /// Inclusive pixel bounds: minimum X/Y, maximum X/Y.
    pub bounds: [u32; 4],
    pub shape_id: u32,
    pub class_id: u32,
    pub pixel_count: u32,
    pub reserved: u32,
}

impl BufferElement for DeviceBoundingBox {
    type Gpu = Self;
    fn to_gpu(&self) -> Self {
        *self
    }
    fn from_gpu(value: &Self) -> Self {
        *value
    }
}

impl DeviceBoundingBox {
    pub fn to_cpu(self) -> Option<BoundingBox2D> {
        if self.pixel_count == 0 {
            return None;
        }
        Some(BoundingBox2D {
            x_min: self.bounds[0],
            y_min: self.bounds[1],
            x_max: self.bounds[2],
            y_max: self.bounds[3],
            shape_id: self.shape_id,
            class_id: self.class_id,
            pixel_count: self.pixel_count,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnnotationCounts {
    pub labelled_pixels: u32,
    pub unlabelled_pixels: u32,
    pub missed_pixels: u32,
    pub visible_instances: u32,
}

impl AnnotationCounts {
    pub fn from_words(words: &[u32]) -> Result<Self> {
        let &[
            labelled_pixels,
            unlabelled_pixels,
            missed_pixels,
            visible_instances,
        ] = words
        else {
            return Err(Error::Solver(
                "annotation: counts require four words".into(),
            ));
        };
        Ok(Self {
            labelled_pixels,
            unlabelled_pixels,
            missed_pixels,
            visible_instances,
        })
    }
}

pub struct AnnotationJob<'a> {
    pub shapes: ReadbackSource<'a>,
    pub width: u32,
    pub height: u32,
    pub stamp: ReadbackStamp,
}

pub struct PendingAnnotationJob<'a> {
    pub frame: &'a EncodedSensorFrame,
    pub channel: usize,
    /// Pixel indices relative to the pending channel's active range.
    pub pixels: Range<usize>,
    pub width: u32,
    pub height: u32,
    pub stamp: ReadbackStamp,
}

pub struct EncodedAnnotations {
    pub frame: EncodedSensorFrame,
    pub layouts: Vec<AnnotationLayout>,
    pub label_revision: u64,
}

impl<'a> AnnotationJob<'a> {
    pub fn from_camera(image: &CameraImage<'a>, simulation_time: Option<f64>) -> Result<Self> {
        Ok(Self {
            shapes: image.readback_source(CameraChannels::SHAPE_INDEX)?,
            width: image.layout().width,
            height: image.layout().height,
            stamp: image.readback_stamp(CameraChannels::SHAPE_INDEX, simulation_time)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AnnotationConfig {
    pub max_pixels: u32,
    pub max_images: u32,
    pub max_labels: u32,
    pub max_buffer_bytes: u64,
}

impl Default for AnnotationConfig {
    fn default() -> Self {
        Self {
            max_pixels: 4_000_000,
            max_images: 16,
            max_labels: 1024,
            max_buffer_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnnotationStats {
    pub frames: u64,
    pub submissions: u64,
    pub copied_bytes: u64,
    pub kernel_dispatches: u64,
    pub buffer_allocations: u64,
    pub buffer_bytes: u64,
    pub binding_preparations: u64,
    pub label_uploads: u64,
    /// Successful direct encodes, including subsequently cancelled batches.
    pub encoded_frames: u64,
}

#[derive(Debug, Clone)]
pub struct AnnotationLayout {
    pub stamp: ReadbackStamp,
    pub width: u32,
    pub height: u32,
    pub pixels: Range<usize>,
    pub boxes: Range<usize>,
}

pub struct AnnotationDeviceFrame<'a> {
    processor: &'a AnnotationProcessor,
}

impl<'a> AnnotationDeviceFrame<'a> {
    pub fn layouts(&self) -> &[AnnotationLayout] {
        &self.processor.layouts
    }
    pub fn label_revision(&self) -> u64 {
        self.processor.label_revision
    }
    fn layout(&self, image: usize) -> Result<&AnnotationLayout> {
        self.processor
            .layouts
            .get(image)
            .ok_or_else(|| Error::Solver("annotation: unknown image".into()))
    }
    /// Full-width shape-instance IDs, with `u32::MAX` for misses and original IDs for unlabelled shapes.
    pub fn shape_source(&self, image: usize, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        self.processor
            .shapes
            .readback_source(subrange(&self.layout(image)?.pixels, range)?)
    }
    pub fn class_source(&self, image: usize, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        self.processor
            .classes
            .readback_source(subrange(&self.layout(image)?.pixels, range)?)
    }
    /// Visible prefix is sorted by shape ID; the remaining active label extent is zeroed.
    pub fn boxes_source(&self, image: usize, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        self.processor
            .boxes
            .readback_source(subrange(&self.layout(image)?.boxes, range)?)
    }
    pub fn counts_source(&self, image: usize) -> Result<ReadbackSource<'a>> {
        self.layout(image)?;
        self.processor
            .counts
            .readback_source(image * 4..image * 4 + 4)
    }
    pub fn box_count_source(&self, image: usize) -> Result<ReadbackSource<'a>> {
        self.layout(image)?;
        self.processor
            .counts
            .readback_source(image * 4 + 3..image * 4 + 4)
    }
}

fn subrange(active: &Range<usize>, range: Range<usize>) -> Result<Range<usize>> {
    if range.start > range.end || range.end > active.len() {
        return Err(Error::Solver(
            "annotation: range exceeds active image".into(),
        ));
    }
    Ok(active.start + range.start..active.start + range.end)
}

pub struct AnnotationProcessor {
    storage_usage: ProcessorStorageUsage,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: AnnotationConfig,
    shapes: DeviceBuffer<u32>,
    classes: DeviceBuffer<u32>,
    boxes: DeviceBuffer<DeviceBoundingBox>,
    counts: DeviceBuffer<u32>,
    labels: wgpu::Buffer,
    descriptors: wgpu::Buffer,
    work: wgpu::Buffer,
    kernel: AnnotationKernel,
    direct: BindingsCache<AnnotationBindings>,
    label_count: u32,
    label_revision: u64,
    layouts: Vec<AnnotationLayout>,
    stats: AnnotationStats,
}

impl AnnotationProcessor {
    /// Requested construction payload, excluding driver staging, metadata and output pools.
    pub fn storage_usage(&self) -> ProcessorStorageUsage {
        self.storage_usage
    }

    pub fn new(engine: &Engine, config: AnnotationConfig) -> Result<Self> {
        Self::new_with_storage_limits(engine, config, ProcessorStorageLimits::default())
    }

    /// Validate device budgets and combined construction payload before allocating arrays.
    pub fn new_with_storage_limits(
        engine: &Engine,
        config: AnnotationConfig,
        limits: ProcessorStorageLimits,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("annotation requires GPU".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("annotation requires queue".into()))?
            .clone();
        let storage_usage = ProcessorStorageUsage::for_annotation(config, &device.limits())?;
        limits.check(storage_usage)?;
        let bytes = storage_usage.device_bytes;
        let shapes = DeviceBuffer::from_host(
            engine,
            filled(config.max_pixels as usize, 0u32)?,
            BufferUsage::StorageReadWrite,
        )?;
        let classes = DeviceBuffer::from_host(
            engine,
            filled(config.max_pixels as usize, 0u32)?,
            BufferUsage::StorageReadWrite,
        )?;
        let boxes = DeviceBuffer::from_host(
            engine,
            filled(
                config.max_images as usize * config.max_labels as usize,
                DeviceBoundingBox::default(),
            )?,
            BufferUsage::StorageReadWrite,
        )?;
        let counts = DeviceBuffer::from_host(
            engine,
            filled(config.max_images as usize * 4, 0u32)?,
            BufferUsage::StorageReadWrite,
        )?;
        let storage = |size, label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let labels = storage(config.max_labels as u64 * 8, "molla::annotation labels");
        let descriptors = storage(config.max_images as u64 * 16, "molla::annotation images");
        let work = storage(
            config.max_images as u64 * config.max_labels as u64 * 32,
            "molla::annotation working boxes",
        );
        let kernel = AnnotationKernel::new(
            &device,
            [
                shapes.device_buffer()?,
                classes.device_buffer()?,
                &labels,
                &descriptors,
                &work,
                boxes.device_buffer()?,
                counts.device_buffer()?,
            ],
            [config.max_pixels, config.max_images, config.max_labels],
        )?;
        Ok(Self {
            storage_usage,
            device,
            queue,
            config,
            shapes,
            classes,
            boxes,
            counts,
            labels,
            descriptors,
            work,
            kernel,
            direct: BindingsCache::with_capacity(128),
            label_count: 0,
            label_revision: 0,
            layouts: Vec::new(),
            stats: AnnotationStats {
                buffer_allocations: 8,
                buffer_bytes: bytes,
                binding_preparations: 1,
                ..Default::default()
            },
        })
    }

    pub fn stats(&self) -> AnnotationStats {
        self.stats
    }
    pub fn label_revision(&self) -> u64 {
        self.label_revision
    }

    fn output_specs(&self) -> [SensorChannelSpec; 4] {
        [
            SensorChannelSpec::of::<u32>(self.config.max_pixels as usize),
            SensorChannelSpec::of::<u32>(self.config.max_pixels as usize),
            SensorChannelSpec::of::<DeviceBoundingBox>(
                self.config.max_images as usize * self.config.max_labels as usize,
            ),
            SensorChannelSpec::of::<u32>(self.config.max_images as usize * 4),
        ]
    }

    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        self.shapes.check_device()?;
        engine.check_device()?;
        if !Arc::ptr_eq(&self.device, engine.device().expect("checked engine")) {
            return Err(Error::Gpu("annotation: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Encode device-source images into leased shapes, classes, boxes and counts.
    /// Source jobs share the batch's 64-external-input bound.
    ///
    /// ```compile_fail
    /// use molla_sensors::{AnnotationJob, AnnotationProcessor, SensorCommandBatch, SensorFramePool};
    /// fn relabel<'a>(p: &'a mut AnnotationProcessor, b: &mut SensorCommandBatch<'a>, jobs: &'a [AnnotationJob<'a>], pool: &mut SensorFramePool) {
    ///     p.encode(b, jobs, pool, Default::default()).unwrap();
    ///     p.set_labels(&[]).unwrap();
    ///     b.stats();
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sensors::{AnnotationJob, AnnotationProcessor, SensorCommandBatch, SensorFramePool};
    /// use molla_sensors::readback::DeviceBuffer;
    /// fn mutate(engine: &Engine, p: &mut AnnotationProcessor, input: &mut DeviceBuffer<u32>, pool: &mut SensorFramePool) {
    ///     let jobs = [AnnotationJob { shapes: input.readback_source(0..1).unwrap(), width: 1, height: 1, stamp: Default::default() }];
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     p.encode(&mut batch, &jobs, pool, Default::default()).unwrap();
    ///     input.host_mut().unwrap()[0] = 0;
    ///     batch.submit().unwrap();
    /// }
    /// ```
    pub fn encode<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        jobs: &'a [AnnotationJob<'a>],
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedAnnotations>> {
        self.shapes.check_device()?;
        batch.check_device(&self.device)?;
        batch.check_source_capacity(jobs.len())?;
        let (descriptors, layouts, pixels) = self.plan(jobs)?;
        let specs = self.output_specs();
        let ranges = self.output_ranges(pixels, jobs.len());
        let frame =
            batch.encode_output(pool, &specs, &ranges, stamp, |_, _, encoder, outputs| {
                for (job, layout) in jobs.iter().zip(&layouts) {
                    if job.shapes.channel.bytes != 0 {
                        encoder.copy_buffer_to_buffer(
                            job.shapes.buffer.expect("validated shape source"),
                            job.shapes.offset,
                            &outputs[0],
                            layout.pixels.start as u64 * 4,
                            job.shapes.channel.bytes as u64,
                        );
                    }
                }
                self.encode_direct(encoder, outputs, &descriptors, pixels)
            })?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        for job in jobs {
            batch.retain_source(&job.shapes)?;
        }
        self.stats.encoded_frames += 1;
        self.stats.copied_bytes += pixels as u64 * 4;
        Ok(Some(EncodedAnnotations {
            frame,
            layouts,
            label_revision: self.label_revision,
        }))
    }

    /// Encode up to 1024 images from earlier outputs, deduplicating receipt dependencies.
    ///
    /// ```compile_fail
    /// use molla_sensors::{AnnotationProcessor, PendingAnnotationJob, SensorCommandBatch, SensorFramePool};
    /// fn reuse<'a>(p: &'a mut AnnotationProcessor, b: &mut SensorCommandBatch<'a>, jobs: &[PendingAnnotationJob<'_>], pool: &mut SensorFramePool) {
    ///     p.encode_pending(b, jobs, pool, Default::default()).unwrap();
    ///     p.process(&[]).unwrap();
    ///     b.stats();
    /// }
    /// ```
    pub fn encode_pending<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        jobs: &[PendingAnnotationJob<'_>],
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedAnnotations>> {
        self.shapes.check_device()?;
        batch.check_device(&self.device)?;
        if jobs.len() > self.config.max_images as usize {
            return Err(Error::Solver("annotation: image budget exceeded".into()));
        }
        let mut receipts: Vec<&EncodedSensorFrame> = Vec::with_capacity(64);
        let mut indices = Vec::with_capacity(jobs.len());
        for job in jobs {
            batch.pending_range::<u32>(job.frame, job.channel, job.pixels.clone())?;
            let index = if let Some(index) = receipts
                .iter()
                .position(|frame| frame.same_receipt(job.frame))
            {
                index
            } else {
                if receipts.len() == 64 {
                    return Err(Error::Solver(
                        "annotation: pending dependency budget exceeded".into(),
                    ));
                }
                receipts.push(job.frame);
                receipts.len() - 1
            };
            indices.push(index);
        }
        let (descriptors, layouts, pixels) = self.plan_metadata(jobs.iter().map(|job| {
            (
                job.width,
                job.height,
                job.pixels.len() as u64 * 4,
                job.stamp,
            )
        }))?;
        let specs = self.output_specs();
        let ranges = self.output_ranges(pixels, jobs.len());
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
                    for ((job, layout), &index) in jobs.iter().zip(&layouts).zip(&indices) {
                        let channel = inputs.channel::<u32>(index, job.channel)?;
                        let view = channel.slice(job.pixels.clone())?;
                        if !view.is_empty() {
                            encoder.copy_buffer_to_buffer(
                                view.binding()?.buffer,
                                view.active_range().start as u64 * 4,
                                &outputs[0],
                                layout.pixels.start as u64 * 4,
                                view.len() as u64 * 4,
                            );
                        }
                    }
                    self.encode_direct(encoder, outputs, &descriptors, pixels)
                },
            )?
        };
        let Some(frame) = frame else {
            return Ok(None);
        };
        self.stats.encoded_frames += 1;
        self.stats.copied_bytes += pixels as u64 * 4;
        Ok(Some(EncodedAnnotations {
            frame,
            layouts,
            label_revision: self.label_revision,
        }))
    }

    fn output_ranges(&self, pixels: u32, images: usize) -> [Range<usize>; 4] {
        [
            0..pixels as usize,
            0..pixels as usize,
            0..images * self.label_count as usize,
            0..images * 4,
        ]
    }

    fn encode_direct(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        outputs: &[wgpu::Buffer],
        descriptors: &[[u32; 4]],
        pixels: u32,
    ) -> Result<u64> {
        if descriptors.is_empty() {
            return Ok(0);
        }
        self.queue
            .write_buffer(&self.descriptors, 0, bytemuck::cast_slice(descriptors));
        if pixels == 0 && self.label_count == 0 {
            return Ok(0);
        }
        let buffers = [
            &outputs[0],
            &outputs[1],
            &self.labels,
            &self.descriptors,
            &self.work,
            &outputs[2],
            &outputs[3],
        ];
        let (bindings, prepared) = self.direct.try_get_or_prepare(
            |bindings| bindings.matches(buffers),
            || self.kernel.prepare(&self.device, buffers),
        )?;
        self.stats.binding_preparations += u64::from(prepared);
        let dispatches = self.kernel.encode_prepared(
            &self.queue,
            encoder,
            bindings,
            [pixels, descriptors.len() as u32, self.label_count],
        )?;
        self.stats.kernel_dispatches += dispatches;
        Ok(dispatches)
    }

    /// Replaces the shared label table. Duplicate or miss-sentinel shape IDs are rejected.
    pub fn set_labels(&mut self, labels: &[ShapeLabel]) -> Result<u64> {
        self.shapes.check_device()?;
        if labels.len() > self.config.max_labels as usize {
            return Err(Error::Solver("annotation: label budget exceeded".into()));
        }
        let revision = self
            .label_revision
            .checked_add(1)
            .ok_or_else(|| Error::Solver("annotation: label revision exhausted".into()))?;
        let mut sorted = labels.to_vec();
        sorted.sort_unstable_by_key(|label| label.shape_id);
        if sorted.iter().any(|label| label.shape_id == MISS_SHAPE_ID)
            || sorted
                .windows(2)
                .any(|pair| pair[0].shape_id == pair[1].shape_id)
        {
            return Err(Error::Solver(
                "annotation: duplicate or reserved shape ID".into(),
            ));
        }
        if !sorted.is_empty() {
            self.queue
                .write_buffer(&self.labels, 0, bytemuck::cast_slice(&sorted));
        }
        self.label_count = sorted.len() as u32;
        self.label_revision = revision;
        self.layouts.clear();
        self.stats.label_uploads += u64::from(!sorted.is_empty());
        Ok(revision)
    }

    pub fn process(&mut self, jobs: &[AnnotationJob<'_>]) -> Result<AnnotationDeviceFrame<'_>> {
        self.shapes.check_device()?;
        let (descriptors, layouts, pixels) = self.plan(jobs)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::annotations"),
            });
        encoder.clear_buffer(self.counts.device_buffer()?, 0, None);
        if !descriptors.is_empty() {
            self.queue
                .write_buffer(&self.descriptors, 0, bytemuck::cast_slice(&descriptors));
        }
        for (job, layout) in jobs.iter().zip(&layouts) {
            if job.shapes.channel.bytes != 0 {
                let source = job
                    .shapes
                    .buffer
                    .ok_or_else(|| Error::Solver("annotation: missing shape buffer".into()))?;
                encoder.copy_buffer_to_buffer(
                    source,
                    job.shapes.offset,
                    self.shapes.device_buffer()?,
                    layout.pixels.start as u64 * 4,
                    job.shapes.channel.bytes as u64,
                );
            }
        }
        let dispatches = self.kernel.encode(
            &self.queue,
            &mut encoder,
            [pixels, jobs.len() as u32, self.label_count],
        )?;
        self.queue.submit([encoder.finish()]);
        let leases: Vec<BufferLease> = jobs
            .iter()
            .filter_map(|job| job.shapes.lease.clone())
            .collect();
        if !leases.is_empty() {
            self.queue.on_submitted_work_done(move || drop(leases));
        }
        self.counts.mark_device_dirty();
        if pixels != 0 {
            self.shapes.mark_device_dirty();
            self.classes.mark_device_dirty();
        }
        if !jobs.is_empty() && self.label_count != 0 {
            self.boxes.mark_device_dirty();
        }
        self.layouts = layouts;
        self.stats.frames += 1;
        self.stats.submissions += 1;
        self.stats.kernel_dispatches += dispatches;
        self.stats.copied_bytes += pixels as u64 * 4;
        Ok(AnnotationDeviceFrame { processor: self })
    }

    fn plan(
        &self,
        jobs: &[AnnotationJob<'_>],
    ) -> Result<(Vec<[u32; 4]>, Vec<AnnotationLayout>, u32)> {
        if jobs.len() > self.config.max_images as usize {
            return Err(Error::Solver("annotation: image budget exceeded".into()));
        }
        for job in jobs {
            if job.shapes.channel.kind != TypeId::of::<u32>()
                || job.shapes.channel.element_size != 4
                || !Arc::ptr_eq(&self.device, &job.shapes.device)
                || (job.shapes.channel.bytes != 0 && job.shapes.buffer.is_none())
            {
                return Err(Error::Solver(
                    "annotation: malformed or foreign shape source".into(),
                ));
            }
        }
        self.plan_metadata(jobs.iter().map(|job| {
            (
                job.width,
                job.height,
                job.shapes.channel.bytes as u64,
                job.stamp,
            )
        }))
    }

    fn plan_metadata(
        &self,
        jobs: impl ExactSizeIterator<Item = (u32, u32, u64, ReadbackStamp)>,
    ) -> Result<(Vec<[u32; 4]>, Vec<AnnotationLayout>, u32)> {
        if jobs.len() > self.config.max_images as usize {
            return Err(Error::Solver("annotation: image budget exceeded".into()));
        }
        let mut descriptors = Vec::with_capacity(jobs.len());
        let mut layouts = Vec::with_capacity(jobs.len());
        let mut ids = HashSet::with_capacity(jobs.len());
        let mut total = 0u32;
        for (i, (width, height, bytes, stamp)) in jobs.enumerate() {
            let pixels = width
                .checked_mul(height)
                .ok_or_else(|| Error::Solver("annotation: image size overflow".into()))?;
            let end = total
                .checked_add(pixels)
                .filter(|&n| n <= self.config.max_pixels)
                .ok_or_else(|| Error::Solver("annotation: pixel budget exceeded".into()))?;
            if (width == 0) != (height == 0)
                || bytes != pixels as u64 * 4
                || !ids.insert(stamp.sensor_id)
                || stamp
                    .simulation_time
                    .is_some_and(|t| !t.is_finite() || t < 0.)
            {
                return Err(Error::Solver(
                    "annotation: malformed image, source, identity or timestamp".into(),
                ));
            }
            descriptors.push([total, pixels, width, height]);
            layouts.push(AnnotationLayout {
                stamp,
                width,
                height,
                pixels: total as usize..end as usize,
                boxes: i * self.label_count as usize..(i + 1) * self.label_count as usize,
            });
            total = end;
        }
        Ok((descriptors, layouts, total))
    }
}

pub(crate) fn checked_budget(config: AnnotationConfig, limits: &wgpu::Limits) -> Result<u64> {
    let [p, i, l] = [
        config.max_pixels as u64,
        config.max_images as u64,
        config.max_labels as u64,
    ];
    let slots = i * l;
    if p == 0 || i == 0 || l == 0 || slots > u32::MAX as u64 || i > 1024 {
        return Err(Error::Gpu("annotation: invalid capacities".into()));
    }
    let bytes = 8 * p + 64 * slots + 8 * l + 32 * i + 16;
    if bytes > config.max_buffer_bytes
        || p.max(slots).div_ceil(256) > limits.max_compute_workgroups_per_dimension as u64
        || i > limits.max_compute_workgroups_per_dimension as u64
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_storage_size < 1024
        || limits.max_storage_buffers_per_shader_stage < 7
        || [4 * p, 8 * l, 16 * i, 32 * slots, 16].into_iter().any(|b| {
            b > limits.max_buffer_size || b > limits.max_storage_buffer_binding_size as u64
        })
    {
        return Err(Error::Gpu(
            "annotation: memory or device budget exceeded".into(),
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
        let config = AnnotationConfig {
            max_pixels: 257,
            max_images: 3,
            max_labels: 5,
            max_buffer_bytes: u64::MAX,
        };
        let usage =
            ProcessorStorageUsage::for_annotation(config, &engine.device().unwrap().limits())
                .unwrap();
        assert!(
            AnnotationProcessor::new_with_storage_limits(
                &engine,
                config,
                ProcessorStorageLimits {
                    max_construction_bytes: usage.construction_bytes - 1,
                }
            )
            .is_err()
        );
        let processor = AnnotationProcessor::new_with_storage_limits(
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
                buffer_bytes(&processor.shapes.buffer),
                buffer_bytes(&processor.classes.buffer),
                buffer_bytes(&processor.boxes.buffer),
                buffer_bytes(&processor.counts.buffer),
            ],
        );
    }

    #[test]
    fn annotation_wire_budget_and_capacity_checks() {
        assert_eq!(std::mem::size_of::<ShapeLabel>(), 8);
        assert_eq!(std::mem::size_of::<DeviceBoundingBox>(), 32);
        let limits = wgpu::Limits::default();
        let mut config = AnnotationConfig {
            max_pixels: 257,
            max_images: 3,
            max_labels: 257,
            max_buffer_bytes: 8 * 257 + 64 * 3 * 257 + 8 * 257 + 32 * 3 + 16,
        };
        assert_eq!(
            checked_budget(config, &limits).unwrap(),
            config.max_buffer_bytes
        );
        config.max_buffer_bytes -= 1;
        assert!(checked_budget(config, &limits).is_err());
        assert!(
            checked_budget(
                AnnotationConfig {
                    max_images: u32::MAX,
                    max_labels: u32::MAX,
                    max_buffer_bytes: u64::MAX,
                    ..config
                },
                &limits
            )
            .is_err()
        );
        assert!(
            checked_budget(
                AnnotationConfig {
                    max_labels: 0,
                    ..config
                },
                &limits
            )
            .is_err()
        );
        assert!(AnnotationCounts::from_words(&[0; 3]).is_err());
        assert_eq!(
            AnnotationCounts::from_words(&[1, 2, 3, 4])
                .unwrap()
                .visible_instances,
            4
        );
        assert!(DeviceBoundingBox::default().to_cpu().is_none());
    }
}
