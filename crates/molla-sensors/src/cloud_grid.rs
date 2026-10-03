//! Bounded device-resident spatial membership and per-cell reductions.

use crate::host_arrays::filled;
use crate::{
    CloudPoint,
    device_results::{
        EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
        SensorFramePoolConfig,
    },
    dispatch::BindingsCache,
    readback::{DeviceBuffer, ReadbackSource, ReadbackStamp},
};
use crate::{ProcessorStorageLimits, ProcessorStorageUsage};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{CloudGridBindings, CloudGridKernel, CloudGridParams};
use molla_storage::{BufferElement, BufferUsage};
use std::{any::TypeId, ops::Range, sync::Arc};

/// Half-open bounds `[minimum, maximum)` in the input point coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudGridSpec {
    pub minimum: [f32; 3],
    pub maximum: [f32; 3],
    pub dimensions: [u32; 3],
}

impl CloudGridSpec {
    fn parameters(self, capacity: u32) -> Result<CloudGridParams> {
        let cells = self
            .dimensions
            .into_iter()
            .try_fold(1u32, u32::checked_mul)
            .filter(|&n| n > 0 && n <= capacity)
            .ok_or_else(|| {
                Error::Solver("cloud grid: invalid dimensions or cell capacity".into())
            })?;
        let mut params = CloudGridParams::default();
        for i in 0..3 {
            let min = self.minimum[i];
            let max = self.maximum[i];
            let size = (max - min) / self.dimensions[i] as f32;
            if !min.is_finite() || !max.is_finite() || !size.is_finite() || size < f32::MIN_POSITIVE
            {
                return Err(Error::Solver(
                    "cloud grid: bounds require finite positive normal cell widths".into(),
                ));
            }
            params.minimum[i] = min;
            params.maximum[i] = max;
            params.size[i] = size;
            params.dimensions[i] = self.dimensions[i];
        }
        params.counts[1] = cells;
        Ok(params)
    }
}

/// 64-byte summary; cells are flattened as `(z * ny + y) * nx + x`.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudGridCell {
    /// Index start, member count, numeric-overflow flag (bit 0), reserved.
    pub header: [u32; 4],
    /// Centroid XYZ and mean intensity; zero for empty or overflowed cells.
    pub mean: [f32; 4],
    /// Minimum XYZ and intensity; zero for empty cells.
    pub minimum: [f32; 4],
    /// Maximum XYZ and intensity; zero for empty cells.
    pub maximum: [f32; 4],
}

impl BufferElement for CloudGridCell {
    type Gpu = Self;
    fn to_gpu(&self) -> Self {
        *self
    }
    fn from_gpu(value: &Self) -> Self {
        *value
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CloudGridDiagnostics {
    pub accepted: u32,
    pub misses: u32,
    pub invalid: u32,
    pub outside: u32,
    pub index_overflow: u32,
    pub numeric_overflow_cells: u32,
}

impl CloudGridDiagnostics {
    pub fn from_words(words: &[u32]) -> Result<Self> {
        let &[
            accepted,
            misses,
            invalid,
            outside,
            index_overflow,
            numeric_overflow_cells,
        ] = words
        else {
            return Err(Error::Solver(
                "cloud grid: diagnostics need six words".into(),
            ));
        };
        Ok(Self {
            accepted,
            misses,
            invalid,
            outside,
            index_overflow,
            numeric_overflow_cells,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CloudGridConfig {
    pub max_points: u32,
    pub max_cells: u32,
    pub max_buffer_bytes: u64,
}

impl Default for CloudGridConfig {
    fn default() -> Self {
        Self {
            max_points: 65_536,
            max_cells: 65_536,
            max_buffer_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CloudGridStats {
    pub frames: u64,
    pub submissions: u64,
    pub copied_bytes: u64,
    pub kernel_dispatches: u64,
    pub buffer_allocations: u64,
    pub buffer_bytes: u64,
    pub binding_preparations: u64,
    /// Successful direct encodes, including subsequently cancelled batches.
    pub encoded_frames: u64,
}

pub struct PendingCloudGridInput<'a> {
    pub frame: &'a EncodedSensorFrame,
    pub channel: usize,
    /// Point indices relative to the pending channel's active range.
    pub points: Range<usize>,
}

pub struct EncodedCloudGrid {
    pub frame: EncodedSensorFrame,
    pub spec: CloudGridSpec,
    pub input_count: usize,
    pub cell_count: usize,
}

pub struct CloudGridFrame<'a> {
    grid: &'a CloudGrid,
    pub spec: CloudGridSpec,
    pub stamp: ReadbackStamp,
    input_count: usize,
    cell_count: usize,
}

impl<'a> CloudGridFrame<'a> {
    pub fn input_count(&self) -> usize {
        self.input_count
    }
    pub fn cell_count(&self) -> usize {
        self.cell_count
    }
    pub fn cells_source(&self, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        checked_range(&range, self.cell_count)?;
        self.grid.cells.readback_source(range)
    }
    /// Accepted prefix only; unused active tail is `u32::MAX`. Cell-local order is unspecified.
    pub fn indices_source(&self, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        checked_range(&range, self.input_count)?;
        self.grid.indices.readback_source(range)
    }
    /// Packed input copy addressed by the range-local membership indices.
    pub fn points_source(&self, range: Range<usize>) -> Result<ReadbackSource<'a>> {
        checked_range(&range, self.input_count)?;
        self.grid.points.readback_source(range)
    }
    pub fn diagnostics_source(&self) -> Result<ReadbackSource<'a>> {
        self.grid.diagnostics.readback_source(0..6)
    }
}

fn checked_range(range: &Range<usize>, count: usize) -> Result<()> {
    if range.start > range.end || range.end > count {
        return Err(Error::Solver(
            "cloud grid: range exceeds active extent".into(),
        ));
    }
    Ok(())
}

pub struct CloudGrid {
    storage_usage: ProcessorStorageUsage,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: CloudGridConfig,
    points: DeviceBuffer<CloudPoint>,
    indices: DeviceBuffer<u32>,
    cells: DeviceBuffer<CloudGridCell>,
    diagnostics: DeviceBuffer<u32>,
    kernel: CloudGridKernel,
    direct: BindingsCache<CloudGridBindings>,
    stats: CloudGridStats,
}

impl CloudGrid {
    /// Requested construction payload, excluding driver staging, metadata and output pools.
    pub fn storage_usage(&self) -> ProcessorStorageUsage {
        self.storage_usage
    }

    pub fn new(engine: &Engine, config: CloudGridConfig) -> Result<Self> {
        Self::new_with_storage_limits(engine, config, ProcessorStorageLimits::default())
    }

    /// Validate device budgets and combined construction payload before allocating arrays.
    pub fn new_with_storage_limits(
        engine: &Engine,
        config: CloudGridConfig,
        limits: ProcessorStorageLimits,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("cloud grid requires GPU".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("cloud grid requires queue".into()))?
            .clone();
        let storage_usage = ProcessorStorageUsage::for_grid(config, &device.limits())?;
        limits.check(storage_usage)?;
        let bytes = storage_usage.device_bytes;
        let points = DeviceBuffer::from_host(
            engine,
            filled(config.max_points as usize, CloudPoint::default())?,
            BufferUsage::StorageReadWrite,
        )?;
        let indices = DeviceBuffer::from_host(
            engine,
            filled(config.max_points as usize, 0u32)?,
            BufferUsage::StorageReadWrite,
        )?;
        let cells = DeviceBuffer::from_host(
            engine,
            filled(config.max_cells as usize, CloudGridCell::default())?,
            BufferUsage::StorageReadWrite,
        )?;
        let diagnostics =
            DeviceBuffer::from_host(engine, filled(6, 0u32)?, BufferUsage::StorageReadWrite)?;
        let kernel = CloudGridKernel::new(
            &device,
            [
                points.device_buffer()?,
                indices.device_buffer()?,
                cells.device_buffer()?,
                diagnostics.device_buffer()?,
            ],
            config.max_points,
            config.max_cells,
        )?;
        Ok(Self {
            storage_usage,
            device,
            queue,
            config,
            points,
            indices,
            cells,
            diagnostics,
            kernel,
            direct: BindingsCache::with_capacity(128),
            stats: CloudGridStats {
                buffer_allocations: 8,
                buffer_bytes: bytes,
                binding_preparations: 1,
                ..Default::default()
            },
        })
    }

    pub fn stats(&self) -> CloudGridStats {
        self.stats
    }

    fn output_specs(&self) -> [SensorChannelSpec; 4] {
        [
            SensorChannelSpec::of::<CloudPoint>(self.config.max_points as usize),
            SensorChannelSpec::of::<u32>(self.config.max_points as usize),
            SensorChannelSpec::of::<CloudGridCell>(self.config.max_cells as usize),
            SensorChannelSpec::of::<u32>(6),
        ]
    }

    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        self.points.check_device()?;
        engine.check_device()?;
        if !Arc::ptr_eq(&self.device, engine.device().expect("checked engine")) {
            return Err(Error::Gpu("cloud grid: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Encode a device range into leased points, indices, cells and diagnostics.
    ///
    /// ```compile_fail
    /// use molla_compute::Engine;
    /// use molla_sensors::{CloudGrid, CloudGridSpec, CloudPoint, SensorCommandBatch, SensorFramePool};
    /// use molla_sensors::readback::DeviceBuffer;
    /// fn mutate(engine: &Engine, grid: &mut CloudGrid, points: &mut DeviceBuffer<CloudPoint>, spec: CloudGridSpec, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     grid.encode(&mut batch, points.readback_source(0..1).unwrap(), spec, pool, Default::default()).unwrap();
    ///     points.host_mut().unwrap()[0] = CloudPoint::default();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    pub fn encode<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        source: ReadbackSource<'a>,
        spec: CloudGridSpec,
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedCloudGrid>> {
        self.points.check_device()?;
        batch.check_device(&self.device)?;
        batch.check_source_capacity(1)?;
        let count = self.validate_source(&source)?;
        let params = self.parameters(count, spec, stamp)?;
        let cell_count = params.counts[1] as usize;
        let specs = self.output_specs();
        let ranges = [0..count, 0..count, 0..cell_count, 0..6];
        let frame =
            batch.encode_output(pool, &specs, &ranges, stamp, |_, _, encoder, outputs| {
                if count != 0 {
                    encoder.copy_buffer_to_buffer(
                        source.buffer.expect("validated source"),
                        source.offset,
                        &outputs[0],
                        0,
                        count as u64 * 112,
                    );
                }
                self.encode_direct(encoder, outputs, params)
            })?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        batch.retain_source(&source)?;
        self.stats.encoded_frames += 1;
        self.stats.copied_bytes += count as u64 * 112;
        Ok(Some(EncodedCloudGrid {
            frame,
            spec,
            input_count: count,
            cell_count,
        }))
    }

    /// Encode a grid from an earlier typed cloud channel in this batch.
    ///
    /// ```compile_fail
    /// use molla_sensors::{CloudGrid, CloudGridSpec, PendingCloudGridInput, SensorCommandBatch, SensorFramePool};
    /// fn reuse<'a>(grid: &'a mut CloudGrid, batch: &mut SensorCommandBatch<'a>, input: PendingCloudGridInput<'_>, spec: CloudGridSpec, pool: &mut SensorFramePool) {
    ///     let again = PendingCloudGridInput { frame: input.frame, channel: input.channel, points: input.points.clone() };
    ///     grid.encode_pending(batch, input, spec, pool, Default::default()).unwrap();
    ///     grid.encode_pending(batch, again, spec, pool, Default::default()).unwrap();
    /// }
    /// ```
    pub fn encode_pending<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        input: PendingCloudGridInput<'_>,
        spec: CloudGridSpec,
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedCloudGrid>> {
        self.points.check_device()?;
        batch.check_device(&self.device)?;
        let count = batch
            .pending_range::<CloudPoint>(input.frame, input.channel, input.points.clone())?
            .len();
        let params = self.parameters(count, spec, stamp)?;
        let cell_count = params.counts[1] as usize;
        let specs = self.output_specs();
        let ranges = [0..count, 0..count, 0..cell_count, 0..6];
        let frame = batch.encode_from_pending(
            &[input.frame],
            pool,
            &specs,
            &ranges,
            stamp,
            |encoder, inputs, outputs| {
                let channel = inputs.channel::<CloudPoint>(0, input.channel)?;
                let view = channel.slice(input.points)?;
                if !view.is_empty() {
                    encoder.copy_buffer_to_buffer(
                        view.binding()?.buffer,
                        view.active_range().start as u64 * 112,
                        &outputs[0],
                        0,
                        count as u64 * 112,
                    );
                }
                self.encode_direct(encoder, outputs, params)
            },
        )?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        self.stats.encoded_frames += 1;
        self.stats.copied_bytes += count as u64 * 112;
        Ok(Some(EncodedCloudGrid {
            frame,
            spec,
            input_count: count,
            cell_count,
        }))
    }

    fn encode_direct(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        outputs: &[wgpu::Buffer],
        params: CloudGridParams,
    ) -> Result<u64> {
        if params.counts[0] == 0 {
            return Ok(0);
        }
        let buffers = [&outputs[0], &outputs[1], &outputs[2], &outputs[3]];
        let (bindings, prepared) = self.direct.try_get_or_prepare(
            |bindings| bindings.matches(buffers),
            || self.kernel.prepare(&self.device, buffers),
        )?;
        self.stats.binding_preparations += u64::from(prepared);
        self.kernel
            .encode_prepared(&self.queue, encoder, bindings, params)?;
        self.stats.kernel_dispatches += 5;
        Ok(5)
    }

    fn validate_source(&self, source: &ReadbackSource<'_>) -> Result<usize> {
        if !Arc::ptr_eq(&self.device, &source.device)
            || source.channel.kind != TypeId::of::<CloudPoint>()
            || source.channel.element_size != 112
            || !source.channel.bytes.is_multiple_of(112)
            || (source.channel.bytes != 0 && source.buffer.is_none())
        {
            return Err(Error::Solver(
                "cloud grid: invalid source identity or type".into(),
            ));
        }
        Ok(source.channel.bytes / 112)
    }

    fn parameters(
        &self,
        count: usize,
        spec: CloudGridSpec,
        stamp: ReadbackStamp,
    ) -> Result<CloudGridParams> {
        let mut params = spec.parameters(self.config.max_cells)?;
        if count > self.config.max_points as usize {
            return Err(Error::Solver("cloud grid: point capacity exceeded".into()));
        }
        if stamp
            .simulation_time
            .is_some_and(|t| !t.is_finite() || t < 0.0)
        {
            return Err(Error::Solver("cloud grid: invalid timestamp".into()));
        }
        params.counts[0] = count as u32;
        Ok(params)
    }

    /// Copies the selected packed range, then classifies, prefixes, scatters and reduces on device.
    pub fn process(
        &mut self,
        source: ReadbackSource<'_>,
        spec: CloudGridSpec,
        stamp: ReadbackStamp,
    ) -> Result<CloudGridFrame<'_>> {
        self.points.check_device()?;
        let count = self.validate_source(&source)?;
        let params = self.parameters(count, spec, stamp)?;
        let cell_count = params.counts[1] as usize;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::cloud grid"),
            });
        encoder.clear_buffer(self.diagnostics.device_buffer()?, 0, None);
        if count == 0 {
            encoder.clear_buffer(self.cells.device_buffer()?, 0, Some(cell_count as u64 * 64));
        } else {
            let buffer = source
                .buffer
                .ok_or_else(|| Error::Solver("cloud grid: missing input buffer".into()))?;
            encoder.copy_buffer_to_buffer(
                buffer,
                source.offset,
                self.points.device_buffer()?,
                0,
                source.channel.bytes as u64,
            );
            self.kernel.encode(&self.queue, &mut encoder, params)?;
        }
        self.queue.submit([encoder.finish()]);
        if let Some(lease) = source.lease {
            self.queue.on_submitted_work_done(move || drop(lease));
        }
        self.cells.mark_device_dirty();
        self.diagnostics.mark_device_dirty();
        if count != 0 {
            self.points.mark_device_dirty();
            self.indices.mark_device_dirty();
        }
        self.stats.frames += 1;
        self.stats.submissions += 1;
        self.stats.copied_bytes += source.channel.bytes as u64;
        self.stats.kernel_dispatches += if count == 0 { 0 } else { 5 };
        Ok(CloudGridFrame {
            grid: self,
            spec,
            stamp,
            input_count: count,
            cell_count,
        })
    }
}

pub(crate) fn checked_budget(config: CloudGridConfig, limits: &wgpu::Limits) -> Result<u64> {
    let n = config.max_points as u64;
    let c = config.max_cells as u64;
    let blocks = c.div_ceil(256);
    let bytes = 120 * n + 76 * c + 8 * blocks + 104;
    let groups = limits.max_compute_workgroups_per_dimension as u64;
    if n == 0
        || c == 0
        || n.max(c).div_ceil(256) > groups
        || c > groups * groups
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_storage_size < 13_312
        || limits.max_storage_buffers_per_shader_stage < 7
        || bytes > config.max_buffer_bytes
        || [112 * n, 4 * n, 64 * c, 12 * c, 8 * blocks, 24, 80]
            .into_iter()
            .any(|b| b > limits.max_buffer_size || b > limits.max_storage_buffer_binding_size)
    {
        return Err(Error::Gpu(
            "cloud grid: memory or device budget exceeded".into(),
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
        let config = CloudGridConfig {
            max_points: 257,
            max_cells: 17,
            max_buffer_bytes: u64::MAX,
        };
        let usage =
            ProcessorStorageUsage::for_grid(config, &engine.device().unwrap().limits()).unwrap();
        assert!(
            CloudGrid::new_with_storage_limits(
                &engine,
                config,
                ProcessorStorageLimits {
                    max_construction_bytes: usage.construction_bytes - 1,
                }
            )
            .is_err()
        );
        let processor = CloudGrid::new_with_storage_limits(
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
                buffer_bytes(&processor.points.buffer),
                buffer_bytes(&processor.indices.buffer),
                buffer_bytes(&processor.cells.buffer),
                buffer_bytes(&processor.diagnostics.buffer),
            ],
        );
    }

    #[test]
    fn grid_wire_budget_and_limits() {
        assert_eq!(std::mem::size_of::<CloudGridCell>(), 64);
        assert_eq!(std::mem::size_of::<CloudGridParams>(), 80);
        let mut limits = wgpu::Limits::default();
        let mut config = CloudGridConfig {
            max_points: 257,
            max_cells: 257,
            max_buffer_bytes: 196 * 257 + 16 + 104,
        };
        assert_eq!(
            checked_budget(config, &limits).unwrap(),
            config.max_buffer_bytes
        );
        config.max_buffer_bytes -= 1;
        assert!(checked_budget(config, &limits).is_err());
        config.max_buffer_bytes += 1;
        limits.max_compute_workgroups_per_dimension = 2;
        assert!(checked_budget(config, &limits).is_err());
        assert!(CloudGridDiagnostics::from_words(&[0; 5]).is_err());
        assert_eq!(
            CloudGridDiagnostics::from_words(&[1, 2, 3, 4, 5, 6])
                .unwrap()
                .numeric_overflow_cells,
            6
        );
    }

    #[test]
    fn grid_configuration_rejects_degenerate_and_unrepresentable_cells() {
        let spec = CloudGridSpec {
            minimum: [0.; 3],
            maximum: [1.; 3],
            dimensions: [2; 3],
        };
        assert_eq!(spec.parameters(8).unwrap().size, [0.5, 0.5, 0.5, 0.]);
        assert!(spec.parameters(7).is_err());
        for dimensions in [[0, 1, 1], [u32::MAX; 3]] {
            assert!(
                CloudGridSpec { dimensions, ..spec }
                    .parameters(u32::MAX)
                    .is_err()
            );
        }
        for maximum in [
            [f32::NAN; 3],
            [f32::INFINITY; 3],
            [0.; 3],
            [-1.; 3],
            [f32::MIN_POSITIVE; 3],
        ] {
            assert!(CloudGridSpec { maximum, ..spec }.parameters(8).is_err());
        }
        assert!(
            CloudGridSpec {
                minimum: [-f32::MAX; 3],
                maximum: [f32::MAX; 3],
                ..spec
            }
            .parameters(8)
            .is_err()
        );
    }
}
