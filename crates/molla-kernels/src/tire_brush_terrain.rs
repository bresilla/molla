use super::*;
use molla_compute::{DeviceHealth, Engine};
use std::sync::Arc;

/// Borrowed regular height/friction maps for the unchanged brush model.
pub struct BrushTerrainGpu<'a> {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub heights: &'a wgpu::Buffer,
    pub friction: f32,
    pub friction_map: Option<&'a wgpu::Buffer>,
}

/// Retained brush input/output with caller-encoded copies from GPU wheel producers.
/// Raw buffers/encoders must belong to the engine. GPU rows must satisfy the brush
/// input contract, including terrain mode; map values and sample coordinates must
/// be finite and valid for the legacy sampler. No new physical fault latch is added.
/// Reprepare when terrain metadata, source identities or wheel count change.
pub struct PreparedTireBrushTerrain {
    health: Arc<DeviceHealth>,
    pipeline: wgpu::ComputePipeline,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    bindings: [wgpu::BindGroup; 2],
    count: u32,
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("prepared brush terrain: {message}"))
}

impl PreparedTireBrushTerrain {
    pub fn required_bytes(count: u32) -> u64 {
        224 * u64::from(count) + 36
    }

    /// Prepares fixed-capacity scratch and immutable terrain parameters.
    /// Owned byte accounting excludes borrowed maps and driver allocations.
    pub fn new(
        engine: &Engine,
        kernel: &TireBrushGpuKernel,
        count: u32,
        terrain: Option<&BrushTerrainGpu<'_>>,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU engine required"))?;
        crate::soil_gpu_identity::validate_soil_device(device, &kernel.device)?;
        let limits = device.limits();
        let size = 112 * u64::from(count);
        if count == 0
            || count > limits.max_compute_workgroups_per_dimension
            || Self::required_bytes(count) > max_owned_bytes
            || size > limits.max_buffer_size
            || size > limits.max_storage_buffer_binding_size
        {
            return Err(invalid("wheel count or owned byte budget exceeds limits"));
        }
        let params = if let Some(t) = terrain {
            let cells = u64::from(t.nx) * u64::from(t.nz);
            if cells == 0
                || cells > u64::from(u32::MAX)
                || t.nx > i32::MAX as u32
                || t.nz > i32::MAX as u32
                || !t.spacing.is_finite()
                || t.spacing < 1e-6
                || !t.friction.is_finite()
                || t.origin.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("invalid terrain metadata"));
            }
            for buffer in [Some(t.heights), t.friction_map].into_iter().flatten() {
                if !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                    || buffer.size() < 4 * cells
                    || buffer.size() > limits.max_storage_buffer_binding_size
                {
                    return Err(invalid("terrain storage capacity"));
                }
            }
            TerrainUniformsCpu {
                origin: [t.origin[0], t.origin[1], t.spacing, t.friction.max(1e-6)],
                dims: [t.nx, t.nz, 1, u32::from(t.friction_map.is_some())],
            }
        } else {
            validate_terrain(None, None)?
        };
        let allocate = |size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("prepared brush terrain"),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let input = allocate(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let output = allocate(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let dummy = allocate(4, wgpu::BufferUsages::STORAGE);
        let uniform = allocate(
            32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &uniform,
            0,
            bytemuck::bytes_of(&params)
        );
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let bindings = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prepared brush terrain IO"),
                layout: &kernel.layout_io,
                entries: &[
                    entry(0, &input),
                    entry(1, &output),
                    entry(2, terrain.map_or(&dummy, |t| t.heights)),
                    entry(3, terrain.and_then(|t| t.friction_map).unwrap_or(&dummy)),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prepared brush terrain parameters"),
                layout: &kernel.layout_uniforms,
                entries: &[entry(0, &uniform)],
            }),
        ];
        Ok(Self {
            health: engine.device_health().unwrap().clone(),
            pipeline: kernel.pipeline.clone(),
            input,
            output,
            bindings,
            count,
        })
    }

    pub fn wheel_count(&self) -> usize {
        self.count as usize
    }

    /// Returns a retained GPU handle; subsequent encodes overwrite its contents.
    pub fn output(&self) -> TireBrushResidentOutput {
        TireBrushResidentOutput {
            buffer: self.output.clone(),
            wheel_count: self.wheel_count(),
        }
    }

    pub(crate) fn validate_owner(&self, engine: &Engine, count: usize) -> Result<()> {
        self.health.check()?;
        if count != self.wheel_count()
            || !engine
                .device_health()
                .is_some_and(|h| Arc::ptr_eq(h, &self.health))
        {
            return Err(invalid("foreign engine or changed wheel count"));
        }
        Ok(())
    }

    /// Copies current GPU wheel rows then evaluates the brush without host transfers.
    pub fn encode(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        wheels: &wgpu::Buffer,
    ) -> Result<()> {
        self.encode_dispatch(engine, encoder, wheels, None)
    }

    pub(crate) fn encode_dispatch(
        &self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        wheels: &wgpu::Buffer,
        indirect: Option<(&wgpu::Buffer, u64)>,
    ) -> Result<()> {
        self.validate_owner(engine, self.wheel_count())?;
        if !wheels.usage().contains(wgpu::BufferUsages::COPY_SRC)
            || wheels.size() < self.input.size()
        {
            return Err(invalid("wheel source copy capacity"));
        }
        encoder.copy_buffer_to_buffer(wheels, 0, &self.input, 0, self.input.size());
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings[0], &[]);
        pass.set_bind_group(1, &self.bindings[1], &[]);
        molla_core::gpu_metrics::record_dispatch();
        if let Some((buffer, offset)) = indirect {
            pass.dispatch_workgroups_indirect(buffer, offset);
        } else {
            pass.dispatch_workgroups(self.count, 1, 1);
        }
        Ok(())
    }
}

#[test]
fn prepared_brush_terrain_buffer_contract() {
    assert_eq!(core::mem::size_of::<BrushWheelInput>(), 112);
    assert_eq!(core::mem::size_of::<BrushWheelOutput>(), 112);
    assert_eq!(core::mem::size_of::<TerrainUniformsCpu>(), 32);
    assert_eq!(PreparedTireBrushTerrain::required_bytes(4), 932);
}
