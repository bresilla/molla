//! Shared particle-bin resources and hierarchical scans for terrain export.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};

pub(crate) struct RasterParams {
    pub particles: u32,
    pub nx: u32,
    pub nz: u32,
    pub terrain: [f32; 4],
    pub extra: [f32; 4],
    pub attribute_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    dims: [u32; 4],
    level: [u32; 4],
    terrain: [f32; 4],
    velocity: [f32; 4],
}

pub(crate) struct ParticleRasterKernel {
    uniforms_layout: wgpu::BindGroupLayout,
    io_layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 6],
}

/// Fixed-buffer terrain export with reusable bins and immutable parameters.
///
/// Encode invocations on one queue in order. Input/output buffers and dimensions
/// remain fixed; create another prepared export when their identities change.
pub(crate) struct PreparedParticleRaster {
    pipelines: [wgpu::ComputePipeline; 6],
    io: wgpu::BindGroup,
    levels: Vec<(wgpu::BindGroup, u32)>,
    bins: wgpu::Buffer,
    particles: u32,
    cells: u32,
    max_groups: u32,
}

impl ParticleRasterKernel {
    pub(crate) fn new(device: &wgpu::Device, body: &str) -> Self {
        let uniforms_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil raster uniforms layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding < 3,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let io_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil raster io layout"),
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil raster pipeline layout"),
            bind_group_layouts: &[Some(&uniforms_layout), Some(&io_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil raster shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{body}",
                    include_str!("../shaders/soil_raster_bins.wgsl")
                )
                .into(),
            ),
        });
        let pipelines = [
            "histogram",
            "scan_blocks",
            "scan_fixup",
            "scatter",
            "reduce_cells",
            "reference_cells",
        ]
        .map(|name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Self {
            uniforms_layout,
            io_layout,
            pipelines,
        }
    }

    pub(crate) fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 6],
        params: &RasterParams,
    ) -> Result<PreparedParticleRaster> {
        let cells = params
            .nx
            .checked_mul(params.nz)
            .filter(|&n| n > 0 && params.nx <= i32::MAX as u32 && params.nz <= i32::MAX as u32)
            .ok_or_else(|| Error::Solver("soil raster: invalid dimensions".into()))?;
        if !params
            .terrain
            .iter()
            .chain(&params.extra)
            .all(|v| v.is_finite())
            || params.terrain[2] <= 0.0
        {
            return Err(Error::Solver("soil raster: invalid parameters".into()));
        }
        let limits = device.limits();
        let mut layout = Vec::new();
        let mut base = 0u32;
        let mut len = cells;
        loop {
            layout.push((base, len));
            base = base
                .checked_add(len)
                .ok_or_else(|| Error::Solver("soil raster: scan size overflow".into()))?;
            if len <= 64 {
                break;
            }
            len = len.div_ceil(64);
        }
        let bins_bytes = u64::from(base) * 12;
        let members_bytes = u64::from(params.particles.max(1)) * 4;
        let sizes = [
            u64::from(params.particles.max(1)) * 16,
            params.attribute_bytes,
            u64::from(params.particles.max(1)) * 16,
            u64::from(cells) * 4,
            u64::from(cells) * 4,
            u64::from(cells) * 16,
        ];
        for (buffer, size) in buffers.iter().zip(sizes) {
            if buffer.size() < size
                || buffer.size() > u64::from(limits.max_storage_buffer_binding_size)
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(Error::Solver(
                    "soil raster: invalid storage buffer size/usage".into(),
                ));
            }
        }
        for bytes in [bins_bytes, members_bytes] {
            if bytes > limits.max_buffer_size
                || bytes > u64::from(limits.max_storage_buffer_binding_size)
            {
                return Err(Error::Solver(
                    "soil raster: scratch exceeds device limits".into(),
                ));
            }
        }
        let max_groups = limits.max_compute_workgroups_per_dimension;
        if u64::from(cells.max(params.particles.div_ceil(64))) > u64::from(max_groups).pow(2) {
            return Err(Error::Solver(
                "soil raster: dispatch exceeds device limits".into(),
            ));
        }
        let bins = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil raster bins"),
                size: bins_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let members = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil raster members"),
                size: members_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }
        );
        let entries: Vec<_> = buffers
            .into_iter()
            .chain([&bins, &members])
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil raster prepared io"),
            layout: &self.io_layout,
            entries: &entries,
        });
        let levels = layout
            .iter()
            .enumerate()
            .map(|(i, &(base, len))| {
                let parent = layout.get(i + 1).map(|&(base, _)| base);
                let data = Uniforms {
                    dims: [params.particles, params.nx, params.nz, max_groups],
                    level: [base, len, parent.unwrap_or(0), u32::from(parent.is_some())],
                    terrain: params.terrain,
                    velocity: params.extra,
                };
                let uniform = molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("soil raster immutable params"),
                        size: std::mem::size_of::<Uniforms>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM,
                        mapped_at_creation: true,
                    }
                );
                uniform
                    .slice(..)
                    .get_mapped_range_mut()
                    .copy_from_slice(bytemuck::bytes_of(&data));
                uniform.unmap();
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("soil raster prepared params"),
                    layout: &self.uniforms_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    }],
                });
                (group, len)
            })
            .collect();
        Ok(PreparedParticleRaster {
            pipelines: self.pipelines.clone(),
            io,
            levels,
            bins,
            particles: params.particles,
            cells,
            max_groups,
        })
    }
}

impl PreparedParticleRaster {
    /// Records the exhaustive reference over the same buffers and parameters.
    pub fn encode_reference(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("soil raster exhaustive reference"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.levels[0].0, &[]);
        pass.set_bind_group(1, &self.io, &[]);
        pass.set_pipeline(&self.pipelines[5]);
        self.dispatch(&mut pass, self.cells.div_ceil(64));
    }

    /// Records a complete export without allocation, submission, or readback.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.bins, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("soil raster binned export"),
            timestamp_writes: None,
        });
        pass.set_bind_group(1, &self.io, &[]);
        pass.set_bind_group(0, &self.levels[0].0, &[]);
        pass.set_pipeline(&self.pipelines[0]);
        self.dispatch(&mut pass, self.particles.div_ceil(64));
        pass.set_pipeline(&self.pipelines[1]);
        for (params, len) in &self.levels {
            pass.set_bind_group(0, params, &[]);
            self.dispatch(&mut pass, len.div_ceil(64));
        }
        pass.set_pipeline(&self.pipelines[2]);
        for (params, len) in self.levels.iter().rev().skip(1) {
            pass.set_bind_group(0, params, &[]);
            self.dispatch(&mut pass, len.div_ceil(64));
        }
        pass.set_bind_group(0, &self.levels[0].0, &[]);
        pass.set_pipeline(&self.pipelines[3]);
        self.dispatch(&mut pass, self.particles.div_ceil(64));
        pass.set_pipeline(&self.pipelines[4]);
        self.dispatch(&mut pass, self.cells);
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, groups: u32) {
        if groups != 0 {
            pass.dispatch_workgroups(
                groups.min(self.max_groups),
                groups.div_ceil(self.max_groups),
                1,
            );
            molla_core::gpu_metrics::record_dispatch();
        }
    }
}
