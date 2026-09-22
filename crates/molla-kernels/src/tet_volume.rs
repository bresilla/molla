//! GPU lifetime minimum-volume diagnostics for accepted tetrahedral states.

use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

pub struct TetVolumeBindings<'a> {
    pub positions: &'a wgpu::Buffer,
    pub reference: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    pub inverse_rest: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub particle_count: u32,
    pub tet_count: u32,
}

pub struct TetVolumeMonitor {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    diagnostics: wgpu::Buffer,
}

impl TetVolumeMonitor {
    pub fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..7)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 6 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tet volume bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tet volume layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tet volume"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tet_volume.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tet volume"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("measure"),
            compilation_options: Default::default(),
            cache: None,
        });
        let diagnostics = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tet volume lifetime minimum"),
            contents: bytemuck::cast_slice(&[f32::INFINITY.to_bits(), u32::MAX]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        Self {
            layout,
            pipeline,
            diagnostics,
        }
    }

    /// Minimum positive J as f32 bits, then smallest invalid tet index or u32::MAX.
    /// Invalid samples latch minimum to zero; construction resets the lifetime record.
    pub fn diagnostics(&self) -> &wgpu::Buffer {
        &self.diagnostics
    }

    /// Records J without readback; status ORs bit 8 for invalid/inverted tets, bit 4 for nonfinite J.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        b: TetVolumeBindings<'_>,
    ) -> Result<()> {
        let nodes = b.particle_count.max(1) as u64;
        let tets = b.tet_count.max(1) as u64;
        if b.tet_count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
            || [
                (b.positions, nodes * 16),
                (b.reference, nodes * 16),
                (b.indices, tets * 16),
                (b.inverse_rest, tets * 48),
                (b.status, 4),
            ]
            .iter()
            .any(|(buffer, size)| buffer.size() < *size)
        {
            return Err(Error::Build("invalid GPU tet volume buffers".into()));
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tet volume parameters"),
            contents: bytemuck::cast_slice(&[b.tet_count, b.particle_count, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            b.positions,
            b.reference,
            b.indices,
            b.inverse_rest,
            &self.diagnostics,
            b.status,
            &uniform,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tet volume"),
            layout: &self.layout,
            entries: &entries,
        });
        if b.tet_count > 0 {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(b.tet_count.div_ceil(64), 1, 1);
        }
        Ok(())
    }
}
