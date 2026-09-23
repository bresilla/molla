use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

pub struct SoftMaterialIterationBindings<'a> {
    pub input_velocity: &'a wgpu::Buffer,
    pub predicted_velocity: &'a wgpu::Buffer,
    pub corrected_velocity: &'a wgpu::Buffer,
    pub previous_body_velocity: &'a wgpu::Buffer,
    pub corrected_body_velocity: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub particles: u32,
    pub bodies: u32,
}

/// Accumulates contact velocity increments and monitors the last fixed-point correction.
pub struct SoftMaterialIterationKernel {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 2],
    diagnostics: wgpu::Buffer,
}

impl SoftMaterialIterationKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let entries = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 1..=4),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect::<Vec<_>>();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FEM material contact iteration"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FEM material contact iteration"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/soft_material_iteration.wgsl").into(),
            ),
        });
        let pipelines = ["accumulate", "check_bodies"].map(|name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let diagnostics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FEM material contact residuals"),
            size: 12,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            layout,
            pipelines,
            diagnostics,
        }
    }

    /// Maximum soft linear, rigid linear and rigid angular correction, as three f32 values.
    pub fn diagnostics(&self) -> &wgpu::Buffer {
        &self.diagnostics
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        b: SoftMaterialIterationBindings<'_>,
        linear_tolerance: f32,
        angular_tolerance: f32,
        final_iteration: bool,
    ) -> Result<()> {
        if !linear_tolerance.is_normal()
            || linear_tolerance <= 0.0
            || !angular_tolerance.is_normal()
            || angular_tolerance <= 0.0
            || b.status.size() < 4
            || b.particles == 0
            || b.bodies == 0
            || b.particles.max(b.bodies).div_ceil(64)
                > device.limits().max_compute_workgroups_per_dimension
            || [b.input_velocity, b.predicted_velocity, b.corrected_velocity]
                .iter()
                .any(|v| v.size() < b.particles as u64 * 16)
            || [b.previous_body_velocity, b.corrected_body_velocity]
                .iter()
                .any(|v| v.size() < b.bodies as u64 * 32)
        {
            return Err(Error::Build(
                "invalid FEM material contact iteration inputs".into(),
            ));
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                b.particles,
                b.bodies,
                u32::from(final_iteration),
                0,
                linear_tolerance.to_bits(),
                angular_tolerance.to_bits(),
                0,
                0,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            b.input_velocity,
            b.predicted_velocity,
            b.corrected_velocity,
            b.previous_body_velocity,
            b.corrected_body_velocity,
            b.status,
            &self.diagnostics,
            &uniform,
        ];
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(i, v)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: v.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        encoder.clear_buffer(&self.diagnostics, 0, None);
        for (pipeline, count) in self.pipelines.iter().zip([b.particles, b.bodies]) {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        Ok(())
    }
}
