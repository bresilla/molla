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
    pipelines: [wgpu::ComputePipeline; 3],
    diagnostics: wgpu::Buffer,
    force_layout: wgpu::BindGroupLayout,
    force_pipeline: wgpu::ComputePipeline,
    force_diagnostics: wgpu::Buffer,
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
        let pipelines = ["accumulate", "check_bodies", "check_particles"].map(|name| {
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
        let entries = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 3 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 0,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect::<Vec<_>>();
        let force_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FEM force residual"),
            entries: &entries,
        });
        let force_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&force_layout)],
                immediate_size: 0,
            });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FEM force residual"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/soft_material_force.wgsl").into(),
            ),
        });
        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FEM force residual"),
            layout: Some(&force_pipeline_layout),
            module: &shader,
            entry_point: Some("check_force"),
            compilation_options: Default::default(),
            cache: None,
        });
        let force_diagnostics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FEM maximum force residual"),
            size: 4,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Self {
            layout,
            pipelines,
            diagnostics,
            force_layout,
            force_pipeline,
            force_diagnostics,
        }
    }

    /// Maximum soft linear, rigid linear and rigid angular correction, as three f32 values.
    pub fn diagnostics(&self) -> &wgpu::Buffer {
        &self.diagnostics
    }

    /// Maximum velocity-equivalent constitutive residual, as one f32.
    pub fn force_diagnostics(&self) -> &wgpu::Buffer {
        &self.force_diagnostics
    }

    pub fn encode_force_check(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        values: &wgpu::Buffer,
        status: &wgpu::Buffer,
        count: u32,
        tolerance: f32,
    ) -> Result<()> {
        if count == 0
            || values.size() < count as u64 * 16
            || status.size() < 4
            || count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
            || !tolerance.is_normal()
            || tolerance <= 0.0
        {
            return Err(Error::Build("invalid FEM force residual inputs".into()));
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[count, tolerance.to_bits(), 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let entries = [values, status, &self.force_diagnostics, &uniform]
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.force_layout,
            entries: &entries,
        });
        encoder.clear_buffer(&self.force_diagnostics, 0, None);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.force_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        Ok(())
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
        self.encode_internal(
            device,
            encoder,
            b,
            linear_tolerance,
            angular_tolerance,
            final_iteration,
            true,
        )
    }

    /// Check velocity changes without modifying the accumulated contact input.
    pub fn encode_check(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        b: SoftMaterialIterationBindings<'_>,
        linear_tolerance: f32,
        angular_tolerance: f32,
    ) -> Result<()> {
        self.encode_internal(
            device,
            encoder,
            b,
            linear_tolerance,
            angular_tolerance,
            true,
            false,
        )
    }

    fn encode_internal(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        b: SoftMaterialIterationBindings<'_>,
        linear_tolerance: f32,
        angular_tolerance: f32,
        final_iteration: bool,
        accumulate: bool,
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
        for (index, count) in [(if accumulate { 0 } else { 2 }, b.particles), (1, b.bodies)] {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[index]);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        Ok(())
    }
}
