//! GPU stage-resolved linear momentum diagnostics for particles and rigid COM velocities.

use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

pub struct MomentumBindings<'a> {
    pub particle_mass: &'a wgpu::Buffer,
    pub particle_velocity: &'a wgpu::Buffer,
    pub body_mass: &'a wgpu::Buffer,
    pub body_velocity: &'a wgpu::Buffer,
    pub particles: u32,
    pub bodies: u32,
    pub generalized: Option<GeneralizedMomentumBindings<'a>>,
}

pub struct GeneralizedMomentumBindings<'a> {
    pub joint_velocity: &'a wgpu::Buffer,
    /// Body-major COM Jacobians, six rows by DOF count.
    pub jacobians: &'a wgpu::Buffer,
    pub dofs: u32,
}

pub struct MomentumMonitor {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    diagnostics: wgpu::Buffer,
}

impl MomentumMonitor {
    pub fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 5 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("momentum diagnostics bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("momentum diagnostics layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("momentum diagnostics"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/momentum.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("momentum record"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("record"),
            compilation_options: Default::default(),
            cache: None,
        });
        let diagnostics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("momentum stage totals and deltas"),
            size: 20 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            layout,
            pipeline,
            diagnostics,
        }
    }

    /// Twenty vec4s: five soft/rigid phase pairs, then their cumulative phase deltas (phase zero unused).
    pub fn diagnostics(&self) -> &wgpu::Buffer {
        &self.diagnostics
    }

    /// Record phases 0 through 4 in order for each substep; only diagnostic buffers are modified.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: MomentumBindings<'_>,
        phase: u32,
    ) -> Result<()> {
        let b = bindings;
        if phase > 4
            || b.particle_mass.size() < b.particles.max(1) as u64 * 4
            || b.particle_velocity.size() < b.particles.max(1) as u64 * 16
            || b.body_mass.size() < b.bodies.max(1) as u64 * 4
            || b.body_velocity.size() < b.bodies.max(1) as u64 * 32
        {
            return Err(Error::Build(
                "invalid momentum diagnostics bindings or phase".into(),
            ));
        }
        let (joint_velocity, jacobians, dofs) = match &b.generalized {
            Some(g) => {
                if !(1..=64).contains(&g.dofs)
                    || g.joint_velocity.size() < g.dofs as u64 * 4
                    || g.jacobians.size() < b.bodies.max(1) as u64 * 6 * g.dofs as u64 * 4
                {
                    return Err(Error::Build("invalid generalized momentum bindings".into()));
                }
                (g.joint_velocity, g.jacobians, g.dofs)
            }
            None => (b.body_mass, b.body_mass, 0),
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("momentum phase"),
            contents: bytemuck::cast_slice(&[b.particles, b.bodies, phase, dofs]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            b.particle_mass,
            b.particle_velocity,
            b.body_mass,
            b.body_velocity,
            &self.diagnostics,
            &uniform,
            joint_velocity,
            jacobians,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("momentum record bindings"),
            layout: &self.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("momentum record"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }
}
