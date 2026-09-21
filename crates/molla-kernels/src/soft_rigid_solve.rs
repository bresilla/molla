//! Device-resident articulated FEM contact response and Coulomb iterations.

use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dofs: u32,
    rows: u32,
    particles: u32,
    bodies: u32,
    iterations: u32,
    dt: f32,
    recovery: f32,
    max_recovery: f32,
}

#[derive(Clone, Copy)]
pub struct SoftRigidSolveSettings {
    pub iterations: u32,
    pub dt: f32,
    pub penetration_recovery: f32,
    pub max_recovery_speed: f32,
}

pub struct SoftRigidSolveBindings<'a> {
    pub positions: &'a wgpu::Buffer,
    pub velocities: &'a wgpu::Buffer,
    pub inverse_mass: &'a wgpu::Buffer,
    pub contacts: &'a wgpu::Buffer,
    pub body_poses: &'a wgpu::Buffer,
    pub body_com: &'a wgpu::Buffer,
    /// Body-major, COM point, six rows by articulation DOF count per body.
    pub jacobians: &'a wgpu::Buffer,
    pub mass_matrix: &'a wgpu::Buffer,
    pub rigid_velocities: &'a wgpu::Buffer,
    pub body_velocities: &'a wgpu::Buffer,
    pub particle_count: u32,
    pub body_count: u32,
}

/// One coupled articulation, at most 64 DOFs, with serialized contact sweeps on device.
pub struct SoftRigidSolveKernel {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 4],
    factor: wgpu::Buffer,
    projected: wgpu::Buffer,
    response: wgpu::Buffer,
    rows: wgpu::Buffer,
    status: wgpu::Buffer,
    dofs: u32,
    row_count: u32,
}

impl SoftRigidSolveKernel {
    pub fn new(device: &wgpu::Device, dofs: u32, row_count: u32) -> Result<Self> {
        if !(1..=64).contains(&dofs)
            || row_count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(Error::Build(
                "unsupported GPU FEM contact dimensions".into(),
            ));
        }
        let response_size = row_count.max(1) as u64 * 3 * dofs as u64 * 4;
        if response_size > device.limits().max_storage_buffer_binding_size as u64
            || row_count.max(1) as u64 * 32 > device.limits().max_storage_buffer_binding_size as u64
        {
            return Err(Error::Build(
                "GPU FEM response exceeds buffer limits".into(),
            ));
        }
        let allocate = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let entries: Vec<_> = (0..16)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 14 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: !matches!(binding, 1 | 8..=13 | 15),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FEM two-way contact bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FEM two-way contact layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FEM two-way contact"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soft_rigid_solve.wgsl").into(),
            ),
        });
        let pipelines = ["factor_mass", "prepare", "solve", "recover_body_velocity"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Ok(Self {
            layout,
            pipelines,
            factor: allocate("FEM rigid mass factor", dofs as u64 * dofs as u64 * 4),
            projected: allocate("FEM contact Jacobians", response_size),
            response: allocate("FEM contact responses", response_size),
            rows: allocate("FEM contact impulses", row_count.max(1) as u64 * 32),
            status: allocate("FEM contact solver status", 4),
            dofs,
            row_count,
        })
    }

    /// Sticky status bits: 1 invalid mass, 2 invalid contact/response, 4 nonfinite solve.
    pub fn status(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// Eight floats per contact: normal inverse mass, tangent bound, target, pad, three impulses, pad.
    pub fn row_state(&self) -> &wgpu::Buffer {
        &self.rows
    }

    /// Encode mass factorization, parallel response assembly and contact sweeps without host readback.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
    ) -> Result<()> {
        if settings.iterations == 0
            || !settings.dt.is_finite()
            || settings.dt <= 0.0
            || !settings.penetration_recovery.is_finite()
            || !(0.0..=1.0).contains(&settings.penetration_recovery)
            || !settings.max_recovery_speed.is_finite()
            || settings.max_recovery_speed < 0.0
        {
            return Err(Error::Build("invalid GPU FEM contact settings".into()));
        }
        let b = &bindings;
        let sizes = [
            (b.positions, b.particle_count.max(1) as u64 * 16),
            (b.velocities, b.particle_count.max(1) as u64 * 16),
            (b.inverse_mass, b.particle_count.max(1) as u64 * 4),
            (b.contacts, self.row_count.max(1) as u64 * 48),
            (b.body_poses, b.body_count.max(1) as u64 * 32),
            (b.body_com, b.body_count.max(1) as u64 * 16),
            (
                b.jacobians,
                b.body_count.max(1) as u64 * 6 * self.dofs as u64 * 4,
            ),
            (b.mass_matrix, self.dofs as u64 * self.dofs as u64 * 4),
            (b.rigid_velocities, self.dofs as u64 * 4),
            (b.body_velocities, b.body_count.max(1) as u64 * 32),
        ];
        if sizes.iter().any(|(buffer, size)| buffer.size() < *size) {
            return Err(Error::Build("undersized GPU FEM solve buffer".into()));
        }
        let params = Params {
            dofs: self.dofs,
            rows: self.row_count,
            particles: b.particle_count,
            bodies: b.body_count,
            iterations: settings.iterations,
            dt: settings.dt,
            recovery: settings.penetration_recovery,
            max_recovery: settings.max_recovery_speed,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FEM solve parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            b.positions,
            b.velocities,
            b.inverse_mass,
            b.contacts,
            b.body_poses,
            b.body_com,
            b.jacobians,
            b.mass_matrix,
            b.rigid_velocities,
            &self.factor,
            &self.projected,
            &self.response,
            &self.rows,
            &self.status,
            &uniform,
            b.body_velocities,
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
            label: Some("FEM solve bindings"),
            layout: &self.layout,
            entries: &entries,
        });
        for (pipeline, count) in self.pipelines.iter().zip([
            1,
            self.row_count.div_ceil(64),
            1,
            b.body_count.div_ceil(64),
        ]) {
            if count == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FEM two-way solve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count, 1, 1);
        }
        Ok(())
    }
}
