//! Device-resident articulated FEM contact response and Coulomb iterations.

use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

const PREPARE_BATCH_GROUPS: u32 = 8;

/// GPU start/end ticks for factor, prepare, count, offset, scatter, solve and recovery.
pub struct SoftRigidSolveProfile {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
}

impl SoftRigidSolveProfile {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(Error::Build(
                "contact profiling requires timestamp queries".into(),
            ));
        }
        Ok(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("FEM contact pass timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 14,
            }),
            resolved: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FEM contact pass ticks"),
                size: 14 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        })
    }

    /// Fourteen u64 ticks, resolved when the profiled GPU submission completes.
    pub fn ticks(&self) -> &wgpu::Buffer {
        &self.resolved
    }
}

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
    base_row: u32,
    pad: [u32; 3],
}

#[derive(Clone, Copy)]
pub struct SoftRigidSolveSettings {
    pub iterations: u32,
    pub dt: f32,
    /// Fraction of contact penetration or spherical-anchor separation corrected per step.
    pub penetration_recovery: f32,
    /// Contact-normal speed cap; spherical loops clamp each world-axis component.
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
    pipelines: [wgpu::ComputePipeline; 7],
    factor: wgpu::Buffer,
    projected: wgpu::Buffer,
    response: wgpu::Buffer,
    rows: wgpu::Buffer,
    status: wgpu::Buffer,
    active_rows: wgpu::Buffer,
    dofs: u32,
    row_count: u32,
}

impl SoftRigidSolveKernel {
    pub fn new(device: &wgpu::Device, dofs: u32, row_count: u32) -> Result<Self> {
        Self::new_cooperative(device, dofs, row_count)
    }

    /// Single-invocation reference sweep with storage-resident rigid velocities.
    pub fn new_serial(device: &wgpu::Device, dofs: u32, row_count: u32) -> Result<Self> {
        Self::build(device, dofs, row_count, false)
    }

    /// Cooperative row loading with ordered contact updates and cached rigid velocities.
    pub fn new_cooperative(device: &wgpu::Device, dofs: u32, row_count: u32) -> Result<Self> {
        if device.limits().max_compute_invocations_per_workgroup < 64
            || device.limits().max_compute_workgroup_size_x < 64
            || device.limits().max_compute_workgroup_storage_size < 2048
        {
            return Err(Error::Build(
                "unsupported cooperative contact workgroup".into(),
            ));
        }
        Self::build(device, dofs, row_count, true)
    }

    fn build(device: &wgpu::Device, dofs: u32, row_count: u32, cooperative: bool) -> Result<Self> {
        if !(1..=64).contains(&dofs)
            || row_count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
            || device.limits().max_storage_buffers_per_shader_stage < 16
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
        let entries: Vec<_> = (0..17)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 14 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: !matches!(binding, 1 | 8..=13 | 15..=16),
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
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FEM two-way contact"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    include_str!("../shaders/soft_rigid_solve.wgsl"),
                    include_str!("../shaders/soft_rigid_cooperative.wgsl")
                )
                .into(),
            ),
        });
        let pipelines = [
            "factor_mass",
            "prepare",
            "count_contacts",
            "offset_contacts",
            "scatter_contacts",
            if cooperative {
                "solve_cooperative"
            } else {
                "solve"
            },
            "recover_body_velocity",
        ]
        .map(|entry| {
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
            active_rows: allocate(
                "FEM stable active contact rows",
                (row_count as u64 + 1 + row_count.div_ceil(64) as u64) * 4,
            ),
            dofs,
            row_count,
        })
    }

    /// Sticky bits: 1 invalid mass, 2 invalid input/response, 4 nonfinite solve, 16 contact overflow.
    pub fn status(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// Eight floats per contact: normal inverse mass, tangent bound, target, pad, three impulses, pad.
    /// Ball rows store XYZ effective inverse masses in the first three floats.
    pub fn row_state(&self) -> &wgpu::Buffer {
        &self.rows
    }

    /// Active count followed by contact indices in ascending candidate order; scratch offsets follow.
    pub fn active_rows(&self) -> &wgpu::Buffer {
        &self.active_rows
    }

    /// Encode mass factorization, parallel response assembly and contact sweeps without host readback.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
    ) -> Result<()> {
        self.encode_internal(device, encoder, bindings, settings, 0..=6, None, None)
    }

    /// Encode a nonempty full solve with seven GPU timestamp pairs and no host readback.
    pub fn encode_profiled(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
        profile: &SoftRigidSolveProfile,
    ) -> Result<()> {
        if self.row_count == 0 || bindings.body_count == 0 {
            return Err(Error::Build(
                "contact profiling requires nonempty passes".into(),
            ));
        }
        self.encode_internal(
            device,
            encoder,
            bindings,
            settings,
            0..=6,
            None,
            Some(profile),
        )?;
        encoder.resolve_query_set(&profile.queries, 0..14, &profile.resolved, 0);
        Ok(())
    }

    /// Encode mass factorization and contact responses without applying impulses.
    pub fn encode_preparation(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
    ) -> Result<()> {
        self.encode_internal(device, encoder, bindings, settings, 0..=1, None, None)
    }

    /// Submit individual contact passes and report queue completion without state readback.
    pub fn encode_with_completion(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
        completed: std::sync::Arc<dyn Fn(usize) + Send + Sync>,
    ) -> Result<()> {
        self.encode_internal(
            device,
            encoder,
            bindings,
            settings,
            0..=6,
            Some((queue, completed)),
            None,
        )
    }

    /// Recover COM body velocities from the current Jacobians and generalized velocities.
    pub fn encode_velocity_recovery(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
    ) -> Result<()> {
        self.encode_internal(device, encoder, bindings, settings, 6..=6, None, None)
    }

    fn encode_internal(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bindings: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
        stages: std::ops::RangeInclusive<usize>,
        trace: Option<(&wgpu::Queue, std::sync::Arc<dyn Fn(usize) + Send + Sync>)>,
        profile: Option<&SoftRigidSolveProfile>,
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
            (
                b.contacts,
                self.row_count.max(1) as u64
                    * std::mem::size_of::<crate::soft_rigid_contact::SoftRigidContactGpu>() as u64,
            ),
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
            base_row: 0,
            pad: [0; 3],
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
            &self.active_rows,
        ];
        let bind = |uniform: &wgpu::Buffer| {
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: if i == 14 {
                        uniform.as_entire_binding()
                    } else {
                        b.as_entire_binding()
                    },
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FEM solve bindings"),
                layout: &self.layout,
                entries: &entries,
            })
        };
        let group = bind(&uniform);
        for (index, (pipeline, count)) in self
            .pipelines
            .iter()
            .zip([
                1,
                self.row_count.div_ceil(64),
                self.row_count.div_ceil(64),
                1,
                self.row_count.div_ceil(64),
                1,
                b.body_count.div_ceil(64),
            ])
            .enumerate()
        {
            if count == 0 || !stages.contains(&index) {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FEM two-way solve"),
                timestamp_writes: profile.map(|profile| wgpu::ComputePassTimestampWrites {
                    query_set: &profile.queries,
                    beginning_of_pass_write_index: Some(index as u32 * 2),
                    end_of_pass_write_index: Some(index as u32 * 2 + 1),
                }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            if index == 1 {
                for base in (0..count).step_by(PREPARE_BATCH_GROUPS as usize) {
                    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("FEM response batch parameters"),
                        contents: bytemuck::bytes_of(&Params {
                            base_row: base * 64,
                            ..params
                        }),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                    let group = bind(&uniform);
                    pass.set_bind_group(0, &group, &[]);
                    pass.dispatch_workgroups((count - base).min(PREPARE_BATCH_GROUPS), 1, 1);
                }
            } else {
                pass.dispatch_workgroups(count, 1, 1);
            }
            drop(pass);
            if let Some((queue, completed)) = &trace {
                let pending =
                    std::mem::replace(encoder, device.create_command_encoder(&Default::default()));
                queue.submit([pending.finish()]);
                let completed = completed.clone();
                queue.on_submitted_work_done(move || completed(index));
            }
        }
        Ok(())
    }
}

pub struct SoftParticleDriftKernel {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl SoftParticleDriftKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 4 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 2,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FEM drift bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FEM drift layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FEM drift"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soft_particle_drift.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FEM drift"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("drift"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { layout, pipeline }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        output: &wgpu::Buffer,
        status: &wgpu::Buffer,
        count: u32,
        dt: f32,
    ) -> Result<()> {
        if !dt.is_finite()
            || dt <= 0.0
            || status.size() < 4
            || count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
            || [input, velocities, output]
                .iter()
                .any(|b| b.size() < count.max(1) as u64 * 16)
        {
            return Err(Error::Build("invalid GPU particle drift input".into()));
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FEM drift parameters"),
            contents: bytemuck::cast_slice(&[count, dt.to_bits(), 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [input, velocities, output, status, &uniform];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FEM drift"),
            layout: &self.layout,
            entries: &entries,
        });
        if count > 0 {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        Ok(())
    }
}
