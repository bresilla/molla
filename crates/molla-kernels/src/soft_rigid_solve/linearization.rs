use super::*;

/// Candidates and directions relative to the prepared contact multipliers and rigid reference.
#[derive(Clone, Copy)]
pub struct ContactLinearizationBindings<'a> {
    pub state: SoftRigidSolveBindings<'a>,
    pub rigid_reference: &'a wgpu::Buffer,
    pub soft_direction: &'a wgpu::Buffer,
    pub rigid_direction: &'a wgpu::Buffer,
    pub impulses: &'a wgpu::Buffer,
    pub impulse_direction: &'a wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    particles: u32,
    rows: u32,
    dofs: u32,
    bodies: u32,
    dt: f32,
    recovery: f32,
    max_recovery: f32,
    pad: u32,
}

/// GPU contact residual/action, articulated impulse response and nodal impulse transpose.
///
/// Prepared geometry, row multipliers and rigid reference remain fixed across evaluations.
/// Candidates and directions may change in their bound device buffers. Subtract `soft_terms`
/// from the material residual/action whose inertia target already includes the reference
/// multipliers. `rigid_terms` uses the same reference impulse convention.
pub struct ContactLinearizationKernel {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 9],
    group: Option<wgpu::BindGroup>,
    adjacency: wgpu::Buffer,
    partials: wgpu::Buffer,
    soft: wgpu::Buffer,
    rigid: wgpu::Buffer,
    contact: wgpu::Buffer,
    status: wgpu::Buffer,
    pub(super) particles: u32,
    pub(super) rows: u32,
    pub(super) dofs: u32,
    bodies: u32,
}

impl ContactLinearizationKernel {
    pub fn new(
        device: &wgpu::Device,
        particles: u32,
        rows: u32,
        dofs: u32,
        bodies: u32,
    ) -> Result<Self> {
        let limits = device.limits();
        let adjacency_words = 2 * u64::from(particles) + 1 + 3 * u64::from(rows);
        let sizes = [
            adjacency_words * 4,
            u64::from(rows.div_ceil(64)) * u64::from(dofs) * 8,
            u64::from(particles) * 32,
            u64::from(dofs) * 16,
            u64::from(rows) * 32,
            8,
        ];
        if particles == 0
            || rows == 0
            || bodies == 0
            || !(1..=64).contains(&dofs)
            || adjacency_words > u64::from(u32::MAX)
            || limits.max_storage_buffers_per_shader_stage < 19
            || limits.max_bindings_per_bind_group < 20
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 512
            || dofs > limits.max_compute_workgroups_per_dimension
            || rows.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || particles.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || sizes.iter().any(|size| {
                *size > limits.max_storage_buffer_binding_size || *size > limits.max_buffer_size
            })
        {
            return Err(Error::Build(
                "invalid contact linearization dimensions or device limits".into(),
            ));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("contact linearization"),
            entries: &(0..20)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 19 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 13,
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("contact linearization"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact natural residual and analytic action"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_contact_linearization.wgsl").into(),
            ),
        });
        let pipelines = [
            "count_supports",
            "prefix_supports",
            "scatter_supports",
            "sort_supports",
            "reset_evaluation",
            "rigid_partials",
            "rigid_reduce",
            "soft_transpose",
            "contact_evaluate",
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
        let allocate = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        };
        Ok(Self {
            layout,
            pipelines,
            group: None,
            adjacency: allocate("contact node adjacency", sizes[0]),
            partials: allocate("contact response partials", sizes[1]),
            soft: allocate("contact nodal transpose", sizes[2]),
            rigid: allocate("contact generalized residual", sizes[3]),
            contact: allocate("contact natural residual", sizes[4]),
            status: allocate("contact linearization status", sizes[5]),
            particles,
            rows,
            dofs,
            bodies,
        })
    }

    /// Builds sorted GPU adjacency from the same geometry and timestep as contact preparation.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        solve: &SoftRigidSolveKernel,
        bindings: ContactLinearizationBindings<'_>,
        settings: SoftRigidSolveSettings,
    ) -> Result<()> {
        self.group = None;
        if solve.dofs != self.dofs
            || solve.row_count != self.rows
            || bindings.state.particle_count != self.particles
            || bindings.state.body_count != self.bodies
            || !settings.dt.is_finite()
            || settings.dt <= 0.0
            || !(1.0 / settings.dt).is_finite()
            || !settings.penetration_recovery.is_finite()
            || !(0.0..=1.0).contains(&settings.penetration_recovery)
            || !settings.max_recovery_speed.is_finite()
            || settings.max_recovery_speed < 0.0
        {
            return Err(Error::Build(
                "contact linearization preparation does not match its dimensions or timestep"
                    .into(),
            ));
        }
        let inputs = [
            bindings.state.contacts,
            solve.row_state(),
            solve.projected_rows(),
            solve.row_responses(),
            bindings.state.body_poses,
            bindings.state.inverse_mass,
            bindings.state.velocities,
            bindings.state.rigid_velocities,
            bindings.rigid_reference,
            bindings.soft_direction,
            bindings.rigid_direction,
            bindings.impulses,
            bindings.impulse_direction,
        ];
        let n = u64::from(self.particles);
        let r = u64::from(self.rows);
        let d = u64::from(self.dofs);
        let minimums = [
            r * 80,
            r * 32,
            r * 3 * d * 4,
            r * 3 * d * 4,
            u64::from(self.bodies) * 32,
            n * 4,
            n * 16,
            d * 4,
            d * 4,
            n * 16,
            d * 4,
            r * 16,
            r * 16,
        ];
        let outputs = [
            &self.adjacency,
            &self.partials,
            &self.soft,
            &self.rigid,
            &self.contact,
            &self.status,
        ];
        for (buffer, minimum) in inputs.iter().zip(minimums) {
            if buffer.size() < minimum
                || buffer.size() > device.limits().max_storage_buffer_binding_size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                || outputs.contains(buffer)
            {
                return Err(Error::Build(
                    "invalid or aliased contact linearization input buffer".into(),
                ));
            }
        }
        let uniform = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("contact linearization parameters"),
                usage: wgpu::BufferUsages::UNIFORM,
                contents: bytemuck::bytes_of(&Uniforms {
                    particles: self.particles,
                    rows: self.rows,
                    dofs: self.dofs,
                    bodies: self.bodies,
                    dt: settings.dt,
                    recovery: settings.penetration_recovery,
                    max_recovery: settings.max_recovery_speed,
                    pad: 0,
                }),
            }
        );
        let buffers: Vec<_> = inputs
            .into_iter()
            .chain(outputs)
            .chain([&uniform])
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact linearization"),
            layout: &self.layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        encoder.clear_buffer(&self.adjacency, 0, None);
        encoder.copy_buffer_to_buffer(solve.status(), 0, &self.status, 0, 4);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &group, &[]);
            for (pipeline, count) in [
                (0, self.rows.div_ceil(64)),
                (1, 1),
                (2, self.rows.div_ceil(64)),
                (3, self.particles.div_ceil(64)),
                (4, 1),
            ] {
                pass.set_pipeline(&self.pipelines[pipeline]);
                pass.dispatch_workgroups(count, 1, 1);
            }
        }
        self.group = Some(group);
        Ok(())
    }

    /// Evaluates the fixed-geometry contact operator without updating any candidate or physical state.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        let group = self
            .group
            .as_ref()
            .ok_or_else(|| Error::Build("contact linearization is not prepared".into()))?;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, group, &[]);
        for (pipeline, x, y) in [
            (4, 1, 1),
            (5, self.rows.div_ceil(64), self.dofs),
            (6, self.dofs.div_ceil(64), 1),
            (7, self.particles.div_ceil(64), 1),
            (8, self.rows.div_ceil(64), 1),
        ] {
            pass.set_pipeline(&self.pipelines[pipeline]);
            pass.dispatch_workgroups(x, y, 1);
        }
        Ok(())
    }

    /// N mass-inverse impulse changes followed by N mass-inverse impulse directions, as vec4.
    pub fn soft_terms(&self) -> &wgpu::Buffer {
        &self.soft
    }
    /// Per-DOF vec4: residual, action, impulse response, directional impulse response.
    pub fn rigid_terms(&self) -> &wgpu::Buffer {
        &self.rigid
    }
    /// R natural residuals followed by R directional actions, as vec4; inactive components fix impulses to zero.
    pub fn contact_terms(&self) -> &wgpu::Buffer {
        &self.contact
    }
    /// Two u32 words: preparation status and the latest evaluation status.
    pub fn status(&self) -> &wgpu::Buffer {
        &self.status
    }
}
