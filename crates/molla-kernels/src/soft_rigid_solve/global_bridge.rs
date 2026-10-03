use super::*;

pub struct GlobalContactCommitBindings<'a> {
    pub soft_candidate: &'a wgpu::Buffer,
    pub rigid_candidate: &'a wgpu::Buffer,
    pub soft_terms: &'a wgpu::Buffer,
    pub newton_diagnostics: &'a wgpu::Buffer,
    pub soft_velocity: &'a wgpu::Buffer,
    pub rigid_velocity: &'a wgpu::Buffer,
    pub input_velocity: &'a wgpu::Buffer,
}

/// Bounded GPU compaction of prepared contact rows and validated impulse/state commit.
pub struct GlobalContactBridge {
    source_rows: u32,
    capacity: u32,
    particles: u32,
    dofs: u32,
    compact: SoftRigidSolveKernel,
    contacts: wgpu::Buffer,
    impulses: wgpu::Buffer,
    inverse_diagonal: wgpu::Buffer,
    gate: wgpu::Buffer,
    uniform: wgpu::Buffer,
    layouts: [wgpu::BindGroupLayout; 2],
    pipelines: [[wgpu::ComputePipeline; 3]; 2],
    material_layout: wgpu::BindGroupLayout,
    material_pipeline: wgpu::ComputePipeline,
    projection_pipeline: wgpu::ComputePipeline,
    projection_group: wgpu::BindGroup,
}
impl GlobalContactBridge {
    pub fn new(
        device: &wgpu::Device,
        source_rows: u32,
        capacity: u32,
        particles: u32,
        dofs: u32,
    ) -> Result<Self> {
        let limits = device.limits();
        let dimension = 3 * u64::from(particles) + u64::from(dofs) + 3 * u64::from(capacity);
        if particles == 0
            || capacity == 0
            || capacity > source_rows
            || !(1..=64).contains(&dofs)
            || dimension > u64::from(u32::MAX)
            || dimension.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
            || dimension * 4 > limits.max_storage_buffer_binding_size
            || dimension * 4 > limits.max_buffer_size
            || limits.max_storage_buffers_per_shader_stage < 13
            || limits.max_bindings_per_bind_group < 14
        {
            return Err(Error::Build(
                "invalid global contact bridge dimensions".into(),
            ));
        }
        let compact = SoftRigidSolveKernel::new(device, dofs, capacity)?;
        let layouts = [0, 1].map(|phase| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("global contact bridge"),
                entries: &(0..14)
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        count: None,
                        ty: wgpu::BindingType::Buffer {
                            ty: if binding == 13 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: binding < if phase == 0 { 5 } else { 7 },
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    })
                    .collect::<Vec<_>>(),
            })
        });
        let pipelines = [
            include_str!("../../shaders/fem_global_contact_compact.wgsl"),
            include_str!("../../shaders/fem_global_contact_commit.wgsl"),
        ]
        .into_iter()
        .enumerate()
        .map(|(phase, source)| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layouts[phase])],
                immediate_size: 0,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("global contact bridge"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let entries = if phase == 0 {
                ["initialize", "compact", "preconditioner"]
            } else {
                ["initialize", "validate", "commit"]
            };
            entries.map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
        let allocate = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    mapped_at_creation: false,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                }
            )
        };
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("global material scale"),
            entries: &(0..3)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    count: None,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 2 {
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
                .collect::<Vec<_>>(),
        });
        let material_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("global material scale"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_global_material_scale.wgsl").into(),
            ),
        });
        let material_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("global material scale"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&material_layout)],
                    immediate_size: 0,
                }),
            ),
            module: &material_shader,
            entry_point: Some("pack"),
            compilation_options: Default::default(),
            cache: None,
        });
        let contacts = allocate("compact Newton contacts", u64::from(capacity) * 80);
        let impulses = allocate("Newton contact multipliers", u64::from(capacity) * 16);
        let uniform = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[source_rows, capacity, particles, dofs]),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );
        let projection_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("feasible Newton impulses"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_global_contact_project.wgsl").into(),
            ),
        });
        let projection_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("feasible Newton impulses"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&material_layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &projection_shader,
                entry_point: Some("project"),
                compilation_options: Default::default(),
                cache: None,
            });
        let projection_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("feasible Newton impulses"),
            layout: &material_layout,
            entries: &[&contacts, &impulses, &uniform]
                .into_iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            source_rows,
            capacity,
            particles,
            dofs,
            compact,
            contacts,
            impulses,
            inverse_diagonal: allocate("mixed Newton preconditioner", dimension * 4),
            gate: allocate("global contact commit gate", 4),
            uniform,
            layouts,
            pipelines,
            material_layout,
            material_pipeline,
            projection_pipeline,
            projection_group,
        })
    }
    fn source_matches(&self, source: &SoftRigidSolveKernel) -> Result<()> {
        if source.row_count != self.source_rows || source.dofs != self.dofs {
            return Err(Error::Build(
                "global contact source dimensions changed".into(),
            ));
        }
        Ok(())
    }
    fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        phase: usize,
        buffers: [&wgpu::Buffer; 13],
        counts: [u32; 3],
    ) {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layouts[phase],
            entries: &buffers
                .into_iter()
                .chain([&self.uniform])
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &group, &[]);
        for (pipeline, count) in self.pipelines[phase].iter().zip(counts) {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(count, 1, 1);
        }
    }
    /// Requires a freshly prepared source response and active-row schedule.
    pub fn encode_prepare(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &SoftRigidSolveKernel,
        contacts: &wgpu::Buffer,
    ) -> Result<()> {
        self.source_matches(source)?;
        if contacts.size() < u64::from(self.source_rows) * 80
            || !contacts.usage().contains(wgpu::BufferUsages::STORAGE)
            || contacts == &self.contacts
        {
            return Err(Error::Build(
                "invalid global contact geometry buffer".into(),
            ));
        }
        self.dispatch(
            device,
            encoder,
            0,
            [
                contacts,
                source.active_rows(),
                source.row_state(),
                source.projected_rows(),
                source.row_responses(),
                source.status(),
                &self.contacts,
                self.compact.row_state(),
                self.compact.projected_rows(),
                self.compact.row_responses(),
                self.compact.status(),
                &self.impulses,
                &self.inverse_diagonal,
            ],
            [
                1,
                self.capacity.div_ceil(64),
                (3 * self.particles + self.dofs + 3 * self.capacity).div_ceil(64),
            ],
        );
        Ok(())
    }
    /// Commits accepted candidates and contact-only target increments; faults leave outputs untouched.
    pub fn encode_commit(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &SoftRigidSolveKernel,
        b: GlobalContactCommitBindings<'_>,
    ) -> Result<()> {
        self.source_matches(source)?;
        let external = [
            b.soft_candidate,
            b.rigid_candidate,
            b.soft_terms,
            b.newton_diagnostics,
            b.soft_velocity,
            b.rigid_velocity,
            b.input_velocity,
        ];
        let n = u64::from(self.particles);
        let d = u64::from(self.dofs);
        for (i, (buffer, size)) in external
            .iter()
            .zip([n * 16, d * 4, n * 16, 64, n * 16, d * 4, n * 16])
            .enumerate()
        {
            if buffer.size() < size
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                || external
                    .iter()
                    .enumerate()
                    .any(|(j, other)| i != j && (i >= 4 || j >= 4) && buffer == other)
            {
                return Err(Error::Build(
                    "invalid or aliased global contact commit buffer".into(),
                ));
            }
        }
        let blocks = self
            .particles
            .max(self.dofs)
            .max(self.capacity)
            .div_ceil(64);
        self.dispatch(
            device,
            encoder,
            1,
            [
                source.active_rows(),
                &self.impulses,
                b.soft_candidate,
                b.rigid_candidate,
                b.soft_terms,
                b.newton_diagnostics,
                self.compact.status(),
                source.status(),
                source.row_state(),
                b.soft_velocity,
                b.rigid_velocity,
                b.input_velocity,
                &self.gate,
            ],
            [1, blocks, blocks],
        );
        Ok(())
    }
    pub fn contacts(&self) -> &wgpu::Buffer {
        &self.contacts
    }
    pub fn encode_material_scale(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        diagonal: &wgpu::Buffer,
    ) -> Result<()> {
        if diagonal.size() < u64::from(self.particles) * 16
            || !diagonal.usage().contains(wgpu::BufferUsages::STORAGE)
            || diagonal == &self.inverse_diagonal
        {
            return Err(Error::Build("invalid material scaling buffer".into()));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.material_layout,
            entries: &[diagonal, &self.inverse_diagonal, &self.uniform]
                .into_iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.material_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((3 * self.particles).div_ceil(64), 1, 1);
        Ok(())
    }
    pub fn impulses(&self) -> &wgpu::Buffer {
        &self.impulses
    }
    /// Projects trial impulses before residual evaluation; bilateral rows are unchanged.
    pub fn encode_project(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.projection_pipeline);
        pass.set_bind_group(0, &self.projection_group, &[]);
        pass.dispatch_workgroups(self.capacity.div_ceil(64), 1, 1);
    }
    pub fn compact_solve(&self) -> &SoftRigidSolveKernel {
        &self.compact
    }
    pub fn inverse_diagonal(&self) -> &wgpu::Buffer {
        &self.inverse_diagonal
    }
    pub fn commit_status(&self) -> &wgpu::Buffer {
        &self.gate
    }
}
