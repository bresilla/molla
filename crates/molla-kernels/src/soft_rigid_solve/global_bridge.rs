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
            || dimension * 4 > u64::from(limits.max_storage_buffer_binding_size)
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
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        Ok(Self {
            source_rows,
            capacity,
            particles,
            dofs,
            compact,
            contacts: allocate("compact Newton contacts", u64::from(capacity) * 80),
            impulses: allocate("Newton contact multipliers", u64::from(capacity) * 16),
            inverse_diagonal: allocate("mixed Newton preconditioner", dimension * 4),
            gate: allocate("global contact commit gate", 4),
            uniform: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[source_rows, capacity, particles, dofs]),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            layouts,
            pipelines,
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
    pub fn impulses(&self) -> &wgpu::Buffer {
        &self.impulses
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
