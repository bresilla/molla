use molla_compute::Engine;
use molla_core::{Error, Result};

pub(crate) const NO_GROUP: u32 = u32::MAX;

pub(crate) struct ActivationBuffers<'a> {
    pub rows: &'a wgpu::Buffer,
    pub impacts: &'a wgpu::Buffer,
    pub onsets: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
    pub velocities: &'a [&'a wgpu::Buffer],
    pub accelerations: &'a wgpu::Buffer,
    pub identities: Option<&'a wgpu::Buffer>,
}

pub(crate) struct SelectionMapping {
    group: wgpu::BindGroup,
    count: u32,
}

pub(crate) struct ContactActivationGpu {
    pipelines: [wgpu::ComputePipeline; 2],
    bindings: Vec<wgpu::BindGroup>,
    mask_pipeline: wgpu::ComputePipeline,
    mask_layout: wgpu::BindGroupLayout,
    status: wgpu::Buffer,
    awake: wgpu::Buffer,
    dependencies: wgpu::Buffer,
    watches: wgpu::Buffer,
    params: wgpu::Buffer,
    body_mask: wgpu::Buffer,
    body_mapping: SelectionMapping,
    groups: u32,
    rows: u32,
    bodies: u32,
    identities: u32,
}

fn storage(device: &wgpu::Device, queue: &wgpu::Queue, size: u64, bytes: &[u8]) -> wgpu::Buffer {
    let result = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("contact activation storage"),
            size: size.max(bytes.len() as u64).max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    if !bytes.is_empty() {
        molla_core::gpu_write_buffer!(queue, &result, 0, bytes);
    }
    result
}

fn uniform(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Buffer {
    let result = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("contact activation parameters"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    molla_core::gpu_write_buffer!(queue, &result, 0, bytes);
    result
}

fn layout(device: &wgpu::Device, readonly: &[bool]) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = readonly
        .iter()
        .enumerate()
        .map(|(i, &read_only)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if i + 1 == readonly.len() {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("contact activation layout"),
        entries: &entries,
    })
}

fn group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("contact activation bindings"),
        layout,
        entries: &entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn mapping(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    awake: &wgpu::Buffer,
    status: &wgpu::Buffer,
    sources: &[u32],
    output: &wgpu::Buffer,
) -> SelectionMapping {
    let sources_buffer = storage(device, queue, 4, bytemuck::cast_slice(sources));
    let params = uniform(
        device,
        queue,
        bytemuck::cast_slice(&[sources.len() as u32, 0, 0, 0]),
    );
    SelectionMapping {
        group: group(
            device,
            layout,
            &[awake, &sources_buffer, output, status, &params],
        ),
        count: sources.len() as u32,
    }
}

impl ContactActivationGpu {
    pub(crate) fn new(
        engine: &Engine,
        body_groups: &[u32],
        groups: u32,
        rows: u32,
        buffers: ActivationBuffers<'_>,
    ) -> Result<Self> {
        let invalid = || Error::Gpu("invalid contact activation layout".into());
        let device = engine.device().ok_or_else(invalid)?;
        let queue = engine.queue().unwrap();
        let bodies = u32::try_from(body_groups.len()).map_err(|_| invalid())?;
        let limits = device.limits();
        if buffers.velocities.is_empty()
            || rows == 0
            || rows.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || bodies.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || limits.max_storage_buffers_per_shader_stage < 13
            || body_groups.iter().any(|&g| g != NO_GROUP && g >= groups)
        {
            return Err(invalid());
        }
        for bytes in [
            u64::from(rows) * 128,
            u64::from(rows) * 16,
            u64::from(bodies) * 32,
            u64::from(groups) * 4,
        ] {
            if bytes > limits.max_storage_buffer_binding_size {
                return Err(invalid());
            }
        }
        let check = |buffer: &wgpu::Buffer, bytes: u64| -> Result<()> {
            if buffer.size() < bytes.max(4) || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                Err(invalid())
            } else {
                Ok(())
            }
        };
        check(buffers.rows, u64::from(rows) * 128)?;
        check(buffers.impacts, u64::from(rows) * 128)?;
        check(buffers.onsets, u64::from(rows) * 4)?;
        check(buffers.status, 4)?;
        let identities = if let Some(buffer) = buffers.identities {
            check(buffer, 4)?;
            if buffer.size() > limits.max_storage_buffer_binding_size || buffer.size() % 4 != 0 {
                return Err(invalid());
            }
            u32::try_from(buffer.size() / 4).map_err(|_| invalid())?
        } else {
            0
        };
        check(buffers.accelerations, u64::from(bodies.max(1)) * 32)?;
        for buffer in buffers.velocities {
            check(buffer, u64::from(bodies.max(1)) * 32)?;
        }
        let awake = storage(
            device,
            queue,
            u64::from(groups) * 4,
            bytemuck::cast_slice(&vec![1u32; groups as usize]),
        );
        let roots = storage(device, queue, u64::from(groups) * 4, &[]);
        let component = storage(device, queue, u64::from(groups) * 4, &[]);
        let dependencies = storage(device, queue, u64::from(bodies.max(1)) * 8, &[]);
        let watches = storage(device, queue, u64::from(rows) * 16, &[]);
        let body_groups_buffer = storage(device, queue, 4, bytemuck::cast_slice(body_groups));
        let params = uniform(
            device,
            queue,
            bytemuck::cast_slice(&[groups, rows, bodies, 0, 0, identities, 0, 0]),
        );
        let core_layout = layout(
            device,
            &[
                false, false, false, true, true, false, false, false, true, true, true, false,
                true, true,
            ],
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact activation"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/contact_activation.wgsl").into(),
            ),
        });
        let pipelines =
            ["propagate", "retire"].map(|entry| pipeline(device, &core_layout, &shader, entry));
        let bindings = buffers
            .velocities
            .iter()
            .map(|velocity| {
                group(
                    device,
                    &core_layout,
                    &[
                        &awake,
                        &roots,
                        &component,
                        &dependencies,
                        &watches,
                        buffers.rows,
                        buffers.impacts,
                        buffers.onsets,
                        &body_groups_buffer,
                        velocity,
                        buffers.accelerations,
                        buffers.status,
                        buffers.identities.unwrap_or(&body_groups_buffer),
                        &params,
                    ],
                )
            })
            .collect();
        let mask_layout = layout(device, &[true, true, false, false, true]);
        let mask_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact activation selection"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/contact_activation_mask.wgsl").into(),
            ),
        });
        let mask_pipeline = pipeline(device, &mask_layout, &mask_shader, "select_masks");
        let body_mask = storage(device, queue, u64::from(bodies) * 4, &[]);
        let body_mapping = mapping(
            device,
            queue,
            &mask_layout,
            &awake,
            buffers.status,
            body_groups,
            &body_mask,
        );
        Ok(Self {
            pipelines,
            bindings,
            mask_pipeline,
            mask_layout,
            status: buffers.status.clone(),
            awake,
            dependencies,
            watches,
            params,
            body_mask,
            body_mapping,
            groups,
            rows,
            bodies,
            identities,
        })
    }

    pub(crate) fn prepare(
        &self,
        queue: &wgpu::Queue,
        awake: &[u32],
        dependencies: &[[u32; 2]],
        watches: &[[u32; 4]],
    ) -> Result<()> {
        if awake.len() != self.groups as usize
            || awake.iter().any(|&a| a > 1)
            || dependencies.len() > self.bodies as usize
            || watches.len() > self.rows as usize
            || dependencies.iter().flatten().any(|&g| g >= self.groups)
            || watches.iter().any(|w| {
                w[0] >= self.groups || w[1] == NO_GROUP || w[2] == NO_GROUP || w[1] == w[2]
            })
        {
            return Err(Error::Gpu("invalid contact activation input".into()));
        }
        if !awake.is_empty() {
            molla_core::gpu_write_buffer!(queue, &self.awake, 0, bytemuck::cast_slice(awake));
        }
        if !dependencies.is_empty() {
            molla_core::gpu_write_buffer!(
                queue,
                &self.dependencies,
                0,
                bytemuck::cast_slice(dependencies)
            );
        }
        if !watches.is_empty() {
            molla_core::gpu_write_buffer!(queue, &self.watches, 0, bytemuck::cast_slice(watches));
        }
        molla_core::gpu_write_buffer!(
            queue,
            &self.params,
            0,
            bytemuck::cast_slice(&[
                self.groups,
                self.rows,
                self.bodies,
                dependencies.len() as u32,
                watches.len() as u32,
                self.identities,
                0,
                0,
            ]),
        );
        Ok(())
    }

    pub(crate) fn selection(
        &self,
        engine: &Engine,
        sources: &[u32],
        output: &wgpu::Buffer,
    ) -> Result<SelectionMapping> {
        let device = engine.device().unwrap();
        if sources.iter().any(|&g| g != NO_GROUP && g >= self.groups)
            || sources.len().div_ceil(64)
                > device.limits().max_compute_workgroups_per_dimension as usize
            || output.size() < (sources.len() as u64 * 4).max(4)
            || !output.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(Error::Gpu("invalid contact activation selection".into()));
        }
        Ok(mapping(
            device,
            engine.queue().unwrap(),
            &self.mask_layout,
            &self.awake,
            &self.status,
            sources,
            output,
        ))
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, side: usize) -> Result<()> {
        let bindings = self
            .bindings
            .get(side)
            .ok_or_else(|| Error::Gpu("invalid contact activation state".into()))?;
        for (pipeline, work) in [
            (&self.pipelines[0], 1),
            (&self.pipelines[1], self.rows.div_ceil(64)),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("contact activation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bindings, &[]);
            pass.dispatch_workgroups(work, 1, 1);
        }
        self.encode_selection(encoder, &self.body_mapping);
        Ok(())
    }

    pub(crate) fn encode_selection(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mapping: &SelectionMapping,
    ) {
        if mapping.count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("contact activation masks"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.mask_pipeline);
        pass.set_bind_group(0, &mapping.group, &[]);
        pass.dispatch_workgroups(mapping.count.div_ceil(64), 1, 1);
    }

    pub(crate) fn body_mask(&self) -> &wgpu::Buffer {
        &self.body_mask
    }
    #[cfg(test)]
    pub(crate) fn group_mask(&self) -> &wgpu::Buffer {
        &self.awake
    }
}

#[cfg(test)]
#[path = "rigid_contact_activation_gpu_tests.rs"]
mod tests;
