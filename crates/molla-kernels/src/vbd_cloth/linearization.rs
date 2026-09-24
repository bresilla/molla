use super::*;

pub(super) struct MaterialLinearization {
    layout: wgpu::BindGroupLayout,
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) group: Option<wgpu::BindGroup>,
    pub(super) residual: wgpu::Buffer,
    pub(super) action: wgpu::Buffer,
}

impl MaterialLinearization {
    pub(super) fn new(
        device: &wgpu::Device,
        material: [&wgpu::BindGroupLayout; 6],
        particles: u32,
    ) -> Result<Self> {
        if device.limits().max_bind_groups < 7
            || device.limits().max_storage_buffers_per_shader_stage < 46
        {
            return Err(Error::Build(
                "material linearization requires 7 groups and 46 storage buffers".into(),
            ));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material linearization vectors"),
            entries: &(0..3)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding == 0,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let mut layouts: Vec<_> = material.into_iter().map(Some).collect();
        layouts.push(Some(&layout));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("material linearization"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        let source = concat!(
            include_str!("../../shaders/tet_viscosity.wgsl"),
            "\n",
            include_str!("../../shaders/vbd_cloth.wgsl"),
            "\n",
            include_str!("../../shaders/fem_material_linearization.wgsl"),
        )
        .replace(
            "const CACHE_PREVIOUS_TETS: bool = false;",
            "const CACHE_PREVIOUS_TETS: bool = true;",
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("material residual and analytic tangent action"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("material linearization"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("material_linearization"),
            compilation_options: Default::default(),
            cache: None,
        });
        let allocate = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(particles) * 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            layout,
            pipeline,
            group: None,
            residual: allocate("material signed residual"),
            action: allocate("material tangent action"),
        })
    }

    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        direction: &wgpu::Buffer,
    ) -> Result<()> {
        self.group = None;
        if direction.size() != self.action.size()
            || direction == &self.action
            || direction == &self.residual
            || !direction.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(Error::Build(
                "material direction must be a distinct particle vec4 storage buffer".into(),
            ));
        }
        self.group = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material linearization vectors"),
                layout: &self.layout,
                entries: &[direction, &self.residual, &self.action]
                    .into_iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            }),
        );
        Ok(())
    }
}
