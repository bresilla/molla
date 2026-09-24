use super::*;

pub(super) struct MaterialLinearization {
    layout: wgpu::BindGroupLayout,
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) diagonal_pipeline: wgpu::ComputePipeline,
    pub(super) cache_pipeline: wgpu::ComputePipeline,
    pub(super) cached_action_pipeline: wgpu::ComputePipeline,
    pub(super) group: Option<wgpu::BindGroup>,
    pub(super) material_groups: Option<[wgpu::BindGroup; 6]>,
    pub(super) batches: Vec<(wgpu::BindGroup, u32)>,
    pub(super) cache_batches: Vec<(wgpu::BindGroup, u32)>,
    pub(super) cache_ready: std::sync::atomic::AtomicBool,
    tangent_cache: wgpu::Buffer,
    pub(super) residual: wgpu::Buffer,
    pub(super) action: wgpu::Buffer,
}

impl MaterialLinearization {
    pub(super) fn new(
        device: &wgpu::Device,
        material: [&wgpu::BindGroupLayout; 6],
        particles: u32,
        tets: u32,
    ) -> Result<Self> {
        let cache_bytes = u64::from(tets.max(1)) * 432;
        if device.limits().max_bind_groups < 7
            || device.limits().max_storage_buffers_per_shader_stage < 47
            || cache_bytes > u64::from(device.limits().max_storage_buffer_binding_size)
            || cache_bytes > device.limits().max_buffer_size
        {
            return Err(Error::Build(
                "material linearization requires 7 groups, 47 storage buffers and tangent capacity"
                    .into(),
            ));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material linearization vectors"),
            entries: &(0..4)
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
        let diagonal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("material inverse diagonal"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("material_inverse_diagonal"),
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
        let cache_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cache material tangents"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cache_material_tangents"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cached_action_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cached material action"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("material_cached_action"),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Self {
            layout,
            pipeline,
            diagonal_pipeline,
            cache_pipeline,
            cached_action_pipeline,
            group: None,
            material_groups: None,
            batches: Vec::new(),
            cache_batches: Vec::new(),
            cache_ready: std::sync::atomic::AtomicBool::new(false),
            tangent_cache: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("material stress derivative cache"),
                size: cache_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
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
        self.cache_ready
            .store(false, std::sync::atomic::Ordering::Relaxed);
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
                entries: &[direction, &self.residual, &self.action, &self.tangent_cache]
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
