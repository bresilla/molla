use super::*;
use wgpu::util::DeviceExt;

/// Prepared contact state and the retained material input velocity target.
pub struct MaterialContactPatchBindings<'a> {
    pub rows: &'a wgpu::Buffer,
    pub projected: &'a wgpu::Buffer,
    pub response: &'a wgpu::Buffer,
    pub rigid_velocity: &'a wgpu::Buffer,
    pub contacts: &'a wgpu::Buffer,
    pub schedule: &'a wgpu::Buffer,
    pub input_velocity: &'a wgpu::Buffer,
    pub inverse_mass: &'a wgpu::Buffer,
}

pub(super) struct CoupledPatch {
    layout: wgpu::BindGroupLayout,
    state_layout: wgpu::BindGroupLayout,
    pub(super) pipelines: [wgpu::ComputePipeline; 5],
    state_pipelines: [wgpu::ComputePipeline; 3],
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    groups: Option<(wgpu::BindGroup, wgpu::BindGroup)>,
    target: Option<(wgpu::Buffer, wgpu::Buffer)>,
    pub(super) select_pending: bool,
}

impl CoupledPatch {
    pub(super) fn new(
        device: &wgpu::Device,
        material: [&wgpu::BindGroupLayout; 6],
    ) -> Result<Self> {
        if device.limits().max_bind_groups < 7
            || device.limits().max_storage_buffers_per_shader_stage < 51
        {
            return Err(Error::Build(
                "coupled material patch requires 7 bind groups and 51 storage buffers".into(),
            ));
        }
        let layout_for = |label, writable: &[bool]| {
            let entries: Vec<_> = writable
                .iter()
                .enumerate()
                .map(|(binding, &write)| wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: !write },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            })
        };
        let layout = layout_for(
            "material/contact patch",
            &[true, false, false, true, true, false, true, true],
        );
        let state_layout = layout_for(
            "material/contact patch target",
            &[false, false, true, true, false],
        );
        let mut layouts: Vec<_> = material.iter().copied().map(Some).collect();
        layouts.push(Some(&layout));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("material/contact patch"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        let state_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("material/contact patch target"),
                bind_group_layouts: &[Some(&state_layout)],
                immediate_size: 0,
            });
        let source = concat!(
            include_str!("../../shaders/tet_viscosity.wgsl"),
            "\n",
            include_str!("../../shaders/vbd_cloth.wgsl"),
            "\n",
            include_str!("../../shaders/fem_contact_patch.wgsl")
        )
        .replace(
            "const CACHE_PREVIOUS_TETS: bool = false;",
            "const CACHE_PREVIOUS_TETS: bool = true;",
        )
        .replace(
            "const LOCAL_MATERIAL_TRIAL: bool = false;",
            "const LOCAL_MATERIAL_TRIAL: bool = true;",
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coupled material/contact Newton"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let state_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coupled material/contact target"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_contact_patch_state.wgsl").into(),
            ),
        });
        let pipeline = |layout, shader, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = [
            "patch_select",
            "patch_polish_neighbors",
            "patch_refresh",
            "patch_live_solve",
            "patch_apply",
        ]
        .map(|entry| pipeline(&pipeline_layout, &shader, entry));
        let state_pipelines = ["disable_contacts", "restore_contacts", "sync_target"]
            .map(|entry| pipeline(&state_pipeline_layout, &state_shader, entry));
        let mut metadata = [0.0_f32; 207];
        metadata[190] = 1.0;
        let upload = |label, data: &[u8]| {
            molla_core::gpu_buffer_init!(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                }
            )
        };
        let input = upload(
            "material/contact patch metadata",
            bytemuck::cast_slice(&metadata),
        );
        let output = upload(
            "material/contact patch result",
            bytemuck::cast_slice(&[0.0_f32; 37]),
        );
        Ok(Self {
            layout,
            state_layout,
            pipelines,
            state_pipelines,
            input,
            output,
            groups: None,
            target: None,
            select_pending: false,
        })
    }

    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        bindings: MaterialContactPatchBindings<'_>,
        particles: u32,
    ) -> Result<()> {
        self.groups = None;
        self.target = None;
        let rows = bindings.contacts.size() / 80;
        let dofs = bindings.rigid_velocity.size() / 4;
        let schedule_words = rows + 1 + rows.div_ceil(64) + 132 + u64::from(particles) + 5 * rows;
        if rows == 0
            || rows >= 1 << 24
            || particles >= 1 << 24
            || !(1..=64).contains(&dofs)
            || !bindings.contacts.size().is_multiple_of(80)
            || !bindings.rigid_velocity.size().is_multiple_of(4)
            || bindings.rows.size() != rows * 32
            || bindings.projected.size() != rows * 3 * dofs * 4
            || bindings.response.size() != bindings.projected.size()
            || bindings.schedule.size() < schedule_words * 4
            || bindings.input_velocity.size() < u64::from(particles) * 16
            || bindings.inverse_mass.size() < u64::from(particles) * 4
        {
            return Err(Error::Build(
                "invalid coupled material/contact patch buffers".into(),
            ));
        }
        let bind = |layout, buffers: &[&wgpu::Buffer]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(index, buffer)| wgpu::BindGroupEntry {
                        binding: index as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let group = bind(
            &self.layout,
            &[
                bindings.rows,
                bindings.projected,
                bindings.response,
                bindings.rigid_velocity,
                bindings.contacts,
                bindings.schedule,
                &self.input,
                &self.output,
            ],
        );
        let state = bind(
            &self.state_layout,
            &[
                &self.input,
                &self.output,
                bindings.contacts,
                bindings.input_velocity,
                bindings.inverse_mass,
            ],
        );
        self.groups = Some((group, state));
        self.target = Some((
            bindings.input_velocity.clone(),
            bindings.inverse_mass.clone(),
        ));
        self.select_pending = true;
        Ok(())
    }

    pub(super) fn clear(&mut self) {
        self.groups = None;
        self.target = None;
        self.select_pending = false;
    }
    pub(super) fn group(&self) -> Option<&wgpu::BindGroup> {
        self.groups.as_ref().map(|groups| &groups.0)
    }
    pub(super) fn matches_target(&self, input: &wgpu::Buffer, inverse_mass: &wgpu::Buffer) -> bool {
        self.target
            .as_ref()
            .is_some_and(|(target, mass)| target == input && mass == inverse_mass)
    }
    pub(super) fn diagnostics(&self) -> &wgpu::Buffer {
        &self.output
    }

    pub(super) fn encode_state(&self, encoder: &mut wgpu::CommandEncoder, index: usize) {
        let Some((_, group)) = &self.groups else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.state_pipelines[index]);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}
