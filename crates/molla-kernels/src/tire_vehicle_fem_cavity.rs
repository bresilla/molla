use super::*;
use crate::PreparedTireFemCavity;
use molla_vehicle::TetraTireFemCavitySurface;

pub(super) struct CavityFrame {
    cavities: Vec<PreparedTireFemCavity>,
    pub(super) records: wgpu::Buffer,
    pub(super) control: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

impl CavityFrame {
    pub(super) fn new(
        engine: &Engine,
        owner: &TireResidentVehicleFemAtlasOwner,
        node_base: &wgpu::Buffer,
        node_groups: u32,
        surfaces: &[TetraTireFemCavitySurface<'_>],
        budget: u64,
    ) -> Result<Self> {
        let n = owner.tire_count();
        if surfaces.len() != n {
            return Err(invalid("cavity count"));
        }
        let mut sizes = Vec::with_capacity(n);
        for (i, (surface, row)) in surfaces
            .iter()
            .zip(&owner.fem.atlas.layout.rows)
            .enumerate()
        {
            if surface.node_count != row.node_count {
                return Err(invalid("cavity node count"));
            }
            sizes.push(PreparedTireFemCavity::validate_surface(
                surface,
                row.node_offset as u32,
                2 * i as u32,
            )?);
        }
        let bytes = 64 + 16 * n as u64 + sizes.iter().sum::<u64>();
        let device = engine.device().unwrap();
        let limits = device.limits();
        if bytes > budget
            || limits.max_storage_buffers_per_shader_stage < 3
            || limits.max_bindings_per_bind_group < 3
            || 112 * n as u64 > limits.max_storage_buffer_binding_size
            || 16 * n as u64 > limits.max_buffer_size
        {
            return Err(invalid("cavity budget or device limits"));
        }
        let mut cavities = Vec::with_capacity(n);
        for (i, ((surface, row), size)) in surfaces
            .iter()
            .zip(&owner.fem.atlas.layout.rows)
            .zip(sizes)
            .enumerate()
        {
            cavities.push(PreparedTireFemCavity::new(
                engine,
                [&owner.fem.pos_a, &owner.fem.pos_b],
                &owner.fem.hub_state_buf,
                row.node_offset as u32,
                2 * i as u32,
                surface,
                size,
            )?);
        }
        let allocate = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false
                }
            )
        };
        let records = allocate(
            "vehicle cavity records",
            16 * n as u64,
            wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
        );
        let control = allocate(
            "vehicle cavity admission and indirect work",
            64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let mut words = [0u32; 16];
        words[13] = n as u32;
        words[14] = node_groups;
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &control,
            0,
            bytemuck::cast_slice(&words)
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vehicle cavity admission"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tire_vehicle_fem_cavity.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("vehicle cavity admission"),
            layout: None,
            module: &module,
            entry_point: Some("admit"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[&records, node_base, &control]
                .iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            cavities,
            records,
            control,
            pipeline,
            group,
        })
    }

    pub(super) fn validate_pressures(params: &[TireFemGpuParams]) -> Result<()> {
        if params.iter().any(|p| {
            !p.pressure_pa.is_finite()
                || p.pressure_pa < 0.
                || !(50. * p.pressure_pa.max(1e4)).is_finite()
        }) {
            return Err(invalid("cavity pressure"));
        }
        Ok(())
    }

    pub(super) fn update_pressures(
        &mut self,
        engine: &Engine,
        params: &[TireFemGpuParams],
    ) -> Result<()> {
        Self::validate_pressures(params)?;
        for (c, p) in self.cavities.iter_mut().zip(params) {
            c.update_pressure(engine, p.pressure_pa)?;
        }
        Ok(())
    }

    pub(super) fn refresh(&self, engine: &Engine, parity: bool) {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        for (i, c) in self.cavities.iter().enumerate() {
            c.encode(&mut encoder, parity);
            encoder.copy_buffer_to_buffer(c.output_buffer(), 0, &self.records, 16 * i as u64, 16);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        engine.queue().unwrap().submit([encoder.finish()]);
    }
}

#[test]
fn cavity_frame_indirect_and_parameter_layout() {
    assert_eq!(
        std::mem::offset_of!(TireFemGpuBatchAtlasParamRow, phys3) + 4,
        84
    );
    let source = include_str!("../shaders/tire_vehicle_fem_cavity.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
