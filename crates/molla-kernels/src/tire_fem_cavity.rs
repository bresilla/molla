use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_vehicle::TetraTireFemCavitySurface;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    ranges: [u32; 4],
    gas: [f32; 4],
    shape: [f32; 4],
}

fn invalid() -> Error {
    Error::Solver("FEM cavity: invalid topology, source, configuration or budget".into())
}

fn packed(
    mesh: &TetraTireFemCavitySurface<'_>,
    node_offset: u32,
    hub_row: u32,
) -> Result<(Params, Vec<[u32; 4]>)> {
    let count = mesh
        .tube
        .len()
        .checked_add(mesh.negative_cap.len())
        .and_then(|n| n.checked_add(mesh.positive_cap.len()))
        .ok_or_else(invalid)?;
    if mesh.node_count == 0
        || mesh.node_count > 16384
        || count > 65536
        || node_offset.checked_add(mesh.node_count as u32).is_none()
    {
        return Err(invalid());
    }
    let mut faces = Vec::with_capacity(count.max(1));
    faces.extend(mesh.tube.iter().map(|t| [t[0], t[1], t[2], 0]));
    faces.extend(mesh.negative_cap.iter().map(|e| [e[1], e[0], 0, 1]));
    faces.extend(mesh.positive_cap.iter().map(|e| [e[0], e[1], 0, 2]));
    if faces.iter().any(|f| {
        f[..if f[3] == 0 { 3 } else { 2 }]
            .iter()
            .any(|&i| i as usize >= mesh.node_count)
    }) {
        return Err(invalid());
    }
    if faces.is_empty() {
        faces.push([0; 4]);
    }
    let params = Params {
        ranges: [node_offset, mesh.node_count as u32, hub_row, count as u32],
        gas: [
            mesh.pressure_pa as f32,
            mesh.rest_volume_m3 as f32,
            mesh.gamma as f32,
            mesh.coupling as f32,
        ],
        shape: [mesh.width_m as f32, mesh.area_per_bead_m2 as f32, 0., 0.],
    };
    if params
        .gas
        .iter()
        .chain(&params.shape)
        .any(|x| !x.is_finite() || *x < 0.)
        || params.gas[2] < 0.1
        || params.gas[3] > 1.
        || !(50. * params.gas[0].max(1e4)).is_finite()
    {
        return Err(invalid());
    }
    Ok((params, faces))
}

/// Resident FP32 cavity volume, gas pressure and bead-force evaluation.
/// Output is [volume_m3, pressure_pa, force_per_bead_n, valid]; valid=0 must
/// not be consumed as a physical load. Buffers/encoder must share the engine.
/// Refresh frequency is caller-controlled; no CPU readback or submission occurs.
pub struct PreparedTireFemCavity {
    device: std::sync::Arc<wgpu::Device>,
    params: Params,
    uniform: wgpu::Buffer,
    output: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
    pipeline: wgpu::ComputePipeline,
    bytes: u64,
}

impl PreparedTireFemCavity {
    pub(crate) fn validate_surface(
        mesh: &TetraTireFemCavitySurface<'_>,
        node_offset: u32,
        hub_row: u32,
    ) -> Result<u64> {
        let (params, _) = packed(mesh, node_offset, hub_row)?;
        Self::required_bytes(params.ranges[3] as usize)
    }
    pub fn required_bytes(face_count: usize) -> Result<u64> {
        if face_count > 65536 {
            return Err(invalid());
        }
        Ok(64 + 16 * face_count.max(1) as u64)
    }

    pub fn new(
        engine: &Engine,
        positions: [&wgpu::Buffer; 2],
        hubs: &wgpu::Buffer,
        node_offset: u32,
        hub_row: u32,
        mesh: &TetraTireFemCavitySurface<'_>,
        max_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let (params, faces) = packed(mesh, node_offset, hub_row)?;
        let bytes = Self::required_bytes(params.ranges[3] as usize)?;
        let limits = device.limits();
        let source_sizes = [
            16 * (u64::from(node_offset) + mesh.node_count as u64),
            16 * (u64::from(node_offset) + mesh.node_count as u64),
            16 * (u64::from(hub_row) + 1),
        ];
        if bytes > max_bytes
            || limits.max_storage_buffers_per_shader_stage < 4
            || limits.max_bindings_per_bind_group < 5
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_uniform_buffer_binding_size < 48
            || 16 * faces.len() as u64 > limits.max_buffer_size
            || 16 * faces.len() as u64 > limits.max_storage_buffer_binding_size
            || positions
                .into_iter()
                .chain([hubs])
                .zip(source_sizes)
                .any(|(b, n)| {
                    b.size() < n
                        || b.size() > limits.max_storage_buffer_binding_size
                        || !b.usage().contains(wgpu::BufferUsages::STORAGE)
                })
        {
            return Err(invalid());
        }
        let buffer = |label, size, usage| {
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
        let topology = buffer(
            "cavity faces",
            16 * faces.len() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let uniform = buffer(
            "cavity constants",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let output = buffer(
            "cavity volume pressure force validity",
            16,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let queue = engine.queue().unwrap();
        molla_core::gpu_write_buffer!(queue, &topology, 0, bytemuck::cast_slice(&faces));
        molla_core::gpu_write_buffer!(queue, &uniform, 0, bytemuck::bytes_of(&params));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident cavity gas law"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tire_fem_cavity.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("resident cavity"),
            layout: None,
            module: &module,
            entry_point: Some("cavity"),
            compilation_options: Default::default(),
            cache: None,
        });
        let groups = positions.map(|pos| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cavity sources"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[pos, hubs, &topology, &uniform, &output]
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        Ok(Self {
            device: device.clone(),
            params,
            uniform,
            output,
            groups,
            pipeline,
            bytes,
        })
    }

    /// Updates only reference pressure; rebuild for topology or other constants.
    /// Submit previous encodings before updating this shared configuration.
    pub fn update_pressure(&mut self, engine: &Engine, pressure_pa: f32) -> Result<()> {
        engine.check_device()?;
        crate::soil_gpu_identity::validate_soil_device(
            engine.device().ok_or_else(invalid)?,
            &self.device,
        )?;
        if !pressure_pa.is_finite() || pressure_pa < 0. || !(50. * pressure_pa.max(1e4)).is_finite()
        {
            return Err(invalid());
        }
        if pressure_pa.to_bits() != self.params.gas[0].to_bits() {
            self.params.gas[0] = pressure_pa;
            molla_core::gpu_write_buffer!(
                engine.queue().unwrap(),
                &self.uniform,
                0,
                bytemuck::bytes_of(&self.params)
            );
        }
        Ok(())
    }

    /// Reads the selected position parity and writes one validity-tagged record.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, parity: bool) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.groups[usize::from(parity)], &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}

#[test]
fn cavity_layout_shader_and_validation() {
    assert_eq!(std::mem::size_of::<Params>(), 48);
    assert_eq!(PreparedTireFemCavity::required_bytes(0).unwrap(), 80);
    assert_eq!(
        PreparedTireFemCavity::required_bytes(65536).unwrap(),
        1_048_640
    );
    assert!(PreparedTireFemCavity::required_bytes(65537).is_err());
    let tire = molla_vehicle::TetraTireFem::new(
        0.,
        180_000.,
        molla_vehicle::TetraTireFemParams::realtime_viewer(),
    );
    let mut mesh = tire.cavity_surface();
    let (params, faces) = packed(&mesh, 0, 0).unwrap();
    assert_eq!(
        params.ranges[3] as usize,
        mesh.tube.len() + mesh.negative_cap.len() + mesh.positive_cap.len()
    );
    assert_eq!(
        faces[mesh.tube.len()],
        [mesh.negative_cap[0][1], mesh.negative_cap[0][0], 0, 1]
    );
    assert!(packed(&mesh, u32::MAX, 0).is_err());
    mesh.gamma = f64::NAN;
    assert!(packed(&mesh, 0, 0).is_err());
    mesh.gamma = 1.;
    mesh.node_count = 1;
    assert!(packed(&mesh, 0, 0).is_err());
    let source = include_str!("../shaders/tire_fem_cavity.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
