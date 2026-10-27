use super::{Error, Result, TireFemContactFeedbackGpuTerrain, terrain_feedback};
use molla_compute::{DeviceHealth, Engine};
use std::sync::Arc;

/// One query result: normal/height, velocity/friction, water/sinkage/rolling scale.
/// Status x is 0 before encoding, 1 for success, 2 for invalid coordinates,
/// or 3 for invalid sampled terrain. Failed samples have zero numeric fields.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoilTerrainSample {
    pub normal_height: [f32; 4],
    pub velocity_friction: [f32; 4],
    pub material: [f32; 4],
    pub status: [u32; 4],
}

fn shader() -> String {
    let mut result = include_str!("../shaders/soil_terrain_queries.wgsl").to_owned();
    let original = include_str!("../shaders/tire_fem_contact_feedback.wgsl");
    for name in [
        "safe_normalize",
        "sample_terrain_height",
        "sample_terrain_friction",
        "sample_terrain_water_film",
        "sample_terrain_sinkage",
        "sample_terrain_surface_velocity",
        "sample_terrain_normal",
    ] {
        let marker = format!("fn {name}(");
        assert_eq!(original.matches(&marker).count(), 1);
        let start = original.find(&marker).unwrap();
        let end = start + original[start..].find("\n}").unwrap() + 2;
        result.push('\n');
        result.push_str(&original[start..end]);
    }
    result
}

fn invalid(message: &str) -> Error {
    Error::Solver(format!("soil terrain queries: {message}"))
}

/// Retained queries over a regular terrain raster, not volumetric MPM collision.
/// Inputs are vec4 positions (x/z queried); results use FEM bilinear sampling and
/// finite-difference normals with edge clamping. Raw buffers and encoders must
/// belong to the supplied engine. Encode after terrain and query producers.
/// Reprepare when source identities, terrain parameters or query count change.
pub struct PreparedSoilTerrainQueries {
    health: Arc<DeviceHealth>,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    output: wgpu::Buffer,
    count: u32,
}

impl PreparedSoilTerrainQueries {
    /// Owned buffer budget excludes borrowed inputs/maps and driver allocations.
    pub fn required_bytes(count: u32) -> u64 {
        112 + 64 * u64::from(count.max(1))
    }

    pub fn new(
        engine: &Engine,
        terrain: &TireFemContactFeedbackGpuTerrain<'_>,
        queries: &wgpu::Buffer,
        count: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| invalid("GPU engine required"))?;
        terrain_feedback::terrain_valid(device, terrain)?;
        let limits = device.limits();
        let size = 64 * u64::from(count.max(1));
        if terrain.spacing < 1e-6
            || terrain.nx > 1_000_000_000
            || terrain.nz > 1_000_000_000
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || Self::required_bytes(count) > max_owned_bytes
            || size > limits.max_storage_buffer_binding_size
            || size > limits.max_buffer_size
            || !queries.usage().contains(wgpu::BufferUsages::STORAGE)
            || queries.size() < 16 * u64::from(count.max(1))
            || queries.size() > limits.max_storage_buffer_binding_size
        {
            return Err(invalid("query/terrain capacity, spacing or byte budget"));
        }
        let allocate = |size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("soil terrain query storage"),
                    size,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let output = allocate(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let dummy = allocate(16, wgpu::BufferUsages::STORAGE);
        let uniform = allocate(
            96,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let flags = u32::from(terrain.friction_map.is_some())
            | (u32::from(terrain.water_film_map.is_some()) << 1)
            | (u32::from(terrain.sinkage_map.is_some()) << 2)
            | (u32::from(terrain.surface_velocity_map.is_some()) << 3);
        let f = f32::to_bits;
        let params = [
            count,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            f(terrain.surface_velocity[0]),
            f(terrain.surface_velocity[1]),
            f(terrain.surface_velocity[2]),
            0,
            f(terrain.origin[0]),
            f(terrain.origin[1]),
            f(terrain.spacing),
            f(terrain.friction),
            terrain.nx,
            terrain.nz,
            1,
            flags,
            f(terrain.soil_rolling_resistance_scale.max(0.0)),
            0,
            0,
            0,
        ];
        engine
            .queue()
            .unwrap()
            .write_buffer(&uniform, 0, bytemuck::cast_slice(&params));
        molla_core::gpu_metrics::record_upload(96);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil terrain queries"),
            source: wgpu::ShaderSource::Wgsl(shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil terrain queries"),
            layout: None,
            module: &module,
            entry_point: Some("query"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffers = [
            queries,
            terrain.heights,
            terrain.friction_map.unwrap_or(&dummy),
            terrain.water_film_map.unwrap_or(&dummy),
            terrain.sinkage_map.unwrap_or(&dummy),
            terrain.surface_velocity_map.unwrap_or(&dummy),
            &output,
            &uniform,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil terrain queries"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Ok(Self {
            health: engine.device_health().unwrap().clone(),
            pipeline,
            bindings,
            output,
            count,
        })
    }

    pub fn output(&self) -> &wgpu::Buffer {
        &self.output
    }
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Encodes one retained dispatch with no upload, allocation, submission or wait.
    pub fn encode(&self, engine: &Engine, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        self.health.check()?;
        if !engine
            .device_health()
            .is_some_and(|h| Arc::ptr_eq(h, &self.health))
        {
            return Err(invalid("foreign engine"));
        }
        if self.count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            molla_core::gpu_metrics::record_dispatch();
            pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        }
        Ok(())
    }
}

#[test]
fn soil_terrain_query_layout_and_source() {
    assert_eq!(std::mem::size_of::<SoilTerrainSample>(), 64);
    assert_eq!(PreparedSoilTerrainQueries::required_bytes(0), 176);
    assert_eq!(PreparedSoilTerrainQueries::required_bytes(129), 8368);
    let module = naga::front::wgsl::parse_str(&shader()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
