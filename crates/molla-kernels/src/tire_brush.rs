//! GPU brush-tire kernel (stage (i)).
//!
//! Mirrors `molla_vehicle::real_tire::evaluate_brush_tire` for either
//! a flat terrain or a regular heightfield grid uploaded as a 1-D
//! storage buffer. The CPU side packs the per-wheel rows and the
//! optional terrain payload; the kernel does the per-segment contact
//! work entirely on GPU.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};

const WORKGROUP_SIZE: u32 = 64;
const SEGMENTS_MAX: u32 = 256;

/// Per-wheel input row uploaded to the GPU brush kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct BrushWheelInput {
    /// `(hub.xyz, width_segments)`: integer 1..=16; zero selects three strips.
    pub hub_pos: [f32; 4],
    pub hub_vel: [f32; 4],
    /// `(heading.xyz, structural_lateral_bias_per_normal)`.
    pub heading: [f32; 4],
    /// `(omega rad/s, radius m, width m, pressure Pa)`
    pub spin: [f32; 4],
    /// `(segments, carcass_normal_stiffness, tread_shear_stiffness, relaxation_speed)`
    pub tire: [f32; 4],
    /// `(mu_static, mu_dynamic, bristle_damping, rolling_resistance)`
    pub material: [f32; 4],
    /// `(cohesion, terrain_mode {0=flat,1=heightfield}, mu_factor, camber_angle_rad)`
    pub extra: [f32; 4],
}

/// Per-wheel output row produced by the kernel.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct BrushWheelOutput {
    /// `(Fx_long_raw, normal_load, Fy_lateral, longitudinal_force = Fx - rolling)`
    pub forces: [f32; 4],
    /// `(patch_length, slip_ratio, slip_speed, contact_count)`
    pub diag1: [f32; 4],
    /// `(sliding_count, aligning_moment_z, lateral_slip_speed, slip_angle_rad)`
    pub diag2: [f32; 4],
    /// `(dissipated_power_w, camber_angle_rad, contact_center_x, pneumatic_trail_m)`
    pub diag3: [f32; 4],
    /// `(adhesion_fraction, sliding_fraction, available_friction_budget, consumed_friction_budget)`
    pub diag4: [f32; 4],
    /// `(effective_longitudinal_stiffness, effective_cornering_stiffness, 0, 0)`
    pub diag5: [f32; 4],
    /// `(Fx, Fy, Fz, 0)` world-force vector accumulated from per-contact
    /// terrain normals. This is the GPU counterpart of
    /// `RealTireContact::force`.
    pub world_force: [f32; 4],
}

/// Stage (i) — optional heightfield terrain shared by every wheel in
/// a dispatch. `None` selects the flat-floor path.
#[derive(Clone, Debug)]
pub struct BrushTerrain<'a> {
    /// World `(x, z)` of sample `(0, 0)`.
    pub origin: [f32; 2],
    /// Grid spacing (m).
    pub spacing: f32,
    /// Samples per row.
    pub nx: u32,
    /// Number of rows.
    pub nz: u32,
    /// Row-major heights, length `nx * nz`.
    pub heights: &'a [f32],
    /// Terrain-wide reference friction. Local friction samples are divided by
    /// this value before scaling each wheel's material friction, mirroring
    /// `molla_vehicle::terrain::Terrain::friction_at / friction`.
    pub friction: f32,
    /// Optional row-major friction samples, length `nx * nz`. When absent the
    /// GPU brush path uses the per-wheel material friction unchanged.
    pub friction_map: Option<&'a [f32]>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TerrainUniformsCpu {
    origin: [f32; 4], // (origin_x, origin_z, spacing, friction_ref)
    dims: [u32; 4],   // (nx, nz, enabled, friction_map_enabled)
}

pub struct TireBrushGpuKernel {
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

/// Reusable GPU buffers for repeated brush-contact dispatches.
///
/// The older [`TireBrushGpuKernel::dispatch_blocking`] path allocates fresh
/// buffers for every call. This batch object keeps the wheel, output, terrain,
/// uniform and staging buffers alive across frames, which is the path the
/// viewer/many-wheel runtime should use once the call sites are ready.
pub struct TireBrushGpuBatch {
    capacity: usize,
    terrain_capacity: usize,
    in_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    terrain_buf: wgpu::Buffer,
    friction_buf: wgpu::Buffer,
    terrain_uniform_buf: wgpu::Buffer,
    stage: wgpu::Buffer,
    bg: wgpu::BindGroup,
    bg_u: wgpu::BindGroup,
}

pub struct TireBrushResidentOutput {
    pub buffer: wgpu::Buffer,
    pub wheel_count: usize,
}

impl TireBrushGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_brush"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_brush.wgsl").into()),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_brush io"),
            entries: &[
                storage_entry(0, true),  // wheels
                storage_entry(1, false), // outputs
                storage_entry(2, true),  // terrain_heights
                storage_entry(3, true),  // terrain_friction
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_brush uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_brush layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_brush::brush_step"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("brush_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_io,
            layout_uniforms,
        }
    }

    /// Allocate reusable buffers for up to `capacity` wheels and
    /// `terrain_capacity` height/friction samples.
    pub fn create_batch(
        &self,
        device: &wgpu::Device,
        capacity: usize,
        terrain_capacity: usize,
    ) -> Result<TireBrushGpuBatch> {
        let capacity = capacity.max(1);
        let terrain_capacity = terrain_capacity.max(1);
        let in_bytes = (capacity * core::mem::size_of::<BrushWheelInput>()) as u64;
        let out_bytes = (capacity * core::mem::size_of::<BrushWheelOutput>()) as u64;
        let terrain_bytes = (terrain_capacity * core::mem::size_of::<f32>()) as u64;

        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush persistent in"),
            size: in_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush persistent out"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let terrain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush persistent terrain"),
            size: terrain_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let friction_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush persistent terrain friction"),
            size: terrain_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let terrain_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush persistent terrain uniforms"),
            size: core::mem::size_of::<TerrainUniformsCpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush persistent stage"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_brush persistent bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: terrain_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: friction_buf.as_entire_binding(),
                },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_brush persistent bg uniforms"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_uniform_buf.as_entire_binding(),
            }],
        });

        Ok(TireBrushGpuBatch {
            capacity,
            terrain_capacity,
            in_buf,
            out_buf,
            terrain_buf,
            friction_buf,
            terrain_uniform_buf,
            stage,
            bg,
            bg_u,
        })
    }

    /// Evaluate wheels using a reusable batch buffer set.
    pub fn dispatch_persistent_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireBrushGpuBatch,
        wheels: &[BrushWheelInput],
        terrain: Option<&BrushTerrain<'_>>,
    ) -> Result<Vec<BrushWheelOutput>> {
        validate_wheels(wheels)?;
        let n = wheels.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > batch.capacity {
            return Err(Error::Solver(format!(
                "tire_brush: batch capacity {} is smaller than {n} wheels",
                batch.capacity
            )));
        }
        let terrain_u = validate_terrain(terrain, Some(batch.terrain_capacity))?;
        let wheels_patched = patched_wheels(wheels, terrain.is_some());
        let out_bytes = (n * core::mem::size_of::<BrushWheelOutput>()) as u64;
        queue.write_buffer(&batch.in_buf, 0, bytemuck::cast_slice(&wheels_patched));
        if let Some(t) = terrain {
            queue.write_buffer(&batch.terrain_buf, 0, bytemuck::cast_slice(t.heights));
            if let Some(friction_map) = t.friction_map {
                queue.write_buffer(&batch.friction_buf, 0, bytemuck::cast_slice(friction_map));
            }
        }
        queue.write_buffer(
            &batch.terrain_uniform_buf,
            0,
            bytemuck::bytes_of(&terrain_u),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_brush persistent encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_brush persistent pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &batch.bg, &[]);
            pass.set_bind_group(1, &batch.bg_u, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&batch.out_buf, 0, &batch.stage, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));
        read_outputs(device, &batch.stage, out_bytes)
    }

    /// Evaluate a batch of wheels against a flat floor or a uploaded
    /// heightfield. `terrain == None` selects the flat-floor path. The
    /// wheel's `extra[1]` (terrain_mode) is overridden to match the
    /// dispatch's terrain choice automatically.
    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        wheels: &[BrushWheelInput],
        terrain: Option<&BrushTerrain<'_>>,
    ) -> Result<Vec<BrushWheelOutput>> {
        validate_wheels(wheels)?;
        let n = wheels.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let wheels_patched = patched_wheels(wheels, terrain.is_some());
        let in_bytes = std::mem::size_of_val(wheels) as u64;
        let out_bytes = (n * core::mem::size_of::<BrushWheelOutput>()) as u64;
        let in_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush in"),
            size: in_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush out"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&in_buf, 0, bytemuck::cast_slice(&wheels_patched));

        // Heightfield storage + uniforms.
        let terrain_u = validate_terrain(terrain, None)?;
        let terrain_buf_size = terrain
            .map(|t| (t.nx as u64 * t.nz as u64 * core::mem::size_of::<f32>() as u64).max(4))
            .unwrap_or(4);
        let friction_buf_size = terrain_buf_size;
        let terrain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush terrain"),
            size: terrain_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(t) = terrain {
            queue.write_buffer(&terrain_buf, 0, bytemuck::cast_slice(t.heights));
        }
        let friction_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush terrain friction"),
            size: friction_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(t) = terrain
            && let Some(friction_map) = t.friction_map
        {
            queue.write_buffer(&friction_buf, 0, bytemuck::cast_slice(friction_map));
        }
        let terrain_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush terrain uniforms"),
            size: core::mem::size_of::<TerrainUniformsCpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&terrain_uniform_buf, 0, bytemuck::bytes_of(&terrain_u));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_brush bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: terrain_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: friction_buf.as_entire_binding(),
                },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_brush bg uniforms"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_uniform_buf.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_brush encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_brush pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush stage"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));
        read_outputs(device, &stage, out_bytes)
    }

    /// Evaluate wheels already packed in a GPU-resident
    /// [`BrushWheelInput`] storage buffer.
    ///
    /// Unlike [`Self::dispatch_blocking`] this path does not patch the input
    /// rows on CPU. Callers that generate the rows on GPU must set
    /// `extra[1]` (terrain mode) and `extra[2]` (mu factor) explicitly before
    /// binding the buffer here.
    pub fn dispatch_input_buffer_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        wheel_buffer: &wgpu::Buffer,
        wheel_count: usize,
        terrain: Option<&BrushTerrain<'_>>,
    ) -> Result<Vec<BrushWheelOutput>> {
        if wheel_count == 0 {
            return Ok(Vec::new());
        }
        if wheel_count > u32::MAX as usize {
            return Err(Error::Solver(format!(
                "tire_brush: wheel count {wheel_count} exceeds dispatch range"
            )));
        }
        let out_bytes = (wheel_count * core::mem::size_of::<BrushWheelOutput>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush external-input out"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let terrain_u = validate_terrain(terrain, None)?;
        let terrain_buf_size = terrain
            .map(|t| (t.nx as u64 * t.nz as u64 * core::mem::size_of::<f32>() as u64).max(4))
            .unwrap_or(4);
        let friction_buf_size = terrain_buf_size;
        let terrain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush external-input terrain"),
            size: terrain_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(t) = terrain {
            queue.write_buffer(&terrain_buf, 0, bytemuck::cast_slice(t.heights));
        }
        let friction_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush external-input terrain friction"),
            size: friction_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(t) = terrain
            && let Some(friction_map) = t.friction_map
        {
            queue.write_buffer(&friction_buf, 0, bytemuck::cast_slice(friction_map));
        }
        let terrain_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush external-input terrain uniforms"),
            size: core::mem::size_of::<TerrainUniformsCpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&terrain_uniform_buf, 0, bytemuck::bytes_of(&terrain_u));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_brush external-input bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wheel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: terrain_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: friction_buf.as_entire_binding(),
                },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_brush external-input bg uniforms"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_uniform_buf.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_brush external-input encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_brush external-input pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(wheel_count as u32, 1, 1);
        }
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush external-input stage"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &stage, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));
        read_outputs(device, &stage, out_bytes)
    }

    pub fn dispatch_input_buffer_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        wheel_buffer: &wgpu::Buffer,
        wheel_count: usize,
        terrain: Option<&BrushTerrain<'_>>,
    ) -> Result<TireBrushResidentOutput> {
        if wheel_count == 0 {
            return Err(Error::Solver(
                "tire_brush: resident output requires at least one wheel".into(),
            ));
        }
        if wheel_count > u32::MAX as usize {
            return Err(Error::Solver(format!(
                "tire_brush: wheel count {wheel_count} exceeds dispatch range"
            )));
        }
        let out_bytes = (wheel_count * core::mem::size_of::<BrushWheelOutput>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush resident external-input out"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let terrain_u = validate_terrain(terrain, None)?;
        let terrain_buf_size = terrain
            .map(|t| (t.nx as u64 * t.nz as u64 * core::mem::size_of::<f32>() as u64).max(4))
            .unwrap_or(4);
        let friction_buf_size = terrain_buf_size;
        let terrain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush resident external-input terrain"),
            size: terrain_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(t) = terrain {
            queue.write_buffer(&terrain_buf, 0, bytemuck::cast_slice(t.heights));
        }
        let friction_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush resident external-input terrain friction"),
            size: friction_buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if let Some(t) = terrain
            && let Some(friction_map) = t.friction_map
        {
            queue.write_buffer(&friction_buf, 0, bytemuck::cast_slice(friction_map));
        }
        let terrain_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush resident external-input terrain uniforms"),
            size: core::mem::size_of::<TerrainUniformsCpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&terrain_uniform_buf, 0, bytemuck::bytes_of(&terrain_u));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_brush resident external-input bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wheel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: terrain_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: friction_buf.as_entire_binding(),
                },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_brush resident external-input bg uniforms"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_uniform_buf.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_brush resident external-input encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_brush resident external-input pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups(wheel_count as u32, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireBrushResidentOutput {
            buffer: out_buf,
            wheel_count,
        })
    }

    /// Read a GPU-resident brush output buffer back to typed host rows.
    ///
    /// Viewer/runtime code should keep downstream GPU handoffs resident where
    /// possible, but CPU-driven demos still need a narrow readback seam for
    /// UI telemetry or legacy CPU chassis integration. Keeping this helper next
    /// to the brush kernel avoids each caller reimplementing ad-hoc staging
    /// copies with subtly different row sizes.
    pub fn readback_resident_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: &TireBrushResidentOutput,
    ) -> Result<Vec<BrushWheelOutput>> {
        if output.wheel_count == 0 {
            return Ok(Vec::new());
        }
        let out_bytes = (output.wheel_count * core::mem::size_of::<BrushWheelOutput>()) as u64;
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_brush resident output readback stage"),
            size: out_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_brush resident output readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&output.buffer, 0, &stage, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));
        read_outputs(device, &stage, out_bytes)
    }
}

fn validate_wheels(wheels: &[BrushWheelInput]) -> Result<()> {
    for w in wheels {
        validate_width_segments(w.hub_pos[3])?;
        if w.tire[0] as u32 > SEGMENTS_MAX {
            return Err(Error::Solver(format!(
                "tire_brush: segments {} exceeds GPU max {SEGMENTS_MAX}",
                w.tire[0] as u32
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_width_segments(value: f32) -> Result<()> {
    if !value.is_finite() || !(0.0..=16.0).contains(&value) || value.fract() != 0.0 {
        return Err(Error::Solver(
            "tire_brush: width segments must be an integer 1..=16, or zero for three strips".into(),
        ));
    }
    Ok(())
}

fn patched_wheels(wheels: &[BrushWheelInput], has_terrain: bool) -> Vec<BrushWheelInput> {
    // Patch terrain_mode into each wheel row so callers don't have to remember
    // it. Also default the thermal/wear μ factor to 1.0.
    let mut wheels_patched: Vec<BrushWheelInput> = wheels.to_vec();
    for w in wheels_patched.iter_mut() {
        w.extra[1] = if has_terrain { 1.0 } else { 0.0 };
        if w.extra[2] <= 0.0 {
            w.extra[2] = 1.0;
        }
    }
    wheels_patched
}

fn validate_terrain(
    terrain: Option<&BrushTerrain<'_>>,
    capacity: Option<usize>,
) -> Result<TerrainUniformsCpu> {
    match terrain {
        Some(t) => {
            let expected = (t.nx as usize) * (t.nz as usize);
            if capacity.is_some_and(|cap| expected > cap) {
                return Err(Error::Solver(format!(
                    "tire_brush: terrain sample count {expected} exceeds batch capacity {}",
                    capacity.unwrap()
                )));
            }
            if t.heights.len() < expected {
                return Err(Error::Solver(format!(
                    "tire_brush: heightfield has {} samples, expected {expected}",
                    t.heights.len()
                )));
            }
            if let Some(friction_map) = t.friction_map
                && friction_map.len() < expected
            {
                return Err(Error::Solver(format!(
                    "tire_brush: friction map has {} samples, expected {expected}",
                    friction_map.len()
                )));
            }
            Ok(TerrainUniformsCpu {
                origin: [t.origin[0], t.origin[1], t.spacing, t.friction.max(1.0e-6)],
                dims: [t.nx, t.nz, 1, u32::from(t.friction_map.is_some())],
            })
        }
        None => Ok(TerrainUniformsCpu {
            origin: [0.0, 0.0, 1.0, 0.0],
            dims: [1, 1, 0, 0],
        }),
    }
}

fn read_outputs(
    device: &wgpu::Device,
    stage: &wgpu::Buffer,
    out_bytes: u64,
) -> Result<Vec<BrushWheelOutput>> {
    let slice = stage.slice(0..out_bytes);
    let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("tire_brush readback: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("tire_brush poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("tire_brush recv: {e:?}")))??;
    let data = slice.get_mapped_range();
    let out: Vec<BrushWheelOutput> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    stage.unmap();
    Ok(out)
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// Silence the `WORKGROUP_SIZE` unused warning in the host code — it
// matches the WGSL `@workgroup_size(64)` literal.
#[allow(dead_code)]
const _WG_DOC: u32 = WORKGROUP_SIZE;

#[cfg(test)]
mod tests {
    #[test]
    fn width_resolution_validates_without_changing_the_row_layout() {
        assert_eq!(std::mem::size_of::<super::BrushWheelInput>(), 7 * 16);
        for width in [0.0, 1.0, 3.0, 16.0] {
            assert!(super::validate_width_segments(width).is_ok());
        }
        for width in [-1.0, 0.5, 1.5, 17.0, f32::NAN, f32::INFINITY] {
            assert!(super::validate_width_segments(width).is_err());
        }
    }

    #[test]
    fn tire_brush_shader_carries_structural_lateral_bias_from_heading_w() {
        let shader = include_str!("../shaders/tire_brush.wgsl");
        assert!(
            shader.contains("structural_lateral_bias = clamp(in_.heading.w")
                && shader.contains("structural_bias = structural_lateral_bias * fn_i")
                && shader.contains("+ structural_bias"),
            "brush shader should mirror CPU conicity/ply-steer lateral bias through heading.w"
        );
    }

    #[test]
    fn tire_brush_shader_carries_pressure_stiffness_factor() {
        let shader = include_str!("../shaders/tire_brush.wgsl");
        assert!(
            shader.contains("fn brush_pressure_stiffness_factor")
                && shader.contains("* brush_pressure_stiffness_factor(pressure)"),
            "brush shader should mirror CPU pressure-dependent tread/cornering stiffness"
        );
    }

    #[test]
    fn tire_brush_shader_carries_camber_and_pneumatic_trail_diagnostics() {
        let shader = include_str!("../shaders/tire_brush.wgsl");
        assert!(
            shader.contains("let camber_angle = clamp(in_.extra.w")
                && shader.contains("camber_bias")
                && shader.contains("camber_thrust")
                && shader.contains("pneumatic_trail")
                && shader.contains("diag3 = vec4<f32>(total_power, camber_angle, contact_center_x, pneumatic_trail)"),
            "brush shader should mirror CPU camber thrust/contact-centre/pneumatic-trail diagnostics"
        );
    }

    #[test]
    fn tire_brush_exposes_resident_output_readback_for_viewer_telemetry() {
        let source = include_str!("tire_brush.rs");
        assert!(
            source.contains("pub fn readback_resident_output")
                && source.contains("TireBrushResidentOutput")
                && source.contains("copy_buffer_to_buffer(&output.buffer")
                && source.contains("read_outputs(device, &stage, out_bytes)"),
            "resident brush-output readback should stay centralized for viewer telemetry seams"
        );
    }
}
