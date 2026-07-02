//! GPU tire-to-MPM material-point feedback.
//!
//! This is a first native tire/continuum-soil coupling lane: tire contact
//! cells apply localized force impulses and plastic compaction increments to
//! material points. It is not a full MPM grid transfer/constitutive solve; it
//! is the tire-specific particle feedback pass that can be composed with the
//! existing MPM solvers.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};

use crate::tire_fem::TireFemContactTerrain;

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct TireMpmContactCell {
    /// `(world_x, surface_y, world_z, influence_radius_m)`.
    pub center_radius: [f32; 4],
    /// `(fx_n, fy_n, fz_n, normal_pressure_pa)`.
    pub force_pressure: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct TireMpmGpuInput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    /// `(plastic_compaction, normal_impulse_n_s, shear_impulse_n_s, active_weight)`.
    pub compaction: Vec<[f32; 4]>,
    pub contacts: Vec<TireMpmContactCell>,
    pub particle_mass: f32,
    pub dt: f32,
    pub plastic_compaction_gain: f32,
    pub max_compaction: f32,
    pub damping: f32,
}

#[derive(Clone, Debug)]
pub struct TireMpmGpuOutput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub compaction: Vec<[f32; 4]>,
}

#[derive(Clone, Debug)]
pub struct TireMpmGpuInPlaceParams {
    pub particle_count: u32,
    pub particle_mass: f32,
    pub dt: f32,
    pub plastic_compaction_gain: f32,
    pub max_compaction: f32,
    pub damping: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct TireMpmTerrainRasterParams {
    pub particle_count: u32,
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub rest_height: f32,
}

#[derive(Clone, Debug)]
pub struct TireMpmTerrainRasterGpuOutput {
    pub heights: Vec<f32>,
    pub sinkage: Vec<f32>,
    pub surface_velocity_map: Vec<[f32; 4]>,
}

/// GPU/CPU-owned tire-MPM material-point state plus a heightfield bridge for
/// FEM/viewer terrain contact.
#[derive(Clone, Debug)]
pub struct TireMpmFemTerrainOwner {
    mpm: TireMpmGpuInput,
    origin: [f32; 2],
    spacing: f32,
    nx: u32,
    nz: u32,
    rest_height: f32,
    friction: f32,
    soil_rolling_resistance_scale: f32,
    surface_velocity: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TireMpmUniforms {
    dims: [u32; 4],
    params: [f32; 4],
    params2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TireMpmRasterUniforms {
    dims: [u32; 4],
    terrain0: [f32; 4],
    terrain_velocity: [f32; 4],
}

pub struct TireMpmGpuKernel {
    pipeline: wgpu::ComputePipeline,
    pipeline_in_place: wgpu::ComputePipeline,
    pipeline_raster: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_in_place: wgpu::BindGroupLayout,
    layout_raster: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    layout_raster_uniforms: wgpu::BindGroupLayout,
    layout_empty: wgpu::BindGroupLayout,
}

impl TireMpmGpuInput {
    fn validate(&self) -> Result<usize> {
        if self.positions.len() != self.velocities.len()
            || self.positions.len() != self.compaction.len()
        {
            return Err(Error::Solver(
                "tire_mpm: positions/velocities/compaction length mismatch".into(),
            ));
        }
        if self.positions.is_empty() {
            return Err(Error::Solver(
                "tire_mpm: at least one material point required".into(),
            ));
        }
        if !self.particle_mass.is_finite() || self.particle_mass <= 0.0 {
            return Err(Error::Solver(
                "tire_mpm: particle_mass must be positive".into(),
            ));
        }
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "tire_mpm: dt must be finite and nonnegative".into(),
            ));
        }
        if !self.max_compaction.is_finite() || self.max_compaction < 0.0 {
            return Err(Error::Solver(
                "tire_mpm: max_compaction must be finite and nonnegative".into(),
            ));
        }
        Ok(self.positions.len())
    }
}

impl TireMpmGpuInPlaceParams {
    fn validate(&self) -> Result<usize> {
        if self.particle_count == 0 {
            return Err(Error::Solver(
                "tire_mpm in-place: at least one material point required".into(),
            ));
        }
        if !self.particle_mass.is_finite() || self.particle_mass <= 0.0 {
            return Err(Error::Solver(
                "tire_mpm in-place: particle_mass must be positive".into(),
            ));
        }
        if !self.dt.is_finite() || self.dt < 0.0 {
            return Err(Error::Solver(
                "tire_mpm in-place: dt must be finite and nonnegative".into(),
            ));
        }
        if !self.max_compaction.is_finite() || self.max_compaction < 0.0 {
            return Err(Error::Solver(
                "tire_mpm in-place: max_compaction must be finite and nonnegative".into(),
            ));
        }
        Ok(self.particle_count as usize)
    }
}

impl TireMpmTerrainRasterParams {
    fn validate(&self) -> Result<usize> {
        if self.particle_count == 0 {
            return Err(Error::Solver(
                "tire_mpm raster: at least one material point required".into(),
            ));
        }
        if self.nx == 0 || self.nz == 0 {
            return Err(Error::Solver(
                "tire_mpm raster: nx/nz must be non-zero".into(),
            ));
        }
        if !self.spacing.is_finite() || self.spacing <= 0.0 {
            return Err(Error::Solver(
                "tire_mpm raster: spacing must be finite and positive".into(),
            ));
        }
        if !self.origin[0].is_finite()
            || !self.origin[1].is_finite()
            || !self.rest_height.is_finite()
        {
            return Err(Error::Solver(
                "tire_mpm raster: origin/rest_height must be finite".into(),
            ));
        }
        Ok(self.nx as usize * self.nz as usize)
    }
}

impl TireMpmFemTerrainOwner {
    #[allow(clippy::too_many_arguments)]
    pub fn from_input(
        input: TireMpmGpuInput,
        origin: [f32; 2],
        spacing: f32,
        nx: u32,
        nz: u32,
        rest_height: f32,
        friction: f32,
    ) -> Result<Self> {
        input.validate()?;
        validate_raster(origin, spacing, nx, nz)?;
        Ok(Self {
            mpm: input,
            origin,
            spacing,
            nx,
            nz,
            rest_height,
            friction: friction.max(0.0),
            soil_rolling_resistance_scale: 1.0,
            surface_velocity: [0.0; 3],
        })
    }

    pub fn mpm_input(&self) -> &TireMpmGpuInput {
        &self.mpm
    }

    pub fn positions(&self) -> &[[f32; 4]] {
        &self.mpm.positions
    }

    pub fn velocities(&self) -> &[[f32; 4]] {
        &self.mpm.velocities
    }

    pub fn compaction(&self) -> &[[f32; 4]] {
        &self.mpm.compaction
    }

    pub fn set_soil_rolling_resistance_scale(&mut self, scale: f32) {
        self.soil_rolling_resistance_scale = if scale.is_finite() {
            scale.max(0.0)
        } else {
            0.0
        };
    }

    pub fn set_surface_velocity(&mut self, surface_velocity: [f32; 3]) {
        self.surface_velocity = surface_velocity.map(|v| if v.is_finite() { v } else { 0.0 });
    }

    pub fn apply_contacts_cpu(&mut self, contacts: &[TireMpmContactCell]) -> Result<()> {
        self.mpm.contacts = contacts.to_vec();
        let out = tire_mpm_reference_feedback(&self.mpm)?;
        self.mpm.positions = out.positions;
        self.mpm.velocities = out.velocities;
        self.mpm.compaction = out.compaction;
        Ok(())
    }

    pub fn apply_contacts_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireMpmGpuKernel,
        contacts: &[TireMpmContactCell],
    ) -> Result<()> {
        self.mpm.contacts = contacts.to_vec();
        let out = kernel.dispatch_blocking(device, queue, &self.mpm)?;
        self.mpm.positions = out.positions;
        self.mpm.velocities = out.velocities;
        self.mpm.compaction = out.compaction;
        Ok(())
    }

    pub fn raster_height_sinkage_velocity(&self) -> (Vec<f32>, Vec<f32>, Vec<[f32; 4]>) {
        raster_height_sinkage_velocity(
            &self.mpm.positions,
            &self.mpm.velocities,
            &self.mpm.compaction,
            self.origin,
            self.spacing,
            self.nx,
            self.nz,
            self.rest_height,
            self.surface_velocity,
        )
    }

    pub fn fem_contact_terrain(&self) -> TireFemContactTerrain {
        let (heights, sinkage, surface_velocity_map) = self.raster_height_sinkage_velocity();
        TireFemContactTerrain {
            origin: self.origin,
            spacing: self.spacing,
            nx: self.nx,
            nz: self.nz,
            heights,
            friction: self.friction,
            friction_map: None,
            water_film_map: None,
            sinkage_map: Some(sinkage),
            soil_rolling_resistance_scale: self.soil_rolling_resistance_scale,
            surface_velocity: self.surface_velocity,
            surface_velocity_map: Some(surface_velocity_map),
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        }
    }
}

fn validate_raster(_origin: [f32; 2], spacing: f32, nx: u32, nz: u32) -> Result<()> {
    if nx == 0 || nz == 0 {
        return Err(Error::Solver(
            "tire_mpm terrain owner: nx/nz must be non-zero".into(),
        ));
    }
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(Error::Solver(
            "tire_mpm terrain owner: spacing must be positive".into(),
        ));
    }
    Ok(())
}

fn raster_height_sinkage_velocity(
    positions: &[[f32; 4]],
    velocities: &[[f32; 4]],
    compaction: &[[f32; 4]],
    origin: [f32; 2],
    spacing: f32,
    nx: u32,
    nz: u32,
    rest_height: f32,
    fallback_velocity: [f32; 3],
) -> (Vec<f32>, Vec<f32>, Vec<[f32; 4]>) {
    let n = nx as usize * nz as usize;
    let mut heights = vec![rest_height; n];
    let mut sinkage = vec![0.0_f32; n];
    let mut velocity_sum = vec![[0.0_f32; 3]; n];
    let mut velocity_count = vec![0.0_f32; n];
    for ((p, v), c) in positions
        .iter()
        .zip(velocities.iter())
        .zip(compaction.iter())
    {
        let ix = (((p[0] - origin[0]) / spacing).floor() as i32).clamp(0, nx as i32 - 1) as usize;
        let iz = (((p[2] - origin[1]) / spacing).floor() as i32).clamp(0, nz as i32 - 1) as usize;
        let idx = iz * nx as usize + ix;
        let compact = c[0].max(0.0);
        sinkage[idx] = sinkage[idx].max(compact);
        heights[idx] = heights[idx].min(rest_height - compact);
        velocity_sum[idx][0] += v[0];
        velocity_sum[idx][1] += v[1];
        velocity_sum[idx][2] += v[2];
        velocity_count[idx] += 1.0;
    }
    let fallback = [
        fallback_velocity[0],
        fallback_velocity[1],
        fallback_velocity[2],
        0.0,
    ];
    let surface_velocity_map = velocity_sum
        .iter()
        .zip(velocity_count.iter())
        .map(|(sum, count)| {
            if *count > 0.0 {
                [sum[0] / *count, sum[1] / *count, sum[2] / *count, 0.0]
            } else {
                fallback
            }
        })
        .collect();
    (heights, sinkage, surface_velocity_map)
}

fn tire_mpm_delta(
    position: [f32; 4],
    contacts: &[TireMpmContactCell],
    mass: f32,
    dt: f32,
    compaction_gain: f32,
    damping: f32,
) -> ([f32; 3], [f32; 4], f32) {
    let mut dv = [0.0_f32; 3];
    let mut accum = [0.0_f32; 4];
    let mut damping_weight = 0.0_f32;
    for contact in contacts {
        let [cx, surface_y, cz, influence] = contact.center_radius;
        let influence = influence.max(1.0e-6);
        let dx = position[0] - cx;
        let dz = position[2] - cz;
        let horizontal = (dx * dx + dz * dz).sqrt();
        if horizontal > influence {
            continue;
        }
        let below_surface = surface_y - position[1];
        if below_surface < -influence * 0.5 || below_surface > influence * 2.0 {
            continue;
        }
        let radial_w = (1.0 - horizontal / influence).clamp(0.0, 1.0);
        let depth_w = (1.0 - below_surface.max(0.0) / (influence * 2.0)).clamp(0.10, 1.0);
        let w = radial_w * radial_w * depth_w;
        let shear = (contact.force_pressure[0] * contact.force_pressure[0]
            + contact.force_pressure[2] * contact.force_pressure[2])
            .sqrt();
        let pressure = contact.force_pressure[3].max((-contact.force_pressure[1]).max(0.0));
        dv[0] += contact.force_pressure[0] * w * dt / mass;
        dv[1] += contact.force_pressure[1] * w * dt / mass;
        dv[2] += contact.force_pressure[2] * w * dt / mass;
        accum[0] += pressure * w * compaction_gain * dt;
        accum[1] += (-contact.force_pressure[1]).max(0.0) * w * dt;
        accum[2] += shear * w * dt;
        accum[3] += w;
        damping_weight = damping_weight.max(w);
    }
    (dv, accum, (damping * damping_weight * dt).clamp(0.0, 0.95))
}

/// CPU reference for [`TireMpmGpuKernel`].
pub fn tire_mpm_reference_feedback(input: &TireMpmGpuInput) -> Result<TireMpmGpuOutput> {
    let n = input.validate()?;
    let mut positions = input.positions.clone();
    let mut velocities = input.velocities.clone();
    let mut compaction = input.compaction.clone();
    let mass = input.particle_mass.max(1.0e-9);
    let dt = input.dt.max(0.0);
    let gain = input.plastic_compaction_gain.max(0.0);
    let max_compaction = input.max_compaction.max(0.0);
    let damping = input.damping.max(0.0);

    for i in 0..n {
        let (dv, accum, damp) =
            tire_mpm_delta(positions[i], &input.contacts, mass, dt, gain, damping);
        velocities[i][0] = (velocities[i][0] + dv[0]) * (1.0 - damp);
        velocities[i][1] = (velocities[i][1] + dv[1]) * (1.0 - damp);
        velocities[i][2] = (velocities[i][2] + dv[2]) * (1.0 - damp);
        positions[i][0] += velocities[i][0] * dt;
        positions[i][1] += velocities[i][1] * dt;
        positions[i][2] += velocities[i][2] * dt;
        compaction[i][0] = (compaction[i][0] + accum[0]).min(max_compaction);
        compaction[i][1] += accum[1];
        compaction[i][2] += accum[2];
        compaction[i][3] += accum[3];
    }

    Ok(TireMpmGpuOutput {
        positions,
        velocities,
        compaction,
    })
}

impl TireMpmGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_mpm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_mpm.wgsl").into()),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_mpm io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
            ],
        });
        let layout_in_place = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_mpm in-place io"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, true),
            ],
        });
        let layout_raster = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_mpm raster io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_mpm uniforms"),
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
        let layout_raster_uniforms =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_mpm raster uniforms"),
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
        let layout_empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_mpm empty group"),
            entries: &[],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_mpm layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline_layout_in_place =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_mpm in-place layout"),
                bind_group_layouts: &[&layout_empty, &layout_uniforms, &layout_in_place],
                push_constant_ranges: &[],
            });
        let pipeline_layout_raster =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_mpm raster layout"),
                bind_group_layouts: &[
                    &layout_empty,
                    &layout_empty,
                    &layout_empty,
                    &layout_raster,
                    &layout_raster_uniforms,
                ],
                push_constant_ranges: &[],
            });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_mpm::apply_feedback"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("apply_feedback"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_in_place = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_mpm::apply_feedback_in_place"),
            layout: Some(&pipeline_layout_in_place),
            module: &shader,
            entry_point: Some("apply_feedback_in_place"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_raster = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_mpm::raster_terrain"),
            layout: Some(&pipeline_layout_raster),
            module: &shader,
            entry_point: Some("raster_terrain"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            pipeline_in_place,
            pipeline_raster,
            layout_io,
            layout_in_place,
            layout_raster,
            layout_uniforms,
            layout_raster_uniforms,
            layout_empty,
        }
    }

    pub fn dispatch_raster_to_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        compaction: &wgpu::Buffer,
        heights: &wgpu::Buffer,
        sinkage: &wgpu::Buffer,
        surface_velocity_map: &wgpu::Buffer,
        params: &TireMpmTerrainRasterParams,
        fallback_surface_velocity: [f32; 3],
    ) -> Result<()> {
        let cell_count = params.validate()?;
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_mpm raster uniforms"),
            size: core::mem::size_of::<TireMpmRasterUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireMpmRasterUniforms {
                dims: [params.particle_count, params.nx, params.nz, 0],
                terrain0: [
                    params.origin[0],
                    params.origin[1],
                    params.spacing,
                    params.rest_height,
                ],
                terrain_velocity: [
                    fallback_surface_velocity[0],
                    fallback_surface_velocity[1],
                    fallback_surface_velocity[2],
                    0.0,
                ],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_mpm raster io bg"),
            layout: &self.layout_raster,
            entries: &[
                entry(0, positions),
                entry(1, compaction),
                entry(2, heights),
                entry(3, sinkage),
                entry(4, velocities),
                entry(5, surface_velocity_map),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_mpm raster uniforms bg"),
            layout: &self.layout_raster_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let bg_empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_mpm raster empty bg"),
            layout: &self.layout_empty,
            entries: &[],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_mpm raster encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_mpm raster pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_raster);
            pass.set_bind_group(0, &bg_empty, &[]);
            pass.set_bind_group(1, &bg_empty, &[]);
            pass.set_bind_group(2, &bg_empty, &[]);
            pass.set_bind_group(3, &bg_io, &[]);
            pass.set_bind_group(4, &bg_uniforms, &[]);
            pass.dispatch_workgroups((cell_count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn dispatch_raster_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions_input: &[[f32; 4]],
        velocities_input: &[[f32; 4]],
        compaction_input: &[[f32; 4]],
        params: &TireMpmTerrainRasterParams,
        fallback_surface_velocity: [f32; 3],
    ) -> Result<TireMpmTerrainRasterGpuOutput> {
        let cell_count = params.validate()?;
        if positions_input.len() != params.particle_count as usize
            || velocities_input.len() != params.particle_count as usize
            || compaction_input.len() != params.particle_count as usize
        {
            return Err(Error::Solver(
                "tire_mpm raster: input lengths must match particle_count".into(),
            ));
        }
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let f32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<f32>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let positions = mk_buf(
            "tire_mpm raster positions",
            vec4_bytes(positions_input.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let compaction = mk_buf(
            "tire_mpm raster compaction",
            vec4_bytes(compaction_input.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let velocities = mk_buf(
            "tire_mpm raster velocities",
            vec4_bytes(velocities_input.len()),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let heights = mk_buf(
            "tire_mpm raster heights",
            f32_bytes(cell_count),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let sinkage = mk_buf(
            "tire_mpm raster sinkage",
            f32_bytes(cell_count),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let surface_velocity_map = mk_buf(
            "tire_mpm raster surface velocity",
            vec4_bytes(cell_count),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let stage_heights = mk_buf(
            "tire_mpm raster stage heights",
            f32_bytes(cell_count),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let stage_sinkage = mk_buf(
            "tire_mpm raster stage sinkage",
            f32_bytes(cell_count),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let stage_surface_velocity = mk_buf(
            "tire_mpm raster stage surface velocity",
            vec4_bytes(cell_count),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        queue.write_buffer(&positions, 0, bytemuck::cast_slice(positions_input));
        queue.write_buffer(&velocities, 0, bytemuck::cast_slice(velocities_input));
        queue.write_buffer(&compaction, 0, bytemuck::cast_slice(compaction_input));
        self.dispatch_raster_to_buffers(
            device,
            queue,
            &positions,
            &velocities,
            &compaction,
            &heights,
            &sinkage,
            &surface_velocity_map,
            params,
            fallback_surface_velocity,
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_mpm raster readback encoder"),
        });
        let bytes = f32_bytes(cell_count);
        encoder.copy_buffer_to_buffer(&heights, 0, &stage_heights, 0, bytes);
        encoder.copy_buffer_to_buffer(&sinkage, 0, &stage_sinkage, 0, bytes);
        encoder.copy_buffer_to_buffer(
            &surface_velocity_map,
            0,
            &stage_surface_velocity,
            0,
            vec4_bytes(cell_count),
        );
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireMpmTerrainRasterGpuOutput {
            heights: read_f32(device, &stage_heights, "tire_mpm raster heights")?,
            sinkage: read_f32(device, &stage_sinkage, "tire_mpm raster sinkage")?,
            surface_velocity_map: read_vec4(
                device,
                &stage_surface_velocity,
                "tire_mpm raster surface velocity",
            )?,
        })
    }

    pub fn dispatch_in_place(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        compaction: &wgpu::Buffer,
        contacts_input: &[TireMpmContactCell],
        params: &TireMpmGpuInPlaceParams,
    ) -> Result<()> {
        let n = params.validate()?;
        let contact_len = contacts_input.len().max(1);
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireMpmContactCell>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let contacts = mk_buf(
            "tire_mpm in-place contacts",
            contact_bytes(contact_len),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let uniforms = mk_buf(
            "tire_mpm in-place uniforms",
            core::mem::size_of::<TireMpmUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        if contacts_input.is_empty() {
            queue.write_buffer(
                &contacts,
                0,
                bytemuck::cast_slice(&[TireMpmContactCell::default()]),
            );
        } else {
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(contacts_input));
        }
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireMpmUniforms {
                dims: [n as u32, contacts_input.len() as u32, 0, 0],
                params: [
                    params.particle_mass,
                    params.dt,
                    params.plastic_compaction_gain,
                    params.max_compaction,
                ],
                params2: [params.damping, 0.0, 0.0, 0.0],
            }),
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_mpm in-place io bg"),
            layout: &self.layout_in_place,
            entries: &[
                entry(0, positions),
                entry(1, velocities),
                entry(2, compaction),
                entry(3, &contacts),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_mpm in-place uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let bg_empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_mpm in-place empty bg"),
            layout: &self.layout_empty,
            entries: &[],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_mpm in-place encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_mpm in-place pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_in_place);
            pass.set_bind_group(0, &bg_empty, &[]);
            pass.set_bind_group(2, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireMpmGpuInput,
    ) -> Result<TireMpmGpuOutput> {
        let n = input.validate()?;
        let contact_len = input.contacts.len().max(1);
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireMpmContactCell>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let pos_in = mk_buf("tire_mpm pos in", vec4_bytes(n), usage_ro);
        let vel_in = mk_buf("tire_mpm vel in", vec4_bytes(n), usage_ro);
        let compact_in = mk_buf("tire_mpm compact in", vec4_bytes(n), usage_ro);
        let contacts = mk_buf("tire_mpm contacts", contact_bytes(contact_len), usage_ro);
        let pos_out = mk_buf("tire_mpm pos out", vec4_bytes(n), usage_rw);
        let vel_out = mk_buf("tire_mpm vel out", vec4_bytes(n), usage_rw);
        let compact_out = mk_buf("tire_mpm compact out", vec4_bytes(n), usage_rw);
        let uniforms = mk_buf(
            "tire_mpm uniforms",
            core::mem::size_of::<TireMpmUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let stage_pos = stage_vec4(device, "tire_mpm stage pos", n);
        let stage_vel = stage_vec4(device, "tire_mpm stage vel", n);
        let stage_compact = stage_vec4(device, "tire_mpm stage compact", n);

        queue.write_buffer(&pos_in, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_in, 0, bytemuck::cast_slice(&input.velocities));
        queue.write_buffer(&compact_in, 0, bytemuck::cast_slice(&input.compaction));
        if input.contacts.is_empty() {
            queue.write_buffer(
                &contacts,
                0,
                bytemuck::cast_slice(&[TireMpmContactCell::default()]),
            );
        } else {
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&input.contacts));
        }
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&TireMpmUniforms {
                dims: [n as u32, input.contacts.len() as u32, 0, 0],
                params: [
                    input.particle_mass,
                    input.dt,
                    input.plastic_compaction_gain,
                    input.max_compaction,
                ],
                params2: [input.damping, 0.0, 0.0, 0.0],
            }),
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_mpm io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &pos_in),
                entry(1, &vel_in),
                entry(2, &compact_in),
                entry(3, &contacts),
                entry(4, &pos_out),
                entry(5, &vel_out),
                entry(6, &compact_out),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_mpm uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_mpm encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_mpm pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let bytes = vec4_bytes(n);
        encoder.copy_buffer_to_buffer(&pos_out, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&vel_out, 0, &stage_vel, 0, bytes);
        encoder.copy_buffer_to_buffer(&compact_out, 0, &stage_compact, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireMpmGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_mpm positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_mpm velocities")?,
            compaction: read_vec4(device, &stage_compact, "tire_mpm compaction")?,
        })
    }
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

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn stage_vec4(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read_vec4(device: &wgpu::Device, buf: &wgpu::Buffer, label: &str) -> Result<Vec<[f32; 4]>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| Error::Solver(format!("{label}: map callback dropped")))?
        .map_err(|e| Error::Solver(format!("{label}: map failed: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}

fn read_f32(device: &wgpu::Device, buf: &wgpu::Buffer, label: &str) -> Result<Vec<f32>> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| Error::Solver(format!("{label}: map callback dropped")))?
        .map_err(|e| Error::Solver(format!("{label}: map failed: {e:?}")))?;
    let data = slice.get_mapped_range();
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}
