//! GPU Bekker deformable-terrain writeback for tire contact cells.
//!
//! The CPU [`molla_vehicle::DeformableTerrain`] path already applies tire
//! contact pressure into a plastic Bekker heightfield. This kernel is the
//! GPU counterpart for the high-frequency tire path: it updates row-major
//! height/sinkage maps from tire contact-cell pressure/shear without float
//! atomics by assigning one invocation to each terrain cell.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};
use std::sync::Arc;

use crate::tire_fem::{TireFemContactSdfPrimitive, TireFemContactTerrain, TireFemContactTriangle};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, Default)]
pub struct TireBekkerContactCell {
    /// `(world_x, world_z, normal_pressure_pa, shear_pressure_pa)`.
    pub xz_pressure: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct TireBekkerGpuInput {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub rest_heights: Vec<f32>,
    pub heights: Vec<f32>,
    pub sinkage: Vec<f32>,
    pub contacts: Vec<TireBekkerContactCell>,
    pub stiffness: f32,
    pub max_sinkage: f32,
    pub shear_strength: f32,
    pub shear_sinkage_gain: f32,
    pub bekker_n: f32,
}

#[derive(Clone, Debug)]
pub struct TireBekkerGpuOutput {
    pub heights: Vec<f32>,
    pub sinkage: Vec<f32>,
}

/// GPU-resident Bekker writeback maps ready for FEM contact feedback.
///
/// The height/sinkage buffers are produced by `tire_bekker.wgsl` and can be
/// passed directly to `TireFemContactFeedbackGpuTerrain`. Optional friction and
/// water-film maps are uploaded by the owner so the FEM feedback shader does
/// not have to fall back to CPU terrain export just to see surface state.
pub struct TireBekkerGpuResidentTerrain {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub friction: f32,
    pub heights: wgpu::Buffer,
    pub sinkage: wgpu::Buffer,
    pub friction_map: Option<wgpu::Buffer>,
    pub water_film_map: Option<wgpu::Buffer>,
    pub surface_velocity_map: Option<wgpu::Buffer>,
    pub surface_velocity: [f32; 3],
    pub soil_rolling_resistance_scale: f32,
}

/// GPU-owned Bekker height/sinkage maps for FEM contact terrain feedback.
///
/// This is the first live ownership bridge between the Bekker writeback kernel
/// and the FEM contact kernels: tire contact cells update the plastic height
/// and sinkage maps, then [`Self::fem_contact_terrain`] exports those maps as a
/// [`TireFemContactTerrain`] so GPU FEM reduction/feedback can sample rut depth
/// for rolling loss and contact diagnostics. The CPU writeback method is a
/// deterministic fallback/reference path for tests and non-GPU callers.
#[derive(Clone, Debug)]
pub struct TireBekkerFemTerrainOwner {
    bekker: TireBekkerGpuInput,
    friction: f32,
    friction_map: BekkerMaterialMap<f32>,
    water_film_map: BekkerMaterialMap<f32>,
    surface_velocity_map: BekkerMaterialMap<[f32; 4]>,
    material_kernel: Option<Arc<()>>,
    surface_velocity: [f32; 3],
    soil_rolling_resistance_scale: f32,
    sdf_primitives: Vec<TireFemContactSdfPrimitive>,
    triangles: Vec<TireFemContactTriangle>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BekkerUniforms {
    dims: [u32; 4],
    origin: [f32; 4],
    soil0: [f32; 4],
    soil1: [f32; 4],
}

pub struct TireBekkerGpuKernel {
    material_identity: Arc<()>,
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

/// Output buffers from a Bekker GPU writeback dispatch.
pub struct TireBekkerResidentOutput {
    pub heights: wgpu::Buffer,
    pub sinkage: wgpu::Buffer,
    pub cell_count: usize,
}

#[derive(Clone, Debug, Default)]
struct BekkerMaterialMap<T> {
    values: Option<Vec<T>>,
    uniform: Option<T>,
    buffer: Option<wgpu::Buffer>,
}

impl<T: Pod + PartialEq> BekkerMaterialMap<T> {
    fn set(&mut self, values: Option<Vec<T>>) {
        if self.values != values {
            self.values = values;
            self.uniform = None;
            self.buffer = None;
        }
    }

    fn set_uniform(&mut self, value: T, len: usize) -> bool {
        if self.uniform == Some(value) && self.values.as_ref().is_some_and(|v| v.len() == len) {
            return false;
        }
        if self
            .values
            .as_ref()
            .is_some_and(|v| v.len() == len && v.iter().all(|&x| x == value))
        {
            self.uniform = Some(value);
            return false;
        }
        match self.values.as_mut() {
            Some(values) => {
                values.resize(len, value);
                values.fill(value);
            }
            None => self.values = Some(vec![value; len]),
        }
        self.uniform = Some(value);
        self.buffer = None;
        true
    }

    fn gpu_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
    ) -> Option<wgpu::Buffer> {
        let values = self.values.as_ref()?;
        if self.buffer.is_none() {
            let buffer = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: core::mem::size_of_val(values.as_slice()).max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }
            );
            if !values.is_empty() {
                molla_core::gpu_write_buffer!(queue, &buffer, 0, bytemuck::cast_slice(values));
            }
            self.buffer = Some(buffer);
        }
        self.buffer.clone()
    }
}

impl TireBekkerGpuInput {
    fn cell_count(&self) -> Result<usize> {
        if self.nx == 0 || self.nz == 0 {
            return Err(Error::Solver("tire_bekker: nx/nz must be non-zero".into()));
        }
        let n = self.nx as usize * self.nz as usize;
        if self.rest_heights.len() != n || self.heights.len() != n || self.sinkage.len() != n {
            return Err(Error::Solver(
                "tire_bekker: height/sinkage arrays must match nx*nz".into(),
            ));
        }
        if !self.spacing.is_finite() || self.spacing <= 0.0 {
            return Err(Error::Solver(
                "tire_bekker: spacing must be positive".into(),
            ));
        }
        Ok(n)
    }
}

impl TireBekkerFemTerrainOwner {
    #[allow(clippy::too_many_arguments)]
    pub fn flat(
        origin: [f32; 2],
        spacing: f32,
        nx: u32,
        nz: u32,
        rest_height: f32,
        friction: f32,
        stiffness: f32,
        max_sinkage: f32,
        shear_strength: f32,
        shear_sinkage_gain: f32,
        bekker_n: f32,
    ) -> Result<Self> {
        let n = nx as usize * nz as usize;
        let owner = Self {
            bekker: TireBekkerGpuInput {
                origin,
                spacing,
                nx,
                nz,
                rest_heights: vec![rest_height; n],
                heights: vec![rest_height; n],
                sinkage: vec![0.0; n],
                contacts: Vec::new(),
                stiffness,
                max_sinkage,
                shear_strength,
                shear_sinkage_gain,
                bekker_n,
            },
            friction: friction.max(0.0),
            friction_map: BekkerMaterialMap::default(),
            water_film_map: BekkerMaterialMap::default(),
            surface_velocity_map: BekkerMaterialMap::default(),
            material_kernel: None,
            surface_velocity: [0.0; 3],
            soil_rolling_resistance_scale: 1.0,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        };
        owner.bekker.cell_count()?;
        Ok(owner)
    }

    pub fn from_bekker_input(input: TireBekkerGpuInput, friction: f32) -> Result<Self> {
        input.cell_count()?;
        Ok(Self {
            bekker: input,
            friction: friction.max(0.0),
            friction_map: BekkerMaterialMap::default(),
            water_film_map: BekkerMaterialMap::default(),
            surface_velocity_map: BekkerMaterialMap::default(),
            material_kernel: None,
            surface_velocity: [0.0; 3],
            soil_rolling_resistance_scale: 1.0,
            sdf_primitives: Vec::new(),
            triangles: Vec::new(),
        })
    }

    pub fn set_friction_map(&mut self, friction_map: Option<Vec<f32>>) -> Result<()> {
        if let Some(map) = &friction_map {
            let n = self.bekker.cell_count()?;
            if map.len() != n {
                return Err(Error::Solver(
                    "tire_bekker terrain owner: friction_map must match nx*nz".into(),
                ));
            }
        }
        self.friction_map.set(friction_map);
        Ok(())
    }

    pub fn set_water_film_map(&mut self, water_film_map: Option<Vec<f32>>) -> Result<()> {
        if let Some(map) = &water_film_map {
            let n = self.bekker.cell_count()?;
            if map.len() != n {
                return Err(Error::Solver(
                    "tire_bekker terrain owner: water_film_map must match nx*nz".into(),
                ));
            }
        }
        self.water_film_map.set(water_film_map);
        Ok(())
    }

    pub fn set_surface_velocity_map(
        &mut self,
        surface_velocity_map: Option<Vec<[f32; 4]>>,
    ) -> Result<()> {
        if let Some(map) = &surface_velocity_map {
            let n = self.bekker.cell_count()?;
            if map.len() != n {
                return Err(Error::Solver(
                    "tire_bekker terrain owner: surface_velocity_map must match nx*nz".into(),
                ));
            }
            if map.iter().flatten().any(|v| !v.is_finite()) {
                return Err(Error::Solver(
                    "tire_bekker terrain owner: surface_velocity_map must be finite".into(),
                ));
            }
        }
        self.surface_velocity_map.set(surface_velocity_map);
        Ok(())
    }

    /// Fill the friction map, returning whether its values changed.
    pub fn set_uniform_friction_map(&mut self, friction: f32) -> Result<bool> {
        if !friction.is_finite() {
            return Err(Error::Solver(
                "tire_bekker: uniform friction must be finite".into(),
            ));
        }
        let n = self.bekker.cell_count()?;
        Ok(self.friction_map.set_uniform(friction, n))
    }

    /// Fill the water-film map, returning whether its values changed.
    pub fn set_uniform_water_film_map(&mut self, depth_m: f32) -> Result<bool> {
        if !depth_m.is_finite() {
            return Err(Error::Solver(
                "tire_bekker: uniform water film must be finite".into(),
            ));
        }
        let n = self.bekker.cell_count()?;
        Ok(self.water_film_map.set_uniform(depth_m, n))
    }

    pub fn set_surface_velocity(&mut self, surface_velocity: [f32; 3]) {
        self.surface_velocity = surface_velocity.map(|v| if v.is_finite() { v } else { 0.0 });
    }

    pub fn set_soil_rolling_resistance_scale(&mut self, scale: f32) {
        self.soil_rolling_resistance_scale = if scale.is_finite() {
            scale.max(0.0)
        } else {
            0.0
        };
    }

    pub fn set_sdf_primitives(&mut self, primitives: Vec<TireFemContactSdfPrimitive>) {
        self.sdf_primitives = primitives;
    }

    pub fn set_triangles(&mut self, triangles: Vec<TireFemContactTriangle>) {
        self.triangles = triangles;
    }

    pub fn bekker_input(&self) -> &TireBekkerGpuInput {
        &self.bekker
    }

    pub fn heights(&self) -> &[f32] {
        &self.bekker.heights
    }

    pub fn sinkage(&self) -> &[f32] {
        &self.bekker.sinkage
    }

    pub fn apply_contacts_cpu(&mut self, contacts: &[TireBekkerContactCell]) -> Result<()> {
        self.bekker.contacts = contacts.to_vec();
        let out = tire_bekker_reference_writeback(&self.bekker)?;
        self.bekker.heights = out.heights;
        self.bekker.sinkage = out.sinkage;
        Ok(())
    }

    pub fn apply_contacts_gpu(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireBekkerGpuKernel,
        contacts: &[TireBekkerContactCell],
    ) -> Result<()> {
        self.bekker.contacts = contacts.to_vec();
        let out = kernel.dispatch_blocking(device, queue, &self.bekker)?;
        self.bekker.heights = out.heights;
        self.bekker.sinkage = out.sinkage;
        Ok(())
    }

    /// Apply contacts through the GPU writeback path and keep the produced
    /// height/sinkage maps available as GPU-resident FEM feedback terrain.
    ///
    /// The owner still mirrors those maps back into its CPU arrays for viewer
    /// drawing and reset/fallback determinism, but downstream FEM contact can
    /// consume the returned buffers directly instead of rebuilding a CPU
    /// heightfield first.
    /// Unchanged material maps share buffers; edits create new buffers so
    /// previously returned terrain views retain their material values.
    pub fn apply_contacts_gpu_resident(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireBekkerGpuKernel,
        contacts: &[TireBekkerContactCell],
    ) -> Result<TireBekkerGpuResidentTerrain> {
        self.bekker.contacts = contacts.to_vec();
        let resident = kernel.dispatch_resident(device, queue, &self.bekker)?;
        let out = kernel.readback_resident_output(device, queue, &resident)?;
        self.bekker.heights = out.heights;
        self.bekker.sinkage = out.sinkage;
        Ok(self.resident_terrain_from_buffers(device, queue, kernel, resident))
    }

    fn resident_terrain_from_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        kernel: &TireBekkerGpuKernel,
        resident: TireBekkerResidentOutput,
    ) -> TireBekkerGpuResidentTerrain {
        if self
            .material_kernel
            .as_ref()
            .is_none_or(|id| !Arc::ptr_eq(id, &kernel.material_identity))
        {
            self.friction_map.buffer = None;
            self.water_film_map.buffer = None;
            self.surface_velocity_map.buffer = None;
            self.material_kernel = Some(kernel.material_identity.clone());
        }
        TireBekkerGpuResidentTerrain {
            origin: self.bekker.origin,
            spacing: self.bekker.spacing,
            nx: self.bekker.nx,
            nz: self.bekker.nz,
            friction: self.friction,
            heights: resident.heights,
            sinkage: resident.sinkage,
            friction_map: self.friction_map.gpu_buffer(
                device,
                queue,
                "tire_bekker resident friction map",
            ),
            water_film_map: self.water_film_map.gpu_buffer(
                device,
                queue,
                "tire_bekker resident water film map",
            ),
            surface_velocity_map: self.surface_velocity_map.gpu_buffer(
                device,
                queue,
                "tire_bekker resident surface velocity map",
            ),
            surface_velocity: self.surface_velocity,
            soil_rolling_resistance_scale: self.soil_rolling_resistance_scale,
        }
    }

    pub fn fem_contact_terrain(&self) -> TireFemContactTerrain {
        TireFemContactTerrain {
            origin: self.bekker.origin,
            spacing: self.bekker.spacing,
            nx: self.bekker.nx,
            nz: self.bekker.nz,
            heights: self.bekker.heights.clone(),
            friction: self.friction,
            friction_map: self.friction_map.values.clone(),
            water_film_map: self.water_film_map.values.clone(),
            sinkage_map: Some(self.bekker.sinkage.clone()),
            soil_rolling_resistance_scale: self.soil_rolling_resistance_scale,
            surface_velocity: self.surface_velocity,
            surface_velocity_map: self.surface_velocity_map.values.clone(),
            sdf_primitives: self.sdf_primitives.clone(),
            triangles: self.triangles.clone(),
        }
    }
}

/// CPU reference for the GPU Bekker writeback.
///
/// This is used by tests and by callers that need a deterministic fallback.
pub fn tire_bekker_reference_writeback(input: &TireBekkerGpuInput) -> Result<TireBekkerGpuOutput> {
    let n = input.cell_count()?;
    let mut heights = input.heights.clone();
    let mut sinkage = input.sinkage.clone();
    let stiffness = input.stiffness.max(1.0e-9);
    let max_sinkage = input.max_sinkage.max(0.0);
    let shear_strength = input.shear_strength.max(1.0);
    let shear_gain = input.shear_sinkage_gain.max(0.0);
    let bekker_n = input.bekker_n.max(1.0e-6);

    for idx in 0..n {
        sinkage[idx] = sinkage[idx]
            .max(input.rest_heights[idx] - input.heights[idx])
            .max(0.0);
    }
    for contact in &input.contacts {
        let [x, z, normal_pressure, shear_pressure] = contact.xz_pressure;
        let ix = ((x - input.origin[0]) / input.spacing)
            .round()
            .clamp(0.0, input.nx.saturating_sub(1) as f32) as usize;
        let iz = ((z - input.origin[1]) / input.spacing)
            .round()
            .clamp(0.0, input.nz.saturating_sub(1) as f32) as usize;
        let idx = iz * input.nx as usize + ix;
        let normal_sinkage = (normal_pressure.max(0.0) / stiffness)
            .powf(1.0 / bekker_n)
            .min(max_sinkage);
        let overshear = (shear_pressure.max(0.0) / shear_strength - 1.0).max(0.0);
        let shear_sinkage = (overshear * shear_gain * max_sinkage).min(max_sinkage * 0.45);
        sinkage[idx] = sinkage[idx].max((normal_sinkage + shear_sinkage).min(max_sinkage));
    }
    for idx in 0..n {
        heights[idx] = input.rest_heights[idx] - sinkage[idx];
    }
    Ok(TireBekkerGpuOutput { heights, sinkage })
}

impl TireBekkerGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_bekker"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_bekker.wgsl").into()),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_bekker io"),
            entries: &[
                storage_entry(0, true),  // rest heights
                storage_entry(1, true),  // current heights
                storage_entry(2, true),  // current sinkage
                storage_entry(3, true),  // contact cells
                storage_entry(4, false), // output heights
                storage_entry(5, false), // output sinkage
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_bekker uniforms"),
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
            label: Some("molla::tire_bekker layout"),
            bind_group_layouts: &[Some(&layout_io), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_bekker::writeback"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bekker_writeback"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            material_identity: Arc::new(()),
            pipeline,
            layout_io,
            layout_uniforms,
        }
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireBekkerGpuInput,
    ) -> Result<TireBekkerGpuOutput> {
        let n = input.cell_count()?;
        let contact_len = input.contacts.len().max(1);
        let f32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<f32>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireBekkerContactCell>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes.max(4),
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let rest_buf = mk_buf("tire_bekker rest", f32_bytes(n), usage_ro);
        let height_in_buf = mk_buf("tire_bekker height in", f32_bytes(n), usage_ro);
        let sinkage_in_buf = mk_buf("tire_bekker sinkage in", f32_bytes(n), usage_ro);
        let contact_buf = mk_buf("tire_bekker contacts", contact_bytes(contact_len), usage_ro);
        let height_out_buf = mk_buf("tire_bekker height out", f32_bytes(n), usage_rw);
        let sinkage_out_buf = mk_buf("tire_bekker sinkage out", f32_bytes(n), usage_rw);
        let uniform_buf = mk_buf(
            "tire_bekker uniforms",
            core::mem::size_of::<BekkerUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let stage_height = stage_f32(device, "tire_bekker stage height", n);
        let stage_sinkage = stage_f32(device, "tire_bekker stage sinkage", n);

        molla_core::gpu_write_buffer!(
            queue,
            &rest_buf,
            0,
            bytemuck::cast_slice(&input.rest_heights)
        );
        molla_core::gpu_write_buffer!(
            queue,
            &height_in_buf,
            0,
            bytemuck::cast_slice(&input.heights)
        );
        molla_core::gpu_write_buffer!(
            queue,
            &sinkage_in_buf,
            0,
            bytemuck::cast_slice(&input.sinkage)
        );
        if input.contacts.is_empty() {
            molla_core::gpu_write_buffer!(
                queue,
                &contact_buf,
                0,
                bytemuck::cast_slice(&[TireBekkerContactCell::default()]),
            );
        } else {
            molla_core::gpu_write_buffer!(
                queue,
                &contact_buf,
                0,
                bytemuck::cast_slice(&input.contacts)
            );
        }
        let uniforms = BekkerUniforms {
            dims: [input.nx, input.nz, input.contacts.len() as u32, 0],
            origin: [input.origin[0], input.origin[1], input.spacing, 0.0],
            soil0: [
                input.stiffness,
                input.max_sinkage,
                input.shear_strength,
                input.shear_sinkage_gain,
            ],
            soil1: [input.bekker_n, 0.0, 0.0, 0.0],
        };
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_bekker io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &rest_buf),
                entry(1, &height_in_buf),
                entry(2, &sinkage_in_buf),
                entry(3, &contact_buf),
                entry(4, &height_out_buf),
                entry(5, &sinkage_out_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_bekker uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_bekker encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_bekker pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let bytes = f32_bytes(n);
        encoder.copy_buffer_to_buffer(&height_out_buf, 0, &stage_height, 0, bytes);
        encoder.copy_buffer_to_buffer(&sinkage_out_buf, 0, &stage_sinkage, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireBekkerGpuOutput {
            heights: read_f32(device, &stage_height, "tire_bekker heights")?,
            sinkage: read_f32(device, &stage_sinkage, "tire_bekker sinkage")?,
        })
    }

    /// Run Bekker writeback and leave the updated terrain maps on the GPU.
    ///
    /// This is the path FEM contact feedback should consume when a live
    /// deformable terrain owner is already on the GPU. Use
    /// [`Self::readback_resident_output`] only at explicit viewer/test
    /// telemetry seams.
    pub fn dispatch_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireBekkerGpuInput,
    ) -> Result<TireBekkerResidentOutput> {
        let n = input.cell_count()?;
        let contact_len = input.contacts.len().max(1);
        let f32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<f32>()) as u64;
        let contact_bytes =
            |len: usize| (len.max(1) * core::mem::size_of::<TireBekkerContactCell>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: bytes.max(4),
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let rest_buf = mk_buf("tire_bekker resident rest", f32_bytes(n), usage_ro);
        let height_in_buf = mk_buf("tire_bekker resident height in", f32_bytes(n), usage_ro);
        let sinkage_in_buf = mk_buf("tire_bekker resident sinkage in", f32_bytes(n), usage_ro);
        let contact_buf = mk_buf(
            "tire_bekker resident contacts",
            contact_bytes(contact_len),
            usage_ro,
        );
        let height_out_buf = mk_buf("tire_bekker resident height out", f32_bytes(n), usage_rw);
        let sinkage_out_buf = mk_buf("tire_bekker resident sinkage out", f32_bytes(n), usage_rw);
        let uniform_buf = mk_buf(
            "tire_bekker resident uniforms",
            core::mem::size_of::<BekkerUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        molla_core::gpu_write_buffer!(
            queue,
            &rest_buf,
            0,
            bytemuck::cast_slice(&input.rest_heights)
        );
        molla_core::gpu_write_buffer!(
            queue,
            &height_in_buf,
            0,
            bytemuck::cast_slice(&input.heights)
        );
        molla_core::gpu_write_buffer!(
            queue,
            &sinkage_in_buf,
            0,
            bytemuck::cast_slice(&input.sinkage)
        );
        if input.contacts.is_empty() {
            molla_core::gpu_write_buffer!(
                queue,
                &contact_buf,
                0,
                bytemuck::cast_slice(&[TireBekkerContactCell::default()]),
            );
        } else {
            molla_core::gpu_write_buffer!(
                queue,
                &contact_buf,
                0,
                bytemuck::cast_slice(&input.contacts)
            );
        }
        let uniforms = BekkerUniforms {
            dims: [input.nx, input.nz, input.contacts.len() as u32, 0],
            origin: [input.origin[0], input.origin[1], input.spacing, 0.0],
            soil0: [
                input.stiffness,
                input.max_sinkage,
                input.shear_strength,
                input.shear_sinkage_gain,
            ],
            soil1: [input.bekker_n, 0.0, 0.0, 0.0],
        };
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_bekker resident io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &rest_buf),
                entry(1, &height_in_buf),
                entry(2, &sinkage_in_buf),
                entry(3, &contact_buf),
                entry(4, &height_out_buf),
                entry(5, &sinkage_out_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_bekker resident uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_bekker resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_bekker resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireBekkerResidentOutput {
            heights: height_out_buf,
            sinkage: sinkage_out_buf,
            cell_count: n,
        })
    }

    /// Read GPU-resident Bekker maps back to typed CPU arrays.
    ///
    /// This readback is intentionally separate from [`Self::dispatch_resident`]
    /// so runtime code can keep the hot FEM feedback path on GPU and only mirror
    /// maps for viewer/telemetry/reset state when needed.
    pub fn readback_resident_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: &TireBekkerResidentOutput,
    ) -> Result<TireBekkerGpuOutput> {
        let bytes = (output.cell_count.max(1) * core::mem::size_of::<f32>()) as u64;
        let stage_height = stage_f32(
            device,
            "tire_bekker resident readback height",
            output.cell_count,
        );
        let stage_sinkage = stage_f32(
            device,
            "tire_bekker resident readback sinkage",
            output.cell_count,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_bekker resident readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&output.heights, 0, &stage_height, 0, bytes);
        encoder.copy_buffer_to_buffer(&output.sinkage, 0, &stage_sinkage, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireBekkerGpuOutput {
            heights: read_f32(device, &stage_height, "tire_bekker resident heights")?,
            sinkage: read_f32(device, &stage_sinkage, "tire_bekker resident sinkage")?,
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

fn stage_f32(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: (len.max(1) * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    )
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
    let data = molla_core::gpu_readback!(slice);
    let out = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}
