//! GPU kernel wrapper for volumetric tire tetra-FEM.
//!
//! This is the GPU counterpart to `molla_vehicle::TetraTireFem`'s node update:
//! one WGSL invocation owns one node, loops over incident tetrahedra, applies
//! compressible Neo-Hookean tetra forces, bead/rim coupling, optional pressure
//! traction, and floor contact, then writes next positions/velocities. The
//! first implementation is intentionally deterministic and race-free (no float
//! atomics); later high-throughput versions can add CSR adjacency or per-tet
//! reductions.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    counts: [u32; 4],
    phys0: [f32; 4],
    phys1: [f32; 4],
    phys2: [f32; 4],
    phys3: [f32; 4],
}

/// Host-side scalar parameters for one GPU tire FEM node step.
#[derive(Clone, Copy, Debug)]
pub struct TireFemGpuParams {
    pub dt: f32,
    /// Legacy/fallback scalar material fields. The current kernel uses the
    /// per-tet `tet_mu` / `tet_lambda` buffers for actual constitutive stress.
    pub young_modulus: f32,
    pub poisson_ratio: f32,
    pub pressure_pa: f32,
    pub node_damping: f32,
    pub bead_stiffness: f32,
    pub floor_stiffness: f32,
    pub floor_damping: f32,
    pub pressure_area_per_tread_node: f32,
    pub hub: [f32; 3],
    pub gravity_y: f32,
}

impl Default for TireFemGpuParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 240.0,
            young_modulus: 3.5e5,
            poisson_ratio: 0.46,
            pressure_pa: 180_000.0,
            node_damping: 18.0,
            bead_stiffness: 5.5e4,
            floor_stiffness: 2.8e5,
            floor_damping: 1_200.0,
            pressure_area_per_tread_node: 0.0,
            hub: [0.0, 0.34, 0.0],
            gravity_y: -9.81,
        }
    }
}

/// Complete buffer payload for one blocking GPU tire FEM dispatch.
#[derive(Clone, Debug)]
pub struct TireFemGpuInput {
    pub rest_positions: Vec<[f32; 4]>,
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub masses: Vec<f32>,
    /// Four vertex indices per tet.
    pub tet_indices: Vec<u32>,
    /// Four shape-function gradients per tet, matching `tet_indices`.
    pub tet_grads: Vec<[f32; 4]>,
    pub tet_volumes: Vec<f32>,
    pub tet_mu: Vec<f32>,
    pub tet_lambda: Vec<f32>,
    pub bead_ids: Vec<u32>,
    pub bead_offsets: Vec<[f32; 4]>,
    pub tread_ids: Vec<u32>,
    pub params: TireFemGpuParams,
}

/// Readback from the GPU tire FEM step.
#[derive(Clone, Debug)]
pub struct TireFemGpuOutput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub forces: Vec<[f32; 4]>,
}

pub struct TireFemGpuKernel {
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl TireFemGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_fem"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_fem.wgsl").into()),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem io"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, true),
                storage_entry(9, true),
                storage_entry(10, true),
                storage_entry(11, true),
                storage_entry(12, false),
                storage_entry(13, false),
                storage_entry(14, false),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem uniforms"),
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
            label: Some("molla::tire_fem layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_fem::step_nodes"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("step_nodes"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_io,
            layout_uniforms,
        }
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemGpuInput,
    ) -> Result<TireFemGpuOutput> {
        validate_input(input)?;
        let n = input.positions.len();
        if n == 0 {
            return Ok(TireFemGpuOutput {
                positions: vec![],
                velocities: vec![],
                forces: vec![],
            });
        }

        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let f32_bytes = |len: usize| (len * core::mem::size_of::<f32>()) as u64;
        let u32_bytes = |len: usize| (len * core::mem::size_of::<u32>()) as u64;

        let rest_buf = mk_buf("tire_fem rest", vec4_bytes(n), usage_ro);
        let pos_buf = mk_buf("tire_fem pos", vec4_bytes(n), usage_ro);
        let vel_buf = mk_buf("tire_fem vel", vec4_bytes(n), usage_ro);
        let mass_buf = mk_buf("tire_fem mass", f32_bytes(n), usage_ro);
        let tet_i_buf = mk_buf(
            "tire_fem tet_indices",
            u32_bytes(input.tet_indices.len()),
            usage_ro,
        );
        let tet_g_buf = mk_buf(
            "tire_fem tet_grads",
            vec4_bytes(input.tet_grads.len()),
            usage_ro,
        );
        let tet_v_buf = mk_buf(
            "tire_fem tet_volume",
            f32_bytes(input.tet_volumes.len()),
            usage_ro,
        );
        let tet_mu_buf = mk_buf("tire_fem tet_mu", f32_bytes(input.tet_mu.len()), usage_ro);
        let tet_lambda_buf = mk_buf(
            "tire_fem tet_lambda",
            f32_bytes(input.tet_lambda.len()),
            usage_ro,
        );
        let bead_id_buf = mk_buf(
            "tire_fem bead_ids",
            u32_bytes(input.bead_ids.len()),
            usage_ro,
        );
        let bead_off_buf = mk_buf(
            "tire_fem bead_offsets",
            vec4_bytes(input.bead_offsets.len()),
            usage_ro,
        );
        let tread_id_buf = mk_buf(
            "tire_fem tread_ids",
            u32_bytes(input.tread_ids.len()),
            usage_ro,
        );
        let out_pos = mk_buf("tire_fem out_pos", vec4_bytes(n), usage_rw);
        let out_vel = mk_buf("tire_fem out_vel", vec4_bytes(n), usage_rw);
        let out_force = mk_buf("tire_fem out_force", vec4_bytes(n), usage_rw);

        queue.write_buffer(&rest_buf, 0, bytemuck::cast_slice(&input.rest_positions));
        queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_buf, 0, bytemuck::cast_slice(&input.velocities));
        queue.write_buffer(&mass_buf, 0, bytemuck::cast_slice(&input.masses));
        queue.write_buffer(&tet_i_buf, 0, bytemuck::cast_slice(&input.tet_indices));
        queue.write_buffer(&tet_g_buf, 0, bytemuck::cast_slice(&input.tet_grads));
        queue.write_buffer(&tet_v_buf, 0, bytemuck::cast_slice(&input.tet_volumes));
        queue.write_buffer(&tet_mu_buf, 0, bytemuck::cast_slice(&input.tet_mu));
        queue.write_buffer(&tet_lambda_buf, 0, bytemuck::cast_slice(&input.tet_lambda));
        if !input.bead_ids.is_empty() {
            queue.write_buffer(&bead_id_buf, 0, bytemuck::cast_slice(&input.bead_ids));
            queue.write_buffer(&bead_off_buf, 0, bytemuck::cast_slice(&input.bead_offsets));
        }
        if !input.tread_ids.is_empty() {
            queue.write_buffer(&tread_id_buf, 0, bytemuck::cast_slice(&input.tread_ids));
        }

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &rest_buf),
                entry(1, &pos_buf),
                entry(2, &vel_buf),
                entry(3, &mass_buf),
                entry(4, &tet_i_buf),
                entry(5, &tet_g_buf),
                entry(6, &tet_v_buf),
                entry(7, &tet_mu_buf),
                entry(8, &tet_lambda_buf),
                entry(9, &bead_id_buf),
                entry(10, &bead_off_buf),
                entry(11, &tread_id_buf),
                entry(12, &out_pos),
                entry(13, &out_vel),
                entry(14, &out_force),
            ],
        });

        let tet_count = input.tet_indices.len() / 4;
        let uniforms = Uniforms {
            counts: [
                n as u32,
                tet_count as u32,
                input.bead_ids.len() as u32,
                input.tread_ids.len() as u32,
            ],
            phys0: [
                input.params.dt,
                input.params.young_modulus,
                input.params.poisson_ratio,
                input.params.pressure_pa,
            ],
            phys1: [
                input.params.node_damping,
                input.params.bead_stiffness,
                input.params.floor_stiffness,
                input.params.floor_damping,
            ],
            phys2: [
                input.params.pressure_area_per_tread_node,
                input.params.hub[0],
                input.params.hub[1],
                input.params.hub[2],
            ],
            phys3: [input.params.gravity_y, 0.0, 0.0, 0.0],
        };
        let uniform_buf = mk_buf(
            "tire_fem uniforms",
            core::mem::size_of::<Uniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        let stage_pos = stage_vec4(device, "tire_fem stage pos", n);
        let stage_vel = stage_vec4(device, "tire_fem stage vel", n);
        let stage_force = stage_vec4(device, "tire_fem stage force", n);
        let bytes = vec4_bytes(n);
        encoder.copy_buffer_to_buffer(&out_pos, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&out_vel, 0, &stage_vel, 0, bytes);
        encoder.copy_buffer_to_buffer(&out_force, 0, &stage_force, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireFemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_fem positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_fem velocities")?,
            forces: read_vec4(device, &stage_force, "tire_fem forces")?,
        })
    }
}

fn validate_input(input: &TireFemGpuInput) -> Result<()> {
    let n = input.positions.len();
    if input.rest_positions.len() != n || input.velocities.len() != n || input.masses.len() != n {
        return Err(Error::Solver(
            "tire_fem: rest/position/velocity/mass lengths must match".into(),
        ));
    }
    if input.tet_indices.len() % 4 != 0 {
        return Err(Error::Solver(
            "tire_fem: tet_indices length must be divisible by four".into(),
        ));
    }
    if input.tet_grads.len() != input.tet_indices.len() {
        return Err(Error::Solver(
            "tire_fem: tet_grads must contain four gradients per tet".into(),
        ));
    }
    if input.tet_volumes.len() != input.tet_indices.len() / 4 {
        return Err(Error::Solver(
            "tire_fem: tet_volumes length must match tet count".into(),
        ));
    }
    if input.tet_mu.len() != input.tet_indices.len() / 4
        || input.tet_lambda.len() != input.tet_indices.len() / 4
    {
        return Err(Error::Solver(
            "tire_fem: tet material arrays must match tet count".into(),
        ));
    }
    if input.bead_offsets.len() != input.bead_ids.len() {
        return Err(Error::Solver(
            "tire_fem: bead_offsets length must match bead_ids".into(),
        ));
    }
    Ok(())
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
        size: (len * core::mem::size_of::<[f32; 4]>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn read_vec4(device: &wgpu::Device, buf: &wgpu::Buffer, label: &str) -> Result<Vec<[f32; 4]>> {
    let slice = buf.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
    let label = label.to_string();
    let callback_label = label.clone();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("{callback_label} readback: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("{label} readback poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("{label} readback recv: {e:?}")))??;
    let data = slice.get_mapped_range();
    let out: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}
