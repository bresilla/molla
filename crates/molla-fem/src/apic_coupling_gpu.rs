//! Stage 41 v2 — GPU port of the APIC FEM-MPM hybrid coupling
//! ([`super::apic_coupling::ApicHybrid::step_gpu`]).
//!
//! Mirrors the CPU pipeline in `apic_coupling.rs` step-for-step:
//!
//! 1. `locate_tet`        — brute-force barycentric scan, per particle.
//! 2. `reset_grid`        — zero atomic accumulators.
//! 3. `p2g`               — scatter `(w·m, w·m·(v + C·dx))` into grid.
//! 4. `grid_update`       — `v = mom / mass + dt · gravity`.
//! 5. `viscous_smoothing` — one Jacobi sweep on the mass-weighted
//!    Laplacian over a precomputed node-neighbour graph.
//! 6. `apply_bcs`         — Dirichlet velocity pin.
//! 7. `g2p`               — gather; recompute APIC affine matrix; advect.
//!
//! The neighbour CSR is built once on CPU at construction time
//! ([`ApicGpuState::new`]) and uploaded as `(offsets, indices)` u32
//! buffers. Particle owning-tet + barycentric weights are recomputed
//! each step (the brute-force scan is O(P·T) on the GPU; a BVH-walk
//! version is a v2 follow-up).
//!
//! All buffers are persistent — one allocation per construction, no
//! per-step allocations. Material parameters travel via a single
//! uniform buffer rewritten via `queue.write_buffer`.
//!
//! See `crates/molla-fem/shaders/apic_locate_tet.wgsl` for the kernel
//! sources.

#![cfg(feature = "gpu")]

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::compute::Engine;
use crate::core::{Error, Result};
use crate::fem::apic_coupling::{ApicHybrid, ApicParticle};
use crate::fem::bc::BoundaryCondition;
use crate::fem::mesh::FemCell;
use crate::math::{Mat3, Vec3};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    particle_count: u32,
    vert_count: u32,
    cell_count: u32,
    _pad0: u32,

    dt: f32,
    viscosity: f32,
    gravity_x: f32,
    gravity_y: f32,

    gravity_z: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

const WG_SIZE: u32 = 64;

/// All GPU resources for one [`ApicHybrid`] simulation.
///
/// Built once via [`ApicGpuState::new`] (compiles + uploads + caches
/// neighbour graph). Cloning is **not** supported — wgpu resources
/// are not `Clone`. Drop the state when finished; underlying buffers
/// release via the `Arc` reference count.
#[allow(dead_code)]
pub struct ApicGpuState {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Pipelines (one per WGSL entry point).
    pipe_locate: wgpu::ComputePipeline,
    // Stage 37.6.1 — pressure-projection pipelines.
    pipe_proj_reset_tets: wgpu::ComputePipeline,
    pipe_proj_reset_nodes: wgpu::ComputePipeline,
    pipe_proj_mark_fluid: wgpu::ComputePipeline,
    pipe_proj_setup: wgpu::ComputePipeline,
    pipe_proj_zero_f: wgpu::ComputePipeline,
    pipe_proj_scatter_d: wgpu::ComputePipeline,
    pipe_proj_scatter_q: wgpu::ComputePipeline,
    pipe_proj_gather_sd: wgpu::ComputePipeline,
    pipe_proj_dot_init: wgpu::ComputePipeline,
    pipe_proj_dot_dsd: wgpu::ComputePipeline,
    pipe_proj_dot_rr: wgpu::ComputePipeline,
    pipe_proj_update_qr: wgpu::ComputePipeline,
    pipe_proj_update_d: wgpu::ComputePipeline,
    pipe_proj_correct: wgpu::ComputePipeline,
    pipe_reset: wgpu::ComputePipeline,
    pipe_p2g: wgpu::ComputePipeline,
    pipe_grid_update: wgpu::ComputePipeline,
    pipe_smooth: wgpu::ComputePipeline,
    pipe_swap: wgpu::ComputePipeline,
    pipe_apply_bcs: wgpu::ComputePipeline,
    pipe_g2p: wgpu::ComputePipeline,

    // Bind group layouts.
    layout_uniforms: wgpu::BindGroupLayout,
    layout_mesh: wgpu::BindGroupLayout,
    layout_particles: wgpu::BindGroupLayout,
    layout_grid: wgpu::BindGroupLayout,

    // Mesh buffers (uploaded once at construction).
    mesh_vertices: wgpu::Buffer,
    mesh_cells: wgpu::Buffer,
    mesh_cell_offsets: wgpu::Buffer,

    // Particle buffers.
    p_x: wgpu::Buffer,
    p_v: wgpu::Buffer,
    p_c: wgpu::Buffer,
    p_mass: wgpu::Buffer,
    p_loc: wgpu::Buffer,

    // Grid buffers.
    g_mass: wgpu::Buffer,
    g_mom_x: wgpu::Buffer,
    g_mom_y: wgpu::Buffer,
    g_mom_z: wgpu::Buffer,
    g_vel: wgpu::Buffer,
    g_vel_next: wgpu::Buffer,
    g_neighbour_offsets: wgpu::Buffer,
    g_neighbour_indices: wgpu::Buffer,
    g_bc_mask: wgpu::Buffer,
    g_bc_values: wgpu::Buffer,

    // Persistent uniforms + bind groups.
    uniform_buf: wgpu::Buffer,
    bg_uniforms: wgpu::BindGroup,
    bg_mesh: wgpu::BindGroup,
    bg_particles: wgpu::BindGroup,
    bg_grid: wgpu::BindGroup,

    // Readback staging.
    stage_pos: wgpu::Buffer,
    stage_vel: wgpu::Buffer,
    stage_c: wgpu::Buffer,

    // Sim metadata.
    particle_count: u32,
    vert_count: u32,
    cell_count: u32,
}

impl ApicGpuState {
    /// Build a fresh GPU mirror for `hybrid`. Uploads the mesh,
    /// particle state, and BC list. The BC list is cached — change
    /// the BCs by rebuilding the state. (BCs typically don't change
    /// during a sim; if they do, a follow-up `set_bcs` helper is a
    /// trivial extension.)
    pub fn new(engine: &Engine, hybrid: &ApicHybrid, bcs: &[BoundaryCondition]) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("ApicGpuState: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("ApicGpuState: engine has no GPU queue".into()))?
            .clone();

        let mesh = &hybrid.mesh;
        let vert_count = mesh.vert_count() as u32;
        if vert_count == 0 {
            return Err(Error::Build("ApicGpuState: empty mesh".into()));
        }
        let mut cell_count: u32 = 0;
        for c in &mesh.cells {
            if let FemCell::Tet(_) = c {
                cell_count += 1;
            }
        }
        if cell_count == 0 {
            return Err(Error::Build("ApicGpuState: mesh has no tets".into()));
        }
        let particle_count = hybrid.particles.len() as u32;

        // ── Shader / pipelines ──────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::apic_coupling_gpu"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/apic_locate_tet.wgsl").into(),
            ),
        });

        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::apic uniforms"),
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
        let layout_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::apic mesh"),
            entries: &storage_entries(&[true, true, true]),
        });
        let layout_particles = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::apic particles"),
            // p_x, p_v, p_C, p_mass, p_loc
            entries: &storage_entries(&[false, false, false, true, false]),
        });
        let layout_grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::apic grid"),
            // g_mass, g_mom_x, g_mom_y, g_mom_z, g_vel, g_vel_next,
            // g_neighbour_offsets, g_neighbour_indices, g_bc_mask,
            // g_bc_values
            // + Stage 37.6.1 projection state: t_fluid, t_q, t_r,
            // t_d, t_sd, n_lumped, cg_scalars.
            entries: &storage_entries(&[
                false, false, false, false, false, false, true, true, true, true, false, false,
                false, false, false, false, false,
            ]),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::apic pipeline layout"),
            bind_group_layouts: &[
                &layout_uniforms,
                &layout_mesh,
                &layout_particles,
                &layout_grid,
            ],
            push_constant_ranges: &[],
        });

        let mk_pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipe_locate = mk_pipe("locate_tet");
        let pipe_proj_reset_tets = mk_pipe("proj_reset_tets");
        let pipe_proj_reset_nodes = mk_pipe("proj_reset_nodes");
        let pipe_proj_mark_fluid = mk_pipe("proj_mark_fluid");
        let pipe_proj_setup = mk_pipe("proj_setup");
        let pipe_proj_zero_f = mk_pipe("proj_zero_f");
        let pipe_proj_scatter_d = mk_pipe("proj_scatter_d");
        let pipe_proj_scatter_q = mk_pipe("proj_scatter_q");
        let pipe_proj_gather_sd = mk_pipe("proj_gather_sd");
        let pipe_proj_dot_init = mk_pipe("proj_dot_init");
        let pipe_proj_dot_dsd = mk_pipe("proj_dot_dsd");
        let pipe_proj_dot_rr = mk_pipe("proj_dot_rr");
        let pipe_proj_update_qr = mk_pipe("proj_update_qr");
        let pipe_proj_update_d = mk_pipe("proj_update_d");
        let pipe_proj_correct = mk_pipe("proj_correct");
        let pipe_reset = mk_pipe("reset_grid");
        let pipe_p2g = mk_pipe("p2g");
        let pipe_grid_update = mk_pipe("grid_update");
        let pipe_smooth = mk_pipe("viscous_smoothing");
        let pipe_swap = mk_pipe("velocity_swap");
        let pipe_apply_bcs = mk_pipe("apply_bcs");
        let pipe_g2p = mk_pipe("g2p");

        // ── Mesh upload ────────────────────────────────────────────
        let mut vert_bytes: Vec<f32> = Vec::with_capacity(mesh.vertices.len() * 4);
        for v in &mesh.vertices {
            vert_bytes.push(v.x as f32);
            vert_bytes.push(v.y as f32);
            vert_bytes.push(v.z as f32);
            vert_bytes.push(0.0);
        }
        let mesh_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.mesh_vertices"),
            contents: bytemuck::cast_slice(&vert_bytes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let mut cell_indices: Vec<u32> = Vec::with_capacity(cell_count as usize * 4);
        let mut cell_offsets: Vec<u32> = Vec::with_capacity(cell_count as usize + 1);
        cell_offsets.push(0);
        let mut acc: u32 = 0;
        for c in &mesh.cells {
            if let FemCell::Tet(v) = c {
                cell_indices.extend_from_slice(v);
                acc += 4;
                cell_offsets.push(acc);
            }
        }
        let mesh_cells = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.mesh_cells"),
            contents: bytemuck::cast_slice(&cell_indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let mesh_cell_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.mesh_cell_offsets"),
            contents: bytemuck::cast_slice(&cell_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Particle upload ────────────────────────────────────────
        let mut p_pos_data = Vec::<[f32; 4]>::with_capacity(hybrid.particles.len());
        let mut p_vel_data = Vec::<[f32; 4]>::with_capacity(hybrid.particles.len());
        let mut p_c_data = Vec::<[f32; 4]>::with_capacity(hybrid.particles.len() * 3);
        let mut p_mass_data = Vec::<f32>::with_capacity(hybrid.particles.len());
        for p in &hybrid.particles {
            p_pos_data.push([p.x.x as f32, p.x.y as f32, p.x.z as f32, 0.0]);
            p_vel_data.push([p.v.x as f32, p.v.y as f32, p.v.z as f32, 0.0]);
            let r0 = p.c.x_axis;
            let r1 = p.c.y_axis;
            let r2 = p.c.z_axis;
            // glam DMat3 is column-major; row i = (x_axis[i], y_axis[i], z_axis[i]).
            // We store rows so dot(row, dx) gives (C · dx)[i].
            p_c_data.push([r0.x as f32, r1.x as f32, r2.x as f32, 0.0]);
            p_c_data.push([r0.y as f32, r1.y as f32, r2.y as f32, 0.0]);
            p_c_data.push([r0.z as f32, r1.z as f32, r2.z as f32, 0.0]);
            p_mass_data.push(p.mass as f32);
        }
        // Ensure non-empty (wgpu storage buffers must be ≥ 16 bytes).
        if p_pos_data.is_empty() {
            p_pos_data.push([0.0; 4]);
            p_vel_data.push([0.0; 4]);
            p_c_data.extend_from_slice(&[[0.0; 4]; 3]);
            p_mass_data.push(0.0);
        }
        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let p_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.p_x"),
            contents: bytemuck::cast_slice(&p_pos_data),
            usage: storage_rw,
        });
        let p_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.p_v"),
            contents: bytemuck::cast_slice(&p_vel_data),
            usage: storage_rw,
        });
        let p_c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.p_c"),
            contents: bytemuck::cast_slice(&p_c_data),
            usage: storage_rw,
        });
        let p_mass = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.p_mass"),
            contents: bytemuck::cast_slice(&p_mass_data),
            usage: storage_rw,
        });
        // p_loc: 3 vec4<u32> per particle = 48 bytes per particle.
        let p_loc_bytes = (particle_count.max(1) as u64) * 48;
        let p_loc = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apic.p_loc"),
            size: p_loc_bytes,
            usage: storage_rw,
            mapped_at_creation: false,
        });

        // ── Grid buffers ───────────────────────────────────────────
        let n = vert_count.max(1) as u64;
        let mk_buf = |label: &str, size_bytes: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size_bytes.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        let g_mass = mk_buf("apic.g_mass", n * 4, storage_rw);
        let g_mom_x = mk_buf("apic.g_mom_x", n * 4, storage_rw);
        let g_mom_y = mk_buf("apic.g_mom_y", n * 4, storage_rw);
        let g_mom_z = mk_buf("apic.g_mom_z", n * 4, storage_rw);
        let g_vel = mk_buf("apic.g_vel", n * 16, storage_rw);
        let g_vel_next = mk_buf("apic.g_vel_next", n * 16, storage_rw);

        // Neighbour CSR: walk every tet, collect per-node neighbour
        // sets (sets to dedupe). Mirrors `apply_viscous_smoothing`'s
        // graph build but cached on the device.
        let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); vert_count as usize];
        for cell in &mesh.cells {
            if let FemCell::Tet(v) = cell {
                for &a in v.iter() {
                    for &b in v.iter() {
                        if a != b && !neighbours[a as usize].contains(&b) {
                            neighbours[a as usize].push(b);
                        }
                    }
                }
            }
        }
        let mut off_list: Vec<u32> = Vec::with_capacity(vert_count as usize + 1);
        off_list.push(0);
        let mut idx_list: Vec<u32> = Vec::new();
        for ns in &neighbours {
            for &j in ns {
                idx_list.push(j);
            }
            off_list.push(idx_list.len() as u32);
        }
        if idx_list.is_empty() {
            idx_list.push(0);
        }
        let g_neighbour_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.g_neighbour_offsets"),
            contents: bytemuck::cast_slice(&off_list),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let g_neighbour_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.g_neighbour_indices"),
            contents: bytemuck::cast_slice(&idx_list),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // BCs: mask + values arrays sized to vertex_count.
        let mut bc_mask_data: Vec<u32> = vec![0; vert_count as usize];
        let mut bc_values_data: Vec<[f32; 4]> = vec![[0.0; 4]; vert_count as usize];
        for bc in bcs {
            match bc {
                BoundaryCondition::Dirichlet { vertex, value } => {
                    let idx = *vertex as usize;
                    if idx < vert_count as usize {
                        bc_mask_data[idx] = 1;
                        bc_values_data[idx] = [*value as f32; 4];
                        bc_values_data[idx][3] = 0.0;
                        // Scalar Dirichlet is treated as no-slip
                        // when value == 0 by the CPU mirror.
                        if *value == 0.0 {
                            bc_values_data[idx] = [0.0; 4];
                        }
                    }
                }
                BoundaryCondition::DirichletVector { vertex, values } => {
                    let idx = *vertex as usize;
                    if idx < vert_count as usize && values.len() >= 3 {
                        bc_mask_data[idx] = 1;
                        bc_values_data[idx] =
                            [values[0] as f32, values[1] as f32, values[2] as f32, 0.0];
                    }
                }
                _ => {}
            }
        }
        let g_bc_mask = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.g_bc_mask"),
            contents: bytemuck::cast_slice(&bc_mask_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Stage 37.6.1 — projection state (per-tet CG vectors, lumped
        // node weights, on-device CG scalars).
        let mk_zero = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let tet_bytes = (cell_count.max(1) as u64) * 4;
        let t_fluid = mk_zero("apic.t_fluid", tet_bytes);
        let t_q = mk_zero("apic.t_q", tet_bytes);
        let t_r = mk_zero("apic.t_r", tet_bytes);
        let t_d = mk_zero("apic.t_d", tet_bytes);
        let t_sd = mk_zero("apic.t_sd", tet_bytes);
        let n_lumped = mk_zero("apic.n_lumped", (vert_count.max(1) as u64) * 4);
        let cg_scalars = mk_zero("apic.cg_scalars", 32);

        let g_bc_values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("apic.g_bc_values"),
            contents: bytemuck::cast_slice(&bc_values_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Uniforms ──────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apic.uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apic.bg_uniforms"),
            layout: &layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apic.bg_mesh"),
            layout: &layout_mesh,
            entries: &[
                buffer_entry(0, &mesh_vertices),
                buffer_entry(1, &mesh_cells),
                buffer_entry(2, &mesh_cell_offsets),
            ],
        });
        let bg_particles = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apic.bg_particles"),
            layout: &layout_particles,
            entries: &[
                buffer_entry(0, &p_x),
                buffer_entry(1, &p_v),
                buffer_entry(2, &p_c),
                buffer_entry(3, &p_mass),
                buffer_entry(4, &p_loc),
            ],
        });
        let bg_grid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("apic.bg_grid"),
            layout: &layout_grid,
            entries: &[
                buffer_entry(0, &g_mass),
                buffer_entry(1, &g_mom_x),
                buffer_entry(2, &g_mom_y),
                buffer_entry(3, &g_mom_z),
                buffer_entry(4, &g_vel),
                buffer_entry(5, &g_vel_next),
                buffer_entry(6, &g_neighbour_offsets),
                buffer_entry(7, &g_neighbour_indices),
                buffer_entry(8, &g_bc_mask),
                buffer_entry(9, &g_bc_values),
                buffer_entry(10, &t_fluid),
                buffer_entry(11, &t_q),
                buffer_entry(12, &t_r),
                buffer_entry(13, &t_d),
                buffer_entry(14, &t_sd),
                buffer_entry(15, &n_lumped),
                buffer_entry(16, &cg_scalars),
            ],
        });

        let stage_bytes = (particle_count.max(1) as u64) * 16;
        let stage_pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apic.stage_pos"),
            size: stage_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let stage_vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apic.stage_vel"),
            size: stage_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let stage_c_bytes = (particle_count.max(1) as u64) * 48;
        let stage_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("apic.stage_c"),
            size: stage_c_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipe_locate,
            pipe_proj_reset_tets,
            pipe_proj_reset_nodes,
            pipe_proj_mark_fluid,
            pipe_proj_setup,
            pipe_proj_zero_f,
            pipe_proj_scatter_d,
            pipe_proj_scatter_q,
            pipe_proj_gather_sd,
            pipe_proj_dot_init,
            pipe_proj_dot_dsd,
            pipe_proj_dot_rr,
            pipe_proj_update_qr,
            pipe_proj_update_d,
            pipe_proj_correct,
            pipe_reset,
            pipe_p2g,
            pipe_grid_update,
            pipe_smooth,
            pipe_swap,
            pipe_apply_bcs,
            pipe_g2p,
            layout_uniforms,
            layout_mesh,
            layout_particles,
            layout_grid,
            mesh_vertices,
            mesh_cells,
            mesh_cell_offsets,
            p_x,
            p_v,
            p_c,
            p_mass,
            p_loc,
            g_mass,
            g_mom_x,
            g_mom_y,
            g_mom_z,
            g_vel,
            g_vel_next,
            g_neighbour_offsets,
            g_neighbour_indices,
            g_bc_mask,
            g_bc_values,
            uniform_buf,
            bg_uniforms,
            bg_mesh,
            bg_particles,
            bg_grid,
            stage_pos,
            stage_vel,
            stage_c,
            particle_count,
            vert_count,
            cell_count,
        })
    }

    /// Run one APIC hybrid step on the GPU. Mirrors
    /// [`ApicHybrid::step`] exactly: P2G → grid-update → optional
    /// viscous smoothing → Dirichlet BCs → G2P (which advects).
    ///
    /// Submits a single command encoder; per-step host overhead is
    /// one uniform `write_buffer` + one `queue.submit`.
    pub fn step(&self, hybrid: &ApicHybrid, dt: f32, viscosity: f32) -> Result<()> {
        self.step_projected(hybrid, dt, viscosity, 0)
    }

    /// Stage 37.6.1 — full step with the per-tet (P0) pressure
    /// projection between the grid update and G2P (mirrors
    /// `ApicHybrid::step_projected`; `pressure_iters` CG iterations
    /// recorded entirely on-device — zero readbacks).
    pub fn step_projected(
        &self,
        hybrid: &ApicHybrid,
        dt: f32,
        viscosity: f32,
        pressure_iters: u32,
    ) -> Result<()> {
        if self.particle_count == 0 {
            return Ok(());
        }
        let g = hybrid.gravity;
        let uniforms = Uniforms {
            particle_count: self.particle_count,
            vert_count: self.vert_count,
            cell_count: self.cell_count,
            _pad0: 0,
            dt,
            viscosity,
            gravity_x: g.x as f32,
            gravity_y: g.y as f32,
            gravity_z: g.z as f32,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let groups_p = self.particle_count.div_ceil(WG_SIZE).max(1);
        let groups_v = self.vert_count.div_ceil(WG_SIZE).max(1);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::apic step encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::apic step"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bg_uniforms, &[]);
            pass.set_bind_group(1, &self.bg_mesh, &[]);
            pass.set_bind_group(2, &self.bg_particles, &[]);
            pass.set_bind_group(3, &self.bg_grid, &[]);

            // 1. Locate owning tet per particle.
            pass.set_pipeline(&self.pipe_locate);
            pass.dispatch_workgroups(groups_p, 1, 1);

            // 2. Reset grid mass + momentum + velocity buffers.
            pass.set_pipeline(&self.pipe_reset);
            pass.dispatch_workgroups(groups_v, 1, 1);

            // 3. Particle → grid scatter.
            pass.set_pipeline(&self.pipe_p2g);
            pass.dispatch_workgroups(groups_p, 1, 1);

            // 4. Grid update (mom / mass + gravity).
            pass.set_pipeline(&self.pipe_grid_update);
            pass.dispatch_workgroups(groups_v, 1, 1);

            // 5. One Jacobi sweep of viscous smoothing (g_vel →
            //    g_vel_next), then swap. Skipped when viscosity ≤ 0.
            if viscosity > 0.0 {
                pass.set_pipeline(&self.pipe_smooth);
                pass.dispatch_workgroups(groups_v, 1, 1);
                pass.set_pipeline(&self.pipe_swap);
                pass.dispatch_workgroups(groups_v, 1, 1);
            }

            // 6. Apply Dirichlet BCs.
            pass.set_pipeline(&self.pipe_apply_bcs);
            pass.dispatch_workgroups(groups_v, 1, 1);

            // 6b. Stage 37.6.1 — pressure projection (CG on the
            // lumped-mass Schur operator over deposit-supported fluid
            // tets), fully device-side. Sequential dispatches in one
            // pass have implicit barriers.
            if pressure_iters > 0 {
                let groups_t = self.cell_count.div_ceil(WG_SIZE).max(1);
                pass.set_pipeline(&self.pipe_proj_reset_tets);
                pass.dispatch_workgroups(groups_t, 1, 1);
                pass.set_pipeline(&self.pipe_proj_reset_nodes);
                pass.dispatch_workgroups(groups_v, 1, 1);
                pass.set_pipeline(&self.pipe_proj_mark_fluid);
                pass.dispatch_workgroups(groups_p, 1, 1);
                pass.set_pipeline(&self.pipe_proj_setup);
                pass.dispatch_workgroups(groups_t, 1, 1);
                pass.set_pipeline(&self.pipe_proj_dot_init);
                pass.dispatch_workgroups(1, 1, 1);
                for _ in 0..pressure_iters {
                    pass.set_pipeline(&self.pipe_proj_zero_f);
                    pass.dispatch_workgroups(groups_v, 1, 1);
                    pass.set_pipeline(&self.pipe_proj_scatter_d);
                    pass.dispatch_workgroups(groups_t, 1, 1);
                    pass.set_pipeline(&self.pipe_proj_gather_sd);
                    pass.dispatch_workgroups(groups_t, 1, 1);
                    pass.set_pipeline(&self.pipe_proj_dot_dsd);
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_pipeline(&self.pipe_proj_update_qr);
                    pass.dispatch_workgroups(groups_t, 1, 1);
                    pass.set_pipeline(&self.pipe_proj_dot_rr);
                    pass.dispatch_workgroups(1, 1, 1);
                    pass.set_pipeline(&self.pipe_proj_update_d);
                    pass.dispatch_workgroups(groups_t, 1, 1);
                }
                pass.set_pipeline(&self.pipe_proj_zero_f);
                pass.dispatch_workgroups(groups_v, 1, 1);
                pass.set_pipeline(&self.pipe_proj_scatter_q);
                pass.dispatch_workgroups(groups_t, 1, 1);
                pass.set_pipeline(&self.pipe_proj_correct);
                pass.dispatch_workgroups(groups_v, 1, 1);
                // Re-pin no-slip walls (the correction can add a
                // wall-normal component on boundary nodes) — mirrors
                // the CPU step_projected.
                pass.set_pipeline(&self.pipe_apply_bcs);
                pass.dispatch_workgroups(groups_v, 1, 1);
            }

            // 7. G2P: gather, recompute affine matrix, advect.
            pass.set_pipeline(&self.pipe_g2p);
            pass.dispatch_workgroups(groups_p, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Read particle positions back to host.
    pub fn read_positions(&self) -> Result<Vec<Vec3>> {
        if self.particle_count == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.particle_count as u64) * 16;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apic.read_positions"),
            });
        enc.copy_buffer_to_buffer(&self.p_x, 0, &self.stage_pos, 0, bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = self.stage_pos.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r.map_err(|e| Error::Gpu(format!("apic pos readback: {e:?}"))))
                .ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("apic pos poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("apic pos recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.stage_pos.unmap();
        Ok(raw
            .into_iter()
            .take(self.particle_count as usize)
            .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
            .collect())
    }

    /// Read particle velocities back to host.
    pub fn read_velocities(&self) -> Result<Vec<Vec3>> {
        if self.particle_count == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.particle_count as u64) * 16;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apic.read_velocities"),
            });
        enc.copy_buffer_to_buffer(&self.p_v, 0, &self.stage_vel, 0, bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = self.stage_vel.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r.map_err(|e| Error::Gpu(format!("apic vel readback: {e:?}"))))
                .ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("apic vel poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("apic vel recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.stage_vel.unmap();
        Ok(raw
            .into_iter()
            .take(self.particle_count as usize)
            .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
            .collect())
    }

    /// Read the affine-matrix C buffer back. Returns a `Vec<Mat3>` of
    /// length [`Self::particle_count`].
    pub fn read_affine(&self) -> Result<Vec<Mat3>> {
        if self.particle_count == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.particle_count as u64) * 48;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("apic.read_affine"),
            });
        enc.copy_buffer_to_buffer(&self.p_c, 0, &self.stage_c, 0, bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = self.stage_c.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r.map_err(|e| Error::Gpu(format!("apic c readback: {e:?}"))))
                .ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("apic c poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("apic c recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.stage_c.unmap();
        let mut out = Vec::with_capacity(self.particle_count as usize);
        for i in 0..self.particle_count as usize {
            let r0 = raw[i * 3];
            let r1 = raw[i * 3 + 1];
            let r2 = raw[i * 3 + 2];
            // We stored row-major in WGSL: row0 = (cr0.x, cr0.y, cr0.z).
            // glam Mat3 is column-major, so col i = (row0[i], row1[i], row2[i]).
            let m = Mat3::from_cols(
                Vec3::new(r0[0] as f64, r1[0] as f64, r2[0] as f64),
                Vec3::new(r0[1] as f64, r1[1] as f64, r2[1] as f64),
                Vec3::new(r0[2] as f64, r1[2] as f64, r2[2] as f64),
            );
            out.push(m);
        }
        Ok(out)
    }

    /// Sync the GPU-side particle state back into a host
    /// [`ApicHybrid`]. Combines position / velocity / affine readbacks
    /// into one consistent host snapshot.
    pub fn sync_to_host(&self, hybrid: &mut ApicHybrid) -> Result<()> {
        let pos = self.read_positions()?;
        let vel = self.read_velocities()?;
        let c = self.read_affine()?;
        for (i, p) in hybrid.particles.iter_mut().enumerate() {
            p.x = pos[i];
            p.v = vel[i];
            p.c = c[i];
        }
        Ok(())
    }

    /// Number of GPU-side particles.
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
}

impl ApicHybrid {
    /// Run one APIC hybrid step on the GPU and sync the result back
    /// into `self`. Wraps [`ApicGpuState::step`] + sync; suitable as
    /// a drop-in for [`ApicHybrid::step`] when a GPU adapter is
    /// available.
    ///
    /// For multi-step simulations, prefer building [`ApicGpuState`]
    /// once and calling its `step()` method directly to avoid the
    /// per-call sync; this convenience helper is for parity tests
    /// and one-shot solves.
    pub fn step_gpu(
        &mut self,
        engine: &Engine,
        dt: f32,
        viscosity: f32,
        bcs: &[BoundaryCondition],
    ) -> Result<()> {
        self.step_gpu_projected(engine, dt, viscosity, bcs, 0)
    }

    /// Stage 37.6.1 — GPU step with the per-tet pressure projection
    /// (see [`ApicHybrid::step_projected`] for the formulation; the CG
    /// loop runs entirely on-device).
    pub fn step_gpu_projected(
        &mut self,
        engine: &Engine,
        dt: f32,
        viscosity: f32,
        bcs: &[BoundaryCondition],
        pressure_iters: u32,
    ) -> Result<()> {
        let state = ApicGpuState::new(engine, self, bcs)?;
        state.step_projected(self, dt, viscosity, pressure_iters)?;
        state.sync_to_host(self)
    }
}

// ─── helpers ──────────────────────────────────────────────────────────

fn storage_entries(read_only: &[bool]) -> Vec<wgpu::BindGroupLayoutEntry> {
    read_only
        .iter()
        .enumerate()
        .map(|(i, &ro)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

fn buffer_entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

#[allow(dead_code)]
fn _keep_particle_alive(_: &ApicParticle) {}
