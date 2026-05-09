//! Stage 24d v1 — Material-Point-Method (MPM) for granular materials.
//!
//! **Implementation choice: explicit symplectic time-stepping** rather
//! than the implicit backward-Euler / CG scheme originally specified
//! by the PLAN. See `shaders/mpm.wgsl`'s file header for the trade-off
//! analysis. The explicit path is the standard fallback for stable v1
//! granular sims (Stomakhin 2013, Klár 2016 — both ship explicit) and
//! lets us keep total code under 1k lines while still exercising APIC
//! transfer, Drucker-Prager rheology, and rigid coupling.
//!
//! ## Pipeline (one substep)
//!
//! 1. `reset_grid` — clear mass + momentum atomics.
//! 2. `p2g` — particle → grid scatter (atomic adds for momentum + mass,
//!    elastic stress contribution per-particle via SVD-based PK1).
//! 3. `rigid_imprint` — stamp rigid-body velocities into overlapped
//!    grid cells (penalty coupling).
//! 4. `grid_update` — momentum / mass → velocity, gravity, boundary
//!    friction.
//! 5. `g2p` — grid → particle gather; APIC affine matrix C; F update.
//! 6. `plasticity` — SVD return mapping (Drucker-Prager).
//!
//! ## Storage
//!
//! Dense `R³` grid, `grid_size³` cells. v1 trade-off: a 32³ grid is
//! 32 768 cells × 8 bytes (mass + momentum atomic stores) ≈ 256 KB —
//! comfortable for the tests + viewer demo. A future Stage 24d.x could
//! swap in a Stage-24e sparse-grid backing without touching the
//! kernels.
//!
//! Particles are SoA — separate buffers for `pos`, `vel`, the affine
//! matrix `C`, the elastic deformation gradient `F`, scalar
//! `volume`, `mass`, and `jp` (the plastic-flow scalar that hardens
//! sand under shear).
//!
//! ## Public API
//!
//! ```rust,ignore
//! use molla::compute::{Engine, MpmSimulation, MpmConfig};
//!
//! let engine = Engine::new_gpu()?;
//! let cfg = MpmConfig::sand();
//! let mut sim = MpmSimulation::new(&engine, cfg)?;
//! sim.add_particles_box(/* min */, /* max */, /* count */);
//! for _ in 0..240 {
//!     sim.step(&engine, 1.0 / 240.0)?;
//! }
//! let positions = sim.read_positions(&engine)?;
//! ```
//!
//! Performance v1 (measured on RX 6700 XT, Vulkan via wgpu):
//!   * 10 000 particles + 32³ grid: ≥ 30 Hz @ 1 substep, dt = 1/240 s
//!   * 50 000 particles + 64³ grid is **not** met in v1 due to
//!     wgpu submit-overhead — every substep is one encoder. Fixing
//!     this requires either `wgpu::CommandEncoder::insert_debug_marker`
//!     batching or persistent command buffer ring, both deferred.
//!
//! ## Stage 24d.2 — persistent-buffer dispatcher
//!
//! `step()` still allocates one encoder per call; for high substep
//! counts the per-call submit / encoder cost dominates the actual GPU
//! compute. [`MpmSimulation::step_n`] batches `N` substeps into a
//! single encoder + single `queue.submit`, sharing pre-allocated
//! uniform + readback staging buffers. The viewer + perf tests use
//! this path. See `tests/mpm_perf.rs::mpm_step_n_beats_individual_steps`
//! for the empirical proof.

use crate::compute::Engine;
use crate::core::{Error, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    particle_count: u32,
    rigid_count: u32,
    grid_size: u32,
    grid_total: u32,

    grid_origin: [f32; 3],
    cell_size: f32,

    dt: f32,
    gravity_y: f32,
    youngs: f32,
    poisson: f32,

    dp_alpha: f32,
    cohesion: f32,
    boundary_friction: f32,
    inv_cell_size: f32,

    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Material configuration. Defaults are tuned for dry sand
/// (Klár 2016 §6).
#[derive(Copy, Clone, Debug)]
pub struct MpmConfig {
    /// Cubic grid edge length (cells per side).
    pub grid_size: u32,
    /// World-space size of one grid cell.
    pub cell_size: f32,
    /// Lower-left corner of the grid in world coords.
    pub grid_origin: [f32; 3],
    /// Gravity along `+Y`. Set to `-9.81` for terrestrial sims.
    pub gravity_y: f32,
    /// Young's modulus E (Pa). Sand: `1e5` is typical for "sandbox" toy
    /// rheology; `1e7` for cohesive packings.
    pub youngs: f32,
    /// Poisson's ratio ν.
    pub poisson: f32,
    /// Drucker-Prager friction angle φ (radians). Used to compute the
    /// dimensionless cone constant α host-side.
    pub friction_angle: f32,
    /// Cohesion c (Pa). 0 for granular sand.
    pub cohesion: f32,
    /// Velocity multiplier on the boundary tangent (Coulomb-style).
    /// 0 = sticking, 1 = frictionless slip.
    pub boundary_friction: f32,
    /// Per-particle mass (kg). For uniform fills only — extension
    /// hooks live on `set_particles`.
    pub particle_mass: f32,
    /// Per-particle reference volume (m³).
    pub particle_volume: f32,
    /// Maximum particles the simulation will accept.
    pub max_particles: u32,
    /// Maximum rigid-body imprints per substep.
    pub max_rigid: u32,
}

impl MpmConfig {
    /// Default sand configuration. v1 trade-off: Young's modulus is
    /// kept low (5e3) so the explicit time step stays inside CFL
    /// without sub-millisecond dt. For stiffer rheology the caller
    /// must lower `dt` accordingly (`c = sqrt(E/ρ); dt < dx/c`).
    pub fn sand() -> Self {
        Self {
            grid_size: 32,
            cell_size: 0.05,
            grid_origin: [-0.8, 0.0, -0.8],
            gravity_y: -9.81,
            youngs: 5.0e3,
            poisson: 0.3,
            friction_angle: 30.0_f32.to_radians(),
            cohesion: 0.0,
            boundary_friction: 0.5,
            particle_mass: 1.0e-3,
            particle_volume: 1.0e-6,
            max_particles: 16_384,
            max_rigid: 8,
        }
    }
}

/// Owns particle + grid GPU buffers and the compute pipelines for one
/// MPM simulation. Re-buildable on config change (call [`Self::new`]).
#[allow(dead_code)]
pub struct MpmSimulation {
    cfg: MpmConfig,

    // Pipelines (one shader, six entry points).
    pipe_reset: wgpu::ComputePipeline,
    pipe_p2g: wgpu::ComputePipeline,
    pipe_rigid: wgpu::ComputePipeline,
    pipe_grid: wgpu::ComputePipeline,
    pipe_g2p: wgpu::ComputePipeline,
    pipe_plastic: wgpu::ComputePipeline,

    // Bind group layouts.
    layout_uniforms: wgpu::BindGroupLayout,
    layout_particles: wgpu::BindGroupLayout,
    layout_grid: wgpu::BindGroupLayout,
    layout_rigid: wgpu::BindGroupLayout,

    // Particle buffers.
    p_pos: wgpu::Buffer,
    p_vel: wgpu::Buffer,
    p_c: wgpu::Buffer,
    p_f: wgpu::Buffer,
    p_volume: wgpu::Buffer,
    p_mass: wgpu::Buffer,
    p_jp: wgpu::Buffer,

    // Grid buffers.
    g_mass: wgpu::Buffer,
    g_momentum: wgpu::Buffer,
    g_velocity: wgpu::Buffer,
    g_rigid_mask: wgpu::Buffer,

    // Rigid input buffers (sphere primitives).
    rigid_center: wgpu::Buffer,
    rigid_vel: wgpu::Buffer,

    // Stage 24d.2: persistent uniform buffer + bind group.  Re-uploaded
    // each substep via `queue.write_buffer` rather than re-created.
    uniform_buf: wgpu::Buffer,
    bg_uniforms: wgpu::BindGroup,

    // Stage 24d.2: persistent readback staging buffers (one per
    // particle stream).  Sized to `max_particles * 16` bytes so we never
    // hit a copy-time reallocation.
    stage_pos: wgpu::Buffer,
    stage_vel: wgpu::Buffer,

    // Bind groups (persistent; built once at construction).
    bg_particles: wgpu::BindGroup,
    bg_grid: wgpu::BindGroup,
    bg_rigid: wgpu::BindGroup,

    // Live counters.
    particle_count: u32,
    rigid_count: u32,

    // Host-side mirrors for tests + viewer.
    host_pos: Vec<[f32; 4]>,
    host_vel: Vec<[f32; 4]>,
}

impl MpmSimulation {
    pub fn new(engine: &Engine, cfg: MpmConfig) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("MpmSimulation needs a GPU engine".into()))?
            .clone();
        Self::new_with_device(&device, cfg)
    }

    pub fn new_with_device(device: &wgpu::Device, cfg: MpmConfig) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::mpm"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/mpm.wgsl").into()),
        });

        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpm uniforms"),
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
        let layout_particles = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpm particles"),
            entries: &storage_entries(7, false),
        });
        let layout_grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpm grid"),
            entries: &storage_entries(4, false),
        });
        let layout_rigid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpm rigid"),
            entries: &storage_entries(2, true),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::mpm pipeline layout"),
            bind_group_layouts: &[
                &layout_uniforms,
                &layout_particles,
                &layout_grid,
                &layout_rigid,
            ],
            push_constant_ranges: &[],
        });

        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pipe_reset = mk("reset_grid");
        let pipe_p2g = mk("p2g");
        let pipe_rigid = mk("rigid_imprint");
        let pipe_grid = mk("grid_update");
        let pipe_g2p = mk("g2p");
        let pipe_plastic = mk("plasticity");

        // Allocate buffers.
        let n_particles = cfg.max_particles.max(1) as u64;
        let grid_total = (cfg.grid_size as u64).pow(3);
        let n_rigid = cfg.max_rigid.max(1) as u64;

        let mk_buf = |label: &str, size_bytes: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size_bytes,
                usage,
                mapped_at_creation: false,
            })
        };

        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let p_pos = mk_buf("mpm.p_pos", n_particles * 16, storage);
        let p_vel = mk_buf("mpm.p_vel", n_particles * 16, storage);
        // mat3x3<f32> in WGSL has stride 48 bytes (3 cols × vec3 padded to 16).
        let p_c = mk_buf("mpm.p_C", n_particles * 48, storage);
        let p_f = mk_buf("mpm.p_F", n_particles * 48, storage);
        let p_volume = mk_buf("mpm.p_volume", n_particles * 4, storage);
        let p_mass = mk_buf("mpm.p_mass", n_particles * 4, storage);
        let p_jp = mk_buf("mpm.p_jp", n_particles * 4, storage);

        let g_mass = mk_buf("mpm.g_mass", grid_total * 4, storage);
        let g_momentum = mk_buf("mpm.g_momentum", grid_total * 12, storage);
        let g_velocity = mk_buf("mpm.g_velocity", grid_total * 16, storage);
        let g_rigid_mask = mk_buf("mpm.g_rigid_mask", grid_total * 4, storage);

        let rigid_center = mk_buf("mpm.rigid_center", n_rigid * 16, storage);
        let rigid_vel = mk_buf("mpm.rigid_vel", n_rigid * 16, storage);

        // Stage 24d.2: persistent uniform buffer + readback staging.
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::mpm uniforms (persistent)"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpm uniforms bg (persistent)"),
            layout: &layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let stage_bytes = n_particles * 16;
        let stage_pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::mpm stage_pos (persistent)"),
            size: stage_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let stage_vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::mpm stage_vel (persistent)"),
            size: stage_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Pre-build bind groups for the persistent buffers.
        let bg_particles = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpm particles bg"),
            layout: &layout_particles,
            entries: &[
                buffer_entry(0, &p_pos),
                buffer_entry(1, &p_vel),
                buffer_entry(2, &p_c),
                buffer_entry(3, &p_f),
                buffer_entry(4, &p_volume),
                buffer_entry(5, &p_mass),
                buffer_entry(6, &p_jp),
            ],
        });
        let bg_grid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpm grid bg"),
            layout: &layout_grid,
            entries: &[
                buffer_entry(0, &g_mass),
                buffer_entry(1, &g_momentum),
                buffer_entry(2, &g_velocity),
                buffer_entry(3, &g_rigid_mask),
            ],
        });
        let bg_rigid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpm rigid bg"),
            layout: &layout_rigid,
            entries: &[
                buffer_entry(0, &rigid_center),
                buffer_entry(1, &rigid_vel),
            ],
        });

        Ok(Self {
            cfg,
            pipe_reset,
            pipe_p2g,
            pipe_rigid,
            pipe_grid,
            pipe_g2p,
            pipe_plastic,
            layout_uniforms,
            layout_particles,
            layout_grid,
            layout_rigid,
            p_pos,
            p_vel,
            p_c,
            p_f,
            p_volume,
            p_mass,
            p_jp,
            g_mass,
            g_momentum,
            g_velocity,
            g_rigid_mask,
            rigid_center,
            rigid_vel,
            uniform_buf,
            bg_uniforms,
            stage_pos,
            stage_vel,
            bg_particles,
            bg_grid,
            bg_rigid,
            particle_count: 0,
            rigid_count: 0,
            host_pos: vec![],
            host_vel: vec![],
        })
    }

    pub fn config(&self) -> &MpmConfig {
        &self.cfg
    }

    /// Live particle count.
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    /// Replace the particle set with the given positions. Velocities
    /// start at zero, F = I, jp = 0.
    pub fn set_particles(&mut self, queue: &wgpu::Queue, positions: &[[f32; 3]]) -> Result<()> {
        let n = positions.len();
        if n as u32 > self.cfg.max_particles {
            return Err(Error::Gpu(format!(
                "set_particles: n={} > max_particles={}",
                n, self.cfg.max_particles
            )));
        }
        let pos: Vec<[f32; 4]> = positions.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        let vel: Vec<[f32; 4]> = vec![[0.0; 4]; n];
        // mat3x3 stride 48 bytes. Identity = (1,0,0,_, 0,1,0,_, 0,0,1,_)
        let mut f_data: Vec<[f32; 4]> = Vec::with_capacity(n * 3);
        let mut c_data: Vec<[f32; 4]> = Vec::with_capacity(n * 3);
        for _ in 0..n {
            f_data.push([1.0, 0.0, 0.0, 0.0]);
            f_data.push([0.0, 1.0, 0.0, 0.0]);
            f_data.push([0.0, 0.0, 1.0, 0.0]);
            c_data.push([0.0, 0.0, 0.0, 0.0]);
            c_data.push([0.0, 0.0, 0.0, 0.0]);
            c_data.push([0.0, 0.0, 0.0, 0.0]);
        }
        let volume: Vec<f32> = vec![self.cfg.particle_volume; n];
        let mass: Vec<f32> = vec![self.cfg.particle_mass; n];
        let jp: Vec<f32> = vec![0.0; n];

        queue.write_buffer(&self.p_pos, 0, bytemuck::cast_slice(&pos));
        queue.write_buffer(&self.p_vel, 0, bytemuck::cast_slice(&vel));
        queue.write_buffer(&self.p_c, 0, bytemuck::cast_slice(&c_data));
        queue.write_buffer(&self.p_f, 0, bytemuck::cast_slice(&f_data));
        queue.write_buffer(&self.p_volume, 0, bytemuck::cast_slice(&volume));
        queue.write_buffer(&self.p_mass, 0, bytemuck::cast_slice(&mass));
        queue.write_buffer(&self.p_jp, 0, bytemuck::cast_slice(&jp));

        self.particle_count = n as u32;
        self.host_pos = pos;
        self.host_vel = vel;
        Ok(())
    }

    /// Set rigid-sphere coupling. Each entry is `(center_xyz, radius,
    /// velocity_xyz)`. Cleared on the next call.
    pub fn set_rigid_spheres(
        &mut self,
        queue: &wgpu::Queue,
        spheres: &[([f32; 3], f32, [f32; 3])],
    ) -> Result<()> {
        let n = spheres.len();
        if n as u32 > self.cfg.max_rigid {
            return Err(Error::Gpu(format!(
                "set_rigid_spheres: n={} > max_rigid={}",
                n, self.cfg.max_rigid
            )));
        }
        let centers: Vec<[f32; 4]> = spheres
            .iter()
            .map(|(c, r, _)| [c[0], c[1], c[2], *r])
            .collect();
        let vels: Vec<[f32; 4]> = spheres
            .iter()
            .map(|(_, _, v)| [v[0], v[1], v[2], 0.0])
            .collect();
        if !centers.is_empty() {
            queue.write_buffer(&self.rigid_center, 0, bytemuck::cast_slice(&centers));
            queue.write_buffer(&self.rigid_vel, 0, bytemuck::cast_slice(&vels));
        }
        self.rigid_count = n as u32;
        Ok(())
    }

    /// Update material constants (Young's modulus + friction angle)
    /// without rebuilding the pipelines. Surfaced for the viewer's
    /// live-tuning sliders.
    pub fn set_material(&mut self, youngs: f32, friction_angle: f32) {
        self.cfg.youngs = youngs;
        self.cfg.friction_angle = friction_angle;
    }

    /// Run one substep at `dt`. Caller is responsible for sub-stepping
    /// to satisfy CFL — see the file header.  Back-compat wrapper that
    /// delegates to [`Self::step_n`] with `substeps=1`.
    pub fn step(&mut self, engine: &Engine, dt: f32) -> Result<()> {
        self.step_n(engine, dt, 1)
    }

    /// Back-compat: single-substep variant taking explicit device/queue.
    pub fn step_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
    ) -> Result<()> {
        self.step_n_with(device, queue, dt, 1)
    }

    /// Stage 24d.2: batched substep dispatcher.
    ///
    /// Encode `substeps` MPM substeps into a single command encoder and
    /// submit once.  All six kernel dispatches per substep share the
    /// persistent bind groups and uniform buffer; the host avoids any
    /// intra-step synchronisation.  Materially faster than calling
    /// `step()` `n` times when `n` is large because each `submit` has
    /// fixed wgpu / driver overhead independent of payload size.
    ///
    /// The uniform buffer is written once at the head of the call —
    /// values that vary between substeps would require either a
    /// per-substep `write_buffer` (re-allowed since WGPU 0.20) or
    /// dynamic uniform offsets.  v1 holds `dt`/material constants
    /// constant across the batch, matching the viewer's usage.
    pub fn step_n(&mut self, engine: &Engine, dt: f32, substeps: usize) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("step_n: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("step_n: no GPU queue".into()))?
            .clone();
        self.step_n_with(&device, &queue, dt, substeps)
    }

    pub fn step_n_with(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        substeps: usize,
    ) -> Result<()> {
        if self.particle_count == 0 || substeps == 0 {
            return Ok(());
        }
        let phi = self.cfg.friction_angle.max(1.0_f32.to_radians());
        let sin_phi = phi.sin();
        let dp_alpha =
            (2.0_f32 / 3.0).sqrt() * (2.0 * sin_phi) / (3.0 - sin_phi);

        let uniforms = Uniforms {
            particle_count: self.particle_count,
            rigid_count: self.rigid_count,
            grid_size: self.cfg.grid_size,
            grid_total: self.cfg.grid_size.pow(3),
            grid_origin: self.cfg.grid_origin,
            cell_size: self.cfg.cell_size,
            dt,
            gravity_y: self.cfg.gravity_y,
            youngs: self.cfg.youngs,
            poisson: self.cfg.poisson,
            dp_alpha,
            cohesion: self.cfg.cohesion,
            boundary_friction: self.cfg.boundary_friction,
            inv_cell_size: 1.0 / self.cfg.cell_size,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        // Stage 24d.2: persistent uniform buffer — one upload per call.
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let grid_total = self.cfg.grid_size.pow(3);
        let groups_grid = grid_total.div_ceil(WORKGROUP_SIZE);
        let groups_part = self.particle_count.div_ceil(WORKGROUP_SIZE);

        let mut encoder = _device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpm encoder (step_n)"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::mpm step_n"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bg_uniforms, &[]);
            pass.set_bind_group(1, &self.bg_particles, &[]);
            pass.set_bind_group(2, &self.bg_grid, &[]);
            pass.set_bind_group(3, &self.bg_rigid, &[]);

            for _ in 0..substeps {
                // 1. reset grid
                pass.set_pipeline(&self.pipe_reset);
                pass.dispatch_workgroups(groups_grid.max(1), 1, 1);

                // 2. p2g
                pass.set_pipeline(&self.pipe_p2g);
                pass.dispatch_workgroups(groups_part.max(1), 1, 1);

                // 3. rigid imprint (only if any rigids)
                if self.rigid_count > 0 {
                    pass.set_pipeline(&self.pipe_rigid);
                    pass.dispatch_workgroups(groups_grid.max(1), 1, 1);
                }

                // 4. grid update
                pass.set_pipeline(&self.pipe_grid);
                pass.dispatch_workgroups(groups_grid.max(1), 1, 1);

                // 5. g2p
                pass.set_pipeline(&self.pipe_g2p);
                pass.dispatch_workgroups(groups_part.max(1), 1, 1);

                // 6. plasticity
                pass.set_pipeline(&self.pipe_plastic);
                pass.dispatch_workgroups(groups_part.max(1), 1, 1);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Read particle positions back to host. Synchronous; intended for
    /// tests and the viewer's per-frame draw — high-throughput callers
    /// should mirror the staging-buffer pattern in `dense3.rs`.
    pub fn read_positions(&mut self, engine: &Engine) -> Result<Vec<[f32; 3]>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("read_positions: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("read_positions: no queue".into()))?
            .clone();
        self.read_positions_with(&device, &queue)
    }

    pub fn read_positions_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 3]>> {
        if self.particle_count == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.particle_count as u64) * 16;
        // Stage 24d.2: reuse the persistent staging buffer (sized to
        // max_particles at constructor time).
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpm read_positions encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.p_pos, 0, &self.stage_pos, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.stage_pos.slice(0..bytes);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("mpm readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("mpm readback poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("mpm readback recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.stage_pos.unmap();
        let out: Vec<[f32; 3]> = raw[..self.particle_count as usize]
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        Ok(out)
    }

    /// Read particle velocities back to host. Used by the
    /// momentum-conservation test.
    pub fn read_velocities_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 3]>> {
        if self.particle_count == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.particle_count as u64) * 16;
        // Stage 24d.2: reuse the persistent staging buffer.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpm read_velocities encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.p_vel, 0, &self.stage_vel, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = self.stage_vel.slice(0..bytes);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("mpm vel readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("mpm vel poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("mpm vel recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.stage_vel.unmap();
        let out: Vec<[f32; 3]> = raw[..self.particle_count as usize]
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        Ok(out)
    }
}

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
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

// Silence unused-import warnings on Arc when no test path keeps it
// alive. (Public re-exports use Arc via `Engine` only.)
#[allow(dead_code)]
fn _arc_keep_alive(_a: Arc<wgpu::Device>) {}
