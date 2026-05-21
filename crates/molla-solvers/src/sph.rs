//! Stages F1–F2 — Weakly-compressible SPH (WCSPH) fluid solver.
//!
//! A GPU-resident WCSPH fluid. Particle state lives in raw `wgpu`
//! storage buffers (SoA); neighbour search is a fixed-capacity uniform
//! grid; density/pressure/viscosity/integration run as five WGSL
//! compute kernels (`shaders/sph.wgsl`). The CPU reference for the same
//! math is [`crate::sph_oracle`].
//!
//! F2 adds **static boundary particles**: a scene is fluid particles at
//! indices `[0, n_fluid)` followed by boundary particles at
//! `[n_fluid, n)`. Boundary particles contribute to the density and
//! pressure sums (so fluid feels a real wall) but never move. Use
//! [`SphSimulation::set_scene`] and [`box_tank_walls`].
//!
//! ```rust,ignore
//! use molla_solvers::{SphConfig, SphSimulation, box_tank_walls};
//! let engine = molla_compute::Engine::new_gpu()?;
//! let cfg = SphConfig::water();
//! let mut sim = SphSimulation::new(&engine, cfg)?;
//! let walls = box_tank_walls(cfg.domain_min, cfg.domain_max, cfg.h, 2);
//! sim.set_scene(engine.queue().unwrap(), &fluid, &walls)?;
//! for _ in 0..240 { sim.step_n(&engine, 2.0e-4, 4)?; }
//! ```

use crate::compute::Engine;
use crate::core::{Error, Result};
use bytemuck::{Pod, Zeroable};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    particle_count: u32,
    grid_res: u32,
    grid_total: u32,
    max_per_cell: u32,

    grid_origin: [f32; 3],
    cell_size: f32,

    domain_min: [f32; 3],
    h: f32,

    domain_max: [f32; 3],
    dt: f32,

    mass: f32,
    rest_density: f32,
    stiffness_b: f32,
    gamma: f32,

    viscosity: f32,
    wall_restitution: f32,
    sound_speed: f32,
    n_fluid: u32,

    gravity: [f32; 3],
    friction: f32,
}

/// WCSPH solver configuration. The neighbour grid is derived from
/// `domain_*` and `h` (cell size = `2 * h`, the cubic-spline support).
#[derive(Copy, Clone, Debug)]
pub struct SphConfig {
    /// Buffer capacity — `set_particles`/`set_scene` reject more.
    pub max_particles: u32,
    /// Smoothing length `h` (kernel support radius is `2h`).
    pub h: f32,
    /// Rest density `ρ0` (kg/m³).
    pub rest_density: f32,
    /// Per-particle mass (kg). For a lattice at spacing `h` this should
    /// be `rest_density * h³` so a packed lattice samples `ρ0`.
    pub particle_mass: f32,
    /// Tait equation-of-state stiffness `B` (Pa). Sound speed is
    /// derived as `sqrt(B * gamma / rest_density)`.
    pub stiffness_b: f32,
    /// Tait exponent `gamma` (7 for water).
    pub gamma: f32,
    /// Monaghan artificial-viscosity coefficient `alpha` (~0.1–0.5).
    pub viscosity: f32,
    /// Gravity vector (m/s²).
    pub gravity: [f32; 3],
    /// Domain box minimum corner — fluid is clamped to it as a backstop.
    pub domain_min: [f32; 3],
    /// Domain box maximum corner.
    pub domain_max: [f32; 3],
    /// Wall bounce coefficient (0 = particles stick, 1 = elastic).
    pub wall_restitution: f32,
    /// Fixed per-cell bucket capacity. 64 is safe for WCSPH spacing.
    pub max_per_cell: u32,
    /// Granular Coulomb friction coefficient `mu` (F4). `0` = ordinary
    /// fluid; `> 0` caps inter-particle shear so a pile holds an angle
    /// of repose of roughly `atan(mu)`.
    pub friction: f32,
}

impl SphConfig {
    /// Water-like preset. Stiffness is kept moderate so a 240 Hz inner
    /// step (`dt ≈ 2e-4`) stays inside the WCSPH CFL limit.
    pub fn water() -> Self {
        let h = 0.05_f32;
        let rest_density = 1000.0_f32;
        Self {
            max_particles: 200_000,
            h,
            rest_density,
            particle_mass: rest_density * h * h * h,
            stiffness_b: 50_000.0,
            gamma: 7.0,
            viscosity: 0.25,
            gravity: [0.0, -9.81, 0.0],
            domain_min: [-0.5, 0.0, -0.5],
            domain_max: [0.5, 1.0, 0.5],
            wall_restitution: 0.0,
            max_per_cell: 64,
            friction: 0.0,
        }
    }

    /// Dry-sand-like granular preset (F4). Coulomb friction `mu = 0.6`
    /// targets an angle of repose near `atan(0.6) ≈ 31°`.
    pub fn sand() -> Self {
        Self {
            viscosity: 0.4,
            friction: 0.6,
            ..Self::water()
        }
    }

    /// Sound speed derived from the Tait parameters.
    pub fn sound_speed(&self) -> f32 {
        (self.stiffness_b * self.gamma / self.rest_density).sqrt()
    }

    /// Cubic-spline support radius / neighbour-grid cell size.
    fn cell_size(&self) -> f32 {
        2.0 * self.h
    }
}

/// Generate static boundary particles lining the floor and four side
/// walls of the box `[min, max]` (open top), `layers` particles thick,
/// at `spacing`. Pass the result as the `boundary` argument of
/// [`SphSimulation::set_scene`].
pub fn box_tank_walls(min: [f32; 3], max: [f32; 3], spacing: f32, layers: u32) -> Vec<[f32; 3]> {
    let pad = layers as f32 * spacing;
    // Open top: ceiling stays at max[1] (no pad above) so the fluid has
    // a free surface.
    let lo = [min[0] - pad, min[1] - pad, min[2] - pad];
    let hi = [max[0] + pad, max[1], max[2] + pad];
    let cells = |a: f32, b: f32| (((b - a) / spacing).round() as i32).max(0);
    let eps = spacing * 0.5;
    let mut v = Vec::new();
    for ix in 0..=cells(lo[0], hi[0]) {
        for iy in 0..=cells(lo[1], hi[1]) {
            for iz in 0..=cells(lo[2], hi[2]) {
                let p = [
                    lo[0] + ix as f32 * spacing,
                    lo[1] + iy as f32 * spacing,
                    lo[2] + iz as f32 * spacing,
                ];
                // Interior = the open fluid region. Everything else in
                // the padded box is a wall/floor particle.
                let interior = p[0] > min[0] + eps
                    && p[0] < max[0] - eps
                    && p[1] > min[1] + eps
                    && p[2] > min[2] + eps
                    && p[2] < max[2] - eps;
                if !interior {
                    v.push(p);
                }
            }
        }
    }
    v
}

/// A linear (translation-only) rigid body coupled to the SPH fluid (F3).
///
/// The body's `sample_offsets` (relative to `com`) become boundary
/// particles the fluid pushes on; the summed fluid load moves `com`.
/// Rotation is deferred — both FSI acceptance cases (buoyancy, momentum
/// transfer) are linear phenomena.
#[derive(Clone, Debug)]
pub struct SphRigidBody {
    /// Total body mass (kg). `mass / volume` is the body density — make
    /// it less than the fluid rest density for the body to float.
    pub mass: f32,
    /// Centre-of-mass world position.
    pub com: [f32; 3],
    /// Linear velocity (m/s).
    pub vel: [f32; 3],
    /// Sample-particle offsets relative to `com`.
    pub sample_offsets: Vec<[f32; 3]>,
}

impl SphRigidBody {
    /// A solid box centred at `com`, sampled as a volume lattice at
    /// roughly `spacing`.
    pub fn box_body(com: [f32; 3], half_extent: [f32; 3], spacing: f32, mass: f32) -> Self {
        let mut offsets = Vec::new();
        let n = |e: f32| ((2.0 * e / spacing).round() as i32).max(1);
        let (nx, ny, nz) = (n(half_extent[0]), n(half_extent[1]), n(half_extent[2]));
        for ix in 0..=nx {
            for iy in 0..=ny {
                for iz in 0..=nz {
                    offsets.push([
                        -half_extent[0] + ix as f32 * (2.0 * half_extent[0] / nx as f32),
                        -half_extent[1] + iy as f32 * (2.0 * half_extent[1] / ny as f32),
                        -half_extent[2] + iz as f32 * (2.0 * half_extent[2] / nz as f32),
                    ]);
                }
            }
        }
        Self {
            mass,
            com,
            vel: [0.0; 3],
            sample_offsets: offsets,
        }
    }

    /// World positions of the body's sample particles.
    fn sample_world(&self) -> Vec<[f32; 3]> {
        self.sample_offsets
            .iter()
            .map(|o| [self.com[0] + o[0], self.com[1] + o[1], self.com[2] + o[2]])
            .collect()
    }
}

/// Owns the particle + grid GPU buffers and the WCSPH compute pipelines.
#[allow(dead_code)]
pub struct SphSimulation {
    cfg: SphConfig,
    grid_res: u32,
    grid_origin: [f32; 3],
    cell_size: f32,

    pipe_reset: wgpu::ComputePipeline,
    pipe_build: wgpu::ComputePipeline,
    pipe_density: wgpu::ComputePipeline,
    pipe_stress: wgpu::ComputePipeline,
    pipe_forces: wgpu::ComputePipeline,
    pipe_integrate: wgpu::ComputePipeline,

    layout_uniforms: wgpu::BindGroupLayout,
    layout_particles: wgpu::BindGroupLayout,
    layout_grid: wgpu::BindGroupLayout,

    p_pos: wgpu::Buffer,
    p_vel: wgpu::Buffer,
    p_density: wgpu::Buffer,
    p_pressure: wgpu::Buffer,
    p_accel: wgpu::Buffer,
    p_stress: wgpu::Buffer,

    g_count: wgpu::Buffer,
    g_cell: wgpu::Buffer,

    uniform_buf: wgpu::Buffer,
    bg_uniforms: wgpu::BindGroup,
    bg_particles: wgpu::BindGroup,
    bg_grid: wgpu::BindGroup,

    stage_pos: wgpu::Buffer,
    stage_vel: wgpu::Buffer,
    stage_scalar: wgpu::Buffer,

    /// Fluid particle count (movers). Boundary particles follow.
    n_fluid: u32,
    /// Total particle count (fluid + boundary).
    particle_count: u32,
    /// FSI rigid bodies coupled to the fluid (F3).
    bodies: Vec<SphRigidBody>,
    /// `(start, count)` of each body's sample particles in the buffers.
    body_ranges: Vec<(u32, u32)>,
}

impl SphSimulation {
    /// Build a simulation against the GPU [`Engine`].
    pub fn new(engine: &Engine, cfg: SphConfig) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("SphSimulation needs a GPU engine".into()))?
            .clone();
        Self::new_with_device(&device, cfg)
    }

    /// Build a simulation against a raw `wgpu::Device`.
    pub fn new_with_device(device: &wgpu::Device, cfg: SphConfig) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sph"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sph.wgsl").into()),
        });

        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sph uniforms"),
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
            label: Some("molla::sph particles"),
            entries: &storage_entries(6, false),
        });
        let layout_grid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sph grid"),
            entries: &storage_entries(2, false),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::sph pipeline layout"),
            bind_group_layouts: &[&layout_uniforms, &layout_particles, &layout_grid],
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
        let pipe_build = mk("build_grid");
        let pipe_density = mk("compute_density");
        let pipe_stress = mk("compute_stress");
        let pipe_forces = mk("compute_forces");
        let pipe_integrate = mk("integrate");

        // The neighbour grid covers the domain expanded by a margin so
        // boundary particles placed just outside the fluid domain still
        // land in their own cells (no bucket overflow at the edges).
        let cell_size = cfg.cell_size();
        let margin = 4.0 * cfg.h;
        let grid_origin = [
            cfg.domain_min[0] - margin,
            cfg.domain_min[1] - margin,
            cfg.domain_min[2] - margin,
        ];
        let extent = [
            cfg.domain_max[0] - cfg.domain_min[0] + 2.0 * margin,
            cfg.domain_max[1] - cfg.domain_min[1] + 2.0 * margin,
            cfg.domain_max[2] - cfg.domain_min[2] + 2.0 * margin,
        ];
        let max_ext = extent[0].max(extent[1]).max(extent[2]).max(cell_size);
        let grid_res = ((max_ext / cell_size).ceil() as u32 + 1).max(1);
        let grid_total = (grid_res as u64).pow(3);
        let n = cfg.max_particles.max(1) as u64;

        let mk_buf = |label: &str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let p_pos = mk_buf("sph.p_pos", n * 16, storage);
        let p_vel = mk_buf("sph.p_vel", n * 16, storage);
        let p_density = mk_buf("sph.p_density", n * 4, storage);
        let p_pressure = mk_buf("sph.p_pressure", n * 4, storage);
        let p_accel = mk_buf("sph.p_accel", n * 16, storage);
        // 6 f32 of symmetric deviatoric stress per particle (F4).
        let p_stress = mk_buf("sph.p_stress", n * 24, storage);
        let g_count = mk_buf("sph.g_count", grid_total * 4, storage);
        let g_cell = mk_buf(
            "sph.g_cell",
            grid_total * (cfg.max_per_cell as u64) * 4,
            storage,
        );

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sph uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sph uniforms bg"),
            layout: &layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        let mk_stage = |label: &str, size: u64| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let stage_pos = mk_stage("molla::sph stage_pos", n * 16);
        let stage_vel = mk_stage("molla::sph stage_vel", n * 16);
        let stage_scalar = mk_stage("molla::sph stage_scalar", n * 4);

        let bg_particles = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sph particles bg"),
            layout: &layout_particles,
            entries: &[
                buffer_entry(0, &p_pos),
                buffer_entry(1, &p_vel),
                buffer_entry(2, &p_density),
                buffer_entry(3, &p_pressure),
                buffer_entry(4, &p_accel),
                buffer_entry(5, &p_stress),
            ],
        });
        let bg_grid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sph grid bg"),
            layout: &layout_grid,
            entries: &[buffer_entry(0, &g_count), buffer_entry(1, &g_cell)],
        });

        Ok(Self {
            cfg,
            grid_res,
            grid_origin,
            cell_size,
            pipe_reset,
            pipe_build,
            pipe_density,
            pipe_stress,
            pipe_forces,
            pipe_integrate,
            layout_uniforms,
            layout_particles,
            layout_grid,
            p_pos,
            p_vel,
            p_density,
            p_pressure,
            p_accel,
            p_stress,
            g_count,
            g_cell,
            uniform_buf,
            bg_uniforms,
            bg_particles,
            bg_grid,
            stage_pos,
            stage_vel,
            stage_scalar,
            n_fluid: 0,
            particle_count: 0,
            bodies: Vec::new(),
            body_ranges: Vec::new(),
        })
    }

    /// Solver configuration.
    pub fn config(&self) -> &SphConfig {
        &self.cfg
    }

    /// Live fluid (moving) particle count.
    pub fn fluid_count(&self) -> u32 {
        self.n_fluid
    }

    /// Total particle count (fluid + boundary).
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    /// Replace the scene with fluid particles only (no boundary).
    /// Fluid velocities start at rest.
    pub fn set_particles(&mut self, queue: &wgpu::Queue, positions: &[[f32; 3]]) -> Result<()> {
        self.set_scene(queue, positions, &[])
    }

    /// Replace the scene with `fluid` (moving) and `boundary` (static)
    /// particles. Fluid is stored first, boundary after; all velocities
    /// start at rest.
    pub fn set_scene(
        &mut self,
        queue: &wgpu::Queue,
        fluid: &[[f32; 3]],
        boundary: &[[f32; 3]],
    ) -> Result<()> {
        let nf = fluid.len();
        let total = nf + boundary.len();
        if total as u32 > self.cfg.max_particles {
            return Err(Error::Gpu(format!(
                "set_scene: total={total} > max_particles={}",
                self.cfg.max_particles
            )));
        }
        let mut pos: Vec<[f32; 4]> = Vec::with_capacity(total);
        pos.extend(fluid.iter().map(|p| [p[0], p[1], p[2], 0.0]));
        pos.extend(boundary.iter().map(|p| [p[0], p[1], p[2], 0.0]));
        queue.write_buffer(&self.p_pos, 0, bytemuck::cast_slice(&pos));
        queue.write_buffer(
            &self.p_vel,
            0,
            bytemuck::cast_slice(&vec![[0.0f32; 4]; total]),
        );
        queue.write_buffer(
            &self.p_accel,
            0,
            bytemuck::cast_slice(&vec![[0.0f32; 4]; total]),
        );
        queue.write_buffer(
            &self.p_density,
            0,
            bytemuck::cast_slice(&vec![self.cfg.rest_density; total]),
        );
        queue.write_buffer(
            &self.p_pressure,
            0,
            bytemuck::cast_slice(&vec![0.0f32; total]),
        );
        queue.write_buffer(
            &self.p_stress,
            0,
            bytemuck::cast_slice(&vec![0.0f32; total * 6]),
        );
        self.n_fluid = nf as u32;
        self.particle_count = total as u32;
        self.bodies = Vec::new();
        self.body_ranges = Vec::new();
        Ok(())
    }

    /// Replace the scene with `fluid`, `static_walls`, and FSI rigid
    /// `bodies`. Body sample particles become boundary particles the
    /// fluid pushes on; advance the scene with [`Self::step_fsi`].
    pub fn set_fsi_scene(
        &mut self,
        queue: &wgpu::Queue,
        fluid: &[[f32; 3]],
        static_walls: &[[f32; 3]],
        bodies: Vec<SphRigidBody>,
    ) -> Result<()> {
        let nf = fluid.len();
        let mut boundary: Vec<[f32; 3]> = static_walls.to_vec();
        let mut vel: Vec<[f32; 4]> = vec![[0.0; 4]; nf + static_walls.len()];
        let mut ranges: Vec<(u32, u32)> = Vec::with_capacity(bodies.len());
        for b in &bodies {
            let start = (nf + boundary.len()) as u32;
            let samples = b.sample_world();
            ranges.push((start, samples.len() as u32));
            boundary.extend(samples);
            for _ in 0..b.sample_offsets.len() {
                vel.push([b.vel[0], b.vel[1], b.vel[2], 0.0]);
            }
        }
        let total = nf + boundary.len();
        if total as u32 > self.cfg.max_particles {
            return Err(Error::Gpu(format!(
                "set_fsi_scene: total={total} > max_particles={}",
                self.cfg.max_particles
            )));
        }
        let mut pos: Vec<[f32; 4]> = Vec::with_capacity(total);
        pos.extend(fluid.iter().map(|p| [p[0], p[1], p[2], 0.0]));
        pos.extend(boundary.iter().map(|p| [p[0], p[1], p[2], 0.0]));
        queue.write_buffer(&self.p_pos, 0, bytemuck::cast_slice(&pos));
        queue.write_buffer(&self.p_vel, 0, bytemuck::cast_slice(&vel));
        queue.write_buffer(
            &self.p_accel,
            0,
            bytemuck::cast_slice(&vec![[0.0f32; 4]; total]),
        );
        queue.write_buffer(
            &self.p_density,
            0,
            bytemuck::cast_slice(&vec![self.cfg.rest_density; total]),
        );
        queue.write_buffer(
            &self.p_pressure,
            0,
            bytemuck::cast_slice(&vec![0.0f32; total]),
        );
        queue.write_buffer(
            &self.p_stress,
            0,
            bytemuck::cast_slice(&vec![0.0f32; total * 6]),
        );
        self.n_fluid = nf as u32;
        self.particle_count = total as u32;
        self.bodies = bodies;
        self.body_ranges = ranges;
        Ok(())
    }

    /// Number of coupled FSI rigid bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Access a coupled rigid body's current state.
    pub fn body(&self, i: usize) -> &SphRigidBody {
        &self.bodies[i]
    }

    /// Advance the FSI scene by `substeps` GPU SPH substeps followed by
    /// one rigid-body update from the accumulated fluid load (staggered
    /// coupling — bodies are quasi-static within a batch).
    pub fn step_fsi(&mut self, engine: &Engine, dt: f32, substeps: u32) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("step_fsi: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("step_fsi: no queue".into()))?
            .clone();
        if self.particle_count == 0 || substeps == 0 {
            return Ok(());
        }
        // Push the current body sample positions/velocities.
        for (b, &(start, count)) in self.bodies.iter().zip(self.body_ranges.iter()) {
            let world: Vec<[f32; 4]> = b
                .sample_world()
                .iter()
                .map(|p| [p[0], p[1], p[2], 0.0])
                .collect();
            let vels: Vec<[f32; 4]> = vec![[b.vel[0], b.vel[1], b.vel[2], 0.0]; count as usize];
            queue.write_buffer(
                &self.p_pos,
                (start as u64) * 16,
                bytemuck::cast_slice(&world),
            );
            queue.write_buffer(
                &self.p_vel,
                (start as u64) * 16,
                bytemuck::cast_slice(&vels),
            );
        }
        // GPU SPH substeps.
        self.step_with(&device, &queue, dt, substeps)?;
        // Read the fluid load on every boundary particle.
        let accels = self.read_boundary_accels(&device, &queue)?;
        // Integrate each rigid body from its samples' summed fluid load.
        let eff_dt = dt * substeps as f32;
        let g = self.cfg.gravity;
        let pm = self.cfg.particle_mass;
        let nf = self.n_fluid;
        for (b, &(start, count)) in self.bodies.iter_mut().zip(self.body_ranges.iter()) {
            let base = (start - nf) as usize;
            let mut f = [0.0f32; 3];
            for k in 0..count as usize {
                let a = accels[base + k];
                f[0] += pm * a[0];
                f[1] += pm * a[1];
                f[2] += pm * a[2];
            }
            for axis in 0..3 {
                let acc = f[axis] / b.mass + g[axis];
                b.vel[axis] += acc * eff_dt;
                b.com[axis] += b.vel[axis] * eff_dt;
            }
        }
        Ok(())
    }

    /// Read the fluid load (accel per unit particle mass) on every
    /// boundary particle, indexed `[0, particle_count - n_fluid)`.
    fn read_boundary_accels(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 3]>> {
        let count = self.particle_count - self.n_fluid;
        if count == 0 {
            return Ok(vec![]);
        }
        let bytes = (count as u64) * 16;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sph read_accels"),
        });
        encoder.copy_buffer_to_buffer(
            &self.p_accel,
            (self.n_fluid as u64) * 16,
            &self.stage_pos,
            0,
            bytes,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let data = map_blocking(device, &self.stage_pos, bytes)?;
        let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.stage_pos.unmap();
        Ok(raw[..count as usize]
            .iter()
            .map(|a| [a[0], a[1], a[2]])
            .collect())
    }

    /// Advance one substep.
    pub fn step(&mut self, engine: &Engine, dt: f32) -> Result<()> {
        self.step_n(engine, dt, 1)
    }

    /// Advance `substeps` substeps of size `dt` in one GPU submission.
    pub fn step_n(&mut self, engine: &Engine, dt: f32, substeps: u32) -> Result<()> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("step_n: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("step_n: no queue".into()))?
            .clone();
        self.step_with(&device, &queue, dt, substeps)
    }

    /// Advance `substeps` substeps against raw `wgpu` handles.
    pub fn step_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        substeps: u32,
    ) -> Result<()> {
        if self.particle_count == 0 || substeps == 0 {
            return Ok(());
        }
        let grid_total = (self.grid_res as u64).pow(3) as u32;
        let uniforms = Uniforms {
            particle_count: self.particle_count,
            grid_res: self.grid_res,
            grid_total,
            max_per_cell: self.cfg.max_per_cell,
            grid_origin: self.grid_origin,
            cell_size: self.cell_size,
            domain_min: self.cfg.domain_min,
            h: self.cfg.h,
            domain_max: self.cfg.domain_max,
            dt,
            mass: self.cfg.particle_mass,
            rest_density: self.cfg.rest_density,
            stiffness_b: self.cfg.stiffness_b,
            gamma: self.cfg.gamma,
            viscosity: self.cfg.viscosity,
            wall_restitution: self.cfg.wall_restitution,
            sound_speed: self.cfg.sound_speed(),
            n_fluid: self.n_fluid,
            gravity: self.cfg.gravity,
            friction: self.cfg.friction,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let groups_grid = grid_total.div_ceil(WORKGROUP_SIZE).max(1);
        let groups_part = self.particle_count.div_ceil(WORKGROUP_SIZE).max(1);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sph encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sph step"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bg_uniforms, &[]);
            pass.set_bind_group(1, &self.bg_particles, &[]);
            pass.set_bind_group(2, &self.bg_grid, &[]);
            for _ in 0..substeps {
                pass.set_pipeline(&self.pipe_reset);
                pass.dispatch_workgroups(groups_grid, 1, 1);
                pass.set_pipeline(&self.pipe_build);
                pass.dispatch_workgroups(groups_part, 1, 1);
                pass.set_pipeline(&self.pipe_density);
                pass.dispatch_workgroups(groups_part, 1, 1);
                pass.set_pipeline(&self.pipe_stress);
                pass.dispatch_workgroups(groups_part, 1, 1);
                pass.set_pipeline(&self.pipe_forces);
                pass.dispatch_workgroups(groups_part, 1, 1);
                pass.set_pipeline(&self.pipe_integrate);
                pass.dispatch_workgroups(groups_part, 1, 1);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Read fluid particle positions back to the host (blocking).
    pub fn read_positions(&mut self, engine: &Engine) -> Result<Vec<[f32; 3]>> {
        let (device, queue) = self.engine_handles(engine)?;
        self.read_positions_with(&device, &queue)
    }

    /// Read fluid particle positions back against raw `wgpu` handles.
    pub fn read_positions_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 3]>> {
        let raw = read_vec4(device, queue, &self.p_pos, &self.stage_pos, self.n_fluid)?;
        Ok(raw.iter().map(|p| [p[0], p[1], p[2]]).collect())
    }

    /// Read fluid particle velocities back against raw `wgpu` handles.
    pub fn read_velocities_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 3]>> {
        let raw = read_vec4(device, queue, &self.p_vel, &self.stage_vel, self.n_fluid)?;
        Ok(raw.iter().map(|v| [v[0], v[1], v[2]]).collect())
    }

    /// Read per-fluid-particle densities back to the host (blocking).
    pub fn read_densities(&mut self, engine: &Engine) -> Result<Vec<f32>> {
        let (device, queue) = self.engine_handles(engine)?;
        self.read_densities_with(&device, &queue)
    }

    /// Read per-fluid-particle densities back against raw `wgpu` handles.
    pub fn read_densities_with(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<f32>> {
        if self.n_fluid == 0 {
            return Ok(vec![]);
        }
        let bytes = (self.n_fluid as u64) * 4;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sph read_densities"),
        });
        encoder.copy_buffer_to_buffer(&self.p_density, 0, &self.stage_scalar, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        let data = map_blocking(device, &self.stage_scalar, bytes)?;
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        self.stage_scalar.unmap();
        Ok(out)
    }

    fn engine_handles(
        &self,
        engine: &Engine,
    ) -> Result<(std::sync::Arc<wgpu::Device>, std::sync::Arc<wgpu::Queue>)> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("sph readback: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("sph readback: no queue".into()))?
            .clone();
        Ok((device, queue))
    }
}

fn read_vec4(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    stage: &wgpu::Buffer,
    count: u32,
) -> Result<Vec<[f32; 4]>> {
    if count == 0 {
        return Ok(vec![]);
    }
    let bytes = (count as u64) * 16;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::sph readback"),
    });
    encoder.copy_buffer_to_buffer(src, 0, stage, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let data = map_blocking(device, stage, bytes)?;
    let raw: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    stage.unmap();
    Ok(raw[..count as usize].to_vec())
}

fn map_blocking(
    device: &wgpu::Device,
    stage: &wgpu::Buffer,
    bytes: u64,
) -> Result<wgpu::BufferView> {
    let slice = stage.slice(0..bytes);
    let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("sph readback: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("sph readback poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("sph readback recv: {e:?}")))??;
    Ok(slice.get_mapped_range())
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
