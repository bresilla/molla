//! Stage 40 v2 — Rust dispatcher for the implicit-MPM device kernels.
//!
//! Pairs with `shaders/implicit_mpm.wgsl`. The shader now ships **five**
//! entry points:
//!
//! * `reset_grid` — zero per-cell mass / momentum / 27-stencil.
//! * `p2g` — particle → grid scatter (atomic f32 add for mass, momentum
//!   and the 27×27 = 729-pair Hessian contribution).
//! * `predict_grid` — `v_pred = momentum/m + g·dt` + floor; seeds
//!   `v_grid := v_pred` for the GS sweep.
//! * `gs_sweep` — colored Gauss-Seidel over the full 27-point Hessian
//!   (Stage 40 v1 ported the inner loop only; v2 keeps it identical).
//! * `floor_reapply` — post-solve floor clamp.
//! * `g2p` — grid → particle gather. Updates `p_vel` and the APIC
//!   affine matrix `p_C`. Bingham return mapping still runs CPU-side.
//!
//! The dispatcher owns **persistent GPU buffers** in `GpuStepState`:
//! particles are uploaded once at the start of a substep, the full
//! P2G + predict + GS + floor + G2P chain runs on-device with **zero**
//! intermediate host↔device traffic, and only `p_vel` / `p_C` are read
//! back at the end so the CPU can apply the Bingham projection and
//! advect. v3 will move the Bingham step onto the device too.

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::math::Real;

const WG_SIZE: u32 = 64;
const WG_PARTICLES: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    grid_size: u32,
    color_n: u32,
    use_off_diag: u32,
    particle_count: u32,

    dt: f32,
    cell_size: f32,
    inv_cell_size: f32,
    gravity_y: f32,

    floor_y: f32,
    floor_friction: f32,
    mass_p: f32,
    particle_volume: f32,

    viscosity: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,

    yield_stress: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Compute pipelines + bind-group layouts for the implicit-MPM kernels.
/// Build once per device; reuse across many timesteps.
pub struct ImplicitMpmKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipe_reset: wgpu::ComputePipeline,
    pipe_p2g: wgpu::ComputePipeline,
    pipe_predict: wgpu::ComputePipeline,
    pipe_gs: wgpu::ComputePipeline,
    pipe_floor: wgpu::ComputePipeline,
    pipe_g2p: wgpu::ComputePipeline,
    pipe_bingham: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ImplicitMpmKernel {
    /// Build the kernel against `device` + `queue`.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::implicit_mpm"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/implicit_mpm.wgsl").into(),
            ),
        });

        // Bindings 0, 1, 5 are u32-bitcast atomic-float storage buffers
        // (mass, k_stencil, momentum). Bindings 2 + 4 are vec3<f32>
        // storage. Binding 3 is the color-cells u32 array. Bindings 6,
        // 7 are particle pos / vel (vec3<f32>). Binding 8 is a flat f32
        // array of APIC C matrices (9 entries / particle). Binding 9 is
        // a u32 active flag per cell.
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::implicit_mpm io"),
            entries: &[
                rw_storage(0), // g_mass_atomic
                rw_storage(1), // k_stencil_atomic
                rw_storage(2), // v_pred
                ro_storage(3), // color_cells
                rw_storage(4), // v_grid
                rw_storage(5), // g_momentum_atomic
                rw_storage(6), // p_pos
                rw_storage(7), // p_vel
                rw_storage(8), // p_C
                rw_storage(9), // g_active
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::implicit_mpm uniforms"),
            entries: &[uniform_entry(0)],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::implicit_mpm layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let mk_pipe = |entry: &'static str, label: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipe_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipe_reset = mk_pipe("reset_grid", "molla::implicit_mpm::reset_grid");
        let pipe_p2g = mk_pipe("p2g", "molla::implicit_mpm::p2g");
        let pipe_predict = mk_pipe("predict_grid", "molla::implicit_mpm::predict_grid");
        let pipe_gs = mk_pipe("gs_sweep", "molla::implicit_mpm::gs_sweep");
        let pipe_floor = mk_pipe("floor_reapply", "molla::implicit_mpm::floor_reapply");
        let pipe_g2p = mk_pipe("g2p", "molla::implicit_mpm::g2p");
        let pipe_bingham = mk_pipe("bingham_advect", "molla::implicit_mpm::bingham_advect");

        Self {
            device,
            queue,
            pipe_reset,
            pipe_p2g,
            pipe_predict,
            pipe_gs,
            pipe_floor,
            pipe_g2p,
            pipe_bingham,
            layout_io,
            layout_uniforms,
        }
    }

    /// Owned `device` borrow — kept so callers can build extra buffers
    /// without re-passing it down.
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Owned `queue` borrow.
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}

/// Persistent device-side state for one implicit-MPM scene. Holds all
/// the grid + particle buffers; reused across timesteps so the
/// per-step CPU↔GPU traffic is one upload of `p_pos`/`p_vel`/`p_C`
/// (≤ 36·N bytes for N particles) and one readback of `p_vel`/`p_C`
/// at the end. v3 will keep particles resident too once the Bingham
/// return mapping runs on the device.
pub struct GpuStepState {
    pub grid_size: u32,
    pub max_particles: u32,
    pub g_total: u32,

    // Grid scratch.
    pub g_mass: wgpu::Buffer,
    pub g_momentum: wgpu::Buffer,
    pub k_stencil: wgpu::Buffer,
    pub v_pred: wgpu::Buffer,
    pub v_grid: wgpu::Buffer,
    pub g_active: wgpu::Buffer,

    // Particle state.
    pub p_pos: wgpu::Buffer,
    pub p_vel: wgpu::Buffer,
    pub p_c: wgpu::Buffer,

    // Color groups (one buffer per color, in dispatch order).
    pub color_bufs: Vec<wgpu::Buffer>,
    pub color_lens: Vec<u32>,

    // Per-color uniform buffers (one per color).
    pub color_uniforms: Vec<wgpu::Buffer>,

    // Uniforms shared by reset / p2g / predict / floor / g2p.
    pub uniforms: wgpu::Buffer,

    // Persistent staging buffers for the v3 fused readback (pos + vel).
    // Sized to `max_particles`; reused across substeps so we don't
    // reallocate every step.
    pub staging_pos: wgpu::Buffer,
    pub staging_vel: wgpu::Buffer,
}

impl GpuStepState {
    /// Build all persistent buffers. Sizes scale with `grid_size³` and
    /// `max_particles`; the per-step path keeps these constant.
    pub fn new(kernel: &ImplicitMpmKernel, grid_size: u32, max_particles: u32) -> Self {
        let device = kernel.device();
        let g_total = grid_size * grid_size * grid_size;
        let mk_storage = |label: &str, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let g_mass = mk_storage("molla::implicit_mpm g_mass", g_total as u64 * 4);
        let g_momentum = mk_storage("molla::implicit_mpm g_momentum", g_total as u64 * 3 * 4);
        let k_stencil = mk_storage("molla::implicit_mpm k_stencil", g_total as u64 * 27 * 4);
        let v_pred = mk_storage("molla::implicit_mpm v_pred", g_total as u64 * 16);
        let v_grid = mk_storage("molla::implicit_mpm v_grid", g_total as u64 * 16);
        let g_active = mk_storage("molla::implicit_mpm g_active", g_total as u64 * 4);
        let p_pos = mk_storage("molla::implicit_mpm p_pos", max_particles as u64 * 16);
        let p_vel = mk_storage("molla::implicit_mpm p_vel", max_particles as u64 * 16);
        let p_c = mk_storage("molla::implicit_mpm p_C", max_particles as u64 * 9 * 4);

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::implicit_mpm uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Persistent v3 readback staging. Sized to `max_particles`
        // with 16-byte vec3 stride to match `p_pos`/`p_vel`.
        let staging_bytes = (max_particles as u64 * 16).max(16);
        let staging_pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::implicit_mpm staging pos"),
            size: staging_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::implicit_mpm staging vel"),
            size: staging_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            grid_size,
            max_particles,
            g_total,
            g_mass,
            g_momentum,
            k_stencil,
            v_pred,
            v_grid,
            g_active,
            p_pos,
            p_vel,
            p_c,
            color_bufs: Vec::new(),
            color_lens: Vec::new(),
            color_uniforms: Vec::new(),
            uniforms,
            staging_pos,
            staging_vel,
        }
    }

    /// Upload color groups (run once per scene; the grid coloring
    /// only depends on `grid_size` so cached colors stay valid across
    /// substeps). One u32 buffer + one uniform buffer per color.
    pub fn upload_color_groups(&mut self, kernel: &ImplicitMpmKernel, groups: &[Vec<u32>]) {
        let device = kernel.device();
        let queue = kernel.queue();
        self.color_bufs.clear();
        self.color_lens.clear();
        self.color_uniforms.clear();
        for (c, group) in groups.iter().enumerate() {
            let bytes: Vec<u8> = if group.is_empty() {
                bytemuck::cast_slice::<u32, u8>(&[0u32]).to_vec()
            } else {
                bytemuck::cast_slice::<u32, u8>(group).to_vec()
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("molla::implicit_mpm color {c}")),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, &bytes);
            self.color_bufs.push(buf);
            self.color_lens.push(group.len() as u32);

            // Per-color uniform (only `color_n` differs).
            let u = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("molla::implicit_mpm color-u {c}")),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.color_uniforms.push(u);
        }
    }

    /// Upload all particle state for one substep. Sizes must match
    /// the `max_particles` reserved at construction.
    pub fn upload_particles(
        &self,
        kernel: &ImplicitMpmKernel,
        pos: &[[f32; 3]],
        vel: &[[f32; 3]],
        c_mat: &[f32], // 9*N entries, column-major
    ) {
        let queue = kernel.queue();
        let n = pos.len();
        assert!(n as u32 <= self.max_particles);
        assert_eq!(vel.len(), n);
        assert_eq!(c_mat.len(), n * 9);
        let pad_vec3 = |src: &[[f32; 3]]| -> Vec<f32> {
            let mut padded = Vec::with_capacity(src.len() * 4);
            for v in src {
                padded.push(v[0]);
                padded.push(v[1]);
                padded.push(v[2]);
                padded.push(0.0);
            }
            padded
        };
        let pos_padded = pad_vec3(pos);
        let vel_padded = pad_vec3(vel);
        queue.write_buffer(&self.p_pos, 0, bytemuck::cast_slice(&pos_padded));
        queue.write_buffer(&self.p_vel, 0, bytemuck::cast_slice(&vel_padded));
        queue.write_buffer(&self.p_c, 0, bytemuck::cast_slice(c_mat));
    }

    /// Stage 40 v3 — upload only `p_pos` and `p_vel`, leaving the
    /// device-side `p_C` resident across substeps. Used when the
    /// caller has run `run_full_step_native` previously and the device
    /// already holds the post-Bingham projected affine matrices.
    pub fn upload_pos_vel(
        &self,
        kernel: &ImplicitMpmKernel,
        pos: &[[f32; 3]],
        vel: &[[f32; 3]],
    ) {
        let queue = kernel.queue();
        let n = pos.len();
        assert!(n as u32 <= self.max_particles);
        assert_eq!(vel.len(), n);
        let pad_vec3 = |src: &[[f32; 3]]| -> Vec<f32> {
            let mut padded = Vec::with_capacity(src.len() * 4);
            for v in src {
                padded.push(v[0]);
                padded.push(v[1]);
                padded.push(v[2]);
                padded.push(0.0);
            }
            padded
        };
        let pos_padded = pad_vec3(pos);
        let vel_padded = pad_vec3(vel);
        queue.write_buffer(&self.p_pos, 0, bytemuck::cast_slice(&pos_padded));
        queue.write_buffer(&self.p_vel, 0, bytemuck::cast_slice(&vel_padded));
    }

    /// Zero the resident `p_C` buffer. Call once before the first
    /// substep when starting from an APIC-clean state (all zeros).
    pub fn zero_pc(&self, kernel: &ImplicitMpmKernel) {
        let queue = kernel.queue();
        let zeros = vec![0.0_f32; self.max_particles as usize * 9];
        queue.write_buffer(&self.p_c, 0, bytemuck::cast_slice(&zeros));
    }

    /// Run one full P2G + predict + GS + floor + G2P pass on the GPU,
    /// then read back `p_vel` and `p_C` so the host can do the Bingham
    /// projection and advect.
    ///
    /// `gs_iters` and `use_off_diag` mirror the CPU step. Returns the
    /// new particle velocities + APIC C matrices.
    #[allow(clippy::too_many_arguments)]
    pub fn run_full_step(
        &self,
        kernel: &ImplicitMpmKernel,
        particle_count: u32,
        cfg: &StepUniforms,
        gs_iters: u32,
    ) -> Result<(Vec<[f32; 3]>, Vec<f32>)> {
        let device = kernel.device();
        let queue = kernel.queue();

        // Write the "main" uniform block once. Used by reset/p2g/
        // predict/floor/g2p — color_n is overwritten per dispatch by
        // the color-specific uniforms used by gs_sweep.
        let u = Uniforms {
            grid_size: self.grid_size,
            color_n: 0,
            use_off_diag: u32::from(cfg.use_off_diag),
            particle_count,
            dt: cfg.dt,
            cell_size: cfg.cell_size,
            inv_cell_size: cfg.inv_cell_size,
            gravity_y: cfg.gravity_y,
            floor_y: cfg.floor_y,
            floor_friction: cfg.floor_friction,
            mass_p: cfg.mass_p,
            particle_volume: cfg.particle_volume,
            viscosity: cfg.viscosity,
            grid_origin_x: cfg.grid_origin[0],
            grid_origin_y: cfg.grid_origin[1],
            grid_origin_z: cfg.grid_origin[2],
            yield_stress: cfg.yield_stress,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        for (i, ucb) in self.color_uniforms.iter().enumerate() {
            let mut uc = u;
            uc.color_n = self.color_lens[i];
            queue.write_buffer(ucb, 0, bytemuck::bytes_of(&uc));
        }

        let bg_io = self.make_io_bind_group(kernel, None);
        // GS sweep needs the color's `color_cells` buffer bound at
        // binding 3 — `bg_io` defaults to color 0, so for each color
        // group we build its own bind group with the proper color
        // buffer plugged in.
        let bg_gs_per_color: Vec<wgpu::BindGroup> = (0..self.color_bufs.len())
            .map(|c| self.make_io_bind_group(kernel, Some(c)))
            .collect();
        let bg_u_main = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::implicit_mpm u-main bg"),
            layout: &kernel.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniforms.as_entire_binding(),
            }],
        });
        let bg_u_colors: Vec<wgpu::BindGroup> = self
            .color_uniforms
            .iter()
            .enumerate()
            .map(|(c, ub)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("molla::implicit_mpm u-color {c} bg")),
                    layout: &kernel.layout_uniforms,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    }],
                })
            })
            .collect();

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::implicit_mpm full-step enc"),
        });

        // reset_grid.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm reset pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_reset);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = self.g_total.div_ceil(WG_SIZE).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // p2g.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm p2g pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_p2g);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = particle_count.div_ceil(WG_PARTICLES).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // predict_grid.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm predict pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_predict);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = self.g_total.div_ceil(WG_SIZE).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // GS sweep — one dispatch per color per iteration.
        for _ in 0..gs_iters {
            for (c, group_len) in self.color_lens.iter().enumerate() {
                if *group_len == 0 {
                    continue;
                }
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::implicit_mpm gs pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&kernel.pipe_gs);
                pass.set_bind_group(0, &bg_gs_per_color[c], &[]);
                pass.set_bind_group(1, &bg_u_colors[c], &[]);
                let n = group_len.div_ceil(WG_SIZE).max(1);
                pass.dispatch_workgroups(n, 1, 1);
            }
        }
        // floor_reapply.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm floor pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_floor);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = self.g_total.div_ceil(WG_SIZE).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // g2p.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm g2p pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_g2p);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = particle_count.div_ceil(WG_PARTICLES).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        queue.submit(std::iter::once(enc.finish()));

        // Read back vel + C.
        let vel = readback_vec3_f32(device, queue, &self.p_vel, particle_count as usize)?;
        let c = readback_flat_f32(device, queue, &self.p_c, particle_count as usize * 9)?;
        Ok((vel, c))
    }

    /// **Stage 40 v3** — run the full P2G + predict + GS + floor + G2P
    /// chain *plus* the `bingham_advect` pass on the device. Reads back
    /// only `p_pos` and `p_vel` (the host doesn't need `p_C` for state
    /// reporting), eliminating the v2 per-step `p_C` readback.
    ///
    /// `gs_iters` and `cfg.use_off_diag` mirror the CPU step.
    /// `cfg.yield_stress` must be populated.
    #[allow(clippy::too_many_arguments)]
    pub fn run_full_step_native(
        &self,
        kernel: &ImplicitMpmKernel,
        particle_count: u32,
        cfg: &StepUniforms,
        gs_iters: u32,
    ) -> Result<(Vec<[f32; 3]>, Vec<[f32; 3]>)> {
        let device = kernel.device();
        let queue = kernel.queue();

        let u = Uniforms {
            grid_size: self.grid_size,
            color_n: 0,
            use_off_diag: u32::from(cfg.use_off_diag),
            particle_count,
            dt: cfg.dt,
            cell_size: cfg.cell_size,
            inv_cell_size: cfg.inv_cell_size,
            gravity_y: cfg.gravity_y,
            floor_y: cfg.floor_y,
            floor_friction: cfg.floor_friction,
            mass_p: cfg.mass_p,
            particle_volume: cfg.particle_volume,
            viscosity: cfg.viscosity,
            grid_origin_x: cfg.grid_origin[0],
            grid_origin_y: cfg.grid_origin[1],
            grid_origin_z: cfg.grid_origin[2],
            yield_stress: cfg.yield_stress,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        queue.write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        for (i, ucb) in self.color_uniforms.iter().enumerate() {
            let mut uc = u;
            uc.color_n = self.color_lens[i];
            queue.write_buffer(ucb, 0, bytemuck::bytes_of(&uc));
        }

        let bg_io = self.make_io_bind_group(kernel, None);
        let bg_gs_per_color: Vec<wgpu::BindGroup> = (0..self.color_bufs.len())
            .map(|c| self.make_io_bind_group(kernel, Some(c)))
            .collect();
        let bg_u_main = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::implicit_mpm u-main bg v3"),
            layout: &kernel.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniforms.as_entire_binding(),
            }],
        });
        let bg_u_colors: Vec<wgpu::BindGroup> = self
            .color_uniforms
            .iter()
            .enumerate()
            .map(|(c, ub)| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("molla::implicit_mpm u-color {c} bg v3")),
                    layout: &kernel.layout_uniforms,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    }],
                })
            })
            .collect();

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::implicit_mpm full-step-native enc"),
        });

        // reset_grid.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm v3 reset pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_reset);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = self.g_total.div_ceil(WG_SIZE).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // p2g.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm v3 p2g pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_p2g);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = particle_count.div_ceil(WG_PARTICLES).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // predict_grid.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm v3 predict pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_predict);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = self.g_total.div_ceil(WG_SIZE).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // GS sweep — one dispatch per color per iteration.
        for _ in 0..gs_iters {
            for (c, group_len) in self.color_lens.iter().enumerate() {
                if *group_len == 0 {
                    continue;
                }
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::implicit_mpm v3 gs pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&kernel.pipe_gs);
                pass.set_bind_group(0, &bg_gs_per_color[c], &[]);
                pass.set_bind_group(1, &bg_u_colors[c], &[]);
                let n = group_len.div_ceil(WG_SIZE).max(1);
                pass.dispatch_workgroups(n, 1, 1);
            }
        }
        // floor_reapply.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm v3 floor pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_floor);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = self.g_total.div_ceil(WG_SIZE).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // g2p.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm v3 g2p pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_g2p);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = particle_count.div_ceil(WG_PARTICLES).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }
        // bingham_advect — the v3 closing pass. Eliminates the CPU tail.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::implicit_mpm v3 bingham pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipe_bingham);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u_main, &[]);
            let n = particle_count.div_ceil(WG_PARTICLES).max(1);
            pass.dispatch_workgroups(n, 1, 1);
        }

        // Fused readback: queue both buffer→staging copies inside the
        // SAME command encoder as the compute passes so the
        // submit/wait pair runs once for the whole step. Use the
        // persistent staging buffers built at `GpuStepState::new` so
        // we don't allocate per-step.
        let n_p = particle_count as usize;
        let bytes = (n_p as u64 * 16).max(16);
        enc.copy_buffer_to_buffer(&self.p_pos, 0, &self.staging_pos, 0, bytes);
        enc.copy_buffer_to_buffer(&self.p_vel, 0, &self.staging_vel, 0, bytes);
        queue.submit(std::iter::once(enc.finish()));

        // Map both buffers, then poll once.
        let slice_pos = self.staging_pos.slice(0..bytes);
        let slice_vel = self.staging_vel.slice(0..bytes);
        let (tx_pos, rx_pos) = std::sync::mpsc::channel();
        let (tx_vel, rx_vel) = std::sync::mpsc::channel();
        slice_pos.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_pos.send(r); });
        slice_vel.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_vel.send(r); });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("implicit_mpm v3 poll: {e:?}")))?;
        rx_pos.recv()
            .map_err(|e| Error::Gpu(format!("implicit_mpm v3 recv pos: {e}")))?
            .map_err(|e| Error::Gpu(format!("implicit_mpm v3 map pos: {e:?}")))?;
        rx_vel.recv()
            .map_err(|e| Error::Gpu(format!("implicit_mpm v3 recv vel: {e}")))?
            .map_err(|e| Error::Gpu(format!("implicit_mpm v3 map vel: {e:?}")))?;
        let mapped_pos = slice_pos.get_mapped_range();
        let mapped_vel = slice_vel.get_mapped_range();
        let raw_pos: &[f32] = bytemuck::cast_slice(&mapped_pos);
        let raw_vel: &[f32] = bytemuck::cast_slice(&mapped_vel);
        let mut pos = Vec::with_capacity(n_p);
        let mut vel = Vec::with_capacity(n_p);
        for i in 0..n_p {
            pos.push([raw_pos[i * 4], raw_pos[i * 4 + 1], raw_pos[i * 4 + 2]]);
            vel.push([raw_vel[i * 4], raw_vel[i * 4 + 1], raw_vel[i * 4 + 2]]);
        }
        drop(mapped_pos);
        drop(mapped_vel);
        self.staging_pos.unmap();
        self.staging_vel.unmap();
        Ok((pos, vel))
    }

    /// Build the I/O bind group; `color_idx` selects which color
    /// group's `color_cells` buffer goes into binding 3. Pass `None`
    /// to use any color buffer (binding 3 is unused by reset / p2g /
    /// predict / floor / g2p, but wgpu still requires a valid binding).
    fn make_io_bind_group(
        &self,
        kernel: &ImplicitMpmKernel,
        color_idx: Option<usize>,
    ) -> wgpu::BindGroup {
        let device = kernel.device();
        let dummy_color = if self.color_bufs.is_empty() {
            None
        } else {
            let i = color_idx.unwrap_or(0).min(self.color_bufs.len() - 1);
            Some(&self.color_bufs[i])
        };
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::implicit_mpm color-dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let color = dummy_color.unwrap_or(&scratch);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::implicit_mpm io bg"),
            layout: &kernel.layout_io,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.g_mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.k_stencil.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.v_pred.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: color.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.v_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.g_momentum.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.p_pos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.p_vel.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.p_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.g_active.as_entire_binding() },
            ],
        })
    }
}

/// Plain-old-data view of `ImplicitMpmConfig` used by [`GpuStepState::run_full_step`].
/// All fields are `f32` since they wind up in the device-side uniform buffer.
#[derive(Copy, Clone, Debug)]
pub struct StepUniforms {
    pub dt: f32,
    pub cell_size: f32,
    pub inv_cell_size: f32,
    pub gravity_y: f32,
    pub floor_y: f32,
    pub floor_friction: f32,
    pub mass_p: f32,
    pub particle_volume: f32,
    pub viscosity: f32,
    pub grid_origin: [f32; 3],
    pub use_off_diag: bool,
    /// Bingham yield stress `τ_y` (Pa). Required by the v3
    /// `bingham_advect` device pass; unused by the v2 pipeline.
    pub yield_stress: f32,
}

impl ImplicitMpmKernel {
    /// **Legacy** — run only the inner colored-GS sweep. Kept so the
    /// Stage 40 v1 GS-only parity test still works without rewiring the
    /// caller to the full persistent-buffer path. The signature is the
    /// same as v1 (4 read-only buffers + `v_grid` read-write); the
    /// buffers must be f32-typed but the shader reads them via the
    /// `atomic<u32>` binding (bit-compatible).
    #[allow(clippy::too_many_arguments)]
    pub fn run_gs_sweeps(
        &self,
        grid_size: u32,
        dt: f32,
        use_off_diag: bool,
        max_iters: u32,
        color_groups: &[Vec<u32>],
        g_mass: &wgpu::Buffer,
        k_stencil: &wgpu::Buffer,
        v_pred: &wgpu::Buffer,
        v_grid: &wgpu::Buffer,
    ) -> Result<()> {
        if color_groups.is_empty() || max_iters == 0 {
            return Ok(());
        }
        let g_total = (grid_size * grid_size * grid_size) as u64;
        // Allocate dummies for bindings 5..9 (momentum / p_pos / p_vel /
        // p_C / g_active). They're sized minimally — the GS sweep never
        // touches them.
        let mk_dummy = |label: &str, bytes: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let dummy_mom = mk_dummy("molla::implicit_mpm dummy mom", g_total * 3 * 4);
        let dummy_pp = mk_dummy("molla::implicit_mpm dummy pp", 16);
        let dummy_pv = mk_dummy("molla::implicit_mpm dummy pv", 16);
        let dummy_pc = mk_dummy("molla::implicit_mpm dummy pc", 16);
        let dummy_active = mk_dummy("molla::implicit_mpm dummy active", g_total * 4);

        let mut color_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(color_groups.len());
        for (c, group) in color_groups.iter().enumerate() {
            let label = format!("molla::implicit_mpm color {c}");
            let bytes: Vec<u8> = if group.is_empty() {
                bytemuck::cast_slice::<u32, u8>(&[0u32]).to_vec()
            } else {
                bytemuck::cast_slice::<u32, u8>(group).to_vec()
            };
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, &bytes);
            color_bufs.push(buf);
        }
        let mut uniform_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(color_groups.len());
        for (c, group) in color_groups.iter().enumerate() {
            let label = format!("molla::implicit_mpm uniforms {c}");
            let u = Uniforms {
                grid_size,
                color_n: group.len() as u32,
                use_off_diag: u32::from(use_off_diag),
                particle_count: 0,
                dt,
                cell_size: 1.0,
                inv_cell_size: 1.0,
                gravity_y: 0.0,
                floor_y: 0.0,
                floor_friction: 0.0,
                mass_p: 0.0,
                particle_volume: 0.0,
                viscosity: 0.0,
                grid_origin_x: 0.0,
                grid_origin_y: 0.0,
                grid_origin_z: 0.0,
                yield_stress: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            };
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&u));
            uniform_bufs.push(buf);
        }
        for _ in 0..max_iters {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::implicit_mpm enc"),
            });
            for (c, group) in color_groups.iter().enumerate() {
                if group.is_empty() {
                    continue;
                }
                let bg_io = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("molla::implicit_mpm io bg legacy"),
                    layout: &self.layout_io,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: g_mass.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: k_stencil.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: v_pred.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: color_bufs[c].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: v_grid.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: dummy_mom.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: dummy_pp.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: dummy_pv.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: dummy_pc.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 9, resource: dummy_active.as_entire_binding() },
                    ],
                });
                let bg_u = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("molla::implicit_mpm u bg legacy"),
                    layout: &self.layout_uniforms,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_bufs[c].as_entire_binding(),
                    }],
                });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::implicit_mpm pass legacy"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipe_gs);
                pass.set_bind_group(0, &bg_io, &[]);
                pass.set_bind_group(1, &bg_u, &[]);
                let n_groups = (group.len() as u32).div_ceil(WG_SIZE).max(1);
                pass.dispatch_workgroups(n_groups, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
        }
        Ok(())
    }
}

/// Upload a host `Vec<f32>` (or `Vec<u32>` via bytemuck cast) as a
/// read-write storage buffer. Convenience helper for the parity test.
pub fn upload_storage_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[f32],
) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

/// Upload a host `Vec<vec3<f32>>` (laid out as `[x, y, z, 0]` to match
/// WGSL's `vec3<f32>` storage stride of 16 bytes).
pub fn upload_storage_vec3_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[[f32; 3]],
) -> wgpu::Buffer {
    let mut padded: Vec<f32> = Vec::with_capacity(data.len() * 4);
    for v in data {
        padded.push(v[0]);
        padded.push(v[1]);
        padded.push(v[2]);
        padded.push(0.0);
    }
    let bytes: &[u8] = bytemuck::cast_slice(&padded);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytes);
    buf
}

/// Read back a vec3<f32> storage buffer (with the same 16-byte stride
/// as [`upload_storage_vec3_f32`]) and strip the padding.
pub fn readback_vec3_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    n: usize,
) -> Result<Vec<[f32; 3]>> {
    let bytes = (n * 4 * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::implicit_mpm staging"),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::implicit_mpm readback enc"),
    });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes.max(16));
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("implicit_mpm read poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("implicit_mpm read recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("implicit_mpm read map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let raw: &[f32] = bytemuck::cast_slice(&mapped);
    let mut out: Vec<[f32; 3]> = Vec::with_capacity(n);
    for i in 0..n {
        out.push([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2]]);
    }
    drop(mapped);
    staging.unmap();
    Ok(out)
}

/// Convenience wrapper returning `Vec<[Real; 3]>` (Real = f64 on the
/// host). Kept for the existing Stage 40 v1 parity test.
pub fn readback_vec3_real(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    n: usize,
) -> Result<Vec<[Real; 3]>> {
    let raw = readback_vec3_f32(device, queue, src, n)?;
    Ok(raw
        .into_iter()
        .map(|v| [v[0] as Real, v[1] as Real, v[2] as Real])
        .collect())
}

/// Read back the entire `g_mass` (interpreted as f32 via u32 bitcast)
/// buffer from a `GpuStepState`. Diagnostic only.
pub fn readback_g_mass_f32(state: &GpuStepState, kernel: &ImplicitMpmKernel) -> Result<Vec<f32>> {
    let device = kernel.device();
    let queue = kernel.queue();
    let n = state.g_total as usize;
    let raw = readback_flat_f32(device, queue, &state.g_mass, n)?;
    Ok(raw)
}

/// Read back the entire `v_pred` buffer (vec3<f32>, 16-byte stride).
pub fn readback_v_pred_f32(state: &GpuStepState, kernel: &ImplicitMpmKernel) -> Result<Vec<[f32; 3]>> {
    let device = kernel.device();
    let queue = kernel.queue();
    let n = state.g_total as usize;
    readback_vec3_f32(device, queue, &state.v_pred, n)
}

/// Read back the entire `v_grid` buffer (post-GS).
pub fn readback_v_grid_f32(state: &GpuStepState, kernel: &ImplicitMpmKernel) -> Result<Vec<[f32; 3]>> {
    let device = kernel.device();
    let queue = kernel.queue();
    let n = state.g_total as usize;
    readback_vec3_f32(device, queue, &state.v_grid, n)
}

/// Read back the entire `g_momentum` buffer as flat f32 (3 entries / cell).
pub fn readback_g_momentum_f32(state: &GpuStepState, kernel: &ImplicitMpmKernel) -> Result<Vec<f32>> {
    let device = kernel.device();
    let queue = kernel.queue();
    let n = state.g_total as usize * 3;
    readback_flat_f32(device, queue, &state.g_momentum, n)
}

/// Read back a flat f32 storage buffer with no padding.
pub fn readback_flat_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    n: usize,
) -> Result<Vec<f32>> {
    let bytes = (n * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::implicit_mpm staging flat"),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::implicit_mpm readback enc flat"),
    });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes.max(16));
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("implicit_mpm read poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("implicit_mpm read recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("implicit_mpm read map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let raw: &[f32] = bytemuck::cast_slice(&mapped);
    let out = raw[..n].to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}

// ─── Bind-group entry helpers ───────────────────────────────────────

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
