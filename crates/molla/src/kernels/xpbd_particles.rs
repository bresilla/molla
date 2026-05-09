//! Rust dispatch wrapper for the parallel particle XPBD pipeline in
//! `shaders/xpbd_particles.wgsl`. The kernel is split into seven entry
//! points (`predict`, `reset_lambdas`, `solve_distance`, `strain_zero`,
//! `strain_accumulate`, `strain_apply`, `floor_clamp`,
//! `velocity_recovery`); this wrapper binds the resources once per
//! substep and walks the encoder through the colour passes.
//!
//! Replaces the host-side `apply_particles_post_gpu` shim — particle
//! XPBD now runs end-to-end on the device.

use crate::core::{Error, Result};
use crate::math::{Real, Vec3};
use crate::sim::{Model, State};
use crate::storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    particle_count: u32,
    spring_count: u32,
    iterations: u32,
    dt: f32,
    linear_damping: f32,
    color_index: u32,
    work_count: u32,
    _pad: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Owns the seven compute pipelines (all share one shader module) and
/// the per-substep scratch buffers (λ accumulator + Jacobi
/// deltas/counts). Sized for a specific `(particle_count,
/// spring_count)` — rebuild if either changes.
pub struct XpbdParticlesKernel {
    pipe_predict: wgpu::ComputePipeline,
    pipe_reset_lambdas: wgpu::ComputePipeline,
    pipe_solve_distance: wgpu::ComputePipeline,
    pipe_strain_zero: wgpu::ComputePipeline,
    pipe_strain_accumulate: wgpu::ComputePipeline,
    pipe_strain_apply: wgpu::ComputePipeline,
    pipe_floor: wgpu::ComputePipeline,
    pipe_velocity: wgpu::ComputePipeline,

    layout_state: wgpu::BindGroupLayout,
    layout_springs: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,

    /// Per-spring λ accumulator (size `spring_count.max(1)`).
    lambdas: Buffer<Real>,
    /// Per-particle Jacobi delta scratch.
    deltas: Buffer<Vec3>,
    /// Per-particle Jacobi count scratch.
    counts: Buffer<u32>,

    particle_count: u32,
    spring_count: u32,
}

impl XpbdParticlesKernel {
    pub fn new(
        device: &wgpu::Device,
        particle_count: u32,
        spring_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::xpbd_particles"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/xpbd_particles.wgsl").into(),
            ),
        });

        // State group: 6 read-only inputs.
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles state"),
            entries: &storage_entries(6, true),
        });
        // Springs group: 7 read-only inputs (a, b, rest, comp, max_stretch,
        // color_offsets, indices_by_color).
        let layout_springs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles springs"),
            entries: &storage_entries(7, true),
        });
        // Output / scratch group: 5 read-write (q, qd, lambdas, deltas, counts).
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles out"),
            entries: &storage_entries(5, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles uniforms"),
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
            label: Some("molla::xpbd_particles layout"),
            bind_group_layouts: &[
                &layout_state,
                &layout_springs,
                &layout_out,
                &layout_uniforms,
            ],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::xpbd_particles"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pipe_predict = make_pipeline("predict");
        let pipe_reset_lambdas = make_pipeline("reset_lambdas");
        let pipe_solve_distance = make_pipeline("solve_distance");
        let pipe_strain_zero = make_pipeline("strain_zero");
        let pipe_strain_accumulate = make_pipeline("strain_accumulate");
        let pipe_strain_apply = make_pipeline("strain_apply");
        let pipe_floor = make_pipeline("floor_clamp");
        let pipe_velocity = make_pipeline("velocity_recovery");

        // Scratch buffers. wgpu requires nonzero sizes, so bump zero
        // counts to 1 — the kernel guards against out-of-range indices.
        let pn = particle_count.max(1) as usize;
        let sn = spring_count.max(1) as usize;
        let mut lambdas = Buffer::from_vec(vec![0.0_f64; sn], BufferUsage::StorageReadWrite);
        let mut deltas = Buffer::from_vec(vec![Vec3::ZERO; pn], BufferUsage::StorageReadWrite);
        let mut counts = Buffer::from_vec(vec![0u32; pn], BufferUsage::StorageReadWrite);
        let _ = lambdas.ensure_device(device);
        let _ = deltas.ensure_device(device);
        let _ = counts.ensure_device(device);

        Self {
            pipe_predict,
            pipe_reset_lambdas,
            pipe_solve_distance,
            pipe_strain_zero,
            pipe_strain_accumulate,
            pipe_strain_apply,
            pipe_floor,
            pipe_velocity,
            layout_state,
            layout_springs,
            layout_out,
            layout_uniforms,
            lambdas,
            deltas,
            counts,
            particle_count,
            spring_count,
        }
    }

    /// Run one particle substep: predict → λ-reset → distance solve
    /// (`iterations` × K colour passes) → strain limit
    /// (`iterations*4` × K colour passes) → floor → velocity recovery.
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        dt: Real,
        iterations: u32,
    ) -> Result<()> {
        if model.particle_count == 0 {
            return Ok(());
        }

        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, state_in.particle_q.device_buffer())?,
                buffer_entry(1, state_in.particle_qd.device_buffer())?,
                buffer_entry(2, model.particle_inv_mass.device_buffer())?,
                buffer_entry(3, model.particle_radius.device_buffer())?,
                buffer_entry(4, model.particle_world.device_buffer())?,
                buffer_entry(5, model.gravity.device_buffer())?,
            ],
        });
        let bg_springs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles springs bg"),
            layout: &self.layout_springs,
            entries: &[
                buffer_entry(0, model.spring_a.device_buffer())?,
                buffer_entry(1, model.spring_b.device_buffer())?,
                buffer_entry(2, model.spring_rest_length.device_buffer())?,
                buffer_entry(3, model.spring_compliance.device_buffer())?,
                buffer_entry(4, model.spring_max_stretch.device_buffer())?,
                buffer_entry(5, model.spring_color_offsets.device_buffer())?,
                buffer_entry(6, model.spring_indices_by_color.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, state_out.particle_q.device_buffer())?,
                buffer_entry(1, state_out.particle_qd.device_buffer())?,
                buffer_entry(2, self.lambdas.device_buffer())?,
                buffer_entry(3, self.deltas.device_buffer())?,
                buffer_entry(4, self.counts.device_buffer())?,
            ],
        });

        let particle_count_u = model.particle_count as u32;
        let spring_count_u = model.spring_count as u32;
        let color_count = model.spring_color_count as u32;
        let color_offsets_host = model.spring_color_offsets.host()?.to_vec();

        // We need `2 + iterations + iterations*4` distinct uniform
        // buffers worst case (each colour-pass dispatch picks up a
        // different `color_index` / `work_count`). Allocate them
        // up-front; small (32 B each) and fixed-count.
        let uniforms_per_step = self.uniform_count(iterations, color_count);
        let mut uniform_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(uniforms_per_step);
        let mut bg_uniforms: Vec<wgpu::BindGroup> = Vec::with_capacity(uniforms_per_step);

        let mut push_uniforms = |u: &Uniforms| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::xpbd_particles uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(u));
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::xpbd_particles uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            uniform_bufs.push(buf);
            bg_uniforms.push(bg);
        };

        let base_uniforms = |work_count: u32, color_index: u32| Uniforms {
            particle_count: particle_count_u,
            spring_count: spring_count_u,
            iterations,
            dt: dt as f32,
            linear_damping: model.linear_damping as f32,
            color_index,
            work_count,
            _pad: 0,
        };

        // Build the uniform list in dispatch order so we can index by
        // `bg_uniforms[i]` while encoding.
        push_uniforms(&base_uniforms(particle_count_u, 0));      // 0: predict
        push_uniforms(&base_uniforms(spring_count_u, 0));        // 1: reset_lambdas

        // Distance constraint: iterations × color_count colour passes.
        for _ in 0..iterations {
            for k in 0..color_count {
                let lo = color_offsets_host[k as usize];
                let hi = color_offsets_host[k as usize + 1];
                let nc = hi - lo;
                push_uniforms(&base_uniforms(nc, k));
            }
        }

        // Strain limit: (iterations*4) Jacobi rounds; each round is
        // [zero] [accumulate × K] [apply].
        let strain_iters = iterations * 4;
        for _ in 0..strain_iters {
            push_uniforms(&base_uniforms(particle_count_u, 0)); // strain_zero
            for k in 0..color_count {
                let lo = color_offsets_host[k as usize];
                let hi = color_offsets_host[k as usize + 1];
                let nc = hi - lo;
                push_uniforms(&base_uniforms(nc, k));
            }
            push_uniforms(&base_uniforms(particle_count_u, 0)); // strain_apply
        }

        push_uniforms(&base_uniforms(particle_count_u, 0)); // floor
        push_uniforms(&base_uniforms(particle_count_u, 0)); // velocity recovery

        // ── Encode every dispatch in a single pass. ───────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::xpbd_particles encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd_particles"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_springs, &[]);
            pass.set_bind_group(2, &bg_out, &[]);

            let dispatch_kernel =
                |pass: &mut wgpu::ComputePass<'_>,
                 pipeline: &wgpu::ComputePipeline,
                 uniform_idx: usize,
                 work: u32| {
                    if work == 0 {
                        return;
                    }
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(3, &bg_uniforms[uniform_idx], &[]);
                    let groups = work.div_ceil(WORKGROUP_SIZE);
                    pass.dispatch_workgroups(groups.max(1), 1, 1);
                };

            let mut idx = 0usize;
            // Predict.
            dispatch_kernel(&mut pass, &self.pipe_predict, idx, particle_count_u);
            idx += 1;
            // Reset λ.
            if spring_count_u > 0 {
                dispatch_kernel(&mut pass, &self.pipe_reset_lambdas, idx, spring_count_u);
            }
            idx += 1;
            // Distance constraint.
            for _ in 0..iterations {
                for k in 0..color_count {
                    let lo = color_offsets_host[k as usize];
                    let hi = color_offsets_host[k as usize + 1];
                    let nc = hi - lo;
                    dispatch_kernel(&mut pass, &self.pipe_solve_distance, idx, nc);
                    idx += 1;
                }
            }
            // Strain limit.
            for _ in 0..strain_iters {
                dispatch_kernel(&mut pass, &self.pipe_strain_zero, idx, particle_count_u);
                idx += 1;
                for k in 0..color_count {
                    let lo = color_offsets_host[k as usize];
                    let hi = color_offsets_host[k as usize + 1];
                    let nc = hi - lo;
                    dispatch_kernel(
                        &mut pass,
                        &self.pipe_strain_accumulate,
                        idx,
                        nc,
                    );
                    idx += 1;
                }
                dispatch_kernel(&mut pass, &self.pipe_strain_apply, idx, particle_count_u);
                idx += 1;
            }
            // Floor.
            dispatch_kernel(&mut pass, &self.pipe_floor, idx, particle_count_u);
            idx += 1;
            // Velocity recovery.
            dispatch_kernel(&mut pass, &self.pipe_velocity, idx, particle_count_u);
            idx += 1;
            debug_assert_eq!(idx, uniform_bufs.len());
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.particle_q.mark_device_dirty();
        state_out.particle_qd.mark_device_dirty();
        Ok(())
    }

    fn uniform_count(&self, iterations: u32, color_count: u32) -> usize {
        // predict + reset_lambdas + iterations*K (distance) +
        // strain_iters * (zero + K + apply) + floor + velocity.
        let strain_iters = iterations * 4;
        let dist = iterations * color_count;
        let strain = strain_iters * (color_count + 2);
        (2 + dist + strain + 2) as usize
    }

    /// Number of particles the kernel was built for.
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
    /// Number of springs the kernel was built for.
    pub fn spring_count(&self) -> u32 {
        self.spring_count
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!("xpbd_particles binding {binding}: no device mirror"))
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
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
