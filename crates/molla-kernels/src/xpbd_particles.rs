//! Rust dispatch wrapper for the parallel particle XPBD pipeline in
//! `shaders/xpbd_particles.wgsl`. The kernel is split into seven entry
//! points (`predict`, `reset_lambdas`, `solve_distance`, `strain_zero`,
//! `strain_accumulate`, `strain_apply`, `floor_clamp`,
//! `velocity_recovery`); this wrapper binds the resources once per
//! substep and walks the encoder through the colour passes.
//!
//! Replaces the host-side `apply_particles_post_gpu` shim — particle
//! XPBD now runs end-to-end on the device.

use molla_core::{Error, Result};
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Model, State};
use molla_storage::{Buffer, BufferUsage};

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
    hash_buckets: u32,
    cell_inv_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const WORKGROUP_SIZE: u32 = 64;
/// Number of hash table buckets for self-collision. Power-of-two
/// makes the modulus cheap. Sized so cloth-grade scenes
/// (≤2 k particles) keep average bucket load ≪ MAX_PER_CELL.
const HASH_BUCKETS: u32 = 4096;
/// Per-bucket capacity. Must match the WGSL `MAX_PER_CELL` constant.
const MAX_PER_CELL: u32 = 12;

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
    pipe_shapes: wgpu::ComputePipeline,
    pipe_hash_clear: wgpu::ComputePipeline,
    pipe_hash_insert: wgpu::ComputePipeline,
    pipe_self_collide: wgpu::ComputePipeline,

    layout_state: wgpu::BindGroupLayout,
    layout_springs: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    layout_shapes: wgpu::BindGroupLayout,

    /// Per-spring λ accumulator (size `spring_count.max(1)`).
    lambdas: Buffer<Real>,
    /// Per-particle Jacobi delta scratch.
    deltas: Buffer<Vec3>,
    /// Per-particle Jacobi count scratch.
    counts: Buffer<u32>,
    /// 1-element placeholder buffers used as fallbacks for any
    /// bind group whose model arrays may be empty (no shapes / no
    /// bodies / no springs) — wgpu still requires a non-null binding.
    dummy_u32: Buffer<u32>,
    dummy_real: Buffer<Real>,
    dummy_vec3: Buffer<Vec3>,
    dummy_i32: Buffer<i32>,
    dummy_xform: Buffer<Transform>,

    /// Per-bucket atomic hit count for the spatial hash (size =
    /// `HASH_BUCKETS`). Cleared each substep by `hash_clear`.
    hash_count: Buffer<u32>,
    /// Flat `HASH_BUCKETS * MAX_PER_CELL` index list — bucket `b`
    /// owns slots `b * MAX_PER_CELL .. (b+1) * MAX_PER_CELL`.
    hash_indices: Buffer<u32>,
    /// Inverse cell size (1 / cell_size). 0 disables self-collision.
    cell_inv_size: f32,

    particle_count: u32,
    spring_count: u32,
}

impl XpbdParticlesKernel {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        spring_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::xpbd_particles"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/xpbd_particles.wgsl").into()),
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
        // Shapes + hash group: 5 read-only inputs (geom_kind,
        // geom_data, body, transform, body_q) + 2 read-write
        // (hash_count, hash_indices). Used by `particle_vs_shapes`
        // and the `hash_*` / `self_collide` passes; other entries
        // get the same layout bound but never touch the resources.
        let layout_shapes = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_particles shapes+hash"),
            entries: &mixed_entries(&[true, true, true, true, true, false, false]),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::xpbd_particles layout"),
            bind_group_layouts: &[
                &layout_state,
                &layout_springs,
                &layout_out,
                &layout_uniforms,
                &layout_shapes,
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
        let pipe_shapes = make_pipeline("particle_vs_shapes");
        let pipe_hash_clear = make_pipeline("hash_clear");
        let pipe_hash_insert = make_pipeline("hash_insert");
        let pipe_self_collide = make_pipeline("self_collide");

        // Scratch buffers. wgpu requires nonzero sizes, so bump zero
        // counts to 1 — the kernel guards against out-of-range indices.
        let pn = particle_count.max(1) as usize;
        let sn = spring_count.max(1) as usize;
        // Allocate + upload all scratch / dummy / hash buffers up front.
        // Failures here are unrecoverable (wgpu OOM at construction time)
        // and would otherwise surface as a confusing "missing device
        // mirror" error inside `dispatch`. `.expect()` ensures we crash
        // with a useful message at the actual failure site.
        let mut lambdas = Buffer::from_vec(vec![0.0_f64; sn], BufferUsage::StorageReadWrite);
        let mut deltas = Buffer::from_vec(vec![Vec3::ZERO; pn], BufferUsage::StorageReadWrite);
        let mut counts = Buffer::from_vec(vec![0u32; pn], BufferUsage::StorageReadWrite);
        lambdas
            .ensure_device(device)
            .expect("xpbd_particles: lambdas ensure_device");
        deltas
            .ensure_device(device)
            .expect("xpbd_particles: deltas ensure_device");
        counts
            .ensure_device(device)
            .expect("xpbd_particles: counts ensure_device");

        let mut dummy_u32 = Buffer::from_vec(vec![0u32; 2], BufferUsage::Storage);
        let mut dummy_real = Buffer::from_vec(vec![0.0_f64; 1], BufferUsage::Storage);
        let mut dummy_vec3 = Buffer::from_vec(vec![Vec3::ZERO; 1], BufferUsage::Storage);
        let mut dummy_i32 = Buffer::from_vec(vec![0i32; 1], BufferUsage::Storage);
        let mut dummy_xform = Buffer::from_vec(vec![Transform::IDENTITY; 1], BufferUsage::Storage);
        dummy_u32
            .ensure_device(device)
            .expect("xpbd_particles: dummy_u32 ensure_device");
        dummy_u32
            .upload(queue)
            .expect("xpbd_particles: dummy_u32 upload");
        dummy_real
            .ensure_device(device)
            .expect("xpbd_particles: dummy_real ensure_device");
        dummy_real
            .upload(queue)
            .expect("xpbd_particles: dummy_real upload");
        dummy_vec3
            .ensure_device(device)
            .expect("xpbd_particles: dummy_vec3 ensure_device");
        dummy_vec3
            .upload(queue)
            .expect("xpbd_particles: dummy_vec3 upload");
        dummy_i32
            .ensure_device(device)
            .expect("xpbd_particles: dummy_i32 ensure_device");
        dummy_i32
            .upload(queue)
            .expect("xpbd_particles: dummy_i32 upload");
        dummy_xform
            .ensure_device(device)
            .expect("xpbd_particles: dummy_xform ensure_device");
        dummy_xform
            .upload(queue)
            .expect("xpbd_particles: dummy_xform upload");

        // Spatial hash table for self-collision. 4096 buckets × 12
        // slots = 48 k u32 entries (192 KB) — comfortably small.
        // The bucket count is a constant of the kernel; bumping it
        // means a rebuild.
        let mut hash_count = Buffer::from_vec(
            vec![0u32; HASH_BUCKETS as usize],
            BufferUsage::StorageReadWrite,
        );
        let mut hash_indices = Buffer::from_vec(
            vec![0u32; (HASH_BUCKETS * MAX_PER_CELL) as usize],
            BufferUsage::StorageReadWrite,
        );
        hash_count
            .ensure_device(device)
            .expect("xpbd_particles: hash_count ensure_device");
        hash_count
            .upload(queue)
            .expect("xpbd_particles: hash_count upload");
        hash_indices
            .ensure_device(device)
            .expect("xpbd_particles: hash_indices ensure_device");
        hash_indices
            .upload(queue)
            .expect("xpbd_particles: hash_indices upload");

        Self {
            pipe_predict,
            pipe_reset_lambdas,
            pipe_solve_distance,
            pipe_strain_zero,
            pipe_strain_accumulate,
            pipe_strain_apply,
            pipe_floor,
            pipe_velocity,
            pipe_shapes,
            pipe_hash_clear,
            pipe_hash_insert,
            pipe_self_collide,
            layout_state,
            layout_springs,
            layout_out,
            layout_uniforms,
            layout_shapes,
            lambdas,
            deltas,
            counts,
            dummy_u32,
            dummy_real,
            dummy_vec3,
            dummy_i32,
            dummy_xform,
            hash_count,
            hash_indices,
            cell_inv_size: 0.0,
            particle_count,
            spring_count,
        }
    }

    /// Set the spatial-hash cell size for self-collision. Pass
    /// `2 * particle_radius * 1.1` for cloth grids; `0` disables
    /// self-collision (default).
    pub fn set_self_collision_cell_size(&mut self, cell_size: f32) {
        self.cell_inv_size = if cell_size > 0.0 {
            1.0 / cell_size
        } else {
            0.0
        };
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
        // Springs bind group — falls back to dummy buffers when the
        // model has zero springs so wgpu doesn't see an unbound slot.
        let dummy_u32_buf = self
            .dummy_u32
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_u32 missing".into()))?;
        let dummy_real_buf = self
            .dummy_real
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_real missing".into()))?;
        let sa_buf = model.spring_a.device_buffer().unwrap_or(dummy_u32_buf);
        let sb_buf = model.spring_b.device_buffer().unwrap_or(dummy_u32_buf);
        let rest_buf = model
            .spring_rest_length
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        let comp_buf = model
            .spring_compliance
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        let stretch_buf = model
            .spring_max_stretch
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        let color_off_buf = model
            .spring_color_offsets
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let color_idx_buf = model
            .spring_indices_by_color
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let bg_springs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles springs bg"),
            layout: &self.layout_springs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sa_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rest_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: comp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: stretch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: color_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: color_idx_buf.as_entire_binding(),
                },
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
        // Shapes bind group — used by `particle_vs_shapes`. Bound
        // for every dispatch (cheap; layout is fixed) so the
        // pipeline's group-4 expectation is always satisfied.
        // Empty model arrays (no shapes / no bodies) fall back to
        // 1-element dummy buffers so wgpu doesn't trip on zero-sized
        // bindings.
        let dummy_vec3_buf = self
            .dummy_vec3
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_vec3 missing".into()))?;
        let dummy_i32_buf = self
            .dummy_i32
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_i32 missing".into()))?;
        let dummy_xform_buf = self
            .dummy_xform
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_xform missing".into()))?;
        let kind_buf = model
            .shape_geom_kind
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let data_buf = model
            .shape_geom_data
            .device_buffer()
            .unwrap_or(dummy_vec3_buf);
        let body_buf = model.shape_body.device_buffer().unwrap_or(dummy_i32_buf);
        let xform_buf = model
            .shape_transform
            .device_buffer()
            .unwrap_or(dummy_xform_buf);
        let body_q_buf = state_in.body_q.device_buffer().unwrap_or(dummy_xform_buf);
        let hash_count_buf = self
            .hash_count
            .device_buffer()
            .ok_or_else(|| Error::Gpu("hash_count missing".into()))?;
        let hash_indices_buf = self
            .hash_indices
            .device_buffer()
            .ok_or_else(|| Error::Gpu("hash_indices missing".into()))?;
        let bg_shapes = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_particles shapes+hash bg"),
            layout: &self.layout_shapes,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: kind_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: body_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: xform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: body_q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: hash_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: hash_indices_buf.as_entire_binding(),
                },
            ],
        });
        let shape_count_u = model.shape_count as u32;
        let self_collide_enabled = self.cell_inv_size > 0.0;

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

        let cell_inv_size = self.cell_inv_size;
        let base_uniforms = |work_count: u32, color_index: u32| Uniforms {
            particle_count: particle_count_u,
            spring_count: spring_count_u,
            iterations,
            dt: dt as f32,
            linear_damping: model.linear_damping as f32,
            color_index,
            work_count,
            hash_buckets: HASH_BUCKETS,
            cell_inv_size,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        // Build the uniform list in dispatch order so we can index by
        // `bg_uniforms[i]` while encoding.
        push_uniforms(&base_uniforms(particle_count_u, 0)); // 0: predict
        push_uniforms(&base_uniforms(spring_count_u, 0)); // 1: reset_lambdas

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
        // particle_vs_shapes — uses `color_index` field as
        // `shape_count` so the kernel doesn't need a separate
        // uniform field. work_count = particle_count.
        push_uniforms(&base_uniforms(particle_count_u, shape_count_u));
        // Self-collision pipeline: hash_clear (work=HASH_BUCKETS) →
        // hash_insert (work=particle_count) → self_collide
        // (work=particle_count) → strain_apply (work=particle_count).
        push_uniforms(&base_uniforms(HASH_BUCKETS, 0));
        push_uniforms(&base_uniforms(particle_count_u, 0));
        push_uniforms(&base_uniforms(particle_count_u, 0));
        push_uniforms(&base_uniforms(particle_count_u, 0));
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
            pass.set_bind_group(4, &bg_shapes, &[]);

            let dispatch_kernel = |pass: &mut wgpu::ComputePass<'_>,
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
                    dispatch_kernel(&mut pass, &self.pipe_strain_accumulate, idx, nc);
                    idx += 1;
                }
                dispatch_kernel(&mut pass, &self.pipe_strain_apply, idx, particle_count_u);
                idx += 1;
            }
            // Floor (Y=0 fallback — kept so scenes without an
            // explicit Y-plane still don't drop their cloth into
            // the void).
            dispatch_kernel(&mut pass, &self.pipe_floor, idx, particle_count_u);
            idx += 1;
            // Particle-vs-rigid: project particles out of every
            // collision shape in the model. `shape_count` is
            // smuggled via the `color_index` uniform field.
            if shape_count_u > 0 {
                dispatch_kernel(&mut pass, &self.pipe_shapes, idx, particle_count_u);
            }
            idx += 1;
            // Self-collision: hash_clear → hash_insert →
            // self_collide → strain_apply (reused — applies the
            // averaged correction stored in `deltas`/`counts`).
            // Disabled when cell_inv_size = 0.
            if self_collide_enabled {
                dispatch_kernel(&mut pass, &self.pipe_hash_clear, idx, HASH_BUCKETS);
                idx += 1;
                dispatch_kernel(&mut pass, &self.pipe_hash_insert, idx, particle_count_u);
                idx += 1;
                dispatch_kernel(&mut pass, &self.pipe_self_collide, idx, particle_count_u);
                idx += 1;
                dispatch_kernel(&mut pass, &self.pipe_strain_apply, idx, particle_count_u);
                idx += 1;
            } else {
                idx += 4;
            }
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
        // strain_iters * (zero + K + apply) + floor + shapes +
        // self-collide(4) + velocity.
        let strain_iters = iterations * 4;
        let dist = iterations * color_count;
        let strain = strain_iters * (color_count + 2);
        (2 + dist + strain + 2 + 4 + 1) as usize
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
        Error::Gpu(format!(
            "xpbd_particles binding {binding}: no device mirror"
        ))
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

/// Like `storage_entries` but allows per-binding read/write flags
/// — needed when one bind group mixes read-only model arrays with
/// read-write scratch buffers (e.g. shapes + spatial-hash table).
fn mixed_entries(read_only: &[bool]) -> Vec<wgpu::BindGroupLayoutEntry> {
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
