//! Rust dispatch wrapper for the cloth Vertex Block Descent (VBD)
//! solver in `shaders/vbd_cloth.wgsl`. The kernel is split into four
//! entry points: `predict`, `solve_color`, `apply_shape_contacts`,
//! `velocity_recovery`. Mirrors `XpbdParticlesKernel` (graph-coloured
//! per-pass dispatch, scratch buffers for the per-substep state).
//!
//! See `solvers/vbd.rs` for the algorithm description.

use crate::core::{Error, Result};
use crate::math::{Real, Vec3};
use crate::sim::{Model, State};
use crate::storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    particle_count: u32,
    spring_count: u32,
    color_count: u32,
    color_index: u32,
    work_count: u32,
    dt: f32,
    linear_damping: f32,
    contact_stiffness: f32,
    _pad0: f32,
    _pad1: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Owns the VBD pipelines + per-substep scratch buffers.
/// Sized for a specific `(particle_count, spring_count)` — rebuild
/// the kernel if either changes.
pub struct VbdClothKernel {
    pipe_predict: wgpu::ComputePipeline,
    pipe_solve_color: wgpu::ComputePipeline,
    pipe_shape_contacts: wgpu::ComputePipeline,
    pipe_velocity: wgpu::ComputePipeline,

    layout_state: wgpu::BindGroupLayout,
    layout_topo: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    layout_shapes: wgpu::BindGroupLayout,

    /// Per-spring stiffness (N/m), derived from XPBD compliance at
    /// upload time as `k = 1 / max(α, ε)` and then capped to the
    /// `max_stiffness` solver knob. Owned by the kernel because the
    /// compliance lives in the `Model`; converting once on the host
    /// avoids a per-step conversion in WGSL.
    spring_stiffness: Buffer<Real>,
    /// Per-particle previous-step position scratch.
    particle_q_prev: Buffer<Vec3>,
    /// Per-particle inertial target `y` scratch.
    inertia_y: Buffer<Vec3>,

    /// 1-element placeholder buffers for any bind group whose model
    /// arrays may be empty (no springs / no shapes).
    dummy_u32: Buffer<u32>,
    dummy_real: Buffer<Real>,
    dummy_vec3: Buffer<Vec3>,
    dummy_i32: Buffer<i32>,
    dummy_xform: Buffer<crate::math::Transform>,

    particle_count: u32,
    spring_count: u32,
}

impl VbdClothKernel {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        spring_count: u32,
        per_spring_stiffness: &[Real],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::vbd_cloth"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/vbd_cloth.wgsl").into(),
            ),
        });

        // State group: 6 read-only inputs.
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth state"),
            entries: &storage_entries(6, true),
        });
        // Topology group: 8 read-only inputs.
        let layout_topo = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth topo"),
            entries: &storage_entries(8, true),
        });
        // Output / scratch group: 4 read-write.
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth out"),
            entries: &storage_entries(4, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth uniforms"),
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
        // Shapes group: 5 read-only (kind, data, body, transform, body_q).
        let layout_shapes = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth shapes"),
            entries: &storage_entries(5, true),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::vbd_cloth layout"),
            bind_group_layouts: &[
                &layout_state,
                &layout_topo,
                &layout_out,
                &layout_uniforms,
                &layout_shapes,
            ],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::vbd_cloth"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pipe_predict = make_pipeline("predict");
        let pipe_solve_color = make_pipeline("solve_color");
        let pipe_shape_contacts = make_pipeline("apply_shape_contacts");
        let pipe_velocity = make_pipeline("velocity_recovery");

        let pn = particle_count.max(1) as usize;
        let sn = spring_count.max(1) as usize;
        let stiffness_init = if per_spring_stiffness.len() == sn || per_spring_stiffness.len() == spring_count as usize {
            let mut v = vec![0.0_f64; sn];
            for (dst, src) in v.iter_mut().zip(per_spring_stiffness.iter()) {
                *dst = *src;
            }
            v
        } else {
            vec![0.0_f64; sn]
        };
        let mut spring_stiffness = Buffer::from_vec(stiffness_init, BufferUsage::Storage);
        let mut particle_q_prev =
            Buffer::from_vec(vec![Vec3::ZERO; pn], BufferUsage::StorageReadWrite);
        let mut inertia_y =
            Buffer::from_vec(vec![Vec3::ZERO; pn], BufferUsage::StorageReadWrite);
        let _ = spring_stiffness.ensure_device(device);
        let _ = spring_stiffness.upload(queue);
        let _ = particle_q_prev.ensure_device(device);
        let _ = particle_q_prev.upload(queue);
        let _ = inertia_y.ensure_device(device);
        let _ = inertia_y.upload(queue);

        let mut dummy_u32 = Buffer::from_vec(vec![0u32; 2], BufferUsage::Storage);
        let mut dummy_real = Buffer::from_vec(vec![0.0_f64; 1], BufferUsage::Storage);
        let mut dummy_vec3 = Buffer::from_vec(vec![Vec3::ZERO; 1], BufferUsage::Storage);
        let mut dummy_i32 = Buffer::from_vec(vec![0i32; 1], BufferUsage::Storage);
        let mut dummy_xform = Buffer::from_vec(
            vec![crate::math::Transform::IDENTITY; 1],
            BufferUsage::Storage,
        );
        let _ = dummy_u32.ensure_device(device);
        let _ = dummy_u32.upload(queue);
        let _ = dummy_real.ensure_device(device);
        let _ = dummy_real.upload(queue);
        let _ = dummy_vec3.ensure_device(device);
        let _ = dummy_vec3.upload(queue);
        let _ = dummy_i32.ensure_device(device);
        let _ = dummy_i32.upload(queue);
        let _ = dummy_xform.ensure_device(device);
        let _ = dummy_xform.upload(queue);

        Self {
            pipe_predict,
            pipe_solve_color,
            pipe_shape_contacts,
            pipe_velocity,
            layout_state,
            layout_topo,
            layout_out,
            layout_uniforms,
            layout_shapes,
            spring_stiffness,
            particle_q_prev,
            inertia_y,
            dummy_u32,
            dummy_real,
            dummy_vec3,
            dummy_i32,
            dummy_xform,
            particle_count,
            spring_count,
        }
    }

    /// Push a fresh per-spring stiffness table (host-side) and re-upload
    /// to the device. Useful when the user tweaks stiffness sliders in
    /// the viewer without rebuilding the model.
    pub fn set_spring_stiffness(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        per_spring_stiffness: &[Real],
    ) -> Result<()> {
        let sn = self.spring_count.max(1) as usize;
        let host = self.spring_stiffness.host_mut()?;
        for (dst, src) in host.iter_mut().zip(per_spring_stiffness.iter().chain(std::iter::repeat(&0.0_f64))).take(sn) {
            *dst = *src;
        }
        self.spring_stiffness.ensure_device(device)?;
        self.spring_stiffness.upload(queue)?;
        Ok(())
    }

    /// One VBD substep: predict → (iters × colours × `solve_color`)
    /// → `apply_shape_contacts` → `velocity_recovery`.
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        dt: Real,
        iterations: u32,
        contact_stiffness: f32,
    ) -> Result<()> {
        if model.particle_count == 0 {
            return Ok(());
        }
        let particle_count_u = model.particle_count as u32;
        let spring_count_u = model.spring_count as u32;
        let shape_count_u = model.shape_count as u32;
        let color_count = model.particle_color_count as u32;

        // ── Bind groups ───────────────────────────────────────────
        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth state bg"),
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
        let stiff_buf = self
            .spring_stiffness
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        let vadj_off_buf = model
            .vertex_adjacent_springs_offsets
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let vadj_buf = model
            .vertex_adjacent_springs
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let pcol_off_buf = model
            .particle_color_offsets
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let pcol_idx_buf = model
            .particle_indices_by_color
            .device_buffer()
            .unwrap_or(dummy_u32_buf);

        let bg_topo = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth topo bg"),
            layout: &self.layout_topo,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sa_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sb_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: rest_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: stiff_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: vadj_off_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: vadj_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: pcol_off_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: pcol_idx_buf.as_entire_binding() },
            ],
        });

        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, state_out.particle_q.device_buffer())?,
                buffer_entry(1, state_out.particle_qd.device_buffer())?,
                buffer_entry(2, self.particle_q_prev.device_buffer())?,
                buffer_entry(3, self.inertia_y.device_buffer())?,
            ],
        });

        // Shapes bind group.
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
        let bg_shapes = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth shapes bg"),
            layout: &self.layout_shapes,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: kind_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: body_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: xform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: body_q_buf.as_entire_binding() },
            ],
        });

        // ── Uniforms (one per dispatch) ──────────────────────────
        let color_offsets_host = model.particle_color_offsets.host()?.to_vec();

        let mut uniform_bufs: Vec<wgpu::Buffer> = Vec::new();
        let mut bg_uniforms: Vec<wgpu::BindGroup> = Vec::new();
        let mut push_uniforms = |u: &Uniforms| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::vbd_cloth uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(u));
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::vbd_cloth uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            uniform_bufs.push(buf);
            bg_uniforms.push(bg);
        };

        let base = |work_count: u32, color_index: u32, contact_k: f32| Uniforms {
            particle_count: particle_count_u,
            spring_count: spring_count_u,
            color_count,
            color_index,
            work_count,
            dt: dt as f32,
            linear_damping: model.linear_damping as f32,
            contact_stiffness: contact_k,
            _pad0: 0.0,
            _pad1: 0.0,
        };

        // Predict.
        push_uniforms(&base(particle_count_u, 0, 0.0));
        // iterations × colours × solve_color. Contact penalty lives
        // INSIDE `solve_color` so the per-vertex Newton block sees
        // the contact force/Hessian — applying it as a separate
        // post-pass leaks contact energy directly into velocity.
        for _ in 0..iterations {
            for k in 0..color_count {
                let lo = color_offsets_host[k as usize];
                let hi = color_offsets_host[k as usize + 1];
                push_uniforms(&base(hi - lo, k, contact_stiffness));
            }
        }
        // Final non-penetration projection — runs once at the end
        // of the substep with a *zero* contact stiffness (so it
        // contributes no force-into-velocity), then snaps any
        // remaining penetration depth out hard. Acts as a safety
        // net for stiffness pairs the Newton block under-converges.
        let _ = shape_count_u; // referenced in the apply_shape_contacts
                               // dispatch path; kept for future use.
        // Velocity recovery.
        push_uniforms(&base(particle_count_u, 0, 0.0));

        // ── Encode + dispatch ─────────────────────────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::vbd_cloth encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::vbd_cloth"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_topo, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(4, &bg_shapes, &[]);

            let dispatch =
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
            dispatch(&mut pass, &self.pipe_predict, idx, particle_count_u);
            idx += 1;
            // Outer iterations × colours. Soft rigid contacts are
            // baked into `solve_color` so each per-vertex Newton
            // block sees the contact force + Hessian directly.
            for _ in 0..iterations {
                for k in 0..color_count {
                    let lo = color_offsets_host[k as usize];
                    let hi = color_offsets_host[k as usize + 1];
                    let nc = hi - lo;
                    dispatch(&mut pass, &self.pipe_solve_color, idx, nc);
                    idx += 1;
                }
            }
            // Velocity recovery.
            dispatch(&mut pass, &self.pipe_velocity, idx, particle_count_u);
            idx += 1;
            // `pipe_shape_contacts` kept compiled for future use
            // (warm-start initialiser, hard-mode AVBD); silence the
            // unused-field warning.
            let _ = &self.pipe_shape_contacts;
            debug_assert_eq!(idx, uniform_bufs.len());
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.particle_q.mark_device_dirty();
        state_out.particle_qd.mark_device_dirty();
        Ok(())
    }

    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
    pub fn spring_count(&self) -> u32 {
        self.spring_count
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!("vbd_cloth binding {binding}: no device mirror"))
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
