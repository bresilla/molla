//! Rust dispatch wrapper around `shaders/xpbd.wgsl`.
//!
//! Pipeline:
//! 1. `predict` — semi-implicit prediction of positions / velocities.
//! 2. N iterations of `project_position` — each iteration ping-pongs
//!    between two scratch buffers; the host alternates which one is
//!    bound as "current" via two pre-built bind groups.
//! 3. `recover_velocity` — derives `v_post` from the final
//!    `q_pred` via `(q_pred - q_in) / dt`.
//! 4. M iterations of `project_velocity` — applies restitution +
//!    friction. Mutates `v_post` in place each iteration.
//! 5. `write_final` — copies into the user's `state_out.body_qd`,
//!    preserves angular component.
//!
//! Per-body parallelism (one thread per body, processes its own
//! contacts via per-body CSR) avoids the WGSL `atomic<f32>` gap.

use molla_core::{Error, Result};
use molla_sim::{Contacts, Control, Model, State};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dt: f32,
    angular_damping: f32,
    body_count: u32,
    has_external_force: u32,
    linear_damping: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Compute pipelines + per-solver scratch buffers for GPU XPBD.
pub struct XpbdKernel {
    pipeline_predict: wgpu::ComputePipeline,
    pipeline_project_pos: wgpu::ComputePipeline,
    pipeline_recover_v: wgpu::ComputePipeline,
    pipeline_project_v: wgpu::ComputePipeline,
    pipeline_write_final: wgpu::ComputePipeline,

    layout_model: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_control: wgpu::BindGroupLayout,
    layout_contacts: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,

    /// Scratch buffers sized for `body_count`. Owned by the solver
    /// so we don't reallocate per step.
    body_q_pred: wgpu::Buffer,
    body_v_pred: wgpu::Buffer,
    body_q_alt: wgpu::Buffer,
    body_v_post: wgpu::Buffer,
    body_w_pred: wgpu::Buffer,
    body_w_post: wgpu::Buffer,
    /// 1-element dummy for empty `Control::body_external_f`.
    external_f_dummy: wgpu::Buffer,

    body_count: u32,
}

impl XpbdKernel {
    pub fn new(device: &wgpu::Device, body_count: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::xpbd"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/xpbd.wgsl").into()),
        });

        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd model"),
            entries: &storage_entries(6, true),
        });
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd state"),
            entries: &xpbd_state_entries(),
        });
        let layout_control = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd control"),
            entries: &storage_entries(1, true),
        });
        let layout_contacts = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd contacts"),
            entries: &storage_entries(11, true),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd uniforms"),
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
            label: Some("molla::xpbd layout"),
            bind_group_layouts: &[
                &layout_model,
                &layout_state,
                &layout_control,
                &layout_contacts,
                &layout_uniforms,
            ],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str, label: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let pipeline_predict = make_pipeline("predict", "molla::xpbd::predict");
        let pipeline_project_pos =
            make_pipeline("project_position", "molla::xpbd::project_position");
        let pipeline_recover_v = make_pipeline("recover_velocity", "molla::xpbd::recover_velocity");
        let pipeline_project_v = make_pipeline("project_velocity", "molla::xpbd::project_velocity");
        let pipeline_write_final = make_pipeline("write_final", "molla::xpbd::write_final");

        // Scratch buffers — sized for `body_count` of the device-side
        // mirror types. `Transform` mirror is 32 bytes, `vec4<f32>` is
        // 16 bytes.
        let n = (body_count.max(1)) as u64;
        let body_q_pred = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd body_q_pred"),
            size: 32 * n,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_v_pred = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd body_v_pred"),
            size: 16 * n,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_q_alt = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd body_q_alt"),
            size: 32 * n,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_v_post = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd body_v_post"),
            size: 16 * n,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_w_pred = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd body_w_pred"),
            size: 16 * n,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let body_w_post = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd body_w_post"),
            size: 16 * n,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let external_f_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd dummy external_f"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline_predict,
            pipeline_project_pos,
            pipeline_recover_v,
            pipeline_project_v,
            pipeline_write_final,
            layout_model,
            layout_state,
            layout_control,
            layout_contacts,
            layout_uniforms,
            body_q_pred,
            body_v_pred,
            body_q_alt,
            body_v_post,
            body_w_pred,
            body_w_post,
            external_f_dummy,
            body_count,
        }
    }

    /// Run one full XPBD step on the GPU. Caller must have uploaded
    /// model, state_in, state_out, control, and contacts beforehand.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        position_iterations: usize,
        dt: f64,
        state_out: &mut State,
    ) -> Result<()> {
        if model.body_count == 0 {
            return Ok(());
        }
        if model.body_count as u32 != self.body_count {
            return Err(Error::Gpu(format!(
                "XpbdKernel was built for {} bodies, got {} — recreate the solver",
                self.body_count, model.body_count
            )));
        }

        // Always update the contact CSR placeholder (offsets buffer
        // size depends on body_count and is built each frame; it's
        // already on device after Contacts::upload_to_device).
        let groups = (model.body_count as u32).div_ceil(WORKGROUP_SIZE);

        // ── Bind groups ──────────────────────────────────────────
        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd model bg"),
            layout: &self.layout_model,
            entries: &[
                buffer_entry(0, model.body_inv_mass.device_buffer())?,
                buffer_entry(1, model.body_flags.device_buffer())?,
                buffer_entry(2, model.body_world.device_buffer())?,
                buffer_entry(3, model.gravity.device_buffer())?,
                buffer_entry(4, model.body_com.device_buffer())?,
                buffer_entry(5, model.body_inv_inertia.device_buffer())?,
            ],
        });

        // State bind group needs to swap two of its bindings during
        // ping-pong. Build TWO state bind groups and alternate.
        let state_in_q = state_in
            .body_q
            .device_buffer()
            .ok_or_else(missing_buf("state_in.body_q"))?;
        let state_in_qd = state_in
            .body_qd
            .device_buffer()
            .ok_or_else(missing_buf("state_in.body_qd"))?;
        let state_in_f = state_in
            .body_f
            .device_buffer()
            .ok_or_else(missing_buf("state_in.body_f"))?;
        let state_out_q = state_out
            .body_q
            .device_buffer()
            .ok_or_else(missing_buf("state_out.body_q"))?;
        let state_out_qd = state_out
            .body_qd
            .device_buffer()
            .ok_or_else(missing_buf("state_out.body_qd"))?;

        // Bind group A: q_out = state_out.body_q, q_alt = body_q_alt.
        // Bind group B: q_out = body_q_alt, q_alt = state_out.body_q.
        let make_state_bg = |q_out: &wgpu::Buffer, q_alt: &wgpu::Buffer, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &self.layout_state,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: state_in_q.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: state_in_qd.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: state_in_f.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.body_q_pred.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.body_v_pred.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: q_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: state_out_qd.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: q_alt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.body_v_post.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: self.body_w_pred.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.body_w_post.as_entire_binding(),
                    },
                ],
            })
        };
        let bg_state_a = make_state_bg(state_out_q, &self.body_q_alt, "molla::xpbd state bg A");
        let bg_state_b = make_state_bg(&self.body_q_alt, state_out_q, "molla::xpbd state bg B");

        // Control.
        let (external_buf, has_external) = if control.body_external_f.len() == model.body_count {
            (
                control
                    .body_external_f
                    .device_buffer()
                    .ok_or_else(missing_buf("control.body_external_f"))?,
                1u32,
            )
        } else {
            (&self.external_f_dummy, 0u32)
        };
        let bg_control = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd control bg"),
            layout: &self.layout_control,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: external_buf.as_entire_binding(),
            }],
        });

        // Contacts. The bind group requires non-null device buffers
        // for every binding, including `body_contact_offsets` /
        // `body_contact_indices` which are sized by the contact set
        // (not the body count). Callers MUST run a `CollisionPipeline
        // ::collide` pass at least once before dispatch (even if it
        // produces zero contacts) so the CSR buffers get sized — see
        // `bench_xpbd_pile_gpu` for the canonical pre-collide pattern.
        // Without that pre-collide, `device_buffer()?` on an empty
        // CSR returns "binding 6: no device mirror".
        let bg_contacts = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd contacts bg"),
            layout: &self.layout_contacts,
            entries: &[
                buffer_entry(0, contacts.a_index.device_buffer())?,
                buffer_entry(1, contacts.b_index.device_buffer())?,
                buffer_entry(2, contacts.normal.device_buffer())?,
                buffer_entry(3, contacts.depth.device_buffer())?,
                buffer_entry(4, contacts.friction.device_buffer())?,
                buffer_entry(5, contacts.restitution.device_buffer())?,
                buffer_entry(6, contacts.body_contact_offsets.device_buffer())?,
                buffer_entry(7, contacts.body_contact_indices.device_buffer())?,
                buffer_entry(8, contacts.point.device_buffer())?,
                buffer_entry(9, contacts.rolling_friction.device_buffer())?,
                buffer_entry(10, contacts.torsional_friction.device_buffer())?,
            ],
        });

        // Uniforms.
        let uniforms = Uniforms {
            dt: dt as f32,
            angular_damping: model.angular_damping as f32,
            body_count: model.body_count as u32,
            has_external_force: has_external,
            linear_damping: model.linear_damping as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ── Encode + dispatch ────────────────────────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::xpbd encoder"),
        });

        // 1. Predict — uses bg_state_a (state_out_q is the initial ping).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd::predict"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_predict);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_state_a, &[]);
            pass.set_bind_group(2, &bg_control, &[]);
            pass.set_bind_group(3, &bg_contacts, &[]);
            pass.set_bind_group(4, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // 2. Position projection — N iterations, ping-pong q_out <-> q_alt.
        // After predict, q_out (state_out.body_q) holds the predicted
        // pose. Iteration k uses bg_state_a if k even, bg_state_b if odd.
        // After the loop, the FINAL pose ends up in:
        //   N even (e.g. 4) → state_out.body_q (ping every iteration).
        //
        // Force even iteration count to keep the final result in
        // state_out.body_q without an extra copy.
        let iters = (position_iterations.max(2) + 1) & !1; // round up to even
        for k in 0..iters {
            let bg = if k % 2 == 0 { &bg_state_a } else { &bg_state_b };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd::project_position"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_project_pos);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, bg, &[]);
            pass.set_bind_group(2, &bg_control, &[]);
            pass.set_bind_group(3, &bg_contacts, &[]);
            pass.set_bind_group(4, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // 3. Velocity recovery — reads body_q_out (final), writes body_v_post.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd::recover_velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_recover_v);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_state_a, &[]);
            pass.set_bind_group(2, &bg_control, &[]);
            pass.set_bind_group(3, &bg_contacts, &[]);
            pass.set_bind_group(4, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // 4. Velocity projection — single iteration (in-place mutation
        // is fine because each thread only writes its own v_post).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd::project_velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_project_v);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_state_a, &[]);
            pass.set_bind_group(2, &bg_control, &[]);
            pass.set_bind_group(3, &bg_contacts, &[]);
            pass.set_bind_group(4, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // 5. Write final velocities into state_out.body_qd (preserves
        // angular component). Position is already in state_out.body_q.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd::write_final"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_write_final);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_state_a, &[]);
            pass.set_bind_group(2, &bg_control, &[]);
            pass.set_bind_group(3, &bg_contacts, &[]);
            pass.set_bind_group(4, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        state_out.body_q.mark_device_dirty();
        state_out.body_qd.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| Error::Gpu(format!("xpbd binding {binding}: no device mirror")))?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn missing_buf(name: &'static str) -> impl FnOnce() -> Error {
    move || {
        Error::Gpu(format!(
            "missing device buffer for {name} — call upload_to_device first"
        ))
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

/// State bind group has 11 bindings — first 3 read-only, rest read-write.
fn xpbd_state_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    (0..11u32)
        .map(|i| {
            let ty = if i < 3 { read } else { rw };
            wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect()
}
