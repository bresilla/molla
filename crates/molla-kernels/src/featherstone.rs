//! Rust dispatch wrapper around `shaders/featherstone.wgsl`.
//!
//! Per-articulation parallelism — one workgroup invocation per chain.
//! For molla's typical scenes (one robot per scene) this is single-
//! threaded, but the architecture matches Newton's Featherstone-Warp
//! shape and scales naturally to RL "many robots in parallel" once
//! the articulation grouping is set up.

use molla_core::{Error, Result};
use molla_sim::{Contacts, Control, Model, State};
use molla_storage::{Buffer, BufferElement};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dt: f32,
    joint_damping: f32,
    joint_count: u32,
    articulation_count: u32,
    /// Stage 17.1: 1 if the WGSL kernel should use the rigorous
    /// dense-Cholesky D solve (v.s. v1 sequential rank-1).
    rigorous_d_solve: u32,
    /// Stage 17.1: 1 if Distance joints should use velocity-level
    /// Lagrange-multiplier projection instead of the v1 hard-clamp.
    rigorous_distance: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU Featherstone-style dynamics kernel.
pub struct FeatherstoneKernel {
    pipeline: wgpu::ComputePipeline,
    layout_model: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_control: wgpu::BindGroupLayout,
    layout_arts: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    /// Bind group 5 — per-joint cable spring/damper + per-DOF muscle
    /// torque accumulator (Stage 39E + 42 v3).
    layout_cable_muscle: wgpu::BindGroupLayout,
    /// Articulation CSR — built from `model.articulation_start` at
    /// kernel construction. Length = `articulation_count + 1`.
    art_offsets: wgpu::Buffer,
    /// Zero-filled fallback buffer used when the caller doesn't pass
    /// a custom per-DOF muscle torque buffer. Size = `dof_total *
    /// sizeof(f32)`, allocated once at kernel construction.
    zero_dof_buf: wgpu::Buffer,
    articulation_count: u32,
    joint_count: u32,
    dof_total: u32,
}

impl FeatherstoneKernel {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, model: &Model) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::featherstone"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/featherstone.wgsl").into()),
        });

        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone model"),
            entries: &storage_entries(13, true),
        });
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone state"),
            entries: &state_entries(),
        });
        let layout_control = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone control"),
            entries: &storage_entries(6, true),
        });
        let layout_arts = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone arts"),
            entries: &storage_entries(1, true),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone uniforms"),
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
        // Bind group 5 — 4 cable per-joint buffers + 1 per-DOF muscle
        // torque accumulator. All read-only.
        let layout_cable_muscle =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::featherstone cable+muscle"),
                entries: &storage_entries(5, true),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::featherstone layout"),
            bind_group_layouts: &[
                &layout_model,
                &layout_state,
                &layout_control,
                &layout_arts,
                &layout_uniforms,
                &layout_cable_muscle,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::featherstone"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("featherstone_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Articulation CSR — pull from the model. Falls back to a
        // single articulation covering every joint when
        // `articulation_count == 0` (e.g. an empty/legacy model).
        let joint_count = model.joint_count as u32;
        let art_offsets_data: Vec<u32> = if model.articulation_count == 0 {
            vec![0u32, joint_count]
        } else {
            model
                .articulation_start
                .host()
                .map(|s| s.to_vec())
                .unwrap_or_else(|_| vec![0u32, joint_count])
        };
        let articulation_count = (art_offsets_data.len().saturating_sub(1)) as u32;
        let art_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::featherstone art_offsets"),
            size: (art_offsets_data.len() * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&art_offsets, 0, bytemuck::cast_slice(&art_offsets_data));

        let dof_total = model.joint_dof_total.max(1) as u32;
        let zero_dof_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::featherstone zero_dof"),
            size: (dof_total as u64) * (core::mem::size_of::<f32>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-fill once — never written again.
        let zeros: Vec<u8> = vec![0u8; (dof_total as usize) * core::mem::size_of::<f32>()];
        queue.write_buffer(&zero_dof_buf, 0, &zeros);

        Self {
            pipeline,
            layout_model,
            layout_state,
            layout_control,
            layout_arts,
            layout_uniforms,
            layout_cable_muscle,
            art_offsets,
            zero_dof_buf,
            articulation_count,
            joint_count,
            dof_total,
        }
    }

    /// Borrow the cable+muscle bind-group layout so external callers
    /// (e.g. an integration test) can build their own custom bind
    /// group. The dispatch path uses a private path.
    pub fn cable_muscle_layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout_cable_muscle
    }

    /// Cached joint DoF total for sizing externally-supplied muscle
    /// torque buffers.
    pub fn dof_total(&self) -> u32 {
        self.dof_total
    }

    /// Run one Featherstone step on the GPU. Caller must have
    /// uploaded the model + state + control beforehand.
    ///
    /// `muscle_torque` (optional) is a per-DOF buffer of `f32` of
    /// length `dof_total` containing the pre-computed Hill-type muscle
    /// torque sum for every DoF (zero on non-muscle DoFs). Pass
    /// `None` to use the kernel's internal zero buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        control: &Control,
        _contacts: &Contacts,
        joint_damping: f32,
        dt: f64,
        rigorous_d_solve: bool,
        rigorous_distance: bool,
        state_out: &mut State,
        muscle_torque: Option<&wgpu::Buffer>,
    ) -> Result<()> {
        if model.joint_count == 0 {
            return Ok(());
        }

        // Copy state_in's body_q + joint coords into state_out so the
        // kernel mutates state_out's device buffers in place — same
        // pattern as the XPBD GPU dispatch. body_f is also propagated
        // so substepping callers see the same external wrench across
        // every substep (until they overwrite it).
        copy_buffer(device, queue, &state_in.body_q, &state_out.body_q)?;
        copy_buffer(device, queue, &state_in.joint_q, &state_out.joint_q)?;
        copy_buffer(device, queue, &state_in.joint_qd, &state_out.joint_qd)?;
        copy_buffer(device, queue, &state_in.body_f, &state_out.body_f)?;

        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone model bg"),
            layout: &self.layout_model,
            entries: &[
                buffer_entry(0, model.joint_type.device_buffer())?,
                buffer_entry(1, model.joint_parent.device_buffer())?,
                buffer_entry(2, model.joint_child.device_buffer())?,
                buffer_entry(3, model.joint_x_p.device_buffer())?,
                buffer_entry(4, model.joint_x_c.device_buffer())?,
                buffer_entry(5, model.joint_axis.device_buffer())?,
                buffer_entry(6, model.joint_coord_offset.device_buffer())?,
                buffer_entry(7, model.joint_dof_offset.device_buffer())?,
                buffer_entry(8, model.joint_limit_lower.device_buffer())?,
                buffer_entry(9, model.joint_limit_upper.device_buffer())?,
                buffer_entry(10, model.joint_friction.device_buffer())?,
                buffer_entry(11, model.joint_effort_limit.device_buffer())?,
                buffer_entry(12, model.joint_velocity_limit.device_buffer())?,
            ],
        });

        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, model.body_mass.device_buffer())?,
                buffer_entry(1, model.body_inertia.device_buffer())?,
                buffer_entry(2, model.body_com.device_buffer())?,
                buffer_entry(3, model.body_world.device_buffer())?,
                buffer_entry(4, model.gravity.device_buffer())?,
                buffer_entry(5, state_out.body_q.device_buffer())?,
                buffer_entry(6, state_out.joint_q.device_buffer())?,
                buffer_entry(7, state_out.joint_qd.device_buffer())?,
                buffer_entry(8, state_in.body_f.device_buffer())?,
            ],
        });

        let bg_control = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone control bg"),
            layout: &self.layout_control,
            entries: &[
                buffer_entry(0, control.joint_force.device_buffer())?,
                buffer_entry(1, control.joint_target_mode.device_buffer())?,
                buffer_entry(2, control.joint_target_pos.device_buffer())?,
                buffer_entry(3, control.joint_target_vel.device_buffer())?,
                buffer_entry(4, control.joint_target_kp.device_buffer())?,
                buffer_entry(5, control.joint_target_kd.device_buffer())?,
            ],
        });

        let bg_arts = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone arts bg"),
            layout: &self.layout_arts,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.art_offsets.as_entire_binding(),
            }],
        });

        // Bind group 5 — cable spring/damper + muscle torque accumulator.
        let muscle_buf = muscle_torque.unwrap_or(&self.zero_dof_buf);
        let bg_cable_muscle = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone cable+muscle bg"),
            layout: &self.layout_cable_muscle,
            entries: &[
                buffer_entry(0, model.joint_cable_stiffness_linear.device_buffer())?,
                buffer_entry(1, model.joint_cable_stiffness_bend.device_buffer())?,
                buffer_entry(2, model.joint_cable_damping_linear.device_buffer())?,
                buffer_entry(3, model.joint_cable_damping_bend.device_buffer())?,
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: muscle_buf.as_entire_binding(),
                },
            ],
        });

        let uniforms = Uniforms {
            dt: dt as f32,
            joint_damping,
            joint_count: self.joint_count,
            articulation_count: self.articulation_count,
            rigorous_d_solve: u32::from(rigorous_d_solve),
            rigorous_distance: u32::from(rigorous_distance),
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::featherstone uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::featherstone encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::featherstone pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_state, &[]);
            pass.set_bind_group(2, &bg_control, &[]);
            pass.set_bind_group(3, &bg_arts, &[]);
            pass.set_bind_group(4, &bg_uniforms, &[]);
            pass.set_bind_group(5, &bg_cable_muscle, &[]);
            let groups = self.articulation_count.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.body_q.mark_device_dirty();
        state_out.joint_q.mark_device_dirty();
        state_out.joint_qd.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "featherstone binding {binding}: no device mirror — call upload_to_device first"
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

fn state_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let rw = wgpu::BufferBindingType::Storage { read_only: false };
    (0..9u32)
        .map(|i| {
            // 0..4: read-only model body data; 5..7: rw state;
            // 8: read-only body_f (external wrenches at COM, world).
            let ty = if i < 5 || i == 8 { read } else { rw };
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

/// GPU-to-GPU buffer copy used to seed `state_out` from `state_in`
/// before the kernel mutates the output buffers in place.
fn copy_buffer<T: BufferElement>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &Buffer<T>,
    dst: &Buffer<T>,
) -> Result<()> {
    let s = src
        .device_buffer()
        .ok_or_else(|| Error::Gpu("copy_buffer: src has no device mirror".into()))?;
    let d = dst
        .device_buffer()
        .ok_or_else(|| Error::Gpu("copy_buffer: dst has no device mirror".into()))?;
    let size = s.size().min(d.size());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::featherstone copy"),
    });
    encoder.copy_buffer_to_buffer(s, 0, d, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
