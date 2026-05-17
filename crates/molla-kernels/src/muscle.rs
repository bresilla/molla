//! Rust dispatch wrapper around `shaders/muscle.wgsl`.
//!
//! Stage 42 v3 — Hill-type muscle on the GPU. Provides two compute
//! passes:
//!
//!   * [`MuscleKernel::dispatch_activation`] — first-order low-pass
//!     filter on the per-muscle activation buffer (one thread per
//!     muscle). Mirrors the CPU
//!     `molla_actuators::update_muscle_activation`.
//!
//!   * [`MuscleKernel::dispatch_torque_sum`] — one thread per joint,
//!     sums the Hill three-element torque over every muscle attached
//!     via the CSR `(joint_muscle_offsets, joint_muscle_indices)` and
//!     writes the result into a persistent per-DOF buffer. The cable
//!     joint's linear-stretch DoF receives the sum; every other DoF
//!     stays zero. The GPU Featherstone reads this buffer and adds it
//!     to `τ_raw` before the ABA backward pass.
//!
//! The per-DOF torque buffer (`joint_muscle_torque`) is owned by the
//! kernel so it survives across steps without allocation; callers
//! call [`MuscleKernel::torque_buffer`] to plug it into the
//! Featherstone bind group.

use molla_core::{Error, Result};
use molla_sim::{Model, State};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    muscle_count: u32,
    joint_count: u32,
    dof_total: u32,
    dt: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU Hill-type muscle kernel — owns persistent buffers + pipelines
/// for activation update + per-joint torque sum.
pub struct MuscleKernel {
    pipeline_activation: wgpu::ComputePipeline,
    pipeline_torque: wgpu::ComputePipeline,
    layout_uniforms: wgpu::BindGroupLayout,
    layout_params: wgpu::BindGroupLayout,
    layout_joint: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,

    /// Persistent uniform buffer (rewritten per dispatch).
    uniform_buf: wgpu::Buffer,
    /// Per-DOF muscle torque output. Length = `dof_total`.
    torque_buf: wgpu::Buffer,
    /// Cached `dof_total` so callers can zero / read back the buffer
    /// without holding a `&Model` reference.
    dof_total: u32,
    /// Joint count cached for dispatch sizing.
    joint_count: u32,
}

impl MuscleKernel {
    /// Build the kernel from a model. The model's `joint_dof_offset`
    /// + `muscle_count` are baked into a per-DOF output buffer of
    /// length `model.joint_dof_total`. Caller must have uploaded the
    /// model + muscle buffers before dispatching.
    pub fn new(device: &wgpu::Device, model: &Model) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::muscle"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/muscle.wgsl").into()),
        });

        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::muscle uniforms"),
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
        // Params: 9 read-only + 1 read-write (activation). Both
        // entry points share the same layout — activation update
        // writes to activation; torque-sum only reads it.
        let layout_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::muscle params"),
            entries: &mixed_param_entries(),
        });
        let layout_joint = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::muscle joint"),
            entries: &storage_entries(7, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::muscle out"),
            entries: &storage_entries(1, false),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::muscle layout"),
            bind_group_layouts: &[&layout_uniforms, &layout_params, &layout_joint, &layout_out],
            push_constant_ranges: &[],
        });

        let pipeline_activation =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::muscle::update_muscle_activation"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("update_muscle_activation"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let pipeline_torque = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::muscle::muscle_torque_sum"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("muscle_torque_sum"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let dof_total = model.joint_dof_total.max(1) as u32;
        let joint_count = model.joint_count as u32;
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::muscle uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let torque_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::muscle joint_muscle_torque"),
            size: (dof_total as u64) * (core::mem::size_of::<f32>() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            pipeline_activation,
            pipeline_torque,
            layout_uniforms,
            layout_params,
            layout_joint,
            layout_out,
            uniform_buf,
            torque_buf,
            dof_total,
            joint_count,
        }
    }

    /// Borrow the persistent per-DOF muscle-torque buffer so the
    /// Featherstone kernel can read it. Length = `joint_dof_total`.
    pub fn torque_buffer(&self) -> &wgpu::Buffer {
        &self.torque_buf
    }

    /// Cached `joint_dof_total` for binding-size validation.
    pub fn dof_total(&self) -> u32 {
        self.dof_total
    }

    fn write_uniforms(&self, queue: &wgpu::Queue, muscle_count: u32, dt: f32) {
        let u = Uniforms {
            muscle_count,
            joint_count: self.joint_count,
            dof_total: self.dof_total,
            dt,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
    }

    fn make_bind_groups(
        &self,
        device: &wgpu::Device,
        model: &Model,
        state: &State,
    ) -> Result<(
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
    )> {
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::muscle uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buf.as_entire_binding(),
            }],
        });
        let bg_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::muscle params bg"),
            layout: &self.layout_params,
            entries: &[
                buffer_entry(0, model.muscle_f_max.device_buffer())?,
                buffer_entry(1, model.muscle_l_opt.device_buffer())?,
                buffer_entry(2, model.muscle_v_max.device_buffer())?,
                buffer_entry(3, model.muscle_activation.device_buffer())?,
                buffer_entry(4, model.muscle_excitation.device_buffer())?,
                buffer_entry(5, model.muscle_tau_activation.device_buffer())?,
                buffer_entry(6, model.muscle_l_tendon_slack.device_buffer())?,
                buffer_entry(7, model.muscle_pennation_angle.device_buffer())?,
                buffer_entry(8, model.muscle_polarity.device_buffer())?,
                buffer_entry(9, model.muscle_tendon_stiffness_ratio.device_buffer())?,
            ],
        });
        let bg_joint = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::muscle joint bg"),
            layout: &self.layout_joint,
            entries: &[
                buffer_entry(0, model.joint_type.device_buffer())?,
                buffer_entry(1, model.joint_coord_offset.device_buffer())?,
                buffer_entry(2, model.joint_dof_offset.device_buffer())?,
                buffer_entry(3, model.joint_muscle_offsets.device_buffer())?,
                buffer_entry(4, model.joint_muscle_indices.device_buffer())?,
                buffer_entry(5, state.joint_q.device_buffer())?,
                buffer_entry(6, state.joint_qd.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::muscle out bg"),
            layout: &self.layout_out,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.torque_buf.as_entire_binding(),
            }],
        });
        Ok((bg_uniforms, bg_params, bg_joint, bg_out))
    }

    /// Step the activation filter on every muscle. Reads
    /// `muscle_excitation` + `muscle_activation`, writes back into
    /// `muscle_activation`. No-op when `model.muscle_count == 0`.
    pub fn dispatch_activation(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state: &State,
        dt: f32,
    ) -> Result<()> {
        if model.muscle_count == 0 {
            return Ok(());
        }
        self.write_uniforms(queue, model.muscle_count as u32, dt);
        let (bg_u, bg_p, bg_j, bg_o) = self.make_bind_groups(device, model, state)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::muscle::update_muscle_activation"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::muscle::update_muscle_activation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_activation);
            pass.set_bind_group(0, &bg_u, &[]);
            pass.set_bind_group(1, &bg_p, &[]);
            pass.set_bind_group(2, &bg_j, &[]);
            pass.set_bind_group(3, &bg_o, &[]);
            let groups = (model.muscle_count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Compute the per-joint muscle torque sum into the persistent
    /// per-DOF buffer. Caller must call this after activations are
    /// up to date and before invoking the Featherstone kernel.
    /// Writes one thread per joint; the cable joint's linear-stretch
    /// DoF receives the sum, every other DoF gets zero.
    pub fn dispatch_torque_sum(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state: &State,
    ) -> Result<()> {
        if model.joint_count == 0 {
            return Ok(());
        }
        // dt is unused by the torque-sum entry point — pass 0.
        self.write_uniforms(queue, model.muscle_count as u32, 0.0);
        let (bg_u, bg_p, bg_j, bg_o) = self.make_bind_groups(device, model, state)?;

        // First, zero the entire output buffer so DoFs from joints
        // that aren't visited by the kernel (e.g. when `joint_count`
        // doesn't equal `dof_total / 1`) stay clean. The shader only
        // writes joint's k=0 slot; revolute / prismatic joints that
        // also occupy a single DoF would be overwritten by their own
        // workgroup, but other joint types may leave gaps.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::muscle::torque_zero"),
        });
        // Clear by writing zeros via clear_buffer.
        encoder.clear_buffer(&self.torque_buf, 0, None);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::muscle::muscle_torque_sum"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_torque);
            pass.set_bind_group(0, &bg_u, &[]);
            pass.set_bind_group(1, &bg_p, &[]);
            pass.set_bind_group(2, &bg_j, &[]);
            pass.set_bind_group(3, &bg_o, &[]);
            let groups = (model.joint_count as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "muscle binding {binding}: no device mirror — call upload_to_device first"
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

/// Layout for muscle parameter bind group: 8 read-only buffers plus a
/// read-write activation slot at binding 3 (matches `muscle.wgsl`).
fn mixed_param_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..10u32)
        .map(|i| {
            let read_only = i != 3;
            wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect()
}
