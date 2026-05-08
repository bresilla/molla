//! Rust dispatch wrapper around `shaders/featherstone.wgsl`.
//!
//! Per-articulation parallelism — one workgroup invocation per chain.
//! For molla's typical scenes (one robot per scene) this is single-
//! threaded, but the architecture matches Newton's Featherstone-Warp
//! shape and scales naturally to RL "many robots in parallel" once
//! the articulation grouping is set up.

use crate::core::{Error, Result};
use crate::sim::{Contacts, Control, Model, State};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dt: f32,
    joint_damping: f32,
    joint_count: u32,
    articulation_count: u32,
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
    /// Articulation CSR — one entry per articulation start, last
    /// entry = `joint_count`. Stage 3.3 default: a single
    /// articulation covering every joint.
    art_offsets: wgpu::Buffer,
    articulation_count: u32,
    joint_count: u32,
}

impl FeatherstoneKernel {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, model: &Model) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::featherstone"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/featherstone.wgsl").into(),
            ),
        });

        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone model"),
            entries: &storage_entries(8, true),
        });
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone state"),
            entries: &state_entries(),
        });
        let layout_control = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::featherstone control"),
            entries: &storage_entries(1, true),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::featherstone layout"),
            bind_group_layouts: &[
                &layout_model,
                &layout_state,
                &layout_control,
                &layout_arts,
                &layout_uniforms,
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

        // Default articulation grouping: one articulation owning all
        // joints. CSR = [0, joint_count].
        let joint_count = model.joint_count as u32;
        let art_offsets_data: Vec<u32> = vec![0u32, joint_count];
        let art_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::featherstone art_offsets"),
            size: (art_offsets_data.len() * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&art_offsets, 0, bytemuck::cast_slice(&art_offsets_data));

        Self {
            pipeline,
            layout_model,
            layout_state,
            layout_control,
            layout_arts,
            layout_uniforms,
            art_offsets,
            articulation_count: 1,
            joint_count,
        }
    }

    /// Run one Featherstone step on the GPU. Caller must have
    /// uploaded the model + state + control beforehand.
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
        state_out: &mut State,
    ) -> Result<()> {
        if model.joint_count == 0 {
            return Ok(());
        }

        // Copy state_in's body_q + joint coords into state_out so the
        // kernel mutates state_out's device buffers in place — same
        // pattern as the XPBD GPU dispatch.
        copy_buffer(device, queue, &state_in.body_q, &state_out.body_q)?;
        copy_buffer(device, queue, &state_in.joint_q, &state_out.joint_q)?;
        copy_buffer(device, queue, &state_in.joint_qd, &state_out.joint_qd)?;

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
            ],
        });

        let bg_control = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone control bg"),
            layout: &self.layout_control,
            entries: &[buffer_entry(0, control.joint_force.device_buffer())?],
        });

        let bg_arts = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::featherstone arts bg"),
            layout: &self.layout_arts,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.art_offsets.as_entire_binding(),
            }],
        });

        let uniforms = Uniforms {
            dt: dt as f32,
            joint_damping,
            joint_count: self.joint_count,
            articulation_count: self.articulation_count,
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
    (0..8u32)
        .map(|i| {
            // 0..4: read-only model body data; 5..7: rw state
            let ty = if i < 5 { read } else { rw };
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
fn copy_buffer<T: crate::storage::BufferElement>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &crate::storage::Buffer<T>,
    dst: &crate::storage::Buffer<T>,
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
