//! Rust dispatch wrapper for `shaders/actuator_delay.wgsl`.
//!
//! Exposes two pipelines built from the same shader module:
//! - `delay_push` — rotate the circular buffer + record current
//!   target pos/vel for every actuator in the group.
//! - `delay_read` — sample the per-DOF lag and produce the delayed
//!   targets the controller actually consumes.
//!
//! Both pipelines share one bind-group layout (4 groups), so the
//! caller binds the same buffer set for either dispatch.

use molla_core::{Error, Result};
use molla_storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    actuator_count: u32,
    buf_depth: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct ActuatorDelayKernel {
    push_pipeline: wgpu::ComputePipeline,
    read_pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ActuatorDelayKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::actuator_delay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/actuator_delay.wgsl").into()),
        });

        // Group 0: read-only inputs (target_pos, target_vel,
        // dof_indices, delay_steps). `read_write = false`.
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_delay in"),
            entries: &storage_entries(4, false),
        });
        // Group 1: read-write circular-buffer state.
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_delay state"),
            entries: &storage_entries(4, true),
        });
        // Group 2: read-write delayed outputs (consumed by controller).
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_delay out"),
            entries: &storage_entries(2, true),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_delay uniforms"),
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
            label: Some("molla::actuator_delay layout"),
            bind_group_layouts: &[
                Some(&layout_in),
                Some(&layout_state),
                Some(&layout_out),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
        });

        let push_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::actuator_delay::delay_push"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("delay_push"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let read_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::actuator_delay::delay_read"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("delay_read"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            push_pipeline,
            read_pipeline,
            layout_in,
            layout_state,
            layout_out,
            layout_uniforms,
        }
    }

    /// Bind groups + uniforms; returns the encoder so the caller can
    /// chain stages without flushing in between.
    #[allow(clippy::too_many_arguments)]
    fn make_bind_groups(
        &self,
        device: &wgpu::Device,
        target_pos: &Buffer<molla_math::Real>,
        target_vel: &Buffer<molla_math::Real>,
        dof_indices: &Buffer<u32>,
        delay_steps: &Buffer<u32>,
        buffer_pos: &Buffer<molla_math::Real>,
        buffer_vel: &Buffer<molla_math::Real>,
        num_pushes: &Buffer<u32>,
        write_idx: &Buffer<u32>,
        out_pos: &Buffer<molla_math::Real>,
        out_vel: &Buffer<molla_math::Real>,
        uniform_buf: &wgpu::Buffer,
    ) -> Result<(
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
    )> {
        let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_delay in bg"),
            layout: &self.layout_in,
            entries: &[
                buffer_entry(0, target_pos.device_buffer())?,
                buffer_entry(1, target_vel.device_buffer())?,
                buffer_entry(2, dof_indices.device_buffer())?,
                buffer_entry(3, delay_steps.device_buffer())?,
            ],
        });
        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_delay state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, buffer_pos.device_buffer())?,
                buffer_entry(1, buffer_vel.device_buffer())?,
                buffer_entry(2, num_pushes.device_buffer())?,
                buffer_entry(3, write_idx.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_delay out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, out_pos.device_buffer())?,
                buffer_entry(1, out_vel.device_buffer())?,
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_delay uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        Ok((bg_in, bg_state, bg_out, bg_uniforms))
    }

    /// One push + one read in a single submit. Caller is expected to
    /// hold the buffers and the per-actuator sizing.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_pos: &Buffer<molla_math::Real>,
        target_vel: &Buffer<molla_math::Real>,
        dof_indices: &Buffer<u32>,
        delay_steps: &Buffer<u32>,
        buffer_pos: &mut Buffer<molla_math::Real>,
        buffer_vel: &mut Buffer<molla_math::Real>,
        num_pushes: &mut Buffer<u32>,
        write_idx: &mut Buffer<u32>,
        out_pos: &mut Buffer<molla_math::Real>,
        out_vel: &mut Buffer<molla_math::Real>,
        actuator_count: u32,
        buf_depth: u32,
    ) -> Result<()> {
        if actuator_count == 0 {
            return Ok(());
        }

        let uniforms = Uniforms {
            actuator_count,
            buf_depth,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::actuator_delay uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let (bg_in, bg_state, bg_out, bg_uniforms) = self.make_bind_groups(
            device,
            target_pos,
            target_vel,
            dof_indices,
            delay_steps,
            buffer_pos,
            buffer_vel,
            num_pushes,
            write_idx,
            out_pos,
            out_vel,
            &uniform_buf,
        )?;

        let groups = actuator_count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::actuator_delay encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::actuator_delay::push"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.push_pipeline);
            pass.set_bind_group(0, &bg_in, &[]);
            pass.set_bind_group(1, &bg_state, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::actuator_delay::read"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.read_pipeline);
            pass.set_bind_group(0, &bg_in, &[]);
            pass.set_bind_group(1, &bg_state, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        buffer_pos.mark_device_dirty();
        buffer_vel.mark_device_dirty();
        num_pushes.mark_device_dirty();
        write_idx.mark_device_dirty();
        out_pos.mark_device_dirty();
        out_vel.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "actuator_delay binding {binding}: no device mirror"
        ))
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn storage_entries(count: u32, read_write: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: !read_write,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}
