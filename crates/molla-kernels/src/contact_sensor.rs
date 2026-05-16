//! Rust dispatch wrapper for `shaders/contact_aggregate.wgsl` —
//! the GPU contact-summary kernel.

use molla_core::{Error, Result};
use molla_sim::Contacts;
use molla_storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    body_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct ContactAggregateKernel {
    pipeline: wgpu::ComputePipeline,
    layout_contacts: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ContactAggregateKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::contact_aggregate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/contact_aggregate.wgsl").into(),
            ),
        });

        // 7 read-only storage buffers in the contact bind group.
        let layout_contacts = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::contact_aggregate contacts"),
            entries: &storage_entries(7, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::contact_aggregate out"),
            entries: &storage_entries(4, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::contact_aggregate uniforms"),
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
            label: Some("molla::contact_aggregate layout"),
            bind_group_layouts: &[&layout_contacts, &layout_out, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::contact_aggregate::aggregate_contacts"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("aggregate_contacts"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_contacts,
            layout_out,
            layout_uniforms,
        }
    }

    /// Run the kernel — one thread per body. Output buffers must be
    /// at least `body_count` long. Caller is responsible for having
    /// `Contacts::upload_to_device` already called for the current
    /// frame's contacts.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        contacts: &Contacts,
        body_count: u32,
        out_count: &mut Buffer<u32>,
        out_max_depth: &mut Buffer<molla_math::Real>,
        out_normal_sum: &mut Buffer<molla_math::Vec3>,
        out_centroid: &mut Buffer<molla_math::Vec3>,
    ) -> Result<()> {
        if body_count == 0 {
            return Ok(());
        }

        let bg_contacts = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::contact_aggregate contacts bg"),
            layout: &self.layout_contacts,
            entries: &[
                buffer_entry(0, contacts.a_index.device_buffer())?,
                buffer_entry(1, contacts.b_index.device_buffer())?,
                buffer_entry(2, contacts.normal.device_buffer())?,
                buffer_entry(3, contacts.depth.device_buffer())?,
                buffer_entry(4, contacts.point.device_buffer())?,
                buffer_entry(5, contacts.body_contact_offsets.device_buffer())?,
                buffer_entry(6, contacts.body_contact_indices.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::contact_aggregate out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, out_count.device_buffer())?,
                buffer_entry(1, out_max_depth.device_buffer())?,
                buffer_entry(2, out_normal_sum.device_buffer())?,
                buffer_entry(3, out_centroid.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            body_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::contact_aggregate uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::contact_aggregate uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = body_count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::contact_aggregate encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::contact_aggregate::aggregate_contacts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_contacts, &[]);
            pass.set_bind_group(1, &bg_out, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        out_count.mark_device_dirty();
        out_max_depth.mark_device_dirty();
        out_normal_sum.mark_device_dirty();
        out_centroid.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "contact_aggregate binding {binding}: no device mirror"
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
