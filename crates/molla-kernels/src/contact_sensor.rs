//! Rust dispatch wrapper for `shaders/contact_aggregate.wgsl` —
//! the GPU contact-summary kernel.

use crate::sensor_bindings::{SensorBindings, required_buffers};
use molla_core::Result;
use molla_sim::Contacts;
use molla_storage::Buffer;

/// Retained contact/CSR/output bindings and one parameter buffer.
pub struct ContactAggregateBindings(SensorBindings<11, 2>);

impl ContactAggregateBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 11]) -> bool {
        self.0.matches(buffers)
    }
}

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
            bind_group_layouts: &[
                Some(&layout_contacts),
                Some(&layout_out),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
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

    /// Buffer order: a/b indices, normals, depths, points, CSR offsets/indices,
    /// output counts/depths/normals/centroids. All belong to `device`.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 11],
    ) -> ContactAggregateBindings {
        ContactAggregateBindings(SensorBindings::new(
            device,
            buffers,
            [(&self.layout_contacts, 0..7), (&self.layout_out, 7..11)],
            &self.layout_uniforms,
            "molla::contact_aggregate prepared",
        ))
    }

    /// Encode without submitting. Submit before rewriting these bindings' parameters;
    /// use separate bindings for differing parameters in the same pending batch.
    /// The caller validates active lengths/CSR and marks outputs device-authoritative.
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &ContactAggregateBindings,
        body_count: u32,
    ) {
        if body_count == 0 {
            return;
        }
        let uniforms = Uniforms {
            body_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        bindings.0.encode(
            queue,
            encoder,
            &self.pipeline,
            bytemuck::cast_ref(&uniforms),
            body_count.div_ceil(WORKGROUP_SIZE),
        );
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

        let bindings = self.prepare(
            device,
            required_buffers(
                [
                    contacts.a_index.device_buffer(),
                    contacts.b_index.device_buffer(),
                    contacts.normal.device_buffer(),
                    contacts.depth.device_buffer(),
                    contacts.point.device_buffer(),
                    contacts.body_contact_offsets.device_buffer(),
                    contacts.body_contact_indices.device_buffer(),
                    out_count.device_buffer(),
                    out_max_depth.device_buffer(),
                    out_normal_sum.device_buffer(),
                    out_centroid.device_buffer(),
                ],
                "contact_aggregate",
            )?,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::contact_aggregate encoder"),
        });
        self.encode_prepared(queue, &mut encoder, &bindings, body_count);
        queue.submit(std::iter::once(encoder.finish()));

        out_count.mark_device_dirty();
        out_max_depth.mark_device_dirty();
        out_normal_sum.mark_device_dirty();
        out_centroid.mark_device_dirty();
        Ok(())
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
