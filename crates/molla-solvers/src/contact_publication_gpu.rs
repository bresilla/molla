//! Retained contact-output publication guarded by the shared device status.

use molla_core::{Error, Result};

pub struct ContactPublicationBinding<'a> {
    pub source: &'a wgpu::Buffer,
    pub destination: &'a wgpu::Buffer,
    /// Prefix length in bytes, nonzero and divisible by four.
    pub bytes: u64,
}

/// Copies an output prefix only when the shared status is zero.
/// Encode after the producer and before its consumer, using their shared status.
/// A later fault does not roll back an earlier publication; frame rollback belongs
/// to the caller. All nonzero statuses, including break-stop, suppress publication.
pub struct ResidentContactPublication {
    pipeline: wgpu::ComputePipeline,
    groups: Vec<wgpu::BindGroup>,
    work: Vec<u32>,
}

impl ResidentContactPublication {
    pub fn new(
        device: &wgpu::Device,
        status: &wgpu::Buffer,
        bindings: &[ContactPublicationBinding<'_>],
    ) -> Result<Self> {
        let invalid = || Error::Gpu("invalid contact publication buffers or capacity".into());
        let limits = device.limits();
        if bindings.is_empty()
            || status.size() < 4
            || !status.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(invalid());
        }
        for binding in bindings {
            if binding.bytes == 0
                || binding.bytes % 4 != 0
                || binding.bytes > limits.max_storage_buffer_binding_size
                || (binding.bytes / 4).div_ceil(64)
                    > u64::from(limits.max_compute_workgroups_per_dimension)
                || binding.source == binding.destination
                || binding.destination == status
                || binding.source == status
                || [binding.source, binding.destination].iter().any(|buffer| {
                    buffer.size() < binding.bytes
                        || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                })
            {
                return Err(invalid());
            }
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact publication"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/contact_publication.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("contact publication"),
            layout: None,
            module: &shader,
            entry_point: Some("publish"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let groups = bindings
            .iter()
            .map(|binding| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("contact publication bindings"),
                    layout: &layout,
                    entries: &[
                        (status, 4),
                        (binding.source, binding.bytes),
                        (binding.destination, binding.bytes),
                    ]
                    .into_iter()
                    .enumerate()
                    .map(|(binding, (buffer, bytes))| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(bytes),
                        }),
                    })
                    .collect::<Vec<_>>(),
                })
            })
            .collect();
        Ok(Self {
            pipeline,
            groups,
            work: bindings
                .iter()
                .map(|binding| (binding.bytes / 4).div_ceil(64) as u32)
                .collect(),
        })
    }

    /// Record one prepared publication without allocation or host status polling.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, binding: usize) -> Result<()> {
        let group = self
            .groups
            .get(binding)
            .ok_or_else(|| Error::Gpu("invalid contact publication binding".into()))?;
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.work[binding], 1, 1);
        molla_core::gpu_metrics::record_dispatch();
        Ok(())
    }
}

#[cfg(test)]
#[path = "contact_publication_gpu_tests.rs"]
mod tests;
