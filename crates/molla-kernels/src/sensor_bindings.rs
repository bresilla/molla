use std::ops::Range;

pub(crate) fn required_buffers<'a, const N: usize>(
    buffers: [Option<&'a wgpu::Buffer>; N],
    label: &str,
) -> molla_core::Result<[&'a wgpu::Buffer; N]> {
    if let Some(index) = buffers.iter().position(Option::is_none) {
        return Err(molla_core::Error::Gpu(format!(
            "{label} binding {index}: no device mirror"
        )));
    }
    Ok(buffers.map(|buffer| buffer.expect("validated device mirror")))
}

pub(crate) struct SensorBindings<const N: usize, const G: usize> {
    buffers: [wgpu::Buffer; N],
    groups: [wgpu::BindGroup; G],
    uniform: wgpu::Buffer,
    uniform_group: wgpu::BindGroup,
}

impl<const N: usize, const G: usize> SensorBindings<N, G> {
    pub(crate) fn new(
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; N],
        layouts: [(&wgpu::BindGroupLayout, Range<usize>); G],
        uniform_layout: &wgpu::BindGroupLayout,
        label: &str,
    ) -> Self {
        let groups = layouts.map(|(layout, range)| {
            let entries: Vec<_> = buffers[range]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
        });
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let uniform_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        Self {
            buffers: buffers.map(Clone::clone),
            groups,
            uniform,
            uniform_group,
        }
    }

    pub(crate) fn matches(&self, buffers: [&wgpu::Buffer; N]) -> bool {
        self.buffers
            .iter()
            .zip(buffers)
            .all(|(old, new)| old == new)
    }

    pub(crate) fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        uniforms: &[u8; 16],
        workgroups: u32,
    ) {
        molla_core::gpu_write_buffer!(queue, &self.uniform, 0, uniforms);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::sensor prepared"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        for (index, group) in self.groups.iter().enumerate() {
            pass.set_bind_group(index as u32, group, &[]);
        }
        pass.set_bind_group(G as u32, &self.uniform_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
