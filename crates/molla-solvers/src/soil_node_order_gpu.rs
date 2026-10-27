use super::*;

pub(super) struct SoilNodeOrderGpu {
    input: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    bodies: u32,
}

fn width(capacity: u32, shared_bytes: u32) -> Result<u32> {
    capacity
        .checked_next_power_of_two()
        .filter(|&n| capacity != 0 && u64::from(n) * 4 + 8 <= u64::from(shared_bytes))
        .ok_or_else(|| invalid("canonical node order exceeds workgroup capacity"))
}

fn source(capacity: u32, width: u32) -> String {
    include_str!("../shaders/soil_node_order.wgsl")
        .replace("__CAPACITY__", &capacity.to_string())
        .replace("__WIDTH__", &width.to_string())
}

impl SoilNodeOrderGpu {
    pub(super) fn validate(device: &wgpu::Device, capacity: u32) -> Result<u32> {
        let limits = device.limits();
        if limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
        {
            return Err(invalid("canonical node order requires 64 lanes"));
        }
        width(capacity, limits.max_compute_workgroup_storage_size)
    }

    pub(super) fn new(
        device: &wgpu::Device,
        output: &wgpu::Buffer,
        spans: &wgpu::Buffer,
        status: &wgpu::Buffer,
        capacity: u32,
        bodies: u32,
    ) -> Result<Self> {
        let width = Self::validate(device, capacity)?;
        let input = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("canonical soil node input"),
                size: output.size(),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("canonical soil node order"),
            source: wgpu::ShaderSource::Wgsl(source(capacity, width).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("canonical soil node order"),
            layout: None,
            module: &shader,
            entry_point: Some("canonicalize"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffers = [&input, spans, output, status];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canonical soil node order"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            input,
            pipeline,
            group,
            bodies,
        })
    }

    pub(super) fn input(&self) -> &wgpu::Buffer {
        &self.input
    }

    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.bodies, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}

#[cfg(test)]
#[path = "soil_node_order_tests.rs"]
mod tests;
