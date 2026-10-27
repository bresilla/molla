//! Persistent batched scan-layout and sub-ray selection pipelines.

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LidarScanDescriptor {
    pub counts: [u32; 4],
    pub grid: [u32; 4],
    pub origin: [f32; 4],
    pub forward: [f32; 4],
    pub up: [f32; 4],
    pub right: [f32; 4],
    pub angles: [f32; 4],
    pub extent: [f32; 4],
}

pub struct LidarScanKernel {
    generate: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 4],
    layouts: Vec<wgpu::BindGroupLayout>,
    uniform: wgpu::Buffer,
}

pub struct LidarScanBindings {
    groups: [wgpu::BindGroup; 4],
    buffers: [wgpu::Buffer; 9],
}

impl LidarScanBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 9]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl LidarScanKernel {
    /// Bind descriptors, ray origins/directions, hit t/shape/point/normal, and first/last/count.
    pub fn new(
        device: &wgpu::Device,
        descriptors: &wgpu::Buffer,
        rays: [&wgpu::Buffer; 2],
        hits: [&wgpu::Buffer; 4],
        selections: [&wgpu::Buffer; 3],
    ) -> Self {
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::lidar sizes"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let buffers: [&[&wgpu::Buffer]; 4] = [&[descriptors, &uniform], &rays, &hits, &selections];
        let layouts: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(group, buffers)| {
                let entries: Vec<_> = buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, _)| wgpu::BindGroupLayoutEntry {
                        binding: binding as u32,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: if group == 0 && binding == 1 {
                                wgpu::BufferBindingType::Uniform
                            } else {
                                wgpu::BufferBindingType::Storage {
                                    read_only: group == 0,
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect();
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("molla::lidar bindings"),
                    entries: &entries,
                })
            })
            .collect();
        let groups = std::array::from_fn(|group| {
            let entries: Vec<_> = buffers[group]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::lidar bindings"),
                layout: &layouts[group],
                entries: &entries,
            })
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::lidar pipeline"),
            bind_group_layouts: &layouts.iter().map(Some).collect::<Vec<_>>(),
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::lidar scan"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/lidar_scan.wgsl").into()),
        });
        let pipeline = |name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            generate: pipeline("generate"),
            reduce: pipeline("reduce"),
            groups,
            layouts,
            uniform,
        }
    }

    /// Bind generated rays, hit channels and pulse selections to another output set.
    pub fn prepare(&self, device: &wgpu::Device, buffers: [&wgpu::Buffer; 9]) -> LidarScanBindings {
        let slices = [&buffers[0..2], &buffers[2..6], &buffers[6..9]];
        let groups = std::array::from_fn(|i| {
            if i == 0 {
                return self.groups[0].clone();
            }
            let entries: Vec<_> = slices[i - 1]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::lidar direct bindings"),
                layout: &self.layouts[i],
                entries: &entries,
            })
        });
        LidarScanBindings {
            groups,
            buffers: buffers.map(Clone::clone),
        }
    }

    pub fn encode_generate_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &LidarScanBindings,
        scans: u32,
        rays: u32,
        pulses: u32,
    ) {
        molla_core::gpu_write_buffer!(
            queue,
            &self.uniform,
            0,
            bytemuck::cast_slice(&[scans, rays, pulses, 0]),
        );
        self.encode_groups(encoder, &self.generate, &bindings.groups, rays);
    }

    pub fn encode_reduce_prepared(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &LidarScanBindings,
        pulses: u32,
    ) {
        self.encode_groups(encoder, &self.reduce, &bindings.groups, pulses);
    }

    pub fn generate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scans: u32,
        rays: u32,
        pulses: u32,
    ) {
        molla_core::gpu_write_buffer!(
            queue,
            &self.uniform,
            0,
            bytemuck::cast_slice(&[scans, rays, pulses, 0]),
        );
        self.submit(device, queue, &self.generate, rays);
    }

    pub fn reduce(&self, device: &wgpu::Device, queue: &wgpu::Queue, pulses: u32) {
        self.submit(device, queue, &self.reduce, pulses);
    }

    /// Encode layout generation; submit generation/query/reduction before reuse.
    pub fn encode_generate(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scans: u32,
        rays: u32,
        pulses: u32,
    ) {
        molla_core::gpu_write_buffer!(
            queue,
            &self.uniform,
            0,
            bytemuck::cast_slice(&[scans, rays, pulses, 0]),
        );
        self.encode_count(encoder, &self.generate, rays);
    }

    pub fn encode_reduce(&self, encoder: &mut wgpu::CommandEncoder, pulses: u32) {
        self.encode_count(encoder, &self.reduce, pulses);
    }

    fn encode_count(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        count: u32,
    ) {
        self.encode_groups(encoder, pipeline, &self.groups, count);
    }

    fn encode_groups(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        groups: &[wgpu::BindGroup; 4],
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::lidar scan"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        for (index, group) in groups.iter().enumerate() {
            pass.set_bind_group(index as u32, group, &[]);
        }
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
    }

    fn submit(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &wgpu::ComputePipeline,
        count: u32,
    ) {
        if count == 0 {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::lidar scan"),
        });
        self.encode_count(&mut encoder, pipeline, count);
        queue.submit([encoder.finish()]);
    }
}
