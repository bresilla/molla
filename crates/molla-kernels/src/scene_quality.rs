use super::storage;
use wgpu::util::DeviceExt;

pub(super) struct SceneQualityKernel {
    measure: wgpu::ComputePipeline,
    decide: wgpu::ComputePipeline,
    capture: wgpu::ComputePipeline,
    commit: wgpu::ComputePipeline,
    metrics: wgpu::BindGroup,
    copy: wgpu::BindGroup,
    pub args: wgpu::Buffer,
    pub state: wgpu::Buffer,
    blocks: u32,
    pub device_bytes: u64,
}

impl SceneQualityKernel {
    pub fn new(
        device: &wgpu::Device,
        bounds: [&wgpu::Buffer; 2],
        copies: [&wgpu::Buffer; 2],
        leaves: u32,
        ratio_percent: u32,
    ) -> Self {
        let blocks = (leaves - 1).div_ceil(256);
        let alloc = |size, extra| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::scene quality"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | extra,
                    mapped_at_creation: false,
                }
            )
        };
        let partial = alloc(u64::from(blocks) * 4, wgpu::BufferUsages::empty());
        let state = alloc(32, wgpu::BufferUsages::COPY_SRC);
        let args = alloc(36, wgpu::BufferUsages::INDIRECT);
        let params = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("molla::scene quality parameters"),
                contents: bytemuck::cast_slice(&[
                    leaves,
                    leaves * 2 - 1,
                    blocks,
                    (ratio_percent as f32 / 100.0).to_bits(),
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );
        let mut entries: Vec<_> = (0..5).map(|i| storage(i, i < 2)).collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene quality"),
            entries: &entries,
        });
        let copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene conditional commit"),
            entries: &[storage(0, true), storage(1, false)],
        });
        let bind = |layout, buffers: &[&wgpu::Buffer]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::scene quality"),
                layout,
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let pipeline = |layout, source: &'static str, entry| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("molla::scene quality"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::scene quality"),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::scene quality"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let shader = include_str!("../shaders/scene_quality.wgsl");
        let device_bytes = [&partial, &state, &args, &params]
            .iter()
            .map(|b| b.size())
            .sum();
        Self {
            measure: pipeline(&layout, shader, "measure"),
            decide: pipeline(&layout, shader, "decide"),
            capture: pipeline(&layout, shader, "capture"),
            commit: pipeline(
                &copy_layout,
                include_str!("../shaders/scene_rebuild_commit.wgsl"),
                "commit",
            ),
            metrics: bind(
                &layout,
                &[bounds[0], bounds[1], &partial, &state, &args, &params],
            ),
            copy: bind(&copy_layout, &copies),
            args,
            state,
            blocks,
            device_bytes,
        }
    }

    pub fn encode_measurement(&self, encoder: &mut wgpu::CommandEncoder, capture: bool) {
        for (pipeline, groups) in [
            (&self.measure, self.blocks),
            (if capture { &self.capture } else { &self.decide }, 1),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::scene quality"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.metrics, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }

    pub fn encode_commit(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::scene conditional commit"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.commit);
        pass.set_bind_group(0, &self.copy, &[]);
        pass.dispatch_workgroups_indirect(&self.args, 0);
    }
}
