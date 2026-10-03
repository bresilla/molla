//! Fixed-capacity Morton reordering with retained stable radix-sort bindings.

use wgpu::util::DeviceExt;

#[path = "scene_quality.rs"]
mod quality;
use quality::SceneQualityKernel;

pub struct SceneRebuildKernel {
    morton: wgpu::ComputePipeline,
    reorder: wgpu::ComputePipeline,
    count: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    spatial: wgpu::BindGroup,
    spatial_params: wgpu::BindGroup,
    radix: [wgpu::BindGroup; 2],
    radix_params: [wgpu::BindGroup; 4],
    alternate: wgpu::Buffer,
    destination: wgpu::Buffer,
    leaves: u32,
    payload_bytes: u64,
    quality: Option<SceneQualityKernel>,
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl SceneRebuildKernel {
    /// Inputs are lower/upper bounds, mutable instances and mutable topology; slots follow tree DFS order.
    pub fn new(
        device: &wgpu::Device,
        inputs: [&wgpu::Buffer; 4],
        slots: &[u32],
        root: u32,
    ) -> Self {
        Self::new_with_quality(device, inputs, slots, root, None)
    }

    /// Optional threshold is a percentage of the best normalized internal area since rebuilding.
    pub fn new_with_quality(
        device: &wgpu::Device,
        inputs: [&wgpu::Buffer; 4],
        slots: &[u32],
        root: u32,
        ratio_percent: Option<u32>,
    ) -> Self {
        let n = slots.len() as u32;
        let alloc = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        };
        let alternate = alloc("molla::scene reordered instances", u64::from(n) * 80);
        let keys = alloc("molla::scene Morton keys", u64::from(n) * 4);
        let values = alloc("molla::scene sorted leaf indices", u64::from(n) * 4);
        let capacity = u64::from(n.max(256));
        let scratch_keys = alloc("molla::scene radix keys", capacity * 4);
        let scratch_values = alloc("molla::scene radix values", capacity * 4);
        let bins = alloc("molla::scene radix bins", capacity.div_ceil(256) * 1024);
        let leaf_slots = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("molla::scene leaf destinations"),
                contents: bytemuck::cast_slice(slots),
                usage: wgpu::BufferUsages::STORAGE,
            }
        );
        let payload_bytes = [
            &alternate,
            &keys,
            &values,
            &scratch_keys,
            &scratch_values,
            &bins,
            &leaf_slots,
        ]
        .iter()
        .map(|b| b.size())
        .sum::<u64>()
            + 80;
        let spatial_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene reorder"),
            entries: &std::array::from_fn::<_, 8, _>(|i| storage(i as u32, i < 3 || i == 7)),
        });
        let radix_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene radix"),
            entries: &std::array::from_fn::<_, 5, _>(|i| storage(i as u32, i < 2)),
        });
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene reorder parameters"),
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
        let parameters = |words: [u32; 4]| {
            let buffer = molla_core::gpu_buffer_init!(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("molla::scene reorder immutable parameters"),
                    contents: bytemuck::cast_slice(&words),
                    usage: wgpu::BufferUsages::UNIFORM,
                }
            );
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::scene reorder parameters"),
                layout: &params_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        };
        let bind = |layout, buffers: &[&wgpu::Buffer]| {
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::scene reorder buffers"),
                layout,
                entries: &entries,
            })
        };
        let spatial = bind(
            &spatial_layout,
            &[
                inputs[0],
                inputs[1],
                inputs[2],
                &alternate,
                inputs[3],
                &keys,
                &values,
                &leaf_slots,
            ],
        );
        let radix = [
            bind(
                &radix_layout,
                &[&keys, &values, &scratch_keys, &scratch_values, &bins],
            ),
            bind(
                &radix_layout,
                &[&scratch_keys, &scratch_values, &keys, &values, &bins],
            ),
        ];
        let spatial_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::scene Morton reordering"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/scene_rebuild.wgsl").into()),
        });
        let radix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::scene shared stable radix"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../molla-compute-primitives/shaders/radix_sort.wgsl").into(),
            ),
        });
        let layout = |data| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::scene reorder"),
                bind_group_layouts: &[Some(data), Some(&params_layout)],
                immediate_size: 0,
            })
        };
        let spatial_pipeline_layout = layout(&spatial_layout);
        let radix_pipeline_layout = layout(&radix_layout);
        let pipeline = |layout, module, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::scene rebuild"),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let quality = ratio_percent.map(|ratio| {
            SceneQualityKernel::new(
                device,
                [inputs[0], inputs[1]],
                [&alternate, inputs[2]],
                n,
                ratio,
            )
        });
        let payload_bytes = payload_bytes + quality.as_ref().map_or(0, |q| q.device_bytes);
        Self {
            morton: pipeline(&spatial_pipeline_layout, &spatial_shader, "morton"),
            reorder: pipeline(&spatial_pipeline_layout, &spatial_shader, "reorder"),
            count: pipeline(&radix_pipeline_layout, &radix_shader, "count_block_bins"),
            scan: pipeline(&radix_pipeline_layout, &radix_shader, "scan_global_bins"),
            scatter: pipeline(&radix_pipeline_layout, &radix_shader, "scatter"),
            spatial,
            spatial_params: parameters([n, root, 0, 0]),
            radix,
            radix_params: std::array::from_fn(|i| {
                parameters([n, i as u32 * 8, n.div_ceil(256), 0])
            }),
            alternate,
            destination: inputs[2].clone(),
            leaves: n,
            payload_bytes,
            quality,
        }
    }

    pub fn device_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Borrowed GPU diagnostic words: baseline/current f32, check/rebuild counts and last decision.
    pub fn quality_state(&self) -> Option<&wgpu::Buffer> {
        self.quality.as_ref().map(|q| &q.state)
    }

    /// Capture the baseline after the caller refits the reordered hierarchy.
    pub fn finish_quality(&self, encoder: &mut wgpu::CommandEncoder) -> u64 {
        if let Some(quality) = &self.quality {
            quality.encode_measurement(encoder, true);
            2
        } else {
            0
        }
    }

    /// Encode periodic or GPU-gated key/sort/scatter/commit passes; no submission or queue writes.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) -> u64 {
        if let Some(quality) = &self.quality {
            quality.encode_measurement(encoder, false);
        }
        let mut dispatch = |pipeline, data, params, groups, offset| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::scene rebuild"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, data, &[]);
            pass.set_bind_group(1, params, &[]);
            if let Some(quality) = &self.quality {
                pass.dispatch_workgroups_indirect(&quality.args, offset);
            } else {
                pass.dispatch_workgroups(groups, 1, 1);
            }
        };
        dispatch(
            &self.morton,
            &self.spatial,
            &self.spatial_params,
            self.leaves.div_ceil(64),
            0,
        );
        for pass in 0..4 {
            dispatch(
                &self.count,
                &self.radix[pass % 2],
                &self.radix_params[pass],
                self.leaves.div_ceil(256),
                12,
            );
            dispatch(
                &self.scan,
                &self.radix[pass % 2],
                &self.radix_params[pass],
                1,
                24,
            );
            dispatch(
                &self.scatter,
                &self.radix[pass % 2],
                &self.radix_params[pass],
                self.leaves.div_ceil(256),
                12,
            );
        }
        dispatch(
            &self.reorder,
            &self.spatial,
            &self.spatial_params,
            self.leaves.div_ceil(64),
            0,
        );
        if let Some(quality) = &self.quality {
            quality.encode_commit(encoder);
            return 17;
        }
        encoder.copy_buffer_to_buffer(
            &self.alternate,
            0,
            &self.destination,
            0,
            u64::from(self.leaves) * 80,
        );
        14
    }
}
