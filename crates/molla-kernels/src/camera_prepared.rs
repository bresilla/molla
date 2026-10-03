use super::*;
use std::num::NonZeroU64;

struct UniformSlots {
    buffer: wgpu::Buffer,
    stride: u32,
    count: u32,
    size: u64,
}

impl UniformSlots {
    fn new(device: &wgpu::Device, size: u64, count: u32) -> Result<Self> {
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1) as u64;
        let stride = size.div_ceil(alignment) * alignment;
        let bytes = stride * count as u64;
        if count == 0
            || count > 1024
            || bytes > device.limits().max_buffer_size
            || bytes > u32::MAX as u64
        {
            return Err(Error::Gpu("camera: invalid uniform slot capacity".into()));
        }
        Ok(Self {
            buffer: molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("molla::camera parameter slots"),
                    size: bytes,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            ),
            stride: stride as u32,
            count,
            size,
        })
    }

    fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: NonZeroU64::new(self.size),
        })
    }

    fn write(&self, queue: &wgpu::Queue, slot: u32, bytes: &[u8]) -> Result<u32> {
        if slot >= self.count || bytes.len() as u64 != self.size {
            return Err(Error::Gpu("camera: invalid parameter slot or size".into()));
        }
        let offset = self.stride * slot;
        molla_core::gpu_write_buffer!(queue, &self.buffer, offset as u64, bytes);
        Ok(offset)
    }
}

/// Camera ray-generation parameters for one aperture sample.
#[derive(Clone, Copy, Debug)]
pub struct CameraRaysParams {
    pub width: u32,
    pub height: u32,
    pub camera_count: u32,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub lens_radius: f32,
    pub focal_distance: f32,
    pub aperture_offset: [f32; 2],
}

/// Camera trace/shading parameters for one channel selection.
#[derive(Clone, Copy, Debug)]
pub struct CameraTraceParams {
    pub width: u32,
    pub height: u32,
    pub camera_count: u32,
    pub shape_count: u32,
    pub max_distance: f32,
    pub channels: u32,
    pub ambient: Vec3,
    pub light_dir: Vec3,
    pub light_color: Vec3,
}

/// Retained pose/ray bindings with independently writable aperture parameter slots.
pub struct CameraRaysBindings {
    buffers: [wgpu::Buffer; 3],
    group: wgpu::BindGroup,
    uniforms: UniformSlots,
}

impl CameraRaysBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 3]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

/// Retained scene/ray/channel bindings with independently writable trace parameter slots.
pub struct CameraTraceBindings {
    buffers: [wgpu::Buffer; 15],
    groups: [wgpu::BindGroup; 4],
    uniforms: UniformSlots,
}

impl CameraTraceBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 15]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

/// Shading of shared scene-query results into the five camera channels.
pub struct CameraShadeKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct CameraShadeBindings {
    buffers: [wgpu::Buffer; 9],
    group: wgpu::BindGroup,
    uniforms: UniformSlots,
}

impl CameraShadeBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 9]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl CameraShadeKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let entries: Vec<_> = (0..10)
            .map(|binding| {
                if binding < 9 {
                    storage_entry(binding, binding < 4)
                } else {
                    wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                }
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::camera shade"),
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::camera shade"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/camera_shade.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::camera shade"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::camera shade"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("shade"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Buffers: query distance/shape/normal, shape colors, output depth/color/normal/albedo/ID.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 9],
    ) -> Result<CameraShadeBindings> {
        let uniforms = UniformSlots::new(device, size_of::<TraceUniforms>() as u64, 2)?;
        let mut entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: uniforms.binding(),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera shade"),
            layout: &self.layout,
            entries: &entries,
        });
        Ok(CameraShadeBindings {
            buffers: buffers.map(Clone::clone),
            group,
            uniforms,
        })
    }

    /// Encode with slot 0/1. Submit before changing parameters already used by pending
    /// commands; caller validates buffers/dimensions and tracks output authority.
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &CameraShadeBindings,
        slot: u32,
        params: CameraTraceParams,
    ) -> Result<()> {
        if params.width == 0 || params.height == 0 || params.camera_count == 0 {
            return Ok(());
        }
        let rgba = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
        let uniforms = TraceUniforms {
            width: params.width,
            height: params.height,
            camera_count: params.camera_count,
            shape_count: params.shape_count,
            max_distance: params.max_distance,
            channels: params.channels,
            _pad0: 0.0,
            _pad1: 0.0,
            ambient: rgba(params.ambient),
            light_dir: rgba(params.light_dir),
            light_color: rgba(params.light_color),
        };
        let offset = bindings
            .uniforms
            .write(queue, slot, bytemuck::bytes_of(&uniforms))?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::camera shade"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.group, &[offset]);
        pass.dispatch_workgroups(
            params.width.div_ceil(8),
            params.height.div_ceil(8),
            params.camera_count,
        );
        Ok(())
    }
}

impl CameraRaysKernel {
    /// Buffers: poses, ray origins, ray directions, all on `device`.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 3],
        slots: u32,
    ) -> Result<CameraRaysBindings> {
        let uniforms = UniformSlots::new(device, size_of::<RaysUniforms>() as u64, slots)?;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera rays prepared"),
            layout: &self.layout_main,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniforms.binding(),
                },
            ],
        });
        Ok(CameraRaysBindings {
            buffers: buffers.map(Clone::clone),
            group,
            uniforms,
        })
    }

    /// Encode without submission. Do not overwrite a slot before submitting its
    /// pending commands. Caller validates dimensions/indices and tracks output authority.
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &CameraRaysBindings,
        slot: u32,
        params: CameraRaysParams,
    ) -> Result<()> {
        if params.width == 0 || params.height == 0 || params.camera_count == 0 {
            return Ok(());
        }
        let uniforms = RaysUniforms {
            width: params.width,
            height: params.height,
            camera_count: params.camera_count,
            fov_y: params.fov_y,
            aspect: params.aspect,
            near: params.near,
            lens_radius: params.lens_radius,
            focal_distance: params.focal_distance,
            aperture_offset: params.aperture_offset,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let offset = bindings
            .uniforms
            .write(queue, slot, bytemuck::bytes_of(&uniforms))?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::camera rays"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.group, &[offset]);
        pass.dispatch_workgroups(
            params.width.div_ceil(RAYS_WG_X),
            params.height.div_ceil(RAYS_WG_Y),
            params.camera_count,
        );
        Ok(())
    }
}

impl CameraTraceKernel {
    /// Buffers: shape kind/data/body/pose, body poses, color/metallic/roughness,
    /// ray origins/directions, output depth/color/normal/albedo/shape ID, all on `device`.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 15],
        slots: u32,
    ) -> Result<CameraTraceBindings> {
        let uniforms = UniformSlots::new(device, size_of::<TraceUniforms>() as u64, slots)?;
        let make_group = |layout, range: std::ops::Range<usize>| {
            let entries: Vec<_> = buffers[range]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::camera trace prepared"),
                layout,
                entries: &entries,
            })
        };
        let groups = [
            make_group(&self.layout_model, 0..8),
            make_group(&self.layout_rays, 8..10),
            make_group(&self.layout_out, 10..15),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::camera trace parameter slots"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.binding(),
                }],
            }),
        ];
        Ok(CameraTraceBindings {
            buffers: buffers.map(Clone::clone),
            groups,
            uniforms,
        })
    }

    /// Encode without submission. Do not overwrite a slot before submitting its
    /// pending commands. Caller validates dimensions/indices and tracks output authority.
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &CameraTraceBindings,
        slot: u32,
        params: CameraTraceParams,
    ) -> Result<()> {
        if params.width == 0 || params.height == 0 || params.camera_count == 0 {
            return Ok(());
        }
        let rgba = |v: Vec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
        let uniforms = TraceUniforms {
            width: params.width,
            height: params.height,
            camera_count: params.camera_count,
            shape_count: params.shape_count,
            max_distance: params.max_distance,
            channels: params.channels,
            _pad0: 0.0,
            _pad1: 0.0,
            ambient: rgba(params.ambient),
            light_dir: rgba(params.light_dir),
            light_color: rgba(params.light_color),
        };
        let offset = bindings
            .uniforms
            .write(queue, slot, bytemuck::bytes_of(&uniforms))?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::camera trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for i in 0..3 {
            pass.set_bind_group(i as u32, &bindings.groups[i], &[]);
        }
        pass.set_bind_group(3, &bindings.groups[3], &[offset]);
        pass.dispatch_workgroups(
            params.width.div_ceil(TRACE_WG_X),
            params.height.div_ceil(TRACE_WG_Y),
            params.camera_count,
        );
        Ok(())
    }
}
