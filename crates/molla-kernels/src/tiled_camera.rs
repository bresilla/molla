//! Rust dispatch wrappers for the GPU tiled-camera sensor — two
//! kernels, one pipeline each:
//!
//! * `camera_rays.wgsl` → [`CameraRaysKernel`] — pinhole ray
//!   generation per pixel per camera.
//! * `camera_trace.wgsl` → [`CameraTraceKernel`] — closest-hit
//!   primitive ray-cast + multi-channel writes (depth / colour /
//!   normal / albedo / shape index, gated by a bitmask uniform).
//!
//! Pattern mirrors [`super::RaycastKernel`]: dedicated bind-group
//! layouts per kernel, host-allocated uniform buffer per dispatch.
//!
//! Reference: Newton's `SensorTiledCamera` (Warp/CUDA); molla
//! re-implements the primitive kernels in WGSL and runs Lambertian
//! shading in-line so the output color channel is render-ready.

use molla_core::{Error, Result};
use molla_math::{Real, Transform, Vec3};
use molla_sim::Model;
use molla_storage::Buffer;

// ── Camera-rays kernel ────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RaysUniforms {
    width: u32,
    height: u32,
    camera_count: u32,
    fov_y: f32,
    aspect: f32,
    near: f32,
    _pad0: f32,
    _pad1: f32,
}

const RAYS_WG_X: u32 = 8;
const RAYS_WG_Y: u32 = 8;

pub struct CameraRaysKernel {
    pipeline: wgpu::ComputePipeline,
    layout_main: wgpu::BindGroupLayout,
}

impl CameraRaysKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::camera_rays"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/camera_rays.wgsl").into()),
        });
        let layout_main = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::camera_rays main"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::camera_rays layout"),
            bind_group_layouts: &[&layout_main],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::camera_rays::generate_rays"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("generate_rays"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_main,
        }
    }

    /// Generate `(camera_count × height × width)` rays, written into
    /// `ray_origin` / `ray_direction` row-major (camera-major outer).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_pose: &Buffer<Transform>,
        ray_origin: &mut Buffer<Vec3>,
        ray_direction: &mut Buffer<Vec3>,
        width: u32,
        height: u32,
        camera_count: u32,
        fov_y: f32,
        aspect: f32,
        near: f32,
    ) -> Result<()> {
        if camera_count == 0 || width == 0 || height == 0 {
            return Ok(());
        }

        let uniforms = RaysUniforms {
            width,
            height,
            camera_count,
            fov_y,
            aspect,
            near,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::camera_rays uniforms"),
            size: core::mem::size_of::<RaysUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera_rays bg"),
            layout: &self.layout_main,
            entries: &[
                buffer_entry(0, camera_pose.device_buffer(), "camera_pose")?,
                buffer_entry(1, ray_origin.device_buffer(), "ray_origin")?,
                buffer_entry(2, ray_direction.device_buffer(), "ray_direction")?,
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let groups_x = width.div_ceil(RAYS_WG_X);
        let groups_y = height.div_ceil(RAYS_WG_Y);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::camera_rays encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::camera_rays::generate_rays"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, camera_count);
        }
        queue.submit(std::iter::once(encoder.finish()));
        ray_origin.mark_device_dirty();
        ray_direction.mark_device_dirty();
        Ok(())
    }
}

// ── Camera-trace kernel ───────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TraceUniforms {
    width: u32,
    height: u32,
    camera_count: u32,
    shape_count: u32,
    max_distance: f32,
    channels: u32,
    _pad0: f32,
    _pad1: f32,
    ambient: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
}

const TRACE_WG_X: u32 = 8;
const TRACE_WG_Y: u32 = 8;

pub struct CameraTraceKernel {
    pipeline: wgpu::ComputePipeline,
    layout_model: wgpu::BindGroupLayout,
    layout_rays: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl CameraTraceKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::camera_trace"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/camera_trace.wgsl").into()),
        });
        // 8 read-only storage buffers in model bind group:
        // shape_geom_kind, shape_geom_data, shape_body, shape_transform,
        // body_q, shape_color, shape_metallic, shape_roughness.
        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::camera_trace model"),
            entries: &storage_entries(8, true),
        });
        let layout_rays = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::camera_trace rays"),
            entries: &storage_entries(2, true),
        });
        // 5 read-write outputs: depth, color, normal, albedo, shape_index.
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::camera_trace out"),
            entries: &storage_entries(5, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::camera_trace uniforms"),
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
            label: Some("molla::camera_trace layout"),
            bind_group_layouts: &[&layout_model, &layout_rays, &layout_out, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::camera_trace::trace"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_model,
            layout_rays,
            layout_out,
            layout_uniforms,
        }
    }

    /// Trace the per-pixel rays produced by [`CameraRaysKernel`] and
    /// fill the output channel buffers. Channels not present in the
    /// `channels` bitmask are left untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        body_q: &wgpu::Buffer,
        ray_origin: &Buffer<Vec3>,
        ray_direction: &Buffer<Vec3>,
        out_depth: &mut Buffer<Real>,
        out_color: &mut Buffer<u32>,
        out_normal: &mut Buffer<Vec3>,
        out_albedo: &mut Buffer<u32>,
        out_shape_index: &mut Buffer<u32>,
        width: u32,
        height: u32,
        camera_count: u32,
        max_distance: f32,
        channels: u32,
        ambient: Vec3,
        light_dir: Vec3,
        light_color: Vec3,
    ) -> Result<()> {
        if camera_count == 0 || width == 0 || height == 0 {
            return Ok(());
        }
        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera_trace model bg"),
            layout: &self.layout_model,
            entries: &[
                buffer_entry(0, model.shape_geom_kind.device_buffer(), "shape_geom_kind")?,
                buffer_entry(1, model.shape_geom_data.device_buffer(), "shape_geom_data")?,
                buffer_entry(2, model.shape_body.device_buffer(), "shape_body")?,
                buffer_entry(3, model.shape_transform.device_buffer(), "shape_transform")?,
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: body_q.as_entire_binding(),
                },
                buffer_entry(5, model.shape_color.device_buffer(), "shape_color")?,
                buffer_entry(6, model.shape_metallic.device_buffer(), "shape_metallic")?,
                buffer_entry(7, model.shape_roughness.device_buffer(), "shape_roughness")?,
            ],
        });
        let bg_rays = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera_trace rays bg"),
            layout: &self.layout_rays,
            entries: &[
                buffer_entry(0, ray_origin.device_buffer(), "ray_origin")?,
                buffer_entry(1, ray_direction.device_buffer(), "ray_direction")?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera_trace out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, out_depth.device_buffer(), "out_depth")?,
                buffer_entry(1, out_color.device_buffer(), "out_color")?,
                buffer_entry(2, out_normal.device_buffer(), "out_normal")?,
                buffer_entry(3, out_albedo.device_buffer(), "out_albedo")?,
                buffer_entry(4, out_shape_index.device_buffer(), "out_shape_index")?,
            ],
        });

        let uniforms = TraceUniforms {
            width,
            height,
            camera_count,
            shape_count: model.shape_count as u32,
            max_distance,
            channels,
            _pad0: 0.0,
            _pad1: 0.0,
            ambient: [ambient.x as f32, ambient.y as f32, ambient.z as f32, 0.0],
            light_dir: [
                light_dir.x as f32,
                light_dir.y as f32,
                light_dir.z as f32,
                0.0,
            ],
            light_color: [
                light_color.x as f32,
                light_color.y as f32,
                light_color.z as f32,
                0.0,
            ],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::camera_trace uniforms"),
            size: core::mem::size_of::<TraceUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera_trace uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups_x = width.div_ceil(TRACE_WG_X);
        let groups_y = height.div_ceil(TRACE_WG_Y);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::camera_trace encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::camera_trace::trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_rays, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, camera_count);
        }
        queue.submit(std::iter::once(encoder.finish()));
        out_depth.mark_device_dirty();
        out_color.mark_device_dirty();
        out_normal.mark_device_dirty();
        out_albedo.mark_device_dirty();
        out_shape_index.mark_device_dirty();
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
    name: &'static str,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "camera kernel binding {binding} ({name}): no device mirror"
        ))
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count).map(|i| storage_entry(i, read_only)).collect()
}
