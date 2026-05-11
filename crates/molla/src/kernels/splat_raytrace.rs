//! Rust dispatch wrapper around `shaders/splat_raytrace.wgsl` —
//! forward path of the Stage-44 differentiable 3D-Gaussian splat
//! raytrace renderer.
//!
//! v1 SCOPE: forward only. Walks all splats per pixel, keeps the
//! top-K closest hits in a per-thread mini-sort, then α-composites
//! them front-to-back. No reverse adjoint, no spherical harmonics.
//!
//! Pattern mirrors [`super::RaycastKernel`]: one pipeline, four
//! bind groups (`splats`, `rays`, `outputs`, `uniforms`).

use crate::core::{Error, Result};
use crate::math::{Real, Vec3};
use crate::storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    pixel_count: u32,
    splat_count: u32,
    max_distance: f32,
    min_alpha: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Compute pipeline + bind-group layouts for the GPU splat raytrace
/// kernel. One instance is reusable across many `dispatch` calls;
/// the only per-dispatch state is the uniform buffer.
pub struct SplatRaytraceKernel {
    pipeline: wgpu::ComputePipeline,
    layout_splats: wgpu::BindGroupLayout,
    layout_rays: wgpu::BindGroupLayout,
    layout_outputs: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl SplatRaytraceKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::splat_raytrace"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/splat_raytrace.wgsl").into(),
            ),
        });

        // 5 read-only storage buffers:
        //   splat_pos, splat_scale, splat_rot, splat_color, splat_opacity
        let layout_splats = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::splat_raytrace splats"),
            entries: &storage_entries(5, true),
        });
        // 2 read-only storage buffers: ray_origin, ray_direction.
        let layout_rays = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::splat_raytrace rays"),
            entries: &storage_entries(2, true),
        });
        // 2 read-write outputs: out_rgba, out_depth.
        let layout_outputs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::splat_raytrace outputs"),
            entries: &storage_entries(2, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::splat_raytrace uniforms"),
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
            label: Some("molla::splat_raytrace layout"),
            bind_group_layouts: &[
                &layout_splats,
                &layout_rays,
                &layout_outputs,
                &layout_uniforms,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::splat_raytrace::render"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("render"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_splats,
            layout_rays,
            layout_outputs,
            layout_uniforms,
        }
    }

    /// Dispatch the forward kernel.
    ///
    /// All splat / ray buffers must already have their device mirrors
    /// uploaded. The output buffers will be marked device-dirty so
    /// the caller can `download()` them after this call.
    ///
    /// `pixel_count` must be ≤ the ray buffers' length and ≤ the
    /// output buffers' length. `min_alpha` is the per-splat α
    /// threshold below which a hit is discarded (matches Newton's
    /// 1/255 ≈ 0.0039 default).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        splat_pos: &Buffer<Vec3>,
        splat_scale: &Buffer<Vec3>,
        splat_rot: &Buffer<crate::math::Quat>,
        splat_color: &Buffer<Vec3>,
        splat_opacity: &Buffer<Real>,
        ray_origin: &Buffer<Vec3>,
        ray_direction: &Buffer<Vec3>,
        out_rgba: &mut Buffer<Vec3>,
        out_depth: &mut Buffer<Real>,
        pixel_count: u32,
        max_distance: f32,
        min_alpha: f32,
    ) -> Result<()> {
        if pixel_count == 0 {
            return Ok(());
        }
        let splat_count = splat_pos.len() as u32;
        if splat_count == 0 {
            // Nothing to render — leave outputs as-is. Still ensure
            // the caller gets a clean download by zeroing per-pixel
            // outputs would require a separate clear pass; skipping
            // here is consistent with the raycast kernel's behaviour.
            return Ok(());
        }
        if pixel_count as usize > ray_origin.len() {
            return Err(Error::Gpu(format!(
                "splat_raytrace: pixel_count {pixel_count} exceeds ray buffer len {}",
                ray_origin.len()
            )));
        }
        if pixel_count as usize > out_rgba.len() || pixel_count as usize > out_depth.len() {
            return Err(Error::Gpu(format!(
                "splat_raytrace: pixel_count {pixel_count} exceeds output buffer len"
            )));
        }
        // Sanity-check splat array shapes match.
        if splat_scale.len() != splat_count as usize
            || splat_rot.len() != splat_count as usize
            || splat_color.len() != splat_count as usize
            || splat_opacity.len() != splat_count as usize
        {
            return Err(Error::Gpu(
                "splat_raytrace: splat SoA buffers have mismatched lengths".into(),
            ));
        }

        let groups = pixel_count.div_ceil(WORKGROUP_SIZE);

        let bg_splats = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace splats bg"),
            layout: &self.layout_splats,
            entries: &[
                buffer_entry(0, splat_pos.device_buffer())?,
                buffer_entry(1, splat_scale.device_buffer())?,
                buffer_entry(2, splat_rot.device_buffer())?,
                buffer_entry(3, splat_color.device_buffer())?,
                buffer_entry(4, splat_opacity.device_buffer())?,
            ],
        });
        let bg_rays = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace rays bg"),
            layout: &self.layout_rays,
            entries: &[
                buffer_entry(0, ray_origin.device_buffer())?,
                buffer_entry(1, ray_direction.device_buffer())?,
            ],
        });
        let bg_outputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace outputs bg"),
            layout: &self.layout_outputs,
            entries: &[
                buffer_entry(0, out_rgba.device_buffer())?,
                buffer_entry(1, out_depth.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            pixel_count,
            splat_count,
            max_distance,
            min_alpha,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::splat_raytrace encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::splat_raytrace::render"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_splats, &[]);
            pass.set_bind_group(1, &bg_rays, &[]);
            pass.set_bind_group(2, &bg_outputs, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        out_rgba.mark_device_dirty();
        out_depth.mark_device_dirty();
        Ok(())
    }
}

// ── Helpers (mirror RaycastKernel) ─────────────────────────────────

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!("splat_raytrace binding {binding}: no device mirror"))
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
