//! Rust dispatch wrapper around `shaders/raycast.wgsl` — closest-hit
//! ray-cast against every collision shape in a [`Model`].
//!
//! Pattern mirrors [`super::XpbdKernel`]: one pipeline, four bind
//! groups (`model`, `rays`, `hits`, `uniforms`). Caller supplies the
//! ray-origin + ray-direction buffers and pre-allocated hit buffers;
//! we own only the kernel pipelines and a uniform buffer.
//!
//! Reference: Newton's `_src/geometry/raycast.py` `sensor_raycast_kernel`.

use molla_core::{Error, Result};
use molla_sim::Model;
use molla_storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    ray_count: u32,
    shape_count: u32,
    max_distance: f32,
    _pad: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Compute pipeline + bind-group layouts for the GPU ray-cast kernel.
/// One instance is reusable across many `dispatch` calls; the only
/// per-dispatch state is the uniform buffer.
pub struct RaycastKernel {
    pipeline: wgpu::ComputePipeline,
    layout_model: wgpu::BindGroupLayout,
    layout_rays: wgpu::BindGroupLayout,
    layout_hits: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl RaycastKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::raycast"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/raycast.wgsl").into()),
        });

        // 5 read-only storage buffers in the model bind group:
        //   shape_geom_kind, shape_geom_data, shape_body, shape_transform, body_q
        let layout_model = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::raycast model"),
            entries: &storage_entries(5, true),
        });
        // 2 read-only storage buffers: ray_origin, ray_direction.
        let layout_rays = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::raycast rays"),
            entries: &storage_entries(2, true),
        });
        // 4 read-write outputs: hit_t, hit_shape, hit_point, hit_normal.
        let layout_hits = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::raycast hits"),
            entries: &storage_entries(4, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::raycast uniforms"),
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
            label: Some("molla::raycast layout"),
            bind_group_layouts: &[&layout_model, &layout_rays, &layout_hits, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::raycast::cast_rays"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cast_rays"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_model,
            layout_rays,
            layout_hits,
            layout_uniforms,
        }
    }

    /// Cast `ray_count` rays against the model's *initial* body poses.
    /// For ray-casts against the live simulation state, prefer
    /// [`dispatch_with_body_q`](Self::dispatch_with_body_q).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        ray_count: u32,
        ray_origin: &Buffer<molla_math::Vec3>,
        ray_direction: &Buffer<molla_math::Vec3>,
        hit_t: &mut Buffer<molla_math::Real>,
        hit_shape: &mut Buffer<i32>,
        hit_point: &mut Buffer<molla_math::Vec3>,
        hit_normal: &mut Buffer<molla_math::Vec3>,
        max_distance: f32,
    ) -> Result<()> {
        let body_q = model
            .initial_body_q
            .device_buffer()
            .ok_or_else(missing_buf("initial_body_q"))?;
        self.dispatch_with_body_q(
            device,
            queue,
            model,
            body_q,
            ray_count,
            ray_origin,
            ray_direction,
            hit_t,
            hit_shape,
            hit_point,
            hit_normal,
            max_distance,
        )
    }

    /// Variant of [`dispatch`](Self::dispatch) that uses the caller's
    /// `body_q` buffer (typically `state.body_q.device_buffer()`)
    /// instead of `model.initial_body_q`. `ray_count` is the actual
    /// number of rays to test (≤ buffer capacity); the kernel only
    /// touches the first `ray_count` entries of every ray / hit
    /// buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_with_body_q(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        body_q: &wgpu::Buffer,
        ray_count: u32,
        ray_origin: &Buffer<molla_math::Vec3>,
        ray_direction: &Buffer<molla_math::Vec3>,
        hit_t: &mut Buffer<molla_math::Real>,
        hit_shape: &mut Buffer<i32>,
        hit_point: &mut Buffer<molla_math::Vec3>,
        hit_normal: &mut Buffer<molla_math::Vec3>,
        max_distance: f32,
    ) -> Result<()> {
        if ray_count == 0 {
            return Ok(());
        }
        if ray_count as usize > ray_origin.len() {
            return Err(Error::Gpu(format!(
                "raycast: ray_count {ray_count} exceeds buffer len {}",
                ray_origin.len()
            )));
        }

        let groups = ray_count.div_ceil(WORKGROUP_SIZE);

        let bg_model = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::raycast model bg"),
            layout: &self.layout_model,
            entries: &[
                buffer_entry(0, model.shape_geom_kind.device_buffer())?,
                buffer_entry(1, model.shape_geom_data.device_buffer())?,
                buffer_entry(2, model.shape_body.device_buffer())?,
                buffer_entry(3, model.shape_transform.device_buffer())?,
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: body_q.as_entire_binding(),
                },
            ],
        });
        let bg_rays = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::raycast rays bg"),
            layout: &self.layout_rays,
            entries: &[
                buffer_entry(0, ray_origin.device_buffer())?,
                buffer_entry(1, ray_direction.device_buffer())?,
            ],
        });
        let bg_hits = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::raycast hits bg"),
            layout: &self.layout_hits,
            entries: &[
                buffer_entry(0, hit_t.device_buffer())?,
                buffer_entry(1, hit_shape.device_buffer())?,
                buffer_entry(2, hit_point.device_buffer())?,
                buffer_entry(3, hit_normal.device_buffer())?,
            ],
        });

        // Uniforms — built per-dispatch.
        let uniforms = Uniforms {
            ray_count,
            shape_count: model.shape_count as u32,
            max_distance,
            _pad: 0.0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::raycast uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::raycast uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ── Encode + dispatch ──────────────────────────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::raycast encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::raycast::cast_rays"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_model, &[]);
            pass.set_bind_group(1, &bg_rays, &[]);
            pass.set_bind_group(2, &bg_hits, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        hit_t.mark_device_dirty();
        hit_shape.mark_device_dirty();
        hit_point.mark_device_dirty();
        hit_normal.mark_device_dirty();
        Ok(())
    }
}

// ── Helpers (mirror XpbdKernel's plumbing) ─────────────────────────

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf =
        buf.ok_or_else(|| Error::Gpu(format!("raycast binding {binding}: no device mirror")))?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn missing_buf(name: &'static str) -> impl FnOnce() -> Error {
    move || {
        Error::Gpu(format!(
            "missing device buffer for {name} — call upload_to_device first"
        ))
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
