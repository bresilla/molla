//! Rust dispatch wrapper around `shaders/splat_raytrace.wgsl` (forward)
//! and `shaders/splat_raytrace_backward.wgsl` (Stage 44 v2 reverse
//! adjoint), plus `shaders/splat_raytrace_sh.wgsl` /
//! `shaders/splat_raytrace_sh_backward.wgsl` (Stage 44 v3 view-dep SH).
//!
//! v1 (`dispatch`) walks all splats per pixel, keeps the top-K closest
//! hits in a per-thread mini-sort, and α-composites them front-to-back.
//!
//! v2 (`backward`) replays that same deterministic top-K hit list per
//! pixel and propagates `dL/dC_out` back through the α-composite
//! recurrence, ρ = exp(-½m²), and the canonical-frame chain rule into
//! per-splat (position, scale, rotation, color, opacity) gradients.
//!
//! v3 (`dispatch_sh` / `backward_sh`) extends both passes to evaluate
//! per-splat view-dependent colour from 9 spherical-harmonic
//! coefficient triples (bands 0..=2). The SH coefficient buffer is
//! laid out interleaved per-coefficient: `splat_sh[i*9 + k]` is the
//! k-th coefficient of splat `i` (band-major; see
//! `crate::geometry::GaussianSplatSh3d`). Each coefficient is a vec3
//! padded to vec4 (the buffer is `Buffer<Vec3>`-style stride-4). The
//! backward pass scatters per-coefficient gradients via the same
//! CAS-loop f32 atomic-add as v2.
//!
//! Pattern mirrors [`super::RaycastKernel`]: one pipeline per pass,
//! four bind groups (`splats`, `rays/upstream`, `outputs`,
//! `uniforms`). The backward pipeline reuses the forward's `splats`
//! and `uniforms` layouts; the `rays` bind group is widened to also
//! carry the per-pixel `dL/dC_out` upstream gradient and the
//! `outputs` bind group binds the per-splat atomic-u32 scatter
//! buffers (CAS-loop f32 atomic-add — WGSL has no `atomic<f32>`).

use crate::core::{Error, Result};
use crate::geometry::SH_BAND_2_COUNT;
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
/// kernel. One instance is reusable across many `dispatch` /
/// `backward` calls; the only per-dispatch state is the uniform buffer.
pub struct SplatRaytraceKernel {
    pipeline: wgpu::ComputePipeline,
    layout_splats: wgpu::BindGroupLayout,
    layout_rays: wgpu::BindGroupLayout,
    layout_outputs: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,

    // Stage 44 v2 — reverse adjoint pipeline + layouts.
    pipeline_backward: wgpu::ComputePipeline,
    layout_back_rays: wgpu::BindGroupLayout,
    layout_back_outputs: wgpu::BindGroupLayout,

    // Stage 44 v3 — view-dep SH forward + backward.
    pipeline_sh_forward: wgpu::ComputePipeline,
    pipeline_sh_backward: wgpu::ComputePipeline,
    // The SH pipelines have the same outer bind-group structure as v1/v2;
    // only the splat-side layout differs (one 9-stride SH coeff buffer
    // replaces the constant diffuse RGB buffer). We share the same
    // [`storage_entries(5, true)`] layout — both layouts hold 5 read-only
    // storage buffers in identical positions, so `layout_splats` is reused.
}

/// Per-splat gradient buffers returned by [`SplatRaytraceKernel::backward`].
/// Each `Vec` has length `splat_count`. Quaternion gradient is the
/// natural 4-component derivative `dL/dq` (caller projects onto the
/// tangent of the unit sphere if needed).
pub struct SplatGradients {
    pub d_pos: Vec<Vec3>,
    pub d_scale: Vec<Vec3>,
    pub d_rotation: Vec<crate::math::Quat>,
    pub d_color: Vec<Vec3>,
    pub d_opacity: Vec<Real>,
}

/// Per-splat gradient buffers returned by
/// [`SplatRaytraceKernel::backward_sh`].
///
/// Geometry / opacity gradients are identical to [`SplatGradients`];
/// the colour gradient is replaced by 9 per-band SH coefficient
/// triples per splat (`d_sh.len() == splat_count`, each entry has
/// 9 `Vec3` slots). Quaternion gradient is the natural 4-component
/// derivative.
pub struct SplatShGradients {
    pub d_pos: Vec<Vec3>,
    pub d_scale: Vec<Vec3>,
    pub d_rotation: Vec<crate::math::Quat>,
    pub d_sh: Vec<[Vec3; SH_BAND_2_COUNT]>,
    pub d_opacity: Vec<Real>,
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

        // ── Stage 44 v2 — backward pipeline ──────────────────────
        let shader_back = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::splat_raytrace_backward"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/splat_raytrace_backward.wgsl").into(),
            ),
        });
        // 3 read-only storage buffers: ray_origin, ray_direction, dL/dC.
        let layout_back_rays = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::splat_raytrace_backward rays"),
            entries: &storage_entries(3, true),
        });
        // 5 atomic-u32 read-write outputs: grad_pos / scale / rot /
        // color / opacity.
        let layout_back_outputs = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::splat_raytrace_backward outputs"),
                entries: &storage_entries(5, false),
            },
        );
        let pipeline_layout_back = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("molla::splat_raytrace_backward layout"),
                bind_group_layouts: &[
                    &layout_splats,
                    &layout_back_rays,
                    &layout_back_outputs,
                    &layout_uniforms,
                ],
                push_constant_ranges: &[],
            },
        );
        let pipeline_backward = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::splat_raytrace::backward"),
                layout: Some(&pipeline_layout_back),
                module: &shader_back,
                entry_point: Some("backward"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        // ── Stage 44 v3 — SH forward + backward pipelines ───────
        // Bind-group layouts are bit-identical to v1/v2 (same number
        // of read-only storage buffers in the splat group, same rays
        // group widths, same outputs widths) — only the WGSL bodies
        // and the splat-color binding interpretation differ. We
        // reuse the existing layouts directly.
        let shader_sh = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::splat_raytrace_sh"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/splat_raytrace_sh.wgsl").into(),
            ),
        });
        let pipeline_layout_sh = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::splat_raytrace_sh layout"),
            bind_group_layouts: &[
                &layout_splats,
                &layout_rays,
                &layout_outputs,
                &layout_uniforms,
            ],
            push_constant_ranges: &[],
        });
        let pipeline_sh_forward = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::splat_raytrace_sh::render"),
            layout: Some(&pipeline_layout_sh),
            module: &shader_sh,
            entry_point: Some("render"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let shader_sh_back = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::splat_raytrace_sh_backward"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/splat_raytrace_sh_backward.wgsl").into(),
            ),
        });
        let pipeline_layout_sh_back = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("molla::splat_raytrace_sh_backward layout"),
                bind_group_layouts: &[
                    &layout_splats,
                    &layout_back_rays,
                    &layout_back_outputs,
                    &layout_uniforms,
                ],
                push_constant_ranges: &[],
            },
        );
        let pipeline_sh_backward = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("molla::splat_raytrace_sh::backward"),
                layout: Some(&pipeline_layout_sh_back),
                module: &shader_sh_back,
                entry_point: Some("backward"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        );

        Self {
            pipeline,
            layout_splats,
            layout_rays,
            layout_outputs,
            layout_uniforms,
            pipeline_backward,
            layout_back_rays,
            layout_back_outputs,
            pipeline_sh_forward,
            pipeline_sh_backward,
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

    /// Stage 44 v2 — reverse adjoint dispatch.
    ///
    /// Given the same splat / ray inputs that produced the forward
    /// image and a per-pixel `dL/dC_out` (`d_loss_d_rgb`, padded as
    /// `Vec3` in a `Buffer<Vec3>`), scatter per-splat gradients on
    /// `d_pos`, `d_scale`, `d_rotation`, `d_color`, `d_opacity`.
    /// Returns a [`SplatGradients`] holding the reduced (downloaded)
    /// gradients on the host.
    ///
    /// The kernel re-runs the forward top-K insertion sort per pixel
    /// (deterministic for fixed splat/ray inputs) and walks the hits
    /// back-to-front using the standard 3DGS recurrence. Atomic-u32
    /// CAS adds are used to scatter per-splat scalars — exactly the
    /// same idiom as `fem_assembly.wgsl`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
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
        d_loss_d_rgb: &Buffer<Vec3>,
        pixel_count: u32,
        max_distance: f32,
        min_alpha: f32,
    ) -> Result<SplatGradients> {
        let splat_count = splat_pos.len();
        if splat_count == 0 || pixel_count == 0 {
            return Ok(SplatGradients {
                d_pos: vec![Vec3::ZERO; splat_count],
                d_scale: vec![Vec3::ZERO; splat_count],
                d_rotation: vec![crate::math::Quat::from_xyzw(0.0, 0.0, 0.0, 0.0); splat_count],
                d_color: vec![Vec3::ZERO; splat_count],
                d_opacity: vec![0.0; splat_count],
            });
        }
        if pixel_count as usize > ray_origin.len()
            || pixel_count as usize > d_loss_d_rgb.len()
        {
            return Err(Error::Gpu(format!(
                "splat_raytrace::backward: pixel_count {pixel_count} exceeds ray / dL buffer len"
            )));
        }
        if splat_scale.len() != splat_count
            || splat_rot.len() != splat_count
            || splat_color.len() != splat_count
            || splat_opacity.len() != splat_count
        {
            return Err(Error::Gpu(
                "splat_raytrace::backward: splat SoA buffers have mismatched lengths".into(),
            ));
        }

        // ── Allocate gradient buffers. Each is `splat_count`
        // entries; vec3-padded buffers are stride-4 to match the
        // forward splat SoA. We zero-init via `write_buffer` since
        // wgpu has no native "clear to zero" op for storage buffers
        // (we'd need a separate clear shader to do it on-GPU).
        let pad4_bytes = (splat_count * 4) as u64 * 4;
        let s_bytes = (splat_count) as u64 * 4;
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let grad_pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace::grad_pos"),
            size: pad4_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let grad_scale = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace::grad_scale"),
            size: pad4_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let grad_rot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace::grad_rot"),
            size: pad4_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let grad_color = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace::grad_color"),
            size: pad4_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let grad_opacity = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace::grad_opacity"),
            size: s_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        // Zero-init each gradient buffer.
        let zeros_pad4 = vec![0u32; splat_count * 4];
        let zeros_s = vec![0u32; splat_count];
        queue.write_buffer(&grad_pos, 0, bytemuck::cast_slice(&zeros_pad4));
        queue.write_buffer(&grad_scale, 0, bytemuck::cast_slice(&zeros_pad4));
        queue.write_buffer(&grad_rot, 0, bytemuck::cast_slice(&zeros_pad4));
        queue.write_buffer(&grad_color, 0, bytemuck::cast_slice(&zeros_pad4));
        queue.write_buffer(&grad_opacity, 0, bytemuck::cast_slice(&zeros_s));

        let groups = pixel_count.div_ceil(WORKGROUP_SIZE);

        let bg_splats = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace::backward splats bg"),
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
            label: Some("molla::splat_raytrace::backward rays bg"),
            layout: &self.layout_back_rays,
            entries: &[
                buffer_entry(0, ray_origin.device_buffer())?,
                buffer_entry(1, ray_direction.device_buffer())?,
                buffer_entry(2, d_loss_d_rgb.device_buffer())?,
            ],
        });
        let bg_outputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace::backward outputs bg"),
            layout: &self.layout_back_outputs,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grad_pos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_scale.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_rot.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_color.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_opacity.as_entire_binding() },
            ],
        });
        let uniforms = Uniforms {
            pixel_count,
            splat_count: splat_count as u32,
            max_distance,
            min_alpha,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace::backward uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace::backward uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::splat_raytrace::backward encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::splat_raytrace::backward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_backward);
            pass.set_bind_group(0, &bg_splats, &[]);
            pass.set_bind_group(1, &bg_rays, &[]);
            pass.set_bind_group(2, &bg_outputs, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        // ── Stage the four padded-vec3 gradient buffers + opacity
        // back to host. One readback per buffer (each blocking on
        // its own staging buffer).
        let stage_pad4 = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: pad4_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let stage_s = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: s_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let staging_pos = stage_pad4("molla::splat_raytrace::backward grad_pos staging");
        let staging_scale = stage_pad4("molla::splat_raytrace::backward grad_scale staging");
        let staging_rot = stage_pad4("molla::splat_raytrace::backward grad_rot staging");
        let staging_color = stage_pad4("molla::splat_raytrace::backward grad_color staging");
        let staging_opacity = stage_s("molla::splat_raytrace::backward grad_opacity staging");
        encoder.copy_buffer_to_buffer(&grad_pos, 0, &staging_pos, 0, pad4_bytes);
        encoder.copy_buffer_to_buffer(&grad_scale, 0, &staging_scale, 0, pad4_bytes);
        encoder.copy_buffer_to_buffer(&grad_rot, 0, &staging_rot, 0, pad4_bytes);
        encoder.copy_buffer_to_buffer(&grad_color, 0, &staging_color, 0, pad4_bytes);
        encoder.copy_buffer_to_buffer(&grad_opacity, 0, &staging_opacity, 0, s_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let read_pad4 = |staging: &wgpu::Buffer| -> Result<Vec<Vec3>> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("recv: {e}")))?
                .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&mapped);
            let n = splat_count;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let b = i * 4;
                out.push(Vec3::new(f[b] as Real, f[b + 1] as Real, f[b + 2] as Real));
            }
            drop(mapped);
            staging.unmap();
            Ok(out)
        };
        let read_pad4_quat = |staging: &wgpu::Buffer| -> Result<Vec<crate::math::Quat>> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("recv: {e}")))?
                .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&mapped);
            let n = splat_count;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let b = i * 4;
                out.push(crate::math::Quat::from_xyzw(
                    f[b] as Real,
                    f[b + 1] as Real,
                    f[b + 2] as Real,
                    f[b + 3] as Real,
                ));
            }
            drop(mapped);
            staging.unmap();
            Ok(out)
        };
        let read_s = |staging: &wgpu::Buffer| -> Result<Vec<Real>> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("recv: {e}")))?
                .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&mapped);
            let out = f.iter().map(|x| *x as Real).collect::<Vec<_>>();
            drop(mapped);
            staging.unmap();
            Ok(out)
        };

        let d_pos = read_pad4(&staging_pos)?;
        let d_scale = read_pad4(&staging_scale)?;
        let d_rotation = read_pad4_quat(&staging_rot)?;
        let d_color = read_pad4(&staging_color)?;
        let d_opacity = read_s(&staging_opacity)?;

        Ok(SplatGradients {
            d_pos,
            d_scale,
            d_rotation,
            d_color,
            d_opacity,
        })
    }

    // ── Stage 44 v3 — SH-aware forward ──────────────────────────────
    /// SH-aware forward kernel (Stage 44 v3). Identical to
    /// [`Self::dispatch`] except `splat_color` is replaced by
    /// `splat_sh`, a flat buffer of `splat_count * 9` `Vec3` (one per
    /// SH coefficient, band-major; see
    /// [`crate::geometry::GaussianSplatSh3d`]). The colour read by
    /// the kernel is `Σ_k sh[i*9+k] * Y_k(rd)` where `Y_k` is the
    /// SH basis and `rd` is the pixel's view direction.
    ///
    /// Buffer lengths:
    ///   - `splat_pos`, `splat_scale`, `splat_rot`, `splat_opacity`:
    ///     `splat_count` entries each.
    ///   - `splat_sh`: `splat_count * 9` `Vec3` entries.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_sh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        splat_pos: &Buffer<Vec3>,
        splat_scale: &Buffer<Vec3>,
        splat_rot: &Buffer<crate::math::Quat>,
        splat_sh: &Buffer<Vec3>,
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
            return Ok(());
        }
        if pixel_count as usize > ray_origin.len() {
            return Err(Error::Gpu(format!(
                "splat_raytrace_sh: pixel_count {pixel_count} exceeds ray buffer len {}",
                ray_origin.len()
            )));
        }
        if pixel_count as usize > out_rgba.len() || pixel_count as usize > out_depth.len() {
            return Err(Error::Gpu(format!(
                "splat_raytrace_sh: pixel_count {pixel_count} exceeds output buffer len"
            )));
        }
        let expected_sh_len = splat_count as usize * SH_BAND_2_COUNT;
        if splat_scale.len() != splat_count as usize
            || splat_rot.len() != splat_count as usize
            || splat_opacity.len() != splat_count as usize
        {
            return Err(Error::Gpu(
                "splat_raytrace_sh: splat SoA buffers have mismatched lengths".into(),
            ));
        }
        if splat_sh.len() != expected_sh_len {
            return Err(Error::Gpu(format!(
                "splat_raytrace_sh: sh buffer len {} != {} * {} = {}",
                splat_sh.len(),
                splat_count,
                SH_BAND_2_COUNT,
                expected_sh_len
            )));
        }

        let groups = pixel_count.div_ceil(WORKGROUP_SIZE);

        let bg_splats = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace_sh splats bg"),
            layout: &self.layout_splats,
            entries: &[
                buffer_entry(0, splat_pos.device_buffer())?,
                buffer_entry(1, splat_scale.device_buffer())?,
                buffer_entry(2, splat_rot.device_buffer())?,
                buffer_entry(3, splat_sh.device_buffer())?,
                buffer_entry(4, splat_opacity.device_buffer())?,
            ],
        });
        let bg_rays = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace_sh rays bg"),
            layout: &self.layout_rays,
            entries: &[
                buffer_entry(0, ray_origin.device_buffer())?,
                buffer_entry(1, ray_direction.device_buffer())?,
            ],
        });
        let bg_outputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace_sh outputs bg"),
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
            label: Some("molla::splat_raytrace_sh uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace_sh uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::splat_raytrace_sh encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::splat_raytrace_sh::render"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_sh_forward);
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

    // ── Stage 44 v3 — SH-aware reverse adjoint ──────────────────────
    /// SH-aware backward kernel. Mirrors [`Self::backward`]; the
    /// colour-gradient output is replaced by a per-coefficient SH
    /// gradient (`d_sh.len() == splat_count`, each entry holds 9
    /// `Vec3`). The per-coefficient gradient is the per-pixel
    /// `dL/dC * α_i * T_i * Y_k(rd)` summed over all pixels that hit
    /// splat `i`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_sh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        splat_pos: &Buffer<Vec3>,
        splat_scale: &Buffer<Vec3>,
        splat_rot: &Buffer<crate::math::Quat>,
        splat_sh: &Buffer<Vec3>,
        splat_opacity: &Buffer<Real>,
        ray_origin: &Buffer<Vec3>,
        ray_direction: &Buffer<Vec3>,
        d_loss_d_rgb: &Buffer<Vec3>,
        pixel_count: u32,
        max_distance: f32,
        min_alpha: f32,
    ) -> Result<SplatShGradients> {
        let splat_count = splat_pos.len();
        if splat_count == 0 || pixel_count == 0 {
            return Ok(SplatShGradients {
                d_pos: vec![Vec3::ZERO; splat_count],
                d_scale: vec![Vec3::ZERO; splat_count],
                d_rotation: vec![crate::math::Quat::from_xyzw(0.0, 0.0, 0.0, 0.0); splat_count],
                d_sh: vec![[Vec3::ZERO; SH_BAND_2_COUNT]; splat_count],
                d_opacity: vec![0.0; splat_count],
            });
        }
        if pixel_count as usize > ray_origin.len()
            || pixel_count as usize > d_loss_d_rgb.len()
        {
            return Err(Error::Gpu(format!(
                "splat_raytrace_sh::backward: pixel_count {pixel_count} exceeds ray / dL buffer len"
            )));
        }
        let expected_sh_len = splat_count * SH_BAND_2_COUNT;
        if splat_scale.len() != splat_count
            || splat_rot.len() != splat_count
            || splat_opacity.len() != splat_count
        {
            return Err(Error::Gpu(
                "splat_raytrace_sh::backward: splat SoA buffers have mismatched lengths".into(),
            ));
        }
        if splat_sh.len() != expected_sh_len {
            return Err(Error::Gpu(format!(
                "splat_raytrace_sh::backward: sh buffer len {} != {} * {} = {}",
                splat_sh.len(),
                splat_count,
                SH_BAND_2_COUNT,
                expected_sh_len
            )));
        }

        // ── Allocate gradient buffers. ──────────────────────────────
        let pad4_bytes = (splat_count * 4) as u64 * 4;
        let s_bytes = (splat_count) as u64 * 4;
        let sh_bytes = (splat_count * SH_BAND_2_COUNT * 4) as u64 * 4;
        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let grad_pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace_sh::grad_pos"),
            size: pad4_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let grad_scale = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace_sh::grad_scale"),
            size: pad4_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let grad_rot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace_sh::grad_rot"),
            size: pad4_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let grad_sh = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace_sh::grad_sh"),
            size: sh_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let grad_opacity = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace_sh::grad_opacity"),
            size: s_bytes,
            usage: storage_usage,
            mapped_at_creation: false,
        });
        let zeros_pad4 = vec![0u32; splat_count * 4];
        let zeros_s = vec![0u32; splat_count];
        let zeros_sh = vec![0u32; splat_count * SH_BAND_2_COUNT * 4];
        queue.write_buffer(&grad_pos, 0, bytemuck::cast_slice(&zeros_pad4));
        queue.write_buffer(&grad_scale, 0, bytemuck::cast_slice(&zeros_pad4));
        queue.write_buffer(&grad_rot, 0, bytemuck::cast_slice(&zeros_pad4));
        queue.write_buffer(&grad_sh, 0, bytemuck::cast_slice(&zeros_sh));
        queue.write_buffer(&grad_opacity, 0, bytemuck::cast_slice(&zeros_s));

        let groups = pixel_count.div_ceil(WORKGROUP_SIZE);

        let bg_splats = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace_sh::backward splats bg"),
            layout: &self.layout_splats,
            entries: &[
                buffer_entry(0, splat_pos.device_buffer())?,
                buffer_entry(1, splat_scale.device_buffer())?,
                buffer_entry(2, splat_rot.device_buffer())?,
                buffer_entry(3, splat_sh.device_buffer())?,
                buffer_entry(4, splat_opacity.device_buffer())?,
            ],
        });
        let bg_rays = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace_sh::backward rays bg"),
            layout: &self.layout_back_rays,
            entries: &[
                buffer_entry(0, ray_origin.device_buffer())?,
                buffer_entry(1, ray_direction.device_buffer())?,
                buffer_entry(2, d_loss_d_rgb.device_buffer())?,
            ],
        });
        let bg_outputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace_sh::backward outputs bg"),
            layout: &self.layout_back_outputs,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grad_pos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_scale.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_rot.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_sh.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_opacity.as_entire_binding() },
            ],
        });
        let uniforms = Uniforms {
            pixel_count,
            splat_count: splat_count as u32,
            max_distance,
            min_alpha,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::splat_raytrace_sh::backward uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::splat_raytrace_sh::backward uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::splat_raytrace_sh::backward encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::splat_raytrace_sh::backward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_sh_backward);
            pass.set_bind_group(0, &bg_splats, &[]);
            pass.set_bind_group(1, &bg_rays, &[]);
            pass.set_bind_group(2, &bg_outputs, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }

        let stage_pad4 = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: pad4_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let stage_s = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: s_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let stage_sh = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: sh_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let staging_pos = stage_pad4("molla::splat_raytrace_sh::backward grad_pos staging");
        let staging_scale = stage_pad4("molla::splat_raytrace_sh::backward grad_scale staging");
        let staging_rot = stage_pad4("molla::splat_raytrace_sh::backward grad_rot staging");
        let staging_sh = stage_sh("molla::splat_raytrace_sh::backward grad_sh staging");
        let staging_opacity = stage_s("molla::splat_raytrace_sh::backward grad_opacity staging");
        encoder.copy_buffer_to_buffer(&grad_pos, 0, &staging_pos, 0, pad4_bytes);
        encoder.copy_buffer_to_buffer(&grad_scale, 0, &staging_scale, 0, pad4_bytes);
        encoder.copy_buffer_to_buffer(&grad_rot, 0, &staging_rot, 0, pad4_bytes);
        encoder.copy_buffer_to_buffer(&grad_sh, 0, &staging_sh, 0, sh_bytes);
        encoder.copy_buffer_to_buffer(&grad_opacity, 0, &staging_opacity, 0, s_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        let read_pad4 = |staging: &wgpu::Buffer| -> Result<Vec<Vec3>> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("recv: {e}")))?
                .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&mapped);
            let n = splat_count;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let b = i * 4;
                out.push(Vec3::new(f[b] as Real, f[b + 1] as Real, f[b + 2] as Real));
            }
            drop(mapped);
            staging.unmap();
            Ok(out)
        };
        let read_pad4_quat = |staging: &wgpu::Buffer| -> Result<Vec<crate::math::Quat>> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("recv: {e}")))?
                .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&mapped);
            let n = splat_count;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let b = i * 4;
                out.push(crate::math::Quat::from_xyzw(
                    f[b] as Real,
                    f[b + 1] as Real,
                    f[b + 2] as Real,
                    f[b + 3] as Real,
                ));
            }
            drop(mapped);
            staging.unmap();
            Ok(out)
        };
        let read_s = |staging: &wgpu::Buffer| -> Result<Vec<Real>> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("recv: {e}")))?
                .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&mapped);
            let out = f.iter().map(|x| *x as Real).collect::<Vec<_>>();
            drop(mapped);
            staging.unmap();
            Ok(out)
        };
        let read_sh = |staging: &wgpu::Buffer| -> Result<Vec<[Vec3; SH_BAND_2_COUNT]>> {
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
            rx.recv()
                .map_err(|e| Error::Gpu(format!("recv: {e}")))?
                .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
            let mapped = slice.get_mapped_range();
            let f: &[f32] = bytemuck::cast_slice(&mapped);
            let n = splat_count;
            let mut out: Vec<[Vec3; SH_BAND_2_COUNT]> =
                Vec::with_capacity(n);
            for i in 0..n {
                let mut coeffs = [Vec3::ZERO; SH_BAND_2_COUNT];
                for k in 0..SH_BAND_2_COUNT {
                    let b = (i * SH_BAND_2_COUNT + k) * 4;
                    coeffs[k] = Vec3::new(f[b] as Real, f[b + 1] as Real, f[b + 2] as Real);
                }
                out.push(coeffs);
            }
            drop(mapped);
            staging.unmap();
            Ok(out)
        };

        let d_pos = read_pad4(&staging_pos)?;
        let d_scale = read_pad4(&staging_scale)?;
        let d_rotation = read_pad4_quat(&staging_rot)?;
        let d_sh = read_sh(&staging_sh)?;
        let d_opacity = read_s(&staging_opacity)?;

        Ok(SplatShGradients {
            d_pos,
            d_scale,
            d_rotation,
            d_sh,
            d_opacity,
        })
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
