//! GPU EPA kernel — consumes the per-pair simplex emitted by
//! [`super::GjkKernel::dispatch_with_simplex`] and produces contact
//! manifolds (normal, depth, point_a, point_b) for every pair where
//! GJK reported overlap.
//!
//! Output layout (4 vec4 per pair, see `shaders/epa.wgsl`):
//! - lane 0: `(nx, ny, nz, depth)`
//! - lane 1: `(pa.xyz, _)`
//! - lane 2: `(pb.xyz, _)`
//! - lane 3: `(status_bits_as_f32, _, _, _)` — 1 = OK, 2 = fallback,
//!   0 = pair was disjoint or skipped.
//!
//! Pairs flagged `STATUS_FALLBACK` should be re-run on CPU EPA — the
//! WGSL polytope cap (32 vertices / 64 faces) is enough for every
//! convex-vs-convex test we currently exercise, but degenerate
//! configurations (deeply interpenetrating, near-parallel hulls) hit
//! it sometimes and the host fallback is the cheap escape hatch.

use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::Model;
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    pair_count: u32,
    shape_count: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Output stride (in `Real` / `f32` lanes). Per-pair EPA result =
/// 4 `vec4<f32>` = 16 floats, laid out:
///
/// ```text
/// [0..3]   normal.xyz, depth
/// [4..7]   point_a.xyz, _
/// [8..11]  point_b.xyz, _
/// [12..15] status_bits_as_f32, _, _, _
/// ```
///
/// Stored as `Buffer<Real>` so the `.w` lane round-trips intact
/// (a `Buffer<Vec3>` would discard it on download).
pub const EPA_OUT_STRIDE: usize = 16;

/// Status code for one pair (lane 3, `.x` reinterpret-as-u32).
pub const EPA_STATUS_NONE: u32 = 0;
pub const EPA_STATUS_OK: u32 = 1;
pub const EPA_STATUS_FALLBACK: u32 = 2;

pub struct EpaKernel {
    pipeline: wgpu::ComputePipeline,
    layout_shapes: wgpu::BindGroupLayout,
    layout_pairs: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl EpaKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::epa"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/epa.wgsl").into()),
        });
        let layout_shapes = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::epa shapes"),
            entries: &storage_entries(5, true),
        });
        let layout_pairs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::epa pairs"),
            entries: &storage_entries(4, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::epa out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::epa uniforms"),
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
            label: Some("molla::epa layout"),
            bind_group_layouts: &[
                Some(&layout_shapes),
                Some(&layout_pairs),
                Some(&layout_out),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::epa pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("epa_solve"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_shapes,
            layout_pairs,
            layout_out,
            layout_uniforms,
        }
    }

    /// Allocate the per-pair contact buffer sized for `cap` pairs.
    /// Each pair occupies `EPA_OUT_STRIDE` `f32` lanes.
    pub fn alloc_out(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cap: usize,
    ) -> Result<Buffer<Real>> {
        let cap = cap.max(1);
        let mut buf = Buffer::from_vec(
            vec![0.0 as Real; cap * EPA_OUT_STRIDE],
            BufferUsage::StorageReadWrite,
        );
        buf.ensure_device(device)?;
        buf.upload(queue)?;
        Ok(buf)
    }

    /// Dispatch EPA over all `pair_count` pairs. Reads simplex /
    /// overlap from the buffers populated by [`super::GjkKernel`].
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        body_q_buf: &wgpu::Buffer,
        pair_a: &Buffer<u32>,
        pair_b: &Buffer<u32>,
        overlap_in: &Buffer<u32>,
        gjk_simplex_in: &Buffer<Vec3>,
        epa_out: &mut Buffer<Real>,
        pair_count: u32,
    ) -> Result<()> {
        if pair_count == 0 {
            return Ok(());
        }
        let bg_shapes = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::epa shapes bg"),
            layout: &self.layout_shapes,
            entries: &[
                buffer_entry(0, model.shape_geom_kind.device_buffer())?,
                buffer_entry(1, model.shape_geom_data.device_buffer())?,
                buffer_entry(2, model.shape_body.device_buffer())?,
                buffer_entry(3, model.shape_transform.device_buffer())?,
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: body_q_buf.as_entire_binding(),
                },
            ],
        });
        let bg_pairs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::epa pairs bg"),
            layout: &self.layout_pairs,
            entries: &[
                buffer_entry(0, pair_a.device_buffer())?,
                buffer_entry(1, pair_b.device_buffer())?,
                buffer_entry(2, overlap_in.device_buffer())?,
                buffer_entry(3, gjk_simplex_in.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::epa out bg"),
            layout: &self.layout_out,
            entries: &[buffer_entry(0, epa_out.device_buffer())?],
        });
        let uniforms = Uniforms {
            pair_count,
            shape_count: model.shape_count as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::epa uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::epa uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = pair_count.div_ceil(WORKGROUP_SIZE).max(1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::epa encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::epa pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_shapes, &[]);
            pass.set_bind_group(1, &bg_pairs, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        epa_out.mark_device_dirty();
        Ok(())
    }
}

/// Decoded EPA contact for one pair. `status == EPA_STATUS_OK` means
/// the contact (normal, depth, point_a, point_b) is valid.
#[derive(Copy, Clone, Debug)]
pub struct EpaPairResult {
    pub normal: Vec3,
    pub depth: f32,
    pub point_a: Vec3,
    pub point_b: Vec3,
    pub status: u32,
}

/// Decode the kernel output for `pair_count` pairs. `epa_out` must
/// have been downloaded already (`download(...)` then `host()`).
pub fn decode_epa_out(epa_out: &[Real], pair_count: usize) -> Vec<EpaPairResult> {
    let mut out = Vec::with_capacity(pair_count);
    for i in 0..pair_count {
        let base = i * EPA_OUT_STRIDE;
        let nx = epa_out[base] as f32;
        let ny = epa_out[base + 1] as f32;
        let nz = epa_out[base + 2] as f32;
        let depth = epa_out[base + 3] as f32;
        let pax = epa_out[base + 4] as f32;
        let pay = epa_out[base + 5] as f32;
        let paz = epa_out[base + 6] as f32;
        let pbx = epa_out[base + 8] as f32;
        let pby = epa_out[base + 9] as f32;
        let pbz = epa_out[base + 10] as f32;
        let status_f = epa_out[base + 12] as f32;
        let status = f32::to_bits(status_f);
        out.push(EpaPairResult {
            normal: Vec3::new(nx as Real, ny as Real, nz as Real),
            depth,
            point_a: Vec3::new(pax as Real, pay as Real, paz as Real),
            point_b: Vec3::new(pbx as Real, pby as Real, pbz as Real),
            status,
        });
    }
    out
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| Error::Gpu(format!("epa binding {binding}: no device mirror")))?;
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
