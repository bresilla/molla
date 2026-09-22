//! GPU MPR (Minkowski Portal Refinement) kernel — alternative
//! narrow-phase that produces a contact manifold per shape pair in
//! a single kernel (no separate boolean / penetration-depth split).
//!
//! Output layout matches [`super::EpaKernel`]:
//!
//! ```text
//! [0..3]   normal.xyz, depth
//! [4..7]   point_a.xyz, _
//! [8..11]  point_b.xyz, _
//! [12..15] status_bits_as_f32, _, _, _
//! ```
//!
//! Status codes:
//! - `MPR_STATUS_NONE = 0` — pair was disjoint or skipped.
//! - `MPR_STATUS_OK = 1` — contact valid.
//! - `MPR_STATUS_FALLBACK = 2` — host CPU should re-run (PLANE / SKIP
//!   shape pair, or a bracketing failure during portal seed).
//!
//! Choice of dispatch granularity: one thread per pair (workgroup
//! size 64). The portal-refinement loop is scalar per pair and
//! workgroup-level parallelism wouldn't help.

use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::Model;
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    pair_count: u32,
    shape_count: u32,
    tolerance: f32,
    _pad: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Output stride in `f32` lanes. Same layout as EPA's output.
pub const MPR_OUT_STRIDE: usize = 16;

/// Status code lanes (lane 12 of the per-pair record, reinterpret
/// as `u32` via `f32::to_bits`).
pub const MPR_STATUS_NONE: u32 = 0;
pub const MPR_STATUS_OK: u32 = 1;
pub const MPR_STATUS_FALLBACK: u32 = 2;

/// Default convergence tolerance for the portal-refinement loop.
/// Matches the CPU default in `geometry/mpr.rs`.
pub const MPR_DEFAULT_TOLERANCE: f32 = 1e-4;

pub struct MprKernel {
    pipeline: wgpu::ComputePipeline,
    layout_shapes: wgpu::BindGroupLayout,
    layout_pairs: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl MprKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::mpr"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mpr.wgsl").into()),
        });
        let layout_shapes = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpr shapes"),
            entries: &storage_entries(5, true),
        });
        let layout_pairs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpr pairs"),
            entries: &storage_entries(2, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpr out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mpr uniforms"),
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
            label: Some("molla::mpr layout"),
            bind_group_layouts: &[
                Some(&layout_shapes),
                Some(&layout_pairs),
                Some(&layout_out),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::mpr pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("mpr_collide_pair"),
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

    /// Allocate the per-pair contact output buffer (sized for `cap`
    /// pairs × [`MPR_OUT_STRIDE`] floats each).
    pub fn alloc_out(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cap: usize,
    ) -> Result<Buffer<Real>> {
        let cap = cap.max(1);
        let mut buf = Buffer::from_vec(
            vec![0.0 as Real; cap * MPR_OUT_STRIDE],
            BufferUsage::StorageReadWrite,
        );
        buf.ensure_device(device)?;
        buf.upload(queue)?;
        Ok(buf)
    }

    /// Allocate the pair input buffers sized for `cap` pairs.
    pub fn alloc_pairs(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cap: usize,
    ) -> Result<(Buffer<u32>, Buffer<u32>)> {
        let cap = cap.max(1);
        let mut pair_a = Buffer::from_vec(vec![0u32; cap], BufferUsage::Storage);
        let mut pair_b = Buffer::from_vec(vec![0u32; cap], BufferUsage::Storage);
        pair_a.ensure_device(device)?;
        pair_a.upload(queue)?;
        pair_b.ensure_device(device)?;
        pair_b.upload(queue)?;
        Ok((pair_a, pair_b))
    }

    /// Dispatch MPR on `pair_count` pairs. Writes the per-pair
    /// manifold into `mpr_out`. Blocking — submits and returns after
    /// the queue has accepted the work; the caller still needs to
    /// `download` if it wants host-side access.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        body_q_buf: &wgpu::Buffer,
        pair_a: &Buffer<u32>,
        pair_b: &Buffer<u32>,
        mpr_out: &mut Buffer<Real>,
        pair_count: u32,
        tolerance: f32,
    ) -> Result<()> {
        if pair_count == 0 {
            return Ok(());
        }
        let bg_shapes = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpr shapes bg"),
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
            label: Some("molla::mpr pairs bg"),
            layout: &self.layout_pairs,
            entries: &[
                buffer_entry(0, pair_a.device_buffer())?,
                buffer_entry(1, pair_b.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpr out bg"),
            layout: &self.layout_out,
            entries: &[buffer_entry(0, mpr_out.device_buffer())?],
        });
        let uniforms = Uniforms {
            pair_count,
            shape_count: model.shape_count as u32,
            tolerance,
            _pad: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::mpr uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mpr uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = pair_count.div_ceil(WORKGROUP_SIZE).max(1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mpr encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::mpr pass"),
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
        mpr_out.mark_device_dirty();
        Ok(())
    }
}

/// Decoded MPR contact for one pair. `status == MPR_STATUS_OK` →
/// the manifold (normal, depth, point_a, point_b) is valid.
#[derive(Copy, Clone, Debug)]
pub struct MprPairResult {
    pub normal: Vec3,
    pub depth: f32,
    pub point_a: Vec3,
    pub point_b: Vec3,
    pub status: u32,
}

/// Decode the host-side kernel output for `pair_count` pairs. The
/// caller must have already downloaded `mpr_out` (`.download()` then
/// `.host()`).
pub fn decode_mpr_out(mpr_out: &[Real], pair_count: usize) -> Vec<MprPairResult> {
    let mut out = Vec::with_capacity(pair_count);
    for i in 0..pair_count {
        let base = i * MPR_OUT_STRIDE;
        let nx = mpr_out[base] as f32;
        let ny = mpr_out[base + 1] as f32;
        let nz = mpr_out[base + 2] as f32;
        let depth = mpr_out[base + 3] as f32;
        let pax = mpr_out[base + 4] as f32;
        let pay = mpr_out[base + 5] as f32;
        let paz = mpr_out[base + 6] as f32;
        let pbx = mpr_out[base + 8] as f32;
        let pby = mpr_out[base + 9] as f32;
        let pbz = mpr_out[base + 10] as f32;
        let status_f = mpr_out[base + 12] as f32;
        let status = f32::to_bits(status_f);
        out.push(MprPairResult {
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
    let buf = buf.ok_or_else(|| Error::Gpu(format!("mpr binding {binding}: no device mirror")))?;
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
