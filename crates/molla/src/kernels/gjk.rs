//! GPU GJK boolean overlap test.
//!
//! Given a list of candidate shape pairs (typically from the broad
//! phase) the kernel returns `overlap[i] ∈ {0, 1, 2}`:
//!   * `0` → confirmed disjoint, host can skip narrow phase.
//!   * `1` → confirmed overlapping, host can run EPA on this pair.
//!   * `2` → "fall back to CPU" (Plane / TriMesh / iteration-cap hit).
//!
//! Cuts CPU narrow-phase work for scenes where the broad phase
//! emits many false-positive candidate pairs. EPA stays CPU-side for
//! v1 — porting it requires bounded polytope storage which is its
//! own milestone.

use crate::core::{Error, Result};
use crate::sim::{Contacts, Model};
use crate::storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    pair_count: u32,
    shape_count: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Per-pair simplex output stride (in `vec4<f32>` elements). 4
/// vertices × 3 vec3 (w, a, b) = 12 vec4 lanes per pair.
pub const GJK_SIMPLEX_STRIDE: usize = 12;

pub struct GjkKernel {
    pipeline: wgpu::ComputePipeline,
    layout_shapes: wgpu::BindGroupLayout,
    layout_pairs: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl GjkKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::gjk"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/gjk.wgsl").into()),
        });
        let layout_shapes = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::gjk shapes"),
            entries: &storage_entries(5, true),
        });
        let layout_pairs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::gjk pairs"),
            entries: &storage_entries(2, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::gjk out"),
            entries: &storage_entries(2, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::gjk uniforms"),
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
            label: Some("molla::gjk layout"),
            bind_group_layouts: &[
                &layout_shapes,
                &layout_pairs,
                &layout_out,
                &layout_uniforms,
            ],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::gjk pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("gjk_overlap"),
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

    /// Run GJK overlap on `pair_a[i]` × `pair_b[i]` and write the
    /// result into `overlap_out`. All inputs are device buffers
    /// already populated by the caller. The simplex output is
    /// discarded (this overload allocates a tiny scratch buffer to
    /// satisfy the binding layout — use
    /// [`Self::dispatch_with_simplex`] when EPA needs the simplex).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        body_q_buf: &wgpu::Buffer,
        pair_a: &Buffer<u32>,
        pair_b: &Buffer<u32>,
        overlap_out: &mut Buffer<u32>,
        pair_count: u32,
    ) -> Result<()> {
        // Allocate a stub simplex buffer just large enough to satisfy
        // the binding (one pair worth of vec4 lanes).
        let mut stub = Buffer::<crate::math::Vec3>::from_vec(
            vec![crate::math::Vec3::ZERO; GJK_SIMPLEX_STRIDE.max(pair_count as usize * GJK_SIMPLEX_STRIDE)],
            BufferUsage::StorageReadWrite,
        );
        stub.ensure_device(device)?;
        stub.upload(queue)?;
        self.dispatch_with_simplex(
            device, queue, model, body_q_buf, pair_a, pair_b, overlap_out, &mut stub, pair_count,
        )
    }

    /// Variant of [`Self::dispatch`] that also writes the 4-vertex
    /// Minkowski-difference simplex into `simplex_out` for every pair
    /// where `overlap == 1`. The buffer must hold at least
    /// `pair_count * GJK_SIMPLEX_STRIDE` vec3 elements (each vec3 is
    /// padded to 16 bytes on the GPU side).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_with_simplex(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        body_q_buf: &wgpu::Buffer,
        pair_a: &Buffer<u32>,
        pair_b: &Buffer<u32>,
        overlap_out: &mut Buffer<u32>,
        simplex_out: &mut Buffer<crate::math::Vec3>,
        pair_count: u32,
    ) -> Result<()> {
        if pair_count == 0 {
            return Ok(());
        }
        let bg_shapes = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::gjk shapes bg"),
            layout: &self.layout_shapes,
            entries: &[
                buffer_entry(0, model.shape_geom_kind.device_buffer())?,
                buffer_entry(1, model.shape_geom_data.device_buffer())?,
                buffer_entry(2, model.shape_body.device_buffer())?,
                buffer_entry(3, model.shape_transform.device_buffer())?,
                wgpu::BindGroupEntry { binding: 4, resource: body_q_buf.as_entire_binding() },
            ],
        });
        let bg_pairs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::gjk pairs bg"),
            layout: &self.layout_pairs,
            entries: &[
                buffer_entry(0, pair_a.device_buffer())?,
                buffer_entry(1, pair_b.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::gjk out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, overlap_out.device_buffer())?,
                buffer_entry(1, simplex_out.device_buffer())?,
            ],
        });
        let uniforms = Uniforms {
            pair_count,
            shape_count: model.shape_count as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::gjk uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::gjk uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = pair_count.div_ceil(WORKGROUP_SIZE).max(1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::gjk encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::gjk pass"),
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
        overlap_out.mark_device_dirty();
        simplex_out.mark_device_dirty();
        Ok(())
    }

    /// Allocate a simplex-output buffer sized for `cap` pairs. Each
    /// pair occupies `GJK_SIMPLEX_STRIDE` vec3 elements.
    pub fn alloc_simplex(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cap: usize,
    ) -> Result<Buffer<crate::math::Vec3>> {
        let cap = cap.max(1);
        let mut buf = Buffer::from_vec(
            vec![crate::math::Vec3::ZERO; cap * GJK_SIMPLEX_STRIDE],
            BufferUsage::StorageReadWrite,
        );
        buf.ensure_device(device)?;
        buf.upload(queue)?;
        Ok(buf)
    }

    /// Convenience helper to allocate the input/output buffers sized
    /// for `cap` pairs.
    pub fn alloc_io(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cap: usize,
    ) -> Result<(Buffer<u32>, Buffer<u32>, Buffer<u32>)> {
        let cap = cap.max(1);
        let mut pair_a = Buffer::from_vec(vec![0u32; cap], BufferUsage::Storage);
        let mut pair_b = Buffer::from_vec(vec![0u32; cap], BufferUsage::Storage);
        let mut overlap = Buffer::from_vec(vec![0u32; cap], BufferUsage::StorageReadWrite);
        pair_a.ensure_device(device)?;
        pair_a.upload(queue)?;
        pair_b.ensure_device(device)?;
        pair_b.upload(queue)?;
        overlap.ensure_device(device)?;
        overlap.upload(queue)?;
        Ok((pair_a, pair_b, overlap))
    }
}

// `Contacts` parameter dropped — the GPU GJK pass produces overlap
// flags only, not contacts. Kept the import just in case a follow-up
// wires the kernel into the pipeline.
#[allow(dead_code)]
fn _suppress_contacts(_c: &Contacts) {}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| Error::Gpu(format!("gjk binding {binding}: no device mirror")))?;
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
