//! GPU contact reduction kernel — port of the CPU
//! `molla_geometry::contact_reduction` primitive.
//!
//! Per-pair input layout (host packs the contacts already binned by
//! (a_shape, b_shape)):
//!
//! - `in_contacts` is a flat `vec4<f32>` slab of `group_count *
//!   MAX_PER_GROUP * CONTACT_STRIDE` elements.
//! - For each group `g`, contacts are at offset
//!   `g * MAX_PER_GROUP * CONTACT_STRIDE`, packed as 3 `vec4<f32>`
//!   per contact: `(point.xyz, depth)`, `(normal.xyz, id_lo_f32)`,
//!   `(id_hi_f32, a_shape_f32, b_shape_f32, _)`.
//! - `in_count[g]` is the actual contact count for group `g`
//!   (`<= MAX_PER_GROUP`).
//!
//! Output:
//! - `out_contacts` is sized for `group_count * MAX_OUT_PER_GROUP *
//!   CONTACT_STRIDE` elements (kept canonical set, up to 4 per group).
//! - `out_count[g]` reports how many slots were actually written.
//!
//! Algorithm: one workgroup per group, only thread 0 does the work
//! (per-group n is small — the reduction is cheap, the parallel
//! win is across groups). Documented in `shaders/contact_reduction.wgsl`.

use molla_core::{Error, Result};
use molla_geometry::ReducedContact as CpuContact;
use molla_math::{Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

/// Max input contacts per group. The host must clip to this when
/// packing the input buffers.
pub const MAX_PER_GROUP: usize = 64;

/// Max output (canonical) contacts per group. Mirrors the typical
/// 4-corner manifold size used by the rest of the pipeline.
pub const MAX_OUT_PER_GROUP: usize = 4;

/// `vec4<f32>` elements per contact in either buffer.
pub const CONTACT_STRIDE: usize = 3;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    group_count: u32,
    max_per_pair: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct ContactReductionKernel {
    pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ContactReductionKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::contact_reduction"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/contact_reduction.wgsl").into(),
            ),
        });
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::contact_reduction in"),
            entries: &storage_entries(2, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::contact_reduction out"),
            entries: &storage_entries(2, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::contact_reduction uniforms"),
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
            label: Some("molla::contact_reduction layout"),
            bind_group_layouts: &[&layout_in, &layout_out, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::contact_reduction pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("contact_reduce"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_in,
            layout_out,
            layout_uniforms,
        }
    }

    /// Allocate input buffers for `group_count` groups.
    pub fn alloc_input(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        group_count: usize,
    ) -> Result<(Buffer<Real>, Buffer<u32>)> {
        let group_count = group_count.max(1);
        let contacts_len = group_count * MAX_PER_GROUP * CONTACT_STRIDE * 4;
        let mut in_contacts =
            Buffer::from_vec(vec![0.0 as Real; contacts_len], BufferUsage::Storage);
        let mut in_count = Buffer::from_vec(vec![0u32; group_count], BufferUsage::Storage);
        in_contacts.ensure_device(device)?;
        in_contacts.upload(queue)?;
        in_count.ensure_device(device)?;
        in_count.upload(queue)?;
        Ok((in_contacts, in_count))
    }

    /// Allocate output buffers for `group_count` groups.
    pub fn alloc_output(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        group_count: usize,
    ) -> Result<(Buffer<Real>, Buffer<u32>)> {
        let group_count = group_count.max(1);
        let contacts_len = group_count * MAX_OUT_PER_GROUP * CONTACT_STRIDE * 4;
        let mut out_contacts = Buffer::from_vec(
            vec![0.0 as Real; contacts_len],
            BufferUsage::StorageReadWrite,
        );
        let mut out_count =
            Buffer::from_vec(vec![0u32; group_count], BufferUsage::StorageReadWrite);
        out_contacts.ensure_device(device)?;
        out_contacts.upload(queue)?;
        out_count.ensure_device(device)?;
        out_count.upload(queue)?;
        Ok((out_contacts, out_count))
    }

    /// Dispatch reduction over `group_count` groups, each capped at
    /// `max_per_pair` outputs (must be `1..=MAX_OUT_PER_GROUP`).
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        in_contacts: &Buffer<Real>,
        in_count: &Buffer<u32>,
        out_contacts: &mut Buffer<Real>,
        out_count: &mut Buffer<u32>,
        group_count: u32,
        max_per_pair: u32,
    ) -> Result<()> {
        if group_count == 0 {
            return Ok(());
        }
        let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::contact_reduction in bg"),
            layout: &self.layout_in,
            entries: &[
                buffer_entry(0, in_contacts.device_buffer())?,
                buffer_entry(1, in_count.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::contact_reduction out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, out_contacts.device_buffer())?,
                buffer_entry(1, out_count.device_buffer())?,
            ],
        });
        let uniforms = Uniforms {
            group_count,
            max_per_pair: max_per_pair.min(MAX_OUT_PER_GROUP as u32).max(1),
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::contact_reduction uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::contact_reduction uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });
        // One workgroup per group. Each workgroup's thread 0 does
        // the scalar work — see `shaders/contact_reduction.wgsl`.
        let groups = group_count;
        let _ = WORKGROUP_SIZE; // Workgroup size is set in the shader.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::contact_reduction encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::contact_reduction pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_in, &[]);
            pass.set_bind_group(1, &bg_out, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        out_contacts.mark_device_dirty();
        out_count.mark_device_dirty();
        Ok(())
    }
}

/// Pack a flat slice of contacts into the per-group input slab.
///
/// `groups` is a list of `(a_shape, b_shape, slice-of-input-indices)`
/// triples — the host already binned by pair. Each per-group slice
/// must hold ≤ [`MAX_PER_GROUP`] indices; longer groups are truncated.
///
/// The returned `in_contacts_host` / `in_count_host` vectors can be
/// uploaded directly to the buffers allocated by [`alloc_input`].
pub fn pack_groups_host(
    contacts: &[CpuContact],
    groups: &[(u32, u32, Vec<usize>)],
) -> (Vec<Real>, Vec<u32>) {
    let n_groups = groups.len();
    let mut in_contacts = vec![0.0 as Real; n_groups * MAX_PER_GROUP * CONTACT_STRIDE * 4];
    let mut in_count = vec![0u32; n_groups];
    for (g, (a_shape, b_shape, indices)) in groups.iter().enumerate() {
        let n = indices.len().min(MAX_PER_GROUP) as u32;
        in_count[g] = n;
        for (local, &gi) in indices.iter().take(MAX_PER_GROUP).enumerate() {
            let c = contacts[gi];
            let base = (g * MAX_PER_GROUP + local) * CONTACT_STRIDE * 4;
            // vec4 0: point.xyz, depth
            in_contacts[base + 0] = c.point.x as Real;
            in_contacts[base + 1] = c.point.y as Real;
            in_contacts[base + 2] = c.point.z as Real;
            in_contacts[base + 3] = c.depth as Real;
            // vec4 1: normal.xyz, id_lo (as f32 bit-cast)
            in_contacts[base + 4] = c.normal.x as Real;
            in_contacts[base + 5] = c.normal.y as Real;
            in_contacts[base + 6] = c.normal.z as Real;
            let id_lo = (c.id & 0xFFFF_FFFF) as u32;
            in_contacts[base + 7] = f32::from_bits(id_lo) as Real;
            // vec4 2: id_hi, a_shape, b_shape, _
            let id_hi = (c.id >> 32) as u32;
            in_contacts[base + 8] = f32::from_bits(id_hi) as Real;
            in_contacts[base + 9] = f32::from_bits(*a_shape) as Real;
            in_contacts[base + 10] = f32::from_bits(*b_shape) as Real;
            in_contacts[base + 11] = 0.0;
        }
    }
    (in_contacts, in_count)
}

/// Unpack the output slab back into a flat `Vec<CpuContact>`,
/// preserving group order.
pub fn unpack_output_host(out_contacts: &[Real], out_count: &[u32]) -> Vec<CpuContact> {
    let n_groups = out_count.len();
    let mut out = Vec::new();
    for g in 0..n_groups {
        let n = out_count[g].min(MAX_OUT_PER_GROUP as u32) as usize;
        for local in 0..n {
            let base = (g * MAX_OUT_PER_GROUP + local) * CONTACT_STRIDE * 4;
            let point = Vec3::new(
                out_contacts[base + 0],
                out_contacts[base + 1],
                out_contacts[base + 2],
            );
            let depth = out_contacts[base + 3];
            let normal = Vec3::new(
                out_contacts[base + 4],
                out_contacts[base + 5],
                out_contacts[base + 6],
            );
            let id_lo_f = out_contacts[base + 7] as f32;
            let id_hi_f = out_contacts[base + 8] as f32;
            let id_lo = f32::to_bits(id_lo_f) as u64;
            let id_hi = f32::to_bits(id_hi_f) as u64;
            let id = (id_hi << 32) | id_lo;
            let a_shape_f = out_contacts[base + 9] as f32;
            let b_shape_f = out_contacts[base + 10] as f32;
            let a_shape = f32::to_bits(a_shape_f);
            let b_shape = f32::to_bits(b_shape_f);
            out.push(CpuContact {
                a_shape,
                b_shape,
                id,
                point,
                normal,
                depth,
            });
        }
    }
    out
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "contact_reduction binding {binding}: no device mirror"
        ))
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
