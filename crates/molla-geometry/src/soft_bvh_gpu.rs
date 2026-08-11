//! GPU broad-phase query for [`SoftBvh`](crate::soft_bvh::SoftBvh).
//!
//! `SoftBvh::query_point_swept` returns the tris whose AABB intersects a
//! swept-point AABB; the tree is only an accelerator, so a flat GPU
//! sweep over `tri_aabbs` yields the identical candidate set. This is
//! the Stage 33.2 GPU query path anticipated by the `soft_bvh` module
//! docs (a later LBVH refit can replace the flat sweep for scale without
//! changing the result contract).

use crate::soft_bvh::SoftBvh;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    counts: [u32; 4],
}

/// Batched swept-point query on the GPU. Entry `i` is the sorted list of
/// global tri indices whose AABB intersects `queries[i]`'s swept-point
/// AABB — matching [`SoftBvh::query_point_swept`](crate::soft_bvh::SoftBvh::query_point_swept)
/// as a set.
pub fn soft_bvh_query_points_swept_gpu(
    engine: &molla_compute::Engine,
    bvh: &SoftBvh,
    queries: &[(Vec3, Vec3, Real)],
) -> Result<Vec<Vec<u32>>> {
    let tri_count = bvh.tri_aabbs.len();
    if tri_count == 0 || queries.is_empty() {
        return Ok(vec![Vec::new(); queries.len()]);
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("soft_bvh_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("soft_bvh_gpu: engine has no queue".into()))?
        .clone();

    let q_count = queries.len();
    let pair_count = q_count * tri_count;

    let mut tri_data: Vec<[f32; 4]> = Vec::with_capacity(tri_count * 2);
    for a in &bvh.tri_aabbs {
        let mn = a.min.as_vec3();
        let mx = a.max.as_vec3();
        tri_data.push([mn.x, mn.y, mn.z, 0.0]);
        tri_data.push([mx.x, mx.y, mx.z, 0.0]);
    }
    let tri_buf = make_storage(
        &device,
        "molla::soft_bvh tri",
        bytemuck::cast_slice(&tri_data),
    );
    queue.write_buffer(&tri_buf, 0, bytemuck::cast_slice(&tri_data));

    let mut q_data: Vec<[f32; 4]> = Vec::with_capacity(q_count * 2);
    for (p0, p1, margin) in queries {
        let a = p0.as_vec3();
        let b = p1.as_vec3();
        q_data.push([a.x, a.y, a.z, *margin as f32]);
        q_data.push([b.x, b.y, b.z, 0.0]);
    }
    let q_buf = make_storage(
        &device,
        "molla::soft_bvh query",
        bytemuck::cast_slice(&q_data),
    );
    queue.write_buffer(&q_buf, 0, bytemuck::cast_slice(&q_data));

    let mut out = Buffer::from_vec(vec![0u32; pair_count], BufferUsage::StorageReadWrite);
    out.ensure_device(&device)?;
    out.upload(&queue)?;

    let uniforms = Uniforms {
        counts: [tri_count as u32, q_count as u32, pair_count as u32, 0],
    };
    dispatch(
        &device, &queue, &tri_buf, &q_buf, &mut out, uniforms, pair_count,
    )?;
    out.download(&device, &queue)?;

    let flags = out.host()?;
    let mut result: Vec<Vec<u32>> = Vec::with_capacity(q_count);
    for qi in 0..q_count {
        let mut hits = Vec::new();
        for ti in 0..tri_count {
            if flags[qi * tri_count + ti] != 0 {
                hits.push(ti as u32);
            }
        }
        result.push(hits);
    }
    Ok(result)
}

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tri: &wgpu::Buffer,
    query: &wgpu::Buffer,
    out: &mut Buffer<u32>,
    uniforms: Uniforms,
    pair_count: usize,
) -> Result<()> {
    let kernel = SoftBvhQueryKernel::new(device);
    let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::soft_bvh in bg"),
        layout: &kernel.layout_in,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tri.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: query.as_entire_binding(),
            },
        ],
    });
    let out_buf = out
        .device_buffer()
        .ok_or_else(|| Error::Gpu("soft_bvh out: no device mirror".into()))?;
    let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::soft_bvh out bg"),
        layout: &kernel.layout_out,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::soft_bvh uniforms"),
        size: core::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::soft_bvh uniforms bg"),
        layout: &kernel.layout_uniforms,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let groups = (pair_count as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::soft_bvh encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::soft_bvh pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg_in, &[]);
        pass.set_bind_group(1, &bg_out, &[]);
        pass.set_bind_group(2, &bg_uniforms, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    out.mark_device_dirty();
    Ok(())
}

struct SoftBvhQueryKernel {
    pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl SoftBvhQueryKernel {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::soft_bvh_query"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/soft_bvh_query.wgsl").into()),
        });
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_bvh in"),
            entries: &storage_entries(2, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_bvh out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_bvh uniforms"),
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
            label: Some("molla::soft_bvh layout"),
            bind_group_layouts: &[&layout_in, &layout_out, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::soft_bvh pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("query_swept"),
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
}

fn make_storage(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(16)) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
