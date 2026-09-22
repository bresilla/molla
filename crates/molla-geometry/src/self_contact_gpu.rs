//! GPU vertex-triangle self-contact candidate generation. Mirrors the
//! CPU [`detect_vertex_triangle_self_contacts`](crate::self_contact::detect_vertex_triangle_self_contacts):
//! one thread per (vertex, triangle) pair fills a dense output array,
//! and the host compacts + sorts it into the same candidate list.

use crate::self_contact::SelfContactCandidate;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    counts: [u32; 4],
    params: [f32; 4],
}

const WORKGROUP_SIZE: u32 = 64;
/// Floats per (vertex, triangle) slot in the dense output buffer.
const STRIDE: usize = 12;

/// Reusable pipeline for the self-contact kernel.
pub struct SelfContactKernel {
    pipeline: wgpu::ComputePipeline,
    layout_mesh: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl SelfContactKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::self_contact"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/self_contact.wgsl").into()),
        });
        let layout_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::self_contact mesh"),
            entries: &storage_entries(2, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::self_contact out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::self_contact uniforms"),
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
            label: Some("molla::self_contact layout"),
            bind_group_layouts: &[Some(&layout_mesh), Some(&layout_out), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::self_contact pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("detect_self_contact"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_mesh,
            layout_out,
            layout_uniforms,
        }
    }
}

/// Generate vertex-triangle self-contact candidates on the GPU. The
/// result matches [`detect_vertex_triangle_self_contacts`](crate::self_contact::detect_vertex_triangle_self_contacts)
/// (same ordering) within f32 tolerance.
pub fn detect_vertex_triangle_self_contacts_gpu(
    engine: &molla_compute::Engine,
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    thickness: Real,
) -> Result<Vec<SelfContactCandidate>> {
    if vertices.is_empty() || triangles.is_empty() || thickness < 0.0 {
        return Ok(Vec::new());
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("self_contact_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("self_contact_gpu: engine has no queue".into()))?
        .clone();

    let vertex_count = vertices.len();
    let tri_count = triangles.len();
    let pair_count = vertex_count * tri_count;

    let mut vert_pad: Vec<[f32; 4]> = Vec::with_capacity(vertex_count);
    for v in vertices {
        let g = v.as_vec3();
        vert_pad.push([g.x, g.y, g.z, 0.0]);
    }
    let vertices_buf = make_storage(
        &device,
        "molla::self_contact vertices",
        bytemuck::cast_slice(&vert_pad),
    );
    queue.write_buffer(&vertices_buf, 0, bytemuck::cast_slice(&vert_pad));

    let mut idx_flat: Vec<u32> = Vec::with_capacity(tri_count * 3);
    for tri in triangles {
        idx_flat.extend_from_slice(tri);
    }
    let indices_buf = make_storage(
        &device,
        "molla::self_contact indices",
        bytemuck::cast_slice(&idx_flat),
    );
    queue.write_buffer(&indices_buf, 0, bytemuck::cast_slice(&idx_flat));

    let mut out = Buffer::from_vec(
        vec![-1.0 as Real; pair_count * STRIDE],
        BufferUsage::StorageReadWrite,
    );
    out.ensure_device(&device)?;
    out.upload(&queue)?;

    let kernel = SelfContactKernel::new(&device);
    let uniforms = Uniforms {
        counts: [vertex_count as u32, tri_count as u32, pair_count as u32, 0],
        params: [thickness as f32, 0.0, 0.0, 0.0],
    };
    dispatch(
        &device,
        &queue,
        &kernel,
        &vertices_buf,
        &indices_buf,
        &mut out,
        uniforms,
        pair_count,
    )?;
    out.download(&device, &queue)?;

    // Compact the dense slots, then sort to match the CPU ordering.
    let data = out.host()?;
    let mut candidates = Vec::new();
    for pair in 0..pair_count {
        let base = pair * STRIDE;
        let distance = data[base + 11];
        if distance < 0.0 {
            continue;
        }
        candidates.push(SelfContactCandidate {
            vertex: data[base] as u32,
            triangle: [
                data[base + 1] as u32,
                data[base + 2] as u32,
                data[base + 3] as u32,
            ],
            point: Vec3::new(data[base + 4], data[base + 5], data[base + 6]),
            gap: data[base + 7],
            normal: Vec3::new(data[base + 8], data[base + 9], data[base + 10]),
            distance,
        });
    }
    candidates.sort_by(|a, b| {
        a.vertex
            .cmp(&b.vertex)
            .then_with(|| a.triangle.cmp(&b.triangle))
            .then_with(|| a.distance.partial_cmp(&b.distance).unwrap())
    });
    Ok(candidates)
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kernel: &SelfContactKernel,
    vertices: &wgpu::Buffer,
    indices: &wgpu::Buffer,
    out: &mut Buffer<Real>,
    uniforms: Uniforms,
    pair_count: usize,
) -> Result<()> {
    let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::self_contact mesh bg"),
        layout: &kernel.layout_mesh,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: indices.as_entire_binding(),
            },
        ],
    });
    let out_buf = out
        .device_buffer()
        .ok_or_else(|| Error::Gpu("self_contact out: no device mirror".into()))?;
    let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::self_contact out bg"),
        layout: &kernel.layout_out,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_buf.as_entire_binding(),
        }],
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::self_contact uniforms"),
        size: core::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::self_contact uniforms bg"),
        layout: &kernel.layout_uniforms,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let groups = (pair_count as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::self_contact encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::self_contact pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg_mesh, &[]);
        pass.set_bind_group(1, &bg_out, &[]);
        pass.set_bind_group(2, &bg_uniforms, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    out.mark_device_dirty();
    Ok(())
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
