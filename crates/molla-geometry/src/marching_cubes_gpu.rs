//! GPU marching cubes. Mirrors the CPU [`marching_cubes`](crate::marching_cubes::marching_cubes):
//! one thread per grid cell fills a dense per-cell triangle slot, and the
//! host compacts the cells in order into the same [`TriMesh`].

use crate::marching_cubes::{EDGE_TABLE, TRI_TABLE};
use crate::sdf_bake::SdfGrid;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::TriMesh;
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    dims: [u32; 4],
    origin: [f32; 4],
    iso: [f32; 4],
}

const WORKGROUP_SIZE: u32 = 64;
/// Floats per cell in the dense output: up to 5 triangles × 3 verts × 3.
const CELL_STRIDE: usize = 45;

/// Extract a triangle mesh from `sdf` at `iso` on the GPU — matches the
/// CPU [`marching_cubes`](crate::marching_cubes::marching_cubes) mesh
/// (same triangle order) within f32 tolerance.
pub fn marching_cubes_gpu(
    engine: &molla_compute::Engine,
    sdf: &SdfGrid,
    iso: Real,
) -> Result<TriMesh> {
    let (nx, ny, nz) = (sdf.nx as usize, sdf.ny as usize, sdf.nz as usize);
    if nx < 2 || ny < 2 || nz < 2 {
        return Ok(TriMesh::default());
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("marching_cubes_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("marching_cubes_gpu: engine has no queue".into()))?
        .clone();

    let cell_count = (nx - 1) * (ny - 1) * (nz - 1);

    // SDF grid (f32) + the Lorensen tables, uploaded from the CPU constants.
    let sdf_f32: Vec<f32> = sdf.data.iter().map(|&d| d as f32).collect();
    let sdf_buf = make_storage(&device, "molla::mc sdf", bytemuck::cast_slice(&sdf_f32));
    molla_core::gpu_write_buffer!(queue, &sdf_buf, 0, bytemuck::cast_slice(&sdf_f32));

    let edge_buf = make_storage(&device, "molla::mc edge", bytemuck::cast_slice(&EDGE_TABLE));
    molla_core::gpu_write_buffer!(queue, &edge_buf, 0, bytemuck::cast_slice(&EDGE_TABLE));

    let tri_flat: Vec<i32> = TRI_TABLE.iter().flatten().map(|&x| x as i32).collect();
    let tri_buf = make_storage(&device, "molla::mc tri", bytemuck::cast_slice(&tri_flat));
    molla_core::gpu_write_buffer!(queue, &tri_buf, 0, bytemuck::cast_slice(&tri_flat));

    let mut out_verts = Buffer::from_vec(
        vec![0.0 as Real; cell_count * CELL_STRIDE],
        BufferUsage::StorageReadWrite,
    );
    out_verts.ensure_device(&device)?;
    out_verts.upload(&queue)?;
    let mut out_count = Buffer::from_vec(vec![0u32; cell_count], BufferUsage::StorageReadWrite);
    out_count.ensure_device(&device)?;
    out_count.upload(&queue)?;

    let uniforms = Uniforms {
        dims: [nx as u32, ny as u32, nz as u32, cell_count as u32],
        origin: [
            sdf.origin.x as f32,
            sdf.origin.y as f32,
            sdf.origin.z as f32,
            sdf.cell_size as f32,
        ],
        iso: [iso as f32, 0.0, 0.0, 0.0],
    };
    dispatch(
        &device,
        &queue,
        &sdf_buf,
        &edge_buf,
        &tri_buf,
        &mut out_verts,
        &mut out_count,
        uniforms,
        cell_count,
    )?;
    out_verts.download(&device, &queue)?;
    out_count.download(&device, &queue)?;

    // Compact cells in order — reproduces the CPU emit sequence exactly.
    let verts = out_verts.host()?;
    let counts = out_count.host()?;
    let mut vertices: Vec<Vec3> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    for (cell, &count) in counts.iter().enumerate() {
        for tri in 0..count as usize {
            let b = cell * CELL_STRIDE + tri * 9;
            let base = vertices.len() as u32;
            vertices.push(Vec3::new(verts[b], verts[b + 1], verts[b + 2]));
            vertices.push(Vec3::new(verts[b + 3], verts[b + 4], verts[b + 5]));
            vertices.push(Vec3::new(verts[b + 6], verts[b + 7], verts[b + 8]));
            indices.push([base, base + 1, base + 2]);
        }
    }
    Ok(TriMesh { vertices, indices })
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sdf: &wgpu::Buffer,
    edge: &wgpu::Buffer,
    tri: &wgpu::Buffer,
    out_verts: &mut Buffer<Real>,
    out_count: &mut Buffer<u32>,
    uniforms: Uniforms,
    cell_count: usize,
) -> Result<()> {
    let kernel = MarchingCubesKernel::new(device);
    let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::mc in bg"),
        layout: &kernel.layout_in,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sdf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: edge.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: tri.as_entire_binding(),
            },
        ],
    });
    let verts_buf = out_verts
        .device_buffer()
        .ok_or_else(|| Error::Gpu("mc out_verts: no device mirror".into()))?;
    let count_buf = out_count
        .device_buffer()
        .ok_or_else(|| Error::Gpu("mc out_count: no device mirror".into()))?;
    let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::mc out bg"),
        layout: &kernel.layout_out,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: verts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: count_buf.as_entire_binding(),
            },
        ],
    });
    let uniform_buf = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("molla::mc uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::mc uniforms bg"),
        layout: &kernel.layout_uniforms,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let groups = (cell_count as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::mc encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::mc pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg_in, &[]);
        pass.set_bind_group(1, &bg_out, &[]);
        pass.set_bind_group(2, &bg_uniforms, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    out_verts.mark_device_dirty();
    out_count.mark_device_dirty();
    Ok(())
}

struct MarchingCubesKernel {
    pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl MarchingCubesKernel {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::marching_cubes"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/marching_cubes.wgsl").into()),
        });
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mc in"),
            entries: &storage_entries(3, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mc out"),
            entries: &storage_entries(2, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mc uniforms"),
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
            label: Some("molla::mc layout"),
            bind_group_layouts: &[Some(&layout_in), Some(&layout_out), Some(&layout_uniforms)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::mc pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("march"),
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
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
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
