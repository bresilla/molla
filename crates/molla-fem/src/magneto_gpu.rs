//! GPU per-cell curl recovery `B = ∇×A` from Nédélec edge DoFs. Mirrors
//! the CPU [`compute_curl_per_cell`](crate::magneto::compute_curl_per_cell):
//! one thread per tet. The edge DoF indices + orientation signs are
//! precomputed host-side from the [`NedelecDofMap`]. Tet meshes only.

use crate::mesh::{FemCell, FemMesh};
use crate::nedelec_element::NedelecDofMap;
use crate::p2_element::TET_EDGES;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    counts: [u32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU counterpart of [`compute_curl_per_cell`](crate::magneto::compute_curl_per_cell)
/// for an all-tet mesh. Returns one `B` vector per cell, matching the
/// CPU within f32 tolerance.
pub fn compute_curl_per_cell_gpu(
    engine: &molla_compute::Engine,
    mesh: &FemMesh,
    map: &NedelecDofMap,
    a_global: &[Real],
) -> Result<Vec<Vec3>> {
    let cell_count = mesh.cells.len();
    if cell_count == 0 {
        return Ok(Vec::new());
    }

    // Per-cell tet vertices + precomputed edge DoFs / signs.
    let mut verts: Vec<u32> = Vec::with_capacity(cell_count * 4);
    let mut edge_dofs: Vec<u32> = Vec::with_capacity(cell_count * 6);
    let mut edge_signs: Vec<f32> = Vec::with_capacity(cell_count * 6);
    for cell in &mesh.cells {
        let FemCell::Tet(v) = cell else {
            return Err(Error::Gpu(
                "compute_curl_per_cell_gpu: tet meshes only".into(),
            ));
        };
        verts.extend_from_slice(v);
        for (e, [ai, bi]) in TET_EDGES.iter().enumerate() {
            edge_dofs.push(map.edge_dof(v[*ai], v[*bi]) as u32);
            edge_signs.push(NedelecDofMap::edge_sign(cell, e) as f32);
        }
    }

    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("magneto_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("magneto_gpu: engine has no queue".into()))?
        .clone();

    let mut pos: Vec<[f32; 4]> = Vec::with_capacity(mesh.vertices.len());
    for p in &mesh.vertices {
        let g = p.as_vec3();
        pos.push([g.x, g.y, g.z, 0.0]);
    }
    let pos_buf = make_storage(&device, "molla::curl pos", bytemuck::cast_slice(&pos));
    queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&pos));
    let verts_buf = make_storage(&device, "molla::curl verts", bytemuck::cast_slice(&verts));
    queue.write_buffer(&verts_buf, 0, bytemuck::cast_slice(&verts));
    let dof_buf = make_storage(
        &device,
        "molla::curl dofs",
        bytemuck::cast_slice(&edge_dofs),
    );
    queue.write_buffer(&dof_buf, 0, bytemuck::cast_slice(&edge_dofs));
    let sign_buf = make_storage(
        &device,
        "molla::curl signs",
        bytemuck::cast_slice(&edge_signs),
    );
    queue.write_buffer(&sign_buf, 0, bytemuck::cast_slice(&edge_signs));

    let mut a_buf = Buffer::from_vec(a_global.to_vec(), BufferUsage::Storage);
    a_buf.ensure_device(&device)?;
    a_buf.upload(&queue)?;
    let mut out = Buffer::from_vec(
        vec![0.0 as Real; cell_count * 3],
        BufferUsage::StorageReadWrite,
    );
    out.ensure_device(&device)?;
    out.upload(&queue)?;

    let kernel = CurlKernel::new(&device);
    let a_dev = a_buf
        .device_buffer()
        .ok_or_else(|| Error::Gpu("curl a: no device mirror".into()))?;
    let out_dev = out
        .device_buffer()
        .ok_or_else(|| Error::Gpu("curl out: no device mirror".into()))?;
    let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::curl in bg"),
        layout: &kernel.layout_in,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: verts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pos_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: a_dev.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dof_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: sign_buf.as_entire_binding(),
            },
        ],
    });
    let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::curl out bg"),
        layout: &kernel.layout_out,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out_dev.as_entire_binding(),
        }],
    });
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::curl uniforms"),
        size: core::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &uniform_buf,
        0,
        bytemuck::bytes_of(&Uniforms {
            counts: [cell_count as u32, 0, 0, 0],
        }),
    );
    let bg_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::curl uniform bg"),
        layout: &kernel.layout_uniform,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let groups = (cell_count as u32).div_ceil(WORKGROUP_SIZE).max(1);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::curl encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::curl pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg_in, &[]);
        pass.set_bind_group(1, &bg_out, &[]);
        pass.set_bind_group(2, &bg_uniform, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    out.mark_device_dirty();
    out.download(&device, &queue)?;

    let flat = out.host()?;
    Ok((0..cell_count)
        .map(|c| Vec3::new(flat[3 * c], flat[3 * c + 1], flat[3 * c + 2]))
        .collect())
}

struct CurlKernel {
    pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniform: wgpu::BindGroupLayout,
}

impl CurlKernel {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::magneto_curl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/magneto_curl.wgsl").into()),
        });
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::curl in"),
            entries: &storage_entries(5, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::curl out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniform = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::curl uniform"),
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
            label: Some("molla::curl layout"),
            bind_group_layouts: &[Some(&layout_in), Some(&layout_out), Some(&layout_uniform)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::curl pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("curl"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_in,
            layout_out,
            layout_uniform,
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
