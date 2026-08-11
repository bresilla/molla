//! GPU mesh → voxel signed-distance bake. Mirrors the CPU
//! [`bake_mesh_to_sdf`](crate::sdf_bake::bake_mesh_to_sdf): one thread
//! per voxel computes the unsigned closest-triangle distance and the
//! +X ray-parity sign (see `shaders/sdf_bake.wgsl`).
//!
//! The grid envelope (origin, cell size, resolution) is computed
//! host-side exactly as the CPU path does, so the two grids are sample
//! aligned and their `data` arrays compare directly within the usual
//! f32-narrowing tolerance.

use crate::sdf_bake::SdfGrid;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::TriMesh;
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    origin: [f32; 4],
    dims: [u32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// Reusable pipeline for the SDF bake kernel.
pub struct SdfBakeKernel {
    pipeline: wgpu::ComputePipeline,
    layout_mesh: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl SdfBakeKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sdf_bake"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf_bake.wgsl").into()),
        });
        let layout_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sdf_bake mesh"),
            entries: &storage_entries(2, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sdf_bake out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sdf_bake uniforms"),
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
            label: Some("molla::sdf_bake layout"),
            bind_group_layouts: &[&layout_mesh, &layout_out, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::sdf_bake pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("bake_sdf"),
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

    /// Fill `out` (length `total = res³`) with the signed distance for
    /// every voxel of the grid described by `origin` (xyz + cell size
    /// in `w`), `res` and `tri_count`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        out: &mut Buffer<Real>,
        origin: [f32; 4],
        res: u32,
        tri_count: u32,
        total: u32,
    ) -> Result<()> {
        if total == 0 {
            return Ok(());
        }
        let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sdf_bake mesh bg"),
            layout: &self.layout_mesh,
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
            .ok_or_else(|| Error::Gpu("sdf_bake out: no device mirror".into()))?;
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sdf_bake out bg"),
            layout: &self.layout_out,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            }],
        });
        let uniforms = Uniforms {
            origin,
            dims: [res, tri_count, total, 0],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sdf_bake uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sdf_bake uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = total.div_ceil(WORKGROUP_SIZE).max(1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sdf_bake encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sdf_bake pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_mesh, &[]);
            pass.set_bind_group(1, &bg_out, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        out.mark_device_dirty();
        Ok(())
    }
}

/// Bake `mesh` into a signed-distance grid on the GPU. `resolution`
/// and `padding` have the same meaning as
/// [`bake_mesh_to_sdf`](crate::sdf_bake::bake_mesh_to_sdf); the two
/// produce sample-aligned grids that agree within f32 tolerance.
pub fn bake_mesh_to_sdf_gpu(
    engine: &molla_compute::Engine,
    mesh: &TriMesh,
    resolution: usize,
    padding: Real,
) -> Result<SdfGrid> {
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("sdf_bake_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("sdf_bake_gpu: engine has no queue".into()))?
        .clone();

    let res = resolution.max(2) as u32;
    if mesh.indices.is_empty() || mesh.vertices.is_empty() {
        return Ok(SdfGrid {
            nx: res,
            ny: res,
            nz: res,
            origin: Vec3::ZERO,
            cell_size: 1.0,
            data: vec![Real::INFINITY; (res * res * res) as usize],
        });
    }

    // Grid envelope — identical to the CPU path so the samples align.
    let (mut mn, mut mx) = mesh.vertices.iter().fold(
        (Vec3::splat(Real::INFINITY), Vec3::splat(-Real::INFINITY)),
        |(a, b), v| (a.min(*v), b.max(*v)),
    );
    mn -= Vec3::splat(padding);
    mx += Vec3::splat(padding);
    let extent = mx - mn;
    let max_extent = extent.x.max(extent.y).max(extent.z);
    let centre = (mn + mx) * 0.5;
    let half = Vec3::splat(max_extent * 0.5);
    let mn = centre - half;
    let cell_size = max_extent / (res as Real - 1.0);
    let total = (res * res * res) as usize;

    // Upload mesh: vertices as vec4, triangle indices flattened.
    let mut vert_pad: Vec<[f32; 4]> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let g = v.as_vec3();
        vert_pad.push([g.x, g.y, g.z, 0.0]);
    }
    let vertices_buf = make_storage(
        &device,
        "molla::sdf_bake vertices",
        bytemuck::cast_slice(&vert_pad),
    );
    queue.write_buffer(&vertices_buf, 0, bytemuck::cast_slice(&vert_pad));

    let mut idx_flat: Vec<u32> = Vec::with_capacity(mesh.indices.len() * 3);
    for tri in &mesh.indices {
        idx_flat.extend_from_slice(tri);
    }
    let indices_buf = make_storage(
        &device,
        "molla::sdf_bake indices",
        bytemuck::cast_slice(&idx_flat),
    );
    queue.write_buffer(&indices_buf, 0, bytemuck::cast_slice(&idx_flat));

    let mut out = Buffer::from_vec(vec![0.0 as Real; total], BufferUsage::StorageReadWrite);
    out.ensure_device(&device)?;
    out.upload(&queue)?;

    let kernel = SdfBakeKernel::new(&device);
    let origin = [mn.x as f32, mn.y as f32, mn.z as f32, cell_size as f32];
    kernel.dispatch(
        &device,
        &queue,
        &vertices_buf,
        &indices_buf,
        &mut out,
        origin,
        res,
        mesh.indices.len() as u32,
        total as u32,
    )?;
    out.download(&device, &queue)?;

    let data: Vec<Real> = out.host()?.to_vec();
    Ok(SdfGrid {
        nx: res,
        ny: res,
        nz: res,
        origin: mn,
        cell_size,
        data,
    })
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
