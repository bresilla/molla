//! GPU mesh-BVH ray-cast kernel. Mirrors the CPU
//! [`MeshBvh::ray_cast`](crate::mesh_bvh::MeshBvh::ray_cast)
//! but operates on a flat node array uploaded to the device.
//!
//! Two pieces:
//!  - [`MeshBvhGpu`] flattens a [`crate::MeshBvh`] (and its
//!    backing vertex/index arrays) into device buffers.
//!  - [`MeshBvhRaycastKernel`] dispatches the WGSL kernel against
//!    those buffers + a per-ray origin/direction bundle.
//!
//! Used by Stage 14's camera-sensor mesh-shape rendering and Stage
//! 15's mesh-vs-mesh narrow phase.

use crate::MeshBvh;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    ray_count: u32,
    node_count: u32,
    max_distance: f32,
    _pad: f32,
}

/// Flat-buffer layout of one BVH node — matches the WGSL `BvhNode`:
///   `aabb_min.xyz, is_leaf_as_f32`
///   `aabb_max.xyz, _pad`
///   `prim_start, tri_count, left, right`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBvhNode {
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
    pub payload: [u32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// Per-mesh device buffers needed by the kernel: BVH nodes,
/// triangle-order permutation, flat triangle indices, and vertex
/// positions.
pub struct MeshBvhGpu {
    pub nodes: wgpu::Buffer,
    pub tri_order: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub vertices: wgpu::Buffer,
    pub node_count: u32,
}

impl MeshBvhGpu {
    /// Upload `bvh` plus `(vertices, indices)` to the device.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bvh: &MeshBvh,
        vertices: &[Vec3],
        indices: &[[u32; 3]],
    ) -> Result<Self> {
        let nodes_gpu: Vec<GpuBvhNode> = bvh
            .nodes
            .iter()
            .map(|n| {
                let mn = n.aabb_min.as_vec3();
                let mx = n.aabb_max.as_vec3();
                GpuBvhNode {
                    aabb_min: [mn.x, mn.y, mn.z, n.is_leaf as f32],
                    aabb_max: [mx.x, mx.y, mx.z, 0.0],
                    payload: [n.prim_start, n.tri_count, n.left, n.right],
                }
            })
            .collect();
        let nodes = make_storage(
            device,
            "molla::mesh_bvh nodes",
            bytemuck::cast_slice(&nodes_gpu),
        )?;
        queue.write_buffer(&nodes, 0, bytemuck::cast_slice(&nodes_gpu));

        let tri_order_data: Vec<u32> = bvh.triangle_order.clone();
        let tri_order = make_storage(
            device,
            "molla::mesh_bvh tri_order",
            bytemuck::cast_slice(&tri_order_data),
        )?;
        queue.write_buffer(&tri_order, 0, bytemuck::cast_slice(&tri_order_data));

        let mut idx_flat: Vec<u32> = Vec::with_capacity(indices.len() * 3);
        for tri in indices {
            idx_flat.extend_from_slice(tri);
        }
        let indices_buf = make_storage(
            device,
            "molla::mesh_bvh indices",
            bytemuck::cast_slice(&idx_flat),
        )?;
        queue.write_buffer(&indices_buf, 0, bytemuck::cast_slice(&idx_flat));

        let mut vert_pad: Vec<[f32; 4]> = Vec::with_capacity(vertices.len());
        for v in vertices {
            let g = v.as_vec3();
            vert_pad.push([g.x, g.y, g.z, 0.0]);
        }
        let vertices_buf = make_storage(
            device,
            "molla::mesh_bvh vertices",
            bytemuck::cast_slice(&vert_pad),
        )?;
        queue.write_buffer(&vertices_buf, 0, bytemuck::cast_slice(&vert_pad));

        Ok(Self {
            nodes,
            tri_order,
            indices: indices_buf,
            vertices: vertices_buf,
            node_count: bvh.nodes.len() as u32,
        })
    }
}

fn make_storage(device: &wgpu::Device, label: &str, data: &[u8]) -> Result<wgpu::Buffer> {
    if data.is_empty() {
        // wgpu refuses zero-sized buffers; allocate a 16-byte stub so
        // the binding is valid (the kernel's `node_count == 0` early-out
        // means we never read it).
        return Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    Ok(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: data.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }))
}

pub struct MeshBvhRaycastKernel {
    pipeline: wgpu::ComputePipeline,
    layout_mesh: wgpu::BindGroupLayout,
    layout_rays: wgpu::BindGroupLayout,
    layout_hits: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl MeshBvhRaycastKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::mesh_bvh_raycast"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mesh_bvh_raycast.wgsl").into(),
            ),
        });
        let layout_mesh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mesh_bvh_raycast mesh"),
            entries: &storage_entries(4, true),
        });
        let layout_rays = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mesh_bvh_raycast rays"),
            entries: &storage_entries(2, true),
        });
        let layout_hits = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mesh_bvh_raycast hits"),
            entries: &storage_entries(3, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::mesh_bvh_raycast uniforms"),
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
            label: Some("molla::mesh_bvh_raycast layout"),
            bind_group_layouts: &[&layout_mesh, &layout_rays, &layout_hits, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::mesh_bvh_raycast pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("raycast_mesh"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_mesh,
            layout_rays,
            layout_hits,
            layout_uniforms,
        }
    }

    /// Cast `ray_count` rays against the mesh, writing per-ray hits.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &MeshBvhGpu,
        ray_origin: &Buffer<Vec3>,
        ray_direction: &Buffer<Vec3>,
        hit_t: &mut Buffer<Real>,
        hit_tri: &mut Buffer<i32>,
        hit_normal: &mut Buffer<Vec3>,
        max_distance: f32,
        ray_count: u32,
    ) -> Result<()> {
        if ray_count == 0 {
            return Ok(());
        }
        let bg_mesh = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mesh_bvh_raycast mesh bg"),
            layout: &self.layout_mesh,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh.tri_order.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh.indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mesh.vertices.as_entire_binding(),
                },
            ],
        });
        let bg_rays = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mesh_bvh_raycast rays bg"),
            layout: &self.layout_rays,
            entries: &[
                buffer_entry(0, ray_origin.device_buffer())?,
                buffer_entry(1, ray_direction.device_buffer())?,
            ],
        });
        let bg_hits = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mesh_bvh_raycast hits bg"),
            layout: &self.layout_hits,
            entries: &[
                buffer_entry(0, hit_t.device_buffer())?,
                buffer_entry(1, hit_tri.device_buffer())?,
                buffer_entry(2, hit_normal.device_buffer())?,
            ],
        });
        let uniforms = Uniforms {
            ray_count,
            node_count: mesh.node_count,
            max_distance,
            _pad: 0.0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::mesh_bvh_raycast uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mesh_bvh_raycast uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = ray_count.div_ceil(WORKGROUP_SIZE).max(1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::mesh_bvh_raycast encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::mesh_bvh_raycast pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_mesh, &[]);
            pass.set_bind_group(1, &bg_rays, &[]);
            pass.set_bind_group(2, &bg_hits, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        hit_t.mark_device_dirty();
        hit_tri.mark_device_dirty();
        hit_normal.mark_device_dirty();
        Ok(())
    }

    /// Convenience helper to allocate the per-ray hit buffers sized
    /// for `ray_count` rays.
    pub fn alloc_hits(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ray_count: usize,
    ) -> Result<(Buffer<Real>, Buffer<i32>, Buffer<Vec3>)> {
        let n = ray_count.max(1);
        let mut hit_t = Buffer::from_vec(vec![Real::INFINITY; n], BufferUsage::StorageReadWrite);
        let mut hit_tri = Buffer::from_vec(vec![-1i32; n], BufferUsage::StorageReadWrite);
        let mut hit_normal = Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);
        hit_t.ensure_device(device)?;
        hit_tri.ensure_device(device)?;
        hit_normal.ensure_device(device)?;
        hit_t.upload(queue)?;
        hit_tri.upload(queue)?;
        hit_normal.upload(queue)?;
        Ok((hit_t, hit_tri, hit_normal))
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "mesh_bvh_raycast binding {binding}: no device mirror"
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
