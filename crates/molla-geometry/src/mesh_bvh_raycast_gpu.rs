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
use molla_math::{Real, Transform, Vec3};
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    ray_count: u32,
    node_count: u32,
    max_distance: f32,
    shape_id: u32,
    body_id: i32,
    _pad: [u32; 3],
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
const MAX_TREE_DEPTH: u32 = 64;
const RAYCAST_LEAF_SIZE: usize = 8;

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
    /// Validate local mesh data, build a balanced BVH and upload it once.
    pub fn from_mesh(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[Vec3],
        indices: &[[u32; 3]],
    ) -> Result<Self> {
        let bvh = Self::build_validated(device, vertices, indices)?;
        Self::upload_validated(device, queue, &bvh, vertices, indices)
    }

    /// Build and validate local geometry and tree limits without GPU allocation.
    pub fn build_validated(
        device: &wgpu::Device,
        vertices: &[Vec3],
        indices: &[[u32; 3]],
    ) -> Result<MeshBvh> {
        validate_mesh(device, vertices, indices)?;
        let bvh = MeshBvh::build(vertices, indices, RAYCAST_LEAF_SIZE);
        Self::validate(device, &bvh, vertices, indices)?;
        Ok(bvh)
    }

    /// Exact node count of the median-split raycast tree, without building its arrays.
    pub fn node_count_for_mesh(triangles: usize) -> Result<usize> {
        if triangles > u32::MAX as usize {
            return Err(Error::Gpu("mesh triangle count exceeds u32".into()));
        }
        MeshBvh::planned_node_count(triangles, RAYCAST_LEAF_SIZE)
            .ok_or_else(|| Error::Gpu("mesh node count exceeds usize".into()))
    }

    /// Upload `bvh` plus `(vertices, indices)` to the device.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bvh: &MeshBvh,
        vertices: &[Vec3],
        indices: &[[u32; 3]],
    ) -> Result<Self> {
        Self::validate(device, bvh, vertices, indices)?;
        Self::upload_validated(device, queue, bvh, vertices, indices)
    }

    /// Validate mesh/tree geometry and device-size limits without allocating GPU buffers.
    pub fn validate(
        device: &wgpu::Device,
        bvh: &MeshBvh,
        vertices: &[Vec3],
        indices: &[[u32; 3]],
    ) -> Result<()> {
        validate_mesh(device, vertices, indices)?;
        validate_tree(bvh, indices.len())?;
        for node in bvh.nodes.iter().filter(|node| node.is_leaf == 1) {
            for &triangle in &bvh.triangle_order
                [node.prim_start as usize..(node.prim_start + node.tri_count) as usize]
            {
                for index in indices[triangle as usize] {
                    let vertex = vertices[index as usize];
                    if !node.aabb_min.cmple(vertex).all() || !vertex.cmple(node.aabb_max).all() {
                        return Err(Error::Gpu(
                            "mesh BVH bounds exclude triangle vertices".into(),
                        ));
                    }
                }
            }
        }
        check_bytes(device, bvh.nodes.len(), 48)?;
        check_bytes(device, bvh.triangle_order.len(), 4)?;
        Ok(())
    }

    fn upload_validated(
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
        molla_core::gpu_write_buffer!(queue, &nodes, 0, bytemuck::cast_slice(&nodes_gpu));

        let tri_order_data: Vec<u32> = bvh.triangle_order.clone();
        let tri_order = make_storage(
            device,
            "molla::mesh_bvh tri_order",
            bytemuck::cast_slice(&tri_order_data),
        )?;
        molla_core::gpu_write_buffer!(queue, &tri_order, 0, bytemuck::cast_slice(&tri_order_data));

        let mut idx_flat: Vec<u32> = Vec::with_capacity(indices.len() * 3);
        for tri in indices {
            idx_flat.extend_from_slice(tri);
        }
        let indices_buf = make_storage(
            device,
            "molla::mesh_bvh indices",
            bytemuck::cast_slice(&idx_flat),
        )?;
        molla_core::gpu_write_buffer!(queue, &indices_buf, 0, bytemuck::cast_slice(&idx_flat));

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
        molla_core::gpu_write_buffer!(queue, &vertices_buf, 0, bytemuck::cast_slice(&vert_pad));

        Ok(Self {
            nodes,
            tri_order,
            indices: indices_buf,
            vertices: vertices_buf,
            node_count: bvh.nodes.len() as u32,
        })
    }
}

fn check_bytes(device: &wgpu::Device, count: usize, stride: usize) -> Result<()> {
    let bytes = count
        .checked_mul(stride)
        .ok_or_else(|| Error::Gpu("mesh BVH size overflow".into()))? as u64;
    if bytes > device.limits().max_buffer_size
        || bytes > device.limits().max_storage_buffer_binding_size
    {
        return Err(Error::Gpu("mesh BVH exceeds GPU buffer limits".into()));
    }
    Ok(())
}

fn validate_mesh(device: &wgpu::Device, vertices: &[Vec3], indices: &[[u32; 3]]) -> Result<()> {
    check_bytes(device, vertices.len(), 16)?;
    check_bytes(device, indices.len(), 12)?;
    if vertices.len() > u32::MAX as usize
        || indices.len() > i32::MAX as usize / 3
        || vertices.iter().any(|v| !v.as_vec3().is_finite())
        || indices
            .iter()
            .flatten()
            .any(|&i| i as usize >= vertices.len())
    {
        return Err(Error::Gpu("invalid mesh vertices or indices".into()));
    }
    Ok(())
}

fn validate_tree(bvh: &MeshBvh, triangles: usize) -> Result<()> {
    let bad = || Error::Gpu("invalid or over-depth mesh BVH".into());
    if bvh.nodes.is_empty() {
        return if triangles == 0 && bvh.triangle_order.is_empty() {
            Ok(())
        } else {
            Err(bad())
        };
    }
    if bvh.triangle_order.len() != triangles
        || bvh.triangle_order.iter().any(|&i| i as usize >= triangles)
    {
        return Err(bad());
    }
    let mut visited = vec![false; bvh.nodes.len()];
    let mut ordered = vec![false; triangles];
    for &triangle in &bvh.triangle_order {
        if ordered[triangle as usize] {
            return Err(bad());
        }
        ordered[triangle as usize] = true;
    }
    let mut covered = vec![false; triangles];
    let mut stack = vec![(0usize, 1u32)];
    while let Some((index, depth)) = stack.pop() {
        if index >= bvh.nodes.len() || visited[index] || depth > MAX_TREE_DEPTH {
            return Err(bad());
        }
        visited[index] = true;
        let node = bvh.nodes[index];
        if !node.aabb_min.as_vec3().is_finite()
            || !node.aabb_max.as_vec3().is_finite()
            || !node.aabb_min.cmple(node.aabb_max).all()
        {
            return Err(bad());
        }
        if node.is_leaf == 1 {
            if (node.prim_start as usize)
                .checked_add(node.tri_count as usize)
                .is_none_or(|end| end > triangles)
            {
                return Err(bad());
            }
            for slot in
                &mut covered[node.prim_start as usize..(node.prim_start + node.tri_count) as usize]
            {
                if *slot {
                    return Err(bad());
                }
                *slot = true;
            }
        } else if node.is_leaf == 0 {
            for child in [node.left, node.right] {
                let child = bvh.nodes.get(child as usize).ok_or_else(bad)?;
                if !node.aabb_min.cmple(child.aabb_min).all()
                    || !child.aabb_max.cmple(node.aabb_max).all()
                {
                    return Err(bad());
                }
            }
            stack.push((node.left as usize, depth + 1));
            stack.push((node.right as usize, depth + 1));
        } else {
            return Err(bad());
        }
    }
    if visited.iter().any(|&v| !v) || covered.iter().any(|&v| !v) {
        return Err(bad());
    }
    Ok(())
}

fn make_storage(device: &wgpu::Device, label: &str, data: &[u8]) -> Result<wgpu::Buffer> {
    if data.is_empty() {
        // wgpu refuses zero-sized buffers; allocate a 16-byte stub so
        // the binding is valid (the kernel's `node_count == 0` early-out
        // means we never read it).
        return Ok(molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        ));
    }
    Ok(molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    ))
}

pub struct MeshBvhRaycastKernel {
    pipeline: wgpu::ComputePipeline,
    layout_mesh: wgpu::BindGroupLayout,
    layout_rays: wgpu::BindGroupLayout,
    layout_hits: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    identity: wgpu::Buffer,
}

/// Retained mesh, ray, hit and rigid-instance bindings.
pub struct MeshBvhRaycastBindings {
    buffers: [wgpu::Buffer; 11],
    groups: [wgpu::BindGroup; 4],
    uniform: wgpu::Buffer,
}

impl MeshBvhRaycastBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 11]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl MeshBvhRaycastKernel {
    /// WGSL triangle intersection and normal calculation shared by mesh query kernels.
    pub const TRIANGLE_SHADER: &'static str = include_str!("../shaders/ray_triangle.wgsl");
    /// Bind nodes/order/indices/vertices, rays, t/triangle/normal, body/shape poses.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 11],
    ) -> MeshBvhRaycastBindings {
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::mesh retained uniform"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let make_group = |layout, buffers: &[&wgpu::Buffer]| {
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::mesh retained bindings"),
                layout,
                entries: &entries,
            })
        };
        MeshBvhRaycastBindings {
            groups: [
                make_group(&self.layout_mesh, &buffers[..4]),
                make_group(&self.layout_rays, &buffers[4..6]),
                make_group(&self.layout_hits, &buffers[6..9]),
                make_group(&self.layout_uniforms, &[&uniform, buffers[9], buffers[10]]),
            ],
            buffers: std::array::from_fn(|i| buffers[i].clone()),
            uniform,
        }
    }

    /// Encode one instance with metric hit distances and range clipping.
    /// Submit before reusing its uniform for different parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &MeshBvhRaycastBindings,
        ray_count: u32,
        node_count: u32,
        max_distance: f32,
        shape_id: u32,
        body_id: i32,
    ) {
        if ray_count == 0 {
            return;
        }
        molla_core::gpu_write_buffer!(
            queue,
            &bindings.uniform,
            0,
            bytemuck::bytes_of(&Uniforms {
                ray_count,
                node_count,
                max_distance,
                shape_id,
                body_id,
                _pad: [0; 3],
            }),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::mesh prepared"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (i, group) in bindings.groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
        pass.dispatch_workgroups(ray_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::mesh_bvh_raycast"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    Self::TRIANGLE_SHADER,
                    include_str!("../shaders/mesh_bvh_raycast.wgsl")
                )
                .into(),
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
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entries(3, true)[1],
                storage_entries(3, true)[2],
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::mesh_bvh_raycast layout"),
            bind_group_layouts: &[
                Some(&layout_mesh),
                Some(&layout_rays),
                Some(&layout_hits),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::mesh_bvh_raycast pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("raycast_mesh"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        use wgpu::util::DeviceExt;
        let identity_data: molla_math::GpuTransform = Transform::IDENTITY.into();
        let identity = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("molla::mesh identity"),
                contents: bytemuck::bytes_of(&identity_data),
                usage: wgpu::BufferUsages::STORAGE,
            }
        );
        Self {
            identity,
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
        self.dispatch_instance(
            device,
            queue,
            mesh,
            ray_origin,
            ray_direction,
            hit_t,
            hit_tri,
            hit_normal,
            max_distance,
            ray_count,
            &self.identity,
            &self.identity,
            0,
            -1,
        )
    }

    /// Cast world rays against a local-space mesh at a body-relative rigid shape pose.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_instance(
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
        body_poses: &wgpu::Buffer,
        shape_poses: &wgpu::Buffer,
        shape_id: u32,
        body_id: i32,
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
            shape_id,
            body_id,
            _pad: [0; 3],
        };
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::mesh_bvh_raycast uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::mesh_bvh_raycast uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: body_poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shape_poses.as_entire_binding(),
                },
            ],
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

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn planned_raycast_node_count_matches_median_trees() {
        let vertices = [Vec3::ZERO, Vec3::X, Vec3::Z];
        for count in [
            0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 49, 255, 256, 257, 1000, 4097,
        ] {
            let indices = vec![[0, 1, 2]; count];
            let tree = MeshBvh::build(&vertices, &indices, RAYCAST_LEAF_SIZE);
            assert_eq!(
                MeshBvhGpu::node_count_for_mesh(count).unwrap(),
                tree.nodes.len(),
                "triangles={count}"
            );
        }
        if let Some(too_many) = (u32::MAX as usize).checked_add(1) {
            assert!(MeshBvhGpu::node_count_for_mesh(too_many).is_err());
        }
        assert!(MeshBvhGpu::node_count_for_mesh(u32::MAX as usize).unwrap() < u32::MAX as usize);
    }
    use crate::MeshBvhNode;

    fn chain(levels: usize) -> MeshBvh {
        let leaf = |start| MeshBvhNode {
            aabb_min: Vec3::ZERO,
            aabb_max: Vec3::ONE,
            prim_start: start,
            tri_count: 1,
            left: 0,
            right: 0,
            is_leaf: 1,
        };
        let mut nodes = Vec::new();
        for level in 0..levels {
            let left = nodes.len() as u32 + 1;
            nodes.push(MeshBvhNode {
                is_leaf: 0,
                left,
                right: left + 1,
                ..leaf(0)
            });
            nodes.push(leaf(level as u32));
        }
        nodes.push(leaf(levels as u32));
        MeshBvh {
            nodes,
            triangle_order: (0..levels as u32 + 1).collect(),
        }
    }

    #[test]
    fn upload_validation_bounds_stack_depth_and_matches_shader() {
        assert!(
            include_str!("../shaders/mesh_bvh_raycast.wgsl")
                .contains(&format!("const MAX_STACK: u32 = {MAX_TREE_DEPTH}u;"))
        );
        assert!(validate_tree(&chain(63), 64).is_ok());
        assert!(validate_tree(&chain(64), 65).is_err());
    }

    #[test]
    fn upload_validation_rejects_cycles_ranges_and_nonfinite_bounds() {
        let mut cyclic = chain(2);
        cyclic.nodes[0].right = 0;
        assert!(validate_tree(&cyclic, 3).is_err());
        let mut invalid_leaf = chain(2);
        invalid_leaf.nodes[1].tri_count = 5;
        assert!(validate_tree(&invalid_leaf, 3).is_err());
        let mut invalid_index = chain(2);
        invalid_index.triangle_order[0] = 99;
        assert!(validate_tree(&invalid_index, 3).is_err());
        let mut nonfinite = chain(2);
        nonfinite.nodes[0].aabb_min = Vec3::splat(f64::NAN);
        assert!(validate_tree(&nonfinite, 3).is_err());
        let mut omitted = chain(2);
        omitted.nodes[1].tri_count = 0;
        assert!(validate_tree(&omitted, 3).is_err());
        let mut duplicate = chain(2);
        duplicate.triangle_order[1] = 0;
        assert!(validate_tree(&duplicate, 3).is_err());
    }
}
