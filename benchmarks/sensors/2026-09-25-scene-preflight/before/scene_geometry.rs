//! Packed immutable shape records and shared local-space mesh acceleration.

use molla_core::{Error, Result};
use molla_geometry::{MeshBvh, MeshBvhGpu};
use molla_math::GpuTransform;
use molla_sim::{Model, ShapeGeometry};
use wgpu::util::DeviceExt;

pub(crate) const DEFAULT_QUERY_BYTES: u64 = 256 << 20;
const SHAPE_WORDS: usize = 24;

pub(crate) struct SceneGeometry {
    pub shapes: wgpu::Buffer,
    pub meshes: wgpu::Buffer,
    pub plane_offset: u32,
    pub plane_count: u32,
    pub finite_count: u32,
    pub device_bytes: u64,
}

#[derive(Clone, Copy, Default)]
struct MeshRange {
    nodes: u32,
    order: u32,
    indices: u32,
    vertices: u32,
    node_count: u32,
}

fn words(count: usize, stride: usize) -> Result<usize> {
    count
        .checked_mul(stride)
        .ok_or_else(|| Error::Build("scene query size overflow".into()))
}

fn checked_payload(
    shape_words: usize,
    mesh_words: usize,
    budget: u64,
    limits: &wgpu::Limits,
) -> Result<u64> {
    let sizes = [shape_words.max(SHAPE_WORDS), mesh_words.max(1)];
    let mut total = 0u64;
    for size in sizes {
        let bytes = u64::try_from(size)
            .ok()
            .and_then(|s| s.checked_mul(4))
            .ok_or_else(|| Error::Build("scene query byte size overflow".into()))?;
        if size > u32::MAX as usize
            || bytes > limits.max_buffer_size
            || bytes > limits.max_storage_buffer_binding_size
        {
            return Err(Error::Build(
                "scene query exceeds GPU buffer/address limits".into(),
            ));
        }
        total = total
            .checked_add(bytes)
            .ok_or_else(|| Error::Build("scene query byte sum overflow".into()))?;
    }
    if total > budget {
        return Err(Error::Build("scene query exceeds allocation budget".into()));
    }
    Ok(total)
}

impl SceneGeometry {
    pub fn new(device: &wgpu::Device, model: &Model, budget: u64) -> Result<Self> {
        let kinds = model.shape_geom_kind.host()?;
        let plane_count = kinds.iter().filter(|&&k| k == 1).count();
        let plane_offset = words(model.shape_count, SHAPE_WORDS)?;
        let shape_words = plane_offset
            .checked_add(plane_count)
            .ok_or_else(|| Error::Build("scene shape record size overflow".into()))?;
        let mut mesh_words = 0usize;
        for mesh in &model.meshes {
            mesh_words = mesh_words
                .checked_add(words(mesh.vertices.len(), 4)?)
                .and_then(|s| s.checked_add(mesh.indices.len().checked_mul(4)?))
                .ok_or_else(|| Error::Build("scene mesh payload size overflow".into()))?;
        }
        checked_payload(shape_words, mesh_words, budget, &device.limits())?;
        let mut trees: Vec<MeshBvh> = Vec::with_capacity(model.meshes.len());
        for mesh in &model.meshes {
            let tree = MeshBvhGpu::build_validated(device, &mesh.vertices, &mesh.indices)?;
            mesh_words = mesh_words
                .checked_add(words(tree.nodes.len(), 12)?)
                .ok_or_else(|| Error::Build("scene mesh tree size overflow".into()))?;
            checked_payload(shape_words, mesh_words, budget, &device.limits())?;
            trees.push(tree);
        }
        let device_bytes = checked_payload(shape_words, mesh_words, budget, &device.limits())?;
        let mut packed = Vec::<u32>::with_capacity(mesh_words.max(1));
        let mut ranges = Vec::with_capacity(trees.len());
        for (mesh, tree) in model.meshes.iter().zip(trees) {
            let nodes = packed.len() as u32;
            for node in &tree.nodes {
                let lo = node.aabb_min.as_vec3();
                let hi = node.aabb_max.as_vec3();
                packed.extend([lo.x.to_bits(), lo.y.to_bits(), lo.z.to_bits(), node.is_leaf]);
                packed.extend([hi.x.to_bits(), hi.y.to_bits(), hi.z.to_bits(), 0]);
                packed.extend([node.prim_start, node.tri_count, node.left, node.right]);
            }
            crate::scene_threading::thread_tree(
                (!tree.nodes.is_empty()).then_some(0),
                |i| {
                    let node = tree.nodes[i as usize];
                    (node.is_leaf == 0).then_some([node.right, node.left])
                },
                |i, escape| packed[nodes as usize + i as usize * 12 + 7] = escape,
            );
            let order = packed.len() as u32;
            packed.extend_from_slice(&tree.triangle_order);
            let indices = packed.len() as u32;
            packed.extend(mesh.indices.iter().flatten().copied());
            let vertices = packed.len() as u32;
            for v in &mesh.vertices {
                let v = v.as_vec3();
                packed.extend([v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), 0]);
            }
            ranges.push(MeshRange {
                nodes,
                order,
                indices,
                vertices,
                node_count: tree.nodes.len() as u32,
            });
        }
        debug_assert_eq!(packed.len(), mesh_words);
        packed.resize(mesh_words.max(1), 0);
        let mut shapes = vec![0u32; shape_words.max(SHAPE_WORDS)];
        let data = model.shape_geom_data.host()?;
        let bodies = model.shape_body.host()?;
        let poses = model.shape_transform.host()?;
        let mut plane = plane_offset;
        let mut finite_count = 0;
        for i in 0..model.shape_count {
            let offset = i * SHAPE_WORDS;
            shapes[offset] = kinds[i];
            shapes[offset + 1] = bodies[i] as u32;
            let v = data[i].as_vec3();
            shapes[offset + 4..offset + 8].copy_from_slice(&[
                v.x.to_bits(),
                v.y.to_bits(),
                v.z.to_bits(),
                0,
            ]);
            let pose = GpuTransform::from(poses[i]);
            shapes[offset + 8..offset + 16].copy_from_slice(bytemuck::cast_slice(&[pose]));
            if let ShapeGeometry::TriMesh { mesh_id } = model.shape_geometry[i] {
                let r = ranges[mesh_id as usize];
                shapes[offset + 16..offset + 21].copy_from_slice(&[
                    r.nodes,
                    r.order,
                    r.indices,
                    r.vertices,
                    r.node_count,
                ]);
            }
            if kinds[i] == 1 {
                shapes[plane] = i as u32;
                plane += 1;
            } else if kinds[i] != 0xff || shapes[offset + 20] != 0 {
                finite_count += 1;
            }
        }
        let upload = |label, data: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        Ok(Self {
            shapes: upload("molla::scene shape records", &shapes),
            meshes: upload("molla::scene mesh atlas", &packed),
            plane_offset: plane_offset as u32,
            plane_count: plane_count as u32,
            finite_count,
            device_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_geometry_budget_and_address_boundaries() {
        let limits = wgpu::Limits::default();
        assert_eq!(checked_payload(0, 0, 100, &limits).unwrap(), 100);
        assert!(checked_payload(0, 0, 99, &limits).is_err());
        assert_eq!(checked_payload(49, 123, 688, &limits).unwrap(), 688);
        assert!(checked_payload(49, 123, 687, &limits).is_err());
        assert!(checked_payload(usize::MAX, 1, u64::MAX, &limits).is_err());
        assert!(
            checked_payload(
                24,
                limits.max_storage_buffer_binding_size as usize / 4 + 1,
                u64::MAX,
                &limits
            )
            .is_err()
        );
        assert!(words(usize::MAX, 24).is_err());
    }

    #[test]
    #[ignore = "requires GPU"]
    fn atlas_payload_matches_allocated_buffers_and_shared_mesh_storage() {
        let engine = molla_compute::Engine::new_gpu().unwrap();
        let mut builder = molla_sim::ModelBuilder::new();
        let mesh = builder.add_mesh(molla_sim::TriMesh {
            vertices: vec![
                -molla_math::Vec3::X,
                molla_math::Vec3::X,
                molla_math::Vec3::Z,
            ],
            indices: vec![[0, 1, 2]],
        });
        for i in 0..65 {
            builder
                .add_shape_trimesh(
                    molla_core::BodyId::NONE,
                    mesh,
                    molla_math::Transform::from_translation(molla_math::Vec3::X * i as f64),
                )
                .unwrap();
        }
        builder
            .add_shape_plane(
                molla_core::BodyId::NONE,
                molla_math::Vec3::Y,
                molla_math::Transform::IDENTITY,
            )
            .unwrap();
        let model = builder.build().unwrap();
        let scene = crate::SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let geometry = &scene.geometry;
        assert_eq!(
            geometry.device_bytes,
            geometry.shapes.size() + geometry.meshes.size()
        );
        assert_eq!(geometry.shapes.size(), 66 * 96 + 4);
        assert_eq!(geometry.meshes.size(), 48 + 4 + 12 + 3 * 16);
        assert_eq!(geometry.plane_offset, 66 * 24);
        assert_eq!(geometry.plane_count, 1);
    }
}
