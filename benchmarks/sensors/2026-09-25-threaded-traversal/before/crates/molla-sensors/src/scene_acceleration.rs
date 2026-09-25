//! Fixed-topology instance bounds with GPU-only pose updates and bottom-up refitting.

use crate::{SceneStamp, SensorScene, dispatch::BindingsCache, readback::DeviceBuffer};
use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_kernels::{SceneBoundsBindings, SceneBoundsKernel, SceneRebuildKernel};
use molla_math::{GpuTransform, Transform, Vec3};
use molla_sim::{ShapeGeometry, State};
use molla_storage::BufferUsage;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy)]
pub struct SceneAccelerationConfig {
    pub max_leaves: u32,
    /// Fixed GPU buffer payload only; excludes host storage, staging and pipelines.
    pub max_device_bytes: u64,
}

impl Default for SceneAccelerationConfig {
    fn default() -> Self {
        Self {
            max_leaves: 1_000_000,
            max_device_bytes: 256 << 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneAccelerationStorage {
    pub leaves: u32,
    pub nodes: u32,
    pub levels: u32,
    pub device_bytes: u64,
}

impl SceneAccelerationStorage {
    pub fn for_leaves(leaves: u32) -> Result<Self> {
        let nodes = if leaves == 0 {
            0
        } else {
            leaves
                .checked_mul(2)
                .and_then(|n| n.checked_sub(1))
                .ok_or_else(|| Error::Build("scene hierarchy node count overflow".into()))?
        };
        let levels = if leaves == 0 {
            0
        } else {
            33 - (leaves - 1).leading_zeros()
        };
        let device_bytes = 116 * u64::from(leaves.max(1))
            + 48 * u64::from(nodes.max(1))
            + 32
            + 16 * u64::from(levels);
        Ok(Self {
            leaves,
            nodes,
            levels,
            device_bytes,
        })
    }
}

/// Leaves precede internal nodes. Internal children always have lower indices.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneBvhNode {
    pub left: u32,
    pub right: u32,
    /// Dense shape index for a leaf; u32::MAX for an internal node.
    pub shape: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    lo: [f32; 4],
    hi: [f32; 4],
    pose: GpuTransform,
    ids: [i32; 4],
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SceneAccelerationStats {
    pub encoded_refits: u64,
    pub submissions: u64,
    pub kernel_dispatches: u64,
    pub binding_preparations: u64,
}

/// Borrowed current bounds; enqueue consumers/readback before the next refit.
pub struct SceneAccelerationView<'a> {
    pub scene_stamp: SceneStamp,
    pub topology: &'a [SceneBvhNode],
    pub planes: &'a [u32],
    pub root: Option<u32>,
    pub node_count: usize,
    pub leaf_count: usize,
    /// XYZ bounds; the GPU vec4 padding stores aggregated invalid-leaf flags for traversal.
    pub lower: &'a DeviceBuffer<Vec3>,
    pub upper: &'a DeviceBuffer<Vec3>,
    pub world_poses: &'a DeviceBuffer<Transform>,
    /// Per leaf: 0 valid, 1 invalid pose/data, 2 unsafe arithmetic magnitude.
    pub status: &'a DeviceBuffer<u32>,
}

/// Reusable refit storage tied to one immutable scene revision, not a ray tracer.
pub struct SceneAcceleration {
    device: Arc<wgpu::Device>,
    stamp: SceneStamp,
    body_count: usize,
    storage: SceneAccelerationStorage,
    nodes: Vec<SceneBvhNode>,
    planes: Vec<u32>,
    instances: wgpu::Buffer,
    topology: wgpu::Buffer,
    fallback: DeviceBuffer<Transform>,
    lower: DeviceBuffer<Vec3>,
    upper: DeviceBuffer<Vec3>,
    world_poses: DeviceBuffer<Transform>,
    status: DeviceBuffer<u32>,
    kernel: SceneBoundsKernel,
    bindings: BindingsCache<SceneBoundsBindings>,
    stats: SceneAccelerationStats,
}

impl SceneAcceleration {
    /// Optional CPU seed poses affect tree partitioning only; refits always read live device poses.
    pub fn new(
        engine: &Engine,
        scene: &SensorScene,
        seed_poses: Option<&[Transform]>,
        config: SceneAccelerationConfig,
    ) -> Result<Self> {
        scene.check_engine(engine)?;
        let device = engine.device().expect("checked engine").clone();
        let model = &scene.model;
        if let Some(poses) = seed_poses {
            if poses.len() != model.body_count || poses.iter().any(|&p| !valid_pose(p)) {
                return Err(Error::Build(
                    "scene hierarchy: invalid topology seed poses".into(),
                ));
            }
        }
        let count = model
            .shape_geometry
            .iter()
            .filter(|&&g| finite_geometry(g, model))
            .count();
        let storage = checked_storage(count, config, &device.limits())?;
        let bodies = model.shape_body.host()?;
        let poses = model.shape_transform.host()?;
        let mesh_bounds: Vec<_> = model
            .meshes
            .iter()
            .map(|mesh| {
                mesh.vertices.iter().fold(
                    (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY)),
                    |(lo, hi), &p| (lo.min(p), hi.max(p)),
                )
            })
            .collect();
        let mut instances = Vec::with_capacity(count);
        let mut centers = Vec::with_capacity(count);
        let mut shapes = Vec::with_capacity(count);
        let mut planes = Vec::new();
        for (shape, &geometry) in model.shape_geometry.iter().enumerate() {
            if matches!(geometry, ShapeGeometry::Plane { .. }) {
                planes.push(shape as u32);
                continue;
            }
            if !finite_geometry(geometry, model) {
                continue;
            }
            let (lo, hi) = local_bounds(geometry, &mesh_bounds);
            if !lo.as_vec3().is_finite() || !hi.as_vec3().is_finite() || lo.cmpgt(hi).any() {
                return Err(Error::Build(
                    "scene hierarchy: non-finite local bounds".into(),
                ));
            }
            let pose = poses[shape];
            let mut center = pose.transform_point(lo * 0.5 + hi * 0.5);
            if bodies[shape] >= 0 {
                if let Some(seeds) = seed_poses {
                    center = seeds[bodies[shape] as usize].transform_point(center);
                }
            }
            centers.push(center);
            shapes.push(shape as u32);
            let pad = |v: Vec3, upper: bool| {
                let [x, y, z] = v.as_vec3().to_array().map(|x| {
                    if upper {
                        x.next_up().min(f32::MAX)
                    } else {
                        x.next_down().max(-f32::MAX)
                    }
                });
                [x, y, z, 0.0]
            };
            instances.push(Instance {
                lo: pad(lo, false),
                hi: pad(hi, true),
                pose: pose.into(),
                ids: [shape as i32, bodies[shape], 0, 0],
            });
        }
        let (nodes, levels) = topology(&centers, &shapes);
        let make = |data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("molla::scene hierarchy immutable input"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let dummy = Instance {
            lo: [0.0; 4],
            hi: [0.0; 4],
            pose: Transform::IDENTITY.into(),
            ids: [0; 4],
        };
        let instance_buffer = make(if instances.is_empty() {
            bytemuck::bytes_of(&dummy)
        } else {
            bytemuck::cast_slice(&instances)
        });
        let topology_buffer = make(if nodes.is_empty() {
            &[0; 16]
        } else {
            bytemuck::cast_slice(&nodes)
        });
        let n = nodes.len().max(1);
        Ok(Self {
            kernel: SceneBoundsKernel::new(&device, &levels, model.body_count as u32),
            fallback: DeviceBuffer::from_host(
                engine,
                vec![Transform::IDENTITY],
                BufferUsage::Storage,
            )?,
            lower: DeviceBuffer::from_host(
                engine,
                vec![Vec3::ZERO; n],
                BufferUsage::StorageReadWrite,
            )?,
            upper: DeviceBuffer::from_host(
                engine,
                vec![Vec3::ZERO; n],
                BufferUsage::StorageReadWrite,
            )?,
            world_poses: DeviceBuffer::from_host(
                engine,
                vec![Transform::IDENTITY; count.max(1)],
                BufferUsage::StorageReadWrite,
            )?,
            status: DeviceBuffer::from_host(
                engine,
                vec![0; count.max(1)],
                BufferUsage::StorageReadWrite,
            )?,
            device,
            stamp: scene.stamp(),
            body_count: model.body_count,
            storage,
            nodes,
            planes,
            instances: instance_buffer,
            topology: topology_buffer,
            bindings: BindingsCache::new(),
            stats: Default::default(),
        })
    }

    pub fn topology(&self) -> &[SceneBvhNode] {
        &self.nodes
    }
    pub fn planes(&self) -> &[u32] {
        &self.planes
    }
    pub fn storage(&self) -> SceneAccelerationStorage {
        self.storage
    }
    pub fn stats(&self) -> SceneAccelerationStats {
        self.stats
    }

    /// Query-only leaf permutation; the standalone CPU diagnostic shape table stays unchanged.
    pub(crate) fn query_rebuild_kernel(&self, quality: Option<u32>) -> Result<SceneRebuildKernel> {
        let mut slots = Vec::with_capacity(self.storage.leaves as usize);
        let mut stack = vec![self.storage.nodes - 1];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index as usize];
            if node.height == 0 {
                slots.push(index);
            } else {
                stack.extend([node.right, node.left]);
            }
        }
        Ok(SceneRebuildKernel::new_with_quality(
            &self.device,
            [
                self.lower.device_buffer()?,
                self.upper.device_buffer()?,
                &self.instances,
                &self.topology,
            ],
            &slots,
            self.storage.nodes - 1,
            quality,
        ))
    }

    pub fn dispatch<'a>(
        &'a mut self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
    ) -> Result<SceneAccelerationView<'a>> {
        self.preflight(engine, scene, state)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::scene hierarchy refit"),
            });
        self.encode(&mut encoder, state)?;
        engine
            .queue()
            .expect("checked engine")
            .submit([encoder.finish()]);
        self.stats.submissions += 1;
        Ok(SceneAccelerationView {
            scene_stamp: self.stamp,
            topology: &self.nodes,
            planes: &self.planes,
            root: self.nodes.len().checked_sub(1).map(|n| n as u32),
            node_count: self.nodes.len(),
            leaf_count: self.storage.leaves as usize,
            lower: &self.lower,
            upper: &self.upper,
            world_poses: &self.world_poses,
            status: &self.status,
        })
    }

    pub(crate) fn preflight(
        &self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
    ) -> Result<()> {
        scene.check_engine(engine)?;
        if !Arc::ptr_eq(engine.device().expect("checked engine"), &self.device)
            || scene.stamp() != self.stamp
        {
            return Err(Error::Gpu(
                "scene hierarchy: foreign device or stale scene revision".into(),
            ));
        }
        if self.body_count != 0 && state.body_q.len() != self.body_count {
            return Err(Error::Solver(
                "scene hierarchy: state/body count mismatch".into(),
            ));
        }
        if self.body_count != 0 {
            state.body_q.device_buffer_for(&self.device)?;
        }
        Ok(())
    }

    pub(crate) fn trace_buffers<'a>(&'a self, state: &'a State) -> Result<[&'a wgpu::Buffer; 4]> {
        Ok([
            &self.topology,
            self.lower.device_buffer()?,
            self.upper.device_buffer()?,
            if self.body_count == 0 {
                self.fallback.device_buffer()?
            } else {
                state.body_q.device_buffer_current()?
            },
        ])
    }

    pub(crate) fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        state: &State,
    ) -> Result<()> {
        if self.storage.leaves != 0 {
            let bodies = if self.body_count == 0 {
                self.fallback.device_buffer()?
            } else {
                state.body_q.device_buffer_current()?
            };
            let buffers = [
                &self.instances,
                &self.topology,
                bodies,
                self.lower.device_buffer()?,
                self.upper.device_buffer()?,
                self.world_poses.device_buffer()?,
                self.status.device_buffer()?,
            ];
            let (binding, prepared) = self.bindings.get_or_prepare(
                |b| b.matches(buffers),
                || self.kernel.prepare(&self.device, buffers),
            );
            self.stats.binding_preparations += u64::from(prepared);
            self.stats.kernel_dispatches += self.kernel.encode(encoder, binding);
            self.lower.mark_device_dirty();
            self.upper.mark_device_dirty();
            self.world_poses.mark_device_dirty();
            self.status.mark_device_dirty();
        }
        self.stats.encoded_refits += 1;
        Ok(())
    }
}

fn checked_storage(
    count: usize,
    config: SceneAccelerationConfig,
    limits: &wgpu::Limits,
) -> Result<SceneAccelerationStorage> {
    let leaves = u32::try_from(count)
        .map_err(|_| Error::Build("scene hierarchy leaf count overflow".into()))?;
    let storage = SceneAccelerationStorage::for_leaves(leaves)?;
    let largest = (u64::from(leaves.max(1)) * 80).max(u64::from(storage.nodes.max(1)) * 16);
    if leaves > config.max_leaves
        || storage.device_bytes > config.max_device_bytes
        || largest > limits.max_storage_buffer_binding_size
        || largest > limits.max_buffer_size
        || leaves.div_ceil(64) > limits.max_compute_workgroups_per_dimension
    {
        return Err(Error::Build(
            "scene hierarchy capacity exceeds allocation/device budget".into(),
        ));
    }
    Ok(storage)
}

fn valid_pose(p: Transform) -> bool {
    p.position.as_vec3().is_finite()
        && p.rotation.is_finite()
        && (p.rotation.length_squared() - 1.0).abs() < 1e-6
}

fn finite_geometry(g: ShapeGeometry, model: &molla_sim::Model) -> bool {
    match g {
        ShapeGeometry::Sphere { .. }
        | ShapeGeometry::Box { .. }
        | ShapeGeometry::Capsule { .. }
        | ShapeGeometry::Cylinder { .. }
        | ShapeGeometry::Cone { .. }
        | ShapeGeometry::Ellipsoid { .. } => true,
        ShapeGeometry::TriMesh { mesh_id } => !model.meshes[mesh_id as usize].indices.is_empty(),
        _ => false,
    }
}

fn local_bounds(g: ShapeGeometry, meshes: &[(Vec3, Vec3)]) -> (Vec3, Vec3) {
    let extent = match g {
        ShapeGeometry::Sphere { radius } => Vec3::splat(radius),
        ShapeGeometry::Box { half_extents } => half_extents,
        ShapeGeometry::Ellipsoid { half_axes } => half_axes,
        ShapeGeometry::Capsule {
            half_height,
            radius,
        } => Vec3::new(radius, half_height + radius, radius),
        ShapeGeometry::Cylinder {
            half_height,
            radius,
        }
        | ShapeGeometry::Cone {
            half_height,
            radius,
        } => Vec3::new(radius, half_height, radius),
        ShapeGeometry::TriMesh { mesh_id } => return meshes[mesh_id as usize],
        _ => unreachable!("finite shape"),
    };
    (-extent, extent)
}

fn topology(centers: &[Vec3], shapes: &[u32]) -> (Vec<SceneBvhNode>, Vec<(u32, u32)>) {
    if shapes.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut nodes: Vec<_> = shapes
        .iter()
        .map(|&shape| SceneBvhNode {
            left: u32::MAX,
            right: u32::MAX,
            shape,
            height: 0,
        })
        .collect();
    fn split(ids: &mut [u32], centers: &[Vec3], nodes: &mut Vec<SceneBvhNode>) -> u32 {
        if ids.len() == 1 {
            return ids[0];
        }
        let (lo, hi) = ids.iter().fold(
            (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY)),
            |(lo, hi), &i| (lo.min(centers[i as usize]), hi.max(centers[i as usize])),
        );
        let extent = hi - lo;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        ids.sort_unstable_by(|&a, &b| {
            centers[a as usize][axis]
                .total_cmp(&centers[b as usize][axis])
                .then(a.cmp(&b))
        });
        let mid = ids.len() / 2;
        let (a, b) = ids.split_at_mut(mid);
        let left = split(a, centers, nodes);
        let right = split(b, centers, nodes);
        let height = 1 + nodes[left as usize]
            .height
            .max(nodes[right as usize].height);
        let index = nodes.len() as u32;
        nodes.push(SceneBvhNode {
            left,
            right,
            shape: u32::MAX,
            height,
        });
        index
    }
    let mut ids: Vec<_> = (0..shapes.len() as u32).collect();
    split(&mut ids, centers, &mut nodes);
    let mut order: Vec<_> = (0..nodes.len()).collect();
    order.sort_unstable_by_key(|&i| (nodes[i].height, i));
    let mut remap = vec![0; nodes.len()];
    for (new, &old) in order.iter().enumerate() {
        remap[old] = new as u32;
    }
    let result: Vec<_> = order
        .iter()
        .map(|&i| {
            let mut node = nodes[i];
            if node.height != 0 {
                node.left = remap[node.left as usize];
                node.right = remap[node.right as usize];
            }
            node
        })
        .collect();
    let mut levels = Vec::new();
    for (i, node) in result.iter().enumerate() {
        if node.height as usize == levels.len() {
            levels.push((i as u32, 0));
        }
        levels[node.height as usize].1 += 1;
    }
    (result, levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires GPU"]
    fn estimated_device_payload_matches_constructed_hierarchy_buffers() {
        let engine = Engine::new_gpu().unwrap();
        for count in [0, 1, 17] {
            let mut builder = molla_sim::ModelBuilder::new();
            for i in 0..count {
                builder
                    .add_shape_sphere(
                        molla_core::BodyId::NONE,
                        1.0,
                        Transform::from_translation(Vec3::X * (i as f64 * 3.0)),
                    )
                    .unwrap();
            }
            let model = builder.build().unwrap();
            let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
            let usage = SceneAccelerationStorage::for_leaves(count).unwrap();
            let acceleration = SceneAcceleration::new(
                &engine,
                &scene,
                None,
                SceneAccelerationConfig {
                    max_leaves: count,
                    max_device_bytes: usage.device_bytes,
                },
            )
            .unwrap();
            let actual = acceleration.instances.size()
                + acceleration.topology.size()
                + acceleration.lower.device_buffer().unwrap().size()
                + acceleration.upper.device_buffer().unwrap().size()
                + acceleration.world_poses.device_buffer().unwrap().size()
                + acceleration.status.device_buffer().unwrap().size()
                + acceleration.fallback.device_buffer().unwrap().size()
                + u64::from(usage.levels) * 16;
            assert_eq!(actual, usage.device_bytes);
        }
    }

    #[test]
    fn balanced_topology_covers_every_leaf_and_orders_all_dependencies() {
        for n in [0, 1, 2, 3, 7, 64, 65, 1023, 1024] {
            let centers: Vec<_> = (0..n)
                .map(|i| Vec3::new(((i * 31) % 101) as f64, -4.0, 2.0))
                .collect();
            let shapes: Vec<_> = (0..n).map(|i| i as u32 * 3).collect();
            let (nodes, levels) = topology(&centers, &shapes);
            let expected = SceneAccelerationStorage::for_leaves(n as u32).unwrap();
            assert_eq!(nodes.len(), expected.nodes as usize);
            assert_eq!(levels.len(), expected.levels as usize);
            assert_eq!(levels.iter().map(|(_, n)| n).sum::<u32>(), expected.nodes);
            let mut incoming = vec![0; nodes.len()];
            for (i, node) in nodes.iter().enumerate() {
                if i < n {
                    assert_eq!(node.shape, shapes[i]);
                    assert_eq!(node.height, 0);
                } else {
                    assert_eq!(node.shape, u32::MAX);
                    for child in [node.left, node.right] {
                        assert!((child as usize) < i);
                        assert!(nodes[child as usize].height < node.height);
                        incoming[child as usize] += 1;
                    }
                }
            }
            if let Some((&last, children)) = incoming.split_last() {
                assert_eq!(last, 0);
                assert!(children.iter().all(|&n| n == 1));
                assert_eq!(nodes.last().unwrap().height + 1, expected.levels);
            }
            let mut offset = 0;
            for (height, &(start, count)) in levels.iter().enumerate() {
                assert_eq!(start, offset);
                assert!(
                    nodes[start as usize..(start + count) as usize]
                        .iter()
                        .all(|n| n.height == height as u32)
                );
                offset += count;
            }
        }
    }

    #[test]
    #[ignore = "requires GPU"]
    fn gpu_spatial_rebuild_preserves_leaf_identity_and_improves_scrambled_bounds() {
        fn read(engine: &Engine, source: &wgpu::Buffer, words: usize) -> Vec<u32> {
            let device = engine.device().unwrap();
            let queue = engine.queue().unwrap();
            let mut result =
                molla_storage::Buffer::from_vec(vec![0u32; words], BufferUsage::StorageReadWrite);
            result.ensure_device(device).unwrap();
            result.upload(queue).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(
                source,
                0,
                result.device_buffer().unwrap(),
                0,
                words as u64 * 4,
            );
            queue.submit([encoder.finish()]);
            result.mark_device_dirty();
            result.download(device, queue).unwrap();
            result.host().unwrap().to_vec()
        }
        fn vectors(words: &[u32]) -> Vec<Vec3> {
            words
                .chunks_exact(4)
                .map(|v| {
                    Vec3::new(
                        f32::from_bits(v[0]) as f64,
                        f32::from_bits(v[1]) as f64,
                        f32::from_bits(v[2]) as f64,
                    )
                })
                .collect()
        }
        fn cost(lower: &[Vec3], upper: &[Vec3], leaves: usize) -> f64 {
            lower[leaves..]
                .iter()
                .zip(&upper[leaves..])
                .map(|(&a, &b)| {
                    let d = b - a;
                    d.x * d.y + d.y * d.z + d.z * d.x
                })
                .sum()
        }
        let engine = Engine::new_gpu().unwrap();
        let n = 257usize;
        let mut builder = molla_sim::ModelBuilder::new();
        for _ in 0..n {
            let body = builder.add_body(Default::default());
            builder
                .add_shape_sphere(body, 0.4, Transform::IDENTITY)
                .unwrap();
        }
        let model = builder.build().unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let mut state = model.state().unwrap();
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let mut acceleration =
            SceneAcceleration::new(&engine, &scene, None, Default::default()).unwrap();
        let rebuild = acceleration.query_rebuild_kernel(None).unwrap();
        assert_eq!(
            rebuild.device_bytes(),
            crate::SceneRebuildStorage::for_leaves(n as u32)
                .unwrap()
                .device_bytes
        );
        let nodes = acceleration.storage.nodes as usize;
        let mut costs = Vec::new();
        for frame in 0..3 {
            let poses: Vec<_> = (0..n)
                .map(|i| {
                    Transform::from_translation(if frame == 2 {
                        Vec3::new(5.0, -3.0, 2.0)
                    } else {
                        Vec3::X * ((i * (73 + frame * 34)) % n) as f64 * 3.0
                    })
                })
                .collect();
            let packed: Vec<GpuTransform> = poses.iter().copied().map(Into::into).collect();
            engine.queue().unwrap().write_buffer(
                state.body_q.device_buffer().unwrap(),
                0,
                bytemuck::cast_slice(&packed),
            );
            state.body_q.mark_device_dirty();
            acceleration.dispatch(&engine, &scene, &state).unwrap();
            let lower = vectors(&read(
                &engine,
                acceleration.lower.device_buffer().unwrap(),
                nodes * 4,
            ));
            let upper = vectors(&read(
                &engine,
                acceleration.upper.device_buffer().unwrap(),
                nodes * 4,
            ));
            let before = cost(&lower, &upper, n);
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            assert_eq!(rebuild.encode(&mut encoder), 14);
            acceleration.encode(&mut encoder, &state).unwrap();
            engine.queue().unwrap().submit([encoder.finish()]);
            let lower = vectors(&read(
                &engine,
                acceleration.lower.device_buffer().unwrap(),
                nodes * 4,
            ));
            let upper = vectors(&read(
                &engine,
                acceleration.upper.device_buffer().unwrap(),
                nodes * 4,
            ));
            let topology = read(&engine, &acceleration.topology, nodes * 4);
            let mut identities = Vec::new();
            for (index, node) in topology.chunks_exact(4).enumerate() {
                if index < n {
                    let shape = node[2] as usize;
                    assert!(shape < n);
                    identities.push(shape);
                    let p = poses[shape].position;
                    assert!(lower[index].cmple(p - Vec3::splat(0.4)).all());
                    assert!(upper[index].cmpge(p + Vec3::splat(0.4)).all());
                    assert_eq!(node[3], 0);
                } else {
                    assert_eq!(
                        lower[index],
                        lower[node[0] as usize].min(lower[node[1] as usize])
                    );
                    assert_eq!(
                        upper[index],
                        upper[node[0] as usize].max(upper[node[1] as usize])
                    );
                }
            }
            identities.sort_unstable();
            assert_eq!(identities, (0..n).collect::<Vec<_>>());
            let after = cost(&lower, &upper, n);
            if frame < 2 {
                assert!(after < before * 0.3, "{before} -> {after}");
            } else {
                assert!((after - before).abs() < 1e-5);
            }
            costs.push((before, after));
            assert!(state.body_q.host().is_err());
        }
        eprintln!("scrambled hierarchy internal-area costs: {costs:?}");
    }

    #[test]
    #[ignore = "requires GPU"]
    fn gpu_quality_rebuild_skips_stable_motion_and_repairs_scrambled_topology() {
        fn words(engine: &Engine, source: &wgpu::Buffer, n: usize) -> Vec<u32> {
            let device = engine.device().unwrap();
            let queue = engine.queue().unwrap();
            let mut result =
                molla_storage::Buffer::from_vec(vec![0u32; n], BufferUsage::StorageReadWrite);
            result.ensure_device(device).unwrap();
            result.upload(queue).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(
                source,
                0,
                result.device_buffer().unwrap(),
                0,
                n as u64 * 4,
            );
            queue.submit([encoder.finish()]);
            result.mark_device_dirty();
            result.download(device, queue).unwrap();
            result.host().unwrap().to_vec()
        }
        let engine = Engine::new_gpu().unwrap();
        let n = 513usize;
        let mut builder = molla_sim::ModelBuilder::new();
        for _ in 0..n {
            let body = builder.add_body(Default::default());
            builder
                .add_shape_sphere(body, 0.4, Transform::IDENTITY)
                .unwrap();
        }
        let model = builder.build().unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        let mut state = model.state().unwrap();
        state
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let mut acceleration =
            SceneAcceleration::new(&engine, &scene, None, Default::default()).unwrap();
        let rebuild = acceleration.query_rebuild_kernel(Some(125)).unwrap();
        assert_eq!(
            rebuild.device_bytes(),
            crate::SceneRebuildStorage::for_quality_leaves(n as u32)
                .unwrap()
                .device_bytes
        );
        let nodes = acceleration.storage.nodes as usize;
        let mut previous_topology = Vec::new();
        let mut baseline = 0.0;
        for (frame, expected_rebuilds) in [1, 1, 1, 2, 2, 3, 3, 3, 4, 4].into_iter().enumerate() {
            let poses: Vec<GpuTransform> = (0..n)
                .map(|i| {
                    let slot = if frame >= 8 {
                        (i * 107) % n
                    } else if frame >= 3 {
                        (i * 73) % n
                    } else {
                        i
                    };
                    let p = if matches!(frame, 5 | 6) {
                        Vec3::ZERO
                    } else {
                        Vec3::X * slot as f64 * 3.0
                            + if frame == 2 {
                                Vec3::new(-10.0, 3.0, -2.0)
                            } else {
                                Vec3::ZERO
                            }
                    };
                    Transform::from_translation(p).into()
                })
                .collect();
            engine.queue().unwrap().write_buffer(
                state.body_q.device_buffer().unwrap(),
                0,
                bytemuck::cast_slice(&poses),
            );
            state.body_q.mark_device_dirty();
            if frame == 0 {
                let mut cancelled = engine
                    .device()
                    .unwrap()
                    .create_command_encoder(&Default::default());
                acceleration.encode(&mut cancelled, &state).unwrap();
                rebuild.encode(&mut cancelled);
                drop(cancelled);
                assert_eq!(
                    words(&engine, rebuild.quality_state().unwrap(), 8),
                    vec![0; 8]
                );
            }
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            acceleration.encode(&mut encoder, &state).unwrap();
            assert_eq!(rebuild.encode(&mut encoder), 17);
            acceleration.encode(&mut encoder, &state).unwrap();
            assert_eq!(rebuild.finish_quality(&mut encoder), 2);
            engine.queue().unwrap().submit([encoder.finish()]);
            let quality = words(&engine, rebuild.quality_state().unwrap(), 8);
            assert_eq!(quality[2], frame as u32 + 1);
            assert_eq!(quality[3], expected_rebuilds, "frame {frame}: {quality:?}");
            let current = f32::from_bits(quality[1]);
            assert!(current.is_finite() && current >= 1.0);
            if frame == 0 {
                baseline = current;
            }
            if frame <= 4 || frame >= 7 {
                assert!(
                    current < baseline * 1.05,
                    "repaired cost {current}, baseline {baseline}"
                );
            }
            let topology = words(&engine, &acceleration.topology, nodes * 4);
            if matches!(frame, 1 | 2 | 4 | 6 | 7 | 9) {
                assert_eq!(topology, previous_topology);
            }
            let lower = words(
                &engine,
                acceleration.lower.device_buffer().unwrap(),
                nodes * 4,
            );
            let upper = words(
                &engine,
                acceleration.upper.device_buffer().unwrap(),
                nodes * 4,
            );
            let mut ids = Vec::new();
            let extent = |index: usize, axis: usize| {
                (f32::from_bits(upper[index * 4 + axis]) as f64
                    - f32::from_bits(lower[index * 4 + axis]) as f64)
                    .max(0.0)
            };
            let area = |i: usize| {
                let d = [extent(i, 0), extent(i, 1), extent(i, 2)];
                d[0] * d[1] + d[1] * d[2] + d[2] * d[0]
            };
            let reference = (n..nodes).map(area).sum::<f64>() / area(nodes - 1);
            assert!((current as f64 - reference).abs() < reference * 1e-5);
            for i in 0..n {
                let shape = topology[i * 4 + 2] as usize;
                ids.push(shape);
                for axis in 0..3 {
                    let p = poses[shape].position[axis];
                    assert!(f32::from_bits(lower[i * 4 + axis]) <= p - 0.4);
                    assert!(f32::from_bits(upper[i * 4 + axis]) >= p + 0.4);
                }
            }
            ids.sort_unstable();
            assert_eq!(ids, (0..n).collect::<Vec<_>>());
            previous_topology = topology;
            assert!(state.body_q.host().is_err());
        }
    }

    #[test]
    fn hierarchy_device_budget_and_dispatch_limits_precede_allocation() {
        let limits = wgpu::Limits::default();
        for count in [0, 1, 17, 1_000_000] {
            let usage = SceneAccelerationStorage::for_leaves(count).unwrap();
            let config = SceneAccelerationConfig {
                max_leaves: count,
                max_device_bytes: usage.device_bytes,
            };
            assert_eq!(
                checked_storage(count as usize, config, &limits).unwrap(),
                usage
            );
            assert!(
                checked_storage(
                    count as usize,
                    SceneAccelerationConfig {
                        max_device_bytes: usage.device_bytes - 1,
                        ..config
                    },
                    &limits
                )
                .is_err()
            );
        }
        assert_eq!(
            SceneAccelerationStorage::for_leaves(0)
                .unwrap()
                .device_bytes,
            196
        );
        assert!(SceneAccelerationStorage::for_leaves(u32::MAX).is_err());
        assert!(checked_storage(usize::MAX, Default::default(), &limits).is_err());
        assert!(
            checked_storage(
                2,
                SceneAccelerationConfig {
                    max_leaves: 1,
                    ..Default::default()
                },
                &limits
            )
            .is_err()
        );
        assert!(
            checked_storage(
                65,
                Default::default(),
                &wgpu::Limits {
                    max_compute_workgroups_per_dimension: 1,
                    ..limits.clone()
                }
            )
            .is_err()
        );
        assert!(
            checked_storage(
                0,
                Default::default(),
                &wgpu::Limits {
                    max_storage_buffer_binding_size: 79,
                    ..limits.clone()
                }
            )
            .is_err()
        );
        assert!(
            checked_storage(
                0,
                Default::default(),
                &wgpu::Limits {
                    max_buffer_size: 79,
                    ..limits
                }
            )
            .is_err()
        );
    }

    #[test]
    fn local_bounds_cover_radial_shapes_and_shared_meshes() {
        for (geometry, extent) in [
            (ShapeGeometry::Sphere { radius: 2.0 }, Vec3::splat(2.0)),
            (
                ShapeGeometry::Box {
                    half_extents: Vec3::new(1.0, 2.0, 3.0),
                },
                Vec3::new(1.0, 2.0, 3.0),
            ),
            (
                ShapeGeometry::Capsule {
                    half_height: 3.0,
                    radius: 2.0,
                },
                Vec3::new(2.0, 5.0, 2.0),
            ),
            (
                ShapeGeometry::Cylinder {
                    half_height: 3.0,
                    radius: 2.0,
                },
                Vec3::new(2.0, 3.0, 2.0),
            ),
            (
                ShapeGeometry::Cone {
                    half_height: 3.0,
                    radius: 2.0,
                },
                Vec3::new(2.0, 3.0, 2.0),
            ),
            (
                ShapeGeometry::Ellipsoid {
                    half_axes: Vec3::new(2.0, 3.0, 4.0),
                },
                Vec3::new(2.0, 3.0, 4.0),
            ),
        ] {
            assert_eq!(local_bounds(geometry, &[]), (-extent, extent));
        }
        let meshes = [(Vec3::new(-2.0, -3.0, -4.0), Vec3::new(1.0, 7.0, 8.0))];
        assert_eq!(
            local_bounds(ShapeGeometry::TriMesh { mesh_id: 0 }, &meshes),
            meshes[0]
        );
    }
}
