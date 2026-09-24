//! GPU closest-hit queries over analytical primitives and rigid mesh instances.
//!
//! Mesh candidates are merged on device. CPU inputs are normalized and uploaded
//! once per batch; collection is optional. Local mesh BVHs are reused across poses.
//! Owned scenes share immutable geometry; legacy model queries compare mesh content.

use crate::device_results::{
    EncodedSensorFrame, SensorChannelSpec, SensorCommandBatch, SensorFramePool,
    SensorFramePoolConfig,
};
use crate::dispatch::BindingsCache;
use crate::readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp, ReadbackStats};
use crate::scene::{SceneStamp, SensorScene};
use molla_compute::Engine;
use molla_core::{BodyId, Error, Result};
use molla_geometry::{MeshBvhGpu, MeshBvhRaycastBindings, MeshBvhRaycastKernel};
use molla_kernels::{RaycastBindings, RaycastKernel, RaycastMergeBindings, RaycastMergeKernel};
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Model, ShapeGeometry, State};
use molla_storage::{Buffer, BufferUsage};
use std::{collections::HashMap, sync::Arc, time::Duration};

/// Single closest-hit result.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Index in `Model::shape_geometry`; `None` for a miss.
    pub shape_id: Option<u32>,
    /// Owning body, or `BodyId::NONE` for static geometry and misses.
    pub body_id: BodyId,
    /// Distance in meters along the normalized ray; infinity for a miss.
    pub t: Real,
    /// World-space hit point; zero for a miss.
    pub point: Vec3,
    /// World-space normal; +Y for a miss. Mesh normals oppose the incident ray.
    pub normal: Vec3,
}

/// Borrowed device channels. Only `0..ray_count` contains this batch's results.
/// Submit consumers or enqueue readback on the same engine before the next dispatch.
/// Misses have distance -1, shape -1, point zero, normal +Y.
pub struct RayDeviceHits<'a> {
    pub ray_count: usize,
    pub scene_stamp: Option<SceneStamp>,
    pub origins: &'a DeviceBuffer<Vec3>,
    pub directions: &'a DeviceBuffer<Vec3>,
    pub distance: &'a DeviceBuffer<Real>,
    pub shape: &'a DeviceBuffer<i32>,
    pub point: &'a DeviceBuffer<Vec3>,
    pub normal: &'a DeviceBuffer<Vec3>,
}

/// Pending query identity; channels become available through the submitted frame receipt.
pub struct EncodedRayQuery {
    pub frame: EncodedSensorFrame,
    pub ray_count: usize,
    pub scene_stamp: SceneStamp,
}

/// Query-path preparations and submissions, excluding construction and readback.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RayQueryStats {
    pub encoded_queries: u64,
    pub submissions: u64,
    pub primitive_preparations: u64,
    pub mesh_preparations: u64,
    pub merge_preparations: u64,
}

/// Reusable fixed-capacity ray sensor with device-side closest-hit reduction.
/// Local TriMesh BVHs are shared across rigid instances, including body-attached meshes.
pub struct Raycaster {
    kernel: RaycastKernel,
    merge: RaycastMergeKernel,
    capacity: u32,
    static_body_q: Buffer<Transform>,
    ray_origin: DeviceBuffer<Vec3>,
    ray_direction: DeviceBuffer<Vec3>,
    hit_t: DeviceBuffer<Real>,
    hit_shape: DeviceBuffer<i32>,
    hit_point: DeviceBuffer<Vec3>,
    hit_normal: DeviceBuffer<Vec3>,
    mesh_kernel: MeshBvhRaycastKernel,
    mesh_cache: HashMap<u32, CachedMesh>,
    primitive_bindings: Vec<RaycastBindings>,
    direct_primitives: BindingsCache<RaycastBindings>,
    direct_initial: BindingsCache<RaycastMergeBindings>,
    direct_meshes: HashMap<u32, BindingsCache<DirectMeshBindings>>,
    direct_scene: Option<SceneStamp>,
    upload_origins: Vec<[f32; 4]>,
    upload_directions: Vec<[f32; 4]>,
    mesh_bindings: HashMap<u32, CachedQuery>,
    query_stats: RayQueryStats,
    mesh_epoch: u64,
    mesh_builds: u64,
    active_scene: Option<SceneStamp>,
    mesh_hit_t: Buffer<Real>,
    mesh_hit_tri: Buffer<i32>,
    mesh_hit_normal: Buffer<Vec3>,
    readback: Option<ReadbackRing>,
}

struct CachedMesh {
    gpu: MeshBvhGpu,
    source: molla_sim::TriMesh,
    checked_epoch: u64,
}

struct CachedQuery {
    bindings: Vec<MeshBvhRaycastBindings>,
    merge: RaycastMergeBindings,
    checked_epoch: u64,
}

struct DirectMeshBindings {
    query: MeshBvhRaycastBindings,
    merge: RaycastMergeBindings,
}

impl Raycaster {
    /// Allocate reusable input, output and scratch storage for up to `max_rays`.
    pub fn new(engine: &Engine, max_rays: u32) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("raycaster: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("raycaster: engine has no GPU queue".into()))?
            .clone();
        let cap = max_rays.max(1) as usize;
        let limits = device.limits();
        let bytes = cap as u64 * 16;
        if bytes > limits.max_storage_buffer_binding_size as u64
            || bytes > limits.max_buffer_size
            || max_rays.max(1).div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(Error::Solver(
                "raycaster: ray capacity exceeds GPU limits".into(),
            ));
        }
        let mut static_body_q = Buffer::from_vec(vec![Transform::IDENTITY], BufferUsage::Storage);
        static_body_q.ensure_device(&device)?;
        static_body_q.upload(&queue)?;
        let ray_origin =
            DeviceBuffer::from_host(engine, vec![Vec3::ZERO; cap], BufferUsage::StorageReadWrite)?;
        let ray_direction =
            DeviceBuffer::from_host(engine, vec![Vec3::ZERO; cap], BufferUsage::StorageReadWrite)?;
        let hit_t =
            DeviceBuffer::from_host(engine, vec![-1.0; cap], BufferUsage::StorageReadWrite)?;
        let hit_shape =
            DeviceBuffer::from_host(engine, vec![-1; cap], BufferUsage::StorageReadWrite)?;
        let hit_point =
            DeviceBuffer::from_host(engine, vec![Vec3::ZERO; cap], BufferUsage::StorageReadWrite)?;
        let hit_normal =
            DeviceBuffer::from_host(engine, vec![Vec3::Y; cap], BufferUsage::StorageReadWrite)?;
        let mut mesh_hit_t =
            Buffer::from_vec(vec![Real::INFINITY; cap], BufferUsage::StorageReadWrite);
        let mut mesh_hit_tri = Buffer::from_vec(vec![-1_i32; cap], BufferUsage::StorageReadWrite);
        let mut mesh_hit_normal =
            Buffer::from_vec(vec![Vec3::ZERO; cap], BufferUsage::StorageReadWrite);
        mesh_hit_t.ensure_device(&device)?;
        mesh_hit_tri.ensure_device(&device)?;
        mesh_hit_normal.ensure_device(&device)?;
        mesh_hit_t.upload(&queue)?;
        mesh_hit_tri.upload(&queue)?;
        mesh_hit_normal.upload(&queue)?;
        let merge = RaycastMergeKernel::new(
            &device,
            [
                ray_origin.device_buffer()?,
                ray_direction.device_buffer()?,
                mesh_hit_t.device_buffer_current()?,
                mesh_hit_tri.device_buffer_current()?,
                mesh_hit_normal.device_buffer_current()?,
                hit_t.device_buffer()?,
                hit_shape.device_buffer()?,
                hit_point.device_buffer()?,
                hit_normal.device_buffer()?,
            ],
        );
        Ok(Self {
            kernel: RaycastKernel::new(&device),
            merge,
            capacity: max_rays.max(1),
            static_body_q,
            ray_origin,
            ray_direction,
            hit_t,
            hit_shape,
            hit_point,
            hit_normal,
            mesh_kernel: MeshBvhRaycastKernel::new(&device),
            mesh_cache: HashMap::new(),
            primitive_bindings: Vec::with_capacity(2),
            direct_primitives: BindingsCache::with_capacity(128),
            direct_initial: BindingsCache::with_capacity(64),
            direct_meshes: HashMap::new(),
            direct_scene: None,
            upload_origins: vec![[0.0; 4]; cap],
            upload_directions: vec![[0.0; 4]; cap],
            mesh_bindings: HashMap::new(),
            query_stats: RayQueryStats::default(),
            mesh_epoch: 0,
            mesh_builds: 0,
            active_scene: None,
            mesh_hit_t,
            mesh_hit_tri,
            mesh_hit_normal,
            readback: None,
        })
    }

    /// Release legacy mesh BVHs and mesh query bindings. Owned scene geometry is unaffected.
    pub fn clear_mesh_cache(&mut self) {
        self.mesh_cache.clear();
        self.mesh_bindings.clear();
        self.direct_meshes.clear();
    }

    /// Number of unique local meshes cached by legacy model queries.
    pub fn mesh_cache_len(&self) -> usize {
        self.mesh_cache.len()
    }

    /// Counters for the optional synchronous collector. Device-only dispatch leaves these zero.
    pub fn readback_stats(&self) -> ReadbackStats {
        self.readback
            .as_ref()
            .map(ReadbackRing::stats)
            .unwrap_or_default()
    }

    /// Each preparation allocates one uniform and four bind groups; merge uses one group.
    /// Legacy queries retain two variants; direct queries retain 128 per primitive/mesh
    /// and 64 miss initializers. Direct submissions are counted on their command batch.
    pub fn query_stats(&self) -> RayQueryStats {
        self.query_stats
    }

    fn output_specs(&self) -> [SensorChannelSpec; 6] {
        let n = self.capacity as usize;
        [
            SensorChannelSpec::of::<Vec3>(n),
            SensorChannelSpec::of::<Vec3>(n),
            SensorChannelSpec::of::<Real>(n),
            SensorChannelSpec::of::<i32>(n),
            SensorChannelSpec::of::<Vec3>(n),
            SensorChannelSpec::of::<Vec3>(n),
        ]
    }

    /// Channels: origins, unit directions, distance, shape index, point, normal.
    pub fn output_pool(
        &self,
        engine: &Engine,
        config: SensorFramePoolConfig,
    ) -> Result<SensorFramePool> {
        engine.check_device()?;
        if !Arc::ptr_eq(engine.device().expect("checked engine"), &self.hit_t.device) {
            return Err(Error::Gpu("raycaster: foreign engine".into()));
        }
        SensorFramePool::new(engine, config, &self.output_specs())
    }

    /// Encode closest hits directly into leased scene-query slots without collection/submission.
    /// Host rays are checked and normalized before upload; all six channels use `0..ray_count`.
    /// Scene/producer/state borrows last until submission/drop. Nonempty body states need canonical device ownership.
    /// `None` means backpressure without input uploads, binding preparation or query commands.
    ///
    /// ```compile_fail
    /// use molla_sensors::{Raycaster, SensorScene, SensorCommandBatch, SensorFramePool};
    /// use molla_compute::Engine;
    /// use molla_sim::{Model, State};
    /// fn replace_pending(sensor: &mut Raycaster, engine: &Engine, scene: &mut SensorScene, model: &Model, state: &State, pool: &mut SensorFramePool) {
    ///     let mut batch = SensorCommandBatch::new(engine).unwrap();
    ///     sensor.encode_scene(&mut batch, engine, scene, state, &[], &[], 10.0, pool, Default::default()).unwrap();
    ///     scene.replace_model(engine, model).unwrap();
    ///     batch.submit().unwrap();
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn encode_scene<'a>(
        &'a mut self,
        batch: &mut SensorCommandBatch<'a>,
        engine: &Engine,
        scene: &'a SensorScene,
        state: &'a State,
        origins: &[Vec3],
        directions: &[Vec3],
        max_distance: Real,
        pool: &mut SensorFramePool,
        stamp: ReadbackStamp,
    ) -> Result<Option<EncodedRayQuery>> {
        batch.check_engine(engine)?;
        scene.check_engine(engine)?;
        let device = engine.device().expect("checked engine");
        if !Arc::ptr_eq(device, &self.hit_t.device) {
            return Err(Error::Gpu("raycaster: foreign engine".into()));
        }
        if origins.len() != directions.len() || origins.len() > self.capacity as usize {
            return Err(Error::Solver(
                "raycaster: ray lengths differ or exceed capacity".into(),
            ));
        }
        if !(max_distance as f32).is_finite() || max_distance as f32 <= 0.0 {
            return Err(Error::Solver(
                "raycaster: max_distance must be positive and finite in f32".into(),
            ));
        }
        for (i, (origin, direction)) in origins.iter().zip(directions).enumerate() {
            if !origin.as_vec3().is_finite()
                || !direction.is_finite()
                || direction.abs().max_element() == 0.0
            {
                return Err(Error::Solver(format!(
                    "raycaster: invalid origin or direction at ray {i}"
                )));
            }
        }
        let n = origins.len();
        if n != 0 && scene.model.body_count != 0 {
            if state.body_q.len() != scene.model.body_count {
                return Err(Error::Solver(
                    "raycaster: state/model body counts differ".into(),
                ));
            }
            state.body_q.device_buffer_for(device)?;
        }
        let specs = self.output_specs();
        let ranges = std::array::from_fn::<_, 6, _>(|_| 0..n);
        let frame = batch.encode_output_initialized(
            pool,
            &specs,
            &ranges,
            stamp,
            !3u64,
            |device, queue, encoder, outputs| {
                if n == 0 {
                    return Ok(0);
                }
                for (i, (origin, direction)) in origins.iter().zip(directions).enumerate() {
                    let direction = (*direction / direction.abs().max_element()).normalize();
                    self.upload_origins[i] =
                        [origin.x as f32, origin.y as f32, origin.z as f32, 0.0];
                    self.upload_directions[i] = [
                        direction.x as f32,
                        direction.y as f32,
                        direction.z as f32,
                        0.0,
                    ];
                }
                queue.write_buffer(
                    &outputs[0],
                    0,
                    bytemuck::cast_slice(&self.upload_origins[..n]),
                );
                queue.write_buffer(
                    &outputs[1],
                    0,
                    bytemuck::cast_slice(&self.upload_directions[..n]),
                );
                self.encode_owned_scene(
                    device,
                    queue,
                    encoder,
                    scene,
                    state,
                    outputs,
                    n as u32,
                    max_distance as f32,
                )
            },
        )?;
        let Some(frame) = frame else {
            return Ok(None);
        };
        batch.record_input_upload(n as u64 * 32);
        Ok(Some(EncodedRayQuery {
            frame,
            ray_count: n,
            scene_stamp: scene.stamp(),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_owned_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &SensorScene,
        state: &State,
        outputs: &[wgpu::Buffer],
        count: u32,
        max_distance: f32,
    ) -> Result<u64> {
        if self.direct_scene != Some(scene.stamp()) {
            self.direct_primitives.clear();
            self.direct_meshes.clear();
            self.direct_scene = Some(scene.stamp());
        }
        let model = &scene.model;
        let scratch = [
            self.mesh_hit_t.device_buffer_current()?,
            self.mesh_hit_tri.device_buffer_current()?,
            self.mesh_hit_normal.device_buffer_current()?,
        ];
        let merge_buffers = [
            &outputs[0],
            &outputs[1],
            scratch[0],
            scratch[1],
            scratch[2],
            &outputs[2],
            &outputs[3],
            &outputs[4],
            &outputs[5],
        ];
        let has_primitives = model
            .shape_geom_kind
            .host()?
            .iter()
            .any(|&kind| kind != 0xff);
        if has_primitives {
            let poses = if model.body_count == 0 {
                self.static_body_q.device_buffer_current()?
            } else {
                state.body_q.device_buffer_current()?
            };
            let buffers = [
                model.shape_geom_kind.device_buffer_current()?,
                model.shape_geom_data.device_buffer_current()?,
                model.shape_body.device_buffer_current()?,
                model.shape_transform.device_buffer_current()?,
                poses,
                &outputs[0],
                &outputs[1],
                &outputs[2],
                &outputs[3],
                &outputs[4],
                &outputs[5],
            ];
            let (bindings, prepared) = self.direct_primitives.get_or_prepare(
                |binding| binding.matches(buffers),
                || self.kernel.prepare(device, buffers),
            );
            self.query_stats.primitive_preparations += u64::from(prepared);
            self.kernel.encode_prepared(
                queue,
                encoder,
                bindings,
                count,
                model.shape_count as u32,
                max_distance,
            );
        } else {
            let (bindings, prepared) = self.direct_initial.get_or_prepare(
                |binding| binding.matches(merge_buffers),
                || self.merge.prepare_with_buffers(device, merge_buffers),
            );
            self.query_stats.merge_preparations += u64::from(prepared);
            self.merge
                .encode_prepared(queue, encoder, bindings, count, None, max_distance);
        }
        let mut dispatches = 1;
        let bodies = model.shape_body.host()?;
        for (shape, geometry) in model.shape_geometry.iter().enumerate() {
            let ShapeGeometry::TriMesh { mesh_id } = *geometry else {
                continue;
            };
            let mesh = scene.mesh(mesh_id)?;
            if mesh.node_count == 0 {
                continue;
            }
            let body = bodies[shape];
            let poses = if body < 0 {
                self.static_body_q.device_buffer_current()?
            } else {
                state.body_q.device_buffer_current()?
            };
            let buffers = [
                &mesh.nodes,
                &mesh.tri_order,
                &mesh.indices,
                &mesh.vertices,
                &outputs[0],
                &outputs[1],
                scratch[0],
                scratch[1],
                scratch[2],
                poses,
                model.shape_transform.device_buffer_current()?,
            ];
            let cache = self
                .direct_meshes
                .entry(shape as u32)
                .or_insert_with(|| BindingsCache::with_capacity(128));
            let (bindings, prepared) = cache.get_or_prepare(
                |binding| binding.query.matches(buffers) && binding.merge.matches(merge_buffers),
                || DirectMeshBindings {
                    query: self.mesh_kernel.prepare(device, buffers),
                    merge: self.merge.prepare_with_buffers(device, merge_buffers),
                },
            );
            self.query_stats.mesh_preparations += u64::from(prepared);
            self.query_stats.merge_preparations += u64::from(prepared);
            self.mesh_kernel.encode_prepared(
                queue,
                encoder,
                &bindings.query,
                count,
                mesh.node_count,
                max_distance,
                shape as u32,
                body,
            );
            self.merge.encode_prepared(
                queue,
                encoder,
                &bindings.merge,
                count,
                Some(shape as u32),
                max_distance,
            );
            dispatches += 2;
        }
        self.mesh_hit_t.mark_device_dirty();
        self.mesh_hit_tri.mark_device_dirty();
        self.mesh_hit_normal.mark_device_dirty();
        self.query_stats.encoded_queries += 1;
        Ok(dispatches)
    }

    /// Submit a query without host collection or GPU waits.
    /// Directions must be finite/nonzero; origins and range must remain finite in f32.
    /// Equal-distance ties retain analytical hits first, then the earliest mesh.
    /// Inputs must be uploaded on this engine; raw mesh edits invalidate cached geometry.
    pub fn dispatch(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        origins: &[Vec3],
        directions: &[Vec3],
        max_distance: Real,
    ) -> Result<RayDeviceHits<'_>> {
        engine.check_device()?;
        self.dispatch_for_model(
            engine,
            model,
            state,
            origins,
            directions,
            max_distance,
            None,
        )
    }

    pub fn dispatch_scene(
        &mut self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
        origins: &[Vec3],
        directions: &[Vec3],
        max_distance: Real,
    ) -> Result<RayDeviceHits<'_>> {
        scene.check_engine(engine)?;
        self.dispatch_for_model(
            engine,
            &scene.model,
            state,
            origins,
            directions,
            max_distance,
            Some(scene),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_for_model(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        origins: &[Vec3],
        directions: &[Vec3],
        max_distance: Real,
        scene: Option<&SensorScene>,
    ) -> Result<RayDeviceHits<'_>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("raycaster: engine has no GPU device".into()))?
            .clone();
        if !Arc::ptr_eq(&device, &self.hit_t.device) {
            return Err(Error::Gpu(
                "raycaster: engine differs from sensor device".into(),
            ));
        }
        if origins.len() != directions.len() || origins.len() > self.capacity as usize {
            return Err(Error::Solver(
                "raycaster: ray lengths differ or exceed capacity".into(),
            ));
        }
        if !(max_distance as f32).is_finite() || max_distance as f32 <= 0.0 {
            return Err(Error::Solver(
                "raycaster: max_distance must be positive and finite in f32".into(),
            ));
        }
        for (i, (origin, direction)) in origins.iter().zip(directions).enumerate() {
            if !origin.as_vec3().is_finite()
                || !direction.is_finite()
                || direction.abs().max_element() == 0.0
            {
                return Err(Error::Solver(format!(
                    "raycaster: invalid origin or direction at ray {i}"
                )));
            }
        }
        let n = origins.len();
        if n != 0 {
            self.ray_origin.host_mut()?[..n].copy_from_slice(origins);
            for (out, direction) in self.ray_direction.host_mut()?.iter_mut().zip(directions) {
                *out = (*direction / direction.abs().max_element()).normalize();
            }
            self.ray_origin.upload()?;
            self.ray_direction.upload()?;
        }
        self.query_uploaded(engine, model, state, n, max_distance, scene)
    }

    pub(crate) fn device_hits(&self, n: usize) -> RayDeviceHits<'_> {
        RayDeviceHits {
            ray_count: n,
            scene_stamp: self.active_scene,
            origins: &self.ray_origin,
            directions: &self.ray_direction,
            distance: &self.hit_t,
            shape: &self.hit_shape,
            point: &self.hit_point,
            normal: &self.hit_normal,
        }
    }

    pub fn mesh_build_count(&self) -> u64 {
        self.mesh_builds
    }

    pub(crate) fn set_scene_stamp(&mut self, stamp: Option<SceneStamp>) {
        self.active_scene = stamp;
    }

    pub(crate) fn encode_generated(
        &mut self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        model: &Model,
        state: &State,
        n: usize,
        max_distance: Real,
        scene: Option<&SensorScene>,
    ) -> Result<u64> {
        self.ray_origin.mark_device_dirty();
        self.ray_direction.mark_device_dirty();
        self.encode_uploaded(engine, encoder, model, state, n, max_distance, scene)
    }

    fn query_uploaded(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        n: usize,
        max_distance: Real,
        scene: Option<&SensorScene>,
    ) -> Result<RayDeviceHits<'_>> {
        if n == 0 {
            self.active_scene = scene.map(SensorScene::stamp);
            return Ok(self.device_hits(0));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("raycaster requires GPU".into()))?;
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("raycaster requires queue".into()))?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::raycaster query"),
        });
        self.encode_uploaded(engine, &mut encoder, model, state, n, max_distance, scene)?;
        queue.submit([encoder.finish()]);
        self.query_stats.submissions += 1;
        Ok(self.device_hits(n))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_uploaded(
        &mut self,
        engine: &Engine,
        encoder: &mut wgpu::CommandEncoder,
        model: &Model,
        state: &State,
        n: usize,
        max_distance: Real,
        scene: Option<&SensorScene>,
    ) -> Result<u64> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("raycaster requires GPU".into()))?;
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("raycaster requires queue".into()))?;
        if !Arc::ptr_eq(device, &self.hit_t.device) || n > self.capacity as usize {
            return Err(Error::Gpu("raycaster: invalid device or capacity".into()));
        }
        self.active_scene = scene.map(SensorScene::stamp);
        let mut dispatches = 0;
        if n != 0 {
            let has_analytical_shape = model.shape_geom_kind.host()?.iter().any(|&k| k != 0xff);
            if has_analytical_shape {
                let body_q = if model.body_count == 0 {
                    self.static_body_q.device_buffer_current()?
                } else {
                    if state.body_q.len() != model.body_count {
                        return Err(Error::Solver(
                            "raycaster: state/model body counts differ".into(),
                        ));
                    }
                    state.body_q.device_buffer_current()?
                };
                let buffers = [
                    model.shape_geom_kind.device_buffer_current()?,
                    model.shape_geom_data.device_buffer_current()?,
                    model.shape_body.device_buffer_current()?,
                    model.shape_transform.device_buffer_current()?,
                    body_q,
                    self.ray_origin.device_buffer()?,
                    self.ray_direction.device_buffer()?,
                    self.hit_t.device_buffer()?,
                    self.hit_shape.device_buffer()?,
                    self.hit_point.device_buffer()?,
                    self.hit_normal.device_buffer()?,
                ];
                let index = match self
                    .primitive_bindings
                    .iter()
                    .position(|b| b.matches(buffers))
                {
                    Some(index) => index,
                    None => {
                        if self.primitive_bindings.len() == 2 {
                            self.primitive_bindings.remove(0);
                        }
                        self.primitive_bindings
                            .push(self.kernel.prepare(device, buffers));
                        self.query_stats.primitive_preparations += 1;
                        self.primitive_bindings.len() - 1
                    }
                };
                self.kernel.encode_prepared(
                    queue,
                    encoder,
                    &self.primitive_bindings[index],
                    n as u32,
                    model.shape_count as u32,
                    max_distance as f32,
                );
            } else {
                self.merge
                    .encode_initial(queue, encoder, n as u32, max_distance as f32);
            }
            self.hit_t.mark_device_dirty();
            self.hit_shape.mark_device_dirty();
            self.hit_point.mark_device_dirty();
            self.hit_normal.mark_device_dirty();
            dispatches = 1 + self.run_trimesh_pass(
                &device,
                &queue,
                encoder,
                model,
                state,
                scene,
                n as u32,
                max_distance as f32,
            )?;
            self.query_stats.encoded_queries += 1;
        }
        Ok(dispatches)
    }

    /// Cast and collect all four active hit channels in one grouped readback.
    /// This synchronous convenience call may wait up to 30 seconds for collection.
    pub fn cast(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        origins: &[Vec3],
        directions: &[Vec3],
        max_distance: Real,
    ) -> Result<Vec<RayHit>> {
        self.dispatch(engine, model, state, origins, directions, max_distance)?;
        self.collect_hits(engine, model, origins.len())
    }

    pub fn cast_scene(
        &mut self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
        origins: &[Vec3],
        directions: &[Vec3],
        max_distance: Real,
    ) -> Result<Vec<RayHit>> {
        self.dispatch_scene(engine, scene, state, origins, directions, max_distance)?;
        self.collect_hits(engine, &scene.model, origins.len())
    }

    fn collect_hits(&mut self, engine: &Engine, model: &Model, n: usize) -> Result<Vec<RayHit>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        if self.readback.is_none() {
            let bytes = self.capacity as u64 * 40;
            self.readback = Some(ReadbackRing::new(
                engine,
                ReadbackConfig {
                    slots: 1,
                    bytes_per_slot: bytes,
                    max_total_bytes: bytes,
                },
            )?);
        }
        let ring = self.readback.as_mut().expect("initialized readback");
        let sources = [
            self.hit_t.readback_source(0..n)?,
            self.hit_shape.readback_source(0..n)?,
            self.hit_point.readback_source(0..n)?,
            self.hit_normal.readback_source(0..n)?,
        ];
        let ticket = ring
            .try_enqueue(&sources, ReadbackStamp::default())?
            .ok_or_else(|| Error::Storage("raycaster: synchronous readback slot busy".into()))?;
        let frame = match ring.wait(&ticket, Duration::from_secs(30)) {
            Ok(frame) => frame,
            Err(error) => {
                ring.reset();
                return Err(error);
            }
        };
        let shape_body = model.shape_body.host()?;
        let hits = frame
            .channel_iter::<Real>(0)?
            .zip(frame.channel_iter::<i32>(1)?)
            .zip(frame.channel_iter::<Vec3>(2)?)
            .zip(frame.channel_iter::<Vec3>(3)?)
            .map(|(((t, shape), point), normal)| {
                if shape < 0 || t <= 0.0 {
                    RayHit {
                        shape_id: None,
                        body_id: BodyId::NONE,
                        t: Real::INFINITY,
                        point: Vec3::ZERO,
                        normal: Vec3::Y,
                    }
                } else {
                    let body = shape_body.get(shape as usize).copied().unwrap_or(-1);
                    RayHit {
                        shape_id: Some(shape as u32),
                        body_id: if body >= 0 {
                            BodyId::from_index(body as usize)
                        } else {
                            BodyId::NONE
                        },
                        t,
                        point,
                        normal,
                    }
                }
            })
            .collect();
        Ok(hits)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_trimesh_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        model: &Model,
        state: &State,
        scene: Option<&SensorScene>,
        ray_count: u32,
        max_distance: f32,
    ) -> Result<u64> {
        let mut dispatches = 0;
        self.mesh_epoch = self
            .mesh_epoch
            .checked_add(1)
            .ok_or_else(|| Error::Solver("raycaster mesh epoch exhausted".into()))?;
        let shape_body = model.shape_body.host()?;
        for (shape_id, geometry) in model.shape_geometry.iter().enumerate() {
            let ShapeGeometry::TriMesh { mesh_id } = *geometry else {
                continue;
            };
            let body_id = *shape_body
                .get(shape_id)
                .ok_or_else(|| Error::Solver("raycaster: invalid shape body array".into()))?;
            if body_id < -1 || body_id >= 0 && body_id as usize >= model.body_count {
                return Err(Error::Solver("raycaster: invalid mesh body index".into()));
            }
            let body_poses = if body_id < 0 {
                self.static_body_q.device_buffer_current()?
            } else {
                if state.body_q.len() != model.body_count {
                    return Err(Error::Solver(
                        "raycaster: state/model body counts differ".into(),
                    ));
                }
                state.body_q.device_buffer_current()?
            };
            let shape_poses = model.shape_transform.device_buffer_current()?;
            let mesh = if let Some(scene) = scene {
                scene.mesh(mesh_id)?
            } else {
                let source = model
                    .meshes
                    .get(mesh_id as usize)
                    .ok_or_else(|| Error::Solver("raycaster: invalid mesh id".into()))?;
                let entry = self.mesh_cache.get(&mesh_id);
                let rebuild = entry.is_none_or(|entry| {
                    entry.checked_epoch != self.mesh_epoch
                        && (entry.source.vertices != source.vertices
                            || entry.source.indices != source.indices)
                });
                if rebuild {
                    let gpu =
                        MeshBvhGpu::from_mesh(device, queue, &source.vertices, &source.indices)?;
                    self.mesh_cache.insert(
                        mesh_id,
                        CachedMesh {
                            gpu,
                            source: source.clone(),
                            checked_epoch: self.mesh_epoch,
                        },
                    );
                    self.mesh_builds += 1;
                }
                let entry = self.mesh_cache.get_mut(&mesh_id).expect("initialized mesh");
                entry.checked_epoch = self.mesh_epoch;
                &entry.gpu
            };
            if mesh.node_count == 0 {
                continue;
            }
            let key = u32::try_from(shape_id)
                .ok()
                .filter(|&id| id <= i32::MAX as u32)
                .ok_or_else(|| Error::Solver("raycaster: shape index exceeds i32".into()))?;
            let buffers = [
                &mesh.nodes,
                &mesh.tri_order,
                &mesh.indices,
                &mesh.vertices,
                self.ray_origin.device_buffer()?,
                self.ray_direction.device_buffer()?,
                self.mesh_hit_t.device_buffer_current()?,
                self.mesh_hit_tri.device_buffer_current()?,
                self.mesh_hit_normal.device_buffer_current()?,
                body_poses,
                shape_poses,
            ];
            let entry = self.mesh_bindings.entry(key).or_insert_with(|| {
                self.query_stats.merge_preparations += 1;
                CachedQuery {
                    bindings: Vec::with_capacity(2),
                    merge: self.merge.prepare(device),
                    checked_epoch: 0,
                }
            });
            entry.checked_epoch = self.mesh_epoch;
            let index = match entry.bindings.iter().position(|b| b.matches(buffers)) {
                Some(index) => index,
                None => {
                    if entry.bindings.len() == 2 {
                        entry.bindings.remove(0);
                    }
                    entry
                        .bindings
                        .push(self.mesh_kernel.prepare(device, buffers));
                    self.query_stats.mesh_preparations += 1;
                    entry.bindings.len() - 1
                }
            };
            self.mesh_kernel.encode_prepared(
                queue,
                encoder,
                &entry.bindings[index],
                ray_count,
                mesh.node_count,
                max_distance,
                key,
                body_id,
            );
            self.mesh_hit_t.mark_device_dirty();
            self.mesh_hit_tri.mark_device_dirty();
            self.mesh_hit_normal.mark_device_dirty();
            self.merge.encode_prepared(
                queue,
                encoder,
                &entry.merge,
                ray_count,
                Some(key),
                max_distance,
            );
            dispatches += 2;
        }
        self.mesh_bindings
            .retain(|_, entry| entry.checked_epoch == self.mesh_epoch);
        if scene.is_none() {
            self.mesh_cache
                .retain(|_, entry| entry.checked_epoch == self.mesh_epoch);
        }
        Ok(dispatches)
    }
}
