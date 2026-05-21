//! Public ray-cast sensor — orchestrates a [`RaycastKernel`] dispatch.
//!
//! GPU-first by construction: the heavy lifting (shape iteration,
//! ray-vs-primitive math, normal computation) all runs in
//! `shaders/raycast.wgsl`. This module only owns the per-batch
//! ray/hit buffers and the dispatch wrapper.
//!
//! **H1 — TriMesh support.** Triangle / TriMesh shapes still carry the
//! `0xff` "skip" sentinel in `Model::shape_geom_kind` (the analytical
//! `raycast.wgsl` kernel skips them), but the [`Raycaster`] now also
//! runs a second GPU pass per static TriMesh shape against
//! [`molla_geometry::MeshBvhGpu`] and merges the closest hit into the
//! analytical result. Static TriMesh shapes (`shape_body == -1`) are
//! supported; dynamic TriMesh shapes (attached to a moving body) are
//! still skipped pending a transform-aware mesh-BVH kernel (a later
//! H1 follow-up). The mesh-BVH is built in **world space** at first
//! use and cached per-shape, so each `cast()` call only pays the
//! per-shape dispatch + readback.
//!
//! # Example
//!
//! ```ignore
//! use molla::compute::Engine;
//! use molla::sensors::Raycaster;
//! use molla::math::Vec3;
//!
//! let engine = Engine::new_gpu()?;
//! let mut sensor = Raycaster::new(&engine, /* max_rays = */ 64);
//! let origins = vec![Vec3::new(0.0, 5.0, 0.0); 1];
//! let dirs    = vec![Vec3::new(0.0, -1.0, 0.0); 1];
//! let hits = sensor.cast(&engine, &model, &state, &origins, &dirs, 100.0)?;
//! println!("{:?}", hits[0]); // RayHit { t: 4.5, shape_id: ..., ... }
//! ```

use molla_compute::Engine;
use molla_core::{BodyId, Error, Result};
use molla_geometry::{MeshBvh, MeshBvhGpu, MeshBvhRaycastKernel};
use molla_kernels::RaycastKernel;
use molla_math::{Real, Vec3};
use molla_sim::{Model, ShapeGeometry, State};
use molla_storage::{Buffer, BufferUsage};
use std::collections::HashMap;

/// Single closest-hit result.
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// Index of the hit shape in `Model::shape_geometry`. `None` if
    /// the ray missed every shape within `max_distance`.
    pub shape_id: Option<u32>,
    /// Body that owns the hit shape, or `BodyId::NONE` for static
    /// world geometry. Meaningless when `shape_id.is_none()`.
    pub body_id: BodyId,
    /// Distance along the ray. `f64::INFINITY` if no hit.
    pub t: Real,
    /// World-space hit point (zero when no hit).
    pub point: Vec3,
    /// World-space outward surface normal at the hit (`+Y` when no hit).
    pub normal: Vec3,
}

/// GPU ray-cast batched sensor. Owns the analytical kernel, the mesh
/// BVH kernel, and per-batch ray / hit storage; reuse one instance
/// across many `cast` calls to avoid reallocating GPU buffers each
/// frame. The mesh-BVH cache is keyed per shape so two TriMesh shapes
/// that share a `mesh_id` but live at different poses each get their
/// own world-space BVH.
pub struct Raycaster {
    kernel: RaycastKernel,
    capacity: u32,
    // Ray inputs (shared by both the analytical and the mesh-BVH passes).
    ray_origin: Buffer<Vec3>,
    ray_direction: Buffer<Vec3>,
    // Analytical-pass outputs.
    hit_t: Buffer<Real>,
    hit_shape: Buffer<i32>,
    hit_point: Buffer<Vec3>,
    hit_normal: Buffer<Vec3>,
    // H1 — mesh-BVH pass state.
    mesh_kernel: MeshBvhRaycastKernel,
    /// Per-shape cached world-space mesh BVH. Keyed by `shape_id` so
    /// two shapes that share `mesh_id` but have different poses each
    /// get their own entry. Built lazily on first cast.
    mesh_cache: HashMap<u32, MeshBvhGpu>,
    /// Scratch per-ray mesh-BVH hit buffers, sized to `capacity`.
    mesh_hit_t: Buffer<Real>,
    mesh_hit_tri: Buffer<i32>,
    mesh_hit_normal: Buffer<Vec3>,
}

impl Raycaster {
    /// Build a sensor with capacity for up to `max_rays` rays per
    /// batch. Cheap to construct; reusable.
    pub fn new(engine: &Engine, max_rays: u32) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("raycaster: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("raycaster: engine has no GPU queue".into()))?
            .clone();

        let cap = max_rays.max(1) as usize;
        let mut ray_origin = Buffer::from_vec(vec![Vec3::ZERO; cap], BufferUsage::Storage);
        let mut ray_direction = Buffer::from_vec(vec![Vec3::ZERO; cap], BufferUsage::Storage);
        let mut hit_t = Buffer::from_vec(vec![-1.0_f64; cap], BufferUsage::StorageReadWrite);
        let mut hit_shape = Buffer::from_vec(vec![-1_i32; cap], BufferUsage::StorageReadWrite);
        let mut hit_point = Buffer::from_vec(vec![Vec3::ZERO; cap], BufferUsage::StorageReadWrite);
        let mut hit_normal = Buffer::from_vec(vec![Vec3::Y; cap], BufferUsage::StorageReadWrite);

        // H1 — mesh-pass scratch hit buffers, sized to capacity so the
        // mesh kernel can be dispatched against the same ray batch.
        let mut mesh_hit_t =
            Buffer::from_vec(vec![Real::INFINITY; cap], BufferUsage::StorageReadWrite);
        let mut mesh_hit_tri = Buffer::from_vec(vec![-1_i32; cap], BufferUsage::StorageReadWrite);
        let mut mesh_hit_normal =
            Buffer::from_vec(vec![Vec3::ZERO; cap], BufferUsage::StorageReadWrite);

        // Allocate device mirrors up-front so dispatch never has to.
        ray_origin.ensure_device(&device)?;
        ray_direction.ensure_device(&device)?;
        hit_t.ensure_device(&device)?;
        hit_shape.ensure_device(&device)?;
        hit_point.ensure_device(&device)?;
        hit_normal.ensure_device(&device)?;
        mesh_hit_t.ensure_device(&device)?;
        mesh_hit_tri.ensure_device(&device)?;
        mesh_hit_normal.ensure_device(&device)?;
        ray_origin.upload(&queue)?;
        ray_direction.upload(&queue)?;
        hit_t.upload(&queue)?;
        hit_shape.upload(&queue)?;
        hit_point.upload(&queue)?;
        hit_normal.upload(&queue)?;
        mesh_hit_t.upload(&queue)?;
        mesh_hit_tri.upload(&queue)?;
        mesh_hit_normal.upload(&queue)?;

        Ok(Self {
            kernel: RaycastKernel::new(&device),
            mesh_kernel: MeshBvhRaycastKernel::new(&device),
            mesh_cache: HashMap::new(),
            capacity: max_rays.max(1),
            ray_origin,
            ray_direction,
            hit_t,
            hit_shape,
            hit_point,
            hit_normal,
            mesh_hit_t,
            mesh_hit_tri,
            mesh_hit_normal,
        })
    }

    /// Drop every cached per-shape mesh BVH. Call this when the
    /// model's TriMesh shapes change (vertices edited, mesh swapped,
    /// or a new `Model` reuses old shape indices); a subsequent
    /// `cast` then rebuilds the world-space BVH lazily. Cheap when
    /// the cache is empty.
    pub fn clear_mesh_cache(&mut self) {
        self.mesh_cache.clear();
    }

    /// Number of cached per-shape mesh BVHs. Useful for tests and
    /// memory-budget asserts.
    pub fn mesh_cache_len(&self) -> usize {
        self.mesh_cache.len()
    }

    /// Cast `origins.len()` rays against `model` (using the body
    /// poses in `state`) and return the closest hits within
    /// `max_distance`. The model + state must already have their
    /// device mirrors uploaded.
    ///
    /// Slice lengths must match and not exceed the sensor's
    /// `max_rays` capacity.
    pub fn cast(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        origins: &[Vec3],
        directions: &[Vec3],
        max_distance: Real,
    ) -> Result<Vec<RayHit>> {
        if origins.len() != directions.len() {
            return Err(Error::Solver(format!(
                "raycaster::cast: origins.len() {} != directions.len() {}",
                origins.len(),
                directions.len()
            )));
        }
        if origins.len() as u32 > self.capacity {
            return Err(Error::Solver(format!(
                "raycaster::cast: {} rays exceeds capacity {}",
                origins.len(),
                self.capacity
            )));
        }
        if origins.is_empty() {
            return Ok(Vec::new());
        }

        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("raycaster::cast: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("raycaster::cast: engine has no GPU queue".into()))?
            .clone();

        // Fill the front of the input buffers with the caller's rays;
        // any unused slots stay at the previous frame's values, which
        // is fine because we only consult the first `n` outputs.
        {
            let host = self.ray_origin.host_mut()?;
            host[..origins.len()].copy_from_slice(origins);
            self.ray_origin.upload(&queue)?;
        }
        {
            let host = self.ray_direction.host_mut()?;
            host[..directions.len()].copy_from_slice(directions);
            self.ray_direction.upload(&queue)?;
        }

        // ── Pass 1: analytical primitives (sphere / box / capsule /
        // cylinder / cone / plane / ellipsoid). The kernel skips
        // TriMesh shapes via the `0xff` sentinel. If the model carries
        // no analytical shapes at all — e.g. a TriMesh-only ground
        // scene — skip the analytical dispatch and seed the merge
        // buffers with miss defaults so pass 2 still runs.
        let n = origins.len();
        let has_analytical_shape = {
            let kinds = model.shape_geom_kind.host()?;
            kinds.iter().any(|&k| k != 0xff)
        };
        let mut best_t: Vec<Real>;
        let mut best_shape: Vec<i32>;
        let mut best_point: Vec<Vec3>;
        let mut best_normal: Vec<Vec3>;
        if has_analytical_shape {
            let body_q_buf = state
                .body_q
                .device_buffer()
                .ok_or_else(|| Error::Gpu("raycaster::cast: state.body_q not on device".into()))?;
            self.kernel.dispatch_with_body_q(
                &device,
                &queue,
                model,
                body_q_buf,
                n as u32,
                &self.ray_origin,
                &self.ray_direction,
                &mut self.hit_t,
                &mut self.hit_shape,
                &mut self.hit_point,
                &mut self.hit_normal,
                max_distance as f32,
            )?;
            self.hit_t.download(&device, &queue)?;
            self.hit_shape.download(&device, &queue)?;
            self.hit_point.download(&device, &queue)?;
            self.hit_normal.download(&device, &queue)?;
            best_t = self.hit_t.host()?[..n].to_vec();
            best_shape = self.hit_shape.host()?[..n].to_vec();
            best_point = self.hit_point.host()?[..n].to_vec();
            best_normal = self.hit_normal.host()?[..n].to_vec();
        } else {
            best_t = vec![-1.0; n];
            best_shape = vec![-1; n];
            best_point = vec![Vec3::ZERO; n];
            best_normal = vec![Vec3::Y; n];
        }

        // ── Pass 2: one mesh-BVH dispatch per static TriMesh shape,
        // merged into `best_*` host-side. Dynamic TriMesh shapes
        // (`shape_body != -1`) are skipped — see module docstring.
        self.run_trimesh_pass(
            &device,
            &queue,
            model,
            origins,
            directions,
            max_distance,
            &mut best_t,
            &mut best_shape,
            &mut best_point,
            &mut best_normal,
        )?;

        // ── Assemble the public RayHit vec.
        let shape_body = model.shape_body.host()?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let t = best_t[i];
            let s = best_shape[i];
            if s < 0 || t <= 0.0 {
                out.push(RayHit {
                    shape_id: None,
                    body_id: BodyId::NONE,
                    t: Real::INFINITY,
                    point: Vec3::ZERO,
                    normal: Vec3::Y,
                });
            } else {
                let body = shape_body.get(s as usize).copied().unwrap_or(-1);
                out.push(RayHit {
                    shape_id: Some(s as u32),
                    body_id: if body >= 0 {
                        BodyId::from_index(body as usize)
                    } else {
                        BodyId::NONE
                    },
                    t,
                    point: best_point[i],
                    normal: best_normal[i],
                });
            }
        }

        Ok(out)
    }

    /// H1 — second pass over every static TriMesh shape. The pass
    /// builds the world-space mesh BVH once per shape (cached), then
    /// merges the closest hit into the per-ray running best.
    #[allow(clippy::too_many_arguments)]
    fn run_trimesh_pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        origins: &[Vec3],
        directions: &[Vec3],
        max_distance: Real,
        best_t: &mut [Real],
        best_shape: &mut [i32],
        best_point: &mut [Vec3],
        best_normal: &mut [Vec3],
    ) -> Result<()> {
        let n = origins.len();
        if model.shape_geometry.is_empty() || n == 0 {
            return Ok(());
        }
        let shape_body = model.shape_body.host()?;
        let shape_transform = model.shape_transform.host()?;
        // Snapshot once; the inner loop indexes into these.
        let n_shapes = model.shape_geometry.len();

        for shape_id in 0..n_shapes {
            let mesh_id = match model.shape_geometry[shape_id] {
                ShapeGeometry::TriMesh { mesh_id } => mesh_id,
                _ => continue,
            };
            // v1 scope: static TriMesh only (no moving body).
            // shape_body[s] == -1 ↔ world-static shape. Dynamic
            // TriMesh is logged at scene-build time via the existing
            // `0xff` skip; the mesh kernel here just defers it.
            if shape_body.get(shape_id).copied().unwrap_or(-1) != -1 {
                continue;
            }
            let world_xf = shape_transform
                .get(shape_id)
                .copied()
                .ok_or_else(|| Error::Gpu("raycaster: shape_transform out of range".into()))?;

            // Build & upload the world-space BVH the first time this
            // shape is queried. Vertices are pre-transformed so the
            // mesh kernel can be used as-is (it operates in
            // "mesh-local" = world-space here). `Entry::or_insert_with`
            // avoids the contains_key/insert two-lookup pattern, but
            // we want the build to be fallible — match-and-fill is
            // the readable form.
            let key = shape_id as u32;
            if let std::collections::hash_map::Entry::Vacant(slot) = self.mesh_cache.entry(key) {
                let mesh = model.meshes.get(mesh_id as usize).ok_or_else(|| {
                    Error::Gpu(format!(
                        "raycaster: shape {shape_id} has invalid mesh_id {mesh_id}"
                    ))
                })?;
                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    continue;
                }
                let world_vertices: Vec<Vec3> = mesh
                    .vertices
                    .iter()
                    .map(|v| world_xf.transform_point(*v))
                    .collect();
                let bvh = MeshBvh::build(&world_vertices, &mesh.indices, 8);
                let gpu = MeshBvhGpu::upload(device, queue, &bvh, &world_vertices, &mesh.indices)?;
                slot.insert(gpu);
            }
            let mesh_gpu = self
                .mesh_cache
                .get(&key)
                .expect("just inserted or already cached");

            // Dispatch the mesh-BVH raycast against the same ray batch.
            self.mesh_kernel.dispatch(
                device,
                queue,
                mesh_gpu,
                &self.ray_origin,
                &self.ray_direction,
                &mut self.mesh_hit_t,
                &mut self.mesh_hit_tri,
                &mut self.mesh_hit_normal,
                max_distance as f32,
                n as u32,
            )?;
            self.mesh_hit_t.download(device, queue)?;
            self.mesh_hit_tri.download(device, queue)?;
            self.mesh_hit_normal.download(device, queue)?;

            let mht = self.mesh_hit_t.host()?;
            let mtri = self.mesh_hit_tri.host()?;
            let mnrm = self.mesh_hit_normal.host()?;

            for i in 0..n {
                if mtri[i] < 0 {
                    continue; // miss
                }
                let mt = mht[i];
                // The mesh kernel writes `max_distance` as the miss
                // sentinel for `hit_t` (it's also the initial value),
                // so guard both: only count it as a real hit when the
                // triangle index is non-negative AND `t < max_distance`.
                if !mt.is_finite() || mt >= max_distance || mt <= 0.0 {
                    continue;
                }
                let prev_has_hit = best_shape[i] >= 0 && best_t[i] > 0.0;
                if !prev_has_hit || mt < best_t[i] {
                    let n_geom = mnrm[i];
                    let n_len2 = n_geom.x * n_geom.x + n_geom.y * n_geom.y + n_geom.z * n_geom.z;
                    let outward = if n_len2 > 1.0e-24 {
                        let inv = 1.0 / n_len2.sqrt();
                        let n_unit = Vec3::new(n_geom.x * inv, n_geom.y * inv, n_geom.z * inv);
                        // Flip to outward (against the ray direction).
                        let dir = directions[i];
                        if n_unit.x * dir.x + n_unit.y * dir.y + n_unit.z * dir.z > 0.0 {
                            Vec3::new(-n_unit.x, -n_unit.y, -n_unit.z)
                        } else {
                            n_unit
                        }
                    } else {
                        Vec3::Y
                    };
                    best_t[i] = mt;
                    best_shape[i] = shape_id as i32;
                    // The ray `direction` is the raw vector (not
                    // necessarily unit); both kernels return `t` in
                    // direction-vector units, so the hit point is
                    // `origin + t * direction` regardless.
                    let dir = directions[i];
                    best_point[i] = Vec3::new(
                        origins[i].x + dir.x * mt,
                        origins[i].y + dir.y * mt,
                        origins[i].z + dir.z * mt,
                    );
                    best_normal[i] = outward;
                }
            }
        }
        Ok(())
    }
}
