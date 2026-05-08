//! Public ray-cast sensor — orchestrates a [`RaycastKernel`] dispatch.
//!
//! GPU-first by construction: the heavy lifting (shape iteration,
//! ray-vs-primitive math, normal computation) all runs in
//! `shaders/raycast.wgsl`. This module only owns the per-batch
//! ray/hit buffers and the dispatch wrapper.
//!
//! Triangle / TriMesh shapes carry the `0xff` "skip" sentinel in
//! `Model::shape_geom_kind` — they're skipped on the GPU. The host
//! ray-vs-mesh fallback for those is on hold until the mesh BVH lands
//! in stage 11.
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

use crate::compute::Engine;
use crate::core::{BodyId, Error, Result};
use crate::kernels::RaycastKernel;
use crate::math::{Real, Vec3};
use crate::sim::{Model, State};
use crate::storage::{Buffer, BufferUsage};

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

/// GPU ray-cast batched sensor. Owns the kernel and per-batch ray /
/// hit storage; reuse one instance across many `cast` calls to avoid
/// reallocating GPU buffers each frame.
pub struct Raycaster {
    kernel: RaycastKernel,
    capacity: u32,
    // Ray inputs.
    ray_origin: Buffer<Vec3>,
    ray_direction: Buffer<Vec3>,
    // Outputs.
    hit_t: Buffer<Real>,
    hit_shape: Buffer<i32>,
    hit_point: Buffer<Vec3>,
    hit_normal: Buffer<Vec3>,
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
        let mut ray_direction =
            Buffer::from_vec(vec![Vec3::ZERO; cap], BufferUsage::Storage);
        let mut hit_t = Buffer::from_vec(vec![-1.0_f64; cap], BufferUsage::StorageReadWrite);
        let mut hit_shape =
            Buffer::from_vec(vec![-1_i32; cap], BufferUsage::StorageReadWrite);
        let mut hit_point =
            Buffer::from_vec(vec![Vec3::ZERO; cap], BufferUsage::StorageReadWrite);
        let mut hit_normal =
            Buffer::from_vec(vec![Vec3::Y; cap], BufferUsage::StorageReadWrite);

        // Allocate device mirrors up-front so dispatch never has to.
        ray_origin.ensure_device(&device)?;
        ray_direction.ensure_device(&device)?;
        hit_t.ensure_device(&device)?;
        hit_shape.ensure_device(&device)?;
        hit_point.ensure_device(&device)?;
        hit_normal.ensure_device(&device)?;
        ray_origin.upload(&queue)?;
        ray_direction.upload(&queue)?;
        hit_t.upload(&queue)?;
        hit_shape.upload(&queue)?;
        hit_point.upload(&queue)?;
        hit_normal.upload(&queue)?;

        Ok(Self {
            kernel: RaycastKernel::new(&device),
            capacity: max_rays.max(1),
            ray_origin,
            ray_direction,
            hit_t,
            hit_shape,
            hit_point,
            hit_normal,
        })
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

        // The buffers are sized to `capacity`; we only dispatch
        // `origins.len()` threads via the explicit `ray_count` param.
        // The kernel uses `uniforms.ray_count` for the bound, and
        // entries past `ray_count` are not touched.
        let body_q_buf = state
            .body_q
            .device_buffer()
            .ok_or_else(|| Error::Gpu("raycaster::cast: state.body_q not on device".into()))?;

        self.kernel.dispatch_with_body_q(
            &device,
            &queue,
            model,
            body_q_buf,
            origins.len() as u32,
            &self.ray_origin,
            &self.ray_direction,
            &mut self.hit_t,
            &mut self.hit_shape,
            &mut self.hit_point,
            &mut self.hit_normal,
            max_distance as f32,
        )?;

        // Pull the results back to host.
        self.hit_t.download(&device, &queue)?;
        self.hit_shape.download(&device, &queue)?;
        self.hit_point.download(&device, &queue)?;
        self.hit_normal.download(&device, &queue)?;

        let hit_t = self.hit_t.host()?;
        let hit_shape = self.hit_shape.host()?;
        let hit_point = self.hit_point.host()?;
        let hit_normal = self.hit_normal.host()?;
        let shape_body = model.shape_body.host()?;

        let mut out = Vec::with_capacity(origins.len());
        for i in 0..origins.len() {
            let t = hit_t[i];
            let s = hit_shape[i];
            if s < 0 || t <= 0.0 {
                out.push(RayHit {
                    shape_id: None,
                    body_id: BodyId::NONE,
                    t: Real::INFINITY,
                    point: Vec3::ZERO,
                    normal: Vec3::Y,
                });
            } else {
                let body = shape_body
                    .get(s as usize)
                    .copied()
                    .unwrap_or(-1);
                out.push(RayHit {
                    shape_id: Some(s as u32),
                    body_id: if body >= 0 {
                        BodyId::from_index(body as usize)
                    } else {
                        BodyId::NONE
                    },
                    t,
                    point: hit_point[i],
                    normal: hit_normal[i],
                });
            }
        }

        Ok(out)
    }
}
