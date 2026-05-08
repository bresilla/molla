//! Public frame-transform sensor — orchestrates a
//! [`FrameTransformKernel`] dispatch.
//!
//! Tracks N (body, local-pose) frames per step, returning each
//! frame's world pose, and (optionally) its pose **relative to a
//! reference frame** (also a body+local-pose pair). Useful for
//! end-effector tracking, chase-cam targets, trajectory recording.
//!
//! GPU-first: the per-frame world / relative pose math runs in
//! `shaders/frame_transform.wgsl` (one thread per target). When a
//! reference frame is configured, the caller passes it as **target
//! index 0** and the kernel computes everyone else's pose relative
//! to it.

use crate::compute::Engine;
use crate::core::{BodyId, Error, Result};
use crate::kernels::FrameTransformKernel;
use crate::math::{Quat, Transform, Vec3};
use crate::sim::{Model, State};
use crate::storage::{Buffer, BufferUsage};

/// One frame to track.
#[derive(Debug, Clone, Copy)]
pub struct FrameTarget {
    pub body: BodyId,
    pub local_pose: Transform,
}

/// Per-frame reading.
#[derive(Debug, Clone, Copy)]
pub struct FrameReading {
    /// World-space pose of the target frame.
    pub world: Transform,
    /// Pose of the target frame **expressed in the reference frame**.
    /// Equal to `world` when the sensor was created without a
    /// reference (`reference = None`).
    pub relative: Transform,
}

/// GPU frame-transform sensor batch.
pub struct FrameTransformSensor {
    kernel: FrameTransformKernel,
    has_reference: bool,
    target_count: u32,

    target_body: Buffer<u32>,
    target_local_pos: Buffer<Vec3>,
    target_local_rot: Buffer<Quat>,

    world_pos: Buffer<Vec3>,
    world_rot: Buffer<Quat>,
    relative_pos: Buffer<Vec3>,
    relative_rot: Buffer<Quat>,
}

impl FrameTransformSensor {
    /// Build the sensor. If `reference` is `Some`, every reading's
    /// `relative` field is expressed in the reference frame; if
    /// `None`, `relative == world`.
    pub fn new(
        engine: &Engine,
        model: &Model,
        reference: Option<FrameTarget>,
        targets: &[FrameTarget],
    ) -> Result<Self> {
        if targets.is_empty() {
            return Err(Error::Solver(
                "FrameTransformSensor::new: empty target list".into(),
            ));
        }
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("frame_transform: engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("frame_transform: engine has no GPU queue".into()))?
            .clone();

        // Combined buffer with reference at index 0 (if any) followed
        // by the user's targets.
        let has_ref = reference.is_some();
        let mut all: Vec<FrameTarget> = Vec::with_capacity(targets.len() + usize::from(has_ref));
        if let Some(r) = reference {
            all.push(r);
        }
        all.extend_from_slice(targets);
        for (i, t) in all.iter().enumerate() {
            if t.body.index() >= model.body_count {
                return Err(Error::Solver(format!(
                    "FrameTransformSensor::new: target {i} body {} out of range (count={})",
                    t.body.index(),
                    model.body_count
                )));
            }
        }

        let n = all.len();
        let body_idx: Vec<u32> = all.iter().map(|t| t.body.index() as u32).collect();
        let local_pos: Vec<Vec3> = all.iter().map(|t| t.local_pose.position).collect();
        let local_rot: Vec<Quat> = all.iter().map(|t| t.local_pose.rotation).collect();

        let mut target_body = Buffer::from_vec(body_idx, BufferUsage::Storage);
        let mut target_local_pos = Buffer::from_vec(local_pos, BufferUsage::Storage);
        let mut target_local_rot = Buffer::from_vec(local_rot, BufferUsage::Storage);
        let mut world_pos = Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);
        let mut world_rot =
            Buffer::from_vec(vec![Quat::IDENTITY; n], BufferUsage::StorageReadWrite);
        let mut relative_pos =
            Buffer::from_vec(vec![Vec3::ZERO; n], BufferUsage::StorageReadWrite);
        let mut relative_rot =
            Buffer::from_vec(vec![Quat::IDENTITY; n], BufferUsage::StorageReadWrite);

        target_body.ensure_device(&device)?;
        target_local_pos.ensure_device(&device)?;
        target_local_rot.ensure_device(&device)?;
        world_pos.ensure_device(&device)?;
        world_rot.ensure_device(&device)?;
        relative_pos.ensure_device(&device)?;
        relative_rot.ensure_device(&device)?;
        target_body.upload(&queue)?;
        target_local_pos.upload(&queue)?;
        target_local_rot.upload(&queue)?;
        world_pos.upload(&queue)?;
        world_rot.upload(&queue)?;
        relative_pos.upload(&queue)?;
        relative_rot.upload(&queue)?;

        Ok(Self {
            kernel: FrameTransformKernel::new(&device),
            has_reference: has_ref,
            target_count: n as u32,
            target_body,
            target_local_pos,
            target_local_rot,
            world_pos,
            world_rot,
            relative_pos,
            relative_rot,
        })
    }

    /// Run the kernel against `state` and return one reading per
    /// caller-supplied target (the reference, if any, is **not**
    /// returned — it's an implementation slot).
    pub fn update(&mut self, engine: &Engine, state: &State) -> Result<Vec<FrameReading>> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("frame_transform: no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("frame_transform: no GPU queue".into()))?
            .clone();

        self.kernel.dispatch(
            &device,
            &queue,
            state,
            &self.target_body,
            &self.target_local_pos,
            &self.target_local_rot,
            &mut self.world_pos,
            &mut self.world_rot,
            &mut self.relative_pos,
            &mut self.relative_rot,
            self.target_count,
            self.has_reference,
        )?;
        self.world_pos.download(&device, &queue)?;
        self.world_rot.download(&device, &queue)?;
        self.relative_pos.download(&device, &queue)?;
        self.relative_rot.download(&device, &queue)?;

        let wp = self.world_pos.host()?;
        let wr = self.world_rot.host()?;
        let rp = self.relative_pos.host()?;
        let rr = self.relative_rot.host()?;

        // Skip slot 0 if it's the reference.
        let start = if self.has_reference { 1 } else { 0 };
        let mut out = Vec::with_capacity(self.target_count as usize - start);
        for i in start..self.target_count as usize {
            out.push(FrameReading {
                world: Transform::new(wp[i], wr[i]),
                relative: Transform::new(rp[i], rr[i]),
            });
        }
        Ok(out)
    }
}
