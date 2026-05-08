//! `State` — time-varying positions, velocities, forces.

use crate::math::{Real, SpatialVector, Transform, Vec3};
use crate::storage::{Buffer, BufferUsage};

/// Mutable state advanced by a [`crate::solvers::Solver`] each step.
#[derive(Debug)]
pub struct State {
    /// Particle positions (m).
    pub particle_q: Buffer<Vec3>,
    /// Particle velocities (m/s).
    pub particle_qd: Buffer<Vec3>,
    /// Particle forces (N).
    pub particle_f: Buffer<Vec3>,

    /// Rigid body poses. Translation is the body-frame origin in world,
    /// not the COM (see `Model.body_com` for the COM offset).
    pub body_q: Buffer<Transform>,
    /// Rigid body twists (m/s, rad/s) — linear is COM velocity in world,
    /// angular is body angular velocity in world frame.
    pub body_qd: Buffer<SpatialVector>,
    /// Rigid body applied wrenches (N, N·m) at COM, world frame.
    pub body_f: Buffer<SpatialVector>,

    /// Generalised joint coordinates (m or rad).
    pub joint_q: Buffer<Real>,
    /// Generalised joint velocities (m/s or rad/s).
    pub joint_qd: Buffer<Real>,
}

impl State {
    /// An empty state — zero of everything. Useful as a sentinel and for
    /// tests that don't need a real `Model`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            particle_q: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            particle_qd: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            particle_f: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            body_q: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            body_qd: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            body_f: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            joint_q: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
            joint_qd: Buffer::from_vec(Vec::new(), BufferUsage::StorageReadWrite),
        }
    }

    /// Allocate a state sized for `model`. Initial body poses come from
    /// `model.initial_body_q`; everything else is zero.
    pub fn for_model(model: &super::Model) -> crate::core::Result<Self> {
        let initial = model.initial_body_q.host()?.to_vec();
        Ok(Self {
            particle_q: Buffer::from_vec(
                vec![Vec3::ZERO; model.particle_count],
                BufferUsage::StorageReadWrite,
            ),
            particle_qd: Buffer::from_vec(
                vec![Vec3::ZERO; model.particle_count],
                BufferUsage::StorageReadWrite,
            ),
            particle_f: Buffer::from_vec(
                vec![Vec3::ZERO; model.particle_count],
                BufferUsage::StorageReadWrite,
            ),
            body_q: Buffer::from_vec(initial, BufferUsage::StorageReadWrite),
            body_qd: Buffer::from_vec(
                vec![SpatialVector::ZERO; model.body_count],
                BufferUsage::StorageReadWrite,
            ),
            body_f: Buffer::from_vec(
                vec![SpatialVector::ZERO; model.body_count],
                BufferUsage::StorageReadWrite,
            ),
            joint_q: Buffer::from_vec(
                model.initial_joint_q.host()?.to_vec(),
                BufferUsage::StorageReadWrite,
            ),
            joint_qd: Buffer::from_vec(
                vec![0.0; model.joint_dof_total],
                BufferUsage::StorageReadWrite,
            ),
        })
    }

    /// Zero out all force arrays. Solvers call this at the top of each step.
    pub fn clear_forces(&mut self) -> crate::core::Result<()> {
        for f in self.body_f.host_mut()? {
            *f = SpatialVector::ZERO;
        }
        for f in self.particle_f.host_mut()? {
            *f = Vec3::ZERO;
        }
        Ok(())
    }

    /// Allocate device buffers for every state array and upload host data.
    /// Idempotent: subsequent calls re-upload from host.
    #[cfg(feature = "gpu")]
    pub fn upload_to_device(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> crate::core::Result<()> {
        macro_rules! up {
            ($field:ident) => {{
                self.$field.ensure_device(device)?;
                self.$field.upload(queue)?;
            }};
        }
        up!(particle_q);
        up!(particle_qd);
        up!(particle_f);
        up!(body_q);
        up!(body_qd);
        up!(body_f);
        up!(joint_q);
        up!(joint_qd);
        Ok(())
    }

    /// Read every device buffer back into the host mirror. Blocks until
    /// the transfers complete.
    #[cfg(feature = "gpu")]
    pub fn download_from_device(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> crate::core::Result<()> {
        self.body_q.download(device, queue)?;
        self.body_qd.download(device, queue)?;
        Ok(())
    }
}
