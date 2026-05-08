//! `State` — time-varying positions, velocities, forces.
//!
//! Stage 0: field declarations only.

use crate::math::{Real, SpatialVector, Transform, Vec3};
use crate::storage::Buffer;

/// Mutable state advanced by a [`crate::solvers::Solver`] each step.
#[derive(Debug)]
pub struct State {
    /// Particle positions (m).
    pub particle_q: Buffer<Vec3>,
    /// Particle velocities (m/s).
    pub particle_qd: Buffer<Vec3>,
    /// Particle forces (N).
    pub particle_f: Buffer<Vec3>,

    /// Rigid body poses.
    pub body_q: Buffer<Transform>,
    /// Rigid body twists (m/s, rad/s).
    pub body_qd: Buffer<SpatialVector>,
    /// Rigid body applied wrenches (N, N·m) (world frame, at COM).
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
        use crate::storage::BufferUsage::StorageReadWrite;
        Self {
            particle_q: Buffer::from_vec(Vec::new(), StorageReadWrite),
            particle_qd: Buffer::from_vec(Vec::new(), StorageReadWrite),
            particle_f: Buffer::from_vec(Vec::new(), StorageReadWrite),
            body_q: Buffer::from_vec(Vec::new(), StorageReadWrite),
            body_qd: Buffer::from_vec(Vec::new(), StorageReadWrite),
            body_f: Buffer::from_vec(Vec::new(), StorageReadWrite),
            joint_q: Buffer::from_vec(Vec::new(), StorageReadWrite),
            joint_qd: Buffer::from_vec(Vec::new(), StorageReadWrite),
        }
    }
}
