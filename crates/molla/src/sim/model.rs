//! `Model` — static, immutable simulation description.
//!
//! Stage 0: field declarations only. Field access patterns and the
//! `ModelBuilder::build` path are stage 1.

use crate::math::{Mat3, Real, Transform, Vec3};
use crate::storage::Buffer;

/// Static description of a simulation. Built once from a `ModelBuilder`
/// (added in stage 1) and shared across many [`State`](super::State) instances.
#[derive(Debug)]
pub struct Model {
    // ── Particles ───────────────────────────────────────────────────
    /// Number of particles.
    pub particle_count: usize,
    /// Per-particle mass (kg).
    pub particle_mass: Buffer<Real>,
    /// Per-particle inverse mass (1/kg), 0 for kinematic.
    pub particle_inv_mass: Buffer<Real>,
    /// Per-particle radius (m).
    pub particle_radius: Buffer<Real>,
    /// Per-particle bitflags (see `ParticleFlags` — added in stage 4).
    pub particle_flags: Buffer<u32>,
    /// Per-particle world index (-1 = global).
    pub particle_world: Buffer<i32>,

    // ── Rigid bodies ────────────────────────────────────────────────
    /// Number of rigid bodies.
    pub body_count: usize,
    /// Per-body mass (kg).
    pub body_mass: Buffer<Real>,
    /// Per-body inverse mass (1/kg), 0 for static/kinematic.
    pub body_inv_mass: Buffer<Real>,
    /// Per-body inertia tensor at COM, body frame (kg·m²).
    pub body_inertia: Buffer<Mat3>,
    /// Per-body inverse inertia tensor.
    pub body_inv_inertia: Buffer<Mat3>,
    /// Per-body centre of mass in body frame (m).
    pub body_com: Buffer<Vec3>,
    /// Per-body bitflags (see `BodyFlags` — added in stage 1).
    pub body_flags: Buffer<u32>,
    /// Per-body world index.
    pub body_world: Buffer<i32>,

    // ── Shapes ──────────────────────────────────────────────────────
    /// Number of collision shapes.
    pub shape_count: usize,
    /// Owning body index, or -1 for static-world shapes.
    pub shape_body: Buffer<i32>,
    /// Shape transform in owning body frame.
    pub shape_transform: Buffer<Transform>,
    /// Per-shape bitflags.
    pub shape_flags: Buffer<u32>,

    // ── Joints ──────────────────────────────────────────────────────
    /// Number of joints.
    pub joint_count: usize,
    /// Per-joint parent body index (-1 = world).
    pub joint_parent: Buffer<i32>,
    /// Per-joint child body index.
    pub joint_child: Buffer<i32>,
    /// Joint frame in parent body.
    pub joint_x_p: Buffer<Transform>,
    /// Joint frame in child body.
    pub joint_x_c: Buffer<Transform>,

    // ── Worlds ──────────────────────────────────────────────────────
    /// Number of independent worlds (RL/multi-env grouping).
    pub world_count: usize,
    /// Per-world gravity vector (m/s²).
    pub gravity: Buffer<Vec3>,
}

impl Model {
    /// An empty model: zero particles, zero bodies, zero shapes, one
    /// world with standard gravity. Used as a sentinel and as the seed
    /// for ad-hoc construction in tests.
    #[must_use]
    pub fn empty() -> Self {
        use crate::storage::BufferUsage::Storage;
        Self {
            particle_count: 0,
            particle_mass: Buffer::from_vec(Vec::new(), Storage),
            particle_inv_mass: Buffer::from_vec(Vec::new(), Storage),
            particle_radius: Buffer::from_vec(Vec::new(), Storage),
            particle_flags: Buffer::from_vec(Vec::new(), Storage),
            particle_world: Buffer::from_vec(Vec::new(), Storage),

            body_count: 0,
            body_mass: Buffer::from_vec(Vec::new(), Storage),
            body_inv_mass: Buffer::from_vec(Vec::new(), Storage),
            body_inertia: Buffer::from_vec(Vec::new(), Storage),
            body_inv_inertia: Buffer::from_vec(Vec::new(), Storage),
            body_com: Buffer::from_vec(Vec::new(), Storage),
            body_flags: Buffer::from_vec(Vec::new(), Storage),
            body_world: Buffer::from_vec(Vec::new(), Storage),

            shape_count: 0,
            shape_body: Buffer::from_vec(Vec::new(), Storage),
            shape_transform: Buffer::from_vec(Vec::new(), Storage),
            shape_flags: Buffer::from_vec(Vec::new(), Storage),

            joint_count: 0,
            joint_parent: Buffer::from_vec(Vec::new(), Storage),
            joint_child: Buffer::from_vec(Vec::new(), Storage),
            joint_x_p: Buffer::from_vec(Vec::new(), Storage),
            joint_x_c: Buffer::from_vec(Vec::new(), Storage),

            world_count: 1,
            gravity: Buffer::from_vec(vec![Vec3::new(0.0, -crate::math::GRAVITY, 0.0)], Storage),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_model_has_one_world_with_gravity() {
        let m = Model::empty();
        assert_eq!(m.body_count, 0);
        assert_eq!(m.particle_count, 0);
        assert_eq!(m.world_count, 1);
        let g = m.gravity.host().unwrap();
        assert_eq!(g.len(), 1);
        assert!((g[0].y + crate::math::GRAVITY).abs() < 1e-12);
    }
}
