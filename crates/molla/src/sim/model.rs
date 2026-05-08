//! `Model` — static, immutable simulation description.

use crate::core::Result;
use crate::math::{Mat3, Real, Transform, Vec3};
use crate::storage::{Buffer, BufferUsage};

/// Static description of a simulation. Built once from a `ModelBuilder`
/// and shared across many [`State`](super::State) instances.
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
    /// Per-particle bitflags (see [`super::ParticleFlags`]).
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
    /// Per-body bitflags (see [`super::BodyFlags`]).
    pub body_flags: Buffer<u32>,
    /// Per-body world index.
    pub body_world: Buffer<i32>,
    /// Initial body pose snapshotted at build time. [`super::State::for_model`]
    /// copies this into the working state.
    pub initial_body_q: Buffer<Transform>,

    // ── Shapes ──────────────────────────────────────────────────────
    /// Number of collision shapes.
    pub shape_count: usize,
    /// Owning body index, or -1 for static-world shapes.
    pub shape_body: Buffer<i32>,
    /// Shape transform in owning body frame.
    pub shape_transform: Buffer<Transform>,
    /// Per-shape geometry tags (sphere radius, etc). Stage 4 expands this.
    pub shape_geometry: Vec<super::ShapeGeometry>,
    /// Per-shape bitflags.
    pub shape_flags: Buffer<u32>,

    // ── Joints (placeholder until stage 3) ──────────────────────────
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

    // ── Global solver knobs ─────────────────────────────────────────
    /// Angular damping coefficient applied each step (1/s).
    pub angular_damping: Real,
    /// Velocity cap applied to particles (m/s) for stability.
    pub particle_velocity_max: Real,
}

impl Model {
    /// An empty model: zero particles, zero bodies, zero shapes, one
    /// world with standard gravity.
    #[must_use]
    pub fn empty() -> Self {
        let s = BufferUsage::Storage;
        Self {
            particle_count: 0,
            particle_mass: Buffer::from_vec(Vec::new(), s),
            particle_inv_mass: Buffer::from_vec(Vec::new(), s),
            particle_radius: Buffer::from_vec(Vec::new(), s),
            particle_flags: Buffer::from_vec(Vec::new(), s),
            particle_world: Buffer::from_vec(Vec::new(), s),

            body_count: 0,
            body_mass: Buffer::from_vec(Vec::new(), s),
            body_inv_mass: Buffer::from_vec(Vec::new(), s),
            body_inertia: Buffer::from_vec(Vec::new(), s),
            body_inv_inertia: Buffer::from_vec(Vec::new(), s),
            body_com: Buffer::from_vec(Vec::new(), s),
            body_flags: Buffer::from_vec(Vec::new(), s),
            body_world: Buffer::from_vec(Vec::new(), s),
            initial_body_q: Buffer::from_vec(Vec::new(), s),

            shape_count: 0,
            shape_body: Buffer::from_vec(Vec::new(), s),
            shape_transform: Buffer::from_vec(Vec::new(), s),
            shape_geometry: Vec::new(),
            shape_flags: Buffer::from_vec(Vec::new(), s),

            joint_count: 0,
            joint_parent: Buffer::from_vec(Vec::new(), s),
            joint_child: Buffer::from_vec(Vec::new(), s),
            joint_x_p: Buffer::from_vec(Vec::new(), s),
            joint_x_c: Buffer::from_vec(Vec::new(), s),

            world_count: 1,
            gravity: Buffer::from_vec(vec![Vec3::new(0.0, -crate::math::GRAVITY, 0.0)], s),

            angular_damping: 0.0,
            particle_velocity_max: 1.0e6,
        }
    }

    /// Allocate a fresh [`super::State`] sized for this model.
    pub fn state(&self) -> Result<super::State> {
        super::State::for_model(self)
    }

    /// Allocate a fresh [`super::Control`] sized for this model.
    #[must_use]
    pub fn control(&self) -> super::Control {
        super::Control::for_model(self)
    }

    /// Allocate device buffers for every model array and upload host data.
    ///
    /// The model is read-only at simulation time, so this only needs to
    /// run once before the first GPU step. Re-running is harmless but
    /// re-uploads the same data.
    #[cfg(feature = "gpu")]
    pub fn upload_to_device(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        macro_rules! up {
            ($field:ident) => {{
                self.$field.ensure_device(device)?;
                self.$field.upload(queue)?;
            }};
        }
        up!(particle_mass);
        up!(particle_inv_mass);
        up!(particle_radius);
        up!(particle_flags);
        up!(particle_world);
        up!(body_mass);
        up!(body_inv_mass);
        up!(body_inertia);
        up!(body_inv_inertia);
        up!(body_com);
        up!(body_flags);
        up!(body_world);
        up!(initial_body_q);
        up!(shape_body);
        up!(shape_transform);
        up!(shape_flags);
        up!(joint_parent);
        up!(joint_child);
        up!(joint_x_p);
        up!(joint_x_c);
        up!(gravity);
        Ok(())
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
