//! Semi-implicit Euler integrator for rigid bodies and particles.
//!
//! Direct CPU port of Newton's `integrate_rigid_body` (see
//! `xtra/newton/newton/_src/solvers/solver.py`). Stage 1 implements rigid
//! bodies only; particles arrive in stage 5.
//!
//! Semi-implicit means we update velocity from forces, then position from
//! the **new** velocity:
//!
//! ```text
//!   v_{k+1} = v_k + f * inv_m * dt + g * dt
//!   x_{k+1} = x_k + v_{k+1} * dt
//! ```
//!
//! Angular dynamics are integrated in body frame to keep the inertia
//! tensor diagonal, then rotated back. Quaternions are renormalised every
//! step.

use crate::core::{Error, Result};
use crate::math::{Mat3, Quat, Real, SpatialVector, Transform, Vec3};
use crate::sim::{BodyFlags, Contacts, Control, Model, State};

use super::Solver;

/// Semi-implicit Euler integrator. Stateless — `step` is a pure function
/// of its inputs, modulo the output state arrays.
#[derive(Debug, Default)]
pub struct SemiImplicitSolver;

impl SemiImplicitSolver {
    /// Construct the solver. Cheap; safe to recreate per step.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Solver for SemiImplicitSolver {
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        _contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        if state_in.body_q.len() != model.body_count
            || state_out.body_q.len() != model.body_count
        {
            return Err(Error::Solver(format!(
                "state size mismatch: model has {} bodies, state_in has {}, state_out has {}",
                model.body_count,
                state_in.body_q.len(),
                state_out.body_q.len()
            )));
        }

        let q_in = state_in.body_q.host()?;
        let qd_in = state_in.body_qd.host()?;
        let f_in = state_in.body_f.host()?;
        let mass = model.body_mass.host()?;
        let inv_mass = model.body_inv_mass.host()?;
        let inertia = model.body_inertia.host()?;
        let inv_inertia = model.body_inv_inertia.host()?;
        let com = model.body_com.host()?;
        let flags = model.body_flags.host()?;
        let world_idx = model.body_world.host()?;
        let gravity = model.gravity.host()?;
        let external = if control.body_external_f.len() == model.body_count {
            Some(control.body_external_f.host()?)
        } else {
            None
        };

        // Snapshot all inputs before borrowing state_out mutably (host_mut
        // takes exclusive borrow).
        let mut q_out = vec![Transform::IDENTITY; model.body_count];
        let mut qd_out = vec![SpatialVector::ZERO; model.body_count];

        for tid in 0..model.body_count {
            if (flags[tid] & BodyFlags::KINEMATIC) != 0 {
                q_out[tid] = q_in[tid];
                qd_out[tid] = qd_in[tid];
                continue;
            }

            let mut applied = f_in[tid];
            if let Some(ext) = external {
                applied = applied + ext[tid];
            }

            let world = world_idx[tid].max(0) as usize;
            let g = gravity[world];

            let (q_new, qd_new) = integrate_rigid_body(
                q_in[tid],
                qd_in[tid],
                applied,
                com[tid],
                mass[tid],
                inertia[tid],
                inv_mass[tid],
                inv_inertia[tid],
                g,
                model.angular_damping,
                dt,
            );
            q_out[tid] = q_new;
            qd_out[tid] = qd_new;
        }

        state_out.body_q.host_mut()?.copy_from_slice(&q_out);
        state_out.body_qd.host_mut()?.copy_from_slice(&qd_out);
        // Forces from the input step are not persisted into the next.
        for f in state_out.body_f.host_mut()? {
            *f = SpatialVector::ZERO;
        }

        Ok(())
    }
}

/// Closed-form semi-implicit Euler step for one rigid body.
///
/// Direct port of Newton's `integrate_rigid_body` in
/// `xtra/newton/newton/_src/solvers/solver.py`. Arguments mirror that
/// kernel in the same order.
#[allow(clippy::too_many_arguments, reason = "matches Newton's reference signature")]
fn integrate_rigid_body(
    q: Transform,
    qd: SpatialVector,
    f: SpatialVector,
    com: Vec3,
    _mass: Real,
    inertia: Mat3,
    inv_mass: Real,
    inv_inertia: Mat3,
    gravity: Vec3,
    angular_damping: Real,
    dt: Real,
) -> (Transform, SpatialVector) {
    let x0 = q.position;
    let r0 = q.rotation;

    // Newton's spatial layout: top = linear, bottom = angular.
    let v0 = qd.linear;
    let w0 = qd.angular;
    let f0 = f.linear; // applied force (world frame, at COM)
    let t0 = f.angular; // applied torque (world frame)

    // World position of the COM: body origin + R * com_local.
    let x_com = x0 + r0 * com;

    // Linear part. Gravity acts only on bodies with nonzero inv_mass,
    // matching Newton's `wp.nonzero(inv_mass)` gate.
    let gravity_gate = if inv_mass > 0.0 { 1.0 } else { 0.0 };
    let v1 = v0 + (f0 * inv_mass + gravity * gravity_gate) * dt;
    let x1 = x_com + v1 * dt;

    // Angular part — compute in body frame so inertia is diagonal.
    let r0_inv = r0.conjugate();
    let wb = r0_inv * w0;
    let tb = r0_inv * t0 - (wb).cross(inertia * wb); // Coriolis

    let mut w1 = r0 * (wb + inv_inertia * tb * dt);

    // Quaternion update: r1 = normalise(r0 + 0.5 * dt * (w1 ⊗ r0)).
    // Newton stores the angular velocity quaternion as (w_xyz, 0).
    let w_quat = Quat::from_xyzw(w1.x, w1.y, w1.z, 0.0);
    let r1 = (r0 + w_quat * r0 * (0.5 * dt)).normalize();

    // Angular damping (post-rotation, as Newton does).
    w1 *= 1.0 - angular_damping * dt;

    // Recover body-origin pose: x_com - R1 * com_local.
    let position_new = x1 - r1 * com;

    let q_new = Transform::new(position_new, r1);
    let qd_new = SpatialVector::new(v1, w1);
    (q_new, qd_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::SpatialInertia;
    use crate::sim::{BodyParams, ModelBuilder};

    /// One-body free fall, com=0. After `n` steps from rest under gravity g,
    /// semi-implicit Euler gives:
    ///   v_n = g * n * dt
    ///   x_n = x_0 + g * dt² * n*(n+1)/2
    #[test]
    fn free_fall_matches_closed_form() {
        let mut b = ModelBuilder::new();
        b.set_angular_damping(0.0);
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
                .with_position(Vec3::new(0.0, 10.0, 0.0)),
        );
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);

        let mut solver = SemiImplicitSolver::new();
        let dt = 0.01;
        let n_steps = 100usize;

        for _ in 0..n_steps {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }

        let g = -crate::math::GRAVITY;
        let n = n_steps as Real;
        let expected_v = g * n * dt;
        let expected_x = 10.0 + g * dt * dt * n * (n + 1.0) / 2.0;

        let q = s_a.body_q.host().unwrap();
        let qd = s_a.body_qd.host().unwrap();
        let got_x = q[0].position.y;
        let got_v = qd[0].linear.y;

        assert!(
            (got_x - expected_x).abs() < 1e-9,
            "y mismatch: got {got_x}, expected {expected_x}"
        );
        assert!(
            (got_v - expected_v).abs() < 1e-9,
            "v_y mismatch: got {got_v}, expected {expected_v}"
        );
    }

    #[test]
    fn kinematic_body_passes_through() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_position(Vec3::new(0.0, 5.0, 0.0))
                .with_flags(BodyFlags::KINEMATIC),
        );
        b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = SemiImplicitSolver::new();

        for _ in 0..10 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.01, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.body_q.host().unwrap();
        assert!((q[0].position.y - 5.0).abs() < 1e-12);
    }

    #[test]
    fn zero_mass_body_does_not_fall() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_mass(0.0)
                .with_inertia(Mat3::ZERO)
                .with_position(Vec3::new(0.0, 5.0, 0.0)),
        );
        b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = SemiImplicitSolver::new();

        for _ in 0..50 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.01, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.body_q.host().unwrap();
        assert!((q[0].position.y - 5.0).abs() < 1e-12);
    }
}
