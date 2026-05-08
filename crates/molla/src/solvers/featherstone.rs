//! Featherstone-style forward dynamics for articulated rigid bodies.
//!
//! **Stage 3.1 scope**: per-joint dynamics for single-chain revolute
//! joints with isotropic / aligned-diagonal inertias. The math
//! computes for each revolute joint:
//!
//! - the gravitational torque about the joint axis
//! - the effective inertia about the joint axis (parallel axis)
//! - the joint acceleration `q̈ = τ / I_eff`
//! - semi-implicit Euler integration of `q̇` and `q`
//! - re-evaluates FK at the end so `body_q` reflects the new poses.
//!
//! This is *not* full Featherstone ABA — it ignores cross-coupling
//! between joints in the chain (the bias terms that proper ABA
//! propagates through the backward pass). For a single-link
//! pendulum it is exact; for chains the per-joint approximation
//! drifts compared to Newton's MuJoCo / Featherstone reference.
//! Full ABA + branched trees come in stage 3.x.
//!
//! Joints other than revolute are passed through with no dynamics
//! (their `joint_qd` is left untouched). Fixed joints are correctly
//! handled by FK (zero coordinates → identity local transform).

use crate::core::{Error, Result};
use crate::math::{Mat3, Real, Vec3};
use crate::sim::{eval_fk, Contacts, Control, JointType, Model, State};

use super::Solver;

/// Backend selector for [`FeatherstoneSolver`].
#[allow(clippy::large_enum_variant, reason = "GPU variant holds Arcs / kernel; one solver instance, fine")]
pub enum FeatherstoneBackend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(GpuBackend),
}

#[cfg(feature = "gpu")]
pub struct GpuBackend {
    pub(crate) device: std::sync::Arc<wgpu::Device>,
    pub(crate) queue: std::sync::Arc<wgpu::Queue>,
    pub(crate) kernel: crate::kernels::FeatherstoneKernel,
}

/// Forward-dynamics solver for revolute joint chains.
pub struct FeatherstoneSolver {
    /// Joint-space damping coefficient (1/s) applied to `joint_qd`
    /// each step. Helps keep long-running pendulums numerically calm.
    pub joint_damping: Real,
    backend: FeatherstoneBackend,
}

impl Default for FeatherstoneSolver {
    fn default() -> Self {
        Self::new_cpu()
    }
}

impl FeatherstoneSolver {
    /// CPU backend.
    #[must_use]
    pub fn new_cpu() -> Self {
        Self {
            joint_damping: 0.0,
            backend: FeatherstoneBackend::Cpu,
        }
    }

    /// Backwards-compat shim — same as [`Self::new_cpu`].
    #[must_use]
    pub fn new() -> Self {
        Self::new_cpu()
    }

    /// GPU backend. Caller must already have uploaded `model` /
    /// `state` / `control` device mirrors before stepping.
    #[cfg(feature = "gpu")]
    pub fn new_gpu(engine: &crate::compute::Engine, model: &Model) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("engine has no GPU queue".into()))?
            .clone();
        let kernel = crate::kernels::FeatherstoneKernel::new(&device, &queue, model);
        Ok(Self {
            joint_damping: 0.0,
            backend: FeatherstoneBackend::Gpu(GpuBackend {
                device,
                queue,
                kernel,
            }),
        })
    }

    #[must_use]
    pub fn with_damping(mut self, d: Real) -> Self {
        self.joint_damping = d;
        self
    }

    #[must_use]
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self.backend, FeatherstoneBackend::Gpu(_))
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
}

impl Solver for FeatherstoneSolver {
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        match &mut self.backend {
            FeatherstoneBackend::Cpu => {
                cpu_step(model, state_in, control, contacts, dt, state_out, self.joint_damping)
            }
            #[cfg(feature = "gpu")]
            FeatherstoneBackend::Gpu(gpu) => {
                let GpuBackend { device, queue, kernel } = gpu;
                kernel.dispatch(
                    device,
                    queue,
                    model,
                    state_in,
                    control,
                    contacts,
                    self.joint_damping as f32,
                    dt,
                    state_out,
                )
            }
        }
    }
}

fn cpu_step(
    model: &Model,
    state_in: &State,
    control: &Control,
    _contacts: &Contacts,
    dt: Real,
    state_out: &mut State,
    joint_damping: Real,
) -> Result<()> {
    if model.joint_count == 0 {
        // Nothing to integrate. Pass body state through.
        state_out
            .body_q
            .host_mut()?
            .copy_from_slice(state_in.body_q.host()?);
        state_out
            .body_qd
            .host_mut()?
            .copy_from_slice(state_in.body_qd.host()?);
        return Ok(());
    }

        // Snapshot input coords + recompute FK at the input poses so
        // we have fresh `body_q` for the dynamics calc.
        let q_in = state_in.joint_q.host()?.to_vec();
        let qd_in = state_in.joint_qd.host()?.to_vec();

        // Copy into state_out and run FK on it. After this state_out
        // body_q reflects the input pose; we'll integrate qd/q below
        // and call FK again to publish the new pose.
        state_out
            .joint_q
            .host_mut()?
            .copy_from_slice(&q_in);
        state_out
            .joint_qd
            .host_mut()?
            .copy_from_slice(&qd_in);
        state_out
            .body_q
            .host_mut()?
            .copy_from_slice(state_in.body_q.host()?);
        eval_fk(model, state_out)?;

        // Per-joint dynamics.
        let joint_type = model.joint_type.host()?;
        let joint_parent = model.joint_parent.host()?;
        let joint_child = model.joint_child.host()?;
        let joint_x_p = model.joint_x_p.host()?;
        let joint_axis = model.joint_axis.host()?;
        let joint_coord_offset = model.joint_coord_offset.host()?;
        let joint_dof_offset = model.joint_dof_offset.host()?;

        let body_q = state_out.body_q.host()?.to_vec();
        let body_mass = model.body_mass.host()?;
        let body_inertia = model.body_inertia.host()?;
        let body_com = model.body_com.host()?;
        let body_world = model.body_world.host()?;
        let gravity = model.gravity.host()?;

        let control_force = control.joint_force.host()?;

        let mut q_new = q_in.clone();
        let mut qd_new = qd_in.clone();

        for j in 0..model.joint_count {
            let jt = joint_type_from_u32(joint_type[j]);
            if jt != JointType::Revolute {
                continue;
            }
            let parent = joint_parent[j];
            let child = joint_child[j] as usize;

            // Joint world transform = parent body world * X_p.
            let parent_world = if parent < 0 {
                crate::math::Transform::IDENTITY
            } else {
                body_q[parent as usize]
            };
            let joint_world = parent_world.compose(&joint_x_p[j]);
            let axis_world = joint_world.rotation * joint_axis[j];

            // Child world transform + COM in world.
            let child_world = body_q[child];
            let com_world = child_world.position + child_world.rotation * body_com[child];

            // Lever arm from joint to COM.
            let r = com_world - joint_world.position;

            // Gravity force on COM.
            let world_idx = body_world[child].max(0) as usize;
            let grav_force = gravity[world_idx] * body_mass[child];

            // Gravity torque about joint axis.
            let tau_grav = r.cross(grav_force).dot(axis_world);

            // Applied joint torque from `control`.
            let dof_idx = joint_dof_offset[j] as usize;
            let tau_applied = control_force.get(dof_idx).copied().unwrap_or(0.0);

            // Effective inertia about joint axis (parallel axis):
            //   I_eff = axis^T · (R · I_local · R^T) · axis  +  m · |r|²
            let inertia_along_axis = inertia_about_axis_world(
                child_world.rotation,
                body_inertia[child],
                axis_world,
            );
            let i_eff = inertia_along_axis + body_mass[child] * r.length_squared();
            if i_eff <= 0.0 {
                continue;
            }

            let total_torque = tau_applied + tau_grav;
            let q_dd = total_torque / i_eff;

            // Semi-implicit Euler with joint-space damping.
            let coord_idx = joint_coord_offset[j] as usize;
            let qd_old = qd_in[dof_idx];
            let qd_int = qd_old + q_dd * dt;
            let qd_damped = qd_int * (1.0 - joint_damping * dt).max(0.0);
            qd_new[dof_idx] = qd_damped;
            q_new[coord_idx] = q_in[coord_idx] + qd_damped * dt;
        }

        // Write integrated coordinates and re-evaluate FK so
        // `state_out.body_q` is consistent.
        state_out.joint_q.host_mut()?.copy_from_slice(&q_new);
        state_out.joint_qd.host_mut()?.copy_from_slice(&qd_new);
        eval_fk(model, state_out)?;

        // Body forces are stateless across steps for this solver.
        for f in state_out.body_f.host_mut()? {
            *f = crate::math::SpatialVector::ZERO;
        }
        Ok(())
}

fn joint_type_from_u32(v: u32) -> JointType {
    match v {
        0 => JointType::Prismatic,
        1 => JointType::Revolute,
        4 => JointType::Free,
        _ => JointType::Fixed,
    }
}

/// `axis^T · (R · I · R^T) · axis` — the moment of inertia of a body
/// (with body-frame inertia `I_local`) about a world-space axis.
fn inertia_about_axis_world(rotation: crate::math::Quat, i_local: Mat3, axis_world: Vec3) -> Real {
    let axis_local = rotation.conjugate() * axis_world;
    axis_world.dot(rotation * (i_local * axis_local))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BodyId;
    use crate::math::{SpatialInertia, Transform, Vec3 as V3};
    use crate::sim::{BodyParams, ModelBuilder};

    /// Pendulum at small initial angle should oscillate with the
    /// linearised period `T = 2π·√(I_eff / (m·g·L))`. Check the
    /// joint angle returns close to the initial value after one
    /// period (within 5 % — any energy gain or large drift fails).
    #[test]
    fn pendulum_oscillates_with_correct_period() {
        // Body: unit mass, identity inertia (I_zz_com = 1).
        let mut b = ModelBuilder::new();
        let link = b.add_body(
            BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
        );
        // Joint at world origin, child anchor at (0, 0.5, 0) → COM
        // hangs at (0, -0.5, 0) at q=0. L = 0.5.
        b.add_joint_revolute(
            BodyId::NONE,
            link,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();

        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.joint_q.host_mut().unwrap()[0] = 0.1; // ~5.7° initial swing

        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();

        // I_eff = I_zz_com + m·L² = 1 + 1·0.25 = 1.25
        // T = 2π · √(1.25 / (1·9.81·0.5)) ≈ 3.171 s
        let dt = 0.001_f64;
        let period = 2.0
            * std::f64::consts::PI
            * (1.25_f64 / (1.0 * crate::math::GRAVITY * 0.5)).sqrt();
        let n_steps = (period / dt).round() as usize;

        for _ in 0..n_steps {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }

        let q_final = s_a.joint_q.host().unwrap()[0];
        let drift = (q_final - 0.1).abs();
        assert!(
            drift < 0.01,
            "after one period, q drifted from 0.1 to {q_final} (Δ={drift:.4})"
        );
    }

    /// Pendulum with no initial energy + control torque = 0 stays at q=0.
    #[test]
    fn pendulum_at_rest_stays_at_rest() {
        let mut b = ModelBuilder::new();
        let link = b.add_body(
            BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
        );
        b.add_joint_revolute(
            BodyId::NONE,
            link,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();
        for _ in 0..200 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.joint_q.host().unwrap()[0];
        assert!(q.abs() < 1e-6, "rest state drifted: q={q}");
    }
}
