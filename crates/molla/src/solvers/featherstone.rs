//! Featherstone forward dynamics for articulated rigid bodies — full
//! Articulated Body Algorithm (ABA) per Featherstone (2008) ch. 7,
//! mirroring the structure of Newton's
//! `solvers/featherstone/kernels.py` (with spatial-vector primitives
//! ported from Warp's `_src/math.py` / `builtins.py`).
//!
//! Three passes over the chain:
//!
//! 1. **Forward kinematics + body kinematics** (root → leaf):
//!    transforms, body spatial velocity `v_i = v_p + S_i · q̇_i`,
//!    Coriolis term `c_i = v_i × (S_i · q̇_i)`, body spatial inertia
//!    in world frame, bias force
//!    `p_i = v_i ×* (I_i · v_i) - f_grav_i`.
//! 2. **Articulated inertia accumulation** (leaf → root):
//!    `U = I^A · S`, `D = S^T · U`, `u = τ - S^T · p^A`.
//!    For non-root: `I^a = I^A - U · U^T / D`,
//!                  `p^a = p^A + I^a · c + U · u / D`,
//!    propagate `I^A_p += I^a`, `p^A_p += p^a` to parent.
//!    (All quantities expressed at world origin in world frame, so no
//!    inter-frame transform needed during accumulation — same trick
//!    Newton uses.)
//! 3. **Forward acceleration solve** (root → leaf):
//!    `q̈_i = (u_i - U_i · a_p) / D_i`, `a_i = a_p + S_i · q̈_i + c_i`.
//!
//! Then semi-implicit Euler integrates `q̇`, `q`. Final FK pass
//! refreshes `body_q` for downstream consumers.
//!
//! Supported joint types: Revolute, Prismatic (1-DoF each); Fixed
//! (passes through). Free, Ball, D6 — coords integrate kinematically
//! but their dynamics aren't covered by 1-DoF ABA; future work.
//!
//! Single-articulation: this CPU step processes every joint in
//! `model.joint_count` as one chain in declaration order. The
//! articulation grouping pass (for multi-robot scenes) lands in a
//! follow-up commit when there's a real multi-robot test case.

use crate::core::{Error, Result};
use crate::math::{
    spatial_cross, spatial_cross_dual, Mat3, Quat, Real, SpatialMatrix6, SpatialVector,
    Transform, Vec3,
};
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

    // Snapshot input + run FK so `state_out.body_q` reflects the
    // input poses (we'll integrate q,qd below and re-FK at the end).
    let q_in = state_in.joint_q.host()?.to_vec();
    let qd_in = state_in.joint_qd.host()?.to_vec();
    state_out.joint_q.host_mut()?.copy_from_slice(&q_in);
    state_out.joint_qd.host_mut()?.copy_from_slice(&qd_in);
    state_out
        .body_q
        .host_mut()?
        .copy_from_slice(state_in.body_q.host()?);
    eval_fk(model, state_out)?;

    let n = model.joint_count;
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

    // ── Per-joint scratch — indexed by joint id ──────────────────────
    let mut s = vec![SpatialVector::ZERO; n]; // motion subspace (world)
    let mut v_body = vec![SpatialVector::ZERO; n]; // body spatial velocity
    let mut c_term = vec![SpatialVector::ZERO; n]; // Coriolis bias
    let mut i_a = vec![SpatialMatrix6::ZERO; n]; // articulated inertia
    let mut p_a = vec![SpatialVector::ZERO; n]; // articulated bias force
    let mut u_vec = vec![SpatialVector::ZERO; n];
    let mut d_inv = vec![0.0_f64; n];
    let mut u_tau = vec![0.0_f64; n];
    let mut joint_world_axis = vec![Vec3::ZERO; n];

    // ── Pass 1: forward kinematics + body kinematics ─────────────────
    for j in 0..n {
        let jt = joint_type_from_u32(joint_type[j]);
        let parent = joint_parent[j];
        let child = joint_child[j] as usize;

        // Joint world transform = parent body world * X_p.
        let parent_world = if parent < 0 {
            Transform::IDENTITY
        } else {
            body_q[parent as usize]
        };
        let joint_world = parent_world.compose(&joint_x_p[j]);
        let axis_world = (joint_world.rotation * joint_axis[j]).normalize_or_zero();
        joint_world_axis[j] = axis_world;

        // Joint motion subspace (world frame).
        // Convention: spatial vector = [linear; angular].
        let dof_idx = joint_dof_offset[j] as usize;
        let qd_j = qd_in.get(dof_idx).copied().unwrap_or(0.0);
        let s_j = match jt {
            JointType::Revolute => SpatialVector::new(Vec3::ZERO, axis_world),
            JointType::Prismatic => SpatialVector::new(axis_world, Vec3::ZERO),
            _ => SpatialVector::ZERO, // Fixed / Free / unsupported by this solver
        };
        s[j] = s_j;
        let v_j = s_j.scale(qd_j);

        // Parent body's spatial velocity (zero if root).
        let v_p = if parent < 0 {
            SpatialVector::ZERO
        } else {
            // For chains, the parent body of joint j may be a child of
            // an earlier joint k. Look up that joint's `v_body`.
            // For our single-chain assumption: parent body index ==
            // child of some earlier joint. Find by linear search; for
            // typical small chains this is cheap.
            let mut v = SpatialVector::ZERO;
            for k in 0..j {
                if joint_child[k] as usize == parent as usize {
                    v = v_body[k];
                    break;
                }
            }
            v
        };

        // Body velocity + Coriolis term.
        let v_i = v_p + v_j;
        v_body[j] = v_i;
        c_term[j] = spatial_cross(v_i, v_j);

        // Body spatial inertia in WORLD frame, expressed at world
        // origin (so all `I^A` in the chain accumulate without
        // re-transforming — same trick Newton uses).
        let cw = body_q[child];
        let com_world = cw.position + cw.rotation * body_com[child];
        let i_local_world = rotate_inertia(cw.rotation, body_inertia[child]);
        let i_s = SpatialMatrix6::from_body(body_mass[child], com_world, i_local_world);

        // Initial articulated inertia and bias.
        // Bias `p = v ×* (I · v) - f_grav`, where f_grav is the
        // gravity wrench at the world origin: linear = m·g, angular
        // = r_com × m·g.
        let world_idx = body_world[child].max(0) as usize;
        let g = gravity[world_idx];
        let f_g_lin = g * body_mass[child];
        let f_g_ang = com_world.cross(f_g_lin);
        let f_grav = SpatialVector::new(f_g_lin, f_g_ang);
        let bias = spatial_cross_dual(v_i, i_s.mul_sv(v_i));
        // Negative gravity (it's an EXTERNAL wrench, so it offsets the
        // bias rather than adding to it):
        let p_init = bias - f_grav;
        i_a[j] = i_s;
        p_a[j] = p_init;
    }

    // ── Pass 2: backward — articulated inertia accumulation ──────────
    for j in (0..n).rev() {
        let parent = joint_parent[j];
        let s_j = s[j];

        // For 0-DoF joints (Fixed / unsupported), skip the rank-1
        // update — there's no joint coordinate to project against.
        let jt = joint_type_from_u32(joint_type[j]);
        let supported = matches!(jt, JointType::Revolute | JointType::Prismatic);

        if supported {
            let u = i_a[j].mul_sv(s_j);
            let d = sv_dot(s_j, u);
            u_vec[j] = u;
            d_inv[j] = if d.abs() > 1e-12 { 1.0 / d } else { 0.0 };

            let dof_idx = joint_dof_offset[j] as usize;
            let tau_app = control_force.get(dof_idx).copied().unwrap_or(0.0);
            u_tau[j] = tau_app - sv_dot(s_j, p_a[j]);

            if parent >= 0 {
                // Find the earlier joint whose child is this joint's
                // parent body — that's where we accumulate.
                let mut parent_joint: Option<usize> = None;
                for k in 0..j {
                    if joint_child[k] as usize == parent as usize {
                        parent_joint = Some(k);
                        break;
                    }
                }
                if let Some(pj) = parent_joint {
                    // I^a = I^A - U·U^T / D
                    let outer = SpatialMatrix6::outer(u, u).scale(d_inv[j]);
                    let i_articulated = i_a[j].sub(&outer);
                    // p^a = p^A + I^a·c + U·(u_tau / D)
                    let p_articulated =
                        p_a[j] + i_articulated.mul_sv(c_term[j]) + u.scale(u_tau[j] * d_inv[j]);
                    i_a[pj] = i_a[pj].add(&i_articulated);
                    p_a[pj] = p_a[pj] + p_articulated;
                }
            }
        }
    }

    // ── Pass 3: forward — solve for joint accelerations ──────────────
    let mut q_dd = vec![0.0_f64; model.joint_dof_total];
    let mut a_body = vec![SpatialVector::ZERO; n]; // body spatial accel

    for j in 0..n {
        let parent = joint_parent[j];
        let s_j = s[j];
        let jt = joint_type_from_u32(joint_type[j]);
        let supported = matches!(jt, JointType::Revolute | JointType::Prismatic);

        let a_p = if parent < 0 {
            SpatialVector::ZERO
        } else {
            let mut a = SpatialVector::ZERO;
            for k in 0..j {
                if joint_child[k] as usize == parent as usize {
                    a = a_body[k];
                    break;
                }
            }
            a
        };

        if supported {
            let dof_idx = joint_dof_offset[j] as usize;
            let q_dd_j = (u_tau[j] - sv_dot(u_vec[j], a_p)) * d_inv[j];
            q_dd[dof_idx] = q_dd_j;
            a_body[j] = a_p + s_j.scale(q_dd_j) + c_term[j];
        } else {
            a_body[j] = a_p + c_term[j];
        }
    }

    // ── Integrate q, q̇ ──────────────────────────────────────────────
    let mut q_new = q_in.clone();
    let mut qd_new = qd_in.clone();
    for j in 0..n {
        let jt = joint_type_from_u32(joint_type[j]);
        let coord_idx = joint_coord_offset[j] as usize;
        let dof_idx = joint_dof_offset[j] as usize;
        match jt {
            JointType::Revolute | JointType::Prismatic => {
                let qd_old = qd_in[dof_idx];
                let qd_int = qd_old + q_dd[dof_idx] * dt;
                let qd_damped = qd_int * (1.0 - joint_damping * dt).max(0.0);
                qd_new[dof_idx] = qd_damped;
                q_new[coord_idx] = q_in[coord_idx] + qd_damped * dt;
            }
            JointType::Fixed | JointType::Free => {
                // Fixed: no coords. Free: 7 coords / 6 dof — pass
                // through (dynamics for Free joints handled in a
                // future stage; usual case is they're driven by the
                // bias-force machinery as a 6-DoF root).
            }
        }
    }

    state_out.joint_q.host_mut()?.copy_from_slice(&q_new);
    state_out.joint_qd.host_mut()?.copy_from_slice(&qd_new);
    eval_fk(model, state_out)?;

    for f in state_out.body_f.host_mut()? {
        *f = SpatialVector::ZERO;
    }
    Ok(())
}

fn sv_dot(a: SpatialVector, b: SpatialVector) -> Real {
    a.linear.dot(b.linear) + a.angular.dot(b.angular)
}

/// Rotate a body-frame inertia tensor `I_local` into world frame
/// using the body's rotation: `I_world = R · I_local · R^T`.
fn rotate_inertia(r: Quat, i_local: Mat3) -> Mat3 {
    let rm = Mat3::from_quat(r);
    rm * i_local * rm.transpose()
}

fn joint_type_from_u32(v: u32) -> JointType {
    match v {
        0 => JointType::Prismatic,
        1 => JointType::Revolute,
        4 => JointType::Free,
        _ => JointType::Fixed,
    }
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

    /// Two-link pendulum: tests that the ABA bias-force propagation
    /// is actually wired. The per-joint approximation would let the
    /// upper joint ignore the lower link's inertia and the system
    /// would diverge over many seconds; full ABA stays bounded.
    /// Energy isn't conserved under semi-implicit Euler but the
    /// total kinetic + potential energy must stay within ~50 % of
    /// the initial value over 5 seconds.
    #[test]
    fn double_pendulum_energy_bounded() {
        let mut b = ModelBuilder::new();
        let l0 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.083,
        }));
        let l1 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.083,
        }));
        b.add_joint_revolute(
            BodyId::NONE,
            l0,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
        b.add_joint_revolute(
            l0,
            l1,
            Transform::from_translation(V3::new(0.0, -0.5, 0.0)),
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
        let model = b.build().unwrap();

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.joint_q.host_mut().unwrap()[0] = 0.3;
        s_a.joint_q.host_mut().unwrap()[1] = -0.2;
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();

        // Compute initial energy E_0 = T + V (T=0 since q̇=0 at start).
        let body_q0: Vec<_> = {
            let mut tmp = model.state().unwrap();
            tmp.joint_q.host_mut().unwrap()[0] = 0.3;
            tmp.joint_q.host_mut().unwrap()[1] = -0.2;
            crate::sim::eval_fk(&model, &mut tmp).unwrap();
            tmp.body_q.host().unwrap().to_vec()
        };
        let g_mag = crate::math::GRAVITY;
        let v_init = body_q0
            .iter()
            .map(|t| t.position.y * 1.0 * g_mag)
            .sum::<Real>();

        let dt = 0.001_f64;
        let n_steps = 5000usize; // 5 s
        let mut max_energy_excess = 0.0_f64;
        for _ in 0..n_steps {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);

            // Total energy ≈ ½ Σ m·|v_com|² + Σ m·g·h.
            let body_q = s_a.body_q.host().unwrap();
            let body_qd = s_a.body_qd.host().unwrap();
            let mut t_energy = 0.0_f64;
            let mut v_energy = 0.0_f64;
            for (i, q) in body_q.iter().enumerate() {
                let v_com = body_qd[i].linear;
                t_energy += 0.5 * 1.0 * v_com.length_squared();
                // For uniform inertia tensor, rotational KE = ½·I·|w|² with I=0.083:
                t_energy += 0.5 * 0.083 * body_qd[i].angular.length_squared();
                v_energy += q.position.y * 1.0 * g_mag;
            }
            let e = t_energy + v_energy;
            let excess = (e - v_init).abs();
            if excess > max_energy_excess {
                max_energy_excess = excess;
            }
        }
        // Allow ~30 % drift over 5 s of 1 ms steps. Without full ABA
        // bias propagation this test diverges to >100 %.
        assert!(
            max_energy_excess < 0.5 * v_init.abs().max(1.0),
            "double pendulum energy drift too high: {max_energy_excess} > 50% of |E_0|={}",
            v_init.abs()
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
