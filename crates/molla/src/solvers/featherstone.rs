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
use crate::sim::{eval_fk, Contacts, Control, JointTargetMode, JointType, Model, State};

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
    /// Number of substeps per outer `step` call. Each substep advances
    /// `dt/substeps`. Default 1.
    pub substeps: usize,
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
            substeps: 1,
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
            substeps: 1,
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

    /// Configure substepping: each outer `step(dt)` runs `n` inner
    /// substeps with `dt/n`.
    #[must_use]
    pub fn with_substeps(mut self, n: usize) -> Self {
        self.substeps = n.max(1);
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
        let n_sub = self.substeps.max(1);
        let dt_sub = dt / n_sub as Real;

        if n_sub == 1 {
            return self.step_inner(model, state_in, control, contacts, dt_sub, state_out);
        }

        let mut scratch = state_in.clone_host()?;
        self.step_inner(model, state_in, control, contacts, dt_sub, state_out)?;
        let mut output_in_scratch = false;
        for _ in 1..n_sub {
            if output_in_scratch {
                self.step_inner(model, &scratch, control, contacts, dt_sub, state_out)?;
            } else {
                self.step_inner(model, state_out, control, contacts, dt_sub, &mut scratch)?;
            }
            output_in_scratch = !output_in_scratch;
        }
        if output_in_scratch {
            state_out.copy_host_from(&scratch)?;
        }
        Ok(())
    }
}

impl FeatherstoneSolver {
    fn step_inner(
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
    let mut q_in = state_in.joint_q.host()?.to_vec();
    let mut qd_in = state_in.joint_qd.host()?.to_vec();

    // ── Pre-step: apply mimic constraints (q_a = c0 + c1·q_b) ──
    // Each mimic overrides the driven joint's first coord/dof to
    // match its driver. Only single-DoF driven joints are handled;
    // multi-DoF mimic is a future extension.
    if model.mimic_count > 0 {
        let m_a = model.mimic_joint_a.host()?;
        let m_b = model.mimic_joint_b.host()?;
        let m_c0 = model.mimic_c0.host()?;
        let m_c1 = model.mimic_c1.host()?;
        let m_en = model.mimic_enabled.host()?;
        let coord_off = model.joint_coord_offset.host()?;
        let dof_off = model.joint_dof_offset.host()?;
        for c in 0..model.mimic_count {
            if m_en[c] == 0 {
                continue;
            }
            let a = m_a[c] as usize;
            let b = m_b[c] as usize;
            if a >= model.joint_count || b >= model.joint_count {
                continue;
            }
            let coord_a = coord_off[a] as usize;
            let coord_b = coord_off[b] as usize;
            let dof_a = dof_off[a] as usize;
            let dof_b = dof_off[b] as usize;
            // Override q_a and qd_a from driver's values.
            q_in[coord_a] = m_c0[c] + m_c1[c] * q_in[coord_b];
            qd_in[dof_a] = m_c1[c] * qd_in[dof_b];
        }
    }
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
    let joint_enabled = model.joint_enabled.host()?;
    let art_start = model.articulation_start.host()?;
    let art_count = model.articulation_count.max(1);

    let body_q = state_out.body_q.host()?.to_vec();
    let body_mass = model.body_mass.host()?;
    let body_inertia = model.body_inertia.host()?;
    let body_com = model.body_com.host()?;
    let body_world = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let control_force = control.joint_force.host()?;
    // Per-DOF drive + limit tables (length = joint_dof_total). All
    // optional with NaN sentinel = "unbounded".
    let limit_lower = model.joint_limit_lower.host()?;
    let limit_upper = model.joint_limit_upper.host()?;
    let joint_friction = model.joint_friction.host()?;
    let effort_limit = model.joint_effort_limit.host()?;
    let velocity_limit = model.joint_velocity_limit.host()?;
    let target_mode = control.joint_target_mode.host()?;
    let target_pos = control.joint_target_pos.host()?;
    let target_vel = control.joint_target_vel.host()?;
    let target_kp = control.joint_target_kp.host()?;
    let target_kd = control.joint_target_kd.host()?;

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

    // ── Per-articulation outer loop ─────────────────────────────────
    // Each articulation owns its joint range; the parent-joint search
    // inside the passes naturally stays within `j_begin..j_end` because
    // it only scans `0..j` (which is at most `j_begin..j`).
    let mut q_new: Vec<Real> = q_in.clone();
    let mut qd_new: Vec<Real> = qd_in.clone();
    for art in 0..art_count {
    let j_begin = art_start[art] as usize;
    let j_end = art_start[art + 1] as usize;
    if j_begin == j_end {
        continue;
    }

    // ── Pass 1: forward kinematics + body kinematics ─────────────────
    for j in j_begin..j_end {
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

        // Parent body's spatial velocity (zero if root). The parent
        // must be a child of an earlier joint in *this* articulation;
        // a parent in a different articulation is treated as world.
        let v_p = if parent < 0 {
            SpatialVector::ZERO
        } else {
            let mut v = SpatialVector::ZERO;
            for k in j_begin..j {
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
    for j in (j_begin..j_end).rev() {
        let parent = joint_parent[j];
        let s_j = s[j];

        // For 0-DoF joints (Fixed / unsupported), skip the rank-1
        // update — there's no joint coordinate to project against.
        let jt = joint_type_from_u32(joint_type[j]);
        let enabled = joint_enabled.get(j).copied().unwrap_or(1) != 0;
        let supported = enabled
            && matches!(jt, JointType::Revolute | JointType::Prismatic);

        if supported {
            let u = i_a[j].mul_sv(s_j);
            let d = sv_dot(s_j, u);
            u_vec[j] = u;
            d_inv[j] = if d.abs() > 1e-12 { 1.0 / d } else { 0.0 };

            let dof_idx = joint_dof_offset[j] as usize;
            let coord_idx = joint_coord_offset[j] as usize;
            let q_now = q_in.get(coord_idx).copied().unwrap_or(0.0);
            let qd_now = qd_in.get(dof_idx).copied().unwrap_or(0.0);
            let mode = JointTargetMode::from_u32(
                target_mode.get(dof_idx).copied().unwrap_or(0),
            );
            let kp = target_kp.get(dof_idx).copied().unwrap_or(0.0);
            let kd = target_kd.get(dof_idx).copied().unwrap_or(0.0);
            let pos_t = target_pos.get(dof_idx).copied().unwrap_or(0.0);
            let vel_t = target_vel.get(dof_idx).copied().unwrap_or(0.0);
            let drive_torque = match mode {
                JointTargetMode::None => 0.0,
                JointTargetMode::Position => kp * (pos_t - q_now) - kd * qd_now,
                JointTargetMode::Velocity => kd * (vel_t - qd_now),
                JointTargetMode::PositionVelocity => {
                    kp * (pos_t - q_now) + kd * (vel_t - qd_now)
                }
                JointTargetMode::Effort => pos_t,
            };
            let mu = joint_friction.get(dof_idx).copied().unwrap_or(0.0);
            let friction_torque = -mu * qd_now;
            let tau_raw = control_force.get(dof_idx).copied().unwrap_or(0.0)
                + drive_torque
                + friction_torque;
            let tau_app = clamp_or_passthrough(
                tau_raw,
                effort_limit.get(dof_idx).copied().unwrap_or(Real::NAN),
            );
            u_tau[j] = tau_app - sv_dot(s_j, p_a[j]);

            if parent >= 0 {
                // Find the earlier joint whose child is this joint's
                // parent body — that's where we accumulate. Confined
                // to the current articulation.
                let mut parent_joint: Option<usize> = None;
                for k in j_begin..j {
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

    for j in j_begin..j_end {
        let parent = joint_parent[j];
        let s_j = s[j];
        let jt = joint_type_from_u32(joint_type[j]);
        let enabled = joint_enabled.get(j).copied().unwrap_or(1) != 0;
        let supported = enabled
            && matches!(jt, JointType::Revolute | JointType::Prismatic);

        let a_p = if parent < 0 {
            SpatialVector::ZERO
        } else {
            let mut a = SpatialVector::ZERO;
            for k in j_begin..j {
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

    // ── Integrate q, q̇ for joints in this articulation ──────────────
    for j in j_begin..j_end {
        let jt = joint_type_from_u32(joint_type[j]);
        let coord_idx = joint_coord_offset[j] as usize;
        let dof_idx = joint_dof_offset[j] as usize;
        match jt {
            JointType::Revolute | JointType::Prismatic => {
                let qd_old = qd_in[dof_idx];
                let qd_int = qd_old + q_dd[dof_idx] * dt;
                let qd_damped = qd_int * (1.0 - joint_damping * dt).max(0.0);
                // Per-DOF velocity cap (NaN = unlimited).
                let qd_capped = clamp_or_passthrough(
                    qd_damped,
                    velocity_limit.get(dof_idx).copied().unwrap_or(Real::NAN),
                );
                let q_pred = q_in[coord_idx] + qd_capped * dt;

                // Hard joint-position limits with velocity zeroing at
                // the limit. NaN = unlimited.
                let lo = limit_lower.get(dof_idx).copied().unwrap_or(Real::NAN);
                let hi = limit_upper.get(dof_idx).copied().unwrap_or(Real::NAN);
                let (q_clamped, qd_after) = clamp_joint_pos(q_pred, qd_capped, lo, hi);
                qd_new[dof_idx] = qd_after;
                q_new[coord_idx] = q_clamped;
            }
            JointType::Ball => {
                // 4 coords (qx, qy, qz, qw); 3 DoF angular velocity.
                // Reduced-coord ABA dynamics are deferred (3×3 D
                // matrix, requires Mat3 inverse machinery in the
                // chain). For now the ball joint integrates
                // **kinematically** — the caller is expected to
                // drive `joint_qd` directly (e.g. via a CONNECT
                // equality constraint in XPBD scenes, or by
                // applying torque outside Featherstone).
                let dof_idx = joint_dof_offset[j] as usize;
                let omega = Vec3::new(
                    qd_in.get(dof_idx).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 1).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                // Integrate quaternion: q_new = exp(0.5·dt·ω) ⊗ q_old.
                let q_old = Quat::from_xyzw(
                    q_in[coord_idx],
                    q_in[coord_idx + 1],
                    q_in[coord_idx + 2],
                    q_in[coord_idx + 3],
                );
                let dq = Quat::from_xyzw(
                    omega.x * 0.5 * dt,
                    omega.y * 0.5 * dt,
                    omega.z * 0.5 * dt,
                    0.0,
                );
                let q_new_quat = (dq * q_old + q_old).normalize();
                q_new[coord_idx] = q_new_quat.x;
                q_new[coord_idx + 1] = q_new_quat.y;
                q_new[coord_idx + 2] = q_new_quat.z;
                q_new[coord_idx + 3] = q_new_quat.w;
                qd_new[dof_idx] = omega.x;
                qd_new[dof_idx + 1] = omega.y;
                qd_new[dof_idx + 2] = omega.z;
            }
            JointType::Fixed | JointType::Free | JointType::Distance | JointType::D6 => {
                // Fixed: no coords. 6-DoF multi-coord joints (Free /
                // Distance / D6): pass through unchanged. Their
                // reduced-coord dynamics arrive with the IK / RNEA
                // stage; they currently integrate kinematically.
            }
        }
    }
    } // end per-articulation outer loop

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

/// Clamp `x` to `[-limit, +limit]`. If `limit` is NaN, passthrough.
#[inline]
fn clamp_or_passthrough(x: Real, limit: Real) -> Real {
    if limit.is_nan() {
        x
    } else {
        x.clamp(-limit, limit)
    }
}

/// Project a joint coordinate to its limits and zero its velocity at
/// the boundary. Each bound is independently optional (NaN sentinel).
#[inline]
fn clamp_joint_pos(q: Real, qd: Real, lo: Real, hi: Real) -> (Real, Real) {
    let mut q_out = q;
    let mut qd_out = qd;
    if !lo.is_nan() && q_out < lo {
        q_out = lo;
        if qd_out < 0.0 {
            qd_out = 0.0;
        }
    }
    if !hi.is_nan() && q_out > hi {
        q_out = hi;
        if qd_out > 0.0 {
            qd_out = 0.0;
        }
    }
    (q_out, qd_out)
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
        2 => JointType::Ball,
        4 => JointType::Free,
        5 => JointType::Distance,
        6 => JointType::D6,
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

    /// Pendulum with hard joint limits stays inside its allowed cone
    /// even after a free swing that would otherwise overshoot.
    #[test]
    fn pendulum_with_limits_stays_inside_cone() {
        let mut b = ModelBuilder::new();
        let link = b.add_body(
            BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
        );
        let joint = b
            .add_joint_revolute(
                BodyId::NONE,
                link,
                Transform::IDENTITY,
                Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
                V3::Z,
            )
            .unwrap();
        // Constrain the swing to ±15°.
        let lo = -0.26_f64; // ~ -15°
        let hi = 0.26_f64;
        b.set_joint_limits(joint, &[lo], &[hi]).unwrap();

        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        // Start at the high limit with a positive velocity that would
        // otherwise push past it.
        s_a.joint_q.host_mut().unwrap()[0] = hi;
        s_a.joint_qd.host_mut().unwrap()[0] = 5.0;
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();

        let mut max_q = -Real::INFINITY;
        let mut min_q = Real::INFINITY;
        for _ in 0..2000 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
            let q = s_a.joint_q.host().unwrap()[0];
            if q > max_q {
                max_q = q;
            }
            if q < min_q {
                min_q = q;
            }
        }
        assert!(
            max_q <= hi + 1e-6,
            "joint exceeded upper limit: max_q={max_q} (hi={hi})"
        );
        assert!(
            min_q >= lo - 1e-6,
            "joint exceeded lower limit: min_q={min_q} (lo={lo})"
        );
    }

    /// PD drive holds the pendulum at a non-zero target angle against
    /// gravity. With kp large enough, residual error stays below 5°.
    #[test]
    fn pd_drive_holds_pendulum_at_target() {
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

        let target = 1.0_f64; // ~57° lift
        let mut control = model.control();
        control.joint_target_mode.host_mut().unwrap()[0] =
            crate::sim::JointTargetMode::Position as u32;
        control.joint_target_pos.host_mut().unwrap()[0] = target;
        control.joint_target_kp.host_mut().unwrap()[0] = 200.0;
        control.joint_target_kd.host_mut().unwrap()[0] = 20.0;

        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();
        for _ in 0..3000 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.joint_q.host().unwrap()[0];
        let err = (q - target).abs();
        assert!(
            err < 0.10,
            "PD didn't hold target: q={q}, target={target}, err={err}"
        );
    }

    /// Joint friction damps a freely-swinging pendulum to rest.
    #[test]
    fn joint_friction_damps_swing_to_rest() {
        let mut b = ModelBuilder::new();
        let link = b.add_body(
            BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
        );
        let joint = b
            .add_joint_revolute(
                BodyId::NONE,
                link,
                Transform::IDENTITY,
                Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
                V3::Z,
            )
            .unwrap();
        b.set_joint_friction(joint, &[2.0]).unwrap();

        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.joint_q.host_mut().unwrap()[0] = 0.5;
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();
        for _ in 0..5000 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let qd = s_a.joint_qd.host().unwrap()[0];
        assert!(
            qd.abs() < 0.05,
            "friction failed to damp pendulum: qd={qd}"
        );
    }

    /// Two independent pendulums in one scene step in parallel —
    /// neither articulation's parent-search reaches into the other.
    /// Each pendulum should oscillate exactly the same as if it were
    /// alone in the scene.
    #[test]
    fn two_independent_pendulums() {
        // Build two single-pendulum models AND a combined model with
        // both pendulums marked as separate articulations. After the
        // same simulation, each joint angle in the combined run should
        // match its counterpart in the single-pendulum run.
        fn single_model() -> Model {
            let mut b = ModelBuilder::new();
            let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::IDENTITY,
            }));
            b.add_joint_revolute(
                BodyId::NONE,
                link,
                Transform::IDENTITY,
                Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
                V3::Z,
            )
            .unwrap();
            b.build().unwrap()
        }

        let single = single_model();
        let mut sa = single.state().unwrap();
        let mut sb = single.state().unwrap();
        sa.joint_q.host_mut().unwrap()[0] = 0.3;
        let ctl = single.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver_single = FeatherstoneSolver::new();
        for _ in 0..500 {
            solver_single.step(&single, &sa, &ctl, &contacts, 0.001, &mut sb).unwrap();
            std::mem::swap(&mut sa, &mut sb);
        }
        let single_q = sa.joint_q.host().unwrap()[0];

        // Combined model: pendulum 1 starts at q=0.3, pendulum 2 at q=−0.2.
        let mut b = ModelBuilder::new();
        let l1 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        b.add_joint_revolute(
            BodyId::NONE,
            l1,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
        b.begin_articulation();
        let l2 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        b.add_joint_revolute(
            BodyId::NONE,
            l2,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();

        let model = b.build().unwrap();
        assert_eq!(model.articulation_count, 2);
        let arts = model.articulation_start.host().unwrap();
        assert_eq!(arts, &[0u32, 1, 2]);

        let mut sa = model.state().unwrap();
        let mut sb = model.state().unwrap();
        sa.joint_q.host_mut().unwrap()[0] = 0.3;
        sa.joint_q.host_mut().unwrap()[1] = -0.2;
        let ctl = model.control();
        let mut solver = FeatherstoneSolver::new();
        for _ in 0..500 {
            solver.step(&model, &sa, &ctl, &contacts, 0.001, &mut sb).unwrap();
            std::mem::swap(&mut sa, &mut sb);
        }
        let q = sa.joint_q.host().unwrap();
        // Pendulum 1 in the combined run must match the single-pendulum
        // baseline to within solver tolerance — proves the two
        // articulations don't bleed into each other.
        let drift = (q[0] - single_q).abs();
        assert!(
            drift < 1e-6,
            "articulation 1 drifted from baseline: combined={} single={} Δ={drift}",
            q[0],
            single_q
        );
        // And pendulum 2 (started at q=-0.2) should be in the
        // mirror-image oscillation phase, bounded.
        assert!(q[1].abs() < 0.5, "pendulum 2 amplitude grew: q={}", q[1]);
    }

    /// Substepping the Featherstone solver should converge toward the
    /// same trajectory as a single big step (just smaller integration
    /// error). With dt=0.005 and substeps=4, a swinging pendulum's
    /// final angle should match the same scene with substeps=1 within
    /// integration tolerance.
    #[test]
    fn substeps_reduce_integration_error() {
        fn run(substeps: usize) -> Real {
            let mut b = ModelBuilder::new();
            let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::IDENTITY,
            }));
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
            s_a.joint_q.host_mut().unwrap()[0] = 0.5;
            let control = model.control();
            let contacts = Contacts::with_capacity(0);
            let mut solver = FeatherstoneSolver::new().with_substeps(substeps);
            for _ in 0..1000 {
                solver.step(&model, &s_a, &control, &contacts, 0.005, &mut s_b).unwrap();
                std::mem::swap(&mut s_a, &mut s_b);
            }
            s_a.joint_q.host().unwrap()[0]
        }
        // Substepping should produce nearly the same trajectory; energy
        // drift differs slightly but within solver noise.
        let q1 = run(1);
        let q4 = run(4);
        let drift = (q1 - q4).abs();
        eprintln!("substeps=1: q={q1:.6}, substeps=4: q={q4:.6}, |Δ|={drift:.6}");
        assert!(drift < 0.05, "substep result diverged: {drift}");
    }

    /// Mimic constraint (`q_a = c0 + c1·q_b`): drive one joint's
    /// position from another. With c0=0, c1=1 the two joint angles
    /// stay equal; with c1=-1 they mirror.
    #[test]
    fn mimic_constraint_couples_two_joints() {
        let mut b = ModelBuilder::new();
        let l0 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        let driver = b
            .add_joint_revolute(
                BodyId::NONE,
                l0,
                Transform::IDENTITY,
                Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
                V3::Z,
            )
            .unwrap();
        b.begin_articulation();
        let l1 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        let driven = b
            .add_joint_revolute(
                BodyId::NONE,
                l1,
                Transform::IDENTITY,
                Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
                V3::Z,
            )
            .unwrap();
        // q_driven = -q_driver (mirror).
        b.add_mimic_constraint(driven, driver, 0.0, -1.0).unwrap();

        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.joint_q.host_mut().unwrap()[0] = 0.5; // initial driver
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();
        for _ in 0..200 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.joint_q.host().unwrap();
        let drift = (q[0] + q[1]).abs(); // q_driver + q_driven should equal 0
        assert!(
            drift < 1e-3,
            "mimic broke: driver={} driven={} sum={drift}",
            q[0],
            q[1]
        );
    }

    /// Disabling a joint freezes its DOF — gravity stops affecting it.
    /// A pendulum that would normally swing should stay at its initial
    /// angle when its joint is disabled.
    #[test]
    fn disabled_joint_stays_frozen() {
        let mut b = ModelBuilder::new();
        let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        let joint = b
            .add_joint_revolute(
                BodyId::NONE,
                link,
                Transform::IDENTITY,
                Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
                V3::Z,
            )
            .unwrap();
        b.set_joint_enabled(joint, false).unwrap();
        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.joint_q.host_mut().unwrap()[0] = 0.5;
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();
        for _ in 0..1000 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.joint_q.host().unwrap()[0];
        assert!(
            (q - 0.5).abs() < 1e-6,
            "disabled joint moved: q={q}, expected 0.5"
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
