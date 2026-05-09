//! Featherstone forward dynamics for articulated rigid bodies — full
//! Articulated Body Algorithm (ABA) per Featherstone (2008) ch. 7,
//! mirroring the structure of Newton's
//! `solvers/featherstone/kernels.py` (with spatial-vector primitives
//! ported from Warp's `_src/math.py` / `builtins.py`).
//!
//! Three passes over the chain:
//!
//! 1. **Forward kinematics + body kinematics** (root → leaf):
//!    transforms, body spatial velocity `v_i = v_p + Σ_k S_ik · q̇_ik`,
//!    Coriolis term `c_i = v_i × v_j`, body spatial inertia in world
//!    frame, bias force `p_i = v_i ×* (I_i · v_i) - f_grav_i`.
//! 2. **Articulated inertia accumulation** (leaf → root):
//!    For each DOF column `S_k` of the joint's motion subspace, apply
//!    a sequential rank-1 update `I^A := I^A − U_k U_k^T / D_k`. After
//!    processing all DOFs, propagate `I^A_p += I^a`, `p^A_p += p^a`
//!    to the parent. (Sequential per-column rank-1 updates handle the
//!    multi-DOF case correctly even when the D matrix is non-diagonal.)
//! 3. **Forward acceleration solve** (root → leaf):
//!    For each DOF: `q̈_k = (u_k − U_k · a_p) / D_k`, accumulating
//!    `a_i = a_p + Σ_k S_k · q̈_k + c_i`.
//!
//! Then symplectic Euler integrates `q̇` and `q`. Final FK pass
//! refreshes `body_q` for downstream consumers.
//!
//! Supported joint types: Revolute, Prismatic (1-DOF), Ball (3 angular
//! DOF), Free / Distance (6 DOF), D6 (6 DOF with per-axis locks via
//! `lower == upper == 0` sentinel). Fixed passes through.

use crate::core::{Error, Result};
use crate::math::{
    spatial_cross, spatial_cross_dual, Mat3, Quat, Real, SpatialMatrix6, SpatialVector,
    Transform, Vec3,
};
use crate::sim::{eval_fk, Contacts, Control, JointTargetMode, JointType, Model, State};

use super::Solver;

/// Maximum DOFs in any single joint — Free/Distance/D6 hit 6.
const MAX_JOINT_DOFS: usize = 6;

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

/// Forward-dynamics solver for articulated rigid bodies.
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

/// Build the per-DOF motion subspace (in world frame) for joint `j`.
/// Returns the count of effective DOFs (after lock filtering for D6)
/// and writes columns into `s_cols[0..count]`. Locked DOF indices are
/// recorded as `dof_active[k] = false` so the integrator skips them.
///
/// Conventions:
/// - Revolute: 1 column = (0; axis_world)
/// - Prismatic: 1 column = (axis_world; 0)
/// - Ball: 3 angular columns, joint_world rotation columns ex/ey/ez
/// - Free / Distance: 6 columns = ex/ey/ez (linear) + ex/ey/ez (angular)
///   in the joint-world frame
/// - D6: same as Free, but axes with `lower == upper == 0` (locked)
///   set their column to ZERO and `dof_active[k] = false`.
fn build_motion_subspace(
    jt: JointType,
    joint_world: Transform,
    axis_local: Vec3,
    dof_idx: usize,
    limit_lower: &[Real],
    limit_upper: &[Real],
    s_cols: &mut [SpatialVector; MAX_JOINT_DOFS],
    dof_active: &mut [bool; MAX_JOINT_DOFS],
) -> usize {
    for k in 0..MAX_JOINT_DOFS {
        s_cols[k] = SpatialVector::ZERO;
        dof_active[k] = false;
    }
    let r = joint_world.rotation;
    let ex = (r * Vec3::new(1.0, 0.0, 0.0)).normalize_or_zero();
    let ey = (r * Vec3::new(0.0, 1.0, 0.0)).normalize_or_zero();
    let ez = (r * Vec3::new(0.0, 0.0, 1.0)).normalize_or_zero();
    match jt {
        JointType::Revolute => {
            let axis_world = (r * axis_local).normalize_or_zero();
            s_cols[0] = SpatialVector::new(Vec3::ZERO, axis_world);
            dof_active[0] = true;
            1
        }
        JointType::Prismatic => {
            let axis_world = (r * axis_local).normalize_or_zero();
            s_cols[0] = SpatialVector::new(axis_world, Vec3::ZERO);
            dof_active[0] = true;
            1
        }
        JointType::Ball => {
            s_cols[0] = SpatialVector::new(Vec3::ZERO, ex);
            s_cols[1] = SpatialVector::new(Vec3::ZERO, ey);
            s_cols[2] = SpatialVector::new(Vec3::ZERO, ez);
            dof_active[0] = true;
            dof_active[1] = true;
            dof_active[2] = true;
            3
        }
        JointType::Free | JointType::Distance => {
            // 3 linear + 3 angular columns in joint world frame.
            s_cols[0] = SpatialVector::new(ex, Vec3::ZERO);
            s_cols[1] = SpatialVector::new(ey, Vec3::ZERO);
            s_cols[2] = SpatialVector::new(ez, Vec3::ZERO);
            s_cols[3] = SpatialVector::new(Vec3::ZERO, ex);
            s_cols[4] = SpatialVector::new(Vec3::ZERO, ey);
            s_cols[5] = SpatialVector::new(Vec3::ZERO, ez);
            for k in 0..6 {
                dof_active[k] = true;
            }
            6
        }
        JointType::D6 => {
            // 6 columns; axes locked when lower == upper (== 0 conventionally).
            let columns = [
                SpatialVector::new(ex, Vec3::ZERO),
                SpatialVector::new(ey, Vec3::ZERO),
                SpatialVector::new(ez, Vec3::ZERO),
                SpatialVector::new(Vec3::ZERO, ex),
                SpatialVector::new(Vec3::ZERO, ey),
                SpatialVector::new(Vec3::ZERO, ez),
            ];
            for k in 0..6 {
                let lo = limit_lower.get(dof_idx + k).copied().unwrap_or(Real::NAN);
                let hi = limit_upper.get(dof_idx + k).copied().unwrap_or(Real::NAN);
                let locked = lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12;
                if locked {
                    s_cols[k] = SpatialVector::ZERO;
                    dof_active[k] = false;
                } else {
                    s_cols[k] = columns[k];
                    dof_active[k] = true;
                }
            }
            6
        }
        JointType::Fixed => 0,
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
    let body_f_in = state_in.body_f.host()?.to_vec();
    let body_mass = model.body_mass.host()?;
    let body_inertia = model.body_inertia.host()?;
    let body_com = model.body_com.host()?;
    let body_world = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let control_force = control.joint_force.host()?;
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
    let mut v_body = vec![SpatialVector::ZERO; n]; // body spatial velocity
    let mut c_term = vec![SpatialVector::ZERO; n]; // Coriolis bias
    let mut i_a = vec![SpatialMatrix6::ZERO; n]; // articulated inertia
    let mut p_a = vec![SpatialVector::ZERO; n]; // articulated bias force
    // Per-joint motion-subspace columns + active mask + DOF count.
    let mut s_cols = vec![[SpatialVector::ZERO; MAX_JOINT_DOFS]; n];
    let mut dof_active = vec![[false; MAX_JOINT_DOFS]; n];
    let mut dof_count = vec![0usize; n];
    // Per-DOF accumulators (column-by-column).
    let mut u_per_dof = vec![[SpatialVector::ZERO; MAX_JOINT_DOFS]; n];
    let mut d_inv_per_dof = vec![[0.0_f64; MAX_JOINT_DOFS]; n];
    let mut u_tau_per_dof = vec![[0.0_f64; MAX_JOINT_DOFS]; n];

    // ── Per-articulation outer loop ─────────────────────────────────
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

        // Build motion subspace (per-DOF).
        let dof_idx = joint_dof_offset[j] as usize;
        let n_dof = build_motion_subspace(
            jt,
            joint_world,
            joint_axis[j],
            dof_idx,
            limit_lower,
            limit_upper,
            &mut s_cols[j],
            &mut dof_active[j],
        );
        dof_count[j] = n_dof;

        // Joint velocity = sum of S_k · qd_k.
        let mut v_j = SpatialVector::ZERO;
        for k in 0..n_dof {
            if !dof_active[j][k] {
                continue;
            }
            let qd_k = qd_in.get(dof_idx + k).copied().unwrap_or(0.0);
            v_j = v_j + s_cols[j][k].scale(qd_k);
        }

        // Parent body's spatial velocity.
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

        let v_i = v_p + v_j;
        v_body[j] = v_i;
        c_term[j] = spatial_cross(v_i, v_j);

        // Body spatial inertia in WORLD frame, expressed at world origin.
        let cw = body_q[child];
        let com_world = cw.position + cw.rotation * body_com[child];
        let i_local_world = rotate_inertia(cw.rotation, body_inertia[child]);
        let i_s = SpatialMatrix6::from_body(body_mass[child], com_world, i_local_world);

        // Initial articulated inertia and bias.
        // Bias = v ×* (I·v) − f_grav − f_ext, where f_ext is the
        // user-applied wrench at COM (state.body_f, world frame),
        // re-expressed at the world origin.
        let world_idx = body_world[child].max(0) as usize;
        let g = gravity[world_idx];
        let f_g_lin = g * body_mass[child];
        let f_g_ang = com_world.cross(f_g_lin);
        let f_grav = SpatialVector::new(f_g_lin, f_g_ang);
        // External wrench at COM: caller fills `body_f[child]` with
        // (force, torque_at_COM) in world frame. Re-express at world
        // origin: linear unchanged, angular gets +r_com × F.
        let ext_com = body_f_in.get(child).copied().unwrap_or(SpatialVector::ZERO);
        let f_ext_lin = ext_com.linear;
        let f_ext_ang = ext_com.angular + com_world.cross(ext_com.linear);
        let f_ext = SpatialVector::new(f_ext_lin, f_ext_ang);
        let bias = spatial_cross_dual(v_i, i_s.mul_sv(v_i));
        let p_init = bias - f_grav - f_ext;
        i_a[j] = i_s;
        p_a[j] = p_init;
    }

    // ── Pass 2: backward — articulated inertia accumulation ──────────
    for j in (j_begin..j_end).rev() {
        let parent = joint_parent[j];
        let jt = joint_type_from_u32(joint_type[j]);
        let enabled = joint_enabled.get(j).copied().unwrap_or(1) != 0;
        let n_dof = dof_count[j];
        let supported = enabled && !matches!(jt, JointType::Fixed);

        // Snapshot the joint's spatial inertia for the parent
        // accumulation. We'll deflate it column-by-column.
        let mut i_curr = i_a[j];
        let dof_idx = joint_dof_offset[j] as usize;
        let coord_idx = joint_coord_offset[j] as usize;

        if supported && n_dof > 0 {
            // Sequential rank-1 updates: for each DOF k in 0..n_dof,
            //   U_k = i_curr · S_k
            //   D_k = S_k^T · U_k
            //   u_tau_k = tau_k - S_k^T · p_a[j]
            //   then deflate i_curr := i_curr - U_k · U_k^T / D_k
            // Drive / friction / effort are accumulated per DOF.
            for k in 0..n_dof {
                if !dof_active[j][k] {
                    continue;
                }
                let s_k = s_cols[j][k];
                let u_k = i_curr.mul_sv(s_k);
                let d_k = sv_dot(s_k, u_k);
                let d_inv_k = if d_k.abs() > 1e-12 { 1.0 / d_k } else { 0.0 };
                u_per_dof[j][k] = u_k;
                d_inv_per_dof[j][k] = d_inv_k;

                // Drive / friction / effort. Position-based drives only
                // make sense for 1-DOF coords; for ball/free/d6 we use
                // velocity-based drives (kp ignored).
                let dof_g = dof_idx + k;
                let qd_now = qd_in.get(dof_g).copied().unwrap_or(0.0);
                let q_now = match jt {
                    JointType::Revolute | JointType::Prismatic => {
                        q_in.get(coord_idx + k).copied().unwrap_or(0.0)
                    }
                    _ => 0.0, // multi-DOF coords aren't direct positions
                };
                let mode = JointTargetMode::from_u32(
                    target_mode.get(dof_g).copied().unwrap_or(0),
                );
                let kp = target_kp.get(dof_g).copied().unwrap_or(0.0);
                let kd = target_kd.get(dof_g).copied().unwrap_or(0.0);
                let pos_t = target_pos.get(dof_g).copied().unwrap_or(0.0);
                let vel_t = target_vel.get(dof_g).copied().unwrap_or(0.0);
                let drive_torque = match mode {
                    JointTargetMode::None => 0.0,
                    JointTargetMode::Position => kp * (pos_t - q_now) - kd * qd_now,
                    JointTargetMode::Velocity => kd * (vel_t - qd_now),
                    JointTargetMode::PositionVelocity => {
                        kp * (pos_t - q_now) + kd * (vel_t - qd_now)
                    }
                    JointTargetMode::Effort => pos_t,
                };
                let mu = joint_friction.get(dof_g).copied().unwrap_or(0.0);
                let friction_torque = -mu * qd_now;
                let tau_raw = control_force.get(dof_g).copied().unwrap_or(0.0)
                    + drive_torque
                    + friction_torque;
                let tau_app = clamp_or_passthrough(
                    tau_raw,
                    effort_limit.get(dof_g).copied().unwrap_or(Real::NAN),
                );
                u_tau_per_dof[j][k] = tau_app - sv_dot(s_k, p_a[j]);

                // Deflate i_curr for next column.
                let outer = SpatialMatrix6::outer(u_k, u_k).scale(d_inv_k);
                i_curr = i_curr.sub(&outer);
            }

            // After deflation, i_curr is the articulated inertia to
            // hand to the parent: I^a = I^A - Σ_k U_k U_k^T / D_k.
            // Bias propagation:
            //   p^a = p^A + I^a · c + Σ_k U_k · (u_tau_k / D_k)
            let mut p_articulated = p_a[j] + i_curr.mul_sv(c_term[j]);
            for k in 0..n_dof {
                if !dof_active[j][k] {
                    continue;
                }
                p_articulated = p_articulated
                    + u_per_dof[j][k].scale(u_tau_per_dof[j][k] * d_inv_per_dof[j][k]);
            }

            if parent >= 0 {
                let mut parent_joint: Option<usize> = None;
                for k in j_begin..j {
                    if joint_child[k] as usize == parent as usize {
                        parent_joint = Some(k);
                        break;
                    }
                }
                if let Some(pj) = parent_joint {
                    i_a[pj] = i_a[pj].add(&i_curr);
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
        let jt = joint_type_from_u32(joint_type[j]);
        let enabled = joint_enabled.get(j).copied().unwrap_or(1) != 0;
        let n_dof = dof_count[j];
        let supported = enabled && !matches!(jt, JointType::Fixed);

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

        if supported && n_dof > 0 {
            let dof_idx = joint_dof_offset[j] as usize;
            let mut s_qdd = SpatialVector::ZERO;
            for k in 0..n_dof {
                if !dof_active[j][k] {
                    continue;
                }
                let q_dd_k =
                    (u_tau_per_dof[j][k] - sv_dot(u_per_dof[j][k], a_p)) * d_inv_per_dof[j][k];
                q_dd[dof_idx + k] = q_dd_k;
                s_qdd = s_qdd + s_cols[j][k].scale(q_dd_k);
            }
            a_body[j] = a_p + s_qdd + c_term[j];
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
                let qd_capped = clamp_or_passthrough(
                    qd_damped,
                    velocity_limit.get(dof_idx).copied().unwrap_or(Real::NAN),
                );
                let q_pred = q_in[coord_idx] + qd_capped * dt;
                let lo = limit_lower.get(dof_idx).copied().unwrap_or(Real::NAN);
                let hi = limit_upper.get(dof_idx).copied().unwrap_or(Real::NAN);
                let (q_clamped, qd_after) = clamp_joint_pos(q_pred, qd_capped, lo, hi);
                qd_new[dof_idx] = qd_after;
                q_new[coord_idx] = q_clamped;
            }
            JointType::Ball => {
                // 4 coords (qx, qy, qz, qw) + 3 dof (ω in joint frame).
                // Symplectic Euler with quaternion integration.
                let mut omega = Vec3::new(
                    qd_in.get(dof_idx).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 1).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                let alpha = Vec3::new(
                    q_dd.get(dof_idx).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 1).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                omega = omega + alpha * dt;
                let damp = (1.0 - joint_damping * dt).max(0.0);
                omega = omega * damp;
                let q_old = Quat::from_xyzw(
                    q_in[coord_idx],
                    q_in[coord_idx + 1],
                    q_in[coord_idx + 2],
                    q_in[coord_idx + 3],
                );
                let dq = Quat::from_xyzw(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5, 0.0);
                let q_dot = dq * q_old;
                let q_pred = Quat::from_xyzw(
                    q_old.x + q_dot.x * dt,
                    q_old.y + q_dot.y * dt,
                    q_old.z + q_dot.z * dt,
                    q_old.w + q_dot.w * dt,
                )
                .normalize();
                q_new[coord_idx] = q_pred.x;
                q_new[coord_idx + 1] = q_pred.y;
                q_new[coord_idx + 2] = q_pred.z;
                q_new[coord_idx + 3] = q_pred.w;
                qd_new[dof_idx] = omega.x;
                qd_new[dof_idx + 1] = omega.y;
                qd_new[dof_idx + 2] = omega.z;
            }
            JointType::Free | JointType::Distance | JointType::D6 => {
                // 7 coords [tx, ty, tz, qx, qy, qz, qw], 6 dof
                // [vx, vy, vz, ωx, ωy, ωz] expressed in joint world frame.
                let mut v = Vec3::new(
                    qd_in.get(dof_idx).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 1).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                let mut omega = Vec3::new(
                    qd_in.get(dof_idx + 3).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 4).copied().unwrap_or(0.0),
                    qd_in.get(dof_idx + 5).copied().unwrap_or(0.0),
                );
                let a_lin = Vec3::new(
                    q_dd.get(dof_idx).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 1).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 2).copied().unwrap_or(0.0),
                );
                let alpha = Vec3::new(
                    q_dd.get(dof_idx + 3).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 4).copied().unwrap_or(0.0),
                    q_dd.get(dof_idx + 5).copied().unwrap_or(0.0),
                );
                v = v + a_lin * dt;
                omega = omega + alpha * dt;
                let damp = (1.0 - joint_damping * dt).max(0.0);
                v = v * damp;
                omega = omega * damp;

                let p_old = Vec3::new(q_in[coord_idx], q_in[coord_idx + 1], q_in[coord_idx + 2]);
                let p_new = p_old + v * dt;
                let r_old = Quat::from_xyzw(
                    q_in[coord_idx + 3],
                    q_in[coord_idx + 4],
                    q_in[coord_idx + 5],
                    q_in[coord_idx + 6],
                );
                let dq = Quat::from_xyzw(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5, 0.0);
                let r_dot = dq * r_old;
                let r_new = Quat::from_xyzw(
                    r_old.x + r_dot.x * dt,
                    r_old.y + r_dot.y * dt,
                    r_old.z + r_dot.z * dt,
                    r_old.w + r_dot.w * dt,
                )
                .normalize();
                q_new[coord_idx] = p_new.x;
                q_new[coord_idx + 1] = p_new.y;
                q_new[coord_idx + 2] = p_new.z;
                q_new[coord_idx + 3] = r_new.x;
                q_new[coord_idx + 4] = r_new.y;
                q_new[coord_idx + 5] = r_new.z;
                q_new[coord_idx + 6] = r_new.w;
                qd_new[dof_idx] = v.x;
                qd_new[dof_idx + 1] = v.y;
                qd_new[dof_idx + 2] = v.z;
                qd_new[dof_idx + 3] = omega.x;
                qd_new[dof_idx + 4] = omega.y;
                qd_new[dof_idx + 5] = omega.z;

                // Distance hard-projection: clamp |p_new| to
                // [lower[0], upper[0]] (using DOF 0's limits as a
                // distance limit). NaN sentinel = unlimited.
                if matches!(jt, JointType::Distance) {
                    let lo = limit_lower.get(dof_idx).copied().unwrap_or(Real::NAN);
                    let hi = limit_upper.get(dof_idx).copied().unwrap_or(Real::NAN);
                    let p_curr = Vec3::new(
                        q_new[coord_idx],
                        q_new[coord_idx + 1],
                        q_new[coord_idx + 2],
                    );
                    let dist = p_curr.length();
                    let mut clamped = p_curr;
                    let mut zero_radial_v = false;
                    if !lo.is_nan() && dist < lo && dist > 1e-12 {
                        clamped = p_curr * (lo / dist);
                        zero_radial_v = true;
                    }
                    if !hi.is_nan() && dist > hi && dist > 1e-12 {
                        clamped = p_curr * (hi / dist);
                        zero_radial_v = true;
                    }
                    if zero_radial_v {
                        q_new[coord_idx] = clamped.x;
                        q_new[coord_idx + 1] = clamped.y;
                        q_new[coord_idx + 2] = clamped.z;
                        // Strip radial velocity component.
                        let n = clamped.normalize_or_zero();
                        let radial = n * v.dot(n);
                        let v_proj = v - radial;
                        qd_new[dof_idx] = v_proj.x;
                        qd_new[dof_idx + 1] = v_proj.y;
                        qd_new[dof_idx + 2] = v_proj.z;
                    }
                }

                // D6 per-axis position clamping for un-locked DOFs.
                if matches!(jt, JointType::D6) {
                    // Only the first 3 (linear) DOFs map directly to
                    // coord positions in this layout; angular limits
                    // would need axis-angle decomposition (deferred).
                    for k in 0..3 {
                        let lo = limit_lower.get(dof_idx + k).copied().unwrap_or(Real::NAN);
                        let hi = limit_upper.get(dof_idx + k).copied().unwrap_or(Real::NAN);
                        let p_k = q_new[coord_idx + k];
                        let qd_k = qd_new[dof_idx + k];
                        let (p_c, qd_c) = clamp_joint_pos(p_k, qd_k, lo, hi);
                        q_new[coord_idx + k] = p_c;
                        qd_new[dof_idx + k] = qd_c;
                    }
                }
            }
            JointType::Fixed => {
                // No coords / no dof — pass through.
            }
        }
    }
    } // end per-articulation outer loop

    state_out.joint_q.host_mut()?.copy_from_slice(&q_new);
    state_out.joint_qd.host_mut()?.copy_from_slice(&qd_new);
    eval_fk(model, state_out)?;

    // Propagate input body_f to output so substepping (which swaps
    // state_a / state_b each iteration) sees the same external wrench
    // each substep until the caller overwrites it.
    state_out
        .body_f
        .host_mut()?
        .copy_from_slice(&body_f_in);
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
        s_a.joint_q.host_mut().unwrap()[0] = 0.1;

        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();

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
    /// is actually wired.
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
        let n_steps = 5000usize;
        let mut max_energy_excess = 0.0_f64;
        for _ in 0..n_steps {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);

            let body_q = s_a.body_q.host().unwrap();
            let body_qd = s_a.body_qd.host().unwrap();
            let mut t_energy = 0.0_f64;
            let mut v_energy = 0.0_f64;
            for (i, q) in body_q.iter().enumerate() {
                let v_com = body_qd[i].linear;
                t_energy += 0.5 * 1.0 * v_com.length_squared();
                t_energy += 0.5 * 0.083 * body_qd[i].angular.length_squared();
                v_energy += q.position.y * 1.0 * g_mag;
            }
            let e = t_energy + v_energy;
            let excess = (e - v_init).abs();
            if excess > max_energy_excess {
                max_energy_excess = excess;
            }
        }
        assert!(
            max_energy_excess < 0.5 * v_init.abs().max(1.0),
            "double pendulum energy drift too high: {max_energy_excess} > 50% of |E_0|={}",
            v_init.abs()
        );
    }

    /// Pendulum with hard joint limits stays inside its allowed cone.
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
        let lo = -0.26_f64;
        let hi = 0.26_f64;
        b.set_joint_limits(joint, &[lo], &[hi]).unwrap();

        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
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

    /// PD drive holds the pendulum at a non-zero target angle against gravity.
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

        let target = 1.0_f64;
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
    #[test]
    fn two_independent_pendulums() {
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
        let drift = (q[0] - single_q).abs();
        assert!(
            drift < 1e-6,
            "articulation 1 drifted from baseline: combined={} single={} Δ={drift}",
            q[0],
            single_q
        );
        assert!(q[1].abs() < 0.5, "pendulum 2 amplitude grew: q={}", q[1]);
    }

    /// Substepping the Featherstone solver should converge toward the
    /// same trajectory as a single big step.
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
        let q1 = run(1);
        let q4 = run(4);
        let drift = (q1 - q4).abs();
        eprintln!("substeps=1: q={q1:.6}, substeps=4: q={q4:.6}, |Δ|={drift:.6}");
        assert!(drift < 0.05, "substep result diverged: {drift}");
    }

    /// Mimic constraint (`q_a = c0 + c1·q_b`).
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
        b.add_mimic_constraint(driven, driver, 0.0, -1.0).unwrap();

        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.joint_q.host_mut().unwrap()[0] = 0.5;
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
        let drift = (q[0] + q[1]).abs();
        assert!(
            drift < 1e-3,
            "mimic broke: driver={} driven={} sum={drift}",
            q[0],
            q[1]
        );
    }

    /// Disabling a joint freezes its DOF.
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

    /// Pendulum with no initial energy + no torque stays at q=0.
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

    // ── Stage 17 multi-DOF tests ─────────────────────────────────────

    /// Free-floating body falls under gravity at g (CPU baseline).
    #[test]
    fn free_joint_body_falls_at_g() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.1,
        }));
        b.add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();

        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();

        let dt = 0.001_f64;
        let n_steps = 100;
        for _ in 0..n_steps {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let t = dt * n_steps as f64;
        let expected_y = -0.5 * crate::math::GRAVITY * t * t;
        let actual_y = s_a.joint_q.host().unwrap()[1];
        let err = (actual_y - expected_y).abs();
        assert!(
            err < 0.01,
            "free body fall: expected y={expected_y:.4}, got {actual_y:.4} (Δ={err:.4})"
        );
    }

    /// Spherical pendulum (BALL joint) — body hangs from a ball joint
    /// and swings under gravity. Energy must stay bounded.
    #[test]
    fn ball_joint_spherical_pendulum_energy_bounded() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.05,
        }));
        // Joint at world origin, child anchor +0.5 above so the body
        // hangs at -0.5 in -Y at q=identity.
        b.add_joint_ball(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
        )
        .unwrap();
        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        // Tilt the pendulum 30° about world +Z so it starts at an
        // off-vertical angle.
        let angle = 0.3_f64;
        let tilt = Quat::from_axis_angle(V3::Z, angle);
        let q = s_a.joint_q.host_mut().unwrap();
        q[0] = tilt.x;
        q[1] = tilt.y;
        q[2] = tilt.z;
        q[3] = tilt.w;
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new().with_damping(0.1);

        let dt = 0.001_f64;
        let mut max_y = -Real::INFINITY;
        for _ in 0..2000 {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
            let y = s_a.body_q.host().unwrap()[0].position.y;
            if y > max_y {
                max_y = y;
            }
        }
        // Started at y = -0.5·cos(0.3) ≈ -0.477. Without energy
        // injection, swing should stay below ~+0.0 (bouncing past the
        // horizontal would require energy gain).
        assert!(
            max_y < 0.05,
            "ball pendulum gained energy: max_y={max_y:.4}"
        );
    }

    /// External wrench at COM (`state.body_f`) cancels gravity for a
    /// free body when `f_ext = m·g·ŷ`.
    #[test]
    fn body_f_external_wrench_cancels_gravity() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 2.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.1,
        }));
        b.add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        // Apply +m·g upward to cancel gravity exactly.
        let f = V3::new(0.0, 2.0 * crate::math::GRAVITY, 0.0);
        s_a.body_f.host_mut().unwrap()[0] = SpatialVector::new(f, V3::ZERO);
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();
        for _ in 0..200 {
            // body_f is zeroed at end of each step; re-apply.
            s_a.body_f.host_mut().unwrap()[0] = SpatialVector::new(f, V3::ZERO);
            solver
                .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let y = s_a.joint_q.host().unwrap()[1];
        assert!(
            y.abs() < 1e-3,
            "body_f cancel failed: y drifted to {y}"
        );
    }

    /// D6 with all DOFs locked behaves like a fixed joint.
    #[test]
    fn d6_all_locked_behaves_like_fixed() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.1,
        }));
        let j = b
            .add_joint_d6(
                BodyId::NONE,
                body,
                Transform::IDENTITY,
                Transform::IDENTITY,
            )
            .unwrap();
        // Lock all 6 DOFs by setting lower==upper==0.
        b.set_joint_limits(j, &[0.0; 6], &[0.0; 6]).unwrap();
        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();

        for _ in 0..200 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        // With all linear DOFs locked, the body shouldn't fall.
        let pos = s_a.body_q.host().unwrap()[0].position;
        assert!(
            pos.length() < 1e-3,
            "locked D6 drifted: pos={pos:?}"
        );
    }
}
