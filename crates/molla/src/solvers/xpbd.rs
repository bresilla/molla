//! XPBD (eXtended Position-Based Dynamics) — position-projection
//! solver for rigid bodies with contacts.
//!
//! Stage 4.x implementation:
//! 1. Predict positions + rotations from velocities + gravity
//!    (semi-implicit-ish).
//! 2. **Position projection**: for every contact, push the two
//!    bodies apart along the normal until the recorded penetration
//!    is removed. **With angular response** — the contact point's
//!    lever arm `r = p_contact - body_com_world` produces an
//!    angular delta `Δθ = I⁻¹·(r × n) · λ` alongside the linear
//!    delta `Δp = ±n · (1/m) · λ`. Generalised inverse mass
//!    `w = 1/m + (r × n)ᵀ · I⁻¹ · (r × n)`.
//! 3. **Velocity recovery**: linear from `(p_new - p_old) / dt`,
//!    angular from the small-rotation extracted from the quat delta.
//! 4. **Velocity projection**: along each contact normal, replace
//!    the relative *contact-point* approach velocity by
//!    `-restitution · v_in`. The contact-point velocity includes the
//!    angular contribution `ω × r`.
//!
//! Friction: Coulomb tangential clamp on the velocity-projection
//! pass with the full angular Jacobian.
//!
//! Reference: Müller et al., "Detailed Rigid Body Simulation with
//! Extended Position Based Dynamics" (Eurographics 2020).

use crate::core::{Error, Result};
use crate::geometry::STATIC_BODY;
use crate::math::{Mat3, Quat, Real, SpatialVector, Transform, Vec3};
use crate::sim::{BodyFlags, Contacts, Control, Model, State};

use super::Solver;

/// Backend selector for [`XpbdSolver`].
#[allow(clippy::large_enum_variant, reason = "GPU variant holds Arcs / kernel; one solver instance, fine")]
pub enum XpbdBackend {
    Cpu,
    /// GPU dispatch via wgpu. Built once for a fixed `body_count`;
    /// rebuild the solver if the model changes.
    #[cfg(feature = "gpu")]
    Gpu(GpuBackend),
}

#[cfg(feature = "gpu")]
pub struct GpuBackend {
    pub(crate) device: std::sync::Arc<wgpu::Device>,
    pub(crate) queue: std::sync::Arc<wgpu::Queue>,
    pub(crate) kernel: crate::kernels::XpbdKernel,
    /// GPU-side equality-constraint projection — replaces the
    /// host-side `apply_equality_post_gpu` shim. Built lazily so
    /// solvers without equality constraints don't pay the pipeline
    /// compile cost.
    pub(crate) equality_kernel: crate::kernels::XpbdEqualityKernel,
}

/// Position-based dynamics solver.
pub struct XpbdSolver {
    /// Position-projection iterations per substep. Many iterations
    /// help when many contacts share a body (e.g. a stack), but the
    /// preferred way to scale is to **add substeps** (Müller 2020 §3.2).
    pub position_iterations: usize,
    /// Number of substeps per outer `step` call. The contact set
    /// from the caller is held fixed across all substeps, while
    /// each substep advances `dt/substeps`. Default 1.
    pub substeps: usize,
    backend: XpbdBackend,
}

impl Default for XpbdSolver {
    fn default() -> Self {
        Self::new_cpu()
    }
}

impl XpbdSolver {
    /// CPU backend (used by tests + as a fallback). The molla project
    /// is GPU-first — prefer [`Self::new_gpu`] for any user-facing
    /// scene.
    #[must_use]
    pub fn new_cpu() -> Self {
        Self {
            position_iterations: 4,
            substeps: 1,
            backend: XpbdBackend::Cpu,
        }
    }

    /// Backwards-compat shim. Same as [`Self::new_cpu`].
    #[must_use]
    pub fn new() -> Self {
        Self::new_cpu()
    }

    /// GPU backend. The kernel is built for `body_count` bodies; if
    /// the model later grows, rebuild the solver. Uploads are done by
    /// the caller (`Model::upload_to_device`, etc.).
    #[cfg(feature = "gpu")]
    pub fn new_gpu(engine: &crate::compute::Engine, body_count: u32) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("engine has no GPU queue".into()))?
            .clone();
        let kernel = crate::kernels::XpbdKernel::new(&device, body_count);
        let equality_kernel = crate::kernels::XpbdEqualityKernel::new(&device);
        Ok(Self {
            position_iterations: 4,
            substeps: 1,
            backend: XpbdBackend::Gpu(GpuBackend {
                device,
                queue,
                kernel,
                equality_kernel,
            }),
        })
    }

    #[must_use]
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.position_iterations = n;
        self
    }

    /// Configure substepping: each outer `step(dt)` runs `n` inner
    /// substeps with `dt/n`, **reusing the caller-supplied contact
    /// set across all substeps**.
    ///
    /// **Limitations**: because contacts aren't re-detected per
    /// substep, this is most useful for contact-free workloads. For
    /// the Müller (2020) "many small substeps × few iterations"
    /// stability win on contact-rich stacks, the caller should run
    /// the substep loop externally — `pipeline.collide` + `solver.step`
    /// inside an `n`-iteration inner loop, with `solver.substeps = 1`.
    /// See `tests/xpbd_boxes.rs::external_substeps_box_stack` for the
    /// recommended pattern.
    #[must_use]
    pub fn with_substeps(mut self, n: usize) -> Self {
        self.substeps = n.max(1);
        self
    }

    #[must_use]
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self.backend, XpbdBackend::Gpu(_))
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
}

impl Solver for XpbdSolver {
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
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

        let n_sub = self.substeps.max(1);
        let dt_sub = dt / n_sub as Real;

        // Single substep: no scratch buffer needed, no contact-pose
        // override — cpu_step uses state_in.body_q for p_a_local.
        if n_sub == 1 {
            return self.step_inner(model, state_in, control, contacts, dt_sub, state_out, None);
        }

        // Multi-substep: ping-pong between state_out and a host-side
        // scratch state. The contact set is reused across substeps —
        // Müller (2020) §3.2: contacts captured once at the outer
        // step boundary, refined by N short substeps. The body poses
        // **at contact-detection time** (= caller's state_in.body_q)
        // are passed to every substep so the per-contact body-frame
        // anchor `p_a_local` stays consistent across substeps.
        let q_at_contact = state_in.body_q.host()?.to_vec();
        let mut scratch = state_in.clone_host()?;
        self.step_inner(
            model,
            state_in,
            control,
            contacts,
            dt_sub,
            state_out,
            Some(&q_at_contact),
        )?;
        let mut output_in_scratch = false;
        for _ in 1..n_sub {
            if output_in_scratch {
                self.step_inner(
                    model,
                    &scratch,
                    control,
                    contacts,
                    dt_sub,
                    state_out,
                    Some(&q_at_contact),
                )?;
            } else {
                self.step_inner(
                    model,
                    state_out,
                    control,
                    contacts,
                    dt_sub,
                    &mut scratch,
                    Some(&q_at_contact),
                )?;
            }
            output_in_scratch = !output_in_scratch;
        }
        if output_in_scratch {
            state_out.copy_host_from(&scratch)?;
        }
        Ok(())
    }
}

impl XpbdSolver {
    fn step_inner(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
        q_at_contact: Option<&[Transform]>,
    ) -> Result<()> {
        match &mut self.backend {
            XpbdBackend::Cpu => cpu_step(
                model,
                state_in,
                control,
                contacts,
                dt,
                state_out,
                self.position_iterations,
                q_at_contact,
            ),
            #[cfg(feature = "gpu")]
            XpbdBackend::Gpu(gpu) => {
                // GPU substep with contact-relative pose override is
                // not implemented yet — fall back to using state_in's
                // body_q (works for substeps == 1).
                let _ = q_at_contact;
                let GpuBackend {
                    device,
                    queue,
                    kernel,
                    equality_kernel,
                } = gpu;
                kernel.dispatch(
                    device,
                    queue,
                    model,
                    state_in,
                    control,
                    contacts,
                    self.position_iterations,
                    dt,
                    state_out,
                )?;
                // Equality constraints (CONNECT / WELD) — full GPU
                // dispatch via `XpbdEqualityKernel`; modifies
                // `state_out.body_q` in place and recomputes
                // `state_out.body_qd` from the corrected pose.
                equality_kernel.dispatch(
                    device,
                    queue,
                    model,
                    state_in,
                    state_out,
                    self.position_iterations as u32,
                    dt as f32,
                )?;
                // Particles (cloth/rope/springs) likewise live on the
                // CPU side for now.
                apply_particles_post_gpu(
                    model,
                    state_in,
                    state_out,
                    dt,
                    self.position_iterations,
                    device,
                    queue,
                )
            }
        }
    }
}

fn cpu_step(
    model: &Model,
    state_in: &State,
    control: &Control,
    contacts: &Contacts,
    dt: Real,
    state_out: &mut State,
    position_iterations: usize,
    q_at_contact: Option<&[Transform]>,
) -> Result<()> {
    let n = model.body_count;
    let q_in = state_in.body_q.host()?;
    // Body poses at contact-detection time. Same as `q_in` in single-step
    // mode; supplied by the substepping wrapper otherwise so the
    // body-local contact anchors `p_a_local` stay consistent across
    // substeps.
    let q_anchor = q_at_contact.unwrap_or(q_in);
    let qd_in = state_in.body_qd.host()?;
    let f_in = state_in.body_f.host()?;
    let inv_mass = model.body_inv_mass.host()?;
    let inv_inertia_body = model.body_inv_inertia.host()?;
    let body_com = model.body_com.host()?;
    let flags = model.body_flags.host()?;
    let world_idx = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let external = if control.body_external_f.len() == n {
        Some(control.body_external_f.host()?)
    } else {
        None
    };

    // ── 1. Predict ────────────────────────────────────────────
    // x_new = x + dt·v + dt²·(f/m + g)
    // q_new = quat_integrate(q, ω, dt)
    let mut q_pred: Vec<Transform> = q_in.to_vec();
    let mut v_pred: Vec<Vec3> = (0..n).map(|i| qd_in[i].linear).collect();
    let mut w_pred: Vec<Vec3> = (0..n).map(|i| qd_in[i].angular).collect();

    for i in 0..n {
        if (flags[i] & BodyFlags::KINEMATIC) != 0 {
            continue;
        }
        let mut applied_lin = f_in[i].linear;
        let mut applied_ang = f_in[i].angular;
        if let Some(ext) = external {
            applied_lin += ext[i].linear;
            applied_ang += ext[i].angular;
        }
        let world = world_idx[i].max(0) as usize;
        let g = if inv_mass[i] > 0.0 {
            gravity[world]
        } else {
            Vec3::ZERO
        };
        let v0 = qd_in[i].linear;
        let mut v1 = v0 + (applied_lin * inv_mass[i] + g) * dt;
        // Linear damping: v *= exp(-damp · dt). Small-dt approximation
        // is fine and faster — std exp is overkill for ≤10/s damping.
        let damp = model.linear_damping;
        if damp > 0.0 {
            let factor = (1.0 - damp * dt).max(0.0);
            v1 *= factor;
        }
        v_pred[i] = v1;
        q_pred[i].position = q_in[i].position + v1 * dt;

        // Angular: ω̇ = I⁻¹·(τ - ω × I·ω). For stage 4, with small ω
        // and no torque, we drop the gyroscopic term — boxes don't
        // need it for stacking. Add when needed.
        let i_inv_world = world_inv_inertia(q_in[i].rotation, inv_inertia_body[i]);
        let dw = i_inv_world * applied_ang * dt;
        let w1 = w_pred[i] + dw;
        w_pred[i] = w1;
        // Quat integration: q_new = exp(0.5 · dt · ω) ⊗ q_old.
        q_pred[i].rotation = integrate_rotation(q_in[i].rotation, w1, dt);
    }

    // ── 2. Position projection ────────────────────────────────
    // Jacobi-style relaxation: each contact computes deltas
    // independently, accumulated and averaged per body. Stable for
    // multi-contact bodies like box stacks. Mirrors Newton's
    // `solve_body_contact_positions` + `apply_body_deltas`.
    let n_contacts = contacts.count as usize;

    // Pre-compute body-local contact points (frozen at narrow-phase
    // time). At each iteration we rotate these into world frame using
    // the current pose so the penetration estimate stays faithful even
    // after angular corrections accumulate. Shared between position
    // and velocity projection. Computed against `q_anchor` so the
    // anchors stay consistent across substeps.
    let (p_a_local, p_b_local) = if n_contacts > 0 {
        let a_idx = contacts.a_index.host()?;
        let b_idx = contacts.b_index.host()?;
        let point = contacts.point.host()?;
        let mut pa = Vec::with_capacity(n_contacts);
        let mut pb = Vec::with_capacity(n_contacts);
        for c in 0..n_contacts {
            let a = a_idx[c] as usize;
            let b = b_idx[c] as usize;
            let pw = point[c];
            pa.push(if is_static_or_kinematic(flags, a) {
                Vec3::ZERO
            } else {
                q_anchor[a].rotation.conjugate() * (pw - q_anchor[a].position)
            });
            pb.push(if is_static_or_kinematic(flags, b) || b >= q_anchor.len() {
                Vec3::ZERO
            } else {
                q_anchor[b].rotation.conjugate() * (pw - q_anchor[b].position)
            });
        }
        (pa, pb)
    } else {
        (Vec::new(), Vec::new())
    };

    if n_contacts > 0 {
        let a_idx = contacts.a_index.host()?;
        let b_idx = contacts.b_index.host()?;
        let normal = contacts.normal.host()?;
        let depth = contacts.depth.host()?;
        let point = contacts.point.host()?;

        // Jacobi-style relaxation (matches Newton's
        // `solve_body_contact_positions` + `apply_body_deltas`). Each
        // contact computes its impulse independently, then deltas are
        // averaged per body. Much more stable for multi-contact bodies
        // (e.g. box stacks) than per-contact Gauss-Seidel.
        let mut lin_delta = vec![Vec3::ZERO; n];
        let mut ang_delta = vec![Vec3::ZERO; n];
        let mut weight = vec![0.0_f64; n];

        for _ in 0..position_iterations {
            for d in lin_delta.iter_mut() {
                *d = Vec3::ZERO;
            }
            for d in ang_delta.iter_mut() {
                *d = Vec3::ZERO;
            }
            for w in weight.iter_mut() {
                *w = 0.0;
            }

            for c in 0..n_contacts {
                let a = a_idx[c] as usize;
                let b = b_idx[c] as usize;
                let n_ab = normal[c];
                let initial = depth[c];

                let kin_a = is_static_or_kinematic(flags, a);
                let kin_b = is_static_or_kinematic(flags, b);
                let inv_m_a = if kin_a { 0.0 } else { inv_mass.get(a).copied().unwrap_or(0.0) };
                let inv_m_b = if kin_b { 0.0 } else { inv_mass.get(b).copied().unwrap_or(0.0) };

                let p_a_now = if kin_a {
                    point[c]
                } else {
                    q_pred[a].position + q_pred[a].rotation * p_a_local[c]
                };
                let p_b_now = if kin_b || b >= q_pred.len() {
                    point[c]
                } else {
                    q_pred[b].position + q_pred[b].rotation * p_b_local[c]
                };

                let r_a = if kin_a {
                    Vec3::ZERO
                } else {
                    p_a_now - com_world(q_pred[a], body_com[a])
                };
                let r_b = if kin_b || b >= q_pred.len() {
                    Vec3::ZERO
                } else {
                    p_b_now - com_world(q_pred[b], body_com[b])
                };

                let i_inv_a = if kin_a {
                    Mat3::ZERO
                } else {
                    world_inv_inertia(q_pred[a].rotation, inv_inertia_body[a])
                };
                let i_inv_b = if kin_b || b >= q_pred.len() {
                    Mat3::ZERO
                } else {
                    world_inv_inertia(q_pred[b].rotation, inv_inertia_body[b])
                };

                let ra_x_n = r_a.cross(n_ab);
                let rb_x_n = r_b.cross(n_ab);
                // Jacobian rows: linear_a = -n, angular_a = -r_a × n.
                let denom = inv_m_a
                    + inv_m_b
                    + ra_x_n.dot(i_inv_a * ra_x_n)
                    + rb_x_n.dot(i_inv_b * rb_x_n);
                if denom <= 0.0 {
                    continue;
                }

                let current = initial + (p_a_now - p_b_now).dot(n_ab);
                if current <= 0.0 {
                    continue;
                }

                let lambda = current / denom;

                if !kin_a {
                    lin_delta[a] -= n_ab * lambda;
                    ang_delta[a] -= ra_x_n * lambda;
                    weight[a] += 1.0;
                }
                if !kin_b && b < lin_delta.len() {
                    lin_delta[b] += n_ab * lambda;
                    ang_delta[b] += rb_x_n * lambda;
                    weight[b] += 1.0;
                }
            }

            // Apply averaged per-body deltas.
            for i in 0..n {
                if weight[i] <= 0.0 || (flags[i] & BodyFlags::KINEMATIC) != 0 {
                    continue;
                }
                let inv_w = 1.0 / weight[i];
                let dp = lin_delta[i] * (inv_mass[i] * inv_w);
                let dtheta = world_inv_inertia(q_pred[i].rotation, inv_inertia_body[i])
                    * ang_delta[i]
                    * inv_w;
                q_pred[i].position += dp;
                q_pred[i].rotation = apply_rotation_delta(q_pred[i].rotation, dtheta);
            }
        }
    }

    // ── 2b. Equality-constraint projection (CONNECT / WELD) ──
    // Logic factored into `equality_projection_pass` so the GPU
    // backend can reuse it as a CPU shim until the kernel grows
    // native support.
    equality_projection_pass(model, &mut q_pred, position_iterations)?;

    // ── 3. Velocity recovery ─────────────────────────────────
    // Linear: v_post = (p_new - p_old) / dt.
    // Angular: ω_post = quat_to_axis_angle(q_new · q_old⁻¹) / dt.
    let mut v_post: Vec<Vec3> = (0..n)
        .map(|i| (q_pred[i].position - q_in[i].position) / dt)
        .collect();
    let mut w_post: Vec<Vec3> = (0..n)
        .map(|i| {
            let dq = q_pred[i].rotation * q_in[i].rotation.conjugate();
            quat_to_omega(dq) / dt
        })
        .collect();

    // ── 4. Velocity projection at contacts ───────────────────
    // Jacobi-style: for each contact compute Δv_lin / Δω contributions,
    // accumulate per body, then average. Targets the *pre-projection*
    // contact-point relative normal velocity for restitution.
    if n_contacts > 0 {
        let a_idx = contacts.a_index.host()?;
        let b_idx = contacts.b_index.host()?;
        let normal = contacts.normal.host()?;
        let restitution_arr = contacts.restitution.host()?;
        let friction_arr = contacts.friction.host()?;

        let mut v_lin_delta = vec![Vec3::ZERO; n];
        let mut v_ang_delta = vec![Vec3::ZERO; n];
        let mut v_weight = vec![0.0_f64; n];

        for c in 0..n_contacts {
            let a = a_idx[c] as usize;
            let b = b_idx[c] as usize;
            let n_ab = normal[c];

            let kin_a = is_static_or_kinematic(flags, a);
            let kin_b = is_static_or_kinematic(flags, b);
            let inv_m_a = if kin_a { 0.0 } else { inv_mass.get(a).copied().unwrap_or(0.0) };
            let inv_m_b = if kin_b { 0.0 } else { inv_mass.get(b).copied().unwrap_or(0.0) };

            // Lever arms at the post-projection pose (q_pred), using
            // the body-local contact points frozen at narrow-phase time.
            let r_a = if kin_a {
                Vec3::ZERO
            } else {
                let p_a_now = q_pred[a].position + q_pred[a].rotation * p_a_local[c];
                p_a_now - com_world(q_pred[a], body_com[a])
            };
            let r_b = if kin_b || b >= q_pred.len() {
                Vec3::ZERO
            } else {
                let p_b_now = q_pred[b].position + q_pred[b].rotation * p_b_local[c];
                p_b_now - com_world(q_pred[b], body_com[b])
            };

            let i_inv_a = if kin_a {
                Mat3::ZERO
            } else {
                world_inv_inertia(q_pred[a].rotation, inv_inertia_body[a])
            };
            let i_inv_b = if kin_b || b >= q_pred.len() {
                Mat3::ZERO
            } else {
                world_inv_inertia(q_pred[b].rotation, inv_inertia_body[b])
            };

            let ra_x_n = r_a.cross(n_ab);
            let rb_x_n = r_b.cross(n_ab);
            let denom = inv_m_a
                + inv_m_b
                + ra_x_n.dot(i_inv_a * ra_x_n)
                + rb_x_n.dot(i_inv_b * rb_x_n);
            if denom <= 0.0 {
                continue;
            }

            // Pre-projection contact-point relative velocity.
            let v_a_pre = if kin_a {
                Vec3::ZERO
            } else {
                v_pred[a] + w_pred[a].cross(r_a)
            };
            let v_b_pre = if kin_b || b >= v_pred.len() {
                Vec3::ZERO
            } else {
                v_pred[b] + w_pred[b].cross(r_b)
            };
            let v_n_pre = (v_b_pre - v_a_pre).dot(n_ab);
            if v_n_pre >= 0.0 {
                continue;
            }
            let e = restitution_arr[c];
            let v_n_target = -e * v_n_pre;

            // Current post-projection contact-point rel velocity.
            let v_a = if kin_a {
                Vec3::ZERO
            } else {
                v_post[a] + w_post[a].cross(r_a)
            };
            let v_b = if kin_b || b >= v_post.len() {
                Vec3::ZERO
            } else {
                v_post[b] + w_post[b].cross(r_b)
            };
            let v_rel = v_b - v_a;
            let v_n = v_rel.dot(n_ab);
            let dv_n = v_n_target - v_n;
            let lambda_n = dv_n / denom;

            if !kin_a {
                v_lin_delta[a] -= n_ab * lambda_n;
                v_ang_delta[a] -= ra_x_n * lambda_n;
                v_weight[a] += 1.0;
            }
            if !kin_b && b < v_lin_delta.len() {
                v_lin_delta[b] += n_ab * lambda_n;
                v_ang_delta[b] += rb_x_n * lambda_n;
                v_weight[b] += 1.0;
            }

            // Coulomb tangential — same Jacobi accumulation.
            let mu = friction_arr[c];
            if mu > 0.0 {
                let v_t = v_rel - n_ab * v_n;
                let v_t_len = v_t.length();
                if v_t_len > 1e-9 {
                    let t_hat = v_t / v_t_len;
                    let ra_x_t = r_a.cross(t_hat);
                    let rb_x_t = r_b.cross(t_hat);
                    let denom_t = inv_m_a
                        + inv_m_b
                        + ra_x_t.dot(i_inv_a * ra_x_t)
                        + rb_x_t.dot(i_inv_b * rb_x_t);
                    if denom_t > 0.0 {
                        let impulse_cap = mu * lambda_n.abs();
                        let dv_t = v_t_len.min(impulse_cap.max(0.0));
                        let lambda_t = dv_t / denom_t;
                        if !kin_a {
                            v_lin_delta[a] += t_hat * lambda_t;
                            v_ang_delta[a] += ra_x_t * lambda_t;
                        }
                        if !kin_b && b < v_lin_delta.len() {
                            v_lin_delta[b] -= t_hat * lambda_t;
                            v_ang_delta[b] -= rb_x_t * lambda_t;
                        }
                    }
                }
            }
        }

        for i in 0..n {
            if v_weight[i] <= 0.0 || (flags[i] & BodyFlags::KINEMATIC) != 0 {
                continue;
            }
            let inv_w = 1.0 / v_weight[i];
            v_post[i] += v_lin_delta[i] * (inv_mass[i] * inv_w);
            let i_inv = world_inv_inertia(q_pred[i].rotation, inv_inertia_body[i]);
            w_post[i] += i_inv * v_ang_delta[i] * inv_w;
        }
    }

    // ── Write outputs ────────────────────────────────────────
    let q_out = state_out.body_q.host_mut()?;
    let qd_out = state_out.body_qd.host_mut()?;
    for i in 0..n {
        if (flags[i] & BodyFlags::KINEMATIC) != 0 {
            q_out[i] = q_in[i];
            qd_out[i] = qd_in[i];
            continue;
        }
        q_out[i] = q_pred[i];
        q_out[i].renormalize();
        qd_out[i] = SpatialVector::new(v_post[i], w_post[i]);
    }
    for f in state_out.body_f.host_mut()? {
        *f = SpatialVector::ZERO;
    }

    // Particle XPBD pass — separate from rigid bodies for now.
    if model.particle_count > 0 {
        let p_in = state_in.particle_q.host()?.to_vec();
        let v_in = state_in.particle_qd.host()?.to_vec();
        let mut p_out = vec![Vec3::ZERO; model.particle_count];
        let mut v_out = vec![Vec3::ZERO; model.particle_count];
        step_particles(model, &p_in, &v_in, &mut p_out, &mut v_out, dt, position_iterations)?;
        state_out
            .particle_q
            .host_mut()?
            .copy_from_slice(&p_out);
        state_out
            .particle_qd
            .host_mut()?
            .copy_from_slice(&v_out);
    }
    Ok(())
}

fn is_static_or_kinematic(flags: &[u32], idx: usize) -> bool {
    if idx == STATIC_BODY as usize {
        return true;
    }
    flags
        .get(idx)
        .copied()
        .is_none_or(|f| f & BodyFlags::KINEMATIC != 0)
}

#[inline]
fn com_world(xf: Transform, com: Vec3) -> Vec3 {
    xf.position + xf.rotation * com
}

/// World-frame inverse inertia: R · I_body⁻¹ · Rᵀ.
fn world_inv_inertia(rotation: Quat, body_inv_inertia: Mat3) -> Mat3 {
    let r = Mat3::from_quat(rotation);
    r * body_inv_inertia * r.transpose()
}

/// `q_new = exp(0.5 · dt · ω) ⊗ q_old`, using the small-angle form.
fn integrate_rotation(q: Quat, omega: Vec3, dt: Real) -> Quat {
    let dq = Quat::from_xyzw(
        omega.x * 0.5 * dt,
        omega.y * 0.5 * dt,
        omega.z * 0.5 * dt,
        0.0,
    );
    (dq * q + q).normalize()
}

/// Apply a small rotation `Δθ` (rotation vector) to a unit quaternion.
fn apply_rotation_delta(q: Quat, dtheta: Vec3) -> Quat {
    let dq = Quat::from_xyzw(dtheta.x * 0.5, dtheta.y * 0.5, dtheta.z * 0.5, 0.0);
    (dq * q + q).normalize()
}

/// Recover `ω` from a unit-quaternion delta. Caller divides by `dt`.
fn quat_to_omega(dq: Quat) -> Vec3 {
    // Make sure we take the short way around.
    let sign = if dq.w < 0.0 { -1.0 } else { 1.0 };
    Vec3::new(dq.x * 2.0 * sign, dq.y * 2.0 * sign, dq.z * 2.0 * sign)
}

/// Run `iterations` Jacobi-style passes of the CONNECT / WELD
/// equality constraints over `q_pred`. Each pass reads the per-body
/// flags / inv-mass / inv-inertia from `model` and pulls anchor
/// points together (CONNECT) and aligns frames (WELD). Used by the
/// CPU step directly and by the GPU backend as a host-side shim.
pub(crate) fn equality_projection_pass(
    model: &Model,
    q_pred: &mut [Transform],
    iterations: usize,
) -> Result<()> {
    if model.equality_count == 0 {
        return Ok(());
    }
    let flags = model.body_flags.host()?;
    let inv_mass = model.body_inv_mass.host()?;
    let body_com = model.body_com.host()?;
    let inv_inertia_body = model.body_inv_inertia.host()?;
    let eq_type = model.equality_type.host()?;
    let eq_a = model.equality_body_a.host()?;
    let eq_b = model.equality_body_b.host()?;
    let eq_anchor_a = model.equality_anchor_a.host()?;
    let eq_anchor_b = model.equality_anchor_b.host()?;
    let eq_relrot = model.equality_relative_rotation.host()?;
    let eq_enabled = model.equality_enabled.host()?;
    for _ in 0..iterations.max(1) {
        for c in 0..model.equality_count {
            if eq_enabled[c] == 0 {
                continue;
            }
            let a = eq_a[c];
            let b = eq_b[c];
            let anchor_a_local = eq_anchor_a[c];
            let anchor_b_local = eq_anchor_b[c];

            let kin_a = a < 0
                || ((flags.get(a as usize).copied().unwrap_or(0))
                    & BodyFlags::KINEMATIC
                    != 0);
            let kin_b = b < 0
                || ((flags.get(b as usize).copied().unwrap_or(0))
                    & BodyFlags::KINEMATIC
                    != 0);
            let inv_m_a = if kin_a {
                0.0
            } else {
                inv_mass.get(a as usize).copied().unwrap_or(0.0)
            };
            let inv_m_b = if kin_b {
                0.0
            } else {
                inv_mass.get(b as usize).copied().unwrap_or(0.0)
            };
            if inv_m_a + inv_m_b <= 0.0 {
                continue;
            }
            let xf_a = if a >= 0 {
                q_pred[a as usize]
            } else {
                Transform::IDENTITY
            };
            let xf_b = if b >= 0 {
                q_pred[b as usize]
            } else {
                Transform::IDENTITY
            };
            let p_a = xf_a.transform_point(anchor_a_local);
            let p_b = xf_b.transform_point(anchor_b_local);

            let delta = p_b - p_a;
            let r_a = if kin_a {
                Vec3::ZERO
            } else {
                p_a - com_world(q_pred[a as usize], body_com[a as usize])
            };
            let r_b = if kin_b {
                Vec3::ZERO
            } else {
                p_b - com_world(q_pred[b as usize], body_com[b as usize])
            };
            let len = delta.length();
            if len > 1e-9 {
                let n_dir = delta / len;
                let i_inv_a = if kin_a {
                    Mat3::ZERO
                } else {
                    world_inv_inertia(
                        q_pred[a as usize].rotation,
                        inv_inertia_body[a as usize],
                    )
                };
                let i_inv_b = if kin_b {
                    Mat3::ZERO
                } else {
                    world_inv_inertia(
                        q_pred[b as usize].rotation,
                        inv_inertia_body[b as usize],
                    )
                };
                let ra_x_n = r_a.cross(n_dir);
                let rb_x_n = r_b.cross(n_dir);
                let w_a = inv_m_a + ra_x_n.dot(i_inv_a * ra_x_n);
                let w_b = inv_m_b + rb_x_n.dot(i_inv_b * rb_x_n);
                let w_sum = w_a + w_b;
                if w_sum > 0.0 {
                    let lambda = len / w_sum;
                    if !kin_a {
                        q_pred[a as usize].position += n_dir * (lambda * inv_m_a);
                        let dtheta = i_inv_a * ra_x_n * lambda;
                        q_pred[a as usize].rotation =
                            apply_rotation_delta(q_pred[a as usize].rotation, dtheta);
                    }
                    if !kin_b {
                        q_pred[b as usize].position -= n_dir * (lambda * inv_m_b);
                        let dtheta = i_inv_b * rb_x_n * (-lambda);
                        q_pred[b as usize].rotation =
                            apply_rotation_delta(q_pred[b as usize].rotation, dtheta);
                    }
                }
            }

            if eq_type[c] == crate::sim::EqualityType::Weld as u32 {
                let r_target = eq_relrot[c];
                let q_a_now = q_pred[a as usize].rotation;
                let q_b_now = q_pred[b as usize].rotation;
                let q_b_target = q_a_now * r_target;
                let dq = q_b_target * q_b_now.conjugate();
                let sign = if dq.w < 0.0 { -1.0 } else { 1.0 };
                let theta_full = Vec3::new(
                    dq.x * 2.0 * sign,
                    dq.y * 2.0 * sign,
                    dq.z * 2.0 * sign,
                );
                let tr_a = if kin_a {
                    0.0
                } else {
                    let i_inv = world_inv_inertia(
                        q_pred[a as usize].rotation,
                        inv_inertia_body[a as usize],
                    );
                    i_inv.x_axis.x + i_inv.y_axis.y + i_inv.z_axis.z
                };
                let tr_b = if kin_b {
                    0.0
                } else {
                    let i_inv = world_inv_inertia(
                        q_pred[b as usize].rotation,
                        inv_inertia_body[b as usize],
                    );
                    i_inv.x_axis.x + i_inv.y_axis.y + i_inv.z_axis.z
                };
                let tr_sum = tr_a + tr_b;
                if tr_sum > 0.0 {
                    let frac_a = tr_a / tr_sum;
                    let frac_b = tr_b / tr_sum;
                    if !kin_a {
                        q_pred[a as usize].rotation = apply_rotation_delta(
                            q_pred[a as usize].rotation,
                            -theta_full * frac_a,
                        );
                    }
                    if !kin_b {
                        q_pred[b as usize].rotation = apply_rotation_delta(
                            q_pred[b as usize].rotation,
                            theta_full * frac_b,
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

// `apply_equality_post_gpu` (CPU shim) was removed when
// `XpbdEqualityKernel` (`shaders/xpbd_equality.wgsl`) replaced it
// with a true GPU dispatch. The CPU `equality_projection_pass`
// above is kept — it's still used by the CPU `cpu_step` path and
// by tests as the reference oracle for the WGSL port.

/// Run one full XPBD particle step on host:
/// 1. Predict: `p_new = p + dt·v + dt²·g`, with linear damping.
/// 2. Project distance constraints (springs) using XPBD compliance.
/// 3. Project particle vs static plane Y=0 (snap up to `radius`).
/// 4. Recover velocity: `v_post = (p_new - p) / dt`.
///
/// All three molla scenes that use particles today (cloth) are CPU-
/// only — porting to a WGSL kernel is straightforward but mirrors the
/// rigid-body kernel's effort and isn't on the critical path. For
/// 10k-particle workloads this loop runs in well under a millisecond
/// per step on a modern host CPU thanks to the simple data layout.
pub(crate) fn step_particles(
    model: &Model,
    particle_q_in: &[Vec3],
    particle_qd_in: &[Vec3],
    particle_q_out: &mut [Vec3],
    particle_qd_out: &mut [Vec3],
    dt: Real,
    iterations: usize,
) -> Result<()> {
    if model.particle_count == 0 {
        return Ok(());
    }
    let n = model.particle_count;
    let inv_mass = model.particle_inv_mass.host()?;
    let radius = model.particle_radius.host()?;
    let world_idx = model.particle_world.host()?;
    let gravity = model.gravity.host()?;
    let lin_damp = model.linear_damping;

    // --- 1. Predict -------------------------------------------------
    let mut p: Vec<Vec3> = (0..n).map(|i| particle_q_in[i]).collect();
    let mut v: Vec<Vec3> = (0..n).map(|i| particle_qd_in[i]).collect();
    for i in 0..n {
        if inv_mass[i] <= 0.0 {
            continue; // pinned / kinematic
        }
        let world = world_idx[i].max(0) as usize;
        let g = gravity.get(world).copied().unwrap_or(Vec3::ZERO);
        let mut v1 = v[i] + g * dt;
        if lin_damp > 0.0 {
            v1 *= (1.0 - lin_damp * dt).max(0.0);
        }
        v[i] = v1;
        p[i] += v1 * dt;
    }

    // --- 2. Distance constraint projection (XPBD) ------------------
    // Faithful port of Newton's `solve_springs` (Warp): per-constraint
    // Lagrange multiplier accumulator `λ` reset to zero at the start
    // of each substep. Each iteration computes
    //
    //     Δλ = -(C + α·λ) / (Σw + α)
    //     Δp_i = ±wᵢ · Δλ · n
    //     λ ← λ + Δλ
    //
    // The accumulator is what distinguishes XPBD from plain PBD —
    // without it, the per-iter correction overshoots and the spring
    // oscillates instead of converging. That oscillation looks like
    // "the middle particles wiggling weirdly" when you grab the cloth.
    //
    // Reference: Macklin et al., "XPBD: Position-Based Simulation of
    // Compliant Constrained Dynamics" (2016), eq. 18; Warp's
    // `solve_springs` kernel.
    if model.spring_count > 0 {
        let sa = model.spring_a.host()?;
        let sb = model.spring_b.host()?;
        let rest = model.spring_rest_length.host()?;
        let comp = model.spring_compliance.host()?;

        // λ accumulator — one entry per spring, reset each substep.
        let mut lambdas = vec![0.0; model.spring_count];
        let dt2 = dt * dt;

        for _ in 0..iterations.max(1) {
            for c in 0..model.spring_count {
                let a = sa[c] as usize;
                let b = sb[c] as usize;
                if a >= n || b >= n {
                    continue;
                }
                let wa = inv_mass[a];
                let wb = inv_mass[b];
                let w_sum = wa + wb;
                if w_sum <= 0.0 {
                    continue;
                }
                let delta = p[b] - p[a];
                let len = delta.length();
                if len < 1e-9 {
                    continue;
                }
                let n_dir = delta / len;
                let c_val = len - rest[c];
                // α has units of m/N — XPBD compliance. We store
                // it as a per-spring buffer so different spring
                // populations (cloth vs rope) can have different
                // stiffness without rebuilding the model.
                let alpha = comp[c] / dt2;
                let dlambda = -(c_val + alpha * lambdas[c]) / (w_sum + alpha);
                lambdas[c] += dlambda;
                let dp = n_dir * dlambda;
                // grad_c_xa = -n, grad_c_xb = +n →
                // Δx_a = wa · Δλ · grad_c_xa = -wa · Δλ · n
                // Δx_b = wb · Δλ · grad_c_xb = +wb · Δλ · n
                p[a] -= dp * wa;
                p[b] += dp * wb;
            }
        }

        // --- 2b. Hard strain limit (Jacobi) ---------------------------
        // XPBD with finite iterations leaves residual stretch when a
        // corner is yanked fast. The strain-limit pass enforces
        //   length(a, b) ≤ rest · (1 + max_stretch)
        // as a unilateral position projection.
        //
        // Jacobi-style: each iteration first **proposes** corrections
        // for every violated spring, accumulates them per-particle,
        // and applies the AVERAGED correction once. That avoids the
        // zigzag pattern Gauss-Seidel produces in chained meshes
        // (clamping spring AB shifts particle B, which violates BC,
        // which when clamped shifts B back — visible as the middle
        // particles tangling). Reference: Macklin 2014 PBD §3.2.
        //
        // Only triggers for springs with `max_stretch < 1.0`; large
        // values are the "no cap" sentinel and skip the work.
        let mss = model.spring_max_stretch.host()?;
        let strain_max_iters = iterations * 4;
        let mut deltas: Vec<Vec3> = vec![Vec3::ZERO; n];
        let mut counts: Vec<u32> = vec![0; n];
        for _ in 0..strain_max_iters {
            for d in deltas.iter_mut() {
                *d = Vec3::ZERO;
            }
            for c_count in counts.iter_mut() {
                *c_count = 0;
            }
            let mut violated = false;
            for c in 0..model.spring_count {
                let ms = mss[c];
                if ms >= 1.0 {
                    continue;
                }
                let a = sa[c] as usize;
                let b = sb[c] as usize;
                if a >= n || b >= n {
                    continue;
                }
                let wa = inv_mass[a];
                let wb = inv_mass[b];
                let w_sum = wa + wb;
                if w_sum <= 0.0 {
                    continue;
                }
                let max_len = rest[c] * (1.0 + ms);
                let delta = p[b] - p[a];
                let len = delta.length();
                if len > max_len && len > 1e-9 {
                    violated = true;
                    let n_dir = delta / len;
                    let excess = len - max_len;
                    let dp = n_dir * (excess / w_sum);
                    deltas[a] += dp * wa;
                    deltas[b] -= dp * wb;
                    counts[a] += 1;
                    counts[b] += 1;
                }
            }
            if !violated {
                break;
            }
            // Apply averaged deltas (Jacobi). Skip pinned particles
            // (inv_mass = 0) — they shouldn't be moved.
            for i in 0..n {
                if inv_mass[i] <= 0.0 || counts[i] == 0 {
                    continue;
                }
                p[i] += deltas[i] / counts[i] as Real;
            }
        }
    }

    // --- 3. Particle vs Y=0 plane (cheap floor collision) ----------
    for i in 0..n {
        if inv_mass[i] <= 0.0 {
            continue;
        }
        let r = radius[i];
        if p[i].y < r {
            p[i].y = r;
        }
    }

    // --- 4. Velocity recovery + write out --------------------------
    for i in 0..n {
        if inv_mass[i] <= 0.0 {
            particle_q_out[i] = particle_q_in[i];
            particle_qd_out[i] = Vec3::ZERO;
            continue;
        }
        particle_qd_out[i] = (p[i] - particle_q_in[i]) / dt;
        particle_q_out[i] = p[i];
    }
    Ok(())
}

/// CPU shim that runs `step_particles` after the GPU rigid-body step.
/// Particles aren't yet in the WGSL kernel — same situation as
/// equality constraints. Downloads the existing particle buffers,
/// runs one substep on host, uploads the result.
#[cfg(feature = "gpu")]
pub(crate) fn apply_particles_post_gpu(
    model: &Model,
    state_in: &State,
    state_out: &mut State,
    dt: Real,
    iterations: usize,
    _device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<()> {
    if model.particle_count == 0 {
        return Ok(());
    }
    let p_in = state_in.particle_q.host()?.to_vec();
    let v_in = state_in.particle_qd.host()?.to_vec();

    // Safety: state_out particle buffers are sized to particle_count
    // by State::for_model + clone_host. Just write into them.
    let mut p_out = vec![Vec3::ZERO; model.particle_count];
    let mut v_out = vec![Vec3::ZERO; model.particle_count];
    step_particles(model, &p_in, &v_in, &mut p_out, &mut v_out, dt, iterations)?;

    state_out.particle_q.host_mut()?.copy_from_slice(&p_out);
    state_out.particle_qd.host_mut()?.copy_from_slice(&v_out);
    state_out.particle_q.upload(queue)?;
    state_out.particle_qd.upload(queue)?;
    Ok(())
}
