//! XPBD (eXtended Position-Based Dynamics) — position-projection
//! solver for rigid bodies with contacts.
//!
//! Stage 4.0 implementation:
//! 1. Predict positions from velocities + gravity (semi-implicit-ish).
//! 2. **Position projection**: for every contact, push the two
//!    bodies apart along the normal until the recorded penetration
//!    is removed. Mass-weighted; static / kinematic peers contribute
//!    zero inverse mass and absorb no correction.
//! 3. **Velocity recovery**: `v = (p_new - p_old) / dt` (PBD-classic).
//! 4. **Velocity projection**: along each contact normal, replace
//!    the relative approach velocity by `-restitution * v_in`. With
//!    `restitution = 0` this stops bouncing and the body settles.
//!
//! Friction: Coulomb tangential clamp on the velocity-projection
//! pass. The clamp is `μ` times the normal impulse magnitude.

use crate::core::{Error, Result};
use crate::geometry::STATIC_BODY;
use crate::math::{Real, SpatialVector, Transform, Vec3};
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
}

/// Position-based dynamics solver.
pub struct XpbdSolver {
    /// Position-projection iterations per step. Multiple iterations
    /// help when many contacts share a body (e.g. a stack).
    pub position_iterations: usize,
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
        Ok(Self {
            position_iterations: 4,
            backend: XpbdBackend::Gpu(GpuBackend { device, queue, kernel }),
        })
    }

    #[must_use]
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.position_iterations = n;
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

        match &mut self.backend {
            XpbdBackend::Cpu => cpu_step(model, state_in, control, contacts, dt, state_out, self.position_iterations),
            #[cfg(feature = "gpu")]
            XpbdBackend::Gpu(gpu) => {
                let GpuBackend { device, queue, kernel } = gpu;
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
) -> Result<()> {
    let n = model.body_count;
    let q_in = state_in.body_q.host()?;
        let qd_in = state_in.body_qd.host()?;
        let f_in = state_in.body_f.host()?;
        let inv_mass = model.body_inv_mass.host()?;
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
        // v_new = v + dt·(f/m + g)            (provisional)
        let mut q_pred: Vec<Transform> = q_in.to_vec();
        let mut v_pred: Vec<Vec3> = (0..n).map(|i| qd_in[i].linear).collect();

        for i in 0..n {
            if (flags[i] & BodyFlags::KINEMATIC) != 0 {
                continue;
            }
            let mut applied = f_in[i].linear;
            if let Some(ext) = external {
                applied += ext[i].linear;
            }
            let world = world_idx[i].max(0) as usize;
            let g = if inv_mass[i] > 0.0 {
                gravity[world]
            } else {
                Vec3::ZERO
            };
            let v0 = qd_in[i].linear;
            let v1 = v0 + (applied * inv_mass[i] + g) * dt;
            v_pred[i] = v1;
            q_pred[i].position = q_in[i].position + v1 * dt;
        }

        // ── 2. Position projection ────────────────────────────────
        // We recompute the *current* penetration each iteration as
        //   d_now = d_initial + (Δq_a - Δq_b) · n_ab
        // where Δq_x is each body's accumulated displacement away
        // from the input pose. This corrects two issues:
        //   * The contact's `depth` was measured at `state_in`,
        //     but the predict step moves bodies further into the
        //     contact — we need to resolve the larger penetration.
        //   * For bodies in multiple contacts, earlier corrections
        //     in the same iteration affect the accuracy of later
        //     ones; reading current displacement keeps the constraint
        //     truthful as the loop walks.
        let n_contacts = contacts.count as usize;
        let q_in_pos: Vec<Vec3> = q_in.iter().map(|t| t.position).collect();
        if n_contacts > 0 {
            let a_idx = contacts.a_index.host()?;
            let b_idx = contacts.b_index.host()?;
            let normal = contacts.normal.host()?;
            let depth = contacts.depth.host()?;

            for _ in 0..position_iterations {
                for c in 0..n_contacts {
                    let a = a_idx[c] as usize;
                    let b = b_idx[c] as usize;
                    let n_ab = normal[c]; // points a → b
                    let initial = depth[c];

                    let (w_a, w_b, kin_a, kin_b) = inv_masses(flags, inv_mass, a, b);
                    let w_sum = w_a + w_b;
                    if w_sum <= 0.0 {
                        continue;
                    }

                    let delta_a = if kin_a {
                        Vec3::ZERO
                    } else {
                        q_pred[a].position - q_in_pos[a]
                    };
                    let delta_b = if kin_b || b >= q_pred.len() {
                        Vec3::ZERO
                    } else {
                        q_pred[b].position - q_in_pos[b]
                    };

                    // a moving along +n_ab is moving toward b → depth grows.
                    let current = initial + (delta_a - delta_b).dot(n_ab);
                    if current <= 0.0 {
                        continue;
                    }

                    let lambda = current / w_sum;
                    // Push a along -n_ab (away from b), b along +n_ab.
                    if !kin_a {
                        q_pred[a].position -= n_ab * (lambda * w_a);
                    }
                    if !kin_b && b < q_pred.len() {
                        q_pred[b].position += n_ab * (lambda * w_b);
                    }
                }
            }
        }

        // ── 3. Velocity recovery from position delta ─────────────
        // For unconstrained motion this equals `v_pred`. For bodies
        // that got a position projection, it includes the artificial
        // separating velocity that the projection produced — that's
        // why step 4 is necessary.
        let mut v_post: Vec<Vec3> = (0..n)
            .map(|i| (q_pred[i].position - q_in[i].position) / dt)
            .collect();

        // ── 4. Velocity projection at contacts ───────────────────
        // Target the *pre-projection* normal velocity (`v_pred`), not
        // the post-projection velocity, otherwise the position
        // correction's artificial separation gets read as a real bounce.
        if n_contacts > 0 {
            let a_idx = contacts.a_index.host()?;
            let b_idx = contacts.b_index.host()?;
            let normal = contacts.normal.host()?;
            let restitution_arr = contacts.restitution.host()?;
            let friction_arr = contacts.friction.host()?;

            for c in 0..n_contacts {
                let a = a_idx[c] as usize;
                let b = b_idx[c] as usize;
                let n_ab = normal[c]; // a → b

                let (w_a, w_b, kin_a, kin_b) = inv_masses(flags, inv_mass, a, b);
                let w_sum = w_a + w_b;
                if w_sum <= 0.0 {
                    continue;
                }

                // Pre-projection relative velocity at this contact.
                let v_a_pre = if kin_a { Vec3::ZERO } else { v_pred[a] };
                let v_b_pre = if kin_b || b >= v_pred.len() {
                    Vec3::ZERO
                } else {
                    v_pred[b]
                };
                let v_rel_pre = v_b_pre - v_a_pre;
                let v_n_pre = v_rel_pre.dot(n_ab); // < 0 means approaching

                // Skip pairs that were already separating before the
                // position projection — there's nothing to resolve and
                // we don't want to brake them.
                if v_n_pre >= 0.0 {
                    continue;
                }

                let e = restitution_arr[c];
                // Target normal velocity post-resolution: bounce back
                // with restitution. v_n_pre is negative when
                // approaching, so `-e * v_n_pre` is positive
                // (separating). With e=0 the target is 0 and we just
                // brake the approach component without bouncing.
                let v_n_target = -e * v_n_pre;

                // Current post-projection relative normal velocity.
                let v_a = if kin_a { Vec3::ZERO } else { v_post[a] };
                let v_b = if kin_b || b >= v_post.len() {
                    Vec3::ZERO
                } else {
                    v_post[b]
                };
                let v_rel = v_b - v_a;
                let v_n = v_rel.dot(n_ab);

                let dv_n = v_n_target - v_n;
                let lambda_v = dv_n / w_sum;
                if !kin_a {
                    v_post[a] -= n_ab * (lambda_v * w_a);
                }
                if !kin_b && b < v_post.len() {
                    v_post[b] += n_ab * (lambda_v * w_b);
                }

                // Coulomb friction — clamps the tangential
                // post-projection velocity by `μ · |λ_n|`. Uses the
                // post-projection v_t since friction acts on whatever
                // tangential motion survives the normal correction.
                let mu = friction_arr[c];
                if mu > 0.0 {
                    let v_a2 = if kin_a { Vec3::ZERO } else { v_post[a] };
                    let v_b2 = if kin_b || b >= v_post.len() {
                        Vec3::ZERO
                    } else {
                        v_post[b]
                    };
                    let v_rel2 = v_b2 - v_a2;
                    let v_n2 = v_rel2.dot(n_ab);
                    apply_friction(
                        &mut v_post,
                        n_ab,
                        v_rel2,
                        v_n2,
                        w_a,
                        w_b,
                        w_sum,
                        kin_a,
                        kin_b,
                        a,
                        b,
                        mu * lambda_v.abs(),
                    );
                }
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
            qd_out[i] = SpatialVector::new(v_post[i], qd_in[i].angular);
        }
        for f in state_out.body_f.host_mut()? {
            *f = SpatialVector::ZERO;
        }
        Ok(())
}

fn inv_masses(
    flags: &[u32],
    inv_mass: &[Real],
    a: usize,
    b: usize,
) -> (Real, Real, bool, bool) {
    let kin_a = is_static_or_kinematic(flags, a);
    let kin_b = is_static_or_kinematic(flags, b);
    let w_a = if kin_a {
        0.0
    } else {
        inv_mass.get(a).copied().unwrap_or(0.0)
    };
    let w_b = if kin_b {
        0.0
    } else {
        inv_mass.get(b).copied().unwrap_or(0.0)
    };
    (w_a, w_b, kin_a, kin_b)
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

#[allow(clippy::too_many_arguments, reason = "small helper, all args are direct refs to step-loop state")]
fn apply_friction(
    v_post: &mut [Vec3],
    n_ab: Vec3,
    v_rel: Vec3,
    v_n: Real,
    w_a: Real,
    w_b: Real,
    w_sum: Real,
    kin_a: bool,
    kin_b: bool,
    a: usize,
    b: usize,
    impulse_cap: Real,
) {
    let v_t = v_rel - n_ab * v_n;
    let v_t_len = v_t.length();
    if v_t_len < 1e-9 {
        return;
    }
    let t = v_t / v_t_len;
    let dv_t = v_t_len.min(impulse_cap.max(0.0));
    let lambda_t = dv_t / w_sum;
    if !kin_a {
        v_post[a] += t * (lambda_t * w_a);
    }
    if !kin_b && b < v_post.len() {
        v_post[b] -= t * (lambda_t * w_b);
    }
}
