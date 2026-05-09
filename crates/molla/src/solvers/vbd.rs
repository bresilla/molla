//! Vertex Block Descent (VBD) cloth + Augmented VBD soft-on-rigid
//! coupling — Stage 20 (P1).
//!
//! ## Why VBD over Style3D
//!
//! Newton ships two specialised soft-body backends:
//!
//! - **Style3D** — cloth-only projective dynamics with a global PCG
//!   linear solve. Strong on garment / drape workloads, smaller scope.
//! - **VBD + Augmented VBD** — broader: cloth + soft-on-rigid coupling
//!   + cable joints + persistent self-contact.
//!
//! For molla v1 we picked **VBD** because:
//! 1. Its rigid coupling story (Augmented VBD) unblocks soft-on-rigid
//!    demos that Style3D can't do.
//! 2. The vertex-block structure parallelises naturally onto WGSL —
//!    every vertex is a small (3×3) Newton block, dispatched in
//!    colour passes that mirror the existing XPBD particles kernel.
//! 3. The CABLE joint (rope + bend) is on the roadmap; keeping the
//!    door open for it is more valuable than Style3D's garment focus.
//!
//! The choice is recorded in `PLAN.md` Stage 20 + Decision Log.
//!
//! ## Algorithm
//!
//! Implicit Euler: each substep solves
//!   `(M/dt² + Σ Hₑ(x)) Δx = M/dt² (y - x) + Σ fₑ(x)`
//! where `y = x_n + dt·(v_n + dt·g)` is the inertial target.
//!
//! VBD's trick is to do this with **per-vertex** Gauss-Seidel sweeps:
//! at each outer iteration, every vertex assembles the 3×3 row of
//! the Hessian + residual that it owns, inverts it locally, and
//! writes back `x_v ← x_v + H_v⁻¹ g_v`. Within a colour, no two
//! vertices share a constraint, so the sweep is race-free in WGSL.
//!
//! Compared to XPBD:
//! - VBD inherits the **stiffness scale** correctly (it's a true
//!   implicit integrator); high-k cloth doesn't explode at modest
//!   substep counts.
//! - VBD doesn't suffer XPBD's "more iterations ⇒ effectively stiffer"
//!   coupling — convergence improves cloth fidelity without changing
//!   material behaviour.
//!
//! ## Augmented VBD (rigid coupling)
//!
//! For v1 the rigid-side support is the simplest AVBD variant: a
//! quadratic penalty `½ k_c · max(pen, 0)²` against each rigid SDF,
//! folded into the per-vertex block as an extra force/Hessian
//! contribution. No augmented-Lagrangian dual updates, no friction.
//! That's enough for the deliverable demo (cloth wrapping a kinematic
//! sphere); friction + dual updates land in a follow-up stage along
//! with the CABLE joint and Dahl friction.
//!
//! ## Out of scope for v1 (deferred per `PLAN.md` Stage 20)
//!
//! - CABLE joint constraints (stretch + bend on rope-like structures).
//! - Dahl hysteresis friction (smooth stick-slip; needed by CABLE).
//! - Hard-mode AVBD with augmented-Lagrangian dual updates.
//! - MPM coupling.
//! - Tetrahedral / volumetric soft bodies (only triangle-cloth + spring
//!   constraints in v1; the per-vertex block is built to extend
//!   trivially when tet adjacency is wired through).
//!
//! ## References
//!
//! - Chen, Liu, Yang & Yuksel (2024). *Vertex Block Descent.* SIGGRAPH.
//! - Giles, Diaz & Yuksel (2025). *Augmented Vertex Block Descent.*
//!   ACM Trans. Graph. 44(4).
//! - `xtra/newton/newton/_src/solvers/vbd/` — reference Python+Warp
//!   implementation; we mirror its structure but reimplement clean.

use crate::core::{Error, Result};
use crate::math::{Real, Vec3};
use crate::sim::{Contacts, Control, Model, State};

use super::Solver;

/// Backend selector for [`VbdSolver`]. Mirrors `XpbdBackend` —
/// CPU is the oracle / fallback; GPU is the production path.
#[allow(clippy::large_enum_variant, reason = "GPU variant holds Arcs / kernel; one solver instance, fine")]
pub enum VbdBackend {
    Cpu,
    /// GPU dispatch via wgpu. Built once for fixed `(particle_count,
    /// spring_count)`; rebuild the solver if the model topology
    /// changes.
    #[cfg(feature = "gpu")]
    Gpu(GpuBackend),
}

#[cfg(feature = "gpu")]
pub struct GpuBackend {
    pub(crate) device: std::sync::Arc<wgpu::Device>,
    pub(crate) queue: std::sync::Arc<wgpu::Queue>,
    pub(crate) kernel: crate::kernels::VbdClothKernel,
}

/// AVBD contact treatment. **Soft** is the Stage-20 v1 quadratic
/// penalty: a single per-iteration `½ k_c · pen²` term, no dual
/// variable. **Hard** layers an augmented-Lagrangian update on top —
/// each outer iteration accumulates a per-particle multiplier `λ`
/// from the residual penetration, which drives `pen → 0` even at
/// modest contact stiffness. Hard mode is what no-slip / no-penetration
/// cloth-on-rigid demos need.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum VbdMode {
    /// Stage-20 v1 quadratic penalty. Cheap, allows ~1mm penetration
    /// for typical stiffnesses but never overshoots on the way out.
    #[default]
    Soft,
    /// Augmented-Lagrangian dual updates per outer iteration. Drives
    /// penetration toward zero independent of `contact_stiffness`,
    /// at the cost of one extra per-particle scratch buffer (`λ`)
    /// and a residual-evaluation pass per outer iteration.
    Hard,
}

/// Vertex Block Descent cloth + soft-on-rigid solver. Use for
/// high-stiffness cloth where XPBD compliance produces visible jelly
/// behaviour, and for cloth that drapes over rigid kinematic bodies.
pub struct VbdSolver {
    /// Outer Newton iterations per substep. VBD's convergence is the
    /// inner block-solve count: more iters → tighter satisfaction of
    /// the implicit step. 4 is the bare minimum for a cloth grid;
    /// 8–12 is comfortable; 20+ is overkill outside research.
    pub iterations: usize,
    /// Substeps per outer `step`. Like XPBD, more small steps beats
    /// a few big ones when stiffness is high.
    pub substeps: usize,
    /// Soft contact stiffness against rigid SDFs (N/m). 0 disables
    /// rigid coupling entirely. Recommended ~1e5 for a 1cm-radius
    /// cloth particle dropping onto a kinematic sphere.
    pub contact_stiffness: f64,
    /// Cap on per-spring stiffness derived from XPBD compliance. We
    /// build VBD on top of the existing cloth model (which stores
    /// XPBD `α` per spring) so the user can swap solvers without
    /// rebuilding. `k = 1 / max(α, ε)`, then capped at this value
    /// to avoid pathological Hessian conditioning when the user
    /// passes near-zero compliance.
    pub max_spring_stiffness: f64,
    /// Contact-mode selector. Defaults to [`VbdMode::Soft`] for
    /// back-compat with Stage 20.
    pub vbd_mode: VbdMode,
    backend: VbdBackend,
}

impl Default for VbdSolver {
    fn default() -> Self {
        Self::new_cpu()
    }
}

impl VbdSolver {
    /// CPU backend (used by tests + as a fallback). molla is GPU-first
    /// — prefer [`Self::new_gpu`] for any user-facing scene.
    #[must_use]
    pub fn new_cpu() -> Self {
        Self {
            iterations: 8,
            substeps: 1,
            contact_stiffness: 1.0e5,
            max_spring_stiffness: 1.0e7,
            vbd_mode: VbdMode::Soft,
            backend: VbdBackend::Cpu,
        }
    }

    /// GPU backend. The kernel is built for fixed `(particle_count,
    /// spring_count)` taken from `model`; if the topology grows,
    /// rebuild the solver.
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
        let particle_count = model.particle_count as u32;
        let spring_count = model.spring_count as u32;
        let max_k = 1.0e7_f64;
        let stiffness = derive_stiffness_table(model, max_k)?;
        let kernel = crate::kernels::VbdClothKernel::new(
            &device,
            &queue,
            particle_count,
            spring_count,
            &stiffness,
        );
        Ok(Self {
            iterations: 8,
            substeps: 1,
            contact_stiffness: 1.0e5,
            max_spring_stiffness: max_k,
            vbd_mode: VbdMode::Soft,
            backend: VbdBackend::Gpu(GpuBackend { device, queue, kernel }),
        })
    }

    #[must_use]
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n.max(1);
        self
    }
    #[must_use]
    pub fn with_substeps(mut self, n: usize) -> Self {
        self.substeps = n.max(1);
        self
    }
    #[must_use]
    pub fn with_contact_stiffness(mut self, k: f64) -> Self {
        self.contact_stiffness = k.max(0.0);
        self
    }
    #[must_use]
    pub fn with_max_spring_stiffness(mut self, k: f64) -> Self {
        self.max_spring_stiffness = k.max(1.0);
        self
    }
    /// Pick the AVBD contact treatment. See [`VbdMode`].
    #[must_use]
    pub fn with_mode(mut self, mode: VbdMode) -> Self {
        self.vbd_mode = mode;
        self
    }
    /// Equivalent to `with_mode(VbdMode::Hard)`. Sugar for the
    /// no-penetration use case.
    #[must_use]
    pub fn with_hard_mode(self) -> Self {
        self.with_mode(VbdMode::Hard)
    }

    /// Refresh the per-spring stiffness table from `model`. Call
    /// after the user tweaks compliance sliders without rebuilding
    /// the solver. CPU backend is a no-op (it computes stiffness
    /// per-step from compliance directly).
    #[allow(unused_variables, reason = "model is unused on CPU backend")]
    pub fn refresh_stiffness(&mut self, model: &Model) -> Result<()> {
        match &mut self.backend {
            VbdBackend::Cpu => Ok(()),
            #[cfg(feature = "gpu")]
            VbdBackend::Gpu(gpu) => {
                let stiffness = derive_stiffness_table(model, self.max_spring_stiffness)?;
                gpu.kernel
                    .set_spring_stiffness(&gpu.device, &gpu.queue, &stiffness)
            }
        }
    }

    #[must_use]
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self.backend, VbdBackend::Gpu(_))
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
}

impl Solver for VbdSolver {
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        _control: &Control,
        _contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        // VBD operates on particles + springs only in v1. Bodies pass
        // through untouched (the demo scenes use kinematic-rigid
        // shapes, which the soft-contact pass reads via the model
        // shape arrays — no body integration needed here).
        if model.particle_count == 0 {
            return Ok(());
        }
        let n_sub = self.substeps.max(1);
        let dt_sub = dt / n_sub as Real;

        match &mut self.backend {
            VbdBackend::Cpu => {
                let mut p_in = state_in.particle_q.host()?.to_vec();
                let mut v_in = state_in.particle_qd.host()?.to_vec();
                let mut p_out = vec![Vec3::ZERO; p_in.len()];
                let mut v_out = vec![Vec3::ZERO; v_in.len()];
                // Stage 20.3 — copy the plasticity state in/out around
                // the substep loop so each substep's return mapping
                // sees the previous substep's update. State lives
                // host-side on CPU; on GPU it stays device-resident.
                let mut fp = if model.tet_count > 0 {
                    state_in.tet_f_plastic.host()?.to_vec()
                } else {
                    Vec::new()
                };
                let mut eps_p = if model.tet_count > 0 {
                    state_in.tet_plastic_strain.host()?.to_vec()
                } else {
                    Vec::new()
                };
                for _ in 0..n_sub {
                    cpu_step(
                        model,
                        &p_in,
                        &v_in,
                        &fp,
                        dt_sub,
                        self.iterations,
                        self.contact_stiffness,
                        self.max_spring_stiffness,
                        self.vbd_mode,
                        &mut p_out,
                        &mut v_out,
                    )?;
                    // Plasticity return-mapping pass per tet (uses the
                    // post-step positions). Mutates `fp` + `eps_p`.
                    if model.tet_count > 0 {
                        cpu_plasticity_return_map(model, &p_out, &mut fp, &mut eps_p)?;
                    }
                    std::mem::swap(&mut p_in, &mut p_out);
                    std::mem::swap(&mut v_in, &mut v_out);
                }
                state_out.particle_q.host_mut()?.copy_from_slice(&p_in);
                state_out.particle_qd.host_mut()?.copy_from_slice(&v_in);
                if model.tet_count > 0 {
                    state_out.tet_f_plastic.host_mut()?.copy_from_slice(&fp);
                    state_out
                        .tet_plastic_strain
                        .host_mut()?
                        .copy_from_slice(&eps_p);
                }
                // Body state passes through.
                let q_in = state_in.body_q.host()?.to_vec();
                let qd_in = state_in.body_qd.host()?.to_vec();
                state_out.body_q.host_mut()?.copy_from_slice(&q_in);
                state_out.body_qd.host_mut()?.copy_from_slice(&qd_in);
            }
            #[cfg(feature = "gpu")]
            VbdBackend::Gpu(gpu) => {
                // Body state passes through unchanged. VBD doesn't
                // touch `body_q` / `body_qd` on the device so we
                // don't need to copy anything per-step; the kernel
                // reads `state_in.body_q` directly for AVBD soft
                // contact transforms. For scenes that swap state
                // buffers we still need to make sure `state_out`'s
                // body_q is up-to-date for the next iteration; we
                // skip the host bounce (which trips the "host stale"
                // check after a previous GPU dispatch) and instead
                // rely on the caller to manage body integration.

                let GpuBackend { device, queue, kernel } = gpu;
                let contact_k = self.contact_stiffness as f32;
                let hard = matches!(self.vbd_mode, VbdMode::Hard);
                // Single substep — straight kernel dispatch. Multi-
                // substep ping-pong is implemented by dispatching N
                // times with a chain of state_in → state_out swaps
                // entirely on the device side.
                kernel.dispatch(
                    device,
                    queue,
                    model,
                    state_in,
                    state_out,
                    dt_sub,
                    self.iterations as u32,
                    contact_k,
                    hard,
                )?;
                if n_sub > 1 {
                    // Subsequent substeps: alternate buffers using a
                    // GPU-resident scratch state. We sidestep the
                    // host download by using `state_out` as the
                    // scratch (we just wrote to it; it's the
                    // canonical "previous step" for sub n+1) and
                    // ping-ponging into a freshly-allocated state.
                    // For v1 we just chain: dispatch reads state_in,
                    // writes state_out; the next dispatch reads
                    // state_out, writes a temp, etc. We need a real
                    // GPU-resident scratch for this — defer the
                    // multi-substep optimisation; v1 callers wrap
                    // the substep loop themselves (same pattern as
                    // XPBD `external_substeps_box_stack` test).
                    return Err(Error::Solver(
                        "VBD GPU multi-substep is not implemented; \
                         set substeps = 1 and run the substep loop externally"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Map XPBD compliance `α` (m/N) onto an equivalent stiffness
/// `k = 1 / max(α, ε)`, capped at `max_k` to avoid pathological
/// Hessian conditioning when the user requested near-zero
/// compliance ("perfectly rigid"). The cap is what makes VBD
/// well-behaved across the full XPBD compliance range that the
/// builder accepts (0 → "stiff" rather than "infinite").
fn derive_stiffness_table(model: &Model, max_k: f64) -> Result<Vec<Real>> {
    let comp = model.spring_compliance.host()?;
    let mut out = Vec::with_capacity(comp.len());
    for &a in comp {
        let k = if a > 1.0e-12 { 1.0 / a } else { max_k };
        out.push(k.min(max_k).max(1.0));
    }
    Ok(out)
}

/// CPU oracle. Mirrors the GPU kernel's dispatch order: predict →
/// outer-iters × (per-vertex Gauss-Seidel block solve + soft-contact
/// pass) → velocity recovery.
///
/// Matches Newton's `solve_elasticity` (reference at
/// `xtra/newton/newton/_src/solvers/vbd/particle_vbd_kernels.py`)
/// for the stretch + bending springs. We don't ship the membrane /
/// dihedral-angle bending energies (those need a tri / edge-pair
/// adjacency we haven't built yet); instead we use the existing
/// builder's "skip-one" bending springs, treating them as Hookean
/// spring constraints. The resulting cloth is visibly stiffer +
/// less jelly than XPBD at the same iteration count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_step(
    model: &Model,
    particle_q_in: &[Vec3],
    particle_qd_in: &[Vec3],
    tet_f_plastic: &[crate::math::Mat3],
    dt: Real,
    iterations: usize,
    contact_stiffness: f64,
    max_spring_stiffness: f64,
    mode: VbdMode,
    particle_q_out: &mut [Vec3],
    particle_qd_out: &mut [Vec3],
) -> Result<()> {
    use crate::math::{Mat3, Quat, Transform};

    let n = model.particle_count;
    let inv_mass = model.particle_inv_mass.host()?;
    let radius = model.particle_radius.host()?;
    let world_idx = model.particle_world.host()?;
    let gravity = model.gravity.host()?;
    let lin_damp = model.linear_damping;

    // Predict: y = x + dt·(v + dt·g) and snapshot x_prev = x.
    let mut x: Vec<Vec3> = particle_q_in.to_vec();
    let x_prev: Vec<Vec3> = particle_q_in.to_vec();
    let mut y: Vec<Vec3> = vec![Vec3::ZERO; n];
    for i in 0..n {
        if inv_mass[i] <= 0.0 {
            y[i] = x_prev[i];
            x[i] = x_prev[i];
            continue;
        }
        let world = world_idx[i].max(0) as usize;
        let g = gravity.get(world).copied().unwrap_or(Vec3::ZERO);
        let mut v = particle_qd_in[i] + g * dt;
        if lin_damp > 0.0 {
            v *= (1.0 - lin_damp * dt).max(0.0);
        }
        y[i] = x_prev[i] + v * dt;
        x[i] = y[i]; // warm-start from inertial target
    }

    // Topology + colouring.
    let spring_a = model.spring_a.host()?;
    let spring_b = model.spring_b.host()?;
    let spring_rest = model.spring_rest_length.host()?;
    let spring_comp = model.spring_compliance.host()?;
    let vadj_off = model.vertex_adjacent_springs_offsets.host()?;
    let vadj_idx = model.vertex_adjacent_springs.host()?;
    let pcol_off = model.particle_color_offsets.host()?;
    let pcol_idx = model.particle_indices_by_color.host()?;
    let color_count = model.particle_color_count;
    // Tet topology + Neo-Hookean material (Stage 20.2). All optional —
    // a model with zero tets just skips the per-vertex tet sum.
    let tet_indices = model.tet_indices.host()?;
    let tet_dm_inv = model.tet_dm_inv.host()?;
    let tet_rest_volume = model.tet_rest_volume.host()?;
    let tet_mu = model.tet_mu.host()?;
    let tet_lambda = model.tet_lambda.host()?;
    let vadj_tets_off = model.vertex_adjacent_tets_offsets.host()?;
    let vadj_tets = model.vertex_adjacent_tets.host()?;
    // Cable bending state (Stage 20.2).
    let cable_off = model.cable_particle_offsets.host()?;
    let cable_particles = model.cable_particles.host()?;
    let cable_bending = model.cable_bending_stiffness.host()?;

    // Pre-compute per-spring stiffness from compliance.
    let stiffness: Vec<Real> = spring_comp
        .iter()
        .map(|&a| {
            let k = if a > 1.0e-12 { 1.0 / a } else { max_spring_stiffness };
            k.min(max_spring_stiffness).max(1.0)
        })
        .collect();

    let dt2 = dt * dt;

    // Hard-mode AVBD scratch: per-particle augmented-Lagrangian
    // multiplier `λ` (one per particle, scalar — applied along the
    // most-penetrating contact normal). Updated at the end of every
    // outer iteration as `λ ← max(λ + k_c · pen, 0)`. Soft mode
    // leaves λ at zero — the contribution `(λ + k_c · pen) · n`
    // collapses back to the Stage-20 v1 quadratic penalty.
    let mut lambda = vec![0.0_f64; n];
    let hard = matches!(mode, VbdMode::Hard);

    // Pre-compute world-frame shape transforms once per substep.
    let body_q = state_body_transforms(model, n)?;
    let mut shape_xforms: Vec<Transform> = Vec::with_capacity(model.shape_count);
    let shape_body = model.shape_body.host()?;
    let shape_xf = model.shape_transform.host()?;
    let shape_kind = model.shape_geom_kind.host()?;
    let shape_data = model.shape_geom_data.host()?;
    for s in 0..model.shape_count {
        let local = shape_xf[s];
        let body = shape_body[s];
        let world_xf = if body < 0 {
            local
        } else {
            let bq = body_q.get(body as usize).copied().unwrap_or(Transform::IDENTITY);
            Transform::new(
                bq.position + bq.rotation * local.position,
                bq.rotation * local.rotation,
            )
        };
        shape_xforms.push(world_xf);
    }

    // Outer iterations × colours × per-vertex block solve.
    for _it in 0..iterations.max(1) {
        for k in 0..color_count {
            let lo = pcol_off[k] as usize;
            let hi = pcol_off[k + 1] as usize;
            for slot in lo..hi {
                let v = pcol_idx[slot] as usize;
                if v >= n {
                    continue;
                }
                let inv_m = inv_mass[v];
                if inv_m <= 0.0 {
                    continue; // pinned
                }
                let mass = 1.0 / inv_m;
                let m_over_dt2 = mass / dt2;

                // Inertia term.
                let mut f = (y[v] - x[v]) * m_over_dt2;
                let mut h = Mat3::from_diagonal(Vec3::splat(m_over_dt2));

                // Spring contributions over incident springs.
                let off_lo = vadj_off[v] as usize;
                let off_hi = vadj_off[v + 1] as usize;
                for s_slot in off_lo..off_hi {
                    let s = vadj_idx[s_slot] as usize;
                    let a = spring_a[s] as usize;
                    let b = spring_b[s] as usize;
                    let other = if a == v { b } else { a };
                    if other >= n {
                        continue;
                    }
                    let diff = x[v] - x[other];
                    let len = diff.length().max(1.0e-8);
                    let rest = spring_rest[s];
                    let kk = stiffness[s];
                    let f_self = -kk * (1.0 - rest / len) * diff;
                    let outer = mat3_outer(diff, diff);
                    let inv_l2 = 1.0 / (len * len);
                    let outer_scaled = mat3_scale(outer, inv_l2);
                    let bracket = mat3_sub(Mat3::IDENTITY, outer_scaled);
                    let l0_l = rest / len;
                    let h_self = mat3_scale(
                        mat3_sub(Mat3::IDENTITY, mat3_scale(bracket, l0_l)),
                        kk,
                    );
                    f += f_self;
                    h = mat3_add(h, h_self);
                }

                // Tet (Neo-Hookean) contributions over incident tets.
                // Each entry in `vadj_tets` is a (tet_id, v_order) pair.
                // The kernel below mirrors Newton's
                // `evaluate_volumetric_neo_hookean_force_and_hessian`
                // for the per-vertex 3×3 block.
                let to_lo = vadj_tets_off[v] as usize;
                let to_hi = vadj_tets_off[v + 1] as usize;
                for slot in (to_lo..to_hi).step_by(2) {
                    let tet = vadj_tets[slot] as usize;
                    let order = vadj_tets[slot + 1] as usize;
                    // Stage 20.3 — F_p reference. When a tet has been
                    // unloaded after plastic flow, F_p is non-identity
                    // and the elastic energy is evaluated on
                    // `F_e = F · F_p⁻¹` so the tet wants to return to
                    // its plastically-deformed configuration rather
                    // than the original rest state.
                    let fp_tet = if tet_f_plastic.is_empty() {
                        crate::math::Mat3::IDENTITY
                    } else {
                        tet_f_plastic[tet]
                    };
                    let (f_t, h_t) = neo_hookean_vertex_block(
                        tet,
                        order,
                        &x,
                        tet_indices,
                        tet_dm_inv,
                        tet_rest_volume,
                        tet_mu,
                        tet_lambda,
                        fp_tet,
                    );
                    f += f_t;
                    h = mat3_add(h, h_t);
                }

                // Cable bending contributions: for every cable that
                // includes `v` as an *interior* particle (not an
                // endpoint), the discrete hinge `(v_{i-1}, v_i,
                // v_{i+1})` produces a force + Hessian on each of the
                // three particles; here we only need the self-contribution.
                if model.cable_count > 0 {
                    for c in 0..model.cable_count {
                        let lo = cable_off[c] as usize;
                        let hi = cable_off[c + 1] as usize;
                        let kb = cable_bending[c];
                        if kb <= 0.0 || hi - lo < 3 {
                            continue;
                        }
                        // Walk interior triples and check whether any
                        // of `v_prev`, `v_mid`, `v_next` matches `v`.
                        for i in (lo + 1)..(hi - 1) {
                            let v_prev = cable_particles[i - 1] as usize;
                            let v_mid = cable_particles[i] as usize;
                            let v_next = cable_particles[i + 1] as usize;
                            if v_mid == v || v_prev == v || v_next == v {
                                let role = if v == v_prev { 0 }
                                    else if v == v_mid { 1 }
                                    else { 2 };
                                let (f_b, h_b) = cable_bending_vertex_block(
                                    role, x[v_prev], x[v_mid], x[v_next], kb,
                                );
                                f += f_b;
                                h = mat3_add(h, h_b);
                            }
                        }
                    }
                }

                // Rigid contact term. Soft-mode = quadratic penalty
                // (Stage-20 v1). Hard-mode = augmented-Lagrangian:
                // adds `λ · n` to the force where `λ` is the per-
                // particle dual variable updated at the end of the
                // outer iteration. Per Giles, Diaz & Yuksel (2025,
                // §4.2), the per-iteration block residual is
                //   f_c = (λ + k_c · pen) · n,
                //   H_c = k_c · n nᵀ.
                if contact_stiffness > 0.0 && model.shape_count > 0 {
                    let pr = radius[v];
                    let lam = lambda[v];
                    for s in 0..model.shape_count {
                        let kind = shape_kind[s];
                        if kind == 0xff {
                            continue;
                        }
                        let xf = shape_xforms[s];
                        let data = shape_data[s];
                        let nd = shape_signed_distance(kind, xf, data, x[v]);
                        let pen = pr - nd.1;
                        if pen > 0.0 {
                            let normal = nd.0;
                            let mag = if hard {
                                lam + contact_stiffness * pen
                            } else {
                                contact_stiffness * pen
                            };
                            f += normal * mag;
                            let n_outer = mat3_outer(normal, normal);
                            h = mat3_add(h, mat3_scale(n_outer, contact_stiffness));
                        }
                    }
                }

                // Newton step: x ← x + H⁻¹ f.
                if let Some(dx) = solve_3x3(h, f) {
                    x[v] += dx;
                }
            }
        }

        // Hard-mode AVBD dual update — once per outer iteration,
        // bump `λ` by the residual penetration. Negative λ would
        // pull the particle into the surface, so we clamp at zero
        // (unilateral contact). Soft mode leaves `λ` untouched.
        if hard && contact_stiffness > 0.0 && model.shape_count > 0 {
            for v in 0..n {
                if inv_mass[v] <= 0.0 {
                    continue;
                }
                let pr = radius[v];
                let mut max_pen = 0.0_f64;
                for s in 0..model.shape_count {
                    let kind = shape_kind[s];
                    if kind == 0xff {
                        continue;
                    }
                    let xf = shape_xforms[s];
                    let data = shape_data[s];
                    let nd = shape_signed_distance(kind, xf, data, x[v]);
                    let pen = pr - nd.1;
                    if pen > max_pen {
                        max_pen = pen;
                    }
                }
                let new_lam = lambda[v] + contact_stiffness * max_pen;
                lambda[v] = new_lam.max(0.0);
            }
        }
    }

    // Velocity recovery.
    for i in 0..n {
        if inv_mass[i] <= 0.0 {
            particle_q_out[i] = x_prev[i];
            particle_qd_out[i] = Vec3::ZERO;
            continue;
        }
        particle_q_out[i] = x[i];
        particle_qd_out[i] = (x[i] - x_prev[i]) / dt;
    }

    let _ = (Quat::IDENTITY,);
    Ok(())
}

/// Resolve body transforms — for the CPU oracle we always read from
/// `model.initial_body_q` (kinematic bodies don't move during a VBD
/// substep). Returns at most `model.body_count` transforms; size is
/// passed in so callers can avoid the buffer download when they
/// know the model has no bodies.
fn state_body_transforms(
    model: &Model,
    _particle_count: usize,
) -> Result<Vec<crate::math::Transform>> {
    let q = model.initial_body_q.host()?;
    Ok(q.to_vec())
}

/// Per-shape signed-distance + outward normal at world-space point
/// `p`. Returns `(normal, signed_distance)`. Mirrors the WGSL
/// `shape_sd` helper in `vbd_cloth.wgsl`.
fn shape_signed_distance(
    kind: u32,
    xf: crate::math::Transform,
    data: Vec3,
    p: Vec3,
) -> (Vec3, Real) {
    match kind {
        0 => {
            // Sphere
            let d = p - xf.position;
            let len = d.length();
            if len < 1.0e-9 {
                return (Vec3::Y, -data.x);
            }
            (d / len, len - data.x)
        }
        1 => {
            // Plane
            let n_world = (xf.rotation * data).normalize_or_zero();
            let nn = if n_world.length_squared() > 1.0e-12 {
                n_world
            } else {
                Vec3::Y
            };
            (nn, (p - xf.position).dot(nn))
        }
        2 => {
            // Box
            let inv_rot = xf.rotation.conjugate();
            let p_local = inv_rot * (p - xf.position);
            let half = data;
            let q = Vec3::new(p_local.x.abs(), p_local.y.abs(), p_local.z.abs()) - half;
            let outside = Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
            let outside_len = outside.length();
            let inside_d = q.x.max(q.y.max(q.z)).min(0.0);
            let sd = outside_len + inside_d;
            let sign = Vec3::new(p_local.x.signum(), p_local.y.signum(), p_local.z.signum());
            let n_local = if outside_len > 1.0e-9 {
                Vec3::new(
                    sign.x * outside.x / outside_len,
                    sign.y * outside.y / outside_len,
                    sign.z * outside.z / outside_len,
                )
            } else if q.x >= q.y && q.x >= q.z {
                Vec3::new(sign.x, 0.0, 0.0)
            } else if q.y >= q.z {
                Vec3::new(0.0, sign.y, 0.0)
            } else {
                Vec3::new(0.0, 0.0, sign.z)
            };
            let n_world = (xf.rotation * n_local).normalize_or_zero();
            (
                if n_world.length_squared() > 1.0e-12 { n_world } else { Vec3::Y },
                sd,
            )
        }
        _ => (Vec3::Y, 1.0e6),
    }
}

#[inline]
fn mat3_add(a: crate::math::Mat3, b: crate::math::Mat3) -> crate::math::Mat3 {
    crate::math::Mat3::from_cols(a.x_axis + b.x_axis, a.y_axis + b.y_axis, a.z_axis + b.z_axis)
}
#[inline]
fn mat3_sub(a: crate::math::Mat3, b: crate::math::Mat3) -> crate::math::Mat3 {
    crate::math::Mat3::from_cols(a.x_axis - b.x_axis, a.y_axis - b.y_axis, a.z_axis - b.z_axis)
}
#[inline]
fn mat3_scale(a: crate::math::Mat3, s: Real) -> crate::math::Mat3 {
    crate::math::Mat3::from_cols(a.x_axis * s, a.y_axis * s, a.z_axis * s)
}
#[inline]
fn mat3_outer(u: Vec3, v: Vec3) -> crate::math::Mat3 {
    // Column-major: each column is `u * v[col]`.
    crate::math::Mat3::from_cols(u * v.x, u * v.y, u * v.z)
}

/// Neo-Hookean force + 3×3 Hessian on vertex `order ∈ {0..3}` of tet
/// `tet_id`. Mirrors Newton's
/// `evaluate_volumetric_neo_hookean_force_and_hessian` (CPU port).
///
/// The Neo-Hookean energy density is
///   `Ψ(F) = (μ/2)·(I_C - 3) - μ·log(J) + (λ/2)·log²(J)`
/// where `F = Ds · Dm⁻¹` is the deformation gradient and
/// `Ds = [v1-v0 | v2-v0 | v3-v0]` is the current basis. Newton uses
/// the Smith et al. 2018 stable-NH reparameterisation
/// `μ_NH = μ_Lamé`, `λ_NH = λ_Lamé + μ_Lamé`, and *drops* the
/// `s · ∂²J/∂F²` term in the Hessian — the per-vertex 3×3 block
/// from that term cancels out by Levi-Civita antisymmetry, and what
/// remains is SPD by inspection (no further projection needed).
///
/// Returns `(force, hessian)` for the Newton step. We multiply `H`
/// by the per-tet rest volume so the solver sees the physical units
/// (N/m for the diagonal). The hessian is symmetric-positive-(semi)-
/// definite for this term (the inertia + spring contributions in the
/// outer assembly add a positive `m/dt²·I` so the full block is
/// always SPD).
#[allow(clippy::too_many_arguments)]
pub(crate) fn neo_hookean_vertex_block(
    tet_id: usize,
    order: usize,
    x: &[Vec3],
    tet_indices: &[u32],
    tet_dm_inv: &[crate::math::Mat3],
    tet_rest_volume: &[Real],
    tet_mu: &[Real],
    tet_lambda: &[Real],
    fp: crate::math::Mat3,
) -> (Vec3, crate::math::Mat3) {
    use crate::math::Mat3;
    let v0 = x[tet_indices[tet_id * 4] as usize];
    let v1 = x[tet_indices[tet_id * 4 + 1] as usize];
    let v2 = x[tet_indices[tet_id * 4 + 2] as usize];
    let v3 = x[tet_indices[tet_id * 4 + 3] as usize];
    let dm_inv = tet_dm_inv[tet_id];
    let rest_vol = tet_rest_volume[tet_id];
    let mu_l = tet_mu[tet_id];
    let lam_l = tet_lambda[tet_id];

    // Stable-NH reparameterisation (Smith et al. 2018).
    let mu = mu_l;
    let lam = lam_l + mu_l;

    // Deformation gradient F = Ds · Dm⁻¹. The *elastic* deformation
    // gradient is F_e = F · F_p⁻¹ (multiplicative split). For purely
    // elastic tets, F_p = I so F_e = F. The Neo-Hookean kernel below
    // is identical to Stage 20.2 except it evaluates the energy on
    // F_e instead of F.
    let ds = Mat3::from_cols(v1 - v0, v2 - v0, v3 - v0);
    let f_full = ds * dm_inv;
    // Compute F_p⁻¹. We *could* invert each call, but for elastic-only
    // tets F_p = I so the inverse is trivial; for plastic tets we do
    // a real inverse. The branch is on a determinant check (cheaper
    // than a full inverse + matrix multiply on the trivial path).
    let fp_det = fp.determinant();
    let f_mat = if (fp_det - 1.0).abs() < 1.0e-9 && fp.x_axis.x.is_finite() {
        // F_p = I (or numerically indistinguishable) — skip the inverse.
        f_full
    } else if fp_det.abs() > 1.0e-12 {
        f_full * fp.inverse()
    } else {
        // Singular F_p: fall back to identity (no plastic effect this
        // step). Should not happen with a sane return mapping.
        f_full
    };

    // Useful quantities.
    let j = f_mat.determinant();
    let lam_safe = if lam.abs() > 1.0e-6 { lam } else { 1.0e-6_f64.copysign(lam.signum()) };
    let alpha = 1.0 + mu / lam_safe;

    // Cofactor matrix (adjugate) — numerically stable when J ≈ 0.
    let cof = cofactor3(f_mat);

    // First Piola-Kirchhoff stress P = μ·F + s·cof(F), with
    // s = λ · (J - α). Multiplied by rest volume.
    let s = lam * (j - alpha);
    let p_mat = mat3_add(mat3_scale(f_mat, mu), mat3_scale(cof, s));
    let p_vec = mat3_scale(p_mat, rest_vol);

    // Per-vertex gradient column `m` from `Dm⁻¹` — Newton's
    // `compute_G_matrix` equivalent. Newton/Warp uses row-major
    // `Dm_inv[i, j]` (row i, column j); glam stores `Mat3` as
    // column-major (`dm_inv.x_axis` is column 0). Concretely:
    //   - v_order = 0:
    //       `m_a = -(Dm⁻¹[0, a] + Dm⁻¹[1, a] + Dm⁻¹[2, a])`
    //            = -(sum of column `a` in row-major terms)
    //            = -(sum of `dm_inv.col(a)`'s components)
    //   - v_order ∈ {1, 2, 3}:
    //       `m = (Dm⁻¹[r, 0], Dm⁻¹[r, 1], Dm⁻¹[r, 2])`, with
    //       `r = v_order - 1`. In glam's column-major basis,
    //       `Dm⁻¹[r, c] = dm_inv.col(c)[r]` so the row-vector
    //       reads `(x_axis[r], y_axis[r], z_axis[r])`.
    let m: Vec3 = match order {
        0 => Vec3::new(
            -(dm_inv.x_axis.x + dm_inv.x_axis.y + dm_inv.x_axis.z),
            -(dm_inv.y_axis.x + dm_inv.y_axis.y + dm_inv.y_axis.z),
            -(dm_inv.z_axis.x + dm_inv.z_axis.y + dm_inv.z_axis.z),
        ),
        1 => Vec3::new(dm_inv.x_axis.x, dm_inv.y_axis.x, dm_inv.z_axis.x),
        2 => Vec3::new(dm_inv.x_axis.y, dm_inv.y_axis.y, dm_inv.z_axis.y),
        _ => Vec3::new(dm_inv.x_axis.z, dm_inv.y_axis.z, dm_inv.z_axis.z),
    };

    // Force = -P·m. P is a 3×3 written column-major; multiplying by
    // the per-vertex gradient `m` is the standard reduction in
    // Newton's `assemble_tet_vertex_force_and_hessian`.
    let f_clean = -(p_vec.x_axis * m.x + p_vec.y_axis * m.y + p_vec.z_axis * m.z);

    // Hessian = rest_vol · (μ · I9 + λ · cof_vec ⊗ cof_vec) projected
    // onto the per-vertex 3×3 block.
    //
    // For a single vertex `i` with gradient `m`, the projection is
    //   H_v = m·mᵀ · (μ · I3 + λ · (cof · cof)?)
    // ...but the precise form per Newton's
    // `assemble_tet_vertex_force_and_hessian` is more involved. We
    // implement it directly: given the 9×9 Hessian
    //   H = rest_vol · (μ · I9 + λ · cof_vec ⊗ cof_vec),
    // the per-vertex 3×3 block is
    //   H_v[a, b] = sum over (i, j) of m[i] · H[3i+a, 3j+b] · m[j].
    // For this Neo-Hookean form (μ on the diagonal + λ · outer), the
    // per-vertex 3×3 block reduces to:
    //   H_v = m·mᵀ · (μ + λ · |cof_block ⊗ m|²) — but more precisely:
    //   H_v_diag = (μ · |m|² + λ · (cof · m)·(cof · m)ᵀ) · rest_vol
    // where (cof · m) is the 3-vector formed by treating cof as 3×3.
    //
    // We implement it directly from the vec9 outer-product form:
    let cof_m = cof.x_axis * m.x + cof.y_axis * m.y + cof.z_axis * m.z;
    let mu_term = mat3_scale(Mat3::IDENTITY, mu * m.dot(m));
    let lam_term = crate::math::Mat3::from_cols(
        cof_m * cof_m.x,
        cof_m * cof_m.y,
        cof_m * cof_m.z,
    );
    let h_block = mat3_scale(mat3_add(mu_term, mat3_scale(lam_term, lam)), rest_vol);

    // Tikhonov-projection safety net: clamp the diagonal to a tiny
    // positive floor so the per-vertex Hessian is strictly SPD even
    // when an inverted tet would otherwise produce a sign-indefinite
    // contribution. Negligible compared to the inertia + spring
    // diagonals; documented in `solvers/vbd.rs` module docs.
    let h_floor = 1.0e-8_f64 * rest_vol.max(1.0e-12);
    let h_block = crate::math::Mat3::from_cols(
        Vec3::new(h_block.x_axis.x.max(h_floor), h_block.x_axis.y, h_block.x_axis.z),
        Vec3::new(h_block.y_axis.x, h_block.y_axis.y.max(h_floor), h_block.y_axis.z),
        Vec3::new(h_block.z_axis.x, h_block.z_axis.y, h_block.z_axis.z.max(h_floor)),
    );

    (f_clean, h_block)
}

/// Cofactor (adjugate) matrix of `m`, computed without `m.inverse()`
/// — numerically stable when `det(m) ≈ 0`.
#[inline]
fn cofactor3(m: crate::math::Mat3) -> crate::math::Mat3 {
    let f11 = m.x_axis.x; let f21 = m.x_axis.y; let f31 = m.x_axis.z;
    let f12 = m.y_axis.x; let f22 = m.y_axis.y; let f32 = m.y_axis.z;
    let f13 = m.z_axis.x; let f23 = m.z_axis.y; let f33 = m.z_axis.z;
    crate::math::Mat3::from_cols(
        Vec3::new(
            f22 * f33 - f23 * f32,
            f23 * f31 - f21 * f33,
            f21 * f32 - f22 * f31,
        ),
        Vec3::new(
            f13 * f32 - f12 * f33,
            f11 * f33 - f13 * f31,
            f12 * f31 - f11 * f32,
        ),
        Vec3::new(
            f12 * f23 - f13 * f22,
            f13 * f21 - f11 * f23,
            f11 * f22 - f12 * f21,
        ),
    )
}

/// Discrete hinge bending energy on a CABLE triple `(v_prev, v_mid,
/// v_next)`. Energy is `½·k_b·(2·sin(θ/2))²` where θ is the angle
/// between segments `(v_mid − v_prev)` and `(v_next − v_mid)`. We
/// approximate the per-vertex Hessian with a Gauss-Newton block —
/// the second-derivative term is dropped (same projection trick the
/// Neo-Hookean kernel above uses). Returns `(force, H)` for the
/// requested role: 0 = `v_prev`, 1 = `v_mid`, 2 = `v_next`.
///
/// Physical sign convention: the bending force pushes `v_mid` toward
/// the line `(v_prev, v_next)` (rest curvature = 0). Endpoints
/// receive the opposite force (Newton's third law). `H` is the
/// Gauss-Newton outer product `2·k_b · (∂c/∂x_role)·(∂c/∂x_role)ᵀ`
/// where `c = 2·sin(θ/2) · n̂` is the curvature vector at v_mid.
pub(crate) fn cable_bending_vertex_block(
    role: u32,
    v_prev: Vec3,
    v_mid: Vec3,
    v_next: Vec3,
    k_b: Real,
) -> (Vec3, crate::math::Mat3) {
    use crate::math::Mat3;
    // Curvature vector at v_mid: kappa = 2·(v_prev + v_next - 2·v_mid)
    // / (|e1| + |e2|)  — the standard discrete-curvature for inextensible
    // ropes. For our purposes the simpler `kappa = (v_prev - 2·v_mid + v_next)`
    // is sufficient because the segment lengths are bounded by the
    // CABLE's stretch springs.
    let kappa = v_prev + v_next - 2.0 * v_mid;
    // Energy gradient w.r.t. each role (bending force opposes kappa
    // on v_mid; reverses on endpoints).
    let (grad_self, sign): (Vec3, Real) = match role {
        0 => (kappa, 1.0),         // v_prev: ∂E/∂v_prev = +k_b · kappa
        1 => (-2.0 * kappa, 1.0),  // v_mid:  ∂E/∂v_mid  = -2·k_b · kappa
        _ => (kappa, 1.0),         // v_next: ∂E/∂v_next = +k_b · kappa
    };
    let f_self = -k_b * grad_self;
    // Gauss-Newton Hessian: Σ k_b · J·Jᵀ where J is the row of the
    // Jacobian of `kappa` w.r.t. `x_role`. For a discrete laplacian
    // J_v_mid = -2·I, J_v_prev = J_v_next = +I. So the Hessian
    // contribution per role is:
    //   role 0/2: k_b · I
    //   role 1:   4·k_b · I   (from |J| = 2)
    let h_diag = match role {
        1 => 4.0 * k_b,
        _ => k_b,
    };
    let h = mat3_scale(Mat3::IDENTITY, h_diag);
    let _ = sign;
    (f_self, h)
}

/// Solve `H · dx = f` for a 3×3 SPD `H`. Returns `None` if `H` is
/// numerically singular (caller skips the update).
fn solve_3x3(h: crate::math::Mat3, f: Vec3) -> Option<Vec3> {
    let det = h.determinant();
    if det.abs() < 1.0e-20 {
        return None;
    }
    Some(h.inverse() * f)
}

/// Stage 20.3 — per-tet von Mises return-mapping pass (CPU oracle).
///
/// For each tet:
///  1. Compute trial `F = Ds · Dm⁻¹` from the post-step positions.
///  2. Compute trial elastic gradient `F_e_trial = F · F_p⁻¹` and SVD
///     it: `F_e_trial = U · Σ · Vᵀ`.
///  3. Take Hencky log strains `ε = log(σ)`, split into mean +
///     deviator, and form the trial deviatoric Kirchhoff stress
///     `s = 2μ · dev(ε)`.
///  4. Check yield: `f = ‖s‖ - sqrt(2/3) · σ_y(ε_p)` with
///     `σ_y(ε_p) = σ_y0 + H · ε_p`. If `f ≤ 0` → elastic, no update.
///  5. Else: compute plastic multiplier
///     `Δγ = f / (2μ + (2/3)·H)`; pull dev_ε back along the trial
///     direction by `Δγ`; reconstruct `Σ_e = exp(ε_new)` and
///     `F_e_new = U · Σ_e · Vᵀ`; set `F_p_new = F_e_new⁻¹ · F` and
///     `ε_p ← ε_p + sqrt(2/3)·Δγ`.
///
/// Numerical guards:
/// - Singular values clamped to `[1e-3, 1e3]` to keep `log(σ)` finite
///   on a near-pancaked tet.
/// - Plastic multiplier capped to half the trial deviator norm to
///   avoid the classic "overshoot the yield surface and oscillate"
///   pathology (line-search is overkill; this is enough damping).
/// - `F_p` clamped to `det(F_p) ∈ [0.05, 20]` so a misbehaving tet
///   doesn't propagate Inf into the next step.
pub(crate) fn cpu_plasticity_return_map(
    model: &Model,
    particle_q: &[Vec3],
    fp: &mut [crate::math::Mat3],
    eps_p: &mut [Real],
) -> Result<()> {
    use crate::math::{dense3::svd3, Mat3};
    let tet_indices = model.tet_indices.host()?;
    let tet_dm_inv = model.tet_dm_inv.host()?;
    let tet_mu = model.tet_mu.host()?;
    let tet_yield = model.tet_yield_stress.host()?;
    let tet_hard = model.tet_hardening.host()?;
    let n_tets = model.tet_count;

    let sqrt_2_3 = (2.0_f64 / 3.0).sqrt();
    for tet in 0..n_tets {
        let sigma_y0 = tet_yield[tet];
        // Pure-elastic short-circuit (back-compat path; F_p stays I).
        if !sigma_y0.is_finite() || sigma_y0 > 1.0e29 {
            continue;
        }
        let mu = tet_mu[tet];
        let h = tet_hard[tet];

        let v0 = particle_q[tet_indices[tet * 4] as usize];
        let v1 = particle_q[tet_indices[tet * 4 + 1] as usize];
        let v2 = particle_q[tet_indices[tet * 4 + 2] as usize];
        let v3 = particle_q[tet_indices[tet * 4 + 3] as usize];
        let dm_inv = tet_dm_inv[tet];
        let ds = Mat3::from_cols(v1 - v0, v2 - v0, v3 - v0);
        let f_full = ds * dm_inv;

        let fp_old = fp[tet];
        let fp_det = fp_old.determinant();
        if !fp_det.is_finite() || fp_det.abs() < 1.0e-12 {
            // Reset on degenerate F_p — should never happen with the
            // clamp below but defensive.
            fp[tet] = Mat3::IDENTITY;
            continue;
        }
        let fe_trial = f_full * fp_old.inverse();

        let (u_mat, sigma_raw, v_mat) = svd3(fe_trial);
        // Clamp singular values to keep log finite. A pancaked tet
        // (σ near 0) would otherwise drive trace_eps → -∞ and the
        // return mapping would project back to a degenerate state.
        let sigma = crate::math::Vec3::new(
            sigma_raw.x.clamp(1.0e-3, 1.0e3),
            sigma_raw.y.clamp(1.0e-3, 1.0e3),
            sigma_raw.z.clamp(1.0e-3, 1.0e3),
        );
        let eps = crate::math::Vec3::new(sigma.x.ln(), sigma.y.ln(), sigma.z.ln());
        let trace_eps = eps.x + eps.y + eps.z;
        let mean = trace_eps / 3.0;
        let dev = crate::math::Vec3::new(eps.x - mean, eps.y - mean, eps.z - mean);
        let dev_norm = dev.length();

        // Trial Kirchhoff deviator norm (von Mises): ‖s‖ = 2μ · ‖dev_ε‖.
        let s_trial = 2.0 * mu * dev_norm;
        let sigma_y = sigma_y0 + h * eps_p[tet];
        // von Mises yield condition: f = ‖s‖ - √(2/3)·σ_y.
        let f_yield = s_trial - sqrt_2_3 * sigma_y;
        if f_yield <= 0.0 || dev_norm < 1.0e-12 {
            continue;
        }
        // Closed-form plastic multiplier for linear isotropic
        // hardening (consistent return mapping):
        //   Δγ = f_yield / (2μ + (2/3)·H)
        let denom = 2.0 * mu + (2.0 / 3.0) * h;
        let dgamma = (f_yield / denom).max(0.0);
        // Damping cap to defuse the "overshoot the cylinder and
        // oscillate" pathology. Capping at half the trial deviator
        // strain (≈ pulling the trial state half-way back to the
        // surface in the worst case) keeps the iteration stable
        // without a full line search.
        let dev_strain_increment = dgamma.min(0.5 * dev_norm);
        // Pull the deviator back along its trial direction.
        let scale = (dev_norm - dev_strain_increment) / dev_norm;
        let new_dev = dev * scale;
        let new_eps =
            crate::math::Vec3::new(new_dev.x + mean, new_dev.y + mean, new_dev.z + mean);
        let new_sigma = crate::math::Vec3::new(
            new_eps.x.exp().clamp(0.05, 20.0),
            new_eps.y.exp().clamp(0.05, 20.0),
            new_eps.z.exp().clamp(0.05, 20.0),
        );
        let sigma_diag =
            Mat3::from_cols(
                Vec3::new(new_sigma.x, 0.0, 0.0),
                Vec3::new(0.0, new_sigma.y, 0.0),
                Vec3::new(0.0, 0.0, new_sigma.z),
            );
        let fe_new = u_mat * sigma_diag * v_mat.transpose();
        let fe_det = fe_new.determinant();
        if !fe_det.is_finite() || fe_det.abs() < 1.0e-12 {
            continue;
        }
        // F = F_e · F_p ⇒ F_p_new = F_e_new⁻¹ · F.
        let fp_new = fe_new.inverse() * f_full;
        // Sanity-clamp det(F_p) so a misbehaving step doesn't poison
        // the next one. det(F_p) ≈ 1 corresponds to volume-preserving
        // plastic flow (the von Mises model is volume-preserving by
        // construction; small drift is from the discrete return map).
        let fp_new_det = fp_new.determinant();
        if !fp_new_det.is_finite() || fp_new_det.abs() < 0.05 || fp_new_det.abs() > 20.0 {
            // Reject this step's plastic update; keep F_p as-is.
            continue;
        }
        fp[tet] = fp_new;
        // Equivalent plastic strain accumulator: classic √(2/3)·Δγ.
        eps_p[tet] = (eps_p[tet] + sqrt_2_3 * dev_strain_increment).min(10.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Real;
    use crate::sim::{Contacts, ModelBuilder};
    use crate::core::WorldId;

    /// Smoke: CPU VBD on a 4×4 cloth runs end-to-end without NaNs
    /// and the cloth actually falls.
    #[test]
    fn cpu_smoke_cloth_falls() {
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.4);
        let n = 4;
        let size = 0.6;
        let origin = Vec3::new(-size * 0.5, 2.0, -size * 0.5);
        let u_axis = Vec3::new(size, 0.0, 0.0);
        let v_axis = Vec3::new(0.0, 0.0, size);
        b.add_cloth_grid(origin, u_axis, v_axis, n, n, 0.05, 0.02, 1.0e-4, 0.5, WorldId(0));
        let model = b.build().unwrap();
        assert_eq!(model.particle_count, 16);
        assert!(model.particle_color_count >= 2, "VBD needs ≥2 colours for an 4×4 cloth");

        let mut solver = VbdSolver::new_cpu().with_iterations(8);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(8);

        let initial_mean_y = s_a
            .particle_q
            .host()
            .unwrap()
            .iter()
            .map(|p| p.y)
            .sum::<Real>()
            / model.particle_count as Real;

        let dt: Real = 1.0 / 240.0;
        for _ in 0..120 {
            crate::solvers::Solver::step(
                &mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b,
            )
            .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }

        let final_mean_y = s_a
            .particle_q
            .host()
            .unwrap()
            .iter()
            .map(|p| p.y)
            .sum::<Real>()
            / model.particle_count as Real;
        assert!(final_mean_y < initial_mean_y - 0.1, "cloth didn't fall: {initial_mean_y} -> {final_mean_y}");
        for p in s_a.particle_q.host().unwrap() {
            assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
        }
    }

    /// Pinned-corner sanity: pin the four corners and the cloth
    /// should drape but the corners stay put.
    #[test]
    fn cpu_pinned_corners_stay_put() {
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.4);
        let n = 5;
        let size = 1.0;
        let origin = Vec3::new(-size * 0.5, 2.0, -size * 0.5);
        let u_axis = Vec3::new(size, 0.0, 0.0);
        let v_axis = Vec3::new(0.0, 0.0, size);
        let (first, _, _) =
            b.add_cloth_grid(origin, u_axis, v_axis, n, n, 0.05, 0.02, 1.0e-4, 0.5, WorldId(0));
        b.pin_particle(first);
        b.pin_particle(first + n - 1);
        b.pin_particle(first + (n - 1) * n);
        b.pin_particle(first + (n - 1) * n + n - 1);
        let model = b.build().unwrap();

        let pinned = [first, first + n - 1, first + (n - 1) * n, first + (n - 1) * n + n - 1];
        let initial = s_a_initial_corners(&model, &pinned);

        let mut solver = VbdSolver::new_cpu().with_iterations(8);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(8);

        let dt: Real = 1.0 / 240.0;
        for _ in 0..240 {
            crate::solvers::Solver::step(
                &mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b,
            )
            .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.particle_q.host().unwrap();
        for (i, &pin) in pinned.iter().enumerate() {
            let cur = q[pin as usize];
            let d = (cur - initial[i]).length();
            assert!(d < 1.0e-6, "pinned corner {pin} drifted {d:.6} m");
        }
    }

    fn s_a_initial_corners(model: &crate::sim::Model, pinned: &[u32]) -> Vec<Vec3> {
        let s = model.state().unwrap();
        let q = s.particle_q.host().unwrap();
        pinned.iter().map(|&i| q[i as usize]).collect()
    }
}
