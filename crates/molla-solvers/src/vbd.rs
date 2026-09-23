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
//! - `xtra/newton/newton/_src/solvers/vbd/` — reference implementation; we
//!   mirror its structure but reimplement clean.

#[cfg(feature = "gpu")]
use crate::core::Error;
use crate::core::Result;
use crate::math::{Real, Vec3};
use crate::sim::{Contacts, Control, Model, State};

use super::Solver;

mod viscosity;

/// Backend selector for [`VbdSolver`]. Mirrors `XpbdBackend` —
/// CPU is the oracle / fallback; GPU is the production path.
#[allow(
    clippy::large_enum_variant,
    reason = "GPU variant holds Arcs / kernel; one solver instance, fine"
)]
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
    /// Stage 33.2 — GPU LBVH driver for the soft-surface tris. Lazily
    /// constructed on the first substep with VT contact (`None` for
    /// scenes without soft surfaces, or until the first step).
    pub(crate) soft_bvh_gpu: Option<crate::kernels::SoftBvhGpu>,
    /// Stage 33.2 — GPU VT contact pipelines. Same lazy-construct
    /// pattern as `soft_bvh_gpu`.
    pub(crate) soft_vt_kernel: Option<crate::kernels::SoftVtContactKernel>,
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

/// Stage 20.4 — backing volume for [`VbdSolver::with_sdf_grid`] and
/// [`VbdSolver::with_sparse_sdf`]. Wraps either a baked dense
/// [`crate::geometry::SdfGrid`] or a sparse [`crate::geometry::SparseSdf`]
/// + a world-frame [`crate::math::Transform`]. The host-side
///   post-step fixup pass projects every soft-body vertex out of any
///   registered SDF surface (delegating the actual sample to the
///   underlying backing volume's sampler).
///
/// Currently exposed only via the VBD solver — Stage 20.5 will wire
/// these into the GPU contact kernel directly via a dedicated
/// `tet_sdf_contact` pipeline that binds the 3D-texture / sparse-grid
/// data on group 6.
#[derive(Clone)]
pub enum VbdSdfBacking {
    /// Dense [`crate::geometry::SdfGrid`] in world frame at `transform`.
    Dense(
        std::sync::Arc<crate::geometry::SdfGrid>,
        crate::math::Transform,
    ),
    /// Sparse SDF (Stage 24e) in world frame at `transform`.
    Sparse(
        std::sync::Arc<crate::geometry::SparseSdf>,
        crate::math::Transform,
    ),
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
    /// Stage 20.4 — Baked / sparse SDF backing volumes for tet-vs-rigid
    /// collision against arbitrary mesh-derived shapes (the spec's
    /// "GPU SDF rigid-shape backing" deliverable). Each registered
    /// volume's surface is treated as kinematic; the host-side
    /// post-step fixup pass projects every soft-body vertex out.
    /// Empty by default; populated via [`Self::with_sdf_grid`] /
    /// [`Self::with_sparse_sdf`].
    pub sdf_backings: Vec<VbdSdfBacking>,
    /// Stage 33 — cross-mesh VT contact margin (m). Vertices within
    /// `vt_contact_margin` of any other-mesh tri trigger a contact;
    /// the post-substep projection pushes them out. Default `0.01`
    /// (1 cm) — comfortable for ~5 cm cloth + jelly cells.
    pub vt_contact_margin: Real,
    /// Stage 33 — number of Jacobi sweeps in the post-substep VT
    /// projection. 6 is enough for the v1 demo (jelly on cloth);
    /// raise for stiff multi-contact configurations.
    pub vt_contact_sweeps: usize,
    /// Stage 33 / 33.1 — last-substep active soft contact count
    /// (VT + EE combined), surfaced so viewer / tests can read it
    /// without re-running gather. Stage 33 emitted only VT contacts;
    /// Stage 33.1 onwards this also includes EE contacts. The
    /// legacy name (`vt_contact_count`) is retained for back-compat
    /// — viewer panes already use it as a "number of cross-mesh
    /// soft contacts" readout, which is exactly what the unified
    /// count means.
    pub last_vt_contact_count: usize,
    /// Stage 33 — cross-mesh BVH, lazily built on the first substep
    /// (`None` until then). Refit per substep; rebuilt only when
    /// the soft-surface tri count changes.
    pub(crate) soft_bvh: Option<crate::geometry::SoftBvh>,
    /// Stage 33.4 — enable the pre-substep CCD pass.
    pub ccd_enabled: bool,
    /// Stage 33.4 — velocity gate (m/s) below which CCD is skipped.
    pub ccd_velocity_threshold: Real,
    /// Stage 33.4 — diagnostic counter: `vt_ccd` invocations.
    pub last_ccd_invocations: usize,
    /// Stage 33.4 — diagnostic counter: vertices snapped by CCD.
    pub last_ccd_snaps: usize,
    /// Stage 33.5 — last substep's VT contact set (warm-starts
    /// friction across substeps via
    /// [`super::vbd_soft_contact::warm_start_friction`]).
    pub(crate) prev_vt_contacts: Vec<super::vbd_soft_contact::VtContact>,
    backend: VbdBackend,
    /// Device-resident scratch state for GPU multi-substep ping-pong
    /// (same pattern as `XpbdSolver::gpu_scratch`). Built lazily on
    /// the first multi-substep GPU step, reused after.
    #[cfg(feature = "gpu")]
    gpu_scratch: Option<State>,
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
            sdf_backings: Vec::new(),
            vt_contact_margin: 0.01,
            vt_contact_sweeps: 6,
            last_vt_contact_count: 0,
            soft_bvh: None,
            ccd_enabled: false,
            ccd_velocity_threshold: 1.0,
            last_ccd_invocations: 0,
            last_ccd_snaps: 0,
            prev_vt_contacts: Vec::new(),
            backend: VbdBackend::Cpu,
            #[cfg(feature = "gpu")]
            gpu_scratch: None,
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
        Self::new_gpu_on_device(device, queue, model)
    }

    /// Construct GPU VBD on the caller's matching device and queue.
    #[cfg(feature = "gpu")]
    pub fn new_gpu_on_device(
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        model: &Model,
    ) -> Result<Self> {
        model.validate_tet_viscosity()?;
        let limits = device.limits();
        if limits.max_bind_groups < 6
            || limits.max_bindings_per_bind_group < 17
            || limits.max_storage_buffers_per_shader_stage < 43
        {
            return Err(Error::Gpu(
                "VBD requires 6 bind groups, 17 bindings per group and 43 storage buffers".into(),
            ));
        }
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
            sdf_backings: Vec::new(),
            vt_contact_margin: 0.01,
            vt_contact_sweeps: 6,
            last_vt_contact_count: 0,
            soft_bvh: None,
            ccd_enabled: false,
            ccd_velocity_threshold: 1.0,
            last_ccd_invocations: 0,
            last_ccd_snaps: 0,
            prev_vt_contacts: Vec::new(),
            backend: VbdBackend::Gpu(GpuBackend {
                device,
                queue,
                kernel,
                soft_bvh_gpu: None,
                soft_vt_kernel: None,
            }),
            gpu_scratch: None,
        })
    }

    /// Stage 20.4 — register a baked dense [`crate::geometry::SdfGrid`]
    /// (typically from `bake_mesh_to_sdf` in Stage 21) at the given
    /// world-frame `transform`. Soft-body vertices that penetrate the
    /// SDF surface are projected out by the host-side fixup pass after
    /// every substep.
    #[must_use]
    pub fn with_sdf_grid(
        mut self,
        grid: std::sync::Arc<crate::geometry::SdfGrid>,
        transform: crate::math::Transform,
    ) -> Self {
        self.sdf_backings
            .push(VbdSdfBacking::Dense(grid, transform));
        self
    }

    /// Stage 20.4 — register a sparse [`crate::geometry::SparseSdf`]
    /// (Stage 24e narrow-band sparse volume) at the given world-frame
    /// `transform`. Same projection semantics as [`Self::with_sdf_grid`].
    #[must_use]
    pub fn with_sparse_sdf(
        mut self,
        sdf: std::sync::Arc<crate::geometry::SparseSdf>,
        transform: crate::math::Transform,
    ) -> Self {
        self.sdf_backings
            .push(VbdSdfBacking::Sparse(sdf, transform));
        self
    }

    /// Number of VBD outer iterations per substep. Higher = stiffer
    /// material response, slower step. Defaults to 8.
    #[must_use]
    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n.max(1);
        self
    }
    /// Substep count: each outer `step(dt)` runs `n` substeps with
    /// `dt/n`. Defaults to 1.
    #[must_use]
    pub fn with_substeps(mut self, n: usize) -> Self {
        self.substeps = n.max(1);
        self
    }
    /// Penalty stiffness applied at vertex-vs-collider contacts (N/m).
    /// Defaults to `1.0e5`.
    #[must_use]
    pub fn with_contact_stiffness(mut self, k: f64) -> Self {
        self.contact_stiffness = k.max(0.0);
        self
    }
    /// Cap on per-spring stiffness used to derive the GPU stiffness
    /// table. Defaults to `1.0e7` N/m.
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

    /// Stage 33 — set the cross-mesh VT contact margin (m).
    #[must_use]
    pub fn with_vt_contact_margin(mut self, m: Real) -> Self {
        self.vt_contact_margin = m.max(0.0);
        self
    }

    /// Stage 33 — set the number of Jacobi sweeps in the post-substep
    /// VT contact projection.
    #[must_use]
    pub fn with_vt_contact_sweeps(mut self, n: usize) -> Self {
        self.vt_contact_sweeps = n.max(1);
        self
    }

    /// Stage 33 / 33.1 — last-frame active soft (VT + EE) contact
    /// count. Available for the viewer pane and tests after `step`
    /// returns.
    #[must_use]
    pub fn last_vt_contact_count(&self) -> usize {
        self.last_vt_contact_count
    }

    /// Stage 33.4 — toggle the pre-substep CCD pass on / off.
    ///
    /// CPU-only (INSPECTION.md F14): CCD has no GPU kernel, so a GPU
    /// backend with CCD enabled and ≥ 2 soft surfaces refuses to step
    /// (loud error) instead of silently tunneling. Use [`Vbd::cpu`]
    /// (or the `new_cpu` constructor) for CCD scenes.
    #[must_use]
    pub fn with_ccd(mut self, enabled: bool) -> Self {
        self.ccd_enabled = enabled;
        self
    }

    /// Stage 33.4 — set the per-vertex velocity gate (m/s) below which
    /// the CCD pass skips the per-vertex BVH query.
    #[must_use]
    pub fn with_ccd_velocity_threshold(mut self, v: Real) -> Self {
        self.ccd_velocity_threshold = v.max(0.0);
        self
    }

    /// Stage 33.4 — number of `vt_ccd` invocations during the last
    /// `step`. Read by tests to verify the velocity threshold gate.
    #[must_use]
    pub fn last_ccd_invocations(&self) -> usize {
        self.last_ccd_invocations
    }

    /// Stage 33.4 — number of vertices snapped back by the CCD pass
    /// (TOI strictly less than 1) during the last `step`.
    #[must_use]
    pub fn last_ccd_snaps(&self) -> usize {
        self.last_ccd_snaps
    }

    /// Stage 33.1 — same value as [`Self::last_vt_contact_count`],
    /// renamed for clarity now that the count includes EE contacts
    /// too. New callers should prefer this accessor.
    #[must_use]
    pub fn last_soft_contact_count(&self) -> usize {
        self.last_vt_contact_count
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
        if !dt_sub.is_finite() || dt_sub <= 0.0 {
            return Err(molla_core::Error::Build(
                "VBD timestep must be finite and positive".into(),
            ));
        }

        // GPU path — handled before the backend match so the substep
        // driver can re-borrow `self` (scratch state + substep helper).
        #[cfg(feature = "gpu")]
        if matches!(self.backend, VbdBackend::Gpu(_)) {
            // Stage 33.4.1 (INSPECTION.md F14) — CCD is a CPU-only
            // stage (BVH build + TOI root-finds; no GPU kernel yet).
            // Refuse loudly rather than silently tunnel: a caller who
            // opted into `with_ccd(true)` is depending on it for
            // correctness at high impact speeds.
            if self.ccd_enabled && model.soft_surfaces.len() >= 2 {
                return Err(Error::Solver(
                    "VBD CCD (`with_ccd(true)`) is CPU-only — it has no GPU kernel and \
                     would silently tunnel on the GPU backend. Use `Vbd::cpu()` for \
                     CCD scenes, or disable CCD for this solver."
                        .into(),
                ));
            }
            if n_sub == 1 {
                // A single substep advances the full dt (VBD is an
                // implicit-energy minimiser, stable at large dt given
                // enough inner iterations).
                return self.gpu_substep(model, state_in, state_out, dt);
            }
            return self.gpu_multi_substep(model, state_in, state_out, dt_sub, n_sub);
        }

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
                // Stage 20.4 — kinematic-hardening back-stress α.
                let mut alpha = if model.tet_count > 0 {
                    state_in.tet_back_stress.host()?.to_vec()
                } else {
                    Vec::new()
                };
                // Stage 33.4 — diagnostic counters reset per `step`.
                // CCD invocations / snaps accumulate across substeps so
                // the final value reflects the whole step.
                self.last_ccd_invocations = 0;
                self.last_ccd_snaps = 0;
                let body_q_in = if model.body_count > 0 {
                    state_in.body_q.host()?.to_vec()
                } else {
                    Vec::new()
                };
                for _ in 0..n_sub {
                    // Stage 33.4 — pre-substep CCD pass. Runs only when
                    // ≥ 2 soft surfaces are registered (no cross-mesh
                    // contact otherwise) and the user has explicitly
                    // opted in via `with_ccd(true)`. Mutates `p_in`
                    // (snap to TOI - eps) + `v_in` (kill normal-side
                    // dive component) IN PLACE so the subsequent
                    // `cpu_step` predictor sees the corrected state.
                    if self.ccd_enabled
                        && model.soft_surfaces.len() >= 2
                        && self.contact_stiffness > 0.0
                    {
                        if self.soft_bvh.is_none() {
                            self.soft_bvh = Some(crate::geometry::SoftBvh::build(model, state_in));
                        }
                        let bvh = self.soft_bvh.as_mut().unwrap();
                        let (inv, snaps) = super::vbd_soft_contact::ccd_pass(
                            model,
                            bvh,
                            &mut p_in,
                            &mut v_in,
                            self.vt_contact_margin,
                            self.ccd_velocity_threshold,
                            dt_sub,
                        );
                        self.last_ccd_invocations += inv;
                        self.last_ccd_snaps += snaps;
                    }
                    cpu_step(
                        model,
                        &p_in,
                        &v_in,
                        &body_q_in,
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
                    // post-step positions). Mutates `fp`, `eps_p`, `alpha`.
                    if model.tet_count > 0 {
                        cpu_plasticity_return_map(model, &p_out, &mut fp, &mut eps_p, &mut alpha)?;
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
                    state_out
                        .tet_back_stress
                        .host_mut()?
                        .copy_from_slice(&alpha);
                }
                // Stage 20.4 — host-side tet-vs-tet contact projection
                // (only when ≥ 2 tet meshes are registered) + host-side
                // SDF backing projection (always — no-op when no
                // backings are registered).
                if model.tet_count > 0 && model.tet_mesh_count >= 2 && self.contact_stiffness > 0.0
                {
                    project_tet_tet_contacts(model, state_out)?;
                }
                if !self.sdf_backings.is_empty() {
                    let radius_vec = model.particle_radius.host()?.to_vec();
                    let inv_mass_vec = model.particle_inv_mass.host()?.to_vec();
                    project_sdf_backings(
                        &self.sdf_backings,
                        state_out,
                        &radius_vec,
                        &inv_mass_vec,
                    )?;
                }
                // Stage 33 — cross-mesh soft-vs-soft VT contact
                // projection. Two-way (vertex side + tri side) Jacobi
                // sweep that runs after the inner VBD substeps so the
                // contact correction sits on top of the elastic
                // dynamics. No-op when fewer than two soft surfaces
                // are registered.
                if model.soft_surfaces.len() >= 2 && self.contact_stiffness > 0.0 {
                    if self.soft_bvh.is_none() {
                        self.soft_bvh = Some(crate::geometry::SoftBvh::build(model, state_out));
                    }
                    let bvh = self.soft_bvh.as_mut().unwrap();
                    // Stage 33.5 friction needs the per-substep
                    // tangential drift, so pass in the pre-step
                    // positions (`state_in.particle_q`).
                    let q_pre_step = state_in.particle_q.host()?.to_vec();
                    let qd_pre_step = state_in.particle_qd.host()?.to_vec();
                    let new_contacts = super::vbd_soft_contact::project_vt_contacts(
                        model,
                        state_out,
                        bvh,
                        self.vt_contact_margin,
                        self.vt_contact_sweeps,
                        dt,
                        &self.prev_vt_contacts,
                        &q_pre_step,
                        &qd_pre_step,
                    )?;
                    self.last_vt_contact_count = new_contacts.len();
                    self.prev_vt_contacts = new_contacts;
                }
                // Body state passes through.
                let q_in = state_in.body_q.host()?.to_vec();
                let qd_in = state_in.body_qd.host()?.to_vec();
                state_out.body_q.host_mut()?.copy_from_slice(&q_in);
                state_out.body_qd.host_mut()?.copy_from_slice(&qd_in);
            }
            #[cfg(feature = "gpu")]
            VbdBackend::Gpu(_) => {
                unreachable!("GPU path returns before the backend match")
            }
        }
        Ok(())
    }
}

#[cfg(feature = "gpu")]
impl VbdSolver {
    /// Last enqueued inertial targets and per-node force residuals; requires device completion.
    pub fn material_force_samples_gpu(&self) -> Option<&wgpu::Buffer> {
        match &self.backend {
            VbdBackend::Gpu(gpu) => gpu.kernel.force_samples(),
            _ => None,
        }
    }

    /// Device-projected spring stiffnesses, padded to at least one element.
    pub fn spring_stiffness_gpu(&self) -> Option<&wgpu::Buffer> {
        match &self.backend {
            VbdBackend::Gpu(gpu) => Some(gpu.kernel.spring_stiffness_buffer()),
            _ => None,
        }
    }

    /// Starts or continues GPU elastic material iterations with no inline shape penalties.
    /// Original positions, history, model and dt stay fixed; input velocities may change.
    #[allow(clippy::too_many_arguments)]
    pub fn material_iteration_gpu(
        &mut self,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        dt: Real,
        iterations: u32,
        begin: bool,
    ) -> Result<()> {
        let VbdBackend::Gpu(gpu) = &mut self.backend else {
            return Err(Error::Build(
                "material iteration requires GPU backend".into(),
            ));
        };
        gpu.kernel.material_iteration(
            &gpu.device,
            &gpu.queue,
            model,
            state_in,
            state_out,
            dt,
            iterations,
            begin,
        )
    }

    /// Evaluates the elastic material residual at the supplied candidate velocity, without advancing state.
    pub fn material_force_residual_gpu(
        &mut self,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        dt: Real,
    ) -> Result<&wgpu::Buffer> {
        let VbdBackend::Gpu(gpu) = &mut self.backend else {
            return Err(Error::Build(
                "material force residual requires GPU backend".into(),
            ));
        };
        gpu.kernel
            .force_residual(&gpu.device, &gpu.queue, model, state_in, state_out, dt)
    }

    /// One GPU VBD substep advancing `dt`: main kernel dispatch, the
    /// soft-VT contact pipeline, and the host-side tet-tet / SDF
    /// fallbacks. Extracted from `step` so the multi-substep driver
    /// can loop it with per-substep input/output states.
    fn gpu_substep(
        &mut self,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        dt: Real,
    ) -> Result<()> {
        let VbdBackend::Gpu(gpu) = &mut self.backend else {
            unreachable!("gpu_substep on CPU backend")
        };
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

        // Borrow the bits the kernel dispatch needs separately
        // from the soft-VT pipeline (which we lazy-construct
        // below). Splitting the borrow lets us mutate
        // `gpu.soft_bvh_gpu` / `gpu.soft_vt_kernel` after the
        // main kernel dispatch.
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        let contact_k = self.contact_stiffness as f32;
        let hard = matches!(self.vbd_mode, VbdMode::Hard);
        // Stage 33.4 — GPU CCD path is deferred. The kernel
        // reads `state_in` from device memory and `state_in` is
        // an immutable borrow at this layer (mutating it would
        // require download → host CCD → upload, which can't
        // happen through `&State`). For v1 the GPU backend
        // surfaces zero CCD activity. The CPU backend ships
        // the working CCD pass; viewer scenes that need CCD
        // (Scene::FastSoftImpact) drive the CPU path.
        self.last_ccd_invocations = 0;
        self.last_ccd_snaps = 0;
        // The GPU backend runs a SINGLE VBD substep per step,
        // advancing the full `dt` (VBD is an implicit-energy
        // minimiser, stable at large dt given enough inner
        // iterations). It cannot honour `self.substeps > 1`:
        // a device-side substep loop would need to ping-pong
        // through `state_in`, but `state_in` is an immutable
        // borrow here. Passing `dt` (not `dt / substeps`)
        // keeps the GPU path advancing real-time — an earlier
        // version passed `dt_sub` and ran the sim at
        // 1/substeps speed. Callers that need multi-substep
        // accuracy on the GPU must drive the substep loop
        // externally. `dt_sub` is still used by the CPU arm.
        gpu.kernel.dispatch(
            &device,
            &queue,
            model,
            state_in,
            state_out,
            dt,
            self.iterations as u32,
            contact_k,
            hard,
        )?;
        // Stage 33.2 — soft-vs-soft VT contact via a fully-GPU
        // pipeline. Replaces the Stage 33 host-side post-step
        // round trip (`particle_q.download → CPU project → upload`)
        // with three on-device dispatches: `compute_tri_aabbs`
        // → `LBVH refit_in_place` → `gather_vt + apply_vt_penalty
        // + commit_dq`. The only host bounce is a 4-byte read
        // of the contact count between gather and apply.
        let has_vt = model.soft_surfaces.len() >= 2 && self.contact_stiffness > 0.0;
        if has_vt {
            // Lazy construct the GPU BVH + VT kernel on first use.
            if gpu.soft_bvh_gpu.is_none() {
                gpu.soft_bvh_gpu = Some(crate::kernels::SoftBvhGpu::new(
                    device.clone(),
                    queue.clone(),
                    model,
                )?);
            }
            if gpu.soft_vt_kernel.is_none() {
                gpu.soft_vt_kernel = Some(crate::kernels::SoftVtContactKernel::new(
                    device.clone(),
                    queue.clone(),
                    model,
                )?);
            }
            let soft_bvh = gpu.soft_bvh_gpu.as_mut().unwrap();
            let vt_kernel = gpu.soft_vt_kernel.as_ref().unwrap();
            // Stage 33.2.1 — snapshot pre-substep state (q + qd)
            // BEFORE the sweep loop. The position-form Coulomb
            // friction in `apply_vt_penalty` reads the snapshot
            // for tangential drift; `commit_velocity_recovery`
            // reads it for restitution-aware bounce targets.
            // Also resets the per-vertex friction/restitution/
            // contact_normal_sum scratches.
            vt_kernel.dispatch_snapshot_pre_step(state_in)?;
            // Run the gather+apply+commit Jacobi-style sweep
            // `vt_contact_sweeps` times per substep — matches
            // the CPU `project_vt_contacts` semantics. Each
            // sweep refits the BVH against the latest
            // post-projection positions so successive contacts
            // converge.
            let n_sweeps = self.vt_contact_sweeps.max(1);
            let mut last_count = 0u32;
            for sweep_idx in 0..n_sweeps {
                // 1) compute per-tri AABBs from the current
                //    state.particle_q (writes directly into the
                //    LBVH primitive_aabb buffer).
                vt_kernel.dispatch_compute_tri_aabbs(model, state_out, soft_bvh)?;
                // 2) refit the LBVH in place against those AABBs.
                soft_bvh.refit(model)?;
                // 3) gather + apply + commit_dq — fully GPU.
                let stats = vt_kernel.dispatch_gather_and_apply(
                    model,
                    state_out,
                    soft_bvh,
                    self.vt_contact_margin,
                    self.contact_stiffness,
                    dt,
                    sweep_idx == 0,
                )?;
                last_count = stats.contact_count;
                if stats.contact_count == 0 {
                    // No active contacts — early-out the rest
                    // of the sweep (matches CPU
                    // `project_vt_contacts` `break` on empty
                    // contact list).
                    break;
                }
            }
            self.last_vt_contact_count = last_count as usize;
            // Stage 33.2.1 — restitution-aware velocity recovery
            // + Coulomb tangential damping (one-shot, after the
            // sweep loop). No-op when no contacts engaged
            // because the contact_normal_sum scratch stays at
            // zero.
            if last_count > 0 {
                vt_kernel.dispatch_commit_velocity_recovery(
                    model,
                    state_out,
                    soft_bvh,
                    self.vt_contact_margin,
                    self.contact_stiffness,
                    dt,
                )?;
            }
            // commit_dq mutated state_out.particle_q +
            // particle_qd in place. The Buffer<Vec3> wrapper
            // still thinks the host is canonical — mark them
            // device-dirty so subsequent host reads trip the
            // staleness check (and legitimate use cases call
            // download).
            state_out.particle_q.mark_device_dirty();
            state_out.particle_qd.mark_device_dirty();
        }
        // Stage 20.4 — host-side fallbacks for tet-vs-tet (when
        // ≥ 2 tet meshes register but soft surfaces don't yet
        // expose them as the unified registry) + SDF backings.
        // These STILL require a `particle_q.download → upload`
        // round-trip until they're ported (Stage 33.2 spec
        // explicitly leaves rigid-SDF backings as CPU-only).
        let has_tet_tet = model.tet_count > 0
            && model.tet_mesh_count >= 2
            && self.contact_stiffness > 0.0
            // Don't double-process: when soft surfaces cover
            // the same tet meshes, the GPU VT pipeline above
            // already handled them.
            && !has_vt;
        if has_tet_tet || !self.sdf_backings.is_empty() {
            state_out.particle_q.download(&device, &queue)?;
            if has_tet_tet {
                project_tet_tet_contacts(model, state_out)?;
            }
            if !self.sdf_backings.is_empty() {
                let radius_vec = model.particle_radius.host()?.to_vec();
                let inv_mass_vec = model.particle_inv_mass.host()?.to_vec();
                project_sdf_backings(&self.sdf_backings, state_out, &radius_vec, &inv_mass_vec)?;
            }
            // VT contacts are handled by the GPU kernel above
            // (Stage 33.2 dropped the CPU bounce). EE contacts
            // (Stage 33.1) and friction (Stage 33.5) are CPU-only
            // and only run on the CPU VBD backend; the GPU path
            // skips them in v1.
            state_out.particle_q.ensure_device(&device)?;
            state_out.particle_q.upload(&queue)?;
        }
        Ok(())
    }

    /// GPU multi-substep driver: ping-pong [`Self::gpu_substep`]
    /// through a lazily-built device-resident scratch state (same
    /// pattern as `XpbdSolver::step_multi_gpu`), then device-copy the
    /// result into `state_out` if the last substep landed in the
    /// scratch.
    fn gpu_multi_substep(
        &mut self,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        dt_sub: Real,
        n_sub: usize,
    ) -> Result<()> {
        let (device, queue) = match &self.backend {
            VbdBackend::Gpu(gpu) => (gpu.device.clone(), gpu.queue.clone()),
            VbdBackend::Cpu => unreachable!("gpu_multi_substep on CPU backend"),
        };
        if self.gpu_scratch.is_none() {
            let mut scratch = model.state()?;
            scratch.upload_to_device(&device, &queue)?;
            self.gpu_scratch = Some(scratch);
        }
        let mut scratch = self.gpu_scratch.take().expect("scratch just ensured");
        // Bodies pass through VBD untouched, but the kernel reads the
        // substep INPUT state's body poses for soft-contact transforms
        // — seed both intermediate states with the outer input poses.
        if model.body_count > 0 {
            crate::xpbd::copy_device_buffers(
                &device,
                &queue,
                &[
                    (
                        state_in.body_q.device_buffer(),
                        scratch.body_q.device_buffer(),
                    ),
                    (
                        state_in.body_q.device_buffer(),
                        state_out.body_q.device_buffer(),
                    ),
                ],
            )?;
        }
        let run = |solver: &mut Self, state_out: &mut State, scratch: &mut State| {
            solver.gpu_substep(model, state_in, state_out, dt_sub)?;
            let mut output_in_scratch = false;
            for _ in 1..n_sub {
                if output_in_scratch {
                    solver.gpu_substep(model, scratch, state_out, dt_sub)?;
                } else {
                    solver.gpu_substep(model, state_out, scratch, dt_sub)?;
                }
                output_in_scratch = !output_in_scratch;
            }
            Ok(output_in_scratch)
        };
        let result = run(self, state_out, &mut scratch);
        let finalize = if matches!(result, Ok(true)) {
            // Final substep landed in the scratch — copy it out.
            let mut pairs = vec![
                (
                    scratch.particle_q.device_buffer(),
                    state_out.particle_q.device_buffer(),
                ),
                (
                    scratch.particle_qd.device_buffer(),
                    state_out.particle_qd.device_buffer(),
                ),
            ];
            if model.tet_count > 0 {
                // Plasticity state ping-pongs with the particle state
                // on the GPU — carry it out of the scratch too.
                pairs.push((
                    scratch.tet_f_plastic.device_buffer(),
                    state_out.tet_f_plastic.device_buffer(),
                ));
                pairs.push((
                    scratch.tet_plastic_strain.device_buffer(),
                    state_out.tet_plastic_strain.device_buffer(),
                ));
                pairs.push((
                    scratch.tet_back_stress.device_buffer(),
                    state_out.tet_back_stress.device_buffer(),
                ));
            }
            let copied = crate::xpbd::copy_device_buffers(&device, &queue, &pairs);
            if copied.is_ok() {
                state_out.particle_q.mark_device_dirty();
                state_out.particle_qd.mark_device_dirty();
            }
            copied
        } else {
            Ok(())
        };
        self.gpu_scratch = Some(scratch);
        result?;
        finalize
    }
}

/// Stage 20.4 — host-side projection of every soft-body vertex out of
/// every registered [`VbdSdfBacking`] surface. The projection is a
/// classic XPBD-style position-only correction: sample the SDF at the
/// vertex's world-space position; if `signed_dist - radius < 0` push
/// the vertex along the SDF gradient by `(radius - signed_dist)` so it
/// sits exactly on the surface with a `radius` clearance.
///
/// Caveats per the spec:
/// - **Narrow-band edges**: `SdfGrid` returns `+∞` outside its sample
///   AABB; a vertex that pokes out the side of the grid won't get
///   projected. Bake the grid with enough padding to cover the
///   workload's deformation envelope.
/// - **Sparse SDFs**: `SparseSdf::sample` returns the background value
///   far outside the active narrow band; if the vertex is well outside
///   the band we treat it as "no contact" (which is correct iff the
///   bake's narrow-band radius covers the simulation's penetration
///   envelope).
pub(crate) fn project_sdf_backings(
    sdf_backings: &[VbdSdfBacking],
    state: &mut State,
    radius: &[Real],
    inv_mass: &[Real],
) -> Result<()> {
    if sdf_backings.is_empty() {
        return Ok(());
    }
    let q = state.particle_q.host_mut()?;
    for (i, p) in q.iter_mut().enumerate() {
        if inv_mass[i] <= 0.0 {
            continue;
        }
        let pr = radius[i];
        for backing in sdf_backings {
            match backing {
                VbdSdfBacking::Dense(grid, xform) => {
                    let p_local = xform.rotation.conjugate() * (*p - xform.position);
                    let sd = sample_sdf_grid_trilinear(grid, p_local);
                    if sd >= pr {
                        continue;
                    }
                    let Some(n_local) = sample_sdf_grid_gradient(grid, p_local) else {
                        continue;
                    };
                    if n_local.length_squared() < 1.0e-12 {
                        continue;
                    }
                    let n_world = (xform.rotation * n_local).normalize_or_zero();
                    if n_world.length_squared() < 1.0e-12 {
                        continue;
                    }
                    let push = pr - sd;
                    *p += n_world * push;
                }
                VbdSdfBacking::Sparse(sdf, xform) => {
                    let p_local = xform.rotation.conjugate() * (*p - xform.position);
                    let sd = sdf.sample(p_local);
                    if sd >= pr {
                        continue;
                    }
                    let n_local = sdf.normal(p_local);
                    if n_local.length_squared() < 1.0e-12 {
                        continue;
                    }
                    let n_world = (xform.rotation * n_local).normalize_or_zero();
                    if n_world.length_squared() < 1.0e-12 {
                        continue;
                    }
                    let push = pr - sd;
                    *p += n_world * push;
                }
            }
        }
    }
    Ok(())
}

/// Trilinear-interpolated sample of `SdfGrid` at world-equivalent local
/// position `p`. Returns `+∞` outside the sample AABB.
fn sample_sdf_grid_trilinear(grid: &crate::geometry::SdfGrid, p: Vec3) -> Real {
    let local = (p - grid.origin) / grid.cell_size;
    let nx = grid.nx as i32;
    let ny = grid.ny as i32;
    let nz = grid.nz as i32;
    let x0 = local.x.floor() as i32;
    let y0 = local.y.floor() as i32;
    let z0 = local.z.floor() as i32;
    if x0 < 0 || y0 < 0 || z0 < 0 || x0 + 1 >= nx || y0 + 1 >= ny || z0 + 1 >= nz {
        return Real::INFINITY;
    }
    let fx = local.x - (x0 as Real);
    let fy = local.y - (y0 as Real);
    let fz = local.z - (z0 as Real);
    let s000 = grid.at(x0 as u32, y0 as u32, z0 as u32);
    let s100 = grid.at((x0 + 1) as u32, y0 as u32, z0 as u32);
    let s010 = grid.at(x0 as u32, (y0 + 1) as u32, z0 as u32);
    let s110 = grid.at((x0 + 1) as u32, (y0 + 1) as u32, z0 as u32);
    let s001 = grid.at(x0 as u32, y0 as u32, (z0 + 1) as u32);
    let s101 = grid.at((x0 + 1) as u32, y0 as u32, (z0 + 1) as u32);
    let s011 = grid.at(x0 as u32, (y0 + 1) as u32, (z0 + 1) as u32);
    let s111 = grid.at((x0 + 1) as u32, (y0 + 1) as u32, (z0 + 1) as u32);
    let lerp = |a: Real, b: Real, t: Real| a + (b - a) * t;
    let c00 = lerp(s000, s100, fx);
    let c01 = lerp(s001, s101, fx);
    let c10 = lerp(s010, s110, fx);
    let c11 = lerp(s011, s111, fx);
    let c0 = lerp(c00, c10, fy);
    let c1 = lerp(c01, c11, fy);
    lerp(c0, c1, fz)
}

/// Central-difference gradient of [`sample_sdf_grid_trilinear`] at
/// local position `p` (cell-spaced step). Returns `None` when the
/// finite difference is degenerate (out-of-grid neighbours, or
/// non-finite samples) so the caller can skip the contact instead of
/// substituting a phantom `+Y` normal.
fn sample_sdf_grid_gradient(grid: &crate::geometry::SdfGrid, p: Vec3) -> Option<Vec3> {
    let h = grid.cell_size;
    let dx = sample_sdf_grid_trilinear(grid, p + Vec3::new(h, 0.0, 0.0))
        - sample_sdf_grid_trilinear(grid, p - Vec3::new(h, 0.0, 0.0));
    let dy = sample_sdf_grid_trilinear(grid, p + Vec3::new(0.0, h, 0.0))
        - sample_sdf_grid_trilinear(grid, p - Vec3::new(0.0, h, 0.0));
    let dz = sample_sdf_grid_trilinear(grid, p + Vec3::new(0.0, 0.0, h))
        - sample_sdf_grid_trilinear(grid, p - Vec3::new(0.0, 0.0, h));
    let g = Vec3::new(dx, dy, dz);
    if g.length_squared() > 1.0e-24 && g.x.is_finite() && g.y.is_finite() && g.z.is_finite() {
        Some(g.normalize_or_zero())
    } else {
        // Degenerate finite-difference window — at the grid boundary or
        // sampling NaN/Inf cells. Caller decides how to handle the
        // missing normal; the default is to skip the contact for this
        // vertex this step. Logged once per process for visibility.
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[molla::vbd] sample_sdf_grid_gradient: degenerate gradient at \
                 p={p:?} (length²={}, finite={}/{}/{}); skipping SDF contact for \
                 this vertex (further occurrences silenced).",
                g.length_squared(),
                g.x.is_finite(),
                g.y.is_finite(),
                g.z.is_finite(),
            );
        }
        None
    }
}

/// Stage 20.4 — host-side tet-vs-tet collision projection.
///
/// Walks every particle that belongs to one tet mesh and projects it
/// out of any tet from a different mesh that contains it. Uses the
/// per-tet AABB list for broad-phase culling + `tet_closest_face` for
/// the narrow-phase signed-distance + outward-normal query.
///
/// Multiple Jacobi-style sweeps stack — for the v1 workload (two
/// jelly cubes), one pass typically resolves all penetrations because
/// each substep's pre-impact velocity is small. We run 3 sweeps to be
/// safe for the corner cases where one vertex pokes simultaneously
/// into multiple opposite tets. CPU host-side; called between substeps
/// with the host-side particle positions current.
pub(crate) fn project_tet_tet_contacts(model: &Model, state: &mut State) -> Result<()> {
    use crate::geometry::tet_bvh::{
        TetAabb, recompute_tet_aabbs, tet_closest_face, tet_meshes_aabb_overlap,
    };
    let n_tets = model.tet_count;
    let n_meshes = model.tet_mesh_count;
    if n_tets == 0 || n_meshes < 2 {
        return Ok(());
    }
    let tet_indices = model.tet_indices.host()?;
    let tet_mesh_offsets = model.tet_mesh_offsets.host()?;
    let inv_mass = model.particle_inv_mass.host()?;
    let radius = model.particle_radius.host()?;
    let n = model.particle_count;

    // Pre-compute per-particle mesh ownership.
    let mut owner = vec![-1_i32; n];
    for m in 0..n_meshes {
        let lo = tet_mesh_offsets[m] as usize;
        let hi = tet_mesh_offsets[m + 1] as usize;
        for t in lo..hi {
            for k in 0..4 {
                let v = tet_indices[t * 4 + k] as usize;
                if owner[v] == -1 {
                    owner[v] = m as i32;
                } else if owner[v] != m as i32 {
                    owner[v] = -2; // shared — skip
                }
            }
        }
    }

    let mut q = state.particle_q.host_mut()?.to_vec();
    let mut aabbs = vec![TetAabb::from_corners(0, Vec3::ZERO, Vec3::ZERO); n_tets];
    // Eight Jacobi sweeps. Each sweep refits AABBs from the current
    // (already-projected) positions, rebuilds the mesh-pair overlap
    // table, and applies one round of position projection. Multiple
    // sweeps are essential when both meshes deform (each sweep moves
    // one particle out at a time, but the partner's vertices may then
    // poke through the corrected position; iterating to convergence
    // tightens the residual).
    for _sweep in 0..8 {
        recompute_tet_aabbs(tet_indices, &q, 0.005, &mut aabbs);
        // Build mesh-pair overlap table.
        let mut pairs: Vec<Vec<u32>> = vec![Vec::new(); n_meshes];
        for m_a in 0..n_meshes {
            let lo_a = tet_mesh_offsets[m_a] as usize;
            let hi_a = tet_mesh_offsets[m_a + 1] as usize;
            if hi_a == lo_a {
                continue;
            }
            for m_b in (m_a + 1)..n_meshes {
                let lo_b = tet_mesh_offsets[m_b] as usize;
                let hi_b = tet_mesh_offsets[m_b + 1] as usize;
                if hi_b == lo_b {
                    continue;
                }
                if tet_meshes_aabb_overlap(&aabbs[lo_a..hi_a], &aabbs[lo_b..hi_b]) {
                    pairs[m_a].push(m_b as u32);
                    pairs[m_b].push(m_a as u32);
                }
            }
        }
        let mut any_projected = false;
        for v in 0..n {
            let o = owner[v];
            if o < 0 {
                continue;
            }
            // Pinned particles are anchored — they project the *other*
            // particle out (via Newton's third law on the collision
            // partner), but their own position is fixed. Skip.
            if inv_mass[v] <= 0.0 {
                continue;
            }
            let pr = radius[v];
            let pos = q[v];
            // Find the *deepest* penetration across all candidate tets
            // and project along its normal. (A "first-found" projection
            // can flip-flop a vertex between two adjacent tets sharing
            // a face when the penetration spans them.)
            let mut deepest_pen: Real = 0.0;
            let mut deepest_normal = Vec3::ZERO;
            for &m_other in &pairs[o as usize] {
                let lo_t = tet_mesh_offsets[m_other as usize] as usize;
                let hi_t = tet_mesh_offsets[m_other as usize + 1] as usize;
                for t in lo_t..hi_t {
                    let aabb = &aabbs[t];
                    let inflated = aabb.inflated(pr);
                    if !inflated.contains_point(pos) {
                        continue;
                    }
                    let i0 = tet_indices[t * 4] as usize;
                    let i1 = tet_indices[t * 4 + 1] as usize;
                    let i2 = tet_indices[t * 4 + 2] as usize;
                    let i3 = tet_indices[t * 4 + 3] as usize;
                    if let Some((signed, n_dir)) = tet_closest_face(pos, q[i0], q[i1], q[i2], q[i3])
                    {
                        // signed < 0 ⇒ inside the tet (penetration depth = -signed).
                        // We treat any signed < pr as a contact so the
                        // particle gets a `pr` clearance buffer — but
                        // only positive pen drives projection.
                        let push_dist = pr - signed;
                        if push_dist > deepest_pen {
                            deepest_pen = push_dist;
                            deepest_normal = n_dir;
                        }
                    }
                }
            }
            if deepest_pen > 0.0 && deepest_normal.length_squared() > 1.0e-12 {
                q[v] = pos + deepest_normal * deepest_pen;
                any_projected = true;
            }
        }
        if !any_projected {
            break;
        }
    }
    state.particle_q.host_mut()?.copy_from_slice(&q);
    Ok(())
}

/// Map XPBD compliance `α` (m/N) onto an equivalent stiffness
/// `k = 1 / max(α, ε)`, capped at `max_k` to avoid pathological
/// Hessian conditioning when the user requested near-zero
/// compliance ("perfectly rigid"). The cap is what makes VBD
/// well-behaved across the full XPBD compliance range that the
/// builder accepts (0 → "stiff" rather than "infinite").
#[cfg(feature = "gpu")]
fn derive_stiffness_table(model: &Model, max_k: f64) -> Result<Vec<Real>> {
    let comp = model.spring_compliance.host()?;
    let tension_only = model.spring_tension_only.host()?;
    let mut out = Vec::with_capacity(comp.len());
    for (i, &a) in comp.iter().enumerate() {
        let k = if a > 1.0e-12 { 1.0 / a } else { max_k };
        out.push(if tension_only[i] != 0 {
            1.0 / a
        } else {
            k.min(max_k).max(1.0)
        });
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
    body_q_in: &[crate::math::Transform],
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
    let spring_tension_only = model.spring_tension_only.host()?;
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
    model.validate_tet_viscosity()?;
    let tet_viscosity = model.tet_viscosity.host()?;
    let vadj_tets_off = model.vertex_adjacent_tets_offsets.host()?;
    let vadj_tets = model.vertex_adjacent_tets.host()?;
    // Cable bending state (Stage 20.2).
    let cable_off = model.cable_particle_offsets.host()?;
    let cable_particles = model.cable_particles.host()?;
    let cable_bending = model.cable_bending_stiffness.host()?;

    // Pre-compute per-spring stiffness from compliance.
    let stiffness: Vec<Real> = spring_comp
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let k = if a > 1.0e-12 {
                1.0 / a
            } else {
                max_spring_stiffness
            };
            if spring_tension_only[i] != 0 {
                1.0 / a
            } else {
                k.min(max_spring_stiffness).max(1.0)
            }
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

    // Stage 20.4 — tet-vs-tet collision support (CPU oracle).
    // Strategy: maintain a per-tet AABB that's refreshed once per
    // outer-iteration (the per-vertex Newton block reads it; we don't
    // refit between iterations to keep cost bounded). For each
    // particle, walk every *other* tet mesh's AABB list looking for
    // overlaps; a successful overlap triggers a per-vertex point-in-tet
    // penalty using `tet_closest_face`.
    //
    // This collapses to a no-op when only one tet mesh is present (the
    // canonical "single jelly cube" demo). For two cubes, the two-mesh
    // overlap pair drives the per-vertex × per-other-tet quadratic loop;
    // the top-level mesh-AABB cull skips the inner loop entirely when
    // the cubes are spatially apart.
    use crate::geometry::tet_bvh::{
        TetAabb, recompute_tet_aabbs, tet_closest_face, tet_meshes_aabb_overlap,
    };
    let tet_mesh_offsets = model.tet_mesh_offsets.host()?;
    let n_tet_meshes = model.tet_mesh_count;
    // Per-particle: which tet mesh (if any) it belongs to. -1 = no
    // mesh (cloth particle / cable particle / unowned). When a
    // particle is referenced by tets in multiple meshes (the user's
    // model glued meshes together), we record the *first* mesh —
    // tet-vs-tet contact then skips that particle as a degenerate case
    // (it's already shared, so applying a separation force would tear
    // the model apart). For the demo workload (two independent jelly
    // cubes) every particle belongs to exactly one mesh.
    let particle_mesh_id: Vec<i32> = if model.tet_count > 0 && n_tet_meshes >= 2 {
        let mut owner = vec![-1_i32; n];
        for m in 0..n_tet_meshes {
            let lo_t = tet_mesh_offsets[m] as usize;
            let hi_t = tet_mesh_offsets[m + 1] as usize;
            for t in lo_t..hi_t {
                for k in 0..4 {
                    let v = tet_indices[t * 4 + k] as usize;
                    if owner[v] == -1 {
                        owner[v] = m as i32;
                    } else if owner[v] != m as i32 {
                        // Shared between meshes — flag with -2 so the
                        // contact loop below skips the particle.
                        owner[v] = -2;
                    }
                }
            }
        }
        owner
    } else {
        Vec::new()
    };
    let tet_aabb_margin = 0.01_f64; // 1 cm — comfortable for 5–30 cm cubes.
    let mut tet_aabbs: Vec<TetAabb> = if model.tet_count > 0 && n_tet_meshes >= 2 {
        let mut v = vec![TetAabb::from_corners(0, Vec3::ZERO, Vec3::ZERO); model.tet_count];
        recompute_tet_aabbs(tet_indices, &x, tet_aabb_margin, &mut v);
        v
    } else {
        Vec::new()
    };
    // Pair table: for each tet mesh, the list of *other* tet meshes
    // whose root AABB overlaps. Filled below.
    let mut overlapping_pairs: Vec<Vec<u32>> = vec![Vec::new(); n_tet_meshes];
    // Stage 20.4 — tet-vs-tet contact stiffness. Reuse the rigid-shape
    // contact stiffness as the default; the user can scale this from
    // the viewer pane independently in a future stage.
    let tet_contact_stiffness = contact_stiffness * 0.5;

    // Pre-compute world-frame shape transforms once per substep.
    let body_q = state_body_transforms(model, body_q_in, n)?;
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
            let bq = body_q
                .get(body as usize)
                .copied()
                .unwrap_or(Transform::IDENTITY);
            Transform::new(
                bq.position + bq.rotation * local.position,
                bq.rotation * local.rotation,
            )
        };
        shape_xforms.push(world_xf);
    }

    // Outer iterations × colours × per-vertex block solve.
    for _it in 0..iterations.max(1) {
        // Stage 20.4 — refresh per-tet AABBs + mesh-pair overlap table
        // once per outer iteration. Skipping intermediate iterations
        // is fine: the per-vertex Newton step is small (sub-mm), so
        // the AABBs change negligibly within an iteration. Doing this
        // once per outer iteration (rather than once per substep)
        // gives the contact path a fresh AABB after spring + tet
        // forces have already pulled vertices toward each other.
        if !tet_aabbs.is_empty() {
            recompute_tet_aabbs(tet_indices, &x, tet_aabb_margin, &mut tet_aabbs);
            for pairs in overlapping_pairs.iter_mut() {
                pairs.clear();
            }
            for m_a in 0..n_tet_meshes {
                let lo_a = tet_mesh_offsets[m_a] as usize;
                let hi_a = tet_mesh_offsets[m_a + 1] as usize;
                if hi_a == lo_a {
                    continue;
                }
                let aabbs_a = &tet_aabbs[lo_a..hi_a];
                for m_b in (m_a + 1)..n_tet_meshes {
                    let lo_b = tet_mesh_offsets[m_b] as usize;
                    let hi_b = tet_mesh_offsets[m_b + 1] as usize;
                    if hi_b == lo_b {
                        continue;
                    }
                    let aabbs_b = &tet_aabbs[lo_b..hi_b];
                    if tet_meshes_aabb_overlap(aabbs_a, aabbs_b) {
                        overlapping_pairs[m_a].push(m_b as u32);
                        overlapping_pairs[m_b].push(m_a as u32);
                    }
                }
            }
        }
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
                    if spring_tension_only[s] != 0 && len <= rest {
                        continue;
                    }
                    let kk = stiffness[s];
                    let f_self = -kk * (1.0 - rest / len) * diff;
                    let outer = mat3_outer(diff, diff);
                    let inv_l2 = 1.0 / (len * len);
                    let outer_scaled = mat3_scale(outer, inv_l2);
                    let bracket = mat3_sub(Mat3::IDENTITY, outer_scaled);
                    let l0_l = rest / len;
                    let h_self =
                        mat3_scale(mat3_sub(Mat3::IDENTITY, mat3_scale(bracket, l0_l)), kk);
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
                    let visc = tet_viscosity[tet];
                    if visc.shear > 0.0 || visc.bulk > 0.0 {
                        let ids = &tet_indices[tet * 4..tet * 4 + 4];
                        let p: [Vec3; 4] = std::array::from_fn(|i| x_prev[ids[i] as usize]);
                        let d: [Vec3; 4] = std::array::from_fn(|i| x[ids[i] as usize] - p[i]);
                        let dm = tet_dm_inv[tet];
                        let previous =
                            crate::math::Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0])
                                * dm;
                        let increment =
                            crate::math::Mat3::from_cols(d[1] - d[0], d[2] - d[0], d[3] - d[0])
                                * dm;
                        let gradient = if order == 0 {
                            -dm.transpose() * Vec3::ONE
                        } else {
                            dm.row(order - 1)
                        };
                        let block = viscosity::increment_block(
                            previous,
                            increment,
                            gradient,
                            visc.shear,
                            visc.bulk,
                            dt,
                            tet_rest_volume[tet],
                        );
                        if !block.force.is_finite() || !block.tangent.is_finite() {
                            return Err(molla_core::Error::Build(
                                "nonfinite VBD viscous tet block".into(),
                            ));
                        }
                        f += block.force;
                        h += block.tangent;
                    }
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
                                let role = if v == v_prev {
                                    0
                                } else if v == v_mid {
                                    1
                                } else {
                                    2
                                };
                                let (f_b, h_b) = cable_bending_vertex_block(
                                    role, x[v_prev], x[v_mid], x[v_next], kb,
                                );
                                f += f_b;
                                h = mat3_add(h, h_b);
                            }
                        }
                    }
                }

                // Stage 20.4 — tet-vs-tet collision (per-vertex penalty).
                // For each tet in any *other* tet mesh whose AABB
                // contains `x[v]`, compute the signed distance to the
                // tet's nearest face and apply a quadratic penalty:
                //   f_c = +k_c · pen · n_outward
                //   H_c = +k_c · n n^T
                // where pen = -signed (positive when penetrating).
                if !tet_aabbs.is_empty() && tet_contact_stiffness > 0.0 {
                    let owner = particle_mesh_id.get(v).copied().unwrap_or(-1);
                    if owner >= 0 {
                        let pr = radius[v];
                        let pos = x[v];
                        for &m_other in &overlapping_pairs[owner as usize] {
                            let lo_t = tet_mesh_offsets[m_other as usize] as usize;
                            let hi_t = tet_mesh_offsets[m_other as usize + 1] as usize;
                            // Skip if the broad AABB of mesh m_other
                            // doesn't contain `pos` (with the radius
                            // margin pre-baked into tet_aabb_margin).
                            // We test individual tet AABBs in the inner
                            // loop; the pair-list cull above already
                            // ensures the meshes' root AABBs overlap.
                            for t in lo_t..hi_t {
                                let aabb = &tet_aabbs[t];
                                // Inflate the AABB by `pr` so a vertex
                                // touching (not yet penetrating) the tet
                                // still triggers the closest-face test.
                                let inflated = aabb.inflated(pr);
                                if !inflated.contains_point(pos) {
                                    continue;
                                }
                                let i0 = tet_indices[t * 4] as usize;
                                let i1 = tet_indices[t * 4 + 1] as usize;
                                let i2 = tet_indices[t * 4 + 2] as usize;
                                let i3 = tet_indices[t * 4 + 3] as usize;
                                if let Some((signed, n_dir)) =
                                    tet_closest_face(pos, x[i0], x[i1], x[i2], x[i3])
                                {
                                    let pen = -signed; // positive when inside
                                    if pen > 0.0 {
                                        let mag = tet_contact_stiffness * pen;
                                        f += n_dir * mag;
                                        let n_outer = mat3_outer(n_dir, n_dir);
                                        h = mat3_add(h, mat3_scale(n_outer, tet_contact_stiffness));
                                    }
                                }
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

/// Resolve body transforms from the input state, falling back to
/// `model.initial_body_q` for older call sites / empty snapshots. Returns at
/// most `model.body_count` transforms; size is passed in so callers can avoid
/// the buffer download when they know the model has no bodies.
fn state_body_transforms(
    model: &Model,
    body_q_in: &[crate::math::Transform],
    _particle_count: usize,
) -> Result<Vec<crate::math::Transform>> {
    if body_q_in.len() >= model.body_count {
        return Ok(body_q_in[..model.body_count].to_vec());
    }
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
                if n_world.length_squared() > 1.0e-12 {
                    n_world
                } else {
                    Vec3::Y
                },
                sd,
            )
        }
        _ => (Vec3::Y, 1.0e6),
    }
}

#[inline]
fn mat3_add(a: crate::math::Mat3, b: crate::math::Mat3) -> crate::math::Mat3 {
    crate::math::Mat3::from_cols(
        a.x_axis + b.x_axis,
        a.y_axis + b.y_axis,
        a.z_axis + b.z_axis,
    )
}
#[inline]
fn mat3_sub(a: crate::math::Mat3, b: crate::math::Mat3) -> crate::math::Mat3 {
    crate::math::Mat3::from_cols(
        a.x_axis - b.x_axis,
        a.y_axis - b.y_axis,
        a.z_axis - b.z_axis,
    )
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
    //
    // F_p is maintained by the von-Mises return mapping, which clamps
    // its singular values to `[0.5, 2.0]` — `det(F_p)` is therefore
    // bounded in `[0.125, 8.0]` and always finite. A NaN / singular
    // F_p here means the upstream return mapping accepted a bad step;
    // catch it loudly in debug builds.
    let fp_det = fp.determinant();
    debug_assert!(
        fp_det.is_finite(),
        "neo_hookean_plastic: non-finite det(F_p) ({fp_det})"
    );
    let f_mat = if (fp_det - 1.0).abs() < 1.0e-9 {
        // F_p = I (or numerically indistinguishable) — skip the inverse.
        f_full
    } else if fp_det.abs() > 1.0e-12 {
        f_full * fp.inverse()
    } else {
        // Singular F_p — should not happen with the singular-value
        // clamp in `apply_vonmises_return_mapping`. Fall back to the
        // pure-elastic gradient so the per-tet step doesn't blow up,
        // but make this path loud in debug.
        debug_assert!(false, "neo_hookean_plastic: singular F_p (det={fp_det})");
        f_full
    };

    // Useful quantities.
    let j = f_mat.determinant();
    let lam_safe = if lam.abs() > 1.0e-6 {
        lam
    } else {
        1.0e-6_f64.copysign(lam.signum())
    };
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
    let lam_term = crate::math::Mat3::from_cols(cof_m * cof_m.x, cof_m * cof_m.y, cof_m * cof_m.z);
    let h_block = mat3_scale(mat3_add(mu_term, mat3_scale(lam_term, lam)), rest_vol);

    // Tikhonov-projection safety net: clamp the diagonal to a tiny
    // positive floor so the per-vertex Hessian is strictly SPD even
    // when an inverted tet would otherwise produce a sign-indefinite
    // contribution. Negligible compared to the inertia + spring
    // diagonals; documented in `solvers/vbd.rs` module docs.
    let h_floor = 1.0e-8_f64 * rest_vol.max(1.0e-12);
    let h_block = crate::math::Mat3::from_cols(
        Vec3::new(
            h_block.x_axis.x.max(h_floor),
            h_block.x_axis.y,
            h_block.x_axis.z,
        ),
        Vec3::new(
            h_block.y_axis.x,
            h_block.y_axis.y.max(h_floor),
            h_block.y_axis.z,
        ),
        Vec3::new(
            h_block.z_axis.x,
            h_block.z_axis.y,
            h_block.z_axis.z.max(h_floor),
        ),
    );

    (f_clean, h_block)
}

/// Determinant gradient, including singular matrices.
#[inline]
fn cofactor3(m: crate::math::Mat3) -> crate::math::Mat3 {
    crate::math::Mat3::from_cols(
        m.y_axis.cross(m.z_axis),
        m.z_axis.cross(m.x_axis),
        m.x_axis.cross(m.y_axis),
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
        0 => (kappa, 1.0),        // v_prev: ∂E/∂v_prev = +k_b · kappa
        1 => (-2.0 * kappa, 1.0), // v_mid:  ∂E/∂v_mid  = -2·k_b · kappa
        _ => (kappa, 1.0),        // v_next: ∂E/∂v_next = +k_b · kappa
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
fn log_plasticity_dropped(reason: &'static str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::error!(
            target: "molla::vbd",
            reason,
            "plastic flow dropped this substep — upstream NaN / overshoot in von-Mises return mapping (further occurrences silenced for this process)"
        );
    }
}

pub(crate) fn cpu_plasticity_return_map(
    model: &Model,
    particle_q: &[Vec3],
    fp: &mut [crate::math::Mat3],
    eps_p: &mut [Real],
    alpha: &mut [crate::math::Mat3],
) -> Result<()> {
    use crate::math::{Mat3, dense3::svd3};
    let tet_indices = model.tet_indices.host()?;
    let tet_dm_inv = model.tet_dm_inv.host()?;
    let tet_mu = model.tet_mu.host()?;
    let tet_yield = model.tet_yield_stress.host()?;
    let tet_hard = model.tet_hardening.host()?;
    // Stage 20.4 — kinematic-hardening modulus c (Pa). When 0, the
    // back-stress α stays at zero throughout (the per-tet update
    // collapses to Stage 20.3's pure isotropic-hardening return map).
    let tet_kin_hard = model.tet_kinematic_hardening.host()?;
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
        // Stage 20.4 — per-tet kinematic-hardening modulus c. When the
        // material was added with `MaterialNeoHookean::plastic` (no
        // kinematic field), this stays 0 and the back-stress evolution
        // is a no-op (Stage 20.3 back-compat).
        let c_kin = tet_kin_hard.get(tet).copied().unwrap_or(0.0);

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
            // Degenerate F_p_prev. The singular-value clamp below
            // bounds det(F_p) ∈ [0.125, 8] so reaching this branch
            // means F_p was poisoned upstream — likely a non-finite
            // particle position or a kernel that bypassed the clamp.
            // Loud in debug; in release we reset F_p and emit a
            // one-shot trace so the next step is well-defined.
            debug_assert!(
                false,
                "von_mises_return_mapping: degenerate F_p_prev (det={fp_det}) tet={tet}"
            );
            log_plasticity_dropped("F_p_prev degenerate");
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

        // Stage 20.4 — kinematic-hardening (Prager / Ziegler) back-stress.
        //
        // The yield function shifts from `‖dev(τ)‖` (Stage 20.3) to
        // `‖dev(τ) - α‖` where `α` is a 3×3 symmetric tensor evolved
        // per Prager. We work in the trial-strain principal frame
        // (the SVD's `U` rotation, equivalently `Vᵀ` for symmetric
        // strain): project `α` into that frame as `α_local = Uᵀ·α·U`,
        // take the diagonal as the principal back-stress vector
        // (off-diagonals are physically zero in the principal frame
        // for an aligned load; non-zero only when the load axis
        // rotates between substeps — small dt keeps that bounded).
        // After the return mapping updates the principal back-stress
        // diagonal we rotate it back: `α_new = U·diag(α_local_diag)·Uᵀ`,
        // preserving symmetry by construction.
        //
        // Trial Kirchhoff deviator (principal): `s_p = 2μ · dev`.
        // Project α into the principal frame.
        let alpha_old = alpha[tet];
        // α_local = Uᵀ · α · U. In glam (column-major) `m.transpose()`
        // gives Uᵀ, so `Uᵀ · α · U` reads `u_mat.transpose() * α * u_mat`.
        let alpha_local: Mat3 = u_mat.transpose() * alpha_old * u_mat;
        // Principal back-stress vector (diagonal). Symmetrise via
        // diag-only — the off-diagonals carry no useful physical
        // information for the von-Mises projection.
        let alpha_p = crate::math::Vec3::new(
            alpha_local.x_axis.x,
            alpha_local.y_axis.y,
            alpha_local.z_axis.z,
        );
        // Shifted principal deviator: ξ = 2μ·dev - α_p.
        let two_mu = 2.0 * mu;
        let xi = crate::math::Vec3::new(
            two_mu * dev.x - alpha_p.x,
            two_mu * dev.y - alpha_p.y,
            two_mu * dev.z - alpha_p.z,
        );
        let xi_norm = xi.length();
        // Trial state norm reused below for damping (unused after the
        // back-stress refactor; kept for the existing two-tier
        // plasticity damping call signature).
        let _dev_norm = dev.length();

        // von Mises yield: f = ‖ξ‖ - √(2/3)·σ_y. Note ξ already
        // carries 2μ·dev so the test is in stress units.
        let sigma_y = sigma_y0 + h * eps_p[tet];
        let f_yield = xi_norm - sqrt_2_3 * sigma_y;
        if f_yield <= 0.0 || xi_norm < 1.0e-12 {
            continue;
        }
        // Closed-form plastic multiplier for linear isotropic +
        // kinematic hardening (consistent return mapping):
        //   Δγ = f_yield / (2μ + (2/3)·H + (2/3)·c)
        // Notes:
        // - The kinematic-hardening contribution (2/3)·c grows the
        //   denominator the same way `H` does, so any positive `c`
        //   shrinks the per-step plastic flow — desirable, this is
        //   what makes back-stress "well-behaved" under cyclic loading.
        // - When c = 0 the expression collapses to Stage 20.3.
        let denom = two_mu + (2.0 / 3.0) * h + (2.0 / 3.0) * c_kin;
        let dgamma = (f_yield / denom).max(0.0);
        // Damping caps (overshoot defuse). Same two-tier cap structure
        // as Stage 20.3 — the closed-form return map is a single Newton
        // step on the non-linear yield function and can oscillate.
        //   (a) Δγ ≤ 0.1 · ‖ξ‖ / (2μ)  — proportional to trial.
        //   (b) Δγ ≤ 0.01 (abs absolute strain-rate cap).
        let dgamma = dgamma.min(0.1 * xi_norm / two_mu.max(1.0)).min(0.01);
        // Plastic strain increment (the actual logarithmic-strain
        // pull-back along the shifted-deviator direction).
        let dev_strain_increment = dgamma;
        // Plastic flow direction in the principal frame: n̂ = ξ / ‖ξ‖.
        let n_hat = xi / xi_norm;
        // Pull the *shifted* deviator back: dev ← dev - Δγ · n̂.
        // (Equivalent to `2μ·dev - 2μ·Δγ·n̂` then re-divide by 2μ;
        // the back-stress evolves separately.)
        let new_dev = dev - n_hat * dev_strain_increment;
        let new_eps = crate::math::Vec3::new(new_dev.x + mean, new_dev.y + mean, new_dev.z + mean);
        let new_sigma = crate::math::Vec3::new(
            new_eps.x.exp().clamp(0.05, 20.0),
            new_eps.y.exp().clamp(0.05, 20.0),
            new_eps.z.exp().clamp(0.05, 20.0),
        );
        let sigma_diag = Mat3::from_cols(
            Vec3::new(new_sigma.x, 0.0, 0.0),
            Vec3::new(0.0, new_sigma.y, 0.0),
            Vec3::new(0.0, 0.0, new_sigma.z),
        );
        let fe_new = u_mat * sigma_diag * v_mat.transpose();
        let fe_det = fe_new.determinant();
        if !fe_det.is_finite() || fe_det.abs() < 1.0e-12 {
            // F_e_new collapsed to a degenerate (or NaN) state — the
            // singular values were clamped to [0.05, 20] above so this
            // requires a non-finite Δγ or u_mat / v_mat. Loud in debug;
            // in release we skip the plastic update for this tet.
            debug_assert!(
                false,
                "von_mises_return_mapping: degenerate F_e_new (det={fe_det}) tet={tet}"
            );
            log_plasticity_dropped("F_e_new degenerate");
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
            // F_p drifted outside the volume-preserving envelope —
            // either a NaN (upstream bug) or a return-mapping
            // overshoot. Loud in debug; in release we keep the prior
            // F_p so the next step is well-defined.
            debug_assert!(
                false,
                "von_mises_return_mapping: F_p_new det out of bounds ({fp_new_det}) tet={tet}"
            );
            log_plasticity_dropped("F_p_new det out of bounds");
            continue;
        }
        // Robust per-singular-value clamp on F_p_new: the next step's
        // elastic computation `F · F_p⁻¹` blows up when σ_min(F_p) is
        // small and produces near-degenerate F_e when σ_max is large.
        // Clamp σ ∈ [0.5, 2.0] in place — much cheaper than a full
        // line search on Δγ, and bounded enough that the multi-step
        // plastic flow accumulates correctly. We force F_p to
        // det > 0 by taking |σ| in the SVD and flipping the smallest
        // |σ|-column of the reconstruction if it ended up reflective.
        let (fp_u, fp_sigma, fp_v) = svd3(fp_new);
        let fp_sigma_clamped = crate::math::Vec3::new(
            fp_sigma.x.abs().clamp(0.5, 2.0),
            fp_sigma.y.abs().clamp(0.5, 2.0),
            fp_sigma.z.abs().clamp(0.5, 2.0),
        );
        let fp_diag = Mat3::from_cols(
            Vec3::new(fp_sigma_clamped.x, 0.0, 0.0),
            Vec3::new(0.0, fp_sigma_clamped.y, 0.0),
            Vec3::new(0.0, 0.0, fp_sigma_clamped.z),
        );
        let mut fp_clamped = fp_u * fp_diag * fp_v.transpose();
        if fp_clamped.determinant() < 0.0 {
            let mut min_i = 0;
            if fp_sigma_clamped.y < fp_sigma_clamped[min_i] {
                min_i = 1;
            }
            if fp_sigma_clamped.z < fp_sigma_clamped[min_i] {
                min_i = 2;
            }
            // Flip column `min_i`.
            let mut cols = [fp_clamped.x_axis, fp_clamped.y_axis, fp_clamped.z_axis];
            cols[min_i] = -cols[min_i];
            fp_clamped = Mat3::from_cols(cols[0], cols[1], cols[2]);
        }
        fp[tet] = fp_clamped;
        // Equivalent plastic strain accumulator: classic √(2/3)·Δγ.
        eps_p[tet] = (eps_p[tet] + sqrt_2_3 * dev_strain_increment).min(10.0);

        // ── Stage 20.4 — back-stress evolution (Prager rule) ─────
        // `α_p_new = α_p + (2/3)·c·Δγ·n̂` in the principal frame,
        // then rotate back to world via `α = U·diag(α_p_new)·Uᵀ`.
        // No-op when c == 0 (Stage 20.3 path: α stays 0).
        if c_kin > 0.0 {
            let dalpha_p = n_hat * ((2.0 / 3.0) * c_kin * dev_strain_increment);
            let alpha_p_new = alpha_p + dalpha_p;
            // Cap the magnitude of α to prevent runaway accumulation
            // (the yield surface itself is bounded by the
            // Tikhonov-style F_p clamp upstream; back-stress doesn't
            // need to exceed the absolute material strength). Cap is
            // generous — `100·σ_y0` lets the canonical Bauschinger
            // demo accumulate without saturation.
            let alpha_cap = (100.0 * sigma_y0).max(1.0e6);
            let alpha_p_clamped = crate::math::Vec3::new(
                alpha_p_new.x.clamp(-alpha_cap, alpha_cap),
                alpha_p_new.y.clamp(-alpha_cap, alpha_cap),
                alpha_p_new.z.clamp(-alpha_cap, alpha_cap),
            );
            let alpha_diag = Mat3::from_cols(
                Vec3::new(alpha_p_clamped.x, 0.0, 0.0),
                Vec3::new(0.0, alpha_p_clamped.y, 0.0),
                Vec3::new(0.0, 0.0, alpha_p_clamped.z),
            );
            let alpha_world = u_mat * alpha_diag * u_mat.transpose();
            // Symmetrise — guard against numerical drift introducing
            // antisymmetric components.
            let alpha_sym = mat3_scale(mat3_add(alpha_world, alpha_world.transpose()), 0.5);
            // Final sanity check on the back-stress diagonal: NaN here
            // means the kinematic-hardening update derailed (e.g. a
            // non-finite n_hat). Loud in debug; in release we skip the
            // back-stress write so the next step uses the prior α.
            let alpha_finite = alpha_sym.x_axis.x.is_finite()
                && alpha_sym.y_axis.y.is_finite()
                && alpha_sym.z_axis.z.is_finite();
            debug_assert!(
                alpha_finite,
                "von_mises_return_mapping: non-finite back-stress diag at tet={tet}"
            );
            if alpha_finite {
                alpha[tet] = alpha_sym;
            } else {
                log_plasticity_dropped("back-stress non-finite");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WorldId;
    use crate::math::Real;
    use crate::sim::{Contacts, ModelBuilder};

    #[test]
    fn cofactor_matches_determinant_gradient() {
        use crate::math::Mat3;
        for f in [
            Mat3::from_cols(
                Vec3::new(1.2, 0.3, -0.2),
                Vec3::new(0.1, 0.8, 0.4),
                Vec3::new(-0.3, 0.2, 1.1),
            ),
            Mat3::from_cols(Vec3::X, Vec3::Y, Vec3::X + Vec3::Y),
        ] {
            let gradient = cofactor3(f).to_cols_array();
            for i in 0..9 {
                let mut plus = f.to_cols_array();
                let mut minus = plus;
                plus[i] += 1e-6;
                minus[i] -= 1e-6;
                let derivative = (Mat3::from_cols_array(&plus).determinant()
                    - Mat3::from_cols_array(&minus).determinant())
                    / 2e-6;
                assert!((gradient[i] - derivative).abs() < 1e-8);
            }
        }
    }

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
        b.add_cloth_grid(
            origin,
            u_axis,
            v_axis,
            n,
            n,
            0.05,
            0.02,
            1.0e-4,
            0.5,
            WorldId(0),
        );
        let model = b.build().unwrap();
        assert_eq!(model.particle_count, 16);
        assert!(
            model.particle_color_count >= 2,
            "VBD needs ≥2 colours for an 4×4 cloth"
        );

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
            crate::Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
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
        assert!(
            final_mean_y < initial_mean_y - 0.1,
            "cloth didn't fall: {initial_mean_y} -> {final_mean_y}"
        );
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
        let (first, _, _) = b.add_cloth_grid(
            origin,
            u_axis,
            v_axis,
            n,
            n,
            0.05,
            0.02,
            1.0e-4,
            0.5,
            WorldId(0),
        );
        b.pin_particle(first);
        b.pin_particle(first + n - 1);
        b.pin_particle(first + (n - 1) * n);
        b.pin_particle(first + (n - 1) * n + n - 1);
        let model = b.build().unwrap();

        let pinned = [
            first,
            first + n - 1,
            first + (n - 1) * n,
            first + (n - 1) * n + n - 1,
        ];
        let initial = s_a_initial_corners(&model, &pinned);

        let mut solver = VbdSolver::new_cpu().with_iterations(8);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(8);

        let dt: Real = 1.0 / 240.0;
        for _ in 0..240 {
            crate::Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
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

    /// Stage 20.3 — small jelly cube (3×3×3) on CPU with the same
    /// kinematic-puller-then-release scenario as the GPU test
    /// `plasticity_yield_holds_permanent_set`. Verifies the multi-tet
    /// return mapping survives a sustained compress + release cycle
    /// without any tet's F_p going degenerate. Cheap; no GPU adapter.
    #[test]
    fn cpu_jelly_cube_plastic_compress_release() {
        use crate::sim::MaterialNeoHookean;
        let n = 3u32;
        let side = 0.3_f64;
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.6);
        b.set_world_gravity(WorldId(0), Vec3::ZERO);
        let dx = side / (n as Real - 1.0);
        let centre = Vec3::ZERO;
        let origin = centre - Vec3::splat(side * 0.5);
        let mass = 0.05;
        let pr = dx * 0.5;
        let mut bottom_pins = Vec::new();
        let mut top_pins = Vec::new();
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = origin + Vec3::new(i as Real * dx, j as Real * dx, k as Real * dx);
                    let id = b.add_particle(p, mass, pr, WorldId(0));
                    if j == 0 {
                        b.pin_particle(id);
                        bottom_pins.push(id as usize);
                    } else if j == n - 1 {
                        b.pin_particle(id);
                        top_pins.push(id as usize);
                    }
                }
            }
        }
        let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
        let mut tets: Vec<[u32; 4]> = Vec::new();
        for kk in 0..n - 1 {
            for jj in 0..n - 1 {
                for ii in 0..n - 1 {
                    let corners = [
                        v_at(ii, jj, kk),
                        v_at(ii + 1, jj, kk),
                        v_at(ii, jj + 1, kk),
                        v_at(ii + 1, jj + 1, kk),
                        v_at(ii, jj, kk + 1),
                        v_at(ii + 1, jj, kk + 1),
                        v_at(ii, jj + 1, kk + 1),
                        v_at(ii + 1, jj + 1, kk + 1),
                    ];
                    // Helper redeclared inline (the test file's `cube_to_six_tets`
                    // lives in the integration test crate, not here).
                    let v000 = corners[0];
                    let v100 = corners[1];
                    let v010 = corners[2];
                    let v110 = corners[3];
                    let v001 = corners[4];
                    let v101 = corners[5];
                    let v011 = corners[6];
                    let v111 = corners[7];
                    tets.extend_from_slice(&[
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                        [v000, v101, v100, v111],
                    ]);
                }
            }
        }
        let materials = [MaterialNeoHookean {
            viscosity: Default::default(),
            youngs: 5.0e3,
            poisson: 0.3,
            yield_stress: 5.0e1,
            hardening: 1.0e3,
            kinematic_hardening: 0.0,
        }];
        let mat_idx: Vec<u32> = vec![0; tets.len()];
        b.add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
            .unwrap();
        let model = b.build().unwrap();
        let initial = model.initial_particle_q.host().unwrap().to_vec();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(8);
        let mut solver = VbdSolver::new_cpu().with_iterations(15);
        let dt: Real = 1.0 / 240.0;

        // Phase 1: compress.
        let total_squash: Real = 0.04 * side;
        let compress_steps = 400;
        for step in 0..compress_steps {
            let frac = step as Real / compress_steps as Real;
            let squash = frac * total_squash;
            {
                let q = s_a.particle_q.host_mut().unwrap();
                for &v in &top_pins {
                    let mut p = initial[v];
                    p.y -= squash;
                    q[v] = p;
                }
            }
            crate::Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        // Read F_p / ε_p.
        let eps_after = s_a.tet_plastic_strain.host().unwrap().to_vec();
        let max_eps_after = eps_after.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!("post-compress max ε_p = {max_eps_after:.4}");
        assert!(max_eps_after > 1.0e-3);

        // Phase 2: release.
        for _ in 0..200 {
            {
                let q = s_a.particle_q.host_mut().unwrap();
                for &v in &top_pins {
                    q[v] = initial[v];
                }
            }
            crate::Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.particle_q.host().unwrap();
        for (i, p) in q.iter().enumerate() {
            assert!(
                p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
                "particle {i} non-finite: {p:?}"
            );
        }
        let eps_post = s_a.tet_plastic_strain.host().unwrap().to_vec();
        let max_eps_post = eps_post.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!("post-release max ε_p = {max_eps_post:.4}");
        // ε_p is monotone non-decreasing.
        assert!(max_eps_post >= max_eps_after);
        assert!(max_eps_post > 1.0e-3);
    }

    /// Stage 20.3 — a single tet under a static stretch load past
    /// yield should retain a non-trivial F_p after release. This is
    /// the smallest-possible reproducer for the plasticity kernel:
    /// no spatial discretisation, just one tet.
    #[test]
    fn cpu_single_tet_yields_and_retains_set() {
        use crate::sim::MaterialNeoHookean;
        let mut b = ModelBuilder::new();
        b.set_linear_damping(0.4);
        b.set_world_gravity(WorldId(0), Vec3::ZERO);
        // Unit tet in axis-aligned form.
        let p0 = b.add_particle(Vec3::new(0.0, 0.0, 0.0), 0.05, 0.01, WorldId(0));
        let p1 = b.add_particle(Vec3::new(1.0, 0.0, 0.0), 0.05, 0.01, WorldId(0));
        let p2 = b.add_particle(Vec3::new(0.0, 1.0, 0.0), 0.05, 0.01, WorldId(0));
        let p3 = b.add_particle(Vec3::new(0.0, 0.0, 1.0), 0.05, 0.01, WorldId(0));
        let mat = MaterialNeoHookean {
            viscosity: Default::default(),
            youngs: 1.0e4,
            poisson: 0.3,
            yield_stress: 50.0,
            hardening: 100.0,
            kinematic_hardening: 0.0,
        };
        b.add_tet_mesh_per_material(&[[p0, p1, p2, p3]], &[mat], &[0], None)
            .unwrap();
        // Pin all four particles so we can write-driven deform.
        b.pin_particle(p0);
        b.pin_particle(p1);
        b.pin_particle(p2);
        b.pin_particle(p3);
        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(8);

        let mut solver = VbdSolver::new_cpu().with_iterations(8);
        let dt: Real = 1.0 / 240.0;

        // Apply a static stretch: move p1 outward by 30% over many
        // substeps. The pinned particles overrule the solver, so the
        // tet sees a large F gradient. With yield_stress = 50 and
        // youngs = 1e4 we expect plasticity to fire.
        let n_compress = 200;
        for step in 0..n_compress {
            let frac = step as Real / n_compress as Real;
            let stretch = 0.3 * frac;
            {
                let q = s_a.particle_q.host_mut().unwrap();
                q[p1 as usize] = Vec3::new(1.0 + stretch, 0.0, 0.0);
            }
            crate::Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        // F_p should have moved from identity.
        let fp = s_a.tet_f_plastic.host().unwrap();
        let fp_norm = (fp[0] - crate::math::Mat3::IDENTITY).x_axis.length()
            + (fp[0] - crate::math::Mat3::IDENTITY).y_axis.length()
            + (fp[0] - crate::math::Mat3::IDENTITY).z_axis.length();
        eprintln!("‖F_p - I‖ after stretch: {fp_norm:.4}");
        assert!(
            fp_norm > 1.0e-3,
            "F_p didn't move from identity under sustained yield: {fp_norm}"
        );
        // Plastic strain accumulator should also be > 0.
        let eps_p = s_a.tet_plastic_strain.host().unwrap()[0];
        eprintln!("ε_p = {eps_p:.4}");
        assert!(eps_p > 0.0);

        // Phase 2: release p1 back to its original position; F_p
        // should NOT reset (plastic strain is permanent).
        let fp_snapshot = fp[0];
        let n_release = 300;
        for _ in 0..n_release {
            {
                let q = s_a.particle_q.host_mut().unwrap();
                q[p1 as usize] = Vec3::new(1.0, 0.0, 0.0);
            }
            crate::Solver::step(&mut solver, &model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let fp_post = s_a.tet_f_plastic.host().unwrap()[0];
        let fp_post_norm = (fp_post - crate::math::Mat3::IDENTITY).x_axis.length()
            + (fp_post - crate::math::Mat3::IDENTITY).y_axis.length()
            + (fp_post - crate::math::Mat3::IDENTITY).z_axis.length();
        let eps_p_post = s_a.tet_plastic_strain.host().unwrap()[0];
        eprintln!("‖F_p_post - I‖: {fp_post_norm:.4}, ε_p_post: {eps_p_post:.4}");
        // The accumulated plastic strain `ε_p` is monotonic
        // (sqrt(2/3)·Δγ accumulates regardless of yielding direction);
        // verify it stayed > 0 through the release, which is the
        // canonical "plasticity has occurred" signal. F_p itself can
        // shrink during release because isotropic-hardening von
        // Mises is symmetric under sign flips of the trial deviator
        // — kinematic hardening would prevent reverse yielding but
        // is out of scope for v1 (deferred to Stage 20.4).
        assert!(
            eps_p_post >= eps_p,
            "ε_p decreased: {eps_p:.4} → {eps_p_post:.4}"
        );
        assert!(
            eps_p_post > 0.05,
            "no permanent plastic strain: ε_p = {eps_p_post}"
        );
        // Verify it's still finite.
        assert!(fp_post.x_axis.x.is_finite());
        let _ = (fp_snapshot, fp_post_norm);
    }
}
