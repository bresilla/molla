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
//! `lower == upper == 0` sentinel), Cable (Stage 39E — 2 DOF: 1 linear
//! stretch along joint-local +z + 1 bend about joint-local +x; per-joint
//! spring + damper coefficients live in `Model.joint_cable_*`). Fixed
//! passes through.

use crate::core::Error;
use crate::core::Result;
use crate::math::{
    Mat3, Quat, Real, SpatialMatrix6, SpatialVector, Transform, Vec3, spatial_cross,
    spatial_cross_dual,
};
use crate::sim::{
    BodyFlags, Contacts, Control, JointTargetMode, JointType, Model, State, eval_body_velocities,
    eval_fk,
};

use super::Solver;

#[path = "featherstone_activation.rs"]
mod activation;

#[path = "featherstone_scratch.rs"]
mod scratch;
use scratch::JointScratch;

#[path = "featherstone_singular_root.rs"]
mod singular_root;

#[path = "featherstone_integrate.rs"]
mod integration;

/// Maximum DOFs in any single joint — Free/Distance/D6 hit 6.
pub(crate) const MAX_JOINT_DOFS: usize = 6;

/// Backend selector for [`FeatherstoneSolver`].
#[allow(
    clippy::large_enum_variant,
    reason = "GPU variant holds Arcs / kernel; one solver instance, fine"
)]
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
    /// Stage 42 v3 — pre-pass that fills the per-DOF muscle torque
    /// accumulator before each Featherstone step. `None` when the
    /// model has no muscles, so empty scenes pay zero overhead.
    pub(crate) muscle_kernel: Option<crate::kernels::MuscleKernel>,
}

/// Forward-dynamics solver for articulated rigid bodies.
pub struct FeatherstoneSolver {
    contact_forces: Vec<crate::penalty_contact::ContactForce>,
    /// Joint-space damping coefficient (1/s) applied to `joint_qd`
    /// each step. Helps keep long-running pendulums numerically calm.
    pub joint_damping: Real,
    /// Number of substeps per outer `step` call. Each substep advances
    /// `dt/substeps`. Default 1.
    pub substeps: usize,
    /// CPU feedback gains use a per-DOF effective-inertia backward-Euler denominator.
    pub stable_motor_feedback: bool,
    /// Stage 17.1 polish: when `true`, the backward ABA pass builds the
    /// full per-joint `D = S^T · I^A · S` matrix and solves
    /// `D · q̈ = u_tau` via dense Cholesky instead of the v1 sequential
    /// rank-1 deflation. Required for asymmetric Free-joint inertias
    /// (e.g. CoM offset cross-couples linear/angular DOFs); for joints
    /// where `D` is already diagonal (Ball, Revolute, decoupled Free)
    /// the two paths agree to roundoff. Default `false` for backwards
    /// compatibility.
    pub rigorous_d_solve: bool,
    /// Stage 17.1 polish: when `true`, Distance joints enforce their
    /// length constraint via a velocity-level Lagrange multiplier
    /// (force-level projection) instead of the v1 hard
    /// `p ← p · L_target / |p|` clamp. Default `true` — the v1 clamp
    /// stays only as a fallback when the multiplier is degenerate.
    pub rigorous_distance: bool,
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
            stable_motor_feedback: false,
            rigorous_d_solve: false,
            rigorous_distance: true,
            backend: FeatherstoneBackend::Cpu,
            contact_forces: Vec::new(),
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
        Self::new_gpu_on_device(device, queue, model)
    }

    /// Construct articulated dynamics on the caller's matching device and queue.
    #[cfg(feature = "gpu")]
    pub fn new_gpu_on_device(
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        model: &Model,
    ) -> Result<Self> {
        let kernel = crate::kernels::FeatherstoneKernel::new(&device, &queue, model);
        let muscle_kernel = if model.muscle_count > 0 {
            Some(crate::kernels::MuscleKernel::new(&device, model))
        } else {
            None
        };
        Ok(Self {
            joint_damping: 0.0,
            substeps: 1,
            stable_motor_feedback: false,
            rigorous_d_solve: false,
            rigorous_distance: true,
            backend: FeatherstoneBackend::Gpu(GpuBackend {
                device,
                queue,
                kernel,
                muscle_kernel,
            }),
            contact_forces: Vec::new(),
        })
    }

    /// Integrate input coordinates using output velocities without evaluating dynamics.
    pub fn drift_cpu(
        &self,
        model: &Model,
        state_in: &State,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(Error::Solver("invalid articulated drift timestep".into()));
        }
        let q_in = state_in.joint_q.host()?;
        let qd_in = state_out.joint_qd.host()?.to_vec();
        let mut q = q_in.to_vec();
        let mut qd = qd_in.clone();
        let mut acceleration = vec![0.0; model.joint_dof_total];
        if model.joint_count > 0 {
            state_out.joint_q.host_mut()?.copy_from_slice(q_in);
            eval_fk(model, state_out)?;
            let mut mass = crate::joint_mass::JointMass::new(model, state_out);
            for art in 0..model.articulation_count.max(1) {
                crate::joint_limits::react_scalar_stops(
                    model, state_out, art, &mut acceleration, dt, 0.0, &mut mass,
                )?;
            }
        }
        integration::integrate(
            model, 0..model.joint_count, q_in, &qd_in,
            &acceleration, dt, 0.0,
            self.rigorous_distance, &mut q, &mut qd,
        )?;
        state_out.joint_q.host_mut()?.copy_from_slice(&q);
        state_out.joint_qd.host_mut()?.copy_from_slice(&qd);
        eval_fk(model, state_out)?;
        eval_body_velocities(model, state_out)
    }

    /// Integrate input coordinates using contact-corrected output velocities on the GPU.
    #[cfg(feature = "gpu")]
    pub fn drift_gpu(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        if !dt.is_finite() || !(dt as f32).is_finite() || dt as f32 <= 0.0 {
            return Err(Error::Solver("invalid articulated drift timestep".into()));
        }
        match &mut self.backend {
            FeatherstoneBackend::Gpu(gpu) => gpu.kernel.drift(
                &gpu.device,
                &gpu.queue,
                model,
                state_in,
                control,
                dt,
                self.rigorous_distance,
                state_out,
            ),
            _ => Err(Error::Gpu(
                "articulated drift requires a GPU backend".into(),
            )),
        }
    }

    /// Per-DOF joint damping coefficient (`τ_damp = -d · qd`),
    /// applied uniformly across all joints. Defaults to `0.0`.
    #[must_use]
    pub fn with_damping(mut self, d: Real) -> Self {
        self.joint_damping = d;
        self
    }

    /// Enable CPU effective-inertia stabilization of feedback motors.
    #[must_use]
    pub fn with_stable_motor_feedback(mut self, enabled: bool) -> Self {
        self.stable_motor_feedback = enabled;
        self
    }

    /// Configure substepping: each outer `step(dt)` runs `n` inner
    /// substeps with `dt/n`.
    #[must_use]
    pub fn with_substeps(mut self, n: usize) -> Self {
        self.substeps = n.max(1);
        self
    }

    /// Stage 17.1: enable the rigorous dense-Cholesky `D`-matrix solve
    /// for multi-DOF joints (Free / Distance / D6 / Ball). The default
    /// path uses Stage 17's sequential rank-1 deflation, which is exact
    /// only when `D` is diagonal. Toggle this on for Free joints whose
    /// inertia matrix is non-trivially coupled (e.g. CoM offset cross-
    /// couples linear and angular DOFs).
    #[must_use]
    pub fn with_rigorous_d_solve(mut self, on: bool) -> Self {
        self.rigorous_d_solve = on;
        self
    }

    /// Stage 17.1: toggle the velocity-level Lagrange-multiplier
    /// projection for Distance joints. Default `true`; when `false`
    /// the v1 hard-clamp (`p ← p · L_target / |p|` + radial-velocity
    /// zeroing) is used instead. The `false` setting is preserved for
    /// A/B parity tests.
    #[must_use]
    pub fn with_rigorous_distance(mut self, on: bool) -> Self {
        self.rigorous_distance = on;
        self
    }

    /// In-place setter for [`Self::with_rigorous_d_solve`] — the
    /// `with_*` variant consumes `self` (for builder chains), this one
    /// flips the flag on an already-built solver (used by viewer UI
    /// toggles, e.g. the Quadrotor scene's "rigorous" pane).
    pub fn set_rigorous_d_solve(&mut self, on: bool) {
        self.rigorous_d_solve = on;
    }

    /// In-place setter for [`Self::with_rigorous_distance`] — see
    /// [`Self::set_rigorous_d_solve`] for the `with_*` vs `set_*`
    /// convention rationale.
    pub fn set_rigorous_distance(&mut self, on: bool) {
        self.rigorous_distance = on;
    }

    /// Stage 42 v3 — run the GPU activation filter on every muscle.
    /// Mirrors the CPU [`crate::actuators::update_muscle_activation`]
    /// helper. No-op on the CPU backend (caller should invoke the
    /// host function directly) or on a GPU model without muscles.
    /// Must be called by the user before each step, just like the
    /// CPU path expects.
    ///
    /// # Errors
    ///
    /// Returns `Error::Gpu` if the muscle buffers haven't been
    /// uploaded.
    #[cfg(feature = "gpu")]
    pub fn update_muscle_activation_gpu(
        &mut self,
        model: &Model,
        state: &State,
        dt: Real,
    ) -> Result<()> {
        match &mut self.backend {
            FeatherstoneBackend::Cpu => Ok(()),
            FeatherstoneBackend::Gpu(gpu) => {
                if let Some(mk) = &gpu.muscle_kernel {
                    mk.dispatch_activation(&gpu.device, &gpu.queue, model, state, dt as f32)?;
                }
                Ok(())
            }
        }
    }

    /// Sticky GPU scalar-stop failure bit 32; no state readback is performed.
    #[cfg(feature = "gpu")]
    pub fn gpu_status_buffer(&self) -> Option<&wgpu::Buffer> {
        match &self.backend {
            FeatherstoneBackend::Gpu(gpu) => Some(gpu.kernel.status_buffer()),
            FeatherstoneBackend::Cpu => None,
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn set_gpu_status_buffer(&mut self, buffer: &wgpu::Buffer) -> Result<()> {
        match &mut self.backend {
            FeatherstoneBackend::Gpu(gpu) => gpu.kernel.set_status_buffer(buffer),
            FeatherstoneBackend::Cpu => Err(crate::core::Error::Solver(
                "GPU status requires GPU Featherstone".into(),
            )),
        }
    }

    /// `true` if the solver runs on a GPU backend.
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

    /// Stage 23.1 — same as [`Self::step`], but records the per-step
    /// inputs on a [`crate::autograd::Tape`] for later backward replay
    /// against [`crate::autograd::FeatherstoneRevoluteChain::reverse`].
    /// `chain` must mirror the model's articulation 0 (a serial revolute
    /// chain — multi-DOF joints aren't supported by the v1 adjoint, see
    /// [`crate::autograd::featherstone_adjoint`] doc).
    ///
    /// `q_id` / `qd_id` / `tau_id` must be tape ids whose values were
    /// seeded from the corresponding host buffers via
    /// [`crate::autograd::Tape::alloc`]. The function evaluates the
    /// step on the host (so `state_out` ends up populated for the
    /// downstream rendering / sensors), then emits the three TapeOp
    /// chunks via
    /// [`crate::autograd::record_featherstone_step`].
    ///
    /// Returns `(q_new_id, qd_new_id, ee_id)` — the tape ids the caller
    /// can chain into a downstream loss.
    #[cfg(feature = "autograd")]
    pub fn step_with_tape(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
        tape: &mut crate::autograd::Tape,
        chain: &crate::autograd::FeatherstoneRevoluteChain,
        q_id: crate::autograd::TapeId,
        qd_id: crate::autograd::TapeId,
        tau_id: crate::autograd::TapeId,
    ) -> Result<(
        crate::autograd::TapeId,
        crate::autograd::TapeId,
        crate::autograd::TapeId,
    )> {
        // Forward step into state_out for the downstream sim consumers
        // (rendering, sensors). The tape recorder runs the adjoint-side
        // forward independently against the captured chain.
        self.step(model, state_in, control, contacts, dt, state_out)?;
        let ids = crate::autograd::record_featherstone_step(tape, chain, q_id, qd_id, tau_id);
        Ok(ids)
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
        self.contact_forces.clear();
        if self.is_gpu() && self.stable_motor_feedback {
            return Err(crate::core::Error::Solver(
                "stable motor feedback requires CPU Featherstone".into(),
            ));
        }
        if self.is_gpu()
            && model
                .joint_compliance
                .host()?
                .iter()
                .any(|c| c.enabled != 0)
        {
            return Err(crate::core::Error::Solver(
                "joint compliance requires CPU Featherstone".into(),
            ));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(crate::core::Error::Solver(
                "step dt must be positive and finite".into(),
            ));
        }
        let n_sub = self.substeps.max(1);
        let dt_sub = dt / n_sub as Real;

        if n_sub == 1 {
            return self.step_inner(model, state_in, control, contacts, dt_sub, state_out);
        }

        let mut scratch = state_in.clone_host()?;
        self.step_inner(model, state_in, control, contacts, dt_sub, state_out)?;
        let mut samples = Vec::new();
        samples.append(&mut self.contact_forces);
        let mut output_in_scratch = false;
        for _ in 1..n_sub {
            if output_in_scratch {
                self.step_inner(model, &scratch, control, contacts, dt_sub, state_out)?;
            } else {
                self.step_inner(model, state_out, control, contacts, dt_sub, &mut scratch)?;
            }
            output_in_scratch = !output_in_scratch;
            samples.append(&mut self.contact_forces);
        }
        if output_in_scratch {
            state_out.copy_host_from(&scratch)?;
        }
        self.contact_forces = samples;
        Ok(())
    }
}

impl FeatherstoneSolver {
    /// CPU contact samples from every substep of the latest step; impulses sum over time.
    pub fn contact_forces(&self) -> &[crate::penalty_contact::ContactForce] {
        &self.contact_forces
    }

    /// CPU stepping with fresh collision detection and pair filtering on every substep.
    pub fn step_with_collision(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        pipeline: &mut crate::geometry::CollisionPipeline,
        contacts: &mut Contacts,
        dt: Real,
        state_out: &mut State,
        excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
    ) -> Result<()> {
        self.step_with_collision_and_forces(
            model,
            state_in,
            control,
            pipeline,
            contacts,
            dt,
            state_out,
            excluded,
            |_, _, _, _| Ok(()),
        )
    }

    /// Refresh collision and transient COM wrenches every substep without accumulating them.
    pub fn step_with_collision_and_forces(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        pipeline: &mut crate::geometry::CollisionPipeline,
        contacts: &mut Contacts,
        dt: Real,
        state_out: &mut State,
        excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
        forces: impl FnMut(&Model, &State, Real, &mut [SpatialVector]) -> Result<()>,
    ) -> Result<()> {
        self.step_with_collision_bounds(
            model, state_in, control, pipeline, contacts, dt, state_out, excluded, forces, None,
        )
    }

    pub(crate) fn step_with_collision_bounds(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        pipeline: &mut crate::geometry::CollisionPipeline,
        contacts: &mut Contacts,
        dt: Real,
        state_out: &mut State,
        excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
        mut forces: impl FnMut(&Model, &State, Real, &mut [SpatialVector]) -> Result<()>,
        bounds: Option<&crate::geometry::shape_world::PreparedShapeBounds<'_>>,
    ) -> Result<()> {
        self.step_with_collision_response(model, state_in, control, pipeline, contacts, dt,
            state_out, excluded, |model, state, dt, wrenches, _, _| {
                forces(model, state, dt, wrenches)
            }, bounds)
    }

    pub(crate) fn step_with_collision_response(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        pipeline: &mut crate::geometry::CollisionPipeline,
        contacts: &mut Contacts,
        dt: Real,
        state_out: &mut State,
        mut excluded: impl FnMut(Option<molla_core::BodyId>, Option<molla_core::BodyId>) -> bool,
        mut forces: impl FnMut(&Model, &State, Real, &mut [SpatialVector], &mut crate::joint_mass::MassSeed<'_>, &mut Contacts) -> Result<()>,
        bounds: Option<&crate::geometry::shape_world::PreparedShapeBounds<'_>>,
    ) -> Result<()> {
        self.contact_forces.clear();
        if !dt.is_finite() || dt <= 0.0 {
            return Err(crate::core::Error::Solver(
                "step dt must be positive and finite".into(),
            ));
        }
        if !matches!(self.backend, FeatherstoneBackend::Cpu) {
            return Err(crate::core::Error::Solver(
                "contact-refreshed Featherstone stepping requires the CPU backend".into(),
            ));
        }
        let n = self.substeps.max(1);
        let sub_dt = dt / n as Real;
        let mut scratch = state_in.clone_host()?;
        eval_fk(model, &mut scratch)?;
        eval_body_velocities(model, &mut scratch)?;
        let mut samples = Vec::new();
        let persistent = state_in.body_f.host()?.to_vec();
        let mut applied = persistent.clone();
        let owned_bounds;
        let bounds = match bounds {
            Some(bounds) if std::ptr::eq(bounds.model(), model) => bounds,
            Some(_) => return Err(crate::core::Error::Solver("bounds model mismatch".into())),
            None => {
                owned_bounds = crate::geometry::shape_world::PreparedShapeBounds::new(model);
                &owned_bounds
            }
        };
        for substep in 0..n {
            tracing::debug_span!("collision").in_scope(|| {
                pipeline.collide_prepared(bounds, &scratch, contacts, &mut excluded)
            })?;
            applied.copy_from_slice(&persistent);
            let mut mass_seed = crate::joint_mass::MassSeed::new(model);
            tracing::debug_span!("external_forces")
                .in_scope(|| forces(model, &scratch, sub_dt, &mut applied, &mut mass_seed, contacts))?;
            if applied
                .iter()
                .any(|f| !f.linear.is_finite() || !f.angular.is_finite())
            {
                return Err(crate::core::Error::Solver(
                    "non-finite transient body force".into(),
                ));
            }
            scratch.body_f.host_mut()?.copy_from_slice(&applied);
            tracing::debug_span!("featherstone_substep").in_scope(|| {
                self.step_inner_seeded(model, &scratch, control, contacts, sub_dt, state_out, Some(mass_seed))
            })?;
            state_out.body_f.host_mut()?.copy_from_slice(&persistent);
            samples.append(&mut self.contact_forces);
            if substep + 1 < n {
                std::mem::swap(&mut scratch, state_out);
            }
        }
        self.contact_forces = samples;
        Ok(())
    }

    fn step_inner(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        self.step_inner_seeded(model, state_in, control, contacts, dt, state_out, None)
    }

    fn step_inner_seeded(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
        mass_seed: Option<crate::joint_mass::MassSeed<'_>>,
    ) -> Result<()> {
        self.contact_forces.clear();
        if self.is_gpu() && model.joint_reversed.host()?.iter().any(|&r| r != 0) {
            return Err(crate::core::Error::Solver(
                "reversed joints require CPU Featherstone".into(),
            ));
        }
        if self.is_gpu() && control.joint_motor_model.host()?.iter().any(|&m| m != 0) {
            return Err(crate::core::Error::Solver(
                "acceleration motors require CPU Featherstone".into(),
            ));
        }
        if self.is_gpu()
            && control
                .joint_motor_max_force
                .host()?
                .iter()
                .any(|cap| cap.is_nan() || *cap < 0.0)
        {
            return Err(crate::core::Error::Solver(
                "motor-only caps must be nonnegative and not NaN".into(),
            ));
        }
        match &mut self.backend {
            FeatherstoneBackend::Cpu => {
                cpu_step(
                    model,
                    state_in,
                    control,
                    contacts,
                    dt,
                    state_out,
                    self.joint_damping,
                    self.stable_motor_feedback,
                    self.rigorous_d_solve,
                    self.rigorous_distance,
                    &mut self.contact_forces,
                    None,
                    mass_seed,
                )?;
                for reading in &mut self.contact_forces {
                    reading.sample_dt = dt;
                    reading.normal_impulse = reading.normal_force * dt;
                }
                Ok(())
            }
            #[cfg(feature = "gpu")]
            FeatherstoneBackend::Gpu(gpu) => {
                let GpuBackend {
                    device,
                    queue,
                    kernel,
                    muscle_kernel,
                } = gpu;
                // Stage 42 v3 — Hill-type muscle pre-pass. Compute
                // the per-joint torque sum from the *current*
                // activation (the caller must have already called
                // [`crate::actuators::update_muscle_activation`] —
                // either the CPU helper that mutates host memory, or
                // the GPU [`crate::kernels::MuscleKernel::
                // dispatch_activation`] entry point — to filter the
                // excitation into activation before stepping). The
                // CPU `cpu_step` makes the same assumption, so we
                // intentionally don't run the filter here.
                let muscle_buf = if let Some(mk) = muscle_kernel {
                    mk.dispatch_torque_sum(device, queue, model, state_in)?;
                    Some(mk.torque_buffer())
                } else {
                    None
                };
                kernel.dispatch(
                    device,
                    queue,
                    model,
                    state_in,
                    control,
                    contacts,
                    self.joint_damping as f32,
                    dt,
                    self.rigorous_d_solve,
                    self.rigorous_distance,
                    state_out,
                    muscle_buf,
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
/// - Cable (Stage 39E): 2 columns — col 0 = (ez; 0) linear stretch along
///   the joint-world +z (cable axis), col 1 = (0; ex) bend about the
///   joint-world +x (bend axis).
pub(crate) fn build_motion_subspace(
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
    // Every spatial quantity in this solver (body inertia, gravity
    // wrench, external wrench) is referenced at the WORLD ORIGIN, so
    // the joint motion subspace must be too. A unit rotation `ω` about
    // an axis through the joint origin `p` produces, at the world
    // origin, the spatial velocity `(ω × (-p); ω)` — the linear part
    // is the moment-arm term. Omitting it (treating `S` as the twist
    // at a point *on* the axis) makes the solver origin-dependent and
    // wrong for any joint not sitting at the world origin. Pure
    // translations (prismatic / free-linear / cable stretch) carry no
    // moment arm and are left as `(axis; 0)`. Matches the verified
    // CRBA/RNEA `spatial_jacobian_column`.
    let p = joint_world.position;
    let twist = |axis: Vec3| SpatialVector::new(axis.cross(-p), axis);
    let locked = |k| {
        let lo = limit_lower
            .get(dof_idx + k)
            .copied()
            .unwrap_or(Real::NEG_INFINITY);
        let hi = limit_upper
            .get(dof_idx + k)
            .copied()
            .unwrap_or(Real::INFINITY);
        lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12
    };
    match jt {
        JointType::Revolute => {
            let axis_world = (r * axis_local).normalize_or_zero();
            if !locked(0) {
                s_cols[0] = twist(axis_world);
                dof_active[0] = true;
            }
            1
        }
        JointType::Prismatic => {
            let axis_world = (r * axis_local).normalize_or_zero();
            if !locked(0) {
                s_cols[0] = SpatialVector::new(axis_world, Vec3::ZERO);
                dof_active[0] = true;
            }
            1
        }
        JointType::Ball => {
            for (k, axis) in [ex, ey, ez].into_iter().enumerate() {
                if !locked(k) {
                    s_cols[k] = twist(axis);
                    dof_active[k] = true;
                }
            }
            3
        }
        JointType::Free | JointType::Distance => {
            // 3 linear + 3 angular columns in joint world frame.
            s_cols[0] = SpatialVector::new(ex, Vec3::ZERO);
            s_cols[1] = SpatialVector::new(ey, Vec3::ZERO);
            s_cols[2] = SpatialVector::new(ez, Vec3::ZERO);
            s_cols[3] = twist(ex);
            s_cols[4] = twist(ey);
            s_cols[5] = twist(ez);
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
                twist(ex),
                twist(ey),
                twist(ez),
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
        JointType::Cable => {
            // Stage 39E — 2-column motion subspace:
            //   col 0: linear stretch along joint-world +z (cable axis)
            //   col 1: bend rotation about joint-world +x (bend axis)
            // Both axes are joint-frame canonical (deterministic) — the
            // child body inherits whatever rotation `joint_x_p` applies
            // when laying out the cable in the parent's body frame.
            s_cols[0] = SpatialVector::new(ez, Vec3::ZERO);
            s_cols[1] = twist(ex);
            dof_active[0] = true;
            dof_active[1] = true;
            2
        }
        JointType::Fixed => 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_step(
    model: &Model,
    state_in: &State,
    control: &Control,
    contacts: &Contacts,
    dt: Real,
    state_out: &mut State,
    joint_damping: Real,
    stable_motor_feedback: bool,
    rigorous_d_solve: bool,
    rigorous_distance: bool,
    contact_forces: &mut Vec<crate::penalty_contact::ContactForce>,
    activation: Option<&activation::ActiveSet>,
    mass_seed: Option<crate::joint_mass::MassSeed<'_>>,
) -> Result<()> {
    if model.articulation_count <= 1 {
        cpu_step_inner::<false>(model, state_in, control, contacts, dt, state_out,
            joint_damping, stable_motor_feedback, rigorous_d_solve, rigorous_distance,
            contact_forces, activation, mass_seed)
    } else {
        cpu_step_inner::<true>(model, state_in, control, contacts, dt, state_out,
            joint_damping, stable_motor_feedback, rigorous_d_solve, rigorous_distance,
            contact_forces, activation, mass_seed)
    }
}

fn cpu_step_inner<const LOCAL: bool>(
    model: &Model,
    state_in: &State,
    control: &Control,
    contacts: &Contacts,
    dt: Real,
    state_out: &mut State,
    joint_damping: Real,
    stable_motor_feedback: bool,
    rigorous_d_solve: bool,
    rigorous_distance: bool,
    contact_forces: &mut Vec<crate::penalty_contact::ContactForce>,
    activation: Option<&activation::ActiveSet>,
    mass_seed: Option<crate::joint_mass::MassSeed<'_>>,
) -> Result<()> {
    let setup_span = tracing::debug_span!("aba_setup").entered();
    if model.joint_count == 0 {
        state_out.copy_host_from(state_in)?;
        for (i, &flags) in model.body_flags.host()?.iter().enumerate() {
            if flags & BodyFlags::FIXED != 0 {
                state_out.body_qd.host_mut()?[i] = SpatialVector::ZERO;
            }
        }
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
            if activation.is_some_and(|active| !active.joints[a]) {
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
    state_out
        .body_qd
        .host_mut()?
        .copy_from_slice(state_in.body_qd.host()?);
    eval_fk(model, state_out)?;
    eval_body_velocities(model, state_out)?;

    let mut joint_mass = crate::joint_mass::JointMass::with_seed(model, state_out, mass_seed)?;
    let motor_inertias = effective_drive_inertias(
        model, control, stable_motor_feedback, &mut joint_mass,
        activation.map(|active| active.articulations.as_slice()),
    )?;

    let joint_type = model.joint_type.host()?;
    let joint_parent = model.joint_parent.host()?;
    let joint_child = model.joint_child.host()?;
    let joint_axis = model.joint_axis.host()?;
    let joint_coord_offset = model.joint_coord_offset.host()?;
    let joint_dof_offset = model.joint_dof_offset.host()?;
    let joint_enabled = model.joint_enabled.host()?;
    let art_start = model.articulation_start.host()?;
    let art_count = model.articulation_count.max(1);

    let body_q = state_out.body_q.host()?.to_vec();
    let mut body_f_in = state_in.body_f.host()?.to_vec();
    crate::joint_compliance::apply(model, state_out, &motor_inertias, dt, &mut body_f_in)?;
    *contact_forces =
        crate::penalty_contact::eval_penalty_contacts(model, state_out, contacts, &mut body_f_in)?;
    if let Some(active) = activation {
        active.filter_readings(model, contact_forces)?;
        if !active.joints.iter().any(|&selected| selected) {
            active.restore(model, state_in, state_out)?;
            state_out.body_f.host_mut()?.copy_from_slice(state_in.body_f.host()?);
            return Ok(());
        }
    }
    let body_mass = model.body_mass.host()?;
    let body_flags = model.body_flags.host()?;
    let body_inertia = model.body_inertia.host()?;
    let body_com = model.body_com.host()?;
    let body_world = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let control_force = control.joint_force.host()?;
    let limit_lower = model.joint_limit_lower.host()?;
    let limit_upper = model.joint_limit_upper.host()?;
    let joint_friction = model.joint_friction.host()?;
    let effort_limit = model.joint_effort_limit.host()?;
    let cable_k_lin = model.joint_cable_stiffness_linear.host()?;
    let cable_k_bend = model.joint_cable_stiffness_bend.host()?;
    let cable_c_lin = model.joint_cable_damping_linear.host()?;
    let cable_c_bend = model.joint_cable_damping_bend.host()?;
    // Stage 42 v2 — Hill-type muscle actuators. CSR maps each joint
    // to the list of muscles attached to it (co-contraction summed at
    // the solver level). `joint_muscle_offsets[j+1] − [j]` is the
    // number of muscles on joint `j`.
    let muscle_f_max = model.muscle_f_max.host()?;
    let muscle_l_opt = model.muscle_l_opt.host()?;
    let muscle_v_max = model.muscle_v_max.host()?;
    let muscle_activation = model.muscle_activation.host()?;
    let muscle_l_tendon_slack = model.muscle_l_tendon_slack.host()?;
    let muscle_tendon_stiffness_ratio = model.muscle_tendon_stiffness_ratio.host()?;
    let muscle_pennation_angle = model.muscle_pennation_angle.host()?;
    let muscle_polarity = model.muscle_polarity.host()?;
    let joint_muscle_offsets = model.joint_muscle_offsets.host()?;
    let joint_muscle_indices = model.joint_muscle_indices.host()?;
    let target_mode = control.joint_target_mode.host()?;
    let target_pos = control.joint_target_pos.host()?;
    let target_vel = control.joint_target_vel.host()?;
    let target_kp = control.joint_target_kp.host()?;
    let target_kd = control.joint_target_kd.host()?;
    let motor_max_force = control.joint_motor_max_force.host()?;
    let motor_models = control.joint_motor_model.host()?;

    let scratch_len = if LOCAL {
        (0..art_count)
            .filter(|&art| activation.is_none_or(|active| active.articulations[art]))
            .map(|art| (art_start[art + 1] - art_start[art]) as usize)
            .max()
            .unwrap_or(0)
    } else {
        model.joint_count
    };
    let mut v_body = JointScratch::<_, LOCAL>::new(scratch_len, SpatialVector::ZERO);
    let mut c_term = JointScratch::<_, LOCAL>::new(scratch_len, SpatialVector::ZERO);
    let mut i_a = JointScratch::<_, LOCAL>::new(scratch_len, SpatialMatrix6::ZERO);
    let mut p_a = JointScratch::<_, LOCAL>::new(scratch_len, SpatialVector::ZERO);
    let mut s_cols = JointScratch::<_, LOCAL>::new(scratch_len, [SpatialVector::ZERO; MAX_JOINT_DOFS]);
    let mut dof_active = JointScratch::<_, LOCAL>::new(scratch_len, [false; MAX_JOINT_DOFS]);
    let mut dof_count = JointScratch::<_, LOCAL>::new(scratch_len, 0usize);
    let mut u_per_dof = JointScratch::<_, LOCAL>::new(scratch_len, [SpatialVector::ZERO; MAX_JOINT_DOFS]);
    let mut d_inv_per_dof = JointScratch::<_, LOCAL>::new(scratch_len, [0.0_f64; MAX_JOINT_DOFS]);
    let mut u_tau_per_dof = JointScratch::<_, LOCAL>::new(scratch_len, [0.0_f64; MAX_JOINT_DOFS]);
    let mut l_chol_per_joint = JointScratch::<_, LOCAL>::new(scratch_len, [0.0_f64; MAX_JOINT_DOFS * MAX_JOINT_DOFS]);

    // ── Per-articulation outer loop ─────────────────────────────────
    let mut q_new: Vec<Real> = q_in.clone();
    let mut qd_new: Vec<Real> = qd_in.clone();
    let mut q_dd = vec![0.0_f64; model.joint_dof_total];
    drop(setup_span);
    let mut a_body = JointScratch::<_, LOCAL>::new(scratch_len, SpatialVector::ZERO);
    for art in 0..art_count {
        if activation.is_some_and(|active| !active.articulations[art]) {
            continue;
        }
        let j_begin = art_start[art] as usize;
        let j_end = art_start[art + 1] as usize;
        if j_begin == j_end {
            continue;
        }

        v_body.reset(j_begin, j_end);
        c_term.reset(j_begin, j_end);
        i_a.reset(j_begin, j_end);
        p_a.reset(j_begin, j_end);
        s_cols.reset(j_begin, j_end);
        dof_active.reset(j_begin, j_end);
        dof_count.reset(j_begin, j_end);
        u_per_dof.reset(j_begin, j_end);
        d_inv_per_dof.reset(j_begin, j_end);
        u_tau_per_dof.reset(j_begin, j_end);
        l_chol_per_joint.reset(j_begin, j_end);
        a_body.reset(j_begin, j_end);
        let mut root_applied_tau = [0.0; MAX_JOINT_DOFS];

        // ── Pass 1: forward kinematics + body kinematics ─────────────────
        let kinematics_span = tracing::debug_span!("aba_kinematics").entered();
        for j in j_begin..j_end {
            let jt = joint_type_from_u32(joint_type[j]);
            let parent = joint_parent[j];
            let child = joint_child[j] as usize;

            let (motion_frame, direction) = model.joint_motion_frame(&body_q, &q_in, j)?;

            // Build motion subspace (per-DOF).
            let dof_idx = joint_dof_offset[j] as usize;
            let mut n_dof = build_motion_subspace(
                jt,
                motion_frame,
                joint_axis[j],
                dof_idx,
                limit_lower,
                limit_upper,
                &mut s_cols[j],
                &mut dof_active[j],
            );
            for column in &mut s_cols[j] {
                *column = column.scale(direction);
            }
            if body_flags[child] & BodyFlags::IMMOVABLE != 0 || joint_enabled[j] == 0 {
                n_dof = 0;
                s_cols[j].fill(SpatialVector::ZERO);
                dof_active[j].fill(false);
            }
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

            let v_i = if body_flags[child] & BodyFlags::FIXED != 0 {
                SpatialVector::ZERO
            } else if body_flags[child] & BodyFlags::KINEMATIC != 0 {
                let velocity = state_in.body_qd.host()?[child];
                let com = body_q[child].position + body_q[child].rotation * body_com[child];
                SpatialVector::new(
                    velocity.linear - velocity.angular.cross(com),
                    velocity.angular,
                )
            } else {
                v_p + v_j
            };
            v_body[j] = v_i;
            c_term[j] = spatial_cross(v_i, v_j);
            if matches!(jt, JointType::Free | JointType::Distance | JointType::D6) {
                let translation = motion_frame.rotation
                    * Vec3::new(qd_in[dof_idx], qd_in[dof_idx + 1], qd_in[dof_idx + 2]);
                c_term[j].linear -= v_j.angular.cross(translation);
            }

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
        drop(kinematics_span);

        // ── Pass 2: backward — articulated inertia accumulation ──────────
        let backward_span = tracing::debug_span!("aba_backward").entered();
        for j in (j_begin..j_end).rev() {
            let parent = joint_parent[j];
            let jt = joint_type_from_u32(joint_type[j]);
            let enabled = joint_enabled.get(j).copied().unwrap_or(1) != 0;
            let n_dof = dof_count[j];
            let supported = enabled && !matches!(jt, JointType::Fixed);

            let dof_idx = joint_dof_offset[j] as usize;
            let coord_idx = joint_coord_offset[j] as usize;

            if supported && n_dof > 0 {
                // Compute per-DOF generalised force `u_tau_k = tau_k − S_k^T · p_a[j]`
                // from drives + friction + effort first; this is
                // independent of the deflation strategy below.
                for k in 0..n_dof {
                    if !dof_active[j][k] {
                        u_tau_per_dof[j][k] = 0.0;
                        continue;
                    }
                    let s_k = s_cols[j][k];
                    let dof_g = dof_idx + k;
                    let qd_now = qd_in.get(dof_g).copied().unwrap_or(0.0);
                    let q_now = motor_coordinate(jt, &q_in[coord_idx..], k);
                    let mode =
                        JointTargetMode::from_u32(target_mode.get(dof_g).copied().unwrap_or(0));
                    let kp = target_kp.get(dof_g).copied().unwrap_or(0.0);
                    let kd = target_kd.get(dof_g).copied().unwrap_or(0.0);
                    let pos_t = target_pos.get(dof_g).copied().unwrap_or(0.0);
                    let vel_t = target_vel.get(dof_g).copied().unwrap_or(0.0);
                    let drive = match mode {
                        JointTargetMode::None => 0.0,
                        JointTargetMode::Position => kp * (pos_t - q_now) - kd * qd_now,
                        JointTargetMode::Velocity => kd * (vel_t - qd_now),
                        JointTargetMode::PositionVelocity => {
                            kp * (pos_t - q_now) + kd * (vel_t - qd_now)
                        }
                        JointTargetMode::Effort => pos_t,
                    };
                    let scale =
                        if mode != JointTargetMode::Effort && motor_models.get(dof_g) == Some(&1) {
                            motor_inertias[dof_g]
                        } else {
                            1.0
                        };
                    let implicit_drive = if stable_motor_feedback
                        && (1..=3).contains(&(mode as u32))
                        && motor_inertias[dof_g] > 0.0
                    {
                        let stiffness = if mode == JointTargetMode::Velocity {
                            0.0
                        } else {
                            kp
                        };
                        (drive - stiffness * dt * qd_now) * scale
                            / (1.0 + (kd + stiffness * dt) * scale * dt / motor_inertias[dof_g])
                    } else {
                        drive * scale
                    };
                    let drive_torque = clamp_or_passthrough(
                        implicit_drive,
                        motor_max_force
                            .get(dof_g)
                            .copied()
                            .unwrap_or(Real::INFINITY),
                    );
                    let mu = joint_friction.get(dof_g).copied().unwrap_or(0.0);
                    let friction_torque = -mu * qd_now;
                    // Stage 39E — CABLE per-DoF spring + damper. Only applied
                    // when joint type is CABLE (the buffers are zero for
                    // every other type).
                    let cable_torque = if matches!(jt, JointType::Cable) {
                        let (k_stiff, c_damp) = if k == 0 {
                            (
                                cable_k_lin.get(j).copied().unwrap_or(0.0),
                                cable_c_lin.get(j).copied().unwrap_or(0.0),
                            )
                        } else {
                            (
                                cable_k_bend.get(j).copied().unwrap_or(0.0),
                                cable_c_bend.get(j).copied().unwrap_or(0.0),
                            )
                        };
                        -k_stiff * q_now - c_damp * qd_now
                    } else {
                        0.0
                    };
                    // Stage 42 v2 — Hill-type muscle actuators. Sum over
                    // every muscle attached to this joint via the CSR
                    // (`joint_muscle_offsets[j] .. [j+1]`). Only the
                    // linear stretch DoF (k == 0) is driven; muscles on
                    // CABLE bend would need oblique-fibre pennation we
                    // don't model yet.
                    let muscle_torque = if matches!(jt, JointType::Cable)
                        && k == 0
                        && j + 1 < joint_muscle_offsets.len()
                    {
                        let lo = joint_muscle_offsets[j] as usize;
                        let hi = joint_muscle_offsets[j + 1] as usize;
                        let mut sum: Real = 0.0;
                        for slot in lo..hi {
                            let m_id = joint_muscle_indices.get(slot).copied().unwrap_or(u32::MAX)
                                as usize;
                            if m_id == usize::MAX {
                                continue;
                            }
                            let m = crate::actuators::Muscle {
                                joint: crate::core::JointId::from_index(j),
                                config: crate::actuators::MuscleConfig {
                                    f_max: muscle_f_max.get(m_id).copied().unwrap_or(0.0),
                                    l_opt: muscle_l_opt.get(m_id).copied().unwrap_or(1.0),
                                    v_max: muscle_v_max.get(m_id).copied().unwrap_or(1.0),
                                    activation: muscle_activation.get(m_id).copied().unwrap_or(0.0),
                                    excitation: 0.0,
                                    tau_activation: crate::actuators::DEFAULT_TAU_ACTIVATION,
                                    l_tendon_slack: muscle_l_tendon_slack
                                        .get(m_id)
                                        .copied()
                                        .unwrap_or(0.0),
                                    tendon_stiffness_ratio: muscle_tendon_stiffness_ratio
                                        .get(m_id)
                                        .copied()
                                        .unwrap_or(
                                            crate::actuators::DEFAULT_TENDON_STIFFNESS_RATIO,
                                        ),
                                    pennation_angle: muscle_pennation_angle
                                        .get(m_id)
                                        .copied()
                                        .unwrap_or(0.0),
                                    polarity: muscle_polarity.get(m_id).copied().unwrap_or(1.0),
                                },
                            };
                            // Sign convention: each muscle's `Muscle::torque`
                            // already applies its own `polarity`. Agonists
                            // pull q_l negative, antagonists pull it
                            // positive; balanced pairs cancel.
                            sum += m.torque(q_now, qd_now);
                        }
                        sum
                    } else {
                        0.0
                    };
                    let tau_raw = control_force.get(dof_g).copied().unwrap_or(0.0)
                        + drive_torque
                        + friction_torque
                        + cable_torque
                        + muscle_torque;
                    let tau_app = clamp_or_passthrough(
                        tau_raw,
                        effort_limit.get(dof_g).copied().unwrap_or(Real::NAN),
                    );
                    u_tau_per_dof[j][k] = tau_app - sv_dot(s_k, p_a[j]);
                    if j == j_begin {
                        root_applied_tau[k] = tau_app;
                    }
                }

                if rigorous_d_solve && n_dof > 1 {
                    // Stage 17.1 rigorous path — full dense Cholesky on D.
                    //   U_block = I^A · S        (6 × n_dof)
                    //   D       = S^T · U_block  (n_dof × n_dof)
                    //   I^a     = I^A − U · D^{-1} · U^T (6×6, full block-deflate)
                    //   p^a     = p^A + I^a · c + U · (D^{-1} · u_tau)
                    let i_a_j = i_a[j];
                    // Build U = I^A · S and the lower-right active mask.
                    let mut active_idx: [usize; MAX_JOINT_DOFS] = [0; MAX_JOINT_DOFS];
                    let mut n_act: usize = 0;
                    for k in 0..n_dof {
                        if dof_active[j][k] {
                            u_per_dof[j][k] = i_a_j.mul_sv(s_cols[j][k]);
                            active_idx[n_act] = k;
                            n_act += 1;
                        } else {
                            u_per_dof[j][k] = SpatialVector::ZERO;
                        }
                    }

                    if n_act == 0 {
                        if parent >= 0 {
                            let mut parent_joint: Option<usize> = None;
                            for k in j_begin..j {
                                if joint_child[k] as usize == parent as usize {
                                    parent_joint = Some(k);
                                    break;
                                }
                            }
                            if let Some(pj) = parent_joint {
                                i_a[pj] = i_a[pj].add(&i_a_j);
                                p_a[pj] = p_a[pj] + p_a[j];
                            }
                        }
                        continue;
                    }

                    // Pack n_act × n_act D and n_act-vector u_tau.
                    let mut d_mat = [0.0; MAX_JOINT_DOFS * MAX_JOINT_DOFS];
                    let mut u_tau_packed = [0.0; MAX_JOINT_DOFS];
                    for ai in 0..n_act {
                        let ki = active_idx[ai];
                        let s_ki = s_cols[j][ki];
                        u_tau_packed[ai] = u_tau_per_dof[j][ki];
                        for aj in 0..n_act {
                            let kj = active_idx[aj];
                            d_mat[ai * n_act + aj] = sv_dot(s_ki, u_per_dof[j][kj]);
                        }
                    }

                    // Cholesky factor — fall back to per-DOF rank-1 if D
                    // is not SPD (singular column / locked DOFs).
                    let mut d_chol = d_mat;
                    let chol_ok = dense_chol(&mut d_chol[..n_act * n_act], n_act).is_ok();
                    if !chol_ok {
                        // Degenerate D — fall back to sequential rank-1.
                        apply_rank1_backward(
                            j,
                            n_dof,
                            &mut i_a,
                            &mut p_a,
                            &dof_active,
                            &s_cols,
                            &mut u_per_dof,
                            &mut d_inv_per_dof,
                            &u_tau_per_dof,
                            &c_term,
                            joint_parent,
                            joint_child,
                            j_begin,
                        );
                        continue;
                    }
                    // Stash the Cholesky factor for the forward pass.
                    for r in 0..n_act {
                        for c in 0..n_act {
                            l_chol_per_joint[j][r * MAX_JOINT_DOFS + c] = d_chol[r * n_act + c];
                        }
                    }
                    // Solve D · y = u_tau via the cached Cholesky factor.
                    let d_inv_u_tau = solve_joint_block(&d_chol, &u_tau_packed, n_act);

                    // I^a = I^A − Σ_{i,j} U_i · (D^{-1})_{ij} · U_j^T
                    // Build via the Cholesky factor too: for each column k
                    // of U, solve `D · w_k = U^T_k` and accumulate
                    // `i_a − = Σ_k U_k · w_k^T`. Cheaper expression: form
                    // `W = U · D^{-1}` (6 × n_act); then `i_a = i_a − W · U^T`.
                    // Build I^a directly: I^a = I^A − U_block · D^{-1} · U_block^T
                    // via I^a = I^A − Σ_k U_k · (D^{-1} U^T)_k where the
                    // intermediate is column k of D^{-1} U^T.
                    // Compute cols of D^{-1} U^T: for each spatial component
                    // c in 0..6, pack u_c = (U_0[c], U_1[c], …) and solve.
                    let mut dinv_ut = [[0.0; MAX_JOINT_DOFS]; 6];
                    for cc in 0..6 {
                        let mut rhs = [0.0; MAX_JOINT_DOFS];
                        for ai in 0..n_act {
                            let ki = active_idx[ai];
                            let u_ki = u_per_dof[j][ki];
                            rhs[ai] = sv_component(u_ki, cc);
                        }
                        dinv_ut[cc] = solve_joint_block(&d_chol, &rhs, n_act);
                    }
                    // Now i_a = i_a − Σ_k U_k · row_k(D^{-1} U^T)
                    // row_k(D^{-1} U^T) is a 6-vector: (dinv_ut[0][k], …, dinv_ut[5][k]).
                    let mut i_curr = i_a_j;
                    for ai in 0..n_act {
                        let ki = active_idx[ai];
                        let u_ki = u_per_dof[j][ki];
                        let row = SpatialVector::new(
                            Vec3::new(dinv_ut[0][ai], dinv_ut[1][ai], dinv_ut[2][ai]),
                            Vec3::new(dinv_ut[3][ai], dinv_ut[4][ai], dinv_ut[5][ai]),
                        );
                        let outer = SpatialMatrix6::outer(u_ki, row);
                        i_curr = i_curr.sub(&outer);
                    }

                    // p^a = p^A + I^a · c + Σ_k U_k · (D^{-1} u_tau)_k
                    let mut p_articulated = p_a[j] + i_curr.mul_sv(c_term[j]);
                    for ai in 0..n_act {
                        let ki = active_idx[ai];
                        p_articulated = p_articulated + u_per_dof[j][ki].scale(d_inv_u_tau[ai]);
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
                } else {
                    // v1 sequential rank-1 deflation (default — exact when
                    // D is diagonal, approximate but stable when not).
                    apply_rank1_backward(
                        j,
                        n_dof,
                        &mut i_a,
                        &mut p_a,
                        &dof_active,
                        &s_cols,
                        &mut u_per_dof,
                        &mut d_inv_per_dof,
                        &u_tau_per_dof,
                        &c_term,
                        joint_parent,
                        joint_child,
                        j_begin,
                    );
                }
            } else if parent >= 0
                && body_flags[joint_child[j] as usize] & BodyFlags::IMMOVABLE == 0
                && let Some(pj) = (j_begin..j).find(|&k| joint_child[k] == parent)
            {
                i_a[pj] = i_a[pj].add(&i_a[j]);
                p_a[pj] = p_a[pj] + p_a[j] + i_a[j].mul_sv(c_term[j]);
            }
        }

        drop(backward_span);
        // ── Pass 3: forward — solve for joint accelerations ──────────────
        let acceleration_span = tracing::debug_span!("aba_accelerations").entered();
        for j in j_begin..j_end {
            let parent = joint_parent[j];
            let jt = joint_type_from_u32(joint_type[j]);
            let enabled = joint_enabled.get(j).copied().unwrap_or(1) != 0;
            let n_dof = dof_count[j];
            let supported = enabled && !matches!(jt, JointType::Fixed);

            if body_flags[joint_child[j] as usize] & BodyFlags::IMMOVABLE != 0 {
                continue;
            }

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

            let a_bias = a_p + c_term[j];
            if supported && n_dof > 0 {
                let dof_idx = joint_dof_offset[j] as usize;
                let mut s_qdd = SpatialVector::ZERO;

                if rigorous_d_solve && n_dof > 1 {
                    // Build active list + RHS vector `r_k = u_tau_k − U_k^T · a_p`.
                    let mut active_idx: [usize; MAX_JOINT_DOFS] = [0; MAX_JOINT_DOFS];
                    let mut n_act: usize = 0;
                    for k in 0..n_dof {
                        if dof_active[j][k] {
                            active_idx[n_act] = k;
                            n_act += 1;
                        }
                    }
                    if n_act > 0 {
                        let mut rhs = [0.0; MAX_JOINT_DOFS];
                        for ai in 0..n_act {
                            let ki = active_idx[ai];
                            rhs[ai] = u_tau_per_dof[j][ki] - sv_dot(u_per_dof[j][ki], a_bias);
                        }
                        // Re-pack the cached Cholesky factor into a flat n_act×n_act.
                        let mut l_packed = [0.0; MAX_JOINT_DOFS * MAX_JOINT_DOFS];
                        for r in 0..n_act {
                            for c in 0..n_act {
                                l_packed[r * n_act + c] =
                                    l_chol_per_joint[j][r * MAX_JOINT_DOFS + c];
                            }
                        }
                        // Solve D · q̈ = rhs via the stored Cholesky.
                        // If Cholesky was zeroed (degenerate fallback path),
                        // l_packed.diag is zero → forward_sub guards via 1/L_ii.
                        let degenerate = (0..n_act).any(|i| l_packed[i * n_act + i].abs() < 1e-30);
                        if degenerate {
                            // Fall back to the sequential rank-1 result.
                            for k in 0..n_dof {
                                if !dof_active[j][k] {
                                    continue;
                                }
                                let q_dd_k = (u_tau_per_dof[j][k]
                                    - sv_dot(u_per_dof[j][k], a_bias))
                                    * d_inv_per_dof[j][k];
                                q_dd[dof_idx + k] = q_dd_k;
                                s_qdd = s_qdd + s_cols[j][k].scale(q_dd_k);
                            }
                        } else {
                            let q_dd_active = solve_joint_block(&l_packed, &rhs, n_act);
                            for ai in 0..n_act {
                                let ki = active_idx[ai];
                                let q_dd_k = q_dd_active[ai];
                                q_dd[dof_idx + ki] = q_dd_k;
                                s_qdd = s_qdd + s_cols[j][ki].scale(q_dd_k);
                            }
                        }
                    }
                } else {
                    for k in 0..n_dof {
                        if !dof_active[j][k] {
                            continue;
                        }
                        let q_dd_k = (u_tau_per_dof[j][k] - sv_dot(u_per_dof[j][k], a_bias))
                            * d_inv_per_dof[j][k];
                        q_dd[dof_idx + k] = q_dd_k;
                        s_qdd = s_qdd + s_cols[j][k].scale(q_dd_k);
                    }
                }
                a_body[j] = a_bias + s_qdd;
            } else {
                a_body[j] = a_bias;
            }
        }

        drop(acceleration_span);
        let limits_span = tracing::debug_span!("aba_constraints").entered();
        if j_end == j_begin + 1
            && let Some(acceleration) = singular_root::acceleration(
                model, state_out, art, j_begin, &root_applied_tau,
                &body_f_in, &mut joint_mass,
            )?
        {
            let first = joint_dof_offset[j_begin] as usize;
            q_dd[first..first + acceleration.len()].copy_from_slice(&acceleration);
        }

        crate::joint_limits::react_scalar_stops(
            model,
            state_out,
            art,
            &mut q_dd,
            dt,
            joint_damping,
            &mut joint_mass,
        )?;

        // ── Integrate q, q̇ for joints in this articulation ──────────────
        drop(limits_span);
        let integration_span = tracing::debug_span!("aba_integration").entered();
        integration::integrate(
            model, j_begin..j_end, &q_in, &qd_in, &q_dd, dt,
            joint_damping, rigorous_distance, &mut q_new, &mut qd_new,
        )?;
        drop(integration_span);
    } // end per-articulation outer loop

    let _finalize_span = tracing::debug_span!("aba_finalize").entered();
    state_out.joint_q.host_mut()?.copy_from_slice(&q_new);
    state_out.joint_qd.host_mut()?.copy_from_slice(&qd_new);
    eval_fk(model, state_out)?;
    eval_body_velocities(model, state_out)?;

    if let Some(active) = activation {
        active.restore(model, state_in, state_out)?;
    }

    // Propagate input body_f to output so substepping (which swaps
    // state_a / state_b each iteration) sees the same external wrench
    // each substep until the caller overwrites it.
    state_out
        .body_f
        .host_mut()?
        .copy_from_slice(state_in.body_f.host()?);
    Ok(())
}

fn sv_dot(a: SpatialVector, b: SpatialVector) -> Real {
    a.linear.dot(b.linear) + a.angular.dot(b.angular)
}

/// Index a spatial vector by its 6-dim component index `[lin.x, lin.y,
/// lin.z, ang.x, ang.y, ang.z]`.
#[inline]
fn motor_coordinate(kind: JointType, q: &[Real], axis: usize) -> Real {
    let angular = |offset: usize, axis: usize| {
        let r = Quat::from_xyzw(q[offset], q[offset + 1], q[offset + 2], q[offset + 3]);
        let (x, y, z) = r.to_euler(glam::EulerRot::XYZ);
        [x, y, z][axis]
    };
    match kind {
        JointType::Revolute | JointType::Prismatic | JointType::Cable => q[axis],
        JointType::Ball => angular(0, axis),
        JointType::Free | JointType::D6 | JointType::Distance if axis < 3 => q[axis],
        JointType::Free | JointType::D6 | JointType::Distance => angular(3, axis - 3),
        JointType::Fixed => 0.0,
    }
}

fn project_angular_limits(
    rotation: Quat,
    mut velocity: Vec3,
    offset: usize,
    lower: &[Real],
    upper: &[Real],
) -> (Quat, Vec3) {
    let (x, y, z) = rotation.to_euler(glam::EulerRot::XYZ);
    let mut angles = [x, y, z];
    let mut changed = false;
    for k in 0..3 {
        let (angle, rate) =
            clamp_joint_pos(angles[k], velocity[k], lower[offset + k], upper[offset + k]);
        changed |= angle != angles[k];
        angles[k] = angle;
        velocity[k] = rate;
    }
    let rotation = if changed {
        Quat::from_euler(glam::EulerRot::XYZ, angles[0], angles[1], angles[2]).normalize()
    } else {
        rotation
    };
    (rotation, velocity)
}

/// Compute 1 / (M^-1)[i,i] for active feedback motors and joint springs.
fn effective_drive_inertias(
    model: &Model,
    control: &Control,
    stable_feedback: bool,
    joint_mass: &mut crate::joint_mass::JointMass<'_>,
    active: Option<&[bool]>,
) -> Result<Vec<Real>> {
    let _span = tracing::debug_span!("drive_inertias").entered();
    let mut result = vec![1.0; model.joint_dof_total];
    let models = control.joint_motor_model.host()?;
    let modes = control.joint_target_mode.host()?;
    let compliance = model.joint_compliance.host()?;
    let requested = |dof: usize| {
        ((stable_feedback || models.get(dof) == Some(&1))
            && modes.get(dof).is_some_and(|m| (1..=3).contains(m)))
            || compliance
                .get(dof)
                .is_some_and(|c| c.enabled != 0 && c.natural_frequency > 0.0)
    };
    if !(0..model.joint_dof_total).any(requested) {
        return Ok(result);
    }
    let starts = model.articulation_start.host()?;
    let offsets = model.joint_dof_offset.host()?;
    for articulation in 0..model.articulation_count.max(1) {
        let (begin, end) = if model.articulation_count == 0 {
            (0, model.joint_count)
        } else {
            (
                starts[articulation] as usize,
                starts[articulation + 1] as usize,
            )
        };
        let first = offsets[begin] as usize;
        let count = (offsets[end] - offsets[begin]) as usize;
        if active.is_some_and(|active| !active[articulation]) {
            result[first..first + count].fill(0.0);
            continue;
        }
        if !(first..first + count).any(requested) {
            continue;
        }
        if let Some((scale, factor)) = joint_mass.cached_cholesky(articulation, count) {
            let mut workspace = vec![0.0; count];
            for dof in 0..count {
                if requested(first + dof) {
                    let inverse = unit_inverse_diagonal(factor, count, dof, &mut workspace);
                    result[first + dof] = if inverse > 0.0 { scale / inverse } else { 0.0 };
                }
            }
            continue;
        }
        let mass = joint_mass.matrix(articulation)?;
        if mass.iter().any(|v| !v.is_finite()) {
            return Err(crate::core::Error::Solver(
                "non-finite drive inertia".into(),
            ));
        }
        let scale = mass.iter().copied().map(Real::abs).fold(0.0, Real::max);
        let active: Vec<usize> = (0..count)
            .filter(|&i| mass[i * count + i] > scale * 1e-14)
            .collect();
        let n = active.len();
        let mut matrix = vec![0.0; n * n];
        for (i, &a) in active.iter().enumerate() {
            for (j, &b) in active.iter().enumerate() {
                matrix[i * n + j] = mass[a * count + b] / scale;
            }
        }
        let mut factor = matrix.clone();
        let factored = dense_chol(&mut factor, n).is_ok();
        let finite_factor = factored && factor.iter().all(|v| v.is_finite());
        let mut workspace = vec![0.0; n];
        let mut singular = None;
        for dof in 0..count {
            if !requested(first + dof) {
                continue;
            }
            result[first + dof] = 0.0;
            let Some(index) = active.iter().position(|&i| i == dof) else {
                continue;
            };
            let inverse_diagonal = if finite_factor {
                unit_inverse_diagonal(&factor, n, index, &mut workspace)
            } else {
                workspace.fill(0.0);
                workspace[index] = 1.0;
                if factored {
                    forward_sub_packed(&factor, &workspace, n)
                        .iter()
                        .map(|v| v * v)
                        .sum()
                } else {
                    if singular.is_none() {
                        singular = Some(crate::rigid_impulse::MassFactor::new(&matrix, n)?);
                    }
                    singular.as_ref().unwrap().solve(&workspace)?[index]
                }
            };
            if inverse_diagonal > 0.0 {
                result[first + dof] = scale / inverse_diagonal;
            }
        }
        if factored && n == count {
            joint_mass.retain_cholesky(articulation, factor, count, scale)?;
        }
    }
    Ok(result)
}

/// Diagonal of `(L Lᵀ)⁻¹` from the nonzero suffix of `L⁻¹ e_index`, for finite L.
/// Uses the packed forward solve's `1e-30` pivot cutoff.
fn unit_inverse_diagonal(l: &[Real], n: usize, index: usize, work: &mut [Real]) -> Real {
    for i in index..n {
        let mut value = if i == index { 1.0 } else { 0.0 };
        for (&coefficient, &previous) in l[i * n + index..i * n + i]
            .iter()
            .zip(&work[index..i])
        {
            value -= coefficient * previous;
        }
        let diagonal = l[i * n + i];
        work[i] = if diagonal.abs() < 1e-30 {
            0.0
        } else {
            value / diagonal
        };
    }
    work[index..n].iter().map(|v| v * v).sum()
}

fn sv_component(s: SpatialVector, c: usize) -> Real {
    match c {
        0 => s.linear.x,
        1 => s.linear.y,
        2 => s.linear.z,
        3 => s.angular.x,
        4 => s.angular.y,
        5 => s.angular.z,
        _ => 0.0,
    }
}

/// In-place Cholesky for an n×n SPD matrix in row-major flat storage.
/// Returns Err if the matrix has a non-positive pivot (D singular /
/// indefinite — falls back to sequential rank-1 in the caller). This is
/// a thin wrapper around [`crate::math::dense_solve::dense_chol`] kept
/// local so the featherstone solver can also call it on packed sub-
/// blocks of MAX_JOINT_DOFS² scratch.
fn dense_chol(a: &mut [Real], n: usize) -> Result<()> {
    crate::math::dense_solve::dense_chol(a, n)
}

/// Forward substitution `L · y = b` returning `y`. `l` is row-major
/// `n × n` lower triangular; only the lower triangle (including the
/// diagonal) is read.
fn forward_sub_packed(l: &[Real], b: &[Real], n: usize) -> Vec<Real> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        let l_ii = l[i * n + i];
        if l_ii.abs() < 1e-30 {
            y[i] = 0.0;
        } else {
            y[i] = s / l_ii;
        }
    }
    y
}

/// Solve a packed joint Cholesky block using stack-backed substitutions.
fn solve_joint_block(
    l: &[Real],
    rhs: &[Real; MAX_JOINT_DOFS],
    n: usize,
) -> [Real; MAX_JOINT_DOFS] {
    let mut y = [0.0; MAX_JOINT_DOFS];
    for i in 0..n {
        let mut value = rhs[i];
        for k in 0..i {
            value -= l[i * n + k] * y[k];
        }
        let pivot = l[i * n + i];
        y[i] = if pivot.abs() < 1e-30 { 0.0 } else { value / pivot };
    }
    let mut x = [0.0; MAX_JOINT_DOFS];
    for i in (0..n).rev() {
        let mut value = y[i];
        for k in i + 1..n {
            value -= l[k * n + i] * x[k];
        }
        let pivot = l[i * n + i];
        x[i] = if pivot.abs() < 1e-30 { 0.0 } else { value / pivot };
    }
    x
}

/// Back substitution `Lᵀ · x = y` returning `x`.
#[cfg(test)]
fn back_sub_packed(l: &[Real], y: &[Real], n: usize) -> Vec<Real> {
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        let l_ii = l[i * n + i];
        if l_ii.abs() < 1e-30 {
            x[i] = 0.0;
        } else {
            x[i] = s / l_ii;
        }
    }
    x
}

/// Stage 17's sequential rank-1 deflation, factored out so the rigorous
/// path can fall back to it cleanly when the dense `D` block turns out
/// to be singular (e.g. all DOFs locked through limits).
#[allow(clippy::too_many_arguments)]
fn apply_rank1_backward<const LOCAL: bool>(
    j: usize,
    n_dof: usize,
    i_a: &mut JointScratch<SpatialMatrix6, LOCAL>,
    p_a: &mut JointScratch<SpatialVector, LOCAL>,
    dof_active: &JointScratch<[bool; MAX_JOINT_DOFS], LOCAL>,
    s_cols: &JointScratch<[SpatialVector; MAX_JOINT_DOFS], LOCAL>,
    u_per_dof: &mut JointScratch<[SpatialVector; MAX_JOINT_DOFS], LOCAL>,
    d_inv_per_dof: &mut JointScratch<[Real; MAX_JOINT_DOFS], LOCAL>,
    u_tau_per_dof: &JointScratch<[Real; MAX_JOINT_DOFS], LOCAL>,
    c_term: &JointScratch<SpatialVector, LOCAL>,
    joint_parent: &[i32],
    joint_child: &[i32],
    j_begin: usize,
) {
    let mut i_curr = i_a[j];
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
        let outer = SpatialMatrix6::outer(u_k, u_k).scale(d_inv_k);
        i_curr = i_curr.sub(&outer);
    }

    let mut p_articulated = p_a[j] + i_curr.mul_sv(c_term[j]);
    for k in 0..n_dof {
        if !dof_active[j][k] {
            continue;
        }
        p_articulated =
            p_articulated + u_per_dof[j][k].scale(u_tau_per_dof[j][k] * d_inv_per_dof[j][k]);
    }

    let parent = joint_parent[j];
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

/// Clamp `x` to `[-limit, +limit]`. `limit` must be finite (NaN
/// indicates an upstream bug — the builder substitutes NaN sentinels
/// with `Real::INFINITY` at finalize). `Real::INFINITY` makes the
/// clamp a no-op, expressing "no limit" without a runtime branch.
#[inline]
fn clamp_or_passthrough(x: Real, limit: Real) -> Real {
    debug_assert!(
        !limit.is_nan(),
        "clamp_or_passthrough: limit must not be NaN"
    );
    x.clamp(-limit, limit)
}

/// Project a joint coordinate to its limits and zero its velocity at
/// the boundary. `lo` / `hi` must not be NaN (the builder substitutes
/// `±Real::INFINITY` for the "unset" sentinel at finalize); infinite
/// bounds make the corresponding branch unreachable, matching the
/// "unbounded" semantics without a runtime NaN check.
#[inline]
fn clamp_joint_pos(q: Real, qd: Real, lo: Real, hi: Real) -> (Real, Real) {
    debug_assert!(!lo.is_nan(), "clamp_joint_pos: lower bound must not be NaN");
    debug_assert!(!hi.is_nan(), "clamp_joint_pos: upper bound must not be NaN");
    let mut q_out = q;
    let mut qd_out = qd;
    if q_out < lo {
        q_out = lo;
        if qd_out < 0.0 {
            qd_out = 0.0;
        }
    }
    if q_out > hi {
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
        7 => JointType::Cable,
        _ => JointType::Fixed,
    }
}

#[cfg(test)]
#[path = "featherstone_mass_seed.rs"]
mod mass_seed_tests;

#[cfg(test)]
#[path = "featherstone_factor_seed.rs"]
mod factor_seed_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BodyId;
    use crate::math::{SpatialInertia, Transform, Vec3 as V3};
    use crate::sim::{BodyParams, ModelBuilder};

    #[test]
    fn joint_blocks_match_allocated_substitutions_exactly() {
        for n in 0..=MAX_JOINT_DOFS {
            for pivot in [0.0, 1e-31, 1e-30, 1e-25, -0.7, 0.7, Real::INFINITY, Real::NAN] {
                let mut factor = [Real::NAN; MAX_JOINT_DOFS * MAX_JOINT_DOFS];
                let mut rhs = [Real::NAN; MAX_JOINT_DOFS];
                for i in 0..n {
                    rhs[i] = ((i * 5 + 1) as Real).cos();
                    factor[i * n + i] = if i == n / 2 { pivot } else { 0.5 + i as Real * 0.2 };
                    for k in 0..i {
                        factor[i * n + k] = ((i * 7 + k * 3) as Real).sin() * 0.03;
                    }
                }
                let expected = back_sub_packed(&factor, &forward_sub_packed(&factor, &rhs, n), n);
                let actual = solve_joint_block(&factor, &rhs, n);
                for (&a, &b) in actual[..n].iter().zip(&expected) {
                    assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                        "n={n}, pivot={pivot}: {a} != {b}");
                }
                assert!(actual[n..].iter().all(|&x| x == 0.0));
            }
        }
    }

    #[test]
    fn unit_inverse_diagonals_match_full_forward_substitution() {
        for n in [1, 2, 6, 39, 78] {
            for pivot in [0.0, 1e-31, 1e-25, 0.7] {
                let mut factor = vec![0.0; n * n];
                for i in 0..n {
                    factor[i * n + i] = if i == n / 2 {
                        pivot
                    } else {
                        0.5 + i as Real * 0.02
                    };
                    for k in 0..i {
                        factor[i * n + k] = ((i * 7 + k * 3) as Real).sin() * 0.03;
                    }
                }
                let mut work = vec![Real::NAN; n];
                for index in (0..n).rev() {
                    let mut unit = vec![0.0; n];
                    unit[index] = 1.0;
                    let expected: Real = forward_sub_packed(&factor, &unit, n)
                        .iter()
                        .map(|v| v * v)
                        .sum();
                    let actual = unit_inverse_diagonal(&factor, n, index, &mut work);
                    assert_eq!(actual, expected, "n={n}, index={index}, pivot={pivot}");
                }
            }
        }
    }

    #[test]
    fn drive_inertias_match_full_inverse_diagonals() {
        for scale in [1e-60, 1.0, 1e60] {
            for massless_tip in [false, true] {
                let mut builder = ModelBuilder::new();
                let mut parent = BodyId::NONE;
                for i in 0..5 {
                    let weight = if massless_tip && i == 4 { 0.0 } else { scale };
                    let body = builder.add_body(
                        BodyParams::new().with_spatial_inertia(SpatialInertia {
                            mass: weight * (1.0 + i as Real),
                            com: V3::new(0.1, 0.2, -0.1),
                            inertia: Mat3::IDENTITY * weight,
                        }),
                    );
                    builder
                        .add_joint_revolute(
                            parent,
                            body,
                            Transform::from_translation(V3::new(0.3, 0.5, 0.2)),
                            Transform::IDENTITY,
                            if i % 2 == 0 { V3::Z } else { V3::X },
                        )
                        .unwrap();
                    parent = body;
                }
                let model = builder.build().unwrap();
                let mut state = model.state().unwrap();
                for (i, q) in state.joint_q.host_mut().unwrap().iter_mut().enumerate() {
                    *q = i as Real * 0.2;
                }
                eval_fk(&model, &mut state).unwrap();
                let mut control = model.control();
                control
                    .joint_target_mode
                    .host_mut()
                    .unwrap()
                    .fill(JointTargetMode::Velocity as u32);
                let mut cache = crate::joint_mass::JointMass::new(&model, &state);
                let result = effective_drive_inertias(&model, &control, true, &mut cache, None).unwrap();
                let mass = crate::mass_matrix(&model, &state, 0).unwrap();
                let shared = cache.factor(0, 5).unwrap();
                for (i, actual) in result.iter().enumerate() {
                    let mut unit = vec![0.0; 5];
                    unit[i] = 1.0;
                    let fresh = crate::rigid_impulse::solve_mass(&mass, &unit, 5).unwrap();
                    assert_eq!(shared.solve(&unit).unwrap(), fresh);
                    let inverse = fresh[i];
                    let expected = if inverse > 0.0 { inverse.recip() } else { 0.0 };
                    assert!(
                        (actual - expected).abs() <= 1e-10 * scale,
                        "scale={scale} massless={massless_tip} dof={i}: {actual} != {expected}"
                    );
                }
                assert_eq!(
                    effective_drive_inertias(&model, &control, false, &mut cache, None).unwrap(),
                    vec![1.0; 5]
                );
            }
        }
    }

    #[test]
    fn drive_inertias_preserve_singular_response_fallback() {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(
            BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::from_cols(V3::new(1.0, 1.0, 0.0), V3::new(1.0, 1.0, 0.0), V3::ZERO),
            }),
        );
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        eval_fk(&model, &mut state).unwrap();
        let mut control = model.control();
        control
            .joint_target_mode
            .host_mut()
            .unwrap()
            .fill(JointTargetMode::Velocity as u32);
        let mut cache = crate::joint_mass::JointMass::new(&model, &state);
        let actual = effective_drive_inertias(&model, &control, true, &mut cache, None).unwrap();
        let mass = crate::mass_matrix(&model, &state, 0).unwrap();
        let mut compact = vec![0.0; 25];
        for i in 0..5 {
            for j in 0..5 {
                compact[i * 5 + j] = mass[i * 6 + j];
            }
        }
        assert!(dense_chol(&mut compact, 5).is_err());
        for index in 0..6 {
            let mut unit = vec![0.0; 6];
            unit[index] = 1.0;
            let inverse = crate::rigid_impulse::solve_mass(&mass, &unit, 6).unwrap()[index];
            let expected = if inverse > 0.0 { inverse.recip() } else { 0.0 };
            assert!((actual[index] - expected).abs() < 1e-12);
        }
    }

    /// Pendulum at small initial angle should oscillate with the
    /// linearised period `T = 2π·√(I_eff / (m·g·L))`. Check the
    /// joint angle returns close to the initial value after one
    /// period (within 5 % — any energy gain or large drift fails).
    #[test]
    fn pendulum_oscillates_with_correct_period() {
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
        s_a.joint_q.host_mut().unwrap()[0] = 0.1;

        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new();

        let dt = 0.001_f64;
        let period =
            2.0 * std::f64::consts::PI * (1.25_f64 / (1.0 * crate::math::GRAVITY * 0.5)).sqrt();
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
        assert!(qd.abs() < 0.05, "friction failed to damp pendulum: qd={qd}");
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
            solver_single
                .step(&single, &sa, &ctl, &contacts, 0.001, &mut sb)
                .unwrap();
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
            solver
                .step(&model, &sa, &ctl, &contacts, 0.001, &mut sb)
                .unwrap();
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
                solver
                    .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
                    .unwrap();
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
        assert!(y.abs() < 1e-3, "body_f cancel failed: y drifted to {y}");
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
            .add_joint_d6(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
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
        assert!(pos.length() < 1e-3, "locked D6 drifted: pos={pos:?}");
    }
}
