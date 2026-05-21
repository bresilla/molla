// ────────────────────────────────────────────────────────────────────
// Stage-23.1 differentiable-Featherstone scene state
// ────────────────────────────────────────────────────────────────────

const DIFF_ARM6_OPT_ITERS: usize = 250;
const DIFF_ARM6_PLAYBACK_SPEED: f64 = 0.5;

// Stage 23.1.3 — joint kind selectable from the right-rail picker.
// Each variant maps to a `FeatherstoneRevoluteChain` constructor; the
// chain rebuilds when the user picks a different kind. Kept separate
// from `molla::autograd::JointKind` because that enum is per-link
// metadata; this enum is "which whole demo chain to run".
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DiffArm6Kind {
    /// 6-DOF revolute Franka (Stage 23.1 default).
    Revolute,
    /// 3-DoF Ball-joint spherical pendulum (Stage 23.1.1).
    Ball,
    /// 6-DoF Free-joint floating body (Stage 23.1.1).
    Free,
    /// 3-DoF Distance-joint tethered body (Stage 23.1.3).
    Distance,
    /// 6-DoF D6 joint with per-axis angular limits (Stage 23.1.3).
    D6,
}

impl DiffArm6Kind {
    fn label(self) -> &'static str {
        match self {
            Self::Revolute => "Revolute (6-DOF Franka)",
            Self::Ball => "Ball (spherical pendulum)",
            Self::Free => "Free (floating body)",
            Self::Distance => "Distance (tethered)",
            Self::D6 => "D6 (limited Free)",
        }
    }

    /// Build the chain for this kind. All chains use `dt = 0.005` and a
    /// 60-step horizon so the GD loop dynamics stay comparable across
    /// variants.
    fn build_chain(self) -> FeatherstoneRevoluteChain {
        match self {
            Self::Revolute => {
                let mut c = FeatherstoneRevoluteChain::franka6(0.005, 60);
                c.damping = 1.0;
                c
            }
            Self::Ball => {
                let mut c = FeatherstoneRevoluteChain::spherical_pendulum(0.5, 1.0, 0.005, 60);
                c.damping = 0.5;
                c
            }
            Self::Free => {
                let mut c = FeatherstoneRevoluteChain::free_body(1.0, 0.05, 0.005, 60);
                c.gravity = molla::math::Vec3::ZERO;
                c.damping = 0.2;
                c
            }
            Self::Distance => {
                let mut c =
                    FeatherstoneRevoluteChain::add_joint_distance(1.0, 0.5, 0.005, 60, 1.0e-3);
                c.gravity = molla::math::Vec3::ZERO;
                c.damping = 0.1;
                c
            }
            Self::D6 => FeatherstoneRevoluteChain::add_joint_d6(
                1.0,
                0.05,
                0.005,
                60,
                [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
                [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)],
            ),
        }
    }
}

// Stage-23.1 6-DOF Featherstone differentiable-arm runtime state.
// Mirrors [`DiffArmExampleState`] but for the
// [`FeatherstoneRevoluteChain::franka6`] chain. The pane drives the
// EE target XYZ, displays residual + iters-to-converge, and lets the
// user toggle the EE trajectory overlay.
#[derive(Resource)]
struct DiffArm6ExampleState {
    chain: FeatherstoneRevoluteChain,
    /// Target end-effector position in world coordinates. Default sits
    /// just off the rest pose so the optimiser does meaningful work
    /// the first frame.
    target: MVec3,
    /// Current optimised torque sequence. Length = `chain.horizon`.
    tau: Vec<Vec<f64>>,
    /// EE positions across all optimisation iterations — used for the
    /// "show optimisation trajectory" overlay. Cleared on re-run.
    iter_ee_trace: Vec<MVec3>,
    /// Final residual after the last optimisation pass.
    last_residual: f64,
    /// Iteration count at which the optimisation reached 5 cm of the
    /// target (`None` if it never did).
    converged_iter: Option<usize>,
    /// Total optimisation iterations actually run (for the readout).
    iters_run: usize,
    /// Re-run pending? Toggled by the pane's "Re-run optimisation"
    /// button or by a target slider change. Consumed by
    /// `update_diff_arm6` next frame.
    rerun: bool,
    /// Show the iteration trail of EE positions as a gizmo.
    show_trajectory: bool,
    /// Wall-clock since scene entry — used to animate playback.
    playback_t: f64,
    /// Reusable scratch for per-step Jacobian columns. Lives on the
    /// state so the GD loop doesn't re-allocate every iteration.
    scratch: JacobianScratch,
    /// Stage 23.1.1: kept around for back-compat. Stage 23.1.2 promoted
    /// the boolean to the 4-way `backend` picker below; this field now
    /// just mirrors `backend == GpuFallbackCpu` so any pre-existing code reading
    /// it stays correct.
    use_gpu_adjoint: bool,
    /// Stage 23.1.2 — 4-way backend picker (Auto / CPU CF / CPU AnalyticQd
    /// / CPU FD / GPU). The pane converts user clicks into this enum;
    /// the optimiser reads it once per re-run.
    backend: molla::kernels::AdjointBackend,
    /// Stage 23.1.2 — multi-articulation batch demo toggle. `true` runs
    /// N parallel Franka chains on every re-run via `MultiAdjointBatch`,
    /// which exercises the cross-chain parallelism that's the headline
    /// GPU win. `false` keeps the single-chain optimisation loop.
    use_multi_articulation: bool,
    /// Stage 23.1.2 — N for the multi-articulation batch toggle.
    multi_articulation_n: usize,
    /// Stage 23.1.1 readout: backward-pass / forward-pass time ratio
    /// from the last optimisation run. Demonstrates the closed-form
    /// vs FD perf win visually next to the iter counter.
    last_bwd_fwd_ratio: f64,
    /// Stage 23.1.2 readout: per-chain wall time (microseconds) on the
    /// last multi-articulation batch run. Single-chain runs leave it 0.
    last_per_chain_us: f64,
    /// Stage 23.1.2 readout: parallel speedup vs a single sequential
    /// reverse pass. Computed on the multi-articulation re-run.
    last_parallel_speedup: f64,
    /// Stage 23.1.3 — which joint kind drives the chain. Picker on the
    /// right-rail pane swaps between the Stage-23.1 6-DOF Franka, the
    /// Stage-23.1.1 Ball / Free demos, and the Stage-23.1.3 Distance /
    /// D6 demos so the user can verify the gradient flows for each.
    kind: DiffArm6Kind,
}

impl Default for DiffArm6ExampleState {
    fn default() -> Self {
        let kind = DiffArm6Kind::Revolute;
        let chain = kind.build_chain();
        let n = chain.links.len();
        let h = chain.horizon;
        // Target offset from rest in the X-Z plane (Y is hardest to
        // reach because gravity acts there).
        let rest_ee = chain.end_effector(&vec![0.0; n]);
        let target_world = MVec3::new(
            rest_ee.x as f64 + 0.07,
            rest_ee.y as f64 + DIFF_ARM6_BASE_HEIGHT,
            rest_ee.z as f64 - 0.07,
        );
        Self {
            scratch: JacobianScratch::new(n),
            chain,
            target: target_world,
            tau: vec![vec![0.0; n]; h],
            iter_ee_trace: Vec::new(),
            last_residual: f64::INFINITY,
            converged_iter: None,
            iters_run: 0,
            rerun: true,
            show_trajectory: true,
            playback_t: 0.0,
            use_gpu_adjoint: false,
            backend: molla::kernels::AdjointBackend::Auto,
            use_multi_articulation: false,
            multi_articulation_n: 10,
            last_bwd_fwd_ratio: 0.0,
            last_per_chain_us: 0.0,
            last_parallel_speedup: 0.0,
            kind,
        }
    }
}

// --- Stage-23.1 differentiable Featherstone arm scene ---------------
//
// Six kinematic capsule bodies representing the links of a 6-DOF
// Franka-style serial chain. Same playback model as `DiffArm`: the
// dynamics live entirely host-side in `update_diff_arm6` via
// [`FeatherstoneRevoluteChain`]; here we just declare six kinematic
// bodies whose `body_q` will be rewritten each frame from the
// optimisation playback.
const DIFF_ARM6_BASE_HEIGHT: f64 = 0.5;
const DIFF_ARM6_LINK_RADIUS: f32 = 0.04;

fn build_diff_arm6_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let chain = FeatherstoneRevoluteChain::franka6(0.005, 60);
    let mut visuals = Vec::with_capacity(6);
    for link in &chain.links {
        let len = (link.com_local.length() * 2.0).max(0.06);
        let body = b.add_body(
            mp::BodyParams::new()
                .with_mass(0.0)
                .with_pose(mp::Transform::new(
                    MVec3::new(0.0, DIFF_ARM6_BASE_HEIGHT, 0.0),
                    MQuat::IDENTITY,
                ))
                .with_flags(mp::BodyFlags::KINEMATIC),
        );
        let half_h = (len * 0.5).max(0.04);
        b.add_shape_capsule(
            body,
            half_h,
            DIFF_ARM6_LINK_RADIUS as f64,
            mp::Transform::IDENTITY,
        )
        .expect("diff arm6 link shape");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: DIFF_ARM6_LINK_RADIUS,
        });
    }
    // Static plane so the camera has a ground reference.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("diff arm6 model"), visuals)
}

fn update_diff_arm6(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<DiffArm6ExampleState>,
) {
    if sim.scene != Scene::DiffArm6 {
        state.playback_t = 0.0;
        return;
    }

    let n = state.chain.links.len();
    let h = state.chain.horizon;

    // 1. (Re)run optimisation if requested.
    if state.rerun {
        // Reset torque baseline so each re-run starts fresh.
        state.tau = vec![vec![0.0; n]; h];
        state.iter_ee_trace.clear();

        // Target in chain-local coordinates: shift down by the base
        // height, since the chain's root is at the world origin in its
        // local frame.
        let target_local = MVec3::new(
            state.target.x,
            state.target.y - DIFF_ARM6_BASE_HEIGHT,
            state.target.z,
        );
        let target_chain = molla::math::Vec3::new(target_local.x, target_local.y, target_local.z);

        let q0 = vec![0.0_f64; n];
        let qd0 = vec![0.0_f64; n];
        let lr_init = 0.05_f64;
        let mut lr = lr_init;
        let mut prev = f64::INFINITY;
        let mut last_residual = f64::INFINITY;
        let mut converged: Option<usize> = None;
        // Stage 23.1.1: track backward / forward time ratio across all
        // GD iterations and average. The viewer's pane shows this so
        // the user can compare the closed-form CPU vs FD-CPU perf
        // toggle effects in real time.
        let mut fwd_total_us = 0.0_f64;
        let mut bwd_total_us = 0.0_f64;
        // Stage 23.1.2 — backend picker. The 4-way enum drives the
        // dispatcher directly. `state.use_gpu_adjoint` stays mirrored
        // for back-compat with anything still reading the bool.
        let backend_choice = state.backend;
        state.use_gpu_adjoint = matches!(
            backend_choice,
            molla::kernels::AdjointBackend::GpuFallbackCpu
        );

        // Stage 23.1.2 — multi-articulation batch demo. Runs N parallel
        // chains through the rayon batch dispatcher per re-run and
        // records per-chain wall time + parallel speedup. The
        // single-chain optimisation loop still runs after so the user
        // sees the trajectory animation.
        if state.use_multi_articulation {
            let batch_n = state.multi_articulation_n.max(2);
            let chains: Vec<molla::autograd::FeatherstoneRevoluteChain> =
                (0..batch_n).map(|_| state.chain.clone()).collect();
            let q0_b = vec![0.0_f64; n];
            let qd0_b = vec![0.0_f64; n];
            let tau_b = vec![vec![0.0_f64; n]; h];
            let traces: Vec<_> = chains
                .iter()
                .map(|c| c.rollout(&q0_b, &qd0_b, &tau_b))
                .collect();
            // Use a generic +Z grad seed for the batch demo — the
            // benchmark exercises the per-chain reverse-mode cost, not
            // a specific optimisation step.
            let grad_seed = molla::math::Vec3::Z;
            let inputs: Vec<molla::kernels::MultiAdjointInput<'_>> = (0..batch_n)
                .map(|i| molla::kernels::MultiAdjointInput {
                    chain: &chains[i],
                    trace: &traces[i],
                    grad_ee: grad_seed,
                })
                .collect();
            // Warm.
            let _ = molla::kernels::MultiAdjointBatch::dispatch_cpu(&inputs);
            let (_par, par_wall_us) = molla::kernels::MultiAdjointBatch::dispatch_cpu(&inputs)
                .expect("Stage 23.1 chains are single-DoF");
            let (_ser, ser_wall_us) =
                molla::kernels::MultiAdjointBatch::dispatch_cpu_serial(&inputs)
                    .expect("Stage 23.1 chains are single-DoF");
            state.last_per_chain_us = par_wall_us / batch_n as f64;
            state.last_parallel_speedup = if par_wall_us > 0.0 {
                ser_wall_us / par_wall_us
            } else {
                0.0
            };
            info!(
                "diff_arm6: multi-articulation batch — N={batch_n}, parallel_wall={par_wall_us:.0}µs ({:.0}µs/chain), serial_wall={ser_wall_us:.0}µs, parallel_speedup={:.2}×",
                state.last_per_chain_us, state.last_parallel_speedup,
            );
        } else {
            state.last_per_chain_us = 0.0;
            state.last_parallel_speedup = 0.0;
        }
        for it in 0..DIFF_ARM6_OPT_ITERS {
            let t_fwd = std::time::Instant::now();
            let trace = state.chain.rollout(&q0, &qd0, &state.tau);
            fwd_total_us += t_fwd.elapsed().as_micros() as f64;
            let q_final = &trace.q[h];
            let ee = state.chain.end_effector(q_final);
            let d = ee - target_chain;
            last_residual = d.length();

            // Trail point in WORLD coords.
            state
                .iter_ee_trace
                .push(MVec3::new(ee.x, ee.y + DIFF_ARM6_BASE_HEIGHT, ee.z));

            if !last_residual.is_finite() {
                warn!("diff_arm6: residual diverged at iter {it} — backing off");
                break;
            }
            if converged.is_none() && last_residual < 0.05 {
                converged = Some(it);
            }
            if last_residual < 0.01 {
                state.iters_run = it;
                break;
            }

            // Backoff lr if loss went up (cheap line search).
            if last_residual > prev * 1.05 {
                lr *= 0.5;
            } else if last_residual < prev * 0.99 {
                lr = (lr * 1.05).min(lr_init);
            }
            prev = last_residual;

            // Borrow chain + scratch as disjoint fields so the same
            // struct's immutable + mutable references coexist. Route
            // through the Stage 23.1.1 dispatcher so the GPU / CPU
            // toggle takes effect.
            let res = {
                let DiffArm6ExampleState { chain, scratch, .. } = &mut *state;
                molla::kernels::FeatherstoneAdjointDispatcher::reverse(
                    &*chain,
                    &trace,
                    d * 2.0,
                    scratch,
                    backend_choice,
                )
                .expect("Stage 23.1 single-DoF chain")
            };
            bwd_total_us += res.backward_us;
            let g = res.g_tau;
            for t in 0..h {
                for k in 0..n {
                    let delta = -lr * g[t][k];
                    let clamped = delta.clamp(-0.2, 0.2);
                    state.tau[t][k] += clamped;
                    state.tau[t][k] = state.tau[t][k].clamp(-3.0, 3.0);
                }
            }
            state.iters_run = it + 1;
        }
        state.last_residual = last_residual;
        state.converged_iter = converged;
        state.rerun = false;
        state.playback_t = 0.0;
        state.last_bwd_fwd_ratio = if fwd_total_us > 0.0 {
            bwd_total_us / fwd_total_us
        } else {
            0.0
        };
        info!(
            "diff_arm6: optimisation done — residual {:.4} m, converged at iter {:?}",
            state.last_residual, state.converged_iter,
        );
    }

    // 2. Playback. Advance wall clock + roll out to current playback time.
    let dt = time.delta_secs_f64();
    state.playback_t += dt * DIFF_ARM6_PLAYBACK_SPEED;
    let total = h as f64 * state.chain.dt;
    let t = if total > 0.0 {
        state.playback_t % total
    } else {
        0.0
    };
    let step = ((t / state.chain.dt).floor() as usize).min(h.saturating_sub(1));
    let q0 = vec![0.0_f64; n];
    let qd0 = vec![0.0_f64; n];
    let trace = state.chain.rollout(&q0, &qd0, &state.tau);
    let q = &trace.q[step];

    // FK on the chain → world poses for each link.
    let body_world = state.chain.forward_kinematics(q);
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    let base_lift = MVec3::new(0.0, DIFF_ARM6_BASE_HEIGHT, 0.0);
    for (j, link_t) in body_world.iter().enumerate().take(body_q.len()) {
        let p = MVec3::new(link_t.position.x, link_t.position.y, link_t.position.z) + base_lift;
        let r = MQuat::from_xyzw(
            link_t.rotation.x,
            link_t.rotation.y,
            link_t.rotation.z,
            link_t.rotation.w,
        );
        body_q[j] = mp::Transform::new(p, r);
    }
}

// Arm example — sets per-joint targets (auto-sine or slider) and
// runs the GPU `ControllerPd` to write `Control.joint_force`. The
// physics step (Featherstone) then consumes that force. No-op for

fn draw_diff_arm6_gizmos(sim: Res<Sim>, state: Res<DiffArm6ExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::DiffArm6 {
        return;
    }
    let target = Vec3::new(
        state.target.x as f32,
        state.target.y as f32,
        state.target.z as f32,
    );
    let target_color = Color::srgb(1.0, 0.95, 0.20);
    gizmos.sphere(target, 0.04, target_color);
    let cross = 0.08;
    gizmos.line(
        target - Vec3::X * cross,
        target + Vec3::X * cross,
        target_color,
    );
    gizmos.line(
        target - Vec3::Y * cross,
        target + Vec3::Y * cross,
        target_color,
    );
    gizmos.line(
        target - Vec3::Z * cross,
        target + Vec3::Z * cross,
        target_color,
    );
    if state.show_trajectory && state.iter_ee_trace.len() >= 2 {
        let n = state.iter_ee_trace.len();
        for window in state.iter_ee_trace.windows(2).enumerate() {
            let (i, pair) = window;
            let t = i as f32 / (n - 1).max(1) as f32;
            let r = 0.20 + 0.75 * t;
            let g = 0.40 + 0.10 * (1.0 - t);
            let b = 0.95 - 0.85 * t;
            let color = Color::srgb(r, g, b);
            let a = Vec3::new(pair[0].x as f32, pair[0].y as f32, pair[0].z as f32);
            let b_ = Vec3::new(pair[1].x as f32, pair[1].y as f32, pair[1].z as f32);
            gizmos.line(a, b_, color);
            gizmos.sphere(a, 0.01, color);
        }
        if let Some(last) = state.iter_ee_trace.last() {
            let p = Vec3::new(last.x as f32, last.y as f32, last.z as f32);
            gizmos.sphere(p, 0.018, Color::srgb(0.95, 0.55, 0.15));
        }
    }
}

fn hit_to_world(hit: &RayHit) -> Option<Vec3> {
    if hit.shape_id.is_some() {
        Some(Vec3::new(
            hit.point.x as f32,
            hit.point.y as f32,
            hit.point.z as f32,
        ))
    } else {
        None
    }
}

// Draw the lidar's rays + hit points as Bevy gizmos. Hit rays are
// red with a small white cross at the hit point; missed rays are
// dim cyan extending to the max range. Lidar origin gets a small

fn diff_arm6_pane(ui: &mut egui::Ui, state: &mut DiffArm6ExampleState) {
    ui.add_space(2.0);
    let solver_label = match state.backend {
        molla::kernels::AdjointBackend::Auto => "Featherstone, Auto (Stage 23.1.2 → AnalyticQd)",
        molla::kernels::AdjointBackend::ForceCpuClosedForm => {
            "Featherstone, CPU closed-form (Stage 23.1.1)"
        }
        molla::kernels::AdjointBackend::ForceCpuAnalyticQd => {
            "Featherstone, CPU AnalyticQd (Stage 23.1.2)"
        }
        molla::kernels::AdjointBackend::ForceCpuFiniteDiff => "Featherstone, CPU FD (v1 oracle)",
        molla::kernels::AdjointBackend::GpuFallbackCpu => {
            "Featherstone, GPU (single-chain → CPU AnalyticQd)"
        }
    };
    readout(ui, "Solver", solver_label);
    readout(
        ui,
        "Chain",
        &format!(
            "{}, {} links × {} steps × {:.1} ms",
            state.kind.label(),
            state.chain.links.len(),
            state.chain.horizon,
            state.chain.dt * 1000.0,
        ),
    );

    // Stage 23.1.3 — joint kind picker. Swapping the kind rebuilds
    // the chain, resets τ, and triggers a re-run so the user sees the
    // gradient flow through each joint kind end-to-end.
    let mut new_kind = state.kind;
    egui::ComboBox::from_label("Joint kind")
        .selected_text(state.kind.label())
        .show_ui(ui, |ui| {
            for kind in [
                DiffArm6Kind::Revolute,
                DiffArm6Kind::Ball,
                DiffArm6Kind::Free,
                DiffArm6Kind::Distance,
                DiffArm6Kind::D6,
            ] {
                ui.selectable_value(&mut new_kind, kind, kind.label());
            }
        });
    if new_kind != state.kind {
        state.kind = new_kind;
        state.chain = new_kind.build_chain();
        let n = state.chain.links.len();
        let h = state.chain.horizon;
        state.scratch = JacobianScratch::new(n);
        state.tau = vec![vec![0.0; n]; h];
        state.iter_ee_trace.clear();
        state.last_residual = f64::INFINITY;
        state.converged_iter = None;
        state.iters_run = 0;
        // Re-pick a sensible target near the new chain's rest pose.
        let rest_ee = state.chain.end_effector(&vec![0.0; n]);
        state.target = MVec3::new(
            rest_ee.x as f64 + 0.07,
            rest_ee.y as f64 + DIFF_ARM6_BASE_HEIGHT,
            rest_ee.z as f64 - 0.07,
        );
        state.rerun = true;
    }
    ui.separator();

    ui.label("Optimisation");
    readout(ui, "  iters run", &format!("{}", state.iters_run));
    readout(
        ui,
        "  residual",
        &if state.last_residual.is_finite() {
            format!("{:.4} m", state.last_residual)
        } else {
            "—".into()
        },
    );
    readout(
        ui,
        "  converged",
        &match state.converged_iter {
            Some(it) => format!("at iter {it} (≤ 5 cm)"),
            None if state.iters_run == 0 => "(idle)".into(),
            None => "did NOT converge ≤ 5 cm".into(),
        },
    );
    // Stage 23.1.1: backward / forward ratio readout. Demonstrates the
    // closed-form CPU adjoint perf win next to the iter counter.
    readout(
        ui,
        "  bwd / fwd",
        &if state.last_bwd_fwd_ratio > 0.0 {
            format!("{:.2}×", state.last_bwd_fwd_ratio)
        } else {
            "—".into()
        },
    );
    if ui.button("Re-run optimisation").clicked() {
        state.rerun = true;
    }
    // Stage 23.1.2 — 4-way backend picker (Auto / CPU CF / CPU AnalyticQd
    // / CPU FD / GPU). Promoted from the Stage 23.1.1 boolean checkbox.
    let prev_backend = state.backend;
    let backend_label = state.backend.label();
    egui::ComboBox::from_label("Adjoint backend")
        .selected_text(backend_label)
        .show_ui(ui, |ui| {
            for b in [
                molla::kernels::AdjointBackend::Auto,
                molla::kernels::AdjointBackend::ForceCpuClosedForm,
                molla::kernels::AdjointBackend::ForceCpuAnalyticQd,
                molla::kernels::AdjointBackend::ForceCpuFiniteDiff,
                molla::kernels::AdjointBackend::GpuFallbackCpu,
            ] {
                ui.selectable_value(&mut state.backend, b, b.label());
            }
        });
    if state.backend != prev_backend {
        state.rerun = true;
    }
    // Stage 23.1.2 — multi-articulation batch toggle. Drives N parallel
    // chains through `MultiAdjointBatch` on each re-run; the per-chain
    // wall time + parallel speedup readouts below show the win.
    let prev_multi = state.use_multi_articulation;
    ui.checkbox(
        &mut state.use_multi_articulation,
        "Multi-articulation batch (Stage 23.1.2)",
    );
    if state.use_multi_articulation {
        let mut n = state.multi_articulation_n as i32;
        if ui
            .add(egui::Slider::new(&mut n, 2..=64).text("batch N"))
            .changed()
        {
            state.multi_articulation_n = n.max(2) as usize;
            state.rerun = true;
        }
        readout(
            ui,
            "  per-chain (parallel)",
            &if state.last_per_chain_us > 0.0 {
                format!("{:.0} µs", state.last_per_chain_us)
            } else {
                "—".into()
            },
        );
        readout(
            ui,
            "  parallel speedup",
            &if state.last_parallel_speedup > 0.0 {
                format!("{:.2}×", state.last_parallel_speedup)
            } else {
                "—".into()
            },
        );
    }
    if state.use_multi_articulation != prev_multi {
        state.rerun = true;
    }
    ui.separator();

    ui.label("Target (world XYZ)");
    let mut tx = state.target.x as f32;
    let mut ty = state.target.y as f32;
    let mut tz = state.target.z as f32;
    let mut changed = false;
    let lift = DIFF_ARM6_BASE_HEIGHT as f32;
    if ui
        .add(
            egui::Slider::new(&mut tx, -1.2_f32..=1.2_f32)
                .text("x")
                .step_by(0.01),
        )
        .changed()
    {
        state.target.x = tx as f64;
        changed = true;
    }
    if ui
        .add(
            egui::Slider::new(&mut ty, lift..=(lift + 1.5))
                .text("y")
                .step_by(0.01),
        )
        .changed()
    {
        state.target.y = ty as f64;
        changed = true;
    }
    if ui
        .add(
            egui::Slider::new(&mut tz, -1.2_f32..=1.2_f32)
                .text("z")
                .step_by(0.01),
        )
        .changed()
    {
        state.target.z = tz as f64;
        changed = true;
    }
    if changed {
        state.rerun = true;
    }
    ui.separator();

    ui.checkbox(&mut state.show_trajectory, "Show optimisation trajectory");
    ui.add_space(4.0);
    ui.small(
        "6-DOF Franka-style serial chain. Open-loop torque sequence \
         optimised by gradient descent through the reverse-mode \
         Featherstone adjoint. Stage 23.1.2 default: AnalyticQd CPU \
         path — closed-form M⁻¹ + analytic ∂h/∂qd via chain rule on \
         RNEA + FD-on-RNEA for ∂h/∂q. The multi-articulation batch \
         toggle drives N parallel chains through the rayon batch \
         dispatcher; per-chain wall time + parallel speedup readouts \
         show the headline win. The pure-analytic ∂h/∂q path lives in \
         Stage 23.1.3 (full Featherstone-2008 §10 derivation).",
    );
}


