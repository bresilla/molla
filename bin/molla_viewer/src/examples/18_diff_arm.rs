// --- Differentiable arm (Stage 23) -------------------------------
// World-space anchor height of the planar arm's shoulder. The arm
// hangs in the world XY plane (Z = 0) so the existing camera works.
const DIFF_ARM_BASE_Y: f32 = 1.6;
// Capsule visual radius for the two arm links.
const DIFF_ARM_LINK_RADIUS: f32 = 0.04;
// Optimisation iterations per "Re-run" pass. The example shows the
// solver typically lands within 5 cm in <30 iters; 80 leaves margin
// for unreachable targets where the residual just shrinks.
const DIFF_ARM_OPT_ITERS: usize = 80;
// Gradient-descent learning rate. ~5.0 is the value the example
// uses; reducing it below 1 makes convergence noticeably slower.
const DIFF_ARM_OPT_LR: f64 = 5.0;
// Playback speed multiplier — the trajectory is `arm.dt × arm.horizon`
// = ~0.8 s of dynamics; we replay it at 0.5× to make the arm motion
// readable.
const DIFF_ARM_PLAYBACK_SPEED: f64 = 0.5;

// Stage-23 differentiable-arm demo runtime state. Owns the
// optimisation horizon, target XYZ, the per-step torque buffer
// (cleared and re-optimised on every "Re-run" / target-change), and
// the per-iteration EE trajectory (so the right-rail toggle can
// overlay it as an arrow trace).
#[derive(Resource)]
struct DiffArmExampleState {
    arm: DiffArm,
    /// Target end-effector position in world coordinates. The arm is
    /// anchored at `(0, DIFF_ARM_BASE_Y, 0)` and lives in the world XY
    /// plane (Z = 0); `target.z` is exposed in the UI for curiosity
    /// but stays clamped to 0 for v1 (the arm itself is planar).
    target: MVec3,
    /// Current optimised torque sequence. Length = `arm.horizon`.
    tau: Vec<[f64; 2]>,
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
    /// `update_diff_arm` next frame.
    rerun: bool,
    /// Show the iteration trail of EE positions as a gizmo.
    show_trajectory: bool,
    /// Wall-clock since scene entry — used to animate the arm
    /// playback so the user actually SEES the optimised trajectory
    /// (otherwise body_q would just snap to the final pose).
    playback_t: f64,
}

impl Default for DiffArmExampleState {
    fn default() -> Self {
        let arm = DiffArm::default();
        Self {
            arm,
            // Reachable point well within L1 + L2 = 0.9 m.
            target: MVec3::new(0.45, DIFF_ARM_BASE_Y as f64 - 0.25, 0.0),
            tau: vec![[0.0; 2]; arm.horizon],
            iter_ee_trace: Vec::new(),
            last_residual: f64::INFINITY,
            converged_iter: None,
            iters_run: 0,
            // Re-run on first entry so the user sees the optimisation
            // happen immediately rather than starting from a dead-zero
            // arm.
            rerun: true,
            show_trajectory: true,
            playback_t: 0.0,
        }
    }
}

// --- Stage-23 differentiable arm scene -------------------------------
//
// Two kinematic capsule bodies representing the upper + lower arm of
// a 2-link planar arm. The actual "physics" runs host-side in
// `update_diff_arm` via [`DiffArm`]; here we just declare two
// kinematic bodies whose `body_q` will be rewritten each frame from
// the optimisation playback. No joints — molla's joint system isn't
// needed because we drive both bodies' world poses directly.
fn build_diff_arm_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let arm = DiffArm::default();

    // Body 0: upper arm. Half-length = L1/2 along the body's local +Y;
    // visualised as a capsule. Initial pose = hanging straight down
    // from the shoulder anchor. `update_diff_arm` rewrites this every
    // frame — the initial pose is just a sane starting visual.
    let half1 = arm.l1 * 0.5;
    let upper = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(
                MVec3::new(0.0, DIFF_ARM_BASE_Y as f64 - half1, 0.0),
                MQuat::IDENTITY,
            ))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_capsule(
        upper,
        half1,
        DIFF_ARM_LINK_RADIUS as f64,
        mp::Transform::IDENTITY,
    )
    .expect("diff arm upper shape");

    // Body 1: lower arm.
    let half2 = arm.l2 * 0.5;
    let lower = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(
                MVec3::new(0.0, DIFF_ARM_BASE_Y as f64 - arm.l1 - half2, 0.0),
                MQuat::IDENTITY,
            ))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_capsule(
        lower,
        half2,
        DIFF_ARM_LINK_RADIUS as f64,
        mp::Transform::IDENTITY,
    )
    .expect("diff arm lower shape");

    let visuals = vec![
        BodyVisual::Capsule {
            half_h: half1 as f32,
            radius: DIFF_ARM_LINK_RADIUS,
        },
        BodyVisual::Capsule {
            half_h: half2 as f32,
            radius: DIFF_ARM_LINK_RADIUS,
        },
    ];

    // Static plane so the camera has a ground reference.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("diff arm model"), visuals)
}

fn update_diff_arm(time: Res<Time>, mut sim: ResMut<Sim>, mut state: ResMut<DiffArmExampleState>) {
    if sim.scene != Scene::DiffArm {
        // Reset the per-frame playback so re-entry restarts from
        // the start of the optimised motion. Don't clobber `tau` —
        // the user might want their previous result preserved if
        // they hop out + back without changing target.
        state.playback_t = 0.0;
        return;
    }

    // 1. (Re)run optimisation if requested.
    if state.rerun {
        // Reset the torque buffer to zero so each re-run starts from
        // the same baseline (otherwise sweeping the target slider
        // would silently warm-start from the previous tau and
        // produce path-dependent answers).
        state.tau = vec![[0.0; 2]; state.arm.horizon];
        state.iter_ee_trace.clear();

        // Target in arm-local 2D: shift by the shoulder anchor.
        let target_2d = [state.target.x, state.target.y - DIFF_ARM_BASE_Y as f64];

        // Iterate by hand so we can capture per-iter EE positions
        // for the overlay (DiffArm::gradient_descent doesn't expose
        // intermediates).
        let arm = state.arm;
        let q0 = [0.0; 2];
        let qd0 = [0.0; 2];
        let mut last_residual = f64::INFINITY;
        let mut converged = None;
        for it in 0..DIFF_ARM_OPT_ITERS {
            let trace = arm.rollout(q0, qd0, &state.tau);
            let final_q = trace[trace.len() - 1].0;
            let ee = arm.forward_kinematics(final_q);
            // Trail point in world coords.
            state
                .iter_ee_trace
                .push(MVec3::new(ee[0], ee[1] + DIFF_ARM_BASE_Y as f64, 0.0));
            let dx = ee[0] - target_2d[0];
            let dy = ee[1] - target_2d[1];
            last_residual = (dx * dx + dy * dy).sqrt();
            if converged.is_none() && last_residual < 0.05 {
                converged = Some(it);
            }
            let g_tau = arm.reverse(&trace, [2.0 * dx, 2.0 * dy]);
            for i in 0..arm.horizon {
                state.tau[i][0] -= DIFF_ARM_OPT_LR * g_tau[i][0];
                state.tau[i][1] -= DIFF_ARM_OPT_LR * g_tau[i][1];
            }
        }
        state.last_residual = last_residual;
        state.converged_iter = converged;
        state.iters_run = DIFF_ARM_OPT_ITERS;
        state.rerun = false;
        state.playback_t = 0.0;
        info!(
            "diff_arm: optimisation done — residual {:.4} m, converged at iter {:?}",
            state.last_residual, state.converged_iter,
        );
    }

    // 2. Playback. Advance the wall clock + roll out to the current
    // playback time. We re-roll every frame because the cost is
    // negligible (40 substeps of double-integrator math) and it
    // avoids stashing the trace as resource state.
    let dt = time.delta_secs_f64();
    state.playback_t += dt * DIFF_ARM_PLAYBACK_SPEED;
    let total = state.arm.horizon as f64 * state.arm.dt;
    let t = if total > 0.0 {
        state.playback_t % total
    } else {
        0.0
    };
    let step = ((t / state.arm.dt).floor() as usize).min(state.arm.horizon.saturating_sub(1));
    let trace = state.arm.rollout([0.0; 2], [0.0; 2], &state.tau);
    let q = trace[step].0;

    // Forward kinematics → world poses for the two capsule bodies.
    // Each capsule's local +Y points along the link, so the body
    // rotation must rotate +Y onto the link direction.
    let half1 = state.arm.l1 * 0.5;
    let half2 = state.arm.l2 * 0.5;

    // Upper arm: extends from shoulder down (-Y when q1 = 0). With
    // q1 nonzero, the arm rotates around +Z. The link direction in
    // world coords is `(sin q1, -cos q1, 0)`; the capsule centre is
    // `shoulder + 0.5·L1·dir`.
    let dir1 = MVec3::new(q[0].sin(), -q[0].cos(), 0.0);
    let p1 = MVec3::new(0.0, DIFF_ARM_BASE_Y as f64, 0.0) + dir1 * half1;
    // Rotation: rotate local +Y onto dir1 → angle = q1 around +Z.
    let r1 = MQuat::from_axis_angle(MVec3::Z, q[0]);

    // Lower arm: extends from elbow down (-Y when q1 + q2 = 0).
    let elbow = MVec3::new(0.0, DIFF_ARM_BASE_Y as f64, 0.0) + dir1 * state.arm.l1;
    let qsum = q[0] + q[1];
    let dir2 = MVec3::new(qsum.sin(), -qsum.cos(), 0.0);
    let p2 = elbow + dir2 * half2;
    let r2 = MQuat::from_axis_angle(MVec3::Z, qsum);

    // Stash poses into body_q on the host. `sync_transforms` reads
    // host buffers; no GPU upload needed for this kinematic path.
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    if body_q.len() >= 2 {
        body_q[0] = mp::Transform::new(p1, r1);
        body_q[1] = mp::Transform::new(p2, r2);
    }
}

// Stage-23.1 6-DOF Featherstone diff-arm pre-step. Same shape as
// `update_diff_arm`, but the chain is the [`FeatherstoneRevoluteChain::franka6`]
// 6-DOF Franka stand-in driven by the reverse-mode adjoint from

fn draw_diff_arm_gizmos(sim: Res<Sim>, state: Res<DiffArmExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::DiffArm {
        return;
    }
    // Target marker — yellow sphere + cross-hair lines.
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

    // Trajectory overlay — polyline through the per-iteration EE
    // positions captured during the optimisation. Fade colour from
    // cool blue (iter 0, far from target) to warm orange (final iter,
    // near target).
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
            // Small dot at each iteration so individual GD steps are
            // visible (otherwise the polyline blurs them).
            gizmos.sphere(a, 0.012, color);
        }
        // Last point.
        if let Some(last) = state.iter_ee_trace.last() {
            let p = Vec3::new(last.x as f32, last.y as f32, last.z as f32);
            gizmos.sphere(p, 0.018, Color::srgb(0.95, 0.55, 0.15));
        }
    }
}

// Stage-23.1 differentiable Featherstone arm gizmos. Same content as

fn diff_arm_pane(ui: &mut egui::Ui, state: &mut DiffArmExampleState) {
    ui.add_space(2.0);
    readout(ui, "Solver", "CPU autograd (Stage 23)");
    readout(
        ui,
        "Arm",
        &format!("L1={:.2} m, L2={:.2} m", state.arm.l1, state.arm.l2),
    );
    readout(
        ui,
        "Horizon",
        &format!(
            "{} steps × {:.0} ms",
            state.arm.horizon,
            state.arm.dt * 1000.0
        ),
    );
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
    if ui.button("Re-run optimisation").clicked() {
        state.rerun = true;
    }
    ui.separator();

    ui.label("Target (world XYZ)");
    let mut tx = state.target.x as f32;
    let mut ty = state.target.y as f32;
    let mut tz = state.target.z as f32;
    let mut changed = false;
    if ui
        .add(
            egui::Slider::new(&mut tx, -0.9_f32..=0.9_f32)
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
            egui::Slider::new(&mut ty, (DIFF_ARM_BASE_Y - 0.9)..=(DIFF_ARM_BASE_Y + 0.9))
                .text("y")
                .step_by(0.01),
        )
        .changed()
    {
        state.target.y = ty as f64;
        changed = true;
    }
    // Z is shown for completeness but the planar arm can't reach off-plane.
    if ui
        .add(
            egui::Slider::new(&mut tz, -0.5_f32..=0.5_f32)
                .text("z (cosmetic)")
                .step_by(0.01),
        )
        .changed()
    {
        state.target.z = tz as f64;
        // Z change doesn't affect the optimisation itself (planar
        // arm), so don't trigger a re-run.
    }
    if changed {
        state.rerun = true;
    }
    ui.separator();

    ui.checkbox(&mut state.show_trajectory, "Show optimisation trajectory");
    ui.add_space(4.0);
    ui.small(
        "2-link planar arm. Open-loop torque sequence optimised by \
         gradient descent (closed-form adjoint) to drive the EE to \
         the target. The trajectory overlay shows the EE's location \
         at every optimisation iteration so you can see GD converge.",
    );
}

// Body for the right-rail pane in the Differentiable-Featherstone
// (Stage 23.1) scene. Same shape as [`diff_arm_pane`] — target XYZ
// sliders, residual + iters-to-converge readout, "re-run" button,

