// GPU IK arm runtime state. Lazy-initialised when entering the
// `Scene::GpuIkArm` scene; reset on scene switch. Tracks the
// draggable target position, the most recent solver telemetry, and
// a "use GPU" toggle.
#[derive(Resource)]
struct GpuIkArmState {
    /// Target world position the EE chases. Default = rest EE +
    /// `GPU_IK_DEFAULT_OFFSET`, computed lazily on first frame.
    target: MVec3,
    target_init: bool,
    use_gpu: bool,
    last_iters: usize,
    last_residual: f64,
    last_solve_us: f64,
}

impl Default for GpuIkArmState {
    fn default() -> Self {
        Self {
            target: MVec3::ZERO,
            target_init: false,
            use_gpu: true,
            last_iters: 0,
            last_residual: 0.0,
            last_solve_us: 0.0,
        }
    }
}

// --- Stage-16 GPU IK arm scene ------------------------------------
//
// 7-DOF Franka-style arm rooted at world origin (lifted on a base
// pillar so the arm rests above the ground grid). End-effector chases
// a draggable target (XYZ sliders in the right-rail pane). Per-frame:
// `update_gpu_ik_arm` runs `IkSolver::solve_gpu` with the current
// target as a position objective on the EE body, writes `joint_q` +
// `body_q` on the host, and `step_simulation` skips this scene
// entirely (no integration — purely kinematic IK).

const GPU_IK_BASE_HEIGHT: f64 = 1.0;
const GPU_IK_LINK_RADIUS: f64 = 0.05;
const GPU_IK_LINK_MASS: f64 = 1.0;
// Default reach offset relative to the arm's rest EE position.
const GPU_IK_DEFAULT_OFFSET: MVec3 = MVec3::new(0.25, 0.05, -0.15);

fn build_gpu_ik_arm_model() -> (mp::Model, Vec<BodyVisual>) {
    use mp::*;
    let mut b = ModelBuilder::new();
    let mut visuals = Vec::with_capacity(7);

    let mass = GPU_IK_LINK_MASS;
    let half = MVec3::new(0.04, 0.04, 0.04);
    let h2 = half * half;
    let m_third = mass / 3.0;
    let inertia = molla::math::Mat3::from_diagonal(MVec3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ));
    let make_link = |b: &mut ModelBuilder| {
        b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass,
            com: MVec3::ZERO,
            inertia,
        }))
    };

    b.begin_articulation();
    let axes = [
        MVec3::Z,
        MVec3::Y,
        MVec3::Z,
        MVec3::Y,
        MVec3::Z,
        MVec3::Y,
        MVec3::Z,
    ];
    let link_lens = [0.333_f64, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107];
    let mut prev = molla::core::BodyId::NONE;
    for i in 0..7 {
        let body = make_link(&mut b);
        let x_p = if i == 0 {
            // Lift the arm root above the ground grid so the linkage
            // is visible from the default camera.
            mp::Transform::from_translation(MVec3::new(0.0, GPU_IK_BASE_HEIGHT, 0.0))
        } else {
            mp::Transform::from_translation(MVec3::new(0.0, 0.0, link_lens[i - 1]))
        };
        let x_c = mp::Transform::IDENTITY;
        b.add_joint_revolute(prev, body, x_p, x_c, axes[i]).unwrap();
        // A short capsule running along the link's local +Z. Last
        // link (the EE) gets a slightly wider sphere visual so the
        // tip is easy to spot.
        let half_h = link_lens[i].max(0.06_f64) * 0.5;
        let radius = GPU_IK_LINK_RADIUS;
        // We use Sphere for the EE so the gizmo target dot lines up.
        let visual = if i == 6 {
            BodyVisual::Sphere(0.08)
        } else {
            BodyVisual::Capsule {
                half_h: half_h as f32,
                radius: radius as f32,
            }
        };
        visuals.push(visual);
        prev = body;
    }

    (b.build().expect("gpu ik arm model"), visuals)
}

// --- Stage-17 quadrotor scene --------------------------------------
//
// FREE joint root + 4 fixed-joint rotor bodies at ±arm × {X, Z}.
// PD altitude + attitude control writes per-rotor thrusts as world-
// frame vertical forces into `state.body_f` on each rotor body —
// Featherstone consumes them via the bias term, producing both
// translation (sum of thrusts) and rotation (asymmetric thrusts).
//
// The four rotors are arranged in a "+" shape: rotor 0 = +X, rotor 1
// = -X, rotor 2 = +Z, rotor 3 = -Z. Roll = thrust differential between
// rotors 2/3 (+Z vs -Z); pitch = differential between rotors 0/1.

fn update_gpu_ik_arm(mut sim: ResMut<Sim>, mut state: ResMut<GpuIkArmState>) {
    if sim.scene != Scene::GpuIkArm {
        return;
    }
    // Destructure to keep separate borrows of `model`, `state_a`, `engine`.
    let Sim {
        model,
        state_a,
        engine,
        ..
    } = &mut *sim;

    // EE is the last body (index = body_count - 1) for the 7-DOF arm.
    let ee_body = model.body_count - 1;

    // Lazily seed the target on first frame from the rest pose.
    if !state.target_init {
        molla::sim::eval_fk(model, state_a).expect("eval_fk");
        let bq = state_a.body_q.host().expect("body_q");
        state.target = bq[ee_body].position + GPU_IK_DEFAULT_OFFSET;
        state.target_init = true;
    }

    let q0: Vec<f64> = state_a
        .joint_q
        .host()
        .expect("joint_q").to_vec();
    let solver = IkSolver::new(vec![IkObjective::Position {
        body: ee_body,
        point_local: MVec3::ZERO,
        target_world: state.target,
        weight: 1.0,
    }])
    .with_config(IkConfig {
        max_iter: 30,
        residual_tol: 5.0e-4,
        ..IkConfig::default()
    });

    let t0 = std::time::Instant::now();
    let result = if state.use_gpu {
        solver.solve_gpu(engine, model, state_a, &q0)
    } else {
        solver.solve(model, state_a, &q0)
    };
    state.last_solve_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

    match result {
        Ok(r) => {
            state.last_iters = r.iters;
            state.last_residual = r.residual;
        }
        Err(e) => {
            warn!("gpu_ik_arm: solve failed: {e}");
        }
    }
}

// Stage-23 diff-arm pre-step. Two responsibilities:
//
// 1. **(Re)run optimisation** when `state.rerun` is set. Project the
//    target into the arm's planar coordinates (XY, Z ignored), run
//    [`DiffArm::gradient_descent`] on the torque sequence, capture
//    the per-iter EE positions for the trail overlay.
// 2. **Replay the optimised trajectory** every frame: advance
//    `state.playback_t` by `dt × playback_speed`, look up the
//    `(q1, q2)` at that time from the rolled-out trace, run forward
//    kinematics, write the two body world poses into `state_a.body_q`
//    so `sync_transforms` picks them up.
//

fn gpu_ik_pane(ui: &mut egui::Ui, state: &mut GpuIkArmState, sim: &Sim) {
    ui.add_space(2.0);
    readout(
        ui,
        "Solver",
        if state.use_gpu { "GPU LM" } else { "CPU LM" },
    );
    readout(ui, "DOFs", "7");
    readout(ui, "Iterations", &format!("{}", state.last_iters));
    readout(ui, "Residual", &format!("{:.5}", state.last_residual));
    readout(
        ui,
        "ms / solve",
        &format!("{:.2}", state.last_solve_us / 1000.0),
    );

    let ee_body = sim.model.body_count - 1;
    if let Ok(bq) = sim.state_a.body_q.host()
        && ee_body < bq.len() {
            let p = bq[ee_body].position;
            let err = (p - state.target).length();
            readout(ui, "EE error", &format!("{:.4} m", err));
        }
    ui.separator();

    ui.checkbox(&mut state.use_gpu, "Use GPU IK");
    ui.separator();

    ui.label("Target (world)");
    ui.add(
        egui::Slider::new(&mut state.target.x, -1.5..=1.5)
            .text("x")
            .step_by(0.005),
    );
    ui.add(
        egui::Slider::new(&mut state.target.y, 0.05..=2.5)
            .text("y")
            .step_by(0.005),
    );
    ui.add(
        egui::Slider::new(&mut state.target.z, -1.5..=1.5)
            .text("z")
            .step_by(0.005),
    );
    ui.separator();

    if let Ok(q) = sim.state_a.joint_q.host() {
        ui.label("Current angles (rad)");
        for i in 0..7.min(q.len()) {
            readout(ui, &format!("  q{i}"), &format!("{:.3}", q[i]));
        }
    }

    ui.add_space(4.0);
    ui.small(
        "Levenberg-Marquardt IK; per-iteration Jacobian on the GPU. \
         Drag the X/Y/Z sliders to move the EE target; the arm chases \
         in real time. Toggle 'Use GPU IK' to A/B against the CPU \
         solver.",
    );
}


