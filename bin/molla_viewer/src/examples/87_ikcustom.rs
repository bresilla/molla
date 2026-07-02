// ═══ Scene 87: IkCustom — IK with collision-avoidance objective ═════════
//
// Reuses the Stage-16 7-DOF Franka-style arm geometry (built via
// `build_gpu_ik_arm_model`) and adds a sphere obstacle in front of the
// arm. The standard `IkSolver::solve_gpu` from #10 is reused unchanged;
// per-frame, in addition to the EE-position objective, we synthesise an
// extra `IkObjective::Position` per link whose target is the link's
// current world position projected to lie just outside the sphere when
// the link centre is within `obstacle_margin` of the sphere surface.
// When a link is clear of the sphere, the synthetic target collapses to
// the link's current position (residual = 0).
//
// This is the "shape the residual to encode the constraint" pattern —
// the IkSolver shipped only `Position` and `JointLimit` objective
// kinds, so a collision-avoidance objective is realised as a
// dynamically-targeted Position objective. Newton's `CollisionSphereAvoidObjective`
// uses softplus penetration depth; this maps cleanly because once a
// link is inside the safety margin, the residual norm grows linearly
// with penetration, which is what LM consumes.

struct IkCustomState {
    /// Target world position for the EE.
    target: MVec3,
    /// Obstacle sphere centre (world).
    obstacle_center: MVec3,
    /// Obstacle sphere radius.
    obstacle_radius: f32,
    /// Per-link sphere proxy radius used for collision avoidance.
    link_radius: f32,
    /// Extra margin added to the (link_radius + obstacle_radius) gap
    /// so the solver pulls links well clear of the surface.
    obstacle_margin: f32,
    /// Weight on the collision residuals relative to the EE position
    /// residual (which has weight 1.0).
    collision_weight: f32,
    /// Toggle: when off, only the EE objective runs (so the user can
    /// see the arm slam straight through the obstacle).
    collision_enabled: bool,
    /// Have we initialised `target` from the rest EE pose?
    target_init: bool,
    /// Last solver telemetry (for the pane).
    last_iters: usize,
    last_residual: f64,
    last_solve_us: f64,
    /// Worst-case (smallest) signed-distance from any tracked link to
    /// the obstacle surface this frame. Negative = penetrating.
    worst_link_clearance: f64,
}

impl Default for IkCustomState {
    fn default() -> Self {
        Self {
            target: MVec3::ZERO,
            obstacle_center: MVec3::new(0.30, 1.10, 0.05),
            obstacle_radius: 0.15,
            link_radius: 0.06,
            obstacle_margin: 0.04,
            collision_weight: 3.0,
            collision_enabled: true,
            target_init: false,
            last_iters: 0,
            last_residual: 0.0,
            last_solve_us: 0.0,
            worst_link_clearance: f64::INFINITY,
        }
    }
}

// Indices of the links we treat as collision spheres. We skip body 0
// (the rooted base) since it never moves and would just add a no-op
// row to the IK Jacobian.
const IK_CUSTOM_LINKS: &[usize] = &[1, 2, 3, 4, 5, 6];

fn ik_custom_collision_projection(
    p: MVec3,
    c_obs: MVec3,
    r_obs: f64,
    r_link: f64,
    margin: f64,
) -> (f64, Option<MVec3>) {
    let d = p - c_obs;
    let dist = d.length();
    let signed = dist - (r_obs + r_link);
    let r_min = r_obs + r_link + margin;
    if dist < 1.0e-6 {
        return (signed, Some(c_obs + MVec3::new(0.0, r_min, 0.0)));
    }
    if dist < r_min {
        let n = d / dist;
        (signed, Some(c_obs + n * r_min))
    } else {
        (signed, None)
    }
}

fn update_ik_custom(mut sim: ResMut<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::IkCustom {
        return;
    }
    let st = &mut warp.ik_custom;
    let Sim {
        model,
        state_a,
        engine,
        ..
    } = &mut *sim;

    // EE = last body in the chain (mirrors `update_gpu_ik_arm`).
    let ee_body = model.body_count - 1;

    // Lazy-seed the EE target from rest FK so the arm starts at a
    // reachable pose just in front of the sphere.
    if !st.target_init {
        molla::sim::eval_fk(model, state_a).expect("eval_fk");
        let bq = state_a.body_q.host().expect("body_q");
        // Park the target on the opposite side of the obstacle so the
        // arm has to either go around or burrow through (which the
        // collision objective prevents).
        let rest_ee = bq[ee_body].position;
        st.target = MVec3::new(rest_ee.x + 0.35, rest_ee.y - 0.20, rest_ee.z + 0.15);
        st.target_init = true;
    }

    // Build the objective list. Always include the EE position
    // objective. When collisions are enabled, append a per-link
    // Position objective whose target sits OUTSIDE the obstacle.
    //
    // For each link i, let `p_i` be the link's current world centre.
    // Direction from obstacle centre to the link: `d = p_i − c_obs`.
    // Required minimum distance to keep the link clear:
    //     r_min = r_obs + r_link + margin.
    // If `‖d‖ < r_min`, the synthetic target is at `c_obs +
    // (r_min / ‖d‖) · d` — the radial projection out of the sphere.
    // Else, the target collapses to `p_i` (zero residual).
    let q0: Vec<f64> = state_a
        .joint_q
        .host()
        .expect("joint_q").to_vec();
    let body_q_now = state_a.body_q.host().expect("body_q").to_vec();
    let c_obs = st.obstacle_center;
    let r_obs = st.obstacle_radius as f64;
    let r_link = st.link_radius as f64;
    let margin = st.obstacle_margin as f64;
    let mut worst_clearance: f64 = f64::INFINITY;

    let mut objectives: Vec<IkObjective> = vec![IkObjective::Position {
        body: ee_body,
        point_local: MVec3::ZERO,
        target_world: st.target,
        weight: 1.0,
    }];
    if st.collision_enabled {
        for &li in IK_CUSTOM_LINKS {
            if li >= body_q_now.len() {
                continue;
            }
            let p = body_q_now[li].position;
            let (signed, target) = ik_custom_collision_projection(p, c_obs, r_obs, r_link, margin);
            if signed < worst_clearance {
                worst_clearance = signed;
            }
            if let Some(synth) = target {
                objectives.push(IkObjective::Position {
                    body: li,
                    point_local: MVec3::ZERO,
                    target_world: synth,
                    weight: st.collision_weight as f64,
                });
            }
            // else: link is clear; emit no row (cheaper than a weight-zero
            // residual + Jacobian row).
        }
    }
    st.worst_link_clearance = worst_clearance;

    let solver = IkSolver::new(objectives).with_config(IkConfig {
        max_iter: 30,
        residual_tol: 5.0e-4,
        ..IkConfig::default()
    });

    let t0 = std::time::Instant::now();
    let result = solver.solve_gpu(engine, model, state_a, &q0);
    st.last_solve_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
    match result {
        Ok(r) => {
            st.last_iters = r.iters;
            st.last_residual = r.residual;
        }
        Err(e) => warn!("ik_custom: solve failed: {e}"),
    }
}

fn draw_ik_custom_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::IkCustom {
        return;
    }
    let st = &warp.ik_custom;
    // Target marker (green).
    let target_v = Vec3::new(st.target.x as f32, st.target.y as f32, st.target.z as f32);
    gizmos.sphere(target_v, 0.04, Color::srgb(0.2, 1.0, 0.4));
    // Obstacle sphere — wireframe by stacking three orthogonal circles
    // so users can see the inside of the shell.
    let center = Vec3::new(
        st.obstacle_center.x as f32,
        st.obstacle_center.y as f32,
        st.obstacle_center.z as f32,
    );
    let r = st.obstacle_radius;
    let color = if st.worst_link_clearance < 0.0 {
        Color::srgb(1.0, 0.3, 0.2) // red — link inside
    } else if (st.worst_link_clearance as f32) < st.obstacle_margin {
        Color::srgb(1.0, 0.7, 0.2) // amber — within margin
    } else {
        Color::srgb(0.4, 0.7, 1.0) // blue — clear
    };
    let segs = 48;
    for axis in 0..3 {
        for s in 0..segs {
            let t0 = (s as f32 / segs as f32) * std::f32::consts::TAU;
            let t1 = ((s + 1) as f32 / segs as f32) * std::f32::consts::TAU;
            let (a, b) = match axis {
                0 => (
                    Vec3::new(0.0, r * t0.cos(), r * t0.sin()),
                    Vec3::new(0.0, r * t1.cos(), r * t1.sin()),
                ),
                1 => (
                    Vec3::new(r * t0.cos(), 0.0, r * t0.sin()),
                    Vec3::new(r * t1.cos(), 0.0, r * t1.sin()),
                ),
                _ => (
                    Vec3::new(r * t0.cos(), r * t0.sin(), 0.0),
                    Vec3::new(r * t1.cos(), r * t1.sin(), 0.0),
                ),
            };
            gizmos.line(center + a, center + b, color);
        }
    }
    // Link spheres — outline each tracked link's collision proxy.
    if let Ok(bq) = sim.state_a.body_q.host() {
        for &li in IK_CUSTOM_LINKS {
            if li >= bq.len() {
                continue;
            }
            let p = bq[li].position;
            let pv = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            gizmos.sphere(pv, st.link_radius, Color::srgba(1.0, 0.95, 0.8, 0.5));
        }
    }
}

fn ik_custom_pane(ui: &mut egui::Ui, st: &mut IkCustomState) {
    ui.label("IK custom — Newton-port `ik_custom` (collision objective)");
    ui.separator();
    readout(ui, "Solver", "GPU LM (Stage 16)");
    readout(ui, "DOFs", "7");
    readout(ui, "Iterations", &format!("{}", st.last_iters));
    readout(ui, "Residual", &format!("{:.5}", st.last_residual));
    readout(
        ui,
        "ms / solve",
        &format!("{:.2}", st.last_solve_us / 1000.0),
    );
    let clearance = st.worst_link_clearance;
    let clr_label = if clearance.is_finite() {
        format!("{:+.3} m", clearance)
    } else {
        "n/a".to_string()
    };
    readout(ui, "Worst link gap", &clr_label);
    ui.separator();

    ui.checkbox(&mut st.collision_enabled, "Enable collision objective");
    ui.separator();
    ui.label("EE target (world)");
    ui.add(
        egui::Slider::new(&mut st.target.x, -1.5..=1.5)
            .text("x")
            .step_by(0.005),
    );
    ui.add(
        egui::Slider::new(&mut st.target.y, 0.05..=2.5)
            .text("y")
            .step_by(0.005),
    );
    ui.add(
        egui::Slider::new(&mut st.target.z, -1.5..=1.5)
            .text("z")
            .step_by(0.005),
    );
    ui.separator();
    ui.label("Obstacle sphere centre");
    ui.add(egui::Slider::new(&mut st.obstacle_center.x, -1.0..=1.0).text("cx"));
    ui.add(egui::Slider::new(&mut st.obstacle_center.y, 0.3..=2.0).text("cy"));
    ui.add(egui::Slider::new(&mut st.obstacle_center.z, -1.0..=1.0).text("cz"));
    ui.add(egui::Slider::new(&mut st.obstacle_radius, 0.05..=0.40).text("radius"));
    ui.separator();
    ui.label("Collision tuning");
    ui.add(egui::Slider::new(&mut st.link_radius, 0.02..=0.15).text("link radius"));
    ui.add(egui::Slider::new(&mut st.obstacle_margin, 0.0..=0.20).text("margin"));
    ui.add(
        egui::Slider::new(&mut st.collision_weight, 0.1..=20.0)
            .text("collision weight")
            .logarithmic(true),
    );
    if ui.button("Re-seed target").clicked() {
        st.target_init = false;
    }
    ui.small(
        "Collision avoidance is encoded as per-link Position objectives whose target \
         is the link's centre projected to the obstacle's surface-plus-margin radius. \
         Disable the toggle to see the arm reach straight through.",
    );
}

