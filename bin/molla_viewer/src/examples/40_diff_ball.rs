fn build_diff_ball_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();

    // Body 0 — kinematic ball whose pose is rewritten each frame from
    // the analytical trajectory.
    let ball = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.0, 0.5, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(ball, 0.10, mp::Transform::IDENTITY)
        .expect("ball");

    // Body 1 — kinematic target indicator (small sphere) drawn at the
    // current target position.
    let target = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(3.0, 1.5, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(target, 0.08, mp::Transform::IDENTITY)
        .expect("target");

    let visuals = vec![BodyVisual::Sphere(0.10), BodyVisual::Sphere(0.08)];
    // Static plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("diff ball model"), visuals)
}

#[derive(Resource)]
struct DiffBallExampleState {
    /// Initial speed magnitude (m/s).
    v: f32,
    /// Initial angle from horizontal (degrees).
    angle_deg: f32,
    /// Target XY (Z = 0).
    target: [f32; 2],
    /// Latest computed miss distance (m).
    miss: f32,
    /// Convergence iter count from last "Optimise" run.
    iters_run: u32,
    /// Current sample point along the playback trajectory.
    playback_t: f32,
    /// Triggered by the pane to re-run gradient descent.
    rerun: bool,
    /// Trail of (v, angle) pairs visited during optimisation.
    history: Vec<[f32; 2]>,
    /// Trail of last-rollout (x, y) sample points (for visualization).
    trajectory: Vec<[f32; 2]>,
}

impl Default for DiffBallExampleState {
    fn default() -> Self {
        Self {
            v: 6.0,
            angle_deg: 50.0,
            target: [3.0, 1.5],
            miss: 0.0,
            iters_run: 0,
            playback_t: 0.0,
            rerun: true,
            history: Vec::new(),
            trajectory: Vec::new(),
        }
    }
}


fn update_diff_ball_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<DiffBallExampleState>,
) {
    if sim.scene != Scene::DiffBall {
        return;
    }
    if sim.model.body_count < 2 {
        return;
    }

    // Re-optimise (v, angle) when triggered by a slider change.
    if state.rerun {
        // Closed-form projectile under gravity (no air drag).
        // Position at time t given (v, θ):
        //   x(t) = v cos θ · t
        //   y(t) = y0 + v sin θ · t − ½ g t²
        // Pick the t* that minimises (x(t)−Tx)² + (y(t)−Ty)² and
        // evaluate the loss; back-propagate via finite differences.
        let target_x = state.target[0] as f64;
        let target_y = state.target[1] as f64;
        let g = 9.81;
        let y0 = 0.5;
        // Sample a coarse grid of t to find the closest approach.
        let trajectory_for = |v: f64, ang: f64| -> Vec<[f32; 2]> {
            let cos_a = ang.cos();
            let sin_a = ang.sin();
            let mut pts = Vec::with_capacity(64);
            for k in 0..64 {
                let t = (k as f64) * 0.04;
                let x = v * cos_a * t;
                let y = y0 + v * sin_a * t - 0.5 * g * t * t;
                if y < 0.0 && k > 1 {
                    break;
                }
                pts.push([x as f32, y as f32]);
            }
            pts
        };
        let miss_for = |v: f64, ang: f64| -> f64 {
            let pts = trajectory_for(v, ang);
            let mut best = f64::INFINITY;
            for p in &pts {
                let dx = p[0] as f64 - target_x;
                let dy = p[1] as f64 - target_y;
                let d2 = dx * dx + dy * dy;
                if d2 < best {
                    best = d2;
                }
            }
            best.sqrt()
        };
        let mut v = state.v as f64;
        let mut ang = (state.angle_deg as f64).to_radians();
        let lr_v = 0.3_f64;
        let lr_a = 0.3_f64;
        let mut history = Vec::new();
        for _it in 0..40 {
            history.push([v as f32, ang.to_degrees() as f32]);
            let m0 = miss_for(v, ang);
            let dv = (miss_for(v + 0.01, ang) - m0) / 0.01;
            let da = (miss_for(v, ang + 0.01) - m0) / 0.01;
            v -= lr_v * dv;
            ang -= lr_a * da;
            v = v.clamp(0.5, 20.0);
            ang = ang.clamp(0.05, 1.55);
            if m0 < 0.05 {
                break;
            }
        }
        state.v = v as f32;
        state.angle_deg = ang.to_degrees() as f32;
        state.miss = miss_for(v, ang) as f32;
        state.iters_run = history.len() as u32;
        state.history = history;
        state.trajectory = trajectory_for(v, ang);
        state.playback_t = 0.0;
        state.rerun = false;
    }

    // Playback: animate the ball along the trajectory.
    state.playback_t += time.delta_secs() * 0.5;
    if state.playback_t > 2.0 {
        state.playback_t = 0.0;
    }
    if state.trajectory.is_empty() {
        return;
    }
    let frac = (state.playback_t / 2.0).clamp(0.0, 1.0);
    let idx = ((frac * (state.trajectory.len() - 1) as f32).floor() as usize)
        .min(state.trajectory.len() - 1);
    let p = state.trajectory[idx];
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    body_q[0] = mp::Transform::from_translation(MVec3::new(p[0] as f64, p[1] as f64, 0.0));
    body_q[1] = mp::Transform::from_translation(MVec3::new(
        state.target[0] as f64,
        state.target[1] as f64,
        0.0,
    ));
    let queue = sim.engine.queue().expect("queue").clone();
    let _ = sim.state_a.body_q.upload(&queue);
}

fn diff_ball_pane(ui: &mut egui::Ui, state: &mut DiffBallExampleState, accent: egui::Color32) {
    ui.label("Initial speed (m/s)");
    if ui
        .add(egui::Slider::new(&mut state.v, 1.0..=15.0))
        .changed()
    {
        state.rerun = true;
    }
    ui.label("Launch angle (deg)");
    if ui
        .add(egui::Slider::new(&mut state.angle_deg, 5.0..=85.0))
        .changed()
    {
        state.rerun = true;
    }
    ui.label("Target X (m)");
    if ui
        .add(egui::Slider::new(&mut state.target[0], 0.5..=8.0))
        .changed()
    {
        state.rerun = true;
    }
    ui.label("Target Y (m)");
    if ui
        .add(egui::Slider::new(&mut state.target[1], 0.0..=4.0))
        .changed()
    {
        state.rerun = true;
    }
    ui.separator();
    readout(ui, "Miss", &format!("{:.3} m", state.miss));
    readout(ui, "Iters run", &format!("{}", state.iters_run));
    if ui
        .add_sized(
            [ui.available_width(), 24.0],
            button_widget("Optimise", accent),
        )
        .clicked()
    {
        state.rerun = true;
    }
    ui.separator();
    ui.small("Projectile under gravity. Tape autograd would");
    ui.small("compute ∂miss/∂(v, θ); we use closed-form here.");
}

fn draw_diff_ball_gizmos(sim: Res<Sim>, state: Res<DiffBallExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::DiffBall {
        return;
    }
    if state.trajectory.len() < 2 {
        return;
    }
    let line_color = Color::srgb(0.55, 0.85, 0.20);
    for w in state.trajectory.windows(2) {
        let a = Vec3::new(w[0][0], w[0][1], 0.0);
        let b = Vec3::new(w[1][0], w[1][1], 0.0);
        gizmos.line(a, b, line_color);
    }
    // Target hoop — circle in the XY plane.
    let target = Vec3::new(state.target[0], state.target[1], 0.0);
    let tgt_color = Color::srgb(0.95, 0.30, 0.20);
    let r = 0.18_f32;
    let segs = 24;
    let mut prev = target + Vec3::new(r, 0.0, 0.0);
    for k in 1..=segs {
        let theta = (k as f32 / segs as f32) * std::f32::consts::TAU;
        let next = target + Vec3::new(r * theta.cos(), r * theta.sin(), 0.0);
        gizmos.line(prev, next, tgt_color);
        prev = next;
    }
}

// --- Stage 36 v1 — FEM heat-equation scene (#47) -------------------
//
// Pure-CPU demo of the Stage 36 FEM framework. A 16×16×16 tet-mesh
// of the unit cube is held by `FemHeatExampleState`; each frame we
// step backward-Euler once via `molla::fem::solve_transient`. The
// tet wireframe is drawn as gizmo lines coloured by the per-vertex
// temperature (blue → red).
//

