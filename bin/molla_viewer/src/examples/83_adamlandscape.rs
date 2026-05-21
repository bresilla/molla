// ═══ Scene 83: AdamLandscape (Stage 38c) ═══════════════════════════════
//
// 2D Rosenbrock loss landscape — `J(x, y) = (1 − x)² + 100·(y − x²)²` —
// with a single Adam optimizer chasing the global minimum at (1, 1).
// The landscape is rendered as a grid of small colored spheres at
// height = `log(1 + J(x, y))`; the trajectory is drawn as a line +
// markers from init to current iterate. Pane exposes the standard Adam
// hyperparameters.

const ADAM_LANDSCAPE_GRID: usize = 24;

struct AdamLandscapeState {
    params: [f64; 2],
    init: [f64; 2],
    adam: molla::optim::AdamState,
    cfg: molla::optim::AdamConfig,
    trajectory: Vec<[f64; 2]>,
    loss: f64,
    paused: bool,
    user_reset: bool,
}

impl Default for AdamLandscapeState {
    fn default() -> Self {
        let init = [-1.0, 1.0];
        Self {
            params: init,
            init,
            adam: molla::optim::AdamState::new(2),
            cfg: molla::optim::AdamConfig {
                lr: 0.05,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1.0e-8,
                weight_decay: 0.0,
            },
            trajectory: vec![init],
            loss: rosenbrock_loss(init[0], init[1]),
            paused: false,
            user_reset: false,
        }
    }
}

fn rosenbrock_loss(x: f64, y: f64) -> f64 {
    let a = 1.0 - x;
    let b = y - x * x;
    a * a + 100.0 * b * b
}

fn rosenbrock_grad(x: f64, y: f64) -> [f64; 2] {
    // ∂J/∂x = -2·(1 − x) − 400·(y − x²)·x
    // ∂J/∂y =  200·(y − x²)
    let b = y - x * x;
    let dx = -2.0 * (1.0 - x) - 400.0 * b * x;
    let dy = 200.0 * b;
    [dx, dy]
}

fn update_adam_landscape(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::AdamLandscape {
        return;
    }
    let st = &mut warp.adam_landscape;
    if st.user_reset {
        let init = st.init;
        st.params = init;
        st.adam = molla::optim::AdamState::new(2);
        st.trajectory.clear();
        st.trajectory.push(init);
        st.loss = rosenbrock_loss(init[0], init[1]);
        st.user_reset = false;
        return;
    }
    if run.paused || st.paused {
        return;
    }
    let g = rosenbrock_grad(st.params[0], st.params[1]);
    st.adam.step(&mut st.params, &g, &st.cfg);
    st.loss = rosenbrock_loss(st.params[0], st.params[1]);
    st.trajectory.push(st.params);
    // Trajectory cap so it doesn't grow unbounded on long runs.
    if st.trajectory.len() > 5000 {
        st.trajectory.drain(0..1000);
    }
}

fn adam_landscape_color(loss: f64) -> Color {
    // log(1 + J) maps loss into a roughly visible range.
    let t = (1.0 + loss).ln() / 6.0;
    let t = t.clamp(0.0, 1.0) as f32;
    if t < 0.5 {
        let s = t * 2.0;
        Color::srgb(0.05, 0.20 + 0.65 * s, 1.0 - 0.6 * s)
    } else {
        let s = (t - 0.5) * 2.0;
        Color::srgb(0.05 + 0.95 * s, 0.85 - 0.55 * s, 0.40 - 0.40 * s)
    }
}

fn draw_adam_landscape_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::AdamLandscape {
        return;
    }
    let st = &warp.adam_landscape;
    let n = ADAM_LANDSCAPE_GRID;
    // Sample (x, y) ∈ [-2, 2]² on an n×n grid; colour by log(1 + J).
    let lo = -2.0_f32;
    let hi = 2.0_f32;
    let step = (hi - lo) / n as f32;
    for j in 0..n {
        let y = lo + (j as f32 + 0.5) * step;
        for i in 0..n {
            let x = lo + (i as f32 + 0.5) * step;
            let loss = rosenbrock_loss(x as f64, y as f64);
            let h = ((1.0 + loss).ln() as f32 * 0.08).min(0.6);
            let pos = Vec3::new(x * 0.4, h, y * 0.4);
            gizmos.sphere(pos, 0.02, adam_landscape_color(loss));
        }
    }
    // Trajectory as a line strip, with growing-size markers along the way.
    let mut last: Option<Vec3> = None;
    for (k, p) in st.trajectory.iter().enumerate() {
        let x = p[0] as f32;
        let y = p[1] as f32;
        let loss = rosenbrock_loss(p[0], p[1]);
        let h = ((1.0 + loss).ln() as f32 * 0.08).min(0.6) + 0.04;
        let pos = Vec3::new(x * 0.4, h, y * 0.4);
        if let Some(prev) = last {
            gizmos.line(prev, pos, Color::srgb(1.0, 0.9, 0.3));
        }
        last = Some(pos);
        // Mark every 10th iterate.
        if k % 10 == 0 {
            gizmos.sphere(pos, 0.025, Color::srgb(1.0, 0.95, 0.4));
        }
    }
    // Current iterate as a bright marker + a tiny target marker at (1, 1).
    if let Some(p) = st.trajectory.last() {
        let x = p[0] as f32;
        let y = p[1] as f32;
        let loss = rosenbrock_loss(p[0], p[1]);
        let h = ((1.0 + loss).ln() as f32 * 0.08).min(0.6) + 0.08;
        let pos = Vec3::new(x * 0.4, h, y * 0.4);
        gizmos.sphere(pos, 0.05, Color::srgb(1.0, 1.0, 1.0));
    }
    let target = Vec3::new(0.4, 0.04, 0.4);
    gizmos.sphere(target, 0.04, Color::srgb(0.2, 1.0, 0.4));
}

fn adam_landscape_pane(ui: &mut egui::Ui, st: &mut AdamLandscapeState) {
    ui.label("Adam optimization on 2D Rosenbrock — Stage 38c");
    ui.separator();
    readout(ui, "Iter", &format!("{}", st.adam.step));
    readout(
        ui,
        "Params",
        &format!("({:+.3}, {:+.3})", st.params[0], st.params[1]),
    );
    readout(ui, "Loss J", &format!("{:.3e}", st.loss));
    let dist = ((1.0 - st.params[0]).powi(2) + (1.0 - st.params[1]).powi(2)).sqrt();
    readout(ui, "‖θ − target‖", &format!("{:.4}", dist));
    ui.separator();
    ui.label("Hyperparameters");
    let _ = ui.add(
        egui::Slider::new(&mut st.cfg.lr, 0.001..=0.5)
            .text("lr")
            .logarithmic(true),
    );
    let _ = ui.add(egui::Slider::new(&mut st.cfg.beta1, 0.5..=0.9999).text("β1"));
    let _ = ui.add(egui::Slider::new(&mut st.cfg.beta2, 0.9..=0.99999).text("β2"));
    let _ = ui.add(egui::Slider::new(&mut st.cfg.weight_decay, 0.0..=0.1).text("weight decay"));
    ui.separator();
    ui.label("Initial point");
    let mut init_x = st.init[0];
    let mut init_y = st.init[1];
    let dx = ui
        .add(egui::Slider::new(&mut init_x, -2.0..=2.0).text("x₀"))
        .changed();
    let dy = ui
        .add(egui::Slider::new(&mut init_y, -2.0..=2.0).text("y₀"))
        .changed();
    if dx || dy {
        st.init = [init_x, init_y];
        st.user_reset = true;
    }
    ui.separator();
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset to init").clicked() {
        st.user_reset = true;
    }
    ui.small("Yellow trajectory = Adam path; green sphere = (1, 1) target.");
    ui.small("Adam: m + v moments with bias correction (Kingma & Ba).");
}

