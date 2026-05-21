// ═══ Scene 88: DiffsimDrone — MPC over a point-mass quadrotor ══════════
//
// Open-loop trajectory MPC. State `x ∈ ℝ⁶` = (px, py, pz, vx, vy, vz);
// control `u ∈ ℝ³` = (thrust, ax_lat, az_lat) — thrust along +Y, plus
// two lateral acceleration command channels acting in world +X / +Z so
// we have a simple controllable system without modelling attitude
// explicitly. Dynamics per substep:
//     a = (0, thrust − g, 0) + (ax_lat, 0, az_lat);
//     v ← v + a · dt − k_drag · v · dt;
//     p ← p + v · dt.
// Loss = `‖p_T − target‖²` at the final step.
//
// Adam optimises the open-loop sequence over `DRONE_HORIZON` steps. Because the
// reduced point-mass dynamics are linear in the control accelerations, the
// final-position loss has an analytic reverse-mode gradient. This keeps the
// scene deterministic and cheap while staying honest that the full
// Featherstone/autodiff drone path is outside this reduced viewer case.
//
// Visualisation: cyan trajectory line for the optimised rollout +
// orange drone marker that loops through the trajectory. Adam control
// re-runs when the user touches the target slider.

const DRONE_HORIZON: usize = 50;
const DRONE_DT: f64 = 0.05; // seconds per substep
const DRONE_DRAG: f64 = 0.4; // linear damping (1/s)
const DRONE_GRAVITY: f64 = 9.81;
const DRONE_OPT_ITERS: usize = 40;
#[cfg(test)]
const DRONE_FD_EPS: f64 = 1.0e-3;

struct DiffsimDroneState {
    /// Per-step controls flattened as `[thrust_0, ax_0, az_0, thrust_1, …]`.
    controls: Vec<f64>,
    /// Adam optimiser state matching `controls`.
    adam: molla::optim::AdamState,
    cfg: molla::optim::AdamConfig,
    /// Initial position (world).
    p0: MVec3,
    /// Target position (world).
    target: MVec3,
    /// Cached rollout (one entry per substep, `DRONE_HORIZON + 1` total).
    trajectory: Vec<MVec3>,
    /// Last computed loss.
    last_loss: f64,
    /// Optimisation iters consumed this rebuild.
    iters_run: u32,
    /// Last gradient-eval ms (timing of the full Adam iteration).
    last_iter_us: f64,
    /// Playback head — index into `trajectory` for the drone marker.
    playback_t: f64,
    /// User toggles.
    needs_rerun: bool,
    paused_opt: bool,
}

impl Default for DiffsimDroneState {
    fn default() -> Self {
        let n_params = DRONE_HORIZON * 3;
        let mut controls = vec![0.0_f64; n_params];
        // Hover-feed-forward thrust so the optimiser starts at a
        // sensible operating point rather than ballistic freefall.
        for k in 0..DRONE_HORIZON {
            controls[k * 3] = DRONE_GRAVITY;
        }
        Self {
            controls,
            adam: molla::optim::AdamState::new(n_params),
            cfg: molla::optim::AdamConfig {
                lr: 0.5,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1.0e-8,
                weight_decay: 0.0,
            },
            p0: MVec3::new(0.0, 1.0, 0.0),
            target: MVec3::new(1.2, 1.6, 0.8),
            trajectory: Vec::new(),
            last_loss: f64::INFINITY,
            iters_run: 0,
            last_iter_us: 0.0,
            playback_t: 0.0,
            needs_rerun: true,
            paused_opt: false,
        }
    }
}

// Drone forward rollout. Returns final position + the trajectory.
fn drone_rollout(p0: MVec3, controls: &[f64]) -> Vec<MVec3> {
    let mut p = p0;
    let mut v = MVec3::ZERO;
    let dt = DRONE_DT;
    let mut traj = Vec::with_capacity(DRONE_HORIZON + 1);
    traj.push(p);
    for k in 0..DRONE_HORIZON {
        let thrust = controls[k * 3];
        let ax_lat = controls[k * 3 + 1];
        let az_lat = controls[k * 3 + 2];
        let a = MVec3::new(ax_lat, thrust - DRONE_GRAVITY, az_lat);
        v = v + a * dt - v * (DRONE_DRAG * dt);
        p = p + v * dt;
        traj.push(p);
    }
    traj
}

#[cfg(test)]
fn drone_loss(p0: MVec3, target: MVec3, controls: &[f64]) -> f64 {
    let traj = drone_rollout(p0, controls);
    let pf = traj[traj.len() - 1];
    let d = pf - target;
    d.dot(d)
}

fn drone_loss_and_gradient(p0: MVec3, target: MVec3, controls: &[f64]) -> (f64, Vec<f64>) {
    let traj = drone_rollout(p0, controls);
    let pf = traj[traj.len() - 1];
    let residual = pf - target;
    let loss = residual.dot(residual);

    let beta = 1.0 - DRONE_DRAG * DRONE_DT;
    let dt2 = DRONE_DT * DRONE_DT;
    let mut grads = vec![0.0_f64; controls.len()];
    for k in 0..DRONE_HORIZON {
        let remaining = DRONE_HORIZON - k;
        let gain = if (1.0 - beta).abs() > f64::EPSILON {
            dt2 * (1.0 - beta.powi(remaining as i32)) / (1.0 - beta)
        } else {
            dt2 * remaining as f64
        };
        grads[k * 3] = 2.0 * residual.y * gain;
        grads[k * 3 + 1] = 2.0 * residual.x * gain;
        grads[k * 3 + 2] = 2.0 * residual.z * gain;
    }

    (loss, grads)
}

fn update_diffsim_drone(sim: Res<Sim>, run: Res<SimRunState>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::DiffsimDrone {
        return;
    }
    let st = &mut warp.diffsim_drone;
    if st.needs_rerun {
        // Reset Adam + control sequence to the hover feed-forward.
        let n = DRONE_HORIZON * 3;
        st.controls = vec![0.0_f64; n];
        for k in 0..DRONE_HORIZON {
            st.controls[k * 3] = DRONE_GRAVITY;
        }
        st.adam = molla::optim::AdamState::new(n);
        st.iters_run = 0;
        st.last_loss = f64::INFINITY;
        st.trajectory = drone_rollout(st.p0, &st.controls);
        st.needs_rerun = false;
        st.playback_t = 0.0;
    }
    if run.paused || st.paused_opt {
        // Even when paused, keep the playback advancing so the user
        // can see the current best trajectory animate.
        if !run.paused {
            advance_drone_playback(st);
        }
        return;
    }

    if (st.iters_run as usize) < DRONE_OPT_ITERS {
        let t0 = std::time::Instant::now();
        let (base_loss, grads) = drone_loss_and_gradient(st.p0, st.target, &st.controls);
        st.adam.step(&mut st.controls, &grads, &st.cfg);
        st.last_loss = base_loss;
        st.iters_run += 1;
        st.trajectory = drone_rollout(st.p0, &st.controls);
        st.last_iter_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
    }

    advance_drone_playback(st);
}

fn advance_drone_playback(st: &mut DiffsimDroneState) {
    if st.trajectory.is_empty() {
        return;
    }
    let total = DRONE_DT * DRONE_HORIZON as f64;
    st.playback_t += 1.0 / 60.0;
    if st.playback_t > total + 0.5 {
        st.playback_t = 0.0;
    }
}

fn draw_diffsim_drone_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::DiffsimDrone {
        return;
    }
    let st = &warp.diffsim_drone;
    if st.trajectory.is_empty() {
        return;
    }
    // Start marker (purple), target marker (green).
    let p0v = Vec3::new(st.p0.x as f32, st.p0.y as f32, st.p0.z as f32);
    let tv = Vec3::new(st.target.x as f32, st.target.y as f32, st.target.z as f32);
    gizmos.sphere(p0v, 0.05, Color::srgb(0.7, 0.4, 1.0));
    gizmos.sphere(tv, 0.06, Color::srgb(0.2, 1.0, 0.4));
    // Trajectory polyline (cyan).
    let mut prev: Option<Vec3> = None;
    for p in &st.trajectory {
        let v = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
        if let Some(pv) = prev {
            gizmos.line(pv, v, Color::srgb(0.4, 0.85, 1.0));
        }
        prev = Some(v);
    }
    // Drone marker — linear interpolation along the trajectory based
    // on playback time.
    let total = DRONE_DT * DRONE_HORIZON as f64;
    let t = st.playback_t.clamp(0.0, total);
    let f = (t / DRONE_DT).floor() as usize;
    let frac = (t / DRONE_DT - f as f64) as f32;
    let i = f.min(DRONE_HORIZON);
    let j = (i + 1).min(DRONE_HORIZON);
    let pa = st.trajectory[i];
    let pb = st.trajectory[j];
    let pm = pa + (pb - pa) * frac as f64;
    let pmv = Vec3::new(pm.x as f32, pm.y as f32, pm.z as f32);
    gizmos.sphere(pmv, 0.07, Color::srgb(1.0, 0.6, 0.2));
    // Tiny "rotor" cross around the drone marker.
    let arm = 0.10;
    gizmos.line(
        pmv + Vec3::X * arm,
        pmv - Vec3::X * arm,
        Color::srgb(1.0, 0.4, 0.3),
    );
    gizmos.line(
        pmv + Vec3::Z * arm,
        pmv - Vec3::Z * arm,
        Color::srgb(1.0, 0.4, 0.3),
    );
}

fn diffsim_drone_pane(ui: &mut egui::Ui, st: &mut DiffsimDroneState) {
    ui.label("Diffsim drone — MPC + Adam + analytic gradients (Newton-port)");
    ui.separator();
    readout(ui, "Backend", "point-mass model · analytic ∂L/∂u · Adam");
    readout(
        ui,
        "Horizon",
        &format!("{} × {:.0} ms", DRONE_HORIZON, DRONE_DT * 1000.0),
    );
    readout(
        ui,
        "Iter",
        &format!("{} / {}", st.iters_run, DRONE_OPT_ITERS),
    );
    readout(ui, "Loss", &format!("{:.4e}", st.last_loss));
    if let Some(pf) = st.trajectory.last() {
        let err = (*pf - st.target).length();
        readout(ui, "‖p_T − target‖", &format!("{:.4} m", err));
    }
    readout(
        ui,
        "ms / iter",
        &format!("{:.2}", st.last_iter_us / 1_000.0),
    );
    ui.separator();
    ui.label("Start (world)");
    let mut p0 = st.p0;
    let dx = ui
        .add(egui::Slider::new(&mut p0.x, -2.0..=2.0).text("x₀"))
        .changed();
    let dy = ui
        .add(egui::Slider::new(&mut p0.y, 0.2..=3.0).text("y₀"))
        .changed();
    let dz = ui
        .add(egui::Slider::new(&mut p0.z, -2.0..=2.0).text("z₀"))
        .changed();
    if dx || dy || dz {
        st.p0 = p0;
        st.needs_rerun = true;
    }
    ui.separator();
    ui.label("Target (world)");
    let mut t = st.target;
    let dxt = ui
        .add(egui::Slider::new(&mut t.x, -2.0..=2.0).text("x"))
        .changed();
    let dyt = ui
        .add(egui::Slider::new(&mut t.y, 0.2..=3.0).text("y"))
        .changed();
    let dzt = ui
        .add(egui::Slider::new(&mut t.z, -2.0..=2.0).text("z"))
        .changed();
    if dxt || dyt || dzt {
        st.target = t;
        st.needs_rerun = true;
    }
    ui.separator();
    ui.label("Adam hyperparameters");
    let _ = ui.add(
        egui::Slider::new(&mut st.cfg.lr, 0.001..=2.0)
            .text("lr")
            .logarithmic(true),
    );
    let _ = ui.add(egui::Slider::new(&mut st.cfg.beta1, 0.5..=0.9999).text("β1"));
    let _ = ui.add(egui::Slider::new(&mut st.cfg.beta2, 0.9..=0.99999).text("β2"));
    ui.separator();
    let _ = ui.checkbox(&mut st.paused_opt, "Pause optimisation");
    if ui.button("Re-run from hover").clicked() {
        st.needs_rerun = true;
    }
    ui.small(
        "Open-loop control sequence over a point-mass quadrotor. \
         The full Featherstone quadrotor has no autograd path in scope, \
         so gradients are the analytic closed-form ∂L/∂u of the damped \
         point-mass rollout (validated against finite differences in tests).",
    );
}

