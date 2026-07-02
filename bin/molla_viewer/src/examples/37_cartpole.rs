// CartPole scene — identical topology to `NeuralPolicy` (cart on
// prismatic + pole on revolute) but driven by a hand-written PD on
// the pole angle. Port of `example_robot_cartpole.py` /
// `example_selection_cartpole.py`.
fn build_cartpole_model() -> (mp::Model, Vec<BodyVisual>) {
    // Re-use the existing cart-pole topology — identical bodies and
    // joints, just a different controller in the pre-step system.
    build_neural_policy_model()
}

#[derive(Resource)]
struct CartPoleExampleState {
    /// Single-cart-pole controller (lazy-init).
    controller: Option<ControllerPd>,
    /// PD gains.
    kp: f32,
    kd: f32,
    /// Pole angle target (rad). Default 0 = upright.
    target_theta: f32,
    /// Live readouts.
    last_x: f32,
    last_theta: f32,
    last_force: f32,
    /// Auto perturb timer.
    perturb_timer: f32,
    auto_perturb: bool,
}

impl Default for CartPoleExampleState {
    fn default() -> Self {
        Self {
            controller: None,
            kp: 50.0,
            kd: 8.0,
            target_theta: 0.0,
            last_x: 0.0,
            last_theta: 0.0,
            last_force: 0.0,
            perturb_timer: 0.0,
            auto_perturb: true,
        }
    }
}


fn update_cartpole_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<CartPoleExampleState>,
) {
    let is_swarm = sim.scene == Scene::CartPoleSwarm;
    if sim.scene != Scene::CartPole && !is_swarm {
        if state.controller.is_some() {
            state.controller = None;
        }
        return;
    }
    if sim.model.joint_dof_total < 2 {
        return;
    }
    // Lazy-init: ControllerPd is over-engineered for our needs here —
    // we drive `joint_force` directly from a closed-form LQR-style
    // balance policy below. We still construct it once so subsequent
    // re-entries are cheap; the controller object is dropped on scene
    // switch.
    if state.controller.is_none() {
        let cfg = ControllerPdConfig {
            dof: 0,
            kp: state.kp as f64,
            kd: state.kd as f64,
            max_effort: 60.0,
        };
        match ControllerPd::new(&sim.engine, &sim.model, &[cfg]) {
            Ok(c) => state.controller = Some(c),
            Err(e) => {
                warn!("cartpole: controller init failed: {e}");
                return;
            }
        }
    }

    // Read state. Snapshot to owned Vecs so we can mutate `sim` below.
    let q_vec: Vec<f64> = match sim.state_a.joint_q.host() {
        Ok(q) => q.to_vec(),
        Err(_) => return,
    };
    let qd_vec: Vec<f64> = match sim.state_a.joint_qd.host() {
        Ok(q) => q.to_vec(),
        Err(_) => return,
    };
    if q_vec.len() < 2 || qd_vec.len() < 2 {
        return;
    }
    // Single-pole readouts mirror the first cart-pole.
    let cart_x = q_vec[0] as f32;
    let _cart_xd = qd_vec[0] as f32;
    let theta = q_vec[1] as f32;
    let _theta_d = qd_vec[1] as f32;
    state.last_x = cart_x;
    state.last_theta = theta;

    // Auto-perturb every 3 s — only the first cart in swarm mode
    // (otherwise the swarm jumps in lockstep and looks fake).
    state.perturb_timer += time.delta_secs();
    if state.auto_perturb && state.perturb_timer > 3.0 {
        state.perturb_timer = 0.0;
        let queue = sim.engine.queue().expect("queue").clone();
        if let Ok(qdh) = sim.state_a.joint_qd.host_mut()
            && qdh.len() >= 2 {
                let sign = if theta >= 0.0 { -1.0 } else { 1.0 };
                qdh[1] += sign * 1.5;
            }
        let _ = sim.state_a.joint_qd.upload(&queue);
    }

    // Inverted-pendulum balance law per cart-pole pair.
    let target_theta = state.target_theta;
    let queue = sim.engine.queue().expect("queue").clone();
    if let Ok(jf) = sim.control.joint_force.host_mut() {
        let pairs = jf.len() / 2;
        for i in 0..pairs {
            let xx = q_vec[2 * i] as f32;
            let xxd = qd_vec[2 * i] as f32;
            let th = q_vec[2 * i + 1] as f32;
            let thd = qd_vec[2 * i + 1] as f32;
            let force = -(2.5 * xx + 1.0 * xxd + state.kp * (th - target_theta) + state.kd * thd);
            jf[2 * i] = force.clamp(-60.0, 60.0) as f64;
            // Pole revolute receives no actuated torque.
            jf[2 * i + 1] = 0.0;
        }
        if !jf.is_empty() {
            state.last_force = jf[0] as f32;
        }
    }
    let _ = sim.control.joint_force.upload(&queue);
    let _ = is_swarm;
}

fn cartpole_pane(ui: &mut egui::Ui, state: &mut CartPoleExampleState, _accent: egui::Color32) {
    readout(ui, "cart x", &format!("{:>+5.2} m", state.last_x));
    readout(ui, "pole θ", &format!("{:>+5.2} rad", state.last_theta));
    readout(ui, "force", &format!("{:>+5.2} N", state.last_force));
    ui.separator();
    ui.label("PD gains");
    ui.add(egui::Slider::new(&mut state.kp, 1.0..=300.0).text("kp"));
    ui.add(egui::Slider::new(&mut state.kd, 0.0..=50.0).text("kd"));
    ui.label("Pole target (rad, 0 = upright)");
    ui.add(egui::Slider::new(&mut state.target_theta, -0.5..=0.5));
    ui.checkbox(&mut state.auto_perturb, "Auto-perturb every 3 s");
    ui.separator();
    ui.small("PD on the pole revolute, force applied to cart prismatic.");
}

