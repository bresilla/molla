// Stage-31 neural-LSTM demo runtime state. Mirrors
// [`NeuralPolicyExampleState`], but the controller carries hidden +
// cell state across substeps. The right-rail pane shows both the
// last input vector + the live hidden-state snapshot.
//
// Hidden / output dims are baked here (`HID = 16`, `OUT = 1`) so
// `last_hidden` can be a plain array — keeps the resource bound to
// `Copy` constants without a `Vec<f32>` allocation per frame.
const LSTM_HID_DIM: usize = 16;

#[derive(Resource)]
struct NeuralLstmExampleState {
    actuator: Option<Actuator>,
    /// Most recent input vector (4 floats), for the readout pane.
    last_inputs: [f32; 4],
    /// Most recent network output (1 float — cart-force command).
    last_output: f32,
    /// Most recent delivered force after clamping.
    last_delivered: f32,
    /// Most recent hidden-state vector, for the readout pane. Sized
    /// for `LSTM_HID_DIM`.
    last_hidden: [f32; LSTM_HID_DIM],
    /// Periodic pole kick toggle.
    auto_perturb: bool,
    perturb_timer: f32,
    /// Effort cap (N) for the cart force.
    max_force: f32,
}

impl Default for NeuralLstmExampleState {
    fn default() -> Self {
        Self {
            actuator: None,
            last_inputs: [0.0; 4],
            last_output: 0.0,
            last_delivered: 0.0,
            last_hidden: [0.0; LSTM_HID_DIM],
            auto_perturb: true,
            perturb_timer: 0.0,
            max_force: 25.0,
        }
    }
}

fn update_neural_lstm_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<NeuralLstmExampleState>,
) {
    if sim.scene != Scene::NeuralLstm {
        if state.actuator.is_some() {
            state.actuator = None;
        }
        return;
    }

    // Lazy-init the pipeline + upload weights on first frame in scene.
    if state.actuator.is_none() {
        let cfg = ControllerNeuralLstmConfig {
            input_dim: 4,
            hidden_dim: LSTM_HID_DIM as u32,
            output_dim: 1,
        };
        let dof_indices = vec![0u32]; // cart prismatic = joint_force[0]
        let lstm_res = ControllerNeuralLstm::new(&sim.engine, cfg, &dof_indices);
        let delay_res = Delay::new(
            &sim.engine,
            &[DelayConfig {
                dof: 0,
                delay_steps: 0,
            }],
            1,
        );
        let max_force = state.max_force as f64;
        let clamp_res = ClampingMaxEffort::new(
            &sim.engine,
            &[ClampingMaxEffortConfig {
                dof: 0,
                max_effort: max_force,
            }],
        );
        match (lstm_res, delay_res, clamp_res) {
            (Ok(mut lstm), Ok(delay), Ok(clamp)) => {
                let weights = neural_lstm_lqr_weights();
                let device = sim.engine.device().unwrap().clone();
                let queue = sim.engine.queue().unwrap().clone();
                if let Err(e) = lstm.set_weights(&device, &queue, &weights) {
                    warn!("neural lstm: set_weights failed: {e}");
                    return;
                }
                if let Err(e) = lstm.reset_hidden_state(&queue) {
                    warn!("neural lstm: reset_hidden_state failed: {e}");
                    return;
                }
                state.actuator = Some(Actuator::new(
                    delay,
                    ControllerStage::Lstm(lstm),
                    ClampingStage::MaxEffort(clamp),
                ));
                info!("neural lstm: LSTM online — 4→16→1 LQR-style balance net (recurrent)");
            }
            (l, d, c) => {
                if let Err(e) = l {
                    warn!("neural lstm: lstm init failed: {e}");
                }
                if let Err(e) = d {
                    warn!("neural lstm: delay init failed: {e}");
                }
                if let Err(e) = c {
                    warn!("neural lstm: clamp init failed: {e}");
                }
                return;
            }
        }
    }

    // Pull joint state host-side.
    let (cart_x, cart_xd, theta, theta_d) = {
        let q = sim.state_a.joint_q.host().ok();
        let qd = sim.state_a.joint_qd.host().ok();
        match (q, qd) {
            (Some(q), Some(qd)) if q.len() >= 2 && qd.len() >= 2 => {
                (q[0] as f32, qd[0] as f32, q[1] as f32, qd[1] as f32)
            }
            _ => return,
        }
    };

    // Periodic perturb — same convention as the MLP demo.
    state.perturb_timer += time.delta_secs();
    if state.auto_perturb && state.perturb_timer > 3.0 {
        state.perturb_timer = 0.0;
        let sign = if (theta + 0.001).is_sign_positive() {
            -1.0
        } else {
            1.0
        };
        let queue = sim.engine.queue().expect("queue").clone();
        if let Ok(qd) = sim.state_a.joint_qd.host_mut()
            && qd.len() >= 2 {
                qd[1] += sign * 1.5;
            }
        let _ = sim.state_a.joint_qd.upload(&queue);
    }

    // Same observation scaling as the MLP demo so the hand-rolled
    // gains carry over without retuning.
    let inputs: [f32; 4] = [cart_x * 0.5, cart_xd * 0.2, theta, theta_d * 0.2];
    state.last_inputs = inputs;
    let inputs_f64: Vec<f64> = inputs.iter().map(|v| *v as f64).collect();

    let Sim {
        ref engine,
        ref state_a,
        ref mut control,
        ..
    } = *sim;
    if let Some(act) = state.actuator.as_mut() {
        let queue = engine.queue().expect("queue").clone();
        match act.controller_mut() {
            ControllerStage::Lstm(lstm) => {
                if let Err(e) = lstm.set_inputs(&queue, &inputs_f64) {
                    warn!("neural lstm: set_inputs failed: {e}");
                    return;
                }
            }
            _ => unreachable!("neural lstm scene must use Lstm controller"),
        }
        if let Err(e) = act.step(engine, state_a, control, FIXED_DT) {
            warn!("neural lstm: pipeline step failed: {e}");
            return;
        }

        // Read the network output, hidden state, and post-clamp force.
        // Buffer the downloads into locals first so we don't keep
        // `lstm` borrowed while writing back to `state`.
        let device = engine.device().expect("device").clone();
        let mut new_output: Option<f32> = None;
        let mut new_hidden: Option<Vec<f32>> = None;
        if let ControllerStage::Lstm(lstm) = act.controller_mut() {
            if let Ok(outs) = lstm.download_outputs(&device, &queue)
                && !outs.is_empty() {
                    new_output = Some(outs[0] as f32);
                }
            if let Ok(hs) = lstm.download_hidden_state(&device, &queue) {
                new_hidden = Some(hs.iter().map(|v| *v as f32).collect());
            }
        }
        if let Some(v) = new_output {
            state.last_output = v;
        }
        if let Some(hs) = new_hidden {
            let n = state.last_hidden.len().min(hs.len());
            state.last_hidden[..n].copy_from_slice(&hs[..n]);
        }
        if control.joint_force.download(&device, &queue).is_ok()
            && let Ok(jf) = control.joint_force.host()
                && !jf.is_empty() {
                    state.last_delivered = jf[0] as f32;
                }
    }
}

// Hand-rolled LSTM weights expressing an LQR-style cart-pole balance
// policy. **Random init does NOT balance the pole** — the recurrent
// gates make the optimisation landscape non-trivial. To keep the
// demo functional v1 deliberately suppresses the recurrence so the
// LSTM behaves like a stateful linear-projection wrapper:
//
//   - `f = sigmoid(b_f)` with `b_f = -8` → `f ≈ 0` → forget previous c.
//   - `i = sigmoid(b_i)` with `b_i = +8` → `i ≈ 1` → admit candidate.
//   - `g = tanh(W_g · x_t)` (no recurrence) → cell state ≈ tanh(x).
//   - `o = sigmoid(b_o)` with `b_o = +8` → `o ≈ 1` → output ≈ tanh(c).
//
// Then the linear projection `y = W_y · h_t + b_y` mixes the four
// "input-passthrough" hidden units with LQR gains (matching
// `build_neural_policy_weights`). The remaining 12 hidden units stay
// near zero (their `W_g` row is zero) so they don't pollute the
// output.
//
// Result: the cart-pole balances with the recurrent gates
// instrumented but functionally inert; the `h` / `c` buffers do
// evolve across substeps (visible in the right-rail readout) so the
// demo does prove the persistence path. A "real" trained recurrent
// policy would need offline training + weight import — out of scope
// for v1.
fn neural_lstm_lqr_weights() -> LstmWeights<'static> {
    use std::sync::OnceLock;
    static W: OnceLock<NeuralLstmWeights> = OnceLock::new();
    let weights = W.get_or_init(build_neural_lstm_weights);
    LstmWeights {
        w_i: &weights.w_i,
        u_i: &weights.u_i,
        b_i: &weights.b_i,
        w_f: &weights.w_f,
        u_f: &weights.u_f,
        b_f: &weights.b_f,
        w_g: &weights.w_g,
        u_g: &weights.u_g,
        b_g: &weights.b_g,
        w_o: &weights.w_o,
        u_o: &weights.u_o,
        b_o: &weights.b_o,
        w_y: &weights.w_y,
        b_y: &weights.b_y,
    }
}

struct NeuralLstmWeights {
    w_i: Vec<f64>,
    u_i: Vec<f64>,
    b_i: Vec<f64>,
    w_f: Vec<f64>,
    u_f: Vec<f64>,
    b_f: Vec<f64>,
    w_g: Vec<f64>,
    u_g: Vec<f64>,
    b_g: Vec<f64>,
    w_o: Vec<f64>,
    u_o: Vec<f64>,
    b_o: Vec<f64>,
    w_y: Vec<f64>,
    b_y: Vec<f64>,
}

// The weight fill uses the explicit `row * stride + col` row-major idiom
// so every entry reads as a (row, col) coordinate.
#[allow(clippy::erasing_op, clippy::identity_op)]
fn build_neural_lstm_weights() -> NeuralLstmWeights {
    let in_dim: usize = 4;
    let hid_dim: usize = LSTM_HID_DIM; // 16
    let out_dim: usize = 1;

    // All gate input / recurrent weights start at zero.
    let zero_w = || vec![0.0f64; hid_dim * in_dim];
    let zero_u = || vec![0.0f64; hid_dim * hid_dim];
    let zero_b = || vec![0.0f64; hid_dim];

    let mut w_i = zero_w();
    let u_i = zero_u();
    let mut b_i = zero_b();
    let mut w_f = zero_w();
    let u_f = zero_u();
    let mut b_f = zero_b();
    let mut w_g = zero_w();
    let u_g = zero_u();
    let b_g = zero_b();
    let mut w_o = zero_w();
    let u_o = zero_u();
    let mut b_o = zero_b();

    // Saturate gates per the doc-block above: i≈1, f≈0, o≈1.
    for i in 0..hid_dim {
        b_i[i] = 8.0;
        b_f[i] = -8.0;
        b_o[i] = 8.0;
    }
    // Layer-1 input passthrough on the first 4 hidden units. We
    // intentionally route the input via the candidate gate (g) — that
    // way `c_t = i_t ⊙ g_t ≈ g_t = tanh(W_g · x_t)`. The `w_i` /
    // `w_f` / `w_o` weights stay zero so the saturated bias terms
    // dominate the gate values.
    for i in 0..in_dim {
        w_g[i * in_dim + i] = 1.0;
        // Defensive: also zero w_i / w_f / w_o for these rows
        // (already zero by construction, but explicit).
        w_i[i * in_dim + i] = 0.0;
        w_f[i * in_dim + i] = 0.0;
        w_o[i * in_dim + i] = 0.0;
    }

    // Linear projection. Same per-channel LQR gains as the MLP demo.
    let mut w_y = vec![0.0f64; out_dim * hid_dim];
    let b_y = vec![0.0f64; out_dim];
    let k_x = -1.0;
    let k_v = -2.5;
    let k_t = -55.0;
    let k_w = -10.0;
    w_y[0 * hid_dim + 0] = k_x;
    w_y[0 * hid_dim + 1] = k_v;
    w_y[0 * hid_dim + 2] = k_t;
    w_y[0 * hid_dim + 3] = k_w;

    NeuralLstmWeights {
        w_i,
        u_i,
        b_i,
        w_f,
        u_f,
        b_f,
        w_g,
        u_g,
        b_g,
        w_o,
        u_o,
        b_o,
        w_y,
        b_y,
    }
}

// Convert a millisecond delay into a substep count. The viewer
// runs `SUBSTEPS_PER_FRAME × FIXED_DT` per visible frame, so one
// "actuator substep" is `FIXED_DT` seconds long.
fn ms_to_substeps(ms: f32) -> u32 {
    let dt_ms = (FIXED_DT * 1000.0) as f32;
    if dt_ms <= 0.0 {
        return 0;
    }
    (ms / dt_ms).round().max(0.0) as u32
}

// Quadrotor demo — PD altitude + attitude control. Reads the current
// frame body pose + velocity, computes per-rotor thrusts, writes
// world-frame thrust forces into `state.body_f` on each rotor body.
// Featherstone GPU consumes them via the bias term.
// Drive the kinematic puller body + the puller-pinned cable particle
// in `Scene::Cable` along a circular orbit. Also keeps the active
// VBD solver's `vbd_mode` / `contact_stiffness` / `iterations` in

fn neural_lstm_pane(ui: &mut egui::Ui, state: &mut NeuralLstmExampleState) {
    ui.add_space(2.0);
    readout(ui, "Net", &format!("LSTM 4 → {} → 1", LSTM_HID_DIM));
    readout(ui, "Cell", "single-cell, gates: i / f / g / o");
    readout(ui, "Backend", "GPU (single-workgroup forward)");
    ui.separator();

    ui.label("Inputs (scaled, post-normalisation)");
    readout(
        ui,
        "  · cart_x'",
        &format!("{:>+6.3}", state.last_inputs[0]),
    );
    readout(
        ui,
        "  · cart_v'",
        &format!("{:>+6.3}", state.last_inputs[1]),
    );
    readout(
        ui,
        "  · theta",
        &format!("{:>+6.3} rad", state.last_inputs[2]),
    );
    readout(
        ui,
        "  · theta_d'",
        &format!("{:>+6.3}", state.last_inputs[3]),
    );
    ui.separator();

    ui.label("Hidden state h_t (live, persists across substeps)");
    // Render as a compact 4×4 grid so the pane stays readable.
    egui::Grid::new("lstm_hidden_state_grid")
        .num_columns(4)
        .spacing([8.0, 2.0])
        .show(ui, |ui| {
            for i in 0..LSTM_HID_DIM {
                ui.label(format!("{:>+6.3}", state.last_hidden[i]));
                if (i + 1) % 4 == 0 {
                    ui.end_row();
                }
            }
        });
    ui.separator();

    ui.label("Outputs");
    readout(
        ui,
        "  · LSTM raw",
        &format!("{:>+6.2} N", state.last_output),
    );
    readout(
        ui,
        "  · post-clamp",
        &format!(
            "{:>+6.2} N (cap ±{:.0})",
            state.last_delivered, state.max_force
        ),
    );
    ui.separator();

    ui.checkbox(&mut state.auto_perturb, "Periodic pole kick (every 3 s)");
    if !state.auto_perturb {
        ui.small("(without perturbations the system parks at upright)");
    }
    ui.separator();

    ui.small(
        "Stage-31 LSTM controller. Recurrent gates are saturated (i≈1, \
         f≈0, o≈1) so the cell behaves as `c_t = tanh(W_g·x_t)` — this \
         keeps the v1 demo visibly functional (cart-pole balance) while \
         exercising the full single-cell forward pass + persistent \
         hidden / cell state on-device. The `h_t` readout above does \
         evolve every substep, proving the hidden-state buffer survives \
         across pipeline ticks. A genuinely-trained recurrent policy \
         would need offline training + weight import.",
    );
}


