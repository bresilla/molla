// Stage-18.1 neural-policy demo runtime state. Owns the composable
// [`Actuator`] (built lazily on scene entry) wrapping a
// [`ControllerNeuralMlp`] balance policy + a [`ClampingMaxEffort`]
// stage. Per-frame: gathers `[cart_x, cart_xdot, theta, theta_dot]`
// from `joint_q / joint_qd`, normalises, pushes into the MLP via
// `set_inputs`, the pipeline scatters the single output into
// `joint_force[0]` (cart prismatic).
#[derive(Resource)]
struct NeuralPolicyExampleState {
    actuator: Option<Actuator>,
    /// Most recent input vector (4 floats), for the readout pane.
    last_inputs: [f32; 4],
    /// Most recent network output (1 float — cart force command).
    last_output: f32,
    /// Most recent delivered force after clamping. Diagnostic only.
    last_delivered: f32,
    /// User-toggleable: when true, every couple of seconds we add a
    /// small rotational kick to the pole so the demo is visibly
    /// dynamic instead of asymptoting to perfect upright forever.
    auto_perturb: bool,
    /// Time since last perturb, seconds.
    perturb_timer: f32,
    /// Effort cap (N) for the cart force.
    max_force: f32,
}

impl Default for NeuralPolicyExampleState {
    fn default() -> Self {
        Self {
            actuator: None,
            last_inputs: [0.0; 4],
            last_output: 0.0,
            last_delivered: 0.0,
            auto_perturb: true,
            perturb_timer: 0.0,
            max_force: 25.0,
        }
    }
}

//     • prismatic joint (parent=NONE) along world +X, joint_q[0] = x
//   • body 1 — pole (capsule)
//     • revolute joint (parent=cart) about Z, joint_q[1] = θ
//
// Joint frame for the cart's prismatic: cart spawns at world origin,
// height CART_RAIL_Y; the pole's revolute pivots at the top of the
// cart (where the pole is hinged).
const CART_RAIL_Y: f64 = 1.0;
const CART_HALF_X: f64 = 0.15;
const CART_HALF_Y: f64 = 0.05;
const CART_HALF_Z: f64 = 0.10;
const CART_MASS: f64 = 1.0;
const POLE_HALF_LEN: f64 = 0.5;
const POLE_RADIUS: f64 = 0.03;
const POLE_MASS: f64 = 0.1;

fn build_neural_policy_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    // Light damping so the system doesn't asymptote to zero on its
    // own — the policy has to actually do work.
    b.set_angular_damping(0.02);

    b.begin_articulation();

    // ── Cart (prismatic to world along +X) ───────────────────────────
    let cart_inertia = box_inertia_xyz(CART_MASS, CART_HALF_X, CART_HALF_Y, CART_HALF_Z);
    let cart = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: CART_MASS,
                com: MVec3::ZERO,
                inertia: cart_inertia,
            })
            .with_position(MVec3::new(0.0, CART_RAIL_Y, 0.0)),
    );
    b.add_shape_box(
        cart,
        MVec3::new(CART_HALF_X, CART_HALF_Y, CART_HALF_Z),
        mp::Transform::IDENTITY,
    )
    .expect("cart shape");

    let x_p_cart = mp::Transform::from_translation(MVec3::new(0.0, CART_RAIL_Y, 0.0));
    let x_c_cart = mp::Transform::IDENTITY;
    b.add_joint_prismatic(mp::BodyId::NONE, cart, x_p_cart, x_c_cart, MVec3::X)
        .expect("cart prismatic joint");

    // ── Pole (revolute to cart, axis +Z) ─────────────────────────────
    // Pole rest pose at θ=0: hangs straight DOWN from the cart's top
    // centre. The joint connects the cart-top frame to the pole's +Y
    // end (`x_c_pole` offset below), so the capsule dangles below the
    // pivot at rest — θ=0 is the stable equilibrium, not an inverted
    // one. The controller damps/recenters about this rest pose.
    let pole_inertia = capsule_inertia(POLE_MASS, POLE_HALF_LEN, POLE_RADIUS);
    let pole = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: POLE_MASS,
                com: MVec3::ZERO,
                inertia: pole_inertia,
            })
            // Rest pose visualisation only — Featherstone's FK pass
            // overwrites this on the first step.
            .with_position(MVec3::new(0.0, CART_RAIL_Y - POLE_HALF_LEN, 0.0)),
    );
    b.add_shape_capsule(pole, POLE_HALF_LEN, POLE_RADIUS, mp::Transform::IDENTITY)
        .expect("pole shape");

    // Joint frame in cart-local space: top of the cart (+Y by
    // CART_HALF_Y), no rotation needed since the pole's body-local +Y
    // is already its long axis. The pole hangs at θ=0 (down).
    let x_p_pole = mp::Transform::from_translation(MVec3::new(0.0, CART_HALF_Y, 0.0));
    let x_c_pole = mp::Transform::from_translation(MVec3::new(0.0, POLE_HALF_LEN, 0.0));
    b.add_joint_revolute(cart, pole, x_p_pole, x_c_pole, MVec3::Z)
        .expect("pole revolute joint");

    let visuals = vec![
        BodyVisual::Box(Vec3::new(
            CART_HALF_X as f32,
            CART_HALF_Y as f32,
            CART_HALF_Z as f32,
        )),
        BodyVisual::Capsule {
            half_h: POLE_HALF_LEN as f32,
            radius: POLE_RADIUS as f32,
        },
    ];

    (b.build().expect("neural policy model"), visuals)
}

// Inertia of a uniform box about its centre. Axes: x/y/z extents are
// 2·(hx, hy, hz). Returned as a diagonal `Mat3` per
// [`mp::SpatialInertia::inertia`]. Distinct from the existing
// `box_inertia(mass, half)` helper, which assumes a cube.
fn box_inertia_xyz(mass: f64, hx: f64, hy: f64, hz: f64) -> molla::math::Mat3 {
    let s = mass / 3.0;
    let ixx = s * (hy * hy + hz * hz);
    let iyy = s * (hx * hx + hz * hz);
    let izz = s * (hx * hx + hy * hy);
    molla::math::Mat3::from_diagonal(MVec3::new(ixx, iyy, izz))
}

fn update_neural_policy_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<NeuralPolicyExampleState>,
) {
    if sim.scene != Scene::NeuralPolicy {
        if state.actuator.is_some() {
            state.actuator = None;
        }
        return;
    }

    // Lazy-init the pipeline + upload weights on first frame in scene.
    if state.actuator.is_none() {
        let cfg = ControllerNeuralMlpConfig {
            input_dim: 4,
            hidden_dim: 32,
            output_dim: 1,
        };
        let dof_indices = vec![0u32]; // cart prismatic = joint_force[0]
        let mlp_res = ControllerNeuralMlp::new(&sim.engine, cfg, &dof_indices);
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
        match (mlp_res, delay_res, clamp_res) {
            (Ok(mut mlp), Ok(delay), Ok(clamp)) => {
                let weights = neural_policy_lqr_weights();
                let device = sim.engine.device().unwrap().clone();
                let queue = sim.engine.queue().unwrap().clone();
                if let Err(e) = mlp.set_weights(&device, &queue, &weights) {
                    warn!("neural policy: set_weights failed: {e}");
                    return;
                }
                state.actuator = Some(Actuator::new(
                    delay,
                    ControllerStage::NeuralMlp(mlp),
                    ClampingStage::MaxEffort(clamp),
                ));
                info!("neural policy: MLP online — 4→32→32→1 LQR-style balance net");
            }
            (m, d, c) => {
                if let Err(e) = m {
                    warn!("neural policy: mlp init failed: {e}");
                }
                if let Err(e) = d {
                    warn!("neural policy: delay init failed: {e}");
                }
                if let Err(e) = c {
                    warn!("neural policy: clamp init failed: {e}");
                }
                return;
            }
        }
    }

    // Pull joint state host-side (already downloaded each substep
    // by `step_simulation`).
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

    // Periodic perturb: every 3 s, pulse a small angular velocity into
    // the pole so the user can see the policy recover. Implemented by
    // writing directly into joint_qd[1] on the host then re-uploading.
    state.perturb_timer += time.delta_secs();
    if state.auto_perturb && state.perturb_timer > 3.0 {
        state.perturb_timer = 0.0;
        // Alternate sign so the kicks balance over time.
        let sign = if (theta + 0.001).is_sign_positive() {
            -1.0
        } else {
            1.0
        };
        let queue = sim.engine.queue().expect("queue").clone();
        if let Ok(qd) = sim.state_a.joint_qd.host_mut() {
            if qd.len() >= 2 {
                qd[1] += (sign * 1.5) as f64;
            }
        }
        let _ = sim.state_a.joint_qd.upload(&queue);
    }

    // Observation: scaled to keep tanh in its near-linear regime.
    // Per-component scales picked so a "reasonable" excursion lands
    // |x| ≈ 0.5 — the LQR-equivalent gains compensate via w3.
    let inputs: [f32; 4] = [
        cart_x * 0.5,  // ±2 m → ±1
        cart_xd * 0.2, // ±5 m/s → ±1
        theta,         // already ±π/2 useful range
        theta_d * 0.2, // ±5 rad/s → ±1
    ];
    state.last_inputs = inputs;

    let inputs_f64: Vec<f64> = inputs.iter().map(|v| *v as f64).collect();

    // Borrow split so we can mutate both `control` and `state.actuator`.
    let Sim {
        ref engine,
        ref state_a,
        ref mut control,
        ..
    } = *sim;
    if let Some(act) = state.actuator.as_mut() {
        let queue = engine.queue().expect("queue").clone();
        // Push the observation into the MLP. We need to dive into the
        // controller stage to call `set_inputs` — actuator.step's
        // contract assumes the inputs are already on-device.
        match act.controller_mut() {
            ControllerStage::NeuralMlp(mlp) => {
                if let Err(e) = mlp.set_inputs(&queue, &inputs_f64) {
                    warn!("neural policy: set_inputs failed: {e}");
                    return;
                }
            }
            _ => unreachable!("neural policy scene must use NeuralMlp"),
        }
        if let Err(e) = act.step(engine, state_a, control, FIXED_DT) {
            warn!("neural policy: pipeline step failed: {e}");
            return;
        }

        // Read back the network output + post-clamp force for the UI.
        let device = engine.device().expect("device").clone();
        match act.controller_mut() {
            ControllerStage::NeuralMlp(mlp) => {
                if let Ok(outs) = mlp.download_outputs(&device, &queue) {
                    if !outs.is_empty() {
                        state.last_output = outs[0] as f32;
                    }
                }
            }
            _ => {}
        }
        if control.joint_force.download(&device, &queue).is_ok() {
            if let Ok(jf) = control.joint_force.host() {
                if !jf.is_empty() {
                    state.last_delivered = jf[0] as f32;
                }
            }
        }
    }
}

// Hand-rolled MLP weights that express an LQR-style balance policy
// on the cart-pole. The network maps a 4-element observation
//   [x_scaled, xdot_scaled, theta, theta_d_scaled]
// through two near-identity tanh hidden layers to a single output:
//
//   F = -[K_x · x_scaled + K_v · xdot_scaled
//        + K_θ · theta + K_ω · theta_d_scaled]
//
// In the near-linear regime (|input| ≪ 1) `tanh(z) ≈ z`, so the
// network collapses to the linear policy above. Past saturation the
// tanh caps each channel — desirable behaviour for a balance net.
fn neural_policy_lqr_weights() -> MlpWeights<'static> {
    use std::sync::OnceLock;

    static W: OnceLock<NeuralPolicyWeights> = OnceLock::new();
    let weights = W.get_or_init(build_neural_policy_weights);
    MlpWeights {
        w1: &weights.w1,
        b1: &weights.b1,
        w2: &weights.w2,
        b2: &weights.b2,
        w3: &weights.w3,
        b3: &weights.b3,
    }
}

struct NeuralPolicyWeights {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    w3: Vec<f64>,
    b3: Vec<f64>,
}

// The weight fill uses the explicit `row * stride + col` row-major idiom
// so every entry reads as a (row, col) coordinate.
#[allow(clippy::erasing_op, clippy::identity_op)]
fn build_neural_policy_weights() -> NeuralPolicyWeights {
    // Architecture: 4 → 32 → 32 → 1.
    // Strategy: layer 1 is a slightly-noisy identity passthrough on
    // the first 4 hidden units; remaining 28 units stay zero so they
    // don't perturb the output. Layer 2 is identity. Layer 3 is the
    // LQR linear combination on the first 4 hidden units.
    let in_dim: usize = 4;
    let hid_dim: usize = 32;
    let out_dim: usize = 1;

    let mut w1 = vec![0.0f64; hid_dim * in_dim];
    let b1 = vec![0.0f64; hid_dim];
    let mut w2 = vec![0.0f64; hid_dim * hid_dim];
    let b2 = vec![0.0f64; hid_dim];
    let mut w3 = vec![0.0f64; out_dim * hid_dim];
    let b3 = vec![0.0f64; out_dim];

    // Layer 1: identity passthrough on units 0..4.
    // w1[i, k] for i ∈ [0, in_dim).
    for i in 0..in_dim {
        w1[i * in_dim + i] = 1.0;
    }
    // Layer 2: identity on the 32 hidden units (so the signal carries
    // through unchanged).
    for i in 0..hid_dim {
        w2[i * hid_dim + i] = 1.0;
    }

    // Layer 3: LQR-style gains. We want
    //   F = -k_x·x - k_v·xdot - k_θ·theta - k_ω·theta_d
    // The network sees scaled inputs (see `update_neural_policy_example`
    // for the scale factors). Tuning rationale:
    //   • theta dominates — push hard on pole error.
    //   • theta_d damps oscillation, but lower than k_θ.
    //   • cart_x weak — only reins in long-term drift.
    //   • cart_xdot moderate damping.
    //
    // Per-channel (via the scale factors used in `update_*`):
    //   k_x_eff   = k_x  / 0.5
    //   k_v_eff   = k_v  / 0.2
    //   k_θ_eff   = k_θ
    //   k_ω_eff   = k_ω  / 0.2
    //
    // Picked so theta-only error of 0.2 rad → ~10 N (well under cap).
    let k_x = -1.0;
    let k_v = -2.5;
    let k_t = -55.0;
    let k_w = -10.0;
    w3[0 * hid_dim + 0] = k_x; // x_scaled
    w3[0 * hid_dim + 1] = k_v; // xdot_scaled
    w3[0 * hid_dim + 2] = k_t; // theta
    w3[0 * hid_dim + 3] = k_w; // theta_d_scaled

    NeuralPolicyWeights {
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
    }
}

// ── Stage-31 neural-LSTM demo update ───────────────────────────────
//
// Mirrors `update_neural_policy_example`, but the controller is a
// single-cell LSTM (4 → 16 → 1) instead of an MLP. The weights are
// hand-rolled to express the same LQR-style balance policy: the
// recurrent gates are tuned so the cell barely modulates the signal
// (forget ≈ 0, input ≈ 1, candidate ≈ tanh(x)), and the linear
// projection then mixes the gated state with the same per-component
// gains the MLP demo uses. This way the demo proves the forward-pass

fn neural_policy_pane(ui: &mut egui::Ui, state: &mut NeuralPolicyExampleState) {
    ui.add_space(2.0);
    readout(ui, "Net", "MLP 4 → 32 → 32 → 1");
    readout(ui, "Activation", "tanh / tanh / linear");
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

    ui.label("Outputs");
    readout(ui, "  · MLP raw", &format!("{:>+6.2} N", state.last_output));
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
        "Hard-coded LQR-style balance policy expressed as a 3-layer MLP. \
         Layer 1 + 2 are near-identity passthroughs on the first 4 \
         hidden units; layer 3 is a linear combination producing the \
         cart force. tanh saturation provides a soft envelope; the \
         downstream MaxEffort clamp adds a hard rail.",
    );
}


