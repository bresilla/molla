// Stage-18 actuator pipeline runtime state. Owns the composable
// `Actuator` (built lazily on scene entry), the slider-driven
// per-frame parameters, and the latest commanded vs delivered
// torque samples for the right-rail readout.
#[derive(Resource)]
struct ActuatorExampleState {
    actuator: Option<Actuator>,
    /// Position target [rad] driven by the slider.
    target: f32,
    /// Whether to drive the target with a sine wave instead of a
    /// fixed value (so the user can immediately see the delay
    /// stage's effect).
    sine_enabled: bool,
    sine_amplitude: f32,
    sine_frequency: f32,
    elapsed: f32,
    /// Lag in milliseconds (converted to substeps per scene config).
    delay_ms: f32,
    /// DC-motor parameters (per-frame slider state).
    stall_torque: f32,
    no_load_speed: f32,
    max_motor_effort: f32,
    /// PD gains.
    kp: f32,
    kd: f32,
    /// Latest commanded effort (before clamp) and delivered effort
    /// (after clamp). Computed host-side per frame for the readout
    /// — these are diagnostic, not authoritative.
    last_commanded: f32,
    last_delivered: f32,
    /// Reuploaded params when sliders change.
    params_dirty: bool,
}

impl Default for ActuatorExampleState {
    fn default() -> Self {
        Self {
            actuator: None,
            target: 0.5,
            sine_enabled: true,
            sine_amplitude: 1.0,
            sine_frequency: 0.4,
            elapsed: 0.0,
            delay_ms: 12.0,
            stall_torque: 8.0,
            no_load_speed: 18.0,
            max_motor_effort: 20.0,
            kp: 60.0,
            kd: 4.0,
            last_commanded: 0.0,
            last_delivered: 0.0,
            params_dirty: false,
        }
    }
}

// --- Stage-18 composable actuator pipeline scene ------------------
//
// Single revolute joint mounted on a stationary stub. The joint is
// driven by `Actuator { Delay → ControllerPipelinePd → ClampingDcMotor }`
// every substep — the Delay pushes / reads from a circular buffer
// of depth 32, the PD writes raw effort, the DC-motor clamp turns
// it into a torque-vs-speed envelope.
//
// Visuals:
//   • body 0 — short stub of length 0.3 m anchored at +1.5 m (no
//     parent ⇒ joint to NONE); rotates around world +Z so the swing
//     plays out in the camera-facing X/Y plane.
const ACTUATOR_BASE_HEIGHT: f64 = 1.5;
const ACTUATOR_LINK_HALF_H: f64 = 0.4;
const ACTUATOR_LINK_RADIUS: f64 = 0.05;
const ACTUATOR_LINK_MASS: f64 = 0.5;

fn build_actuator_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    let inertia = capsule_inertia(
        ACTUATOR_LINK_MASS,
        ACTUATOR_LINK_HALF_H,
        ACTUATOR_LINK_RADIUS,
    );

    b.begin_articulation();

    // Anchor the link to world via a revolute joint. Joint frame in
    // parent space lifts to (0, ACTUATOR_BASE_HEIGHT, 0) and the
    // 180° flip around X makes the link hang along world -Y at q=0.
    let body = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: ACTUATOR_LINK_MASS,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(MVec3::new(
                0.0,
                ACTUATOR_BASE_HEIGHT - ACTUATOR_LINK_HALF_H,
                0.0,
            )),
    );
    b.add_shape_capsule(
        body,
        ACTUATOR_LINK_HALF_H,
        ACTUATOR_LINK_RADIUS,
        mp::Transform::IDENTITY,
    )
    .expect("actuator link shape");

    let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
    let x_p = mp::Transform::new(MVec3::new(0.0, ACTUATOR_BASE_HEIGHT, 0.0), flip);
    let x_c = mp::Transform::from_translation(MVec3::new(0.0, -ACTUATOR_LINK_HALF_H, 0.0));
    b.add_joint_revolute(mp::BodyId::NONE, body, x_p, x_c, MVec3::Z)
        .expect("actuator joint");

    let visuals = vec![BodyVisual::Capsule {
        half_h: ACTUATOR_LINK_HALF_H as f32,
        radius: ACTUATOR_LINK_RADIUS as f32,
    }];

    (b.build().expect("actuator model"), visuals)
}

// --- Stage-18.1 neural-policy cart-pole scene ---------------------
//
// Cart slides along world +X via a prismatic joint. Pole hangs above
// the cart and rotates around world +Z via a revolute joint. The
// pre-trained MLP balance policy keeps the pole near upright (θ=0)
// while damping cart drift. With a small initial perturbation in θ
// the system swings back to upright in a few seconds.
//
// Topology:

fn update_actuator_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<ActuatorExampleState>,
) {
    if sim.scene != Scene::Actuator {
        if state.actuator.is_some() {
            state.actuator = None;
        }
        return;
    }

    // Lazy-init the pipeline on first frame in this scene.
    if state.actuator.is_none() {
        let dof = 0u32;
        let delay_steps = ms_to_substeps(state.delay_ms);
        let max_delay = 64u32; // headroom; user-facing slider caps at ~30 ms = 18 substeps.
        let delay_res = Delay::new(&sim.engine, &[DelayConfig { dof, delay_steps }], max_delay);
        let pd_res = ControllerPipelinePd::new(
            &sim.engine,
            &[ControllerPipelinePdConfig {
                dof,
                kp: state.kp as f64,
                kd: state.kd as f64,
            }],
        );
        let clamp_res = ClampingDcMotor::new(
            &sim.engine,
            &[ClampingDcMotorConfig {
                dof,
                stall_torque: state.stall_torque as f64,
                no_load_speed: state.no_load_speed as f64,
                max_motor_effort: state.max_motor_effort as f64,
            }],
        );
        match (delay_res, pd_res, clamp_res) {
            (Ok(delay), Ok(pd), Ok(clamp)) => {
                let act = Actuator::new(
                    delay,
                    ControllerStage::Pd(pd),
                    ClampingStage::DcMotor(clamp),
                );
                info!(
                    "actuator: pipeline online — delay {} substeps, kp={}, kd={}, stall={} N·m, no_load={} rad/s",
                    delay_steps, state.kp, state.kd, state.stall_torque, state.no_load_speed,
                );
                state.actuator = Some(act);
            }
            (d, p, c) => {
                if let Err(e) = d {
                    warn!("actuator: delay init failed: {e}");
                }
                if let Err(e) = p {
                    warn!("actuator: pd init failed: {e}");
                }
                if let Err(e) = c {
                    warn!("actuator: clamp init failed: {e}");
                }
                return;
            }
        }
    }

    state.elapsed += time.delta_secs();

    // Apply slider edits by uploading new params to the GPU buffers.
    if state.params_dirty {
        // Copy slider values out of `state` so the controller-stage
        // closures below don't conflict with the `state.actuator`
        // mutable borrow.
        let delay_steps = ms_to_substeps(state.delay_ms);
        let kp = state.kp as f64;
        let kd = state.kd as f64;
        let stall = state.stall_torque as f64;
        let no_load = state.no_load_speed as f64;
        let max_motor = state.max_motor_effort as f64;
        if let Some(act) = state.actuator.as_mut() {
            let n = act.actuator_count() as usize;
            if let Err(e) = act
                .delay_mut()
                .set_delay_steps(&sim.engine, &vec![delay_steps; n])
            {
                warn!("actuator: set_delay_steps failed: {e}");
            }
            match act.controller_mut() {
                ControllerStage::Pd(pd) => {
                    if let Err(e) = pd.set_gains(&sim.engine, &vec![kp; n], &vec![kd; n]) {
                        warn!("actuator: set_gains failed: {e}");
                    }
                }
                // The Stage-18 actuator pane only constructs a PD
                // controller; PID / MLP / LSTM variants live in their
                // own dedicated demo scenes (NeuralPolicy / NeuralLstm).
                ControllerStage::Pid(_)
                | ControllerStage::NeuralMlp(_)
                | ControllerStage::Lstm(_) => {}
            }
            if let ClampingStage::DcMotor(clamp) = act.clamping_mut()
                && let Err(e) = clamp.set_params(
                    &sim.engine,
                    &vec![stall; n],
                    &vec![no_load; n],
                    &vec![max_motor; n],
                ) {
                    warn!("actuator: set_params failed: {e}");
                }
        }
        state.params_dirty = false;
    }

    // Decide the target for this frame.
    let target = if state.sine_enabled {
        let t = state.elapsed * state.sine_frequency * std::f32::consts::TAU;
        state.sine_amplitude * t.sin()
    } else {
        state.target
    };
    if state.sine_enabled {
        state.target = target;
    }

    // Compute commanded torque host-side (diagnostic): same formula
    // the pipeline uses, but without the DC-motor clamp.
    let q = sim
        .state_a
        .joint_q
        .host()
        .ok()
        .and_then(|s| s.first().copied())
        .unwrap_or(0.0) as f32;
    let qd = sim
        .state_a
        .joint_qd
        .host()
        .ok()
        .and_then(|s| s.first().copied())
        .unwrap_or(0.0) as f32;
    let commanded = state.kp * (target - q) + state.kd * (0.0 - qd);
    state.last_commanded = commanded;

    // Push the target into Control.joint_target_pos on the device.
    {
        let queue = sim.engine.queue().expect("queue").clone();
        let target_pos = sim
            .control
            .joint_target_pos
            .host_mut()
            .expect("joint_target_pos");
        if !target_pos.is_empty() {
            target_pos[0] = target as f64;
        }
        let target_vel = sim
            .control
            .joint_target_vel
            .host_mut()
            .expect("joint_target_vel");
        for v in target_vel.iter_mut() {
            *v = 0.0;
        }
        sim.control
            .joint_target_pos
            .upload(&queue)
            .expect("target_pos upload");
        sim.control
            .joint_target_vel
            .upload(&queue)
            .expect("target_vel upload");
    }

    // Run the actuator pipeline.
    let Sim {
        ref engine,
        ref state_a,
        ref mut control,
        ..
    } = *sim;
    if let Some(act) = state.actuator.as_mut() {
        if let Err(e) = act.step(engine, state_a, control, FIXED_DT) {
            warn!("actuator: pipeline step failed: {e}");
        } else {
            // Read the delivered torque back off the GPU for the
            // readout panel. This is a small download (1 element)
            // so the cost is negligible.
            let device = engine.device().expect("device").clone();
            let queue = engine.queue().expect("queue").clone();
            if control.joint_force.download(&device, &queue).is_ok()
                && let Ok(jf) = control.joint_force.host()
                    && !jf.is_empty() {
                        state.last_delivered = jf[0] as f32;
                    }
        }
    }
}

// ── Stage-18.1 neural-policy demo update ───────────────────────────
//
// Per visible frame:
//   1. Lazy-init the [`Actuator`] containing
//      `Delay(0) → ControllerNeuralMlp → ClampingMaxEffort` once we
//      enter the scene.
//   2. Read `joint_q[0..2]` (cart_x, theta), `joint_qd[0..2]`
//      (cart_xdot, theta_dot) host-side.
//   3. Build the 4-element observation, normalise, push through the
//      MLP via `set_inputs`, then run `act.step` — the GPU dispatches
//      forward + scatter + clamp.
//   4. Periodically inject a small angular perturbation so the
//      visualisation stays dynamic (otherwise the policy parks the
//      pole at exactly upright and nothing moves).
//
// The MLP weights are **hand-rolled** to express a linear LQR-style
// state-feedback law `F = -K · [x, x_dot, theta, theta_dot]` through
// the network. With small inputs, the tanh non-linearities act as
// near-identity, so the MLP collapses to a linear policy. Past a few
// rad of pole error, tanh saturates and the output caps; the

fn actuator_pane(ui: &mut egui::Ui, state: &mut ActuatorExampleState) {
    ui.add_space(2.0);
    readout(ui, "Pipeline", "Delay → PD → DC-Motor");
    let delay_substeps = ms_to_substeps(state.delay_ms);
    readout(
        ui,
        "Delay",
        &format!("{:.1} ms ({} substeps)", state.delay_ms, delay_substeps),
    );
    readout(
        ui,
        "Commanded τ",
        &format!("{:>6.2} N·m", state.last_commanded),
    );
    readout(
        ui,
        "Delivered τ",
        &format!("{:>6.2} N·m", state.last_delivered),
    );
    let saturation = if state.last_commanded.abs() > 1e-3 {
        state.last_delivered / state.last_commanded
    } else {
        1.0
    };
    readout(ui, "Delivered/Cmd", &format!("{:>5.2}", saturation));
    ui.separator();

    ui.checkbox(&mut state.sine_enabled, "Sine target drive");
    if state.sine_enabled {
        ui.add(egui::Slider::new(&mut state.sine_amplitude, 0.0..=2.0).text("amplitude (rad)"));
        ui.add(egui::Slider::new(&mut state.sine_frequency, 0.0..=2.0).text("frequency (Hz)"));
        readout(ui, "Target", &format!("{:>6.3} rad", state.target));
    } else {
        let resp = ui.add(
            egui::Slider::new(
                &mut state.target,
                -std::f32::consts::PI..=std::f32::consts::PI,
            )
            .text("target (rad)"),
        );
        let _ = resp;
    }
    ui.separator();

    ui.label("Delay");
    if ui
        .add(egui::Slider::new(&mut state.delay_ms, 0.0..=30.0).text("delay (ms)"))
        .changed()
    {
        state.params_dirty = true;
    }
    ui.separator();

    ui.label("PD gains");
    if ui
        .add(egui::Slider::new(&mut state.kp, 0.0..=200.0).text("kp"))
        .changed()
    {
        state.params_dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.kd, 0.0..=20.0).text("kd"))
        .changed()
    {
        state.params_dirty = true;
    }
    ui.separator();

    ui.label("DC motor");
    if ui
        .add(egui::Slider::new(&mut state.stall_torque, 0.0..=30.0).text("stall τ (N·m)"))
        .changed()
    {
        state.params_dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.no_load_speed, 0.5..=40.0).text("no-load ω (rad/s)"))
        .changed()
    {
        state.params_dirty = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.max_motor_effort, 0.0..=50.0).text("max τ cap (N·m)"))
        .changed()
    {
        state.params_dirty = true;
    }

    ui.add_space(4.0);
    ui.small(
        "Pipeline = Delay (ring buffer of length ⌈delay_ms/sub_dt⌉) → \
         pipeline-PD (no cap) → DC-motor envelope. The clamp's \
         effort ceiling drops linearly with |qd| toward 0 at no-load \
         speed. Watch 'Delivered τ / Commanded τ' shrink as the link \
         spins up.",
    );
}


