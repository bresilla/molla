// Arm example runtime state. Owns the GPU PD controller and the
// per-joint target inputs (slider value, optional auto-sine).
#[derive(Resource)]
struct ArmExampleState {
    controller: Option<ControllerPd>,
    /// Per-joint target position [rad]. Driven by the right-rail
    /// sliders or by the auto-sine generator below.
    targets: [f32; ARM_LINKS],
    /// Whether we've logged the first successful PD dispatch — saves
    /// the user from "is this thing even running?" guesswork.
    logged_init: bool,
    /// Auto-sine drive: when enabled, each joint follows
    /// `amplitude · sin(2π·frequency·t + phase[i])`. Disabled by
    /// default so the user can drag the sliders manually.
    sine_enabled: bool,
    sine_amplitude: f32,
    sine_frequency: f32,
    /// Wall-clock since the scene started, for the sine generator.
    elapsed: f32,
    /// Stage-32 joint-mimic toggle: when on, the arm is rebuilt with
    /// an additional auxiliary joint that mimics joint 0 via the
    /// linear-identity polynomial `[0, 1, 0, 0, 0]`. The viewer
    /// renders both joints' angles in the right rail so the user can
    /// confirm the mimic stays tight under the same PD drive.
    mimic_enabled: bool,
    /// Set to `true` whenever `mimic_enabled` is toggled; consumed
    /// by `apply_scene_switch` to know it must rebuild the arm
    /// model. The toggle handler in `arm_pane` flips this.
    pub mimic_dirty: bool,
}

impl Default for ArmExampleState {
    fn default() -> Self {
        Self {
            controller: None,
            // 6-element initial target — the rest get clobbered by
            // the auto-sine pass anyway, this is just for the manual
            // slider mode.
            targets: [0.0; ARM_LINKS],
            sine_enabled: true,
            sine_amplitude: 0.5,
            sine_frequency: 0.3,
            elapsed: 0.0,
            logged_init: false,
            mimic_enabled: false,
            mimic_dirty: false,
        }
    }
}

const ARM_LINKS: usize = 6;
// Length of each arm link (m).
const ARM_LINK_LEN: f32 = 0.6;
const ARM_LINK_RADIUS: f32 = 0.06;
const ARM_BASE_HEIGHT: f32 = 5.0;
// Per-link mass [kg]. Mass × kp dictate how snappy the arm looks;
// too low + high-kp diverges, too high + low-kp barely moves under
// the chained-inertia load at the shoulder. 0.5 kg with kp=120 is
// stable at 240 Hz substeps and produces visible swings.
const ARM_LINK_MASS: f64 = 0.5;
// PD gains. `max_effort` caps a step's torque so a far-from-target
// joint can't accelerate to 10⁵ rad/s² in a single substep.
const ARM_KP: f64 = 120.0;
const ARM_KD: f64 = 12.0;
const ARM_MAX_EFFORT: f64 = 80.0;

// 6-link revolute arm anchored at `(0, ARM_BASE_HEIGHT, 0)`. The
// shoulder joint rotates around `+Z` and every elbow rotates around
// `+X`, which gives the arm two visible degrees of swing — a "wave"
// in the XY plane (shoulder) plus folding in YZ (elbows). Default
// joint orientation has each link extending along the world
// `-Y` direction, so the rest pose is a hanging arm; gravity adds
// to the PD torque to make swings dramatic.
//
// When `mimic` is true (Stage 32), a 7th body + revolute joint is
// appended in a *separate* articulation alongside the arm and
// coupled to joint 0 by a `JointMimic` equality with the linear-
// identity polynomial `[0, 1, 0, 0, 0]`. The mimic joint's link
// hangs visually offset along `+X` so the user can see both joints
// move in lockstep.
#[allow(
    dead_code,
    reason = "kept as a back-compat shim — Stage 32 routed callers through `build_arm_model_with` to thread the mimic toggle"
)]
fn build_arm_model() -> (mp::Model, Vec<BodyVisual>) {
    build_arm_model_with(false)
}

// Stage-32 mimic-toggle accessor. The toggle lives in
// `ArmExampleState`; we mirror it into a static atomic so the
// `Sim::build_scene` (which has no SystemParam access) can read it
// when the scene is rebuilt. `arm_pane` is the only writer; the
// scene-switch handler reads + clears the dirty flag.
static ARM_MIMIC_FLAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn arm_mimic_flag() -> bool {
    ARM_MIMIC_FLAG.load(std::sync::atomic::Ordering::Relaxed)
}

fn set_arm_mimic_flag(v: bool) {
    ARM_MIMIC_FLAG.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn build_arm_model_with(mimic: bool) -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.1);

    let mut visuals = Vec::with_capacity(ARM_LINKS);
    let half_h = (ARM_LINK_LEN as f64) * 0.5;
    let radius = ARM_LINK_RADIUS as f64;

    let inertia = capsule_inertia(ARM_LINK_MASS, half_h, radius);

    b.begin_articulation();

    let mut prev_body = mp::BodyId::NONE;
    // Initial body_q is just a placeholder — Featherstone's Pass 4 FK
    // overwrites these on the first step.
    let mut next_pos = MVec3::new(0.0, ARM_BASE_HEIGHT as f64 - half_h, 0.0);

    for i in 0..ARM_LINKS {
        let body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: ARM_LINK_MASS,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(next_pos),
        );
        b.add_shape_capsule(body, half_h, radius, mp::Transform::IDENTITY)
            .expect("arm link shape");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });

        // Joint anchor in PARENT frame: world base for the shoulder,
        // the +Y tip of the previous link for elbows. For the
        // shoulder we also rotate the joint frame so axis +Z
        // (camera-facing) gives a rest pose where body 0 hangs along
        // world -Y. For elbows the parent's frame already has +Y
        // pointing along the chain so no extra rotation is needed.
        let x_p = if i == 0 {
            // 180° around X flips the joint-local +Y to world -Y, so
            // a revolute joint with axis +Z + q=0 leaves the link
            // hanging straight down from the shoulder.
            let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
            mp::Transform::new(MVec3::new(0.0, ARM_BASE_HEIGHT as f64, 0.0), flip)
        } else {
            mp::Transform::from_translation(MVec3::new(0.0, half_h, 0.0))
        };
        let x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));

        // Shoulder swings in the world XY plane (axis +Z); the elbows
        // bend in YZ (axis +X). Mixing axes gives the arm two
        // independent "directions" of motion, which reads better
        // than every joint sweeping the same plane.
        let axis = if i == 0 { MVec3::Z } else { MVec3::X };
        let _jid = b
            .add_joint_revolute(prev_body, body, x_p, x_c, axis)
            .expect("arm joint");

        prev_body = body;
        next_pos = next_pos - MVec3::new(0.0, ARM_LINK_LEN as f64, 0.0);
    }

    // Stage 32 — optional joint-mimic side chain. Append a single
    // capsule body in a SECOND articulation that hangs offset along
    // +X and is coupled to joint 0 (the shoulder) by a `JointMimic`
    // equality with the linear-identity polynomial. Both joints
    // remain dynamic at the Featherstone level; the XPBD equality
    // pass enforces `q_mimic = q_shoulder` each substep when the
    // XPBD path is active. With Featherstone-only solvers the
    // existing `add_mimic_constraint` linear pre-step keeps it in
    // sync (same effect, separate code path).
    if mimic {
        b.begin_articulation();
        let offset = MVec3::new(1.5, ARM_BASE_HEIGHT as f64 - half_h, 0.0);
        let mimic_body = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: ARM_LINK_MASS,
                    com: MVec3::ZERO,
                    inertia,
                })
                .with_position(offset),
        );
        b.add_shape_capsule(mimic_body, half_h, radius, mp::Transform::IDENTITY)
            .expect("mimic link shape");
        visuals.push(BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        });
        let flip = MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI);
        let x_p_m = mp::Transform::new(MVec3::new(1.5, ARM_BASE_HEIGHT as f64, 0.0), flip);
        let x_c_m = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
        let mimic_joint = b
            .add_joint_revolute(mp::BodyId::NONE, mimic_body, x_p_m, x_c_m, MVec3::Z)
            .expect("mimic joint");
        // Driver = joint 0 (the shoulder). Identity linear mimic.
        let shoulder = mp::JointId(0);
        b.add_equality_joint_mimic(shoulder, mimic_joint, [0.0, 1.0, 0.0, 0.0, 0.0])
            .expect("mimic equality");
        // Also wire the legacy linear-mimic path so the Featherstone
        // pre-step keeps the joint synced even when the XPBD
        // equality pass isn't invoked (the arm scene runs on
        // Featherstone GPU; the JointMimic projection lands when XPBD
        // is in the loop). Both paths agree on the linear-identity
        // case so the duplicate is harmless.
        b.add_mimic_constraint(mimic_joint, shoulder, 0.0, 1.0)
            .expect("mimic legacy");
    }

    (b.build().expect("arm model"), visuals)
}

fn update_arm_example(time: Res<Time>, mut sim: ResMut<Sim>, mut state: ResMut<ArmExampleState>) {
    if sim.scene != Scene::Arm {
        if state.controller.is_some() {
            state.controller = None;
        }
        return;
    }

    if state.controller.is_none() {
        // 1 actuator per joint DOF. Featherstone's joint_dof_total
        // covers all joint DOFs — for an N-revolute arm, that's N.
        let configs: Vec<ControllerPdConfig> = (0..ARM_LINKS)
            .map(|i| ControllerPdConfig {
                dof: i as u32,
                kp: ARM_KP,
                kd: ARM_KD,
                max_effort: ARM_MAX_EFFORT,
            })
            .collect();
        match ControllerPd::new(&sim.engine, &sim.model, &configs) {
            Ok(c) => {
                info!(
                    "arm: ControllerPd online — {} actuators, kp={ARM_KP}, kd={ARM_KD}, max_effort={ARM_MAX_EFFORT}",
                    configs.len(),
                );
                state.controller = Some(c);
            }
            Err(e) => {
                warn!("arm: controller init failed: {e}");
                return;
            }
        }
    }

    state.elapsed += time.delta_secs();

    // Determine target positions for this frame.
    let mut targets = state.targets;
    if state.sine_enabled {
        let t = state.elapsed * state.sine_frequency * std::f32::consts::TAU;
        for (i, target) in targets.iter_mut().enumerate() {
            // Phase-stagger each joint by 2π/N so the arm looks
            // wavey rather than flapping in unison.
            let phase = (i as f32) * std::f32::consts::TAU / (ARM_LINKS as f32);
            *target = state.sine_amplitude * (t + phase).sin();
        }
        // Reflect the auto-generated targets back into the slider
        // values so the UI stays in sync.
        state.targets = targets;
    }

    // Push the targets into Control.joint_target_pos. Targets for
    // joint i live at the start-of-DOF index for that joint —
    // joint_dof_offset[i]..[i+1]. For revolute joints that's a
    // single DOF per joint, so dof = i. We also zero target_vel
    // so the derivative term tracks the joint coming to rest at
    // the target. We do NOT touch joint_force here — the PD
    // kernel writes it on the device (and marks it device-dirty),
    // so a host-side write would conflict. The kernel does
    // `joint_force[dof] = effort` (assignment, not accumulate)
    // and our 3 actuators cover all 3 DOFs of this scene, so no
    // stale values leak through.
    {
        let queue = sim.engine.queue().expect("queue").clone();
        let target_pos = sim
            .control
            .joint_target_pos
            .host_mut()
            .expect("joint_target_pos");
        for (i, t) in targets.iter().enumerate() {
            if i < target_pos.len() {
                target_pos[i] = *t as f64;
            }
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

    // Run the PD controller — writes effort into Control.joint_force.
    let Sim {
        ref engine,
        ref state_a,
        ref mut control,
        ..
    } = *sim;
    if let Some(controller) = state.controller.as_mut() {
        match controller.step(engine, state_a, control) {
            Ok(()) => {
                if !state.logged_init {
                    info!("arm: first PD dispatch OK");
                    state.logged_init = true;
                }
            }
            Err(e) => warn!("arm: controller step failed: {e}"),
        }
    }
}

// Stage-18 actuator pipeline example — wires the composable
// `Actuator { Delay → ControllerPipelinePd → ClampingDcMotor }` to
// drive the single revolute joint. Reads slider state per frame
// and pushes the target into `Control.joint_target_pos`. No-op for

fn arm_pane(
    ui: &mut egui::Ui,
    state: &mut ArmExampleState,
    sim: &Sim,
    pending: &mut PendingSceneSwitch,
) {
    ui.add_space(2.0);
    readout(ui, "Solver", "Featherstone GPU");
    readout(ui, "Controller", "GPU PD");
    readout(ui, "Joints", &format!("{ARM_LINKS}"));
    readout(ui, "Effort cap", &format!("{:.0} N·m", ARM_MAX_EFFORT));
    ui.separator();

    // Live joint state — read joint_q (already host-mirrored after
    // the Featherstone download in step_simulation).
    if let Ok(q) = sim.state_a.joint_q.host() {
        ui.label("Current angles");
        for i in 0..ARM_LINKS {
            if i < q.len() {
                readout(ui, &format!("  q{i}"), &format!("{:.3} rad", q[i]));
            }
        }
        ui.separator();
    }

    // Body world poses — FK output. If these stay frozen at the
    // initial values from `build_arm_model` while joint_q is changing,
    // Featherstone's FK pass isn't reaching the host (or the kernel's
    // not writing to body_q at all).
    if let Ok(bq) = sim.state_a.body_q.host() {
        ui.label("Body Y positions (FK)");
        for i in 0..ARM_LINKS {
            if i < bq.len() {
                readout(
                    ui,
                    &format!("  body{i}.y"),
                    &format!("{:.3} m", bq[i].position.y),
                );
            }
        }
        ui.separator();
    }

    ui.checkbox(&mut state.sine_enabled, "Auto-sine drive");
    if state.sine_enabled {
        ui.add(egui::Slider::new(&mut state.sine_amplitude, 0.0..=1.4).text("amplitude (rad)"));
        ui.add(egui::Slider::new(&mut state.sine_frequency, 0.0..=2.0).text("frequency (Hz)"));
    } else {
        ui.label("Targets (rad)");
        for i in 0..ARM_LINKS {
            ui.add(
                egui::Slider::new(
                    &mut state.targets[i],
                    -std::f32::consts::FRAC_PI_2..=std::f32::consts::FRAC_PI_2,
                )
                .text(format!("q{i} target")),
            );
        }
    }

    ui.add_space(4.0);
    ui.small(
        "PD effort = kp·(target − q) + kd·(0 − qd), capped to ±max_effort. \
         Watch the arm settle on slider changes; at high gains it's stiff, \
         at low gains it overshoots.",
    );

    // Stage 32 — joint-mimic toggle. Adds a 7th capsule body in a
    // separate articulation, coupled to joint 0 (the shoulder) by a
    // `JointMimic` equality with the linear-identity polynomial.
    // Toggling forces a scene rebuild so the mimic body / joint /
    // equality enter/exit the model cleanly.
    ui.separator();
    let prev = state.mimic_enabled;
    if ui
        .checkbox(&mut state.mimic_enabled, "Mimic joint (Stage 32)")
        .changed()
        && state.mimic_enabled != prev
    {
        state.mimic_dirty = true;
        set_arm_mimic_flag(state.mimic_enabled);
    }
    if state.mimic_enabled {
        ui.small(
            "Linear-identity polynomial [0, 1, 0, 0, 0]. The 7th body \
             on +X tracks the shoulder via JointMimic equality (XPBD \
             projection) + Featherstone pre-step (legacy linear \
             mimic). Watch the offset capsule mirror the shoulder's \
             angle as the sine drive sweeps.",
        );
    }
    if state.mimic_dirty {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Toggle changed — rebuild the arm scene to apply.",
        );
        if ui.button("Rebuild arm").clicked() {
            // Re-enter the current scene unconditionally; the
            // hotkey path (`handle_input`) skips this when the
            // scene is already active, so the button is the only
            // way to force a rebuild from the pane.
            pending.0 = Some(sim.scene);
        }
    }
}


