// Reference double-pendulum demo runtime state. Kept as a native Featherstone
// procedural demo inside the physics-only surface.
#[derive(Resource)]
struct ReferencePendulumState {
    /// Smoothed wall-clock per `solver.step()` call (µs). Updated by
    /// `step_simulation` when this scene is active. Useful for users
    /// to compare against Featherstone's timing.
    smoothed_step_us: f64,
    /// Whether the user has been informed via `info!` that the scene now uses
    /// the native solver — only logged once to avoid spamming.
    notified_unavailable: bool,
    /// Cached info-style status string for the right-rail pane. Built
    /// the first time the scene is entered.
    status: String,
}

impl Default for ReferencePendulumState {
    fn default() -> Self {
        Self {
            smoothed_step_us: 0.0,
            notified_unavailable: false,
            status: "Native Featherstone procedural demo".to_owned(),
        }
    }
}

// --- Native reference double-pendulum scene -----------------------
//
// Double pendulum (4-link arm would also work). The topology is clean:
// two revolute joints + two capsule links, stepped by native Featherstone.
// Same body / joint construction as a slice of `build_arm_model`,
// just shorter and without the FK-on-elbow rotation tricks.
const REFERENCE_PENDULUM_LINK_LEN: f32 = 0.5;
const REFERENCE_PENDULUM_LINK_RADIUS: f32 = 0.05;
const REFERENCE_PENDULUM_LINK_MASS: f64 = 0.6;
const REFERENCE_PENDULUM_BASE_HEIGHT: f32 = 4.0;

fn build_reference_pendulum_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);

    let half_h = (REFERENCE_PENDULUM_LINK_LEN as f64) * 0.5;
    let radius = REFERENCE_PENDULUM_LINK_RADIUS as f64;
    let inertia = capsule_inertia(REFERENCE_PENDULUM_LINK_MASS, half_h, radius);

    b.begin_articulation();

    // Upper link.
    let upper_pos = MVec3::new(0.0, REFERENCE_PENDULUM_BASE_HEIGHT as f64 - half_h, 0.0);
    let upper = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: REFERENCE_PENDULUM_LINK_MASS,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(upper_pos),
    );
    b.add_shape_capsule(upper, half_h, radius, mp::Transform::IDENTITY)
        .expect("upper link");

    // Shoulder revolute joint about world +Z, anchored at the base
    // height. The PI rotation around X mirrors the arm scene's trick
    // for hanging the link along world -Y at q=0.
    let shoulder_x_p = mp::Transform::new(
        MVec3::new(0.0, REFERENCE_PENDULUM_BASE_HEIGHT as f64, 0.0),
        MQuat::from_axis_angle(MVec3::X, std::f64::consts::PI),
    );
    let shoulder_x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
    b.add_joint_revolute(
        mp::BodyId::NONE,
        upper,
        shoulder_x_p,
        shoulder_x_c,
        MVec3::Z,
    )
    .expect("shoulder joint");

    // Lower link.
    let lower_pos = upper_pos - MVec3::new(0.0, REFERENCE_PENDULUM_LINK_LEN as f64, 0.0);
    let lower = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: REFERENCE_PENDULUM_LINK_MASS,
                com: MVec3::ZERO,
                inertia,
            })
            .with_position(lower_pos),
    );
    b.add_shape_capsule(lower, half_h, radius, mp::Transform::IDENTITY)
        .expect("lower link");

    let elbow_x_p = mp::Transform::from_translation(MVec3::new(0.0, half_h, 0.0));
    let elbow_x_c = mp::Transform::from_translation(MVec3::new(0.0, -half_h, 0.0));
    b.add_joint_revolute(upper, lower, elbow_x_p, elbow_x_c, MVec3::X)
        .expect("elbow joint");

    let visuals = vec![
        BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        },
        BodyVisual::Capsule {
            half_h: half_h as f32,
            radius: radius as f32,
        },
    ];

    (b.build().expect("reference pendulum model"), visuals)
}

fn update_reference_pendulum_example(sim: Res<Sim>, mut state: ResMut<ReferencePendulumState>) {
    if sim.scene != Scene::ReferencePendulum {
        return;
    }
    if !state.notified_unavailable {
        info!("Reference-pendulum scene uses native Featherstone only.");
        state.notified_unavailable = true;
    }
}

// Convert a quaternion to YXZ Euler (yaw about Y, pitch about X, roll
// about Z). Used by the quadrotor pane for a human-readable readout
// of the attitude error.
fn quat_to_rpy(q: MQuat) -> (f64, f64, f64) {
    // glam (re-exported via molla::math) lives at bevy::math too,
    // but we go through the molla shim to avoid pulling glam directly.
    let (yaw, pitch, roll) = q.to_euler(bevy::math::EulerRot::YXZ);
    (roll, pitch, yaw)
}

// Contact example — runs the GPU `ContactSensor` against the
// current `Contacts` and stashes the readings. No-op for any other

fn reference_pendulum_pane(ui: &mut egui::Ui, state: &mut ReferencePendulumState, sim: &Sim) {
    ui.add_space(2.0);
    let backend_label = match &sim.solver {
        SolverBackend::Featherstone(_) => "Featherstone (native)",
        _ => "(unexpected solver)",
    };
    readout(ui, "Backend", backend_label);
    readout(ui, "Status", &state.status);
    readout(ui, "Bodies", &format!("{}", sim.model.body_count));
    readout(ui, "Joints", &format!("{}", sim.model.joint_count));
    readout(
        ui,
        "Solve / frame",
        &format!("{:>5.0} µs", state.smoothed_step_us),
    );
    ui.separator();

    ui.label("This procedural double pendulum runs on native Featherstone.");
    ui.separator();

    ui.small("Hotkey: J  ·  Stage 22 (P2)");
    ui.small("Reference: PLAN.md Stage 22, GAP.md §2 / §9.");
    ui.small("Double pendulum — same topology as the Featherstone");
    ui.small("multi-DOF parity test, makes A/B compare easy.");
}

// Re-derive per-spring compliance from the user-chosen stiffness
// sliders and re-upload to the solver. Cheap: spring count is O(few
// thousand) and the compliance buffer is a single device upload.
fn apply_vbd_stiffness(sim: &mut Sim, state: &VbdClothExampleState) {
    let spring_count = sim.model.spring_count;
    if spring_count == 0 {
        return;
    }
    let stretch_alpha = (1.0 / state.stretch_stiffness.max(1.0)) as f64;
    let bend_alpha = (1.0 / state.bend_stiffness.max(1.0)) as f64;
    // Compliance ordering: the cloth-grid builder appends springs in
    // (structural, shear, bending) groups; bending springs are the
    // "skip-one" entries with rest length = 2 × spacing. Round-tripping
    // via the rest-length sentinel is fragile; instead we rebuild the
    // distinction from the per-spring `max_stretch` flag (the builder
    // sets `BEND_STRETCH_DISABLED ≈ 2.0` for bending springs and a
    // user-facing value < 1.0 for the others).
    let max_stretch = sim
        .model
        .spring_max_stretch
        .host()
        .ok()
        .map(|s| s.to_vec())
        .unwrap_or_default();
    let comp_host = sim
        .model
        .spring_compliance
        .host_mut()
        .expect("spring_compliance host");
    for (i, c) in comp_host.iter_mut().enumerate() {
        let is_bend = max_stretch.get(i).is_some_and(|s| *s >= 1.0);
        *c = if is_bend { bend_alpha } else { stretch_alpha };
    }
    // Push the new compliance to the device + refresh the VBD
    // kernel's stiffness mirror.
    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();
    sim.model
        .spring_compliance
        .ensure_device(&device)
        .expect("ensure_device");
    sim.model
        .spring_compliance
        .upload(&queue)
        .expect("upload compliance");
    // Re-derive VBD's stiffness mirror.
    if let SolverBackend::Vbd(s) = &mut sim.solver {
        let _ = s.refresh_stiffness(&sim.model);
    }
}


