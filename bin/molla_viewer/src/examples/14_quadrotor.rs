// Quadrotor demo runtime state. Tracks PD-control gains, target
// hover altitude, and per-rotor thrust readouts. Reset on scene
// switch.
#[derive(Resource)]
struct QuadrotorState {
    /// Target hover altitude (m).
    target_altitude: f32,
    /// PD altitude loop gains (force/m, force·s/m).
    altitude_kp: f32,
    altitude_kd: f32,
    /// PD attitude loop gains (torque/rad, torque·s/rad).
    attitude_kp: f32,
    attitude_kd: f32,
    /// Most recent per-rotor thrust commands (N), for UI readout.
    last_thrusts: [f32; 4],
    /// Most recent altitude (m).
    last_altitude: f32,
    /// Most recent attitude angles (roll, pitch, yaw) in radians.
    last_attitude: [f32; 3],
    /// Whether we've initialised the body-mass / arm-length cache.
    cached: bool,
    total_mass: f32,
    arm_length: f32,
    /// Stage 17.1: enable the rigorous dense-Cholesky D solve. v1
    /// uses sequential rank-1 deflation (exact when D is diagonal,
    /// approximate but stable when not). For symmetric quadrotor
    /// inertia the two paths agree closely; toggle on to demo the
    /// asymmetric-inertia case (e.g. add a payload offset).
    rigorous_d_solve: bool,
    /// Last applied flag value (so we don't re-stamp the solver every
    /// frame — only when the user flips the toggle).
    rigorous_d_solve_applied: bool,
}

impl Default for QuadrotorState {
    fn default() -> Self {
        Self {
            target_altitude: 1.5,
            altitude_kp: 30.0,
            altitude_kd: 8.0,
            attitude_kp: 4.0,
            attitude_kd: 0.6,
            last_thrusts: [0.0; 4],
            last_altitude: 0.0,
            last_attitude: [0.0; 3],
            cached: false,
            total_mass: 0.0,
            arm_length: 0.0,
            rigorous_d_solve: false,
            rigorous_d_solve_applied: false,
        }
    }
}

const QUAD_ARM: f64 = 0.25;
const QUAD_FRAME_MASS: f64 = 1.0;
const QUAD_ROTOR_MASS: f64 = 0.05;
const QUAD_FRAME_HALF: MVec3 = MVec3::new(0.18, 0.04, 0.18);
const QUAD_ROTOR_RADIUS: f64 = 0.06;
const QUAD_INITIAL_HEIGHT: f64 = 1.0;

fn build_quadrotor_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);
    b.set_angular_damping(0.6);
    b.begin_articulation();

    let mut visuals = Vec::with_capacity(5);

    // Frame body — central cuboid; FREE joint to world.
    let half = QUAD_FRAME_HALF;
    let m = QUAD_FRAME_MASS;
    let i_frame = molla::math::Mat3::from_diagonal(MVec3::new(
        m * (half.y * half.y + half.z * half.z) / 3.0,
        m * (half.x * half.x + half.z * half.z) / 3.0,
        m * (half.x * half.x + half.y * half.y) / 3.0,
    ));
    let frame = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia {
                mass: m,
                com: MVec3::ZERO,
                inertia: i_frame,
            })
            .with_position(MVec3::new(0.0, QUAD_INITIAL_HEIGHT, 0.0)),
    );
    b.add_shape_box(frame, half, mp::Transform::IDENTITY)
        .expect("frame box");
    visuals.push(BodyVisual::Box(Vec3::new(
        half.x as f32,
        half.y as f32,
        half.z as f32,
    )));
    // FREE joint anchored at world origin; child anchor at frame origin.
    b.add_joint_free(
        mp::BodyId::NONE,
        frame,
        mp::Transform::IDENTITY,
        mp::Transform::IDENTITY,
    )
    .expect("free root");

    // Rotor offsets in body frame (at ±arm along X and Z).
    let rotor_offsets = [
        MVec3::new(QUAD_ARM, 0.0, 0.0),  // 0: +X
        MVec3::new(-QUAD_ARM, 0.0, 0.0), // 1: -X
        MVec3::new(0.0, 0.0, QUAD_ARM),  // 2: +Z
        MVec3::new(0.0, 0.0, -QUAD_ARM), // 3: -Z
    ];
    let rotor_mass = QUAD_ROTOR_MASS;
    let r2 = QUAD_ROTOR_RADIUS * QUAD_ROTOR_RADIUS;
    let i_rotor = molla::math::Mat3::from_diagonal(MVec3::new(
        rotor_mass * r2 * 0.25,
        rotor_mass * r2 * 0.5,
        rotor_mass * r2 * 0.25,
    ));
    for off in rotor_offsets.iter() {
        let rotor = b.add_body(
            mp::BodyParams::new()
                .with_spatial_inertia(mp::SpatialInertia {
                    mass: rotor_mass,
                    com: MVec3::ZERO,
                    inertia: i_rotor,
                })
                .with_position(MVec3::new(off.x, QUAD_INITIAL_HEIGHT + 0.02, off.z)),
        );
        b.add_shape_sphere(rotor, QUAD_ROTOR_RADIUS, mp::Transform::IDENTITY)
            .expect("rotor sphere");
        visuals.push(BodyVisual::Sphere(QUAD_ROTOR_RADIUS as f32));
        // Fixed joint to the frame at the rotor's offset.
        b.add_joint_fixed(
            frame,
            rotor,
            mp::Transform::from_translation(*off + MVec3::Y * 0.02),
            mp::Transform::IDENTITY,
        )
        .expect("rotor fixed joint");
    }

    (b.build().expect("quadrotor model"), visuals)
}


fn update_quadrotor(mut sim: ResMut<Sim>, mut state: ResMut<QuadrotorState>) {
    if sim.scene != Scene::Quadrotor {
        return;
    }
    if sim.model.body_count < 5 {
        return;
    }
    // Cache per-scene mass + arm length once on entry.
    if !state.cached {
        let masses = sim.model.body_mass.host().expect("body_mass");
        let total: f64 = masses.iter().take(5).sum();
        state.total_mass = total as f32;
        state.arm_length = QUAD_ARM as f32;
        state.cached = true;
    }

    // Stage 17.1 — propagate the right-rail "rigorous" toggle into the
    // active Featherstone solver. Only re-stamp on change so we don't
    // touch the solver every frame.
    if state.rigorous_d_solve != state.rigorous_d_solve_applied {
        if let SolverBackend::Featherstone(s) = &mut sim.solver {
            s.set_rigorous_d_solve(state.rigorous_d_solve);
        }
        state.rigorous_d_solve_applied = state.rigorous_d_solve;
    }

    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();

    // Frame body = body 0; rotors = 1..=4.
    // Pull the latest frame pose + twist from the host. After every
    // frame, `step_simulation` downloads `body_q` + `body_qd` from the
    // device, so this is a fresh read.
    let (frame_pos, frame_rot, frame_v, frame_w) = {
        let bq = sim.state_a.body_q.host().expect("body_q");
        let bqd = sim.state_a.body_qd.host().expect("body_qd");
        (
            bq[0].position,
            bq[0].rotation,
            bqd[0].linear,
            bqd[0].angular,
        )
    };

    let body_y_world = frame_rot * MVec3::Y;
    let thrusts = quadrotor_pd_thrusts(
        state.total_mass as f64,
        state.arm_length as f64,
        state.target_altitude as f64,
        frame_pos.y,
        frame_v.y,
        body_y_world,
        frame_w,
        state.altitude_kp as f64,
        state.altitude_kd as f64,
        state.attitude_kp as f64,
        state.attitude_kd as f64,
    );

    // Update UI state.
    for i in 0..4 {
        state.last_thrusts[i] = thrusts[i] as f32;
    }
    state.last_altitude = frame_pos.y as f32;
    let (roll, pitch, yaw) = quat_to_rpy(frame_rot);
    state.last_attitude = [roll as f32, pitch as f32, yaw as f32];

    // Write thrust as world-frame +Y force on each rotor body. The
    // FREE root + fixed-rotor topology means each rotor's wrench is
    // transmitted into the frame via its joint constraint.
    {
        let bf = sim.state_a.body_f.host_mut().expect("body_f");
        // Frame body: zero (we drive via the rotors).
        bf[0] = mp::SpatialVector::ZERO;
        for i in 0..4 {
            let f = mp::SpatialVector::new(MVec3::new(0.0, thrusts[i], 0.0), MVec3::ZERO);
            bf[1 + i] = f;
        }
        // Upload to device so Featherstone reads it.
        sim.state_a.body_f.upload(&queue).expect("body_f upload");
    }

    // Touch device so future downloads see the host write.
    let _ = device;
}

// Reference-pendulum update. Logs once that the procedural double pendulum is

fn quadrotor_pane(ui: &mut egui::Ui, state: &mut QuadrotorState) {
    ui.add_space(2.0);
    readout(ui, "Solver", "Featherstone-GPU (Stage 17)");
    readout(ui, "Mass", &format!("{:.3} kg", state.total_mass));
    readout(ui, "Arm", &format!("{:.3} m", state.arm_length));
    ui.separator();

    ui.label("Live state");
    readout(ui, "  altitude", &format!("{:>6.3} m", state.last_altitude));
    readout(
        ui,
        "  roll",
        &format!("{:>6.2}°", state.last_attitude[0].to_degrees()),
    );
    readout(
        ui,
        "  pitch",
        &format!("{:>6.2}°", state.last_attitude[1].to_degrees()),
    );
    readout(
        ui,
        "  yaw",
        &format!("{:>6.2}°", state.last_attitude[2].to_degrees()),
    );
    ui.separator();

    ui.label("Per-rotor thrust (N)");
    readout(ui, "  rotor +X", &format!("{:>6.2}", state.last_thrusts[0]));
    readout(ui, "  rotor -X", &format!("{:>6.2}", state.last_thrusts[1]));
    readout(ui, "  rotor +Z", &format!("{:>6.2}", state.last_thrusts[2]));
    readout(ui, "  rotor -Z", &format!("{:>6.2}", state.last_thrusts[3]));
    let total: f32 = state.last_thrusts.iter().sum();
    readout(ui, "  Σ thrust", &format!("{:>6.2}", total));
    ui.separator();

    ui.label("Target altitude (m)");
    ui.add(
        egui::Slider::new(&mut state.target_altitude, 0.2..=4.0)
            .text("z_t")
            .step_by(0.05),
    );
    ui.separator();

    ui.label("Altitude PD");
    ui.add(egui::Slider::new(&mut state.altitude_kp, 0.0..=120.0).text("kp"));
    ui.add(egui::Slider::new(&mut state.altitude_kd, 0.0..=40.0).text("kd"));
    ui.separator();

    ui.label("Attitude PD");
    ui.add(egui::Slider::new(&mut state.attitude_kp, 0.0..=20.0).text("kp"));
    ui.add(egui::Slider::new(&mut state.attitude_kd, 0.0..=4.0).text("kd"));
    ui.separator();

    // Stage 17.1 — rigorous toggle.
    ui.label("Stage 17.1 polish");
    ui.checkbox(
        &mut state.rigorous_d_solve,
        "rigorous dense-Cholesky D solve",
    );
    ui.small(
        "v1 sequential rank-1 deflation is exact when the joint's D \
         matrix is diagonal. For asymmetric inertias (e.g. rotor mass \
         offset cross-coupling linear/angular DOFs) the rigorous path \
         is force-correct. For this symmetric quadrotor the two paths \
         agree closely; toggle to A/B against your own asymmetric \
         payload.",
    );

    ui.add_space(4.0);
    ui.small(
        "FREE root + 4 fixed-rotor bodies. Per-rotor thrust commanded \
         via state.body_f world-frame +Y forces; Featherstone GPU \
         solves the 6-DOF base + 4 rigid attachments. Yaw is free \
         (single-axis thrust can't generate yaw torque).",
    );
}

// Body for the right-rail pane in the Differentiable-arm (Stage 23)
// scene. Owns the target XYZ sliders, the iteration counter +
// residual readout, the "Re-run" button, and the trajectory-overlay

