// DiffSpring scene — particle inside a "cage" of springs, all
// anchored at fixed cube corners. The host runs gradient descent on
// per-spring rest lengths so the particle ends up at the target.
// Visualised as kinematic anchor bodies + a particle body + line
// gizmos for the springs. Port of `example_diffsim_spring_cage.py`.
const DIFF_SPRING_CAGE_HALF: f64 = 0.6;
const DIFF_SPRING_CENTRE_Y: f64 = 1.5;
fn build_diff_spring_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();

    let mut visuals = Vec::with_capacity(9);
    // 8 cage-corner anchors (kinematic spheres). Order matches the
    // anchor list initialised on first frame in update_diff_spring.
    for sx in &[-1.0_f64, 1.0] {
        for sy in &[-1.0_f64, 1.0] {
            for sz in &[-1.0_f64, 1.0] {
                let pos = MVec3::new(
                    sx * DIFF_SPRING_CAGE_HALF,
                    DIFF_SPRING_CENTRE_Y + sy * DIFF_SPRING_CAGE_HALF,
                    sz * DIFF_SPRING_CAGE_HALF,
                );
                let body = b.add_body(
                    mp::BodyParams::new()
                        .with_mass(0.0)
                        .with_inertia(molla::math::Mat3::ZERO)
                        .with_position(pos)
                        .with_flags(mp::BodyFlags::KINEMATIC),
                );
                b.add_shape_sphere(body, 0.04, mp::Transform::IDENTITY)
                    .expect("anchor sphere");
                visuals.push(BodyVisual::Sphere(0.04));
            }
        }
    }
    // Body 8 — kinematic particle (the variable being optimised). The
    // host writes its position each frame from the gradient-descent
    // result.
    let particle = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_inertia(molla::math::Mat3::ZERO)
            .with_position(MVec3::new(0.0, DIFF_SPRING_CENTRE_Y, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(particle, 0.08, mp::Transform::IDENTITY)
        .expect("particle");
    visuals.push(BodyVisual::Sphere(0.08));

    // Static plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("diff spring model"), visuals)
}

const DIFFSIM_CLOTH_NX: u32 = 12;
const DIFFSIM_CLOTH_SIZE: f64 = 1.2;
const DIFFSIM_CLOTH_TOP_Y: f64 = 2.8;

#[derive(Resource)]
struct DiffSpringExampleState {
    /// Number of springs in the cage (kept fixed at build time; UI
    /// just shows it).
    n_springs: u32,
    /// Per-spring rest length (current optimised value).
    rest_lengths: Vec<f32>,
    /// Cage anchor points (world coords) — fixed.
    anchors: Vec<[f32; 3]>,
    /// Particle position the optimiser is solving for.
    particle: [f32; 3],
    /// Target position (the particle should land here).
    target: [f32; 3],
    /// Last residual (m).
    last_residual: f32,
    /// Iterations of gradient descent run.
    iters_run: u32,
    /// Re-run trigger.
    rerun: bool,
}

impl Default for DiffSpringExampleState {
    fn default() -> Self {
        Self {
            n_springs: 0,
            rest_lengths: Vec::new(),
            anchors: Vec::new(),
            particle: [0.0, 1.5, 0.0],
            target: [0.6, 1.0, 0.2],
            last_residual: 0.0,
            iters_run: 0,
            rerun: true,
        }
    }
}

// ── Update-system stubs for Newton-port scenes (filled in below
// scene-by-scene). Each is a no-op when the active scene differs.

fn update_diff_spring_example(mut sim: ResMut<Sim>, mut state: ResMut<DiffSpringExampleState>) {
    if sim.scene != Scene::DiffSpring {
        return;
    }
    if sim.model.body_count < 9 {
        return;
    }

    // Lazy-init: cache anchor positions + initialise rest lengths.
    if state.anchors.is_empty() {
        state.anchors.clear();
        for sx in &[-1.0_f64, 1.0] {
            for sy in &[-1.0_f64, 1.0] {
                for sz in &[-1.0_f64, 1.0] {
                    state.anchors.push([
                        (sx * DIFF_SPRING_CAGE_HALF) as f32,
                        (DIFF_SPRING_CENTRE_Y + sy * DIFF_SPRING_CAGE_HALF) as f32,
                        (sz * DIFF_SPRING_CAGE_HALF) as f32,
                    ]);
                }
            }
        }
        state.n_springs = state.anchors.len() as u32;
        // Initial rest length = distance from cage centre to anchor.
        state.rest_lengths = state
            .anchors
            .iter()
            .map(|a| {
                let dx = a[0];
                let dy = a[1] - DIFF_SPRING_CENTRE_Y as f32;
                let dz = a[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        state.particle = [0.0, DIFF_SPRING_CENTRE_Y as f32, 0.0];
    }

    // Re-optimise rest lengths to drive the particle to the target.
    if state.rerun {
        // Equilibrium: particle position is the weighted centroid of
        // anchor positions, where weights are 1/rest_length. We
        // gradient-descend on rest lengths so the equilibrium centroid
        // matches the target.
        let target = state.target;
        // Helper: compute current particle equilibrium (closed form).
        let equilibrium = |rls: &[f32], anchors: &[[f32; 3]]| -> [f32; 3] {
            let mut sum_w = 0.0_f32;
            let mut sum_x = [0.0_f32; 3];
            for (a, l) in anchors.iter().zip(rls.iter()) {
                let w = 1.0 / l.max(1e-3);
                sum_w += w;
                sum_x[0] += a[0] * w;
                sum_x[1] += a[1] * w;
                sum_x[2] += a[2] * w;
            }
            if sum_w < 1e-6 {
                [0.0; 3]
            } else {
                [sum_x[0] / sum_w, sum_x[1] / sum_w, sum_x[2] / sum_w]
            }
        };
        let loss = |rls: &[f32], anchors: &[[f32; 3]], tgt: [f32; 3]| -> f32 {
            let p = equilibrium(rls, anchors);
            let dx = p[0] - tgt[0];
            let dy = p[1] - tgt[1];
            let dz = p[2] - tgt[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let lr = 0.1_f32;
        let mut rls = state.rest_lengths.clone();
        for _it in 0..80 {
            let m0 = loss(&rls, &state.anchors, target);
            // FD gradient on each rest length.
            let mut grads = vec![0.0_f32; rls.len()];
            for i in 0..rls.len() {
                let saved = rls[i];
                rls[i] = saved + 0.005;
                let m1 = loss(&rls, &state.anchors, target);
                grads[i] = (m1 - m0) / 0.005;
                rls[i] = saved;
            }
            for i in 0..rls.len() {
                rls[i] -= lr * grads[i];
                rls[i] = rls[i].max(0.05).min(3.0);
            }
            if m0 < 0.01 {
                break;
            }
        }
        let final_p = equilibrium(&rls, &state.anchors);
        state.rest_lengths = rls;
        state.particle = final_p;
        state.last_residual = loss(&state.rest_lengths, &state.anchors, target);
        state.iters_run = 80;
        state.rerun = false;
    }

    // Push particle pose into body[8] for the renderer.
    let body_q = match sim.state_a.body_q.host_mut() {
        Ok(b) => b,
        Err(_) => return,
    };
    if body_q.len() >= 9 {
        body_q[8] = mp::Transform::from_translation(MVec3::new(
            state.particle[0] as f64,
            state.particle[1] as f64,
            state.particle[2] as f64,
        ));
    }
    let queue = sim.engine.queue().expect("queue").clone();
    let _ = sim.state_a.body_q.upload(&queue);
}

// ── Pane stubs for the rest of the Newton-port scenes — replaced
// scene-by-scene below.

fn diff_spring_pane(ui: &mut egui::Ui, state: &mut DiffSpringExampleState, accent: egui::Color32) {
    readout(ui, "Springs", &format!("{}", state.n_springs));
    readout(ui, "Iters run", &format!("{}", state.iters_run));
    readout(ui, "Residual", &format!("{:.4} m", state.last_residual));
    ui.separator();
    ui.label("Target X / Y / Z");
    if ui
        .add(egui::Slider::new(&mut state.target[0], -1.0..=1.0))
        .changed()
    {
        state.rerun = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.target[1], 0.5..=2.5))
        .changed()
    {
        state.rerun = true;
    }
    if ui
        .add(egui::Slider::new(&mut state.target[2], -1.0..=1.0))
        .changed()
    {
        state.rerun = true;
    }
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
    ui.small("Particle in a cage of springs. Gradient descent");
    ui.small("on rest lengths drives the particle to the target.");
}

// Stage 33.4 — FastSoftImpact right-rail pane. Carries:
// - CCD on/off toggle (re-launches the cube on flip).
// - Impact-speed slider (m/s).
// - Live readout of CCD invocations / snaps.

fn draw_diff_spring_gizmos(sim: Res<Sim>, state: Res<DiffSpringExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::DiffSpring {
        return;
    }
    if state.anchors.is_empty() {
        return;
    }
    let p = Vec3::new(state.particle[0], state.particle[1], state.particle[2]);
    let target = Vec3::new(state.target[0], state.target[1], state.target[2]);
    let line_color = Color::srgb(0.85, 0.85, 0.20);
    for a in &state.anchors {
        let av = Vec3::new(a[0], a[1], a[2]);
        gizmos.line(av, p, line_color);
    }
    // Target marker (cross gizmo).
    let tgt_color = Color::srgb(0.95, 0.30, 0.20);
    let r = 0.10;
    gizmos.line(target - Vec3::X * r, target + Vec3::X * r, tgt_color);
    gizmos.line(target - Vec3::Y * r, target + Vec3::Y * r, tgt_color);
    gizmos.line(target - Vec3::Z * r, target + Vec3::Z * r, tgt_color);
}

