// ═══ Scene 84: ClothPokerCards (Newton port) ═══════════════════════════
//
// 52 stiff VBD cloth "cards" stacked into a deck on top of a static box,
// with a kinematic sphere swinging in from the side to knock the stack
// over. Each card is a small high-stiffness cloth grid (4×6 cells, 5×7
// particles). Pure stress test of the VBD cloth solver — no plumbing
// beyond `add_cloth_grid_with_bending` + the existing VBD pipeline.
//
// Newton ref: `xtra/newton/newton/examples/cloth/example_cloth_poker_cards.py`.

const POKER_NUM_CARDS: u32 = 52;
const POKER_CARD_W: f64 = 0.0635; // 6.35 cm
const POKER_CARD_H: f64 = 0.0889; // 8.89 cm
const POKER_CARD_DIM_X: u32 = 4; // cells along width
const POKER_CARD_DIM_Y: u32 = 6; // cells along height
const POKER_CUBE_HALF: f64 = 0.10; // static cube half-extent
const POKER_CUBE_Y: f64 = 0.10; // cube centre Y
const POKER_DROP_BASE: f64 = 0.25; // first card spawn Y
const POKER_CARD_SPACING: f64 = 0.0015;
const POKER_SPHERE_RADIUS: f32 = 0.025;
const POKER_SPHERE_Y: f32 = 0.225;
const POKER_SPHERE_START_X: f32 = -0.40;
const POKER_SPHERE_END_X: f32 = 0.40;
const POKER_SPHERE_VEL: f32 = 0.40;
const POKER_PARTICLE_RADIUS: f64 = 0.003;
const POKER_PARTICLE_MASS: f64 = 1.8e-3 / 35.0;
// Stiff cards — very small stretch compliance and moderate bending.
const POKER_STRETCH_COMPLIANCE: f64 = 5.0e-8;
const POKER_BEND_COMPLIANCE: f64 = 5.0e-4;

struct ClothPokerCardsState {
    /// Live sphere X. Reset to `POKER_SPHERE_START_X` when the scene
    /// switches or the user clicks "Reset sphere".
    sphere_x: f32,
    sphere_vel: f32,
    /// User-controlled pause for the sphere only (physics still runs).
    sphere_paused: bool,
    user_reset: bool,
}

impl Default for ClothPokerCardsState {
    fn default() -> Self {
        Self {
            sphere_x: POKER_SPHERE_START_X,
            sphere_vel: POKER_SPHERE_VEL,
            sphere_paused: false,
            user_reset: false,
        }
    }
}

// Build 52 stiff cloth grids stacked above a static cube + a kinematic
// sphere off to the left. The sphere body is recorded as the LAST
// body added so `update_cloth_poker_cards` can address it via
// `body_count - 1`.
fn build_cloth_poker_cards_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    // Static cube — body with KINEMATIC flag so it stays put.
    let cube = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(1.0, POKER_CUBE_HALF))
            .with_position(MVec3::new(0.0, POKER_CUBE_Y, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_box(
        cube,
        MVec3::new(POKER_CUBE_HALF, POKER_CUBE_HALF, POKER_CUBE_HALF),
        mp::Transform::IDENTITY,
    )
    .expect("poker cube");

    // 52 cards stacked above the cube. Tiny random-ish offsets so they
    // settle into a slightly disordered stack (deterministic LCG).
    let mut rng_state: u64 = 0x1234_5678_9ABC_DEF0;
    let rng = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    for i in 0..POKER_NUM_CARDS {
        let off_x = rng(&mut rng_state) * 0.004;
        let off_z = rng(&mut rng_state) * 0.004;
        let z = POKER_DROP_BASE + i as f64 * POKER_CARD_SPACING;
        let origin = MVec3::new(-POKER_CARD_W * 0.5 + off_x, z, -POKER_CARD_H * 0.5 + off_z);
        let u_axis = MVec3::new(POKER_CARD_W, 0.0, 0.0);
        let v_axis = MVec3::new(0.0, 0.0, POKER_CARD_H);
        let _ = b.add_cloth_grid_with_bending(
            origin,
            u_axis,
            v_axis,
            POKER_CARD_DIM_X + 1,
            POKER_CARD_DIM_Y + 1,
            POKER_PARTICLE_MASS,
            POKER_PARTICLE_RADIUS,
            POKER_STRETCH_COMPLIANCE,
            0.02,
            POKER_BEND_COMPLIANCE,
            mp::WorldId(0),
        );
    }

    // Kinematic wrecking sphere — added LAST so it's at body index
    // `body_count - 1` and `update_cloth_poker_cards` can find it.
    let sphere = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(1.0, POKER_SPHERE_RADIUS as f64))
            .with_position(MVec3::new(
                POKER_SPHERE_START_X as f64,
                POKER_SPHERE_Y as f64,
                0.0,
            ))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(sphere, POKER_SPHERE_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("poker sphere");

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("poker plane");

    let visuals = vec![
        BodyVisual::Box(Vec3::new(
            POKER_CUBE_HALF as f32,
            POKER_CUBE_HALF as f32,
            POKER_CUBE_HALF as f32,
        )),
        BodyVisual::Sphere(POKER_SPHERE_RADIUS),
    ];
    (b.build().expect("cloth poker cards model"), visuals)
}

fn update_cloth_poker_cards(
    mut sim: ResMut<Sim>,
    time: Res<Time>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::ClothPokerCards {
        return;
    }
    let st = &mut warp.cloth_poker_cards;
    if st.user_reset {
        st.sphere_x = POKER_SPHERE_START_X;
        st.user_reset = false;
    }
    if !run.paused && !st.sphere_paused {
        st.sphere_x += st.sphere_vel * time.delta_secs();
        if st.sphere_x > POKER_SPHERE_END_X {
            st.sphere_x = POKER_SPHERE_END_X;
        }
    }
    // Sphere body is the LAST body in the model (added after the cube).
    let n_bodies = sim.model.body_count;
    if n_bodies < 2 {
        return;
    }
    let pose =
        mp::Transform::from_translation(MVec3::new(st.sphere_x as f64, POKER_SPHERE_Y as f64, 0.0));
    if let Ok(bq) = sim.state_a.body_q.host_mut() {
        if let Some(b) = bq.last_mut() {
            *b = pose;
        }
    }
    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.state_a.body_q.ensure_device(&device);
        let _ = sim.state_a.body_q.upload(&queue);
    }
}

fn draw_cloth_poker_cards_gizmos(sim: Res<Sim>, mut gizmos: Gizmos) {
    if sim.scene != Scene::ClothPokerCards {
        return;
    }
    // Draw each cloth-grid card as a wireframe of its tri edges (so the
    // user sees 52 stacked little quads). Cards are laid out
    // sequentially in `particle_q` starting at index 0; each card has
    // `(DIM_X+1) * (DIM_Y+1)` particles.
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    let nx = POKER_CARD_DIM_X + 1;
    let ny = POKER_CARD_DIM_Y + 1;
    let per_card = (nx * ny) as usize;
    let max_card = (sim.model.particle_count / per_card).min(POKER_NUM_CARDS as usize);
    for c in 0..max_card {
        let base = c * per_card;
        // Hue cycles around the deck for visual separation.
        let t = c as f32 / POKER_NUM_CARDS as f32;
        let r = 0.85 - 0.5 * t;
        let g = 0.30 + 0.5 * (t * std::f32::consts::TAU).sin().abs();
        let b = 0.20 + 0.7 * t;
        let col = Color::srgb(r, g, b);
        for iy in 0..ny {
            for ix in 0..nx {
                let i = base + (iy * nx + ix) as usize;
                if ix + 1 < nx {
                    let j = base + (iy * nx + ix + 1) as usize;
                    let a = pq[i];
                    let bp = pq[j];
                    gizmos.line(
                        Vec3::new(a.x as f32, a.y as f32, a.z as f32),
                        Vec3::new(bp.x as f32, bp.y as f32, bp.z as f32),
                        col,
                    );
                }
                if iy + 1 < ny {
                    let j = base + ((iy + 1) * nx + ix) as usize;
                    let a = pq[i];
                    let bp = pq[j];
                    gizmos.line(
                        Vec3::new(a.x as f32, a.y as f32, a.z as f32),
                        Vec3::new(bp.x as f32, bp.y as f32, bp.z as f32),
                        col,
                    );
                }
            }
        }
    }
}

fn cloth_poker_cards_pane(ui: &mut egui::Ui, st: &mut ClothPokerCardsState) {
    ui.label("Newton port — 52 stiff VBD cards on a stacked deck.");
    ui.separator();
    readout(ui, "Cards", &format!("{}", POKER_NUM_CARDS));
    readout(
        ui,
        "Cells / card",
        &format!("{}×{}", POKER_CARD_DIM_X, POKER_CARD_DIM_Y),
    );
    readout(
        ui,
        "Particles / card",
        &format!("{}", (POKER_CARD_DIM_X + 1) * (POKER_CARD_DIM_Y + 1)),
    );
    readout(ui, "Sphere x", &format!("{:+.3} m", st.sphere_x));
    ui.separator();
    ui.label("Wrecking sphere");
    let _ = ui.add(egui::Slider::new(&mut st.sphere_vel, -1.0..=1.0).text("velocity (m/s)"));
    let _ = ui.checkbox(&mut st.sphere_paused, "Pause sphere");
    if ui.button("Reset sphere to start").clicked() {
        st.user_reset = true;
    }
    ui.small("Stack settles for a beat, then the sphere swings");
    ui.small("through it. Bending stiffness is high so cards stay rigid.");
}

