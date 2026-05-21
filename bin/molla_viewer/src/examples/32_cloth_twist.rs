// ClothTwist scene — a vertical cloth strip pinned along TOP and
// BOTTOM edges. The bottom edge is rotated each frame around the
// strip's vertical axis (Y) so the cloth twists. Port of
// `xtra/newton/newton/examples/cloth/example_cloth_twist.py`.
const CLOTH_TWIST_GRID_X: u32 = 18;
const CLOTH_TWIST_GRID_Y: u32 = 24;
const CLOTH_TWIST_W: f64 = 1.0;
const CLOTH_TWIST_H: f64 = 1.6;
const CLOTH_TWIST_TOP_Y: f64 = 4.0;
const CLOTH_TWIST_CENTRE_X: f64 = 0.0;
fn build_cloth_twist_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    let origin = MVec3::new(
        CLOTH_TWIST_CENTRE_X - CLOTH_TWIST_W * 0.5,
        CLOTH_TWIST_TOP_Y,
        0.0,
    );
    let u_axis = MVec3::new(CLOTH_TWIST_W, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, -CLOTH_TWIST_H, 0.0);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        CLOTH_TWIST_GRID_X,
        CLOTH_TWIST_GRID_Y,
        0.04,   // particle mass
        0.025,  // particle radius
        5.0e-6, // stretch compliance
        0.05,   // max stretch ratio
        mp::WorldId(0),
    );
    // Pin top row (iy = 0).
    for ix in 0..CLOTH_TWIST_GRID_X {
        b.pin_particle(first + ix);
    }
    // Pin bottom row (iy = CLOTH_TWIST_GRID_Y - 1) — driven by
    // `update_cloth_twist_example`.
    let bottom_iy = CLOTH_TWIST_GRID_Y - 1;
    for ix in 0..CLOTH_TWIST_GRID_X {
        b.pin_particle(first + bottom_iy * CLOTH_TWIST_GRID_X + ix);
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cloth twist model"), Vec::new())
}
// CablePile scene — drop a handful of cable polylines into a "bowl"
// (4 walls + floor) and watch them pile / coil. Port of
// `xtra/newton/newton/examples/cable/example_cable_pile.py` (simpler

#[derive(Resource)]
struct ClothTwistExampleState {
    /// Rotation rate (rad/s) for the bottom edge.
    rotation_rate: f32,
    /// Accumulated angle (rad).
    angle: f32,
    /// Cached top-edge particle indices (pinned).
    top_indices: Vec<u32>,
    /// Cached bottom-edge particle indices (rotated).
    bottom_indices: Vec<u32>,
    /// Cached initial position of every bottom particle (relative to
    /// the rotation axis — used to recompute the rotated position).
    bottom_rest: Vec<MVec3>,
    /// Rotation axis centre (Y-axis through the cloth centre).
    centre: MVec3,
}

impl Default for ClothTwistExampleState {
    fn default() -> Self {
        Self {
            rotation_rate: 0.4,
            angle: 0.0,
            top_indices: Vec::new(),
            bottom_indices: Vec::new(),
            bottom_rest: Vec::new(),
            centre: MVec3::ZERO,
        }
    }
}


fn update_cloth_twist_example(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut state: ResMut<ClothTwistExampleState>,
) {
    if sim.scene != Scene::ClothTwist {
        return;
    }
    if sim.model.particle_count == 0 {
        return;
    }
    // Lazy-cache the bottom-row particle indices and their initial
    // positions relative to the rotation axis on first frame.
    if state.bottom_indices.is_empty() {
        let bottom_iy = CLOTH_TWIST_GRID_Y - 1;
        let centre = MVec3::new(CLOTH_TWIST_CENTRE_X, CLOTH_TWIST_TOP_Y - CLOTH_TWIST_H, 0.0);
        state.centre = centre;
        let q = match sim.state_a.particle_q.host() {
            Ok(q) => q,
            Err(_) => return,
        };
        for ix in 0..CLOTH_TWIST_GRID_X {
            let i = bottom_iy * CLOTH_TWIST_GRID_X + ix;
            state.bottom_indices.push(i);
            state.bottom_rest.push(q[i as usize] - centre);
            // Top row indices for the readout (no rewrite needed —
            // they're pinned at their initial positions automatically).
            state.top_indices.push(ix);
        }
    }
    // Advance the angle by rotation_rate per second.
    state.angle += state.rotation_rate * time.delta_secs();
    let cos_a = state.angle.cos() as f64;
    let sin_a = state.angle.sin() as f64;

    let q = match sim.state_a.particle_q.host_mut() {
        Ok(q) => q,
        Err(_) => return,
    };
    for (k, &idx) in state.bottom_indices.iter().enumerate() {
        let rest = state.bottom_rest[k];
        // Rotate around Y axis (X' = X·cos + Z·sin, Z' = -X·sin + Z·cos).
        let xnew = rest.x * cos_a + rest.z * sin_a;
        let znew = -rest.x * sin_a + rest.z * cos_a;
        q[idx as usize] = state.centre + MVec3::new(xnew, rest.y, znew);
    }
    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.state_a.particle_q.ensure_device(&device);
        let _ = sim.state_a.particle_q.upload(&queue);
    }
}

// Stage 33.3 — drive the cloth-collar's iy=NX-1 row downward each
// frame so the cloth folds onto itself. Reads the live puller Y

fn cloth_twist_pane(ui: &mut egui::Ui, state: &mut ClothTwistExampleState) {
    ui.label("Bottom-edge rotation rate (rad/s)");
    ui.add(egui::Slider::new(&mut state.rotation_rate, -2.0..=2.0));
    readout(ui, "Twist angle", &format!("{:>+5.2} rad", state.angle));
    ui.separator();
    ui.small("Top edge pinned, bottom edge rotated.");
}

