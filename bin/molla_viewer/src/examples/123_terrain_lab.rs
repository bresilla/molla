// ── 123 Terrain lab ────────────────────────────────────────────────
//
// Same FEM tire as scene 122 (`molla_vehicle::TetraTireFem`), but now
// physically rolling forward over a procedural heightfield with named
// zones — flat run-in, three smooth speed-bumps, a bowl-shaped hole, a
// tapered washboard section, and a ramp.
//
// The tire is driven by `external_force_xyz`: ↑ pushes a +X force per
// node, ↓ pushes -X. Vertical load is applied the same way. The tire
// itself integrates everything via `step_on_terrain`, so what you see is
// the actual FEM mesh moving through world space and deforming over
// terrain — no cosmetic wheel ring on top.

const TLAB_TIRE_RADIUS: f64 = 0.32;
const TLAB_TIRE_INNER_RADIUS: f64 = 0.12;
const TLAB_TIRE_WIDTH: f64 = 0.185;
const TLAB_TIRE_SEGMENTS: usize = 24;

const TLAB_PRESSURE_PA: f64 = 180_000.0;
const TLAB_VERTICAL_LOAD_N: f64 = 2_600.0;

const TLAB_TILE_NX: usize = 161;
const TLAB_TILE_NZ: usize = 41;
const TLAB_TILE_SPACING: f64 = 0.30;
const TLAB_BUMP_HEIGHT_M: f64 = 0.06;
const TLAB_BUMP_HALF_WIDTH_M: f64 = 0.45;
const TLAB_BUMP_CENTERS_X: [f64; 3] = [-12.0, -10.0, -8.0];
const TLAB_HOLE_CENTER_X: f64 = -4.0;
const TLAB_HOLE_HALF_WIDTH_M: f64 = 0.90;
const TLAB_HOLE_DEPTH_M: f64 = 0.18;
const TLAB_WASHBOARD_RANGE_X: (f64, f64) = (1.0, 6.0);
const TLAB_WASHBOARD_AMPLITUDE_M: f64 = 0.04;
const TLAB_WASHBOARD_WAVELENGTH_M: f64 = 0.55;
const TLAB_RAMP_START_X: f64 = 9.0;
const TLAB_RAMP_LENGTH_M: f64 = 6.0;
const TLAB_RAMP_HEIGHT_M: f64 = 0.50;

/// Per-frame state for [`Scene::TerrainLab`].
#[derive(Resource)]
#[derive(Default)]
struct TerrainLabExampleState {
    tire: Option<molla_vehicle::TetraTireFem>,
    terrain: Option<molla_vehicle::HeightfieldTerrain>,
    grid_segments: Vec<(Vec3, Vec3)>,
    was_active: bool,
}


const TLAB_DRIVE_TORQUE_N_M: f64 = 160.0;
const TLAB_BRAKE_TORQUE_N_M: f64 = 240.0;

/// Composite procedural terrain. World X = drive direction.
fn tlab_height(x: f64, _z: f64) -> f64 {
    let mut h = 0.0;
    for &cx in &TLAB_BUMP_CENTERS_X {
        let d = (x - cx) / TLAB_BUMP_HALF_WIDTH_M;
        if d.abs() < 1.0 {
            h += TLAB_BUMP_HEIGHT_M * 0.5 * (1.0 + (std::f64::consts::PI * d).cos());
        }
    }
    {
        let d = (x - TLAB_HOLE_CENTER_X) / TLAB_HOLE_HALF_WIDTH_M;
        if d.abs() < 1.0 {
            h -= TLAB_HOLE_DEPTH_M * 0.5 * (1.0 + (std::f64::consts::PI * d).cos());
        }
    }
    let (wb_x0, wb_x1) = TLAB_WASHBOARD_RANGE_X;
    if x > wb_x0 && x < wb_x1 {
        let center = 0.5 * (wb_x0 + wb_x1);
        let half_width = 0.5 * (wb_x1 - wb_x0);
        let env_t = ((x - center) / half_width).clamp(-1.0, 1.0);
        let env = 0.5 * (1.0 + (std::f64::consts::PI * env_t).cos());
        let phase = (x - wb_x0) * std::f64::consts::TAU / TLAB_WASHBOARD_WAVELENGTH_M;
        h += env * TLAB_WASHBOARD_AMPLITUDE_M * (1.0 - phase.cos()) * 0.5;
    }
    if x >= TLAB_RAMP_START_X {
        let along = ((x - TLAB_RAMP_START_X) / TLAB_RAMP_LENGTH_M).min(1.0);
        h += TLAB_RAMP_HEIGHT_M * along;
    }
    h
}

fn build_tlab_heightfield() -> molla_vehicle::HeightfieldTerrain {
    let nx = TLAB_TILE_NX;
    let nz = TLAB_TILE_NZ;
    let spacing = TLAB_TILE_SPACING;
    let origin = [
        -(nx as f64 - 1.0) * spacing * 0.5,
        -(nz as f64 - 1.0) * spacing * 0.5,
    ];
    let mut heights = Vec::with_capacity(nx * nz);
    for iz in 0..nz {
        let z = origin[1] + iz as f64 * spacing;
        for ix in 0..nx {
            let x = origin[0] + ix as f64 * spacing;
            heights.push(tlab_height(x, z));
        }
    }
    molla_vehicle::HeightfieldTerrain::new(origin, spacing, nx, nz, heights, 0.95)
}

fn build_tlab_grid_segments(
    terrain: &molla_vehicle::HeightfieldTerrain,
) -> Vec<(Vec3, Vec3)> {
    use molla_vehicle::Terrain;
    let mut out = Vec::new();
    for iz in 0..terrain.nz {
        let z = terrain.origin[1] + iz as f64 * terrain.spacing;
        for ix in 0..(terrain.nx - 1) {
            let x0 = terrain.origin[0] + ix as f64 * terrain.spacing;
            let x1 = x0 + terrain.spacing;
            let h0 = terrain.height(x0, z);
            let h1 = terrain.height(x1, z);
            out.push((
                Vec3::new(x0 as f32, h0 as f32, z as f32),
                Vec3::new(x1 as f32, h1 as f32, z as f32),
            ));
        }
    }
    for ix in 0..terrain.nx {
        let x = terrain.origin[0] + ix as f64 * terrain.spacing;
        for iz in 0..(terrain.nz - 1) {
            let z0 = terrain.origin[1] + iz as f64 * terrain.spacing;
            let z1 = z0 + terrain.spacing;
            let h0 = terrain.height(x, z0);
            let h1 = terrain.height(x, z1);
            out.push((
                Vec3::new(x as f32, h0 as f32, z0 as f32),
                Vec3::new(x as f32, h1 as f32, z1 as f32),
            ));
        }
    }
    out
}

fn tlab_tire_params() -> molla_vehicle::TetraTireFemParams {
    molla_vehicle::TetraTireFemParams {
        outer_radius: TLAB_TIRE_RADIUS,
        inner_radius: TLAB_TIRE_INNER_RADIUS,
        width: TLAB_TIRE_WIDTH,
        segments: TLAB_TIRE_SEGMENTS,
        radial_layers: 3,
        width_layers: 3,
        pressure_pa: TLAB_PRESSURE_PA,
        downforce: TLAB_VERTICAL_LOAD_N,
        young_modulus: 4.0e6,
        sidewall_young_modulus: 2.0e6,
        tread_young_modulus: 8.0e6,
        bead_young_modulus: 2.5e7,
        density: 1_100.0,
        node_damping: 55.0,
        bead_stiffness: 5.8e4,
        ..Default::default()
    }
}

fn update_terrain_lab_example(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    sim: Res<Sim>,
    mut state: ResMut<TerrainLabExampleState>,
) {
    if sim.scene != Scene::TerrainLab {
        state.was_active = false;
        return;
    }
    if !state.was_active {
        let terrain = build_tlab_heightfield();
        state.grid_segments = build_tlab_grid_segments(&terrain);
        state.terrain = Some(terrain);
        // Spawn the FEM tire at X = -16 m, in the flat run-in zone.
        let mut tire = molla_vehicle::TetraTireFem::new(
            -16.0,
            TLAB_PRESSURE_PA,
            tlab_tire_params(),
        );
        tire.external_force_xyz = [0.0, -TLAB_VERTICAL_LOAD_N, 0.0];
        // PLAN_TOURQUE Stage E — opt into dynamic-hub integration so
        // contact reaction physically rolls the tire forward instead
        // of the host prescribing hub X.
        tire.set_hub_dynamic();
        state.tire = Some(tire);
        state.was_active = true;
    }
    // Fixed physics dt with a frame-derived substep count (scene-110
    // pattern) — feeding render-frame dt straight into the solver made
    // the demo frame-rate-dependent (2026-07-05 tire audit).
    let frame = (time.delta_secs() as f64).clamp(0.0, 1.0 / 30.0);
    let dt = 1.0 / 240.0;
    let steps = ((frame / dt).round() as u32).clamp(1, 8);
    // PLAN_TOURQUE Stage E — arrow keys → drive/brake torque at the
    // wheel hub. Throttle ↑ adds drive torque, brake ↓ adds brake
    // torque (the FEM tire's soft-sign saturator keeps brake from
    // overshooting through zero). No manual forward_speed integration,
    // no set_hub_kinematic — the tire rolls because contact friction
    // pushes the hub.
    let throttle = keys.pressed(KeyCode::ArrowUp) as i8 as f64;
    let brake = keys.pressed(KeyCode::ArrowDown) as i8 as f64;
    let mut tire = state.tire.take();
    let terrain = state.terrain.take();
    if let (Some(t), Some(tr)) = (tire.as_mut(), terrain.as_ref()) {
        t.set_drive_torque_n_m(throttle * TLAB_DRIVE_TORQUE_N_M);
        t.set_brake_torque_n_m(brake * TLAB_BRAKE_TORQUE_N_M);
        for _ in 0..steps {
            t.step_on_terrain(tr as &dyn molla_vehicle::Terrain, dt);
        }
    }
    state.tire = tire;
    state.terrain = terrain;
}

fn draw_terrain_lab_gizmos(
    sim: Res<Sim>,
    state: Res<TerrainLabExampleState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::TerrainLab {
        return;
    }
    // Render the heightfield at its actual world position. Zone-coloured
    // so bumps / hole / washboard / ramp read distinctly.
    let bumps_color = Color::srgb(0.40, 0.60, 0.85);
    let hole_color = Color::srgb(0.30, 0.18, 0.10);
    let washboard_color = Color::srgb(0.85, 0.45, 0.25);
    let ramp_color = Color::srgb(0.55, 0.78, 0.40);
    let flat_color = Color::srgb(0.45, 0.50, 0.55);
    let pick_color = |x_world: f32| -> Color {
        let xf = x_world as f64;
        if (xf - TLAB_HOLE_CENTER_X).abs() < TLAB_HOLE_HALF_WIDTH_M {
            return hole_color;
        }
        if TLAB_BUMP_CENTERS_X
            .iter()
            .any(|&cx| (xf - cx).abs() < TLAB_BUMP_HALF_WIDTH_M)
        {
            return bumps_color;
        }
        let (wb_x0, wb_x1) = TLAB_WASHBOARD_RANGE_X;
        if xf > wb_x0 && xf < wb_x1 {
            return washboard_color;
        }
        if xf >= TLAB_RAMP_START_X {
            return ramp_color;
        }
        flat_color
    };
    for (a, b) in &state.grid_segments {
        gizmos.line(*a, *b, pick_color(0.5 * (a.x + b.x)));
    }

    // Render the FEM tire as a wireframe of its surface edges so the
    // user sees the actual deforming mesh, not a cosmetic ring.
    //
    // The FEM nodes don't physically rotate — only the contact-bristle
    // slip is computed *as if* the tread were spinning, then the hub
    // translates from the bristle reaction. Rolling is shown with the
    // BeamNG/AC render-time rotation: `nodes_rendered()` rigidly spins
    // the deformed mesh about the hub by `hub_angle` for display only,
    // so the wheel visibly rolls while the physics stays in the
    // non-rotating tire frame. Trade-off (PROGRESS.md): the deformation
    // rotates with the mesh instead of staying glued to world-bottom;
    // elastic relaxation makes that read as material flowing through
    // the contact patch. The contact highlight below keys off the
    // rendered height, which still bottoms out at the terrain.
    let Some(tire) = state.tire.as_ref() else {
        return;
    };
    let nodes = tire.nodes_rendered();
    let surface = tire.surface_edges();
    let body = Color::srgb(0.85, 0.85, 0.90);
    let contact = Color::srgb(1.0, 0.92, 0.30);
    for &(a, b) in surface {
        let pa = nodes[a];
        let pb = nodes[b];
        // Light up tread-band edges when the node is in or near contact
        // with the terrain so the contact patch is obvious as it crosses
        // bumps.
        let color = if pa[1] <= 0.04 || pb[1] <= 0.04 {
            contact
        } else {
            body
        };
        gizmos.line(
            Vec3::new(pa[0] as f32, pa[1] as f32, pa[2] as f32),
            Vec3::new(pb[0] as f32, pb[1] as f32, pb[2] as f32),
            color,
        );
    }

    // PLAN_TOURQUE Stage E HUD — make the torque API observable:
    //   1. A spinning spoke at the hub, rotated by `hub_angle`. When
    //      `hub_omega > 0`, the spoke visibly turns — that is the spin
    //      DOF doing its job.
    //   2. A drive-torque arrow on the spin axis (orange) sized by
    //      |drive_torque| with its tip pointed forward.
    //   3. A contact-reaction arrow (cyan) drawn from the contact patch
    //      centroid, length proportional to |last_hub_reaction_xyz|,
    //      direction = the published reaction's XZ direction.
    //   4. The standard heading badge above the hub.
    let hub = tire.hub();
    let hub_pos = Vec3::new(hub[0] as f32, hub[1] as f32, hub[2] as f32);
    let spoke_radius = (TLAB_TIRE_RADIUS * 0.92) as f32;
    // hub_angle integrates from positive omega (forward rolling). The
    // visual rotation here mirrors the spin convention from the CPU
    // contact code (positive ω → top of wheel +X), so the matrix
    // applied below is `[cos, sin; −sin, cos]` (clockwise looking
    // from +Z).
    let theta = tire.hub_angle() as f32;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    // Six radial spokes so the rotation is unambiguous even at low
    // frame rates. They DO NOT touch the FEM mesh — they are a
    // separate visual indicator. The FEM mesh keeps its physics-true
    // orientation (deformation stays at the bottom).
    let spoke_color = Color::srgb(1.0, 0.85, 0.20);
    let spoke_count = 6;
    let mid_color = Color::srgb(0.30, 0.85, 0.95); // every other spoke
    for k in 0..spoke_count {
        let base_a = k as f32 * std::f32::consts::TAU / spoke_count as f32;
        // Rotate the base offset by the spin matrix.
        let bx = spoke_radius * base_a.cos();
        let by = spoke_radius * base_a.sin();
        let rx = cos_t * bx + sin_t * by;
        let ry = -sin_t * bx + cos_t * by;
        let tip = hub_pos + Vec3::new(rx, ry, 0.0);
        let color = if k % 2 == 0 { spoke_color } else { mid_color };
        gizmos.line(hub_pos, tip, color);
    }

    // Drive-torque arrow (forward of the hub, length scales with torque
    // magnitude). Red = brake, green = drive, blank when both are zero.
    let drive_torque = tire.drive_torque_n_m() as f32;
    let brake_torque = tire.brake_torque_n_m() as f32;
    let torque_color = if drive_torque > 0.0 {
        Color::srgb(0.40, 0.85, 0.45)
    } else if brake_torque > 0.0 {
        Color::srgb(0.85, 0.45, 0.40)
    } else {
        Color::srgb(0.5, 0.5, 0.5)
    };
    let torque_len = ((drive_torque + brake_torque).abs() / 200.0).clamp(0.0, 0.6);
    if torque_len > 0.02 {
        let tip = hub_pos + Vec3::new(torque_len, 0.0, 0.0);
        gizmos.line(hub_pos, tip, torque_color);
    }

    // Contact-reaction arrow from the contact patch centroid in XZ.
    let reaction = tire.last_hub_reaction_xyz();
    let reaction_xz = [reaction[0] as f32, reaction[2] as f32];
    let reaction_mag = (reaction_xz[0] * reaction_xz[0] + reaction_xz[1] * reaction_xz[1]).sqrt();
    if reaction_mag > 1.0 {
        let scale = (reaction_mag / 1500.0).clamp(0.05, 0.8);
        let dir_x = reaction_xz[0] / reaction_mag;
        let dir_z = reaction_xz[1] / reaction_mag;
        let foot = Vec3::new(hub_pos.x, 0.005, hub_pos.z);
        let tip = foot + Vec3::new(dir_x * scale, 0.0, dir_z * scale);
        gizmos.line(foot, tip, Color::srgb(0.30, 0.85, 0.95));
    }

    // Heading badge above the hub.
    let above = hub_pos + Vec3::new(0.0, 0.55, 0.0);
    gizmos.line(
        above,
        above + Vec3::new(0.80, 0.0, 0.0),
        Color::srgb(0.95, 0.95, 0.97),
    );
}
