// S1 — Bekker pressure-sinkage curve. A loading plate is pressed
// into a `DeformableTerrain` patch at increasing vertical load; the
// resulting (pressure, sinkage) samples are drawn as a 3-D line
// strip so the canonical `z = (p / stiffness)^(1/n)` curve becomes
// visible. The SCM terrain extends with the Bekker `n` exponent so
// the demo reproduces the textbook concave curve for sandy soil.

fn build_bekker_sinkage_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("bekker sinkage model"), Vec::new())
}

/// Per-frame state for `Scene::BekkerSinkage`.
#[derive(Resource)]
struct BekkerSinkageState {
    /// SCM deformable patch. Built on first entry; rebuilt on reset.
    terrain: Option<molla_vehicle::DeformableTerrain>,
    /// Current vertical load on the plate (N). Cycles 0 → max → 0.
    fz: f64,
    /// Sweep direction (`+1` ramps up, `-1` ramps back).
    sweep_dir: f64,
    /// Recorded `(pressure_kPa, sinkage_m)` samples — drawn as the
    /// canonical Bekker curve.
    curve: Vec<(f32, f32)>,
    /// Loading-plate side length (m). Match the SCM cell spacing so
    /// the load lands in a single cell (`apply_load` is per-cell).
    plate_side: f32,
    /// Live sinkage at the plate centre (m). Set every frame.
    sinkage: f32,
    /// Bekker exponent `n` used by the terrain.
    bekker_n: f64,
}

impl Default for BekkerSinkageState {
    fn default() -> Self {
        Self {
            terrain: None,
            fz: 0.0,
            sweep_dir: 1.0,
            curve: Vec::new(),
            plate_side: 0.30, // matches the SCM cell spacing below
            sinkage: 0.0,
            bekker_n: 0.8, // sandy soil — sub-linear curve
        }
    }
}

// Drive the SCM patch: ramp the load up over ~6 s, hold, ramp down,
// recording the (pressure, sinkage) pair every frame.
fn update_bekker_sinkage_example(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut state: ResMut<BekkerSinkageState>,
) {
    if sim.scene != Scene::BekkerSinkage {
        return;
    }
    if state.terrain.is_none() {
        let mut t = molla_vehicle::DeformableTerrain::flat(
            [-4.0, -4.0],
            0.30,
            27,
            27,
            0.0,
            8.0e5,
            0.9,
        );
        t.bekker_n = state.bekker_n;
        t.max_sinkage = 0.18;
        state.terrain = Some(t);
        state.curve.clear();
        state.fz = 0.0;
        state.sweep_dir = 1.0;
    }
    if run.paused {
        return;
    }
    let dt = time.delta_secs() as f64;
    let max_fz = 9.0e4_f64;
    let rate = max_fz / 6.0;
    state.fz += state.sweep_dir * rate * dt;
    if state.fz > max_fz {
        state.fz = max_fz;
        state.sweep_dir = -1.0;
        state.curve.clear();
    } else if state.fz < 0.0 {
        state.fz = 0.0;
        state.sweep_dir = 1.0;
    }
    // Snapshot scalars first so we can mutate `state.terrain` and
    // `state.curve` without aliasing other fields.
    let fz = state.fz;
    let plate_side = state.plate_side;
    let (sinkage_now, pressure_kpa) = {
        let Some(t) = state.terrain.as_mut() else {
            return;
        };
        t.apply_load(0.0, 0.0, fz);
        let s = t.sinkage_at(0.0, 0.0);
        let area = (plate_side as f64).powi(2);
        (s as f32, ((fz / area) / 1000.0) as f32)
    };
    state.sinkage = sinkage_now;
    state.curve.push((pressure_kpa, sinkage_now));
    if state.curve.len() > 1200 {
        let drop = state.curve.len() - 1200;
        state.curve.drain(0..drop);
    }
}

fn draw_bekker_sinkage_gizmos(
    sim: Res<Sim>,
    state: Res<BekkerSinkageState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::BekkerSinkage {
        return;
    }
    // ── The loading plate ──────────────────────────────────────────
    let half = state.plate_side * 0.5;
    let plate_y = -state.sinkage;
    let corners = [
        Vec3::new(-half, plate_y, -half),
        Vec3::new(half, plate_y, -half),
        Vec3::new(half, plate_y, half),
        Vec3::new(-half, plate_y, half),
    ];
    let plate_color = Color::srgb(0.90, 0.80, 0.35);
    for i in 0..4 {
        gizmos.line(corners[i], corners[(i + 1) % 4], plate_color);
    }
    // Vertical load arrow above the plate.
    let arrow_top = Vec3::new(0.0, 1.2, 0.0);
    let arrow_tip = Vec3::new(0.0, plate_y + 0.05, 0.0);
    let arrow_color = Color::srgb(0.95, 0.45, 0.30);
    gizmos.line(arrow_top, arrow_tip, arrow_color);
    // ── The deformed soil surface ──────────────────────────────────
    if let Some(t) = state.terrain.as_ref() {
        use molla_vehicle::Terrain;
        let row_color = Color::srgb(0.55, 0.45, 0.32);
        let half_extent_cells = 18i64;
        for ix in -half_extent_cells..half_extent_cells {
            let x0 = (ix as f64) * t.spacing;
            let x1 = ((ix + 1) as f64) * t.spacing;
            let h0 = t.height(x0, 0.0);
            let h1 = t.height(x1, 0.0);
            gizmos.line(
                Vec3::new(x0 as f32, h0 as f32, 0.0),
                Vec3::new(x1 as f32, h1 as f32, 0.0),
                row_color,
            );
        }
        for iz in -half_extent_cells..half_extent_cells {
            let z0 = (iz as f64) * t.spacing;
            let z1 = ((iz + 1) as f64) * t.spacing;
            let h0 = t.height(0.0, z0);
            let h1 = t.height(0.0, z1);
            gizmos.line(
                Vec3::new(0.0, h0 as f32, z0 as f32),
                Vec3::new(0.0, h1 as f32, z1 as f32),
                row_color,
            );
        }
    }
    // ── The (pressure, sinkage) curve ──────────────────────────────
    let plot_origin = Vec3::new(5.0, 1.6, 0.0);
    let plot_w = 4.0_f32; // metres along Z (pressure axis)
    let plot_h = 1.5_f32; // metres along Y (sinkage axis, plotted downward)
    let pressure_max_kpa = 1100.0_f32;
    let sinkage_max = state.terrain.as_ref().map_or(0.18, |t| t.max_sinkage as f32);
    let axis_color = Color::srgb(0.55, 0.55, 0.60);
    gizmos.line(
        plot_origin,
        plot_origin + Vec3::new(0.0, 0.0, plot_w),
        axis_color,
    );
    gizmos.line(
        plot_origin,
        plot_origin - Vec3::new(0.0, plot_h, 0.0),
        axis_color,
    );
    if state.curve.len() >= 2 {
        let curve_color = Color::srgb(0.30, 0.90, 0.65);
        let map = |p: f32, s: f32| -> Vec3 {
            let z = (p / pressure_max_kpa).clamp(0.0, 1.0) * plot_w;
            let y = -(s / sinkage_max).clamp(0.0, 1.0) * plot_h;
            plot_origin + Vec3::new(0.0, y, z)
        };
        let mut prev = map(state.curve[0].0, state.curve[0].1);
        for &(p, s) in &state.curve[1..] {
            let here = map(p, s);
            gizmos.line(prev, here, curve_color);
            prev = here;
        }
        // Current-sample marker.
        let last = state.curve.last().copied().unwrap();
        gizmos.sphere(map(last.0, last.1), 0.04, Color::srgb(1.0, 1.0, 0.4));
    }
}
