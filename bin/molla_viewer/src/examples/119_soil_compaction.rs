// S2 — DEM multi-pass soil compaction. A heavy kinematic roller
// sweeps back and forth across a `GranularBed` patch; on every
// completed pass we sample the local rut depth at the centre and
// plot the rut-depth-vs-pass curve. The bed compacts more on each
// pass with diminishing returns — the canonical proctor-style
// compaction curve made visible.

fn build_soil_compaction_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("soil compaction model"), Vec::new())
}

/// Per-frame state for `Scene::SoilCompaction`.
#[derive(Resource)]
struct SoilCompactionState {
    bed: Option<molla_vehicle::GranularBed>,
    /// Initial mean grain-top height (m) — measured at scene entry
    /// so per-pass rut depth is referenced to the loose surface.
    initial_surface: f64,
    /// Kinematic wheel/roller world position (m).
    roller_pos: [f64; 3],
    /// Roller travel direction along X (`+1` or `-1`).
    roller_dir: f64,
    /// Roller radius (m).
    roller_radius: f64,
    /// Visual/contact width along Z (m).  The DEM backend takes a
    /// spherical obstacle, but the viewer draws it as a wheel/roller.
    roller_width: f64,
    /// Small penetration below the loose surface so the wheel actually
    /// compacts instead of hovering through `height()` sampling holes.
    roller_sinkage: f64,
    /// Roller travel speed (m/s).
    roller_speed: f64,
    /// Rolling phase, used only for visual spokes/tread motion.
    roller_angle: f64,
    /// Soft-x bounds of the roller sweep.
    x_min: f64,
    x_max: f64,
    /// Number of completed half-passes (one direction reversal each).
    pass_count: u32,
    /// `(pass_index, rut_depth)` samples plotted as a bar chart.
    rut_history: Vec<(u32, f32)>,
    /// Have we settled the bed under gravity yet?
    settled: bool,
    /// Frames spent settling so far.
    settle_frames: u32,
}

impl Default for SoilCompactionState {
    fn default() -> Self {
        Self {
            bed: None,
            initial_surface: 0.0,
            roller_pos: [-1.2, 0.75, 0.0],
            roller_dir: 1.0,
            roller_radius: 0.28,
            roller_width: 0.42,
            roller_sinkage: 0.055,
            roller_speed: 1.1,
            roller_angle: 0.0,
            x_min: -1.25,
            x_max: 1.25,
            pass_count: 0,
            rut_history: Vec::new(),
            settled: false,
            settle_frames: 0,
        }
    }
}

const COMPACTION_DT: f64 = 5.0e-4;
const COMPACTION_SUBSTEPS: u32 = 16;

fn soil_compaction_surface_under_wheel(
    bed: &molla_vehicle::GranularBed,
    x: f64,
    z: f64,
    width: f64,
) -> f64 {
    // `GranularBed::height` intentionally queries a very narrow column.
    // A moving wheel rarely lands exactly on a grain centre, so use the
    // whole contact footprint for the visual/roller height.  This keeps
    // the wheel visibly above the soil while still pressing the top layer.
    let x_window = (bed.radius * 2.4).max(0.10);
    let z_window = (width * 0.55).max(bed.radius);
    bed.grains()
        .iter()
        .filter(|g| (g[0] - x).abs() <= x_window && (g[2] - z).abs() <= z_window)
        .map(|g| g[1] + bed.radius)
        .fold(bed.floor_y, f64::max)
}

fn update_soil_compaction_example(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut state: ResMut<SoilCompactionState>,
) {
    if sim.scene != Scene::SoilCompaction {
        return;
    }
    if state.bed.is_none() {
        // 2.4 m × 0.6 m × 0.30 m bed of grain radius 0.05 m → roughly
        // 24 × 6 × 3 = 432 grains. A bit of jitter so the lattice
        // isn't artificially stable.
        let mut bed = molla_vehicle::GranularBed::lattice(
            [-1.2, 0.05, -0.30],
            [1.2, 0.50, 0.30],
            0.05,
            1.5, // grain mass
            0.65,
        );
        bed.jitter(0.008);
        state.bed = Some(bed);
        state.initial_surface = 0.0;
        state.pass_count = 0;
        state.rut_history.clear();
        state.settled = false;
        state.settle_frames = 0;
        state.roller_pos = [state.x_min, 0.75, 0.0];
        state.roller_dir = 1.0;
        state.roller_angle = 0.0;
    }
    if run.paused {
        return;
    }
    // Phase 1 — let the loose bed settle under gravity (no roller)
    // for ~0.6 s; then sample the initial surface height at x = 0.
    if !state.settled {
        let mut settled_now = false;
        let mut initial_surface = 0.0_f64;
        if let Some(b) = state.bed.as_mut() {
            for _ in 0..COMPACTION_SUBSTEPS {
                b.step(&[], COMPACTION_DT);
            }
        }
        state.settle_frames += 1;
        if state.settle_frames >= 80 {
            if let Some(b) = state.bed.as_ref() {
                initial_surface =
                    soil_compaction_surface_under_wheel(b, 0.0, 0.0, state.roller_width);
                settled_now = true;
            }
        }
        if settled_now {
            state.initial_surface = initial_surface;
            state.settled = true;
            state.rut_history.push((0, 0.0));
        }
        return;
    }
    // Phase 2 — sweep the roller back and forth. Each direction
    // change increments the pass counter and samples rut depth.
    let frame_dt = (COMPACTION_SUBSTEPS as f64) * COMPACTION_DT;
    let dx = state.roller_dir * state.roller_speed * frame_dt;
    let prev_x = state.roller_pos[0];
    let prev_y = state.roller_pos[1];
    let mut new_x = prev_x + dx;
    let mut crossed_end = false;
    let x_max = state.x_max;
    let x_min = state.x_min;
    if new_x >= x_max {
        new_x = x_max;
        state.roller_dir = -1.0;
        crossed_end = true;
    } else if new_x <= x_min {
        new_x = x_min;
        state.roller_dir = 1.0;
        crossed_end = true;
    }
    state.roller_pos[0] = new_x;
    state.roller_angle -= dx / state.roller_radius;
    // Keep the wheel centre above the sampled footprint surface.  The
    // small `roller_sinkage` means the bottom of the wheel presses into
    // the grains just enough to compact them, but the user still sees a
    // wheel rolling *over* the bed rather than a hidden plough.
    let roller_radius = state.roller_radius;
    let roller_x = state.roller_pos[0];
    if let Some(b) = state.bed.as_ref() {
        let surface =
            soil_compaction_surface_under_wheel(b, roller_x, 0.0, state.roller_width);
        state.roller_pos[1] = surface + roller_radius - state.roller_sinkage;
    }
    // Step the bed with the roller as an obstacle.
    let obstacle = molla_vehicle::GrainObstacle {
        center: state.roller_pos,
        radius: state.roller_radius,
        vel: [
            state.roller_dir * state.roller_speed,
            (state.roller_pos[1] - prev_y) / frame_dt,
            0.0,
        ],
    };
    if let Some(b) = state.bed.as_mut() {
        for _ in 0..COMPACTION_SUBSTEPS {
            b.step(std::slice::from_ref(&obstacle), COMPACTION_DT);
        }
    }
    if crossed_end {
        state.pass_count += 1;
        let initial = state.initial_surface;
        let pass = state.pass_count;
        let rut = if let Some(b) = state.bed.as_ref() {
            let surface_now = soil_compaction_surface_under_wheel(b, 0.0, 0.0, state.roller_width);
            (initial - surface_now).max(0.0) as f32
        } else {
            0.0
        };
        state.rut_history.push((pass, rut));
        if state.rut_history.len() > 24 {
            let drop = state.rut_history.len() - 24;
            state.rut_history.drain(0..drop);
        }
    }
}

fn draw_soil_compaction_wheel(
    gizmos: &mut Gizmos,
    centre: Vec3,
    radius: f32,
    width: f32,
    angle: f32,
    color: Color,
) {
    let rim_color = color;
    let side_color = Color::srgb(0.30, 0.34, 0.40);
    let tread_color = Color::srgb(0.10, 0.12, 0.14);
    let hub_color = Color::srgb(0.85, 0.85, 0.90);
    let half_w = width * 0.5;
    let segments = 48;
    for side_z in [-half_w, half_w] {
        for i in 0..segments {
            let a0 = angle + (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a1 = angle + ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;
            let p0 = centre + Vec3::new(radius * a0.cos(), radius * a0.sin(), side_z);
            let p1 = centre + Vec3::new(radius * a1.cos(), radius * a1.sin(), side_z);
            gizmos.line(p0, p1, rim_color);
        }
    }
    for i in 0..12 {
        let a = angle + (i as f32 / 12.0) * std::f32::consts::TAU;
        let left = centre + Vec3::new(radius * a.cos(), radius * a.sin(), -half_w);
        let right = centre + Vec3::new(radius * a.cos(), radius * a.sin(), half_w);
        gizmos.line(left, right, tread_color);
    }
    for i in 0..8 {
        let a = angle + (i as f32 / 8.0) * std::f32::consts::TAU;
        let rim = centre + Vec3::new(radius * 0.88 * a.cos(), radius * 0.88 * a.sin(), 0.0);
        gizmos.line(centre, rim, side_color);
    }
    gizmos.sphere(centre, radius * 0.16, hub_color);
}

fn draw_soil_compaction_gizmos(
    sim: Res<Sim>,
    state: Res<SoilCompactionState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::SoilCompaction {
        return;
    }
    // ── The grains ─────────────────────────────────────────────────
    if let Some(b) = state.bed.as_ref() {
        let grain_color = Color::srgb(0.78, 0.68, 0.48);
        let r = b.radius as f32 * 0.85;
        for g in b.grains() {
            gizmos.sphere(
                Vec3::new(g[0] as f32, g[1] as f32, g[2] as f32),
                r,
                grain_color,
            );
        }
    }
    // ── The moving wheel/roller above the soil ─────────────────────
    let roller_center = Vec3::new(
        state.roller_pos[0] as f32,
        state.roller_pos[1] as f32,
        state.roller_pos[2] as f32,
    );
    draw_soil_compaction_wheel(
        &mut gizmos,
        roller_center,
        state.roller_radius as f32,
        state.roller_width as f32,
        state.roller_angle as f32,
        Color::srgb(0.55, 0.65, 0.95),
    );
    // A faint contact patch marker makes it obvious what portion of
    // the soil is being compacted this frame.
    gizmos.line(
        roller_center + Vec3::new(0.0, -(state.roller_radius as f32), -(state.roller_width as f32) * 0.5),
        roller_center + Vec3::new(0.0, -(state.roller_radius as f32), (state.roller_width as f32) * 0.5),
        Color::srgb(1.0, 0.75, 0.25),
    );
    // Travel-line under the roller showing the sweep range.
    let path_color = Color::srgb(0.30, 0.35, 0.40);
    gizmos.line(
        Vec3::new(state.x_min as f32, 0.001, 0.0),
        Vec3::new(state.x_max as f32, 0.001, 0.0),
        path_color,
    );
    // ── Rut-depth-vs-pass bar chart ─────────────────────────────────
    if !state.rut_history.is_empty() {
        let plot_origin = Vec3::new(3.0, 1.4, 0.0);
        let plot_w = 4.0_f32; // along Z (pass-index axis)
        let plot_h = 1.0_f32; // along Y (rut-depth axis, plotted downward)
        let pass_max = state.rut_history.last().map(|(p, _)| *p).unwrap_or(1) as f32;
        let rut_max = state
            .rut_history
            .iter()
            .map(|&(_, r)| r)
            .fold(0.05_f32, f32::max);
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
        let bar_color = Color::srgb(0.95, 0.62, 0.20);
        for &(pass, rut) in &state.rut_history {
            let z = (pass as f32 / pass_max.max(1.0)) * plot_w;
            let bar_top = plot_origin + Vec3::new(0.0, 0.0, z);
            let bar_bot =
                plot_origin + Vec3::new(0.0, -(rut / rut_max).clamp(0.0, 1.0) * plot_h, z);
            gizmos.line(bar_top, bar_bot, bar_color);
        }
    }
    // ── Initial-surface reference line ─────────────────────────────
    let ref_color = Color::srgb(0.30, 0.85, 0.55);
    gizmos.line(
        Vec3::new(state.x_min as f32, state.initial_surface as f32, 0.0),
        Vec3::new(state.x_max as f32, state.initial_surface as f32, 0.0),
        ref_color,
    );
}
