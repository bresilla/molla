// ═══ Scene 94: WarpStableFluid (Stage 39C) ═════════════════════════════
//
// 2-D Stam stable-fluid solver on a periodic 32×32 grid. A central dye
// source injects density each frame while a swirling body force keeps
// the velocity field unsteady. The velocity-magnitude scalar is drawn
// as a height-field of small upright gizmo segments, colour-mapped
// blue → red along the speed range. Pane: viscosity, source, swirl,
// reset.

const STABLE_FLUID_GRID: usize = 32;
const STABLE_FLUID_PLANE_HALF: f32 = 0.50;

struct WarpStableFluidState {
    state: molla::compute::stam_fluid::StamFluidState,
    cfg: molla::compute::stam_fluid::StamFluidConfig,
    source_amp: f32,
    swirl_amp: f32,
    paused: bool,
    user_reset: bool,
    steps: usize,
    last_max_speed: f32,
}

impl Default for WarpStableFluidState {
    fn default() -> Self {
        let mut cfg = molla::compute::stam_fluid::StamFluidConfig::default();
        cfg.grid_n = STABLE_FLUID_GRID;
        cfg.viscosity = 5.0e-4;
        cfg.dt = 1.0 / 60.0;
        let state = molla::compute::stam_fluid::StamFluidState::new(STABLE_FLUID_GRID);
        Self {
            state,
            cfg,
            source_amp: 6.0,
            swirl_amp: 4.0,
            paused: false,
            user_reset: false,
            steps: 0,
            last_max_speed: 0.0,
        }
    }
}

fn update_warp_stable_fluid(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpStableFluid {
        return;
    }
    let st = &mut warp.stable_fluid;
    if st.user_reset {
        st.state = molla::compute::stam_fluid::StamFluidState::new(st.cfg.grid_n);
        st.steps = 0;
        st.last_max_speed = 0.0;
        st.user_reset = false;
        return;
    }
    if run.paused || st.paused {
        return;
    }
    let n = st.cfg.grid_n;
    let mut force = vec![[0.0_f32; 2]; n * n];
    let mut source = vec![0.0_f32; n * n];
    // Centre source — small disk emitting dye.
    let cx = n as f32 * 0.5;
    let cy = n as f32 * 0.5;
    let inv_n = 1.0 / n as f32;
    for j in 0..n {
        for i in 0..n {
            let dx = i as f32 + 0.5 - cx;
            let dy = j as f32 + 0.5 - cy;
            let r2 = dx * dx + dy * dy;
            if r2 < 9.0 {
                source[j * n + i] = st.source_amp;
            }
            // Swirling body force: tangential to centre.
            let r = r2.sqrt().max(1.0);
            let tx = -dy / r;
            let ty = dx / r;
            let mag = st.swirl_amp * (-(r * inv_n).powi(2) * 8.0).exp();
            force[j * n + i][0] += tx * mag;
            force[j * n + i][1] += ty * mag;
        }
    }
    st.state.step_cpu(&st.cfg, &force, &source);
    st.steps += 1;
    let mut mx = 0.0_f32;
    for v in &st.state.u {
        let s = (v[0] * v[0] + v[1] * v[1]).sqrt();
        if s > mx {
            mx = s;
        }
    }
    st.last_max_speed = if st.last_max_speed == 0.0 {
        mx
    } else {
        0.9 * st.last_max_speed + 0.1 * mx
    };
}

fn draw_warp_stable_fluid_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpStableFluid {
        return;
    }
    let st = &warp.stable_fluid;
    let n = st.cfg.grid_n;
    let half = STABLE_FLUID_PLANE_HALF;
    let cell = (2.0 * half) / n as f32;
    let peak = st.last_max_speed.max(0.05);
    let outline = Color::srgb(0.30, 0.30, 0.35);
    let corners = [
        Vec3::new(-half, 0.0, -half),
        Vec3::new(half, 0.0, -half),
        Vec3::new(half, 0.0, half),
        Vec3::new(-half, 0.0, half),
    ];
    for i in 0..4 {
        gizmos.line(corners[i], corners[(i + 1) % 4], outline);
    }
    for j in 0..n {
        for i in 0..n {
            let v = st.state.u[j * n + i];
            let s = (v[0] * v[0] + v[1] * v[1]).sqrt();
            let h = (s / peak).clamp(0.0, 1.0);
            let t = h;
            let col = Color::srgb(0.15 + 0.80 * t, 0.30 + 0.20 * t, 0.95 - 0.80 * t);
            let x = -half + (i as f32 + 0.5) * cell;
            let z = -half + (j as f32 + 0.5) * cell;
            let base = Vec3::new(x, 0.0, z);
            let top = base + Vec3::new(0.0, h * 0.35, 0.0);
            gizmos.line(base, top, col);
        }
    }
    let stride = (n / 8).max(1);
    let arrow_col = Color::srgb(0.95, 0.95, 0.4);
    for j in (0..n).step_by(stride) {
        for i in (0..n).step_by(stride) {
            let v = st.state.u[j * n + i];
            let s = (v[0] * v[0] + v[1] * v[1]).sqrt();
            if s < 1.0e-4 {
                continue;
            }
            let x = -half + (i as f32 + 0.5) * cell;
            let z = -half + (j as f32 + 0.5) * cell;
            let scale = 0.02 / s.max(1.0e-3);
            let a = Vec3::new(x, 0.005, z);
            let b = a + Vec3::new(v[0] * scale, 0.0, v[1] * scale);
            gizmos.line(a, b, arrow_col);
        }
    }
}

fn warp_stable_fluid_pane(ui: &mut egui::Ui, st: &mut WarpStableFluidState) {
    ui.label("Stage 39C — 2D Stam stable fluid (periodic grid)");
    ui.separator();
    readout(ui, "Grid", &format!("{0}×{0}", st.cfg.grid_n));
    readout(ui, "Step", &format!("{}", st.steps));
    readout(ui, "Max |v|", &format!("{:.3}", st.last_max_speed));
    ui.separator();
    let _ = ui.add(
        egui::Slider::new(&mut st.cfg.viscosity, 1.0e-5..=1.0e-2)
            .text("ν viscosity")
            .logarithmic(true),
    );
    let _ = ui.add(egui::Slider::new(&mut st.source_amp, 0.0..=20.0).text("source amp"));
    let _ = ui.add(egui::Slider::new(&mut st.swirl_amp, 0.0..=20.0).text("swirl amp"));
    let _ = ui.add(egui::Slider::new(&mut st.cfg.dt, 0.001..=0.05).text("dt (s)"));
    ui.separator();
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Vertical bars = |velocity| height-field; yellow arrows trace \
         flow direction on a coarse sub-grid. A central disk emits dye \
         while a tangential body force shears the field.",
    );
}

