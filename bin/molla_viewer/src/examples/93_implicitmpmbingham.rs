// ═══ Scene 93: ImplicitMpmBingham (Stage 39A) ═══════════════════════════
//
// A Bingham-fluid blob (CPU implicit MPM) drops onto a fixed floor.
// The pane exposes `yield_stress` + `viscosity` sliders, a Reset
// button, and a Pause toggle. Particles are rendered as small colored
// spheres via `gizmos.sphere`; colour goes from blue (rigid plug) to
// red (flowing) based on the local strain-rate magnitude. The grid is
// not drawn — only the particle cloud.

struct ImplicitMpmBinghamState {
    cfg: molla::compute::implicit_mpm::ImplicitMpmConfig,
    state: molla::compute::implicit_mpm::MpmState,
    paused: bool,
    user_reset: bool,
    n_steps: usize,
    last_ke: f64,
    last_yielded: usize,
    last_step_us: f64,
}

impl Default for ImplicitMpmBinghamState {
    fn default() -> Self {
        let cfg = implicit_mpm_bingham_default_cfg();
        let state = implicit_mpm_bingham_initial_state(&cfg);
        Self {
            cfg,
            state,
            paused: false,
            user_reset: false,
            n_steps: 0,
            last_ke: 0.0,
            last_yielded: 0,
            last_step_us: 0.0,
        }
    }
}

fn implicit_mpm_bingham_default_cfg() -> molla::compute::implicit_mpm::ImplicitMpmConfig {
    molla::compute::implicit_mpm::ImplicitMpmConfig {
        grid_size: 24,
        cell_size: 0.05,
        grid_origin: [-0.60, -0.05, -0.60],
        dt: 1.0 / 60.0,
        gravity_y: -9.81,
        yield_stress: 500.0,
        viscosity: 20.0,
        density: 1000.0,
        particle_volume: 1.0e-4,
        floor_y: 0.0,
        floor_friction: 0.6,
        max_gs_iters: 16,
        gs_tol: 1.0e-3,
        max_newton_iters: 1,
    }
}

fn implicit_mpm_bingham_initial_state(
    cfg: &molla::compute::implicit_mpm::ImplicitMpmConfig,
) -> molla::compute::implicit_mpm::MpmState {
    // Sphere of radius 0.12 centred at (0, 0.30, 0); sample on a
    // regular cell-aligned grid inside the sphere.
    let center = MVec3::new(0.0, 0.30, 0.0);
    let radius = 0.12_f64;
    let dx = cfg.cell_size * 0.6;
    let n = (radius / dx).ceil() as i32;
    let mut pos: Vec<MVec3> = Vec::new();
    for iz in -n..=n {
        for iy in -n..=n {
            for ix in -n..=n {
                let off = MVec3::new(ix as f64 * dx, iy as f64 * dx, iz as f64 * dx);
                if off.length() <= radius {
                    pos.push(center + off);
                }
            }
        }
    }
    molla::compute::implicit_mpm::MpmState::new(&pos, &[])
}

fn update_implicit_mpm_bingham(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::ImplicitMpmBingham {
        return;
    }
    let st = &mut warp.implicit_mpm_bingham;
    if st.user_reset {
        st.state = implicit_mpm_bingham_initial_state(&st.cfg);
        st.n_steps = 0;
        st.last_ke = 0.0;
        st.last_yielded = 0;
        st.user_reset = false;
        return;
    }
    if run.paused || st.paused {
        return;
    }
    let t0 = std::time::Instant::now();
    let report = molla::compute::implicit_mpm::implicit_mpm_step(&mut st.state, &st.cfg, &mut []);
    let dt_us = t0.elapsed().as_micros() as f64;
    // EMA on step time so the readout doesn't flicker.
    st.last_step_us = if st.last_step_us == 0.0 {
        dt_us
    } else {
        0.9 * st.last_step_us + 0.1 * dt_us
    };
    st.n_steps += 1;
    st.last_ke = st.state.kinetic_energy(st.cfg.particle_mass());
    st.last_yielded = report.yielded_count;
}

fn implicit_mpm_bingham_particle_color(speed: f64, peak_speed: f64) -> Color {
    // Map speed [0, peak_speed] → blue (plug) → red (flowing).
    let t = (speed / peak_speed.max(1.0e-6)).clamp(0.0, 1.0) as f32;
    Color::srgb(0.15 + 0.80 * t, 0.30, 0.85 - 0.70 * t)
}

fn draw_implicit_mpm_bingham_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::ImplicitMpmBingham {
        return;
    }
    let st = &warp.implicit_mpm_bingham;
    // Floor plane outline (thin grey rectangle one cell above floor).
    let half = (st.cfg.grid_size as f64 * st.cfg.cell_size * 0.5) as f32;
    let fy = st.cfg.floor_y as f32;
    let grey = Color::srgb(0.45, 0.45, 0.5);
    let corners = [
        Vec3::new(-half, fy, -half),
        Vec3::new(half, fy, -half),
        Vec3::new(half, fy, half),
        Vec3::new(-half, fy, half),
    ];
    for i in 0..4 {
        gizmos.line(corners[i], corners[(i + 1) % 4], grey);
    }
    // Particles, colored by local |v|.
    let peak_speed = st
        .state
        .vel
        .iter()
        .map(|v| v.length())
        .fold(0.0_f64, f64::max)
        .max(0.1);
    for (i, p) in st.state.pos.iter().enumerate() {
        let v = st.state.vel[i];
        let pos = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
        let color = implicit_mpm_bingham_particle_color(v.length(), peak_speed);
        gizmos.sphere(pos, 0.012, color);
    }
    // Bounding box of the grid in dim grey.
    let g = st.cfg.grid_size as f32 * st.cfg.cell_size as f32;
    let o = Vec3::new(
        st.cfg.grid_origin[0] as f32,
        st.cfg.grid_origin[1] as f32,
        st.cfg.grid_origin[2] as f32,
    );
    let dim = Color::srgb(0.25, 0.25, 0.30);
    // Draw 4 vertical edges + top/bottom rectangles.
    let bottom = [
        o,
        o + Vec3::X * g,
        o + Vec3::X * g + Vec3::Z * g,
        o + Vec3::Z * g,
    ];
    let top: [Vec3; 4] = bottom.map(|p| p + Vec3::Y * g);
    for i in 0..4 {
        gizmos.line(bottom[i], bottom[(i + 1) % 4], dim);
        gizmos.line(top[i], top[(i + 1) % 4], dim);
        gizmos.line(bottom[i], top[i], dim);
    }
}

fn implicit_mpm_bingham_pane(ui: &mut egui::Ui, st: &mut ImplicitMpmBinghamState) {
    ui.label("Implicit MPM with Bingham rheology — Stage 39A");
    ui.separator();
    readout(ui, "Particles", &format!("{}", st.state.n()));
    readout(ui, "Step", &format!("{}", st.n_steps));
    readout(ui, "KE total", &format!("{:.3e}", st.last_ke));
    readout(ui, "Yielded (plug)", &format!("{}", st.last_yielded));
    readout(ui, "ms/step", &format!("{:.2}", st.last_step_us * 1.0e-3));
    ui.separator();
    ui.label("Bingham rheology");
    let _ = ui.add(
        egui::Slider::new(&mut st.cfg.yield_stress, 0.0..=2000.0)
            .text("τ_y (Pa)")
            .logarithmic(true),
    );
    let _ = ui.add(
        egui::Slider::new(&mut st.cfg.viscosity, 0.1..=200.0)
            .text("μ_b (Pa·s)")
            .logarithmic(true),
    );
    let _ = ui.add(egui::Slider::new(&mut st.cfg.gravity_y, -30.0..=0.0).text("gravity_y"));
    let _ = ui.add(
        egui::Slider::new(&mut st.cfg.dt, 0.001..=0.05)
            .text("dt (s)")
            .logarithmic(true),
    );
    let _ = ui.add(egui::Slider::new(&mut st.cfg.floor_friction, 0.0..=1.0).text("floor friction"));
    ui.separator();
    ui.label("Solver");
    let _ = ui.add(egui::Slider::new(&mut st.cfg.max_gs_iters, 1..=64).text("max GS iters"));
    let _ = ui.add(
        egui::Slider::new(&mut st.cfg.gs_tol, 1.0e-6..=1.0e-1)
            .text("GS tol")
            .logarithmic(true),
    );
    ui.separator();
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset blob").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Yellow→red particles = flowing; blue = rigid plug. The implicit \
         grid solve uses a colored Gauss-Seidel relaxation over the 7-pt \
         stencil; per-particle Bingham return mapping projects trial \
         strain rates onto the yield surface and a Bingham-aware Coulomb \
         floor friction arrests residual lateral flow at rest.",
    );
}

