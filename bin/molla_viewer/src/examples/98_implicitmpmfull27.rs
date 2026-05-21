// ═══ Scene 98: ImplicitMpmFull27 (Stage 40) ═════════════════════════════
//
// Same Bingham blob as Scene 93 but with denser particles and the
// full 27-point Newton Hessian (`HessianStencil::Full27`). The
// Stage 40 GPU device kernel is the v3 follow-up; this scene uses
// the CPU path that mirrors the kernel's per-iter cost.

struct ImplicitMpmFull27State {
    cfg: molla::compute::implicit_mpm::ImplicitMpmConfig,
    state: molla::compute::implicit_mpm::MpmState,
    paused: bool,
    user_reset: bool,
    n_steps: usize,
    last_ke: f64,
    last_yielded: usize,
    last_step_us: f64,
}

impl Default for ImplicitMpmFull27State {
    fn default() -> Self {
        let cfg = implicit_mpm_full27_default_cfg();
        let state = implicit_mpm_full27_initial_state(&cfg);
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

fn implicit_mpm_full27_default_cfg() -> molla::compute::implicit_mpm::ImplicitMpmConfig {
    molla::compute::implicit_mpm::ImplicitMpmConfig {
        grid_size: 28,
        cell_size: 0.045,
        grid_origin: [-0.63, -0.05, -0.63],
        dt: 1.0 / 60.0,
        gravity_y: -9.81,
        yield_stress: 350.0,
        viscosity: 15.0,
        density: 1000.0,
        particle_volume: 1.0e-4,
        floor_y: 0.0,
        floor_friction: 0.55,
        max_gs_iters: 24,
        gs_tol: 1.0e-3,
        max_newton_iters: 1,
    }
}

fn implicit_mpm_full27_initial_state(
    cfg: &molla::compute::implicit_mpm::ImplicitMpmConfig,
) -> molla::compute::implicit_mpm::MpmState {
    let center = MVec3::new(0.0, 0.35, 0.0);
    let radius = 0.15_f64;
    let dx = cfg.cell_size * 0.45;
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

fn update_implicit_mpm_full27(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::ImplicitMpmFull27 {
        return;
    }
    let st = &mut warp.implicit_mpm_full27;
    if st.user_reset {
        st.state = implicit_mpm_full27_initial_state(&st.cfg);
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
    let report = molla::compute::implicit_mpm::implicit_mpm_step_with_stencil(
        &mut st.state,
        &st.cfg,
        &mut [],
        molla::compute::implicit_mpm::HessianStencil::Full27,
    );
    let dt_us = t0.elapsed().as_micros() as f64;
    st.last_step_us = if st.last_step_us == 0.0 {
        dt_us
    } else {
        0.9 * st.last_step_us + 0.1 * dt_us
    };
    st.n_steps += 1;
    st.last_ke = st.state.kinetic_energy(st.cfg.particle_mass());
    st.last_yielded = report.yielded_count;
}

fn draw_implicit_mpm_full27_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::ImplicitMpmFull27 {
        return;
    }
    let st = &warp.implicit_mpm_full27;
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
        let t = (v.length() / peak_speed.max(1.0e-6)).clamp(0.0, 1.0) as f32;
        let color = Color::srgb(0.20 + 0.75 * t, 0.55 + 0.10 * (1.0 - t), 0.85 - 0.70 * t);
        gizmos.sphere(pos, 0.010, color);
    }
}

fn implicit_mpm_full27_pane(ui: &mut egui::Ui, st: &mut ImplicitMpmFull27State) {
    ui.label("Stage 40 — implicit MPM, full 27-point Newton Hessian");
    ui.separator();
    readout(ui, "Particles", &format!("{}", st.state.n()));
    readout(ui, "Step", &format!("{}", st.n_steps));
    readout(ui, "KE total", &format!("{:.3e}", st.last_ke));
    readout(ui, "Yielded (plug)", &format!("{}", st.last_yielded));
    readout(ui, "ms/step", &format!("{:.2}", st.last_step_us * 1.0e-3));
    ui.separator();
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
    let _ = ui.add(egui::Slider::new(&mut st.cfg.max_gs_iters, 1..=64).text("max GS iters"));
    ui.separator();
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset blob").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Same Bingham blob as #93, ~5k particles, full 27-point Newton \
         Hessian. Stage 40 v3 wires the device-side colored-GS sweep \
         into this exact path.",
    );
}

