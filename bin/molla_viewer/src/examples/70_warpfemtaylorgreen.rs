// ── Scene 70: WarpFemTaylorGreen (Stokes vortex IC) ───────────────
struct WarpFemTaylorGreenState {
    sim_t: f64,
    nu: f64,
    /// Decay magnitude (analytic): exp(-8 π² ν t).
    decay: f64,
    user_reset: bool,
}
impl Default for WarpFemTaylorGreenState {
    fn default() -> Self {
        Self {
            sim_t: 0.0,
            nu: 0.01,
            decay: 1.0,
            user_reset: false,
        }
    }
}

fn update_warp_fem_taylor(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemTaylorGreen {
        return;
    }
    let st = &mut warp.fem_taylor;
    if st.user_reset {
        st.sim_t = 0.0;
        st.user_reset = false;
    }
    if run.paused {
        return;
    }
    st.sim_t += time.delta_secs() as f64;
    let p = std::f64::consts::PI;
    st.decay = (-8.0 * p * p * st.nu * st.sim_t).exp();
}

fn draw_warp_fem_taylor_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpFemTaylorGreen {
        return;
    }
    let st = &warp.fem_taylor;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    let n = 16;
    let p = std::f64::consts::PI;
    for j in 0..n {
        for i in 0..n {
            let x = (i as f64 + 0.5) / n as f64;
            let y = (j as f64 + 0.5) / n as f64;
            let u = -((2.0 * p * x).cos() * (2.0 * p * y).sin()) * st.decay;
            let v = ((2.0 * p * x).sin() * (2.0 * p * y).cos()) * st.decay;
            let from = origin + Vec3::new(x as f32, 0.0, y as f32);
            let to = from + Vec3::new(u as f32 * 0.05, 0.0, v as f32 * 0.05);
            gizmos.line(from, to, Color::srgb(0.4, 0.95, 0.6));
        }
    }
}

fn warp_fem_taylor_pane(ui: &mut egui::Ui, st: &mut WarpFemTaylorGreenState) {
    ui.label("Warp FEM Taylor-Green — port of fem/example_taylor_green.py");
    ui.separator();
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    readout(ui, "Decay exp(-8π²νt)", &format!("{:.3e}", st.decay));
    ui.label("Viscosity ν");
    let _ = ui.add(egui::Slider::new(&mut st.nu, 0.001..=0.5).logarithmic(true));
    if ui.button("Reset t").clicked() {
        st.user_reset = true;
    }
    ui.small("Analytic Taylor-Green vortex. Velocity arrows shrink");
    ui.small("with the exact decay envelope.");
}

