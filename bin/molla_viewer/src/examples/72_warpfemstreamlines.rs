// ── Scene 72: WarpFemStreamlines ──────────────────────────────────
struct WarpFemStreamlinesState {
    /// Tracer positions.
    tracers: Vec<Vec3>,
    sim_t: f64,
    user_reset: bool,
}
impl Default for WarpFemStreamlinesState {
    fn default() -> Self {
        Self {
            tracers: Vec::new(),
            sim_t: 0.0,
            user_reset: true,
        }
    }
}

fn warp_streamlines_velocity(p: Vec3) -> Vec3 {
    // Simple analytic vortex (poor man's lid-driven cavity).
    let r = Vec3::new(p.x - 0.5, 0.0, p.z - 0.5);
    Vec3::new(-r.z, 0.0, r.x) * 0.6
}

fn update_warp_fem_streamlines(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpFemStreamlines {
        return;
    }
    let st = &mut warp.fem_streamlines;
    if st.user_reset {
        st.tracers.clear();
        for j in 0..8 {
            for i in 0..8 {
                let x = (i as f32 + 0.5) / 8.0;
                let z = (j as f32 + 0.5) / 8.0;
                st.tracers.push(Vec3::new(x, 0.05, z));
            }
        }
        st.sim_t = 0.0;
        st.user_reset = false;
    }
    if run.paused {
        return;
    }
    let dt = time.delta_secs().min(0.05);
    st.sim_t += dt as f64;
    for p in st.tracers.iter_mut() {
        let v = warp_streamlines_velocity(*p);
        *p += v * dt;
        // Wrap into [0,1]^2 in xz.
        if p.x < 0.0 {
            p.x += 1.0;
        }
        if p.x > 1.0 {
            p.x -= 1.0;
        }
        if p.z < 0.0 {
            p.z += 1.0;
        }
        if p.z > 1.0 {
            p.z -= 1.0;
        }
    }
}

fn draw_warp_fem_streamlines_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemStreamlines {
        return;
    }
    let st = &warp.fem_streamlines;
    let origin = Vec3::new(-0.5, 0.05, -0.5);
    for &p in &st.tracers {
        let world = origin + Vec3::new(p.x, 0.0, p.z);
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(world),
            0.015,
            Color::srgb(0.3, 0.7, 0.95),
        );
    }
}

fn warp_fem_streamlines_pane(ui: &mut egui::Ui, st: &mut WarpFemStreamlinesState) {
    ui.label("Warp FEM streamlines — port of fem/example_streamlines.py");
    ui.separator();
    readout(ui, "Tracers", &format!("{}", st.tracers.len()));
    readout(ui, "t", &format!("{:.2} s", st.sim_t));
    if ui.button("Reset tracers").clicked() {
        st.user_reset = true;
    }
    ui.small("Tracers advect through analytic 2D vortex velocity field.");
}

