// ── Scene 110: WarpFemMixedElasticity ─────────────────────────────
//
// Reduced pressure/displacement split for Warp's mixed-elasticity example.
// The displacement-only proxy deliberately exhibits volumetric locking as
// ν→0.5; the mixed proxy keeps the deviatoric response finite and reports a
// bounded pressure residual.
struct WarpFemMixedElasticityState {
    poisson: f64,
    load: f64,
    disp_only_tip: f64,
    mixed_tip: f64,
    pressure_residual: f64,
    user_reset: bool,
}

impl Default for WarpFemMixedElasticityState {
    fn default() -> Self {
        let mut me = Self {
            poisson: 0.49,
            load: 1.0,
            disp_only_tip: 0.0,
            mixed_tip: 0.0,
            pressure_residual: 0.0,
            user_reset: true,
        };
        me.recompute();
        me
    }
}

fn mixed_elasticity_reduced_response(poisson: f64, load: f64) -> (f64, f64, f64) {
    let nu = poisson.clamp(0.0, 0.499);
    let youngs = 1.0;
    let mu = youngs / (2.0 * (1.0 + nu));
    let lambda = youngs * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    // Volumetric locking proxy: displacement-only stiffness grows with λ.
    let disp_only_stiffness = 12.0 * mu + lambda;
    let disp_only_tip = load / disp_only_stiffness.max(1.0e-12);

    // Mixed method proxy: pressure carries the volumetric part, so the tip
    // response is governed mostly by deviatoric stiffness and remains finite
    // as λ→∞. The residual scales with 1/λ and therefore shrinks near the
    // incompressible limit.
    let mixed_stiffness = 12.0 * mu + 0.25;
    let mixed_tip = load / mixed_stiffness.max(1.0e-12);
    let pressure_residual = (load.abs() / (lambda + 12.0 * mu)).max(0.0);
    (disp_only_tip, mixed_tip, pressure_residual)
}

impl WarpFemMixedElasticityState {
    fn recompute(&mut self) {
        let (disp, mixed, pres) = mixed_elasticity_reduced_response(self.poisson, self.load);
        self.disp_only_tip = disp;
        self.mixed_tip = mixed;
        self.pressure_residual = pres;
        self.user_reset = false;
    }
}

fn update_warp_fem_mixed_elasticity(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemMixedElasticity {
        return;
    }
    let st = &mut warp.fem_mixed_elasticity;
    if st.user_reset {
        st.recompute();
    }
}

fn draw_warp_fem_mixed_elasticity_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpFemMixedElasticity {
        return;
    }
    let st = &warp.fem_mixed_elasticity;
    let base_y = 0.35_f32;
    let beam_len = 1.2_f32;
    let scale = 1.6_f32;
    for (row, (z, tip, col)) in [
        (
            -0.25_f32,
            st.disp_only_tip as f32,
            Color::srgb(0.95, 0.35, 0.25),
        ),
        (0.25_f32, st.mixed_tip as f32, Color::srgb(0.25, 0.75, 0.95)),
    ]
    .iter()
    .enumerate()
    {
        let y_tip = base_y - tip * scale;
        let x0 = -beam_len * 0.5;
        let x1 = beam_len * 0.5;
        let z = *z;
        let y0 = base_y;
        let label_tick = if row == 0 { 0.05 } else { -0.05 };
        gizmos.line(Vec3::new(x0, y0, z), Vec3::new(x1, y_tip, z), *col);
        gizmos.line(
            Vec3::new(x0, y0 - 0.12, z),
            Vec3::new(x0, y0 + 0.12, z),
            Color::srgb(0.7, 0.7, 0.75),
        );
        gizmos.line(
            Vec3::new(x1, y_tip, z - label_tick),
            Vec3::new(x1, y_tip, z + label_tick),
            *col,
        );
    }
}

fn warp_fem_mixed_elasticity_pane(ui: &mut egui::Ui, st: &mut WarpFemMixedElasticityState) {
    ui.label("Warp FEM mixed elasticity — reduced pressure split");
    ui.separator();
    readout(ui, "ν", &format!("{:.3}", st.poisson));
    readout(ui, "u tip only", &format!("{:.4}", st.disp_only_tip));
    readout(ui, "u tip mixed", &format!("{:.4}", st.mixed_tip));
    readout(ui, "p residual", &format!("{:.2e}", st.pressure_residual));
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut st.poisson, 0.0..=0.499).text("Poisson ν"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut st.load, 0.1..=2.0).text("load"))
        .changed();
    if changed || ui.button("Recompute").clicked() {
        st.user_reset = true;
    }
    ui.small("Red = displacement-only locking proxy; blue = mixed pressure/displacement proxy.");
    ui.small("Full mixed FE pressure solve and GPU parity remain future work.");
}

