// ── Scene 75: WarpFemElasticOpt ───────────────────────────────────
struct WarpFemElasticOptState {
    /// Beam height parameter (optimised).
    height: f64,
    /// Stress proxy (smaller = better).
    last_stress: f64,
    iter: usize,
    learning_rate: f64,
    paused: bool,
    user_reset: bool,
}
impl Default for WarpFemElasticOptState {
    fn default() -> Self {
        Self {
            height: 0.4,
            last_stress: 0.0,
            iter: 0,
            learning_rate: 0.01,
            paused: false,
            user_reset: true,
        }
    }
}

fn update_warp_fem_elastic_opt(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemElasticOpt {
        return;
    }
    let st = &mut warp.fem_elastic_opt;
    if st.user_reset {
        st.height = 0.4;
        st.iter = 0;
        st.last_stress = 0.0;
        st.user_reset = false;
    }
    if st.paused {
        return;
    }
    // Surrogate stress model: small for h=0.5, growing for h<0.2 or h>0.8.
    let s = (st.height - 0.5).powi(2) * 100.0 + 1.0 / st.height.max(0.05);
    let grad = 200.0 * (st.height - 0.5) - 1.0 / st.height.max(0.05).powi(2);
    st.last_stress = s;
    st.height = (st.height - st.learning_rate * grad).clamp(0.1, 0.9);
    st.iter += 1;
}

fn warp_fem_elastic_opt_pane(ui: &mut egui::Ui, st: &mut WarpFemElasticOptState) {
    ui.label("Warp FEM elastic shape opt — port of fem/example_elastic_shape_optimization.py");
    ui.separator();
    readout(ui, "Iter", &format!("{}", st.iter));
    readout(ui, "Beam height h", &format!("{:.3} m", st.height));
    readout(ui, "Stress proxy", &format!("{:.3e}", st.last_stress));
    ui.label("Learning rate");
    let _ = ui.add(egui::Slider::new(&mut st.learning_rate, 0.001..=0.1).logarithmic(true));
    ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small("Surrogate model: optimal h ≈ 0.5; gradient descent.");
}

