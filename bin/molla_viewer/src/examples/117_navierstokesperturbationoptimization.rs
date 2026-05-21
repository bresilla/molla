// ═══ Scene 117: NavierStokesPerturbationOptimization — reduced NS opt ═

const NS_PERT_N: usize = 18;
const NS_PERT_ITERS: usize = 36;

struct NsPerturbationState {
    amp: f64,
    target_response: f64,
    lr: f64,
    response: f64,
    loss: f64,
    initial_loss: f64,
    field: Vec<f64>,
    iterations: usize,
    needs_rerun: bool,
}

impl Default for NsPerturbationState {
    fn default() -> Self {
        Self {
            amp: 0.05,
            target_response: 0.40,
            lr: 1.2,
            response: 0.0,
            loss: f64::INFINITY,
            initial_loss: f64::INFINITY,
            field: vec![0.0; NS_PERT_N * NS_PERT_N],
            iterations: 0,
            needs_rerun: true,
        }
    }
}

fn ns_perturbation_pattern(n: usize) -> Vec<f64> {
    let two_pi = std::f64::consts::TAU;
    let mut out = Vec::with_capacity(n * n);
    for j in 0..n {
        let y = j as f64 / (n - 1) as f64;
        for i in 0..n {
            let x = i as f64 / (n - 1) as f64;
            let shear = (std::f64::consts::PI * y).sin();
            let perturb = (two_pi * x).sin() * (two_pi * y).sin();
            out.push(shear * perturb);
        }
    }
    out
}

fn ns_perturbation_rollout(amp: f64, n: usize) -> Vec<f64> {
    let pattern = ns_perturbation_pattern(n);
    let mut field = vec![0.0; n * n];
    for step in 0..NS_PERT_ITERS {
        let decay = 0.985_f64.powi(step as i32);
        let old = field.clone();
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                let left = old[j * n + (i + n - 1) % n];
                let right = old[j * n + (i + 1) % n];
                let up = old[((j + 1) % n) * n + i];
                let down = old[((j + n - 1) % n) * n + i];
                let lap = left + right + up + down - 4.0 * old[idx];
                // Source coupling 0.13: the response is linear in
                // `amp`, so `unit = response(amp=1)` is set by this
                // factor. It was 0.01, giving `unit ≈ 0.022` — with
                // `amp` clamped to ±5 the reachable response saturated
                // near 0.11, far below the slider's target range, and
                // the optimiser plateaued without ever converging.
                field[idx] = 0.992 * old[idx] + 0.035 * lap + amp * decay * pattern[idx] * 0.13;
            }
        }
    }
    field
}

fn ns_perturbation_response(field: &[f64]) -> f64 {
    let n = (field.len() as f64).sqrt() as usize;
    let pattern = ns_perturbation_pattern(n);
    field
        .iter()
        .zip(pattern.iter())
        .map(|(u, p)| u * p)
        .sum::<f64>()
        / field.len() as f64
}

fn ns_perturbation_loss_and_gradient(amp: f64, target_response: f64) -> (f64, f64, f64) {
    let field = ns_perturbation_rollout(amp, NS_PERT_N);
    let response = ns_perturbation_response(&field);
    let unit = ns_perturbation_response(&ns_perturbation_rollout(1.0, NS_PERT_N));
    let err = response - target_response;
    // Light L2 regularisation on `amp`. Kept small relative to `unit²`
    // so the regularised optimum still reaches the target response
    // rather than being pulled short by the penalty.
    let loss = 0.5 * err * err + 0.001 * amp * amp;
    let grad = err * unit + 0.002 * amp;
    (loss, grad, response)
}

fn ns_perturbation_optimize(
    amp0: f64,
    target_response: f64,
    lr: f64,
    iters: usize,
) -> (f64, f64, f64) {
    let mut amp = amp0;
    for _ in 0..iters {
        let (_loss, grad, _response) = ns_perturbation_loss_and_gradient(amp, target_response);
        amp = (amp - lr * grad).clamp(-5.0, 5.0);
    }
    let (loss, _grad, response) = ns_perturbation_loss_and_gradient(amp, target_response);
    (amp, loss, response)
}

fn update_ns_perturbation(mut warp: ResMut<WarpExampleStates>, sim: Res<Sim>) {
    if sim.scene != Scene::NavierStokesPerturbationOptimization {
        return;
    }
    let st = &mut warp.ns_perturbation;
    if !st.needs_rerun {
        return;
    }
    st.initial_loss = ns_perturbation_loss_and_gradient(st.amp, st.target_response).0;
    let (amp, loss, response) =
        ns_perturbation_optimize(st.amp, st.target_response, st.lr, NS_PERT_ITERS);
    st.amp = amp;
    st.loss = loss;
    st.response = response;
    st.field = ns_perturbation_rollout(st.amp, NS_PERT_N);
    st.iterations += NS_PERT_ITERS;
    st.needs_rerun = false;
}

fn draw_ns_perturbation_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::NavierStokesPerturbationOptimization {
        return;
    }
    let st = &warp.ns_perturbation;
    if st.field.len() != NS_PERT_N * NS_PERT_N {
        return;
    }
    let origin = Vec3::new(-0.65, 0.08, -0.65);
    let scale = 1.3 / (NS_PERT_N - 1) as f32;
    for j in 0..NS_PERT_N {
        for i in 0..NS_PERT_N {
            let v = st.field[j * NS_PERT_N + i] as f32;
            let p = origin + Vec3::new(i as f32 * scale, v * 2.0, j as f32 * scale);
            let t = (v * 20.0).clamp(-1.0, 1.0);
            let color = if t >= 0.0 {
                Color::srgb(0.25 + 0.65 * t, 0.35, 0.95 - 0.55 * t)
            } else {
                Color::srgb(0.25, 0.35 + 0.45 * -t, 0.95)
            };
            gizmos.sphere(p, 0.010, color);
            if i + 1 < NS_PERT_N {
                let v2 = st.field[j * NS_PERT_N + i + 1] as f32;
                let p2 = origin + Vec3::new((i + 1) as f32 * scale, v2 * 2.0, j as f32 * scale);
                gizmos.line(p, p2, color);
            }
        }
    }
}

fn ns_perturbation_pane(ui: &mut egui::Ui, st: &mut NsPerturbationState) {
    ui.label("Navier-Stokes perturbation optimization — reduced response");
    ui.separator();
    readout(ui, "Amplitude", &format!("{:.4}", st.amp));
    readout(ui, "Response", &format!("{:.4}", st.response));
    readout(ui, "Target", &format!("{:.4}", st.target_response));
    readout(ui, "Initial loss", &format!("{:.3e}", st.initial_loss));
    readout(ui, "Loss", &format!("{:.3e}", st.loss));
    readout(ui, "Iterations", &format!("{}", st.iterations));
    let changed_target = ui
        .add(egui::Slider::new(&mut st.target_response, -0.6..=0.6).text("target response"))
        .changed();
    let changed_lr = ui
        .add(egui::Slider::new(&mut st.lr, 0.1..=4.0).text("learning rate"))
        .changed();
    if changed_target || changed_lr || ui.button("Re-run optimization").clicked() {
        st.needs_rerun = true;
    }
    ui.small("Reduced procedural flow response; gradient is analytic for the scalar perturbation.");
    ui.small("Forward NS GPU lanes are named in the audit; full GPU adjoint remains future work.");
}

