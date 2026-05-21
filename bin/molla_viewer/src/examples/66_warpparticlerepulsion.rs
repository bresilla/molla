// ── Scene 66: WarpParticleRepulsion ───────────────────────────────
struct WarpRepulsionState {
    n: u32,
    pos: Vec<Vec3>,
    /// Iteration count of gradient descent already executed.
    iter: usize,
    /// Last loss value.
    last_loss: f64,
    learning_rate: f32,
    paused: bool,
    user_reset: bool,
}
impl Default for WarpRepulsionState {
    fn default() -> Self {
        Self {
            n: 32,
            pos: Vec::new(),
            iter: 0,
            last_loss: 0.0,
            learning_rate: 0.05,
            paused: false,
            user_reset: true,
        }
    }
}

fn update_warp_repulsion(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpParticleRepulsion {
        return;
    }
    let st = &mut warp.repulsion;
    if st.user_reset || st.pos.len() != st.n as usize {
        let n = st.n.max(2) as usize;
        st.pos = Vec::with_capacity(n);
        let mut seed = 0xCAFE_u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as i32 as f32) / 32768.0 + 0.5
        };
        for _ in 0..n {
            st.pos.push(Vec3::new(rng() * 2.0, 1.0, rng() * 2.0));
        }
        st.iter = 0;
        st.last_loss = 0.0;
        st.user_reset = false;
    }
    if st.paused {
        return;
    }
    // Stage 37.8 — gradient comes from the molla `tile_mcgp` primitive
    // (SPSA stochastic estimate) instead of the inline analytic
    // `r / d³` accumulation. Same closed-form loss; the gradient
    // estimate is noisy by construction, so the descent path is more
    // jittery than the analytic version — that's the demo, end users
    // should reach for analytic gradients when they have them.
    let lr = st.learning_rate;

    // Flatten positions to a `Vec<f64>` for the loss closure.
    let params: Vec<f64> = st
        .pos
        .iter()
        .flat_map(|p| [p.x as f64, p.y as f64, p.z as f64])
        .collect();
    let inverse_distance_loss = |p: &[f64]| -> f64 {
        let mut s = 0.0_f64;
        let nn = p.len() / 3;
        for i in 0..nn {
            for j in (i + 1)..nn {
                let dx = p[i * 3 + 0] - p[j * 3 + 0];
                let dy = p[i * 3 + 1] - p[j * 3 + 1];
                let dz = p[i * 3 + 2] - p[j * 3 + 2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-3);
                s += 1.0 / d;
            }
        }
        s
    };

    // Single forward pass for the loss readout.
    let loss_now = inverse_distance_loss(&params);
    st.last_loss = loss_now;

    // 64 SPSA samples per descent step — keeps the per-frame budget
    // bounded while letting the noise average down across frames.
    let cfg = molla::compute::tile_mcgp::McgpConfig {
        n_samples: 64,
        epsilon: 5.0e-3,
        seed: 0x9E3779B9_u64.wrapping_add(st.iter as u64),
    };
    let grad = molla::compute::tile_mcgp::mcgp_estimate(&inverse_distance_loss, &params, &cfg);

    // Gradient descent: p -= lr · ∇L. Pack back into Vec3.
    for (i, p) in st.pos.iter_mut().enumerate() {
        let gx = grad[i * 3 + 0] as f32;
        let gy = grad[i * 3 + 1] as f32;
        let gz = grad[i * 3 + 2] as f32;
        p.x -= lr * gx;
        p.y -= lr * gy;
        p.z -= lr * gz;
        // Periodic-style soft wrap to a 2×2×2 box.
        for axis in 0..3 {
            let v = match axis {
                0 => p.x,
                1 => p.y,
                _ => p.z,
            };
            let w = ((v + 1.0).rem_euclid(2.0)) - 1.0;
            match axis {
                0 => p.x = w,
                1 => p.y = w,
                _ => p.z = w,
            }
        }
        p.y = p.y.abs() + 0.5; // keep above ground
    }
    st.iter += 1;
}

fn draw_warp_repulsion_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpParticleRepulsion {
        return;
    }
    for &p in &warp.repulsion.pos {
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(p),
            0.04,
            Color::srgb(0.4, 0.8, 0.95),
        );
    }
}

fn warp_repulsion_pane(ui: &mut egui::Ui, st: &mut WarpRepulsionState) {
    ui.label("Warp particle repulsion — port of optim/example_particle_repulsion.py");
    ui.separator();
    readout(ui, "Particles", &format!("{}", st.pos.len()));
    readout(ui, "Iterations", &format!("{}", st.iter));
    readout(ui, "Loss", &format!("{:.3e}", st.last_loss));
    ui.separator();
    ui.label("Learning rate");
    let _ = ui.add(egui::Slider::new(&mut st.learning_rate, 0.001..=0.5).logarithmic(true));
    if ui
        .add(egui::Slider::new(&mut st.n, 4..=128).text("N"))
        .changed()
    {
        st.user_reset = true;
    }
    ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Inverse-distance pair potential, gradient descent on positions. \
         Gradient estimated by `molla::compute::tile_mcgp` SPSA \
         (Stage 37.8) — noisy by construction, so the descent path \
         jitters compared to the analytic-gradient placeholder.",
    );
}

