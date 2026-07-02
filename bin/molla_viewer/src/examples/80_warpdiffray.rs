// ═══ Scene 80: WarpDiffRay (Stage 37.7) ════════════════════════════════
//
// Single sphere; gradient descent on its position to match a target depth
// image rendered with the sphere at a different position.

struct WarpDiffRayExampleState {
    scene: molla::sensors::raycast_adjoint::RayScene,
    rays: Vec<molla::sensors::raycast_adjoint::Ray>,
    target_image: Vec<f64>,
    params: Vec<f64>,
    target_params: Vec<f64>,
    lr: f64,
    iter: u32,
    last_loss: f64,
    run_descent: bool,
    user_reset: bool,
}

impl Default for WarpDiffRayExampleState {
    fn default() -> Self {
        use molla::math::Vec3 as MVec3;
        use molla::sensors::raycast_adjoint::{Ray, RayScene};
        let mut scene = RayScene::new();
        // Sphere takes 3 params (xyz) starting at offset 0; radius 0.5.
        scene.add_sphere(0, 0.5);
        scene.add_plane_z(-2.0);
        // 32×32 orthographic-down-Z rays.
        let n = 32usize;
        let half = 1.5_f64;
        let mut rays = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                let uu = (i as f64 + 0.5) / n as f64;
                let vv = (j as f64 + 0.5) / n as f64;
                rays.push(Ray::new(
                    MVec3::new((uu - 0.5) * 2.0 * half, (vv - 0.5) * 2.0 * half, 5.0),
                    MVec3::new(0.0, 0.0, -1.0),
                ));
            }
        }
        let target_params = vec![0.2_f64, 0.2_f64, 0.0_f64];
        let target_image = scene.trace(&target_params, &rays);
        let params = vec![0.0_f64; 3];
        Self {
            scene,
            rays,
            target_image,
            params,
            target_params,
            lr: 5e-4,
            iter: 0,
            last_loss: 0.0,
            run_descent: true,
            user_reset: false,
        }
    }
}

fn update_warp_diff_ray(sim: Res<Sim>, run: Res<SimRunState>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpDiffRay || run.paused {
        return;
    }
    let st = &mut warp.diff_ray;
    if st.user_reset {
        *st = WarpDiffRayExampleState::default();
        return;
    }
    let current = st.scene.trace(&st.params, &st.rays);
    st.last_loss = molla::sensors::raycast_adjoint::compute_loss_l2(&current, &st.target_image);
    if st.run_descent {
        let grad = molla::sensors::raycast_adjoint::compute_grad_raycast(
            &st.scene,
            &mut st.params,
            &st.target_image,
            &st.rays,
            1e-3,
        );
        for i in 0..st.params.len() {
            st.params[i] -= st.lr * grad[i];
        }
        st.iter += 1;
    }
}

fn draw_warp_diff_ray_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpDiffRay {
        return;
    }
    let st = &warp.diff_ray;
    let cur = Vec3::new(
        st.params[0] as f32,
        st.params[1] as f32,
        st.params[2] as f32,
    );
    let tgt = Vec3::new(
        st.target_params[0] as f32,
        st.target_params[1] as f32,
        st.target_params[2] as f32,
    );
    gizmos.sphere(cur, 0.5, Color::srgb(0.30, 0.65, 0.95));
    gizmos.sphere(tgt, 0.5, Color::srgb(0.95, 0.40, 0.30));
}

fn warp_diff_ray_pane(ui: &mut egui::Ui, state: &mut WarpDiffRayExampleState) {
    ui.label("Warp differentiable raycast (port of example_diffray.py)");
    ui.separator();
    readout(ui, "Iteration", &format!("{}", state.iter));
    readout(ui, "Loss", &format!("{:.4e}", state.last_loss));
    readout(
        ui,
        "Position",
        &format!(
            "({:.3}, {:.3}, {:.3})",
            state.params[0], state.params[1], state.params[2]
        ),
    );
    readout(
        ui,
        "Target",
        &format!(
            "({:.3}, {:.3}, {:.3})",
            state.target_params[0], state.target_params[1], state.target_params[2]
        ),
    );
    ui.separator();
    ui.label("Target X");
    let _ = ui.add(egui::Slider::new(&mut state.target_params[0], -1.0..=1.0));
    ui.label("Target Y");
    let _ = ui.add(egui::Slider::new(&mut state.target_params[1], -1.0..=1.0));
    ui.label("Learning rate");
    let _ = ui.add(egui::Slider::new(&mut state.lr, 1e-5..=1e-2).logarithmic(true));
    ui.separator();
    ui.checkbox(&mut state.run_descent, "Run descent");
    if ui.button("Reset (random init)").clicked() {
        state.user_reset = true;
    }
    ui.small("Blue = current sphere; red = target.");
    ui.small("Vanilla SGD on FD adjoint of L2 depth-image loss.");
    ui.small("Z-axis is hostile; XY converges within 50 iters at lr=5e-4.");
}

