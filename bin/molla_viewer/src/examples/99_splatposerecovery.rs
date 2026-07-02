// ═══ Scene 99: SplatPoseRecovery (Stage 44 v2) ══════════════════════════
//
// A small Gaussian-splat scene (96 splats) rendered through a pinhole
// camera at a known truth pose. The "current" pose is perturbed and
// FD-driven gradient descent runs on the 3-translation + 3-rotation
// parameters each frame. The loss is the L2 image difference between
// the current and target 16×16 RGB image. The pane shows both images
// as live thumbnails plus the loss curve.

const SPR_N_SPLATS: usize = 96;
const SPR_IMG_W: usize = 16;
const SPR_IMG_H: usize = 16;
const SPR_FOV_Y: f64 = 1.0;
const SPR_LR: f64 = 4.0e-3;

#[derive(Clone)]
struct SprPose {
    t: MVec3,
    r: MVec3,
}

struct SplatPoseRecoveryState {
    splats: Vec<molla::geometry::GaussianSplat3d>,
    truth: SprPose,
    current: SprPose,
    target_image: Vec<f32>,
    current_image: Vec<f32>,
    loss_history: Vec<f64>,
    iter: usize,
    paused: bool,
    user_reset: bool,
    last_loss: f64,
    last_t_err: f64,
    last_r_err: f64,
}

impl Default for SplatPoseRecoveryState {
    fn default() -> Self {
        let splats = spr_make_scene(0xCAFEBABE);
        let truth = SprPose {
            t: MVec3::new(0.0, 0.0, 0.0),
            r: MVec3::ZERO,
        };
        let current = SprPose {
            t: MVec3::new(0.02, 0.0, 0.0),
            r: MVec3::new(0.0, 0.0, 0.05),
        };
        let target_image = spr_render(&splats, &truth);
        let current_image = spr_render(&splats, &current);
        let last_loss = spr_loss(&current_image, &target_image);
        Self {
            splats,
            truth,
            current,
            target_image,
            current_image,
            loss_history: vec![last_loss],
            iter: 0,
            paused: false,
            user_reset: false,
            last_loss,
            last_t_err: 0.02,
            last_r_err: 0.05,
        }
    }
}

fn spr_lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn spr_make_scene(seed: u64) -> Vec<molla::geometry::GaussianSplat3d> {
    let mut s = seed.max(1);
    let mut out = Vec::with_capacity(SPR_N_SPLATS);
    for _ in 0..SPR_N_SPLATS {
        let pos = MVec3::new(
            -0.7 + 1.4 * spr_lcg_next(&mut s),
            -0.5 + 1.0 * spr_lcg_next(&mut s),
            0.6 + 1.8 * spr_lcg_next(&mut s),
        );
        let r = 0.05 + 0.05 * spr_lcg_next(&mut s);
        let scale = MVec3::new(r, r, r);
        let q = MQuat::IDENTITY;
        let color = MVec3::new(
            spr_lcg_next(&mut s),
            spr_lcg_next(&mut s),
            spr_lcg_next(&mut s),
        );
        let opacity = 0.4 + 0.5 * spr_lcg_next(&mut s);
        out.push(molla::geometry::GaussianSplat3d::new(
            pos, scale, q, color, opacity,
        ));
    }
    out
}

fn spr_pose_quat(r: MVec3) -> MQuat {
    let angle = r.length();
    if angle < 1.0e-9 {
        MQuat::IDENTITY
    } else {
        let axis = r / angle;
        MQuat::from_axis_angle(axis, angle)
    }
}

fn spr_render(splats: &[molla::geometry::GaussianSplat3d], pose: &SprPose) -> Vec<f32> {
    let q_inv = spr_pose_quat(pose.r).inverse();
    let t_inv = q_inv * (-pose.t);
    let xformed: Vec<molla::geometry::GaussianSplat3d> = splats
        .iter()
        .map(|s| {
            let pos = q_inv * s.position + t_inv;
            let rot = q_inv * s.rotation;
            molla::geometry::GaussianSplat3d::new(
                pos,
                s.scale,
                rot,
                s.color_rgb(),
                s.opacity,
            )
        })
        .collect();
    let mut out = vec![0.0_f32; SPR_IMG_W * SPR_IMG_H * 3];
    let aspect = SPR_IMG_W as f64 / SPR_IMG_H as f64;
    let tan_half = (SPR_FOV_Y * 0.5).tan();
    for j in 0..SPR_IMG_H {
        for i in 0..SPR_IMG_W {
            let nx = (2.0 * (i as f64 + 0.5) / SPR_IMG_W as f64 - 1.0) * aspect * tan_half;
            let ny = (1.0 - 2.0 * (j as f64 + 0.5) / SPR_IMG_H as f64) * tan_half;
            let dir = MVec3::new(nx, ny, 1.0).normalize();
            let rgb = molla::geometry::composite_rgb_cpu(MVec3::ZERO, dir, &xformed, 20.0, 1.0e-3);
            let k = (j * SPR_IMG_W + i) * 3;
            out[k] = rgb.x as f32;
            out[k + 1] = rgb.y as f32;
            out[k + 2] = rgb.z as f32;
        }
    }
    out
}

fn spr_loss(a: &[f32], b: &[f32]) -> f64 {
    let mut s = 0.0_f64;
    for k in 0..a.len() {
        let d = (a[k] - b[k]) as f64;
        s += d * d;
    }
    s / a.len() as f64
}

fn spr_loss_at(splats: &[molla::geometry::GaussianSplat3d], pose: &SprPose, target: &[f32]) -> f64 {
    let img = spr_render(splats, pose);
    spr_loss(&img, target)
}

fn update_splat_pose_recovery(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::SplatPoseRecovery {
        return;
    }
    let st = &mut warp.splat_pose;
    if st.user_reset {
        *st = SplatPoseRecoveryState::default();
        return;
    }
    if run.paused || st.paused {
        return;
    }
    let h: f64 = 5.0e-3;
    let mut grad = [0.0_f64; 6];
    let base_loss = st.last_loss;
    for k in 0..6 {
        let mut probe = st.current.clone();
        match k {
            0 => probe.t.x += h,
            1 => probe.t.y += h,
            2 => probe.t.z += h,
            3 => probe.r.x += h,
            4 => probe.r.y += h,
            _ => probe.r.z += h,
        };
        let l_plus = spr_loss_at(&st.splats, &probe, &st.target_image);
        grad[k] = (l_plus - base_loss) / h;
    }
    st.current.t.x -= SPR_LR * grad[0];
    st.current.t.y -= SPR_LR * grad[1];
    st.current.t.z -= SPR_LR * grad[2];
    st.current.r.x -= SPR_LR * grad[3];
    st.current.r.y -= SPR_LR * grad[4];
    st.current.r.z -= SPR_LR * grad[5];
    st.current_image = spr_render(&st.splats, &st.current);
    st.last_loss = spr_loss(&st.current_image, &st.target_image);
    st.loss_history.push(st.last_loss);
    if st.loss_history.len() > 120 {
        st.loss_history.remove(0);
    }
    st.last_t_err = (st.current.t - st.truth.t).length();
    st.last_r_err = (st.current.r - st.truth.r).length();
    st.iter += 1;
}

fn draw_splat_pose_recovery_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::SplatPoseRecovery {
        return;
    }
    let st = &warp.splat_pose;
    let cell = 0.025_f32;
    let half = SPR_IMG_W as f32 * 0.5 * cell;
    let origin_target = Vec3::new(-0.55, 0.0, 0.0);
    let origin_current = Vec3::new(0.10, 0.0, 0.0);
    let draw_image = |gizmos: &mut Gizmos, origin: Vec3, img: &[f32]| {
        for j in 0..SPR_IMG_H {
            for i in 0..SPR_IMG_W {
                let k = (j * SPR_IMG_W + i) * 3;
                let r = img[k].clamp(0.0, 1.0);
                let g = img[k + 1].clamp(0.0, 1.0);
                let b = img[k + 2].clamp(0.0, 1.0);
                let col = Color::srgb(r, g, b);
                let x = origin.x - half + (i as f32 + 0.5) * cell;
                let y = origin.y + half - (j as f32 + 0.5) * cell;
                let c = Vec3::new(x, y, origin.z);
                let r2 = cell * 0.40;
                gizmos.line(
                    c - Vec3::new(r2, 0.0, 0.0),
                    c + Vec3::new(r2, 0.0, 0.0),
                    col,
                );
                gizmos.line(
                    c - Vec3::new(0.0, r2, 0.0),
                    c + Vec3::new(0.0, r2, 0.0),
                    col,
                );
            }
        }
    };
    draw_image(&mut gizmos, origin_target, &st.target_image);
    draw_image(&mut gizmos, origin_current, &st.current_image);
    let curve_y = -0.55_f32;
    let curve_h = 0.20_f32;
    let curve_w = 1.20_f32;
    let curve_x0 = -0.60_f32;
    let max_loss = st
        .loss_history
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(1.0e-9);
    let n = st.loss_history.len();
    let axis_col = Color::srgb(0.35, 0.35, 0.40);
    gizmos.line(
        Vec3::new(curve_x0, curve_y, 0.0),
        Vec3::new(curve_x0 + curve_w, curve_y, 0.0),
        axis_col,
    );
    gizmos.line(
        Vec3::new(curve_x0, curve_y, 0.0),
        Vec3::new(curve_x0, curve_y + curve_h, 0.0),
        axis_col,
    );
    let curve_col = Color::srgb(0.95, 0.75, 0.25);
    if n > 1 {
        for i in 0..n - 1 {
            let x0 = curve_x0 + (i as f32 / (n - 1) as f32) * curve_w;
            let x1 = curve_x0 + ((i + 1) as f32 / (n - 1) as f32) * curve_w;
            let y0 = curve_y + (st.loss_history[i] / max_loss) as f32 * curve_h;
            let y1 = curve_y + (st.loss_history[i + 1] / max_loss) as f32 * curve_h;
            gizmos.line(Vec3::new(x0, y0, 0.0), Vec3::new(x1, y1, 0.0), curve_col);
        }
    }
}

fn splat_pose_recovery_pane(ui: &mut egui::Ui, st: &mut SplatPoseRecoveryState) {
    ui.label("Stage 44 v2 — Gaussian-splat pose recovery (CPU FD)");
    ui.separator();
    readout(ui, "Splats", &format!("{}", st.splats.len()));
    readout(ui, "Image", &format!("{}×{}", SPR_IMG_W, SPR_IMG_H));
    readout(ui, "Iter", &format!("{}", st.iter));
    readout(ui, "Loss (MSE)", &format!("{:.3e}", st.last_loss));
    readout(ui, "‖Δt‖", &format!("{:.4} m", st.last_t_err));
    readout(ui, "‖Δω‖", &format!("{:.4} rad", st.last_r_err));
    ui.separator();
    let _ = ui.checkbox(&mut st.paused, "Pause");
    if ui.button("Reset (re-perturb)").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "Left thumbnail = target view; right = current view; gold curve \
         = loss history. The host-side path uses a forward-difference \
         gradient on the 6-vector pose and `composite_rgb_cpu`. The \
         GPU/release path (Stage 44 v2 in molla) uses analytic adjoints \
         + Adam — see `splat_pose_recovery.rs` integration test.",
    );
}

