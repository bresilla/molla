// ── Scene 53: WarpRaymarch ────────────────────────────────────────
struct WarpRaymarchState {
    width: u32,
    height: u32,
    cam_yaw: f32,
    image: Vec<u8>,
    /// EGui texture id for the image (reuse across frames).
    tex: Option<egui::TextureHandle>,
    last_render_ms: f64,
}
impl Default for WarpRaymarchState {
    fn default() -> Self {
        Self {
            width: 128,
            height: 96,
            cam_yaw: 0.0,
            image: Vec::new(),
            tex: None,
            last_render_ms: 0.0,
        }
    }
}

fn warp_raymarch_sdf(p: bevy::math::Vec3) -> f32 {
    // Sphere centred at (0, 0.75, 0), radius 0.75, minus a box.
    let sp_c = bevy::math::Vec3::new(0.0, 0.75, 0.0);
    let sp = (p - sp_c).length() - 0.75;
    let q = bevy::math::Vec3::new(p.x.abs() - 1.0, p.y.abs() - 0.5, p.z.abs() - 0.5);
    let qp = bevy::math::Vec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
    let bx = qp.length() + q.x.max(q.y.max(q.z)).min(0.0);
    let d = (-bx).max(sp);
    // Ground plane y = -1.
    let plane = p.y + 1.0;
    d.min(plane)
}

fn update_warp_raymarch(time: Res<Time>, sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpRaymarch {
        return;
    }
    let st = &mut warp.raymarch;
    st.cam_yaw += time.delta_secs() * 0.2;
    let w = st.width.max(8) as usize;
    let h = st.height.max(8) as usize;
    let total = w * h * 4;
    if st.image.len() != total {
        st.image = vec![0u8; total];
    }
    let cam = bevy::math::Vec3::new(st.cam_yaw.cos() * 3.0, 1.5, st.cam_yaw.sin() * 3.0);
    let target = bevy::math::Vec3::new(0.0, 0.75, 0.0);
    let forward = (target - cam).normalize_or_zero();
    let right = forward.cross(bevy::math::Vec3::Y).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let t0 = std::time::Instant::now();
    for j in 0..h {
        for i in 0..w {
            let sx = (2.0 * (i as f32 + 0.5) / w as f32 - 1.0) * 0.7;
            let sy = (2.0 * (j as f32 + 0.5) / h as f32 - 1.0) * 0.5;
            let dir = (forward + right * sx + up * sy).normalize_or_zero();
            let mut t = 0.0_f32;
            let mut hit = false;
            for _ in 0..64 {
                let p = cam + dir * t;
                let d = warp_raymarch_sdf(p);
                if d < 0.001 {
                    hit = true;
                    break;
                }
                t += d;
                if t > 20.0 {
                    break;
                }
            }
            let idx = (j * w + i) * 4;
            if hit {
                let p = cam + dir * t;
                let eps = 0.005;
                let nx = warp_raymarch_sdf(p + bevy::math::Vec3::X * eps)
                    - warp_raymarch_sdf(p - bevy::math::Vec3::X * eps);
                let ny = warp_raymarch_sdf(p + bevy::math::Vec3::Y * eps)
                    - warp_raymarch_sdf(p - bevy::math::Vec3::Y * eps);
                let nz = warp_raymarch_sdf(p + bevy::math::Vec3::Z * eps)
                    - warp_raymarch_sdf(p - bevy::math::Vec3::Z * eps);
                let n = bevy::math::Vec3::new(nx, ny, nz).normalize_or_zero();
                let l = bevy::math::Vec3::new(0.6, 0.4, 0.5).normalize_or_zero();
                let diff = n.dot(l).max(0.0);
                let r = (diff * 230.0 + 25.0) as u8;
                let g = (diff * 200.0 + 30.0) as u8;
                let bb = (diff * 150.0 + 60.0) as u8;
                st.image[idx] = r;
                st.image[idx + 1] = g;
                st.image[idx + 2] = bb;
                st.image[idx + 3] = 255;
            } else {
                st.image[idx] = 80;
                st.image[idx + 1] = 90;
                st.image[idx + 2] = 110;
                st.image[idx + 3] = 255;
            }
        }
    }
    st.last_render_ms = t0.elapsed().as_secs_f64() * 1000.0;
}

fn warp_raymarch_pane(ui: &mut egui::Ui, st: &mut WarpRaymarchState) {
    ui.label("Warp raymarch — port of core/example_raymarch.py");
    ui.separator();
    readout(ui, "Resolution", &format!("{}×{}", st.width, st.height));
    readout(ui, "Render time", &format!("{:.2} ms", st.last_render_ms));
    ui.separator();
    let w = st.width as usize;
    let h = st.height as usize;
    if !st.image.is_empty() && st.image.len() == w * h * 4 {
        let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &st.image);
        let tex = match st.tex.as_mut() {
            Some(t) => {
                t.set(img, egui::TextureOptions::LINEAR);
                t.clone()
            }
            None => {
                let t = ui
                    .ctx()
                    .load_texture("warp_raymarch", img, egui::TextureOptions::LINEAR);
                st.tex = Some(t.clone());
                t
            }
        };
        ui.image(egui::load::SizedTexture::new(
            tex.id(),
            egui::vec2(w as f32 * 1.5, h as f32 * 1.5),
        ));
    }
    ui.small("CPU SDF raymarch (sphere ∖ box + ground plane).");
}

