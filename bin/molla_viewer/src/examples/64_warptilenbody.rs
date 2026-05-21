// ── Scene 64: WarpTileNbody ───────────────────────────────────────
struct WarpTileNbodyState {
    n: u32,
    pos: Vec<Vec3>,
    vel: Vec<Vec3>,
    g: f32,
    softening: f32,
    user_reset: bool,
}
impl Default for WarpTileNbodyState {
    fn default() -> Self {
        Self {
            n: 64,
            pos: Vec::new(),
            vel: Vec::new(),
            g: 0.4,
            softening: 0.05,
            user_reset: true,
        }
    }
}

fn warp_tile_nbody_step(pos: &mut [Vec3], vel: &mut [Vec3], g: f32, softening: f32, dt: f32) {
    assert_eq!(
        pos.len(),
        vel.len(),
        "N-body position/velocity length mismatch"
    );
    let n = pos.len();
    let soft_sq = softening * softening;
    let mut acc = vec![Vec3::ZERO; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let r = pos[j] - pos[i];
            let d2 = r.length_squared() + soft_sq;
            let inv_d = d2.sqrt().recip();
            let a_pair = r * (inv_d * inv_d * inv_d * g);
            acc[i] += a_pair;
            acc[j] -= a_pair;
        }
    }
    for i in 0..n {
        vel[i] += acc[i] * dt;
    }
    for i in 0..n {
        pos[i] += vel[i] * dt;
    }
}

fn update_warp_tile_nbody(time: Res<Time>, sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpTileNbody {
        return;
    }
    let st = &mut warp.tile_nbody;
    if st.user_reset || st.pos.len() != st.n as usize {
        let n = st.n.max(2) as usize;
        st.pos = Vec::with_capacity(n);
        st.vel = Vec::with_capacity(n);
        let mut seed = 0xBEEF_u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) as i32 as f32) / 32768.0
        };
        for _ in 0..n {
            // Random surface of a unit sphere shell.
            let phi = (1.0 - 2.0 * (rng() * 0.5 + 0.5)).acos();
            let theta = rng() * std::f32::consts::PI * 2.0;
            let p = Vec3::new(
                phi.sin() * theta.cos(),
                1.0 + phi.sin() * theta.sin(),
                phi.cos(),
            );
            st.pos.push(p);
        }
        // Seed a coherent tangential velocity — solid-body rotation
        // about the +Y axis through the cloud centre — so the system
        // forms a rotating N-body cloud instead of collapsing straight
        // to a point under its own gravity. The Warp reference
        // (example_tile_nbody.py) likewise gives each body an initial
        // `v_scale * cross(pos, axis)`. `omega ~ sqrt(G·N)` keeps the
        // seed near orbital speed as the G / N sliders change.
        let center = Vec3::new(0.0, 1.0, 0.0);
        let omega = 0.45 * (st.g * n as f32).sqrt();
        for idx in 0..n {
            let r = st.pos[idx] - center;
            st.vel.push(omega * Vec3::new(r.z, 0.0, -r.x));
        }
        st.user_reset = false;
    }
    let dt = time.delta_secs().min(0.033);
    warp_tile_nbody_step(&mut st.pos, &mut st.vel, st.g, st.softening, dt);
}

fn draw_warp_tile_nbody_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpTileNbody {
        return;
    }
    for &p in &warp.tile_nbody.pos {
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(p),
            0.04,
            Color::srgb(0.95, 0.85, 0.3),
        );
    }
}

fn warp_tile_nbody_pane(ui: &mut egui::Ui, st: &mut WarpTileNbodyState) {
    ui.label("Warp tile N-body — port of tile/example_tile_nbody.py");
    ui.separator();
    readout(ui, "Bodies", &format!("{}", st.pos.len()));
    ui.label("Gravitational constant G");
    let _ = ui.add(egui::Slider::new(&mut st.g, 0.05..=2.0));
    ui.label("Softening ε");
    let _ = ui.add(egui::Slider::new(&mut st.softening, 0.005..=0.5).logarithmic(true));
    if ui
        .add(egui::Slider::new(&mut st.n, 8..=256).text("N"))
        .changed()
    {
        st.user_reset = true;
    }
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small(
        "All-pairs Newtonian gravity (host O(N²) loop). Stage 37.8 did \
         not add an N-body primitive — kept on CPU for the demo.",
    );
}

