// Warp Raycast scene — a small set of kinematic primitives the GPU
// raycaster sweeps with a 16×16 ray grid each frame. Visualised as
// gizmo lines from origin to hit point. The ray sweep is done in
// `update_warp_raycast` and overlays via `draw_warp_raycast_gizmos`.
fn build_warp_raycast_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    let kin = |pos: MVec3| {
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(pos, MQuat::IDENTITY))
            .with_flags(mp::BodyFlags::KINEMATIC)
    };
    // Bunny stand-in: a sphere + a couple of nearby boxes.
    let s = b.add_body(kin(MVec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(s, 0.7, mp::Transform::IDENTITY)
        .expect("sphere");
    visuals.push(BodyVisual::Sphere(0.7));
    let bx1 = b.add_body(kin(MVec3::new(-1.6, 0.5, 0.4)));
    b.add_shape_box(bx1, MVec3::new(0.4, 0.5, 0.4), mp::Transform::IDENTITY)
        .expect("box");
    visuals.push(BodyVisual::Box(Vec3::new(0.4, 0.5, 0.4)));
    let bx2 = b.add_body(kin(MVec3::new(1.6, 0.4, -0.6)));
    b.add_shape_box(bx2, MVec3::new(0.5, 0.4, 0.5), mp::Transform::IDENTITY)
        .expect("box");
    visuals.push(BodyVisual::Box(Vec3::new(0.5, 0.4, 0.5)));
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("warp raycast model"), visuals)
}
// ── Scene 52: WarpRaycast ─────────────────────────────────────────
struct WarpRaycastState {
    raycaster: Option<Raycaster>,
    ray_grid_n: u32,
    spin: f32,
    last_hit_count: usize,
    last_ms: f64,
    rays: Vec<(Vec3, Vec3, Option<Vec3>)>,
    user_reset: bool,
}
impl Default for WarpRaycastState {
    fn default() -> Self {
        Self {
            raycaster: None,
            ray_grid_n: 16,
            spin: 0.0,
            last_hit_count: 0,
            last_ms: 0.0,
            rays: Vec::new(),
            user_reset: false,
        }
    }
}

fn update_warp_raycast(time: Res<Time>, sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpRaycast {
        if !warp.raycast.rays.is_empty() {
            warp.raycast.rays.clear();
        }
        return;
    }
    let st = &mut warp.raycast;
    if st.user_reset {
        st.spin = 0.0;
        st.user_reset = false;
    }
    st.spin += time.delta_secs() * 0.4;
    let n = st.ray_grid_n.max(2);
    let cap = (n * n).div_ceil(64) * 64;
    if st.raycaster.is_none() {
        match Raycaster::new(&sim.engine, cap) {
            Ok(rc) => st.raycaster = Some(rc),
            Err(e) => {
                warn!("warp raycast: raycaster init failed: {e}");
                return;
            }
        }
    }
    // Camera at origin spinning around +Y, fov ~90°, looking at origin.
    let cam = Vec3::new(st.spin.cos() * 4.0, 2.0, st.spin.sin() * 4.0);
    let target = Vec3::new(0.0, 1.0, 0.0);
    let forward = (target - cam).normalize_or_zero();
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    let fov = 70f32.to_radians();
    let half = (fov * 0.5).tan();
    let mut origins: Vec<MVec3> = Vec::with_capacity((n * n) as usize);
    let mut dirs: Vec<MVec3> = Vec::with_capacity((n * n) as usize);
    for iy in 0..n {
        for ix in 0..n {
            let sx = (2.0 * (ix as f32 + 0.5) / n as f32 - 1.0) * half;
            let sy = (2.0 * (iy as f32 + 0.5) / n as f32 - 1.0) * half;
            let dir = (forward + right * sx + up * sy).normalize_or_zero();
            origins.push(MVec3::new(cam.x as f64, cam.y as f64, cam.z as f64));
            dirs.push(MVec3::new(dir.x as f64, dir.y as f64, dir.z as f64));
        }
    }
    let t0 = std::time::Instant::now();
    let hits = match st.raycaster.as_mut().unwrap().cast(
        &sim.engine,
        &sim.model,
        &sim.state_a,
        &origins,
        &dirs,
        50.0,
    ) {
        Ok(h) => h,
        Err(e) => {
            warn!("warp raycast: cast failed: {e}");
            return;
        }
    };
    st.last_ms = t0.elapsed().as_secs_f64() * 1000.0;
    st.rays.clear();
    let mut hit_count = 0;
    for (i, h) in hits.iter().enumerate() {
        let o = origins[i];
        let d = dirs[i];
        let o_b = Vec3::new(o.x as f32, o.y as f32, o.z as f32);
        let d_b = Vec3::new(d.x as f32, d.y as f32, d.z as f32);
        if h.shape_id.is_some() {
            hit_count += 1;
            let p = Vec3::new(h.point.x as f32, h.point.y as f32, h.point.z as f32);
            st.rays.push((o_b, d_b, Some(p)));
        } else {
            st.rays.push((o_b, d_b, None));
        }
    }
    st.last_hit_count = hit_count;
}

fn draw_warp_raycast_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpRaycast {
        return;
    }
    for (o, d, hit) in &warp.raycast.rays {
        match hit {
            Some(p) => {
                gizmos.line(*o, *p, Color::srgb(0.9, 0.7, 0.2));
                gizmos.sphere(
                    bevy::math::Isometry3d::from_translation(*p),
                    0.025,
                    Color::srgb(0.9, 0.4, 0.1),
                );
            }
            None => {
                let far = *o + *d * 10.0;
                gizmos.line(*o, far, Color::srgba(0.4, 0.4, 0.5, 0.15));
            }
        }
    }
}

fn warp_raycast_pane(ui: &mut egui::Ui, st: &mut WarpRaycastState) {
    ui.label("Warp raycast — port of core/example_raycast.py");
    ui.separator();
    readout(ui, "Ray grid", &format!("{0}×{0}", st.ray_grid_n));
    readout(ui, "Hits", &format!("{}", st.last_hit_count));
    readout(ui, "Cast time", &format!("{:.2} ms", st.last_ms));
    ui.separator();
    ui.label("Ray grid N");
    let _ = ui.add(egui::Slider::new(&mut st.ray_grid_n, 4..=64));
    if ui.button("Reset spin").clicked() {
        st.user_reset = true;
    }
    ui.small("GPU `Raycaster` sweeps an orbiting camera grid.");
}

