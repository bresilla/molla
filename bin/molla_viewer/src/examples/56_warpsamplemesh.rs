// ── Scene 56: WarpSampleMesh ──────────────────────────────────────
struct WarpSampleMeshState {
    sample_count: u32,
    /// Cached samples (world-space points on the icosphere surface).
    samples: Vec<Vec3>,
    user_reset: bool,
}
impl Default for WarpSampleMeshState {
    fn default() -> Self {
        Self {
            sample_count: 200,
            samples: Vec::new(),
            user_reset: true,
        }
    }
}

fn update_warp_sample_mesh(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpSampleMesh {
        return;
    }
    let st = &mut warp.sample_mesh;
    if !st.user_reset && st.samples.len() == st.sample_count as usize {
        return;
    }
    let mesh = warp_icosphere_trimesh(0.6);
    // CDF over triangle areas.
    let mut areas: Vec<f64> = Vec::with_capacity(mesh.indices.len());
    let mut total = 0.0;
    for tri in &mesh.indices {
        let a = mesh.vertices[tri[0] as usize];
        let bb = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        let area = 0.5 * (bb - a).cross(c - a).length();
        total += area;
        areas.push(total);
    }
    if total < 1e-12 {
        st.samples.clear();
        st.user_reset = false;
        return;
    }
    // Simple LCG for deterministic sampling.
    let mut seed = 0xDEADBEEF_u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f64) / (u32::MAX as f64)
    };
    st.samples.clear();
    st.samples.reserve(st.sample_count as usize);
    let origin = bevy::math::Vec3::new(0.0, 1.0, 0.0);
    for _ in 0..st.sample_count {
        let r = rng() * total;
        let tri_idx = match areas.binary_search_by(|x| x.partial_cmp(&r).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(mesh.indices.len() - 1),
        };
        let mut u = rng();
        let mut v = rng();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let w = 1.0 - u - v;
        let tri = mesh.indices[tri_idx];
        let pa = mesh.vertices[tri[0] as usize];
        let pb = mesh.vertices[tri[1] as usize];
        let pc = mesh.vertices[tri[2] as usize];
        let p = pa * w + pb * u + pc * v;
        st.samples
            .push(origin + Vec3::new(p.x as f32, p.y as f32, p.z as f32));
    }
    st.user_reset = false;
}

fn draw_warp_sample_mesh_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpSampleMesh {
        return;
    }
    for &p in &warp.sample_mesh.samples {
        gizmos.sphere(
            bevy::math::Isometry3d::from_translation(p),
            0.012,
            Color::srgb(0.9, 0.6, 0.2),
        );
    }
}

fn warp_sample_mesh_pane(ui: &mut egui::Ui, st: &mut WarpSampleMeshState) {
    ui.label("Warp sample mesh — port of core/example_sample_mesh.py");
    ui.separator();
    readout(ui, "Samples", &format!("{}", st.samples.len()));
    ui.label("Sample count");
    if ui
        .add(egui::Slider::new(&mut st.sample_count, 10..=2000))
        .changed()
    {
        st.user_reset = true;
    }
    if ui.button("Resample").clicked() {
        st.user_reset = true;
    }
    ui.small("CDF-weighted barycentric sampling on icosphere surface.");
}

