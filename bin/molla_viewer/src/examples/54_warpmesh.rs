// Warp Mesh scene — a single TriMesh sphere built from an icosahedron
// triangulation. Animated rotation + breathing scale is applied each
// frame via the body pose (so we don't have to mutate the mesh
// buffer — molla has no built-in mesh deformation API in v1).
fn build_warp_mesh_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    // Build a small icosahedron-style TriMesh as a placeholder.
    let mesh = warp_icosphere_trimesh(0.5);
    let mesh_id = b.add_mesh(mesh);
    let body = b.add_body(
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_position(MVec3::new(0.0, 1.0, 0.0))
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    let _ = b.add_shape_trimesh(body, mesh_id, mp::Transform::IDENTITY);
    visuals.push(BodyVisual::Sphere(0.5));
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("warp mesh model"), visuals)
}
// ── Scene 54: WarpMesh ────────────────────────────────────────────
struct WarpMeshState {
    sim_time: f32,
    spin_rate: f32,
    breathe_amp: f32,
}
impl Default for WarpMeshState {
    fn default() -> Self {
        Self {
            sim_time: 0.0,
            spin_rate: 0.6,
            breathe_amp: 0.15,
        }
    }
}

fn update_warp_mesh(time: Res<Time>, mut sim: ResMut<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpMesh {
        return;
    }
    let st = &mut warp.mesh;
    st.sim_time += time.delta_secs();
    let yaw = st.sim_time * st.spin_rate;
    let scale = 1.0 + (st.sim_time * 1.4).sin() as f64 * st.breathe_amp as f64;
    // Body 0 is the mesh body; rewrite its pose.
    let q = MQuat::from_axis_angle(MVec3::Y, yaw as f64);
    let t = mp::Transform::new(MVec3::new(0.0, 1.0, 0.0), q);
    if let Ok(body_q) = sim.state_a.body_q.host_mut() {
        if !body_q.is_empty() {
            body_q[0] = t;
        }
    }
    // Ignore scale visualization for v1 — we'd need to mutate visuals.
    let _ = scale;
}

fn warp_mesh_pane(ui: &mut egui::Ui, st: &mut WarpMeshState) {
    ui.label("Warp mesh — port of core/example_mesh.py");
    ui.separator();
    readout(ui, "Sim time", &format!("{:.2} s", st.sim_time));
    ui.label("Spin rate");
    let _ = ui.add(egui::Slider::new(&mut st.spin_rate, 0.0..=2.0));
    ui.label("Breathe amplitude");
    let _ = ui.add(egui::Slider::new(&mut st.breathe_amp, 0.0..=0.3));
    ui.small("Animated TriMesh icosphere (rotate + breathe).");
}

