// Warp Mesh Intersect scene — two axis-aligned cube TriMeshes that
// translate toward/away from each other; the pane reports the
// centre-gap distance and an INTERSECTING/separate status.
fn build_warp_mesh_intersect_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    let mut visuals = Vec::new();
    let kin_at = |pos: MVec3| {
        mp::BodyParams::new()
            .with_mass(0.0)
            .with_pose(mp::Transform::new(pos, MQuat::IDENTITY))
            .with_flags(mp::BodyFlags::KINEMATIC)
    };
    let h = MVec3::new(0.5, 0.5, 0.5);
    let a = b.add_body(kin_at(MVec3::new(-0.4, 1.0, 0.0)));
    b.add_shape_box(a, h, mp::Transform::IDENTITY)
        .expect("box A");
    visuals.push(BodyVisual::Box(Vec3::new(0.5, 0.5, 0.5)));
    let c = b.add_body(kin_at(MVec3::new(0.4, 1.0, 0.0)));
    b.add_shape_box(c, h, mp::Transform::IDENTITY)
        .expect("box B");
    visuals.push(BodyVisual::Box(Vec3::new(0.5, 0.5, 0.5)));
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");
    (b.build().expect("warp mesh-intersect"), visuals)
}
// ── Scene 55: WarpMeshIntersect ───────────────────────────────────
struct WarpMeshIntersectState {
    sim_time: f32,
    overlap_dist: f64,
    spin_rate: f32,
    intersecting: bool,
}
impl Default for WarpMeshIntersectState {
    fn default() -> Self {
        Self {
            sim_time: 0.0,
            overlap_dist: 0.0,
            spin_rate: 0.4,
            intersecting: false,
        }
    }
}

fn update_warp_mesh_intersect(
    time: Res<Time>,
    mut sim: ResMut<Sim>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpMeshIntersect {
        return;
    }
    let st = &mut warp.mesh_intersect;
    st.sim_time += time.delta_secs();
    // Animate the second box closer / farther.
    let phase = (st.sim_time * st.spin_rate).sin() as f64;
    let dx = 0.4 + phase * 0.6;
    if let Ok(body_q) = sim.state_a.body_q.host_mut()
        && body_q.len() >= 2 {
            body_q[0].position = MVec3::new(-dx * 0.5, 1.0, 0.0);
            body_q[1].position = MVec3::new(dx * 0.5, 1.0, 0.0);
        }
    // Approximate "overlap" measure as the gap between centres.
    st.overlap_dist = dx;
    st.intersecting = dx < 1.0;
}

fn warp_mesh_intersect_pane(ui: &mut egui::Ui, st: &mut WarpMeshIntersectState) {
    ui.label("Warp mesh intersect — port of core/example_mesh_intersect.py");
    ui.separator();
    readout(ui, "Centre gap", &format!("{:.3} m", st.overlap_dist));
    readout(
        ui,
        "Status",
        if st.intersecting {
            "INTERSECTING"
        } else {
            "separate"
        },
    );
    ui.label("Spin rate");
    let _ = ui.add(egui::Slider::new(&mut st.spin_rate, 0.05..=2.0));
    ui.small("Two axis-aligned cube TriMeshes translate toward/away from");
    ui.small("each other; overlap is the centre-gap test (touch at gap 1.0).");
}

