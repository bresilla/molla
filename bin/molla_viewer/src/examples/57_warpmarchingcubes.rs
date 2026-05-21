// ── Scene 57: WarpMarchingCubes ───────────────────────────────────
struct WarpMarchingCubesState {
    grid: u32,
    sim_time: f32,
    extracted_tri_count: usize,
    extracted_vert_count: usize,
    last_extract_ms: f64,
    /// Cached triangle list (each entry is a triple of points).
    tris: Vec<[Vec3; 3]>,
    user_reset: bool,
}
impl Default for WarpMarchingCubesState {
    fn default() -> Self {
        Self {
            grid: 32,
            sim_time: 0.0,
            extracted_tri_count: 0,
            extracted_vert_count: 0,
            last_extract_ms: 0.0,
            tris: Vec::new(),
            user_reset: true,
        }
    }
}

fn update_warp_marching_cubes(time: Res<Time>, sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpMarchingCubes {
        return;
    }
    let st = &mut warp.marching_cubes;
    st.sim_time += time.delta_secs();
    // Re-extract every 0.25 s (cheap: 32³ + MC).
    static mut LAST: f32 = -1.0;
    let recompute = unsafe {
        if st.user_reset || (st.sim_time - LAST) > 0.25 {
            LAST = st.sim_time;
            true
        } else {
            false
        }
    };
    if !recompute {
        return;
    }
    let dim = st.grid.max(8) as usize;
    let cell = 2.0 / dim as f64;
    let mut data = vec![0.0_f64; dim * dim * dim];
    let amp = 0.4 * (st.sim_time as f64 * 0.8).sin() + 0.6;
    for k in 0..dim {
        for j in 0..dim {
            for i in 0..dim {
                let p = MVec3::new(
                    -1.0 + (i as f64 + 0.5) * cell,
                    -1.0 + (j as f64 + 0.5) * cell,
                    -1.0 + (k as f64 + 0.5) * cell,
                );
                // Smooth-min of sphere + box.
                let sp = (p - MVec3::new(0.0, 0.0, 0.0)).length() - amp;
                let q = MVec3::new(p.x.abs() - 0.5, p.y.abs() - 0.5, p.z.abs() - 0.5);
                let qp = MVec3::new(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0));
                let bx = qp.length() + q.x.max(q.y.max(q.z)).min(0.0);
                let d = sp.min(bx);
                data[i + j * dim + k * dim * dim] = d;
            }
        }
    }
    let sdf = molla::geometry::SdfGrid {
        nx: dim as u32,
        ny: dim as u32,
        nz: dim as u32,
        origin: MVec3::new(-1.0, -1.0, -1.0),
        cell_size: cell,
        data,
    };
    let t0 = std::time::Instant::now();
    let tri_mesh = molla::geometry::marching_cubes(&sdf, 0.0);
    st.last_extract_ms = t0.elapsed().as_secs_f64() * 1000.0;
    st.extracted_vert_count = tri_mesh.vertices.len();
    st.extracted_tri_count = tri_mesh.indices.len();
    st.tris.clear();
    let origin = Vec3::new(0.0, 1.5, 0.0);
    for tri in &tri_mesh.indices {
        let a = tri_mesh.vertices[tri[0] as usize];
        let b = tri_mesh.vertices[tri[1] as usize];
        let c = tri_mesh.vertices[tri[2] as usize];
        let to = |p: MVec3| origin + Vec3::new(p.x as f32, p.y as f32, p.z as f32);
        st.tris.push([to(a), to(b), to(c)]);
    }
    st.user_reset = false;
}

fn draw_warp_marching_cubes_gizmos(
    sim: Res<Sim>,
    warp: Res<WarpExampleStates>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::WarpMarchingCubes {
        return;
    }
    for tri in &warp.marching_cubes.tris {
        let c = Color::srgb(0.4, 0.8, 0.95);
        gizmos.line(tri[0], tri[1], c);
        gizmos.line(tri[1], tri[2], c);
        gizmos.line(tri[2], tri[0], c);
    }
}

fn warp_marching_cubes_pane(ui: &mut egui::Ui, st: &mut WarpMarchingCubesState) {
    ui.label("Warp marching cubes — port of core/example_marching_cubes.py");
    ui.separator();
    readout(ui, "Grid", &format!("{0}³", st.grid));
    readout(ui, "Tris", &format!("{}", st.extracted_tri_count));
    readout(ui, "Verts", &format!("{}", st.extracted_vert_count));
    readout(ui, "Extract time", &format!("{:.2} ms", st.last_extract_ms));
    ui.separator();
    if ui
        .add(egui::Slider::new(&mut st.grid, 16..=64).text("Grid N"))
        .changed()
    {
        st.user_reset = true;
    }
    if ui.button("Re-extract").clicked() {
        st.user_reset = true;
    }
    ui.small("Animated SDF (sphere ⊓ box) → marching cubes wire mesh.");
}

