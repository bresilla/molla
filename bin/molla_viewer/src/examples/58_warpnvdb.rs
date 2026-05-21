// ── Scene 58: WarpNvdb ────────────────────────────────────────────
struct WarpNvdbState {
    insert_count: u32,
    last_insert_ms: f64,
    last_lookup_ms: f64,
    last_grid_voxel_count: usize,
}
impl Default for WarpNvdbState {
    fn default() -> Self {
        Self {
            insert_count: 256,
            last_insert_ms: 0.0,
            last_lookup_ms: 0.0,
            last_grid_voxel_count: 0,
        }
    }
}

fn update_warp_nvdb(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpNvdb {
        return;
    }
    let st = &mut warp.nvdb;
    let n = st.insert_count.max(1) as usize;
    let mut seed = 0xCAFEBABE_u32;
    let mut rng = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed
    };
    let mut points: Vec<MVec3> = Vec::with_capacity(n);
    for _ in 0..n {
        let r1 = rng();
        let r2 = rng();
        let r3 = rng();
        let x = ((r1 % 1000) as f64 / 1000.0 - 0.5) * 4.0;
        let y = ((r2 % 1000) as f64 / 1000.0 - 0.5) * 4.0;
        let z = ((r3 % 1000) as f64 / 1000.0 - 0.5) * 4.0;
        points.push(MVec3::new(x, y, z));
    }
    // Insert: bake a tiny sphere mesh's SDF into a sparse SDF.
    let t0 = std::time::Instant::now();
    let sphere_mesh = warp_icosphere_trimesh(0.4);
    let sparse = molla::geometry::bake_mesh_to_sparse_sdf(&sphere_mesh, 32, 0.05, 4);
    st.last_insert_ms = t0.elapsed().as_secs_f64() * 1000.0;
    st.last_grid_voxel_count = (sparse.nx as usize) * (sparse.ny as usize) * (sparse.nz as usize);
    // Lookup: query at random points (just hash buckets).
    let t1 = std::time::Instant::now();
    let mut hits = 0;
    for p in &points {
        let voxel_size = sparse.cell_size;
        let origin = sparse.origin;
        let ix = ((p.x - origin.x) / voxel_size) as i32;
        let iy = ((p.y - origin.y) / voxel_size) as i32;
        let iz = ((p.z - origin.z) / voxel_size) as i32;
        if ix >= 0
            && iy >= 0
            && iz >= 0
            && (ix as u32) < sparse.nx
            && (iy as u32) < sparse.ny
            && (iz as u32) < sparse.nz
        {
            hits += 1;
        }
    }
    let _ = hits;
    st.last_lookup_ms = t1.elapsed().as_secs_f64() * 1000.0;
}

fn warp_nvdb_pane(ui: &mut egui::Ui, st: &mut WarpNvdbState) {
    ui.label("Warp NanoVDB — port of core/example_nvdb.py");
    ui.separator();
    readout(
        ui,
        "Sparse voxels",
        &format!("{}", st.last_grid_voxel_count),
    );
    readout(ui, "Insert (bake)", &format!("{:.2} ms", st.last_insert_ms));
    readout(ui, "Lookup ({})", &format!("{:.2} ms", st.last_lookup_ms));
    ui.separator();
    let _ = ui.add(egui::Slider::new(&mut st.insert_count, 16..=4096).text("Lookup N"));
    ui.small("Sparse SDF baked from icosphere; lookup at random pts.");
}

