// Stage-21 mesh-tools demo state. Toggles for which artefacts are
// visible in the viewport. The actual meshes are built once at scene
// entry by `apply_scene_switch` and stashed as Bevy entities tagged
// with one of the [`MeshToolPanel`] markers; toggling the booleans
// just flips the `Visibility` of those entities.
#[derive(Resource)]
struct MeshToolsExampleState {
    /// Show the original (unsimplified) icosphere on the left.
    show_original: bool,
    /// Show the QEM-decimated mesh in the centre.
    show_decimated: bool,
    /// Show the marching-cubes surface extracted from a baked SDF
    /// on the right.
    show_marching_cubes: bool,
    /// Overlay the convex-hull wireframe on the original mesh.
    show_hull: bool,
    /// Telemetry surfaced in the right-rail pane (filled in by
    /// `apply_scene_switch` when the meshes are built).
    original_tris: usize,
    decimated_tris: usize,
    mc_tris: usize,
    hull_verts: usize,
    sdf_resolution: u32,
    /// Stage-24e: when true, the right panel renders the marching-cubes
    /// surface from a *sparse* SDF instead of the dense one. Both are
    /// baked at scene-entry; the toggle picks which one the surface +
    /// storage-cost readout reflects.
    use_sparse_sdf: bool,
    /// Stage-24e telemetry (filled in at scene-entry):
    dense_sdf_bytes: usize,
    sparse_sdf_bytes: usize,
    sparse_sdf_leaves: usize,
    sparse_sdf_active_voxels: u64,
    sparse_sdf_band: u32,
    /// Stage-24e.1 telemetry — root-table load factor + average
    /// probe length for the GPU root hash (filled in at scene-entry
    /// when the sparse SDF is built).
    sparse_root_table_capacity: u32,
    sparse_root_load_factor: f32,
    sparse_root_avg_probes: f32,
    sparse_root_max_probes: u32,
    /// Set to `true` when the user touches a control whose effect is
    /// only realised on a scene rebuild (`use_sparse_sdf`). Cleared on
    /// rebuild. Surfaces a "Rebuild mesh tools" button so the pane
    /// doesn't lie about needing the user to re-press M (the hotkey
    /// is a no-op when the scene is already active).
    rebuild_pending: bool,
}

impl Default for MeshToolsExampleState {
    fn default() -> Self {
        Self {
            show_original: true,
            show_decimated: true,
            show_marching_cubes: true,
            show_hull: false,
            original_tris: 0,
            decimated_tris: 0,
            mc_tris: 0,
            hull_verts: 0,
            sdf_resolution: 32,
            use_sparse_sdf: false,
            dense_sdf_bytes: 0,
            sparse_sdf_bytes: 0,
            sparse_sdf_leaves: 0,
            sparse_sdf_active_voxels: 0,
            sparse_sdf_band: 4,
            sparse_root_table_capacity: 0,
            sparse_root_load_factor: 0.0,
            sparse_root_avg_probes: 0.0,
            sparse_root_max_probes: 0,
            rebuild_pending: false,
        }
    }
}

impl MeshToolsExampleState {
    /// Width of the sparse-SDF narrow band, in voxels. Centralised so
    /// the bake function and the pane readout agree.
    fn sdf_sparse_band(&self) -> u32 {
        self.sparse_sdf_band
    }
}

// Stage-24e helper — sample the sparse SDF back into a dense grid so
// the existing `marching_cubes` consumer (which only takes
// `SdfGrid`) can re-extract the surface. Used only by the viewer's
// "use sparse SDF" toggle. Production MC over sparse SDFs should
// iterate active leaves directly (deferred to Stage 24e.1).
fn sparse_sdf_to_dense_for_mc(
    sparse: &molla::geometry::SparseSdf,
    template: &molla::geometry::SdfGrid,
) -> molla::geometry::SdfGrid {
    let mut data = Vec::with_capacity(template.data.len());
    for z in 0..template.nz {
        for y in 0..template.ny {
            for x in 0..template.nx {
                let v = sparse.grid.lookup(x as i32, y as i32, z as i32) as f64;
                data.push(v);
            }
        }
    }
    molla::geometry::SdfGrid {
        nx: template.nx,
        ny: template.ny,
        nz: template.nz,
        origin: template.origin,
        cell_size: template.cell_size,
        data,
    }
}

// Marker for the original (high-res) icosphere in `Scene::MeshTools`.
#[derive(Component)]
struct MeshToolOriginal;

// Marker for the QEM-decimated icosphere.
#[derive(Component)]
struct MeshToolDecimated;

// Marker for the marching-cubes-extracted surface.
#[derive(Component)]
struct MeshToolMarchingCubes;

// Marker for the convex-hull wireframe overlay.
#[derive(Component)]
struct MeshToolHull;

// Stage-21 mesh-tools demo. The "model" is empty (no physics step
// runs for this scene); all the visual work happens in
// `apply_scene_switch` via [`spawn_mesh_tools_entities`], which
// generates an icosphere and runs decimation / SDF-bake / marching-
// cubes / hull on it. This builder just provides the requisite
// `(Model, Vec<BodyVisual>)` shape so [`Sim::build_scene`] can use it
// uniformly with every other scene.
fn build_mesh_tools_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    (b.build().expect("empty model"), Vec::new())
}

// Stage-21 mesh-tools demo; isolated so the same path is reused for
// the original / decimated / marching-cubes meshes.
fn molla_trimesh_to_bevy_mesh(tm: &molla::sim::TriMesh, double_sided: bool) -> Mesh {
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, PrimitiveTopology};
    let _ = double_sided;
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    let positions: Vec<[f32; 3]> = tm
        .vertices
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; positions.len()];
    let mut idx_flat: Vec<u32> = Vec::with_capacity(tm.indices.len() * 3);
    for tri in &tm.indices {
        idx_flat.push(tri[0]);
        idx_flat.push(tri[1]);
        idx_flat.push(tri[2]);
    }
    for chunk in idx_flat.chunks(3) {
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
        for &k in &[i0, i1, i2] {
            let cur = normals[k];
            normals[k] = [cur[0] + n.x, cur[1] + n.y, cur[2] + n.z];
        }
    }
    for n in normals.iter_mut() {
        let v = Vec3::from_array(*n).normalize_or_zero();
        *n = [v.x, v.y, v.z];
    }
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(idx_flat));
    mesh
}

// Generate a unit-radius icosphere as a molla `TriMesh` with the
// requested subdivision level. Mirrors the test helper in
// `tests/mesh_utils.rs` — kept duplicated here so the viewer doesn't
// need to depend on test-only code.
fn icosphere_mesh(subdivisions: u32, radius: f64) -> molla::sim::TriMesh {
    use std::collections::HashMap;
    let t: f64 = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let raw = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ];
    let mut verts: Vec<MVec3> = raw
        .iter()
        .map(|(x, y, z)| {
            let v = MVec3::new(*x, *y, *z);
            v / v.length()
        })
        .collect();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<MVec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&i) = cache.get(&key) {
                return i;
            }
            let mid = (verts[a as usize] + verts[b as usize]) * 0.5;
            let mid = mid / mid.length();
            verts.push(mid);
            let idx = (verts.len() - 1) as u32;
            cache.insert(key, idx);
            idx
        };
        let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
        for tri in &faces {
            let a = midpoint(tri[0], tri[1], &mut verts);
            let b = midpoint(tri[1], tri[2], &mut verts);
            let c = midpoint(tri[2], tri[0], &mut verts);
            new_faces.push([tri[0], a, c]);
            new_faces.push([tri[1], b, a]);
            new_faces.push([tri[2], c, b]);
            new_faces.push([a, b, c]);
        }
        faces = new_faces;
    }
    for v in &mut verts {
        *v *= radius;
    }
    let mut m = molla::sim::TriMesh {
        vertices: verts,
        indices: faces,
    };
    m.build_bvh(4);
    m
}

// Spawn the four Stage-21 visualisation entities (original /
// decimated / marching-cubes / hull) for `Scene::MeshTools`. Updates
// the `mesh_state` telemetry fields so the right-rail pane can show
// triangle counts. Each entity is tagged with `ClothObstacle` so
// `apply_scene_switch` despawns it on the next scene change AND with
// a per-panel marker (`MeshToolOriginal`, etc.) so visibility
// toggles in the pane can find it.
fn spawn_mesh_tools_entities(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    mesh_state: &mut MeshToolsExampleState,
) {
    use molla::geometry::{
        bake_mesh_to_sdf, convex_hull, decimate, marching_cubes, sparse_sdf_from_dense,
    };

    // Build the source mesh — a mildly noise-displaced icosphere
    // (3 subdivisions = 1280 tris). The noise breaks the icosphere's
    // perfect symmetry so decimation has to make non-trivial choices,
    // but stays small enough that the mesh is still recognisably
    // spherical (and the SDF bake doesn't run away).
    let mut source = icosphere_mesh(3, 1.0); // 1280 tris
    let mut s: u32 = 0xA5A5_BEEF;
    let mut next_noise = || {
        s = s.wrapping_mul(2654435761).wrapping_add(0x12345678);
        ((s as f64) / (u32::MAX as f64)) * 2.0 - 1.0
    };
    for v in source.vertices.iter_mut() {
        // glam's f64 DVec3 has `.normalize()` not `.normalize_or_zero()` —
        // safe here since vertex magnitudes are all ~1.0 by construction.
        let n = *v / v.length();
        let r = 1.0 + next_noise() * 0.10;
        *v = n * r;
    }
    source.build_bvh(4);

    let dec = decimate(&source, source.indices.len() / 4); // ~25 % LOD
    let sdf = bake_mesh_to_sdf(&source, mesh_state.sdf_resolution as usize, 0.15);

    // Stage-24e: bake a sparse SDF in parallel — same dense bake →
    // narrow-band filter. Cheap (re-uses the dense data we already
    // have) and lets the user toggle between dense / sparse storage
    // in the right-rail pane to see the byte delta on a real asset.
    let sparse = sparse_sdf_from_dense(&sdf, mesh_state.sdf_sparse_band());
    mesh_state.dense_sdf_bytes = sdf.data.len() * std::mem::size_of::<f64>();
    mesh_state.sparse_sdf_bytes = sparse.grid.storage_bytes();
    mesh_state.sparse_sdf_leaves = sparse.grid.leaf_count();
    mesh_state.sparse_sdf_active_voxels = sparse.grid.active_voxel_count();

    // Stage 24e.1 — record GPU root hash table stats so the pane
    // can show the load factor + average probe length. The host-
    // side replay of the WGSL probing in
    // `SparseGridGpu::root_hash_probe_stats` is byte-identical to
    // what the GPU does at lookup time (same Murmur3 mix), so this
    // is a faithful telemetry source even when the rest of the
    // pipeline runs on the CPU.
    let packed = molla::geometry::SparseGridGpu::from_grid(&sparse.grid);
    mesh_state.sparse_root_table_capacity = packed.root_table_capacity;
    mesh_state.sparse_root_load_factor = packed.root_table_load_factor();
    let (avg, max) = packed.root_hash_probe_stats();
    mesh_state.sparse_root_avg_probes = avg;
    mesh_state.sparse_root_max_probes = max;

    // The MC surface re-extracted from the sparse SDF needs a
    // dense buffer (marching_cubes only consumes `SdfGrid`); for
    // the toggle's "sparse" case we sample the sparse SDF back into
    // a dense scratch grid so the MC consumer is agnostic. This is a
    // viewer-only convenience — production MC over sparse SDFs would
    // iterate active leaves directly (deferred to Stage 24e.1+).
    let mc_source: molla::geometry::SdfGrid = if mesh_state.use_sparse_sdf {
        sparse_sdf_to_dense_for_mc(&sparse, &sdf)
    } else {
        sdf.clone()
    };
    let mc = marching_cubes(&mc_source, 0.0);

    let hull = convex_hull(&source.vertices);

    mesh_state.original_tris = source.indices.len();
    mesh_state.decimated_tris = dec.indices.len();
    mesh_state.mc_tris = mc.indices.len();
    mesh_state.hull_verts = hull.vertices.len();

    // Lay the three primary panels out side-by-side along +X. The MC
    // overlay uses the same world position as the original so the
    // user can compare them directly.
    let original_offset = Vec3::new(-2.5, 1.5, 0.0);
    let decimated_offset = Vec3::new(0.0, 1.5, 0.0);
    let mc_offset = Vec3::new(2.5, 1.5, 0.0);

    let mat_original = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.55, 0.30),
        metallic: 0.05,
        perceptual_roughness: 0.55,
        cull_mode: None,
        ..default()
    });
    let mat_decimated = materials.add(StandardMaterial {
        base_color: Color::srgb(0.40, 0.70, 0.85),
        metallic: 0.05,
        perceptual_roughness: 0.55,
        cull_mode: None,
        ..default()
    });
    let mat_mc = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.85, 0.40),
        metallic: 0.05,
        perceptual_roughness: 0.55,
        cull_mode: None,
        ..default()
    });
    let mat_hull = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.95, 0.30, 0.6),
        unlit: true,
        cull_mode: None,
        ..default()
    });

    // Original.
    let orig_mesh = molla_trimesh_to_bevy_mesh(&source, true);
    commands.spawn((
        ClothObstacle,
        MeshToolOriginal,
        Mesh3d(meshes.add(orig_mesh)),
        MeshMaterial3d(mat_original),
        Transform::from_translation(original_offset),
        Visibility::Visible,
    ));

    // Decimated.
    let dec_mesh = molla_trimesh_to_bevy_mesh(&dec, true);
    commands.spawn((
        ClothObstacle,
        MeshToolDecimated,
        Mesh3d(meshes.add(dec_mesh)),
        MeshMaterial3d(mat_decimated),
        Transform::from_translation(decimated_offset),
        Visibility::Visible,
    ));

    // Marching-cubes surface.
    let mc_mesh = molla_trimesh_to_bevy_mesh(&mc, true);
    commands.spawn((
        ClothObstacle,
        MeshToolMarchingCubes,
        Mesh3d(meshes.add(mc_mesh)),
        MeshMaterial3d(mat_mc),
        Transform::from_translation(mc_offset),
        Visibility::Visible,
    ));

    // Hull — Stage 21.1 swaps the previous yellow-point markers for a
    // real LineList wireframe built from the hull's edge set. One
    // entity covers the full overlay (cheap; tens of edges for the
    // demo icosphere). Defaults to hidden — toggled on via the
    // right-rail pane so the user sees the original first.
    let hull_wire_mesh = meshes.add(build_hull_wireframe_mesh(&hull));
    commands.spawn((
        ClothObstacle,
        MeshToolHull,
        Mesh3d(hull_wire_mesh),
        MeshMaterial3d(mat_hull),
        Transform::from_translation(original_offset),
        Visibility::Hidden,
    ));
}

// Build a Bevy `LineList` mesh tracing every unique edge of the
// hull. Uses the post-Stage-21.1 [`molla::geometry::ConvexHullMesh`]

fn mesh_tools_pane(
    ui: &mut egui::Ui,
    state: &mut MeshToolsExampleState,
    pending: &mut PendingSceneSwitch,
    scene: Scene,
) {
    ui.label("Source: noise-displaced icosphere (3 subdivs)");
    ui.separator();
    readout(ui, "Original tris", &format!("{}", state.original_tris));
    readout(ui, "Decimated tris", &format!("{}", state.decimated_tris));
    readout(ui, "MC tris", &format!("{}", state.mc_tris));
    readout(ui, "Hull verts", &format!("{}", state.hull_verts));
    readout(ui, "SDF resolution", &format!("{}³", state.sdf_resolution));
    ui.separator();

    ui.label("Visibility:");
    ui.checkbox(&mut state.show_original, "Original (left, orange)");
    ui.checkbox(&mut state.show_decimated, "Decimated (centre, blue)");
    ui.checkbox(
        &mut state.show_marching_cubes,
        "Marching cubes (right, green)",
    );
    ui.checkbox(
        &mut state.show_hull,
        "Hull overlay (yellow wireframe on original)",
    );
    ui.separator();

    // Stage-24e — sparse SDF storage delta. The toggle reflects what
    // the MC surface currently shows; flipping it requires a scene
    // rebuild (press M to re-enter), which the small print below
    // calls out so the user isn't surprised.
    ui.label("Stage 24e: sparse vs dense SDF");
    if ui
        .checkbox(
            &mut state.use_sparse_sdf,
            "Use sparse SDF (re-extract MC) — rebuild to apply",
        )
        .changed()
    {
        state.rebuild_pending = true;
    }
    let dense_kib = state.dense_sdf_bytes as f32 / 1024.0;
    let sparse_kib = state.sparse_sdf_bytes as f32 / 1024.0;
    let ratio = if state.dense_sdf_bytes > 0 {
        sparse_kib / dense_kib
    } else {
        0.0
    };
    readout(ui, "Dense SDF", &format!("{dense_kib:.1} KiB"));
    readout(ui, "Sparse SDF", &format!("{sparse_kib:.1} KiB"));
    readout(ui, "Sparse / dense", &format!("{:.1} %", ratio * 100.0));
    readout(ui, "Active leaves", &format!("{}", state.sparse_sdf_leaves));
    readout(
        ui,
        "Active voxels",
        &format!("{}", state.sparse_sdf_active_voxels),
    );
    readout(
        ui,
        "Narrow band",
        &format!("{} voxels", state.sparse_sdf_band),
    );
    ui.separator();

    if state.sparse_root_table_capacity > 0 {
        readout(
            ui,
            "Root hash capacity",
            &format!("{}", state.sparse_root_table_capacity),
        );
        readout(
            ui,
            "Root load factor",
            &format!("{:.2}", state.sparse_root_load_factor),
        );
        readout(
            ui,
            "Root probes",
            &format!(
                "avg {:.2}, max {}",
                state.sparse_root_avg_probes, state.sparse_root_max_probes
            ),
        );
    }
    if state.rebuild_pending {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Rebuild required to apply scene-build toggles.",
        );
        if ui.button("Rebuild mesh tools").clicked() {
            // Re-enter the current scene unconditionally; the
            // hotkey path (`handle_input`) skips this when the
            // scene is already active.
            pending.0 = Some(scene);
            state.rebuild_pending = false;
        }
    }
    ui.separator();

    ui.label("");
    ui.label("Pipeline runs once on scene entry:");
    ui.label("• decimate(): QEM edge collapses → 25 % LOD");
    ui.label("• bake_mesh_to_sdf(): voxelise + sign by parity");
    ui.label("• sparse_sdf_from_dense(): drop voxels outside band");
    ui.label("• marching_cubes(): re-extract surface");
    ui.label("• convex_hull(): QuickHull from vertex cloud");
    ui.label("");
    ui.small("Hotkey: M");
}

// Per-frame system: read [`MeshToolsExampleState`] and flip the
// `Visibility` of every entity tagged with one of the
// `MeshTool*` markers. Cheap (≤ a few hundred entities); skipping
// the marker queries when `show_*` flags are unchanged would be a
// premature optimisation.
fn update_mesh_tools_visibility(
    state: Res<MeshToolsExampleState>,
    mut q_orig: Query<
        &mut Visibility,
        (
            With<MeshToolOriginal>,
            Without<MeshToolDecimated>,
            Without<MeshToolMarchingCubes>,
            Without<MeshToolHull>,
        ),
    >,
    mut q_dec: Query<
        &mut Visibility,
        (
            With<MeshToolDecimated>,
            Without<MeshToolOriginal>,
            Without<MeshToolMarchingCubes>,
            Without<MeshToolHull>,
        ),
    >,
    mut q_mc: Query<
        &mut Visibility,
        (
            With<MeshToolMarchingCubes>,
            Without<MeshToolOriginal>,
            Without<MeshToolDecimated>,
            Without<MeshToolHull>,
        ),
    >,
    mut q_hull: Query<
        &mut Visibility,
        (
            With<MeshToolHull>,
            Without<MeshToolOriginal>,
            Without<MeshToolDecimated>,
            Without<MeshToolMarchingCubes>,
        ),
    >,
) {
    let v = |on: bool| {
        if on {
            Visibility::Visible
        } else {
            Visibility::Hidden
        }
    };
    for mut vis in q_orig.iter_mut() {
        *vis = v(state.show_original);
    }
    for mut vis in q_dec.iter_mut() {
        *vis = v(state.show_decimated);
    }
    for mut vis in q_mc.iter_mut() {
        *vis = v(state.show_marching_cubes);
    }
    for mut vis in q_hull.iter_mut() {
        *vis = v(state.show_hull);
    }
}

// Body for the right-rail pane in the VBD-cloth scene. Sliders for
// stretch / bend / contact stiffness + iteration count, plus a

