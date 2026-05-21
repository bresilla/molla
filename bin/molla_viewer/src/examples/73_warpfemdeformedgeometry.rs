// ── Scene 73: WarpFemDeformedGeometry (curved annular mesh) ──────
struct WarpFemDeformedState {
    /// Annular vertex positions for visualisation.
    verts: Vec<MVec3>,
    /// Annular triangle indices.
    tris: Vec<[u32; 3]>,
    /// Per-vertex temperature.
    u: Vec<f64>,
    user_reset: bool,
}
impl Default for WarpFemDeformedState {
    fn default() -> Self {
        Self {
            verts: Vec::new(),
            tris: Vec::new(),
            u: Vec::new(),
            user_reset: true,
        }
    }
}

fn update_warp_fem_deformed(sim: Res<Sim>, mut warp: ResMut<WarpExampleStates>) {
    if sim.scene != Scene::WarpFemDeformedGeometry {
        return;
    }
    let st = &mut warp.fem_deformed;
    if !st.user_reset {
        return;
    }
    // Build a curved (annular) tri mesh by mapping the unit square
    // (x,y) → (r sin t, r cos t) with r = y + 0.5, t = π x / 2.
    let nx = 12;
    let ny = 8;
    st.verts.clear();
    st.tris.clear();
    for j in 0..=ny {
        for i in 0..=nx {
            let x = i as f64 / nx as f64;
            let y = j as f64 / ny as f64;
            let r = y + 0.5;
            let t = 0.5 * std::f64::consts::PI * x;
            st.verts.push(MVec3::new(r * t.sin(), 0.0, r * t.cos()));
        }
    }
    let stride = nx + 1;
    for j in 0..ny {
        for i in 0..nx {
            let a = (j * stride + i) as u32;
            let b = a + 1;
            let c = a + stride as u32;
            let d = c + 1;
            st.tris.push([a, b, d]);
            st.tris.push([a, d, c]);
        }
    }
    // T = boundary value: 1 on outer arc, 0 on inner arc; linear blend
    // by radial coordinate (cheap analytic approximation).
    st.u = st
        .verts
        .iter()
        .map(|p| {
            let r = (p.x * p.x + p.z * p.z).sqrt();
            ((r - 0.5) / 1.0).clamp(0.0, 1.0)
        })
        .collect();
    st.user_reset = false;
}

fn draw_warp_fem_deformed_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpFemDeformedGeometry {
        return;
    }
    let st = &warp.fem_deformed;
    let origin = Vec3::new(0.0, 0.05, 0.0);
    for tri in &st.tris {
        let pa = st.verts[tri[0] as usize];
        let pb = st.verts[tri[1] as usize];
        let pc = st.verts[tri[2] as usize];
        let to = |p: MVec3| origin + Vec3::new(p.x as f32, 0.0, p.z as f32);
        let avg = (st.u[tri[0] as usize] + st.u[tri[1] as usize] + st.u[tri[2] as usize]) / 3.0;
        let col = fem_heat_colour(avg);
        gizmos.line(to(pa), to(pb), col);
        gizmos.line(to(pb), to(pc), col);
        gizmos.line(to(pc), to(pa), col);
    }
}

fn warp_fem_deformed_pane(ui: &mut egui::Ui, st: &mut WarpFemDeformedState) {
    ui.label("Warp FEM deformed geometry — port of fem/example_deformed_geometry.py");
    ui.separator();
    readout(ui, "Triangles", &format!("{}", st.tris.len()));
    if ui.button("Rebuild").clicked() {
        st.user_reset = true;
    }
    ui.small("Annular (curved) mesh from unit-square parameterisation;");
    ui.small("temperature = radial blend (analytic).");
}

