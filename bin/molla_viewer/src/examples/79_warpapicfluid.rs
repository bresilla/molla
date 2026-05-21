// ── Scene 79: WarpApicFluid (Stage 37.6 — APIC FEM-MPM hybrid) ────
//
// 200 particles fall under gravity into a tet-meshed unit cube box
// with no-slip walls. The FEM mesh provides the velocity field; the
// MPM particles deposit + retrieve momentum via APIC weights. Render
// particles as gizmo points + tet wireframe in muted grey.
//
// All compute lives on CPU through `molla::fem::ApicHybrid`; physics
// is short-circuited in `step_simulation`.

const WARP_APIC_GRID: usize = 6;
// Origin offset for the unit-cube mesh — sit it just above the
// ground plane so the gizmo wireframe doesn't clip.
const WARP_APIC_ORIGIN: Vec3 = Vec3::new(-0.5, 0.05, -0.5);
const WARP_APIC_DEFAULT_VISCOSITY: f64 = 0.5;
const WARP_APIC_DEFAULT_PARTICLE_COUNT: usize = 200;

struct WarpApicFluidState {
    /// The hybrid carries both the mesh and the particles.
    hybrid: fem::apic_coupling::ApicHybrid,
    /// Cached no-slip BCs (every boundary vertex). Rebuilt only when
    /// the mesh changes (which it doesn't in v1).
    bcs: Vec<fem::BoundaryCondition>,
    /// Viscosity slider (kinematic, m²/s).
    viscosity: f64,
    /// Live mean-KE (per particle) readout.
    mean_kinetic_energy: f64,
    /// Step counter — surfaced in the pane.
    step_count: u64,
    /// "Reset" button trigger.
    user_reset: bool,
    /// `apply_scene_switch` stamps this so the next update tick re-
    /// initialises the particle layout.
    needs_reset: bool,
}

impl Default for WarpApicFluidState {
    fn default() -> Self {
        let (hybrid, bcs) = build_apic_fluid_state(WARP_APIC_DEFAULT_PARTICLE_COUNT);
        Self {
            hybrid,
            bcs,
            viscosity: WARP_APIC_DEFAULT_VISCOSITY,
            mean_kinetic_energy: 0.0,
            step_count: 0,
            user_reset: false,
            // Don't reset on first build — the constructor already did it.
            needs_reset: false,
        }
    }
}

// Build a fresh `ApicHybrid` (cube mesh + `count` particles in the
// upper half) plus the matching no-slip boundary list.
fn build_apic_fluid_state(
    count: usize,
) -> (fem::apic_coupling::ApicHybrid, Vec<fem::BoundaryCondition>) {
    let mesh = fem::FemMesh::tet_grid_3d(
        WARP_APIC_GRID,
        WARP_APIC_GRID,
        WARP_APIC_GRID,
        1.0,
        1.0,
        1.0,
    );
    // Pseudo-random sample inside the upper half of the cube.
    let mut s: u64 = 0xCAFE_F00D_DEAD_BEEF;
    let next = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f64) / ((1u64 << 31) as f64)
    };
    let mut particles = Vec::with_capacity(count);
    for _ in 0..count {
        let x = molla::math::Vec3::new(
            0.10 + 0.80 * next(&mut s),
            0.10 + 0.80 * next(&mut s),
            // Upper half of the box so they have room to fall.
            0.55 + 0.40 * next(&mut s),
        );
        particles.push(fem::ApicParticle::at_rest(x, 1.0));
    }
    let mut bcs: Vec<fem::BoundaryCondition> = Vec::new();
    for (i, p) in mesh.vertices.iter().enumerate() {
        let on_boundary = p.x <= 1e-9
            || p.x >= 1.0 - 1e-9
            || p.y <= 1e-9
            || p.y >= 1.0 - 1e-9
            || p.z <= 1e-9
            || p.z >= 1.0 - 1e-9;
        if on_boundary {
            bcs.push(fem::BoundaryCondition::DirichletVector {
                vertex: i as u32,
                values: vec![0.0, 0.0, 0.0],
            });
        }
    }
    let mut hybrid = fem::apic_coupling::ApicHybrid::new(mesh, particles);
    hybrid.gravity = molla::math::Vec3::new(0.0, 0.0, -2.0);
    (hybrid, bcs)
}

fn update_warp_apic_fluid(
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut warp: ResMut<WarpExampleStates>,
) {
    if sim.scene != Scene::WarpApicFluid {
        return;
    }
    let st = &mut warp.apic_fluid;
    if st.user_reset || st.needs_reset {
        let count = st.hybrid.particle_count();
        let (h, bcs) = build_apic_fluid_state(count);
        st.hybrid = h;
        st.bcs = bcs;
        st.step_count = 0;
        st.user_reset = false;
        st.needs_reset = false;
    }
    if run.paused {
        return;
    }
    let dt: f64 = 1.0 / 60.0;
    st.hybrid.step(dt, st.viscosity, &st.bcs);
    // Clamp to AABB so the explicit advect doesn't push particles
    // through a no-slip wall when the velocity exceeds (cell_size / dt).
    st.hybrid.clamp_to_aabb(
        molla::math::Vec3::new(0.001, 0.001, 0.001),
        molla::math::Vec3::new(0.999, 0.999, 0.999),
    );
    let n = st.hybrid.particle_count().max(1) as f64;
    st.mean_kinetic_energy = st.hybrid.total_kinetic_energy() / n;
    st.step_count += 1;
}

fn draw_warp_apic_fluid_gizmos(sim: Res<Sim>, warp: Res<WarpExampleStates>, mut gizmos: Gizmos) {
    if sim.scene != Scene::WarpApicFluid {
        return;
    }
    let st = &warp.apic_fluid;
    let origin = WARP_APIC_ORIGIN;
    // Tet wireframe in muted grey.
    use std::collections::HashSet;
    let mesh = &st.hybrid.mesh;
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(mesh.cells.len() * 4);
    let wire_colour = Color::srgb(0.30, 0.30, 0.34);
    for cell in &mesh.cells {
        if let fem::FemCell::Tet(v) = cell {
            let edges = [
                (v[0], v[1]),
                (v[0], v[2]),
                (v[0], v[3]),
                (v[1], v[2]),
                (v[1], v[3]),
                (v[2], v[3]),
            ];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                if !seen.insert(key) {
                    continue;
                }
                let pa = mesh.vertices[a as usize];
                let pb = mesh.vertices[b as usize];
                let av = origin + Vec3::new(pa.x as f32, pa.z as f32, pa.y as f32);
                let bv = origin + Vec3::new(pb.x as f32, pb.z as f32, pb.y as f32);
                gizmos.line(av, bv, wire_colour);
            }
        }
    }
    // Particles as small gizmo crosses in vivid cyan; brighter when
    // moving fast so the viewer can see the velocity field.
    for particle in &st.hybrid.particles {
        let speed = particle.v.length();
        let t = (speed / 2.0).clamp(0.0, 1.0) as f32;
        let colour = Color::srgb(0.40 + 0.55 * t, 0.85, 0.95 - 0.55 * t);
        let p = particle.x;
        let centre = origin + Vec3::new(p.x as f32, p.z as f32, p.y as f32);
        let r = 0.012;
        gizmos.line(
            centre - Vec3::new(r, 0.0, 0.0),
            centre + Vec3::new(r, 0.0, 0.0),
            colour,
        );
        gizmos.line(
            centre - Vec3::new(0.0, r, 0.0),
            centre + Vec3::new(0.0, r, 0.0),
            colour,
        );
        gizmos.line(
            centre - Vec3::new(0.0, 0.0, r),
            centre + Vec3::new(0.0, 0.0, r),
            colour,
        );
    }
}

fn warp_apic_fluid_pane(ui: &mut egui::Ui, st: &mut WarpApicFluidState) {
    ui.label("Warp APIC fluid — Stage 37.6 FEM-MPM hybrid");
    ui.separator();
    readout(
        ui,
        "Mesh",
        &format!(
            "{0}×{0}×{0} tets ({1} verts)",
            WARP_APIC_GRID,
            st.hybrid.mesh.vert_count()
        ),
    );
    readout(ui, "Particles", &format!("{}", st.hybrid.particle_count()));
    readout(ui, "Step", &format!("{}", st.step_count));
    readout(
        ui,
        "Mean KE / particle",
        &format!("{:.4e} J", st.mean_kinetic_energy),
    );
    ui.separator();
    ui.label("Viscosity ν (m²/s)");
    let _ = ui.add(egui::Slider::new(&mut st.viscosity, 0.0..=2.0).text("smoothing"));
    ui.separator();
    if ui.button("Reset").clicked() {
        st.user_reset = true;
    }
    ui.small("Particles → mesh nodes via APIC weights;");
    ui.small("mesh applies gravity + viscous Laplacian + no-slip;");
    ui.small("particles read updated node velocities back.");
    ui.small("Pure Stokes pressure-Poisson is Stage 37.6.1.");
}

