// T1/T2 — Volumetric tetra-FEM tire carcass dynamics with inflation pressure.
//
// Five tires sit side-by-side: four at fixed inflation pressures (0, 80,
// 200, 320 kPa) plus one whose pressure cycles 0 → 350 kPa → 0 so the
// audience sees one full inflation/deflation transition. Each tire is a true
// volumetric tetrahedral FEM carcass from `molla_vehicle::TetraTireFem`, with
// an annular tire volume split into tetrahedra:
//
//   * circumferential × radial × width nodes form a real 3-D tire body.
//   * each hexahedral parameter cell is split into tetrahedra.
//   * linear tetra FEM strain/stress produces the internal rubber forces.
//   * the bead/inner radius is coupled to a 1-DoF loaded hub.
//   * outer tread nodes receive pressure traction and floor contact.
//   * a constant downward hub load (2.6 kN) — fixed so the only visible
//     variable between tires is inflation pressure.
//   * pressure also raises the membrane tangent stiffness, so high-pressure
//     tires remain rounder while low-pressure tires squash into wide patches.
//
// The hub is locked in X / Z (1-DoF, free in Y) — what you see is the
// dynamic deflection of the carcass under a fixed load at the five
// pressures. Higher pressure ⇒ stiffer casing ⇒ less deflection ⇒ small
// contact patch. Lower pressure ⇒ casing folds, tire pancakes, wide
// patch. The leftmost "deflated" tire flattens toward the rim while every
// tire keeps the same undeformed radius.
//
// This is *not* a Pacejka force-element model — it's the rubber +
// air carcass FEM behind it. The contact patch and carcass motion are
// emergent from the tetrahedral volume dynamics.

const TIRE_SEGMENTS: usize = 32;
const TIRE_RADIUS: f64 = 0.32; // 13" rim + 195/65/R13 tyre, roughly
const TIRE_INNER_RADIUS: f64 = 0.12;
const HUB_RADIUS: f64 = 0.07; // visual only — sidewall extends to here
const TIRE_WIDTH: f64 = 0.185; // effective tire width (m)
const REFERENCE_PRESSURE_PA: f64 = 320_000.0;

fn tire_carcass_params(pressure: f64) -> molla_vehicle::TetraTireFemParams {
    molla_vehicle::TetraTireFemParams {
        outer_radius: TIRE_RADIUS,
        inner_radius: TIRE_INNER_RADIUS,
        width: TIRE_WIDTH,
        segments: TIRE_SEGMENTS,
        radial_layers: 3,
        width_layers: 3,
        pressure_pa: pressure,
        downforce: 2_600.0,
        young_modulus: 4.0e6,
        sidewall_young_modulus: 2.0e6,
        tread_young_modulus: 8.0e6,
        bead_young_modulus: 2.5e7,
        density: 1_100.0,
        node_damping: 55.0,
        bead_stiffness: 5.8e4,
        pressure_load_scale: 0.0,
        ..Default::default()
    }
}

fn tire_visual_hub_height(pressure: f64) -> f64 {
    let params = molla_vehicle::RealTireParams {
        radius: TIRE_RADIUS,
        width: TIRE_WIDTH,
        pressure_pa: pressure,
        segments: 64,
        ..Default::default()
    };
    let terrain = molla_vehicle::FlatTerrain::new(0.0, 0.9);
    molla_vehicle::solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 2_600.0)
}

fn build_tire_dynamics_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("tire dynamics floor");
    (b.build().expect("tire dynamics model"), Vec::new())
}

/// Per-frame state for `Scene::TireDynamics`. Four volumetric tetra-FEM
/// carcasses at fixed inflation pressures plus a sweep counter for one
/// animated "demo" tire that cycles its pressure to make the inflation/
/// deflation transition visible.
#[derive(Resource)]
struct TireDynamicsState {
    tires: Vec<molla_vehicle::TetraTireFem>,
    /// Viewer runs the tetra node update on the new WGSL kernel by default.
    use_gpu: bool,
    kernel: Option<molla::kernels::TireFemGpuKernel>,
    last_backend: &'static str,
    last_gpu_error: Option<String>,
    /// Pressures (Pa) the four `tires` were created with. The fifth
    /// entry (`animated_pressure`) drives `tires[4]`.
    pressures_pa: [f64; 5],
    /// Elapsed scene time (s) — drives the animated tire's sweep.
    elapsed: f32,
}

impl Default for TireDynamicsState {
    fn default() -> Self {
        Self {
            tires: Vec::new(),
            use_gpu: true,
            kernel: None,
            last_backend: "GPU tetra FEM",
            last_gpu_error: None,
            pressures_pa: [0.0, 80_000.0, 200_000.0, 320_000.0, 0.0],
            elapsed: 0.0,
        }
    }
}

fn update_tire_dynamics_example(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut state: ResMut<TireDynamicsState>,
) {
    if sim.scene != Scene::TireDynamics {
        return;
    }
    if state.tires.is_empty() {
        // Hubs spaced 1.5 m apart along X. Snapshot the pressures
        // first so the push loop doesn't alias `state`.
        let xs = [-2.5_f64, -1.0, 0.5, 2.0, 3.5];
        let pressures = state.pressures_pa;
        for (i, &x) in xs.iter().enumerate() {
            state.tires.push(molla_vehicle::TetraTireFem::new(
                x,
                pressures[i],
                tire_carcass_params(pressures[i]),
            ));
        }
        state.elapsed = 0.0;
    }
    if run.paused {
        return;
    }
    state.elapsed += time.delta_secs();
    // Animated tire pressure cycles 0 → 350 kPa → 0 with a 12-s period
    // so the audience sees one full inflate/deflate transition.
    let phase = (state.elapsed as f64 / 12.0).rem_euclid(1.0);
    let pulse = if phase < 0.5 {
        phase * 2.0
    } else {
        2.0 - phase * 2.0
    };
    let animated = pulse * 350_000.0;
    state.pressures_pa[4] = animated;
    if let Some(t) = state.tires.get_mut(4) {
        t.set_pressure(animated);
    }
    let dt = 1.0 / 120.0;
    if state.use_gpu {
        let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) else {
            state.last_backend = "CPU fallback (no GPU queue)";
            for tire in state.tires.iter_mut() {
                tire.step(dt);
            }
            return;
        };
        if state.kernel.is_none() {
            state.kernel = Some(molla::kernels::TireFemGpuKernel::new(device));
        }
        let Some(kernel) = state.kernel.take() else {
            return;
        };
        let mut gpu_ok = true;
        let mut gpu_error = None;
        let gpu_substeps = 24;
        let gpu_dt = dt / gpu_substeps as f64;
        for tire in state.tires.iter_mut() {
            // Physical pressure is no longer scaled down, so the explicit GPU
            // reference kernel uses many small node steps for stability.
            for _ in 0..gpu_substeps {
                if let Err(err) = step_tetra_tire_gpu(&kernel, device, queue, tire, gpu_dt) {
                    gpu_ok = false;
                    gpu_error = Some(err);
                    break;
                }
            }
            if !gpu_ok {
                break;
            }
        }
        if gpu_ok {
            state.last_backend = "GPU tetra FEM";
            state.last_gpu_error = None;
        } else {
            state.last_backend = "CPU fallback (GPU error)";
            state.last_gpu_error = gpu_error;
            for tire in state.tires.iter_mut() {
                tire.step(dt);
            }
        }
        state.kernel = Some(kernel);
    } else {
        state.last_backend = "CPU tetra FEM";
        state.last_gpu_error = None;
        for tire in state.tires.iter_mut() {
            tire.step(dt);
        }
    }
}

fn step_tetra_tire_gpu(
    kernel: &molla::kernels::TireFemGpuKernel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tire: &mut molla_vehicle::TetraTireFem,
    dt: f64,
) -> Result<(), String> {
    let snap = tire.gpu_snapshot();
    let hub = tire.hub();
    let input = molla::kernels::TireFemGpuInput {
        rest_positions: snap.rest_positions,
        positions: snap.positions,
        velocities: snap.velocities,
        masses: snap.masses,
        tet_indices: snap.tet_indices,
        tet_grads: snap.tet_grads,
        tet_volumes: snap.tet_volumes,
        tet_mu: snap.tet_mu,
        tet_lambda: snap.tet_lambda,
        bead_ids: snap.bead_ids,
        bead_offsets: snap.bead_offsets,
        tread_ids: snap.tread_ids,
        params: molla::kernels::TireFemGpuParams {
            dt: dt as f32,
            young_modulus: 4.0e6,
            poisson_ratio: 0.46,
            pressure_pa: tire.pressure() as f32,
            node_damping: 55.0,
            bead_stiffness: 5.8e4,
            floor_stiffness: 2.8e5,
            floor_damping: 1_200.0,
            pressure_area_per_tread_node: snap.pressure_area_per_tread_node,
            hub: [hub[0] as f32, hub[1] as f32, hub[2] as f32],
            gravity_y: -9.81,
        },
    };
    let out = kernel
        .dispatch_blocking(device, queue, &input)
        .map_err(|e| e.to_string())?;
    tire.apply_gpu_node_step(&out.positions, &out.velocities, dt)
        .map_err(str::to_owned)
}

fn point3(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1].max(0.0) as f32, p[2] as f32)
}

fn draw_tetra_tire_fem_3d(
    gizmos: &mut Gizmos,
    tire: &molla_vehicle::TetraTireFem,
    ring_color: Color,
    hub_color: Color,
    spoke_color: Color,
    contact_color: Color,
) {
    let n = tire.segments();
    let nodes = tire.nodes();
    let hub = tire.hub();

    // Draw the actual outer/side/bead surface edges of the tetra mesh.
    for &(a, b) in tire.surface_edges() {
        let color = if nodes[a][1] < 1.0e-3 || nodes[b][1] < 1.0e-3 {
            contact_color
        } else if (a + b) % 2 == 0 {
            Color::srgb(0.18, 0.22, 0.28)
        } else {
            ring_color
        };
        gizmos.line(point3(nodes[a]), point3(nodes[b]), color);
    }

    // Sparse spokes from hub to tread nodes so the loaded rim/tire relation
    // remains readable without hiding the tetra surface wireframe.
    let half_w = tire.width() as f32 * 0.5;
    let hub_left = Vec3::new(hub[0] as f32, hub[1] as f32, -half_w);
    let hub_right = Vec3::new(hub[0] as f32, hub[1] as f32, half_w);
    for (k, &id) in tire.tread_nodes().iter().enumerate().step_by(6) {
        let hub_side = if k % 2 == 0 { hub_left } else { hub_right };
        gizmos.line(hub_side, point3(nodes[id]), spoke_color);
    }
    gizmos.line(hub_left, hub_right, hub_color);
    gizmos.sphere(
        Vec3::new(hub[0] as f32, hub[1] as f32, 0.0),
        HUB_RADIUS as f32 * 1.5,
        hub_color,
    );

    // Highlight actual floor-contact points on the tread surface.
    for &id in tire.tread_nodes() {
        let p = nodes[id];
        if p[1] < 1.0e-3 {
            gizmos.sphere(point3(p), 0.025, contact_color);
        }
    }

    // Tiny vertical tick with tetra count encoded by the surface density.
    let _ = n;
}

fn draw_tire_dynamics_gizmos(
    sim: Res<Sim>,
    state: Res<TireDynamicsState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::TireDynamics {
        return;
    }
    let ring_color = Color::srgb(0.28, 0.55, 0.95);
    let hub_color = Color::srgb(0.85, 0.85, 0.92);
    let contact_color = Color::srgb(0.95, 0.50, 0.30);
    let spoke_color = Color::srgb(0.45, 0.45, 0.55);
    for (idx, tire) in state.tires.iter().enumerate() {
        draw_tetra_tire_fem_3d(
            &mut gizmos,
            tire,
            ring_color,
            hub_color,
            spoke_color,
            contact_color,
        );
        // Contact-patch width indicator: short horizontal segment at
        // y = 0 spanning the leftmost-to-rightmost contact particle on
        // both side circles, so the patch reads in 3D.
        let patch = tire.contact_patch_width();
        if patch > 1.0e-3 {
            let xs: Vec<f64> = tire
                .tread_nodes()
                .iter()
                .map(|&id| tire.nodes()[id])
                .filter(|p| p[1] < 1.0e-3)
                .map(|p| p[0])
                .collect();
            let xmin = xs.iter().cloned().fold(f64::INFINITY, f64::min);
            let xmax = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let half_w = TIRE_WIDTH as f32 * 0.5;
            for &z in &[-half_w, half_w] {
                gizmos.line(
                    Vec3::new(xmin as f32, 0.0, z),
                    Vec3::new(xmax as f32, 0.0, z),
                    contact_color,
                );
            }
        }
        // Pressure bar above each tire — height ∝ pressure / 400 kPa.
        let p_kpa = state.pressures_pa[idx] / 1000.0;
        let bar_h = (p_kpa / 400.0).clamp(0.0, 1.5) as f32;
        let hub = tire.hub();
        let bar_base = Vec3::new(hub[0] as f32, 1.5, 0.0);
        let bar_top = bar_base + Vec3::new(0.0, bar_h, 0.0);
        gizmos.line(bar_base, bar_top, Color::srgb(0.95, 0.85, 0.30));
    }
    // Pressure-axis ticks above the tires (50, 150, 300 kPa references).
    let tick_color = Color::srgb(0.4, 0.4, 0.45);
    for &kpa in &[100.0_f32, 200.0, 300.0] {
        let y = 1.5 + (kpa / 400.0).clamp(0.0, 1.5);
        gizmos.line(
            Vec3::new(-3.0, y, 0.0),
            Vec3::new(4.0, y, 0.0),
            tick_color,
        );
    }
}

fn tire_dynamics_pane(ui: &mut egui::Ui, state: &mut TireDynamicsState) {
    ui.label("Tire tetra FEM backend");
    let was_gpu = state.use_gpu;
    let _ = ui.checkbox(&mut state.use_gpu, "Use GPU tire FEM kernel");
    if was_gpu != state.use_gpu {
        state.last_gpu_error = None;
    }
    readout(ui, "Backend", state.last_backend);
    readout(ui, "Tires", &format!("{}", state.tires.len()));
    if let Some(tire) = state.tires.first() {
        readout(ui, "Tets/tire", &format!("{}", tire.tetra_count()));
    }
    if let Some(err) = &state.last_gpu_error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 80), err);
    }
    ui.small("GPU path updates FEM nodes; hub reaction reduction is still CPU-side until the next kernel.");
}
