// --- Stage-20.2 JELLY (tetrahedral soft body) scene ----------------
// 8×8×8 tet jelly cube drops onto a static plane and bounces against
// a kinematic "puller" body. Right-rail pane carries Young's modulus
// + Poisson-ratio sliders + a wireframe toggle.
const JELLY_GRID: u32 = 8;
const JELLY_SIDE: f32 = 0.8;
const JELLY_TOP_Y: f32 = 3.0;
const JELLY_PARTICLE_MASS: f64 = 0.05;
// Default Young's modulus (Pa). Soft rubber regime — gives a
// visibly squishy fall + bounce.
const JELLY_YOUNGS: f64 = 5.0e5;
// Default Poisson ratio. ν = 0.4 is the typical rubber default.
const JELLY_POISSON: f64 = 0.4;
// Soft-contact stiffness (N/m).
const JELLY_CONTACT_STIFFNESS: f64 = 5.0e4;
// Kinematic puller body pose — sits a metre to the +X of the cube
// drop point. The viewer doesn't animate it (yet); the user can
// drag-and-throw via the existing mouse-grab if they want.
const JELLY_PULLER_RADIUS: f32 = 0.18;
const JELLY_PULLER_POS: bevy::math::Vec3 = bevy::math::Vec3::new(0.6, 0.18, 0.0);

// Stage-20.2 JELLY tetrahedral soft-body demo runtime state. Driven
// by the right-rail sliders; applied to the active solver each frame
// in `update_jelly_example`. Lazy-initialised on entry to
// `Scene::Jelly`; survives scene switches so the user's tuning
// persists.
//
// Stage 20.3 extensions:
// - per-region material picker (`material_preset`) — chooses one of
//   the `JellyMaterialPreset` named recipes when the model is built,
//   driving the per-tet `(E, ν, σ_y, H)` parameters via
//   `add_tet_mesh_per_material`.
// - plasticity sliders (`yield_stress`, `hardening`).
// - "compress" trigger — the kinematic puller is briefly accelerated
//   downward to drive the cube past yield and exhibit permanent set.
// - plastic-strain heat-map toggle (`show_plastic_strain`).
#[derive(Resource)]
struct JellyExampleState {
    /// Per-tet Young's modulus (Pa). Surfaced via the slider.
    youngs_modulus: f64,
    /// Per-tet Poisson ratio. ν ∈ [-0.49, 0.49] enforced when applied
    /// (≥ 0.5 is the incompressible limit and stiffens the volumetric
    /// Hessian unboundedly).
    poisson_ratio: f64,
    /// Soft-contact stiffness (N/m).
    contact_stiffness: f64,
    /// VBD outer-iteration count per substep.
    inner_iterations: u32,
    /// Show the per-tet wireframe overlay (gizmo lines along tet
    /// edges, deduped at draw time).
    show_wireframe: bool,

    // ── Stage 20.3 — multi-material + plasticity ─────────────────
    /// Material preset index — Soft / Stiff / Rigid. Re-builds the
    /// model with the new per-tet `MaterialNeoHookean` when changed.
    material_preset: JellyMaterialPreset,
    /// von Mises yield stress σ_y0 (Pa). `Real::INFINITY` (encoded
    /// here as 0.0 sentinel = "elastic only") disables plasticity.
    yield_stress: f64,
    /// Linear hardening modulus H (Pa).
    hardening: f64,
    /// Toggle the plastic-strain colour overlay on the tet wireframe.
    /// When on, draws each tet's edge with a colour proportional to
    /// its accumulated equivalent plastic strain (cool → hot).
    show_plastic_strain: bool,
    /// Latest max-ε_p readout (filled in by `update_jelly_example`).
    last_max_plastic_strain: f64,
    /// One-shot trigger fired by the "Compress" button. The puller
    /// body's kinematic Y-position is pulled down rapidly by
    /// `update_jelly_example` while this is non-zero, then decays
    /// back. A small impulsive squash is enough to drive the cube
    /// past yield with the default plasticity settings.
    compress_t: f32,

    // ── Stage 20.4 extensions ────────────────────────────────────
    /// Spawn a second jelly cube above the first to demonstrate
    /// tet-vs-tet collision. Triggers a model rebuild on toggle.
    second_cube: bool,
    /// Toggle kinematic-hardening (Stage 20.4) for the per-material
    /// `c` modulus. When on, the back-stress α evolves per Prager;
    /// when off, the model uses Stage-20.3-style isotropic hardening
    /// only. Triggers a model rebuild on toggle.
    kinematic_hardening_on: bool,
    /// Per-tet kinematic-hardening modulus c (Pa). Surfaced when
    /// `kinematic_hardening_on` is true.
    kinematic_hardening_modulus: f64,
    /// Latest max ‖α‖ (Frobenius) across all tets — readout only.
    last_max_back_stress: f64,
    /// SDF-backed rigid shape selection. `None` keeps the analytic
    /// puller sphere; the other variants register a baked / sparse
    /// SDF on solver re-creation.
    sdf_shape: JellySdfShape,
    /// Set to `true` when the user touches a control whose effect is
    /// only realised on a scene rebuild (`second_cube`, `sdf_shape`).
    /// Cleared on rebuild. Surfaces a "Rebuild jelly" button so the
    /// pane doesn't lie about needing the user to re-press Y (the
    /// hotkey is a no-op when the scene is already active).
    rebuild_pending: bool,
}

// Stage 20.4 — picks which `SdfGrid` (or none) to register with the
// VBD solver as a rigid contact target. Triggers a re-create of the
// solver when changed (the SDF backings live on the solver).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
enum JellySdfShape {
    /// No SDF backing — use the analytic puller sphere only.
    #[default]
    None,
    /// Baked dense SDF of a sphere (validates the SdfGrid path).
    BakedSphere,
    /// Baked dense SDF of a torus (qualitative demo of mesh-derived
    /// SDF contact).
    BakedTorus,
}

// Material recipe for `JellyExampleState::material_preset`. Each
// option maps to one or more `MaterialNeoHookean` instances at
// model-build time. v1 ships three named presets — easy to extend
// (add a variant + a `materials_for_preset` arm).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
enum JellyMaterialPreset {
    #[default]
    Soft,
    Stiff,
    /// Soft interior with stiff corner cells — multi-material recipe
    /// that drives `add_tet_mesh_per_material` end-to-end.
    Mixed,
}

impl Default for JellyExampleState {
    fn default() -> Self {
        Self {
            youngs_modulus: JELLY_YOUNGS,
            poisson_ratio: JELLY_POISSON,
            contact_stiffness: JELLY_CONTACT_STIFFNESS,
            inner_iterations: 10,
            show_wireframe: true,
            material_preset: JellyMaterialPreset::Soft,
            yield_stress: 0.0, // 0 = "elastic only" (sentinel, see jelly_pane)
            hardening: 0.0,
            show_plastic_strain: false,
            last_max_plastic_strain: 0.0,
            compress_t: 0.0,
            second_cube: false,
            kinematic_hardening_on: false,
            kinematic_hardening_modulus: 5.0e3,
            last_max_back_stress: 0.0,
            sdf_shape: JellySdfShape::None,
            rebuild_pending: false,
        }
    }
}

fn build_jelly_model_with_preset(preset: JellyMaterialPreset) -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.4);

    let n = JELLY_GRID;
    let side = JELLY_SIDE as f64;
    let dx = side / (n as f64 - 1.0);
    let centre = MVec3::new(0.0, JELLY_TOP_Y as f64, 0.0);
    let origin = centre - MVec3::splat(side * 0.5);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + MVec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                b.add_particle(p, JELLY_PARTICLE_MASS, dx * 0.5, mp::WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { (k * n + j) * n + i };
    let mut tets = Vec::new();
    let mut mat_idx: Vec<u32> = Vec::new();
    let last = n - 2;
    for kk in 0..n - 1 {
        for jj in 0..n - 1 {
            for ii in 0..n - 1 {
                let corners = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                let cell_tets = jelly_cube_to_six_tets(corners);
                let mi: u32 = match preset {
                    JellyMaterialPreset::Soft => 0,
                    JellyMaterialPreset::Stiff => 0,
                    JellyMaterialPreset::Mixed => {
                        // Stiff (1) on the four vertical corner pillars;
                        // soft (0) elsewhere.
                        if (ii == 0 || ii == last) && (kk == 0 || kk == last) {
                            1
                        } else {
                            0
                        }
                    }
                };
                for tet in cell_tets {
                    tets.push(tet);
                    mat_idx.push(mi);
                }
            }
        }
    }
    let materials: Vec<mp::MaterialNeoHookean> = match preset {
        JellyMaterialPreset::Soft => {
            vec![mp::MaterialNeoHookean::elastic(JELLY_YOUNGS, JELLY_POISSON)]
        }
        JellyMaterialPreset::Stiff => vec![mp::MaterialNeoHookean::elastic(
            JELLY_YOUNGS * 100.0,
            JELLY_POISSON,
        )],
        JellyMaterialPreset::Mixed => vec![
            mp::MaterialNeoHookean::elastic(JELLY_YOUNGS, JELLY_POISSON),
            mp::MaterialNeoHookean::elastic(JELLY_YOUNGS * 100.0, JELLY_POISSON),
        ],
    };
    let _ = b
        .add_tet_mesh_per_material(&tets, &materials, &mat_idx, None)
        .expect("add_tet_mesh_per_material");

    // Floor plane.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    // Kinematic puller body — small sphere off to +X. The cube falls
    // against it on its way down, demonstrating soft-on-rigid contact.
    let puller_pos = MVec3::new(
        JELLY_PULLER_POS.x as f64,
        JELLY_PULLER_POS.y as f64,
        JELLY_PULLER_POS.z as f64,
    );
    let puller = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(0.5, JELLY_PULLER_RADIUS as f64))
            .with_position(puller_pos)
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(puller, JELLY_PULLER_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("puller shape");

    let model = b.build().expect("jelly model");
    let visuals = vec![BodyVisual::Sphere(JELLY_PULLER_RADIUS)];
    (model, visuals)
}

// Back-compat shim — Stage 20.2 callers used `build_jelly_model()`
// without preset. Default to Soft.
fn build_jelly_model() -> (mp::Model, Vec<BodyVisual>) {
    build_jelly_model_with_preset(JellyMaterialPreset::Soft)
}

// Per-tet Young's-modulus multiplier for a jelly material preset, in
// the exact tet order `build_jelly_model_with_preset` emits them
// (k → j → i, six tets per cell). `update_jelly_example` scales the
// slider-derived (μ, λ) by this so the Mixed preset's stiff corner
// pillars (and the Stiff preset's ×100) survive the per-frame
// re-stamp instead of being flattened to a single uniform material.
fn jelly_tet_e_multipliers(preset: JellyMaterialPreset) -> Vec<f64> {
    let n = JELLY_GRID;
    let last = n - 2;
    let mut mul = Vec::with_capacity(((n - 1) * (n - 1) * (n - 1) * 6) as usize);
    for kk in 0..n - 1 {
        for _jj in 0..n - 1 {
            for ii in 0..n - 1 {
                let m = match preset {
                    JellyMaterialPreset::Soft => 1.0,
                    JellyMaterialPreset::Stiff => 100.0,
                    JellyMaterialPreset::Mixed => {
                        if (ii == 0 || ii == last) && (kk == 0 || kk == last) {
                            100.0
                        } else {
                            1.0
                        }
                    }
                };
                for _ in 0..6 {
                    mul.push(m);
                }
            }
        }
    }
    mul
}

fn update_jelly_example(mut sim: ResMut<Sim>, mut state: ResMut<JellyExampleState>) {
    if sim.scene != Scene::Jelly {
        return;
    }
    if let SolverBackend::Vbd(s) = &mut sim.solver {
        s.contact_stiffness = state.contact_stiffness;
        s.iterations = state.inner_iterations as usize;
    }
    // Push fresh (E, ν) → (μ, λ) into the model's per-tet buffers.
    // Stage 20.3 — also push fresh per-tet yield + hardening so the
    // plasticity sliders take effect without rebuilding the model.
    // Multi-material presets are baked at scene-build time, so the
    // per-tet uniformity here applies on top of whatever per-tet
    // material-id breakdown the preset chose.
    let nu = state.poisson_ratio.clamp(-0.49, 0.49);
    let tet_count = sim.model.tet_count;
    if tet_count == 0 {
        return;
    }
    // Re-stamp per-tet (μ, λ) from the sliders. The slider sets the
    // BASE Young's modulus; the active preset's per-tet multiplier is
    // applied on top, so the Mixed preset's stiff corner pillars
    // (E ×100) and the Stiff preset's uniform ×100 survive. A flat
    // overwrite here used to collapse every preset to one material on
    // the first frame, making the multi-material picker a no-op.
    let e_base = state.youngs_modulus;
    let e_mul = jelly_tet_e_multipliers(state.material_preset);
    let lame = |mul: f64| -> (f64, f64) {
        let e = (e_base * mul).max(1.0).min(1.0e9);
        let mu = e / (2.0 * (1.0 + nu));
        let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        (mu, lam)
    };
    {
        let mu_host = sim.model.tet_mu.host_mut().expect("tet_mu host");
        for (idx, v) in mu_host.iter_mut().enumerate() {
            *v = lame(e_mul.get(idx).copied().unwrap_or(1.0)).0;
        }
    }
    {
        let lambda_host = sim.model.tet_lambda.host_mut().expect("tet_lambda host");
        for (idx, v) in lambda_host.iter_mut().enumerate() {
            *v = lame(e_mul.get(idx).copied().unwrap_or(1.0)).1;
        }
    }
    // Plasticity uniforms — `yield_stress = 0` is the UI sentinel for
    // "elastic only"; we encode that on the device side as a very
    // large finite value (>1e29) to short-circuit the WGSL plasticity
    // kernel. Real::INFINITY tripped some f32 cast issues so we use
    // a large number directly.
    let sigma_y_device = if state.yield_stress > 0.0 {
        state.yield_stress
    } else {
        1.0e30
    };
    let h_device = state.hardening.max(0.0);
    {
        let yield_host = sim
            .model
            .tet_yield_stress
            .host_mut()
            .expect("tet_yield_stress host");
        for v in yield_host.iter_mut() {
            *v = sigma_y_device;
        }
    }
    {
        let hard_host = sim
            .model
            .tet_hardening
            .host_mut()
            .expect("tet_hardening host");
        for v in hard_host.iter_mut() {
            *v = h_device;
        }
    }
    if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
        let device = device.clone();
        let queue = queue.clone();
        let _ = sim.model.tet_mu.ensure_device(&device);
        let _ = sim.model.tet_mu.upload(&queue);
        let _ = sim.model.tet_lambda.ensure_device(&device);
        let _ = sim.model.tet_lambda.upload(&queue);
        let _ = sim.model.tet_yield_stress.ensure_device(&device);
        let _ = sim.model.tet_yield_stress.upload(&queue);
        let _ = sim.model.tet_hardening.ensure_device(&device);
        let _ = sim.model.tet_hardening.upload(&queue);
    }
    // Compress button: drive the kinematic puller body downward
    // briefly when `compress_t` was bumped by the UI. Pull rate
    // decays geometrically each frame — total impulse is roughly
    // `compress_t / 60` revolutions of the puller body.
    if state.compress_t > 0.0 {
        if sim.model.body_count > 0 {
            let bq_host = sim.state_a.body_q.host_mut().expect("body_q");
            if let Some(b) = bq_host.first_mut() {
                let dy = -0.02 * state.compress_t as f64;
                b.position.y += dy;
            }
        }
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.body_q.ensure_device(&device);
            let _ = sim.state_a.body_q.upload(&queue);
        }
        state.compress_t = (state.compress_t - 0.1).max(0.0);
    }
    // Read back the max plastic-strain accumulator for the readout.
    if state.show_plastic_strain || state.last_max_plastic_strain > 0.0 {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.tet_plastic_strain.download(&device, &queue);
        }
        if let Ok(eps) = sim.state_a.tet_plastic_strain.host() {
            state.last_max_plastic_strain = eps.iter().cloned().fold(0.0_f64, f64::max);
        }
    }
    // Stage 20.4 — back-stress readout. Only download when the
    // kinematic-hardening toggle is on (avoid the per-frame device
    // download cost for the default-off case).
    if state.kinematic_hardening_on && sim.model.tet_count > 0 {
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.state_a.tet_back_stress.download(&device, &queue);
        }
        if let Ok(alphas) = sim.state_a.tet_back_stress.host() {
            // Frobenius norm — same metric as the v4 GPU test.
            let mut max_norm: f64 = 0.0;
            for a in alphas {
                let frob = (a.x_axis.length_squared()
                    + a.y_axis.length_squared()
                    + a.z_axis.length_squared())
                .sqrt();
                if frob > max_norm {
                    max_norm = frob;
                }
            }
            state.last_max_back_stress = max_norm;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn quadrotor_pd_thrusts(
    total_mass: f64,
    arm_length: f64,
    target_altitude: f64,
    frame_pos_y: f64,
    frame_vel_y: f64,
    body_y_world: MVec3,
    frame_w: MVec3,
    altitude_kp: f64,
    altitude_kd: f64,
    attitude_kp: f64,
    attitude_kd: f64,
) -> [f64; 4] {
    let g = molla::math::GRAVITY;
    let alt_err = target_altitude - frame_pos_y;
    let total_thrust = total_mass * g + altitude_kp * alt_err - altitude_kd * frame_vel_y;
    let baseline = (total_thrust / 4.0).max(0.0);

    let tilt = body_y_world.cross(MVec3::Y); // small-angle: ~(roll, _, pitch) error
    let roll_err = tilt.x; // rotation about world +X
    let pitch_err = tilt.z; // rotation about world +Z
    let roll_cmd = attitude_kp * roll_err - attitude_kd * frame_w.x;
    let pitch_cmd = attitude_kp * pitch_err - attitude_kd * frame_w.z;
    let l = arm_length.max(1.0e-9);
    // The ±X (pitch) and ±Z (roll) rotor pairs sit on perpendicular
    // lever arms, so `r × F` has OPPOSITE sign for the two pairs: the
    // +X rotor pushed up yields +Z torque, but the +Z rotor pushed up
    // yields −X torque. The roll pair must therefore be split the
    // opposite way from the pitch pair, or the attitude loop becomes
    // positive feedback on roll and the quadrotor tumbles.
    let mut thrusts = [
        baseline + 0.5 * pitch_cmd / l, // rotor 0 (+X)
        baseline - 0.5 * pitch_cmd / l, // rotor 1 (-X)
        baseline - 0.5 * roll_cmd / l,  // rotor 2 (+Z)
        baseline + 0.5 * roll_cmd / l,  // rotor 3 (-Z)
    ];
    for t in thrusts.iter_mut() {
        if *t < 0.0 {
            *t = 0.0;
        }
    }
    thrusts
}

fn draw_jelly_gizmos(sim: Res<Sim>, state: Res<JellyExampleState>, mut gizmos: Gizmos) {
    if !matches!(
        sim.scene,
        Scene::Jelly
            | Scene::DiffsimSoftBody
            | Scene::SoftbodyDrop
            | Scene::FastSoftImpact
            | Scene::SoftBrake
    ) {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    // For SoftbodyDrop / FastSoftImpact / SoftBrake the jelly tet
    // wireframe is always shown — those demos' whole point is soft-
    // on-soft contact.
    if sim.scene == Scene::Jelly && !state.show_wireframe {
        return;
    }
    let Ok(tet_idx) = sim.model.tet_indices.host() else {
        return;
    };
    let n_tets = sim.model.tet_count;
    if n_tets == 0 {
        return;
    }
    // Stage 20.3 — plastic-strain heat-map. When the toggle is on we
    // colour each tet's edges by its accumulated equivalent plastic
    // strain ε_p (cool blue → hot red over [0, 1]). Without the
    // toggle we use the Stage 20.2 uniform green and dedup edges
    // across tets to keep the line count manageable.
    let plastic_strain = if state.show_plastic_strain {
        sim.state_a
            .tet_plastic_strain
            .host()
            .ok()
            .map(|s| s.to_vec())
    } else {
        None
    };
    use std::collections::HashSet;
    let mut seen = HashSet::<(u32, u32)>::with_capacity(n_tets * 4);
    let default_edge_color = Color::srgb(0.40, 0.85, 0.40);
    for t in 0..n_tets {
        let v0 = tet_idx[t * 4];
        let v1 = tet_idx[t * 4 + 1];
        let v2 = tet_idx[t * 4 + 2];
        let v3 = tet_idx[t * 4 + 3];
        let edge_color = if let Some(eps) = &plastic_strain {
            let e = eps.get(t).copied().unwrap_or(0.0).clamp(0.0, 1.0) as f32;
            // Cool → hot blend (blue → red).
            Color::srgb(e, 0.10 + 0.20 * (1.0 - e), 1.0 - e)
        } else {
            default_edge_color
        };
        for &(a, b) in &[(v0, v1), (v0, v2), (v0, v3), (v1, v2), (v1, v3), (v2, v3)] {
            let key = if a < b { (a, b) } else { (b, a) };
            // When the heat-map is on we don't dedup — every tet
            // contributes its colour to every shared edge — but we
            // still skip if we already drew it FOR THIS TET'S
            // colour (avoid four overlapping draws per shared edge).
            if plastic_strain.is_some() || seen.insert(key) {
                let pa = pq[a as usize];
                let pb = pq[b as usize];
                gizmos.line(
                    Vec3::new(pa.x as f32, pa.y as f32, pa.z as f32),
                    Vec3::new(pb.x as f32, pb.y as f32, pb.z as f32),
                    edge_color,
                );
            }
        }
    }
}

// While dragging, draw a yellow line from the picked anchor (body or
// cloth particle) to the cursor target so the user can see what

fn jelly_pane(
    ui: &mut egui::Ui,
    state: &mut JellyExampleState,
    sim: &mut Sim,
    pending: &mut PendingSceneSwitch,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Tets", &format!("{}", sim.model.tet_count));
    readout(ui, "Tet meshes", &format!("{}", sim.model.tet_mesh_count));
    readout(
        ui,
        "Colours",
        &format!(
            "{} (particle), {} (spring)",
            sim.model.particle_color_count, sim.model.spring_color_count
        ),
    );
    ui.separator();

    ui.label("Young's modulus E (Pa)");
    let _ = ui.add(egui::Slider::new(&mut state.youngs_modulus, 1.0e3..=1.0e8).logarithmic(true));
    ui.label("Poisson ratio ν");
    let _ = ui.add(egui::Slider::new(&mut state.poisson_ratio, -0.49..=0.49));
    ui.label("Soft contact stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.contact_stiffness, 1.0e3..=1.0e6).logarithmic(true))
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = state.contact_stiffness;
        }
    }
    ui.label("Inner iterations");
    if ui
        .add(egui::Slider::new(&mut state.inner_iterations, 1..=30))
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.iterations = state.inner_iterations as usize;
        }
    }
    ui.separator();

    ui.checkbox(&mut state.show_wireframe, "Show tet wireframe");
    ui.separator();

    // ── Stage 20.3 — multi-material picker ──────────────────────
    ui.label("Material preset");
    let mut preset_changed = false;
    egui::ComboBox::from_id_salt("jelly-material-preset")
        .selected_text(match state.material_preset {
            JellyMaterialPreset::Soft => "Soft (uniform)",
            JellyMaterialPreset::Stiff => "Stiff (uniform)",
            JellyMaterialPreset::Mixed => "Mixed (soft + stiff pillars)",
        })
        .show_ui(ui, |ui| {
            for (variant, label) in [
                (JellyMaterialPreset::Soft, "Soft (uniform)"),
                (JellyMaterialPreset::Stiff, "Stiff (uniform)"),
                (JellyMaterialPreset::Mixed, "Mixed (soft + stiff pillars)"),
            ] {
                if ui
                    .selectable_value(&mut state.material_preset, variant, label)
                    .clicked()
                {
                    preset_changed = true;
                }
            }
        });
    if preset_changed {
        // The per-tet material assignment is baked at scene-build
        // time. To avoid a full Bevy entity rebuild, we re-stamp the
        // per-tet (μ, λ) buffers in place. Compute the per-tet
        // breakdown in a Vec first (decoupled from any host_mut
        // borrows) so we don't run into glam's "two host_mut at
        // once" aliasing.
        let n = JELLY_GRID;
        let (e_soft, e_stiff) = match state.material_preset {
            JellyMaterialPreset::Soft => (state.youngs_modulus, state.youngs_modulus),
            JellyMaterialPreset::Stiff => {
                (state.youngs_modulus * 100.0, state.youngs_modulus * 100.0)
            }
            JellyMaterialPreset::Mixed => (state.youngs_modulus, state.youngs_modulus * 100.0),
        };
        let nu = state.poisson_ratio.clamp(-0.49, 0.49);
        let lame = |e: f64| {
            let e = e.max(1.0).min(1.0e9);
            let mu = e / (2.0 * (1.0 + nu));
            let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            (mu, lam)
        };
        let (mu_soft, lam_soft) = lame(e_soft);
        let (mu_stiff, lam_stiff) = lame(e_stiff);
        let last = n - 2;
        // The 6 tets per cell are stored contiguously in tet order
        // matching `build_jelly_model_with_preset`'s loop nesting
        // (k → j → i → 6 tets per cell).
        let tet_count = sim.model.tet_count;
        let mut new_mu = vec![mu_soft; tet_count];
        let mut new_lam = vec![lam_soft; tet_count];
        let mut tet_id = 0usize;
        for kk in 0..n - 1 {
            for _jj in 0..n - 1 {
                for ii in 0..n - 1 {
                    let stiff_cell = (ii == 0 || ii == last) && (kk == 0 || kk == last);
                    let (mu_c, lam_c) = match state.material_preset {
                        JellyMaterialPreset::Mixed if stiff_cell => (mu_stiff, lam_stiff),
                        JellyMaterialPreset::Stiff => (mu_stiff, lam_stiff),
                        _ => (mu_soft, lam_soft),
                    };
                    for _ in 0..6 {
                        if tet_id < tet_count {
                            new_mu[tet_id] = mu_c;
                            new_lam[tet_id] = lam_c;
                            tet_id += 1;
                        }
                    }
                }
            }
        }
        if let Ok(mus) = sim.model.tet_mu.host_mut() {
            for (dst, src) in mus.iter_mut().zip(new_mu.iter()) {
                *dst = *src;
            }
        }
        if let Ok(lams) = sim.model.tet_lambda.host_mut() {
            for (dst, src) in lams.iter_mut().zip(new_lam.iter()) {
                *dst = *src;
            }
        }
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.model.tet_mu.ensure_device(&device);
            let _ = sim.model.tet_mu.upload(&queue);
            let _ = sim.model.tet_lambda.ensure_device(&device);
            let _ = sim.model.tet_lambda.upload(&queue);
        }
    }
    ui.separator();

    // ── Stage 20.3 — plasticity sliders ─────────────────────────
    ui.label("Yield stress σ_y (Pa) — 0 = elastic only");
    let _ = ui.add(egui::Slider::new(&mut state.yield_stress, 0.0..=1.0e5).logarithmic(true));
    ui.label("Hardening modulus H (Pa)");
    let _ = ui.add(egui::Slider::new(&mut state.hardening, 0.0..=1.0e5).logarithmic(true));
    ui.checkbox(
        &mut state.show_plastic_strain,
        "Show plastic-strain heat-map",
    );
    readout(
        ui,
        "Max ε_p",
        &format!("{:.4}", state.last_max_plastic_strain),
    );
    if ui
        .add_sized(
            [ui.available_width(), 24.0],
            button_widget("Compress (drives plasticity)", accent),
        )
        .clicked()
    {
        // Bump the per-frame compress driver — `update_jelly_example`
        // pulls the puller body downward by `0.02 m * compress_t`
        // each frame and decays `compress_t` geometrically.
        state.compress_t = 1.0;
    }
    ui.separator();

    // ── Stage 20.4 toggles ──────────────────────────────────────
    ui.label("Stage 20.4 (tet-vs-tet, kinematic, SDF)");
    // Stage 20.5 will wire the second-cube + SDF rebuilds through
    // apply_scene_switch; v1 surfaces them as informational toggles
    // (changes require a scene re-entry via Y to apply).
    if ui
        .checkbox(
            &mut state.second_cube,
            "Second jelly cube (rebuild to apply)",
        )
        .changed()
    {
        state.rebuild_pending = true;
    }
    let mut kin_changed = false;
    if ui
        .checkbox(
            &mut state.kinematic_hardening_on,
            "Kinematic hardening (back-stress α)",
        )
        .changed()
    {
        kin_changed = true;
    }
    if state.kinematic_hardening_on {
        ui.label("Kinematic hardening modulus c (Pa)");
        if ui
            .add(
                egui::Slider::new(&mut state.kinematic_hardening_modulus, 1.0e2..=1.0e5)
                    .logarithmic(true),
            )
            .changed()
        {
            kin_changed = true;
        }
        readout(ui, "Max ‖α‖", &format!("{:.2}", state.last_max_back_stress));
    } else {
        readout(ui, "Max ‖α‖", "0.00 (off)");
    }
    if kin_changed {
        // Re-stamp the per-tet kinematic-hardening buffer in place;
        // re-upload to the device. Same shape as the Mixed-preset
        // re-stamp above. When the toggle is off, set every entry
        // to 0 (Stage 20.3 path).
        let c = if state.kinematic_hardening_on {
            state.kinematic_hardening_modulus
        } else {
            0.0
        };
        if let Ok(buf) = sim.model.tet_kinematic_hardening.host_mut() {
            for v in buf.iter_mut() {
                *v = c;
            }
        }
        if let (Some(device), Some(queue)) = (sim.engine.device(), sim.engine.queue()) {
            let device = device.clone();
            let queue = queue.clone();
            let _ = sim.model.tet_kinematic_hardening.ensure_device(&device);
            let _ = sim.model.tet_kinematic_hardening.upload(&queue);
        }
    }
    ui.label("Rigid SDF backing (rebuild to apply)");
    egui::ComboBox::from_id_salt("jelly-sdf-shape")
        .selected_text(match state.sdf_shape {
            JellySdfShape::None => "None",
            JellySdfShape::BakedSphere => "Baked sphere SDF",
            JellySdfShape::BakedTorus => "Baked torus SDF",
        })
        .show_ui(ui, |ui| {
            for (variant, label) in [
                (JellySdfShape::None, "None"),
                (JellySdfShape::BakedSphere, "Baked sphere SDF"),
                (JellySdfShape::BakedTorus, "Baked torus SDF"),
            ] {
                if ui
                    .selectable_value(&mut state.sdf_shape, variant, label)
                    .clicked()
                {
                    state.rebuild_pending = true;
                }
            }
        });
    if state.rebuild_pending {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Rebuild required to apply scene-build toggles.",
        );
        if ui.button("Rebuild jelly").clicked() {
            // Re-enter the current scene unconditionally; the
            // hotkey path (`handle_input`) skips this when the
            // scene is already active.
            pending.0 = Some(sim.scene);
            state.rebuild_pending = false;
        }
    }
    ui.separator();

    if ui
        .add_sized(
            [ui.available_width(), 28.0],
            button_widget("Reset jelly", accent),
        )
        .clicked()
    {
        sim.reset();
    }
    ui.add_space(6.0);
    ui.small("Hotkey: Y");
    ui.small("Stage 20.4: tet-vs-tet, kinematic hardening, baked SDF.");
}

// Tiny helper that returns an egui Button styled the same as the
// global `button` widget — needed because the existing `button`
// helper's API takes the ui directly, but we want a sized button.
fn button_widget(text: &str, _accent: egui::Color32) -> egui::Button<'static> {
    egui::Button::new(text.to_owned())
}


