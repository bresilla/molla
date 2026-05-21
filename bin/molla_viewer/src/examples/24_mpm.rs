// Stage-24d.1 MPM (sand / snow) demo state. The actual GPU sim lives
// on `Sim::mpm` (lazy-initialised on first entry to `Scene::Mpm`);
// this resource holds only the user-facing tuning sliders + reset
// triggers for the right-rail UI.
//
// Stage-24d.1 additions: `rheology_idx` (0 = sand, 1 = snow,
// 2 = viscous),
// `time_step_implicit` + `jacobi_iters` (semi-implicit toggle + iter
// slider), `add_rigid_sphere` button (drops a kinematic sphere to
// demonstrate two-way coupling), `pile_slope_deg` live readout
// (computed each frame from the particle positions).
#[derive(Resource)]
struct MpmExampleState {
    /// Drucker-Prager friction angle (degrees). Sand: 25°-40°.
    friction_angle_deg: f32,
    /// Young's modulus (Pa). Higher = stiffer / more like rock.
    youngs: f32,
    /// Per-axis particle count for the initial pour (count = N³).
    pour_axis_count: u32,
    /// Live readout: live particle count (set by the update system).
    particle_count: u32,
    /// Trigger flag: when true, the update system rebuilds the
    /// particle pour and clears it.
    needs_reset: bool,
    /// Spout-y offset for the initial pour.
    spout_y: f32,
    /// Stage 24d.2: substeps batched into a single `step_n` call per
    /// frame.  Higher = more physical fidelity per visible frame +
    /// directly demonstrates the batched-dispatcher throughput win.
    substeps_per_frame: u32,
    /// Live readout (ms): wall-clock for the most recent `step_n`
    /// dispatch + frame-end readback.
    last_frame_ms: f32,

    // Stage 24d.1.
    /// 0 = sand (Drucker-Prager), 1 = snow (Stomakhin), 2 = viscous.
    rheology_idx: u32,
    /// Snow critical compression (only used when `rheology_idx = 1`).
    snow_theta_c: f32,
    /// Snow critical stretch.
    snow_theta_s: f32,
    /// Snow hardening exponent.
    snow_hardening: f32,
    /// Viscous rheology linear velocity damping (1/s).
    viscous_velocity_damping: f32,
    /// Viscous rheology APIC affine damping (1/s).
    viscous_affine_damping: f32,
    /// Use the semi-implicit time-step.
    time_step_implicit: bool,
    /// Number of Jacobi sweeps per substep when `time_step_implicit`.
    jacobi_iters: u32,
    /// "Add rigid sphere" button trigger — when true, the update
    /// system drops a kinematic sphere into the scene.
    add_rigid_sphere: bool,
    /// Live readout: equilibrium pile slope, measured from the
    /// particle positions each frame.
    pile_slope_deg: f32,
    /// Live readout: max-magnitude two-way wrench |F| (N) on the
    /// rigid sphere, when one exists.
    rigid_force_mag: f32,
}

impl Default for MpmExampleState {
    fn default() -> Self {
        Self {
            friction_angle_deg: 30.0,
            youngs: 5.0e3,
            pour_axis_count: 12,
            particle_count: 0,
            needs_reset: true, // initial pour
            spout_y: 0.6,
            // 4 substeps was the pre-24d.2 default (loop of `step()`);
            // higher values exercise the batched dispatcher.
            substeps_per_frame: 4,
            last_frame_ms: 0.0,
            rheology_idx: 0,
            snow_theta_c: 2.5e-2,
            snow_theta_s: 7.5e-3,
            snow_hardening: 10.0,
            viscous_velocity_damping: 4.0,
            viscous_affine_damping: 8.0,
            time_step_implicit: false,
            jacobi_iters: 3,
            add_rigid_sphere: false,
            pile_slope_deg: 0.0,
            rigid_force_mag: 0.0,
        }
    }
}

fn build_mpm_model() -> (mp::Model, Vec<BodyVisual>) {
    let b = mp::ModelBuilder::new();
    // No bodies — the floor of the MPM grid serves as the implicit
    // boundary.
    let model = b.build().expect("mpm trivial model build");
    (model, Vec::new())
}

fn mpm_default_particle_layout(state: &MpmExampleState) -> ([f32; 3], [f32; 3], Option<(f32, f32, f32, f32)>) {
    (
        [-0.18, state.spout_y - 0.18, -0.18],
        [0.18, state.spout_y + 0.18, 0.18],
        None,
    )
}

// Drives the MPM simulation each frame. Lazy-initialises the
// `MpmSimulation` on first entry to `Scene::Mpm`; on `needs_reset`
// rebuilds the particle pour. Steps the sim in lock-step with
// the viewer's fixed-dt cadence.
//
// Stage 24d.1 — also pushes rheology + time-step + rigid-sphere
// state from the right-rail UI, and reads back the per-rigid wrench
// for the live `rigid_force_mag` readout.
fn update_mpm_example(
    mut sim: ResMut<Sim>,
    mut state: ResMut<MpmExampleState>,
    run: Res<SimRunState>,
) {
    if !matches!(
        sim.scene,
        Scene::Mpm
            | Scene::MpmSnowBall
            | Scene::MpmTwoWay
            | Scene::MpmMultiMaterial
            | Scene::MpmBeamTwist
            | Scene::WarpDem
            | Scene::WarpFluid
            | Scene::WarpSph
    ) {
        return;
    }

    // Lazy initialise the MPM sim on first entry.
    if sim.mpm.is_none() {
        let cfg = MpmConfig {
            grid_size: 32,
            cell_size: 0.05,
            grid_origin: [-0.8, 0.0, -0.8],
            gravity_y: -9.81,
            youngs: state.youngs,
            poisson: 0.3,
            friction_angle: state.friction_angle_deg.to_radians(),
            cohesion: 0.0,
            boundary_friction: 0.5,
            particle_mass: 1.0e-3,
            particle_volume: 1.0e-6,
            max_particles: 32_768,
            max_rigid: 4,
            rheology: MpmRheology::Sand {
                friction_angle: state.friction_angle_deg.to_radians(),
            },
            time_step: MpmTimeStep::Explicit,
        };
        let s = MpmSimulation::new(&sim.engine, cfg).expect("mpm new");
        sim.mpm = Some(s);
        state.needs_reset = true;
    }

    // Pull material parameters + rheology + time-step from sliders.
    if let Some(m) = sim.mpm.as_mut() {
        m.set_material(state.youngs, state.friction_angle_deg.to_radians());
        let rheology = match state.rheology_idx {
            1 => MpmRheology::Snow {
                theta_c: state.snow_theta_c,
                theta_s: state.snow_theta_s,
                hardening: state.snow_hardening,
            },
            2 => MpmRheology::Viscous {
                velocity_damping: state.viscous_velocity_damping,
                affine_damping: state.viscous_affine_damping,
            },
            _ => MpmRheology::Sand {
                friction_angle: state.friction_angle_deg.to_radians(),
            },
        };
        m.set_rheology(rheology);
        let ts = if state.time_step_implicit {
            MpmTimeStep::Implicit {
                jacobi_iters: state.jacobi_iters,
            }
        } else {
            MpmTimeStep::Explicit
        };
        m.set_time_step(ts);
    }

    // Reset / re-pour.
    if state.needs_reset {
        let device = sim.engine.device().unwrap().clone();
        let queue = sim.engine.queue().unwrap().clone();
        let n_axis = state.pour_axis_count.max(2) as usize;
        let mut particles = Vec::with_capacity(n_axis.pow(3));
        // Per-scene particle layouts:
        //   * `Mpm`         — column pour (default).
        //   * `MpmSnowBall` — spherical ball at `spout_y` (we cull
        //                     particles outside the sphere).
        //   * `MpmTwoWay`   — wider, shallower pile so the rigid
        //                     sphere has room to plunge in.
        //   * `MpmMultiMaterial` — three adjacent material bands. The current
        //                         solver is single-rheology; the bands are
        //                         visual/audit regions tied to the reduced
        //                         multi-material branch oracle.
        //   * `MpmBeamTwist` — elongated beam block with a procedural twist
        //                      stamped into the initial particle positions.
        let (lo, hi, cull_sphere) = match sim.scene {
            Scene::MpmSnowBall => mpm_snow_ball_particle_layout(&state),
            Scene::MpmTwoWay => mpm_two_way_particle_layout(),
            Scene::MpmMultiMaterial => mpm_multi_material_particle_layout(&state),
            Scene::MpmBeamTwist => mpm_beam_twist_particle_layout(&state),
            _ => mpm_default_particle_layout(&state),
        };
        let dx = (hi[0] - lo[0]) / n_axis as f32;
        let dy = (hi[1] - lo[1]) / n_axis as f32;
        let dz = (hi[2] - lo[2]) / n_axis as f32;
        for ix in 0..n_axis {
            for iy in 0..n_axis {
                for iz in 0..n_axis {
                    let mut p = [
                        lo[0] + (ix as f32 + 0.5) * dx,
                        lo[1] + (iy as f32 + 0.5) * dy,
                        lo[2] + (iz as f32 + 0.5) * dz,
                    ];
                    if sim.scene == Scene::MpmBeamTwist {
                        p = mpm_beam_twist_position(p);
                    }
                    if let Some((cx, cy, cz, r)) = cull_sphere {
                        let dx0 = p[0] - cx;
                        let dy0 = p[1] - cy;
                        let dz0 = p[2] - cz;
                        if dx0 * dx0 + dy0 * dy0 + dz0 * dz0 > r * r {
                            continue;
                        }
                    }
                    particles.push(p);
                }
            }
        }
        if let Some(m) = sim.mpm.as_mut() {
            m.set_particles(&queue, &particles).expect("set particles");
        }
        sim.mpm_rigid_spheres.clear();
        state.particle_count = particles.len() as u32;
        sim.mpm_particles = particles;
        // For MpmTwoWay, auto-spawn the rigid sphere on first frame so
        // the user immediately sees the two-way coupling demo.
        if sim.scene == Scene::MpmTwoWay {
            sim.mpm_rigid_spheres.push(mpm_two_way_initial_rigid_sphere());
        }
        state.needs_reset = false;
        // First read so the viewer doesn't briefly show empty.
        if let Some(m) = sim.mpm.as_mut() {
            sim.mpm_particles = m.read_positions_with(&device, &queue).unwrap_or_default();
        }
        return;
    }

    // "Add rigid sphere" button — drop a kinematic sphere from above.
    if state.add_rigid_sphere {
        state.add_rigid_sphere = false;
        sim.mpm_rigid_spheres
            .push(([0.0, 0.7, 0.0], 0.10, [0.0, -2.0, 0.0]));
    }

    if run.paused {
        return;
    }

    // Stage 24d.2: batch the substeps into a single `step_n` call
    // (one wgpu submit + one frame-end readback). Stage 24d.1 added
    // rigid-sphere integration + two-way wrench readback for the
    // demo path. Combined here.
    let device = sim.engine.device().unwrap().clone();
    let queue = sim.engine.queue().unwrap().clone();

    // Integrate kinematic-sphere positions forward (purely host-side
    // since we don't have a full rigid-body backend wired here; this
    // is the v1 demo path so the viewer shows the rigid sphere
    // descending into the sand). The two-way wrench read-back lets
    // the user observe the reaction force the sand applies.
    if !sim.mpm_rigid_spheres.is_empty() {
        let dt_total = 1.0 / 60.0; // viewer-cadence
        for sph in sim.mpm_rigid_spheres.iter_mut() {
            sph.0[0] += sph.2[0] * dt_total;
            sph.0[1] += sph.2[1] * dt_total;
            sph.0[2] += sph.2[2] * dt_total;
        }
        // Push to the GPU.
        let spheres_snapshot: Vec<_> = sim.mpm_rigid_spheres.clone();
        if let Some(m) = sim.mpm.as_mut() {
            let _ = m.set_rigid_spheres(&queue, &spheres_snapshot);
        }
    }

    let substeps = state.substeps_per_frame.max(1) as usize;
    let t_frame = std::time::Instant::now();
    let has_rigid = !sim.mpm_rigid_spheres.is_empty();
    let (new_pos, new_rigid_force_mag) = if let Some(m) = sim.mpm.as_mut() {
        m.step_n_with(&device, &queue, 1.0 / 480.0, substeps)
            .expect("mpm step_n");
        // Single frame-end readback; the viewer only needs particle
        // positions for the gizmo draw, not the intermediate substeps.
        let positions = m.read_positions_with(&device, &queue).unwrap_or_default();
        let force_mag = if has_rigid {
            m.read_rigid_wrenches_with(&device, &queue)
                .map(|wrenches| {
                    wrenches
                        .iter()
                        .map(|(f, _)| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt())
                        .fold(0.0_f32, f32::max)
                })
                .unwrap_or(0.0)
        } else {
            0.0
        };
        (positions, force_mag)
    } else {
        (sim.mpm_particles.clone(), 0.0)
    };
    sim.mpm_particles = new_pos;
    state.last_frame_ms = t_frame.elapsed().as_secs_f32() * 1000.0;
    state.rigid_force_mag = new_rigid_force_mag;

    // Update the live pile-slope readout.
    state.pile_slope_deg = measure_pile_slope_for_viewer(&sim.mpm_particles);
}

// Stage 24d.1 — quick-and-dirty pile-slope estimator for the viewer.
// Same algorithm as the test-side `measure_pile_slope_deg` but
// inlined here to avoid pulling tests/ into the lib build path.
fn measure_pile_slope_for_viewer(positions: &[[f32; 3]]) -> f32 {
    if positions.len() < 16 {
        return 0.0;
    }
    let mut ys: Vec<f32> = positions.iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let floor_y = ys[ys.len() / 20];
    let radial: Vec<(f32, f32)> = positions
        .iter()
        .map(|p| {
            let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
            ((p[1] - floor_y).max(0.0), r)
        })
        .collect();
    let r_max = radial.iter().map(|(_, r)| *r).fold(0.0_f32, f32::max);
    if r_max < 1.0e-3 {
        return 0.0;
    }
    let n_bins = 8;
    let mut bin_max_h = vec![0.0_f32; n_bins];
    let mut bin_count = vec![0u32; n_bins];
    for (h, r) in &radial {
        let bin = ((*r / r_max) * n_bins as f32).floor() as usize;
        let bin = bin.min(n_bins - 1);
        if *h > bin_max_h[bin] {
            bin_max_h[bin] = *h;
        }
        bin_count[bin] += 1;
    }
    let mut inner: Option<(f32, f32)> = None;
    let mut outer: Option<(f32, f32)> = None;
    for b in 0..n_bins {
        if bin_count[b] < 2 {
            continue;
        }
        let r_centre = (b as f32 + 0.5) / n_bins as f32 * r_max;
        let entry = (r_centre, bin_max_h[b]);
        if inner.is_none() {
            inner = Some(entry);
        }
        outer = Some(entry);
    }
    match (inner, outer) {
        (Some((r_in, h_in)), Some((r_out, h_out))) if r_out > r_in + 1.0e-4 => {
            let dh = (h_in - h_out).max(0.0);
            let dr = r_out - r_in;
            dh.atan2(dr).to_degrees()
        }
        _ => 0.0,
    }
}

// Right-rail pane for the MPM scene. Stage 24d.1 — adds rheology
// dropdown (Sand/Snow/Viscous), implicit time-step toggle + jacobi_iters
// slider, "Add rigid sphere" button, live pile-slope + reaction-force
// readouts.
fn mpm_pane(ui: &mut egui::Ui, state: &mut MpmExampleState, sim: &mut Sim, accent: egui::Color32) {
    ui.add_space(2.0);
    readout(ui, "Particles", &format!("{}", state.particle_count));
    if let Some(m) = sim.mpm.as_ref() {
        readout(ui, "Live count", &format!("{}", m.particle_count()));
        readout(
            ui,
            "Grid",
            &format!("{}³ × {:.2} m", m.config().grid_size, m.config().cell_size),
        );
    }
    readout(ui, "Pile slope", &format!("{:.1}°", state.pile_slope_deg));
    if !sim.mpm_rigid_spheres.is_empty() {
        readout(
            ui,
            "Reaction |F|",
            &format!("{:.2} N", state.rigid_force_mag),
        );
    }
    ui.separator();

    // Rheology dropdown (Sand / Snow / Viscous).
    ui.label("Rheology");
    egui::ComboBox::from_id_salt("mpm_rheology_combo")
        .selected_text(match state.rheology_idx {
            1 => "Snow",
            2 => "Viscous",
            _ => "Sand",
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut state.rheology_idx, 0, "Sand (Drucker-Prager)");
            ui.selectable_value(&mut state.rheology_idx, 1, "Snow (Stomakhin)");
            ui.selectable_value(&mut state.rheology_idx, 2, "Viscous damping");
        });

    match state.rheology_idx {
        1 => {
            ui.label("Snow θ_c (critical compression)");
            ui.add(egui::Slider::new(&mut state.snow_theta_c, 1.0e-3..=1.0e-1).logarithmic(true));
            ui.label("Snow θ_s (critical stretch)");
            ui.add(egui::Slider::new(&mut state.snow_theta_s, 1.0e-3..=5.0e-2).logarithmic(true));
            ui.label("Snow hardening exponent");
            ui.add(egui::Slider::new(&mut state.snow_hardening, 0.0..=20.0));
        }
        2 => {
            ui.label("Velocity damping (1/s)");
            ui.add(egui::Slider::new(
                &mut state.viscous_velocity_damping,
                0.0..=20.0,
            ));
            ui.label("Affine damping (1/s)");
            ui.add(egui::Slider::new(
                &mut state.viscous_affine_damping,
                0.0..=40.0,
            ));
        }
        _ => {
            ui.label("Friction angle (°)");
            ui.add(egui::Slider::new(
                &mut state.friction_angle_deg,
                10.0..=50.0,
            ));
        }
    }

    ui.label("Young's modulus (Pa)");
    ui.add(egui::Slider::new(&mut state.youngs, 1.0e3..=1.0e6).logarithmic(true));

    ui.label("Pour axis count");
    ui.add(egui::Slider::new(&mut state.pour_axis_count, 4..=20));

    ui.separator();
    // Time-step picker.
    ui.label("Time step");
    ui.horizontal(|ui| {
        ui.selectable_value(&mut state.time_step_implicit, false, "Explicit");
        ui.selectable_value(&mut state.time_step_implicit, true, "Implicit");
    });
    if state.time_step_implicit {
        ui.label("Jacobi iterations");
        ui.add(egui::Slider::new(&mut state.jacobi_iters, 1..=20));
    }

    // Stage 24d.2: batched-dispatcher slider.  Drag this up and watch
    // `Frame` stay relatively flat — that's the win from collapsing N
    // submits into one.
    ui.label("Substeps / frame (Stage 24d.2)");
    ui.add(egui::Slider::new(&mut state.substeps_per_frame, 1..=120));
    readout(ui, "Frame (ms)", &format!("{:.2}", state.last_frame_ms));

    ui.separator();
    ui.horizontal(|ui| {
        if button(ui, "Reset", accent).clicked() {
            state.needs_reset = true;
        }
        if button(ui, "Add rigid sphere", accent).clicked() {
            state.add_rigid_sphere = true;
        }
    });
    ui.add_space(2.0);
    ui.small("Sand: Drucker-Prager friction-cone projection.");
    ui.small("Snow: Stomakhin elastic-envelope clamping.");
    ui.small("Implicit: Jacobi-relaxation grid step (semi-implicit fallback).");
    ui.small("Hotkey: P (Particle/MPM)");
}

// Draw the 12 edges of an axis-aligned box `[lo, hi]` as gizmo lines.
// (`Gizmos::cuboid` exists in some Bevy versions but the crate version
// we vendor doesn't expose it; line-loops are portable.)
fn draw_aabb_edges(gizmos: &mut Gizmos, lo: Vec3, hi: Vec3, color: Color) {
    let corners = [
        Vec3::new(lo.x, lo.y, lo.z),
        Vec3::new(hi.x, lo.y, lo.z),
        Vec3::new(hi.x, hi.y, lo.z),
        Vec3::new(lo.x, hi.y, lo.z),
        Vec3::new(lo.x, lo.y, hi.z),
        Vec3::new(hi.x, lo.y, hi.z),
        Vec3::new(hi.x, hi.y, hi.z),
        Vec3::new(lo.x, hi.y, hi.z),
    ];
    let edges: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    for (a, b) in edges {
        gizmos.line(corners[a], corners[b], color);
    }
}

// Renders MPM + SPH particles as point gizmos. Cheap; doesn't allocate.
fn draw_mpm_gizmos(sim: Res<Sim>, mut gizmos: Gizmos) {
    if !matches!(
        sim.scene,
        Scene::Mpm
            | Scene::MpmSnowBall
            | Scene::MpmTwoWay
            | Scene::MpmMultiMaterial
            | Scene::MpmBeamTwist
            | Scene::SphDamBreak
    ) {
        return;
    }
    // SPH dam-break: draw the open-top tank AABB so the fluid surge has
    // a visible container.
    if sim.scene == Scene::SphDamBreak {
        let lo = Vec3::new(SPH_TANK_MIN[0], SPH_TANK_MIN[1], SPH_TANK_MIN[2]);
        let hi = Vec3::new(SPH_TANK_MAX[0], SPH_TANK_MAX[1], SPH_TANK_MAX[2]);
        draw_aabb_edges(&mut gizmos, lo, hi, Color::srgb(0.40, 0.40, 0.45));
    }
    if sim.mpm_particles.is_empty() {
        return;
    }
    // Draw the grid AABB as 12 lines so the user sees the simulation
    // domain. (`Gizmos::cuboid` exists in some Bevy versions but the
    // crate version we vendor doesn't expose it; line-loops are
    // portable.)
    if let Some(m) = sim.mpm.as_ref() {
        let cfg = m.config();
        let lo = Vec3::new(cfg.grid_origin[0], cfg.grid_origin[1], cfg.grid_origin[2]);
        let extent = cfg.grid_size as f32 * cfg.cell_size;
        let hi = lo + Vec3::splat(extent);
        draw_aabb_edges(&mut gizmos, lo, hi, Color::srgb(0.40, 0.40, 0.45));
    }
    let sand_color = Color::srgb(0.85, 0.74, 0.52);
    let water_color = Color::srgb(0.25, 0.55, 0.95);
    for p in &sim.mpm_particles {
        let pos = Vec3::new(p[0], p[1], p[2]);
        let color = match sim.scene {
            Scene::MpmMultiMaterial => mpm_multi_material_color(mpm_multi_material_region(*p)),
            Scene::MpmBeamTwist => mpm_beam_twist_color(*p),
            Scene::SphDamBreak => water_color,
            _ => sand_color,
        };
        gizmos.sphere(pos, 0.012, color);
    }
    // Stage 24d.1 — render kinematic rigid spheres for the two-way
    // coupling demo as wireframe spheres.
    let sphere_color = Color::srgb(0.20, 0.55, 0.95);
    for (centre, radius, _vel) in &sim.mpm_rigid_spheres {
        let p = Vec3::new(centre[0], centre[1], centre[2]);
        gizmos.sphere(p, *radius, sphere_color);
    }
}
