// --- Stage-20.1 CABLE scene ----------------------------------------
// 32-particle polyline pinned at one end + an orbiting kinematic
// "puller" body at the other. A static cylinder sits between them so
// the cable can drape over it, exercising VBD soft-on-rigid contact +
// the new Dahl friction kernel.
const CABLE_PARTICLES: u32 = 32;
const CABLE_PINNED_AT: bevy::math::Vec3 = bevy::math::Vec3::new(-1.2, 1.8, 0.0);
const CABLE_PARTICLE_MASS: f64 = 0.04;
const CABLE_PARTICLE_RADIUS: f64 = 0.022;
// Cable stretch stiffness (N/m). At 1e6 N/m the cable is effectively
// inextensible — XPBD compliance `α = 1/k` < 1e-6.
const CABLE_STIFFNESS: f64 = 1.0e6;
// Soft-contact stiffness against the static cylinder + the orbiting
// puller. Tuneable.
const CABLE_CONTACT_STIFFNESS: f64 = 5.0e4;
// Static cylinder pose — the cable drapes over its top edge.
const CABLE_CYLINDER_RADIUS: f32 = 0.30;
const CABLE_CYLINDER_HALF_HEIGHT: f32 = 0.50;
const CABLE_CYLINDER_POS: bevy::math::Vec3 = bevy::math::Vec3::new(0.0, 1.2, 0.0);
// Puller orbit radius and rate (rad/s).
const CABLE_PULLER_ORBIT_R: f32 = 1.1;
const CABLE_PULLER_ORBIT_RATE: f32 = 0.4;
const CABLE_PULLER_HEIGHT: f32 = 1.6;
const CABLE_PULLER_RADIUS: f32 = 0.08;

// Stage-20.1 CABLE polyline demo runtime state. Driven by the
// right-rail sliders; applied to the active solver each frame in
// `update_cable_example`. Lazy-initialised on entry to `Scene::Cable`;
// survives scene switches so the user's tuning persists.
#[derive(Resource)]
struct CableExampleState {
    /// Cable stretch stiffness (N/m). Re-applied to the per-spring
    /// XPBD compliance buffer when the user drags the slider.
    stretch_stiffness: f64,
    /// Soft-contact stiffness against the cylinder + puller (N/m).
    contact_stiffness: f64,
    /// VBD outer-iteration count per substep.
    inner_iterations: u32,
    /// AVBD contact-mode toggle: hard mode = augmented-Lagrangian
    /// dual updates, soft mode = quadratic penalty (Stage 20).
    use_hard_mode: bool,
    /// Show the pinned-particle marker as a yellow gizmo.
    show_pinned: bool,
    /// Internal sim-time accumulator so the puller orbits at a steady
    /// rate (we can't read SimRunState here without bringing it into
    /// the system param tuple).
    sim_time: f32,
}

impl Default for CableExampleState {
    fn default() -> Self {
        Self {
            stretch_stiffness: CABLE_STIFFNESS,
            contact_stiffness: CABLE_CONTACT_STIFFNESS,
            inner_iterations: 12,
            use_hard_mode: true,
            show_pinned: true,
            sim_time: 0.0,
        }
    }
}

// Stage-20.1 CABLE polyline demo. A 32-particle cable pinned at one
// end, the other end driven kinematically by `update_cable_example`.
// A static cylinder sits between the pin and the puller so the cable
// drapes over it (soft-on-rigid AVBD + Dahl friction territory).
//
// The puller body is a kinematic sphere so it shows up in the
// renderer's body list (we want the user to see what's pulling the
// cable). The cable itself isn't a body — it's particles + springs,
// rendered as a gizmo polyline by `draw_cable_gizmos`.
fn build_cable_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_linear_damping(0.6);

    let pin_pos = MVec3::new(
        CABLE_PINNED_AT.x as f64,
        CABLE_PINNED_AT.y as f64,
        CABLE_PINNED_AT.z as f64,
    );
    // Initial puller position — start at the +X side of the orbit.
    let puller_pos = MVec3::new(CABLE_PULLER_ORBIT_R as f64, CABLE_PULLER_HEIGHT as f64, 0.0);
    let (first, n, _cable_id) = b.add_cable_polyline(
        pin_pos,
        puller_pos,
        CABLE_PARTICLES,
        CABLE_PARTICLE_MASS,
        CABLE_PARTICLE_RADIUS,
        CABLE_STIFFNESS,
        // Hard cap on per-segment stretch — 0 means inextensible
        // after the soft pass. We rely on VBD's stiffness for the
        // soft pass, so the cap mainly catches numerical drift.
        0.02,
        mp::WorldId(0),
    );
    // Pin both endpoints. The puller end is "kinematic-pinned": its
    // mass stays at zero (so the integrator skips it) and the system
    // `update_cable_example` rewrites its position each frame.
    b.pin_particle(first);
    b.pin_particle(first + n - 1);

    // Static cylinder between pin and puller (the drape obstacle).
    b.add_shape_cylinder(
        mp::BodyId::NONE,
        CABLE_CYLINDER_HALF_HEIGHT as f64,
        CABLE_CYLINDER_RADIUS as f64,
        mp::Transform::from_translation(MVec3::new(
            CABLE_CYLINDER_POS.x as f64,
            CABLE_CYLINDER_POS.y as f64,
            CABLE_CYLINDER_POS.z as f64,
        )),
    )
    .expect("cable cylinder shape");

    // Kinematic puller body — a small sphere we can render. Its
    // pose is overwritten by `update_cable_example` each frame.
    let puller = b.add_body(
        mp::BodyParams::new()
            .with_spatial_inertia(mp::SpatialInertia::sphere(0.1, CABLE_PULLER_RADIUS as f64))
            .with_position(puller_pos)
            .with_flags(mp::BodyFlags::KINEMATIC),
    );
    b.add_shape_sphere(puller, CABLE_PULLER_RADIUS as f64, mp::Transform::IDENTITY)
        .expect("puller shape");

    // Floor catch-all so the cable doesn't fall forever if the user
    // pulls too hard and a particle escapes.
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("floor plane");

    let model = b.build().expect("cable model");
    let visuals = vec![BodyVisual::Sphere(CABLE_PULLER_RADIUS)];
    (model, visuals)
}

// Tetrahedrise an axis-aligned cube cell into 6 tets sharing the
// body diagonal `(v000, v111)`. Standard "diagonal split" — every
// tet has positive rest volume under the standard `+x +y +z` cube
// vertex ordering. Mirrors the helper in `tests/vbd_tets_gpu.rs`.
fn jelly_cube_to_six_tets(corners: [u32; 8]) -> [[u32; 4]; 6] {
    let v000 = corners[0];
    let v100 = corners[1];
    let v010 = corners[2];
    let v110 = corners[3];
    let v001 = corners[4];
    let v101 = corners[5];
    let v011 = corners[6];
    let v111 = corners[7];
    [
        [v000, v100, v110, v111],
        [v000, v110, v010, v111],
        [v000, v010, v011, v111],
        [v000, v011, v001, v111],
        [v000, v001, v101, v111],
        [v000, v101, v100, v111],
    ]
}

// Stage-20.2/20.3 JELLY scene. An 8×8×8 tet jelly cube falls onto a
// static plane. A kinematic puller body sits at +X for the cube to
// (optionally) bounce against. The cube renders as gizmo edges on
// the surface tris (computed at draw time from the tet topology).
//
// Stage 20.3 — uses `add_tet_mesh_per_material` to assign different
// `MaterialNeoHookean` recipes per cube cell based on the
// `JellyMaterialPreset`. The Soft preset gives every cell the same
// modest E; Stiff gives every cell a high E; Mixed alternates soft

fn update_cable_example(
    mut sim: ResMut<Sim>,
    mut state: ResMut<CableExampleState>,
    time: Res<Time>,
    run: Res<SimRunState>,
) {
    if sim.scene != Scene::Cable {
        return;
    }
    if !run.paused {
        state.sim_time += time.delta_secs();
    }
    let t = state.sim_time;
    let theta = t * CABLE_PULLER_ORBIT_RATE;
    let puller_pos = MVec3::new(
        (CABLE_PULLER_ORBIT_R * theta.cos()) as f64,
        CABLE_PULLER_HEIGHT as f64,
        (CABLE_PULLER_ORBIT_R * theta.sin()) as f64,
    );

    // Update the kinematic puller body's pose. body 0 is the puller
    // (added via `BodyParams::new().with_flags(KINEMATIC)` in
    // `build_cable_model` after the cable particles).
    if sim.model.body_count > 0
        && let Ok(bq) = sim.state_a.body_q.host_mut()
            && let Some(body) = bq.last_mut() {
                *body = mp::Transform::from_translation(puller_pos);
            }

    // Update the puller-pinned cable particle. It's the LAST particle
    // pushed by `add_cable_polyline`, so index = particle_count - 1.
    let idx = sim.model.particle_count;
    if idx > 0
        && let Ok(pq) = sim.state_a.particle_q.host_mut() {
            pq[idx - 1] = puller_pos;
        }

    // Apply the right-rail solver tunings.
    if let SolverBackend::Vbd(s) = &mut sim.solver {
        s.contact_stiffness = state.contact_stiffness;
        s.iterations = state.inner_iterations as usize;
        s.vbd_mode = if state.use_hard_mode {
            mp::VbdMode::Hard
        } else {
            mp::VbdMode::Soft
        };
    }
}

// Stage-20.2 JELLY scene: keep the active VBD solver's contact
// stiffness, iteration count, and per-tet material in sync with the
// right-rail sliders. The Young's modulus / Poisson sliders re-derive
// the per-element Lamé parameters and overwrite the model's `tet_mu`
// / `tet_lambda` host buffers (which then get re-uploaded on the

fn draw_cable_gizmos(sim: Res<Sim>, state: Res<CableExampleState>, mut gizmos: Gizmos) {
    let multi_cable = matches!(
        sim.scene,
        Scene::CablePile | Scene::CableYJunction | Scene::CableBundleHysteresis,
    );
    if sim.scene != Scene::Cable && !multi_cable {
        return;
    }
    let Ok(pq) = sim.state_a.particle_q.host() else {
        return;
    };
    let n = sim.model.particle_count;
    if n < 2 {
        return;
    }
    let cable_color = if state.use_hard_mode {
        Color::srgb(0.95, 0.55, 0.20) // hard mode = warm orange
    } else {
        Color::srgb(0.40, 0.70, 0.95) // soft mode = cool blue
    };
    // For multi-cable scenes, walk the per-cable particle lists so we
    // don't draw a stray segment between cable N's last particle and
    // cable (N+1)'s first.
    if multi_cable && sim.model.cable_count > 0 {
        let offsets = match sim.model.cable_particle_offsets.host() {
            Ok(o) => o,
            Err(_) => return,
        };
        let particles = match sim.model.cable_particles.host() {
            Ok(p) => p,
            Err(_) => return,
        };
        for c in 0..sim.model.cable_count {
            let start = offsets[c] as usize;
            let end = offsets[c + 1] as usize;
            for k in start..end.saturating_sub(1) {
                let i_a = particles[k] as usize;
                let i_b = particles[k + 1] as usize;
                let a = pq[i_a];
                let b = pq[i_b];
                gizmos.line(
                    Vec3::new(a.x as f32, a.y as f32, a.z as f32),
                    Vec3::new(b.x as f32, b.y as f32, b.z as f32),
                    cable_color,
                );
            }
        }
        return;
    }
    for i in 0..n - 1 {
        let a = pq[i];
        let b = pq[i + 1];
        gizmos.line(
            Vec3::new(a.x as f32, a.y as f32, a.z as f32),
            Vec3::new(b.x as f32, b.y as f32, b.z as f32),
            cable_color,
        );
    }
    if state.show_pinned {
        // First and last particles are the pin anchors.
        let pin_color = Color::srgb(1.0, 0.95, 0.20);
        for &i in [0_usize, n - 1].iter() {
            let p = pq[i];
            let pos = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
            gizmos.sphere(pos, 0.04, pin_color);
        }
    }
}

// Stage-20.2 JELLY scene gizmos. Two parts:
//   1. Tet wireframe — every unique edge of every tet drawn as a
//      green line segment (deduped via a hash set so shared edges
//      don't overdraw).
//   2. Optional CABLE comparison overlay — a small bending cable

fn cable_pane(
    ui: &mut egui::Ui,
    state: &mut CableExampleState,
    sim: &mut Sim,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    readout(ui, "Particles", &format!("{}", sim.model.particle_count));
    readout(ui, "Springs", &format!("{}", sim.model.spring_count));
    readout(ui, "Cables", &format!("{}", sim.model.cable_count));
    if sim.model.cable_count > 0
        && let Ok(rl) = sim.model.cable_rest_length.host() {
            readout(ui, "Rest length", &format!("{:.3} m", rl[0]));
        }
    ui.separator();

    ui.label("Stretch stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.stretch_stiffness, 1.0e3..=1.0e7).logarithmic(true))
        .changed()
    {
        // Mirror the new stiffness into the per-spring compliance
        // buffer so VBD's `refresh_stiffness` (called below) picks it
        // up. The model's compliance buffer is host-mut-friendly
        // because we own it; any GPU re-upload happens inside
        // `refresh_stiffness` on the solver.
        let alpha = if state.stretch_stiffness > 1.0 {
            1.0 / state.stretch_stiffness
        } else {
            1.0
        };
        if let Ok(comp) = sim.model.spring_compliance.host_mut() {
            for c in comp.iter_mut() {
                *c = alpha;
            }
        }
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            let _ = s.refresh_stiffness(&sim.model);
        }
    }

    ui.label("Soft contact stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.contact_stiffness, 1.0e3..=1.0e6).logarithmic(true))
        .changed()
        && let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = state.contact_stiffness;
        }
    ui.label("Inner iterations");
    if ui
        .add(egui::Slider::new(&mut state.inner_iterations, 1..=30))
        .changed()
        && let SolverBackend::Vbd(s) = &mut sim.solver {
            s.iterations = state.inner_iterations as usize;
        }
    ui.separator();

    if ui
        .checkbox(&mut state.use_hard_mode, "Hard-mode AVBD (no penetration)")
        .changed()
        && let SolverBackend::Vbd(s) = &mut sim.solver {
            s.vbd_mode = if state.use_hard_mode {
                mp::VbdMode::Hard
            } else {
                mp::VbdMode::Soft
            };
        }
    ui.checkbox(&mut state.show_pinned, "Show pinned points");
    ui.separator();

    if ui
        .add_sized(
            [ui.available_width(), 28.0],
            button_widget("Reset cable", accent),
        )
        .clicked()
    {
        sim.reset();
        state.sim_time = 0.0;
    }
    ui.add_space(6.0);
    ui.small("Hotkey: C");
}

// Right-rail pane for the Stage-20.2 JELLY tetrahedral soft-body
// demo. Carries Young's-modulus + Poisson-ratio sliders (per-tet
// material), inner-iter / contact-stiffness sliders, and a wireframe

