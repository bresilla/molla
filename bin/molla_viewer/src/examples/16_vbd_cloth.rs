// --- Stage-20 VBD cloth scene ---------------------------------------
// Higher resolution than the XPBD demo + tighter compliance — this
// is exactly the regime where XPBD wobbles and VBD shines.
const VBD_CLOTH_GRID: u32 = 32;
const VBD_CLOTH_SIZE: f32 = 1.8;
const VBD_CLOTH_TOP_Y: f32 = 4.5;
const VBD_CLOTH_PARTICLE_MASS: f64 = 0.04;
const VBD_CLOTH_PARTICLE_RADIUS: f64 = 0.025;
// XPBD compliance for the underlying cloth model (= 1 / k for
// stretch springs). VBD reads this and inverts to a per-spring
// stiffness (see `derive_stiffness_table`); the smaller the value
// the stiffer the cloth.
const VBD_CLOTH_COMPLIANCE: f64 = 5.0e-6;
const VBD_CLOTH_MAX_STRETCH: f64 = 0.10;
const VBD_CLOTH_SPHERE_RADIUS: f32 = 0.6;
const VBD_CLOTH_SPHERE_Y: f32 = 1.6;
// Soft-contact stiffness against the kinematic sphere. Stiff enough
// to stop a 4 cm/s falling cloth particle in ~3 substeps at 480 Hz
// without producing visible jitter. Tuneable from the right-rail.
const VBD_CLOTH_CONTACT_STIFFNESS: f64 = 5.0e4;

// Stage-20 VBD cloth runtime state. Driven by the right-rail sliders;
// applied to the active solver each frame in `apply_vbd_settings`.
// Lazy-initialised on entry to `Scene::VbdCloth`; survives scene
// switches so the user's tuning persists when they jump out and back.
#[derive(Resource)]
struct VbdClothExampleState {
    /// Per-spring stretch stiffness (N/m). Mapped onto VBD via
    /// `1 / k = compliance` then re-uploaded.
    stretch_stiffness: f32,
    /// Per-spring bend stiffness (N/m). The cloth builder lays down
    /// "skip-one" bending springs at compliance = 30 × stretch_compliance
    /// by default; we use the same ratio when the user drags this
    /// slider.
    bend_stiffness: f32,
    /// Soft-contact stiffness against the rigid sphere (N/m).
    contact_stiffness: f32,
    /// Per-substep VBD outer-iteration count. More = tighter Newton
    /// convergence, more compute per substep.
    inner_iterations: u32,
    /// True → use VBD, false → use the Stage-12 XPBD solver. Toggling
    /// hot-swaps the solver in place (model + state are unchanged) so
    /// the user can A/B compare the two without losing the drape.
    use_vbd: bool,
}

impl Default for VbdClothExampleState {
    fn default() -> Self {
        let stretch = 1.0 / VBD_CLOTH_COMPLIANCE as f32;
        Self {
            stretch_stiffness: stretch,
            bend_stiffness: stretch / 30.0,
            contact_stiffness: VBD_CLOTH_CONTACT_STIFFNESS as f32,
            inner_iterations: 10,
            use_vbd: true,
        }
    }
}

fn build_vbd_cloth_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    let nx = VBD_CLOTH_GRID;
    let ny = VBD_CLOTH_GRID;
    let size = VBD_CLOTH_SIZE as f64;
    let origin = MVec3::new(-size * 0.5, VBD_CLOTH_TOP_Y as f64, -size * 0.5);
    let u_axis = MVec3::new(size, 0.0, 0.0);
    let v_axis = MVec3::new(0.0, 0.0, size);

    let _ = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        nx,
        ny,
        VBD_CLOTH_PARTICLE_MASS,
        VBD_CLOTH_PARTICLE_RADIUS,
        VBD_CLOTH_COMPLIANCE,
        VBD_CLOTH_MAX_STRETCH,
        mp::WorldId(0),
    );

    b.add_shape_sphere(
        mp::BodyId::NONE,
        VBD_CLOTH_SPHERE_RADIUS as f64,
        mp::Transform::from_translation(MVec3::new(0.0, VBD_CLOTH_SPHERE_Y as f64, 0.0)),
    )
    .expect("sphere shape");
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane shape");

    let model = b.build().expect("model");
    (model, Vec::new())
}

fn vbd_cloth_pane(
    ui: &mut egui::Ui,
    state: &mut VbdClothExampleState,
    sim: &mut Sim,
    accent: egui::Color32,
) {
    ui.add_space(2.0);
    let active_solver = match &sim.solver {
        SolverBackend::Vbd(_) => "VBD",
        SolverBackend::Xpbd(_) => "XPBD (Stage 12 baseline)",
        _ => "(unknown)",
    };
    readout(ui, "Solver", active_solver);
    readout(
        ui,
        "Cloth",
        &format!(
            "{}² ({} particles, {} springs)",
            VBD_CLOTH_GRID, sim.model.particle_count, sim.model.spring_count,
        ),
    );
    readout(
        ui,
        "Colours",
        &format!(
            "{} (particle), {} (spring)",
            sim.model.particle_color_count, sim.model.spring_color_count,
        ),
    );
    ui.separator();

    // VBD ↔ XPBD toggle.
    let mut want_vbd = state.use_vbd;
    if ui
        .checkbox(&mut want_vbd, "Use VBD (uncheck for XPBD)")
        .changed()
    {
        state.use_vbd = want_vbd;
        let kind = if want_vbd {
            SolverKind::Vbd
        } else {
            SolverKind::Xpbd
        };
        sim.swap_solver_to(kind);
        sim.reset();
    }
    ui.separator();

    let mut dirty = false;

    ui.label("Stretch stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.stretch_stiffness, 1.0e3..=1.0e7).logarithmic(true))
        .changed()
    {
        dirty = true;
    }
    ui.label("Bend stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.bend_stiffness, 1.0e1..=1.0e6).logarithmic(true))
        .changed()
    {
        dirty = true;
    }
    ui.label("Soft contact stiffness (N/m)");
    if ui
        .add(egui::Slider::new(&mut state.contact_stiffness, 1.0e3..=1.0e6).logarithmic(true))
        .changed()
    {
        if let SolverBackend::Vbd(s) = &mut sim.solver {
            s.contact_stiffness = state.contact_stiffness as f64;
        }
    }
    ui.label("Inner iterations");
    if ui
        .add(egui::Slider::new(&mut state.inner_iterations, 1..=30))
        .changed()
    {
        match &mut sim.solver {
            SolverBackend::Vbd(s) => s.iterations = state.inner_iterations as usize,
            SolverBackend::Xpbd(s) => s.position_iterations = state.inner_iterations as usize,
            _ => {}
        }
    }
    ui.separator();

    if dirty {
        // Stretch / bend stiffness changes need to be reflected in
        // the per-spring compliance buffer the kernel reads. We
        // mutate the model directly (it's immutable from a borrow-
        // checker perspective but the compliance buffer is fine to
        // overwrite — VBD reads it once at upload time and again
        // when we call refresh_stiffness).
        apply_vbd_stiffness(sim, state);
    }

    if ui
        .add_sized(
            [ui.available_width(), 28.0],
            button_widget("Reset cloth", accent),
        )
        .clicked()
    {
        sim.reset();
    }

    ui.add_space(6.0);
    ui.small("Hotkey: V");
    ui.small("Compare: toggle VBD off → XPBD baseline.");
    ui.small("XPBD wobbles at high stiffness; VBD settles.");
}

// Right-rail pane for the Stage-20.1 CABLE demo. Carries the
// stretch + friction + contact-stiffness sliders, the AVBD

