// ClothBending scene — two hanging cloth strips side by side, each
// built with a different bending compliance so the user can A/B
// compare the bending response. Port of
// `xtra/newton/newton/examples/cloth/example_cloth_bending.py`.
const CLOTH_BEND_GRID_X: u32 = 12;
const CLOTH_BEND_GRID_Y: u32 = 18;
const CLOTH_BEND_W: f64 = 0.9;
const CLOTH_BEND_H: f64 = 1.4;
const CLOTH_BEND_TOP_Y: f64 = 4.0;
// Static slot for the cloth-bending compliance values, so
// `Sim::build_scene` (no SystemParam access) can read them on rebuild.
// `cloth_bending_pane` writes the bits when the user moves the
// sliders; `apply_scene_switch` triggers a rebuild via `pending`.
static CLOTH_BEND_COMP_LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static CLOTH_BEND_COMP_RIGHT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn cloth_bend_left() -> f32 {
    let bits = CLOTH_BEND_COMP_LEFT.load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 {
        0.05
    } else {
        f32::from_bits(bits)
    }
}
fn cloth_bend_right() -> f32 {
    let bits = CLOTH_BEND_COMP_RIGHT.load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 { 5.0 } else { f32::from_bits(bits) }
}
fn set_cloth_bend(left: f32, right: f32) {
    CLOTH_BEND_COMP_LEFT.store(left.to_bits(), std::sync::atomic::Ordering::Relaxed);
    CLOTH_BEND_COMP_RIGHT.store(right.to_bits(), std::sync::atomic::Ordering::Relaxed);
}
fn build_cloth_bending_model() -> (mp::Model, Vec<BodyVisual>) {
    build_cloth_bending_with(cloth_bend_left(), cloth_bend_right())
}

fn build_cloth_bending_with(comp_left: f32, comp_right: f32) -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.set_angular_damping(0.05);
    b.set_linear_damping(0.4);

    // Two strips. Both pinned along their top edges so they hang.
    for (side_idx, x_offset) in [-1.0_f64, 1.0].iter().enumerate() {
        let comp = if side_idx == 0 { comp_left } else { comp_right };
        let origin = MVec3::new(*x_offset - CLOTH_BEND_W * 0.5, CLOTH_BEND_TOP_Y, 0.0);
        let u_axis = MVec3::new(CLOTH_BEND_W, 0.0, 0.0);
        let v_axis = MVec3::new(0.0, -CLOTH_BEND_H, 0.0);
        let (first, n, _) = b.add_cloth_grid_with_bending(
            origin,
            u_axis,
            v_axis,
            CLOTH_BEND_GRID_X,
            CLOTH_BEND_GRID_Y,
            0.04,        // particle mass
            0.025,       // particle radius
            5.0e-6,      // stretch compliance (very stiff)
            0.05,        // max stretch ratio
            comp as f64, // bending compliance — this is the A/B knob
            mp::WorldId(0),
        );
        let _ = n;
        // Pin top row only (iy = 0 → indices first..first+CLOTH_BEND_GRID_X).
        for ix in 0..CLOTH_BEND_GRID_X {
            b.pin_particle(first + ix);
        }
    }

    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("plane");

    (b.build().expect("cloth bending model"), Vec::new())
}

#[derive(Resource)]
struct ClothBendingExampleState {
    /// Bend compliance for left strip (m/N) — small = stiff.
    k_b_left: f32,
    /// Bend compliance for right strip.
    k_b_right: f32,
    /// Whether sliders moved since last build — triggers rebuild.
    rebuild_pending: bool,
}

impl Default for ClothBendingExampleState {
    fn default() -> Self {
        Self {
            k_b_left: 0.05,
            k_b_right: 5.0,
            rebuild_pending: false,
        }
    }
}

fn cloth_bending_pane(
    ui: &mut egui::Ui,
    state: &mut ClothBendingExampleState,
    pending: &mut PendingSceneSwitch,
    scene: Scene,
) {
    ui.label("Bend compliance — left strip (m/N)");
    if ui
        .add(egui::Slider::new(&mut state.k_b_left, 1e-3..=1e3).logarithmic(true))
        .changed()
    {
        state.rebuild_pending = true;
    }
    ui.label("Bend compliance — right strip");
    if ui
        .add(egui::Slider::new(&mut state.k_b_right, 1e-3..=1e3).logarithmic(true))
        .changed()
    {
        state.rebuild_pending = true;
    }
    ui.separator();
    if state.rebuild_pending {
        ui.colored_label(
            egui::Color32::YELLOW,
            "Rebuild required to apply bending changes.",
        );
        if ui.button("Rebuild scene").clicked() {
            set_cloth_bend(state.k_b_left, state.k_b_right);
            pending.0 = Some(scene);
            state.rebuild_pending = false;
        }
    }
    ui.small("Two cloth strips. Smaller compliance = stiffer.");
    ui.small("Pinned along top edges; gravity does the rest.");
}

