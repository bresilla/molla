fn build_tire_lab_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("tire lab floor");
    (b.build().expect("tire lab model"), Vec::new())
}

// Per-frame state for [`Scene::TireModelLab`]. Live knobs feed the
// per-frame curve evaluation in `update_tire_lab_example`.
#[derive(Resource)]
struct TireLabExampleState {
    /// Normal load on the tire (N). 1000–10000 covers the load
    /// range of a passenger car wheel.
    pub fz: f64,
    /// Ground friction coefficient (Coulomb μ).
    pub friction: f64,
    /// `slip_ratio → Fx` curves for the tire models, sampled
    /// over `slip ∈ [-1, 1]` in `N_SLIP_SAMPLES` steps. Order:
    /// Rigid, TMsimple, Pac89, native real-brush adapter.
    pub fx_curves: [Vec<f32>; TIRE_LAB_MODEL_COUNT],
    /// `slip_angle → Fy` curves for the tire models, sampled
    /// over `α ∈ [-30°, 30°]` in `N_SLIP_SAMPLES` steps.
    pub fy_curves: [Vec<f32>; TIRE_LAB_MODEL_COUNT],
}

impl Default for TireLabExampleState {
    fn default() -> Self {
        Self {
            fz: 4000.0,
            friction: 1.0,
            fx_curves: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            fy_curves: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        }
    }
}

const N_TIRE_LAB_SAMPLES: usize = 121;
const TIRE_LAB_MODEL_COUNT: usize = 4;
const TIRE_LAB_PLOT_HALF_WIDTH: f32 = 5.0; // metres along X
const TIRE_LAB_PLOT_HEIGHT: f32 = 1.5; // visible Fx / Fy span in metres

// V1/V2 plus the native real-brush adapter — evaluate every tire model over
// a fixed slip sweep, store the curves for [`draw_tire_lab_gizmos`].
fn update_tire_lab_example(sim: Res<Sim>, mut state: ResMut<TireLabExampleState>) {
    if sim.scene != Scene::TireModelLab {
        return;
    }
    use molla_vehicle::{Pac89Tire, RealBrushTireModel, RigidTire, TireModel, TmSimpleTire};
    let models: [Box<dyn TireModel>; TIRE_LAB_MODEL_COUNT] = [
        Box::new(RigidTire::default()),
        Box::new(TmSimpleTire::default()),
        Box::new(Pac89Tire::default()),
        Box::new(RealBrushTireModel::passenger_radial()),
    ];
    for (m_idx, model) in models.iter().enumerate() {
        let mut fx = Vec::with_capacity(N_TIRE_LAB_SAMPLES);
        let mut fy = Vec::with_capacity(N_TIRE_LAB_SAMPLES);
        for i in 0..N_TIRE_LAB_SAMPLES {
            let t = (i as f64) / ((N_TIRE_LAB_SAMPLES - 1) as f64);
            let slip = -1.0 + 2.0 * t; // [-1, 1]
            let alpha = (-30.0 + 60.0 * t).to_radians(); // [-30°, 30°]
            let (fx_i, _fy_zero, _mz) = model.evaluate(slip, 0.0, state.fz, state.friction);
            let (_fx_zero, fy_i, _mz) = model.evaluate(0.0, alpha, state.fz, state.friction);
            fx.push(fx_i as f32);
            fy.push(fy_i as f32);
        }
        state.fx_curves[m_idx] = fx;
        state.fy_curves[m_idx] = fy;
    }
}

// Render the tire-model curves as two side-by-side 3-D line plots:
// Fx vs slip ratio (left, around `x = -5`) and Fy vs slip angle
// (right, around `x = +5`). The Y axis is the force scaled to the
// plot height; the Z axis carries the slip parameter.
fn draw_tire_lab_gizmos(sim: Res<Sim>, state: Res<TireLabExampleState>, mut gizmos: Gizmos) {
    if sim.scene != Scene::TireModelLab {
        return;
    }
    let colors = [
        Color::srgb(0.85, 0.45, 0.35), // Rigid
        Color::srgb(0.40, 0.85, 0.50), // TMsimple
        Color::srgb(0.45, 0.55, 0.95), // Pac89
        Color::srgb(0.95, 0.85, 0.25), // native real-brush adapter
    ];
    let plot_origin_fx = Vec3::new(-TIRE_LAB_PLOT_HALF_WIDTH, 1.0, 0.0);
    let plot_origin_fy = Vec3::new(TIRE_LAB_PLOT_HALF_WIDTH, 1.0, 0.0);
    let plot_half_z = 2.0_f32;
    let limit = (state.fz * state.friction) as f32;
    let scale = if limit > 1.0 {
        TIRE_LAB_PLOT_HEIGHT / limit
    } else {
        TIRE_LAB_PLOT_HEIGHT
    };
    // Frame each plot with its axes.
    let axis_color = Color::srgb(0.5, 0.5, 0.55);
    for origin in [plot_origin_fx, plot_origin_fy] {
        gizmos.line(
            origin - Vec3::new(0.0, 0.0, plot_half_z),
            origin + Vec3::new(0.0, 0.0, plot_half_z),
            axis_color,
        );
        gizmos.line(
            origin - Vec3::new(0.0, TIRE_LAB_PLOT_HEIGHT, 0.0),
            origin + Vec3::new(0.0, TIRE_LAB_PLOT_HEIGHT, 0.0),
            axis_color,
        );
    }
    let line_strip = |gizmos: &mut Gizmos, origin: Vec3, samples: &[f32], color: Color| {
        if samples.len() < 2 {
            return;
        }
        let dz = (2.0 * plot_half_z) / (samples.len() as f32 - 1.0);
        let mut prev = origin + Vec3::new(0.0, samples[0] * scale, -plot_half_z + 0.0 * dz);
        for (i, &v) in samples.iter().enumerate().skip(1) {
            let here = origin + Vec3::new(0.0, v * scale, -plot_half_z + i as f32 * dz);
            gizmos.line(prev, here, color);
            prev = here;
        }
    };
    for (i, samples) in state.fx_curves.iter().enumerate() {
        line_strip(&mut gizmos, plot_origin_fx, samples, colors[i]);
    }
    for (i, samples) in state.fy_curves.iter().enumerate() {
        line_strip(&mut gizmos, plot_origin_fy, samples, colors[i]);
    }
}
