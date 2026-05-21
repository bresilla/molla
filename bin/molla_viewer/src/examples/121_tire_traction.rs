// T4 — Tire traction under load.
//
// Three simple car rigs get the same motor torque on a low-friction/ice surface.
// The top car runs over-inflated tires: tiny contact patch, low effective
// traction, lots of wheel spin. The middle car runs partially-inflated tires:
// larger patch, enough traction to move the payload. The bottom car is
// user-controlled from the right rail: pressure and payload weight sliders let
// the user find the slip/move boundary interactively.
//
// This is intentionally a viewer-side reduced model rather than a full vehicle
// solver. It exists to make the pressure/contact-patch tradeoff obvious.

const TRACTION_WHEEL_RADIUS: f32 = TIRE_RADIUS as f32;
const TRACTION_WHEEL_WIDTH: f32 = TIRE_WIDTH as f32;
const TRACTION_DEFAULT_WEIGHT_N: f32 = 18_000.0;
const TRACTION_DRIVE_TORQUE_PER_REAR_WHEEL: f32 = 1_150.0;
const TRACTION_WHEEL_INERTIA: f32 = 1.15;

fn traction_surface_material(slipperiness: f32) -> molla_vehicle::SurfaceMaterial {
    // Interpolate loose dirt toward polished ice. At the ice end the friction
    // cap dominates everything: even a wide contact patch still visibly spins.
    let s = slipperiness.clamp(0.0, 1.0) as f64;
    let lerp = |dirt: f64, ice: f64| dirt + (ice - dirt) * s;
    molla_vehicle::SurfaceMaterial {
        mu_static: lerp(0.30, 0.025),
        mu_dynamic: lerp(0.22, 0.014),
        bristle_damping: lerp(5.0e4, 2.0e3),
        rolling_resistance: lerp(0.035, 0.006),
        cohesion: lerp(9_000.0, 0.0),
    }
}

fn build_tire_traction_model() -> (mp::Model, Vec<BodyVisual>) {
    let mut b = mp::ModelBuilder::new();
    b.add_shape_plane(mp::BodyId::NONE, MVec3::Y, mp::Transform::IDENTITY)
        .expect("tire traction floor");
    (b.build().expect("tire traction model"), Vec::new())
}

#[derive(Clone, Copy)]
struct TractionCar {
    pressure_pa: f32,
    x: f32,
    v: f32,
    wheel_angle: f32,
    wheel_omega: f32,
    slip: f32,
    last_traction_n: f32,
    last_patch_m: f32,
    last_contacts: usize,
    last_sliding: usize,
    last_normal_load_n: f32,
    lane_z: f32,
    weight_n: f32,
    material_tension: f32,
}

impl TractionCar {
    fn new(pressure_pa: f32, lane_z: f32, weight_n: f32, material_tension: f32) -> Self {
        Self {
            pressure_pa,
            x: -2.4,
            v: 0.0,
            wheel_angle: 0.0,
            wheel_omega: 0.0,
            slip: 0.0,
            last_traction_n: 0.0,
            last_patch_m: 0.0,
            last_contacts: 0,
            last_sliding: 0,
            last_normal_load_n: 0.0,
            lane_z,
            weight_n,
            material_tension,
        }
    }

    fn reset_pose(&mut self) {
        self.x = -2.4;
        self.v = 0.0;
        self.wheel_angle = 0.0;
        self.wheel_omega = 0.0;
        self.slip = 0.0;
        self.last_traction_n = 0.0;
        self.last_patch_m = 0.0;
        self.last_contacts = 0;
        self.last_sliding = 0;
        self.last_normal_load_n = 0.0;
    }

    fn patch_factor(self) -> f32 {
        (self.last_patch_m / (2.0 * TRACTION_WHEEL_RADIUS)).clamp(0.0, 1.25)
    }

    fn hub_y(self) -> f32 {
        tire_visual_hub_height(self.pressure_pa as f64) as f32
    }

    fn step(&mut self, dt: f32, material: molla_vehicle::SurfaceMaterial) {
        let terrain = molla_vehicle::FlatTerrain::new(0.0, material.mu_static);
        let params = molla_vehicle::RealTireParams {
            radius: TRACTION_WHEEL_RADIUS as f64,
            width: TRACTION_WHEEL_WIDTH as f64,
            pressure_pa: self.pressure_pa as f64,
            segments: 64,
            carcass_normal_stiffness: 4.0e5 * self.material_tension as f64,
            tread_shear_stiffness: 2.6e6 * self.material_tension as f64,
            ..Default::default()
        };
        let per_wheel_load = (self.weight_n / 4.0) as f64;
        let hub_y = molla_vehicle::solve_loaded_tire_hub_height(
            &params,
            &terrain,
            self.x as f64,
            self.lane_z as f64,
            per_wheel_load,
        );
        let rear = molla_vehicle::evaluate_brush_tire(
            &params,
            &molla_vehicle::RealWheelState {
                hub_pos: [self.x as f64, hub_y, self.lane_z as f64],
                hub_vel: [self.v as f64, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: self.wheel_omega as f64,
            },
            &terrain,
            &material,
        );

        self.last_patch_m = rear.patch_length as f32;
        self.last_contacts = rear.contact_count;
        self.last_sliding = rear.sliding_count;
        self.last_normal_load_n = rear.normal_load as f32;
        self.last_traction_n = (2.0 * rear.longitudinal_force as f32).max(0.0);
        self.slip = rear.slip_ratio.abs().clamp(0.0, 8.0) as f32 / 8.0;

        // Driven rear wheels: motor torque is opposed by the real brush tire
        // force at the contact patch. If the patch cannot transmit torque,
        // wheel speed rises and slip becomes visible.
        let omega_dot = (TRACTION_DRIVE_TORQUE_PER_REAR_WHEEL
            - (rear.longitudinal_force as f32).max(0.0) * TRACTION_WHEEL_RADIUS)
            / TRACTION_WHEEL_INERTIA;
        self.wheel_omega = (self.wheel_omega + omega_dot * dt).clamp(0.0, 180.0);

        let front_rolling =
            2.0 * material.rolling_resistance as f32 * (self.weight_n / 4.0) * self.v.signum().max(0.0);
        let f_net = (self.last_traction_n - front_rolling).max(0.0);
        self.v += (f_net / (self.weight_n / 9.81).max(1.0)) * dt;
        // Heavy load on slippery ground: keep it slow and readable.
        self.v = self.v.clamp(0.0, 1.4);
        self.x += self.v * dt;

        self.wheel_angle -= self.wheel_omega * dt;
    }
}

#[derive(Resource)]
struct TireTractionState {
    cars: [TractionCar; 3],
    user_pressure_kpa: f32,
    user_weight_kn: f32,
    user_material_tension: f32,
    surface_slipperiness: f32,
    elapsed: f32,
    reset_requested: bool,
}

impl Default for TireTractionState {
    fn default() -> Self {
        Self {
            cars: [
                // High pressure: small patch, spins on low-friction ground.
                TractionCar::new(360_000.0, -1.15, TRACTION_DEFAULT_WEIGHT_N, 1.0),
                // Partially inflated: larger patch, same torque/load moves.
                TractionCar::new(90_000.0, 0.0, TRACTION_DEFAULT_WEIGHT_N, 1.0),
                // User-controlled car.
                TractionCar::new(160_000.0, 1.15, TRACTION_DEFAULT_WEIGHT_N, 1.0),
            ],
            user_pressure_kpa: 160.0,
            user_weight_kn: TRACTION_DEFAULT_WEIGHT_N / 1000.0,
            user_material_tension: 1.0,
            surface_slipperiness: 0.88,
            elapsed: 0.0,
            reset_requested: false,
        }
    }
}

impl TireTractionState {
    fn reset_run(&mut self) {
        self.elapsed = 0.0;
        self.reset_requested = false;
        for car in &mut self.cars {
            car.reset_pose();
        }
    }
}

fn update_tire_traction_example(
    time: Res<Time>,
    sim: Res<Sim>,
    run: Res<SimRunState>,
    mut state: ResMut<TireTractionState>,
) {
    if sim.scene != Scene::TireTraction {
        return;
    }
    state.cars[2].pressure_pa = state.user_pressure_kpa * 1000.0;
    state.cars[2].weight_n = state.user_weight_kn * 1000.0;
    state.cars[2].material_tension = state.user_material_tension;
    let material = traction_surface_material(state.surface_slipperiness);
    if state.elapsed > 18.0 || state.cars.iter().any(|c| c.x > 3.3) {
        state.reset_run();
    }
    if state.reset_requested {
        state.reset_run();
    }
    if run.paused {
        return;
    }
    let dt = time.delta_secs().min(1.0 / 30.0);
    state.elapsed += dt;
    for car in &mut state.cars {
        car.step(dt, material);
    }
}

fn draw_traction_wheel(
    gizmos: &mut Gizmos,
    centre: Vec3,
    pressure_pa: f32,
    angle: f32,
    ring_color: Color,
    contact_color: Color,
) {
    let q = (pressure_pa / REFERENCE_PRESSURE_PA as f32).clamp(0.0, 1.0);
    let half_w = TRACTION_WHEEL_WIDTH * 0.5;
    let hub_y = tire_visual_hub_height(pressure_pa as f64) as f32;
    let segments = TIRE_SEGMENTS;
    let side_z = [-half_w, half_w];
    let mut left = Vec::with_capacity(segments);
    let mut right = Vec::with_capacity(segments);
    for i in 0..segments {
        let theta = angle + (i as f32 / segments as f32) * std::f32::consts::TAU;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let bottom_weight = (-sin_t).max(0.0).powf(1.7);
        let side_bulge = (1.0 - q) * 0.11 * bottom_weight;
        let x = centre.x + (TRACTION_WHEEL_RADIUS + side_bulge) * cos_t;
        let mut y = centre.y + TRACTION_WHEEL_RADIUS * sin_t;
        if y < 0.0 {
            y = 0.0;
        }
        // Use the target pressure's hub height to keep the same flattened
        // profile as `TireDynamics`, even when drawn under a moving chassis.
        let p_mid = Vec3::new(x, y + centre.y - hub_y, centre.z);
        left.push(Vec3::new(p_mid.x, p_mid.y.max(0.0), centre.z + side_z[0]));
        right.push(Vec3::new(p_mid.x, p_mid.y.max(0.0), centre.z + side_z[1]));
    }
    for i in 0..segments {
        let j = (i + 1) % segments;
        gizmos.line(left[i], left[j], ring_color);
        gizmos.line(right[i], right[j], ring_color);
        let bar_color = if left[i].y <= 1.0e-3 || right[i].y <= 1.0e-3 {
            contact_color
        } else {
            Color::srgb(0.16, 0.18, 0.22)
        };
        gizmos.line(left[i], right[i], bar_color);
        if i % 3 == 0 {
            gizmos.line(
                centre,
                0.5 * (left[i] + right[i]),
                Color::srgb(0.42, 0.42, 0.48),
            );
        }
    }
    gizmos.sphere(centre, HUB_RADIUS as f32 * 1.35, Color::srgb(0.82, 0.84, 0.88));
}

fn draw_traction_car(gizmos: &mut Gizmos, car: TractionCar, color: Color) {
    let hub_y = car.hub_y();
    let chassis_center = Vec3::new(car.x, hub_y + 0.50, car.lane_z);
    let half = Vec3::new(0.82, 0.22, 0.42);
    draw_aabb_edges(gizmos, chassis_center - half, chassis_center + half, color);
    // Heavy payload block sitting above the suspension.
    let payload_half = Vec3::new(0.50, 0.24, 0.30);
    let payload_center = chassis_center + Vec3::new(0.0, 0.45, 0.0);
    draw_aabb_edges(
        gizmos,
        payload_center - payload_half,
        payload_center + payload_half,
        Color::srgb(0.55, 0.50, 0.45),
    );

    let wheel_x = [-0.48, 0.48];
    let wheel_z = [-0.30, 0.30];
    for &wx in &wheel_x {
        for &wz in &wheel_z {
            let centre = Vec3::new(car.x + wx, hub_y, car.lane_z + wz);
            draw_traction_wheel(
                gizmos,
                centre,
                car.pressure_pa,
                car.wheel_angle,
                color,
                Color::srgb(1.0, 0.46, 0.18),
            );
            gizmos.line(
                centre,
                chassis_center + Vec3::new(wx, -0.10, wz),
                Color::srgb(0.45, 0.45, 0.50),
            );
        }
    }

    // Motion trail and traction/slip bar. High pressure should show a long
    // red slip bar but little forward displacement; partial pressure should
    // show a green traction bar and visible progress.
    gizmos.line(
        Vec3::new(-2.4, 0.02, car.lane_z),
        Vec3::new(car.x, 0.02, car.lane_z),
        color,
    );
    let bar_base = Vec3::new(-3.1, 0.25, car.lane_z);
    gizmos.line(
        bar_base,
        bar_base + Vec3::new(car.patch_factor() * 0.7, 0.0, 0.0),
        Color::srgb(0.25, 0.90, 0.35),
    );
    gizmos.line(
        bar_base + Vec3::new(0.0, 0.08, 0.0),
        bar_base + Vec3::new(car.slip * 0.7, 0.08, 0.0),
        Color::srgb(1.0, 0.22, 0.12),
    );
}

fn draw_tire_traction_gizmos(
    sim: Res<Sim>,
    state: Res<TireTractionState>,
    mut gizmos: Gizmos,
) {
    if sim.scene != Scene::TireTraction {
        return;
    }
    // Low-friction track lanes.
    for &z in &[-1.15_f32, 0.0, 1.15] {
        draw_aabb_edges(
            &mut gizmos,
            Vec3::new(-2.8, 0.0, z - 0.50),
            Vec3::new(3.8, 0.01, z + 0.50),
            Color::srgb(0.20, 0.22, 0.25),
        );
    }
    draw_traction_car(
        &mut gizmos,
        state.cars[0],
        Color::srgb(0.95, 0.35, 0.25),
    );
    draw_traction_car(
        &mut gizmos,
        state.cars[1],
        Color::srgb(0.25, 0.70, 1.00),
    );
    draw_traction_car(
        &mut gizmos,
        state.cars[2],
        Color::srgb(0.65, 0.95, 0.25),
    );
}

fn tire_traction_pane(ui: &mut egui::Ui, state: &mut TireTractionState, accent: egui::Color32) {
    ui.label("User car controls");
    ui.separator();
    ui.label("Tire inflation pressure");
    let _ = ui.add(
        egui::Slider::new(&mut state.user_pressure_kpa, 20.0..=420.0)
            .suffix(" kPa")
            .logarithmic(true),
    );
    ui.label("Payload + vehicle weight");
    let _ = ui.add(egui::Slider::new(&mut state.user_weight_kn, 4.0..=35.0).suffix(" kN"));
    ui.label("Tire material tension / stiffness");
    let _ = ui.add(
        egui::Slider::new(&mut state.user_material_tension, 0.25..=4.0)
            .suffix("×")
            .logarithmic(true),
    );
    ui.label("Surface slipperiness");
    let _ = ui.add(egui::Slider::new(&mut state.surface_slipperiness, 0.0..=1.0).text("dirt → ice"));
    let material = traction_surface_material(state.surface_slipperiness);
    readout(ui, "Surface μ static", &format!("{:.3}", material.mu_static));
    readout(ui, "Surface μ dynamic", &format!("{:.3}", material.mu_dynamic));
    ui.separator();
    if button(ui, "Reset run", accent).clicked() {
        state.reset_requested = true;
    }
    if let Some(car) = state.cars.get(2) {
        readout(ui, "User speed", &format!("{:.2} m/s", car.v));
        readout(ui, "User slip", &format!("{:.0}%", car.slip * 100.0));
        readout(ui, "Patch factor", &format!("{:.2}×", car.patch_factor()));
        readout(ui, "Contacts", &format!("{}", car.last_contacts));
        readout(ui, "Sliding cells", &format!("{}", car.last_sliding));
        readout(
            ui,
            "Traction",
            &format!("{:.1} kN", car.last_traction_n / 1000.0),
        );
    }
    ui.separator();
    ui.small("Red: fixed high pressure, slips.");
    ui.small("Blue: fixed partial pressure, moves.");
    ui.small("Green: pressure + weight + tire material tension controlled here.");
    ui.small("Push slipperiness toward ice to force visible wheel spin.");
}
