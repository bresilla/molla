// T4 — Tire traction under load.
//
// Three simple car rigs get the same motor torque on a low-friction/ice surface.
// The top car runs over-inflated tires: tiny contact patch, low effective
// traction, lots of wheel spin. The middle car runs partially-inflated tires:
// larger patch, enough traction to move the payload. The bottom car is
// user-controlled from the right rail: pressure and payload weight sliders let
// the user find the slip/move boundary interactively.
//
// This is intentionally a compact tire/vehicle rig rather than a full vehicle
// solver. The wheel contact still goes through the real brush model, pressure
// patch sizing, thermal state, and wear state so the sliders affect tire
// physics instead of only moving viewer geometry.

const TRACTION_WHEEL_RADIUS: f32 = TIRE_RADIUS as f32;
const TRACTION_WHEEL_WIDTH: f32 = TIRE_WIDTH as f32;
const TRACTION_DEFAULT_WEIGHT_N: f32 = 18_000.0;
const TRACTION_DRIVE_TORQUE_PER_REAR_WHEEL: f32 = 1_150.0;
const TRACTION_WHEEL_INERTIA: f32 = 1.15;
const TRACTION_DEBUG_CELLS: usize = 24;

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
    /// Lateral world position (Z). Updated from the brush model's
    /// `lateral_force` once steering is engaged.
    z: f32,
    v: f32,
    /// Lateral velocity (m/s in world Z).
    v_lat: f32,
    wheel_angle: f32,
    wheel_omega: f32,
    slip: f32,
    last_traction_n: f32,
    last_force_x_n: f32,
    last_force_z_n: f32,
    /// Stage (i) ext — lateral grip force (N) reported by the brush
    /// model. Drives `v_lat` and is shown in the readout.
    last_lateral_n: f32,
    /// Slip angle at the contact patch (radians).
    last_slip_angle: f32,
    last_patch_m: f32,
    last_contacts: usize,
    last_sliding: usize,
    last_normal_load_n: f32,
    rear_transient: molla_vehicle::RealTireTransientState,
    front_transient: molla_vehicle::RealTireTransientState,
    camber_rad: f32,
    last_raw_slip_angle: f32,
    last_effective_slip_angle: f32,
    last_raw_slip_ratio: f32,
    last_effective_slip_ratio: f32,
    last_camber_rad: f32,
    last_pneumatic_trail_m: f32,
    last_contact_center_x: f32,
    last_aligning_moment_nm: f32,
    last_chassis_yaw_moment_nm: f32,
    last_adhesion_fraction: f32,
    last_sliding_fraction: f32,
    last_friction_budget_n: f32,
    last_consumed_friction_n: f32,
    last_longitudinal_stiffness: f32,
    last_cornering_stiffness: f32,
    last_gpu_delta_valid: bool,
    last_gpu_delta_force_n: f32,
    last_gpu_delta_patch_m: f32,
    last_gpu_delta_slip: f32,
    last_gpu_delta_budget_n: f32,
    last_gpu_delta_moment_nm: f32,
    last_vehicle_handoff_wheels: usize,
    last_vehicle_handoff_force_n: f32,
    last_vehicle_handoff_delta_n: f32,
    last_debug_wheels: [TireDebugWheel; 2],
    last_debug_cells: [TireDebugContactCell; TRACTION_DEBUG_CELLS],
    last_debug_cell_count: usize,
    /// Stage (j) — tread temperature fed by brush dissipation.
    thermal: molla_vehicle::TireThermalState,
    /// Stage (j) — accumulated tread wear fed by brush dissipation.
    wear: molla_vehicle::TireWearState,
    last_mu_factor: f32,
    last_dissipated_w: f32,
    /// Starting lane Z (used to reset back to the centre line).
    lane_z: f32,
    weight_n: f32,
    material_tension: f32,
    /// Stage (i) ext — heading angle in world XZ plane (rad). 0 = +X.
    heading_rad: f32,
    /// Stage (i) ext — steering input ∈ `[-1, 1]`. The first two
    /// reference cars hold this at zero; the user car reads it from
    /// the right rail.
    steer_input: f32,
    /// Actual yaw rate from tire forces (rad/s), not just a visual offset.
    yaw_rate: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct TireDebugContactCell {
    local_x: f32,
    normal_load: f32,
    tangential_force: f32,
    lateral_force: f32,
    normal: [f32; 3],
    sliding: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct TireDebugWheel {
    axle_x: f32,
    steer_angle_rad: f32,
    patch_length: f32,
    contact_center_x: f32,
    pneumatic_trail_m: f32,
    force_x_n: f32,
    force_z_n: f32,
    lateral_force_n: f32,
    aligning_moment_nm: f32,
    debug_cells: [TireDebugContactCell; TRACTION_DEBUG_CELLS],
    debug_cell_count: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct BrushEval {
    longitudinal_force: f32,
    lateral_force: f32,
    force_x: f32,
    force_z: f32,
    normal_load: f32,
    patch_length: f32,
    contact_count: usize,
    sliding_count: usize,
    slip_ratio: f32,
    slip_angle_rad: f32,
    dissipated_power: f32,
    aligning_moment_z: f32,
    camber_angle_rad: f32,
    pneumatic_trail_m: f32,
    contact_center_x: f32,
    adhesion_fraction: f32,
    sliding_fraction: f32,
    friction_budget_n: f32,
    consumed_friction_n: f32,
    effective_longitudinal_stiffness: f32,
    effective_cornering_stiffness: f32,
    debug_cells: [TireDebugContactCell; TRACTION_DEBUG_CELLS],
    debug_cell_count: usize,
}

fn traction_transient_gpu_state(
    transient: &molla_vehicle::RealTireTransientState,
) -> molla::kernels::TireTransientGpuState {
    let layer = [
        transient.kappa_eff as f32,
        transient.alpha_eff as f32,
        transient.tread_yaw as f32,
        0.0,
    ];
    molla::kernels::TireTransientGpuState {
        tread: layer,
        belt: layer,
        carcass: layer,
    }
}

fn traction_transient_gpu_input(
    params: &molla_vehicle::RealTireParams,
    transient: &molla_vehicle::RealTireTransientState,
    wheel: &molla_vehicle::RealWheelState,
    dt: f32,
) -> molla::kernels::TireTransientGpuInput {
    let h_len = (wheel.heading[0] * wheel.heading[0] + wheel.heading[2] * wheel.heading[2])
        .sqrt()
        .max(1.0e-9);
    let h = [wheel.heading[0] / h_len, 0.0, wheel.heading[2] / h_len];
    let side = [-h[2], 0.0, h[0]];
    let v_long = wheel.hub_vel[0] * h[0] + wheel.hub_vel[2] * h[2];
    let v_lat = wheel.hub_vel[0] * side[0] + wheel.hub_vel[2] * side[2];
    let denom = v_long.abs().max(params.relaxation_speed.max(1.0e-4));
    let raw_kappa = ((wheel.omega * params.radius - v_long) / denom).clamp(-8.0, 8.0);
    let raw_alpha = v_lat.atan2(denom).clamp(-1.2, 1.2);
    let speed = (v_long * v_long + v_lat * v_lat)
        .sqrt()
        .max(params.relaxation_speed);
    let scale = molla_vehicle::TireRelaxationContext::default()
        .relaxation_length_scale(params)
        .clamp(0.05, 20.0);
    molla::kernels::TireTransientGpuInput {
        command: [
            raw_kappa as f32,
            raw_alpha as f32,
            (speed * dt as f64) as f32,
            params.radius as f32,
        ],
        // Invert the shader's tread/carcass layer scale factors so a simple
        // RealTireTransientState maps to identical tread/belt/carcass layers.
        tread_lengths: [
            (transient.relaxation_length_longitudinal / 0.85) as f32,
            (transient.relaxation_length_lateral / 0.85) as f32,
            (transient.tread_yaw_relaxation_length / 0.85) as f32,
            scale as f32,
        ],
        belt_lengths: [
            transient.relaxation_length_longitudinal as f32,
            transient.relaxation_length_lateral as f32,
            transient.tread_yaw_relaxation_length as f32,
            0.0,
        ],
        carcass_lengths: [
            (transient.relaxation_length_longitudinal / 1.15) as f32,
            (transient.relaxation_length_lateral / 1.15) as f32,
            (transient.tread_yaw_relaxation_length / 1.15) as f32,
            0.0,
        ],
    }
}

fn traction_effective_wheel_from_gpu_transient(
    params: &molla_vehicle::RealTireParams,
    wheel: &molla_vehicle::RealWheelState,
    out: molla::kernels::TireTransientGpuOutput,
) -> molla_vehicle::RealWheelState {
    let h_len = (wheel.heading[0] * wheel.heading[0] + wheel.heading[2] * wheel.heading[2])
        .sqrt()
        .max(1.0e-9);
    let h = [wheel.heading[0] / h_len, 0.0, wheel.heading[2] / h_len];
    let side = [-h[2], 0.0, h[0]];
    let v_long = wheel.hub_vel[0] * h[0] + wheel.hub_vel[2] * h[2];
    let denom = v_long.abs().max(params.relaxation_speed.max(1.0e-4));
    let kappa_eff = out.effective[0] as f64;
    let alpha_eff = out.effective[1] as f64;
    let v_lat_eff = alpha_eff.tan().clamp(-4.0, 4.0) * denom;
    let omega_eff = (v_long + kappa_eff * denom) / params.radius.max(1.0e-6);
    molla_vehicle::RealWheelState {
        hub_pos: wheel.hub_pos,
        hub_vel: [
            h[0] * v_long + side[0] * v_lat_eff,
            0.0,
            h[2] * v_long + side[2] * v_lat_eff,
        ],
        heading: h,
        omega: omega_eff,
    }
}

fn traction_apply_transient_gpu_output(
    transient: &mut molla_vehicle::RealTireTransientState,
    out: molla::kernels::TireTransientGpuOutput,
) {
    transient.kappa_eff = out.effective[0] as f64;
    transient.alpha_eff = out.effective[1] as f64;
    transient.tread_yaw = out.effective[2] as f64;
    transient.longitudinal_deflection = out.deflection[0] as f64;
    transient.lateral_deflection = out.deflection[1] as f64;
}

impl TractionCar {
    fn new(pressure_pa: f32, lane_z: f32, weight_n: f32, material_tension: f32) -> Self {
        Self {
            pressure_pa,
            x: -2.4,
            z: lane_z,
            v: 0.0,
            v_lat: 0.0,
            wheel_angle: 0.0,
            wheel_omega: 0.0,
            slip: 0.0,
            last_traction_n: 0.0,
            last_force_x_n: 0.0,
            last_force_z_n: 0.0,
            last_lateral_n: 0.0,
            last_slip_angle: 0.0,
            last_patch_m: 0.0,
            last_contacts: 0,
            last_sliding: 0,
            last_normal_load_n: 0.0,
            rear_transient: Default::default(),
            front_transient: Default::default(),
            camber_rad: 0.0,
            last_raw_slip_angle: 0.0,
            last_effective_slip_angle: 0.0,
            last_raw_slip_ratio: 0.0,
            last_effective_slip_ratio: 0.0,
            last_camber_rad: 0.0,
            last_pneumatic_trail_m: 0.0,
            last_contact_center_x: 0.0,
            last_aligning_moment_nm: 0.0,
            last_chassis_yaw_moment_nm: 0.0,
            last_adhesion_fraction: 1.0,
            last_sliding_fraction: 0.0,
            last_friction_budget_n: 0.0,
            last_consumed_friction_n: 0.0,
            last_longitudinal_stiffness: 0.0,
            last_cornering_stiffness: 0.0,
            last_gpu_delta_valid: false,
            last_gpu_delta_force_n: 0.0,
            last_gpu_delta_patch_m: 0.0,
            last_gpu_delta_slip: 0.0,
            last_gpu_delta_budget_n: 0.0,
            last_gpu_delta_moment_nm: 0.0,
            last_vehicle_handoff_wheels: 0,
            last_vehicle_handoff_force_n: 0.0,
            last_vehicle_handoff_delta_n: 0.0,
            last_debug_wheels: [TireDebugWheel::default(); 2],
            last_debug_cells: [TireDebugContactCell::default(); TRACTION_DEBUG_CELLS],
            last_debug_cell_count: 0,
            thermal: molla_vehicle::TireThermalState {
                temperature_c: 55.0,
                ..Default::default()
            },
            wear: Default::default(),
            last_mu_factor: 1.0,
            last_dissipated_w: 0.0,
            lane_z,
            weight_n,
            material_tension,
            heading_rad: 0.0,
            steer_input: 0.0,
            yaw_rate: 0.0,
        }
    }

    fn reset_pose(&mut self) {
        self.x = -2.4;
        self.z = self.lane_z;
        self.v = 0.0;
        self.v_lat = 0.0;
        self.wheel_angle = 0.0;
        self.wheel_omega = 0.0;
        self.slip = 0.0;
        self.last_traction_n = 0.0;
        self.last_force_x_n = 0.0;
        self.last_force_z_n = 0.0;
        self.last_lateral_n = 0.0;
        self.last_slip_angle = 0.0;
        self.last_patch_m = 0.0;
        self.last_contacts = 0;
        self.last_sliding = 0;
        self.last_normal_load_n = 0.0;
        self.rear_transient.reset();
        self.front_transient.reset();
        self.last_raw_slip_angle = 0.0;
        self.last_effective_slip_angle = 0.0;
        self.last_raw_slip_ratio = 0.0;
        self.last_effective_slip_ratio = 0.0;
        self.last_camber_rad = 0.0;
        self.last_pneumatic_trail_m = 0.0;
        self.last_contact_center_x = 0.0;
        self.last_aligning_moment_nm = 0.0;
        self.last_chassis_yaw_moment_nm = 0.0;
        self.last_adhesion_fraction = 1.0;
        self.last_sliding_fraction = 0.0;
        self.last_friction_budget_n = 0.0;
        self.last_consumed_friction_n = 0.0;
        self.last_longitudinal_stiffness = 0.0;
        self.last_cornering_stiffness = 0.0;
        self.last_gpu_delta_valid = false;
        self.last_gpu_delta_force_n = 0.0;
        self.last_gpu_delta_patch_m = 0.0;
        self.last_gpu_delta_slip = 0.0;
        self.last_gpu_delta_budget_n = 0.0;
        self.last_gpu_delta_moment_nm = 0.0;
        self.last_vehicle_handoff_wheels = 0;
        self.last_vehicle_handoff_force_n = 0.0;
        self.last_vehicle_handoff_delta_n = 0.0;
        self.last_debug_wheels = [TireDebugWheel::default(); 2];
        self.last_debug_cells = [TireDebugContactCell::default(); TRACTION_DEBUG_CELLS];
        self.last_debug_cell_count = 0;
        self.thermal = molla_vehicle::TireThermalState {
            temperature_c: 55.0,
            ..Default::default()
        };
        self.wear = Default::default();
        self.last_mu_factor = 1.0;
        self.last_dissipated_w = 0.0;
        self.heading_rad = 0.0;
        self.yaw_rate = 0.0;
    }

    /// Visualization/readout-only patch scale derived from the real contact
    /// telemetry. It must never feed the force path.
    fn visual_patch_factor(self) -> f32 {
        (self.last_patch_m / (2.0 * TRACTION_WHEEL_RADIUS)).clamp(0.0, 1.25)
    }

    fn hub_y(self) -> f32 {
        tire_visual_hub_height_for_load(self.pressure_pa as f64, (self.weight_n / 4.0) as f64) as f32
    }

    fn step(
        &mut self,
        dt: f32,
        material: molla_vehicle::SurfaceMaterial,
        gpu: Option<(
            &molla::kernels::TireBrushGpuKernel,
            &molla::kernels::TireVehicleYawGpuKernel,
            &wgpu::Device,
            &wgpu::Queue,
        )>,
        transient_gpu: Option<(
            &molla::kernels::TireTransientGpuKernel,
            &molla::kernels::TireTransientGpuBatch,
            &wgpu::Device,
            &wgpu::Queue,
        )>,
    ) -> (bool, Option<String>) {
        let mut live_material = material;
        self.last_mu_factor = (self.thermal.mu_factor() * self.wear.mu_factor()) as f32;
        live_material.mu_static *= self.last_mu_factor as f64;
        live_material.mu_dynamic *= self.last_mu_factor as f64;

        let terrain = molla_vehicle::FlatTerrain::new(0.0, live_material.mu_static);
        let mut params = molla_vehicle::RealTireParams::passenger_radial()
            .with_geometry_and_pressure(
                TRACTION_WHEEL_RADIUS as f64,
                TRACTION_WHEEL_WIDTH as f64,
                self.pressure_pa as f64,
            )
            .with_segments(64);
        params.carcass_normal_stiffness = 4.0e5 * self.material_tension as f64;
        params.tread_shear_stiffness = 2.6e6 * self.material_tension as f64;
        let per_wheel_load = (self.weight_n / 4.0) as f64;
        let hub_y = molla_vehicle::solve_loaded_tire_hub_height(
            &params,
            &terrain,
            self.x as f64,
            self.z as f64,
            per_wheel_load,
        );

        // Stage (i) ext — the chassis heading is now integrated from
        // tire forces below. Steering first rotates the front tire
        // basis, then the brush model produces lateral force/yaw moment.
        let steer_angle = self.steer_angle_rad();
        let heading_vec = [
            self.heading_rad.cos() as f64,
            0.0,
            self.heading_rad.sin() as f64,
        ];
        let front_heading = [
            (self.heading_rad + steer_angle).cos() as f64,
            0.0,
            (self.heading_rad + steer_angle).sin() as f64,
        ];
        let world_vel = [self.v as f64, 0.0, self.v_lat as f64];
        let rear_state = molla_vehicle::RealWheelState {
            hub_pos: [self.x as f64, hub_y, self.z as f64],
            hub_vel: world_vel,
            heading: heading_vec,
            omega: self.wheel_omega as f64,
        };
        let passive_front_omega = {
            let v_forward = self.v as f64 * front_heading[0] + self.v_lat as f64 * front_heading[2];
            (v_forward / TRACTION_WHEEL_RADIUS as f64).max(0.0)
        };
        let front_state = molla_vehicle::RealWheelState {
            hub_pos: [self.x as f64, hub_y, self.z as f64],
            hub_vel: world_vel,
            heading: front_heading,
            omega: passive_front_omega,
        };
        let slip_diag = |wheel: &molla_vehicle::RealWheelState| -> (f32, f32) {
            let h = wheel.heading;
            let side = [-h[2], 0.0, h[0]];
            let v_long = wheel.hub_vel[0] * h[0] + wheel.hub_vel[2] * h[2];
            let v_lat = wheel.hub_vel[0] * side[0] + wheel.hub_vel[2] * side[2];
            let denom = v_long
                .abs()
                .max(params.relaxation_speed.max(1.0e-4));
            (
                ((wheel.omega * params.radius - v_long) / denom) as f32,
                v_lat.atan2(denom) as f32,
            )
        };
        let (raw_kappa, raw_alpha) = slip_diag(&front_state);
        let mut used_transient_gpu = false;
        let mut transient_gpu_error = None;
        let (rear_eff, front_eff) =
            if let Some((kernel, batch, device, queue)) = transient_gpu {
                let inputs = [
                    traction_transient_gpu_input(&params, &self.rear_transient, &rear_state, dt),
                    traction_transient_gpu_input(&params, &self.front_transient, &front_state, dt),
                ];
                let states = [
                    traction_transient_gpu_state(&self.rear_transient),
                    traction_transient_gpu_state(&self.front_transient),
                ];
                match kernel.dispatch_persistent_blocking(device, queue, batch, &inputs, &states) {
                    Ok((_states, outputs)) if outputs.len() == 2 => {
                        traction_apply_transient_gpu_output(&mut self.rear_transient, outputs[0]);
                        traction_apply_transient_gpu_output(&mut self.front_transient, outputs[1]);
                        used_transient_gpu = true;
                        (
                            traction_effective_wheel_from_gpu_transient(
                                &params, &rear_state, outputs[0],
                            ),
                            traction_effective_wheel_from_gpu_transient(
                                &params,
                                &front_state,
                                outputs[1],
                            ),
                        )
                    }
                    Ok((_states, outputs)) => {
                        transient_gpu_error = Some(format!(
                            "GPU transient returned {} rows, expected 2",
                            outputs.len()
                        ));
                        (
                            self.rear_transient.step(
                                &params,
                                &rear_state,
                                [0.0, 1.0, 0.0],
                                dt as f64,
                            ),
                            self.front_transient.step(
                                &params,
                                &front_state,
                                [0.0, 1.0, 0.0],
                                dt as f64,
                            ),
                        )
                    }
                    Err(err) => {
                        transient_gpu_error =
                            Some(format!("GPU transient dispatch failed: {err}"));
                        (
                            self.rear_transient.step(
                                &params,
                                &rear_state,
                                [0.0, 1.0, 0.0],
                                dt as f64,
                            ),
                            self.front_transient.step(
                                &params,
                                &front_state,
                                [0.0, 1.0, 0.0],
                                dt as f64,
                            ),
                        )
                    }
                }
            } else {
                (
                    self.rear_transient
                        .step(&params, &rear_state, [0.0, 1.0, 0.0], dt as f64),
                    self.front_transient.step(
                        &params,
                        &front_state,
                        [0.0, 1.0, 0.0],
                        dt as f64,
                    ),
                )
            };
        let orientation = molla_vehicle::RealWheelOrientation {
            wheel_up: [
                0.0,
                (self.camber_rad as f64).cos(),
                (self.camber_rad as f64).sin(),
            ],
        };

        let mut gpu_delta: Option<(f32, f32, f32, f32, f32)> = None;
        let (rear, front, used_gpu, gpu_error) = match gpu {
            Some((brush_kernel, vehicle_kernel, device, queue)) => {
                match eval_brush_pair_gpu_vehicle_handoff(
                    vehicle_kernel,
                    brush_kernel,
                    device,
                    queue,
                    &params,
                    &rear_eff,
                    &front_eff,
                    &live_material,
                    self.camber_rad,
                ) {
                    Ok(mut pair) => {
                        let cpu_rear = eval_brush_cpu(
                            &params,
                            &rear_eff,
                            &terrain,
                            &live_material,
                            &orientation,
                        );
                        let cpu_front = eval_brush_cpu(
                            &params,
                            &front_eff,
                            &terrain,
                            &live_material,
                            &orientation,
                        );
                        self.last_vehicle_handoff_wheels = 2;
                        self.last_vehicle_handoff_force_n =
                            2.0 * (pair[0].force_x + pair[1].force_x);
                        self.last_vehicle_handoff_delta_n = (self.last_vehicle_handoff_force_n
                            - 2.0 * (cpu_rear.force_x + cpu_front.force_x))
                            .abs();
                        gpu_delta = Some(brush_pair_delta(pair, [cpu_rear, cpu_front]));
                        pair[0].debug_cells = cpu_rear.debug_cells;
                        pair[0].debug_cell_count = cpu_rear.debug_cell_count;
                        pair[1].debug_cells = cpu_front.debug_cells;
                        pair[1].debug_cell_count = cpu_front.debug_cell_count;
                        (pair[0], pair[1], true, None)
                    }
                    Err(vehicle_err) => {
                        self.last_vehicle_handoff_wheels = 0;
                        self.last_vehicle_handoff_force_n = 0.0;
                        self.last_vehicle_handoff_delta_n = 0.0;
                        match eval_brush_pair_gpu(
                            brush_kernel,
                            device,
                            queue,
                            &params,
                            &rear_eff,
                            &front_eff,
                            &live_material,
                            self.camber_rad,
                        ) {
                            Ok(mut pair) => {
                                let cpu_rear = eval_brush_cpu(
                                    &params,
                                    &rear_eff,
                                    &terrain,
                                    &live_material,
                                    &orientation,
                                );
                                let cpu_front = eval_brush_cpu(
                                    &params,
                                    &front_eff,
                                    &terrain,
                                    &live_material,
                                    &orientation,
                                );
                                gpu_delta = Some(brush_pair_delta(pair, [cpu_rear, cpu_front]));
                                pair[0].debug_cells = cpu_rear.debug_cells;
                                pair[0].debug_cell_count = cpu_rear.debug_cell_count;
                                pair[1].debug_cells = cpu_front.debug_cells;
                                pair[1].debug_cell_count = cpu_front.debug_cell_count;
                                (pair[0], pair[1], true, Some(vehicle_err))
                            }
                            Err(err) => {
                                let rear = eval_brush_cpu(
                                    &params,
                                    &rear_eff,
                                    &terrain,
                                    &live_material,
                                    &orientation,
                                );
                                let front = eval_brush_cpu(
                                    &params,
                                    &front_eff,
                                    &terrain,
                                    &live_material,
                                    &orientation,
                                );
                                (
                                    rear,
                                    front,
                                    false,
                                    Some(format!(
                                        "{vehicle_err}; direct GPU brush fallback also failed: {err}"
                                    )),
                                )
                            }
                        }
                    }
                }
            }
            None => {
                self.last_vehicle_handoff_wheels = 0;
                self.last_vehicle_handoff_force_n = 0.0;
                self.last_vehicle_handoff_delta_n = 0.0;
                (
                    eval_brush_cpu(&params, &rear_eff, &terrain, &live_material, &orientation),
                    eval_brush_cpu(&params, &front_eff, &terrain, &live_material, &orientation),
                    false,
                    None,
                )
            }
        };

        self.last_patch_m = (rear.patch_length + front.patch_length) * 0.5;
        self.last_contacts = rear.contact_count + front.contact_count;
        self.last_sliding = rear.sliding_count + front.sliding_count;
        self.last_normal_load_n = rear.normal_load + front.normal_load;
        self.last_traction_n = (2.0 * rear.longitudinal_force).max(0.0);
        self.last_force_x_n = 2.0 * (rear.force_x + front.force_x);
        self.last_force_z_n = 2.0 * (rear.force_z + front.force_z);
        self.last_lateral_n =
            (2.0 * rear.lateral_force + 2.0 * front.lateral_force).clamp(-1.0e6, 1.0e6);
        self.last_slip_angle = front.slip_angle_rad;
        self.last_raw_slip_angle = raw_alpha;
        self.last_effective_slip_angle = self.front_transient.alpha_eff as f32;
        self.last_raw_slip_ratio = raw_kappa;
        self.last_effective_slip_ratio = self.front_transient.kappa_eff as f32;
        self.last_camber_rad = front.camber_angle_rad;
        self.last_pneumatic_trail_m = front.pneumatic_trail_m;
        self.last_contact_center_x = front.contact_center_x;
        self.last_aligning_moment_nm = 2.0 * (rear.aligning_moment_z + front.aligning_moment_z);
        self.last_debug_wheels = [
            debug_wheel_from_eval(-0.48, 0.0, rear),
            debug_wheel_from_eval(0.48, steer_angle, front),
        ];
        self.last_debug_cells = front.debug_cells;
        self.last_debug_cell_count = front.debug_cell_count;
        self.last_sliding_fraction = (0.5 * (rear.sliding_fraction + front.sliding_fraction))
            .clamp(0.0, 1.0);
        self.last_adhesion_fraction = (0.5 * (rear.adhesion_fraction + front.adhesion_fraction))
            .clamp(0.0, 1.0);
        self.last_friction_budget_n = 2.0 * (rear.friction_budget_n + front.friction_budget_n);
        self.last_consumed_friction_n =
            2.0 * (rear.consumed_friction_n + front.consumed_friction_n);
        self.last_longitudinal_stiffness = 0.5
            * (rear.effective_longitudinal_stiffness + front.effective_longitudinal_stiffness);
        self.last_cornering_stiffness =
            0.5 * (rear.effective_cornering_stiffness + front.effective_cornering_stiffness);
        if let Some((force, patch, slip, budget, moment)) = gpu_delta {
            self.last_gpu_delta_valid = true;
            self.last_gpu_delta_force_n = force;
            self.last_gpu_delta_patch_m = patch;
            self.last_gpu_delta_slip = slip;
            self.last_gpu_delta_budget_n = budget;
            self.last_gpu_delta_moment_nm = moment;
        } else {
            self.last_gpu_delta_valid = false;
            self.last_gpu_delta_force_n = 0.0;
            self.last_gpu_delta_patch_m = 0.0;
            self.last_gpu_delta_slip = 0.0;
            self.last_gpu_delta_budget_n = 0.0;
            self.last_gpu_delta_moment_nm = 0.0;
        }
        self.slip = rear
            .slip_ratio
            .abs()
            .max(front.slip_ratio.abs())
            .clamp(0.0, 8.0)
            / 8.0;
        self.last_dissipated_w = 2.0 * (rear.dissipated_power + front.dissipated_power);
        self.thermal
            .step(self.last_dissipated_w as f64, dt as f64);
        self.wear.step(self.last_dissipated_w as f64, dt as f64);

        // Driven rear wheels: motor torque is opposed by the brush
        // longitudinal force; lateral force pulls the car sideways.
        let omega_dot = (TRACTION_DRIVE_TORQUE_PER_REAR_WHEEL
            - rear.longitudinal_force.max(0.0) * TRACTION_WHEEL_RADIUS)
            / TRACTION_WHEEL_INERTIA;
        self.wheel_omega = (self.wheel_omega + omega_dot * dt).clamp(0.0, 180.0);

        let front_rolling = 2.0
            * live_material.rolling_resistance as f32
            * (self.weight_n / 4.0)
            * self.v.signum().max(0.0);
        let total_fx = 2.0 * rear.force_x + 2.0 * front.force_x - front_rolling;
        let total_fz = 2.0 * rear.force_z + 2.0 * front.force_z;
        let mass = (self.weight_n / 9.81).max(1.0);
        self.v += (total_fx / mass) * dt;
        self.v_lat += (total_fz / mass) * dt;
        // Light velocity damping so the car doesn't slide forever
        // even on slick floors when steering is zero.
        let lateral_drag = 1.5_f32;
        self.v_lat -= self.v_lat * lateral_drag * dt;
        // Heavy load on slippery ground: keep it slow and readable.
        self.v = self.v.clamp(-0.2, 1.4);
        self.v_lat = self.v_lat.clamp(-1.2, 1.2);

        let wheelbase = 0.96_f32;
        let front_arm = Vec3::new(
            self.heading_rad.cos() * (wheelbase * 0.5),
            0.0,
            self.heading_rad.sin() * (wheelbase * 0.5),
        );
        let rear_arm = -front_arm;
        let front_force = Vec3::new(2.0 * front.force_x, 0.0, 2.0 * front.force_z);
        let rear_force = Vec3::new(2.0 * rear.force_x, 0.0, 2.0 * rear.force_z);
        let yaw_moment =
            front_arm.z * front_force.x - front_arm.x * front_force.z
                + rear_arm.z * rear_force.x
                - rear_arm.x * rear_force.z;
        self.last_chassis_yaw_moment_nm = yaw_moment;
        let yaw_inertia = (mass * (wheelbase * wheelbase + 0.60 * 0.60) / 12.0).max(1.0);
        self.yaw_rate += (yaw_moment / yaw_inertia) * dt;
        self.yaw_rate -= self.yaw_rate * 2.0 * dt;
        self.yaw_rate = self.yaw_rate.clamp(-1.4, 1.4);
        self.heading_rad += self.yaw_rate * dt;
        self.heading_rad = self.heading_rad.clamp(-1.2, 1.2);

        self.x += self.v * dt;
        self.z += self.v_lat * dt;
        // Soft pull-back toward the lane centre line so the demo
        // doesn't wander off-screen when the user releases the slider.
        self.z += (self.lane_z - self.z) * 0.4 * dt;

        self.wheel_angle -= self.wheel_omega * dt;
        (used_gpu || used_transient_gpu, gpu_error.or(transient_gpu_error))
    }

    fn steer_angle_rad(self) -> f32 {
        self.steer_input.clamp(-1.0, 1.0) * 0.58
    }
}

fn debug_cells_from_contact(
    contact: &molla_vehicle::RealTireContact,
) -> ([TireDebugContactCell; TRACTION_DEBUG_CELLS], usize) {
    let mut cells = [TireDebugContactCell::default(); TRACTION_DEBUG_CELLS];
    let count = contact.cells.len().min(TRACTION_DEBUG_CELLS);
    for (dst, src) in cells.iter_mut().zip(contact.cells.iter()).take(count) {
        *dst = TireDebugContactCell {
            local_x: src.local_x as f32,
            normal_load: src.normal_load as f32,
            tangential_force: src.tangential_force as f32,
            lateral_force: src.lateral_force as f32,
            normal: [
                src.normal[0] as f32,
                src.normal[1] as f32,
                src.normal[2] as f32,
            ],
            sliding: src.sliding,
        };
    }
    (cells, count)
}

fn debug_wheel_from_eval(axle_x: f32, steer_angle_rad: f32, eval: BrushEval) -> TireDebugWheel {
    TireDebugWheel {
        axle_x,
        steer_angle_rad,
        patch_length: eval.patch_length,
        contact_center_x: eval.contact_center_x,
        pneumatic_trail_m: eval.pneumatic_trail_m,
        force_x_n: eval.force_x,
        force_z_n: eval.force_z,
        lateral_force_n: eval.lateral_force,
        aligning_moment_nm: eval.aligning_moment_z,
        debug_cells: eval.debug_cells,
        debug_cell_count: eval.debug_cell_count,
    }
}

fn eval_brush_cpu(
    params: &molla_vehicle::RealTireParams,
    wheel: &molla_vehicle::RealWheelState,
    terrain: &molla_vehicle::FlatTerrain,
    material: &molla_vehicle::SurfaceMaterial,
    orientation: &molla_vehicle::RealWheelOrientation,
) -> BrushEval {
    let contact =
        molla_vehicle::evaluate_brush_tire_oriented(params, wheel, terrain, material, orientation);
    let (debug_cells, debug_cell_count) = debug_cells_from_contact(&contact);
    BrushEval {
        longitudinal_force: contact.longitudinal_force as f32,
        lateral_force: contact.lateral_force as f32,
        force_x: contact.force[0] as f32,
        force_z: contact.force[2] as f32,
        normal_load: contact.normal_load as f32,
        patch_length: contact.patch_length as f32,
        contact_count: contact.contact_count,
        sliding_count: contact.sliding_count,
        slip_ratio: contact.slip_ratio as f32,
        slip_angle_rad: contact.slip_angle_rad as f32,
        dissipated_power: contact.dissipated_power as f32,
        aligning_moment_z: contact.aligning_moment_z as f32,
        camber_angle_rad: contact.camber_angle_rad as f32,
        pneumatic_trail_m: contact.pneumatic_trail_m as f32,
        contact_center_x: contact.contact_center_x as f32,
        adhesion_fraction: contact.adhesion_fraction as f32,
        sliding_fraction: contact.sliding_fraction as f32,
        friction_budget_n: contact.available_friction_budget as f32,
        consumed_friction_n: contact.consumed_friction_budget as f32,
        effective_longitudinal_stiffness: contact.effective_longitudinal_stiffness as f32,
        effective_cornering_stiffness: contact.effective_cornering_stiffness as f32,
        debug_cells,
        debug_cell_count,
    }
}

fn brush_pair_delta(gpu: [BrushEval; 2], cpu: [BrushEval; 2]) -> (f32, f32, f32, f32, f32) {
    let mut force_delta = 0.0_f32;
    let mut patch_delta = 0.0_f32;
    let mut slip_delta = 0.0_f32;
    let mut budget_delta = 0.0_f32;
    let mut moment_delta = 0.0_f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let df = ((g.force_x - c.force_x).powi(2) + (g.force_z - c.force_z).powi(2)).sqrt();
        force_delta = force_delta.max(df);
        patch_delta = patch_delta.max((g.patch_length - c.patch_length).abs());
        slip_delta = slip_delta
            .max((g.slip_ratio - c.slip_ratio).abs())
            .max((g.slip_angle_rad - c.slip_angle_rad).abs());
        budget_delta = budget_delta
            .max((g.friction_budget_n - c.friction_budget_n).abs())
            .max((g.consumed_friction_n - c.consumed_friction_n).abs());
        moment_delta = moment_delta.max((g.aligning_moment_z - c.aligning_moment_z).abs());
    }
    (
        force_delta,
        patch_delta,
        slip_delta,
        budget_delta,
        moment_delta,
    )
}

fn pack_brush_gpu_input(
    params: &molla_vehicle::RealTireParams,
    wheel: &molla_vehicle::RealWheelState,
    material: &molla_vehicle::SurfaceMaterial,
    camber_rad: f32,
) -> molla::kernels::BrushWheelInput {
    molla::kernels::BrushWheelInput {
        hub_pos: [
            wheel.hub_pos[0] as f32,
            wheel.hub_pos[1] as f32,
            wheel.hub_pos[2] as f32,
            0.0,
        ],
        hub_vel: [
            wheel.hub_vel[0] as f32,
            wheel.hub_vel[1] as f32,
            wheel.hub_vel[2] as f32,
            0.0,
        ],
        heading: [
            wheel.heading[0] as f32,
            wheel.heading[1] as f32,
            wheel.heading[2] as f32,
            params.structural_lateral_bias_per_normal as f32,
        ],
        spin: [
            wheel.omega as f32,
            params.radius as f32,
            params.width as f32,
            params.pressure_pa as f32,
        ],
        tire: [
            params.segments as f32,
            params.carcass_normal_stiffness as f32,
            params.tread_shear_stiffness as f32,
            params.relaxation_speed as f32,
        ],
        material: [
            material.mu_static as f32,
            material.mu_dynamic as f32,
            material.bristle_damping as f32,
            material.rolling_resistance as f32,
        ],
        // `material` already contains heat/wear μ feedback here; keep
        // GPU `mu_factor` at 1 so the correction is not applied twice.
        extra: [material.cohesion as f32, 0.0, 1.0, camber_rad],
    }
}

fn brush_eval_from_gpu(
    out: molla::kernels::BrushWheelOutput,
    heading: [f64; 3],
) -> BrushEval {
    let hx = heading[0] as f32;
    let hz = heading[2] as f32;
    let sx = -hz;
    let sz = hx;
    let longitudinal_force = out.forces[3];
    let lateral_force = out.forces[2];
    BrushEval {
        longitudinal_force,
        lateral_force,
        force_x: hx * longitudinal_force + sx * lateral_force,
        force_z: hz * longitudinal_force + sz * lateral_force,
        normal_load: out.forces[1],
        patch_length: out.diag1[0],
        contact_count: out.diag1[3].max(0.0).round() as usize,
        sliding_count: out.diag2[0].max(0.0).round() as usize,
        slip_ratio: out.diag1[1],
        slip_angle_rad: out.diag2[3],
        dissipated_power: out.diag3[0],
        aligning_moment_z: out.diag2[1],
        camber_angle_rad: out.diag3[1],
        contact_center_x: out.diag3[2],
        pneumatic_trail_m: out.diag3[3],
        adhesion_fraction: out.diag4[0],
        sliding_fraction: out.diag4[1],
        friction_budget_n: out.diag4[2],
        consumed_friction_n: out.diag4[3],
        effective_longitudinal_stiffness: out.diag5[0],
        effective_cornering_stiffness: out.diag5[1],
        debug_cells: [TireDebugContactCell::default(); TRACTION_DEBUG_CELLS],
        debug_cell_count: 0,
    }
}

fn eval_brush_pair_gpu(
    kernel: &molla::kernels::TireBrushGpuKernel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    params: &molla_vehicle::RealTireParams,
    rear: &molla_vehicle::RealWheelState,
    front: &molla_vehicle::RealWheelState,
    material: &molla_vehicle::SurfaceMaterial,
    camber_rad: f32,
) -> Result<[BrushEval; 2], String> {
    let inputs = [
        pack_brush_gpu_input(params, rear, material, camber_rad),
        pack_brush_gpu_input(params, front, material, camber_rad),
    ];
    let out = kernel
        .dispatch_blocking(device, queue, &inputs, None)
        .map_err(|err| format!("GPU brush dispatch failed: {err}"))?;
    if out.len() != 2 {
        return Err(format!("GPU brush returned {} rows, expected 2", out.len()));
    }
    Ok([
        brush_eval_from_gpu(out[0], rear.heading),
        brush_eval_from_gpu(out[1], front.heading),
    ])
}

fn vehicle_brush_rows_input(
    params: &molla_vehicle::RealTireParams,
    rear: &molla_vehicle::RealWheelState,
    front: &molla_vehicle::RealWheelState,
    material: &molla_vehicle::SurfaceMaterial,
    camber_rad: f32,
) -> molla::kernels::TireVehicleBrushRowsGpuInput {
    let yaw = rear.heading[2].atan2(rear.heading[0]) as f32;
    let front_yaw = front.heading[2].atan2(front.heading[0]) as f32;
    let mut steer = front_yaw - yaw;
    while steer > std::f32::consts::PI {
        steer -= std::f32::consts::TAU;
    }
    while steer < -std::f32::consts::PI {
        steer += std::f32::consts::TAU;
    }
    let travel = (params.radius as f32 - rear.hub_pos[1] as f32).max(0.0);
    let tire_row = [
        params.segments as f32,
        params.carcass_normal_stiffness as f32,
        params.tread_shear_stiffness as f32,
        params.relaxation_speed as f32,
    ];
    let material_row = [
        material.mu_static as f32,
        material.mu_dynamic as f32,
        material.bristle_damping as f32,
        material.rolling_resistance as f32,
    ];
    molla::kernels::TireVehicleBrushRowsGpuInput {
        chassis_pose: [
            rear.hub_pos[0] as f32,
            rear.hub_pos[1] as f32,
            rear.hub_pos[2] as f32,
            yaw,
        ],
        chassis_velocity: [
            rear.hub_vel[0] as f32,
            rear.hub_vel[1] as f32,
            rear.hub_vel[2] as f32,
            0.0,
        ],
        wheel_mounts: vec![[0.0, 0.0, 0.0, params.radius as f32]; 2],
        wheel_state: vec![
            [
                rear.omega as f32,
                0.0,
                params.width as f32,
                params.pressure_pa as f32,
            ],
            [
                front.omega as f32,
                steer,
                params.width as f32,
                params.pressure_pa as f32,
            ],
        ],
        tire_rows: vec![tire_row; 2],
        material_rows: vec![material_row; 2],
        extra_rows: vec![[material.cohesion as f32, 0.0, 1.0, camber_rad]; 2],
        suspension_rows: vec![[travel, 0.0, 0.0, 1.0]; 2],
    }
}

fn eval_brush_pair_gpu_vehicle_handoff(
    vehicle_kernel: &molla::kernels::TireVehicleYawGpuKernel,
    brush_kernel: &molla::kernels::TireBrushGpuKernel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    params: &molla_vehicle::RealTireParams,
    rear: &molla_vehicle::RealWheelState,
    front: &molla_vehicle::RealWheelState,
    material: &molla_vehicle::SurfaceMaterial,
    camber_rad: f32,
) -> Result<[BrushEval; 2], String> {
    let input = vehicle_brush_rows_input(params, rear, front, material, camber_rad);
    let brush_rows = vehicle_kernel
        .dispatch_brush_rows_resident(device, queue, &input)
        .map_err(|err| format!("GPU vehicle→brush row dispatch failed: {err}"))?;
    let resident = brush_kernel
        .dispatch_input_buffer_resident(device, queue, &brush_rows.buffer, brush_rows.wheel_count, None)
        .map_err(|err| format!("GPU resident vehicle→brush dispatch failed: {err}"))?;
    let out = brush_kernel
        .readback_resident_output(device, queue, &resident)
        .map_err(|err| format!("GPU resident vehicle→brush readback failed: {err}"))?;
    if out.len() != 2 {
        return Err(format!(
            "GPU resident vehicle→brush returned {} rows, expected 2",
            out.len()
        ));
    }
    Ok([
        brush_eval_from_gpu(out[0], rear.heading),
        brush_eval_from_gpu(out[1], front.heading),
    ])
}

#[derive(Resource)]
struct TireTractionState {
    cars: [TractionCar; 3],
    use_gpu: bool,
    gpu_enabled_now: bool,
    brush_kernel: Option<molla::kernels::TireBrushGpuKernel>,
    vehicle_kernel: Option<molla::kernels::TireVehicleYawGpuKernel>,
    transient_kernel: Option<molla::kernels::TireTransientGpuKernel>,
    transient_batch: Option<molla::kernels::TireTransientGpuBatch>,
    gpu_error: Option<String>,
    user_pressure_kpa: f32,
    user_weight_kn: f32,
    user_material_tension: f32,
    /// Stage (i) ext — steering input from the right rail. The user
    /// car reads this each frame; the two reference cars hold zero.
    user_steer: f32,
    user_camber_deg: f32,
    surface_slipperiness: f32,
    show_debug_overlays: bool,
    elapsed: f32,
    reset_requested: bool,
    replay_panel: TireRigReplayPanelState,
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
            use_gpu: true,
            gpu_enabled_now: false,
            brush_kernel: None,
            vehicle_kernel: None,
            transient_kernel: None,
            transient_batch: None,
            gpu_error: None,
            user_pressure_kpa: 160.0,
            user_weight_kn: TRACTION_DEFAULT_WEIGHT_N / 1000.0,
            user_material_tension: 1.0,
            user_steer: 0.0,
            user_camber_deg: 0.0,
            surface_slipperiness: 0.88,
            show_debug_overlays: true,
            elapsed: 0.0,
            reset_requested: false,
            replay_panel: Default::default(),
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
    _time: Res<Time>,
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
    state.cars[2].steer_input = state.user_steer.clamp(-1.0, 1.0);
    state.cars[2].camber_rad = state.user_camber_deg.to_radians().clamp(-0.35, 0.35);
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
    let device_queue = if state.use_gpu {
        match (sim.engine.device(), sim.engine.queue()) {
            (Some(device), Some(queue)) => {
                if state.brush_kernel.is_none() {
                    state.brush_kernel = Some(molla::kernels::TireBrushGpuKernel::new(device));
                }
                if state.vehicle_kernel.is_none() {
                    state.vehicle_kernel = Some(molla::kernels::TireVehicleYawGpuKernel::new(device));
                }
                if state.transient_kernel.is_none() {
                    state.transient_kernel =
                        Some(molla::kernels::TireTransientGpuKernel::new(device));
                }
                if state.transient_batch.is_none()
                    && let Some(kernel) = state.transient_kernel.as_ref() {
                        match kernel.create_batch(device, 16) {
                            Ok(batch) => {
                                state.transient_batch = Some(batch);
                            }
                            Err(err) => {
                                state.gpu_error =
                                    Some(format!("GPU transient batch allocation failed: {err}"));
                            }
                        }
                    }
                Some((device.clone(), queue.clone()))
            }
            _ => {
                state.gpu_error = Some("GPU brush requested but engine has no GPU device".into());
                None
            }
        }
    } else {
        None
    };
    // Fixed physics dt: render-frame dt made the demo frame-rate-
    // dependent (2026-07-05 tire audit; scene 120 already did this).
    let dt = 1.0 / 120.0;
    state.elapsed += dt;
    state.gpu_enabled_now = false;
    state.gpu_error = None;
    let mut used_any_gpu = false;
    let mut last_gpu_error = None;
    let brush_kernel = state.brush_kernel.take();
    let vehicle_kernel = state.vehicle_kernel.take();
    let gpu_refs = match (&brush_kernel, &vehicle_kernel, &device_queue) {
        (Some(brush), Some(vehicle), Some((device, queue))) if state.use_gpu => {
            Some((brush, vehicle, device.as_ref(), queue.as_ref()))
        }
        _ => None,
    };
    let transient_kernel = state.transient_kernel.take();
    let transient_batch = state.transient_batch.take();
    let transient_refs = match (&transient_kernel, &transient_batch, &device_queue) {
        (Some(kernel), Some(batch), Some((device, queue))) if state.use_gpu => {
            Some((kernel, batch, device.as_ref(), queue.as_ref()))
        }
        _ => None,
    };
    for car in &mut state.cars {
        let (used_gpu, gpu_error) = car.step(dt, material, gpu_refs, transient_refs);
        used_any_gpu |= used_gpu;
        if gpu_error.is_some() {
            last_gpu_error = gpu_error;
        }
    }
    state.gpu_enabled_now = used_any_gpu;
    state.gpu_error = last_gpu_error;
    state.brush_kernel = brush_kernel;
    state.vehicle_kernel = vehicle_kernel;
    state.transient_kernel = transient_kernel;
    state.transient_batch = transient_batch;
}

fn draw_traction_wheel(
    gizmos: &mut Gizmos,
    centre: Vec3,
    pressure_pa: f32,
    angle: f32,
    ring_color: Color,
    contact_color: Color,
) {
    draw_traction_wheel_oriented(
        gizmos,
        centre,
        pressure_pa,
        angle,
        0.0,
        ring_color,
        contact_color,
    );
}

pub(crate) fn draw_traction_wheel_oriented(
    gizmos: &mut Gizmos,
    centre: Vec3,
    pressure_pa: f32,
    angle: f32,
    yaw_rad: f32,
    ring_color: Color,
    contact_color: Color,
) {
    draw_traction_wheel_oriented_camber(
        gizmos,
        centre,
        pressure_pa,
        angle,
        yaw_rad,
        0.0,
        ring_color,
        contact_color,
    );
}

fn draw_traction_wheel_oriented_camber(
    gizmos: &mut Gizmos,
    centre: Vec3,
    pressure_pa: f32,
    angle: f32,
    yaw_rad: f32,
    camber_rad: f32,
    ring_color: Color,
    contact_color: Color,
) {
    let q = (pressure_pa / REFERENCE_PRESSURE_PA as f32).clamp(0.0, 1.0);
    let half_w = TRACTION_WHEEL_WIDTH * 0.5;
    let segments = TIRE_SEGMENTS;
    let fwd = Vec3::new(yaw_rad.cos(), 0.0, yaw_rad.sin());
    let base_side = Vec3::new(-yaw_rad.sin(), 0.0, yaw_rad.cos());
    let up = Vec3::Y * camber_rad.cos() + base_side * camber_rad.sin();
    let side = base_side * camber_rad.cos() - Vec3::Y * camber_rad.sin();
    let mut left = Vec::with_capacity(segments);
    let mut right = Vec::with_capacity(segments);
    for i in 0..segments {
        let theta = angle + (i as f32 / segments as f32) * std::f32::consts::TAU;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let bottom_weight = (-sin_t).max(0.0).powf(1.7);
        let side_bulge = (1.0 - q) * 0.11 * bottom_weight;
        let x = (TRACTION_WHEEL_RADIUS + side_bulge) * cos_t;
        let p_mid_raw = centre + fwd * x + up * (TRACTION_WHEEL_RADIUS * sin_t);
        let p_mid = Vec3::new(p_mid_raw.x, p_mid_raw.y.max(0.0), p_mid_raw.z);
        let mut p_left = p_mid - side * half_w;
        let mut p_right = p_mid + side * half_w;
        p_left.y = p_left.y.max(0.0);
        p_right.y = p_right.y.max(0.0);
        left.push(p_left);
        right.push(p_right);
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

    // Bright rolling-direction line: front wheels now visibly toe left/right.
    gizmos.line(
        centre - fwd * TRACTION_WHEEL_RADIUS * 0.70,
        centre + fwd * TRACTION_WHEEL_RADIUS * 0.90,
        Color::srgb(1.0, 0.95, 0.30),
    );
}

fn draw_oriented_box_edges(
    gizmos: &mut Gizmos,
    center: Vec3,
    half: Vec3,
    yaw_rad: f32,
    color: Color,
) {
    let fwd = Vec3::new(yaw_rad.cos(), 0.0, yaw_rad.sin());
    let side = Vec3::new(-yaw_rad.sin(), 0.0, yaw_rad.cos());
    let corner = |sx: f32, sy: f32, sz: f32| -> Vec3 {
        center + fwd * (sx * half.x) + Vec3::Y * (sy * half.y) + side * (sz * half.z)
    };
    let corners = [
        corner(-1.0, -1.0, -1.0),
        corner(1.0, -1.0, -1.0),
        corner(1.0, 1.0, -1.0),
        corner(-1.0, 1.0, -1.0),
        corner(-1.0, -1.0, 1.0),
        corner(1.0, -1.0, 1.0),
        corner(1.0, 1.0, 1.0),
        corner(-1.0, 1.0, 1.0),
    ];
    for (a, b) in [
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
    ] {
        gizmos.line(corners[a], corners[b], color);
    }
}

fn draw_traction_car(gizmos: &mut Gizmos, car: TractionCar, color: Color) {
    let hub_y = car.hub_y();
    let chassis_center = Vec3::new(car.x, hub_y + 0.50, car.z);
    let half = Vec3::new(0.82, 0.22, 0.42);
    // Stage (i) ext — rotate chassis around the heading axis.
    let cos_h = car.heading_rad.cos();
    let sin_h = car.heading_rad.sin();
    let rotate_xz = |dx: f32, dz: f32| -> Vec3 {
        Vec3::new(cos_h * dx - sin_h * dz, 0.0, sin_h * dx + cos_h * dz)
    };
    draw_oriented_box_edges(gizmos, chassis_center, half, car.heading_rad, color);
    // Heavy payload block sitting above the suspension.
    let payload_half = Vec3::new(0.50, 0.24, 0.30);
    let payload_center = chassis_center + Vec3::new(0.0, 0.45, 0.0);
    draw_oriented_box_edges(
        gizmos,
        payload_center,
        payload_half,
        car.heading_rad,
        Color::srgb(0.55, 0.50, 0.45),
    );

    let wheel_x = [-0.48, 0.48];
    let wheel_z = [-0.30, 0.30];
    for &wx in &wheel_x {
        for &wz in &wheel_z {
            let offset = rotate_xz(wx, wz);
            let centre = Vec3::new(car.x + offset.x, hub_y, car.z + offset.z);
            let front_steer = if wx > 0.0 { car.steer_angle_rad() } else { 0.0 };
            draw_traction_wheel_oriented_camber(
                gizmos,
                centre,
                car.pressure_pa,
                car.wheel_angle,
                car.heading_rad + front_steer,
                car.camber_rad,
                color,
                Color::srgb(1.0, 0.46, 0.18),
            );
            gizmos.line(
                centre,
                chassis_center + Vec3::new(offset.x, -0.10, offset.z),
                Color::srgb(0.45, 0.45, 0.50),
            );
        }
    }

    // Heading arrow — short ray from the chassis centre along the
    // current heading direction so cornering is visible at a glance.
    let arrow_tip = chassis_center + rotate_xz(0.95, 0.0);
    gizmos.line(chassis_center, arrow_tip, Color::srgb(1.0, 0.85, 0.25));

    // Motion trail and traction/slip bar. The trail follows the actual
    // (x, z) trajectory so cornering shows up here too.
    gizmos.line(
        Vec3::new(-2.4, 0.02, car.lane_z),
        Vec3::new(car.x, 0.02, car.z),
        color,
    );
    let bar_base = Vec3::new(-3.1, 0.25, car.lane_z);
    gizmos.line(
        bar_base,
        bar_base + Vec3::new(car.visual_patch_factor() * 0.7, 0.0, 0.0),
        Color::srgb(0.25, 0.90, 0.35),
    );
    gizmos.line(
        bar_base + Vec3::new(0.0, 0.08, 0.0),
        bar_base + Vec3::new(car.slip * 0.7, 0.08, 0.0),
        Color::srgb(1.0, 0.22, 0.12),
    );
}

fn draw_yaw_moment_glyph(
    gizmos: &mut Gizmos,
    center: Vec3,
    fwd: Vec3,
    side: Vec3,
    moment_nm: f32,
    color: Color,
) {
    let magnitude = moment_nm.abs();
    if magnitude < 1.0 {
        return;
    }
    let sign = if moment_nm >= 0.0 { 1.0 } else { -1.0 };
    let radius = (0.11 + (magnitude / 8_000.0).sqrt() * 0.18).clamp(0.11, 0.42);
    let sweep = std::f32::consts::TAU * 0.72 * sign;
    let steps = 24;
    let mut prev = center + fwd * radius;
    for i in 1..=steps {
        let a = sweep * (i as f32 / steps as f32);
        let p = center + fwd * (radius * a.cos()) + side * (radius * a.sin());
        gizmos.line(prev, p, color);
        prev = p;
    }

    let tangent = (-fwd * sweep.sin() + side * sweep.cos()) * sign;
    let radial = fwd * sweep.cos() + side * sweep.sin();
    let head = prev;
    gizmos.line(head - tangent * 0.16, head, color);
    gizmos.line(head, head - tangent * 0.08 + radial * 0.055, color);
    gizmos.line(head, head - tangent * 0.08 - radial * 0.055, color);
}

fn draw_tire_debug_overlays(gizmos: &mut Gizmos, car: TractionCar) {
    let hub_y = car.hub_y();
    let cos_h = car.heading_rad.cos();
    let sin_h = car.heading_rad.sin();
    let fwd = Vec3::new(cos_h, 0.0, sin_h);
    let side = Vec3::new(-sin_h, 0.0, cos_h);
    let chassis = Vec3::new(car.x, hub_y + 0.84, car.z);

    // Wheel-level force arrow from reusable tire telemetry/readouts.
    let force = Vec3::new(car.last_force_x_n, 0.0, car.last_force_z_n);
    let force_scale = 1.0 / 18_000.0;
    let force_tip = chassis + force * force_scale;
    gizmos.line(chassis, force_tip, Color::srgb(1.0, 0.20, 0.15));
    gizmos.sphere(force_tip, 0.045, Color::srgb(1.0, 0.20, 0.15));

    // All-wheel contact overlays. The physics evaluates one wheel per axle and
    // mirrors it to the left/right tire, matching the 2× force accounting used
    // by the compact vehicle rig. This makes every visible tire show contact
    // cells, patch bounds, force vectors, pneumatic trail, and per-contact
    // moment contribution instead of only debugging the user-car front tire.
    for wheel in car.last_debug_wheels {
        let wheel_heading = car.heading_rad + wheel.steer_angle_rad;
        let wfwd = Vec3::new(wheel_heading.cos(), 0.0, wheel_heading.sin());
        let wside = Vec3::new(-wheel_heading.sin(), 0.0, wheel_heading.cos());
        for &track_offset in &[-0.30_f32, 0.30_f32] {
            let axle_center =
                Vec3::new(car.x, 0.055, car.z) + fwd * wheel.axle_x + side * track_offset;
            let contact_center = axle_center + wfwd * wheel.contact_center_x;
            let resultant_center = contact_center - wfwd * wheel.pneumatic_trail_m;
            gizmos.sphere(contact_center, 0.030, Color::srgb(0.20, 1.0, 0.35));
            gizmos.sphere(resultant_center, 0.024, Color::srgb(1.0, 0.95, 0.15));
            gizmos.line(contact_center, resultant_center, Color::srgb(1.0, 0.95, 0.15));

            let half_patch = wheel.patch_length.max(0.0) * 0.5;
            let left = axle_center - wfwd * half_patch;
            let right = axle_center + wfwd * half_patch;
            gizmos.line(left, right, Color::srgb(0.35, 1.0, 0.95));
            for marker in [left, right, contact_center] {
                gizmos.line(
                    marker,
                    marker + Vec3::Y * 0.18,
                    Color::srgb(0.35, 1.0, 0.95),
                );
            }

            let wheel_force = Vec3::new(wheel.force_x_n, 0.0, wheel.force_z_n);
            let force_base = axle_center + Vec3::Y * 0.28;
            let force_tip = force_base + wheel_force * (1.0 / 9_000.0);
            gizmos.line(force_base, force_tip, Color::srgb(1.0, 0.20, 0.15));
            gizmos.sphere(force_tip, 0.024, Color::srgb(1.0, 0.20, 0.15));
            let lateral_tip = force_base + wside * (wheel.lateral_force_n / 9_000.0);
            gizmos.line(force_base, lateral_tip, Color::srgb(0.45, 0.75, 1.0));

            draw_yaw_moment_glyph(
                gizmos,
                axle_center + Vec3::Y * 0.35,
                wfwd,
                wside,
                wheel.aligning_moment_nm,
                Color::srgb(1.0, 0.30, 0.95),
            );

            // Per-contact bristle shear debug from reusable brush contact
            // cells. Green points are adhesive cells, red/orange points are
            // sliding cells. The small yellow cross-bars are each cell's yaw
            // moment contribution (`-local_x * Fy`), so a bad moment sign is
            // visible directly in the contact patch.
            for (i, cell) in wheel
                .debug_cells
                .iter()
                .take(wheel.debug_cell_count.min(TRACTION_DEBUG_CELLS))
                .enumerate()
            {
                let width_offset = ((i % 3) as f32 - 1.0) * 0.030;
                let base = axle_center + wfwd * cell.local_x + wside * width_offset + Vec3::Y * 0.08;
                let color = if cell.sliding {
                    Color::srgb(1.0, 0.28, 0.08)
                } else {
                    Color::srgb(0.20, 1.0, 0.32)
                };
                let load_radius =
                    (0.010 + (cell.normal_load / 3_000.0).sqrt() * 0.016).clamp(0.010, 0.040);
                gizmos.sphere(base, load_radius, color);

                let shear_force = wfwd * cell.tangential_force + wside * cell.lateral_force;
                let shear_len = shear_force.length();
                if shear_len > 1.0 {
                    let shear_tip =
                        base + shear_force / shear_len * (shear_len / 4_500.0).clamp(0.030, 0.20);
                    gizmos.line(base, shear_tip, color);
                    gizmos.sphere(shear_tip, 0.010, color);
                }

                let normal = Vec3::new(cell.normal[0], cell.normal[1], cell.normal[2]);
                if normal.length_squared() > 1.0e-6 {
                    gizmos.line(
                        base,
                        base + normal.normalize() * 0.11,
                        Color::srgb(0.35, 1.0, 0.95),
                    );
                }

                let cell_mz = -cell.local_x * cell.lateral_force;
                if cell_mz.abs() > 2.0 {
                    let sign = if cell_mz >= 0.0 { 1.0 } else { -1.0 };
                    let len = (cell_mz.abs() / 1_500.0).clamp(0.025, 0.12);
                    let moment_center = base + Vec3::Y * 0.12;
                    let tangent = wside * sign;
                    gizmos.line(
                        moment_center - tangent * len * 0.5,
                        moment_center + tangent * len * 0.5,
                        Color::srgb(1.0, 0.95, 0.15),
                    );
                }
            }
        }
    }

    // Lateral force component, separated from total force for steering debug.
    let lateral_tip = chassis + side * (car.last_lateral_n / 18_000.0);
    gizmos.line(chassis, lateral_tip, Color::srgb(0.45, 0.75, 1.0));

    // Orange is the total chassis yaw moment from axle force arms. Per-wheel
    // magenta aligning-moment glyphs are drawn at each tire above.
    draw_yaw_moment_glyph(
        gizmos,
        chassis + Vec3::Y * 0.16,
        fwd,
        side,
        car.last_chassis_yaw_moment_nm,
        Color::srgb(1.0, 0.62, 0.18),
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
    if state.show_debug_overlays {
        for car in state.cars {
            draw_tire_debug_overlays(&mut gizmos, car);
        }
    }
}

fn tire_traction_pane(ui: &mut egui::Ui, state: &mut TireTractionState, accent: egui::Color32) {
    ui.label("User car controls");
    ui.separator();
    let was_gpu = state.use_gpu;
    let _ = ui.checkbox(&mut state.use_gpu, "Use GPU transient + brush tire contact");
    if was_gpu != state.use_gpu {
        state.gpu_error = None;
        state.gpu_enabled_now = false;
    }
    let backend = if state.gpu_enabled_now {
        "GPU transient + brush"
    } else if state.use_gpu {
        "CPU fallback"
    } else {
        "CPU brush"
    };
    readout(ui, "Contact backend", backend);
    if let Some(err) = &state.gpu_error {
        ui.small(format!("GPU fallback: {err}"));
    }
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
    ui.label("Steering (user car)");
    let _ = ui.add(
        egui::Slider::new(&mut state.user_steer, -1.0..=1.0).text("left ↔ right"),
    );
    ui.label("Camber (user car)");
    let _ = ui.add(egui::Slider::new(&mut state.user_camber_deg, -20.0..=20.0).suffix("°"));
    ui.checkbox(
        &mut state.show_debug_overlays,
        "Show all tire debug overlays",
    );
    let material = traction_surface_material(state.surface_slipperiness);
    readout(ui, "Surface μ static", &format!("{:.3}", material.mu_static));
    readout(ui, "Surface μ dynamic", &format!("{:.3}", material.mu_dynamic));
    ui.separator();
    if button(ui, "Reset run", accent).clicked() {
        state.reset_requested = true;
    }
    if let Some(car) = state.cars.get(2) {
        readout(ui, "User speed", &format!("{:.2} m/s", car.v));
        readout(ui, "Lateral vel", &format!("{:.2} m/s", car.v_lat));
        readout(ui, "User slip", &format!("{:.0}%", car.slip * 100.0));
        readout(
            ui,
            "Slip angle",
            &format!("{:.1}°", car.last_slip_angle.to_degrees()),
        );
        readout(ui, "Heading", &format!("{:.0}°", car.heading_rad.to_degrees()));
        readout(
            ui,
            "Steer angle",
            &format!("{:.0}°", car.steer_angle_rad().to_degrees()),
        );
        readout(ui, "Yaw rate", &format!("{:.2} rad/s", car.yaw_rate));
        readout(
            ui,
            "Align Mz",
            &format!("{:.1} N·m", car.last_aligning_moment_nm),
        );
        readout(
            ui,
            "Chassis Mz",
            &format!("{:.1} N·m", car.last_chassis_yaw_moment_nm),
        );
        readout(
            ui,
            "Raw slip α",
            &format!("{:.1}°", car.last_raw_slip_angle.to_degrees()),
        );
        readout(
            ui,
            "Effective α",
            &format!("{:.1}°", car.last_effective_slip_angle.to_degrees()),
        );
        readout(ui, "Raw κ", &format!("{:.2}", car.last_raw_slip_ratio));
        readout(
            ui,
            "Effective κ",
            &format!("{:.2}", car.last_effective_slip_ratio),
        );
        readout(
            ui,
            "Camber",
            &format!("{:.1}°", car.last_camber_rad.to_degrees()),
        );
        readout(
            ui,
            "Pneum. trail",
            &format!("{:.1} mm", car.last_pneumatic_trail_m * 1000.0),
        );
        readout(
            ui,
            "Contact centre",
            &format!("{:.1} mm", car.last_contact_center_x * 1000.0),
        );
        readout(
            ui,
            "Visual patch factor",
            &format!("{:.2}×", car.visual_patch_factor()),
        );
        readout(ui, "Contacts", &format!("{}", car.last_contacts));
        readout(ui, "Sliding cells", &format!("{}", car.last_sliding));
        readout(
            ui,
            "Sliding frac",
            &format!("{:.0}%", car.last_sliding_fraction * 100.0),
        );
        readout(
            ui,
            "Friction budget",
            &format!(
                "{:.1}/{:.1} kN",
                car.last_consumed_friction_n / 1000.0,
                car.last_friction_budget_n / 1000.0
            ),
        );
        readout(
            ui,
            "Traction",
            &format!("{:.1} kN", car.last_traction_n / 1000.0),
        );
        readout(
            ui,
            "World Fx/Fz",
            &format!(
                "{:.1}/{:.1} kN",
                car.last_force_x_n / 1000.0,
                car.last_force_z_n / 1000.0
            ),
        );
        readout(
            ui,
            "Lateral force",
            &format!("{:.1} kN", car.last_lateral_n / 1000.0),
        );
        readout(
            ui,
            "Tread temp",
            &format!("{:.1} °C", car.thermal.temperature_c),
        );
        readout(
            ui,
            "Tread wear",
            &format!("{:.2} mm", car.wear.depth_m * 1000.0),
        );
        readout(ui, "μ heat/wear", &format!("{:.2}×", car.last_mu_factor));
        readout(
            ui,
            "Dissipation",
            &format!("{:.0} W", car.last_dissipated_w),
        );
        readout(
            ui,
            "Kx / Ca",
            &format!(
                "{:.0} kN/m  {:.0} kN/rad",
                car.last_longitudinal_stiffness / 1000.0,
                car.last_cornering_stiffness / 1000.0
            ),
        );
        if car.last_gpu_delta_valid {
            readout(
                ui,
                "GPU Δ force",
                &format!("{:.1} N", car.last_gpu_delta_force_n),
            );
            readout(
                ui,
                "GPU Δ patch",
                &format!("{:.2} mm", car.last_gpu_delta_patch_m * 1000.0),
            );
            readout(ui, "GPU Δ slip", &format!("{:.4}", car.last_gpu_delta_slip));
            readout(
                ui,
                "GPU Δ budget",
                &format!("{:.1} N", car.last_gpu_delta_budget_n),
            );
            readout(
                ui,
                "GPU Δ Mz",
                &format!("{:.2} N·m", car.last_gpu_delta_moment_nm),
            );
        } else if state.use_gpu {
            readout(ui, "GPU Δ", "not available (CPU fallback)");
        }
        if car.last_vehicle_handoff_wheels > 0 {
            readout(
                ui,
                "Vehicle→brush rows",
                &format!(
                    "{} wheels, Fx {:.0} N, Δ {:.1} N",
                    car.last_vehicle_handoff_wheels,
                    car.last_vehicle_handoff_force_n,
                    car.last_vehicle_handoff_delta_n
                ),
            );
        } else if state.use_gpu {
            readout(ui, "Vehicle→brush rows", "not available");
        }
    }
    ui.separator();
    tire_rig_replay_panel(ui, &mut state.replay_panel);
    ui.separator();
    ui.small("Red: fixed high pressure, slips.");
    ui.small("Blue: fixed partial pressure, moves.");
    ui.small("Green: pressure + weight + tire material tension controlled here.");
    ui.small("Push slipperiness toward ice to force visible wheel spin.");
    ui.small("Drag steering: transient tire state lags raw slip before yaw builds.");
    ui.small("Camber adds real brush camber thrust + pneumatic-trail diagnostics.");
    ui.small("Load sensitivity + smooth combined-slip saturation come from the reusable brush law.");
    ui.small("Heat/wear now feed back into μ; high slip raises dissipation.");
}
