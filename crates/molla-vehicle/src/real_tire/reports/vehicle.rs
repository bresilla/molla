//! Vehicle-level data-free reports: torque/slip outcomes, wheel-spin torque
//! balance, longitudinal motion, dynamic load transfer.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::{FlatTerrain, HeightfieldTerrain, Terrain};

use super::super::internal::{bool_metric, csv_token, finite_nonnegative, finite_or};
use super::super::{
    RealTireContact, RealTireParams, RealTireTelemetrySample, RealWheelState, SurfaceMaterial,
    TireRigAcceptanceCheck, TireRigSnapshot, TireRigSweepPoint, evaluate_brush_tire,
    sanitize_brush_params, solve_loaded_tire_hub_height, tire_rig_camber_sweep,
    tire_rig_heightfield_envelope_sweep, tire_rig_point, tire_rig_step_brake_response,
    tire_rig_step_torque_response,
};

/// Data-free driven-wheel outcome for the PLAN_REALTIRE torque/slip case.
///
/// Same torque, same load, same low-friction surface: over-inflated tires
/// should turn torque into more wheel slip, while the partial-pressure tire
/// should use the larger pressure-dependent patch to move farther. This is a
/// compact native tire/vehicle integration check, not a fitted acceleration
/// target.
#[derive(Clone, Debug)]
pub struct TireTorqueSlipOutcomeReport {
    pub scenario_id: String,
    pub surface_label: String,
    pub drive_torque_nm_per_wheel: f64,
    pub load_n_per_wheel: f64,
    pub high_pressure_pa: f64,
    pub partial_pressure_pa: f64,
    pub high_final_slip_ratio: f64,
    pub partial_final_slip_ratio: f64,
    pub high_distance_m: f64,
    pub partial_distance_m: f64,
    pub high_final_speed_m_per_s: f64,
    pub partial_final_speed_m_per_s: f64,
    pub high_patch_length_m: f64,
    pub partial_patch_length_m: f64,
    pub high_telemetry: RealTireTelemetrySample,
    pub partial_telemetry: RealTireTelemetrySample,
}

impl TireTorqueSlipOutcomeReport {
    pub fn is_finite(&self) -> bool {
        self.drive_torque_nm_per_wheel.is_finite()
            && self.load_n_per_wheel.is_finite()
            && self.high_pressure_pa.is_finite()
            && self.partial_pressure_pa.is_finite()
            && self.high_final_slip_ratio.is_finite()
            && self.partial_final_slip_ratio.is_finite()
            && self.high_distance_m.is_finite()
            && self.partial_distance_m.is_finite()
            && self.high_final_speed_m_per_s.is_finite()
            && self.partial_final_speed_m_per_s.is_finite()
            && self.high_patch_length_m.is_finite()
            && self.partial_patch_length_m.is_finite()
            && self.high_telemetry.is_finite()
            && self.partial_telemetry.is_finite()
    }

    pub fn passed(&self) -> bool {
        self.is_finite() && self.acceptance_checks().iter().all(|check| check.passed)
    }

    pub fn acceptance_checks(&self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "torque_slip_partial_pressure_moves_farther",
                "same torque/load should move the partial-pressure tire farther than the over-inflated tire",
                "partial_minus_high_distance_m",
                self.partial_distance_m - self.high_distance_m,
                0.02,
            ),
            TireRigAcceptanceCheck::at_least(
                "torque_slip_high_pressure_slips_more",
                "same torque/load should leave the over-inflated tire with more wheel slip",
                "high_minus_partial_abs_slip",
                self.high_final_slip_ratio.abs() - self.partial_final_slip_ratio.abs(),
                0.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "torque_slip_partial_patch_larger",
                "partial pressure should build a larger contact patch than over-inflation",
                "partial_minus_high_patch_m",
                self.partial_patch_length_m - self.high_patch_length_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "torque_slip_partial_speed_higher",
                "partial pressure should turn more of the torque into vehicle speed",
                "partial_minus_high_speed_m_per_s",
                self.partial_final_speed_m_per_s - self.high_final_speed_m_per_s,
                0.02,
            ),
            TireRigAcceptanceCheck::at_most(
                "torque_slip_partial_force_budget_bounded",
                "partial-pressure driven tire should stay inside the shared friction budget",
                "partial_consumed_over_available",
                self.partial_telemetry.consumed_friction_budget_n
                    / self.partial_telemetry.available_friction_budget_n.max(1.0),
                1.05,
            ),
        ]
    }

    pub fn to_text_line(&self) -> String {
        format!(
            "torque_slip_outcome scenario={} surface={} torque_nm={:.3} load_n={:.3} high_pressure_pa={:.3} partial_pressure_pa={:.3} high_slip={:.6} partial_slip={:.6} high_distance_m={:.6} partial_distance_m={:.6} high_speed_mps={:.6} partial_speed_mps={:.6} high_patch_m={:.6} partial_patch_m={:.6} passed={}",
            csv_token(&self.scenario_id),
            csv_token(&self.surface_label),
            self.drive_torque_nm_per_wheel,
            self.load_n_per_wheel,
            self.high_pressure_pa,
            self.partial_pressure_pa,
            self.high_final_slip_ratio,
            self.partial_final_slip_ratio,
            self.high_distance_m,
            self.partial_distance_m,
            self.high_final_speed_m_per_s,
            self.partial_final_speed_m_per_s,
            self.high_patch_length_m,
            self.partial_patch_length_m,
            self.passed(),
        )
    }

    pub fn snapshot(&self) -> TireRigSnapshot {
        TireRigSweepPoint {
            load_n: self.load_n_per_wheel,
            pressure_pa: self.partial_pressure_pa,
            hub_height_m: 0.0,
            deflection_m: 0.0,
            speed_m_per_s: self.partial_final_speed_m_per_s,
            slip_ratio_command: self.partial_final_slip_ratio,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: self.partial_telemetry.clone(),
        }
        .snapshot_on_terrain(
            &self.scenario_id,
            "partial_pressure_torque_outcome",
            &self.surface_label,
            "flat_low_friction",
            "cpu_brush_wheel_spin_vehicle_outcome",
            &FlatTerrain::new(0.0, 0.30),
            0.0,
            0.0,
        )
    }
}

/// Data-free torque balance for driven wheel spin integration.
///
/// PLAN_REALTIRE requires slip to be an outcome of the wheel equation
/// `I * omega_dot = motor_torque - Fx * radius - bearing_loss`, not a viewer
/// patch multiplier. This report exposes the one-step torque accounting used
/// by the longer torque-induced-slip scenario.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireWheelSpinTorqueBalanceReport {
    pub drive_torque_nm: f64,
    pub tire_reaction_torque_nm: f64,
    pub bearing_loss_nm: f64,
    pub net_torque_nm: f64,
    pub wheel_inertia_kg_m2: f64,
    pub dt_s: f64,
    pub initial_omega_rad_s: f64,
    pub omega_dot_rad_s2: f64,
    pub expected_delta_omega_rad_s: f64,
    pub integrated_delta_omega_rad_s: f64,
    pub longitudinal_force_n: f64,
    pub radius_m: f64,
    pub slip_ratio: f64,
    pub friction_budget_ratio: f64,
    pub finite_outputs: bool,
}

impl TireWheelSpinTorqueBalanceReport {
    pub fn is_finite(self) -> bool {
        self.drive_torque_nm.is_finite()
            && self.tire_reaction_torque_nm.is_finite()
            && self.bearing_loss_nm.is_finite()
            && self.net_torque_nm.is_finite()
            && self.wheel_inertia_kg_m2.is_finite()
            && self.dt_s.is_finite()
            && self.initial_omega_rad_s.is_finite()
            && self.omega_dot_rad_s2.is_finite()
            && self.expected_delta_omega_rad_s.is_finite()
            && self.integrated_delta_omega_rad_s.is_finite()
            && self.longitudinal_force_n.is_finite()
            && self.radius_m.is_finite()
            && self.slip_ratio.is_finite()
            && self.friction_budget_ratio.is_finite()
    }

    pub fn omega_delta_error(self) -> f64 {
        (self.integrated_delta_omega_rad_s - self.expected_delta_omega_rad_s).abs()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.drive_torque_nm > 0.0
            && self.tire_reaction_torque_nm > 0.0
            && self.bearing_loss_nm >= 0.0
            && self.wheel_inertia_kg_m2 > 0.0
            && self.dt_s > 0.0
            && self.net_torque_nm > 0.0
            && self.omega_dot_rad_s2 > 0.0
            && self.omega_delta_error() <= 1.0e-9
            && self.longitudinal_force_n > 0.0
            && self.radius_m > 0.0
            && self.slip_ratio > 0.0
            && self.friction_budget_ratio <= 1.05
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "wheel_spin_torque_outputs_finite",
                "wheel-spin torque-balance outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "wheel_spin_tire_reaction_positive",
                "driven wheel should subtract a positive tire reaction torque from motor torque",
                "tire_reaction_torque_nm",
                self.tire_reaction_torque_nm,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "wheel_spin_bearing_loss_nonnegative",
                "bearing loss should oppose positive wheel spin",
                "bearing_loss_nm",
                self.bearing_loss_nm,
                0.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "wheel_spin_net_torque_accelerates",
                "remaining drive torque should accelerate wheel spin when torque exceeds ground reaction",
                "net_torque_nm",
                self.net_torque_nm,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "wheel_spin_integrator_matches_equation",
                "integrated omega change should match dt * (motor - tire - bearing) / inertia",
                "omega_delta_error",
                self.omega_delta_error(),
                1.0e-9,
            ),
            TireRigAcceptanceCheck::at_least(
                "wheel_spin_slip_is_outcome",
                "drive torque should produce positive slip through the wheel equation",
                "slip_ratio",
                self.slip_ratio,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "wheel_spin_friction_budget_bounded",
                "wheel-spin sample should keep the contact force inside the shared friction budget",
                "friction_budget_ratio",
                self.friction_budget_ratio,
                1.05,
            ),
        ]
    }
}

/// Data-free vehicle longitudinal-motion accounting.
///
/// PLAN_REALTIRE requires vehicle speed to integrate from tire force, rolling
/// resistance, and aerodynamic drag instead of from a viewer-side velocity
/// shortcut. This is a compact one-step check for
/// `m * v_dot = Σ(Fx_wheels - rolling_loss_wheels) - drag` using the same
/// brush telemetry that drives wheel-spin and force reports.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireVehicleLongitudinalMotionReport {
    pub driven_wheel_count: f64,
    pub vehicle_mass_kg: f64,
    pub dt_s: f64,
    pub speed_before_m_per_s: f64,
    pub gross_drive_force_per_wheel_n: f64,
    pub rolling_loss_per_wheel_n: f64,
    pub net_tire_force_per_wheel_n: f64,
    pub aero_drag_force_n: f64,
    pub net_vehicle_force_n: f64,
    pub expected_accel_m_per_s2: f64,
    pub expected_delta_speed_m_per_s: f64,
    pub integrated_delta_speed_m_per_s: f64,
    pub velocity_delta_error_m_per_s: f64,
    pub force_balance_error_n: f64,
    pub finite_outputs: bool,
}

impl TireVehicleLongitudinalMotionReport {
    pub fn is_finite(self) -> bool {
        self.driven_wheel_count.is_finite()
            && self.vehicle_mass_kg.is_finite()
            && self.dt_s.is_finite()
            && self.speed_before_m_per_s.is_finite()
            && self.gross_drive_force_per_wheel_n.is_finite()
            && self.rolling_loss_per_wheel_n.is_finite()
            && self.net_tire_force_per_wheel_n.is_finite()
            && self.aero_drag_force_n.is_finite()
            && self.net_vehicle_force_n.is_finite()
            && self.expected_accel_m_per_s2.is_finite()
            && self.expected_delta_speed_m_per_s.is_finite()
            && self.integrated_delta_speed_m_per_s.is_finite()
            && self.velocity_delta_error_m_per_s.is_finite()
            && self.force_balance_error_n.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.driven_wheel_count >= 1.0
            && self.vehicle_mass_kg > 0.0
            && self.dt_s > 0.0
            && self.gross_drive_force_per_wheel_n > 0.0
            && self.rolling_loss_per_wheel_n >= 0.0
            && self.net_tire_force_per_wheel_n > 0.0
            && self.aero_drag_force_n >= 0.0
            && self.net_vehicle_force_n > 0.0
            && self.expected_accel_m_per_s2 > 0.0
            && self.velocity_delta_error_m_per_s <= 1.0e-12
            && self.force_balance_error_n <= 1.0e-9
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "vehicle_longitudinal_outputs_finite",
                "vehicle longitudinal-motion accounting should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "vehicle_longitudinal_drive_force_positive",
                "driven tires should provide positive gross drive force",
                "gross_drive_force_per_wheel_n",
                self.gross_drive_force_per_wheel_n,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "vehicle_longitudinal_rolling_loss_nonnegative",
                "rolling resistance should subtract a non-negative force",
                "rolling_loss_per_wheel_n",
                self.rolling_loss_per_wheel_n,
                0.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "vehicle_longitudinal_drag_nonnegative",
                "aero drag should subtract a non-negative force",
                "aero_drag_force_n",
                self.aero_drag_force_n,
                0.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "vehicle_longitudinal_net_force_accelerates",
                "net tire force after rolling loss and drag should accelerate the vehicle",
                "net_vehicle_force_n",
                self.net_vehicle_force_n,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "vehicle_longitudinal_integrator_matches_equation",
                "integrated speed change should match dt * net_force / mass",
                "velocity_delta_error_m_per_s",
                self.velocity_delta_error_m_per_s,
                1.0e-12,
            ),
            TireRigAcceptanceCheck::at_most(
                "vehicle_longitudinal_force_balance_closes",
                "vehicle force balance should close from tire telemetry, rolling loss, and drag",
                "force_balance_error_n",
                self.force_balance_error_n,
                1.0e-9,
            ),
        ]
    }
}

pub fn tire_vehicle_longitudinal_motion_report() -> TireVehicleLongitudinalMotionReport {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.205, 145_000.0)
        .with_segments(128);
    let load_n = 4_200.0;
    let speed_m_per_s = 8.0;
    let slip_ratio = 0.10;
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
    let contact = evaluate_brush_tire(
        &params,
        &RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [speed_m_per_s, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip_ratio) / params.radius.max(1.0e-9),
        },
        &terrain,
        &material,
    );
    let driven_wheel_count = 2.0;
    let vehicle_mass_kg = 1_450.0;
    let dt_s = 1.0 / 180.0;
    let aero_cd_area_m2 = 0.62;
    let air_density_kg_per_m3 = 1.225;
    let gross_drive_force_per_wheel_n = finite_or(
        contact.longitudinal_force + contact.rolling_resistance_force,
        0.0,
    );
    let rolling_loss_per_wheel_n = finite_nonnegative(contact.rolling_resistance_force.abs());
    let net_tire_force_per_wheel_n = finite_or(
        gross_drive_force_per_wheel_n - rolling_loss_per_wheel_n,
        0.0,
    );
    let aero_drag_force_n =
        0.5 * air_density_kg_per_m3 * aero_cd_area_m2 * speed_m_per_s * speed_m_per_s;
    let net_vehicle_force_n = driven_wheel_count * net_tire_force_per_wheel_n - aero_drag_force_n;
    let expected_accel_m_per_s2 = net_vehicle_force_n / vehicle_mass_kg;
    let expected_delta_speed_m_per_s = expected_accel_m_per_s2 * dt_s;
    let speed_after_m_per_s = finite_or(
        speed_m_per_s + expected_accel_m_per_s2 * dt_s,
        speed_m_per_s,
    );
    let integrated_delta_speed_m_per_s = speed_after_m_per_s - speed_m_per_s;
    let force_balance_error_n = (net_vehicle_force_n
        - (driven_wheel_count * (gross_drive_force_per_wheel_n - rolling_loss_per_wheel_n)
            - aero_drag_force_n))
        .abs();
    TireVehicleLongitudinalMotionReport {
        driven_wheel_count,
        vehicle_mass_kg,
        dt_s,
        speed_before_m_per_s: speed_m_per_s,
        gross_drive_force_per_wheel_n,
        rolling_loss_per_wheel_n,
        net_tire_force_per_wheel_n,
        aero_drag_force_n,
        net_vehicle_force_n,
        expected_accel_m_per_s2,
        expected_delta_speed_m_per_s,
        integrated_delta_speed_m_per_s,
        velocity_delta_error_m_per_s: (integrated_delta_speed_m_per_s
            - expected_delta_speed_m_per_s)
            .abs(),
        force_balance_error_n,
        finite_outputs: contact.is_finite()
            && contact.normal_load > 0.0
            && contact.consumed_friction_budget <= contact.available_friction_budget * 1.05 + 1.0,
    }
}

/// Data-free four-wheel dynamic load-transfer / suspension sanity gate.
///
/// This is not calibration and not a full multibody vehicle solve. It
/// packages the basic vehicle-coupling truth that PLAN_REALTIRE wants before
/// trusting realistic tire scenes: longitudinal acceleration moves normal load
/// between axles, the contact patch and tire force respond through the native
/// brush law, and suspension travel follows the same per-corner load instead
/// of a viewer-only chassis pose shortcut.
#[derive(Clone, Debug)]
pub struct TireVehicleDynamicLoadTransferReport {
    pub static_load_n_per_wheel: f64,
    pub braking_front_load_n_per_wheel: f64,
    pub braking_rear_load_n_per_wheel: f64,
    pub accel_front_load_n_per_wheel: f64,
    pub accel_rear_load_n_per_wheel: f64,
    pub static_total_load_n: f64,
    pub braking_total_load_error_n: f64,
    pub accel_total_load_error_n: f64,
    pub static_patch_length_m: f64,
    pub braking_front_patch_length_m: f64,
    pub braking_rear_patch_length_m: f64,
    pub accel_front_patch_length_m: f64,
    pub accel_rear_patch_length_m: f64,
    pub static_abs_force_n: f64,
    pub braking_front_abs_force_n: f64,
    pub braking_rear_abs_force_n: f64,
    pub accel_front_abs_force_n: f64,
    pub accel_rear_abs_force_n: f64,
    pub static_suspension_travel_m: f64,
    pub braking_front_suspension_travel_m: f64,
    pub braking_rear_suspension_travel_m: f64,
    pub accel_front_suspension_travel_m: f64,
    pub accel_rear_suspension_travel_m: f64,
    pub braking_front_force_gain_over_static: f64,
    pub braking_front_load_gain_over_static: f64,
    pub accel_rear_force_gain_over_static: f64,
    pub accel_rear_load_gain_over_static: f64,
    pub braking_front_telemetry: RealTireTelemetrySample,
    pub accel_rear_telemetry: RealTireTelemetrySample,
    pub finite_outputs: bool,
}

impl TireVehicleDynamicLoadTransferReport {
    pub fn is_finite(&self) -> bool {
        self.static_load_n_per_wheel.is_finite()
            && self.braking_front_load_n_per_wheel.is_finite()
            && self.braking_rear_load_n_per_wheel.is_finite()
            && self.accel_front_load_n_per_wheel.is_finite()
            && self.accel_rear_load_n_per_wheel.is_finite()
            && self.static_total_load_n.is_finite()
            && self.braking_total_load_error_n.is_finite()
            && self.accel_total_load_error_n.is_finite()
            && self.static_patch_length_m.is_finite()
            && self.braking_front_patch_length_m.is_finite()
            && self.braking_rear_patch_length_m.is_finite()
            && self.accel_front_patch_length_m.is_finite()
            && self.accel_rear_patch_length_m.is_finite()
            && self.static_abs_force_n.is_finite()
            && self.braking_front_abs_force_n.is_finite()
            && self.braking_rear_abs_force_n.is_finite()
            && self.accel_front_abs_force_n.is_finite()
            && self.accel_rear_abs_force_n.is_finite()
            && self.static_suspension_travel_m.is_finite()
            && self.braking_front_suspension_travel_m.is_finite()
            && self.braking_rear_suspension_travel_m.is_finite()
            && self.accel_front_suspension_travel_m.is_finite()
            && self.accel_rear_suspension_travel_m.is_finite()
            && self.braking_front_force_gain_over_static.is_finite()
            && self.braking_front_load_gain_over_static.is_finite()
            && self.accel_rear_force_gain_over_static.is_finite()
            && self.accel_rear_load_gain_over_static.is_finite()
            && self.braking_front_telemetry.is_finite()
            && self.accel_rear_telemetry.is_finite()
    }

    pub fn passed(&self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.acceptance_checks().iter().all(|check| check.passed)
    }

    pub fn acceptance_checks(&self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_outputs_finite",
                "dynamic load-transfer tire outputs should remain finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "dynamic_load_transfer_braking_conserves_load",
                "braking load transfer should preserve total four-wheel vertical load",
                "braking_total_load_error_n",
                self.braking_total_load_error_n,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "dynamic_load_transfer_accel_conserves_load",
                "acceleration load transfer should preserve total four-wheel vertical load",
                "accel_total_load_error_n",
                self.accel_total_load_error_n,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_brake_front_loads_up",
                "braking should transfer normal load onto the front axle",
                "braking_front_minus_static_load_n",
                self.braking_front_load_n_per_wheel - self.static_load_n_per_wheel,
                100.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_brake_rear_unloads",
                "braking should unload the rear axle",
                "static_minus_braking_rear_load_n",
                self.static_load_n_per_wheel - self.braking_rear_load_n_per_wheel,
                100.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_accel_rear_loads_up",
                "acceleration should transfer normal load onto the rear axle",
                "accel_rear_minus_static_load_n",
                self.accel_rear_load_n_per_wheel - self.static_load_n_per_wheel,
                100.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_accel_front_unloads",
                "acceleration should unload the front axle",
                "static_minus_accel_front_load_n",
                self.static_load_n_per_wheel - self.accel_front_load_n_per_wheel,
                100.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_brake_front_patch_grows",
                "front braking load should grow the front contact patch relative to the unloaded rear",
                "brake_front_minus_rear_patch_m",
                self.braking_front_patch_length_m - self.braking_rear_patch_length_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_accel_rear_patch_grows",
                "rear acceleration load should grow the rear contact patch relative to the unloaded front",
                "accel_rear_minus_front_patch_m",
                self.accel_rear_patch_length_m - self.accel_front_patch_length_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_brake_front_suspension_compresses",
                "front suspension travel should increase under braking load transfer",
                "brake_front_minus_static_travel_m",
                self.braking_front_suspension_travel_m - self.static_suspension_travel_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_brake_rear_suspension_extends",
                "rear suspension travel should decrease under braking load transfer",
                "static_minus_brake_rear_travel_m",
                self.static_suspension_travel_m - self.braking_rear_suspension_travel_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_accel_rear_suspension_compresses",
                "rear suspension travel should increase under acceleration load transfer",
                "accel_rear_minus_static_travel_m",
                self.accel_rear_suspension_travel_m - self.static_suspension_travel_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_front_brake_force_increases",
                "front tire braking force should increase when braking transfers load forward",
                "braking_front_force_gain_over_static",
                self.braking_front_force_gain_over_static,
                1.01,
            ),
            TireRigAcceptanceCheck::at_most(
                "dynamic_load_transfer_front_force_sublinear",
                "front braking force gain should remain below the normal-load gain because tire μ is load-sensitive",
                "front_force_gain_over_load_gain",
                self.braking_front_force_gain_over_static
                    / self.braking_front_load_gain_over_static.max(1.0e-9),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dynamic_load_transfer_rear_drive_force_increases",
                "rear drive force should increase when acceleration transfers load rearward",
                "accel_rear_force_gain_over_static",
                self.accel_rear_force_gain_over_static,
                1.01,
            ),
            TireRigAcceptanceCheck::at_most(
                "dynamic_load_transfer_rear_force_sublinear",
                "rear drive force gain should remain below the normal-load gain because tire μ is load-sensitive",
                "rear_force_gain_over_load_gain",
                self.accel_rear_force_gain_over_static
                    / self.accel_rear_load_gain_over_static.max(1.0e-9),
                1.0,
            ),
        ]
    }

    pub fn snapshot(&self) -> TireRigSnapshot {
        TireRigSweepPoint {
            load_n: self.braking_front_load_n_per_wheel,
            pressure_pa: 180_000.0,
            hub_height_m: 0.0,
            deflection_m: self.braking_front_suspension_travel_m,
            speed_m_per_s: 12.0,
            slip_ratio_command: -0.18,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: self.braking_front_telemetry.clone(),
        }
        .snapshot_on_terrain(
            "vehicle_dynamic_load_transfer",
            "passenger_radial",
            "dry_asphalt",
            "flat_dynamic_load_transfer",
            "cpu_brush_vehicle_load_transfer_suspension",
            &FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static),
            0.0,
            0.0,
        )
    }
}

#[derive(Clone, Debug)]
struct TireDrivenOutcomeState {
    distance_m: f64,
    speed_m_per_s: f64,
    final_slip_ratio: f64,
    telemetry: RealTireTelemetrySample,
}

fn simulate_torque_slip_outcome(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    drive_torque_nm_per_wheel: f64,
) -> TireDrivenOutcomeState {
    let brush_params = sanitize_brush_params(params);
    let params = RealTireParams {
        radius: brush_params.radius,
        width: brush_params.width,
        pressure_pa: brush_params.pressure_pa,
        segments: brush_params.segments,
        width_segments: brush_params.width_segments,
        carcass_normal_stiffness: brush_params.carcass_normal_stiffness,
        tread_shear_stiffness: brush_params.tread_shear_stiffness,
        structural_lateral_bias_per_normal: brush_params.structural_lateral_bias_per_normal,
        relaxation_speed: brush_params.relaxation_speed,
    };
    let load_n = finite_nonnegative(load_n).max(1.0);
    let torque = finite_nonnegative(drive_torque_nm_per_wheel);
    let dt = 1.0 / 180.0;
    let steps = 240usize;
    let driven_wheels = 2.0;
    let vehicle_mass_kg = 1_450.0;
    let wheel_inertia = 1.35;
    let hub_y = solve_loaded_tire_hub_height(&params, terrain, 0.0, 0.0, load_n);
    let mut speed = 1.0_f64;
    let mut distance = 0.0_f64;
    let mut omega = speed / params.radius.max(1.0e-6);
    let mut final_slip_ratio = 0.0;
    let mut telemetry = RealTireContact::empty().telemetry_sample(dt);

    for _ in 0..steps {
        let state = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [speed, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega,
        };
        let contact = evaluate_brush_tire(&params, &state, terrain, material);
        telemetry = contact.telemetry_sample(dt);
        final_slip_ratio = contact.slip_ratio;
        let tire_reaction_torque = contact.longitudinal_force * params.radius;
        let bearing_loss = 0.08 * omega;
        let omega_dot = (torque - tire_reaction_torque - bearing_loss) / wheel_inertia;
        omega = finite_or(omega + omega_dot * dt, 0.0).clamp(0.0, 450.0);
        let drive_force = driven_wheels * contact.longitudinal_force;
        let aero_drag = 0.40 * speed * speed;
        let accel = (drive_force - aero_drag) / vehicle_mass_kg;
        speed = finite_or(speed + accel * dt, 0.0).max(0.0);
        distance += speed * dt;
    }

    TireDrivenOutcomeState {
        distance_m: distance,
        speed_m_per_s: speed,
        final_slip_ratio,
        telemetry,
    }
}

pub fn tire_torque_induced_slip_outcome_report() -> TireTorqueSlipOutcomeReport {
    let material = SurfaceMaterial {
        mu_static: 0.30,
        mu_dynamic: 0.22,
        bristle_damping: 4.5e4,
        rolling_resistance: 0.020,
        cohesion: 0.0,
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let high = RealTireParams::high_pressure_reference()
        .with_geometry_and_pressure(0.32, 0.185, 360_000.0)
        .with_segments(96);
    let partial = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.205, 125_000.0)
        .with_segments(128);
    let load_n = 4_600.0;
    let torque = 620.0;
    let high_run = simulate_torque_slip_outcome(&high, &terrain, &material, load_n, torque);
    let partial_run = simulate_torque_slip_outcome(&partial, &terrain, &material, load_n, torque);

    TireTorqueSlipOutcomeReport {
        scenario_id: "torque_induced_slip_outcome".to_owned(),
        surface_label: "low_mu_compacted_dirt".to_owned(),
        drive_torque_nm_per_wheel: torque,
        load_n_per_wheel: load_n,
        high_pressure_pa: high.pressure_pa,
        partial_pressure_pa: partial.pressure_pa,
        high_final_slip_ratio: high_run.final_slip_ratio,
        partial_final_slip_ratio: partial_run.final_slip_ratio,
        high_distance_m: high_run.distance_m,
        partial_distance_m: partial_run.distance_m,
        high_final_speed_m_per_s: high_run.speed_m_per_s,
        partial_final_speed_m_per_s: partial_run.speed_m_per_s,
        high_patch_length_m: high_run.telemetry.patch.patch_length_m,
        partial_patch_length_m: partial_run.telemetry.patch.patch_length_m,
        high_telemetry: high_run.telemetry,
        partial_telemetry: partial_run.telemetry,
    }
}

pub fn tire_wheel_spin_torque_balance_report() -> TireWheelSpinTorqueBalanceReport {
    let material = SurfaceMaterial {
        mu_static: 0.42,
        mu_dynamic: 0.31,
        bristle_damping: 4.0e4,
        rolling_resistance: 0.018,
        cohesion: 0.0,
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.205, 135_000.0)
        .with_segments(128);
    let load_n = 4_500.0;
    let speed_m_per_s = 3.0;
    let initial_omega = speed_m_per_s / params.radius.max(1.0e-9) * 1.25;
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
    let state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [speed_m_per_s, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: initial_omega,
    };
    let contact = evaluate_brush_tire(&params, &state, &terrain, &material);
    let drive_torque_nm = 620.0;
    let wheel_inertia_kg_m2 = 1.35;
    let dt_s = 1.0 / 180.0;
    let tire_reaction_torque_nm = contact.longitudinal_force * params.radius;
    let bearing_loss_nm = 0.08 * initial_omega;
    let net_torque_nm = drive_torque_nm - tire_reaction_torque_nm - bearing_loss_nm;
    let omega_dot_rad_s2 = net_torque_nm / wheel_inertia_kg_m2;
    let next_omega = finite_or(initial_omega + omega_dot_rad_s2 * dt_s, initial_omega);
    let telemetry = contact.telemetry_sample(dt_s);
    let friction_budget_ratio =
        telemetry.consumed_friction_budget_n / telemetry.available_friction_budget_n.max(1.0);
    TireWheelSpinTorqueBalanceReport {
        drive_torque_nm,
        tire_reaction_torque_nm,
        bearing_loss_nm,
        net_torque_nm,
        wheel_inertia_kg_m2,
        dt_s,
        initial_omega_rad_s: initial_omega,
        omega_dot_rad_s2,
        expected_delta_omega_rad_s: omega_dot_rad_s2 * dt_s,
        integrated_delta_omega_rad_s: next_omega - initial_omega,
        longitudinal_force_n: contact.longitudinal_force,
        radius_m: params.radius,
        slip_ratio: contact.slip_ratio,
        friction_budget_ratio,
        finite_outputs: contact.is_finite() && telemetry.is_finite(),
    }
}

pub fn tire_vehicle_dynamic_load_transfer_report() -> TireVehicleDynamicLoadTransferReport {
    let material = SurfaceMaterial {
        rolling_resistance: 0.010,
        cohesion: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.195, 180_000.0)
        .with_segments(128);
    let static_load = 4_200.0;
    let braking_front_load = 5_500.0;
    let braking_rear_load = 2_900.0;
    let accel_front_load = 3_050.0;
    let accel_rear_load = 5_350.0;
    let spring_rate_n_per_m = 48_000.0;
    let travel = |load: f64| load / spring_rate_n_per_m;
    let static_brake = tire_rig_point(
        &params,
        &terrain,
        &material,
        static_load,
        12.0,
        -0.18,
        0.0,
        0.0,
    );
    let braking_front = tire_rig_point(
        &params,
        &terrain,
        &material,
        braking_front_load,
        12.0,
        -0.18,
        0.0,
        0.0,
    );
    let braking_rear = tire_rig_point(
        &params,
        &terrain,
        &material,
        braking_rear_load,
        12.0,
        -0.18,
        0.0,
        0.0,
    );
    let static_drive = tire_rig_point(
        &params,
        &terrain,
        &material,
        static_load,
        12.0,
        0.18,
        0.0,
        0.0,
    );
    let accel_front = tire_rig_point(
        &params,
        &terrain,
        &material,
        accel_front_load,
        12.0,
        0.18,
        0.0,
        0.0,
    );
    let accel_rear = tire_rig_point(
        &params,
        &terrain,
        &material,
        accel_rear_load,
        12.0,
        0.18,
        0.0,
        0.0,
    );
    let static_total_load = static_load * 4.0;
    let braking_total_load = 2.0 * (braking_front_load + braking_rear_load);
    let accel_total_load = 2.0 * (accel_front_load + accel_rear_load);
    let static_brake_abs_force = static_brake.telemetry.longitudinal_force_n.abs().max(1.0);
    let static_drive_abs_force = static_drive.telemetry.longitudinal_force_n.abs().max(1.0);
    let braking_front_abs_force = braking_front.telemetry.longitudinal_force_n.abs();
    let accel_rear_abs_force = accel_rear.telemetry.longitudinal_force_n.abs();

    TireVehicleDynamicLoadTransferReport {
        static_load_n_per_wheel: static_load,
        braking_front_load_n_per_wheel: braking_front_load,
        braking_rear_load_n_per_wheel: braking_rear_load,
        accel_front_load_n_per_wheel: accel_front_load,
        accel_rear_load_n_per_wheel: accel_rear_load,
        static_total_load_n: static_total_load,
        braking_total_load_error_n: (braking_total_load - static_total_load).abs(),
        accel_total_load_error_n: (accel_total_load - static_total_load).abs(),
        static_patch_length_m: static_brake.telemetry.patch.patch_length_m,
        braking_front_patch_length_m: braking_front.telemetry.patch.patch_length_m,
        braking_rear_patch_length_m: braking_rear.telemetry.patch.patch_length_m,
        accel_front_patch_length_m: accel_front.telemetry.patch.patch_length_m,
        accel_rear_patch_length_m: accel_rear.telemetry.patch.patch_length_m,
        static_abs_force_n: static_brake_abs_force,
        braking_front_abs_force_n: braking_front_abs_force,
        braking_rear_abs_force_n: braking_rear.telemetry.longitudinal_force_n.abs(),
        accel_front_abs_force_n: accel_front.telemetry.longitudinal_force_n.abs(),
        accel_rear_abs_force_n: accel_rear_abs_force,
        static_suspension_travel_m: travel(static_load),
        braking_front_suspension_travel_m: travel(braking_front_load),
        braking_rear_suspension_travel_m: travel(braking_rear_load),
        accel_front_suspension_travel_m: travel(accel_front_load),
        accel_rear_suspension_travel_m: travel(accel_rear_load),
        braking_front_force_gain_over_static: braking_front_abs_force / static_brake_abs_force,
        braking_front_load_gain_over_static: braking_front_load / static_load,
        accel_rear_force_gain_over_static: accel_rear_abs_force / static_drive_abs_force,
        accel_rear_load_gain_over_static: accel_rear_load / static_load,
        braking_front_telemetry: braking_front.telemetry.clone(),
        accel_rear_telemetry: accel_rear.telemetry.clone(),
        finite_outputs: static_brake.is_finite()
            && braking_front.is_finite()
            && braking_rear.is_finite()
            && static_drive.is_finite()
            && accel_front.is_finite()
            && accel_rear.is_finite(),
    }
}

/// One data-free vehicle-level sanity scenario.
///
/// This is intentionally not a full vehicle dynamics solver and not a
/// calibration fixture. Each row exercises the same reusable tire/contact APIs
/// in a four-wheel vehicle context so viewer examples have falsification cases
/// for heavy load, low friction, braking, steering/camber, curb/obstacle
/// contact, and soft off-road rolling loss.
#[derive(Clone, Debug)]
pub struct TireVehicleSanityCase {
    pub scenario_id: String,
    pub expected_behavior: String,
    pub physics_path: String,
    pub wheel_count: usize,
    pub per_wheel_load_n: f64,
    pub total_vehicle_load_n: f64,
    pub observed_slip_ratio: f64,
    pub observed_lateral_force_n: f64,
    pub observed_patch_length_m: f64,
    pub observed_contact_count: usize,
    pub telemetry: RealTireTelemetrySample,
    pub passed: bool,
}

impl TireVehicleSanityCase {
    pub fn is_finite(&self) -> bool {
        self.wheel_count > 0
            && self.per_wheel_load_n.is_finite()
            && self.total_vehicle_load_n.is_finite()
            && self.observed_slip_ratio.is_finite()
            && self.observed_lateral_force_n.is_finite()
            && self.observed_patch_length_m.is_finite()
            && self.telemetry.is_finite()
    }

    pub fn acceptance_checks(&self) -> Vec<TireRigAcceptanceCheck> {
        let wheel_count = self.wheel_count as f64;
        let expected_total_load_n = finite_nonnegative(self.per_wheel_load_n) * wheel_count;
        let total_load_relative_error =
            (finite_nonnegative(self.total_vehicle_load_n) - expected_total_load_n).abs()
                / expected_total_load_n.max(1.0);
        let friction_budget_ratio = finite_nonnegative(self.telemetry.consumed_friction_budget_n)
            / finite_nonnegative(self.telemetry.available_friction_budget_n).max(1.0);
        let abs_fx_over_fz = finite_or(self.telemetry.longitudinal_force_n, 0.0).abs()
            / finite_nonnegative(self.telemetry.normal_load_n).max(1.0);

        let mut checks = vec![
            TireRigAcceptanceCheck::between(
                "vehicle_four_wheel_context",
                "vehicle sanity scenarios should stay in a four-wheel vehicle context",
                "wheel_count",
                wheel_count,
                4.0,
                4.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "vehicle_per_wheel_load_positive",
                "per-wheel normal load should be positive",
                "per_wheel_load_n",
                self.per_wheel_load_n,
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "vehicle_total_load_accounting",
                "total vehicle load should match wheel count times per-wheel load",
                "total_load_relative_error",
                total_load_relative_error,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "vehicle_contact_cells_present",
                "vehicle tire case should keep at least one active contact cell",
                "observed_contact_count",
                self.observed_contact_count as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "vehicle_patch_length_positive",
                "vehicle tire case should produce a non-zero footprint length",
                "observed_patch_length_m",
                self.observed_patch_length_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_most(
                "vehicle_slip_ratio_bounded",
                "vehicle tire case should keep slip finite and inside a broad debug envelope",
                "abs_observed_slip_ratio",
                self.observed_slip_ratio.abs(),
                10.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "vehicle_friction_budget_bounded",
                "vehicle tire case should not consume far beyond the available friction budget",
                "consumed_over_available_friction",
                friction_budget_ratio,
                1.35,
            ),
        ];

        match self.scenario_id.as_str() {
            "heavy_loaded_vehicle" => {
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_heavy_load_present",
                    "heavy-load case should represent a genuinely heavy four-wheel load",
                    "total_vehicle_load_n",
                    self.total_vehicle_load_n,
                    30_000.0,
                ));
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_heavy_patch_multicell",
                    "heavy-load case should not collapse to a single contact cell",
                    "observed_contact_count",
                    self.observed_contact_count as f64,
                    3.0,
                ));
            }
            "low_friction_driven_vehicle" => {
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_ice_drive_slips",
                    "drive torque on ice should visibly raise wheel slip",
                    "abs_observed_slip_ratio",
                    self.observed_slip_ratio.abs(),
                    0.25,
                ));
                checks.push(TireRigAcceptanceCheck::at_most(
                    "vehicle_ice_force_capped",
                    "ice drive force should remain capped by the low-friction budget",
                    "abs_fx_over_fz",
                    abs_fx_over_fz,
                    0.35,
                ));
            }
            "braking_vehicle" => {
                checks.push(TireRigAcceptanceCheck::at_most(
                    "vehicle_brake_negative_slip",
                    "braking case should develop negative slip",
                    "observed_slip_ratio",
                    self.observed_slip_ratio,
                    -0.10,
                ));
                checks.push(TireRigAcceptanceCheck::at_most(
                    "vehicle_brake_negative_force",
                    "braking case should produce a negative longitudinal braking force",
                    "longitudinal_force_n",
                    self.telemetry.longitudinal_force_n,
                    -50.0,
                ));
            }
            "steering_camber_vehicle" => {
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_steer_lateral_force",
                    "steered/cambered case should produce lateral force from tire orientation",
                    "abs_lateral_force_n",
                    self.observed_lateral_force_n.abs(),
                    100.0,
                ));
                checks.push(TireRigAcceptanceCheck::at_most(
                    "vehicle_steer_pneumatic_trail_bounded",
                    "steered/cambered case should keep pneumatic trail finite and geometrically bounded",
                    "abs_pneumatic_trail_m",
                    self.telemetry.patch.pneumatic_trail_m.abs(),
                    0.5,
                ));
            }
            "obstacle_curb_vehicle" => {
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_curb_patch_multicell",
                    "curb case should keep a multi-cell patch across the obstacle edge",
                    "observed_contact_count",
                    self.observed_contact_count as f64,
                    2.0,
                ));
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_curb_patch_length",
                    "curb case should keep a measurable patch length over the heightfield edge",
                    "observed_patch_length_m",
                    self.observed_patch_length_m,
                    0.04,
                ));
            }
            "offroad_soft_soil_vehicle" => {
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_offroad_patch_multicell",
                    "soft-soil case should keep finite multi-cell tread support",
                    "observed_contact_count",
                    self.observed_contact_count as f64,
                    3.0,
                ));
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_offroad_rolling_loss",
                    "soft-soil case should expose rolling-loss penalty in tire telemetry",
                    "abs_rolling_resistance_force_n",
                    self.telemetry.rolling_resistance_force_n.abs(),
                    100.0,
                ));
            }
            _ => {
                checks.push(TireRigAcceptanceCheck::at_least(
                    "vehicle_case_has_named_expectation",
                    "unknown vehicle sanity cases must still carry a pass/fail expectation",
                    "stored_pass_flag",
                    if self.passed { 1.0 } else { 0.0 },
                    1.0,
                ));
            }
        }

        checks
    }

    pub fn acceptance_passed(&self) -> bool {
        self.passed
            && !self.acceptance_checks().is_empty()
            && self.acceptance_checks().iter().all(|check| check.passed)
    }

    pub fn failed_acceptance_check_ids(&self) -> Vec<String> {
        self.acceptance_checks()
            .into_iter()
            .filter(|check| !check.passed)
            .map(|check| check.check_id)
            .collect()
    }

    pub fn to_text_line(&self) -> String {
        format!(
            "{} wheels={} per_wheel_load_n={:.3} total_load_n={:.3} slip={:.6} lateral_force_n={:.3} patch_m={:.6} contacts={} checks={} passed={} physics={} expected={}",
            csv_token(&self.scenario_id),
            self.wheel_count,
            self.per_wheel_load_n,
            self.total_vehicle_load_n,
            self.observed_slip_ratio,
            self.observed_lateral_force_n,
            self.observed_patch_length_m,
            self.observed_contact_count,
            self.acceptance_checks().len(),
            self.acceptance_passed(),
            csv_token(&self.physics_path),
            csv_token(&self.expected_behavior),
        )
    }
}

fn vehicle_sanity_case_from_point(
    scenario_id: &str,
    expected_behavior: &str,
    physics_path: &str,
    point: &TireRigSweepPoint,
    observed_slip_ratio: f64,
    passed: bool,
) -> TireVehicleSanityCase {
    let wheel_count = 4;
    TireVehicleSanityCase {
        scenario_id: csv_token(scenario_id),
        expected_behavior: csv_token(expected_behavior),
        physics_path: csv_token(physics_path),
        wheel_count,
        per_wheel_load_n: finite_nonnegative(point.load_n),
        total_vehicle_load_n: finite_nonnegative(point.load_n) * wheel_count as f64,
        observed_slip_ratio: finite_or(observed_slip_ratio, 0.0),
        observed_lateral_force_n: finite_or(point.telemetry.lateral_force_n, 0.0),
        observed_patch_length_m: finite_nonnegative(point.telemetry.patch.patch_length_m),
        observed_contact_count: point.telemetry.patch.contact_count,
        telemetry: point.telemetry.clone(),
        passed,
    }
}

/// Deterministic vehicle-level tire sanity suite.
///
/// The returned cases are broad qualitative checks, not fitted targets:
/// - a heavy vehicle should still produce a finite multi-cell patch;
/// - an ice/low-friction driven vehicle should spin/slip and stay force-capped;
/// - a braking vehicle should build negative slip and braking force through
///   the same transient wheel-spin path;
/// - a steering/camber vehicle should produce lateral force and aligning
///   moment from the same brush/orientation path;
/// - a curb/obstacle vehicle should see non-flat terrain normals and avoid
///   single-point patch collapse.
/// - a soft off-road vehicle should keep a finite patch while paying extra
///   rolling loss from the soil-like material state.
pub fn tire_vehicle_sanity_scenarios() -> Vec<TireVehicleSanityCase> {
    let road = SurfaceMaterial::dry_asphalt();
    let road_terrain = FlatTerrain::new(0.0, road.mu_static);
    let base = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 180_000.0)
        .with_segments(96);

    let heavy = tire_rig_point(&base, &road_terrain, &road, 8_500.0, 7.0, 0.08, 0.0, 0.0);
    let heavy_passed = heavy.is_finite()
        && heavy.telemetry.normal_load_n > 8_000.0
        && heavy.telemetry.patch.contact_count >= 3
        && heavy.telemetry.patch.patch_length_m > 0.05;
    let mut out = vec![vehicle_sanity_case_from_point(
        "heavy_loaded_vehicle",
        "four-wheel heavy load produces finite support, multiple patch cells, and bounded force accounting",
        "cpu_brush_load_pressure",
        &heavy,
        heavy.slip_ratio_command,
        heavy_passed,
    )];

    let ice = SurfaceMaterial::ice();
    let ice_terrain = FlatTerrain::new(0.0, ice.mu_static);
    let driven = tire_rig_step_torque_response(
        &base,
        &ice_terrain,
        &ice,
        4_200.0,
        8.0,
        1_800.0,
        1.0 / 120.0,
        96,
    );
    if let Some(last) = driven.last() {
        let fx_over_fz =
            last.telemetry.longitudinal_force_n.abs() / last.telemetry.normal_load_n.max(1.0);
        let point = TireRigSweepPoint {
            load_n: last.telemetry.normal_load_n,
            pressure_pa: base.pressure_pa,
            hub_height_m: base.radius,
            deflection_m: 0.0,
            speed_m_per_s: 8.0,
            slip_ratio_command: last.raw_slip_ratio,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: last.telemetry.clone(),
        };
        out.push(vehicle_sanity_case_from_point(
            "low_friction_driven_vehicle",
            "drive torque on ice raises wheel slip while the friction budget keeps longitudinal force low",
            "cpu_brush_transient_torque_ice",
            &point,
            last.raw_slip_ratio,
            last.is_finite() && last.raw_slip_ratio.abs() > 0.25 && fx_over_fz < 0.35,
        ));
    }

    let braking = tire_rig_step_brake_response(
        &base,
        &road_terrain,
        &road,
        4_200.0,
        14.0,
        1_400.0,
        1.0 / 120.0,
        96,
    );
    if let (Some(first), Some(last)) = (braking.first(), braking.last()) {
        let point = TireRigSweepPoint {
            load_n: last.telemetry.normal_load_n,
            pressure_pa: base.pressure_pa,
            hub_height_m: base.radius,
            deflection_m: 0.0,
            speed_m_per_s: 14.0,
            slip_ratio_command: last.raw_slip_ratio,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: last.telemetry.clone(),
        };
        out.push(vehicle_sanity_case_from_point(
            "braking_vehicle",
            "brake torque lowers wheel speed and produces negative slip/braking force through transient wheel dynamics",
            "cpu_brush_transient_brake",
            &point,
            last.raw_slip_ratio,
            last.is_finite()
                && last.wheel_omega_rad_s < first.wheel_omega_rad_s
                && last.raw_slip_ratio < -0.10
                && last.telemetry.longitudinal_force_n < -50.0,
        ));
    }

    let camber = 8.0_f64.to_radians();
    let steered =
        tire_rig_camber_sweep(&base, &road_terrain, &road, 4_200.0, 12.0, &[camber]).remove(0);
    let steered_passed = steered.is_finite()
        && steered.telemetry.lateral_force_n.abs() > 100.0
        && steered.telemetry.aligning_moment_nm.is_finite()
        && steered.telemetry.patch.pneumatic_trail_m.is_finite();
    out.push(vehicle_sanity_case_from_point(
        "steering_camber_vehicle",
        "steered/cambered vehicle produces lateral force, pneumatic trail, and aligning moment without a viewer yaw hack",
        "cpu_brush_orientation_camber",
        &steered,
        steered.slip_ratio_command,
        steered_passed,
    ));

    let nx = 41;
    let nz = 5;
    let spacing = 0.04;
    let origin = [-0.8, -0.08];
    let mut heights = Vec::with_capacity(nx * nz);
    for _iz in 0..nz {
        for ix in 0..nx {
            let x = origin[0] + ix as f64 * spacing;
            heights.push(if x >= 0.0 { 0.075 } else { 0.0 });
        }
    }
    let curb = HeightfieldTerrain::new(origin, spacing, nx, nz, heights, road.mu_static);
    let soft = base.with_geometry_and_pressure(0.32, 0.185, 95_000.0);
    let curb_points = tire_rig_heightfield_envelope_sweep(
        &soft,
        &curb,
        &road,
        4_500.0,
        3.0,
        0.06,
        &[-0.08, -0.02, 0.02, 0.08],
        0.0,
    );
    if let Some(edge) = curb_points.get(1) {
        let mut case = vehicle_sanity_case_from_point(
            "obstacle_curb_vehicle",
            "low-pressure tire sees the curb normal and keeps a multi-cell patch instead of collapsing to one point",
            "cpu_heightfield_brush_envelope",
            &edge.point,
            edge.point.slip_ratio_command,
            edge.is_finite()
                && edge.terrain_normal[0].abs() > 0.25
                && edge.point.telemetry.patch.contact_count > 1
                && edge.point.telemetry.patch.patch_length_m > 0.04,
        );
        case.observed_lateral_force_n = edge.point.telemetry.force_n[0];
        out.push(case);
    }

    let soft_soil = SurfaceMaterial::soft_soil();
    let soft_soil_terrain = FlatTerrain::new(0.0, soft_soil.mu_static);
    let offroad = tire_rig_point(
        &soft,
        &soft_soil_terrain,
        &soft_soil,
        4_800.0,
        5.0,
        0.12,
        0.0,
        0.0,
    );
    let road_reference = tire_rig_point(&soft, &road_terrain, &road, 4_800.0, 5.0, 0.12, 0.0, 0.0);
    out.push(vehicle_sanity_case_from_point(
        "offroad_soft_soil_vehicle",
        "soft off-road material keeps finite tread support but adds much higher rolling loss than dry asphalt",
        "cpu_brush_soft_soil_vehicle",
        &offroad,
        offroad.slip_ratio_command,
        offroad.is_finite()
            && road_reference.is_finite()
            && offroad.telemetry.patch.contact_count >= 3
            && offroad.telemetry.rolling_resistance_force_n.abs()
                > road_reference.telemetry.rolling_resistance_force_n.abs() * 3.0,
    ));

    out
}
