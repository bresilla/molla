//! `tire_rig_standard_non_calibration_run` and `tire_rig_default_non_calibration_run`
//! — the big PLAN_REALTIRE deterministic rig assembly.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

#![allow(unused_imports)]

use crate::terrain::{FlatTerrain, HeightfieldTerrain, Terrain};

use super::super::internal::{csv_token, finite_positive_or};
use super::super::*;

/// Run the default non-calibration tire rig suite.
pub fn tire_rig_default_non_calibration_run() -> TireRigRunOutput {
    tire_rig_standard_non_calibration_run(&TireRigRunConfig::default())
}

/// Built-in deterministic tire-rig runner for PLAN_REALTIRE falsification.
///
/// The suite covers the non-calibration cases that are easy to break while
/// iterating on the tire model: pressure/load support, dry-vs-ice slip, camber
/// and pneumatic trail, wet-film response, loose-soil response, curb/heightfield
/// envelope contact, and thermal/wear state. The generated snapshots are
/// reusable artifacts for tests and viewer replay; no external datasets or
/// target curves are involved.
pub fn tire_rig_standard_non_calibration_run(config: &TireRigRunConfig) -> TireRigRunOutput {
    let config = TireRigRunConfig {
        seed: config.seed,
        tire_preset: csv_token(&config.tire_preset),
        high_pressure_tire_preset: csv_token(&config.high_pressure_tire_preset),
        low_pressure_tire_preset: csv_token(&config.low_pressure_tire_preset),
        dry_surface_preset: csv_token(&config.dry_surface_preset),
        ice_surface_preset: csv_token(&config.ice_surface_preset),
        wet_surface_preset: csv_token(&config.wet_surface_preset),
        soil_surface_preset: csv_token(&config.soil_surface_preset),
        fixed_dt_s: finite_positive_or(config.fixed_dt_s, 1.0 / 120.0, 1.0e-6).min(0.10),
    };
    let dry = SurfaceMaterial::dry_asphalt();
    let ice = SurfaceMaterial::ice();
    let wet = SurfaceFilmState::wet_road().apply_to(dry, 12.0, &TireWearState::default());
    let soil = SurfaceMaterial::soft_soil();
    let dry_terrain = FlatTerrain::new(0.0, dry.mu_static);
    let ice_terrain = FlatTerrain::new(0.0, ice.mu_static);
    let soil_terrain = FlatTerrain::new(0.0, soil.mu_static);
    let moving_belt_terrain = TireRigSurfaceTransitionTerrain::new(
        0.0,
        dry,
        vec![
            TireRigSurfaceTransitionBand::new("moving_belt", -1.0, 1.0, dry)
                .with_surface_velocity([3.0, 0.0, 0.0]),
        ],
    )
    .with_transition_width(0.0);
    let base = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 180_000.0)
        .with_segments(96);
    let high = RealTireParams::high_pressure_reference()
        .with_geometry_and_pressure(0.32, 0.185, 260_000.0)
        .with_segments(96);
    let low = RealTireParams::offroad_soft()
        .with_geometry_and_pressure(0.32, 0.205, 95_000.0)
        .with_segments(128);

    let mut snapshots = Vec::new();

    let load_n = 4_500.0;
    let high_load = tire_rig_point(&high, &dry_terrain, &dry, load_n, 0.0, 0.0, 0.0, 0.0);
    let low_load = tire_rig_point(&low, &dry_terrain, &dry, load_n, 0.0, 0.0, 0.0, 0.0);
    snapshots.push(high_load.snapshot_on_terrain(
        "load_pressure_high",
        &config.high_pressure_tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_brush_load_pressure",
        &dry_terrain,
        0.0,
        0.0,
    ));
    snapshots.push(low_load.snapshot_on_terrain(
        "load_pressure_low",
        &config.low_pressure_tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_brush_load_pressure",
        &dry_terrain,
        0.0,
        0.0,
    ));

    let dry_slip = tire_rig_point(&base, &dry_terrain, &dry, load_n, 10.0, 0.18, 0.0, 0.0);
    let ice_slip = tire_rig_point(&base, &ice_terrain, &ice, load_n, 10.0, 0.18, 0.0, 0.0);
    snapshots.push(dry_slip.snapshot_on_terrain(
        "longitudinal_slip_dry",
        &config.tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_brush_longitudinal_slip",
        &dry_terrain,
        0.0,
        0.0,
    ));
    snapshots.push(ice_slip.snapshot_on_terrain(
        "longitudinal_slip_ice",
        &config.tire_preset,
        &config.ice_surface_preset,
        "flat",
        "cpu_brush_longitudinal_slip",
        &ice_terrain,
        0.0,
        0.0,
    ));

    let camber = tire_rig_point(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        12.0,
        0.02,
        4.0_f64.to_radians(),
        7.0_f64.to_radians(),
    );
    snapshots.push(camber.snapshot_on_terrain(
        "camber_pneumatic_trail",
        &config.tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_brush_orientation_camber",
        &dry_terrain,
        0.0,
        0.0,
    ));

    let wet_slip = tire_rig_point(&base, &dry_terrain, &wet, load_n, 12.0, 0.18, 0.0, 0.0);
    snapshots.push(wet_slip.snapshot_on_terrain(
        "wet_film_slip",
        &config.tire_preset,
        &config.wet_surface_preset,
        "flat_wet_film",
        "cpu_brush_surface_film",
        &dry_terrain,
        0.0,
        0.0,
    ));

    let soil_slip = tire_rig_point(&low, &soil_terrain, &soil, load_n, 6.0, 0.25, 0.0, 0.0);
    snapshots.push(soil_slip.snapshot_on_terrain(
        "loose_soil_slip",
        &config.low_pressure_tire_preset,
        &config.soil_surface_preset,
        "flat_loose_soil",
        "cpu_brush_soil_material",
        &soil_terrain,
        0.0,
        0.0,
    ));

    let moving_belt = tire_rig_point(
        &base,
        &moving_belt_terrain,
        &dry,
        load_n,
        3.0,
        0.0,
        0.0,
        0.0,
    );
    let moving_belt_snapshot = moving_belt.snapshot_on_terrain(
        "moving_ground_belt",
        &config.tire_preset,
        &config.dry_surface_preset,
        "moving_flat_belt",
        "cpu_brush_moving_ground",
        &moving_belt_terrain,
        0.0,
        0.0,
    );
    snapshots.push(moving_belt_snapshot.clone());
    let moving_belt_power_w = moving_belt_snapshot.moving_ground_power_w;

    let hv_nx = 5;
    let hv_nz = 3;
    let hv_heights = vec![0.0; hv_nx * hv_nz];
    let mut hv_surface_velocity_map = vec![[0.0, 0.0, 0.0]; hv_nx * hv_nz];
    for iz in 0..hv_nz {
        for ix in 3..hv_nx {
            hv_surface_velocity_map[iz * hv_nx + ix] = [-4.0, 0.0, 0.0];
        }
    }
    let heightfield_velocity_map =
        HeightfieldTerrain::new([-1.0, -0.5], 0.5, hv_nx, hv_nz, hv_heights, dry.mu_static)
            .with_surface_velocity_map(hv_surface_velocity_map);
    let heightfield_velocity_point = tire_rig_point_at(
        &base,
        &heightfield_velocity_map,
        &dry,
        [0.75, 0.0],
        load_n,
        10.0,
        0.0,
        0.0,
        0.0,
    );
    let heightfield_velocity_snapshot = heightfield_velocity_point.snapshot_on_terrain(
        "heightfield_surface_velocity_map",
        &config.tire_preset,
        &config.dry_surface_preset,
        "heightfield_surface_velocity_map",
        "cpu_heightfield_surface_velocity_map",
        &heightfield_velocity_map,
        0.75,
        0.0,
    );
    snapshots.push(heightfield_velocity_snapshot.clone());

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
    let curb = HeightfieldTerrain::new(origin, spacing, nx, nz, heights, dry.mu_static);
    let curb_point =
        tire_rig_heightfield_envelope_sweep(&low, &curb, &dry, load_n, 3.0, 0.06, &[-0.02], 0.0)
            .remove(0);
    let high_curb_point =
        tire_rig_heightfield_envelope_sweep(&high, &curb, &dry, load_n, 3.0, 0.06, &[-0.02], 0.0)
            .remove(0);
    snapshots.push(curb_point.point.snapshot_on_terrain(
        "curb_envelope_contact",
        &config.low_pressure_tire_preset,
        &config.dry_surface_preset,
        "heightfield_curb",
        "cpu_heightfield_brush_envelope",
        &curb,
        curb_point.x_m,
        curb_point.z_m,
    ));

    let cleat_nx = 51;
    let cleat_nz = 5;
    let cleat_spacing = 0.025;
    let cleat_origin = [-0.60, -0.05];
    let mut cleat_heights = Vec::with_capacity(cleat_nx * cleat_nz);
    for _iz in 0..cleat_nz {
        for ix in 0..cleat_nx {
            let x = cleat_origin[0] + ix as f64 * cleat_spacing;
            cleat_heights.push(if x.abs() <= 0.035 { 0.055 } else { 0.0 });
        }
    }
    let cleat = HeightfieldTerrain::new(
        cleat_origin,
        cleat_spacing,
        cleat_nx,
        cleat_nz,
        cleat_heights,
        dry.mu_static,
    );
    let cleat_point =
        tire_rig_heightfield_envelope_sweep(&low, &cleat, &dry, load_n, 3.0, 0.05, &[-0.03], 0.0)
            .remove(0);
    snapshots.push(cleat_point.point.snapshot_on_terrain(
        "cleat_envelope_contact",
        &config.low_pressure_tire_preset,
        &config.dry_surface_preset,
        "heightfield_cleat",
        "cpu_heightfield_brush_envelope",
        &cleat,
        cleat_point.x_m,
        cleat_point.z_m,
    ));

    let obstacle_nx = 61;
    let obstacle_nz = 5;
    let obstacle_spacing = 0.035;
    let obstacle_origin = [-0.75, -0.07];
    let mut obstacle_heights = Vec::with_capacity(obstacle_nx * obstacle_nz);
    for _iz in 0..obstacle_nz {
        for ix in 0..obstacle_nx {
            let x = obstacle_origin[0] + ix as f64 * obstacle_spacing;
            obstacle_heights.push(0.07 * (-(x / 0.12).powi(2)).exp());
        }
    }
    let obstacle = HeightfieldTerrain::new(
        obstacle_origin,
        obstacle_spacing,
        obstacle_nx,
        obstacle_nz,
        obstacle_heights,
        dry.mu_static,
    );
    let rolling_params = TetraTireFemParams::realtime_viewer()
        .with_resolution(16, 3, 3)
        .with_pressure_and_load(105_000.0, 4_800.0);
    let rolling_speed_m_per_s = 2.2;
    let rolling_slip_ratio = 0.05;
    let rolling_sequence = tire_rig_rolling_obstacle_sequence(
        rolling_params,
        &obstacle,
        &dry,
        -0.38,
        0.0,
        rolling_speed_m_per_s,
        rolling_slip_ratio,
        config.fixed_dt_s.min(1.0 / 120.0),
        43,
    );
    if let (Some(approach), Some(exit)) = (rolling_sequence.first(), rolling_sequence.last()) {
        let crest = rolling_sequence
            .iter()
            .max_by(|a, b| {
                a.terrain_height_m
                    .partial_cmp(&b.terrain_height_m)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(approach);
        snapshots.push(approach.snapshot(
            "rolling_obstacle_approach",
            &config.low_pressure_tire_preset,
            &config.dry_surface_preset,
            "heightfield_gaussian_bump",
            "cpu_tetra_fem_rolling_obstacle_sequence",
            rolling_params.pressure_pa,
            rolling_params.downforce,
            rolling_speed_m_per_s,
            rolling_slip_ratio,
        ));
        snapshots.push(crest.snapshot(
            "rolling_obstacle_crest",
            &config.low_pressure_tire_preset,
            &config.dry_surface_preset,
            "heightfield_gaussian_bump",
            "cpu_tetra_fem_rolling_obstacle_sequence",
            rolling_params.pressure_pa,
            rolling_params.downforce,
            rolling_speed_m_per_s,
            rolling_slip_ratio,
        ));
        snapshots.push(exit.snapshot(
            "rolling_obstacle_exit",
            &config.low_pressure_tire_preset,
            &config.dry_surface_preset,
            "heightfield_gaussian_bump",
            "cpu_tetra_fem_rolling_obstacle_sequence",
            rolling_params.pressure_pa,
            rolling_params.downforce,
            rolling_speed_m_per_s,
            rolling_slip_ratio,
        ));
    }

    let thermal = tire_rig_thermal_wear_cycle(
        &base,
        &dry_terrain,
        &dry,
        config.fixed_dt_s,
        &[
            TireRigThermalWearPhase::sustained_slip(1.0, load_n, 12.0, 0.20),
            TireRigThermalWearPhase::cooling_recovery(0.5, load_n),
        ],
    );
    let thermal_heat_peak = thermal
        .iter()
        .filter(|point| point.phase_index == 0)
        .max_by(|a, b| {
            a.thermal
                .surface_c
                .partial_cmp(&b.thermal.surface_c)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();
    let thermal_cooling_final = thermal.last().cloned();
    if let Some(heat_peak) = &thermal_heat_peak {
        let point = TireRigSweepPoint {
            load_n,
            pressure_pa: base.pressure_pa,
            hub_height_m: base.radius,
            deflection_m: 0.0,
            speed_m_per_s: 12.0,
            slip_ratio_command: 0.20,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: heat_peak.telemetry.clone(),
        };
        snapshots.push(
            point
                .snapshot_on_terrain(
                    "thermal_wear_heat_peak",
                    &config.tire_preset,
                    &config.dry_surface_preset,
                    "flat",
                    "cpu_brush_layered_thermal_wear",
                    &dry_terrain,
                    0.0,
                    0.0,
                )
                .with_thermal_wear(&heat_peak.thermal, &heat_peak.wear),
        );
    }
    if let Some(cooling_final) = &thermal_cooling_final {
        let point = TireRigSweepPoint {
            load_n,
            pressure_pa: base.pressure_pa,
            hub_height_m: base.radius,
            deflection_m: 0.0,
            speed_m_per_s: 0.0,
            slip_ratio_command: 0.0,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: cooling_final.telemetry.clone(),
        };
        snapshots.push(
            point
                .snapshot_on_terrain(
                    "thermal_wear_cooling_recovery",
                    &config.tire_preset,
                    &config.dry_surface_preset,
                    "flat",
                    "cpu_brush_layered_thermal_wear",
                    &dry_terrain,
                    0.0,
                    0.0,
                )
                .with_thermal_wear(&cooling_final.thermal, &cooling_final.wear),
        );
    }
    if let Some(last) = thermal.last() {
        let point = TireRigSweepPoint {
            load_n,
            pressure_pa: base.pressure_pa,
            hub_height_m: base.radius,
            deflection_m: 0.0,
            speed_m_per_s: 0.0,
            slip_ratio_command: 0.0,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: last.telemetry.clone(),
        };
        snapshots.push(
            point
                .snapshot_on_terrain(
                    "thermal_wear_cycle",
                    &config.tire_preset,
                    &config.dry_surface_preset,
                    "flat",
                    "cpu_brush_layered_thermal_wear",
                    &dry_terrain,
                    0.0,
                    0.0,
                )
                .with_thermal_wear(&last.thermal, &last.wear),
        );
    }

    let fem_params = TetraTireFemParams {
        xpbd_volume_iterations: 2,
        xpbd_structural_iterations: 3,
        ..TetraTireFemParams::realtime_viewer()
            .with_resolution(16, 4, 4)
            .with_pressure_and_load(155_000.0, load_n)
    };
    let mut fem_tire = TetraTireFem::new(0.0, fem_params.pressure_pa, fem_params);
    let fem_dt = config.fixed_dt_s.min(1.0 / 120.0);
    for _ in 0..3 {
        fem_tire.step_on_terrain(&dry_terrain, fem_dt);
    }
    let fem_speed = 6.0;
    let fem_slip = 0.08;
    let fem_state = RealWheelState {
        hub_pos: fem_tire.hub(),
        hub_vel: [fem_speed, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: fem_speed * (1.0 + fem_slip) / fem_tire.outer_radius().max(1.0e-6),
    };
    let fem_preview_contact = fem_tire.evaluate_tread_contact(&fem_state, &dry_terrain, &dry);
    let fem_contact_source_nodes: Vec<usize> = fem_preview_contact
        .cells
        .iter()
        .filter_map(|cell| cell.node_index)
        .collect();
    let fem_avg_contact_node_vx = |tire: &TetraTireFem, node_ids: &[usize]| -> f64 {
        if node_ids.is_empty() {
            0.0
        } else {
            node_ids
                .iter()
                .map(|&node_id| tire.vel[node_id][0])
                .sum::<f64>()
                / node_ids.len() as f64
        }
    };
    let fem_contact_node_vx_before = fem_avg_contact_node_vx(&fem_tire, &fem_contact_source_nodes);
    let fem_contact = fem_tire.apply_tread_contact_wrench(&fem_state, &dry_terrain, &dry, fem_dt);
    let fem_contact_node_vx_after = fem_avg_contact_node_vx(&fem_tire, &fem_contact_source_nodes);
    let fem_contact_feedback_delta_vx = fem_contact_node_vx_after - fem_contact_node_vx_before;
    let fem_telemetry = fem_contact.telemetry_sample(fem_dt);
    let fem_substeps_per_frame = (fem_dt / 1.25e-4).ceil().max(1.0) as usize;
    let fem_budget = fem_tire.solver_budget_telemetry(fem_substeps_per_frame, &fem_telemetry);
    let fem_solver_snapshot = TireRigSweepPoint {
        load_n: fem_tire.downforce(),
        pressure_pa: fem_tire.pressure(),
        hub_height_m: fem_tire.hub()[1],
        deflection_m: fem_tire.hub_deflection_m(),
        speed_m_per_s: fem_speed,
        slip_ratio_command: fem_slip,
        slip_angle_command_rad: 0.0,
        camber_angle_command_rad: 0.0,
        telemetry: fem_telemetry,
    }
    .snapshot_on_terrain(
        "fem_solver_budget",
        &config.tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_tetra_fem_solver_budget",
        &dry_terrain,
        0.0,
        0.0,
    )
    .with_solver_budget(fem_budget);
    snapshots.push(fem_solver_snapshot.clone());
    let solver_stiffness = tetra_fem_solver_stiffness_report();
    let solver_stiffness_snapshot = TireRigSnapshot {
        scenario_id: "fem_solver_stiffness_stabilizer".to_owned(),
        physics_path: "cpu_tetra_fem_semi_implicit_stiffness".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(solver_stiffness_snapshot.clone());
    let diagonal_newton = tetra_fem_diagonal_newton_report();
    let diagonal_newton_snapshot = TireRigSnapshot {
        scenario_id: "fem_diagonal_newton_solver".to_owned(),
        physics_path: "cpu_scalar_backward_euler_newton_tire_mode".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(diagonal_newton_snapshot.clone());
    let element_newton = tetra_fem_element_newton_report();
    let element_newton_snapshot = TireRigSnapshot {
        scenario_id: "fem_element_newton_solver".to_owned(),
        physics_path: "cpu_coupled_tetra_element_backward_euler_newton".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(element_newton_snapshot.clone());
    let patch_newton = tetra_fem_patch_newton_report();
    let patch_newton_snapshot = TireRigSnapshot {
        scenario_id: "fem_patch_newton_solver".to_owned(),
        physics_path: "cpu_multi_tetra_patch_backward_euler_newton".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(patch_newton_snapshot.clone());
    let tiny_mesh_newton = tetra_fem_tiny_mesh_newton_report();
    let tiny_mesh_newton_snapshot = TireRigSnapshot {
        scenario_id: "fem_tiny_mesh_newton_solver".to_owned(),
        physics_path: "cpu_tiny_tire_mesh_backward_euler_newton".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(tiny_mesh_newton_snapshot.clone());
    let matrix_free_newton = tetra_fem_matrix_free_newton_scale_report();
    let matrix_free_newton_snapshot = TireRigSnapshot {
        scenario_id: "fem_matrix_free_newton_scale".to_owned(),
        physics_path: "cpu_matrix_free_newton_jv_scale_probe".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(matrix_free_newton_snapshot.clone());
    let matrix_free_newton_solve = tetra_fem_matrix_free_newton_solve_report();
    let matrix_free_newton_solve_snapshot = TireRigSnapshot {
        scenario_id: "fem_matrix_free_newton_solve".to_owned(),
        physics_path: "cpu_matrix_free_newton_gmres_correction".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(matrix_free_newton_solve_snapshot.clone());
    let matrix_free_newton_preconditioner = tetra_fem_matrix_free_newton_preconditioner_report();
    let matrix_free_newton_preconditioner_snapshot = TireRigSnapshot {
        scenario_id: "fem_matrix_free_newton_preconditioner".to_owned(),
        physics_path: "cpu_matrix_free_newton_jacobi_preconditioner".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(matrix_free_newton_preconditioner_snapshot.clone());
    let matrix_free_newton_step = tetra_fem_matrix_free_newton_step_report();
    let matrix_free_newton_step_snapshot = TireRigSnapshot {
        scenario_id: "fem_matrix_free_newton_step".to_owned(),
        physics_path: "cpu_matrix_free_newton_preconditioned_implicit_step".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(matrix_free_newton_step_snapshot.clone());
    let matrix_free_newton_timestep = tetra_fem_matrix_free_newton_timestep_report();
    let matrix_free_newton_timestep_snapshot = TireRigSnapshot {
        scenario_id: "fem_matrix_free_newton_timestep".to_owned(),
        physics_path: "cpu_tetra_fem_matrix_free_newton_debug_timestep".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(matrix_free_newton_timestep_snapshot.clone());
    if let Some(soil_snapshot) = snapshots
        .iter()
        .find(|snapshot| snapshot.scenario_id == "loose_soil_slip")
        .cloned()
    {
        snapshots.push(TireRigSnapshot {
            scenario_id: "soil_normal_shear_coupling".to_owned(),
            physics_path: "cpu_bekker_deformable_soil_normal_shear".to_owned(),
            ..soil_snapshot.clone()
        });
        snapshots.push(TireRigSnapshot {
            scenario_id: "dem_soil_feedback".to_owned(),
            physics_path: "cpu_dem_tire_contact_feedback".to_owned(),
            ..soil_snapshot
        });
    }
    if let Some(soil_snapshot) = snapshots
        .iter()
        .find(|snapshot| snapshot.scenario_id == "loose_soil_slip")
        .cloned()
    {
        snapshots.push(TireRigSnapshot {
            scenario_id: "mpm_soil_feedback".to_owned(),
            physics_path: "cpu_mpm_tire_contact_feedback".to_owned(),
            ..soil_snapshot
        });
    }

    let mut reports = Vec::new();
    for snapshot in &snapshots {
        reports.push(snapshot.clone().dimensionless_sanity_report(base.radius));
        reports.push(snapshot.clone().energy_sign_report());
    }
    if let Some(thermal_snapshot) = snapshots
        .iter()
        .find(|snapshot| snapshot.scenario_id == "thermal_wear_cycle")
    {
        reports.push(TireRigAcceptanceReport::new(
            thermal_snapshot.clone(),
            tire_thermal_condition_sanity_report(&thermal).acceptance_checks(),
        ));
    }
    if let (Some(heat_peak), Some(heat_peak_snapshot)) = (
        thermal_heat_peak.as_ref(),
        snapshots
            .iter()
            .find(|snapshot| snapshot.scenario_id == "thermal_wear_heat_peak"),
    ) {
        reports.push(TireRigAcceptanceReport::new(
            heat_peak_snapshot.clone(),
            vec![
                TireRigAcceptanceCheck::at_least(
                    "thermal_heat_peak_above_ambient",
                    "sustained slip should raise surface temperature above ambient",
                    "surface_minus_ambient_c",
                    heat_peak.thermal.surface_c - heat_peak.thermal.ambient_c,
                    0.0,
                ),
                TireRigAcceptanceCheck::at_least(
                    "thermal_heat_peak_has_wear",
                    "sustained slip should not reduce accumulated tread wear",
                    "tread_wear_depth_m",
                    heat_peak.wear.depth_m,
                    0.0,
                ),
                TireRigAcceptanceCheck::at_least(
                    "thermal_heat_peak_dissipates_energy",
                    "sliding heat peak should carry non-negative dissipated power",
                    "dissipated_power_w",
                    heat_peak.telemetry.dissipated_power_w,
                    0.0,
                ),
            ],
        ));
    }
    if let (Some(heat_peak), Some(cooling_final), Some(cooling_snapshot)) = (
        thermal_heat_peak.as_ref(),
        thermal_cooling_final.as_ref(),
        snapshots
            .iter()
            .find(|snapshot| snapshot.scenario_id == "thermal_wear_cooling_recovery"),
    ) {
        reports.push(TireRigAcceptanceReport::new(
            cooling_snapshot.clone(),
            vec![
                TireRigAcceptanceCheck::at_least(
                    "thermal_cooling_reduces_surface_temperature",
                    "cooling replay should reduce surface temperature from the heat peak",
                    "heat_peak_minus_final_surface_c",
                    heat_peak.thermal.surface_c - cooling_final.thermal.surface_c,
                    0.0,
                ),
                TireRigAcceptanceCheck::at_least(
                    "thermal_cooling_does_not_heal_wear",
                    "cooling replay should preserve accumulated tread wear",
                    "final_minus_peak_wear_depth_m",
                    cooling_final.wear.depth_m - heat_peak.wear.depth_m,
                    0.0,
                ),
                TireRigAcceptanceCheck::at_most(
                    "thermal_cooling_zero_slip_command",
                    "cooling replay snapshot should be a zero-slip coast/cool state",
                    "abs_slip_ratio_command",
                    cooling_snapshot.slip_ratio_command.abs(),
                    1.0e-9,
                ),
            ],
        ));
    }
    if let Some(dem_snapshot) = snapshots
        .iter()
        .find(|snapshot| snapshot.scenario_id == "soil_normal_shear_coupling")
    {
        reports.push(TireRigAcceptanceReport::new(
            dem_snapshot.clone(),
            tire_soil_normal_shear_coupling_report().acceptance_checks(),
        ));
    }
    if let Some(dem_snapshot) = snapshots
        .iter()
        .find(|snapshot| snapshot.scenario_id == "dem_soil_feedback")
    {
        reports.push(TireRigAcceptanceReport::new(
            dem_snapshot.clone(),
            tire_dem_soil_feedback_report().acceptance_checks(),
        ));
    }
    if let Some(mpm_snapshot) = snapshots
        .iter()
        .find(|snapshot| snapshot.scenario_id == "mpm_soil_feedback")
    {
        reports.push(TireRigAcceptanceReport::new(
            mpm_snapshot.clone(),
            tire_mpm_soil_feedback_report().acceptance_checks(),
        ));
    }
    reports.push(TireRigAcceptanceReport::new(
        fem_solver_snapshot.clone(),
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_contact_node_sources_present",
                "FEM contact cells should come from actual tread nodes, not idealized wheel samples",
                "node_sourced_contact_count",
                fem_contact_source_nodes.len() as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_multicell_patch",
                "FEM contact should retain a distributed patch instead of collapsing to one point",
                "contact_count",
                fem_contact.contact_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_feedback_accelerates_tread",
                "drive-slip FEM contact feedback should push source tread-node velocities forward",
                "contact_node_vx_delta_m_per_s",
                fem_contact_feedback_delta_vx,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_contact_shared_friction_budget",
                "FEM tread feedback should stay inside the shared material friction budget",
                "consumed_over_available",
                fem_contact.consumed_friction_budget
                    / fem_contact.available_friction_budget.max(1.0),
                1.05,
            ),
        ],
    ));
    reports.push(TireRigAcceptanceReport::new(
        solver_stiffness_snapshot,
        solver_stiffness.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        diagonal_newton_snapshot,
        diagonal_newton.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        element_newton_snapshot,
        element_newton.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        patch_newton_snapshot,
        patch_newton.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        tiny_mesh_newton_snapshot,
        tiny_mesh_newton.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        matrix_free_newton_snapshot,
        matrix_free_newton.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        matrix_free_newton_solve_snapshot,
        matrix_free_newton_solve.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        matrix_free_newton_preconditioner_snapshot,
        matrix_free_newton_preconditioner.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        matrix_free_newton_step_snapshot,
        matrix_free_newton_step.acceptance_checks(),
    ));
    reports.push(TireRigAcceptanceReport::new(
        matrix_free_newton_timestep_snapshot,
        matrix_free_newton_timestep.acceptance_checks(),
    ));
    let deformation_force = tetra_fem_deformation_force_report();
    let deformation_force_snapshot = TireRigSnapshot {
        scenario_id: "fem_deformation_force_shape".to_owned(),
        physics_path: "cpu_tetra_fem_deformation_force_shape".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(deformation_force_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        deformation_force_snapshot,
        deformation_force.acceptance_checks(),
    ));
    let contact_slip_patch = tetra_fem_contact_slip_patch_report();
    let contact_slip_patch_snapshot = TireRigSnapshot {
        scenario_id: "fem_contact_slip_patch".to_owned(),
        physics_path: "cpu_tetra_fem_contact_slip_patch".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(contact_slip_patch_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        contact_slip_patch_snapshot,
        contact_slip_patch.acceptance_checks(),
    ));
    if let (Some(approach), Some(crest), Some(exit)) = (
        snapshots
            .iter()
            .find(|snapshot| snapshot.scenario_id == "rolling_obstacle_approach"),
        snapshots
            .iter()
            .find(|snapshot| snapshot.scenario_id == "rolling_obstacle_crest"),
        snapshots
            .iter()
            .find(|snapshot| snapshot.scenario_id == "rolling_obstacle_exit"),
    ) {
        reports.push(TireRigAcceptanceReport::new(
            TireRigSnapshot {
                scenario_id: "rolling_obstacle_sequence".to_owned(),
                physics_path: "cpu_tetra_fem_rolling_obstacle_sequence".to_owned(),
                ..crest.clone()
            },
            vec![
                TireRigAcceptanceCheck::at_least(
                    "rolling_obstacle_crest_above_approach",
                    "ordered rolling obstacle should climb from approach to bump crest",
                    "crest_minus_approach_height_m",
                    crest.terrain_height_m - approach.terrain_height_m,
                    0.04,
                ),
                TireRigAcceptanceCheck::at_least(
                    "rolling_obstacle_crest_above_exit",
                    "ordered rolling obstacle should descend again after the crest",
                    "crest_minus_exit_height_m",
                    crest.terrain_height_m - exit.terrain_height_m,
                    0.04,
                ),
                TireRigAcceptanceCheck::at_least(
                    "rolling_obstacle_multi_cell_patch",
                    "crest contact should remain distributed instead of collapsing to one FEM node",
                    "crest_contact_count",
                    crest.telemetry.patch.contact_count as f64,
                    2.0,
                ),
                TireRigAcceptanceCheck::at_least(
                    "rolling_obstacle_patch_length",
                    "crest contact patch should stay extended while the tire wraps the bump",
                    "crest_patch_length_m",
                    crest.telemetry.patch.patch_length_m,
                    0.035,
                ),
                TireRigAcceptanceCheck::at_least(
                    "rolling_obstacle_horizontal_force",
                    "non-flat bump normals should produce a visible horizontal tire force",
                    "crest_abs_force_x_n",
                    crest.telemetry.force_n[0].abs(),
                    10.0,
                ),
            ],
        ));
    }
    reports.push(TireRigAcceptanceReport::new(
        TireRigSnapshot {
            scenario_id: "curb_envelope_contact".to_owned(),
            physics_path: "cpu_heightfield_brush_envelope".to_owned(),
            ..snapshots
                .iter()
                .find(|snapshot| snapshot.scenario_id == "curb_envelope_contact")
                .cloned()
                .unwrap_or_else(|| {
                    curb_point.point.snapshot_on_terrain(
                        "curb_envelope_contact",
                        &config.low_pressure_tire_preset,
                        &config.dry_surface_preset,
                        "heightfield_curb",
                        "cpu_heightfield_brush_envelope",
                        &curb,
                        curb_point.x_m,
                        curb_point.z_m,
                    )
                })
        },
        vec![
            TireRigAcceptanceCheck::at_least(
                "curb_envelope_nonvertical_normal",
                "heightfield curb should expose a non-vertical contact normal",
                "abs_terrain_normal_x",
                curb_point.terrain_normal[0].abs(),
                0.30,
            ),
            TireRigAcceptanceCheck::at_least(
                "curb_envelope_multicell_patch",
                "curb contact should remain distributed instead of collapsing to one point",
                "low_pressure_contact_count",
                curb_point.point.telemetry.patch.contact_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "curb_envelope_patch_length",
                "low-pressure tire should keep an extended contact patch over the curb edge",
                "low_pressure_patch_length_m",
                curb_point.point.telemetry.patch.patch_length_m,
                0.04,
            ),
            TireRigAcceptanceCheck::at_least(
                "curb_envelope_horizontal_force",
                "curb normal should feed a visible horizontal tire force",
                "abs_force_x_n",
                curb_point.point.telemetry.force_n[0].abs(),
                50.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "curb_low_pressure_conforms_more_than_high",
                "low pressure should conform to the curb with a longer patch than high pressure",
                "low_minus_high_patch_length_m",
                curb_point.point.telemetry.patch.patch_length_m
                    - high_curb_point.point.telemetry.patch.patch_length_m,
                1.0e-4,
            ),
        ],
    ));
    reports.push(TireRigAcceptanceReport::new(
        TireRigSnapshot {
            scenario_id: "cleat_envelope_contact".to_owned(),
            physics_path: "cpu_heightfield_brush_envelope".to_owned(),
            ..snapshots
                .iter()
                .find(|snapshot| snapshot.scenario_id == "cleat_envelope_contact")
                .cloned()
                .unwrap_or_else(|| {
                    cleat_point.point.snapshot_on_terrain(
                        "cleat_envelope_contact",
                        &config.low_pressure_tire_preset,
                        &config.dry_surface_preset,
                        "heightfield_cleat",
                        "cpu_heightfield_brush_envelope",
                        &cleat,
                        cleat_point.x_m,
                        cleat_point.z_m,
                    )
                })
        },
        vec![
            TireRigAcceptanceCheck::at_least(
                "cleat_envelope_raised_contact",
                "heightfield cleat should present a raised obstacle to the tire patch",
                "terrain_height_m",
                cleat_point.terrain_height_m,
                0.020,
            ),
            TireRigAcceptanceCheck::at_least(
                "cleat_envelope_multicell_patch",
                "cleat contact should stay distributed instead of collapsing to one point",
                "contact_count",
                cleat_point.point.telemetry.patch.contact_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "cleat_envelope_patch_length",
                "low-pressure tire should keep an extended patch while climbing a narrow cleat",
                "patch_length_m",
                cleat_point.point.telemetry.patch.patch_length_m,
                0.035,
            ),
            TireRigAcceptanceCheck::at_least(
                "cleat_envelope_horizontal_force",
                "cleat edge should feed a visible horizontal tire force",
                "abs_force_x_n",
                cleat_point.point.telemetry.force_n[0].abs(),
                10.0,
            ),
        ],
    ));
    let bead_sanity = tetra_cavity_bead_sanity_report();
    let bead_snapshot = TireRigSnapshot {
        scenario_id: "cavity_bead_sanity".to_owned(),
        physics_path: "cpu_tetra_fem_cavity_bead_sanity".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(bead_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        bead_snapshot,
        bead_sanity.acceptance_checks(),
    ));
    let cavity_coupling = tetra_cavity_coupling_effect_report();
    let cavity_coupling_snapshot = TireRigSnapshot {
        scenario_id: "cavity_coupling_effect".to_owned(),
        physics_path: "cpu_tetra_fem_cavity_coupling_effect".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(cavity_coupling_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        cavity_coupling_snapshot,
        cavity_coupling.acceptance_checks(),
    ));
    let bead_load_cases = tetra_bead_seat_load_case_report();
    let bead_load_snapshot = TireRigSnapshot {
        scenario_id: "bead_seat_load_cases".to_owned(),
        physics_path: "cpu_tetra_fem_bead_seat_load_cases".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(bead_load_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        bead_load_snapshot,
        bead_load_cases.acceptance_checks(),
    ));
    let sidewall_validation_params = TetraTireFemParams::realtime_viewer()
        .with_resolution(20, 5, 5)
        .with_pressure_and_load(base.pressure_pa, load_n);
    let sidewall_validation_tire = TetraTireFem::new(
        0.0,
        sidewall_validation_params.pressure_pa,
        sidewall_validation_params,
    );
    let sidewall_snapshot = TireRigSnapshot {
        scenario_id: "sidewall_laminate_sanity".to_owned(),
        physics_path: "cpu_tetra_fem_sidewall_laminate_sanity".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(sidewall_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        sidewall_snapshot,
        sidewall_validation_tire
            .sidewall_laminate_sanity_report()
            .acceptance_checks(),
    ));
    let sidewall_dynamic = tetra_sidewall_dynamic_load_report();
    let sidewall_dynamic_snapshot = TireRigSnapshot {
        scenario_id: "sidewall_dynamic_load_cases".to_owned(),
        physics_path: "cpu_tetra_fem_sidewall_dynamic_load_cases".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(sidewall_dynamic_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        sidewall_dynamic_snapshot,
        sidewall_dynamic.acceptance_checks(),
    ));
    let sidewall_coupling = tetra_sidewall_laminate_coupling_report();
    let sidewall_coupling_snapshot = TireRigSnapshot {
        scenario_id: "sidewall_laminate_coupling".to_owned(),
        physics_path: "cpu_tetra_fem_sidewall_laminate_coupling".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(sidewall_coupling_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        sidewall_coupling_snapshot,
        sidewall_coupling.acceptance_checks(),
    ));
    let sidewall_ply_xpbd = tetra_sidewall_ply_strain_xpbd_report();
    let sidewall_ply_xpbd_snapshot = TireRigSnapshot {
        scenario_id: "sidewall_ply_strain_xpbd".to_owned(),
        physics_path: "cpu_tetra_fem_sidewall_ply_strain_xpbd".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(sidewall_ply_xpbd_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        sidewall_ply_xpbd_snapshot,
        sidewall_ply_xpbd.acceptance_checks(),
    ));
    let sidewall_dihedral_xpbd = tetra_sidewall_dihedral_xpbd_effect_report();
    let sidewall_dihedral_xpbd_snapshot = TireRigSnapshot {
        scenario_id: "sidewall_dihedral_xpbd_effect".to_owned(),
        physics_path: "cpu_tetra_fem_sidewall_dihedral_xpbd_effect".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(sidewall_dihedral_xpbd_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        sidewall_dihedral_xpbd_snapshot,
        sidewall_dihedral_xpbd.acceptance_checks(),
    ));
    let construction_sanity = tetra_construction_sanity_report();
    let construction_snapshot = TireRigSnapshot {
        scenario_id: "construction_sanity".to_owned(),
        physics_path: "cpu_tetra_fem_construction_sanity".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(construction_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        construction_snapshot,
        construction_sanity.acceptance_checks(),
    ));
    let realism_sources = tire_realism_source_classification_report();
    let realism_snapshot = TireRigSnapshot {
        scenario_id: "realism_source_classification".to_owned(),
        physics_path: "cpu_report_realism_source_classification".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(realism_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        realism_snapshot,
        realism_sources.acceptance_checks(),
    ));
    let parameter_observability = tire_parameter_observability_report();
    let parameter_observability_snapshot = TireRigSnapshot {
        scenario_id: "parameter_observability".to_owned(),
        physics_path: "cpu_gpu_parameter_observability_audit".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(parameter_observability_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        parameter_observability_snapshot,
        parameter_observability.acceptance_checks(),
    ));
    let triangle_seam = tire_triangle_seam_manifold_report();
    let triangle_snapshot = TireRigSnapshot {
        scenario_id: "triangle_seam_manifold".to_owned(),
        physics_path: "cpu_triangle_mesh_tread_manifold_sanity".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(triangle_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        triangle_snapshot,
        triangle_seam.acceptance_checks(),
    ));
    let triangle_surface = tire_triangle_surface_stress_report();
    let triangle_surface_snapshot = TireRigSnapshot {
        scenario_id: "triangle_surface_stress".to_owned(),
        physics_path: "cpu_triangle_surface_stress".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(triangle_surface_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        triangle_surface_snapshot,
        triangle_surface.acceptance_checks(),
    ));
    let triangle_force = tire_triangle_surface_force_report();
    let triangle_force_snapshot = TireRigSnapshot {
        scenario_id: "triangle_surface_force".to_owned(),
        physics_path: "cpu_triangle_surface_force_bristle".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(triangle_force_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        triangle_force_snapshot,
        triangle_force.acceptance_checks(),
    ));
    let triangle_density = tire_triangle_density_convergence_report();
    let triangle_density_snapshot = TireRigSnapshot {
        scenario_id: "triangle_density_convergence".to_owned(),
        physics_path: "cpu_triangle_density_convergence".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(triangle_density_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        triangle_density_snapshot,
        triangle_density.acceptance_checks(),
    ));
    let triangle_large = tire_triangle_large_manifold_report();
    let triangle_large_snapshot = TireRigSnapshot {
        scenario_id: "triangle_large_manifold".to_owned(),
        physics_path: "cpu_triangle_large_manifold_fem_feedback".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(triangle_large_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        triangle_large_snapshot,
        triangle_large.acceptance_checks(),
    ));
    let low_speed_robustness = tire_low_speed_robustness_report();
    let low_speed_snapshot = TireRigSnapshot {
        scenario_id: "low_speed_robustness".to_owned(),
        physics_path: "cpu_brush_transient_low_speed".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(low_speed_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        low_speed_snapshot,
        low_speed_robustness.acceptance_checks(),
    ));
    let frame_convention = tire_frame_convention_report();
    let frame_snapshot = TireRigSnapshot {
        scenario_id: "frame_sign_convention".to_owned(),
        physics_path: "cpu_brush_frame_sign_convention".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(frame_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        frame_snapshot,
        frame_convention.acceptance_checks(),
    ));
    let patch_convergence = tire_contact_patch_convergence_report();
    let patch_convergence_snapshot = TireRigSnapshot {
        scenario_id: "contact_patch_convergence".to_owned(),
        physics_path: "cpu_brush_patch_segment_convergence".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(patch_convergence_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        patch_convergence_snapshot,
        patch_convergence.acceptance_checks(),
    ));
    let fem_substep_convergence = tetra_fem_substep_convergence_report();
    let fem_substep_snapshot = TireRigSnapshot {
        scenario_id: "fem_substep_convergence".to_owned(),
        physics_path: "cpu_tetra_fem_substep_convergence".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(fem_substep_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        fem_substep_snapshot,
        fem_substep_convergence.acceptance_checks(),
    ));
    let heightfield_resolution = tire_heightfield_resolution_convergence_report();
    let heightfield_resolution_snapshot = TireRigSnapshot {
        scenario_id: "heightfield_resolution_convergence".to_owned(),
        physics_path: "cpu_heightfield_resolution_convergence".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(heightfield_resolution_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        heightfield_resolution_snapshot,
        heightfield_resolution.acceptance_checks(),
    ));
    let thermal_wear_convergence = tire_thermal_wear_convergence_report();
    let thermal_wear_convergence_point =
        tire_rig_point(&base, &dry_terrain, &dry, load_n, 12.0, 0.16, 0.0, 0.0);
    let thermal_wear_convergence_snapshot = thermal_wear_convergence_point.snapshot_on_terrain(
        "thermal_wear_convergence",
        &config.tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_thermal_wear_convergence",
        &dry_terrain,
        0.0,
        0.0,
    );
    snapshots.push(thermal_wear_convergence_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        thermal_wear_convergence_snapshot,
        thermal_wear_convergence.acceptance_checks(),
    ));
    let road_condition_state = tire_road_condition_state_report();
    let road_condition_snapshot = TireRigSnapshot {
        scenario_id: "road_condition_state".to_owned(),
        physics_path: "cpu_brush_road_condition_state".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(road_condition_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        road_condition_snapshot,
        road_condition_state.acceptance_checks(),
    ));
    let layer_ablation = tire_layer_ablation_debug_report();
    let layer_ablation_snapshot = TireRigSnapshot {
        scenario_id: "layer_ablation_debug".to_owned(),
        physics_path: "cpu_brush_fem_thermal_soil_layer_ablation_debug".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(layer_ablation_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        layer_ablation_snapshot,
        layer_ablation.acceptance_checks(),
    ));
    let runtime_gate = tire_runtime_production_gate_report();
    let runtime_gate_snapshot = TireRigSnapshot {
        scenario_id: "runtime_production_gate".to_owned(),
        physics_path: "gpu_first_runtime_gate_checklist".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(runtime_gate_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        runtime_gate_snapshot,
        runtime_gate.acceptance_checks(),
    ));
    let thermal_abuse = tire_thermal_abuse_envelope_report();
    let thermal_abuse_snapshot = TireRigSnapshot {
        scenario_id: "thermal_abuse_envelope".to_owned(),
        physics_path: "cpu_thermal_wear_abuse_envelope".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(thermal_abuse_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        thermal_abuse_snapshot,
        thermal_abuse.acceptance_checks(),
    ));
    let wear_power_temperature = tire_wear_power_temperature_report();
    let wear_power_temperature_wear = TireWearState {
        depth_m: wear_power_temperature.high_power_hot_wear_m,
        ..Default::default()
    };
    let wear_power_temperature_thermal = TireThermalLayers {
        surface_c: wear_power_temperature.hot_surface_c,
        tread_c: wear_power_temperature.hot_surface_c,
        carcass_c: 95.0,
        rim_c: 55.0,
        ..TireThermalLayers::passenger_road()
    };
    let wear_power_temperature_point = TireRigSweepPoint {
        load_n,
        pressure_pa: base.pressure_pa,
        hub_height_m: base.radius,
        deflection_m: 0.0,
        speed_m_per_s: 12.0,
        slip_ratio_command: 0.20,
        slip_angle_command_rad: 0.0,
        camber_angle_command_rad: 0.0,
        telemetry: dry_slip.telemetry.clone(),
    };
    let wear_power_temperature_snapshot = wear_power_temperature_point
        .snapshot_on_terrain(
            "thermal_wear_power_temperature",
            &config.tire_preset,
            &config.dry_surface_preset,
            "flat",
            "cpu_thermal_wear_power_temperature_sensitivity",
            &dry_terrain,
            0.0,
            0.0,
        )
        .with_thermal_wear(
            &wear_power_temperature_thermal,
            &wear_power_temperature_wear,
        );
    snapshots.push(wear_power_temperature_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        wear_power_temperature_snapshot,
        wear_power_temperature.acceptance_checks(),
    ));
    let worn_surface = tire_worn_surface_sensitivity_report();
    let worn_surface_point = TireRigSweepPoint {
        load_n,
        pressure_pa: base.pressure_pa,
        hub_height_m: base.radius,
        deflection_m: 0.0,
        speed_m_per_s: 13.0,
        slip_ratio_command: 0.14,
        slip_angle_command_rad: 0.0,
        camber_angle_command_rad: 0.0,
        telemetry: dry_slip.telemetry.clone(),
    };
    let worn_surface_snapshot = worn_surface_point.snapshot_on_terrain(
        "worn_surface_sensitivity",
        &config.tire_preset,
        "dry_wet_ice_loose_soil",
        "flat_surface_matrix",
        "cpu_thermal_wear_surface_sensitivity",
        &dry_terrain,
        0.0,
        0.0,
    );
    snapshots.push(worn_surface_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        worn_surface_snapshot,
        worn_surface.acceptance_checks(),
    ));
    let condition_failure = tire_condition_failure_mode_report();
    let condition_failure_snapshot = TireRigSnapshot {
        scenario_id: "condition_failure_modes".to_owned(),
        physics_path: "cpu_thermal_wear_condition_failure_modes".to_owned(),
        ..fem_solver_snapshot.clone()
    };
    snapshots.push(condition_failure_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        condition_failure_snapshot,
        condition_failure.acceptance_checks(),
    ));
    let thermal_rolling_carcass = tire_thermal_rolling_carcass_report();
    let thermal_rolling_point = tire_rig_point(
        &base,
        &dry_terrain,
        &dry,
        thermal_rolling_carcass.load_n,
        thermal_rolling_carcass.speed_m_per_s,
        0.0,
        0.0,
        0.0,
    );
    let thermal_rolling_snapshot = thermal_rolling_point
        .snapshot_on_terrain(
            "thermal_rolling_carcass",
            &config.tire_preset,
            &config.dry_surface_preset,
            "flat",
            "cpu_thermal_rolling_carcass",
            &dry_terrain,
            0.0,
            0.0,
        )
        .with_thermal_wear(
            &thermal_rolling_carcass.final_thermal,
            &thermal_rolling_carcass.final_wear,
        );
    snapshots.push(thermal_rolling_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        thermal_rolling_snapshot,
        thermal_rolling_carcass.acceptance_checks(),
    ));
    let thermal_brake_rim = tire_thermal_brake_rim_report();
    let thermal_brake_point = tire_rig_point(&base, &dry_terrain, &dry, load_n, 0.0, 0.0, 0.0, 0.0);
    let thermal_brake_snapshot = thermal_brake_point
        .snapshot_on_terrain(
            "thermal_brake_rim",
            &config.tire_preset,
            &config.dry_surface_preset,
            "flat",
            "cpu_thermal_brake_rim",
            &dry_terrain,
            0.0,
            0.0,
        )
        .with_thermal_wear(
            &thermal_brake_rim.final_thermal,
            &thermal_brake_rim.final_wear,
        );
    snapshots.push(thermal_brake_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        thermal_brake_snapshot,
        thermal_brake_rim.acceptance_checks(),
    ));
    let pressure_patch_snapshot = low_load.snapshot_on_terrain(
        "pressure_patch_envelope",
        &config.low_pressure_tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_brush_load_pressure",
        &dry_terrain,
        0.0,
        0.0,
    );
    snapshots.push(pressure_patch_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        pressure_patch_snapshot,
        vec![TireRigAcceptanceCheck::at_least(
            "lower_pressure_grows_patch",
            "lower pressure should lengthen the contact patch under the same load",
            "low_patch_minus_high_patch_m",
            low_load.telemetry.patch.patch_length_m - high_load.telemetry.patch.patch_length_m,
            1.0e-4,
        )],
    ));
    let surface_grip_snapshot = ice_slip.snapshot_on_terrain(
        "surface_grip_envelope",
        &config.tire_preset,
        &config.ice_surface_preset,
        "flat",
        "cpu_brush_longitudinal_slip",
        &ice_terrain,
        0.0,
        0.0,
    );
    snapshots.push(surface_grip_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        surface_grip_snapshot,
        vec![TireRigAcceptanceCheck::at_most(
            "ice_caps_longitudinal_force",
            "ice should cap longitudinal force far below dry asphalt at the same command",
            "ice_over_dry_abs_fx",
            ice_slip.telemetry.longitudinal_force_n.abs()
                / dry_slip.telemetry.longitudinal_force_n.abs().max(1.0),
            0.45,
        )],
    ));
    let ice_large_patch = tire_ice_large_patch_report();
    let ice_large_patch_point =
        tire_rig_point(&low, &ice_terrain, &ice, load_n, 8.0, 0.24, 0.0, 0.0);
    let ice_large_patch_snapshot = ice_large_patch_point.snapshot_on_terrain(
        "ice_large_patch_grip",
        &config.low_pressure_tire_preset,
        &config.ice_surface_preset,
        "flat_ice_large_patch",
        "cpu_brush_ice_large_patch",
        &ice_terrain,
        0.0,
        0.0,
    );
    snapshots.push(ice_large_patch_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        ice_large_patch_snapshot,
        ice_large_patch.acceptance_checks(),
    ));
    let adhesion_sliding = tire_adhesion_sliding_continuity_report();
    let adhesion_sliding_point =
        tire_rig_point(&base, &dry_terrain, &dry, load_n, 11.0, 0.105, 0.075, 0.0);
    let adhesion_sliding_snapshot = adhesion_sliding_point.snapshot_on_terrain(
        "adhesion_sliding_continuity",
        &config.tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_brush_adhesion_sliding_continuity",
        &dry_terrain,
        0.0,
        0.0,
    );
    snapshots.push(adhesion_sliding_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        adhesion_sliding_snapshot,
        adhesion_sliding.acceptance_checks(),
    ));
    let torque_slip = tire_torque_induced_slip_outcome_report();
    let torque_slip_snapshot = torque_slip.snapshot();
    snapshots.push(torque_slip_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        torque_slip_snapshot,
        torque_slip.acceptance_checks(),
    ));
    let wheel_spin_balance = tire_wheel_spin_torque_balance_report();
    let wheel_spin_point = tire_rig_point(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        3.0,
        wheel_spin_balance.slip_ratio,
        0.0,
        0.0,
    );
    let wheel_spin_snapshot = wheel_spin_point.snapshot_on_terrain(
        "wheel_spin_torque_balance",
        &config.tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_brush_wheel_spin_torque_balance",
        &dry_terrain,
        0.0,
        0.0,
    );
    snapshots.push(wheel_spin_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        wheel_spin_snapshot,
        wheel_spin_balance.acceptance_checks(),
    ));
    let vehicle_longitudinal = tire_vehicle_longitudinal_motion_report();
    let vehicle_longitudinal_point = tire_rig_point(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        vehicle_longitudinal.speed_before_m_per_s,
        0.10,
        0.0,
        0.0,
    );
    let vehicle_longitudinal_snapshot = vehicle_longitudinal_point.snapshot_on_terrain(
        "vehicle_longitudinal_motion",
        &config.tire_preset,
        &config.dry_surface_preset,
        "flat",
        "cpu_brush_vehicle_longitudinal_motion",
        &dry_terrain,
        0.0,
        0.0,
    );
    snapshots.push(vehicle_longitudinal_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        vehicle_longitudinal_snapshot,
        vehicle_longitudinal.acceptance_checks(),
    ));
    let dynamic_load_transfer = tire_vehicle_dynamic_load_transfer_report();
    let dynamic_load_transfer_snapshot = dynamic_load_transfer.snapshot();
    snapshots.push(dynamic_load_transfer_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        dynamic_load_transfer_snapshot,
        dynamic_load_transfer.acceptance_checks(),
    ));
    let wet_film_snapshot = wet_slip.snapshot_on_terrain(
        "wet_film_envelope",
        &config.tire_preset,
        &config.wet_surface_preset,
        "flat_wet_film",
        "cpu_brush_surface_film",
        &dry_terrain,
        0.0,
        0.0,
    );
    snapshots.push(wet_film_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        wet_film_snapshot,
        vec![TireRigAcceptanceCheck::at_most(
            "wet_film_reduces_force",
            "wet film should reduce longitudinal force relative to dry asphalt",
            "wet_over_dry_abs_fx",
            wet_slip.telemetry.longitudinal_force_n.abs()
                / dry_slip.telemetry.longitudinal_force_n.abs().max(1.0),
            0.95,
        )],
    ));
    reports.push(TireRigAcceptanceReport::new(
        moving_belt_snapshot,
        vec![TireRigAcceptanceCheck::at_least(
            "moving_ground_work_positive",
            "moving ground should report positive work on the driven tire state",
            "moving_ground_power_w",
            moving_belt_power_w,
            1.0,
        )],
    ));
    reports.push(TireRigAcceptanceReport::new(
        heightfield_velocity_snapshot,
        tire_heightfield_surface_velocity_map_report().acceptance_checks(),
    ));

    let transition_terrain = TireRigSurfaceTransitionTerrain::new(
        0.0,
        dry,
        vec![
            TireRigSurfaceTransitionBand::new("dry_asphalt", -5.0, -3.0, dry),
            TireRigSurfaceTransitionBand::new("ice", -2.5, -0.8, ice),
            TireRigSurfaceTransitionBand::new("wet_asphalt", -0.2, 1.4, dry)
                .with_water_film_depth(0.0020),
            TireRigSurfaceTransitionBand::new(
                "packed_dirt",
                1.8,
                2.5,
                SurfaceMaterial::loose_dirt(),
            ),
            TireRigSurfaceTransitionBand::new("loose_soil", 3.1, 4.0, soil),
            TireRigSurfaceTransitionBand::new("moving_belt", 4.7, 6.2, dry)
                .with_surface_velocity([-4.0, 0.0, 0.0]),
        ],
    )
    .with_transition_width(0.25);
    let transition_sequence = tire_rig_surface_transition_sequence(
        &base,
        &transition_terrain,
        load_n,
        10.0,
        0.16,
        &[-4.0, -2.8, -1.6, -0.35, 0.7, 2.1, 3.5, 4.9, 5.6],
        0.0,
    );
    let find_transition = |needle: &str| {
        transition_sequence
            .iter()
            .find(|p| p.surface_label.contains(needle))
    };
    if let (
        Some(dry_point),
        Some(ice_point),
        Some(wet_point),
        Some(dirt_point),
        Some(soil_point),
        Some(belt_point),
    ) = (
        find_transition("dry_asphalt"),
        find_transition("ice"),
        find_transition("wet_asphalt"),
        find_transition("packed_dirt"),
        find_transition("loose_soil"),
        find_transition("moving_belt"),
    ) {
        let max_transition_jump = transition_sequence
            .windows(2)
            .map(|pair| {
                (pair[1].telemetry.longitudinal_force_n - pair[0].telemetry.longitudinal_force_n)
                    .abs()
            })
            .fold(0.0_f64, f64::max);
        let moving_belt_force_delta_n = (belt_point.telemetry.longitudinal_force_n
            - dry_point.telemetry.longitudinal_force_n)
            .abs();
        let moving_belt_power_w = dot(
            belt_point.telemetry.force_n,
            belt_point.surface_velocity_m_per_s,
        )
        .abs();
        let anchor = TireRigSweepPoint {
            load_n,
            pressure_pa: base.pressure_pa,
            hub_height_m: solve_loaded_tire_hub_height(
                &base,
                &transition_terrain,
                belt_point.x_m,
                belt_point.z_m,
                load_n,
            ),
            deflection_m: 0.0,
            speed_m_per_s: 10.0,
            slip_ratio_command: 0.16,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: belt_point.telemetry.clone(),
        };
        let surface_transition_snapshot = anchor.snapshot_on_terrain(
            "surface_transition_sequence",
            &config.tire_preset,
            "dry_ice_wet_dirt_soil_belt",
            "banded_flat_surface",
            "cpu_brush_surface_transition",
            &transition_terrain,
            belt_point.x_m,
            belt_point.z_m,
        );
        snapshots.push(surface_transition_snapshot.clone());
        reports.push(TireRigAcceptanceReport::new(
            surface_transition_snapshot,
            vec![
                TireRigAcceptanceCheck::at_most(
                    "surface_transition_ice_caps_force",
                    "ice band should cap force below the dry-asphalt band",
                    "ice_over_dry_abs_fx",
                    ice_point.telemetry.longitudinal_force_n.abs()
                        / dry_point.telemetry.longitudinal_force_n.abs().max(1.0),
                    0.45,
                ),
                TireRigAcceptanceCheck::at_most(
                    "surface_transition_wet_reduces_force",
                    "wet-film band should record water and reduce force below dry asphalt",
                    "wet_over_dry_abs_fx",
                    wet_point.telemetry.longitudinal_force_n.abs()
                        / dry_point.telemetry.longitudinal_force_n.abs().max(1.0),
                    0.98,
                ),
                TireRigAcceptanceCheck::at_least(
                    "surface_transition_soil_rolling_loss",
                    "loose-soil band should carry more rolling resistance than dry asphalt",
                    "soil_minus_dry_rolling_resistance",
                    soil_point.material.rolling_resistance - dry_point.material.rolling_resistance,
                    1.0e-6,
                ),
                TireRigAcceptanceCheck::at_least(
                    "surface_transition_packed_dirt_loss",
                    "packed-dirt band should bridge dry asphalt and loose soil rolling loss",
                    "packed_dirt_minus_dry_rolling_resistance",
                    dirt_point.material.rolling_resistance - dry_point.material.rolling_resistance,
                    1.0e-6,
                ),
                TireRigAcceptanceCheck::at_least(
                    "surface_transition_loose_soil_over_packed_dirt_loss",
                    "loose soil should add rolling loss beyond packed dirt",
                    "loose_soil_minus_packed_dirt_rolling_resistance",
                    soil_point.material.rolling_resistance - dirt_point.material.rolling_resistance,
                    1.0e-6,
                ),
                TireRigAcceptanceCheck::at_least(
                    "surface_transition_moving_belt_velocity",
                    "moving-belt band should expose non-zero surface velocity",
                    "abs_surface_velocity_x_m_per_s",
                    belt_point.surface_velocity_m_per_s[0].abs(),
                    1.0,
                ),
                TireRigAcceptanceCheck::at_least(
                    "surface_transition_moving_belt_changes_force",
                    "moving-belt band should alter tire force through relative ground velocity",
                    "moving_belt_force_delta_n",
                    moving_belt_force_delta_n,
                    500.0,
                ),
                TireRigAcceptanceCheck::at_least(
                    "surface_transition_moving_belt_work_nonzero",
                    "moving-belt band should report non-zero terrain work from tire force dotted with belt velocity",
                    "abs_moving_belt_power_w",
                    moving_belt_power_w,
                    1.0,
                ),
                TireRigAcceptanceCheck::at_most(
                    "surface_transition_force_jump_bounded",
                    "smoothed surface transitions should avoid teleport-scale force jumps",
                    "max_adjacent_force_jump_over_budget",
                    max_transition_jump / dry_point.telemetry.available_friction_budget_n.max(1.0),
                    1.0,
                ),
            ],
        ));
    }

    let loss_speeds = [6.0, 12.0, 24.0];
    let dry_loss = tire_rig_rolling_loss_sweep(
        &base,
        &dry_terrain,
        &config.dry_surface_preset,
        &dry,
        load_n,
        &loss_speeds,
        1_400.0,
        0.62,
        1.225,
    );
    let soil_loss = tire_rig_rolling_loss_sweep(
        &base,
        &soil_terrain,
        &config.soil_surface_preset,
        &soil,
        load_n,
        &loss_speeds,
        1_400.0,
        0.62,
        1.225,
    );
    let standing_water =
        SurfaceFilmState::standing_water().apply_to(dry, 24.0, &TireWearState::default());
    let wet_loss = tire_rig_rolling_loss_sweep(
        &base,
        &dry_terrain,
        "standing_water",
        &standing_water,
        load_n,
        &[24.0],
        1_400.0,
        0.62,
        1.225,
    );
    if dry_loss.len() == 3 && soil_loss.len() == 3 && wet_loss.len() == 1 {
        let anchor = &dry_loss[2];
        let rolling_loss_snapshot = TireRigSweepPoint {
            load_n,
            pressure_pa: base.pressure_pa,
            hub_height_m: solve_loaded_tire_hub_height(&base, &dry_terrain, 0.0, 0.0, load_n),
            deflection_m: 0.0,
            speed_m_per_s: anchor.speed_m_per_s,
            slip_ratio_command: 0.0,
            slip_angle_command_rad: 0.0,
            camber_angle_command_rad: 0.0,
            telemetry: anchor.telemetry.clone(),
        }
        .snapshot_on_terrain(
            "rolling_loss_drag",
            &config.tire_preset,
            &config.dry_surface_preset,
            "flat",
            "cpu_brush_rolling_loss_drag",
            &dry_terrain,
            0.0,
            0.0,
        );
        snapshots.push(rolling_loss_snapshot.clone());
        reports.push(TireRigAcceptanceReport::new(
            rolling_loss_snapshot,
            vec![
                TireRigAcceptanceCheck::at_least(
                    "rolling_loss_soft_soil_exceeds_dry",
                    "soft-soil rolling resistance should exceed dry asphalt at the same speed",
                    "soil_over_dry_rolling_force",
                    soil_loss[2].rolling_resistance_force_n
                        / anchor.rolling_resistance_force_n.max(1.0),
                    3.0,
                ),
                TireRigAcceptanceCheck::between(
                    "aero_drag_force_scales_speed_squared",
                    "aero drag force should scale approximately with speed squared",
                    "drag_force_24_over_12_mps",
                    dry_loss[2].aero_drag_force_n / dry_loss[1].aero_drag_force_n.max(1.0e-9),
                    3.75,
                    4.25,
                ),
                TireRigAcceptanceCheck::between(
                    "aero_drag_power_scales_speed_cubed",
                    "aero drag power should scale approximately with speed cubed",
                    "drag_power_24_over_12_mps",
                    dry_loss[2].aero_drag_power_w / dry_loss[1].aero_drag_power_w.max(1.0e-9),
                    7.40,
                    8.60,
                ),
                TireRigAcceptanceCheck::at_least(
                    "standing_water_adds_rolling_power",
                    "standing water should add water-displacement rolling power loss",
                    "wet_over_dry_rolling_power",
                    wet_loss[0].rolling_resistance_power_w
                        / anchor.rolling_resistance_power_w.max(1.0),
                    1.50,
                ),
                TireRigAcceptanceCheck::at_least(
                    "coastdown_loss_positive",
                    "explicit rolling/aero losses should imply positive coastdown deceleration",
                    "coastdown_deceleration_m_per_s2",
                    anchor.coastdown_deceleration_m_per_s2,
                    1.0e-6,
                ),
            ],
        ));
    }
    let longitudinal_shape_sweep = tire_rig_longitudinal_slip_sweep(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        14.0,
        &[0.0, 0.02, 0.05, 0.10, 0.18, 0.30, 0.45],
    );
    if let Some(anchor) = longitudinal_shape_sweep.get(3) {
        let longitudinal_shape_snapshot = anchor.snapshot_on_terrain(
            "reference_shape_longitudinal_dry",
            &config.tire_preset,
            &config.dry_surface_preset,
            "flat",
            "cpu_brush_reference_shape",
            &dry_terrain,
            0.0,
            0.0,
        );
        snapshots.push(longitudinal_shape_snapshot.clone());
        reports.push(
            tire_rig_longitudinal_reference_shape_diagnostics(
                "reference_shape_longitudinal_dry",
                &longitudinal_shape_sweep,
            )
            .report(longitudinal_shape_snapshot),
        );
    }
    let combined_shape_sweep = tire_rig_combined_slip_sweep(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        14.0,
        &[0.0, 0.22],
        &[0.0, 0.12],
    );
    if combined_shape_sweep.len() == 4 {
        let pure_lateral = &combined_shape_sweep[1];
        let pure_longitudinal = &combined_shape_sweep[2];
        let combined = &combined_shape_sweep[3];
        let combined_shape_snapshot = combined.snapshot_on_terrain(
            "reference_shape_combined_dry",
            &config.tire_preset,
            &config.dry_surface_preset,
            "flat",
            "cpu_brush_reference_shape",
            &dry_terrain,
            0.0,
            0.0,
        );
        snapshots.push(combined_shape_snapshot.clone());
        reports.push(
            tire_rig_combined_reference_shape_diagnostics(
                "reference_shape_combined_dry",
                pure_longitudinal,
                pure_lateral,
                combined,
            )
            .report(combined_shape_snapshot),
        );
    }
    let load_combined_law = tire_load_combined_slip_law_report();
    let load_combined_law_snapshot =
        tire_rig_point(&base, &dry_terrain, &dry, load_n, 10.0, 0.22, 0.12, 0.0)
            .snapshot_on_terrain(
                "load_combined_slip_law",
                &config.tire_preset,
                &config.dry_surface_preset,
                "flat",
                "cpu_brush_load_combined_slip_law",
                &dry_terrain,
                0.0,
                0.0,
            );
    snapshots.push(load_combined_law_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        load_combined_law_snapshot,
        load_combined_law.acceptance_checks(),
    ));
    let layered_transient = tire_layered_transient_envelope_report();
    let layered_transient_snapshot =
        tire_rig_point(&base, &dry_terrain, &dry, load_n, 10.0, 0.35, 0.22, 0.0)
            .snapshot_on_terrain(
                "layered_transient_envelope",
                &config.tire_preset,
                &config.dry_surface_preset,
                "flat",
                "cpu_brush_layered_transient_envelope",
                &dry_terrain,
                0.0,
                0.0,
            );
    snapshots.push(layered_transient_snapshot.clone());
    reports.push(TireRigAcceptanceReport::new(
        layered_transient_snapshot,
        layered_transient.acceptance_checks(),
    ));

    let step_torque = tire_rig_step_torque_response(
        &base,
        &dry_terrain,
        &dry,
        3_800.0,
        10.0,
        800.0,
        config.fixed_dt_s,
        120,
    );
    if let (Some(first), Some(last)) = (
        step_torque
            .iter()
            .find(|p| p.drive_torque_nm > 0.0 && p.raw_slip_ratio > 0.03),
        step_torque.last(),
    ) {
        let step_torque_snapshot = tire_rig_step_response_snapshot(
            "step_torque_response",
            &config.tire_preset,
            &config.dry_surface_preset,
            "cpu_brush_transient_torque_step",
            &base,
            &dry_terrain,
            3_800.0,
            10.0,
            last,
        );
        snapshots.push(step_torque_snapshot.clone());
        reports.push(TireRigAcceptanceReport::new(
            step_torque_snapshot,
            vec![
                TireRigAcceptanceCheck::at_most(
                    "step_torque_effective_slip_lags_raw",
                    "effective slip should lag the raw torque step instead of teleporting",
                    "first_effective_over_raw_slip",
                    first.effective_slip_ratio.abs() / first.raw_slip_ratio.abs().max(1.0e-9),
                    0.80,
                ),
                TireRigAcceptanceCheck::at_least(
                    "step_torque_force_builds",
                    "longitudinal force should build after the first slipping sample",
                    "last_over_first_abs_fx",
                    last.telemetry.longitudinal_force_n.abs()
                        / first.telemetry.longitudinal_force_n.abs().max(1.0),
                    1.05,
                ),
            ],
        ));
    }

    let step_brake = tire_rig_step_brake_response(
        &base,
        &dry_terrain,
        &dry,
        3_800.0,
        12.0,
        900.0,
        config.fixed_dt_s,
        120,
    );
    if let (Some(first), Some(last)) = (
        step_brake
            .iter()
            .find(|p| p.brake_torque_nm > 0.0 && p.raw_slip_ratio < -0.03),
        step_brake.last(),
    ) {
        let initial_omega = step_brake
            .first()
            .map(|p| p.wheel_omega_rad_s)
            .unwrap_or(last.wheel_omega_rad_s);
        let budget_margin = step_brake
            .iter()
            .map(|p| {
                p.telemetry.available_friction_budget_n * 1.05 + 1.0
                    - p.telemetry.consumed_friction_budget_n
            })
            .fold(f64::INFINITY, f64::min);
        let step_brake_snapshot = tire_rig_step_response_snapshot(
            "step_brake_response",
            &config.tire_preset,
            &config.dry_surface_preset,
            "cpu_brush_transient_brake_step",
            &base,
            &dry_terrain,
            3_800.0,
            12.0,
            last,
        );
        snapshots.push(step_brake_snapshot.clone());
        reports.push(TireRigAcceptanceReport::new(
            step_brake_snapshot,
            vec![
                TireRigAcceptanceCheck::at_most(
                    "step_brake_effective_slip_lags_raw",
                    "effective slip should lag the raw braking step instead of teleporting",
                    "first_effective_over_raw_brake_slip",
                    first.effective_slip_ratio.abs() / first.raw_slip_ratio.abs().max(1.0e-9),
                    0.80,
                ),
                TireRigAcceptanceCheck::at_least(
                    "step_brake_reduces_wheel_speed",
                    "brake torque should reduce wheel angular speed over the response",
                    "initial_minus_final_omega",
                    initial_omega - last.wheel_omega_rad_s,
                    0.10,
                ),
                TireRigAcceptanceCheck::at_least(
                    "step_brake_friction_budget_margin",
                    "step brake response should keep consumed friction inside the reported budget",
                    "minimum_budget_margin_n",
                    budget_margin,
                    -1.0e-6,
                ),
            ],
        ));
    }

    let step_steer = tire_rig_step_steer_response(
        &base,
        &dry_terrain,
        &dry,
        3_800.0,
        14.0,
        0.20,
        config.fixed_dt_s,
        140,
    );
    if let (Some(first), Some(last)) = (
        step_steer
            .iter()
            .find(|p| p.steer_angle_rad.abs() > 0.0 && p.raw_slip_angle_rad.abs() > 0.05),
        step_steer.last(),
    ) {
        let step_steer_snapshot = tire_rig_step_response_snapshot(
            "step_steer_response",
            &config.tire_preset,
            &config.dry_surface_preset,
            "cpu_brush_transient_steer_step",
            &base,
            &dry_terrain,
            3_800.0,
            14.0,
            last,
        );
        snapshots.push(step_steer_snapshot.clone());
        reports.push(TireRigAcceptanceReport::new(
            step_steer_snapshot,
            vec![
                TireRigAcceptanceCheck::at_most(
                    "step_steer_effective_angle_lags_raw",
                    "effective slip angle should lag the raw steering step instead of teleporting",
                    "first_effective_over_raw_alpha",
                    first.effective_slip_angle_rad.abs()
                        / first.raw_slip_angle_rad.abs().max(1.0e-9),
                    0.80,
                ),
                TireRigAcceptanceCheck::at_least(
                    "step_steer_lateral_force_builds",
                    "lateral force should build after the first steering sample",
                    "last_over_first_abs_fy",
                    last.telemetry.lateral_force_n.abs()
                        / first.telemetry.lateral_force_n.abs().max(1.0),
                    1.05,
                ),
                TireRigAcceptanceCheck::at_least(
                    "step_steer_tread_yaw_builds",
                    "tread yaw state should build across the steering step",
                    "last_minus_first_abs_tread_yaw_rad",
                    last.tread_yaw_rad.abs() - first.tread_yaw_rad.abs(),
                    1.0e-5,
                ),
            ],
        ));
    }

    for snapshot in &snapshots {
        let has_dimensionless = reports.iter().any(|report| {
            report.snapshot.scenario_id == snapshot.scenario_id
                && report
                    .checks
                    .iter()
                    .any(|check| check.check_id == "fx_over_fz_sane")
        });
        if !has_dimensionless {
            reports.push(snapshot.clone().dimensionless_sanity_report(base.radius));
        }
        let has_energy_sign = reports.iter().any(|report| {
            report.snapshot.scenario_id == snapshot.scenario_id
                && report
                    .checks
                    .iter()
                    .any(|check| check.check_id == "dissipated_power_nonnegative")
        });
        if !has_energy_sign {
            reports.push(snapshot.clone().energy_sign_report());
        }
    }

    TireRigRunOutput {
        runner_id: "standard_non_calibration_tire_rig".to_owned(),
        seed: config.seed,
        snapshots,
        reports,
        sensitivity_points: tire_rig_standard_sensitivity_map(),
        vehicle_sanity_cases: tire_vehicle_sanity_scenarios(),
    }
}

pub(crate) fn tire_rig_point(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratio_command: f64,
    slip_angle_command_rad: f64,
    camber_angle_command_rad: f64,
) -> TireRigSweepPoint {
    tire_rig_point_at(
        params,
        terrain,
        material,
        [0.0, 0.0],
        load_n,
        speed_m_per_s,
        slip_ratio_command,
        slip_angle_command_rad,
        camber_angle_command_rad,
    )
}

pub(crate) fn tire_rig_point_at(
    params: &RealTireParams,
    terrain: &dyn Terrain,
    material: &SurfaceMaterial,
    xz_m: [f64; 2],
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratio_command: f64,
    slip_angle_command_rad: f64,
    camber_angle_command_rad: f64,
) -> TireRigSweepPoint {
    let load_n = finite_nonnegative(load_n);
    let speed_m_per_s = finite_or(speed_m_per_s, 0.0);
    let slip_ratio_command = finite_or(slip_ratio_command, 0.0);
    let slip_angle_command_rad = finite_or(slip_angle_command_rad, 0.0).clamp(-1.2, 1.2);
    let camber_angle_command_rad = finite_or(camber_angle_command_rad, 0.0).clamp(-0.7, 0.7);
    let x = finite_or(xz_m[0], 0.0);
    let z = finite_or(xz_m[1], 0.0);
    let hub_height_m = solve_loaded_tire_hub_height(params, terrain, x, z, load_n);
    let road_height = terrain.height(x, z);
    let deflection_m = (params.radius - (hub_height_m - road_height)).max(0.0);
    let normal = terrain.normal(x, z);
    let heading = normalize(project_to_ground([1.0, 0.0, 0.0], normal));
    let side = normalize(cross(heading, normal));
    let v_long = speed_m_per_s;
    let v_lat = v_long.abs() * slip_angle_command_rad.tan();
    let wheel_up = normalize(add(
        scale(normal, camber_angle_command_rad.cos()),
        scale(side, camber_angle_command_rad.sin()),
    ));
    let state = RealWheelState {
        hub_pos: [x, hub_height_m, z],
        hub_vel: add(scale(heading, v_long), scale(side, v_lat)),
        heading,
        omega: v_long * (1.0 + slip_ratio_command) / params.radius.max(1.0e-6),
    };
    let contact = evaluate_brush_tire_oriented(
        params,
        &state,
        terrain,
        material,
        &RealWheelOrientation { wheel_up },
    );
    TireRigSweepPoint {
        load_n,
        pressure_pa: finite_nonnegative(params.pressure_pa),
        hub_height_m,
        deflection_m,
        speed_m_per_s,
        slip_ratio_command,
        slip_angle_command_rad,
        camber_angle_command_rad,
        telemetry: contact.telemetry_sample(0.0),
    }
}
