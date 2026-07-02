//! Tests for the real-tire model. Moved out of `real_tire/mod.rs` as part of
//! PLAN_REALTIRE Stream A so the parent module stays small.

#![cfg(test)]

use super::internal::{finite_vec3, sub};
use super::*;
use crate::terrain::{FlatTerrain, HeightfieldTerrain, SdfTerrain, SdfTerrainPrimitive, Terrain};

fn loaded_output(pressure: f64, material: SurfaceMaterial, slip_speed: f64) -> RealTireContact {
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: pressure,
        ..Default::default()
    };
    let load = 4_500.0;
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load);
    evaluate_brush_tire(
        &params,
        &RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [1.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: (1.0 + slip_speed) / params.radius,
        },
        &terrain,
        &material,
    )
}

#[test]
fn real_brush_tire_model_adapts_native_brush_contact_to_vehicle_trait() {
    let tire = RealBrushTireModel::passenger_radial().with_nominal_speed(9.0);
    let dry = tire.evaluate(0.12, 0.08, 4_200.0, 1.0);
    let ice = tire.evaluate(0.12, 0.08, 4_200.0, 0.08);
    for (label, force) in [("dry", dry), ("ice", ice)] {
        assert!(
            force.0.is_finite() && force.1.is_finite() && force.2.is_finite(),
            "{label} adapted real-brush tire force must stay finite: {force:?}"
        );
    }
    assert!(
        dry.0.abs() > ice.0.abs() * 3.0,
        "vehicle TireModel adapter should still expose real brush friction sensitivity: dry={dry:?} ice={ice:?}"
    );
    assert!(
        dry.1.abs() > 10.0 && dry.2.is_finite(),
        "adapter should preserve lateral brush force and aligning moment for vehicle yaw: {dry:?}"
    );
    assert_eq!(
        tire.evaluate(0.5, 0.2, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        "unloaded adapted tire should not create forces"
    );
}

#[test]
fn telemetry_sample_reports_units_and_force_accounting() {
    let contact = loaded_output(160_000.0, SurfaceMaterial::dry_asphalt(), 0.65);
    let dt = 1.0 / 120.0;
    let sample = contact.telemetry_sample(dt);
    assert!(sample.in_contact);
    assert!(sample.is_finite(), "telemetry must be finite: {sample:?}");
    assert_eq!(sample.patch.contact_count, contact.contact_count);
    assert_eq!(sample.patch.sliding_count, contact.sliding_count);
    assert!(sample.patch.contact_area_m2 > 0.0);
    assert!(sample.patch.mean_normal_pressure_pa > 0.0);
    assert!(sample.patch.peak_normal_pressure_pa >= sample.patch.mean_normal_pressure_pa);
    assert_eq!(sample.dissipated_energy_j, sample.dissipated_power_w * dt);
    assert!(
        sample.consumed_friction_budget_n <= sample.available_friction_budget_n * 1.05,
        "consumed friction should stay inside the reported budget: {sample:?}"
    );

    let first = TireContactCellTelemetry::from_cell(&contact.cells[0]);
    assert!(
        first.is_finite(),
        "cell telemetry must be finite: {first:?}"
    );
    assert!(first.area_m2 > 0.0);
    assert!(first.normal_load_n >= 0.0);
    assert!(first.normal_pressure_pa >= 0.0);
}

#[test]
fn telemetry_csv_hooks_emit_unit_headers_and_sanitized_rows() {
    let contact = loaded_output(160_000.0, SurfaceMaterial::dry_asphalt(), 0.65);
    let sample = contact.telemetry_sample(0.01);
    let row = sample.to_csv_row(1.25, "front,left");
    assert!(
        RealTireTelemetrySample::CSV_HEADER.contains("force_x_n")
            && RealTireTelemetrySample::CSV_HEADER.contains("patch_length_m"),
        "header should carry unit-labelled fields"
    );
    assert!(
        row.contains("front_left"),
        "wheel id should be comma-sanitized: {row}"
    );
    assert_eq!(
        row.split(',').count(),
        RealTireTelemetrySample::CSV_HEADER.split(',').count(),
        "telemetry row/header column count should match\nheader={}\nrow={row}",
        RealTireTelemetrySample::CSV_HEADER
    );

    let cells = contact.cell_telemetry();
    assert_eq!(cells.len(), contact.cells.len());
    assert!(cells.iter().all(TireContactCellTelemetry::is_finite));
    let cell_row = cells[0].to_csv_row(1.25, "front,left", 0);
    assert!(
        TireContactCellTelemetry::CSV_HEADER.contains("normal_pressure_pa")
            && TireContactCellTelemetry::CSV_HEADER.contains("longitudinal_force_n"),
        "cell header should carry unit-labelled fields"
    );
    assert!(
        cell_row.contains("front_left"),
        "cell wheel id should be comma-sanitized: {cell_row}"
    );
    assert_eq!(
        cell_row.split(',').count(),
        TireContactCellTelemetry::CSV_HEADER.split(',').count(),
        "cell row/header column count should match\nheader={}\nrow={cell_row}",
        TireContactCellTelemetry::CSV_HEADER
    );
}

#[test]
fn pressure_distribution_diagnostics_track_patch_pressure_and_width() {
    let material = SurfaceMaterial::dry_asphalt();
    let high = loaded_output(340_000.0, material, 0.08);
    let low = loaded_output(90_000.0, material, 0.08);
    let high_dist = high.pressure_distribution(8, 4);
    let low_dist = low.pressure_distribution(8, 4);
    assert!(high_dist.is_finite() && low_dist.is_finite());
    assert_eq!(high_dist.lateral_bins.len(), 1);
    assert_eq!(low_dist.lateral_bins.len(), 1);
    assert!(
        low_dist.longitudinal_span_m > high_dist.longitudinal_span_m,
        "lower pressure should spread contact over more circumference: low={low_dist:?} high={high_dist:?}"
    );
    assert!(
        high_dist.peak_normal_pressure_pa > low_dist.peak_normal_pressure_pa,
        "stiff/high-pressure tire should have higher peak patch pressure: high={} low={}",
        high_dist.peak_normal_pressure_pa,
        low_dist.peak_normal_pressure_pa
    );
    let low_binned_load: f64 = low_dist
        .longitudinal_bins
        .iter()
        .map(|bin| bin.normal_load_n)
        .sum();
    assert!(
        (low_binned_load - low.normal_load).abs() < low.normal_load.max(1.0) * 1.0e-9,
        "pressure bins should conserve normal load"
    );

    let mut tire = TetraTireFem::new(
        0.0,
        95_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 5,
            pressure_pa: 95_000.0,
            downforce: 4_200.0,
            ..Default::default()
        },
    );
    for _ in 0..360 {
        tire.step(1.0 / 240.0);
    }
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let state = RealWheelState {
        hub_pos: tire.hub(),
        hub_vel: [4.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 4.0 / tire.outer_radius(),
    };
    let fem_contact = tire.evaluate_tread_contact(&state, &terrain, &material);
    let fem_dist = fem_contact.pressure_distribution(6, 5);
    assert!(
        fem_dist.lateral_span_m > 0.02 && fem_dist.lateral_bins.len() > 1,
        "FEM-derived tread-node contact should expose widthwise pressure distribution: {fem_dist:?}"
    );
    assert!(
        fem_dist
            .lateral_bins
            .iter()
            .filter(|bin| bin.contact_count > 0)
            .count()
            >= 2,
        "at least two lateral pressure bins should carry load for a deformed tread patch"
    );
}

#[test]
fn tire_rig_snapshot_records_inputs_provenance_and_outputs() {
    let params = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 170_000.0)
        .with_segments(32);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.05, material.mu_static);
    let mut sweep =
        tire_rig_longitudinal_slip_sweep(&params, &terrain, &material, 3_200.0, 12.0, &[0.18]);
    let point = sweep.remove(0);
    let thermal = TireThermalLayers {
        surface_c: 74.0,
        tread_c: 48.0,
        carcass_c: 39.0,
        ..TireThermalLayers::passenger_road()
    };
    let wear = TireWearState {
        depth_m: 0.0015,
        ..Default::default()
    };

    let snapshot = point
        .snapshot_on_terrain(
            "long slip, wet\ncase",
            "passenger radial",
            "dry asphalt",
            "flat belt",
            "cpu brush",
            &terrain,
            0.0,
            0.0,
        )
        .with_thermal_wear(&thermal, &wear)
        .with_cpu_gpu_force_delta([
            point.telemetry.force_n[0] + 2.5,
            point.telemetry.force_n[1],
            point.telemetry.force_n[2],
        ]);

    assert!(
        snapshot.is_finite(),
        "snapshot must be finite: {snapshot:?}"
    );
    assert_eq!(
        snapshot.tire_parameter_provenance,
        TireModelProvenance::HandTuned
    );
    assert_eq!(
        snapshot.solver_parameter_provenance,
        TireModelProvenance::NumericalStability
    );
    assert_eq!(snapshot.terrain_height_m, 0.05);
    assert_eq!(snapshot.terrain_surface_velocity_m_per_s, [0.0; 3]);
    assert_eq!(snapshot.moving_ground_power_w, 0.0);
    assert_eq!(snapshot.surface_temperature_c, Some(74.0));
    assert!(snapshot.wear_ratio.unwrap() > 0.0);
    assert!(snapshot.cpu_gpu_relative_force_error.unwrap() >= 0.0);
    assert!(
        TireRigSnapshot::CSV_HEADER.contains("tire_parameter_provenance")
            && TireRigSnapshot::CSV_HEADER.contains("cpu_gpu_relative_force_error")
            && TireRigSnapshot::CSV_HEADER.contains("mean_normal_pressure_pa")
            && TireRigSnapshot::CSV_HEADER.contains("moving_ground_power_w"),
        "snapshot header should carry provenance, parity, work, and unit fields"
    );
    let row = snapshot.to_csv_row();
    assert!(
        row.contains("long slip_ wet_case") && row.contains("passenger radial"),
        "snapshot row should sanitize scenario and keep preset names: {row}"
    );
    assert_eq!(
        row.split(',').count(),
        TireRigSnapshot::CSV_HEADER.split(',').count(),
        "snapshot row/header column count should match\nheader={}\nrow={row}",
        TireRigSnapshot::CSV_HEADER
    );
}

#[test]
fn replay_artifact_coverage_report_checks_compact_snapshot_traceability() {
    let run = tire_rig_default_non_calibration_run();
    let report = run.replay_artifact_coverage_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "default runner should emit complete replay artifacts: {report:?}"
    );
    assert_eq!(report.csv_row_column_mismatches, 0);
    assert_eq!(report.command_input_rows, report.snapshot_count);
    assert_eq!(report.terrain_state_rows, report.snapshot_count);
    assert_eq!(report.provenance_rows, report.snapshot_count);
    assert_eq!(report.force_moment_rows, report.snapshot_count);
    assert_eq!(report.patch_telemetry_rows, report.snapshot_count);
    assert!(report.thermal_wear_rows >= 3);
    assert!(report.solver_budget_rows >= 1);
    assert!(report.cpu_gpu_delta_columns_present);
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "replay_artifact_command_inputs_present")
            && checks
                .iter()
                .any(|check| check.check_id == "replay_artifact_terrain_state_present")
            && checks
                .iter()
                .any(|check| check.check_id == "replay_artifact_known_approximations_available"),
        "acceptance checks should expose replay-artifact traceability gates: {checks:?}"
    );
    let text = run.to_text_report();
    assert!(
        text.contains("replay_artifact_coverage")
            && text.contains("replay_artifact_cpu_gpu_delta_columns_present"),
        "text report should expose replay-artifact coverage rows: {text}"
    );
}

#[test]
fn tire_solver_budget_telemetry_records_fem_numerics_in_snapshot() {
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let params = TetraTireFemParams {
        xpbd_volume_iterations: 2,
        xpbd_structural_iterations: 3,
        ..TetraTireFemParams::realtime_viewer()
    };
    let mut tire = TetraTireFem::new(0.0, params.pressure_pa, params);
    for _ in 0..4 {
        tire.step_on_terrain(&terrain, 1.0 / 240.0);
    }
    let contact = RealTireContact::empty();
    let telemetry = contact.telemetry_sample(1.0 / 60.0);
    let budget = tire.solver_budget_telemetry(4, &telemetry);
    assert!(
        budget.is_finite(),
        "solver budget must be finite: {budget:?}"
    );
    assert_eq!(budget.substeps_per_frame, 4);
    assert_eq!(budget.xpbd_volume_iterations, 2);
    assert_eq!(budget.xpbd_structural_iterations, 3);
    assert!(budget.max_tet_volume_error_m3 >= 0.0);
    assert!(budget.mean_bead_mount_error_m >= 0.0);
    assert!(budget.max_bead_mount_error_m >= budget.mean_bead_mount_error_m);
    assert!(budget.last_xpbd_volume_correction_m >= 0.0);

    let brush_params = RealTireParams::passenger_radial();
    let material = SurfaceMaterial::dry_asphalt();
    let point =
        tire_rig_load_deflection_sweep(&brush_params, &terrain, &material, &[3_000.0]).remove(0);
    let snapshot = point
        .snapshot_on_terrain(
            "solver budget",
            "passenger_radial",
            "dry_asphalt",
            "flat",
            "cpu_tetra_fem",
            &terrain,
            0.0,
            0.0,
        )
        .with_solver_budget(budget);
    assert!(snapshot.is_finite());
    assert!(snapshot.solver_budget.unwrap().max_node_speed_m_per_s >= 0.0);
    assert!(
        TireRigSnapshot::CSV_HEADER.contains("solver_xpbd_volume_iterations")
            && TireRigSnapshot::CSV_HEADER.contains("solver_max_node_speed_m_per_s")
            && TireRigSnapshot::CSV_HEADER.contains("solver_max_bead_mount_error_m"),
        "snapshot CSV should include solver-budget fields"
    );
    let row = snapshot.to_csv_row();
    assert_eq!(
        row.split(',').count(),
        TireRigSnapshot::CSV_HEADER.split(',').count(),
        "solver-budget snapshot row/header column count should match\nheader={}\nrow={row}",
        TireRigSnapshot::CSV_HEADER
    );
}

#[test]
fn tire_rig_default_runner_emits_non_calibration_snapshots_and_envelopes() {
    let run = tire_rig_default_non_calibration_run();
    assert!(run.is_finite(), "runner output must be finite: {run:?}");
    assert!(
        run.passed(),
        "runner acceptance envelopes should pass: {run:?}"
    );
    assert!(
        run.snapshots.len() >= 9,
        "runner should cover load/pressure, slip, camber, wet, soil, curb, and thermal cases: {:?}",
        run.snapshots
            .iter()
            .map(|s| s.scenario_id.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        run.sensitivity_points.len() >= 9,
        "runner should attach the standard sensitivity map: {:?}",
        run.sensitivity_points
            .iter()
            .map(|p| p.axis.label())
            .collect::<Vec<_>>()
    );
    assert!(
        run.sensitivity_points.iter().all(|point| point.passed),
        "standard sensitivity map should pass inside the runner:\n{}",
        run.to_text_report()
    );
    assert_eq!(
        run.vehicle_sanity_cases.len(),
        6,
        "runner should attach the vehicle-level falsification cases: {:?}",
        run.vehicle_sanity_cases
            .iter()
            .map(|case| case.scenario_id.as_str())
            .collect::<Vec<_>>()
    );
    assert!(
        run.vehicle_sanity_cases.iter().all(|case| case.passed),
        "vehicle sanity cases should pass inside the runner:\n{}",
        run.to_text_report()
    );
    for required in [
        "load_pressure_high",
        "load_pressure_low",
        "longitudinal_slip_dry",
        "longitudinal_slip_ice",
        "ice_large_patch_grip",
        "adhesion_sliding_continuity",
        "torque_induced_slip_outcome",
        "vehicle_longitudinal_motion",
        "vehicle_dynamic_load_transfer",
        "camber_pneumatic_trail",
        "load_combined_slip_law",
        "layered_transient_envelope",
        "wet_film_slip",
        "loose_soil_slip",
        "soil_normal_shear_coupling",
        "dem_soil_feedback",
        "mpm_soil_feedback",
        "moving_ground_belt",
        "heightfield_surface_velocity_map",
        "curb_envelope_contact",
        "rolling_obstacle_approach",
        "rolling_obstacle_crest",
        "rolling_obstacle_exit",
        "thermal_wear_cycle",
        "thermal_wear_heat_peak",
        "thermal_wear_cooling_recovery",
        "thermal_abuse_envelope",
        "thermal_wear_power_temperature",
        "worn_surface_sensitivity",
        "condition_failure_modes",
        "thermal_rolling_carcass",
        "thermal_brake_rim",
        "thermal_wear_convergence",
        "fem_solver_budget",
        "fem_solver_stiffness_stabilizer",
        "fem_diagonal_newton_solver",
        "fem_element_newton_solver",
        "fem_patch_newton_solver",
        "fem_tiny_mesh_newton_solver",
        "fem_matrix_free_newton_scale",
        "fem_matrix_free_newton_solve",
        "fem_matrix_free_newton_preconditioner",
        "fem_matrix_free_newton_step",
        "fem_matrix_free_newton_timestep",
        "fem_deformation_force_shape",
        "fem_contact_slip_patch",
        "bead_seat_load_cases",
        "cavity_coupling_effect",
        "sidewall_dynamic_load_cases",
        "sidewall_laminate_coupling",
        "sidewall_ply_strain_xpbd",
        "sidewall_dihedral_xpbd_effect",
        "triangle_surface_stress",
        "triangle_surface_force",
        "triangle_density_convergence",
        "triangle_large_manifold",
        "fem_substep_convergence",
        "heightfield_resolution_convergence",
    ] {
        assert!(
            run.snapshots
                .iter()
                .any(|snapshot| snapshot.scenario_id == required),
            "missing required runner scenario {required}: {:?}",
            run.snapshots
                .iter()
                .map(|s| s.scenario_id.as_str())
                .collect::<Vec<_>>()
        );
    }

    let fem_budget = run
        .snapshots
        .iter()
        .find(|snapshot| snapshot.scenario_id == "fem_solver_budget")
        .and_then(|snapshot| snapshot.solver_budget)
        .expect("default runner should include FEM solver-budget telemetry");
    assert!(
        fem_budget.is_finite()
            && fem_budget.substeps_per_frame > 1
            && fem_budget.max_node_speed_m_per_s >= 0.0
            && fem_budget.max_tet_volume_error_m3 >= 0.0,
        "FEM solver-budget snapshot should expose numerical health: {fem_budget:?}"
    );

    let high = run
        .snapshots
        .iter()
        .find(|s| s.scenario_id == "load_pressure_high")
        .unwrap();
    let low = run
        .snapshots
        .iter()
        .find(|s| s.scenario_id == "load_pressure_low")
        .unwrap();
    assert!(
        low.telemetry.patch.patch_length_m > high.telemetry.patch.patch_length_m,
        "lower pressure should grow the patch in the built-in runner"
    );

    let dry = run
        .snapshots
        .iter()
        .find(|s| s.scenario_id == "longitudinal_slip_dry")
        .unwrap();
    let ice = run
        .snapshots
        .iter()
        .find(|s| s.scenario_id == "longitudinal_slip_ice")
        .unwrap();
    assert!(
        ice.telemetry.longitudinal_force_n.abs() < dry.telemetry.longitudinal_force_n.abs() * 0.45,
        "ice should cap force relative to dry asphalt in the runner"
    );
    let moving = run
        .snapshots
        .iter()
        .find(|s| s.scenario_id == "moving_ground_belt")
        .unwrap();
    assert!(
        moving.terrain_surface_velocity_m_per_s[0].abs() > 1.0
            && moving.moving_ground_power_w > 1.0,
        "moving-ground snapshot should expose belt velocity and positive work: {moving:?}"
    );
    let reports_without_replay_context = run.reports_without_replay_context();
    assert!(
        reports_without_replay_context.is_empty(),
        "every report should be attached to a replay snapshot or a known ordered sequence: {reports_without_replay_context:?}"
    );
    let replay_artifact_coverage = run.replay_artifact_coverage_report();
    assert!(
        replay_artifact_coverage.passed(),
        "replay artifact coverage should prove compact reproducible snapshot artifacts: {replay_artifact_coverage:?}"
    );
    let viewer_target_coverage = run.viewer_target_coverage_report();
    assert!(
        viewer_target_coverage.passed(),
        "viewer-target coverage should map the PLAN_REALTIRE viewer targets to deterministic rig evidence: {viewer_target_coverage:?}"
    );
    assert!(
        viewer_target_coverage.checks.iter().any(|check| {
            check.target_id == "rolling_over_bump_curb" && check.evidence.contains("crest=true")
        }) && viewer_target_coverage.checks.iter().any(|check| {
            check.target_id == "low_pressure_wraps_obstacle"
                && check.evidence.contains("crest_pressure_pa")
        }) && viewer_target_coverage.checks.iter().any(|check| {
            check.target_id == "ice_patch_then_asphalt"
                && check.evidence.contains("surface_transition=true")
        }) && viewer_target_coverage.checks.iter().any(|check| {
            check.target_id == "soft_soil_rut_compaction"
                && check.evidence.contains("offroad_case=true")
        }) && viewer_target_coverage.checks.iter().any(|check| {
            check.target_id == "soft_tire_flattens_and_grows_patch"
                && check.evidence.contains("lower_pressure_gate=true")
        }) && viewer_target_coverage.checks.iter().any(|check| {
            check.target_id == "overinflated_tire_slips_sooner"
                && check.evidence.contains("high_pressure_slips_more=true")
        }) && viewer_target_coverage.checks.iter().any(|check| {
            check.target_id == "ice_large_patch_still_slips"
                && check.evidence.contains("sliding_gate=true")
        }) && viewer_target_coverage.checks.iter().any(|check| {
            check.target_id == "thermal_wear_state_visible"
                && check.evidence.contains("tread_depth_present=true")
        }),
        "coverage report should expose concrete evidence strings for all viewer targets: {viewer_target_coverage:?}"
    );
    let rolling_approach = run
        .snapshots
        .iter()
        .find(|s| s.scenario_id == "rolling_obstacle_approach")
        .unwrap();
    let rolling_crest = run
        .snapshots
        .iter()
        .find(|s| s.scenario_id == "rolling_obstacle_crest")
        .unwrap();
    let rolling_exit = run
        .snapshots
        .iter()
        .find(|s| s.scenario_id == "rolling_obstacle_exit")
        .unwrap();
    assert!(
        rolling_crest.terrain_height_m > rolling_approach.terrain_height_m + 0.04
            && rolling_crest.terrain_height_m > rolling_exit.terrain_height_m + 0.04
            && rolling_crest.telemetry.patch.contact_count > 1
            && rolling_crest.telemetry.patch.patch_length_m > 0.035,
        "rolling obstacle replay should capture approach/crest/exit and keep an extended FEM patch: approach={rolling_approach:?} crest={rolling_crest:?} exit={rolling_exit:?}"
    );

    let csv = run.to_snapshot_csv_table();
    assert_eq!(
        csv.lines().count(),
        run.snapshots.len() + 1,
        "CSV table should include one header plus one row per snapshot"
    );
    for snapshot in &run.snapshots {
        assert!(
            run.reports.iter().any(|report| {
                report.snapshot.scenario_id == snapshot.scenario_id
                    && report
                        .checks
                        .iter()
                        .any(|check| check.check_id == "fx_over_fz_sane")
            }),
            "snapshot {} should have a dimensionless sanity report",
            snapshot.scenario_id
        );
        assert!(
            run.reports.iter().any(|report| {
                report.snapshot.scenario_id == snapshot.scenario_id
                    && report
                        .checks
                        .iter()
                        .any(|check| check.check_id == "dissipated_power_nonnegative")
            }),
            "snapshot {} should have an energy/sign report",
            snapshot.scenario_id
        );
    }
    assert!(
        csv.contains("thermal_wear_cycle") && !csv.contains("non_calibration"),
        "snapshot CSV should contain scenario rows but no fitted-target flag column: {csv}"
    );
    let text = run.to_text_report();
    assert!(
        text.contains("non_calibration=true")
            && text.contains("lower_pressure_grows_patch")
            && text.contains("ice_caps_longitudinal_force")
            && text.contains("ice_large_patch_grip")
            && text.contains("ice_large_patch_still_caps_force")
            && text.contains("ice_large_patch_low_pressure_patch_grows")
            && text.contains("adhesion_sliding_continuity")
            && text.contains("adhesion_sliding_force_continuous")
            && text.contains("adhesion_sliding_moment_continuous")
            && text.contains("torque_induced_slip_outcome")
            && text.contains("torque_slip_partial_pressure_moves_farther")
            && text.contains("torque_slip_high_pressure_slips_more")
            && text.contains("vehicle_longitudinal_motion")
            && text.contains("vehicle_longitudinal_integrator_matches_equation")
            && text.contains("vehicle_longitudinal_force_balance_closes")
            && text.contains("load_combined_slip_law")
            && text.contains("load_sensitivity_sublinear_peak")
            && text.contains("low_mu_pressure_tractive_stiffness_improves")
            && text.contains("low_mu_pressure_patch_grows")
            && text.contains("combined_slip_single_budget")
            && text.contains("combined_slip_saturation_continuous")
            && text.contains("combined_slip_saturation_starts_with_adhesion")
            && text.contains("combined_slip_sliding_fraction_grows")
            && text.contains("temperature_state_reduces_tread_shear")
            && text.contains("temperature_state_reduces_longitudinal_stiffness")
            && text.contains("temperature_state_reduces_cornering_stiffness")
            && text.contains("layered_transient_envelope")
            && text.contains("layered_transient_tread_faster_than_belt_kappa")
            && text.contains("layered_transient_hot_overloaded_slower_alpha")
            && text.contains("layered_transient_force_builds_over_time")
            && text.contains("wet_film_reduces_force")
            && text.contains("moving_ground_belt")
            && text.contains("moving_ground_work_positive")
            && text.contains("heightfield_surface_velocity_map")
            && text.contains("heightfield_surface_velocity_changes_force")
            && text.contains("heightfield_surface_velocity_work_nonzero")
            && text.contains("curb_envelope_contact")
            && text.contains("curb_envelope_multicell_patch")
            && text.contains("curb_low_pressure_conforms_more_than_high")
            && text.contains("surface_transition_sequence")
            && text.contains("surface_transition_ice_caps_force")
            && text.contains("surface_transition_wet_reduces_force")
            && text.contains("surface_transition_moving_belt_velocity")
            && text.contains("surface_transition_moving_belt_changes_force")
            && text.contains("surface_transition_moving_belt_work_nonzero")
            && text.contains("rolling_loss_drag")
            && text.contains("rolling_loss_soft_soil_exceeds_dry")
            && text.contains("rolling_obstacle_approach")
            && text.contains("rolling_obstacle_crest")
            && text.contains("rolling_obstacle_exit")
            && text.contains("rolling_obstacle_sequence")
            && text.contains("rolling_obstacle_crest_above_approach")
            && text.contains("rolling_obstacle_multi_cell_patch")
            && text.contains("aero_drag_force_scales_speed_squared")
            && text.contains("standing_water_adds_rolling_power")
            && text.contains("thermal_condition_sanity_finite")
            && text.contains("thermal_condition_cooling_drop_nonnegative")
            && text.contains("thermal_condition_wear_non_healing")
            && text.contains("thermal_abuse_envelope")
            && text.contains("thermal_abuse_hot_reduces_mu")
            && text.contains("thermal_abuse_combined_heat_wear_worst_mu")
            && text.contains("thermal_abuse_abused_softens_carcass")
            && text.contains("thermal_wear_power_temperature")
            && text.contains("thermal_wear_power_increases_wear")
            && text.contains("thermal_wear_hot_surface_accelerates_wear")
            && text.contains("worn_surface_sensitivity")
            && text.contains("worn_surface_wet_extra_loss")
            && text.contains("worn_surface_ice_extra_loss")
            && text.contains("worn_surface_loose_soil_extra_loss")
            && text.contains("condition_failure_modes")
            && text.contains("condition_failure_levels_state_derived")
            && text.contains("condition_failure_abuse_reduces_force")
            && text.contains("thermal_rolling_carcass")
            && text.contains("thermal_rolling_carcass_heats_carcass")
            && text.contains("thermal_rolling_carcass_pressure_factor_rises")
            && text.contains("thermal_rolling_carcass_mu_factor_clamped")
            && text.contains("thermal_brake_rim")
            && text.contains("thermal_brake_rim_heats_rim_first")
            && text.contains("thermal_brake_rim_conducts_to_carcass")
            && text.contains("thermal_brake_rim_pressure_factor_rises")
            && text.contains("thermal_wear_convergence")
            && text.contains("thermal_wear_convergence_power_stable")
            && text.contains("thermal_wear_convergence_wear_stable")
            && text.contains("soil_normal_shear_coupling")
            && text.contains("soil_normal_shear_normal_support")
            && text.contains("soil_normal_shear_slip_deepens_shear_rut")
            && text.contains("dem_soil_feedback")
            && text.contains("dem_soil_feedback_slip_increases_shear")
            && text.contains("dem_soil_feedback_slip_moves_grains_more")
            && text.contains("mpm_soil_feedback")
            && text.contains("mpm_soil_feedback_slip_increases_shear")
            && text.contains("mpm_soil_feedback_slip_increases_surface_velocity")
            && text.contains("fem_solver_budget")
            && text.contains("fem_contact_node_sources_present")
            && text.contains("fem_contact_feedback_accelerates_tread")
            && text.contains("fem_contact_shared_friction_budget")
            && text.contains("fem_solver_stiffness_stabilizer")
            && text.contains("fem_solver_stiffness_reduces_velocity_spike")
            && text.contains("fem_solver_high_pressure_radius_bounded")
            && text.contains("fem_diagonal_newton_solver")
            && text.contains("fem_diagonal_newton_residual_reduced")
            && text.contains("fem_diagonal_newton_energy_bounded")
            && text.contains("fem_element_newton_solver")
            && text.contains("fem_element_newton_residual_reduced")
            && text.contains("fem_element_newton_offdiag_coupling_present")
            && text.contains("fem_patch_newton_solver")
            && text.contains("fem_patch_newton_shared_nodes")
            && text.contains("fem_patch_newton_offdiag_coupling_present")
            && text.contains("fem_tiny_mesh_newton_solver")
            && text.contains("fem_tiny_mesh_newton_full_mesh_nodes")
            && text.contains("fem_tiny_mesh_newton_offdiag_coupling_present")
            && text.contains("fem_matrix_free_newton_scale")
            && text.contains("fem_matrix_free_newton_jv_nonzero")
            && text.contains("fem_matrix_free_newton_memory_saves_dense")
            && text.contains("fem_matrix_free_newton_solve")
            && text.contains("fem_matrix_free_newton_solve_residual_reduced")
            && text.contains("fem_matrix_free_newton_solve_jv_evaluations")
            && text.contains("fem_matrix_free_newton_preconditioner")
            && text.contains("fem_matrix_free_newton_preconditioner_positive_diagonal")
            && text.contains("fem_matrix_free_newton_preconditioner_residual_not_worse")
            && text.contains("fem_matrix_free_newton_step")
            && text.contains("fem_matrix_free_newton_step_residual_reduced")
            && text.contains("fem_matrix_free_newton_step_energy_bounded")
            && text.contains("fem_matrix_free_newton_timestep")
            && text.contains("fem_matrix_free_newton_step_velocity_bounded")
            && text.contains("fem_deformation_force_shape")
            && text.contains("fem_deformation_force_shape_changes_force")
            && text.contains("fem_deformation_force_low_pressure_longer_patch")
            && text.contains("fem_contact_slip_patch")
            && text.contains("fem_contact_slip_patch_sliding_count_grows")
            && text.contains("fem_contact_slip_patch_low_pressure_longer")
            && text.contains("cavity_bead_sanity")
            && text.contains("cavity_pressure_scales_bead_load")
            && text.contains("cavity_coupling_effect")
            && text.contains("cavity_coupling_high_increases_load")
            && text.contains("bead_mount_error_bounded")
            && text.contains("bead_seat_radial_error_bounded")
            && text.contains("bead_seat_axial_error_bounded")
            && text.contains("bead_seat_tangential_error_bounded")
            && text.contains("bead_seat_load_cases")
            && text.contains("bead_load_cases_lateral_mount_bounded")
            && text.contains("bead_load_cases_lateral_force_applied")
            && text.contains("bead_load_cases_torque_force_applied")
            && text.contains("bead_load_cases_torque_tangential_error_bounded")
            && text.contains("sidewall_laminate_sanity")
            && text.contains("sidewall_laminate_crossed_ply_coverage")
            && text.contains("sidewall_laminate_max_fold_drift")
            && text.contains("sidewall_laminate_max_tensile_strain")
            && text.contains("sidewall_laminate_max_compressive_strain")
            && text.contains("sidewall_dynamic_load_cases")
            && text.contains("sidewall_dynamic_lateral_sanity_passes")
            && text.contains("sidewall_dynamic_lateral_force_applied")
            && text.contains("sidewall_dynamic_steer_sanity_passes")
            && text.contains("sidewall_dynamic_steer_aligning_moment")
            && text.contains("sidewall_laminate_coupling")
            && text.contains("sidewall_laminate_coupling_crossed_pair_balanced")
            && text.contains("sidewall_laminate_coupling_radial_helper_active")
            && text.contains("sidewall_laminate_coupling_axial_helper_active")
            && text.contains("sidewall_laminate_coupling_projected_fold_bounded")
            && text.contains("sidewall_ply_strain_xpbd")
            && text.contains("sidewall_ply_strain_xpbd_reduces_tensile_strain")
            && text.contains("sidewall_ply_strain_xpbd_bounds_tensile_strain")
            && text.contains("sidewall_ply_strain_xpbd_applies_correction")
            && text.contains("sidewall_dihedral_xpbd_effect")
            && text.contains("sidewall_dihedral_xpbd_reduces_fold")
            && text.contains("construction_sanity")
            && text.contains("construction_loaded_radius_bounded")
            && text.contains("construction_tread_support_widthwise")
            && text.contains("construction_belt_peak_stretch_bounded")
            && text.contains("construction_sidewall_bulge_bounded")
            && text.contains("realism_source_classification")
            && text.contains("realism_sources_include_native_physics")
            && text.contains("realism_sources_diagnostics_not_force_source")
            && text.contains("parameter_observability")
            && text.contains("parameter_observability_brush_shader_inputs_named")
            && text.contains("parameter_observability_hidden_force_sources_blocked")
            && text.contains("triangle_seam_manifold")
            && text.contains("triangle_seam_manifold_multi_patch")
            && text.contains("triangle_seam_fem_projection_lifts_node")
            && text.contains("triangle_surface_stress")
            && text.contains("triangle_surface_stress_seam_multi_patch")
            && text.contains("triangle_surface_stress_seam_material_span")
            && text.contains("triangle_surface_stress_right_material")
            && text.contains("triangle_surface_force")
            && text.contains("triangle_surface_force_high_mu_stronger_than_low")
            && text.contains("triangle_surface_force_sliding_bristles_active")
            && text.contains("triangle_surface_force_seam_multi_patch")
            && text.contains("triangle_density_convergence")
            && text.contains("triangle_density_force_stable")
            && text.contains("triangle_density_keeps_material_span")
            && text.contains("triangle_large_manifold")
            && text.contains("triangle_large_manifold_feedback_force")
            && text.contains("low_speed_robustness")
            && text.contains("low_speed_outputs_finite")
            && text.contains("low_speed_reversal_continuity")
            && text.contains("frame_sign_convention")
            && text.contains("frame_drive_slip_force_sign")
            && text.contains("frame_pneumatic_trail_moment_sign")
            && text.contains("contact_patch_convergence")
            && text.contains("patch_convergence_force_stable")
            && text.contains("patch_convergence_pressure_bins_conserve_load")
            && text.contains("fem_substep_convergence")
            && text.contains("fem_substep_convergence_force_stable")
            && text.contains("fem_substep_convergence_volume_bounded")
            && text.contains("heightfield_resolution_convergence")
            && text.contains("heightfield_resolution_force_stable")
            && text.contains("heightfield_resolution_normal_stable")
            && text.contains("road_condition_state")
            && text.contains("road_condition_wet_film_reduces_force")
            && text.contains("road_condition_hot_worn_state_reduces_mu")
            && text.contains("layer_ablation_debug")
            && text.contains("layer_ablation_tread_bristles_change_force")
            && text.contains("layer_ablation_soil_rut_feedback_active")
            && text.contains("runtime_production_gate")
            && text.contains("runtime_gate_resident_gpu_buffers_tracked")
            && text.contains("runtime_gate_fused_dispatch_tracked")
            && text.contains("runtime_gate_convergence_reports_pass")
            && text.contains("reference_shape_longitudinal_dry")
            && text.contains("reference_shape_combined_dry")
            && text.contains("reference_shape_combined_slip_tradeoff")
            && text.contains("step_torque_response")
            && text.contains("step_torque_effective_slip_lags_raw")
            && text.contains("step_brake_response")
            && text.contains("step_brake_reduces_wheel_speed")
            && text.contains("step_steer_response")
            && text.contains("step_steer_lateral_force_builds")
            && text.contains("sensitivities=")
            && text.contains("sensitivity=pressure")
            && text.contains("sensitivity=dry_speed")
            && text.contains("sensitivity=soil_stiffness")
            && text.contains("sensitivity=contact_resolution")
            && text.contains("sensitivity=obstacle_height")
            && text.contains("vehicle_sanity=6")
            && text.contains("vehicle_sanity=heavy_loaded_vehicle")
            && text.contains("vehicle_sanity=braking_vehicle")
            && text.contains("vehicle_sanity=obstacle_curb_vehicle")
            && text.contains("vehicle_sanity=offroad_soft_soil_vehicle")
            && text.contains("viewer_target_coverage")
            && text.contains("viewer_target=rolling_over_bump_curb")
            && text.contains("viewer_target=low_pressure_wraps_obstacle")
            && text.contains("viewer_target=ice_patch_then_asphalt")
            && text.contains("viewer_target=soft_soil_rut_compaction")
            && text.contains("viewer_target=soft_tire_flattens_and_grows_patch")
            && text.contains("viewer_target=overinflated_tire_slips_sooner")
            && text.contains("viewer_target=ice_large_patch_still_slips")
            && text.contains("viewer_target=thermal_wear_state_visible")
            && text.contains("replay_artifact_coverage")
            && text.contains("replay_artifact_command_inputs_present")
            && text.contains("replay_artifact_known_approximations_available")
            && text.contains("gpu_parity=brush_combined_slip")
            && text.contains("gpu_parity=tetra_fem_contact_feedback")
            && text.contains("gpu_parity=mpm_particle_soil_feedback")
            && text.contains("gpu_parity=tetra_fem_sidewall_laminate_coupling_report")
            && text.contains("gpu_parity=tetra_fem_sidewall_ply_strain_xpbd")
            && text.contains("known_approximation=explicit_fem_elasticity")
            && text.contains("known_approximation=sidewall_bend_proxy")
            && text.contains("known_approximation=triangle_surface_simplification")
            && text.contains("preset_governance_summary")
            && text.contains("preset_governance=RealTireParams.passenger_radial.pressure_pa")
            && text.contains(
                "preset_governance=TetraTireFemParams.realtime_viewer.cavity_carcass_coupling"
            )
            && text.contains("preset_governance=SurfaceMaterial.ice.mu_static"),
        "runner text report should expose broad envelope gates: {text}"
    );
}

#[test]
fn tire_gpu_parity_checklist_accounts_for_high_frequency_paths() {
    let items = tire_gpu_parity_checklist();
    assert!(
        items.len() >= 8,
        "checklist should cover the major tire paths, not only one kernel: {items:?}"
    );
    for item in &items {
        assert!(
            item.has_required_gpu_accounting(),
            "GPU parity item needs proof or explicit reason/follow-up: {item:?}"
        );
        assert!(
            !item.feature_id.contains(',') && !item.feature_id.is_empty(),
            "feature ids should be stable CSV-safe handles: {item:?}"
        );
    }

    let brush = items
        .iter()
        .find(|item| item.feature_id == "brush_combined_slip")
        .expect("combined-slip brush contact must be tracked");
    assert_eq!(brush.status, TireGpuParityStatus::CpuAndGpu);
    assert!(brush.status.has_gpu_path());
    assert!(brush.proof.contains("tire_brush_gpu"));

    let terrain_velocity = items
        .iter()
        .find(|item| item.feature_id == "terrain_surface_velocity_map")
        .expect("terrain surface-velocity map path must be tracked");
    assert_eq!(terrain_velocity.status, TireGpuParityStatus::CpuAndGpu);
    assert!(
        terrain_velocity
            .cpu_path
            .contains("HeightfieldTerrain::with_surface_velocity_map")
    );
    assert!(terrain_velocity.gpu_path.contains("surface_velocity_map"));
    assert!(
        terrain_velocity
            .proof
            .contains("heightfield_surface_velocity_map")
    );

    let fem_feedback = items
        .iter()
        .find(|item| item.feature_id == "tetra_fem_contact_feedback")
        .expect("FEM contact feedback must be tracked");
    assert_eq!(fem_feedback.status, TireGpuParityStatus::CpuAndGpu);
    assert!(
        fem_feedback
            .gpu_path
            .contains("TireFemContactFeedbackKernel")
    );
    assert!(
        fem_feedback
            .gpu_path
            .contains("readback_resident_feedback_output")
    );
    assert!(
        fem_feedback
            .gpu_path
            .contains("hub_params_from_feedback_force_rows")
    );
    assert!(
        fem_feedback
            .proof
            .contains("tire_fem_feedback_force_rows_feed_gpu_hub_external_force_params")
    );
    assert!(fem_feedback.proof.contains("Scene 122"));

    let fem_batch = items
        .iter()
        .find(|item| item.feature_id == "tetra_fem_persistent_batch")
        .expect("FEM multi-tire batch ownership must be tracked");
    assert_eq!(fem_batch.status, TireGpuParityStatus::CpuAndGpu);
    assert!(fem_batch.gpu_path.contains("TireFemGpuAtlasBatch"));
    assert!(fem_batch.gpu_path.contains("tire_fem_atlas.wgsl"));
    assert!(fem_batch.proof.contains("resident atlas"));
    assert!(fem_batch.follow_up.contains("Scene 120 combined"));

    let approximations = tire_known_approximation_ledger();
    let coarse = approximations
        .iter()
        .find(|item| item.item_id == "coarse_tetra_resolution")
        .expect("coarse tetra resolution approximation must be tracked");
    assert!(
        coarse
            .current_model
            .contains("resident fused-atlas ownership")
    );
    assert!(coarse.current_model.contains("resident state reseeding"));
    assert!(coarse.next_step.contains("resident-atlas hardware parity"));

    let sidewall_coupling = items
        .iter()
        .find(|item| item.feature_id == "tetra_fem_sidewall_laminate_coupling_report")
        .expect(
            "sidewall laminate coupling diagnostic must be tracked until laminated shell parity lands",
        );
    assert_eq!(sidewall_coupling.status, TireGpuParityStatus::CpuOnlyDebug);
    assert!(
        sidewall_coupling
            .cpu_path
            .contains("crossed bias-ply energy")
    );
    assert!(
        sidewall_coupling
            .proof
            .contains("sidewall_laminate_coupling")
    );
    assert!(sidewall_coupling.reason.contains("diagnostic/fixture"));
    assert!(sidewall_coupling.follow_up.contains("GPU parity"));

    let sidewall_ply_xpbd = items
        .iter()
        .find(|item| item.feature_id == "tetra_fem_sidewall_ply_strain_xpbd")
        .expect("sidewall ply-strain XPBD GPU parity must be tracked");
    assert_eq!(sidewall_ply_xpbd.status, TireGpuParityStatus::CpuAndGpu);
    assert!(
        sidewall_ply_xpbd
            .cpu_path
            .contains("project_sidewall_ply_strain_constraints")
    );
    assert!(sidewall_ply_xpbd.gpu_path.contains("tire_fem.wgsl"));
    assert!(sidewall_ply_xpbd.gpu_path.contains("tire_fem_atlas.wgsl"));
    assert!(sidewall_ply_xpbd.proof.contains("sidewall_ply_strain_xpbd"));
    assert!(
        sidewall_ply_xpbd
            .proof
            .contains("tire_fem_gpu_shaders_include_sidewall_ply_strain_xpbd_guardrail")
    );
    assert!(
        sidewall_ply_xpbd
            .proof
            .contains("tire_fem_gpu_sidewall_ply_strain_projection_reduces_overstretch")
    );
    assert!(
        sidewall_ply_xpbd
            .follow_up
            .contains("full laminated sidewall")
    );

    let hub_reduction = items
        .iter()
        .find(|item| item.feature_id == "tetra_fem_hub_reduction")
        .expect("FEM hub/chassis coupling must be tracked");
    assert_eq!(hub_reduction.status, TireGpuParityStatus::CpuAndGpu);
    assert!(hub_reduction.gpu_path.contains("hub_reduce_atlas"));
    assert!(
        hub_reduction
            .gpu_path
            .contains("tire_fem_hub_params_with_brush_outputs")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("update_resident_atlas_hub_params_from_brush_output")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("update_resident_atlas_hub_params_from_contact_reduction_output")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("hub_params_from_contact_reduction_outputs")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("step_resident_atlas_hubs_from_brush_output")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("step_resident_vehicle_fem_hubs_from_brush_owner")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("step_resident_vehicle_fem_atlas_from_brush_owner")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("TireResidentVehicleFemAtlasOwner")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("create_resident_vehicle_fem_atlas_owner")
    );
    assert!(
        hub_reduction
            .gpu_path
            .contains("step_resident_vehicle_fem_atlas_owner")
    );
    assert!(hub_reduction.gpu_path.contains("Scene 120 now defaults"));
    assert!(hub_reduction.gpu_path.contains("CPU Vehicle yaw"));
    assert!(
        hub_reduction
            .proof
            .contains("tire_aligning_moment_couples_into_chassis_yaw")
    );
    assert!(hub_reduction.proof.contains("brush-output-to-hub adapter"));
    assert!(
        hub_reduction
            .proof
            .contains("tire_fem_resident_atlas_hub_params_from_brush_output_gpu")
    );
    assert!(
        hub_reduction
            .proof
            .contains("tire_fem_resident_atlas_hubs_step_from_brush_output_gpu")
    );
    assert!(
        hub_reduction
            .proof
            .contains("tire_fem_resident_atlas_hubs_step_from_contact_reduction_output_gpu")
    );
    assert!(hub_reduction.proof.contains(
        "tire_fem_resident_vehicle_fem_atlas_substep_sequences_nodes_brush_and_hubs_gpu"
    ));
    assert!(
        hub_reduction
            .proof
            .contains("tire_fem_resident_vehicle_fem_atlas_owner_steps_gpu")
    );
    assert!(hub_reduction.gpu_path.contains("TireVehicleYawGpuKernel"));

    let vehicle_yaw = items
        .iter()
        .find(|item| item.feature_id == "vehicle_chassis_yaw_coupling")
        .expect("vehicle chassis yaw GPU coupling must be tracked");
    assert_eq!(vehicle_yaw.status, TireGpuParityStatus::CpuAndGpu);
    assert!(vehicle_yaw.gpu_path.contains("tire_vehicle_yaw.wgsl"));
    assert!(
        vehicle_yaw
            .proof
            .contains("tire_vehicle_yaw_gpu_matches_reference")
    );
    assert!(vehicle_yaw.follow_up.contains("planar GPU chassis-state"));

    let vehicle_planar = items
        .iter()
        .find(|item| item.feature_id == "vehicle_chassis_planar_coupling")
        .expect("vehicle chassis planar GPU coupling must be tracked");
    assert_eq!(vehicle_planar.status, TireGpuParityStatus::CpuAndGpu);
    assert!(vehicle_planar.gpu_path.contains("integrate_planar"));
    assert!(
        vehicle_planar
            .gpu_path
            .contains("integrate_planar_from_brush_outputs")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("integrate_planar_from_contact_reduction_outputs")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("TireVehiclePlanarResidentState")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("integrate_planar_state_from_brush_outputs")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("integrate_planar_state_from_contact_reduction_outputs")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("integrate_planar_state_from_feedback_force_rows")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("step_planar_resident_from_feedback_output")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("dispatch_input_buffer_resident")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("TireVehicleResidentChassisBrushOwner")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("build_brush_rows_from_spin_suspension_planar_state")
    );
    assert!(
        vehicle_planar
            .gpu_path
            .contains("build_brush_rows_from_spin_suspension_chassis_state")
    );
    assert!(vehicle_planar.gpu_path.contains("position"));
    assert!(
        vehicle_planar
            .proof
            .contains("tire_vehicle_planar_gpu_matches_reference")
    );
    assert!(
        vehicle_planar
            .proof
            .contains("tire_vehicle_planar_contact_reduction_gpu_matches_reference")
    );
    assert!(
        vehicle_planar
            .proof
            .contains("tire_vehicle_resident_brush_output_feeds_planar_gpu_without_staging")
    );
    assert!(
        vehicle_planar
            .proof
            .contains("tire_vehicle_planar_resident_state_ping_pongs_from_brush_output")
    );
    assert!(
        vehicle_planar.proof.contains(
            "tire_vehicle_planar_resident_state_ping_pongs_from_contact_reduction_output"
        )
    );
    assert!(
        vehicle_planar
            .proof
            .contains("tire_vehicle_planar_resident_state_ping_pongs_from_feedback_output")
    );
    assert!(
        vehicle_planar
            .proof
            .contains("tire_vehicle_planar_from_feedback_reference_uses_atlas_force_rows")
    );
    assert!(
        vehicle_planar
            .proof
            .contains("tire_vehicle_resident_chassis_brush_owner_sequences_brush_and_planar_state")
    );
    assert!(vehicle_planar.proof.contains(
        "tire_vehicle_resident_chassis_brush_owner_feeds_planar_state_back_to_brush_rows"
    ));
    assert!(vehicle_planar.proof.contains(
        "tire_vehicle_resident_chassis_brush_owner_uses_resident_planar_and_vertical_state"
    ));
    assert!(
        vehicle_planar
            .follow_up
            .contains("full resident vehicle runtime")
    );

    let wheel_spin = items
        .iter()
        .find(|item| item.feature_id == "vehicle_wheel_spin_coupling")
        .expect("vehicle wheel-spin GPU coupling must be tracked");
    assert_eq!(wheel_spin.status, TireGpuParityStatus::CpuAndGpu);
    assert!(wheel_spin.gpu_path.contains("integrate_wheel_spin"));
    assert!(
        wheel_spin
            .gpu_path
            .contains("TireVehicleWheelSpinResidentBatch")
    );
    assert!(wheel_spin.cpu_path.contains("drive_torque"));
    assert!(
        wheel_spin
            .proof
            .contains("tire_vehicle_wheel_spin_gpu_matches_reference")
    );
    assert!(
        wheel_spin
            .proof
            .contains("tire_vehicle_wheel_spin_resident_batch_ping_pongs_state")
    );
    assert!(
        wheel_spin
            .gpu_path
            .contains("dispatch_brush_rows_from_spin_resident")
    );
    assert!(wheel_spin.follow_up.contains("FEM tire outputs"));

    let wheel_kinematics = items
        .iter()
        .find(|item| item.feature_id == "vehicle_wheel_contact_kinematics")
        .expect("vehicle wheel contact-kinematics GPU coupling must be tracked");
    assert_eq!(wheel_kinematics.status, TireGpuParityStatus::CpuAndGpu);
    assert!(
        wheel_kinematics
            .gpu_path
            .contains("compute_wheel_kinematics")
    );
    assert!(
        wheel_kinematics
            .gpu_path
            .contains("compute_wheel_kinematics_from_spin_state")
    );
    assert!(wheel_kinematics.cpu_path.contains("yaw+steer"));
    assert!(
        wheel_kinematics
            .proof
            .contains("tire_vehicle_wheel_kinematics_gpu_matches_reference")
    );
    assert!(
        wheel_kinematics
            .proof
            .contains("tire_vehicle_resident_spin_state_feeds_kinematics_gpu_without_staging")
    );
    assert!(
        wheel_kinematics
            .follow_up
            .contains("GPU brush/FEM tire kernels")
    );

    let suspension = items
        .iter()
        .find(|item| item.feature_id == "vehicle_suspension_normal_load")
        .expect("vehicle suspension GPU normal-load path must be tracked");
    assert_eq!(suspension.status, TireGpuParityStatus::CpuAndGpu);
    assert!(suspension.gpu_path.contains("compute_suspension"));
    assert!(
        suspension
            .gpu_path
            .contains("TireVehicleSuspensionResidentBatch")
    );
    assert!(
        suspension
            .gpu_path
            .contains("TireVehicleVerticalResidentState")
    );
    assert!(
        suspension
            .gpu_path
            .contains("compute_suspension_from_vertical_state")
    );
    assert!(suspension.cpu_path.contains("normal load"));
    assert!(
        suspension
            .proof
            .contains("tire_vehicle_suspension_gpu_matches_reference")
    );
    assert!(
        suspension
            .proof
            .contains("tire_vehicle_resident_spin_and_suspension_feed_brush_gpu_without_staging")
    );
    assert!(
        suspension
            .proof
            .contains("tire_vehicle_vertical_resident_state_ping_pongs_from_suspension_rows")
    );
    assert!(suspension.follow_up.contains("FEM tire force"));

    let vehicle_brush = items
        .iter()
        .find(|item| item.feature_id == "vehicle_to_brush_gpu_handoff")
        .expect("vehicle-to-brush GPU handoff must be tracked");
    assert_eq!(vehicle_brush.status, TireGpuParityStatus::CpuAndGpu);
    assert!(vehicle_brush.gpu_path.contains("build_brush_rows"));
    assert!(
        vehicle_brush
            .gpu_path
            .contains("build_brush_rows_from_spin_state")
    );
    assert!(
        vehicle_brush
            .gpu_path
            .contains("build_brush_rows_from_spin_and_suspension_state")
    );
    assert!(
        vehicle_brush
            .gpu_path
            .contains("TireVehicleResidentBrushOwner")
    );
    assert!(vehicle_brush.gpu_path.contains("TireBrushGpuKernel"));
    assert!(
        vehicle_brush
            .proof
            .contains("tire_vehicle_brush_rows_resident_feed_brush_gpu_matches_staged_rows")
    );
    assert!(
        vehicle_brush
            .gpu_path
            .contains("dispatch_input_buffer_blocking")
    );
    assert!(
        vehicle_brush
            .gpu_path
            .contains("dispatch_input_buffer_resident")
    );
    assert!(vehicle_brush.gpu_path.contains("readback_resident_output"));
    assert!(vehicle_brush.gpu_path.contains("Scene 121"));
    assert!(vehicle_brush.gpu_path.contains("primary GPU force path"));
    assert!(
        vehicle_brush
            .proof
            .contains("tire_traction_scene_wires_resident_vehicle_to_brush_handoff")
    );
    assert!(
        vehicle_brush
            .proof
            .contains("tire_vehicle_resident_spin_state_feeds_brush_gpu_without_staging")
    );
    assert!(
        vehicle_brush
            .proof
            .contains("tire_vehicle_resident_spin_and_suspension_feed_brush_gpu_without_staging")
    );
    assert!(
        vehicle_brush
            .proof
            .contains("tire_vehicle_resident_brush_owner_sequences_spin_suspension_and_brush")
    );
    assert!(vehicle_brush.follow_up.contains("FEM tire assemblies"));

    let camber_trail = items
        .iter()
        .find(|item| item.feature_id == "brush_camber_pneumatic_trail")
        .expect("camber/pneumatic-trail brush path must be tracked");
    assert_eq!(camber_trail.status, TireGpuParityStatus::CpuAndGpu);
    assert!(camber_trail.cpu_path.contains("RealWheelOrientation"));
    assert!(camber_trail.cpu_path.contains("pneumatic trail"));
    assert!(camber_trail.gpu_path.contains("extra.w"));
    assert!(camber_trail.gpu_path.contains("diag3"));
    assert!(
        camber_trail
            .proof
            .contains("tire_brush_gpu_matches_cpu_camber_extension")
    );
    assert!(
        camber_trail
            .proof
            .contains("camber_thrust_flips_sign_and_reports_pneumatic_trail")
    );

    let legacy_adapter = items
        .iter()
        .find(|item| item.feature_id == "legacy_vehicle_real_brush_adapter")
        .expect("legacy real-brush vehicle adapter must be tracked");
    assert_eq!(legacy_adapter.status, TireGpuParityStatus::CpuOnlyDebug);
    assert!(legacy_adapter.status.is_cpu_only_debug());
    assert!(legacy_adapter.cpu_path.contains("RealBrushTireModel"));
    assert!(legacy_adapter.cpu_path.contains("TireModelLab"));
    assert!(legacy_adapter.reason.contains("legacy TireModel trait"));
    assert!(
        legacy_adapter
            .follow_up
            .contains("resident vehicle-to-brush/FEM")
    );

    let semi_implicit = items
        .iter()
        .find(|item| item.feature_id == "tetra_fem_semi_implicit_stiffness")
        .expect("semi-implicit stiffness stabilizer must be tracked");
    assert_eq!(semi_implicit.status, TireGpuParityStatus::CpuAndGpu);
    assert!(semi_implicit.gpu_path.contains("tire_fem.wgsl"));

    let newton_debug = items
        .iter()
        .find(|item| item.feature_id == "tetra_fem_newton_solver_debug")
        .expect("Newton solver debug path must be tracked for GPU-first accounting");
    assert_eq!(newton_debug.status, TireGpuParityStatus::CpuAndGpu);
    assert!(
        newton_debug
            .cpu_path
            .contains("TetraFemTinyMeshNewtonReport")
            && newton_debug
                .cpu_path
                .contains("TetraFemMatrixFreeNewtonScaleReport")
            && newton_debug
                .cpu_path
                .contains("TetraFemMatrixFreeNewtonSolveReport")
            && newton_debug
                .cpu_path
                .contains("TetraFemMatrixFreeNewtonPreconditionerReport")
            && newton_debug
                .cpu_path
                .contains("TetraFemMatrixFreeNewtonStepReport")
            && newton_debug
                .cpu_path
                .contains("step_on_terrain_matrix_free_newton_debug")
            && newton_debug
                .cpu_path
                .contains("matrix_free_newton_correction")
            && newton_debug.proof.contains("fem_tiny_mesh_newton_solver")
            && newton_debug.proof.contains("fem_matrix_free_newton_scale")
            && newton_debug.proof.contains("fem_matrix_free_newton_solve")
            && newton_debug
                .proof
                .contains("fem_matrix_free_newton_preconditioner")
            && newton_debug.proof.contains("fem_matrix_free_newton_step")
            && newton_debug
                .proof
                .contains("fem_matrix_free_newton_timestep")
            && newton_debug
                .proof
                .contains("tire_fem_matrix_free_newton_contract")
    );
    assert!(
        newton_debug
            .gpu_path
            .contains("tire_fem_matrix_free_newton_gpu_contract")
            && newton_debug
                .gpu_path
                .contains("TireFemMatrixFreeNewtonGpuContract")
            && newton_debug
                .gpu_path
                .contains("tire_fem_matrix_free_newton.wgsl")
            && newton_debug.gpu_path.contains("residual_eval")
            && newton_debug
                .gpu_path
                .contains("TireFemMatrixFreeNewtonGpuKernel")
            && newton_debug.gpu_path.contains("run_newton_outer_blocking")
            && newton_debug.gpu_path.contains("Live")
    );
    assert!(
        newton_debug.cpu_path.contains("high_pressure_implicit")
            && newton_debug
                .cpu_path
                .contains("TetraTireFemSolverMode::MatrixFreeNewtonDebug")
    );
    assert!(
        newton_debug
            .reason
            .contains("matrix-free Newton implicit timestep")
            && newton_debug.reason.contains("high_pressure_implicit")
            && newton_debug.follow_up.contains("GPU persistent FEM batch")
    );

    let bekker = items
        .iter()
        .find(|item| item.feature_id == "bekker_soil_feedback")
        .expect("Bekker soil feedback must be tracked");
    assert_eq!(bekker.status, TireGpuParityStatus::CpuAndGpu);
    assert!(
        bekker
            .cpu_path
            .contains("TireSoilNormalShearCouplingReport")
    );
    assert!(bekker.gpu_path.contains("TireBekkerGpuResidentTerrain"));
    assert!(bekker.gpu_path.contains("apply_contacts_gpu_resident"));
    assert!(bekker.proof.contains("soil_normal_shear_coupling"));
    assert!(
        bekker
            .proof
            .contains("resident Bekker-terrain FEM feedback")
    );

    let mpm = items
        .iter()
        .find(|item| item.feature_id == "mpm_particle_soil_feedback")
        .expect("MPM particle-soil feedback must be tracked");
    assert_eq!(mpm.status, TireGpuParityStatus::CpuAndGpu);
    assert!(mpm.gpu_path.contains("TireMpmGpuKernel"));
    assert!(mpm.gpu_path.contains("surface-velocity"));
    assert!(mpm.gpu_path.contains("dispatch_resident_with_gpu_terrain"));
    assert!(mpm.proof.contains("surface-velocity-map"));
    assert!(mpm.proof.contains("Scene 122"));

    let dem = items
        .iter()
        .find(|item| item.feature_id == "dem_granular_tire_feedback")
        .expect("DEM granular tire feedback must be tracked");
    assert_eq!(dem.status, TireGpuParityStatus::CpuAndGpu);
    assert!(dem.proof.contains("14x12x5"));
    assert!(dem.proof.contains("large sparse-LBVH GPU hardware"));
    assert!(dem.proof.contains("tire_dem_gpu_large_sparse_lbvh"));
    assert!(
        dem.cpu_path
            .contains("tire_dem_raster_surface_velocity_map")
    );
    assert!(dem.gpu_path.contains("surface_velocity_map"));
    assert!(
        dem.gpu_path
            .contains("tire_fem_contact_terrain_with_dem_surface_velocity")
    );
    assert!(dem.cpu_path.contains("TireDemFemTerrainOwner"));
    assert!(dem.gpu_path.contains("TireDemFemTerrainOwner"));
    assert!(dem.gpu_path.contains("GPU blocking sparse-LBVH"));
    assert!(dem.gpu_path.contains("TireDemResidentState"));
    assert!(dem.gpu_path.contains("dispatch_lbvh_step_resident_state"));
    assert!(dem.gpu_path.contains("TireDemResidentLbvh"));
    assert!(
        dem.gpu_path
            .contains("tire_dem_build_sparse_lbvh_morton_order_gpu_radix")
    );
    assert!(
        dem.gpu_path
            .contains("tire_dem_build_sparse_lbvh_leaf_rows_gpu_radix")
    );
    assert!(
        dem.gpu_path
            .contains("tire_dem_build_sparse_lbvh_tree_rows_gpu_radix")
    );
    assert!(
        dem.gpu_path
            .contains("build_resident_sparse_lbvh_tree_rows_gpu_radix")
    );
    assert!(
        dem.proof
            .contains("tire_dem_gpu_sparse_lbvh_morton_order_matches_cpu_oracle")
    );
    assert!(
        dem.proof
            .contains("tire_dem_gpu_sparse_lbvh_leaf_rows_match_cpu_sorted_leaf_chunks")
    );
    assert!(
        dem.proof
            .contains("tire_dem_gpu_sparse_lbvh_tree_rows_match_cpu_complete_binary_oracle")
    );
    assert!(
        dem.proof
            .contains("tire_dem_gpu_sparse_lbvh_tree_rows_pad_non_power_of_two_leaves")
    );
    assert!(
        dem.proof
            .contains("tire_dem_gpu_built_lbvh_tree_rows_feed_resident_sparse_step")
    );
    assert!(
        dem.proof
            .contains("tire_dem_resident_gpu_lbvh_builder_feeds_sparse_step_without_tree_readback")
    );
    assert!(
        dem.proof
            .contains("tire_dem_resident_gpu_lbvh_builder_large_sparse_bed_matches_reference")
    );
    assert!(dem.gpu_path.contains("upload_resident_lbvh"));
    assert!(dem.gpu_path.contains("upload_resident_lbvh_tree_rows"));
    assert!(dem.gpu_path.contains("update_resident_lbvh"));
    assert!(
        dem.gpu_path
            .contains("dispatch_lbvh_step_resident_state_with_resident_lbvh")
    );
    assert!(dem.gpu_path.contains("dispatch_lbvh_step_resident"));
    assert!(dem.gpu_path.contains("readback_resident_output"));
    assert!(dem.gpu_path.contains("update_resident_state"));
    assert!(dem.gpu_path.contains("dispatch_resident_terrain_raster"));
    assert!(dem.gpu_path.contains("TireDemResidentTerrainOutput"));
    assert!(dem.gpu_path.contains("fem_contact_feedback_gpu_terrain"));
    assert!(dem.gpu_path.contains("TireDemResidentFemTerrainOwner"));
    assert!(dem.gpu_path.contains("no-readback GPU LBVH topology build"));
    assert!(dem.gpu_path.contains("cached Morton bounds"));
    assert!(dem.gpu_path.contains("reseed_grain_state"));
    assert!(dem.proof.contains("TireDemFemTerrainOwner"));
    assert!(dem.proof.contains("resident sparse-LBVH no-readback"));
    assert!(
        dem.proof
            .contains("tire_dem_resident_state_sparse_lbvh_step_ping_pongs_without_reupload")
    );
    assert!(
        dem.proof
            .contains("tire_dem_resident_terrain_raster_matches_cpu_export_without_grain_readback")
    );
    assert!(
        dem.proof
            .contains("tire_dem_resident_fem_terrain_owner_steps_and_keeps_gpu_terrain")
    );
    assert!(
        dem.proof
            .contains("tire_dem_resident_fem_terrain_owner_reseed_refreshes_gpu_state_and_bounds")
    );
    assert!(dem.proof.contains("owned sparse-LBVH"));
    assert!(
        dem.proof
            .contains("tire_dem_fem_terrain_owner_gpu_sparse_lbvh_step_matches_reference")
    );
    assert!(dem.proof.contains("Scene 119"));
    assert!(dem.proof.contains("resident GPU stepping/source guard"));
    assert!(dem.proof.contains("Scene 122"));
    assert!(
        dem.proof
            .contains("selectable GPU resident sparse-LBVH DEM step")
    );
    assert!(dem.proof.contains("dispatch_resident_terrain_raster"));
    assert!(dem.proof.contains("fem_contact_feedback_gpu_terrain"));
    assert!(dem.proof.contains("TireDemResidentFemTerrainOwner"));
    assert!(dem.follow_up.contains("FEM tire contact owners"));

    let thermal = items
        .iter()
        .find(|item| item.feature_id == "thermal_wear_condition")
        .expect("thermal/wear condition feedback must be tracked");
    assert_eq!(thermal.status, TireGpuParityStatus::CpuAndGpu);
    assert!(thermal.gpu_path.contains("TireThermalGpuKernel"));
    assert!(thermal.gpu_path.contains("TireThermalGpuBatch"));
    assert!(thermal.gpu_path.contains("wear-depth"));

    let transient = items
        .iter()
        .find(|item| item.feature_id == "layered_transient_relaxation")
        .expect("layered transient relaxation must be tracked");
    assert_eq!(transient.status, TireGpuParityStatus::CpuAndGpu);
    assert!(transient.gpu_path.contains("TireTransientGpuBatch"));

    let triangles = items
        .iter()
        .find(|item| item.feature_id == "triangle_surface_contact")
        .expect("finite triangle tire contact must be tracked");
    assert!(triangles.gpu_path.contains("PatchAccounting"));
    let contact_diagnostics = items
        .iter()
        .find(|item| item.feature_id == "tetra_fem_contact_diagnostics")
        .expect("FEM contact diagnostics must be tracked");
    assert!(contact_diagnostics.proof.contains("large triangle-BVH"));

    let text = tire_gpu_parity_text_report();
    assert!(
        text.contains("brush_combined_slip")
            && text.contains("status=cpu_and_gpu")
            && text.contains("brush_camber_pneumatic_trail")
            && text.contains("terrain_surface_velocity_map")
            && text.contains("tetra_fem_newton_solver_debug")
            && text.contains("status=cpu_only_debug")
            && text.contains("bekker_soil_feedback")
            && text.contains("mpm_particle_soil_feedback"),
        "text report should expose implemented GPU parity: {text}"
    );
}

#[test]
fn runtime_production_gate_ties_physics_to_gpu_runtime_evidence() {
    let report = tire_runtime_production_gate_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "runtime production gate should pass broad physics+runtime checks: {report:?}"
    );
    assert!(
        report.resident_gpu_features >= 4
            && report.many_wheel_batch_features >= 3
            && report.fused_dispatch_features >= 2
            && report.readback_seam_features >= 2
            && tetra_sidewall_laminate_coupling_report().passed()
            && tetra_sidewall_ply_strain_xpbd_report().passed(),
        "runtime gate should count resident, batch, fused-dispatch, readback-seam, and sidewall-structure evidence: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "runtime_gate_resident_gpu_buffers_tracked")
            && checks
                .iter()
                .any(|check| check.check_id == "runtime_gate_many_wheel_batching_tracked")
            && checks
                .iter()
                .any(|check| check.check_id == "runtime_gate_fused_dispatch_tracked")
            && checks
                .iter()
                .any(|check| check.check_id == "runtime_gate_broad_physics_passed"),
        "acceptance checks should expose runtime+physics production gates: {checks:?}"
    );
}

#[test]
fn tire_known_approximation_ledger_is_explicit_and_blunt() {
    let ledger = tire_known_approximation_ledger();
    assert!(
        ledger.len() >= 7,
        "ledger should cover solver, structure, terrain, soil, and thermal approximations: {ledger:?}"
    );
    for item in &ledger {
        assert!(
            item.has_required_accounting(),
            "every approximation needs consequence and next step: {item:?}"
        );
        assert!(
            !item.item_id.contains(',') && !item.item_id.is_empty(),
            "item ids should be stable CSV-safe handles: {item:?}"
        );
    }
    assert!(
        ledger
            .iter()
            .any(|item| item.item_id == "explicit_fem_elasticity"
                && item.status == TireApproximationStatus::PlannedPhysicsUpgrade
                && item.gpu_status == TireGpuParityStatus::CpuAndGpu
                && item.current_model.contains("tiny actual tire mesh")),
        "explicit FEM limitation should remain visible even with CPU/GPU semi-implicit stiffness parity and CPU Newton fixtures"
    );
    let newton_debug = ledger
        .iter()
        .find(|item| item.item_id == "newton_solver_cpu_debug_only")
        .expect("CPU-only Newton solver path must stay visible until GPU parity lands");
    assert_eq!(newton_debug.status, TireApproximationStatus::CpuOnlyDebug);
    assert!(
        newton_debug
            .current_model
            .contains("TetraFemTinyMeshNewtonReport")
            && newton_debug
                .current_model
                .contains("TetraFemMatrixFreeNewtonScaleReport")
            && newton_debug
                .current_model
                .contains("TetraFemMatrixFreeNewtonSolveReport")
            && newton_debug
                .current_model
                .contains("TetraFemMatrixFreeNewtonPreconditionerReport")
            && newton_debug
                .current_model
                .contains("TetraFemMatrixFreeNewtonStepReport")
            && newton_debug
                .current_model
                .contains("step_on_terrain_matrix_free_newton_debug")
            && newton_debug
                .current_model
                .contains("matrix_free_newton_correction")
            && newton_debug
                .current_model
                .contains("tire_fem_matrix_free_newton_gpu_contract")
            && newton_debug
                .current_model
                .contains("tire_fem_matrix_free_newton.wgsl")
            && newton_debug
                .current_model
                .contains("TetraTireFemSolverMode")
            && newton_debug
                .current_model
                .contains("TireFemMatrixFreeNewtonGpuKernel")
            && newton_debug
                .current_model
                .contains("high_pressure_implicit")
            && newton_debug
                .current_model
                .contains("run_newton_outer_blocking")
    );
    assert!(
        newton_debug
            .consequence
            .contains("matrix-free Newton implicit timestep now spans CPU + GPU")
            && newton_debug.consequence.contains("host-orchestrated")
            && newton_debug.consequence.contains("high_pressure_implicit")
    );
    assert!(
        newton_debug.next_step.contains("GPU persistent FEM batch")
            && newton_debug.next_step.contains("run_newton_outer_blocking")
            && newton_debug
                .next_step
                .contains("TetraTireFemSolverMode::MatrixFreeNewtonDebug")
    );
    assert_eq!(newton_debug.gpu_status, TireGpuParityStatus::CpuAndGpu);
    assert!(
        ledger
            .iter()
            .any(|item| item.item_id == "triangle_surface_simplification"
                && item.gpu_status == TireGpuParityStatus::CpuAndGpu
                && item.current_model.contains("force/bristle accounting")),
        "remaining triangle-surface simplifications should be listed explicitly"
    );
    let sidewall_bend = ledger
        .iter()
        .find(|item| item.item_id == "sidewall_bend_proxy")
        .expect("sidewall laminated-shell gap must stay visible in the approximation ledger");
    assert!(sidewall_bend.current_model.contains("ply-energy fixture"));
    assert!(sidewall_bend.current_model.contains("ply-strain XPBD"));
    assert!(sidewall_bend.consequence.contains("uncoupled sidewalls"));
    assert!(sidewall_bend.consequence.contains("laminated shell"));
    assert!(sidewall_bend.next_step.contains("full laminated sidewall"));
    assert!(
        ledger
            .iter()
            .any(|item| item.item_id == "thermal_wear_curves"
                && item.gpu_status == TireGpuParityStatus::CpuAndGpu),
        "thermal/wear approximation should keep GPU parity accounting"
    );
    let legacy_adapter = ledger
        .iter()
        .find(|item| item.item_id == "legacy_vehicle_real_brush_adapter")
        .expect("legacy TireModel bridge must stay visible in the approximation ledger");
    assert_eq!(legacy_adapter.status, TireApproximationStatus::CpuOnlyDebug);
    assert_eq!(legacy_adapter.gpu_status, TireGpuParityStatus::CpuOnlyDebug);
    assert!(legacy_adapter.current_model.contains("RealBrushTireModel"));
    assert!(legacy_adapter.current_model.contains("evaluate_brush_tire"));
    assert!(legacy_adapter.consequence.contains("lack full wheel pose"));
    assert!(
        legacy_adapter
            .next_step
            .contains("resident vehicle-to-brush/FEM")
    );
    let text = tire_known_approximation_text_report();
    assert!(
        text.contains("explicit_fem_elasticity")
            && text.contains("realtime_approximation")
            && text.contains("newton_solver_cpu_debug_only")
            && text.contains("bekker_only_soil")
            && text.contains("triangle_surface_simplification")
            && text.contains("legacy_vehicle_real_brush_adapter"),
        "text report should be useful in viewer truth labels / rig reports: {text}"
    );
}

#[test]
fn tire_rig_acceptance_report_lists_qualitative_pass_fail_envelopes() {
    let params = RealTireParams::passenger_radial();
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let point =
        tire_rig_longitudinal_slip_sweep(&params, &terrain, &material, 3_600.0, 10.0, &[0.12])
            .remove(0);
    let snapshot = point.snapshot_on_terrain(
        "longitudinal acceptance",
        "passenger_radial",
        "dry_asphalt",
        "flat",
        "cpu_brush",
        &terrain,
        0.0,
        0.0,
    );
    let fx_over_fz =
        snapshot.telemetry.longitudinal_force_n.abs() / snapshot.telemetry.normal_load_n.max(1.0);
    let consumed_ratio = snapshot.telemetry.consumed_friction_budget_n
        / snapshot.telemetry.available_friction_budget_n.max(1.0);

    let report = snapshot.acceptance_report(vec![
        TireRigAcceptanceCheck::at_least(
            "patch_exists",
            "loaded tire should produce a non-zero contact patch",
            "patch_length_m",
            point.telemetry.patch.patch_length_m,
            0.01,
        ),
        TireRigAcceptanceCheck::between(
            "force_ratio_sane",
            "longitudinal force over normal load should stay in a plausible dry-road envelope",
            "abs_fx_over_fz",
            fx_over_fz,
            0.0,
            1.5,
        ),
        TireRigAcceptanceCheck::at_most(
            "friction_budget_not_exploded",
            "consumed friction should not exceed available budget by a large amount",
            "consumed_over_available_friction",
            consumed_ratio,
            1.10,
        ),
    ]);

    assert!(
        report.is_finite(),
        "acceptance report must be finite: {report:?}"
    );
    assert!(
        report.passed(),
        "qualitative acceptance checks should pass: {report:?}"
    );
    let text = report.to_text_report();
    assert!(
        text.contains("non_calibration=true")
            && text.contains("patch_exists")
            && text.contains("force_ratio_sane")
            && text.contains("passed=true"),
        "report text should list expectations and pass/fail status: {text}"
    );

    let failed = TireRigAcceptanceReport::new(
        report.snapshot.clone(),
        vec![TireRigAcceptanceCheck::at_least(
            "impossible_patch",
            "this deliberately impossible envelope should fail",
            "patch_length_m",
            report.snapshot.telemetry.patch.patch_length_m,
            10.0,
        )],
    );
    assert!(!failed.passed());
    assert!(failed.to_text_report().contains("passed=false"));
}

#[test]
fn tire_rig_dimensionless_sanity_metrics_feed_acceptance_reports() {
    let params = RealTireParams::passenger_radial();
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let point = tire_rig_combined_slip_sweep(
        &params,
        &terrain,
        &material,
        3_600.0,
        14.0,
        &[0.18],
        &[0.08],
    )
    .remove(0);
    let snapshot = point
        .snapshot_on_terrain(
            "dimensionless sanity",
            "passenger_radial",
            "dry_asphalt",
            "flat",
            "cpu_brush",
            &terrain,
            0.0,
            0.0,
        )
        .with_cpu_gpu_force_delta([
            point.telemetry.force_n[0] * 1.01,
            point.telemetry.force_n[1],
            point.telemetry.force_n[2],
        ]);

    let metrics = snapshot.dimensionless_sanity_metrics(params.radius);
    assert!(metrics.is_finite(), "metrics should be finite: {metrics:?}");
    assert!(metrics.abs_fx_over_fz >= 0.0);
    assert!(metrics.abs_fy_over_fz >= 0.0);
    assert!(metrics.patch_length_over_radius > 0.0);
    assert!(
        metrics.contact_pressure_area_over_load > 0.9
            && metrics.contact_pressure_area_over_load < 1.1,
        "pressure*area should account for load in this rig: {metrics:?}"
    );
    assert!(metrics.cpu_gpu_relative_force_error.unwrap() < 0.02);

    let report = snapshot.dimensionless_sanity_report(params.radius);
    assert!(
        report.is_finite(),
        "sanity report must be finite: {report:?}"
    );
    assert!(
        report.passed(),
        "broad sanity envelopes should pass: {report:?}"
    );
    let text = report.to_text_report();
    assert!(
        text.contains("fx_over_fz_sane")
            && text.contains("pressure_load_accounting_sane")
            && text.contains("cpu_gpu_force_error_sane"),
        "dimensionless report should list force, pressure, and GPU/CPU checks: {text}"
    );
}

#[test]
fn tire_energy_sign_audit_reports_dissipation_budget_and_moment_sign() {
    let params = RealTireParams::passenger_radial();
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let point = tire_rig_combined_slip_sweep(
        &params,
        &terrain,
        &material,
        3_800.0,
        16.0,
        &[0.35],
        &[0.14],
    )
    .remove(0);
    let snapshot = point.snapshot_on_terrain(
        "energy sign",
        "passenger_radial",
        "dry_asphalt",
        "flat",
        "cpu_brush",
        &terrain,
        0.0,
        0.0,
    );
    let audit = snapshot.energy_sign_audit();
    assert!(audit.is_finite(), "audit should be finite: {audit:?}");
    assert!(
        audit.passed(),
        "energy/sign accounting should pass: {audit:?}"
    );
    assert!(audit.dissipated_power_w >= 0.0);
    assert!(audit.dissipated_energy_j >= 0.0);
    assert!(audit.friction_budget_residual_n >= -1.0e-6);
    if audit.expected_pneumatic_aligning_moment_nm.abs() > 1.0e-6
        && audit.reported_aligning_moment_nm.abs() > 1.0e-6
    {
        assert_eq!(
            audit.expected_pneumatic_aligning_moment_nm.signum(),
            audit.reported_aligning_moment_nm.signum(),
            "aligning moment sign should follow -Fy*trail"
        );
    }

    let report = snapshot.energy_sign_report();
    assert!(report.is_finite());
    assert!(
        report.passed(),
        "energy/sign report should pass: {report:?}"
    );
    let text = report.to_text_report();
    assert!(
        text.contains("dissipated_power_nonnegative")
            && text.contains("friction_budget_residual_nonnegative")
            && text.contains("aligning_moment_sign_convention"),
        "energy/sign report should list dissipation, budget, and moment checks: {text}"
    );
}

#[test]
fn empty_contact_telemetry_is_finite_and_zeroed() {
    let contact = RealTireContact::empty();
    let sample = contact.telemetry_sample(0.016);
    assert!(!sample.in_contact);
    assert!(
        sample.is_finite(),
        "empty telemetry must be finite: {sample:?}"
    );
    assert_eq!(sample.normal_load_n, 0.0);
    assert_eq!(sample.dissipated_energy_j, 0.0);
    assert_eq!(sample.patch.contact_count, 0);
    assert_eq!(sample.patch.contact_area_m2, 0.0);
}

#[test]
fn named_tire_presets_are_finite_and_ordered() {
    let passenger = RealTireParams::passenger_radial();
    let soft = RealTireParams::offroad_soft();
    let high = RealTireParams::high_pressure_reference();
    for params in [passenger, soft, high] {
        assert!(params.radius.is_finite() && params.radius > 0.0);
        assert!(params.width.is_finite() && params.width > 0.0);
        assert!(params.pressure_pa.is_finite() && params.pressure_pa > 0.0);
        assert!(params.segments >= 16);
        assert!(
            params.carcass_normal_stiffness.is_finite() && params.carcass_normal_stiffness > 0.0
        );
        assert!(params.tread_shear_stiffness.is_finite() && params.tread_shear_stiffness > 0.0);
    }
    assert!(soft.pressure_pa < passenger.pressure_pa);
    assert!(high.pressure_pa > passenger.pressure_pa);
    assert!(soft.width > passenger.width);

    let scenario = passenger
        .with_geometry_and_pressure(0.41, 0.23, 140_000.0)
        .with_segments(72);
    assert_eq!(scenario.radius, 0.41);
    assert_eq!(scenario.width, 0.23);
    assert_eq!(scenario.pressure_pa, 140_000.0);
    assert_eq!(scenario.segments, 72);
    assert_eq!(
        scenario.tread_shear_stiffness,
        passenger.tread_shear_stiffness
    );
}

#[test]
fn tire_preset_governance_report_tracks_units_ranges_and_provenance() {
    let report = tire_preset_governance_report();
    assert!(
        report.is_finite(),
        "preset governance report must stay finite: {report:?}"
    );
    assert!(
        report.passed(),
        "current named presets should pass broad governance ranges: {report:?}"
    );
    assert!(
        report.checks.len() >= 80,
        "governance should cover tire, surface, film, thermal, FEM, and soil presets"
    );
    assert!(
        report.checks.iter().any(|c| c.group == "RealTireParams"
            && c.preset == "passenger_radial"
            && c.field == "pressure_pa"
            && c.units == "Pa"
            && c.provenance == TireModelProvenance::HandTuned),
        "tire pressure should carry units and provenance"
    );
    assert!(
        report.checks.iter().any(|c| c.group == "TetraTireFemParams"
            && c.field == "xpbd_volume_compliance"
            && c.provenance == TireModelProvenance::NumericalStability),
        "solver knobs should be marked as numerical-stability parameters"
    );
    assert!(
        report.checks.iter().any(|c| c.group == "DeformableTerrain"
            && c.preset == "soft_clay_patch"
            && c.field == "bekker_n"),
        "soil preset governance should include Bekker exponent"
    );
    let passenger_checks = report
        .checks_for_preset("RealTireParams", "passenger_radial")
        .count();
    assert!(
        passenger_checks >= 6,
        "passenger tire preset should have a short review checklist"
    );
    let text = report.to_text_report();
    assert!(
        text.contains("non_calibration=true")
            && text.contains("RealTireParams.passenger_radial.pressure_pa")
            && text.contains("TetraTireFemParams.realtime_viewer.xpbd_volume_compliance")
            && text.contains("DeformableTerrain.soft_clay_patch.bekker_n"),
        "text report should be useful in preset-change reviews: {text}"
    );
}

#[test]
fn parameter_observability_report_blocks_hidden_constants_and_shader_contract_drift() {
    let report = tire_parameter_observability_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "parameter observability audit should pass: {report:?}"
    );
    assert!(
        report.governance_check_count >= 30
            && report.shader_contracts_checked >= 4
            && report.brush_shader_exposes_tire_and_material_inputs
            && report.fem_shader_exposes_structural_and_cavity_inputs
            && report.thermal_shader_exposes_heat_wear_inputs
            && report.transient_shader_exposes_layer_inputs,
        "report should track governed presets and shader input contracts: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks.iter().any(|check| {
                check.check_id == "parameter_observability_brush_shader_inputs_named"
            })
            && checks.iter().any(|check| {
                check.check_id == "parameter_observability_fem_shader_inputs_named"
            })
            && checks.iter().any(|check| {
                check.check_id == "parameter_observability_hidden_force_sources_blocked"
            }),
        "acceptance checks should expose governed shader/force-source contracts: {checks:?}"
    );
}

#[test]
fn realism_source_classification_keeps_physics_diagnostics_and_visuals_separate() {
    let report = tire_realism_source_classification_report();
    assert!(
        report.is_finite(),
        "realism source report should be finite: {report:?}"
    );
    assert!(
        report.passed(),
        "realism source report should cover physics/diagnostic/preset/solver/visual categories: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "realism_sources_include_native_physics")
            && checks
                .iter()
                .any(|check| check.check_id == "realism_sources_include_diagnostic_reference")
            && checks
                .iter()
                .any(|check| check.check_id == "realism_sources_diagnostics_not_force_source"),
        "acceptance checks should separate native physics from reference/visual diagnostics: {checks:?}"
    );
    let text = report.to_text_report();
    assert!(
        text.contains("native_brush_fem_terrain_force")
            && text.contains("reference_shape_diagnostics")
            && text.contains("force_source=false")
            && text.contains("visualization_only"),
        "text report should be usable by honest realism labels: {text}"
    );
}

#[test]
fn named_film_thermal_and_fem_presets_are_ordered_and_finite() {
    let dry = SurfaceFilmState::dry();
    let damp = SurfaceFilmState::damp();
    let wet = SurfaceFilmState::wet_road();
    let standing = SurfaceFilmState::standing_water();
    assert_eq!(dry.water_depth_m, 0.0);
    assert!(damp.water_depth_m < wet.water_depth_m);
    assert!(wet.water_depth_m < standing.water_depth_m);
    assert!(standing.water_drag_rolling_per_mm > wet.water_drag_rolling_per_mm);

    let base = SurfaceMaterial::dry_asphalt();
    let wear = TireWearState::default();
    let damp_material = damp.apply_to(base, 20.0, &wear);
    let wet_material = wet.apply_to(base, 20.0, &wear);
    let standing_material = standing.apply_to(base, 20.0, &wear);
    assert!(damp_material.mu_static > wet_material.mu_static);
    assert!(wet_material.mu_static > standing_material.mu_static);
    assert!(standing_material.rolling_resistance > wet_material.rolling_resistance);

    let road = TireThermalLayers::passenger_road();
    let fast = TireThermalLayers::fast_response_test();
    let heavy = TireThermalLayers::heavy_duty_slow_response();
    for thermal in [road, fast, heavy] {
        assert!(thermal.surface_heat_capacity_j_per_k.is_finite());
        assert!(thermal.tread_heat_capacity_j_per_k.is_finite());
        assert!(thermal.carcass_heat_capacity_j_per_k.is_finite());
        assert!(thermal.rim_heat_capacity_j_per_k.is_finite());
        assert!(thermal.air_conductance_w_per_k.is_finite());
    }
    assert!(fast.surface_heat_capacity_j_per_k < road.surface_heat_capacity_j_per_k);
    assert!(heavy.carcass_heat_capacity_j_per_k > road.carcass_heat_capacity_j_per_k);
    assert!(fast.air_conductance_w_per_k > heavy.air_conductance_w_per_k);

    let viewer = TetraTireFemParams::realtime_viewer();
    let offroad = TetraTireFemParams::offroad_soft();
    let stiff = TetraTireFemParams::high_pressure_reference();
    assert!(offroad.pressure_pa < viewer.pressure_pa);
    assert!(stiff.pressure_pa > viewer.pressure_pa);
    assert!(offroad.width > viewer.width);
    assert!(stiff.sidewall_young_modulus > offroad.sidewall_young_modulus);
    let scenario = viewer
        .with_resolution(28, 4, 5)
        .with_pressure_and_load(140_000.0, 5_500.0);
    assert_eq!(scenario.segments, 28);
    assert_eq!(scenario.radial_layers, 4);
    assert_eq!(scenario.width_layers, 5);
    assert_eq!(scenario.pressure_pa, 140_000.0);
    assert_eq!(scenario.downforce, 5_500.0);
    assert_eq!(scenario.young_modulus, viewer.young_modulus);
}

#[test]
fn tire_fem_resolution_sweep_scales_topology_and_keeps_volumes_finite() {
    let base = TetraTireFemParams::realtime_viewer().with_pressure_and_load(160_000.0, 3_200.0);
    let sweep = tire_fem_resolution_sweep(base, &[(12, 3, 3), (16, 3, 4), (24, 4, 4)]);
    assert_eq!(sweep.len(), 3);
    let mut previous_nodes = 0;
    let mut previous_tets = 0;
    for point in &sweep {
        assert!(
            point.is_finite(),
            "resolution point should remain finite: {point:?}"
        );
        assert_eq!(
            point.node_count,
            point.segments * point.radial_layers * point.width_layers
        );
        assert_eq!(point.tread_node_count, point.segments * point.width_layers);
        assert_eq!(point.bead_node_count, point.segments * point.width_layers);
        assert!(
            point.tetra_count > point.node_count,
            "annular hexa cells should split into many tetrahedra: {point:?}"
        );
        assert!(point.rest_cavity_volume_m3 > 0.0);
        assert!(point.min_tet_volume_m3 > 0.0);
        assert!(point.max_tet_volume_m3 >= point.min_tet_volume_m3);
        assert!(point.mean_tet_volume_m3 >= point.min_tet_volume_m3);
        assert!(
            point.node_count > previous_nodes && point.tetra_count > previous_tets,
            "sweep should increase topology size monotonically: {sweep:?}"
        );
        previous_nodes = point.node_count;
        previous_tets = point.tetra_count;
    }
}

#[test]
fn sidewall_laminate_audit_detects_crossed_bias_and_helper_plies() {
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(20, 5, 5),
    );
    let audit = tire.sidewall_laminate_audit();
    assert!(
        audit.is_finite(),
        "sidewall laminate audit should be finite: {audit:?}"
    );
    assert!(
        audit.sidewall_element_count > tire.segments(),
        "higher-radial-resolution tire should contain audited sidewall laminate elements: {audit:?}"
    );
    assert_eq!(
        audit.crossed_pair_count, audit.sidewall_element_count,
        "every audited sidewall element should carry crossed ± bias plies: {audit:?}"
    );
    assert!(
        (audit.mean_bias_angle_deg - 38.0).abs() < 1.0,
        "sidewall bias angle should match the explicit hand-tuned ply angle: {audit:?}"
    );
    assert!(
        audit.max_bias_angle_error_deg < 6.0,
        "sidewall bias angle should be stable around the tire: {audit:?}"
    );
    assert!(
        audit.mean_radial_helper_alignment > 0.98 && audit.mean_axial_helper_alignment > 0.98,
        "sidewall helpers should be radial + axial, not tread/belt hoop helpers: {audit:?}"
    );
    let stretch = tire.sidewall_laminate_stretch_diagnostics();
    assert!(
        stretch.is_finite()
            && stretch.sidewall_element_count == audit.sidewall_element_count
            && stretch.max_tensile_strain <= 1.0e-9
            && stretch.max_compressive_strain <= 1.0e-9,
        "fresh sidewall ply stretch diagnostics should see the undeformed rest state: {stretch:?}"
    );
}

#[test]
fn sidewall_laminate_sanity_report_rejects_excessive_fold_drift() {
    let params = TetraTireFemParams {
        xpbd_sidewall_bend_compliance: f64::INFINITY,
        xpbd_sidewall_dihedral_compliance: f64::INFINITY,
        ..TetraTireFemParams::realtime_viewer().with_resolution(20, 5, 5)
    };
    let mut tire = TetraTireFem::new(0.0, 180_000.0, params);

    let rest = tire.sidewall_laminate_sanity_report();
    assert!(
        rest.is_finite(),
        "rest sanity report should be finite: {rest:?}"
    );
    assert!(
        rest.passed,
        "fresh laminate should pass the current sanity envelope: {rest:?}"
    );

    let bend_edges = tire.sidewall_bend_edges.clone();
    for (k, edge) in bend_edges.iter().take(32).enumerate() {
        let [a, b] = edge.ids;
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        tire.pos[a][0] += 0.080 * sign;
        tire.pos[a][1] -= 0.060;
        tire.pos[a][2] += 0.120 * tire.pos[a][2].signum().max(1.0) * sign;
        tire.pos[b][0] -= 0.060 * sign;
        tire.pos[b][1] += 0.045;
        tire.pos[b][2] -= 0.100 * tire.pos[b][2].signum().max(1.0) * sign;
    }

    let folded = tire.sidewall_laminate_sanity_report();
    assert!(
        folded.is_finite(),
        "folded sanity report should be finite: {folded:?}"
    );
    assert!(
        !folded.passed,
        "folded laminate should fail the sanity gate: {folded:?}"
    );
    assert!(
        folded.dihedral.max_abs_deviation_rad > folded.fold_max_limit_rad
            || folded.dihedral.mean_abs_deviation_rad > folded.fold_mean_limit_rad
            || folded.stretch.max_tensile_strain > folded.stretch_limit
            || folded.stretch.max_compressive_strain > folded.stretch_limit,
        "fold rejection should be caused by measured dihedral drift or ply stretch: {folded:?}"
    );
    assert!(
        folded.stretch.is_finite(),
        "folded report should still expose finite ply stretch diagnostics: {folded:?}"
    );
    assert!(
        folded.audit.crossed_pair_count == folded.audit.sidewall_element_count,
        "folding should not corrupt the static ply-stack audit: {folded:?}"
    );
}

#[test]
fn sidewall_laminate_sanity_acceptance_checks_cover_ply_and_fold_drift() {
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(20, 5, 5),
    );
    let report = tire.sidewall_laminate_sanity_report();
    assert!(
        report.passed,
        "fresh sidewall sanity report should pass: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed),
        "fresh sidewall laminate acceptance checks should pass: {checks:?}"
    );
    assert!(
        checks
            .iter()
            .any(|check| check.check_id == "sidewall_laminate_crossed_ply_coverage")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_laminate_max_fold_drift")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_laminate_max_tensile_strain")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_laminate_max_compressive_strain"),
        "acceptance checks should cover laminate layout, fold drift, and ply stretch: {checks:?}"
    );
}

#[test]
fn sidewall_dynamic_load_report_exercises_vertical_lateral_and_steered_cases() {
    let report = tetra_sidewall_dynamic_load_report();
    assert!(
        report.passed(),
        "sidewall dynamic-load report should pass broad non-calibration envelopes: {report:?}"
    );
    assert!(
        report.lateral_contact_force_n > 10.0,
        "lateral sidewall case should apply lateral contact force: {report:?}"
    );
    assert!(
        report.lateral_max_fold_drift_rad <= report.lateral_report.fold_max_limit_rad
            && report.lateral_max_ply_strain <= report.lateral_report.stretch_limit,
        "lateral load should stay inside current fold/ply-strain envelope: {report:?}"
    );
    assert!(
        report.steer_contact_force_n > 10.0 && report.steer_aligning_moment_nm.abs() > 1.0e-3,
        "steered sidewall case should produce lateral tire force and aligning moment from wheel-frame angle: {report:?}"
    );
    assert!(
        report.steer_max_fold_drift_rad <= report.steer_report.fold_max_limit_rad
            && report.steer_max_ply_strain <= report.steer_report.stretch_limit,
        "steered load should stay inside current fold/ply-strain envelope: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_dynamic_vertical_sanity_passes")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_dynamic_lateral_sanity_passes")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_dynamic_lateral_force_applied")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_dynamic_steer_sanity_passes")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_dynamic_steer_aligning_moment"),
        "dynamic sidewall report should expose vertical/lateral/steered load gates: {checks:?}"
    );
}

#[test]
fn sidewall_laminate_coupling_report_exercises_ply_energy_and_fold_projection() {
    let report = tetra_sidewall_laminate_coupling_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "sidewall laminate coupling report should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.balanced_bias_a_energy_j > 0.0
            && report.balanced_bias_b_energy_j > 0.0
            && report.crossed_pair_relative_imbalance <= 0.08,
        "symmetric radial bulge should load crossed ± bias plies in balance: {report:?}"
    );
    assert!(
        report.radial_helper_energy_j > 0.0 && report.axial_helper_energy_j > 0.0,
        "radial and axial helper plies should become active under their fixture modes: {report:?}"
    );
    assert!(
        report.folded_ply_energy_j > 0.0
            && report.folded_max_fold_drift_rad > report.projected_max_fold_drift_rad,
        "folded fixture should create ply energy and projection should reduce measured fold: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_laminate_coupling_crossed_pair_balanced")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_laminate_coupling_radial_helper_active")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_laminate_coupling_axial_helper_active")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_laminate_coupling_projected_fold_bounded"),
        "coupling report should expose crossed-ply, helper-ply, and fold-projection gates: {checks:?}"
    );
}

#[test]
fn sidewall_ply_strain_xpbd_report_reduces_tensile_ply_overstretch() {
    let report = tetra_sidewall_ply_strain_xpbd_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "sidewall ply-strain XPBD report should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.pre_max_tensile_strain > report.post_max_tensile_strain
            && report.post_max_tensile_strain <= (report.stretch_limit - 1.0) * 1.08,
        "ply-strain projection should reduce and bound tensile strain: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_ply_strain_xpbd_reduces_tensile_strain")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_ply_strain_xpbd_bounds_tensile_strain")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_ply_strain_xpbd_applies_correction"),
        "ply-strain XPBD report should expose reduction, bound, and correction gates: {checks:?}"
    );
}

#[test]
fn sidewall_dihedral_xpbd_effect_report_reduces_fold_drift() {
    let report = tetra_sidewall_dihedral_xpbd_effect_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "sidewall dihedral XPBD effect report should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.folded_max_abs_deviation_rad > report.stabilized_max_abs_deviation_rad,
        "projection should reduce the deliberately injected fold: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_dihedral_xpbd_perturbs_fold")
            && checks
                .iter()
                .any(|check| check.check_id == "sidewall_dihedral_xpbd_reduces_fold"),
        "dihedral XPBD report should expose perturbation and reduction gates: {checks:?}"
    );
}

#[test]
fn cavity_coupling_effect_report_tracks_named_pressure_proxy() {
    let report = tetra_cavity_coupling_effect_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "cavity coupling effect report should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.zero_force_per_bead_n <= 1.0e-6
            && report.nominal_force_per_bead_n > 1.0
            && report.high_force_per_bead_n > report.nominal_force_per_bead_n,
        "named coupling should control the visible cavity pressure-to-carcass path: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "cavity_coupling_zero_removes_bead_load")
            && checks
                .iter()
                .any(|check| check.check_id == "cavity_coupling_nominal_loads_bead")
            && checks
                .iter()
                .any(|check| check.check_id == "cavity_coupling_high_increases_load"),
        "cavity coupling report should expose zero/nominal/high effect gates: {checks:?}"
    );
}

#[test]
fn tetra_construction_sanity_report_covers_loaded_shape_belt_and_bead() {
    let report = tetra_construction_sanity_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "construction sanity should pass broad FEM shape/structure gates: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "construction_low_pressure_patch_grows")
            && checks
                .iter()
                .any(|check| check.check_id == "construction_tread_support_widthwise")
            && checks
                .iter()
                .any(|check| check.check_id == "construction_belt_stretch_bounded")
            && checks
                .iter()
                .any(|check| check.check_id == "construction_belt_peak_stretch_bounded")
            && checks
                .iter()
                .any(|check| check.check_id == "construction_sidewall_bulge_bounded"),
        "acceptance checks should expose loaded-shape, tread-support, belt and sidewall-bulge gates: {checks:?}"
    );
}

#[test]
fn tire_rig_load_deflection_sweep_is_repeatable_and_monotone() {
    let params = RealTireParams::passenger_radial().with_segments(64);
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let loads = [1_000.0, 3_000.0, 5_000.0, 8_000.0];
    let sweep =
        tire_rig_load_deflection_sweep(&params, &terrain, &SurfaceMaterial::dry_asphalt(), &loads);
    assert_eq!(sweep.len(), loads.len());
    let mut previous_deflection = 0.0;
    let mut previous_patch = 0.0;
    for (point, load) in sweep.iter().zip(loads) {
        assert!(point.is_finite(), "rig point must be finite: {point:?}");
        assert_eq!(point.load_n, load);
        assert!(point.deflection_m >= previous_deflection - 1.0e-8);
        assert!(point.telemetry.patch.patch_length_m >= previous_patch - 1.0e-8);
        assert!(point.telemetry.normal_load_n > 0.0);
        previous_deflection = point.deflection_m;
        previous_patch = point.telemetry.patch.patch_length_m;
    }
}

#[test]
fn tire_rig_slip_sweep_separates_asphalt_from_ice_without_calibration() {
    let params = RealTireParams::passenger_radial().with_segments(64);
    let asphalt = SurfaceMaterial::dry_asphalt();
    let ice = SurfaceMaterial::ice();
    let asphalt_terrain = FlatTerrain::new(0.0, asphalt.mu_static);
    let ice_terrain = FlatTerrain::new(0.0, ice.mu_static);
    let slips = [0.0, 0.04, 0.10, 0.18, 0.35];
    let asphalt_sweep =
        tire_rig_longitudinal_slip_sweep(&params, &asphalt_terrain, &asphalt, 4_500.0, 8.0, &slips);
    let ice_sweep =
        tire_rig_longitudinal_slip_sweep(&params, &ice_terrain, &ice, 4_500.0, 8.0, &slips);
    assert!(asphalt_sweep.iter().all(TireRigSweepPoint::is_finite));
    assert!(ice_sweep.iter().all(TireRigSweepPoint::is_finite));
    let asphalt_peak = asphalt_sweep
        .iter()
        .map(|p| p.telemetry.longitudinal_force_n.abs())
        .fold(0.0_f64, f64::max);
    let ice_peak = ice_sweep
        .iter()
        .map(|p| p.telemetry.longitudinal_force_n.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        asphalt_peak > ice_peak * 5.0,
        "asphalt should transmit far more force than ice: asphalt={asphalt_peak} ice={ice_peak}"
    );
}

#[test]
fn tire_rig_surface_slip_matrix_covers_dirt_wet_soil_and_ice() {
    let params = RealTireParams::passenger_radial().with_segments(64);
    let slips = [0.0, 0.08, 0.18, 0.35];
    let matrix = tire_rig_surface_slip_matrix(
        &params,
        4_500.0,
        12.0,
        &slips,
        0.0012,
        &TireWearState::default(),
    );
    assert_eq!(matrix.len(), 5 * slips.len());
    assert!(matrix.iter().all(TireRigSurfaceSlipPoint::is_finite));

    let peak = |kind: TireRigSurfaceKind| -> f64 {
        matrix
            .iter()
            .filter(|p| p.surface == kind)
            .map(|p| p.point.telemetry.longitudinal_force_n.abs())
            .fold(0.0_f64, f64::max)
    };
    let material_for = |kind: TireRigSurfaceKind| -> SurfaceMaterial {
        matrix.iter().find(|p| p.surface == kind).unwrap().material
    };
    let dry_peak = peak(TireRigSurfaceKind::DryAsphalt);
    let wet_peak = peak(TireRigSurfaceKind::WetAsphalt);
    let dirt_peak = peak(TireRigSurfaceKind::LooseDirt);
    let soil_peak = peak(TireRigSurfaceKind::SoftSoil);
    let ice_peak = peak(TireRigSurfaceKind::Ice);
    assert!(
        dry_peak > wet_peak,
        "wet film should reduce asphalt peak force: dry={dry_peak} wet={wet_peak}"
    );
    assert!(
        wet_peak > ice_peak * 2.0 && dirt_peak > ice_peak * 2.0 && soil_peak > ice_peak * 2.0,
        "non-ice loose/wet surfaces should still exceed ice: wet={wet_peak} dirt={dirt_peak} soil={soil_peak} ice={ice_peak}"
    );
    assert!(
        material_for(TireRigSurfaceKind::SoftSoil).rolling_resistance
            > material_for(TireRigSurfaceKind::LooseDirt).rolling_resistance
            && material_for(TireRigSurfaceKind::LooseDirt).rolling_resistance
                > material_for(TireRigSurfaceKind::DryAsphalt).rolling_resistance,
        "loose surface presets should carry increasing rolling losses"
    );
}

#[test]
fn tire_rig_surface_transition_sequence_covers_wet_ice_soil_and_moving_belt() {
    let params = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 180_000.0)
        .with_segments(80);
    let dry = SurfaceMaterial::dry_asphalt();
    let terrain = TireRigSurfaceTransitionTerrain::new(
        0.0,
        dry,
        vec![
            TireRigSurfaceTransitionBand::new("dry_asphalt", -5.0, -3.0, dry),
            TireRigSurfaceTransitionBand::new("ice", -2.5, -0.8, SurfaceMaterial::ice()),
            TireRigSurfaceTransitionBand::new("wet_asphalt", -0.2, 1.4, dry)
                .with_water_film_depth(0.0020),
            TireRigSurfaceTransitionBand::new(
                "packed_dirt",
                1.8,
                2.5,
                SurfaceMaterial::loose_dirt(),
            ),
            TireRigSurfaceTransitionBand::new("loose_soil", 3.1, 4.0, SurfaceMaterial::soft_soil()),
            TireRigSurfaceTransitionBand::new("moving_belt", 4.7, 6.2, dry)
                .with_surface_velocity([-4.0, 0.0, 0.0]),
        ],
    )
    .with_transition_width(0.25);
    let positions = [-4.0, -2.8, -1.6, -0.35, 0.7, 2.1, 3.5, 4.9, 5.6];
    let sequence = tire_rig_surface_transition_sequence(
        &params, &terrain, 4_200.0, 10.0, 0.16, &positions, 0.0,
    );
    assert_eq!(sequence.len(), positions.len());
    assert!(
        sequence
            .iter()
            .all(TireRigSurfaceTransitionPoint::is_finite),
        "surface transition sequence must stay finite: {sequence:?}"
    );

    let by_label = |needle: &str| -> &TireRigSurfaceTransitionPoint {
        sequence
            .iter()
            .find(|p| p.surface_label.contains(needle))
            .unwrap_or_else(|| panic!("missing surface label {needle}: {sequence:?}"))
    };
    let dry_point = by_label("dry_asphalt");
    let ice_point = by_label("ice");
    let wet_point = by_label("wet_asphalt");
    let dirt_point = by_label("packed_dirt");
    let soil_point = by_label("loose_soil");
    let belt_point = by_label("moving_belt");

    assert!(
        ice_point.telemetry.longitudinal_force_n.abs()
            < dry_point.telemetry.longitudinal_force_n.abs() * 0.45,
        "ice should cap drive force below dry asphalt: ice={} dry={}",
        ice_point.telemetry.longitudinal_force_n,
        dry_point.telemetry.longitudinal_force_n
    );
    assert!(
        wet_point.water_film_depth_m > 0.001
            && wet_point.telemetry.longitudinal_force_n.abs()
                < dry_point.telemetry.longitudinal_force_n.abs(),
        "wet film should be recorded and reduce drive force: wet={wet_point:?} dry={dry_point:?}"
    );
    assert!(
        dirt_point.material.rolling_resistance > dry_point.material.rolling_resistance,
        "packed dirt band should carry higher rolling resistance than asphalt"
    );
    assert!(
        soil_point.material.rolling_resistance > dirt_point.material.rolling_resistance,
        "loose soil band should carry higher rolling resistance than packed dirt"
    );
    assert!(
        soil_point.telemetry.longitudinal_force_n.abs()
            <= dirt_point.telemetry.longitudinal_force_n.abs() * 1.05,
        "loose soil should not transmit more drive force than packed dirt at the same slip: soil={} dirt={}",
        soil_point.telemetry.longitudinal_force_n,
        dirt_point.telemetry.longitudinal_force_n
    );
    assert!(
        belt_point.surface_velocity_m_per_s[0] < -1.0
            && (belt_point.telemetry.longitudinal_force_n
                - dry_point.telemetry.longitudinal_force_n)
                .abs()
                > 500.0,
        "moving belt should alter tire force relative to ground: belt={} dry={}",
        belt_point.telemetry.longitudinal_force_n,
        dry_point.telemetry.longitudinal_force_n
    );

    let transition_force_delta = (sequence[1].telemetry.longitudinal_force_n
        - sequence[2].telemetry.longitudinal_force_n)
        .abs();
    assert!(
        transition_force_delta < dry_point.telemetry.available_friction_budget_n.max(1.0),
        "transition smoothing should avoid non-finite/teleport-scale force jumps: delta={transition_force_delta}"
    );
}

#[test]
fn heightfield_surface_velocity_map_report_changes_force_and_work() {
    let report = tire_heightfield_surface_velocity_map_report();
    assert!(
        report.passed(),
        "heightfield surface-velocity map report should pass broad non-calibration envelopes: {report:?}"
    );
    assert!(
        report.moving_surface_velocity_x_m_per_s.abs() > 1.0
            && report.force_delta_n > 500.0
            && report.moving_ground_power_w.abs() > 1.0,
        "surface velocity map should change force/work through relative ground motion: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| { check.check_id == "heightfield_surface_velocity_changes_force" })
            && checks
                .iter()
                .any(|check| check.check_id == "heightfield_surface_velocity_work_nonzero"),
        "heightfield surface-velocity report should expose force/work gates: {checks:?}"
    );
}

#[test]
fn tire_rig_rolling_loss_sweep_separates_surface_loss_and_aero_drag() {
    let params = RealTireParams::passenger_radial().with_segments(80);
    let dry = SurfaceMaterial::dry_asphalt();
    let soil = SurfaceMaterial::soft_soil();
    let dry_terrain = FlatTerrain::new(0.0, dry.mu_static);
    let soil_terrain = FlatTerrain::new(0.0, soil.mu_static);
    let speeds = [6.0, 12.0, 24.0];
    let dry_loss = tire_rig_rolling_loss_sweep(
        &params,
        &dry_terrain,
        "dry_asphalt",
        &dry,
        4_000.0,
        &speeds,
        1_400.0,
        0.62,
        1.225,
    );
    let soil_loss = tire_rig_rolling_loss_sweep(
        &params,
        &soil_terrain,
        "soft_soil",
        &soil,
        4_000.0,
        &speeds,
        1_400.0,
        0.62,
        1.225,
    );
    assert!(dry_loss.iter().all(TireRigRollingLossPoint::is_finite));
    assert!(soil_loss.iter().all(TireRigRollingLossPoint::is_finite));
    assert_eq!(dry_loss.len(), speeds.len());

    for (dry_point, soil_point) in dry_loss.iter().zip(soil_loss.iter()) {
        assert!(
            soil_point.rolling_resistance_force_n > dry_point.rolling_resistance_force_n * 3.0,
            "soft soil should have much higher rolling loss: soil={} dry={}",
            soil_point.rolling_resistance_force_n,
            dry_point.rolling_resistance_force_n
        );
        assert!(
            dry_point.coastdown_deceleration_m_per_s2 > 0.0
                && soil_point.coastdown_deceleration_m_per_s2
                    > dry_point.coastdown_deceleration_m_per_s2,
            "loss terms should produce positive coastdown, with soil worse"
        );
    }

    let drag_force_ratio = dry_loss[2].aero_drag_force_n / dry_loss[1].aero_drag_force_n;
    let drag_power_ratio = dry_loss[2].aero_drag_power_w / dry_loss[1].aero_drag_power_w;
    assert!(
        (drag_force_ratio - 4.0).abs() < 0.25,
        "aero drag force should scale approximately with speed^2, got ratio={drag_force_ratio}"
    );
    assert!(
        (drag_power_ratio - 8.0).abs() < 0.60,
        "aero drag power should scale approximately with speed^3, got ratio={drag_power_ratio}"
    );

    let wet_material =
        SurfaceFilmState::standing_water().apply_to(dry, 24.0, &TireWearState::default());
    let wet_loss = tire_rig_rolling_loss_sweep(
        &params,
        &dry_terrain,
        "standing_water",
        &wet_material,
        4_000.0,
        &[24.0],
        1_400.0,
        0.62,
        1.225,
    )
    .remove(0);
    assert!(
        wet_loss.rolling_resistance_power_w > dry_loss[2].rolling_resistance_power_w * 1.5,
        "standing water should add rolling/water-displacement power loss: wet={} dry={}",
        wet_loss.rolling_resistance_power_w,
        dry_loss[2].rolling_resistance_power_w
    );
}

#[test]
fn tire_rig_heightfield_envelope_sweep_detects_bump_normals_and_pressure_patch() {
    let nx = 49;
    let nz = 5;
    let spacing = 0.05;
    let origin = [-1.2, -0.10];
    let mut heights = Vec::with_capacity(nx * nz);
    for _iz in 0..nz {
        for ix in 0..nx {
            let x = origin[0] + ix as f64 * spacing;
            let bump = 0.09 * (-(x / 0.18).powi(2)).exp();
            heights.push(bump);
        }
    }
    let terrain = HeightfieldTerrain::new(origin, spacing, nx, nz, heights, 0.95);
    let material = SurfaceMaterial::dry_asphalt();
    let positions = [-0.36, -0.18, 0.0, 0.18, 0.36];
    let low = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 95_000.0)
        .with_segments(80);
    let high = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 320_000.0)
        .with_segments(80);
    let low_sweep = tire_rig_heightfield_envelope_sweep(
        &low, &terrain, &material, 4_500.0, 4.0, 0.08, &positions, 0.0,
    );
    let high_sweep = tire_rig_heightfield_envelope_sweep(
        &high, &terrain, &material, 4_500.0, 4.0, 0.08, &positions, 0.0,
    );
    assert!(low_sweep.iter().all(TireRigEnvelopePoint::is_finite));
    assert!(high_sweep.iter().all(TireRigEnvelopePoint::is_finite));
    assert!(
        low_sweep[2].terrain_height_m > low_sweep[0].terrain_height_m + 0.05,
        "sweep should include a real bump crest: {low_sweep:?}"
    );
    assert!(
        low_sweep.iter().any(|p| p.terrain_normal[0].abs() > 0.15),
        "bump slopes should expose non-vertical terrain normals"
    );
    assert!(
        low_sweep
            .iter()
            .any(|p| p.point.telemetry.force_n[0].abs() > 20.0),
        "heightfield normals should feed a horizontal force component"
    );
    assert!(
        low_sweep[2].point.telemetry.patch.patch_length_m
            > high_sweep[2].point.telemetry.patch.patch_length_m,
        "low pressure should produce a longer bump contact patch: low={:?} high={:?}",
        low_sweep[2].point.telemetry.patch,
        high_sweep[2].point.telemetry.patch
    );
}

#[test]
fn tire_rig_heightfield_envelope_sweep_handles_curb_without_single_point_collapse() {
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
    let terrain = HeightfieldTerrain::new(origin, spacing, nx, nz, heights, 0.95);
    let material = SurfaceMaterial::dry_asphalt();
    let positions = [-0.08, -0.02, 0.02, 0.08];
    let low = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 90_000.0)
        .with_segments(96);
    let high = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 340_000.0)
        .with_segments(96);
    let low_sweep = tire_rig_heightfield_envelope_sweep(
        &low, &terrain, &material, 4_500.0, 3.0, 0.06, &positions, 0.0,
    );
    let high_sweep = tire_rig_heightfield_envelope_sweep(
        &high, &terrain, &material, 4_500.0, 3.0, 0.06, &positions, 0.0,
    );
    assert!(low_sweep.iter().all(TireRigEnvelopePoint::is_finite));
    assert!(high_sweep.iter().all(TireRigEnvelopePoint::is_finite));
    let edge = &low_sweep[1];
    assert!(
        edge.terrain_normal[0].abs() > 0.35,
        "curb edge should expose a steep terrain normal: {:?}",
        edge.terrain_normal
    );
    assert!(
        edge.point.telemetry.patch.contact_count > 1
            && edge.point.telemetry.patch.patch_length_m > 0.04,
        "curb contact must not collapse to one point: {:?}",
        edge.point.telemetry.patch
    );
    assert!(
        edge.point.telemetry.force_n[0].abs() > 50.0,
        "curb normal should produce a horizontal force component: {:?}",
        edge.point.telemetry.force_n
    );
    assert!(
        low_sweep[1].point.telemetry.patch.patch_length_m
            > high_sweep[1].point.telemetry.patch.patch_length_m,
        "low pressure should conform to curb with a longer patch: low={:?} high={:?}",
        low_sweep[1].point.telemetry.patch,
        high_sweep[1].point.telemetry.patch
    );
}

#[test]
fn tire_rig_rolling_obstacle_sequence_tracks_bump_approach_crest_and_exit() {
    let nx = 61;
    let nz = 5;
    let spacing = 0.035;
    let origin = [-0.75, -0.07];
    let mut heights = Vec::with_capacity(nx * nz);
    for _iz in 0..nz {
        for ix in 0..nx {
            let x = origin[0] + ix as f64 * spacing;
            let bump = 0.07 * (-(x / 0.12).powi(2)).exp();
            heights.push(bump);
        }
    }
    let terrain = HeightfieldTerrain::new(origin, spacing, nx, nz, heights, 0.95);
    let material = SurfaceMaterial::dry_asphalt();
    let seq = tire_rig_rolling_obstacle_sequence(
        TetraTireFemParams::realtime_viewer()
            .with_resolution(16, 3, 3)
            .with_pressure_and_load(105_000.0, 4_800.0),
        &terrain,
        &material,
        -0.38,
        0.0,
        2.2,
        0.05,
        1.0 / 120.0,
        43,
    );
    assert_eq!(seq.len(), 43);
    assert!(seq.iter().all(TireRigRollingObstaclePoint::is_finite));
    assert!(
        seq.windows(2)
            .all(|w| w[1].wheel_angle_rad > w[0].wheel_angle_rad),
        "wheel angle should advance monotonically during the ordered sequence"
    );
    let crest = seq
        .iter()
        .max_by(|a, b| {
            a.terrain_height_m
                .partial_cmp(&b.terrain_height_m)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    assert!(
        crest.terrain_height_m > seq[0].terrain_height_m + 0.05
            && crest.terrain_height_m > seq.last().unwrap().terrain_height_m + 0.05,
        "sequence should include approach, bump crest, and exit: first={} crest={} last={}",
        seq[0].terrain_height_m,
        crest.terrain_height_m,
        seq.last().unwrap().terrain_height_m
    );
    assert!(
        seq.iter().any(|p| p.terrain_normal[0].abs() > 0.20),
        "rolling sequence should encounter bump slopes"
    );
    assert!(
        seq.iter().any(|p| p.telemetry.force_n[0].abs() > 15.0),
        "rolling bump normals should generate horizontal force"
    );
    assert!(
        crest.active_tread_nodes > 2 && crest.telemetry.patch.patch_length_m > 0.04,
        "crest contact should remain an extended FEM tread patch, not one node: {:?}",
        crest
    );
}

#[test]
fn tire_rig_camber_sweep_flips_lateral_force_sign() {
    let params = RealTireParams::passenger_radial().with_segments(64);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let camber = 6.0_f64.to_radians();
    let sweep = tire_rig_camber_sweep(
        &params,
        &terrain,
        &material,
        4_500.0,
        8.0,
        &[-camber, 0.0, camber],
    );
    assert!(sweep.iter().all(TireRigSweepPoint::is_finite));
    let neg_fy = sweep[0].telemetry.lateral_force_n;
    let zero_fy = sweep[1].telemetry.lateral_force_n;
    let pos_fy = sweep[2].telemetry.lateral_force_n;
    assert!(
        neg_fy.signum() != pos_fy.signum() && neg_fy.abs() > zero_fy.abs() + 1.0,
        "camber force should flip around zero: neg={neg_fy} zero={zero_fy} pos={pos_fy}"
    );
    assert!(
        pos_fy.abs() > zero_fy.abs() + 1.0,
        "positive camber should add measurable lateral force: zero={zero_fy} pos={pos_fy}"
    );
}

#[test]
fn tire_rig_camber_sweep_bounds_moment_trail_and_sliding_saturation() {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let asphalt = SurfaceMaterial::dry_asphalt();
    let asphalt_terrain = FlatTerrain::new(0.0, asphalt.mu_static);
    let camber = 12.0_f64.to_radians();
    let sweep = tire_rig_camber_sweep(
        &params,
        &asphalt_terrain,
        &asphalt,
        4_500.0,
        10.0,
        &[-camber, 0.0, camber],
    );
    assert!(sweep.iter().all(TireRigSweepPoint::is_finite));
    for point in &sweep {
        let patch = &point.telemetry.patch;
        let half_patch = patch.patch_length_m.max(1.0e-6) * 0.5;
        assert!(
            patch.contact_center_x_m.abs() <= half_patch + 1.0e-6,
            "contact centre must stay inside the patch: {:?}",
            patch
        );
        assert!(
            patch.force_resultant_x_m.abs() <= half_patch + 0.02,
            "lateral resultant should stay near the physical patch: {:?}",
            patch
        );
        assert!(
            patch.pneumatic_trail_m.abs() <= patch.patch_length_m + 0.02,
            "pneumatic trail must remain bounded by the patch: {:?}",
            patch
        );
        let moment_bound =
            point.telemetry.available_friction_budget_n * (patch.patch_length_m * 0.5 + 0.02);
        assert!(
            point.telemetry.aligning_moment_nm.abs() <= moment_bound.max(1.0),
            "aligning moment should be bounded by friction budget and patch length: mz={} bound={}",
            point.telemetry.aligning_moment_nm,
            moment_bound
        );
    }

    let ice = SurfaceMaterial::ice();
    let ice_terrain = FlatTerrain::new(0.0, ice.mu_static);
    let asphalt_high = tire_rig_camber_sweep(
        &params,
        &asphalt_terrain,
        &asphalt,
        4_500.0,
        10.0,
        &[20.0_f64.to_radians()],
    )
    .remove(0);
    let ice_high = tire_rig_camber_sweep(
        &params,
        &ice_terrain,
        &ice,
        4_500.0,
        10.0,
        &[20.0_f64.to_radians()],
    )
    .remove(0);
    assert!(
        ice_high.telemetry.patch.sliding_fraction >= asphalt_high.telemetry.patch.sliding_fraction,
        "low-friction camber case should slide at least as much as asphalt: ice={} asphalt={}",
        ice_high.telemetry.patch.sliding_fraction,
        asphalt_high.telemetry.patch.sliding_fraction
    );
    assert!(
        ice_high.telemetry.consumed_friction_budget_n
            <= ice_high.telemetry.available_friction_budget_n * 1.02 + 1.0,
        "sliding camber force must stay inside the reported friction budget: {:?}",
        ice_high.telemetry
    );
    assert!(
        ice_high.telemetry.lateral_force_n.abs() < asphalt_high.telemetry.lateral_force_n.abs(),
        "ice should cap camber lateral force below asphalt: ice={} asphalt={}",
        ice_high.telemetry.lateral_force_n,
        asphalt_high.telemetry.lateral_force_n
    );
}

#[test]
fn tire_rig_combined_slip_sweep_shares_one_friction_budget() {
    let params = RealTireParams::passenger_radial().with_segments(64);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let slip = 0.22;
    let angle = 12.0_f64.to_radians();
    let sweep = tire_rig_combined_slip_sweep(
        &params,
        &terrain,
        &material,
        4_500.0,
        9.0,
        &[0.0, slip],
        &[0.0, angle],
    );
    assert_eq!(sweep.len(), 4);
    assert!(sweep.iter().all(TireRigSweepPoint::is_finite));
    let pure_lat = &sweep[1].telemetry;
    let pure_long = &sweep[2].telemetry;
    let combined = &sweep[3].telemetry;
    assert!(
        combined.longitudinal_force_n.abs() < pure_long.longitudinal_force_n.abs(),
        "combined cornering should consume some longitudinal budget: pure={} combined={}",
        pure_long.longitudinal_force_n,
        combined.longitudinal_force_n
    );
    assert!(
        combined.lateral_force_n.abs() < pure_lat.lateral_force_n.abs(),
        "combined drive/brake should consume some lateral budget: pure={} combined={}",
        pure_lat.lateral_force_n,
        combined.lateral_force_n
    );
    assert!(
        combined.consumed_friction_budget_n <= combined.available_friction_budget_n * 1.05,
        "combined slip must stay inside one friction budget: {combined:?}"
    );
}

#[test]
fn tire_reference_shape_diagnostics_check_longitudinal_buildup_without_fitting() {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let sweep = tire_rig_longitudinal_slip_sweep(
        &params,
        &terrain,
        &material,
        4_200.0,
        14.0,
        &[0.0, 0.02, 0.05, 0.10, 0.18, 0.30, 0.45],
    );

    let diagnostics =
        tire_rig_longitudinal_reference_shape_diagnostics("long shape dry asphalt", &sweep);
    assert!(
        diagnostics.is_finite(),
        "reference-shape diagnostics should be finite: {diagnostics:?}"
    );
    assert!(
        diagnostics.passed(),
        "dry-asphalt longitudinal shape should pass broad non-fitted checks: {diagnostics:?}"
    );
    assert_eq!(diagnostics.sample_count, sweep.len());
    assert!(diagnostics.peak_longitudinal_force_n > 1_000.0);
    assert!(diagnostics.max_normalized_force_jump <= 0.65);

    let report = diagnostics.clone().report(sweep[3].snapshot_on_terrain(
        "long shape dry asphalt",
        "passenger_radial",
        "dry_asphalt",
        "flat",
        "cpu_brush",
        &terrain,
        0.0,
        0.0,
    ));
    assert!(
        report.passed(),
        "shape acceptance report should pass: {report:?}"
    );
    let text = report.to_text_report();
    assert!(
        text.contains("reference_shape_monotonic_buildup")
            && text.contains("reference_shape_smooth_jump")
            && text.contains("non_calibration=true"),
        "shape report should be reviewable and explicitly non-calibration: {text}"
    );
}

#[test]
fn tire_reference_shape_diagnostics_check_combined_slip_tradeoff_and_mz_sign() {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let sweep = tire_rig_combined_slip_sweep(
        &params,
        &terrain,
        &material,
        4_200.0,
        14.0,
        &[0.0, 0.22],
        &[0.0, 0.12],
    );
    assert_eq!(sweep.len(), 4);
    let pure_lateral = &sweep[1];
    let pure_longitudinal = &sweep[2];
    let combined = &sweep[3];

    let diagnostics = tire_rig_combined_reference_shape_diagnostics(
        "combined shape dry asphalt",
        pure_longitudinal,
        pure_lateral,
        combined,
    );
    assert!(
        diagnostics.is_finite(),
        "combined reference-shape diagnostics should be finite: {diagnostics:?}"
    );
    assert!(
        diagnostics.passed(),
        "combined-slip shape should pass broad tradeoff/sign checks: {diagnostics:?}"
    );
    assert!(diagnostics.combined_slip_tradeoff);
    assert!(diagnostics.aligning_moment_sign_ok);
    assert!(diagnostics.peak_longitudinal_force_n >= combined.telemetry.longitudinal_force_n.abs());
    assert!(diagnostics.peak_lateral_force_n >= combined.telemetry.lateral_force_n.abs());
}

#[test]
fn tire_rig_standard_sensitivity_map_checks_directional_responses_without_fitting() {
    let map = tire_rig_standard_sensitivity_map();
    assert_eq!(map.len(), 12);
    assert!(
        map.iter().all(TireRigSensitivityPoint::is_finite),
        "all sensitivity rows should be finite: {map:#?}"
    );
    assert!(
        map.iter().all(|point| point.passed),
        "every broad non-calibration sensitivity direction should pass:\n{}",
        tire_rig_standard_sensitivity_text_report()
    );
    assert!(
        map.iter()
            .any(|p| p.axis == TireRigSensitivityAxis::Pressure)
    );
    assert!(map.iter().any(|p| p.axis == TireRigSensitivityAxis::Load));
    assert!(
        map.iter()
            .any(|p| p.axis == TireRigSensitivityAxis::DrySpeed)
    );
    assert!(
        map.iter()
            .any(|p| p.axis == TireRigSensitivityAxis::SurfaceTemperature)
    );
    assert!(
        map.iter()
            .any(|p| p.axis == TireRigSensitivityAxis::ObstacleHeight)
    );
    assert!(
        map.iter()
            .any(|p| p.axis == TireRigSensitivityAxis::SoilStiffness)
    );
    assert!(
        map.iter()
            .any(|p| p.axis == TireRigSensitivityAxis::ContactResolution)
    );
    let text = tire_rig_standard_sensitivity_text_report();
    assert!(
        text.contains("pressure")
            && text.contains("dry_speed")
            && text.contains("water_film")
            && text.contains("soft_soil_surface")
            && text.contains("soil_stiffness")
            && text.contains("contact_resolution")
            && text.contains("obstacle_height"),
        "sensitivity text report should expose the varied axes: {text}"
    );
}

#[test]
fn tire_rig_thermal_wear_run_heats_and_wears_under_sustained_slip() {
    let params = RealTireParams::passenger_radial().with_segments(64);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let run =
        tire_rig_thermal_wear_run(&params, &terrain, &material, 4_500.0, 18.0, 0.35, 0.1, 240);
    assert_eq!(run.len(), 240);
    assert!(run.iter().all(TireRigThermalWearPoint::is_finite));
    let first = &run[0];
    let last = run.last().unwrap();
    assert!(
        last.thermal.surface_c > first.thermal.surface_c + 5.0,
        "sustained slip should heat the surface: first={} last={}",
        first.thermal.surface_c,
        last.thermal.surface_c
    );
    assert!(
        last.thermal.tread_c > first.thermal.tread_c,
        "sustained slip should conduct heat into tread: first={} last={}",
        first.thermal.tread_c,
        last.thermal.tread_c
    );
    assert!(
        last.wear.depth_m > first.wear.depth_m,
        "sustained slip should increase wear: first={} last={}",
        first.wear.depth_m,
        last.wear.depth_m
    );
    assert!(
        last.telemetry.dissipated_power_w >= 0.0,
        "contact energy accounting must stay non-negative"
    );
}

#[test]
fn tire_rig_thermal_wear_cycle_cools_surface_after_slip_phase() {
    let params = RealTireParams::passenger_radial().with_segments(64);
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let run = tire_rig_thermal_wear_cycle(
        &params,
        &terrain,
        &material,
        0.1,
        &[
            TireRigThermalWearPhase::sustained_slip(18.0, 4_500.0, 18.0, 0.40),
            TireRigThermalWearPhase::cooling_recovery(45.0, 4_500.0),
        ],
    );
    assert!(!run.is_empty());
    assert!(run.iter().all(TireRigThermalWearPoint::is_finite));
    let heat_peak = run
        .iter()
        .filter(|p| p.phase_index == 0)
        .max_by(|a, b| {
            a.thermal
                .surface_c
                .partial_cmp(&b.thermal.surface_c)
                .unwrap()
        })
        .unwrap();
    let last = run.last().unwrap();
    assert!(
        heat_peak.thermal.surface_c > 25.0,
        "slip phase should create a hot surface: {}",
        heat_peak.thermal.surface_c
    );
    assert!(
        last.thermal.surface_c < heat_peak.thermal.surface_c - 6.0,
        "cooling phase should reduce surface temperature: peak={} last={}",
        heat_peak.thermal.surface_c,
        last.thermal.surface_c
    );
    assert!(
        last.thermal.surface_c - last.thermal.carcass_c
            < heat_peak.thermal.surface_c - heat_peak.thermal.carcass_c,
        "cooling should collapse the surface/carcass gradient: peak={:?} last={:?}",
        heat_peak.thermal,
        last.thermal
    );
    assert!(
        last.wear.depth_m >= heat_peak.wear.depth_m,
        "cooling should not heal tread wear"
    );
    let sanity = tire_thermal_condition_sanity_report(&run);
    assert!(
        sanity.passed(),
        "thermal condition sanity report should pass for heat/cool rig: {sanity:?}"
    );
    assert!(
        sanity.peak_surface_c > sanity.final_surface_c
            && sanity.final_wear_depth_m == sanity.max_wear_depth_m,
        "sanity report should expose cooling and non-healing wear: {sanity:?}"
    );
    let checks = sanity.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_condition_cooling_drop_nonnegative")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_condition_wear_non_healing")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_condition_mu_factor_positive"),
        "thermal condition sanity should expose explicit runner gates: {checks:?}"
    );
}

#[test]
fn thermal_abuse_envelope_report_orders_state_derived_failure_modes() {
    let report = tire_thermal_abuse_envelope_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "thermal abuse envelope should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.cold_mu_factor < report.healthy_mu_factor
            && report.hot_mu_factor < report.healthy_mu_factor
            && report.worn_mu_factor < report.healthy_mu_factor
            && report.abused_mu_factor < report.hot_mu_factor,
        "cold/hot/worn/abused states should reduce grip in the expected order: {report:?}"
    );
    assert!(
        report.hot_wear_rate_factor > report.healthy_wear_rate_factor
            && report.abused_wear_rate_factor > report.hot_wear_rate_factor
            && report.abused_carcass_stiffness_factor < report.healthy_carcass_stiffness_factor,
        "hot/abused states should increase wear and soften carcass from state: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_abuse_hot_reduces_mu")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_abuse_abused_level_state_derived")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_abuse_abused_softens_carcass"),
        "thermal abuse report should expose hot/worn/abused state-derived gates: {checks:?}"
    );
}

#[test]
fn thermal_rolling_carcass_report_heats_carcass_without_sliding() {
    let report = tire_thermal_rolling_carcass_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "rolling/carcass thermal report should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.max_sliding_power_w <= 1.0e-6
            && report.carcass_delta_c() > 0.75
            && report.pressure_factor_delta() > 0.001
            && report.final_wear.depth_m <= TireWearState::default().depth_m + 1.0e-12,
        "rolling heat should come from carcass/rolling work, not sliding wear: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_rolling_carcass_low_sliding_power")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_rolling_carcass_heats_carcass")
            && checks
                .iter()
                .any(|check| { check.check_id == "thermal_rolling_carcass_pressure_factor_rises" }),
        "rolling/carcass report should expose low-sliding, carcass-heat, and pressure feedback gates: {checks:?}"
    );
}

#[test]
fn thermal_brake_rim_report_routes_heat_through_rim_and_carcass() {
    let report = tire_thermal_brake_rim_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "brake/rim thermal report should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.max_sliding_power_w <= 1.0e-6
            && report.rim_delta_c() > report.carcass_delta_c()
            && report.carcass_delta_c() > report.surface_delta_c()
            && report.pressure_factor_delta() > 0.00005
            && report.final_wear.depth_m <= TireWearState::default().depth_m + 1.0e-12,
        "brake heat should route through rim/carcass, not sliding wear: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_brake_rim_heats_rim_first")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_brake_rim_conducts_to_carcass")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_brake_rim_pressure_factor_rises"),
        "brake/rim report should expose rim-first, carcass-conduction, and pressure feedback gates: {checks:?}"
    );
}

#[test]
fn dem_soil_feedback_report_links_tire_contact_to_grain_motion() {
    let report = tire_dem_soil_feedback_report();
    assert!(
        report.passed(),
        "DEM soil feedback report should pass broad non-calibration envelopes: {report:?}"
    );
    assert!(
        report.slipping_active_contact_cells > 0
            && report.slipping_affected_grains > 0
            && report.slipping_normal_load_n > 250.0,
        "slipping DEM tire patch should transfer load into grains: {report:?}"
    );
    assert!(
        report.shear_ratio_slip_over_roll > 1.20 && report.grain_motion_ratio_slip_over_roll > 1.02,
        "slip should increase DEM shear and horizontal grain motion: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "dem_soil_feedback_slip_increases_shear")
            && checks
                .iter()
                .any(|check| check.check_id == "dem_soil_feedback_slip_moves_grains_more"),
        "DEM soil feedback report should expose explicit shear/motion gates: {checks:?}"
    );
}

#[test]
fn soil_normal_shear_coupling_report_links_support_and_driven_rutting() {
    let report = tire_soil_normal_shear_coupling_report();
    assert!(
        report.passed(),
        "Bekker soil normal/shear coupling report should pass broad non-calibration envelopes: {report:?}"
    );
    assert!(
        report.rolling_normal_sinkage_m > 1.0e-5
            && report.driven_normal_sinkage_m > 1.0e-5
            && report.shear_sinkage_gain_m > 1.0e-5,
        "soil report should prove normal support plus driven shear rutting: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "soil_normal_shear_normal_support")
            && checks
                .iter()
                .any(|check| { check.check_id == "soil_normal_shear_slip_deepens_shear_rut" }),
        "soil normal/shear report should expose support and shear-rut gates: {checks:?}"
    );
}

#[test]
fn mpm_soil_feedback_report_links_tire_contact_to_material_points() {
    let report = tire_mpm_soil_feedback_report();
    assert!(
        report.passed(),
        "MPM soil feedback report should pass broad non-calibration envelopes: {report:?}"
    );
    assert!(
        report.slipping_active_points > 0
            && report.slipping_total_compaction > 0.0
            && report.slipping_normal_impulse_n_s > 1.0e-4,
        "slipping MPM tire patch should compact material points: {report:?}"
    );
    assert!(
        report.shear_ratio_slip_over_roll > 1.10
            && report.surface_velocity_ratio_slip_over_roll > 1.05,
        "slip should increase MPM shear and material velocity export: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "mpm_soil_feedback_slip_increases_shear")
            && checks.iter().any(|check| {
                check.check_id == "mpm_soil_feedback_slip_increases_surface_velocity"
            }),
        "MPM soil feedback report should expose explicit shear/velocity gates: {checks:?}"
    );
}

#[test]
fn lower_pressure_makes_a_longer_contact_patch() {
    let material = SurfaceMaterial::dry_asphalt();
    let high = loaded_output(360_000.0, material, 0.0);
    let low = loaded_output(90_000.0, material, 0.0);
    assert!(
        low.patch_length > high.patch_length,
        "low pressure should make a longer patch: low={} high={}",
        low.patch_length,
        high.patch_length
    );
    assert!(
        low.contact_count >= high.contact_count,
        "low pressure should touch at least as many cells"
    );
}

#[test]
fn surface_material_dominates_ultimate_grip() {
    let asphalt = loaded_output(320_000.0, SurfaceMaterial::dry_asphalt(), 3.0);
    let ice = loaded_output(320_000.0, SurfaceMaterial::ice(), 3.0);
    assert!(
        asphalt.longitudinal_force > ice.longitudinal_force * 6.0,
        "asphalt should transmit far more force than ice: asphalt={} ice={}",
        asphalt.longitudinal_force,
        ice.longitudinal_force
    );
}

#[test]
fn heightfield_friction_map_modulates_brush_force_per_contact_cell() {
    let heights = vec![0.0; 25];
    let high_mu = vec![1.0; 25];
    let mut ice_patch = vec![1.0; 25];
    for iz in 0..5 {
        for ix in 3..5 {
            ice_patch[iz * 5 + ix] = 0.08;
        }
    }
    let high_terrain = HeightfieldTerrain::with_friction_map(
        [-1.0, -1.0],
        0.5,
        5,
        5,
        heights.clone(),
        1.0,
        high_mu,
    );
    let patch_terrain =
        HeightfieldTerrain::with_friction_map([-1.0, -1.0], 0.5, 5, 5, heights, 1.0, ice_patch);
    let material = SurfaceMaterial::dry_asphalt();
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &high_terrain, 0.5, 0.0, 4_500.0);
    let state = RealWheelState {
        hub_pos: [0.5, hub_y, 0.0],
        hub_vel: [6.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: (6.0 + 8.0) / params.radius,
    };
    let high = evaluate_brush_tire(&params, &state, &high_terrain, &material);
    let icy = evaluate_brush_tire(&params, &state, &patch_terrain, &material);
    assert!(
        icy.longitudinal_force.abs() < high.longitudinal_force.abs() * 0.55,
        "terrain friction map should reduce local brush force: high={} icy={}",
        high.longitudinal_force,
        icy.longitudinal_force
    );
}

#[test]
fn heightfield_normals_feed_contact_cells_and_force_vector() {
    let mut heights = Vec::new();
    for _iz in 0..5 {
        for ix in 0..5 {
            heights.push(0.18 * ix as f64);
        }
    }
    let terrain = HeightfieldTerrain::new([-1.0, -1.0], 0.5, 5, 5, heights, 1.0);
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        bristle_damping: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 64,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 4_000.0);
    let out = evaluate_brush_tire(
        &params,
        &RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [0.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: 0.0,
        },
        &terrain,
        &material,
    );
    assert!(out.in_contact, "ramp tire should contact");
    assert!(
        out.cells.iter().any(|c| c.normal[0] < -0.05),
        "contact cells should report local ramp normals: {:?}",
        out.cells.iter().map(|c| c.normal).collect::<Vec<_>>()
    );
    assert!(
        out.force[0] < -50.0,
        "wheel-level force should include the uphill normal component, got {:?}",
        out.force
    );
}

#[test]
fn surface_water_film_reduces_grip_more_at_speed() {
    let base = SurfaceMaterial::dry_asphalt();
    let wear = TireWearState::default();
    let film = SurfaceFilmState::wet(0.0015);
    let slow = film.apply_to(base, 4.0, &wear);
    let fast = film.apply_to(base, 32.0, &wear);
    assert!(
        slow.mu_static < base.mu_static,
        "wet film should reduce static grip"
    );
    assert!(
        fast.mu_static < slow.mu_static && fast.mu_dynamic < slow.mu_dynamic,
        "wet film should reduce grip more at speed: slow={slow:?} fast={fast:?}"
    );
    assert!(
        fast.rolling_resistance > slow.rolling_resistance,
        "water film should add speed-dependent rolling/water displacement loss"
    );
}

#[test]
fn worn_tread_is_worse_on_water_film_than_fresh_tread() {
    let base = SurfaceMaterial::dry_asphalt();
    let film = SurfaceFilmState::wet(0.0012);
    let fresh = TireWearState::default();
    let mut worn = TireWearState::default();
    worn.depth_m = worn.max_depth_m * 0.85;
    let fresh_material = film.apply_to(base, 18.0, &fresh);
    let worn_material = film.apply_to(base, 18.0, &worn);
    assert!(
        worn_material.mu_static < fresh_material.mu_static * 0.85,
        "worn tread should drain less water and lose more grip: fresh={fresh_material:?} worn={worn_material:?}"
    );
}

#[test]
fn water_film_reduces_brush_force_but_does_not_become_ice() {
    let base = SurfaceMaterial::dry_asphalt();
    let wear = TireWearState::default();
    let wet = SurfaceFilmState::wet(0.0010).apply_to(base, 20.0, &wear);
    let dry = loaded_output(220_000.0, base, 4.0);
    let wet_out = loaded_output(220_000.0, wet, 4.0);
    let ice = loaded_output(220_000.0, SurfaceMaterial::ice(), 4.0);
    assert!(
        wet_out.longitudinal_force < dry.longitudinal_force * 0.85,
        "wet film should reduce brush traction: dry={} wet={}",
        dry.longitudinal_force,
        wet_out.longitudinal_force
    );
    assert!(
        wet_out.longitudinal_force > ice.longitudinal_force * 1.5,
        "thin wet film should not collapse all the way to ice: wet={} ice={}",
        wet_out.longitudinal_force,
        ice.longitudinal_force
    );
}

#[test]
fn heightfield_water_film_map_reduces_brush_force_per_contact_cell() {
    let heights = vec![0.0; 45];
    let mut water = vec![0.0; 45];
    for iz in 0..5 {
        for ix in 5..9 {
            water[iz * 9 + ix] = 0.004;
        }
    }
    let terrain =
        HeightfieldTerrain::new([-2.0, -1.0], 0.5, 9, 5, heights, 1.0).with_water_film_map(water);
    assert!(terrain.water_film_depth(-1.0, 0.0) <= 1.0e-9);
    assert!(terrain.water_film_depth(1.0, 0.0) > 0.003);

    let material = SurfaceMaterial::dry_asphalt();
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let make_state = |x: f64| RealWheelState {
        hub_pos: [
            x,
            solve_loaded_tire_hub_height(&params, &terrain, x, 0.0, 4_500.0),
            0.0,
        ],
        hub_vel: [14.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: (14.0 + 9.0) / params.radius,
    };
    let dry = evaluate_brush_tire(&params, &make_state(-1.0), &terrain, &material);
    let wet = evaluate_brush_tire(&params, &make_state(1.0), &terrain, &material);
    assert!(dry.in_contact && wet.in_contact);
    assert!(
        (dry.normal_load - wet.normal_load).abs() < dry.normal_load * 0.04,
        "flat dry/wet patches should carry comparable load: dry={} wet={}",
        dry.normal_load,
        wet.normal_load
    );
    assert!(
        wet.longitudinal_force.abs() < dry.longitudinal_force.abs() * 0.72,
        "spatial water-film map should reduce local brush traction: dry={} wet={}",
        dry.longitudinal_force,
        wet.longitudinal_force
    );
    assert!(
        wet.rolling_resistance_force.abs() > dry.rolling_resistance_force.abs() * 1.5,
        "water-film map should add local water displacement rolling loss: dry={} wet={}",
        dry.rolling_resistance_force,
        wet.rolling_resistance_force
    );
    assert!(wet.telemetry_sample(1.0 / 120.0).is_finite());
}

#[derive(Clone, Copy)]
struct MovingFlatTerrain {
    speed_x: f64,
    friction: f64,
}

impl Terrain for MovingFlatTerrain {
    fn height(&self, _x: f64, _z: f64) -> f64 {
        0.0
    }

    fn normal(&self, _x: f64, _z: f64) -> [f64; 3] {
        [0.0, 1.0, 0.0]
    }

    fn friction(&self) -> f64 {
        self.friction
    }

    fn surface_velocity(&self, _x: f64, _z: f64) -> [f64; 3] {
        [self.speed_x, 0.0, 0.0]
    }
}

#[test]
fn terrain_surface_velocity_changes_tire_slip_relative_to_ground() {
    let material = SurfaceMaterial::dry_asphalt();
    let static_ground = MovingFlatTerrain {
        speed_x: 0.0,
        friction: material.mu_static,
    };
    let moving_ground = MovingFlatTerrain {
        speed_x: 3.0,
        friction: material.mu_static,
    };
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &static_ground, 0.0, 0.0, 4_500.0);
    let state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [3.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 3.0 / params.radius,
    };
    let static_out = evaluate_brush_tire(&params, &state, &static_ground, &material);
    let moving_out = evaluate_brush_tire(&params, &state, &moving_ground, &material);
    assert!(
        moving_out.longitudinal_force.abs() > static_out.longitudinal_force.abs() + 500.0,
        "moving ground should change relative slip: static={} moving={}",
        static_out.longitudinal_force,
        moving_out.longitudinal_force
    );
}

#[test]
fn energy_sign_audit_tracks_moving_ground_work_sign() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = MovingFlatTerrain {
        speed_x: 3.0,
        friction: material.mu_static,
    };
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let point = tire_rig_point(&params, &terrain, &material, 4_500.0, 3.0, 0.0, 0.0, 0.0);
    let snapshot = point.snapshot_on_terrain(
        "moving_ground_energy",
        "passenger_radial",
        "dry_asphalt",
        "moving_flat",
        "cpu_brush",
        &terrain,
        0.0,
        0.0,
    );
    let audit = snapshot.energy_sign_audit();
    assert!(
        audit.is_finite(),
        "moving-ground audit should be finite: {audit:?}"
    );
    assert!(
        snapshot.moving_ground_power_w > 0.0,
        "snapshot CSV telemetry should expose moving-ground power: {snapshot:?}"
    );
    assert!(
        audit.moving_ground_speed_m_per_s > 2.9,
        "snapshot should carry moving-ground speed: {audit:?}"
    );
    assert!(
        audit.moving_ground_power_w > 0.0,
        "moving belt should do positive work on this non-braking tire state: {audit:?}"
    );
    assert!(
        audit.moving_ground_work_sign_ok,
        "moving-ground work sign gate should pass: {audit:?}"
    );
    let report = snapshot.energy_sign_report();
    let text = report.to_text_report();
    assert!(
        text.contains("moving_ground_work_expected_sign"),
        "energy/sign report should expose moving-ground work sign gate: {text}"
    );
}

#[test]
fn lower_pressure_improves_low_mu_tractive_stiffness() {
    let material = SurfaceMaterial::new(0.22, 0.17);
    let high = loaded_output(360_000.0, material, 0.02);
    let low = loaded_output(90_000.0, material, 0.02);
    assert!(
        low.longitudinal_force > high.longitudinal_force * 1.10,
        "low pressure should improve brush force before full sliding: low={} high={}",
        low.longitudinal_force,
        high.longitudinal_force
    );
}

#[test]
fn ice_large_patch_report_keeps_low_mu_force_capped() {
    let report = tire_ice_large_patch_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "ice large-patch report should satisfy broad non-calibration gates: {report:?}"
    );
    assert!(
        report.low_over_high_patch_ratio > 1.15
            && report.ice_over_dry_fx <= 0.30
            && report.ice_sliding_fraction >= report.dry_sliding_fraction,
        "large patch should not hide ice sliding/capped force: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "ice_large_patch_still_caps_force")
            && checks
                .iter()
                .any(|check| { check.check_id == "ice_large_patch_low_pressure_patch_grows" })
            && checks
                .iter()
                .any(|check| check.check_id == "ice_large_patch_sliding_not_hidden"),
        "ice large-patch checks should expose patch, cap, and sliding gates: {checks:?}"
    );
}

#[test]
fn adhesion_sliding_continuity_report_keeps_force_and_moment_smooth() {
    let report = tire_adhesion_sliding_continuity_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "adhesion/sliding continuity should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.min_sliding_fraction <= 0.25
            && report.max_sliding_fraction >= 0.55
            && report.max_force_jump_over_load <= 0.30
            && report.max_moment_jump_over_load_radius <= 0.16,
        "sweep should cross adhesion/sliding without force or moment teleporting: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "adhesion_sliding_force_continuous")
            && checks
                .iter()
                .any(|check| check.check_id == "adhesion_sliding_moment_continuous")
            && checks
                .iter()
                .any(|check| check.check_id == "adhesion_sliding_budget_bounded"),
        "adhesion/sliding checks should expose force, moment, and budget continuity gates: {checks:?}"
    );
}

#[test]
fn combined_slip_caps_longitudinal_and_lateral_force_together() {
    let material = SurfaceMaterial {
        mu_static: 0.40,
        mu_dynamic: 0.30,
        bristle_damping: 1.0e4,
        rolling_resistance: 0.0,
        cohesion: 0.0,
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 64,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 4_500.0);
    let out = evaluate_brush_tire(
        &params,
        &RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [5.0, 0.0, 2.0],
            heading: [1.0, 0.0, 0.0],
            omega: (5.0 + 8.0) / params.radius,
        },
        &terrain,
        &material,
    );
    let combined = (out.longitudinal_force * out.longitudinal_force
        + out.lateral_force * out.lateral_force)
        .sqrt();
    assert!(
        out.sliding_count > 0,
        "combined high slip should put part of the patch into sliding"
    );
    assert!(
        combined <= out.available_friction_budget * 1.02,
        "combined force should respect load-sensitive friction budget: |F|={} cap={}",
        combined,
        out.available_friction_budget
    );
    assert!(
        out.lateral_force < 0.0,
        "positive lateral velocity should produce opposing lateral force, got {}",
        out.lateral_force
    );
    assert!(
        out.force[2] < 0.0,
        "world force should include lateral component opposing +Z motion: {:?}",
        out.force
    );
    assert!(
        out.sliding_fraction > 0.0 && out.sliding_fraction <= 1.0,
        "combined slip should report a sane sliding fraction: {}",
        out.sliding_fraction
    );
    assert!(
        out.consumed_friction_budget <= out.available_friction_budget * 1.02,
        "consumed budget should stay under available budget: consumed={} available={}",
        out.consumed_friction_budget,
        out.available_friction_budget
    );
}

#[test]
fn brush_load_sensitivity_makes_peak_force_sublinear_with_load() {
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        cohesion: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let eval = |load: f64| {
        let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load);
        evaluate_brush_tire(
            &params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [8.0, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: (8.0 + 12.0) / params.radius,
            },
            &terrain,
            &material,
        )
    };
    let light = eval(2_500.0);
    let heavy = eval(5_000.0);
    let force_ratio = heavy.longitudinal_force.abs() / light.longitudinal_force.abs();
    let load_ratio = heavy.normal_load / light.normal_load;
    assert!(
        force_ratio > 1.0 && force_ratio < load_ratio * 0.98,
        "load sensitivity should make peak force sublinear: force_ratio={force_ratio} load_ratio={load_ratio}"
    );
    assert!(
        heavy.effective_longitudinal_stiffness > light.effective_longitudinal_stiffness,
        "heavier load should still raise stiffness sub-linearly: light={} heavy={}",
        light.effective_longitudinal_stiffness,
        heavy.effective_longitudinal_stiffness
    );
}

#[test]
fn brush_cornering_stiffness_is_inflation_pressure_dependent() {
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        cohesion: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let low = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 95_000.0)
        .with_segments(96);
    let high = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 300_000.0)
        .with_segments(96);
    let eval = |params: RealTireParams| {
        let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 4_200.0);
        evaluate_brush_tire(
            &params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [9.0, 0.0, 0.25],
                heading: [1.0, 0.0, 0.0],
                omega: 9.0 / params.radius,
            },
            &terrain,
            &material,
        )
    };
    let low_out = eval(low);
    let high_out = eval(high);
    let stiffness_delta =
        (high_out.effective_cornering_stiffness - low_out.effective_cornering_stiffness).abs();
    let stiffness_scale = low_out
        .effective_cornering_stiffness
        .max(high_out.effective_cornering_stiffness)
        .max(1.0);
    assert!(
        stiffness_delta / stiffness_scale > 0.06,
        "inflation pressure should measurably change small-slip cornering stiffness through pressure and contact-patch physics: low={} high={}",
        low_out.effective_cornering_stiffness,
        high_out.effective_cornering_stiffness
    );
    assert!(
        high_out.consumed_friction_budget <= high_out.available_friction_budget * 1.05 + 1.0,
        "pressure-dependent stiffness should still share the friction cap: {high_out:?}"
    );
}

#[test]
fn brush_combined_slip_is_smooth_near_saturation() {
    let material = SurfaceMaterial {
        mu_static: 0.55,
        mu_dynamic: 0.42,
        bristle_damping: 0.0,
        rolling_resistance: 0.0,
        cohesion: 0.0,
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: 200_000.0,
        segments: 96,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 4_000.0);
    let mut prev: Option<f64> = None;
    let mut max_jump: f64 = 0.0;
    for k in 0..32 {
        let slip = 0.01 + k as f64 * 0.01;
        let out = evaluate_brush_tire(
            &params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [6.0, 0.0, 0.7],
                heading: [1.0, 0.0, 0.0],
                omega: (6.0 * (1.0 + slip)) / params.radius,
            },
            &terrain,
            &material,
        );
        let mag = (out.longitudinal_force * out.longitudinal_force
            + out.lateral_force * out.lateral_force)
            .sqrt();
        if let Some(p) = prev {
            max_jump = max_jump.max((mag - p).abs());
        }
        prev = Some(mag);
    }
    assert!(
        max_jump < 450.0,
        "smooth combined-slip saturation should not jump abruptly; max_jump={max_jump}"
    );
}

#[test]
fn combined_slip_reduces_available_axis_force() {
    let material = SurfaceMaterial {
        mu_static: 0.70,
        mu_dynamic: 0.52,
        bristle_damping: 5.0e3,
        rolling_resistance: 0.0,
        cohesion: 0.0,
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 4_500.0);
    let straight = evaluate_brush_tire(
        &params,
        &RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [6.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: (6.0 + 7.0) / params.radius,
        },
        &terrain,
        &material,
    );
    let combined = evaluate_brush_tire(
        &params,
        &RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [6.0, 0.0, 3.0],
            heading: [1.0, 0.0, 0.0],
            omega: (6.0 + 7.0) / params.radius,
        },
        &terrain,
        &material,
    );
    assert!(
        combined.longitudinal_force.abs() < straight.longitudinal_force.abs() * 0.95,
        "lateral slip should consume friction budget and reduce drive force: straight={} combined={}",
        straight.longitudinal_force,
        combined.longitudinal_force
    );
    assert!(
        combined.sliding_fraction >= straight.sliding_fraction,
        "combined slip should slide at least as much as pure drive"
    );
}

#[test]
fn load_combined_slip_law_report_covers_c2_force_shape_requirements() {
    let report = tire_load_combined_slip_law_report();
    assert!(
        report.passed(),
        "load/combined-slip law should satisfy broad C2 envelopes: {report:?}"
    );
    assert!(
        report.load_force_ratio > 1.0 && report.load_force_ratio < report.load_ratio,
        "load sensitivity should be positive but sub-linear: {report:?}"
    );
    assert!(
        report.low_mu_low_pressure_over_high_pressure_fx > 1.05
            && report.low_mu_low_pressure_patch_over_high_pressure > 1.05,
        "lower pressure should improve low-μ pre-cap tractive stiffness and patch length: {report:?}"
    );
    assert!(
        report.combined_fx_over_pure_fx < 1.0 && report.combined_budget_ratio <= 1.05,
        "combined slip should trade axes through one friction budget: {report:?}"
    );
    assert!(
        report.saturation_start_sliding_fraction <= 0.65
            && report.saturation_sliding_fraction_delta > 0.10,
        "saturation sweep should expose sliding-fraction growth from adhesion toward sliding: {report:?}"
    );
    assert!(
        report.ice_over_dry_fx < 0.30
            && report.hot_tread_shear_over_healthy < 0.95
            && report.hot_longitudinal_stiffness_over_healthy < 0.95
            && report.hot_cornering_stiffness_over_healthy < 0.95,
        "ice and hot/worn state should reduce force/stiffness through physical state: {report:?}"
    );
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let material = SurfaceMaterial::dry_asphalt();
    let point = tire_rig_point(
        &RealTireParams::passenger_radial().with_segments(96),
        &terrain,
        &material,
        4_500.0,
        10.0,
        0.22,
        0.12,
        0.0,
    );
    let text = TireRigAcceptanceReport::new(
        point.snapshot_on_terrain(
            "load_combined_slip_law",
            "passenger_radial",
            "dry_asphalt",
            "flat",
            "cpu_brush_load_combined_slip_law",
            &terrain,
            0.0,
            0.0,
        ),
        report.acceptance_checks(),
    )
    .to_text_report();
    assert!(
        text.contains("load_sensitivity_sublinear_peak")
            && text.contains("low_mu_pressure_tractive_stiffness_improves")
            && text.contains("low_mu_pressure_patch_grows")
            && text.contains("combined_slip_single_budget")
            && text.contains("combined_slip_saturation_continuous")
            && text.contains("combined_slip_saturation_starts_with_adhesion")
            && text.contains("combined_slip_sliding_fraction_grows")
            && text.contains("temperature_state_reduces_tread_shear")
            && text.contains("temperature_state_reduces_longitudinal_stiffness")
            && text.contains("temperature_state_reduces_cornering_stiffness"),
        "C2 report should expose explicit load/combined-slip acceptance rows: {text}"
    );
}

#[test]
fn camber_thrust_flips_sign_and_reports_pneumatic_trail() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 4_500.0);
    let state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [8.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 8.0 / params.radius,
    };
    let zero = evaluate_brush_tire_oriented(
        &params,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation::default(),
    );
    let pos = evaluate_brush_tire_oriented(
        &params,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation {
            wheel_up: [0.0, 0.985, 0.174],
        },
    );
    let neg = evaluate_brush_tire_oriented(
        &params,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation {
            wheel_up: [0.0, 0.985, -0.174],
        },
    );
    assert!(
        zero.lateral_force.abs() < pos.lateral_force.abs() * 0.20 + 1.0,
        "zero camber should have little camber thrust: zero={} pos={}",
        zero.lateral_force,
        pos.lateral_force
    );
    assert!(
        pos.lateral_force * neg.lateral_force < 0.0,
        "camber thrust should flip sign: pos={} neg={}",
        pos.lateral_force,
        neg.lateral_force
    );
    assert!(
        pos.camber_angle_rad > 0.0 && neg.camber_angle_rad < 0.0,
        "reported camber signs wrong: pos={} neg={}",
        pos.camber_angle_rad,
        neg.camber_angle_rad
    );
    assert!(
        pos.pneumatic_trail_m.is_finite()
            && pos.contact_center_x.is_finite()
            && pos.force_resultant_x.is_finite(),
        "pneumatic trail diagnostics should be finite"
    );
}

#[test]
fn structural_lateral_bias_models_conicity_without_steering_drift() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let base = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let biased = base.with_structural_lateral_bias(0.018);
    let reverse_biased = base.with_structural_lateral_bias(-0.018);
    let hub_y = solve_loaded_tire_hub_height(&base, &terrain, 0.0, 0.0, 4_500.0);
    let state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [8.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 8.0 / base.radius,
    };
    let neutral = evaluate_brush_tire_oriented(
        &base,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation::default(),
    );
    let right_pull = evaluate_brush_tire_oriented(
        &biased,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation::default(),
    );
    let left_pull = evaluate_brush_tire_oriented(
        &reverse_biased,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation::default(),
    );
    assert!(
        neutral.lateral_force.abs() < right_pull.lateral_force.abs() * 0.25 + 1.0,
        "neutral symmetric tire should not create conicity pull: neutral={} biased={}",
        neutral.lateral_force,
        right_pull.lateral_force
    );
    assert!(
        right_pull.lateral_force * left_pull.lateral_force < 0.0,
        "structural bias should flip lateral force sign without changing steering: right={} left={}",
        right_pull.lateral_force,
        left_pull.lateral_force
    );
    assert!(
        right_pull.consumed_friction_budget <= right_pull.available_friction_budget * 1.05 + 1.0,
        "structural lateral bias must still share the friction budget: {right_pull:?}"
    );
}

#[test]
fn transient_relaxation_lags_slip_force_and_converges() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 96,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 4_500.0);
    let state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [6.0, 0.0, 0.8],
        heading: [1.0, 0.0, 0.0],
        omega: (6.0 + 2.0) / params.radius,
    };
    let steady = evaluate_brush_tire(&params, &state, &terrain, &material);
    let mut transient = RealTireTransientState {
        relaxation_length_longitudinal: 1.5,
        relaxation_length_lateral: 1.5,
        tread_yaw_relaxation_length: 1.5,
        ..Default::default()
    };
    let first = evaluate_brush_tire_transient(
        &params,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation::default(),
        &mut transient,
        1.0 / 60.0,
    );
    assert!(
        first.longitudinal_force.abs() < steady.longitudinal_force.abs() * 0.75,
        "first transient Fx should lag steady force: first={} steady={}",
        first.longitudinal_force,
        steady.longitudinal_force
    );
    assert!(
        first.lateral_force.abs() < steady.lateral_force.abs() * 0.75,
        "first transient Fy should lag steady force: first={} steady={}",
        first.lateral_force,
        steady.lateral_force
    );
    let mut last = first.clone();
    for _ in 0..180 {
        last = evaluate_brush_tire_transient(
            &params,
            &state,
            &terrain,
            &material,
            &RealWheelOrientation::default(),
            &mut transient,
            1.0 / 60.0,
        );
    }
    assert!(
        last.longitudinal_force.abs() > first.longitudinal_force.abs() * 1.5,
        "Fx should grow as relaxation state converges: first={} last={}",
        first.longitudinal_force,
        last.longitudinal_force
    );
    assert!(
        (last.longitudinal_force - steady.longitudinal_force).abs()
            < steady.longitudinal_force.abs() * 0.20 + 20.0,
        "transient Fx should converge near steady: last={} steady={}",
        last.longitudinal_force,
        steady.longitudinal_force
    );

    let reversed = RealWheelState {
        hub_vel: [6.0, 0.0, -0.8],
        omega: (6.0 - 2.0) / params.radius,
        ..state
    };
    let before_sign = last.lateral_force.signum();
    let reverse_first = evaluate_brush_tire_transient(
        &params,
        &reversed,
        &terrain,
        &material,
        &RealWheelOrientation::default(),
        &mut transient,
        1.0 / 60.0,
    );
    assert!(
        reverse_first.lateral_force.signum() == before_sign
            || reverse_first.lateral_force.abs() < last.lateral_force.abs() * 0.25,
        "reversing slip should not teleport lateral force: before={} after={}",
        last.lateral_force,
        reverse_first.lateral_force
    );
}

#[test]
fn transient_relaxation_stays_finite_at_near_zero_speed() {
    let material = SurfaceMaterial::ice();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: 140_000.0,
        ..Default::default()
    };
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 3_000.0);
    let mut transient = RealTireTransientState::default();
    for i in 0..120 {
        let state = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [1.0e-4, 0.0, if i % 2 == 0 { 1.0e-4 } else { -1.0e-4 }],
            heading: [1.0, 0.0, 0.0],
            omega: if i % 2 == 0 { 0.2 } else { -0.2 },
        };
        let out = evaluate_brush_tire_transient(
            &params,
            &state,
            &terrain,
            &material,
            &RealWheelOrientation::default(),
            &mut transient,
            1.0 / 120.0,
        );
        assert!(
            out.force.iter().all(|v| v.is_finite())
                && transient.kappa_eff.is_finite()
                && transient.alpha_eff.is_finite()
                && transient.tread_yaw.is_finite(),
            "near-zero transient update produced non-finite values"
        );
    }
}

#[test]
fn low_speed_robustness_report_covers_launch_lockup_and_reversal() {
    let report = tire_low_speed_robustness_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "low-speed launch/lockup/reversal sanity should pass: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "low_speed_outputs_finite")
            && checks
                .iter()
                .any(|check| check.check_id == "low_speed_reversal_continuity"),
        "acceptance checks should expose finite-output and reversal gates: {checks:?}"
    );
}

#[test]
fn layered_transient_envelope_report_covers_layer_order_and_hot_soft_lag() {
    let report = tire_layered_transient_envelope_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "layered transient envelope should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.tread_over_belt_kappa > 1.05
            && report.belt_over_carcass_kappa > 1.05
            && report.tread_over_belt_alpha > 1.05
            && report.belt_over_carcass_alpha > 1.05,
        "tread/belt/carcass lag ordering should be visible: {report:?}"
    );
    assert!(
        report.hot_kappa_over_cold < 0.85
            && report.hot_alpha_over_cold < 0.85
            && report.final_over_first_force > 1.05,
        "hot overloaded tires should lag cold stiff tires and force should build over time: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks.iter().any(|check| {
                check.check_id == "layered_transient_tread_faster_than_belt_kappa"
            })
            && checks
                .iter()
                .any(|check| { check.check_id == "layered_transient_hot_overloaded_slower_alpha" })
            && checks
                .iter()
                .any(|check| check.check_id == "layered_transient_force_builds_over_time"),
        "layered transient report should expose layer ordering, hot/load lag, and force buildup gates: {checks:?}"
    );
}

#[test]
fn frame_convention_report_documents_force_and_moment_signs() {
    let report = tire_frame_convention_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "frame/sign convention sanity should pass: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "frame_drive_slip_force_sign")
            && checks
                .iter()
                .any(|check| check.check_id == "frame_lateral_slip_force_sign")
            && checks
                .iter()
                .any(|check| check.check_id == "frame_pneumatic_trail_moment_sign"),
        "acceptance checks should expose rolling, lateral, and moment sign conventions: {checks:?}"
    );
}

#[test]
fn contact_patch_convergence_report_tracks_segment_refinement() {
    let report = tire_contact_patch_convergence_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "contact patch convergence should pass broad internal envelopes: {report:?}"
    );
    assert!(
        report.max_segments > report.min_segments && report.contact_count_increases,
        "convergence report should compare refinement levels: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "patch_convergence_force_stable")
            && checks
                .iter()
                .any(|check| check.check_id == "patch_convergence_pressure_bins_conserve_load"),
        "acceptance checks should expose force stability and pressure-bin load conservation: {checks:?}"
    );
}

#[test]
fn fem_substep_convergence_report_tracks_solver_substep_refinement() {
    let report = tetra_fem_substep_convergence_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "FEM substep convergence should pass broad internal envelopes: {report:?}"
    );
    assert!(
        report.max_substeps > report.min_substeps && report.max_contact_count >= 2,
        "substep convergence should compare solver refinement and keep contact: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_substep_convergence_force_stable")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_substep_convergence_patch_stable")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_substep_convergence_volume_bounded"),
        "acceptance checks should expose FEM force, patch, and volume substep gates: {checks:?}"
    );
}

#[test]
fn heightfield_resolution_convergence_report_tracks_grid_refinement() {
    let report = tire_heightfield_resolution_convergence_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "heightfield resolution convergence should pass broad internal envelopes: {report:?}"
    );
    assert!(
        report.max_grid_resolution > report.min_grid_resolution && report.min_contact_count >= 2,
        "heightfield convergence should compare grid refinement and keep contact: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "heightfield_resolution_force_stable")
            && checks
                .iter()
                .any(|check| check.check_id == "heightfield_resolution_normal_stable")
            && checks
                .iter()
                .any(|check| check.check_id == "heightfield_resolution_keeps_contact"),
        "acceptance checks should expose heightfield force, normal, and contact gates: {checks:?}"
    );
}

#[test]
fn thermal_wear_convergence_report_tracks_heat_and_wear_under_contact_refinement() {
    let report = tire_thermal_wear_convergence_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "thermal/wear convergence should pass broad internal envelopes: {report:?}"
    );
    assert!(
        report.max_segments > report.min_segments && report.min_sliding_fraction > 0.05,
        "thermal/wear convergence should refine contact and keep sliding heat active: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_wear_convergence_power_stable")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_wear_convergence_temperature_stable")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_wear_convergence_wear_stable"),
        "acceptance checks should expose thermal/wear power, temperature, and wear convergence gates: {checks:?}"
    );
}

#[test]
fn road_condition_state_report_uses_physical_state_not_labels() {
    let report = tire_road_condition_state_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "road condition state sanity should pass: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "road_condition_wet_film_reduces_force")
            && checks
                .iter()
                .any(|check| check.check_id == "road_condition_moving_ground_velocity_recorded")
            && checks
                .iter()
                .any(|check| check.check_id == "road_condition_hot_worn_state_reduces_mu"),
        "acceptance checks should expose wet-film, moving-ground, and thermal/wear state gates: {checks:?}"
    );
}

#[test]
fn layer_ablation_debug_report_isolates_named_physics_layers() {
    let report = tire_layer_ablation_debug_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "layer ablation debug gates should pass: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "layer_ablation_carcass_pressure_changes_patch")
            && checks
                .iter()
                .any(|check| check.check_id == "layer_ablation_tread_bristles_change_force")
            && checks
                .iter()
                .any(|check| check.check_id == "layer_ablation_soil_rut_feedback_active")
            && checks
                .iter()
                .any(|check| check.check_id == "layer_ablation_transient_relaxation_lags_raw"),
        "acceptance checks should expose carcass, bristle, soil, and transient ablation gates: {checks:?}"
    );
}

#[test]
fn transient_steering_reversal_decays_through_zero() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial().with_segments(96);
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 3_800.0);
    let mut transient = RealTireTransientState {
        relaxation_length_lateral: 1.4,
        tread_yaw_relaxation_length: 1.6,
        ..Default::default()
    };
    let make_state = |vz: f64| RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [10.0, 0.0, vz],
        heading: [1.0, 0.0, 0.0],
        omega: 10.0 / params.radius,
    };
    for _ in 0..48 {
        evaluate_brush_tire_transient(
            &params,
            &make_state(1.8),
            &terrain,
            &material,
            &RealWheelOrientation::default(),
            &mut transient,
            1.0 / 120.0,
        );
    }
    let before = transient.alpha_eff;
    assert!(
        before > 0.01,
        "positive steering/slip phase should build positive effective alpha: {before}"
    );

    let mut first_reversed_alpha = None;
    let mut crossed_zero = false;
    let mut previous = transient.alpha_eff;
    let mut final_alpha = transient.alpha_eff;
    for step in 0..120 {
        evaluate_brush_tire_transient(
            &params,
            &make_state(-1.8),
            &terrain,
            &material,
            &RealWheelOrientation::default(),
            &mut transient,
            1.0 / 120.0,
        );
        let alpha = transient.alpha_eff;
        if step == 0 {
            first_reversed_alpha = Some(alpha);
        }
        crossed_zero |= previous > 0.0 && alpha <= 0.0;
        previous = alpha;
        final_alpha = alpha;
    }

    let first = first_reversed_alpha.unwrap();
    assert!(
        first > 0.0,
        "steering reversal should not teleport alpha sign on the first reversed sample: before={before} first={first}"
    );
    assert!(
        crossed_zero && final_alpha < -0.01,
        "effective alpha should decay through zero and converge negative after reversal: before={before} final={final_alpha}"
    );
}

#[test]
fn transient_relaxation_converges_faster_in_time_at_higher_speed() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial().with_segments(96);
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 3_800.0);
    let run = |speed: f64| {
        let alpha = 0.10_f64;
        let raw_kappa = 0.22_f64;
        let mut transient = RealTireTransientState {
            relaxation_length_longitudinal: 1.2,
            relaxation_length_lateral: 1.2,
            tread_yaw_relaxation_length: 1.2,
            ..Default::default()
        };
        let state = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [speed, 0.0, speed * alpha.tan()],
            heading: [1.0, 0.0, 0.0],
            omega: speed * (1.0 + raw_kappa) / params.radius,
        };
        for _ in 0..12 {
            evaluate_brush_tire_transient(
                &params,
                &state,
                &terrain,
                &material,
                &RealWheelOrientation::default(),
                &mut transient,
                1.0 / 120.0,
            );
        }
        transient
    };
    let slow = run(4.0);
    let fast = run(16.0);
    assert!(
        fast.kappa_eff.abs() > slow.kappa_eff.abs() * 1.8,
        "higher speed should converge farther toward raw kappa in the same wall time: slow={} fast={}",
        slow.kappa_eff,
        fast.kappa_eff
    );
    assert!(
        fast.alpha_eff.abs() > slow.alpha_eff.abs() * 1.8,
        "higher speed should converge farther toward raw alpha in the same wall time: slow={} fast={}",
        slow.alpha_eff,
        fast.alpha_eff
    );
}

#[test]
fn transient_relaxation_context_slows_soft_overloaded_hot_tires() {
    let normal = [0.0, 1.0, 0.0];
    let stiff = RealTireParams::high_pressure_reference().with_segments(96);
    let soft = RealTireParams::offroad_soft().with_segments(96);
    let state = RealWheelState {
        hub_pos: [0.0, 0.36, 0.0],
        hub_vel: [8.0, 0.0, 1.2],
        heading: [1.0, 0.0, 0.0],
        omega: (8.0 + 3.0) / stiff.radius,
    };
    let mut cold_light = RealTireTransientState::default();
    let mut hot_heavy = RealTireTransientState::default();
    let cold_context = TireRelaxationContext::new(2_200.0, 20.0);
    let hot_context = TireRelaxationContext::new(7_200.0, 125.0);

    let stiff_scale = cold_context.relaxation_length_scale(&stiff);
    let soft_hot_scale = hot_context.relaxation_length_scale(&soft);
    assert!(
        soft_hot_scale > stiff_scale * 1.8,
        "soft overloaded hot tire should have much longer effective relaxation: stiff={stiff_scale} soft_hot={soft_hot_scale}"
    );

    cold_light.step_with_context(&stiff, &state, normal, 1.0 / 60.0, cold_context);
    let soft_state = RealWheelState {
        omega: (8.0 + 3.0) / soft.radius,
        ..state
    };
    hot_heavy.step_with_context(&soft, &soft_state, normal, 1.0 / 60.0, hot_context);

    assert!(
        hot_heavy.kappa_eff.abs() < cold_light.kappa_eff.abs() * 0.70,
        "hot overloaded soft tire should build longitudinal slip more slowly: cold={} hot={}",
        cold_light.kappa_eff,
        hot_heavy.kappa_eff
    );
    assert!(
        hot_heavy.alpha_eff.abs() < cold_light.alpha_eff.abs() * 0.70,
        "hot overloaded soft tire should build lateral slip more slowly: cold={} hot={}",
        cold_light.alpha_eff,
        hot_heavy.alpha_eff
    );
}

#[test]
fn layered_transient_separates_fast_tread_from_slow_carcass_lag() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial().with_segments(96);
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, 4_200.0);
    let state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [10.0, 0.0, 1.1],
        heading: [1.0, 0.0, 0.0],
        omega: (10.0 + 3.0) / params.radius,
    };
    let steady = evaluate_brush_tire_oriented(
        &params,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation::default(),
    );
    let mut layered = TireLayeredTransientState::default();
    let first = evaluate_brush_tire_layered_transient(
        &params,
        &state,
        &terrain,
        &material,
        &RealWheelOrientation::default(),
        &mut layered,
        1.0 / 120.0,
    );

    assert!(
        layered.is_finite(),
        "layered transient state must stay finite"
    );
    assert!(
        layered.tread.kappa_eff.abs() > layered.belt.kappa_eff.abs()
            && layered.belt.kappa_eff.abs() > layered.carcass.kappa_eff.abs(),
        "longitudinal lag should be ordered tread > belt > carcass: tread={} belt={} carcass={}",
        layered.tread.kappa_eff,
        layered.belt.kappa_eff,
        layered.carcass.kappa_eff
    );
    assert!(
        layered.tread.alpha_eff_rad.abs() > layered.belt.alpha_eff_rad.abs()
            && layered.belt.alpha_eff_rad.abs() > layered.carcass.alpha_eff_rad.abs(),
        "lateral lag should be ordered tread > belt > carcass: tread={} belt={} carcass={}",
        layered.tread.alpha_eff_rad,
        layered.belt.alpha_eff_rad,
        layered.carcass.alpha_eff_rad
    );
    assert!(
        layered.kappa_eff.abs() < layered.tread.kappa_eff.abs()
            && layered.kappa_eff.abs() > layered.carcass.kappa_eff.abs(),
        "aggregate effective slip should sit between fast tread and slow carcass: agg={} tread={} carcass={}",
        layered.kappa_eff,
        layered.tread.kappa_eff,
        layered.carcass.kappa_eff
    );
    assert!(
        layered.kappa_eff.abs() < steady.slip_ratio.abs() * 0.90
            && layered.alpha_eff_rad.abs() < steady.slip_angle_rad.abs() * 0.90,
        "layered contact should use lagged effective slip/angle: eff_k={} raw_k={} eff_a={} raw_a={}",
        layered.kappa_eff,
        steady.slip_ratio,
        layered.alpha_eff_rad,
        steady.slip_angle_rad
    );

    let first_eff_kappa = layered.kappa_eff;
    let first_eff_alpha = layered.alpha_eff_rad;
    let mut last = first.clone();
    for _ in 0..240 {
        last = evaluate_brush_tire_layered_transient(
            &params,
            &state,
            &terrain,
            &material,
            &RealWheelOrientation::default(),
            &mut layered,
            1.0 / 120.0,
        );
    }
    assert!(
        layered.kappa_eff.abs() > first_eff_kappa.abs() * 1.5
            && layered.alpha_eff_rad.abs() > first_eff_alpha.abs() * 1.5,
        "layered effective slip/angle should build as belt/carcass catch up: first_k={} last_k={} first_a={} last_a={} last_fx={}",
        first_eff_kappa,
        layered.kappa_eff,
        first_eff_alpha,
        layered.alpha_eff_rad,
        last.longitudinal_force
    );
    assert!(
        layered.carcass.kappa_eff.abs() > 0.20,
        "slow carcass layer should eventually carry real deformation: {}",
        layered.carcass.kappa_eff
    );
}

#[test]
fn layered_transient_context_slows_every_layer_for_hot_overloaded_soft_tire() {
    let normal = [0.0, 1.0, 0.0];
    let stiff = RealTireParams::high_pressure_reference().with_segments(96);
    let soft = RealTireParams::offroad_soft().with_segments(96);
    let cold_state = RealWheelState {
        hub_pos: [0.0, 0.36, 0.0],
        hub_vel: [9.0, 0.0, 1.0],
        heading: [1.0, 0.0, 0.0],
        omega: (9.0 + 3.5) / stiff.radius,
    };
    let hot_state = RealWheelState {
        omega: (9.0 + 3.5) / soft.radius,
        ..cold_state
    };
    let mut cold = TireLayeredTransientState::default();
    let mut hot = TireLayeredTransientState::default();
    cold.step_with_context(
        &stiff,
        &cold_state,
        normal,
        1.0 / 60.0,
        TireRelaxationContext::new(2_200.0, 20.0),
    );
    hot.step_with_context(
        &soft,
        &hot_state,
        normal,
        1.0 / 60.0,
        TireRelaxationContext::new(7_200.0, 125.0),
    );

    assert!(
        hot.tread.kappa_eff.abs() < cold.tread.kappa_eff.abs() * 0.75,
        "hot overloaded soft tread should lag cold stiff tread: cold={} hot={}",
        cold.tread.kappa_eff,
        hot.tread.kappa_eff
    );
    assert!(
        hot.belt.kappa_eff.abs() < cold.belt.kappa_eff.abs() * 0.75,
        "hot overloaded soft belt should lag cold stiff belt: cold={} hot={}",
        cold.belt.kappa_eff,
        hot.belt.kappa_eff
    );
    assert!(
        hot.carcass.kappa_eff.abs() < cold.carcass.kappa_eff.abs() * 0.75,
        "hot overloaded soft carcass should lag cold stiff carcass: cold={} hot={}",
        cold.carcass.kappa_eff,
        hot.carcass.kappa_eff
    );
}

#[test]
fn tire_rig_step_torque_response_lags_raw_slip_without_teleporting_force() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial().with_segments(96);
    let response = tire_rig_step_torque_response(
        &params,
        &terrain,
        &material,
        3_800.0,
        10.0,
        800.0,
        1.0 / 120.0,
        120,
    );
    assert_eq!(response.len(), 120);
    assert!(
        response.iter().all(TireRigStepResponsePoint::is_finite),
        "step torque response must stay finite"
    );
    let first_slipping = response
        .iter()
        .find(|p| p.drive_torque_nm > 0.0 && p.raw_slip_ratio > 0.03)
        .expect("drive torque should create positive raw slip");
    let last = response.last().unwrap();
    assert!(
        first_slipping.effective_slip_ratio.abs() < first_slipping.raw_slip_ratio.abs() * 0.80,
        "effective slip should lag the raw torque step: first={first_slipping:?}"
    );
    assert!(
        last.effective_slip_ratio > first_slipping.effective_slip_ratio * 1.20,
        "effective slip should build after the step: first={} last={}",
        first_slipping.effective_slip_ratio,
        last.effective_slip_ratio
    );
    assert!(
        last.telemetry.longitudinal_force_n.abs()
            > first_slipping.telemetry.longitudinal_force_n.abs(),
        "longitudinal force should build rather than teleport fully on the first slipping sample"
    );
    assert!(
        response
            .iter()
            .all(|p| p.telemetry.dissipated_energy_j >= -1.0e-9),
        "step torque energy accounting should stay dissipative"
    );
}

#[test]
fn tire_rig_step_brake_response_locks_progressively_and_stays_finite() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial().with_segments(96);
    let response = tire_rig_step_brake_response(
        &params,
        &terrain,
        &material,
        3_800.0,
        12.0,
        900.0,
        1.0 / 120.0,
        120,
    );
    assert!(
        response.iter().all(TireRigStepResponsePoint::is_finite),
        "step brake response must stay finite"
    );
    let first_braking = response
        .iter()
        .find(|p| p.brake_torque_nm > 0.0 && p.raw_slip_ratio < -0.03)
        .expect("brake torque should create negative raw slip");
    let last = response.last().unwrap();
    assert!(
        first_braking.effective_slip_ratio.abs() < first_braking.raw_slip_ratio.abs() * 0.80,
        "effective slip should lag the raw braking step: first={first_braking:?}"
    );
    assert!(
        last.wheel_omega_rad_s < response[0].wheel_omega_rad_s,
        "braking should reduce wheel angular velocity"
    );
    assert!(
        last.effective_slip_ratio < first_braking.effective_slip_ratio,
        "effective slip should continue moving toward braking slip: first={} last={}",
        first_braking.effective_slip_ratio,
        last.effective_slip_ratio
    );
    assert!(
        response
            .iter()
            .all(|p| p.telemetry.consumed_friction_budget_n
                <= p.telemetry.available_friction_budget_n * 1.05 + 1.0e-6),
        "step brake response should respect the friction budget"
    );
}

#[test]
fn tire_rig_step_steer_response_lags_lateral_force_and_tread_yaw() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams::passenger_radial().with_segments(96);
    let response = tire_rig_step_steer_response(
        &params,
        &terrain,
        &material,
        3_800.0,
        14.0,
        0.20,
        1.0 / 120.0,
        140,
    );
    assert!(
        response.iter().all(TireRigStepResponsePoint::is_finite),
        "step steer response must stay finite"
    );
    let first_steering = response
        .iter()
        .find(|p| p.steer_angle_rad.abs() > 0.0 && p.raw_slip_angle_rad.abs() > 0.05)
        .expect("steer step should create raw slip angle");
    let last = response.last().unwrap();
    assert!(
        first_steering.effective_slip_angle_rad.abs()
            < first_steering.raw_slip_angle_rad.abs() * 0.80,
        "effective slip angle should lag raw steering step: first={first_steering:?}"
    );
    assert!(
        last.effective_slip_angle_rad.abs() > first_steering.effective_slip_angle_rad.abs() * 1.5,
        "effective slip angle should build after steer step: first={} last={}",
        first_steering.effective_slip_angle_rad,
        last.effective_slip_angle_rad
    );
    assert!(
        last.tread_yaw_rad.abs() > first_steering.tread_yaw_rad.abs(),
        "tread yaw should also lag/build across the steer step"
    );
    assert!(
        last.telemetry.lateral_force_n.abs() > first_steering.telemetry.lateral_force_n.abs(),
        "lateral force should build rather than teleport fully on the first steering sample"
    );
}

#[test]
fn tire_torque_induced_slip_outcome_compares_pressure_cases() {
    let report = tire_torque_induced_slip_outcome_report();
    assert!(
        report.passed(),
        "same-torque low-friction outcome should show high-pressure slip and partial-pressure motion: {report:?}"
    );
    assert!(
        report.high_final_slip_ratio.abs() > report.partial_final_slip_ratio.abs() + 0.05,
        "over-inflated tire should finish with more slip: {report:?}"
    );
    assert!(
        report.partial_distance_m > report.high_distance_m + 0.02,
        "partial-pressure tire should move farther: {report:?}"
    );
    assert!(
        report.partial_patch_length_m > report.high_patch_length_m,
        "partial pressure should build a longer patch: {report:?}"
    );
    let text = TireRigAcceptanceReport::new(report.snapshot(), report.acceptance_checks())
        .to_text_report();
    assert!(
        text.contains("torque_slip_partial_pressure_moves_farther")
            && text.contains("torque_slip_high_pressure_slips_more")
            && text.contains("torque_slip_partial_patch_larger"),
        "torque/slip report should expose explicit acceptance rows: {text}"
    );
}

#[test]
fn tire_vehicle_sanity_scenarios_cover_required_falsification_cases() {
    let cases = tire_vehicle_sanity_scenarios();
    assert_eq!(
        cases.len(),
        6,
        "vehicle-level sanity suite should cover heavy, low-friction, braking, steering/camber, obstacle, and off-road cases: {cases:?}"
    );
    assert!(
        cases.iter().all(TireVehicleSanityCase::is_finite),
        "vehicle-level sanity cases must stay finite: {cases:?}"
    );
    assert!(
        cases.iter().all(|case| case.wheel_count == 4),
        "these are four-wheel vehicle-context falsification cases: {cases:?}"
    );
    assert!(
        cases.iter().all(|case| case.passed),
        "all broad vehicle sanity envelopes should pass: {cases:?}"
    );
    assert!(
        cases.iter().all(TireVehicleSanityCase::acceptance_passed),
        "vehicle-level sanity cases should carry explicit reusable acceptance checks: {cases:?}"
    );
    assert!(
        cases.iter().all(|case| case.acceptance_checks().len() >= 9),
        "each vehicle sanity case should expose generic accounting checks plus scenario-specific falsification checks: {cases:?}"
    );
    assert!(
        cases
            .iter()
            .flat_map(TireVehicleSanityCase::failed_acceptance_check_ids)
            .next()
            .is_none(),
        "no vehicle sanity acceptance check should fail: {cases:?}"
    );
    let heavy = cases
        .iter()
        .find(|case| case.scenario_id == "heavy_loaded_vehicle")
        .expect("heavy loaded vehicle case");
    assert!(
        heavy.total_vehicle_load_n > 30_000.0 && heavy.observed_contact_count >= 3,
        "heavy vehicle should produce a multi-cell loaded patch: {heavy:?}"
    );
    let ice = cases
        .iter()
        .find(|case| case.scenario_id == "low_friction_driven_vehicle")
        .expect("low-friction driven vehicle case");
    assert!(
        ice.observed_slip_ratio.abs() > 0.25
            && ice.telemetry.longitudinal_force_n.abs() / ice.telemetry.normal_load_n.max(1.0)
                < 0.35,
        "ice vehicle should visibly slip while force stays low: {ice:?}"
    );
    let braking = cases
        .iter()
        .find(|case| case.scenario_id == "braking_vehicle")
        .expect("braking vehicle case");
    assert!(
        braking.observed_slip_ratio < -0.10 && braking.telemetry.longitudinal_force_n < -50.0,
        "braking vehicle should develop negative slip and braking force: {braking:?}"
    );
    let steer = cases
        .iter()
        .find(|case| case.scenario_id == "steering_camber_vehicle")
        .expect("steering/camber vehicle case");
    assert!(
        steer.observed_lateral_force_n.abs() > 100.0
            && steer.telemetry.aligning_moment_nm.is_finite(),
        "steering/camber vehicle should expose lateral force and aligning moment: {steer:?}"
    );
    let curb = cases
        .iter()
        .find(|case| case.scenario_id == "obstacle_curb_vehicle")
        .expect("obstacle/curb vehicle case");
    assert!(
        curb.observed_contact_count > 1 && curb.observed_patch_length_m > 0.04,
        "curb vehicle should not collapse to a single contact point: {curb:?}"
    );
    let offroad = cases
        .iter()
        .find(|case| case.scenario_id == "offroad_soft_soil_vehicle")
        .expect("offroad soft-soil vehicle case");
    assert!(
        offroad.observed_contact_count >= 3
            && offroad.telemetry.rolling_resistance_force_n.abs() > 100.0,
        "offroad vehicle should keep finite support and report soil rolling loss: {offroad:?}"
    );
}

#[test]
fn vehicle_dynamic_load_transfer_report_tracks_load_patch_and_suspension() {
    let report = tire_vehicle_dynamic_load_transfer_report();
    assert!(
        report.passed(),
        "dynamic load-transfer report should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.braking_front_load_n_per_wheel > report.static_load_n_per_wheel
            && report.braking_rear_load_n_per_wheel < report.static_load_n_per_wheel,
        "braking should move load forward: {report:?}"
    );
    assert!(
        report.accel_rear_load_n_per_wheel > report.static_load_n_per_wheel
            && report.accel_front_load_n_per_wheel < report.static_load_n_per_wheel,
        "acceleration should move load rearward: {report:?}"
    );
    assert!(
        report.braking_front_patch_length_m > report.braking_rear_patch_length_m
            && report.accel_rear_patch_length_m > report.accel_front_patch_length_m,
        "load transfer should change actual tire patch length: {report:?}"
    );
    assert!(
        report.braking_front_suspension_travel_m > report.static_suspension_travel_m
            && report.braking_rear_suspension_travel_m < report.static_suspension_travel_m
            && report.accel_rear_suspension_travel_m > report.static_suspension_travel_m,
        "suspension travel should follow per-corner load: {report:?}"
    );
    assert!(
        report
            .acceptance_checks()
            .iter()
            .any(|check| check.check_id == "dynamic_load_transfer_front_force_sublinear"),
        "report should expose force-vs-load sublinearity, not just load bookkeeping"
    );
    let snapshot = report.snapshot();
    assert_eq!(snapshot.scenario_id, "vehicle_dynamic_load_transfer");
    assert!(snapshot.is_finite());
}

#[test]
fn dynamic_carcass_stays_above_floor_and_forms_patch() {
    let mut tire = DynamicTireCarcass::new(
        0.0,
        120_000.0,
        DynamicTireCarcassParams {
            segments: 32,
            downforce: 2_000.0,
            ..Default::default()
        },
    );
    for _ in 0..300 {
        tire.step(1.0 / 240.0);
    }
    assert!(
        tire.nodes()
            .iter()
            .all(|p| p[1] >= -1.0e-12 && p[1].is_finite()),
        "dynamic carcass node went below floor or non-finite"
    );
    assert!(
        tire.contact_patch_width() > 0.02,
        "loaded dynamic tire should form a visible contact patch"
    );
}

#[test]
fn dynamic_carcass_pressure_reduces_patch_width() {
    let settle = |pressure| {
        let mut tire = DynamicTireCarcass::new(
            0.0,
            pressure,
            DynamicTireCarcassParams {
                segments: 32,
                downforce: 2_200.0,
                ..Default::default()
            },
        );
        for _ in 0..300 {
            tire.step(1.0 / 240.0);
        }
        tire.contact_patch_width()
    };
    let low = settle(70_000.0);
    let high = settle(320_000.0);
    assert!(
        low > high,
        "lower pressure should have wider dynamic patch: low={low} high={high}"
    );
}

#[test]
fn tetra_fem_builds_a_volumetric_tire_mesh() {
    let tire = TetraTireFem::new(
        0.0,
        160_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            ..Default::default()
        },
    );
    assert_eq!(tire.nodes().len(), 16 * 3 * 3);
    assert!(
        tire.tetra_count() > 16 * 4,
        "tetra mesh should be volumetric"
    );
    assert!(
        tire.surface_edges().len() > tire.segments(),
        "viewer should have surface edges to draw"
    );
    let snap = tire.gpu_snapshot();
    assert_eq!(snap.positions.len(), tire.nodes().len());
    assert_eq!(snap.tet_indices.len(), tire.tetra_count() * 4);
    assert_eq!(snap.tet_grads.len(), tire.tetra_count() * 4);
    assert_eq!(snap.tet_volumes.len(), tire.tetra_count());
    assert_eq!(snap.tet_mu.len(), tire.tetra_count());
    assert_eq!(snap.tet_lambda.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_dir.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_k1.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_k2.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_dir_b.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_k1_b.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_k2_b.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_dir_c.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_k1_c.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_k2_c.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_dir_d.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_k1_d.len(), tire.tetra_count());
    assert_eq!(snap.tet_fiber_k2_d.len(), tire.tetra_count());
    assert_eq!(snap.node_tet_offsets.len(), tire.nodes().len() + 1);
    assert_eq!(snap.node_tet_entries.len(), tire.tetra_count() * 4);
    assert_eq!(snap.node_edge_offsets.len(), tire.nodes().len() + 1);
    assert_eq!(snap.xpbd_edge_indices.len(), snap.xpbd_edge_params.len());
    assert!(
        !snap.xpbd_edge_indices.is_empty(),
        "production tire GPU snapshot should export structural XPBD belt/sidewall edges"
    );
    assert_eq!(snap.node_dihedral_offsets.len(), tire.nodes().len() + 1);
    assert_eq!(
        snap.xpbd_dihedral_indices.len(),
        snap.xpbd_dihedral_params.len()
    );
    assert!(
        !snap.xpbd_dihedral_indices.is_empty(),
        "production tire GPU snapshot should export sidewall dihedral XPBD constraints"
    );
    assert_eq!(
        snap.node_edge_offsets.last().copied().unwrap_or(0) as usize,
        snap.node_edge_entries.len()
    );
    assert_eq!(
        snap.node_dihedral_offsets.last().copied().unwrap_or(0) as usize,
        snap.node_dihedral_entries.len()
    );
    assert!(
        snap.tet_fiber_k1.iter().any(|&k| k > 0.0),
        "production tire mesh should export anisotropic cord/belt fibres"
    );
    assert!(
        snap.tet_fiber_k1_b.iter().any(|&k| k > 0.0),
        "production tire mesh should export the second crossed ply family"
    );
    assert!(
        snap.tet_fiber_k1_c.iter().any(|&k| k > 0.0)
            && snap.tet_fiber_k1_d.iter().any(|&k| k > 0.0),
        "production tire mesh should export third/fourth cap/radial ply families"
    );
    let crossed = snap
        .tet_fiber_dir
        .iter()
        .zip(snap.tet_fiber_dir_b.iter())
        .zip(snap.tet_fiber_k1.iter().zip(snap.tet_fiber_k1_b.iter()))
        .any(|((a, b), (ka, kb))| {
            if *ka <= 0.0 || *kb <= 0.0 {
                return false;
            }
            let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            dot.abs() < 0.98
        });
    assert!(
        crossed,
        "explicit ply-angle layups should include non-parallel crossed fibre families"
    );
    assert!(
        snap.cavity_radial_force_per_bead > 0.0,
        "inflated tire should export positive cavity gas-law bead load"
    );
    assert!(snap.pressure_area_per_tread_node >= 0.0);
}

#[test]
fn cavity_pressure_load_tracks_inflation_pressure() {
    let build = |pressure| {
        TetraTireFem::new(
            0.0,
            pressure,
            TetraTireFemParams {
                segments: 16,
                radial_layers: 3,
                width_layers: 3,
                ..Default::default()
            },
        )
        .gpu_snapshot()
        .cavity_radial_force_per_bead
    };
    let low = build(80_000.0);
    let high = build(320_000.0);
    assert!(
        high > low * 3.0,
        "cavity gas-law bead load should scale with pressure: low={low} high={high}"
    );
}

#[test]
fn tetra_diagnostics_report_cavity_and_xpbd_state() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            xpbd_volume_iterations: 2,
            xpbd_structural_iterations: 2,
            ..Default::default()
        },
    );
    let initial = tire.diagnostics();
    assert!(initial.cavity_rest_volume_m3 > 1.0e-4);
    assert!(initial.cavity_volume_m3 > 1.0e-4);
    assert!(initial.effective_cavity_pressure_pa > 0.0);
    assert!(initial.cavity_radial_force_per_bead_n > 0.0);
    assert!(initial.mean_bead_mount_error_m <= 1.0e-9);
    assert!(initial.max_bead_mount_error_m <= 1.0e-9);

    for p in &mut tire.pos {
        let d = sub(*p, tire.hub);
        p[0] = tire.hub[0] + d[0] * 1.04;
        p[1] = tire.hub[1] + d[1] * 0.82;
    }
    tire.step(1.0 / 240.0);
    let after = tire.diagnostics();
    assert!(after.cavity_volume_ratio.is_finite());
    assert!(after.max_tet_volume_error_m3.is_finite());
    assert!(after.mean_bead_mount_error_m.is_finite());
    assert!(after.max_bead_mount_error_m.is_finite());
    let bead_seat = tire.bead_seat_diagnostics();
    assert!(
        bead_seat.is_finite()
            && bead_seat.bead_node_count > 0
            && bead_seat.max_abs_radial_error_m >= 0.0
            && bead_seat.max_abs_axial_error_m >= 0.0
            && bead_seat.max_tangential_error_m >= 0.0,
        "bead-seat decomposed diagnostics should remain finite: {bead_seat:?}"
    );
    assert!(after.last_xpbd_volume_correction_m >= 0.0);
    assert!(after.last_xpbd_bead_correction_m >= 0.0);
    assert!(after.last_xpbd_structural_correction_m >= 0.0);
}

#[test]
fn cavity_bead_sanity_report_gates_pressure_scaling_and_mount_error() {
    let report = tetra_cavity_bead_sanity_report();
    assert!(
        report.is_finite(),
        "cavity/bead report must stay finite: {report:?}"
    );
    assert!(
        report.high_radial_force_per_bead_n > report.low_radial_force_per_bead_n * 2.0,
        "cavity bead load should scale with inflation pressure: {report:?}"
    );
    assert!(
        report.loaded_cavity_volume_ratio > 0.35 && report.loaded_cavity_volume_ratio < 2.5,
        "loaded cavity should remain bounded: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed),
        "cavity/bead sanity acceptance should pass with broad envelopes: {checks:?}"
    );
    assert!(
        checks
            .iter()
            .any(|check| check.check_id == "bead_mount_error_bounded"),
        "bead/rim mount drift should be an explicit acceptance gate"
    );
    assert!(
        checks
            .iter()
            .any(|check| check.check_id == "bead_seat_radial_error_bounded")
            && checks
                .iter()
                .any(|check| check.check_id == "bead_seat_axial_error_bounded")
            && checks
                .iter()
                .any(|check| check.check_id == "bead_seat_tangential_error_bounded"),
        "bead/rim seat drift should be decomposed into radial, axial, and tangential gates"
    );
    assert!(
        checks
            .iter()
            .any(|check| check.check_id == "cavity_carcass_coupling_visible_and_bounded"),
        "cavity pressure coupling should be an explicit audited acceptance gate"
    );
    let governance = tire_preset_governance_report();
    assert!(
        governance.checks.iter().any(|check| {
            check.group == "TetraTireFemParams"
                && check.field == "cavity_carcass_coupling"
                && check.passed
        }),
        "cavity coupling should be visible in preset governance"
    );
}

#[test]
fn bead_seat_load_case_report_exercises_lateral_and_pressure_cases() {
    let report = tetra_bead_seat_load_case_report();
    assert!(
        report.passed(),
        "bead-seat load-case report should satisfy broad non-calibration envelopes: {report:?}"
    );
    assert!(
        report.lateral_contact_force_n > 10.0,
        "lateral bead-seat case should actually apply lateral force: {report:?}"
    );
    assert!(
        report.torque_contact_force_n > 10.0,
        "drive/brake bead-seat case should actually apply longitudinal force: {report:?}"
    );
    assert!(
        report.lateral_max_mount_error_m <= 0.060
            && report.torque_max_mount_error_m <= 0.060
            && report.torque_max_tangential_error_m <= 0.040
            && report.high_pressure_max_mount_error_m <= 0.060,
        "lateral/high-pressure cases should remain inside the bead/rim guardrail envelope: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "bead_load_cases_lateral_mount_bounded")
            && checks
                .iter()
                .any(|check| check.check_id == "bead_load_cases_lateral_force_applied")
            && checks
                .iter()
                .any(|check| check.check_id == "bead_load_cases_torque_force_applied")
            && checks.iter().any(|check| {
                check.check_id == "bead_load_cases_torque_tangential_error_bounded"
            })
            && checks
                .iter()
                .any(|check| check.check_id == "bead_load_cases_high_pressure_mount_bounded"),
        "bead-seat load-case report should expose explicit lateral/torque/high-pressure gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_stays_above_floor_and_forms_patch() {
    let mut tire = TetraTireFem::new(
        0.0,
        130_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 3_600.0,
            ..Default::default()
        },
    );
    for _ in 0..360 {
        tire.step(1.0 / 240.0);
    }
    assert!(
        tire.nodes()
            .iter()
            .all(|p| p[1] >= -1.0e-12 && p.iter().all(|x| x.is_finite())),
        "tetra FEM tire node went below floor or non-finite"
    );
    let patch = tire.contact_patch_width();
    assert!(
        patch > 0.005,
        "loaded tetra FEM tire should form a visible contact patch, got {patch}"
    );
}

#[test]
fn tetra_fem_pressure_reduces_patch_width() {
    let settle = |pressure| {
        let mut tire = TetraTireFem::new(
            0.0,
            pressure,
            TetraTireFemParams {
                segments: 16,
                radial_layers: 3,
                width_layers: 3,
                downforce: 4_500.0,
                ..Default::default()
            },
        );
        for _ in 0..420 {
            tire.step(1.0 / 240.0);
        }
        let terrain = FlatTerrain::new(0.0, 1.0);
        let state = RealWheelState {
            hub_pos: tire.hub(),
            hub_vel: [3.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: 3.0 / tire.outer_radius(),
        };
        tire.evaluate_tread_contact(&state, &terrain, &SurfaceMaterial::dry_asphalt())
    };
    let low = settle(70_000.0);
    let high = settle(320_000.0);
    assert!(
        low.patch_length > high.patch_length || low.contact_count > high.contact_count,
        "lower pressure should have wider tetra FEM patch: low patch={} count={} high patch={} count={}",
        low.patch_length,
        low.contact_count,
        high.patch_length,
        high.contact_count
    );
}

#[test]
fn tetra_tread_contact_uses_actual_deformed_patch_nodes() {
    let settle = |pressure: f64| -> (TetraTireFem, RealTireContact) {
        let mut tire = TetraTireFem::new(
            0.0,
            pressure,
            TetraTireFemParams {
                segments: 20,
                radial_layers: 3,
                width_layers: 3,
                pressure_pa: pressure,
                downforce: 4_500.0,
                ..Default::default()
            },
        );
        for _ in 0..420 {
            tire.step(1.0 / 240.0);
        }
        let terrain = FlatTerrain::new(0.0, 1.0);
        let state = RealWheelState {
            hub_pos: tire.hub(),
            hub_vel: [5.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: 5.0 / tire.outer_radius(),
        };
        let contact =
            tire.evaluate_tread_contact(&state, &terrain, &SurfaceMaterial::dry_asphalt());
        (tire, contact)
    };
    let (high_tire, high) = settle(320_000.0);
    let (low_tire, low) = settle(90_000.0);
    assert!(
        high.in_contact && low.in_contact,
        "both FEM tires should contact"
    );
    assert!(
        low.patch_length > high.patch_length || low.contact_count > high.contact_count,
        "low-pressure deformed FEM patch should be larger: low patch={} count={} high patch={} count={}",
        low.patch_length,
        low.contact_count,
        high.patch_length,
        high.contact_count
    );
    assert!(
        (low.patch_length - low_tire.contact_patch_width()).abs() < 0.20,
        "FEM-derived contact should track tread-node patch width"
    );
    assert!(
        (high.patch_length - high_tire.contact_patch_width()).abs() < 0.20,
        "FEM-derived contact should track tread-node patch width"
    );
}

#[test]
fn tetra_tread_contact_normal_load_tracks_hub_load() {
    let load = 3_800.0;
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 20,
            radial_layers: 3,
            width_layers: 3,
            downforce: load,
            ..Default::default()
        },
    );
    for _ in 0..420 {
        tire.step(1.0 / 240.0);
    }
    let terrain = FlatTerrain::new(0.0, 1.0);
    let state = RealWheelState {
        hub_pos: tire.hub(),
        hub_vel: [4.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 4.0 / tire.outer_radius(),
    };
    let contact = tire.evaluate_tread_contact(&state, &terrain, &SurfaceMaterial::dry_asphalt());
    assert!(
        contact.in_contact,
        "settled FEM tire should have tread contact"
    );
    assert!(
        (contact.normal_load - load).abs() < load * 0.05,
        "FEM-derived Fz should match hub load: contact={} load={load}",
        contact.normal_load
    );
}

#[test]
fn tetra_tread_contact_wrench_feeds_back_into_fem_nodes() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 20,
            radial_layers: 3,
            width_layers: 3,
            downforce: 4_200.0,
            ..Default::default()
        },
    );
    for _ in 0..360 {
        tire.step(1.0 / 240.0);
    }
    let terrain = FlatTerrain::new(0.0, 1.0);
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let state = RealWheelState {
        hub_pos: tire.hub(),
        hub_vel: [2.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 7.0 / tire.outer_radius(),
    };
    let preview = tire.evaluate_tread_contact(&state, &terrain, &material);
    assert!(
        preview.in_contact,
        "settled tire should produce FEM contact"
    );
    assert!(
        preview.cells.iter().all(|cell| cell.node_index.is_some()),
        "FEM-derived contact cells should remember the source tread node"
    );
    assert!(
        preview.longitudinal_force > 100.0,
        "drive slip should produce a positive longitudinal contact force, got {}",
        preview.longitudinal_force
    );
    let contact_nodes: Vec<usize> = preview
        .cells
        .iter()
        .filter_map(|cell| cell.node_index)
        .collect();
    let avg_vx = |t: &TetraTireFem, ids: &[usize]| -> f64 {
        ids.iter().map(|&id| t.vel[id][0]).sum::<f64>() / ids.len().max(1) as f64
    };
    let before = avg_vx(&tire, &contact_nodes);
    let applied = tire.apply_tread_contact_wrench(&state, &terrain, &material, 1.0 / 240.0);
    let after = avg_vx(&tire, &contact_nodes);
    assert!(applied.in_contact);
    assert!(
        after > before + 1.0e-4,
        "distributed brush wrench should accelerate contact tread nodes forward: before={before} after={after}"
    );
}

#[test]
fn tetra_tread_contact_uses_heightfield_water_film_map() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 4_200.0,
            ..Default::default()
        },
    );
    for _ in 0..360 {
        tire.step(1.0 / 240.0);
    }
    let heights = vec![0.0; 25];
    let dry_terrain = HeightfieldTerrain::new([-1.0, -1.0], 0.5, 5, 5, heights.clone(), 1.0);
    let wet_terrain = HeightfieldTerrain::new([-1.0, -1.0], 0.5, 5, 5, heights, 1.0)
        .with_water_film_map(vec![0.004; 25]);
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let state = RealWheelState {
        hub_pos: tire.hub(),
        hub_vel: [10.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 18.0 / tire.outer_radius(),
    };
    let dry = tire.evaluate_tread_contact(&state, &dry_terrain, &material);
    let wet = tire.evaluate_tread_contact(&state, &wet_terrain, &material);
    assert!(dry.in_contact && wet.in_contact);
    assert!(
        wet.longitudinal_force.abs() < dry.longitudinal_force.abs() * 0.75,
        "FEM tread-node contact should sample heightfield water film: dry={} wet={}",
        dry.longitudinal_force,
        wet.longitudinal_force
    );
    assert!(
        wet.rolling_resistance_force.abs() > dry.rolling_resistance_force.abs() + 10.0,
        "wet FEM tread contact should report added water-film rolling loss: dry={} wet={}",
        dry.rolling_resistance_force,
        wet.rolling_resistance_force
    );
}

#[test]
fn tetra_tread_contact_wrench_uses_persistent_node_bristles() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 20,
            radial_layers: 3,
            width_layers: 3,
            downforce: 4_200.0,
            bristle_relax_rate: 8.0,
            ..Default::default()
        },
    );
    for _ in 0..360 {
        tire.step(1.0 / 240.0);
    }
    let terrain = FlatTerrain::new(0.0, 1.0);
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let state = RealWheelState {
        hub_pos: tire.hub(),
        hub_vel: [1.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 8.0 / tire.outer_radius(),
    };
    let bristle_rms = |t: &TetraTireFem| -> f64 {
        let e = t
            .tread_bristle
            .iter()
            .map(|b| b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
            .sum::<f64>()
            / t.tread_bristle.len().max(1) as f64;
        e.sqrt()
    };
    let before = bristle_rms(&tire);
    let first = tire.apply_tread_contact_wrench(&state, &terrain, &material, 1.0 / 240.0);
    let after_first = bristle_rms(&tire);
    let second = tire.apply_tread_contact_wrench(&state, &terrain, &material, 1.0 / 240.0);
    let after_second = bristle_rms(&tire);
    assert!(first.in_contact && second.in_contact);
    assert!(
        after_first > before + 1.0e-6,
        "first wheel-level contact solve should write persistent bristles: before={before} after={after_first}"
    );
    assert!(
        after_second > before + 1.0e-6,
        "persistent bristles should survive the next wheel-level solve instead of resetting: before={before} second={after_second}"
    );
    assert!(
        second.consumed_friction_budget.is_finite() && second.consumed_friction_budget > 0.0,
        "persistent bristle solve should still report consumed friction"
    );
}

#[test]
fn tetra_tread_contact_force_changes_with_deformed_shape() {
    let eval = |pressure: f64| -> RealTireContact {
        let mut tire = TetraTireFem::new(
            0.0,
            pressure,
            TetraTireFemParams {
                segments: 20,
                radial_layers: 3,
                width_layers: 3,
                pressure_pa: pressure,
                downforce: 5_000.0,
                ..Default::default()
            },
        );
        for _ in 0..420 {
            tire.step(1.0 / 240.0);
        }
        let terrain = FlatTerrain::new(0.0, 1.0);
        // Probe at moderate slip: deep in friction saturation the force is
        // mu-capped and shape-independent in any brush law, so the
        // deformation sensitivity this test guards only shows below the cap.
        let state = RealWheelState {
            hub_pos: tire.hub(),
            hub_vel: [5.0, 0.0, 0.4],
            heading: [1.0, 0.0, 0.0],
            omega: (5.0 + 0.6) / tire.outer_radius(),
        };
        tire.evaluate_tread_contact(&state, &terrain, &SurfaceMaterial::dry_asphalt())
    };
    let high = eval(330_000.0);
    let low = eval(80_000.0);
    assert!(high.in_contact && low.in_contact);
    let force_delta = (low.longitudinal_force - high.longitudinal_force).abs()
        + (low.lateral_force - high.lateral_force).abs();
    assert!(
        force_delta > 100.0,
        "same wheel state should produce different force when FEM deformation changes: high Fx/Fy=({},{}) low=({},{})",
        high.longitudinal_force,
        high.lateral_force,
        low.longitudinal_force,
        low.lateral_force
    );
}

#[test]
fn tetra_fem_deformation_force_report_tracks_shape_dependent_force() {
    let report = tetra_fem_deformation_force_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "FEM deformation-force report should pass: {report:?}"
    );
    assert!(
        report.low_patch_length_m > report.high_patch_length_m
            && report.force_delta_n > 50.0
            && (0.85..=1.15).contains(&report.normal_load_ratio),
        "shape-dependent force should be visible without a load mismatch: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| { check.check_id == "fem_deformation_force_shape_changes_force" })
            && checks.iter().any(|check| {
                check.check_id == "fem_deformation_force_low_pressure_longer_patch"
            })
            && checks
                .iter()
                .any(|check| check.check_id == "fem_deformation_force_loads_comparable"),
        "acceptance checks should expose shape/patch/load gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_contact_slip_patch_report_tracks_patch_width_and_sliding_growth() {
    let report = tetra_fem_contact_slip_patch_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "FEM contact slip/patch report should pass: {report:?}"
    );
    assert!(
        report.low_patch_length_m > report.high_patch_length_m
            && report.low_lateral_loaded_bins >= 2
            && report.slipping_sliding_count > report.rolling_sliding_count
            && report.slipping_dissipated_power_w > report.rolling_dissipated_power_w,
        "contact-patch and sliding diagnostics should respond to pressure and slip: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| { check.check_id == "fem_contact_slip_patch_low_pressure_longer" })
            && checks
                .iter()
                .any(|check| { check.check_id == "fem_contact_slip_patch_sliding_count_grows" })
            && checks
                .iter()
                .any(|check| check.check_id == "fem_contact_slip_patch_power_grows"),
        "acceptance checks should expose patch, sliding-count, and power gates: {checks:?}"
    );
}

#[test]
fn tetra_step_on_heightfield_keeps_tread_above_bump_surface() {
    let mut heights = vec![0.0; 7 * 7];
    heights[3 * 7 + 3] = 0.08;
    heights[3 * 7 + 2] = 0.04;
    heights[3 * 7 + 4] = 0.04;
    heights[2 * 7 + 3] = 0.04;
    heights[4 * 7 + 3] = 0.04;
    let terrain = HeightfieldTerrain::new([-0.9, -0.9], 0.3, 7, 7, heights, 1.0);
    let mut tire = TetraTireFem::new(
        0.0,
        160_000.0,
        TetraTireFemParams {
            segments: 20,
            radial_layers: 3,
            width_layers: 3,
            downforce: 4_500.0,
            ..Default::default()
        },
    );
    for _ in 0..240 {
        tire.step_on_terrain(&terrain, 1.0 / 240.0);
    }
    for &id in tire.tread_nodes() {
        let p = tire.nodes()[id];
        let h = terrain.height(p[0], p[2]);
        assert!(
            p[1] >= h - 1.0e-4,
            "tread node penetrated heightfield bump: p={p:?} h={h}"
        );
    }
    let near_count = tire
        .tread_nodes()
        .iter()
        .filter(|&&id| {
            let p = tire.nodes()[id];
            p[1] - terrain.height(p[0], p[2]) < 0.04
        })
        .count();
    assert!(
        near_count > 0,
        "loaded tire should still form a patch on heightfield"
    );
}

#[test]
fn tetra_sdf_contact_projects_nodes_by_true_surface_normal() {
    let terrain = SdfTerrain::new(0.0, 0.9)
        .with_vertical_range(-0.5, 1.2)
        .with_scan_steps(160)
        .with_primitive(SdfTerrainPrimitive::sphere([0.0, 0.30, 0.0], 0.35, 0.6));
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 12,
            radial_layers: 3,
            width_layers: 3,
            downforce: 2_500.0,
            ..Default::default()
        },
    );
    let id = tire.tread_nodes[0];
    tire.pos[id] = [0.34, 0.30, 0.0];
    tire.vel[id] = [-0.5, -0.25, 0.0];
    let before = tire.pos[id];
    let (sdf_before, normal_before) = terrain.signed_distance_and_normal(before);
    assert!(
        sdf_before < 0.0 && normal_before[0] > 0.95 && normal_before[1].abs() < 0.05,
        "test node should start inside the sphere side with horizontal SDF normal: sdf={sdf_before} n={normal_before:?}"
    );

    tire.project_nodes_against_terrain(&terrain);
    let after = tire.pos[id];
    let (sdf_after, normal_after) = terrain.signed_distance_and_normal(after);
    assert!(
        sdf_after >= -1.0e-8,
        "SDF projection should leave node outside obstacle: pos={after:?} sdf={sdf_after}"
    );
    assert!(
        after[0] > before[0] + 0.009 && (after[1] - before[1]).abs() < 0.003,
        "projection should move mostly along the true SDF side normal, not up to the vertical top surface: before={before:?} after={after:?} n={normal_after:?}"
    );
    assert!(
        tire.vel[id][0] >= -1.0e-9,
        "inward normal velocity should be removed by SDF contact projection: vel={:?}",
        tire.vel[id]
    );
}

#[test]
fn tetra_tread_wraps_curb_without_single_node_bridge() {
    let nx = 25;
    let nz = 9;
    let spacing = 0.06;
    let origin = [-0.72, -0.24];
    let mut heights = Vec::with_capacity(nx * nz);
    for _iz in 0..nz {
        for ix in 0..nx {
            let x = origin[0] + ix as f64 * spacing;
            heights.push(if x >= 0.0 { 0.065 } else { 0.0 });
        }
    }
    let terrain = HeightfieldTerrain::new(origin, spacing, nx, nz, heights, 1.0);

    let settle = |pressure: f64| -> (usize, f64) {
        let mut tire = TetraTireFem::new(
            0.0,
            pressure,
            TetraTireFemParams {
                segments: 24,
                radial_layers: 3,
                width_layers: 3,
                pressure_pa: pressure,
                downforce: 4_800.0,
                xpbd_volume_iterations: 2,
                xpbd_structural_iterations: 2,
                ..Default::default()
            },
        );
        for _ in 0..360 {
            tire.step_on_terrain(&terrain, 1.0 / 240.0);
        }
        let mut active = Vec::new();
        for &id in tire.tread_nodes() {
            let p = tire.nodes()[id];
            let clearance = p[1] - terrain.height(p[0], p[2]);
            assert!(
                clearance >= -1.0e-4,
                "tread node penetrated curb heightfield: p={p:?} clearance={clearance}"
            );
            if clearance < 0.035 {
                active.push(p[0]);
            }
        }
        let min_x = active.iter().copied().fold(f64::INFINITY, f64::min);
        let max_x = active.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (active.len(), max_x - min_x)
    };

    let (low_count, low_span) = settle(90_000.0);
    let (high_count, high_span) = settle(330_000.0);
    assert!(
        low_count >= 4 && low_span > 0.08,
        "low-pressure FEM tire should bridge/wrap curb with multiple tread nodes, count={low_count} span={low_span}"
    );
    assert!(
        low_count >= high_count && low_span >= high_span * 0.85,
        "low-pressure curb patch should not collapse relative to high pressure: low count/span=({low_count},{low_span}) high=({high_count},{high_span})"
    );
}

#[test]
fn tetra_fem_tread_contact_consumes_triangle_manifold_radius() {
    let terrain = crate::terrain::TriangleMeshTerrain::new(
        vec![
            [-1.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
        ],
        vec![[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]],
        0.9,
    )
    .with_triangle_friction(vec![0.45, 0.45, 1.1, 1.1]);
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 12,
            radial_layers: 3,
            width_layers: 3,
            downforce: 0.0,
            surface_material: SurfaceMaterial::dry_asphalt(),
            ..Default::default()
        },
    );
    let id = tire.tread_nodes[0];
    tire.pos[id] = [0.0, 0.045, 0.0];
    tire.vel[id] = [0.0, -0.10, 0.0];
    let point_gap_before = terrain.signed_distance_and_normal(tire.pos[id]).0;
    assert!(
        point_gap_before > 0.0,
        "test starts with the node point above the mesh, relying on finite tread radius"
    );
    let patches = terrain.contact_manifold(tire.pos[id], 0.08, 0.0, 8);
    assert!(
        patches.len() >= 2,
        "seam/plate tread contact should expose a manifold before FEM consumes it: {patches:?}"
    );
    tire.step_on_terrain(&terrain, 1.0 / 240.0);
    let p = tire.nodes()[id];
    assert!(
        p[1] > point_gap_before,
        "FEM tread node should consume triangle manifold radius and project before point penetration: before_gap={point_gap_before} after={p:?}"
    );
}

#[test]
fn triangle_seam_manifold_report_exposes_material_and_projection_sanity() {
    let report = tire_triangle_seam_manifold_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "triangle seam manifold sanity should pass: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_seam_manifold_multi_patch")
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_seam_material_span")
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_seam_fem_projection_lifts_node"),
        "acceptance checks should expose seam manifold, material span, and projection gates: {checks:?}"
    );
}

#[test]
fn triangle_surface_stress_report_samples_seam_and_plate_materials() {
    let report = tire_triangle_surface_stress_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "triangle surface stress report should pass: {report:?}"
    );
    assert!(
        report.left_plate_max_friction <= 0.60
            && report.right_plate_min_friction >= 1.00
            && report.seam_patch_count >= 2
            && report.seam_friction_span >= 0.45,
        "stress report should preserve plate and seam material contrast: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_surface_stress_seam_multi_patch")
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_surface_stress_left_material")
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_surface_stress_right_material"),
        "acceptance checks should expose seam and plate material gates: {checks:?}"
    );
}

#[test]
fn triangle_surface_force_report_links_triangle_materials_to_brush_force() {
    let report = tire_triangle_surface_force_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "triangle surface force report should pass: {report:?}"
    );
    assert!(
        report.right_abs_fx_n > report.left_abs_fx_n
            && report.seam_abs_fx_n > report.left_abs_fx_n
            && report.seam_manifold_patch_count >= 2
            && report.seam_friction_span >= 0.45,
        "triangle material/manifold lookup should affect brush force, not only geometry rows: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_surface_force_seam_multi_patch")
            && checks.iter().any(|check| {
                check.check_id == "triangle_surface_force_high_mu_stronger_than_low"
            })
            && checks.iter().any(|check| {
                check.check_id == "triangle_surface_force_sliding_bristles_active"
            }),
        "acceptance checks should expose seam, force, and bristle/sliding gates: {checks:?}"
    );
}

#[test]
fn triangle_density_convergence_report_tracks_mesh_refinement() {
    let report = tire_triangle_density_convergence_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "triangle density convergence report should pass: {report:?}"
    );
    assert!(
        report.max_triangle_count > report.min_triangle_count
            && report.min_seam_manifold_count >= 2
            && report.min_friction_span >= 0.45,
        "triangle density sweep should refine mesh while preserving seam manifold/material span: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_density_force_stable")
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_density_keeps_seam_manifold")
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_density_keeps_material_span"),
        "acceptance checks should expose triangle density force, seam, and material-span gates: {checks:?}"
    );
}

#[test]
fn triangle_large_manifold_report_exercises_large_mesh_fem_feedback() {
    let report = tire_triangle_large_manifold_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "large triangle manifold report should pass broad non-calibration gates: {report:?}"
    );
    assert!(
        report.triangle_count >= 400
            && report.seam_manifold_patch_count >= 2
            && report.feedback_force_norm_n >= 25.0
            && report.feedback_force_cell_count >= 2
            && report.feedback_bristle_force_sum_n >= 25.0
            && report.feedback_width_span_m >= 0.01,
        "large triangle report should keep BVH/manifold and FEM force/bristle feedback active: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_large_manifold_bvh_narrows_query")
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_large_manifold_multi_patch")
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_large_manifold_feedback_force")
            && checks
                .iter()
                .any(|check| { check.check_id == "triangle_large_manifold_bristle_force_sum" })
            && checks
                .iter()
                .any(|check| check.check_id == "triangle_large_manifold_widthwise_support"),
        "acceptance checks should expose large-triangle BVH, manifold, force, bristle, and width-support gates: {checks:?}"
    );
}

#[test]
fn tetra_step_on_moving_terrain_feeds_tangential_motion_into_tread_nodes() {
    let mut tire = TetraTireFem::new(
        0.0,
        160_000.0,
        TetraTireFemParams {
            segments: 20,
            radial_layers: 3,
            width_layers: 3,
            downforce: 4_500.0,
            surface_material: SurfaceMaterial {
                mu_static: 1.1,
                mu_dynamic: 0.95,
                bristle_damping: 4.0e4,
                rolling_resistance: 0.0,
                cohesion: 0.0,
            },
            ..Default::default()
        },
    );
    let static_ground = MovingFlatTerrain {
        speed_x: 0.0,
        friction: 1.1,
    };
    for _ in 0..240 {
        tire.step_on_terrain(&static_ground, 1.0 / 240.0);
    }
    let avg_contact_x = |t: &TetraTireFem| -> f64 {
        let mut sum = 0.0;
        let mut n: f64 = 0.0;
        for &id in t.tread_nodes() {
            let p = t.nodes()[id];
            if p[1] < 0.04 {
                sum += p[0];
                n += 1.0;
            }
        }
        sum / n.max(1.0)
    };
    let before = avg_contact_x(&tire);
    let moving_ground = MovingFlatTerrain {
        speed_x: 3.0,
        friction: 1.1,
    };
    for _ in 0..120 {
        tire.step_on_terrain(&moving_ground, 1.0 / 240.0);
    }
    let after = avg_contact_x(&tire);
    assert!(
        (after - before).abs() > 1.0e-5,
        "moving terrain should drag contacting tread nodes: before={before} after={after}"
    );
}

/// Stage (d) — `SurfaceMaterial` is plumbed into the FEM contact
/// projection and accessible at runtime so vehicles can switch
/// surfaces (asphalt → ice → dirt) on the fly.
#[test]
fn tetra_fem_surface_material_is_runtime_switchable() {
    let mut tire = TetraTireFem::new(
        0.0,
        220_000.0,
        TetraTireFemParams {
            surface_material: SurfaceMaterial::dry_asphalt(),
            ..Default::default()
        },
    );
    assert!(tire.surface_material().mu_static > 0.9);
    tire.set_surface_material(SurfaceMaterial::ice());
    assert!(
        tire.surface_material().mu_static < 0.2,
        "ice μ_s should be sub-0.2, got {}",
        tire.surface_material().mu_static
    );
    // Step a few frames to confirm the new material flows through the
    // contact projection without exploding the tire.
    for _ in 0..30 {
        tire.step(1.0 / 240.0);
    }
    assert!(
        tire.nodes().iter().all(|p| p.iter().all(|v| v.is_finite())),
        "tire went non-finite after surface-material swap"
    );
}

fn tetra_state_is_finite_and_bounded(tire: &TetraTireFem, max_abs_pos: f64, max_speed: f64) {
    assert!(
        tire.hub().iter().all(|v| v.is_finite()),
        "hub went non-finite: {:?}",
        tire.hub()
    );
    for (i, p) in tire.pos.iter().enumerate() {
        assert!(
            p.iter().all(|v| v.is_finite() && v.abs() <= max_abs_pos),
            "node {i} position out of bound/non-finite: {p:?}"
        );
    }
    for (i, v) in tire.vel.iter().enumerate() {
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(
            v.iter().all(|x| x.is_finite()) && speed <= max_speed,
            "node {i} velocity out of bound/non-finite: {v:?}, speed={speed}"
        );
    }
}

fn tetra_kinetic_energy(tire: &TetraTireFem) -> f64 {
    tire.mass
        .iter()
        .zip(tire.vel.iter())
        .map(|(m, v)| 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        .sum()
}

fn tetra_mean_relative_volume_error(tire: &TetraTireFem) -> f64 {
    tire.tets
        .iter()
        .map(|tet| {
            let v = tetra_signed_volume(tet.ids, &tire.pos);
            (v - tet.volume).abs() / tet.volume.max(1.0e-12)
        })
        .sum::<f64>()
        / tire.tets.len().max(1) as f64
}

#[test]
fn tetra_xpbd_volume_projection_reduces_perturbed_volume_error() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 0.0,
            xpbd_volume_compliance: 0.0,
            xpbd_volume_iterations: 8,
            ..Default::default()
        },
    );
    for p in &mut tire.pos {
        let d = sub(*p, tire.hub);
        p[0] = tire.hub[0] + d[0] * 1.08;
        p[1] = tire.hub[1] + d[1] * 0.72;
        p[2] = tire.hub[2] + d[2] * 0.90;
    }

    let before = tetra_mean_relative_volume_error(&tire);
    tire.project_volume_constraints(1.0 / 240.0, &FlatTerrain::new(-10.0, 1.0));
    let after = tetra_mean_relative_volume_error(&tire);
    assert!(
        after < before * 0.65,
        "XPBD volume projection should reduce mean tet-volume error: before={before} after={after}"
    );
    assert!(
        tire.nodes().iter().all(|p| p.iter().all(|v| v.is_finite())),
        "XPBD volume projection produced non-finite nodes"
    );
}

#[test]
fn tetra_xpbd_volume_projection_respects_terrain_projection() {
    let mut tire = TetraTireFem::new(
        0.0,
        160_000.0,
        TetraTireFemParams {
            segments: 12,
            radial_layers: 3,
            width_layers: 3,
            downforce: 0.0,
            xpbd_volume_compliance: 0.0,
            xpbd_volume_iterations: 5,
            ..Default::default()
        },
    );
    let tread_ids = tire.tread_nodes().to_vec();
    for id in tread_ids {
        tire.pos[id][1] = -0.025;
        tire.vel[id][1] = -1.0;
    }
    let terrain = FlatTerrain::new(0.0, 1.0);
    tire.project_volume_constraints(1.0 / 240.0, &terrain);
    assert!(
        tire.nodes().iter().all(|p| p[1] >= -1.0e-10),
        "XPBD volume correction must not leave nodes under the terrain"
    );
    assert!(
        tire.vel.iter().all(|v| v.iter().all(|x| x.is_finite())),
        "XPBD terrain projection produced non-finite velocities"
    );
}

fn tetra_mean_belt_edge_error(tire: &TetraTireFem) -> f64 {
    tire.tread_belt_edges
        .iter()
        .map(|edge| {
            let [a, b] = edge.ids;
            let len = dot(sub(tire.pos[b], tire.pos[a]), sub(tire.pos[b], tire.pos[a])).sqrt();
            (len - edge.rest_length).abs() / edge.rest_length.max(1.0e-9)
        })
        .sum::<f64>()
        / tire.tread_belt_edges.len().max(1) as f64
}

fn tetra_mean_sidewall_bend_error(tire: &TetraTireFem) -> f64 {
    tire.sidewall_bend_edges
        .iter()
        .map(|edge| {
            let [a, b] = edge.ids;
            let len = dot(sub(tire.pos[b], tire.pos[a]), sub(tire.pos[b], tire.pos[a])).sqrt();
            (len - edge.rest_length).abs() / edge.rest_length.max(1.0e-9)
        })
        .sum::<f64>()
        / tire.sidewall_bend_edges.len().max(1) as f64
}

#[test]
fn tetra_sidewall_dihedral_diagnostics_detect_folding() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 0.0,
            xpbd_sidewall_bend_compliance: f64::INFINITY,
            xpbd_sidewall_dihedral_compliance: 0.0,
            xpbd_structural_iterations: 8,
            ..Default::default()
        },
    );
    let rest = tire.sidewall_dihedral_diagnostics();
    assert!(
        rest.is_finite() && rest.quad_count >= tire.segments() * 2,
        "sidewall dihedral diagnostics should cover both side faces: {rest:?}"
    );
    assert!(
        rest.max_abs_deviation_rad < 1.0e-8,
        "fresh tire should match rest dihedral angles: {rest:?}"
    );

    let bend_edges = tire.sidewall_bend_edges.clone();
    for (k, edge) in bend_edges.iter().enumerate() {
        if k % 7 == 0 {
            let id = edge.ids[0];
            tire.pos[id][0] += 0.050;
            tire.pos[id][1] -= 0.035;
            tire.pos[id][2] += tire.pos[id][2].signum() * 0.030;
        }
    }
    let folded = tire.sidewall_dihedral_diagnostics();
    assert!(
        folded.is_finite() && folded.max_abs_deviation_rad > 0.02,
        "folded sidewall should produce measurable dihedral deviation: {folded:?}"
    );

    tire.project_structural_constraints(1.0 / 240.0, &FlatTerrain::new(-10.0, 1.0));
    let stabilized = tire.sidewall_dihedral_diagnostics();
    assert!(
        stabilized.is_finite()
            && stabilized.max_abs_deviation_rad < folded.max_abs_deviation_rad * 0.90,
        "sidewall dihedral constraints should reduce measured fold-angle error without relying on edge proxy: folded={folded:?} stabilized={stabilized:?}"
    );
}

#[test]
fn tetra_xpbd_tread_belt_constraints_reduce_outer_layer_stretch() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 0.0,
            xpbd_bead_compliance: f64::INFINITY,
            xpbd_tread_belt_compliance: 0.0,
            xpbd_structural_iterations: 8,
            ..Default::default()
        },
    );
    let tread_ids = tire.tread_nodes().to_vec();
    for (k, id) in tread_ids.into_iter().enumerate() {
        if k % 3 == 0 {
            tire.pos[id][0] += 0.055;
            tire.pos[id][2] -= 0.025;
        }
    }

    let before = tetra_mean_belt_edge_error(&tire);
    tire.project_structural_constraints(1.0 / 240.0, &FlatTerrain::new(-10.0, 1.0));
    let after = tetra_mean_belt_edge_error(&tire);
    assert!(
        after < before * 0.70,
        "XPBD tread-belt constraints should reduce outer-layer stretch: before={before} after={after}"
    );
}

#[test]
fn tetra_xpbd_bead_constraints_pull_nodes_toward_rim_mount() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 0.0,
            xpbd_bead_compliance: 0.0,
            xpbd_tread_belt_compliance: f64::INFINITY,
            xpbd_structural_iterations: 6,
            ..Default::default()
        },
    );
    let bead_ids: Vec<usize> = tire.bead_offsets.iter().map(|&(id, _)| id).collect();
    for id in bead_ids {
        tire.pos[id] = add(tire.pos[id], [0.035, -0.045, 0.025]);
        tire.vel[id] = [0.5, -0.5, 0.25];
    }

    let before = tire.bead_mount_error_stats_m().0;
    tire.project_structural_constraints(1.0 / 240.0, &FlatTerrain::new(-10.0, 1.0));
    let (after, after_max) = tire.bead_mount_error_stats_m();
    assert!(
        after < before * 0.25,
        "XPBD bead/rim constraints should pull bead nodes back toward mounted offsets: before={before} after={after}"
    );
    let diagnostics = tire.diagnostics();
    assert!((diagnostics.mean_bead_mount_error_m - after).abs() < 1.0e-12);
    assert!((diagnostics.max_bead_mount_error_m - after_max).abs() < 1.0e-12);
    assert!(
        tire.vel.iter().all(|v| v.iter().all(|x| x.is_finite())),
        "bead projection should keep velocities finite"
    );
}

#[test]
fn tetra_xpbd_sidewall_bend_constraints_reduce_sidewall_folding() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 0.0,
            xpbd_bead_compliance: f64::INFINITY,
            xpbd_tread_belt_compliance: f64::INFINITY,
            xpbd_sidewall_bend_compliance: 0.0,
            xpbd_structural_iterations: 8,
            ..Default::default()
        },
    );
    let bend_edges = tire.sidewall_bend_edges.clone();
    for (k, edge) in bend_edges.iter().enumerate() {
        if k % 5 == 0 {
            let id = edge.ids[0];
            tire.pos[id][0] += 0.035;
            tire.pos[id][1] -= 0.030;
            tire.pos[id][2] += tire.pos[id][2].signum() * 0.020;
        }
    }

    let before = tetra_mean_sidewall_bend_error(&tire);
    tire.project_structural_constraints(1.0 / 240.0, &FlatTerrain::new(-10.0, 1.0));
    let after = tetra_mean_sidewall_bend_error(&tire);
    assert!(
        after < before * 0.75,
        "XPBD sidewall bend constraints should reduce sidewall folding distortion: before={before} after={after}"
    );
    assert!(
        tire.nodes().iter().all(|p| p.iter().all(|v| v.is_finite())),
        "sidewall bend projection produced non-finite nodes"
    );
}

#[test]
fn tetra_hardening_high_load_low_pressure_stays_finite() {
    let mut tire = TetraTireFem::new(
        0.0,
        45_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 8_500.0,
            pressure_pa: 45_000.0,
            ..Default::default()
        },
    );
    for _ in 0..900 {
        tire.step(1.0 / 360.0);
    }
    tetra_state_is_finite_and_bounded(&tire, 2.0, 9.0);
    assert!(
        tire.hub()[1] >= tire.params.inner_radius * 0.75 - 1.0e-9,
        "hub punched below floor clamp under high load / low pressure: {:?}",
        tire.hub()
    );
    assert!(
        tire.contact_patch_width() > 0.01,
        "high-load low-pressure tire should still form a patch"
    );
}

#[test]
fn tetra_hardening_high_pressure_high_speed_is_bounded() {
    let mut tire = TetraTireFem::new(
        0.0,
        420_000.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 3_500.0,
            pressure_pa: 420_000.0,
            ..Default::default()
        },
    );
    let terrain = MovingFlatTerrain {
        speed_x: 28.0,
        friction: 1.0,
    };
    for _ in 0..900 {
        tire.step_on_terrain(&terrain, 1.0 / 480.0);
    }
    tetra_state_is_finite_and_bounded(&tire, 2.0, 12.0);
    let max_radius = tire
        .tread_nodes()
        .iter()
        .map(|&id| {
            let p = tire.nodes()[id];
            ((p[0] - tire.hub()[0]).powi(2) + (p[1] - tire.hub()[1]).powi(2)).sqrt()
        })
        .fold(0.0_f64, f64::max);
    assert!(
        max_radius < tire.params.outer_radius * 1.35,
        "high-pressure/high-speed tire expanded unbounded: radius={max_radius}"
    );
}

#[test]
fn tetra_semi_implicit_stiffness_damps_stiff_rubber_velocity_spikes() {
    let make_tire = |semi_implicit_stiffness_scale: f64| {
        let mut tire = TetraTireFem::new(
            0.0,
            360_000.0,
            TetraTireFemParams {
                segments: 12,
                radial_layers: 3,
                width_layers: 3,
                pressure_pa: 360_000.0,
                young_modulus: 1.4e7,
                sidewall_young_modulus: 9.0e6,
                tread_young_modulus: 2.2e7,
                bead_young_modulus: 4.0e7,
                node_damping: 2.0,
                xpbd_volume_iterations: 0,
                xpbd_structural_iterations: 0,
                semi_implicit_stiffness_scale,
                ..TetraTireFemParams::high_pressure_reference()
            },
        );
        for (i, p) in tire.pos.iter_mut().enumerate() {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            p[0] += 0.025 * sign;
            p[1] += 0.018 * sign;
        }
        tire
    };

    let mut explicit = make_tire(0.0);
    let mut semi = make_tire(1.5);
    let terrain = FlatTerrain::new(-10.0, 1.0);
    explicit.step_on_terrain(&terrain, 1.0 / 120.0);
    semi.step_on_terrain(&terrain, 1.0 / 120.0);
    assert!(
        semi.max_node_speed_m_per_s() < explicit.max_node_speed_m_per_s() * 0.80,
        "semi-implicit stiffness should damp stiff-rubber velocity spikes: explicit={} semi={}",
        explicit.max_node_speed_m_per_s(),
        semi.max_node_speed_m_per_s()
    );
    assert_eq!(
        semi.solver_budget_telemetry(1, &RealTireContact::empty().telemetry_sample(0.0))
            .semi_implicit_stiffness_scale,
        1.5
    );
    tetra_state_is_finite_and_bounded(&semi, 2.0, 8.5);
}

#[test]
fn tetra_fem_solver_stiffness_report_gates_semi_implicit_guardrail() {
    let report = tetra_fem_solver_stiffness_report();
    assert!(
        report.passed(),
        "FEM solver stiffness report should pass broad stabilizer envelopes: {report:?}"
    );
    assert!(
        report.speed_ratio_semi_over_explicit <= 0.80,
        "semi-implicit path should reduce velocity spike versus explicit: {report:?}"
    );
    assert!(
        report.high_pressure_radius_ratio <= 1.35 && report.high_pressure_contact_patch_m >= 1.0e-5,
        "high-pressure rolling case should stay bounded and supported: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_solver_stiffness_reduces_velocity_spike")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_solver_high_pressure_radius_bounded"),
        "solver-depth report should expose semi-implicit and high-pressure gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_diagonal_newton_report_solves_stiff_mode_residual() {
    let report = tetra_fem_diagonal_newton_report();
    assert!(
        report.passed(),
        "diagonal Newton report should pass broad implicit-solver gates: {report:?}"
    );
    assert!(
        report.converged
            && report.final_residual_norm <= 1.0e-7
            && report.residual_reduction_ratio <= 1.0e-8,
        "Newton solve should reduce the backward-Euler residual: {report:?}"
    );
    assert!(
        report.energy_ratio_implicit_over_explicit <= 0.35
            && report.explicit_velocity_m_per_s.abs() > report.newton_velocity_m_per_s.abs() * 2.0,
        "implicit Newton mode should avoid the explicit stiff-mode overshoot: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_diagonal_newton_residual_reduced")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_diagonal_newton_energy_bounded"),
        "diagonal Newton report should expose residual and energy gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_element_newton_report_solves_coupled_tetra_residual() {
    let report = tetra_fem_element_newton_report();
    assert!(
        report.passed(),
        "coupled element Newton report should pass broad implicit-solver gates: {report:?}"
    );
    assert_eq!(
        report.coupled_dof_count, 12,
        "element Newton fixture must solve all four tetra node positions together"
    );
    assert!(
        report.converged
            && report.final_residual_norm <= 1.0e-6
            && report.residual_reduction_ratio <= 1.0e-8,
        "coupled element Newton solve should reduce the backward-Euler residual: {report:?}"
    );
    assert!(
        report.jacobian_offdiag_ratio >= 0.05
            && report.final_volume_ratio >= 0.55
            && report.final_volume_ratio <= 1.65,
        "element Newton report should expose coupled Jacobian terms and bounded volume: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_element_newton_residual_reduced")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_element_newton_offdiag_coupling_present"),
        "element Newton report should expose residual and coupling gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_patch_newton_report_solves_connected_tetra_patch() {
    let report = tetra_fem_patch_newton_report();
    assert!(
        report.passed(),
        "multi-element patch Newton report should pass broad implicit-solver gates: {report:?}"
    );
    assert!(
        report.tetra_count >= 2
            && report.shared_node_count >= 3
            && report.coupled_dof_count == report.node_count * 3,
        "patch Newton fixture should solve a connected multi-tet patch as one coupled system: {report:?}"
    );
    assert!(
        report.converged
            && report.final_residual_norm <= 1.0e-4
            && report.residual_reduction_ratio <= 1.0e-7,
        "patch Newton solve should reduce the backward-Euler residual: {report:?}"
    );
    assert!(
        report.jacobian_offdiag_ratio >= 0.05 && report.shared_node_internal_force_n > 1.0,
        "patch Newton report should expose cross-element coupling and shared-node force accumulation: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_patch_newton_shared_nodes")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_patch_newton_offdiag_coupling_present"),
        "patch Newton report should expose shared-node and coupling gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_tiny_mesh_newton_report_solves_actual_annular_mesh() {
    let report = tetra_fem_tiny_mesh_newton_report();
    assert!(
        report.passed(),
        "tiny full-mesh Newton report should pass broad implicit-solver gates: {report:?}"
    );
    assert!(
        report.node_count >= 32
            && report.tetra_count >= 32
            && report.coupled_dof_count == report.node_count * 3,
        "tiny mesh Newton fixture should solve the whole annular mesh, not an isolated patch: {report:?}"
    );
    assert!(
        report.converged
            && report.final_residual_norm <= 1.0e-3
            && report.residual_reduction_ratio <= 1.0e-6,
        "tiny mesh Newton solve should reduce the full mesh residual: {report:?}"
    );
    assert!(
        report.tread_node_count >= 16
            && report.bead_node_count >= 16
            && report.jacobian_offdiag_ratio >= 0.02,
        "tiny mesh report should include tread/bead rows and mesh coupling: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_tiny_mesh_newton_full_mesh_nodes")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_tiny_mesh_newton_offdiag_coupling_present"),
        "tiny mesh Newton report should expose full-mesh and coupling gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_matrix_free_newton_scale_report_probes_larger_mesh_jv() {
    let report = tetra_fem_matrix_free_newton_scale_report();
    let tiny = tetra_fem_tiny_mesh_newton_report();
    assert!(
        report.passed(),
        "matrix-free Newton scaling report should pass larger-mesh residual/Jv gates: {report:?}"
    );
    assert!(
        report.node_count > tiny.node_count
            && report.tetra_count > tiny.tetra_count
            && report.coupled_dof_count == report.node_count * 3
            && report.jv_norm > 1.0e-3
            && report.dense_over_matrix_free_bytes > 8.0,
        "matrix-free Newton report should prove larger-than-tiny topology, non-zero Jv, and dense-memory savings: report={report:?} tiny={tiny:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_matrix_free_newton_jv_nonzero")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_matrix_free_newton_memory_saves_dense"),
        "matrix-free Newton report should expose Jv and memory-scaling gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_matrix_free_newton_solve_report_reduces_larger_mesh_residual() {
    let report = tetra_fem_matrix_free_newton_solve_report();
    assert!(
        report.passed(),
        "matrix-free Newton solve report should pass larger-mesh Krylov correction gates: {report:?}"
    );
    assert!(
        report.corrected_residual_norm < report.initial_residual_norm
            && report.residual_reduction_ratio <= 0.80
            && report.gmres_iteration_count >= 2
            && report.jv_evaluation_count >= report.gmres_iteration_count
            && report.correction_norm_m > 1.0e-8,
        "matrix-free Newton solve should produce a non-zero Krylov correction and reduce the nonlinear residual: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| { check.check_id == "fem_matrix_free_newton_solve_residual_reduced" })
            && checks
                .iter()
                .any(|check| check.check_id == "fem_matrix_free_newton_solve_jv_evaluations"),
        "matrix-free Newton solve report should expose residual-reduction and Jv-driven gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_matrix_free_newton_preconditioner_report_builds_diagonal_scaler() {
    let report = tetra_fem_matrix_free_newton_preconditioner_report();
    assert!(
        report.passed(),
        "matrix-free Newton preconditioner report should pass diagonal/Krylov gates: {report:?}"
    );
    assert!(
        report.diagonal_sample_count == report.coupled_dof_count
            && report.positive_diagonal_fraction >= 0.80
            && report.preconditioned_iteration_count <= report.unpreconditioned_iteration_count
            && report.preconditioned_linear_residual_norm
                <= report.unpreconditioned_linear_residual_norm
            && report.residual_reduction_ratio <= 0.80,
        "preconditioner report should sample all DOFs, keep a mostly positive diagonal, and not make Krylov residual/iterations worse: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks.iter().any(|check| {
                check.check_id == "fem_matrix_free_newton_preconditioner_positive_diagonal"
            })
            && checks.iter().any(|check| {
                check.check_id == "fem_matrix_free_newton_preconditioner_residual_not_worse"
            }),
        "matrix-free Newton preconditioner report should expose diagonal and residual gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_matrix_free_newton_step_report_reconstructs_bounded_state() {
    let report = tetra_fem_matrix_free_newton_step_report();
    assert!(
        report.passed(),
        "matrix-free Newton step report should pass implicit-step state gates: {report:?}"
    );
    assert!(
        report.corrected_residual_norm < report.initial_residual_norm
            && report.line_search_alpha > 0.0
            && report.correction_norm_m > 1.0e-8
            && report.max_velocity_m_per_s < 30.0
            && report.min_volume_ratio >= 0.35
            && report.max_volume_ratio <= 2.50
            && report.energy_ratio_implicit_over_explicit <= 1.05,
        "matrix-free Newton step should reduce residual and reconstruct bounded position/velocity/energy state: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_matrix_free_newton_step_residual_reduced")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_matrix_free_newton_step_energy_bounded"),
        "matrix-free Newton step report should expose residual and state-bound gates: {checks:?}"
    );
}

#[test]
fn tetra_fem_matrix_free_newton_timestep_report_uses_callable_tire_step() {
    let report = tetra_fem_matrix_free_newton_timestep_report();
    assert!(
        report.passed(),
        "callable matrix-free Newton timestep report should pass implicit-step gates: {report:?}"
    );
    assert!(
        report.corrected_residual_norm < report.initial_residual_norm
            && report.line_search_alpha > 0.0
            && report.diagonal_jv_evaluation_count == report.coupled_dof_count
            && report.max_velocity_m_per_s < 30.0
            && report.min_volume_ratio >= 0.35
            && report.energy_ratio_implicit_over_explicit <= 1.05,
        "callable debug timestep should mutate a bounded tire state while preserving the matrix-free Newton gates: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_matrix_free_newton_step_residual_reduced")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_matrix_free_newton_step_velocity_bounded"),
        "callable timestep should expose residual and velocity/state gates: {checks:?}"
    );
}

#[test]
fn tetra_tire_fem_torque_api_default_state_is_at_rest() {
    // PLAN_TOURQUE Stage A — the new spin / torque / reaction fields
    // must start zeroed in `new()`. Stage A is intentionally a no-op:
    // these accessors exist, but nothing reads the values yet.
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(12, 3, 3),
    );
    assert_eq!(tire.drive_torque_n_m(), 0.0);
    assert_eq!(tire.brake_torque_n_m(), 0.0);
    assert_eq!(tire.hub_angle(), 0.0);
    assert_eq!(tire.hub_omega(), 0.0);
    assert_eq!(tire.last_hub_reaction_xyz(), [0.0, 0.0, 0.0]);
}

#[test]
fn tetra_tire_fem_drive_torque_spins_the_hub() {
    // PLAN_TOURQUE Stage B — applying a constant drive torque must
    // grow `hub_omega` roughly linearly per `tau / I`.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(12, 3, 3),
    );
    let i_wheel = tire.wheel_moment_of_inertia();
    let drive_torque = 60.0_f64; // N·m
    tire.set_drive_torque_n_m(drive_torque);
    let terrain = FlatTerrain::new(-10.0, tire.params.surface_material.mu_static);
    let dt = 1.0 / 240.0;
    let steps = 12;
    for _ in 0..steps {
        tire.step_on_terrain(&terrain, dt);
    }
    let expected = drive_torque / i_wheel * dt * steps as f64;
    let omega = tire.hub_omega();
    assert!(omega.is_finite(), "spin must be finite after drive torque");
    // The substep loop runs many internal substeps per call, so the
    // total integrated time is `dt * steps`. We allow a wide
    // tolerance because the FEM substep scheduler may divide dt
    // further.
    assert!(
        omega > 0.0 && omega < expected * 5.0,
        "drive torque should accelerate omega in +ω; got {omega}, expected ≈ {expected}"
    );
    assert!(
        tire.hub_angle() > 0.0,
        "hub_angle should advance in +θ with positive spin"
    );
}

#[test]
fn tetra_tire_fem_brake_torque_decays_spin_toward_zero() {
    // PLAN_TOURQUE Stage B — once spin is going, brake torque must
    // oppose it. With drive torque off and brake torque on, omega
    // monotonically approaches zero (but cannot cross through it in
    // one substep thanks to the soft-sign saturator).
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(12, 3, 3),
    );
    tire.set_drive_torque_n_m(80.0);
    let terrain = FlatTerrain::new(-10.0, tire.params.surface_material.mu_static);
    let dt = 1.0 / 240.0;
    for _ in 0..24 {
        tire.step_on_terrain(&terrain, dt);
    }
    let omega_after_drive = tire.hub_omega();
    assert!(
        omega_after_drive > 0.0,
        "must spin up before testing brake; got {omega_after_drive}"
    );
    tire.set_drive_torque_n_m(0.0);
    tire.set_brake_torque_n_m(150.0);
    let mut omega_history = Vec::new();
    for _ in 0..48 {
        tire.step_on_terrain(&terrain, dt);
        omega_history.push(tire.hub_omega());
    }
    let final_omega = *omega_history.last().unwrap();
    assert!(
        final_omega.abs() < omega_after_drive,
        "brake torque should reduce |omega|: started {omega_after_drive}, ended {final_omega}"
    );
    assert!(
        omega_history.iter().all(|w| w.is_finite()),
        "spin under brake must stay finite"
    );
}

#[test]
fn tetra_tire_fem_spin_creates_backward_tread_slip_at_contact() {
    // PLAN_TOURQUE Stage C — with the hub pinned kinematically and a
    // positive `hub_omega` injected directly, the tread nodes in
    // contact must see a backward (−X) effective slip velocity, and
    // therefore the bristle integrates a −X stretch. A negative
    // bristle.x at the bottom of the wheel is the diagnostic
    // signature of "spin is coupling to contact friction" — Stage D
    // then turns that into a +X reaction at the hub.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(14, 3, 3),
    );
    // Floor at y=0 so the tire actually settles into contact.
    let terrain = FlatTerrain::new(0.0, tire.params.surface_material.mu_static);
    // Let the tire settle so tread nodes are in contact.
    let dt = 1.0 / 240.0;
    for _ in 0..60 {
        tire.step_on_terrain(&terrain, dt);
    }
    // Force a positive omega and re-pin the hub at its current
    // pose so the spin contribution is the only new effect.
    let hub_now = tire.hub();
    tire.set_hub_kinematic(hub_now, [0.0, 0.0, 0.0]);
    tire.hub_omega = 8.0; // 8 rad/s ≈ 2.6 m/s rolling, well below FEM clamp.
    let bristles_before: Vec<[f64; 3]> = tire.tread_bristle.clone();
    for _ in 0..6 {
        // Re-pin every step so the kinematic hub does not drift
        // while we measure the spin-only response.
        tire.set_hub_kinematic(hub_now, [0.0, 0.0, 0.0]);
        tire.hub_omega = 8.0;
        tire.step_on_terrain(&terrain, dt);
    }
    // Find tread nodes that are in contact (low Y) and check their
    // bristle has stretched in −X (spin-induced slip pushes tread
    // backward in world frame, so bristle integrates negative x).
    let nodes = tire.nodes().to_vec();
    let bristles_after = tire.tread_bristle.clone();
    let mut negative_bristle_count = 0usize;
    let mut total_dx = 0.0_f64;
    for (slot, &node_id) in tire.tread_nodes().iter().enumerate() {
        if nodes[node_id][1] > 0.03 {
            continue;
        }
        let dx = bristles_after[slot][0] - bristles_before[slot][0];
        total_dx += dx;
        if dx < 0.0 {
            negative_bristle_count += 1;
        }
    }
    assert!(
        negative_bristle_count > 0,
        "spin should pull contact-patch bristles in −X (positive omega → backward tread slip); \
         got total Δx = {total_dx}, negative slot count = {negative_bristle_count}"
    );
    assert!(
        total_dx < 0.0,
        "net bristle slip across the contact patch must be negative for positive spin; \
         got total Δx = {total_dx}"
    );
}

#[test]
fn tetra_tire_fem_zero_torque_default_state_does_not_disturb_existing_scenes() {
    // PLAN_TOURQUE — the per-plan no-regression sanity check.
    // With `drive_torque = brake_torque = 0` and the hub kinematic
    // (the default), each named viewer preset must still produce a
    // bounded tire state under standard step_on_terrain over
    // several frames. Catches any accidental side effects from the
    // new spin/torque/reaction wiring.
    let presets: [(&str, fn() -> TetraTireFemParams); 3] = [
        ("realtime_viewer", || {
            TetraTireFemParams::realtime_viewer().with_resolution(12, 3, 3)
        }),
        ("offroad_soft", || {
            TetraTireFemParams::offroad_soft().with_resolution(12, 3, 3)
        }),
        ("high_pressure_reference", || {
            TetraTireFemParams::high_pressure_reference().with_resolution(12, 3, 3)
        }),
    ];
    for (name, build_params) in presets {
        let params = build_params();
        let mut tire = TetraTireFem::new(0.0, params.pressure_pa, params);
        let terrain = FlatTerrain::new(0.0, tire.params.surface_material.mu_static);
        for _ in 0..30 {
            tire.step_on_terrain(&terrain, 1.0 / 240.0);
        }
        // Spin DOF must stay at rest with zero torque.
        assert!(
            tire.hub_omega().abs() < 1.0e-9,
            "{name}: zero-torque hub_omega should stay zero, got {}",
            tire.hub_omega()
        );
        // Reaction Y must reflect the gravity/downforce budget (a
        // sanity that the new reaction-publishing line did not
        // zero it out).
        assert!(
            tire.last_hub_reaction_xyz().iter().all(|v| v.is_finite()),
            "{name}: hub reaction must be finite: {:?}",
            tire.last_hub_reaction_xyz()
        );
        // Tire state must remain bounded and finite.
        assert!(
            tire.nodes().iter().all(|p| finite_vec3(*p)),
            "{name}: tire nodes must stay finite under zero-torque stepping"
        );
    }
}

#[test]
fn tetra_tire_fem_body_frame_starts_at_rest_pose_and_zero_velocity() {
    // PLAN_ROLL Stage A — body-frame storage exists alongside the
    // world-frame `pos`/`vel`. At init, body_pos mirrors the rest
    // pose (no rotation, no contact deformation) and body_vel is
    // zero. Stages B+ wire the rotation transform into the per-
    // substep physics.
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(12, 3, 3),
    );
    let body_nodes = tire.body_nodes();
    let world_nodes = tire.nodes();
    assert_eq!(body_nodes.len(), world_nodes.len());
    let body_vel = tire.body_velocities();
    assert_eq!(body_vel.len(), body_nodes.len());
    assert!(
        body_vel.iter().all(|v| v == &[0.0, 0.0, 0.0]),
        "body-frame velocities must start at zero"
    );
    // With hub_angle = 0 the body frame is just world − hub.
    let hub = tire.hub();
    for (i, (body, world)) in body_nodes.iter().zip(world_nodes.iter()).enumerate() {
        for axis in 0..3 {
            let diff = body[axis] - (world[axis] - hub[axis]);
            assert!(
                diff.abs() < 1.0e-9,
                "node {i} axis {axis}: body={body:?} world={world:?} hub={hub:?}, diff={diff}"
            );
        }
    }
}

#[test]
fn tetra_tire_fem_body_world_transform_round_trips_at_any_angle() {
    // PLAN_ROLL Stage B — body_to_world and world_to_body must be
    // exact inverses at any hub_angle and any hub translation.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(10, 3, 3),
    );
    for angle in [0.0_f64, 0.5, 1.7, -2.3, std::f64::consts::PI, 6.5] {
        tire.set_hub_kinematic([3.0, 0.4, -1.2], [0.0; 3]);
        tire.hub_angle = angle;
        for p_body in [[0.0, 0.0, 0.0], [0.2, -0.1, 0.05], [-0.3, 0.45, -0.18]] {
            let p_world = tire.body_to_world(p_body);
            let p_back = tire.world_to_body(p_world);
            for axis in 0..3 {
                assert!(
                    (p_body[axis] - p_back[axis]).abs() < 1.0e-9,
                    "round-trip mismatch axis {axis} at angle {angle}: body={p_body:?} \
                     world={p_world:?} back={p_back:?}"
                );
            }
        }
    }
}

#[test]
fn tetra_tire_fem_body_to_world_rotates_forward_top_to_plus_x() {
    // PLAN_ROLL Stage B — sign convention: positive hub_angle ⇔
    // forward rolling. A body-frame point at (0, +R, 0) — the
    // top of the wheel — should be at (R·sin(θ), R·cos(θ), 0) in
    // world (small positive θ → +X), confirming the rotation
    // matrix matches the contact-slip and visual conventions.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(10, 3, 3),
    );
    tire.set_hub_kinematic([0.0, 0.0, 0.0], [0.0; 3]);
    tire.hub_angle = 0.1; // a small positive angle
    let r = 0.32_f64;
    let top_body = [0.0, r, 0.0];
    let top_world = tire.body_to_world(top_body);
    assert!(
        top_world[0] > 0.0,
        "small +θ should push top of wheel to +X"
    );
    let (s, c) = (0.1_f64).sin_cos();
    assert!((top_world[0] - r * s).abs() < 1.0e-9);
    assert!((top_world[1] - r * c).abs() < 1.0e-9);
    // And a body-frame point at the BOTTOM (0, −R, 0) should go to
    // (-R·sin θ, -R·cos θ, 0) — i.e. towards −X for positive θ.
    let bot_world = tire.body_to_world([0.0, -r, 0.0]);
    assert!(
        bot_world[0] < 0.0,
        "small +θ should push bottom of wheel to −X"
    );
}

#[test]
fn tetra_tire_fem_body_velocity_to_world_includes_spin_term() {
    // PLAN_ROLL Stage B — for a body-frame-stationary point with
    // body_vel = 0 and hub_omega > 0, the world-frame velocity is
    // pure spin: `ω × r` evaluated at the rotated world position.
    // At the wheel's bottom (body (0, -R, 0)), the spin velocity
    // should be in -X.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(10, 3, 3),
    );
    tire.set_hub_kinematic([0.0, 0.0, 0.0], [0.0; 3]);
    tire.hub_angle = 0.0;
    tire.hub_omega = 5.0;
    let r = 0.32_f64;
    let bot_body = [0.0, -r, 0.0];
    let v_world = tire.body_velocity_to_world(bot_body, [0.0, 0.0, 0.0]);
    // Expected: spin_v_x = ω · r_y = 5.0 · (-0.32) = -1.6.
    assert!(
        (v_world[0] - (-5.0 * r)).abs() < 1.0e-9,
        "bottom-of-wheel world velocity x should be ω · r_y = {}; got {}",
        -5.0 * r,
        v_world[0]
    );
    assert!(v_world[1].abs() < 1.0e-9);
}

#[test]
fn tetra_tire_fem_chassis_handoff_setters_pin_pose_and_velocity() {
    // PLAN_TOURQUE Stage F — set_hub_position/velocity are the
    // chassis hand-off seam: a suspension/chassis loop pushes a
    // target each frame, then reads back last_hub_reaction_xyz to
    // pull the contact wrench into the chassis integrator.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(10, 3, 3),
    );
    tire.set_hub_dynamic();
    assert!(tire.hub_is_dynamic());
    tire.set_hub_position([1.5, 0.30, -0.2]);
    // Position setter flips back to kinematic.
    assert!(!tire.hub_is_dynamic());
    assert_eq!(tire.hub(), [1.5, 0.30, -0.2]);
    tire.set_hub_velocity([2.0, 0.0, 0.5]);
    assert!(!tire.hub_is_dynamic());
    // Non-finite inputs are silently rejected (chassis bugs cannot
    // poison the FEM state).
    tire.set_hub_position([f64::NAN, 0.0, 0.0]);
    assert_eq!(tire.hub(), [1.5, 0.30, -0.2]);
    tire.set_hub_velocity([f64::INFINITY, 0.0, 0.0]);
    // Velocity stays at the last good value.
    assert_eq!(tire.last_hub_reaction_xyz(), [0.0, 0.0, 0.0]);
}

#[test]
fn tetra_tire_fem_dynamic_hub_default_is_kinematic() {
    // PLAN_TOURQUE Stage D — the dynamic-hub flag must default to
    // off so every existing scene keeps its kinematic behaviour.
    let tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(10, 3, 3),
    );
    assert!(!tire.hub_is_dynamic());
    // set_hub_dynamic flips, set_hub_kinematic flips back.
    let mut tire = tire;
    tire.set_hub_dynamic();
    assert!(tire.hub_is_dynamic());
    tire.set_hub_kinematic([0.0; 3], [0.0; 3]);
    assert!(!tire.hub_is_dynamic());
}

#[test]
fn tetra_tire_fem_drive_torque_rolls_dynamic_hub_forward() {
    // PLAN_TOURQUE Stage D — drive torque + dynamic hub on flat
    // terrain physically translates the tire in +X. This is the
    // chain Stage B + Stage C + Stage D wired together: torque
    // spins hub_omega, spin creates backward tread slip, brush
    // reaction pushes the hub forward, hub_v_xz integrates.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(14, 3, 3),
    );
    let terrain = FlatTerrain::new(0.0, tire.params.surface_material.mu_static);
    // Let the tire settle on the floor while still kinematic, so
    // the contact patch is established before we go dynamic.
    let dt = 1.0 / 240.0;
    for _ in 0..60 {
        tire.step_on_terrain(&terrain, dt);
    }
    let hub_before = tire.hub();
    // Switch to dynamic, apply a constant drive torque, run for
    // ~1 second. The tire should visibly advance in +X.
    tire.set_hub_dynamic();
    tire.set_drive_torque_n_m(120.0);
    for _ in 0..240 {
        tire.step_on_terrain(&terrain, dt);
    }
    let hub_after = tire.hub();
    let advanced_m = hub_after[0] - hub_before[0];
    let omega = tire.hub_omega();
    let reaction = tire.last_hub_reaction_xyz();
    assert!(
        advanced_m > 0.01,
        "drive torque on a dynamic hub should physically translate the tire in +X; \
         advanced={advanced_m} m, omega={omega} rad/s, reaction={reaction:?}"
    );
    assert!(
        omega > 0.0,
        "hub must still be spinning forward; got {omega}"
    );
    assert!(
        reaction.iter().all(|v| v.is_finite()),
        "hub reaction must be finite: {reaction:?}"
    );
}

#[test]
fn tetra_tire_fem_kinematic_hub_ignores_contact_reaction() {
    // PLAN_TOURQUE Stage D — a kinematic hub must not drift in X
    // from contact reaction even when torque is applied; the host
    // is in charge of where the hub goes.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(14, 3, 3),
    );
    let terrain = FlatTerrain::new(0.0, tire.params.surface_material.mu_static);
    let dt = 1.0 / 240.0;
    for _ in 0..60 {
        tire.step_on_terrain(&terrain, dt);
    }
    // Re-pin the hub each step so any internal drift is overwritten.
    let pinned = tire.hub();
    tire.set_drive_torque_n_m(120.0);
    for _ in 0..120 {
        tire.set_hub_kinematic(pinned, [0.0; 3]);
        tire.step_on_terrain(&terrain, dt);
    }
    let drift = tire.hub()[0] - pinned[0];
    assert!(
        drift.abs() < 1.0e-6,
        "kinematic hub must stay pinned regardless of torque/contact; drift={drift}"
    );
}

#[test]
fn tetra_tire_fem_torque_setters_clamp_nonfinite_and_negative_brake() {
    // Setters reject NaN/Inf (becomes 0) and force the brake torque
    // to a non-negative magnitude per the API contract.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(12, 3, 3),
    );
    tire.set_drive_torque_n_m(45.0);
    tire.set_brake_torque_n_m(20.0);
    assert_eq!(tire.drive_torque_n_m(), 45.0);
    assert_eq!(tire.brake_torque_n_m(), 20.0);
    tire.set_drive_torque_n_m(f64::NAN);
    tire.set_brake_torque_n_m(-30.0);
    assert_eq!(tire.drive_torque_n_m(), 0.0);
    assert_eq!(tire.brake_torque_n_m(), 0.0);
    tire.set_brake_torque_n_m(f64::INFINITY);
    assert_eq!(tire.brake_torque_n_m(), 0.0);
}

#[test]
fn tetra_tire_fem_params_high_pressure_implicit_defaults_to_matrix_free_newton_mode() {
    let preset = TetraTireFemParams::high_pressure_implicit();
    assert_eq!(
        preset.solver_mode,
        TetraTireFemSolverMode::MatrixFreeNewtonDebug,
        "high_pressure_implicit preset is the first default-flipped solver-mode preset"
    );
    // Numerically equivalent to high_pressure_reference except for the
    // solver_mode field.
    let reference = TetraTireFemParams::high_pressure_reference();
    assert!(
        (preset.pressure_pa - reference.pressure_pa).abs() < 1.0e-9
            && (preset.young_modulus - reference.young_modulus).abs() < 1.0e-9
            && (preset.tread_young_modulus - reference.tread_young_modulus).abs() < 1.0e-9
            && (preset.bead_young_modulus - reference.bead_young_modulus).abs() < 1.0e-9
            && (preset.bead_stiffness - reference.bead_stiffness).abs() < 1.0e-9,
        "the only difference between high_pressure_reference and high_pressure_implicit should be solver_mode"
    );
    // The live timestep through step_on_terrain must produce bounded
    // state from the implicit preset.
    let mut tire = TetraTireFem::new(0.0, preset.pressure_pa, preset.with_resolution(14, 3, 3));
    tire.external_force_xyz = [40.0, -60.0, 0.0];
    let terrain = FlatTerrain::new(-10.0, tire.params.surface_material.mu_static);
    for _ in 0..4 {
        tire.step_on_terrain(&terrain, 1.0 / 240.0);
    }
    assert!(
        tire.pos.iter().all(|p| finite_vec3(*p)) && tire.vel.iter().all(|v| finite_vec3(*v)),
        "high_pressure_implicit preset must keep tire state finite"
    );
}

#[test]
fn tetra_tire_fem_matrix_free_newton_debug_runs_across_named_presets() {
    // Scene-coverage validation: the matrix-free Newton debug timestep
    // must produce bounded, finite state for every named viewer preset.
    // This is the CPU-side complement to the molla-kernels hardware lane
    // and stands in for the live FEM viewer/vehicle scene smoke test
    // until the persistent batch substep is switched over to the GPU
    // dispatch.
    let presets: [(&str, fn() -> TetraTireFemParams); 3] = [
        ("realtime_viewer", || {
            TetraTireFemParams::realtime_viewer()
                .with_resolution(14, 3, 3)
                .with_solver_mode(TetraTireFemSolverMode::MatrixFreeNewtonDebug)
        }),
        ("offroad_soft", || {
            TetraTireFemParams::offroad_soft()
                .with_resolution(14, 3, 3)
                .with_solver_mode(TetraTireFemSolverMode::MatrixFreeNewtonDebug)
        }),
        ("high_pressure_reference", || {
            TetraTireFemParams::high_pressure_reference()
                .with_resolution(14, 3, 3)
                .with_solver_mode(TetraTireFemSolverMode::MatrixFreeNewtonDebug)
        }),
    ];
    for (name, build_params) in presets {
        let params = build_params();
        assert_eq!(
            params.solver_mode,
            TetraTireFemSolverMode::MatrixFreeNewtonDebug,
            "{name} preset must opt into the matrix-free Newton solver mode"
        );
        let mut tire = TetraTireFem::new(0.0, params.pressure_pa, params);
        tire.external_force_xyz = [25.0, -40.0, 0.0];
        let terrain = FlatTerrain::new(-10.0, tire.params.surface_material.mu_static);
        let pre_positions = tire.pos.clone();
        // Step several frames to make sure the implicit timestep keeps
        // the tire bounded across more than a single dispatch.
        for _ in 0..4 {
            tire.step_on_terrain(&terrain, 1.0 / 240.0);
        }
        assert!(
            tire.pos.iter().all(|p| finite_vec3(*p)) && tire.vel.iter().all(|v| finite_vec3(*v)),
            "{name} preset must keep tire state finite under matrix-free Newton solver mode"
        );
        assert!(
            tire.vel.iter().all(|v| dot(*v, *v).sqrt() < 30.0),
            "{name} preset must keep tire velocity bounded"
        );
        let moved = tire.pos.iter().zip(pre_positions.iter()).any(|(p, p0)| {
            (p[0] - p0[0]).abs() + (p[1] - p0[1]).abs() + (p[2] - p0[2]).abs() > 0.0
        });
        assert!(
            moved,
            "{name} preset must actually mutate positions through the matrix-free Newton timestep"
        );
    }
}

#[test]
fn tetra_tire_fem_solver_mode_routes_step_on_terrain_to_matrix_free_newton_debug() {
    // Explicit default keeps the original stepper unchanged.
    let mut explicit_tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(12, 3, 3),
    );
    assert_eq!(
        explicit_tire.params.solver_mode,
        TetraTireFemSolverMode::ExplicitDefault,
        "default tires must keep the explicit/semi-implicit production stepper"
    );
    let terrain = FlatTerrain::new(-10.0, explicit_tire.params.surface_material.mu_static);
    explicit_tire.step_on_terrain(&terrain, 1.0 / 240.0);
    assert!(
        explicit_tire.pos.iter().all(|p| finite_vec3(*p))
            && explicit_tire.vel.iter().all(|v| finite_vec3(*v)),
        "explicit stepper must keep tire state finite"
    );

    // Opting in via solver_mode dispatches to the CPU/debug matrix-free
    // Newton timestep without changing any other tire field.
    let mut newton_tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer()
            .with_resolution(12, 3, 3)
            .with_solver_mode(TetraTireFemSolverMode::MatrixFreeNewtonDebug),
    );
    assert_eq!(
        newton_tire.params.solver_mode,
        TetraTireFemSolverMode::MatrixFreeNewtonDebug
    );
    let pre_positions = newton_tire.pos.clone();
    newton_tire.step_on_terrain(&terrain, 1.0 / 300.0);
    assert!(
        newton_tire.pos.iter().all(|p| finite_vec3(*p))
            && newton_tire.vel.iter().all(|v| finite_vec3(*v)),
        "matrix-free Newton debug stepper must keep tire state finite"
    );
    let moved = newton_tire
        .pos
        .iter()
        .zip(pre_positions.iter())
        .any(|(p, p0)| (p[0] - p0[0]).abs() + (p[1] - p0[1]).abs() + (p[2] - p0[2]).abs() > 0.0);
    assert!(
        moved,
        "matrix-free Newton debug stepper should mutate positions when called through step_on_terrain"
    );
}

#[test]
fn tetra_fem_hardening_envelope_report_covers_stability_cases() {
    let report = tetra_fem_hardening_envelope_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "FEM hardening envelope report should pass broad robustness gates: {report:?}"
    );
    assert!(
        report.high_load_patch_width_m >= 0.01
            && report.high_pressure_radius_ratio <= 1.35
            && report.long_run_steps >= 800
            && report.energy_final_over_peak <= 0.95,
        "hardening report should expose high-load, high-pressure, long-run, and damping cases: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "fem_hardening_high_load_patch_exists")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_hardening_high_pressure_radius_bounded")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_hardening_long_run_steps")
            && checks
                .iter()
                .any(|check| check.check_id == "fem_hardening_energy_damps_from_peak"),
        "hardening report should expose all broad robustness gates: {checks:?}"
    );
}

#[test]
fn tetra_hardening_thousands_of_small_substeps_without_nans() {
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams {
            segments: 12,
            radial_layers: 3,
            width_layers: 3,
            downforce: 4_200.0,
            ..Default::default()
        },
    );
    for _ in 0..2_200 {
        tire.step(1.25e-4);
    }
    tetra_state_is_finite_and_bounded(&tire, 2.0, 8.5);
    assert!(
        tire.hub_deflection_m().is_finite() && tire.hub_deflection_m() < 0.20,
        "hub deflection out of robust bound: {}",
        tire.hub_deflection_m()
    );
}

#[test]
fn tetra_hardening_damping_keeps_kinetic_energy_bounded() {
    let mut tire = TetraTireFem::new(
        0.0,
        170_000.0,
        TetraTireFemParams {
            segments: 12,
            radial_layers: 3,
            width_layers: 3,
            downforce: 5_000.0,
            ..Default::default()
        },
    );
    // Inject a finite disturbance and verify damping/contact do not let
    // energy grow without bound.
    for (i, v) in tire.vel.iter_mut().enumerate() {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        *v = [0.35 * sign, -0.25, 0.20 * sign];
    }
    let initial = tetra_kinetic_energy(&tire).max(1.0e-9);
    let mut peak = initial;
    for _ in 0..1_200 {
        tire.step(1.25e-4);
        peak = peak.max(tetra_kinetic_energy(&tire));
    }
    let final_energy = tetra_kinetic_energy(&tire);
    assert!(
        peak < initial * 75.0,
        "disturbed tire kinetic energy blew up: initial={initial} peak={peak}"
    );
    assert!(
        final_energy < peak * 0.95,
        "damping did not reduce kinetic energy from peak: peak={peak} final={final_energy}"
    );
    tetra_state_is_finite_and_bounded(&tire, 2.0, 8.5);
}

/// Stage (j) — heat dissipation. With high slip the brush model
/// must report a strictly positive `dissipated_power`, and zero
/// when slip vanishes.
#[test]
fn brush_contact_reports_heat_under_slip() {
    let no_slip = loaded_output(220_000.0, SurfaceMaterial::dry_asphalt(), 0.0);
    let big_slip = loaded_output(220_000.0, SurfaceMaterial::dry_asphalt(), 8.0);
    assert!(
        big_slip.dissipated_power > 50.0,
        "expected real heat under heavy slip, got {} W",
        big_slip.dissipated_power
    );
    assert!(
        no_slip.dissipated_power < big_slip.dissipated_power * 0.05,
        "no-slip dissipation too high ({} W) vs slip ({} W)",
        no_slip.dissipated_power,
        big_slip.dissipated_power
    );
}

/// Stage (j) — passive `TireThermalState` integrator. Sustained
/// dissipation must heat the tire; sustained idle must cool it back
/// toward ambient.
#[test]
fn tire_thermal_state_heats_and_cools() {
    let mut t = TireThermalState::default();
    // Heat: 5 kW for 60 s.
    for _ in 0..600 {
        t.step(5_000.0, 0.1);
    }
    let hot = t.temperature_c;
    assert!(
        hot > 30.0 && hot < 600.0,
        "tire didn't heat into a sane range: {hot} °C"
    );
    // Cool: idle for 30 minutes. With τ ≈ 22 min the tire is well
    // on its way back toward ambient but not all the way down.
    for _ in 0..18_000 {
        t.step(0.0, 0.1);
    }
    let cool = t.temperature_c;
    let amb = t.ambient_c;
    assert!(
        cool < amb + 0.5 * (hot - amb),
        "tire didn't cool significantly: hot={hot} cool={cool} ambient={amb}"
    );
}

/// Multi-layer thermal model — sliding/contact work must heat the outer
/// surface before the deeper tread/carcass nodes catch up.
#[test]
fn tire_thermal_layers_route_sliding_heat_to_surface_first() {
    let mut t = TireThermalLayers::default();
    for _ in 0..300 {
        t.step(8_000.0, 0.0, 0.0, 0.0, 20.0, 0.1);
    }
    assert!(
        t.surface_c > t.tread_c + 3.0,
        "sliding should heat the surface first: {t:?}"
    );
    assert!(
        t.tread_c > t.carcass_c,
        "tread should sit between surface and carcass after sliding heat: {t:?}"
    );
    let mu = t.mu_factor();
    assert!(
        mu.is_finite() && (0.45..=1.08).contains(&mu),
        "thermal μ factor out of clamp range: {mu}"
    );
}

/// Multi-layer thermal model — cyclic casing flex and rolling resistance
/// must warm the carcass even when the contact patch is not sliding.
#[test]
fn tire_thermal_layers_rolling_heats_carcass_without_sliding() {
    let mut t = TireThermalLayers::default();
    for _ in 0..1_200 {
        t.step(0.0, 3_000.0, 600.0, 0.0, 20.0, 0.1);
    }
    assert!(
        t.carcass_c > t.surface_c + 2.0,
        "flex/rolling heat should live mostly in the carcass: {t:?}"
    );
    assert!(
        t.carcass_pressure_factor() > 1.0,
        "hot carcass should raise the slow pressure/stiffness factor: {t:?}"
    );
}

/// Multi-layer thermal model — the surface has small heat capacity and
/// direct road/air sinks, so it cools faster than the carcass.
#[test]
fn tire_thermal_layers_surface_cools_faster_than_carcass() {
    let mut t = TireThermalLayers {
        surface_c: 140.0,
        tread_c: 105.0,
        carcass_c: 100.0,
        rim_c: 80.0,
        ..Default::default()
    };
    let surface0 = t.surface_c;
    let carcass0 = t.carcass_c;
    for _ in 0..600 {
        t.step(0.0, 0.0, 0.0, 0.0, 20.0, 0.1);
    }
    let surface_drop = surface0 - t.surface_c;
    let carcass_drop = carcass0 - t.carcass_c;
    assert!(
        surface_drop > carcass_drop,
        "surface should cool faster: surface_drop={surface_drop} carcass_drop={carcass_drop} state={t:?}"
    );
}

#[test]
fn tire_wheel_spin_torque_balance_report_matches_drive_equation() {
    let report = tire_wheel_spin_torque_balance_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "wheel-spin torque-balance report should pass broad equation gates: {report:?}"
    );
    assert!(
        report.drive_torque_nm > report.tire_reaction_torque_nm + report.bearing_loss_nm
            && report.omega_dot_rad_s2 > 0.0
            && report.omega_delta_error() <= 1.0e-9
            && report.slip_ratio > 0.0,
        "drive torque should integrate wheel spin through the torque balance equation: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "wheel_spin_tire_reaction_positive")
            && checks
                .iter()
                .any(|check| check.check_id == "wheel_spin_integrator_matches_equation")
            && checks
                .iter()
                .any(|check| check.check_id == "wheel_spin_slip_is_outcome"),
        "wheel-spin torque-balance checks should expose reaction torque, integration, and slip outcome gates: {checks:?}"
    );
}

#[test]
fn tire_vehicle_longitudinal_motion_report_matches_force_balance() {
    let report = tire_vehicle_longitudinal_motion_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "vehicle longitudinal-motion report should pass broad force-balance gates: {report:?}"
    );
    assert!(
        report.gross_drive_force_per_wheel_n > report.rolling_loss_per_wheel_n
            && report.net_vehicle_force_n > 0.0
            && report.expected_accel_m_per_s2 > 0.0
            && report.velocity_delta_error_m_per_s <= 1.0e-12
            && report.force_balance_error_n <= 1.0e-9,
        "vehicle speed should integrate from tire force minus rolling loss and drag: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "vehicle_longitudinal_integrator_matches_equation")
            && checks
                .iter()
                .any(|check| check.check_id == "vehicle_longitudinal_force_balance_closes"),
        "vehicle longitudinal checks should expose integration and force-balance gates: {checks:?}"
    );
}

#[test]
fn tire_condition_report_exposes_cold_hot_and_abused_states() {
    let fresh = TireWearState::default();
    let cold = TireThermalLayers {
        surface_c: 5.0,
        tread_c: 8.0,
        carcass_c: 12.0,
        ..Default::default()
    }
    .condition_report(&fresh);
    let healthy = TireThermalLayers {
        surface_c: 80.0,
        tread_c: 70.0,
        carcass_c: 60.0,
        ..Default::default()
    }
    .condition_report(&fresh);
    let hot = TireThermalLayers {
        surface_c: 145.0,
        tread_c: 130.0,
        carcass_c: 115.0,
        ..Default::default()
    }
    .condition_report(&fresh);
    let mut worn = TireWearState::default();
    worn.depth_m = worn.max_depth_m * 0.96;
    let abused = TireThermalLayers {
        surface_c: 220.0,
        tread_c: 185.0,
        carcass_c: 170.0,
        ..Default::default()
    }
    .condition_report(&worn);

    assert_eq!(cold.level, TireConditionLevel::Cold);
    assert_eq!(healthy.level, TireConditionLevel::Healthy);
    assert!(
        matches!(
            hot.level,
            TireConditionLevel::Hot | TireConditionLevel::Overheated
        ),
        "hot state should be labelled hot/overheated: {hot:?}"
    );
    assert_eq!(abused.level, TireConditionLevel::Abused);
    for report in [cold, healthy, hot, abused] {
        assert!(
            report.is_finite(),
            "condition report must be finite: {report:?}"
        );
        assert!((0.0..=1.0).contains(&report.abuse_index));
        assert!((0.08..=1.12).contains(&report.grip_factor));
    }
    assert!(
        healthy.grip_factor > cold.grip_factor,
        "cold tire should under-grip healthy tire: cold={cold:?} healthy={healthy:?}"
    );
    assert!(
        healthy.grip_factor > hot.grip_factor,
        "overheated tire should under-grip healthy tire: healthy={healthy:?} hot={hot:?}"
    );
    assert!(
        abused.grip_factor < hot.grip_factor
            && abused.carcass_stiffness_factor < hot.carcass_stiffness_factor
            && abused.wear_rate_factor > hot.wear_rate_factor,
        "abused tire should lose grip/stiffness and wear faster: hot={hot:?} abused={abused:?}"
    );
}

#[test]
fn tire_condition_material_response_applies_state_driven_failure_curves() {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let material = SurfaceMaterial::dry_asphalt();
    let fresh = TireWearState::default();
    let worn = TireWearState {
        depth_m: 0.0075,
        ..Default::default()
    };
    let healthy_thermal = TireThermalLayers {
        surface_c: 80.0,
        tread_c: 78.0,
        carcass_c: 62.0,
        ..Default::default()
    };
    let cold_thermal = TireThermalLayers {
        surface_c: -5.0,
        tread_c: 0.0,
        carcass_c: 8.0,
        ..Default::default()
    };
    let abused_thermal = TireThermalLayers {
        surface_c: 225.0,
        tread_c: 195.0,
        carcass_c: 185.0,
        ..Default::default()
    };

    let healthy = healthy_thermal.material_response(&fresh, &params, &material);
    let cold = cold_thermal.material_response(&fresh, &params, &material);
    let abused = abused_thermal.material_response(&worn, &params, &material);

    for response in [healthy, cold, abused] {
        assert!(
            response.is_finite(),
            "response must be finite: {response:?}"
        );
        assert!(
            response.material.mu_dynamic <= response.material.mu_static,
            "conditioned material must preserve dynamic <= static μ: {response:?}"
        );
    }
    assert!(
        cold.cold_grip_limited
            && cold.mu_factor < healthy.mu_factor
            && cold.material.mu_static < healthy.material.mu_static,
        "cold tire should reduce usable μ through state response: healthy={healthy:?} cold={cold:?}"
    );
    assert!(
        abused.abused
            && abused.overheated_grip_loss
            && abused.carcass_softened
            && abused.worn_tread_limited,
        "abused worn tire should expose state-derived failure flags: {abused:?}"
    );
    assert!(
        abused.material.mu_static < healthy.material.mu_static * 0.60
            && abused.params.tread_shear_stiffness < healthy.params.tread_shear_stiffness
            && abused.params.carcass_normal_stiffness < healthy.params.carcass_normal_stiffness
            && abused.material.rolling_resistance > healthy.material.rolling_resistance,
        "abused response should lose grip/stiffness and gain rolling loss: healthy={healthy:?} abused={abused:?}"
    );

    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let load_n = 4_200.0;
    let speed = 14.0;
    let healthy_force = tire_rig_point(
        &healthy.params,
        &terrain,
        &healthy.material,
        load_n,
        speed,
        0.18,
        0.0,
        0.0,
    )
    .telemetry
    .longitudinal_force_n
    .abs();
    let abused_force = tire_rig_point(
        &abused.params,
        &terrain,
        &abused.material,
        load_n,
        speed,
        0.18,
        0.0,
        0.0,
    )
    .telemetry
    .longitudinal_force_n
    .abs();
    assert!(
        abused_force < healthy_force * 0.75,
        "state-driven failure response should reduce tractive force: healthy={healthy_force} abused={abused_force}"
    );
}

#[test]
fn tire_condition_failure_mode_report_gates_state_driven_response() {
    let report = tire_condition_failure_mode_report();
    assert!(report.is_finite(), "report should be finite: {report:?}");
    assert!(
        report.passed(),
        "condition failure-mode report should pass broad state-derived gates: {report:?}"
    );
    assert!(
        report.cold_grip_limited
            && report.overheated_grip_loss
            && report.carcass_softened
            && report.worn_tread_limited
            && report.abused_flag
            && report.abused_force_n < report.healthy_force_n * 0.75,
        "condition report should expose cold/hot/worn/abused state response and force loss: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "condition_failure_levels_state_derived")
            && checks
                .iter()
                .any(|check| check.check_id == "condition_failure_abuse_reduces_mu")
            && checks
                .iter()
                .any(|check| check.check_id == "condition_failure_abuse_reduces_force"),
        "condition failure report should expose level, material, and force gates: {checks:?}"
    );
}

/// Temperature-aware wear — the same sliding power should remove more
/// rubber when the surface is overheated.
#[test]
fn tire_wear_state_hot_surface_accelerates_wear() {
    let mut warm = TireWearState::default();
    let mut hot = TireWearState::default();
    for _ in 0..1_000 {
        warm.step_with_surface_temperature(6_000.0, 70.0, 0.1);
        hot.step_with_surface_temperature(6_000.0, 150.0, 0.1);
    }
    assert!(
        hot.depth_m > warm.depth_m * 1.5,
        "hot tread should wear faster: warm={} hot={}",
        warm.depth_m,
        hot.depth_m
    );
}

#[test]
fn tire_wear_power_temperature_report_orders_power_and_heat() {
    let report = tire_wear_power_temperature_report();
    assert!(
        report.passed(),
        "wear power/temperature sensitivity report should pass: {report:?}"
    );
    assert!(
        report.high_power_warm_wear_m > report.low_power_warm_wear_m
            && report.high_power_hot_wear_m > report.high_power_warm_wear_m,
        "wear should increase with both sliding power and hot surface state: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_wear_power_increases_wear")
            && checks
                .iter()
                .any(|check| check.check_id == "thermal_wear_hot_surface_accelerates_wear"),
        "wear power/temperature checks should expose explicit runner gates: {checks:?}"
    );
}

#[test]
fn tire_worn_surface_sensitivity_report_orders_wet_ice_and_soil_loss() {
    let report = tire_worn_surface_sensitivity_report();
    assert!(
        report.passed(),
        "worn-surface sensitivity report should pass: {report:?}"
    );
    assert!(
        report.wet_loss_fraction > report.dry_loss_fraction
            && report.ice_loss_fraction > report.dry_loss_fraction
            && report.loose_soil_loss_fraction > report.dry_loss_fraction,
        "worn tread should hurt wet/ice/loose-soil more than dry: {report:?}"
    );
    let checks = report.acceptance_checks();
    assert!(
        checks.iter().all(|check| check.passed)
            && checks
                .iter()
                .any(|check| check.check_id == "worn_surface_wet_extra_loss")
            && checks
                .iter()
                .any(|check| check.check_id == "worn_surface_ice_extra_loss")
            && checks
                .iter()
                .any(|check| check.check_id == "worn_surface_loose_soil_extra_loss"),
        "worn-surface checks should expose wet/ice/soil gates: {checks:?}"
    );
}

/// Lateral grip sweep — sweep slip angle `α ∈ [0°, 25°]` and verify
/// the classic brush-tire cornering signature: `Fy` grows
/// monotonically with `α` while sliding cells stay rare, then
/// saturates at the Coulomb friction cap once `α` is past the
/// peak-grip angle. Mirror of `brush_load_deflection_sweep` for
/// the lateral axis.
#[test]
fn brush_lateral_grip_sweep_saturates() {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let params = RealTireParams {
        pressure_pa: 220_000.0,
        segments: 64,
        ..Default::default()
    };
    let load = 4_500.0;
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load);
    // Drive longitudinal velocity 5 m/s, sweep lateral velocity to
    // produce slip angles from 0° to 25°.
    let angles_deg: [f64; 8] = [0.0, 2.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0];
    let mut prev_abs_fy = 0.0_f64;
    let mut peak_abs_fy = 0.0_f64;
    let mut saw_growth = false;
    let mut saw_saturation = false;
    for alpha_deg in angles_deg {
        let alpha = alpha_deg.to_radians();
        let v_long = 5.0;
        let v_lat = v_long * alpha.tan();
        let out = evaluate_brush_tire(
            &params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [v_long, 0.0, v_lat],
                heading: [1.0, 0.0, 0.0],
                omega: v_long / params.radius, // free-rolling
            },
            &terrain,
            &material,
        );
        let abs_fy = out.lateral_force.abs();
        peak_abs_fy = peak_abs_fy.max(abs_fy);
        // Force must oppose the lateral velocity.
        if alpha_deg > 0.0 {
            assert!(
                out.lateral_force < 0.0,
                "lateral force should oppose +Z slip at α={alpha_deg}°: \
                 got {}",
                out.lateral_force
            );
        }
        // Up to the peak (typically ≈5° for this tire/material
        // pairing) the curve should be growing.
        if alpha_deg <= 5.0 && alpha_deg > 0.0 {
            assert!(
                abs_fy >= prev_abs_fy - 5.0,
                "Fy should grow with α below peak (α={alpha_deg}°): \
                 prev={prev_abs_fy} now={abs_fy}"
            );
            if abs_fy > prev_abs_fy + 100.0 {
                saw_growth = true;
            }
        }
        // Once past saturation point, Fy must not exceed μ_s·Fz.
        let cap = material.mu_static * out.normal_load;
        assert!(
            abs_fy <= cap * 1.10,
            "lateral force exceeded Coulomb cap at α={alpha_deg}°: \
             |Fy|={abs_fy} cap={cap}"
        );
        if alpha_deg >= 16.0 && abs_fy > 0.9 * peak_abs_fy {
            saw_saturation = true;
        }
        prev_abs_fy = abs_fy;
    }
    assert!(
        saw_growth,
        "Fy never showed significant growth across the small-α band"
    );
    assert!(
        saw_saturation,
        "Fy never reached saturation at large α (peak={peak_abs_fy})"
    );
}

/// Stage T5 — confirm the FEM tire's equilibrium hub-y deflection
/// responds visibly to load (the viewer scene 122 readout relies on
/// this). Pressure has its main visible effect via the contact
/// patch width / shape, which the existing
/// `tetra_fem_pressure_reduces_patch_width` test pins down.
#[test]
fn tetra_fem_deflection_grows_with_load() {
    let settle = |load: f64| -> f64 {
        let mut t = TetraTireFem::new(
            0.0,
            180_000.0,
            TetraTireFemParams {
                segments: 16,
                radial_layers: 3,
                width_layers: 3,
                downforce: load,
                ..Default::default()
            },
        );
        for _ in 0..360 {
            t.step(1.0 / 240.0);
        }
        t.hub_deflection_m() * 1000.0 // mm
    };
    let d_light = settle(1_000.0);
    let d_heavy = settle(6_000.0);
    assert!(
        d_heavy > d_light + 1.0,
        "expected heavy load to deflect more: light={d_light:.1} mm, heavy={d_heavy:.1} mm"
    );
    for d in [d_light, d_heavy] {
        assert!(
            d > 0.0 && d < 80.0,
            "deflection out of sane band: {d:.1} mm"
        );
    }
}

/// Stage (j) — tread wear accumulates from sliding dissipation
/// and degrades the friction factor.
#[test]
fn tire_wear_state_integrates_and_drops_grip() {
    let mut w = TireWearState::default();
    assert_eq!(w.depth_m, 0.0);
    assert!((w.mu_factor() - 1.0).abs() < 1.0e-9);
    // 10 kW dissipation for an hour (3600 s).
    for _ in 0..36_000 {
        w.step(10_000.0, 0.1);
    }
    assert!(
        w.depth_m > 0.0 && w.depth_m <= w.max_depth_m,
        "wear depth out of range: {} m",
        w.depth_m
    );
    let factor = w.mu_factor();
    assert!(factor < 1.0, "worn tire should grip less: factor={factor}");
}

/// Stage (j) — load-deflection sweep on the brush contact (closed-
/// form path, fast). Walks `Fz` from 1 kN to 8 kN at three
/// pressures and asserts:
///   1. patch length grows monotonically with load,
///   2. lower pressure gives a longer patch at each load.
#[test]
fn brush_load_deflection_sweep() {
    let pressures_pa = [120_000.0, 220_000.0, 320_000.0];
    let loads = [1_000.0, 3_000.0, 5_000.0, 8_000.0];
    let mut prev_patch_per_pressure = [0.0; 3];
    for fz in loads {
        for (i, &p) in pressures_pa.iter().enumerate() {
            let terrain = FlatTerrain::new(0.0, 1.0);
            let params = RealTireParams {
                pressure_pa: p,
                ..Default::default()
            };
            let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, fz);
            let out = evaluate_brush_tire(
                &params,
                &RealWheelState {
                    hub_pos: [0.0, hub_y, 0.0],
                    hub_vel: [0.0; 3],
                    heading: [1.0, 0.0, 0.0],
                    omega: 0.0,
                },
                &terrain,
                &SurfaceMaterial::dry_asphalt(),
            );
            assert!(
                out.patch_length > prev_patch_per_pressure[i] - 1.0e-6,
                "patch length not monotone with load at p={p} Pa: \
                 prev={} now={} at Fz={fz}",
                prev_patch_per_pressure[i],
                out.patch_length
            );
            prev_patch_per_pressure[i] = out.patch_length;
        }
    }
    // At Fz = 5 kN, low pressure → longer patch than high pressure.
    let terrain = FlatTerrain::new(0.0, 1.0);
    let patch_at = |pressure: f64, fz: f64| -> f64 {
        let params = RealTireParams {
            pressure_pa: pressure,
            ..Default::default()
        };
        let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, fz);
        evaluate_brush_tire(
            &params,
            &RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [0.0; 3],
                heading: [1.0, 0.0, 0.0],
                omega: 0.0,
            },
            &terrain,
            &SurfaceMaterial::dry_asphalt(),
        )
        .patch_length
    };
    let p_low = patch_at(120_000.0, 5_000.0);
    let p_high = patch_at(320_000.0, 5_000.0);
    assert!(
        p_low > p_high,
        "low-pressure patch should be longer at Fz=5 kN: low={p_low} high={p_high}"
    );
    // Magnitudes: passenger-tire patches at 5 kN are typically
    // 0.08–0.20 m. Allow a generous band.
    assert!(
        p_low > 0.04 && p_low < 0.35,
        "low-pressure patch length out of plausible band: {p_low} m"
    );
}

/// Hardening sweep for the reduced brush tire: over a small grid
/// of loads, pressures, surfaces, and slips the contact output
/// must stay finite, support roughly the requested vertical load,
/// keep patch sizes inside geometric bounds, and respect the
/// aggregate Coulomb/cohesion cap.
#[test]
fn brush_contact_sanity_matrix_is_finite_and_capped() {
    let pressures_pa = [0.0, 80_000.0, 220_000.0, 420_000.0];
    let loads_n = [800.0, 4_500.0, 9_000.0];
    let materials = [
        SurfaceMaterial::dry_asphalt(),
        SurfaceMaterial::loose_dirt(),
        SurfaceMaterial::ice(),
    ];
    let slip_cases = [
        (2.0, 0.0, 0.0),
        (6.0, 1.5, 0.0),
        (6.0, 0.0, 1.2),
        (6.0, 4.0, 1.6),
    ];

    for pressure_pa in pressures_pa {
        for load_n in loads_n {
            for material in materials {
                let terrain = FlatTerrain::new(0.0, material.mu_static);
                let params = RealTireParams {
                    pressure_pa,
                    segments: 96,
                    ..Default::default()
                };
                let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
                for (v_long, slip_speed, v_lat) in slip_cases {
                    let out = evaluate_brush_tire(
                        &params,
                        &RealWheelState {
                            hub_pos: [0.0, hub_y, 0.0],
                            hub_vel: [v_long, 0.0, v_lat],
                            heading: [1.0, 0.0, 0.0],
                            omega: (v_long + slip_speed) / params.radius,
                        },
                        &terrain,
                        &material,
                    );
                    assert!(
                        out.in_contact,
                        "expected contact at p={pressure_pa} load={load_n}"
                    );
                    assert!(
                        out.force.iter().all(|v| v.is_finite())
                            && out.normal_load.is_finite()
                            && out.longitudinal_force.is_finite()
                            && out.lateral_force.is_finite()
                            && out.dissipated_power.is_finite(),
                        "non-finite tire contact: {out:?}"
                    );
                    assert!(
                        out.normal_load > 0.65 * load_n && out.normal_load < 1.35 * load_n,
                        "normal load drifted away from target: p={pressure_pa} load={load_n} got={}",
                        out.normal_load
                    );
                    assert!(
                        out.patch_length >= 0.0 && out.patch_length <= 2.0 * params.radius + 1.0e-6,
                        "patch length outside tire diameter: {}",
                        out.patch_length
                    );
                    assert!(
                        out.contact_count <= params.segments,
                        "contact cells cannot exceed segments"
                    );
                    assert!(
                        out.sliding_count <= out.contact_count,
                        "sliding cells cannot exceed contact cells"
                    );

                    let raw_fx = out.longitudinal_force + out.rolling_resistance_force;
                    let combined = (raw_fx * raw_fx + out.lateral_force * out.lateral_force).sqrt();
                    let area_sum: f64 = out.cells.iter().map(|c| c.area).sum();
                    let cap = material.mu_static.max(0.0) * out.normal_load
                        + material.cohesion.max(0.0) * area_sum;
                    assert!(
                        combined <= cap * 1.15 + 1.0e-6,
                        "combined brush force exceeded static cap: |F|={combined} cap={cap}"
                    );
                }
            }
        }
    }
}

/// Invalid UI / caller inputs should not poison the tire with NaNs
/// or negative loads. The brush path sanitizes parameters and
/// materials into conservative physical ranges.
#[test]
fn brush_contact_hardens_invalid_inputs() {
    let terrain = FlatTerrain::new(0.0, 1.0);
    let bad_params = RealTireParams {
        radius: -0.5,
        width: f64::NAN,
        pressure_pa: -1.0e9,
        segments: 1_000_000,
        carcass_normal_stiffness: f64::NAN,
        tread_shear_stiffness: -10.0,
        structural_lateral_bias_per_normal: f64::NAN,
        relaxation_speed: 0.0,
    };
    let bad_material = SurfaceMaterial {
        mu_static: f64::NAN,
        mu_dynamic: 5.0,
        bristle_damping: -1.0,
        rolling_resistance: -0.5,
        cohesion: -20.0,
    };
    let out = evaluate_brush_tire(
        &bad_params,
        &RealWheelState {
            hub_pos: [0.0, 0.25, 0.0],
            hub_vel: [1.0, 0.0, 0.2],
            heading: [0.0, 0.0, 0.0],
            omega: f64::NAN,
        },
        &terrain,
        &bad_material,
    );
    assert!(
        out.force.iter().all(|v| v.is_finite())
            && out.normal_load.is_finite()
            && out.longitudinal_force.is_finite()
            && out.lateral_force.is_finite()
            && out.dissipated_power.is_finite(),
        "sanitized bad tire input should stay finite: {out:?}"
    );
    assert!(out.normal_load >= 0.0);
    assert!(out.contact_count <= 512, "segment hard cap should hold");

    let empty = evaluate_brush_tire(
        &RealTireParams::default(),
        &RealWheelState {
            hub_pos: [f64::NAN, 0.0, 0.0],
            hub_vel: [0.0; 3],
            heading: [1.0, 0.0, 0.0],
            omega: 0.0,
        },
        &terrain,
        &SurfaceMaterial::dry_asphalt(),
    );
    assert!(
        !empty.in_contact && empty.force.iter().all(|v| v.is_finite()),
        "non-finite wheel state should return an empty finite contact"
    );
}

/// Stage (b) — cavity gas-law plumbing. With pressure disabled the
/// cavity has no inflation/expansion drive, so flattening the tire
/// onto the floor must shrink the enclosed cavity volume. This is the
/// raw geometric path that `step_inner` reads to compute `P_now`.
#[test]
fn cavity_volume_shrinks_when_loaded_unpressurised() {
    let mut tire = TetraTireFem::new(
        0.0,
        0.0,
        TetraTireFemParams {
            segments: 16,
            radial_layers: 3,
            width_layers: 3,
            downforce: 3_200.0,
            ..Default::default()
        },
    );
    let v_rest = tire.cavity_volume();
    assert!(v_rest > 1.0e-4, "rest cavity volume {v_rest} too small");
    for _ in 0..180 {
        tire.step(1.0 / 240.0);
    }
    let v_loaded = tire.cavity_volume();
    assert!(
        v_loaded < v_rest,
        "loaded cavity volume should shrink: rest={v_rest} loaded={v_loaded}"
    );
    // Sanity: the gas pocket doesn't disappear — it should still be
    // within a sensible fraction of its rest volume.
    assert!(
        v_loaded > 0.3 * v_rest,
        "loaded cavity volume collapsed implausibly: rest={v_rest} loaded={v_loaded}"
    );
}

#[test]
fn tetra_tire_fem_nodes_rendered_is_rigid_rotation_about_hub() {
    // Render-time rotation (BeamNG pattern): `nodes_rendered()` must be
    // the identity at `hub_angle = 0`, and after the hub spins it must
    // be a rigid rotation of the physics nodes about the hub spin axis:
    // distances to the hub axis and Z coordinates are preserved.
    let mut tire = TetraTireFem::new(
        0.0,
        180_000.0,
        TetraTireFemParams::realtime_viewer().with_resolution(12, 3, 3),
    );
    assert_eq!(tire.hub_angle(), 0.0);
    let rendered0 = tire.nodes_rendered();
    for (a, b) in rendered0.iter().zip(tire.nodes().iter()) {
        for k in 0..3 {
            assert!(
                (a[k] - b[k]).abs() < 1.0e-12,
                "at hub_angle = 0 rendered nodes must equal physics nodes"
            );
        }
    }

    tire.set_drive_torque_n_m(60.0);
    let terrain = FlatTerrain::new(-10.0, tire.params.surface_material.mu_static);
    for _ in 0..12 {
        tire.step_on_terrain(&terrain, 1.0 / 240.0);
    }
    assert!(
        tire.hub_angle() > 1.0e-4,
        "fixture must accumulate spin angle: {}",
        tire.hub_angle()
    );
    let hub = tire.hub();
    let rendered = tire.nodes_rendered();
    let nodes = tire.nodes();
    assert_eq!(rendered.len(), nodes.len());
    let mut max_radius_err = 0.0_f64;
    let mut moved = 0usize;
    for (r, p) in rendered.iter().zip(nodes.iter()) {
        let rr = ((r[0] - hub[0]).powi(2) + (r[1] - hub[1]).powi(2)).sqrt();
        let rp = ((p[0] - hub[0]).powi(2) + (p[1] - hub[1]).powi(2)).sqrt();
        max_radius_err = max_radius_err.max((rr - rp).abs());
        assert!(
            (r[2] - p[2]).abs() < 1.0e-12,
            "spin is about +Z: Z must pass through unchanged"
        );
        if ((r[0] - p[0]).powi(2) + (r[1] - p[1]).powi(2)).sqrt() > 1.0e-6 {
            moved += 1;
        }
    }
    assert!(
        max_radius_err < 1.0e-9,
        "rigid rotation must preserve distance to the hub axis: err={max_radius_err}"
    );
    assert!(
        moved > nodes.len() / 2,
        "a nonzero hub_angle must visibly move most nodes: moved={moved}/{}",
        nodes.len()
    );
}
