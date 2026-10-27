//! Viewer-target coverage check + report.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::internal::csv_token;
use super::{TireRigAcceptanceReport, TireRigRunOutput, TireVehicleSanityCase};

/// One viewer-target coverage check tied back to deterministic rig evidence.
///
/// These rows keep the PLAN_REALTIRE viewer targets auditable from exported
/// rig artifacts: the viewer can still be richer than the rig, but each target
/// must have at least one deterministic replay or vehicle-sanity artifact that
/// makes the expected behavior falsifiable without eyeballing the scene.
#[derive(Clone, Debug)]
pub struct TireViewerTargetCoverageCheck {
    pub target_id: String,
    pub expectation: String,
    pub evidence: String,
    pub passed: bool,
}

impl TireViewerTargetCoverageCheck {
    pub fn new(target_id: &str, expectation: &str, evidence: String, passed: bool) -> Self {
        Self {
            target_id: csv_token(target_id),
            expectation: csv_token(expectation),
            evidence: csv_token(&evidence),
            passed,
        }
    }

    pub fn to_text_line(&self) -> String {
        format!(
            "{} passed={} evidence={} expected={}",
            self.target_id, self.passed, self.evidence, self.expectation
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct TireViewerTargetCoverageReport {
    pub checks: Vec<TireViewerTargetCoverageCheck>,
}

impl TireViewerTargetCoverageReport {
    pub fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|check| check.passed)
    }

    pub fn from_run(run: &TireRigRunOutput) -> Self {
        let snapshot = |id: &str| {
            run.snapshots
                .iter()
                .find(|snapshot| snapshot.scenario_id == id)
        };
        let report_passed = |id: &str| {
            run.reports
                .iter()
                .filter(|report| report.snapshot.scenario_id == id)
                .any(TireRigAcceptanceReport::passed)
        };
        let check_passed = |id: &str, check_id: &str| {
            run.reports
                .iter()
                .filter(|report| report.snapshot.scenario_id == id)
                .flat_map(|report| report.checks.iter())
                .any(|check| check.check_id == check_id && check.passed)
        };
        let case = |id: &str| {
            run.vehicle_sanity_cases
                .iter()
                .find(|case| case.scenario_id == id)
        };

        let rolling_approach = snapshot("rolling_obstacle_approach");
        let rolling_crest = snapshot("rolling_obstacle_crest");
        let rolling_exit = snapshot("rolling_obstacle_exit");
        let curb = snapshot("curb_envelope_contact");
        let bump_or_curb_passed = report_passed("rolling_obstacle_sequence")
            && rolling_approach.is_some()
            && rolling_crest.is_some()
            && rolling_exit.is_some()
            && curb.is_some();
        let bump_evidence = format!(
            "rolling_sequence={} approach={} crest={} exit={} curb={}",
            report_passed("rolling_obstacle_sequence"),
            rolling_approach.is_some(),
            rolling_crest.is_some(),
            rolling_exit.is_some(),
            curb.is_some()
        );

        let low_pressure_wrap_passed = rolling_crest.is_some_and(|crest| {
            crest.pressure_pa <= 130_000.0
                && crest.telemetry.patch.contact_count > 1
                && crest.telemetry.patch.patch_length_m > 0.035
        });
        let low_pressure_evidence = rolling_crest
            .map(|crest| {
                format!(
                    "crest_pressure_pa={:.1} contacts={} patch_m={:.5}",
                    crest.pressure_pa,
                    crest.telemetry.patch.contact_count,
                    crest.telemetry.patch.patch_length_m
                )
            })
            .unwrap_or_else(|| "missing_rolling_obstacle_crest".to_owned());

        let ice_transition = snapshot("surface_transition_sequence");
        let ice_patch_passed = report_passed("surface_transition_sequence")
            && snapshot("longitudinal_slip_ice").is_some();
        let ice_evidence = format!(
            "surface_transition={} longitudinal_ice={} aggregate_snapshot={}",
            report_passed("surface_transition_sequence"),
            snapshot("longitudinal_slip_ice").is_some(),
            ice_transition.is_some()
        );

        let offroad = case("offroad_soft_soil_vehicle");
        let soil_rut_passed = offroad.is_some_and(TireVehicleSanityCase::acceptance_passed)
            && (report_passed("road_condition_state")
                || report_passed("rolling_loss_drag")
                || snapshot("loose_soil_slip").is_some());
        let soil_evidence = format!(
            "offroad_case={} road_condition_report={} rolling_loss_report={} loose_soil_snapshot={}",
            offroad
                .map(TireVehicleSanityCase::acceptance_passed)
                .unwrap_or(false),
            report_passed("road_condition_state"),
            report_passed("rolling_loss_drag"),
            snapshot("loose_soil_slip").is_some()
        );

        let pressure_patch = snapshot("pressure_patch_envelope");
        let pressure_patch_passed = report_passed("pressure_patch_envelope")
            && check_passed("pressure_patch_envelope", "lower_pressure_grows_patch")
            && pressure_patch.is_some_and(|snapshot| {
                snapshot.telemetry.patch.contact_count >= 2
                    && snapshot.telemetry.patch.patch_length_m > 0.0
            });
        let pressure_patch_evidence = pressure_patch
            .map(|snapshot| {
                format!(
                    "pressure_patch_report={} lower_pressure_gate={} contacts={} patch_m={:.5}",
                    report_passed("pressure_patch_envelope"),
                    check_passed("pressure_patch_envelope", "lower_pressure_grows_patch"),
                    snapshot.telemetry.patch.contact_count,
                    snapshot.telemetry.patch.patch_length_m
                )
            })
            .unwrap_or_else(|| "missing_pressure_patch_envelope".to_owned());

        let torque_slip = snapshot("torque_induced_slip_outcome");
        let torque_slip_passed = report_passed("torque_induced_slip_outcome")
            && check_passed(
                "torque_induced_slip_outcome",
                "torque_slip_high_pressure_slips_more",
            )
            && check_passed(
                "torque_induced_slip_outcome",
                "torque_slip_partial_patch_larger",
            );
        let torque_slip_evidence = format!(
            "torque_snapshot={} high_pressure_slips_more={} partial_patch_larger={} partial_moves_farther={}",
            torque_slip.is_some(),
            check_passed(
                "torque_induced_slip_outcome",
                "torque_slip_high_pressure_slips_more"
            ),
            check_passed(
                "torque_induced_slip_outcome",
                "torque_slip_partial_patch_larger"
            ),
            check_passed(
                "torque_induced_slip_outcome",
                "torque_slip_partial_pressure_moves_farther"
            )
        );

        let ice_large_patch = snapshot("ice_large_patch_grip");
        let ice_large_patch_passed = report_passed("ice_large_patch_grip")
            && check_passed("ice_large_patch_grip", "ice_large_patch_still_caps_force")
            && check_passed("ice_large_patch_grip", "ice_large_patch_sliding_not_hidden");
        let ice_large_patch_evidence = ice_large_patch
            .map(|snapshot| {
                format!(
                    "ice_large_patch_report={} cap_gate={} sliding_gate={} contacts={} sliding_fraction={:.5}",
                    report_passed("ice_large_patch_grip"),
                    check_passed("ice_large_patch_grip", "ice_large_patch_still_caps_force"),
                    check_passed("ice_large_patch_grip", "ice_large_patch_sliding_not_hidden"),
                    snapshot.telemetry.patch.contact_count,
                    snapshot.telemetry.patch.sliding_fraction
                )
            })
            .unwrap_or_else(|| "missing_ice_large_patch_grip".to_owned());

        let thermal = snapshot("thermal_wear_cycle");
        let thermal_wear_passed = report_passed("thermal_wear_cycle")
            && check_passed("thermal_wear_cycle", "thermal_condition_wear_non_healing")
            && thermal.is_some_and(|snapshot| {
                snapshot.surface_temperature_c.is_some()
                    && snapshot.tread_temperature_c.is_some()
                    && snapshot.carcass_temperature_c.is_some()
                    && snapshot.tread_depth_m.is_some()
            });
        let thermal_wear_evidence = thermal
            .map(|snapshot| {
                format!(
                    "thermal_report={} wear_non_healing={} surface_c_present={} tread_depth_present={}",
                    report_passed("thermal_wear_cycle"),
                    check_passed("thermal_wear_cycle", "thermal_condition_wear_non_healing"),
                    snapshot.surface_temperature_c.is_some(),
                    snapshot.tread_depth_m.is_some()
                )
            })
            .unwrap_or_else(|| "missing_thermal_wear_cycle".to_owned());

        Self {
            checks: vec![
                TireViewerTargetCoverageCheck::new(
                    "rolling_over_bump_curb",
                    "viewer/replay evidence should include a tire rolling over a bump or curb with ordered approach/crest/exit and static curb support",
                    bump_evidence,
                    bump_or_curb_passed,
                ),
                TireViewerTargetCoverageCheck::new(
                    "low_pressure_wraps_obstacle",
                    "low-pressure tire evidence should wrap the obstacle with a multi-cell extended patch",
                    low_pressure_evidence,
                    low_pressure_wrap_passed,
                ),
                TireViewerTargetCoverageCheck::new(
                    "ice_patch_then_asphalt",
                    "surface-transition evidence should include asphalt-to-ice behavior with capped ice force",
                    ice_evidence,
                    ice_patch_passed,
                ),
                TireViewerTargetCoverageCheck::new(
                    "soft_soil_rut_compaction",
                    "soft-soil evidence should include off-road support plus rut/rolling-loss/road-condition feedback",
                    soil_evidence,
                    soil_rut_passed,
                ),
                TireViewerTargetCoverageCheck::new(
                    "soft_tire_flattens_and_grows_patch",
                    "viewer/replay evidence should show softer/lower-pressure tires flattening into a larger distributed patch",
                    pressure_patch_evidence,
                    pressure_patch_passed,
                ),
                TireViewerTargetCoverageCheck::new(
                    "overinflated_tire_slips_sooner",
                    "same torque/load evidence should show over-inflated tires slipping more while partial pressure moves farther",
                    torque_slip_evidence,
                    torque_slip_passed,
                ),
                TireViewerTargetCoverageCheck::new(
                    "ice_large_patch_still_slips",
                    "ice evidence should still cap force and expose sliding even when low pressure creates a large patch",
                    ice_large_patch_evidence,
                    ice_large_patch_passed,
                ),
                TireViewerTargetCoverageCheck::new(
                    "thermal_wear_state_visible",
                    "thermal/wear evidence should expose state-derived heat and non-healing tread wear in replay telemetry",
                    thermal_wear_evidence,
                    thermal_wear_passed,
                ),
            ],
        }
    }
}
