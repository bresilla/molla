//! Tetra-FEM tire report builders + supporting fixtures and helpers
//! (hardening envelope, deformation force, contact slip patch,
//! construction sanity, cavity bead/coupling, bead-seat load, sidewall
//! dynamic load + laminate energy, sidewall-laminate coupling /
//! ply-strain XPBD / dihedral-XPBD-effect builders).
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::FlatTerrain;

use super::super::internal::{add, bool_metric, dot, finite_or, scale, sub};
use super::super::{
    RealTireContact, RealWheelState, SurfaceMaterial, TetraBeadSeatDiagnostics,
    TetraBeadSeatLoadCaseReport, TetraCavityBeadSanityReport, TetraCavityCouplingEffectReport,
    TetraElement, TetraSidewallDihedralXpbdEffectReport, TetraSidewallDynamicLoadReport,
    TetraSidewallLaminateCouplingReport, TetraSidewallLaminateSanityReport,
    TetraSidewallPlyStrainXpbdReport, TetraTireFem, TetraTireFemDiagnostics, TetraTireFemParams,
    TetraTireFemResolutionPoint, TetraTreadBeltStretchDiagnostics, TireRigAcceptanceCheck, det3,
    mat_vec, project_sidewall_ply_strain_constraints, tire_nodes_finite,
};

/// Broad hardening envelope for the compact tetra-FEM tire solver.
///
/// This packages the hardening scenarios that used to live only as unit-test
/// assertions: high load at low pressure, high pressure at high road speed,
/// many small substeps, and damping after a finite disturbance. It is still a
/// data-free robustness gate, not a fitted tire validation.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemHardeningEnvelopeReport {
    pub high_load_hub_floor_margin_m: f64,
    pub high_load_patch_width_m: f64,
    pub high_load_max_node_speed_m_per_s: f64,
    pub high_pressure_radius_ratio: f64,
    pub high_pressure_max_node_speed_m_per_s: f64,
    pub long_run_steps: usize,
    pub long_run_hub_deflection_m: f64,
    pub long_run_max_node_speed_m_per_s: f64,
    pub energy_peak_over_initial: f64,
    pub energy_final_over_peak: f64,
    pub max_abs_node_position_m: f64,
    pub finite_outputs: bool,
}

impl TetraFemHardeningEnvelopeReport {
    pub fn is_finite(self) -> bool {
        self.high_load_hub_floor_margin_m.is_finite()
            && self.high_load_patch_width_m.is_finite()
            && self.high_load_max_node_speed_m_per_s.is_finite()
            && self.high_pressure_radius_ratio.is_finite()
            && self.high_pressure_max_node_speed_m_per_s.is_finite()
            && self.long_run_hub_deflection_m.is_finite()
            && self.long_run_max_node_speed_m_per_s.is_finite()
            && self.energy_peak_over_initial.is_finite()
            && self.energy_final_over_peak.is_finite()
            && self.max_abs_node_position_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.high_load_hub_floor_margin_m >= -1.0e-9
            && self.high_load_patch_width_m >= 0.01
            && self.high_load_max_node_speed_m_per_s <= 9.0
            && self.high_pressure_radius_ratio <= 1.35
            && self.high_pressure_max_node_speed_m_per_s <= 12.0
            && self.long_run_steps >= 800
            && self.long_run_hub_deflection_m <= 0.20
            && self.long_run_max_node_speed_m_per_s <= 8.5
            && self.energy_peak_over_initial <= 75.0
            && self.energy_final_over_peak <= 0.95
            && self.max_abs_node_position_m <= 2.0
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_hardening_outputs_finite",
                "all FEM hardening scenarios should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_hardening_high_load_supported",
                "high-load low-pressure tire should not punch through the rim/floor support envelope",
                "high_load_hub_floor_margin_m",
                self.high_load_hub_floor_margin_m,
                -1.0e-9,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_hardening_high_load_patch_exists",
                "high-load low-pressure tire should still form a supported contact patch",
                "high_load_patch_width_m",
                self.high_load_patch_width_m,
                0.01,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_hardening_high_load_speed_bounded",
                "high-load low-pressure tire node speeds should stay inside the realtime envelope",
                "high_load_max_node_speed_m_per_s",
                self.high_load_max_node_speed_m_per_s,
                9.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_hardening_high_pressure_radius_bounded",
                "high-pressure high-speed tire should not balloon unbounded",
                "high_pressure_radius_ratio",
                self.high_pressure_radius_ratio,
                1.35,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_hardening_high_pressure_speed_bounded",
                "high-pressure high-speed tire node speeds should stay bounded",
                "high_pressure_max_node_speed_m_per_s",
                self.high_pressure_max_node_speed_m_per_s,
                12.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_hardening_long_run_steps",
                "long-run hardening scenario should exercise many small FEM substeps",
                "long_run_steps",
                self.long_run_steps as f64,
                800.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_hardening_long_run_deflection_bounded",
                "many-substep hardening scenario should keep hub deflection bounded",
                "long_run_hub_deflection_m",
                self.long_run_hub_deflection_m,
                0.20,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_hardening_energy_peak_bounded",
                "disturbed tire kinetic energy should not explode before damping acts",
                "energy_peak_over_initial",
                self.energy_peak_over_initial,
                75.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_hardening_energy_damps_from_peak",
                "damping should reduce kinetic energy from the disturbed peak",
                "energy_final_over_peak",
                self.energy_final_over_peak,
                0.95,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_hardening_positions_bounded",
                "hardening scenarios should keep node positions inside a broad finite envelope",
                "max_abs_node_position_m",
                self.max_abs_node_position_m,
                1.0,
            ),
        ]
    }
}

/// Non-calibration gate proving FEM deformation shape affects tire force.
///
/// A reusable brush tire can only be believable for scene 122 if the same
/// commanded wheel state produces different contact force when the actual
/// tetra-FEM shape is different. This report compares a stiff/high-pressure
/// and a soft/low-pressure tire after settling under the same load, then
/// evaluates the same drive+cornering command against each deformed shape.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemDeformationForceReport {
    pub high_pressure_pa: f64,
    pub low_pressure_pa: f64,
    pub high_patch_length_m: f64,
    pub low_patch_length_m: f64,
    pub high_contact_count: usize,
    pub low_contact_count: usize,
    pub high_longitudinal_force_n: f64,
    pub low_longitudinal_force_n: f64,
    pub high_lateral_force_n: f64,
    pub low_lateral_force_n: f64,
    pub high_normal_load_n: f64,
    pub low_normal_load_n: f64,
    pub force_delta_n: f64,
    pub normal_load_ratio: f64,
    pub finite_outputs: bool,
}

impl TetraFemDeformationForceReport {
    pub fn is_finite(self) -> bool {
        self.high_pressure_pa.is_finite()
            && self.low_pressure_pa.is_finite()
            && self.high_patch_length_m.is_finite()
            && self.low_patch_length_m.is_finite()
            && self.high_longitudinal_force_n.is_finite()
            && self.low_longitudinal_force_n.is_finite()
            && self.high_lateral_force_n.is_finite()
            && self.low_lateral_force_n.is_finite()
            && self.high_normal_load_n.is_finite()
            && self.low_normal_load_n.is_finite()
            && self.force_delta_n.is_finite()
            && self.normal_load_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.high_pressure_pa > self.low_pressure_pa
            && self.high_contact_count >= 2
            && self.low_contact_count >= 2
            && self.low_patch_length_m > self.high_patch_length_m + 1.0e-4
            && self.force_delta_n > 50.0
            && self.normal_load_ratio >= 0.85
            && self.normal_load_ratio <= 1.15
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_deformation_force_outputs_finite",
                "FEM deformation-force report should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_deformation_force_pressure_order",
                "FEM deformation-force report should compare high and low pressure tires",
                "high_minus_low_pressure_pa",
                self.high_pressure_pa - self.low_pressure_pa,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_deformation_force_contacts_present",
                "both deformed FEM shapes should form distributed contact",
                "min_contact_count",
                self.high_contact_count.min(self.low_contact_count) as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_deformation_force_low_pressure_longer_patch",
                "low-pressure FEM shape should produce a longer contact patch",
                "low_minus_high_patch_length_m",
                self.low_patch_length_m - self.high_patch_length_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_deformation_force_shape_changes_force",
                "same wheel command should produce different force when FEM deformation changes",
                "force_delta_n",
                self.force_delta_n,
                50.0,
            ),
            TireRigAcceptanceCheck::between(
                "fem_deformation_force_loads_comparable",
                "force-shape comparison should not be only a load mismatch",
                "normal_load_ratio",
                self.normal_load_ratio,
                0.85,
                1.15,
            ),
        ]
    }
}

/// Non-calibration gate for FEM contact-patch/sliding diagnostics.
///
/// This packages two plan-D regression requirements into the standard runner:
/// a lower-pressure deformed tire should form a longer/wider actual tread-node
/// patch, and the sliding/bristle contact diagnostics should grow when the
/// same FEM shape is driven with high slip.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemContactSlipPatchReport {
    pub low_pressure_pa: f64,
    pub high_pressure_pa: f64,
    pub low_patch_length_m: f64,
    pub high_patch_length_m: f64,
    pub low_lateral_loaded_bins: usize,
    pub high_lateral_loaded_bins: usize,
    pub rolling_contact_count: usize,
    pub slipping_contact_count: usize,
    pub rolling_sliding_count: usize,
    pub slipping_sliding_count: usize,
    pub rolling_sliding_fraction: f64,
    pub slipping_sliding_fraction: f64,
    pub rolling_dissipated_power_w: f64,
    pub slipping_dissipated_power_w: f64,
    pub force_budget_ratio: f64,
    pub finite_outputs: bool,
}

impl TetraFemContactSlipPatchReport {
    pub fn is_finite(self) -> bool {
        self.low_pressure_pa.is_finite()
            && self.high_pressure_pa.is_finite()
            && self.low_patch_length_m.is_finite()
            && self.high_patch_length_m.is_finite()
            && self.rolling_sliding_fraction.is_finite()
            && self.slipping_sliding_fraction.is_finite()
            && self.rolling_dissipated_power_w.is_finite()
            && self.slipping_dissipated_power_w.is_finite()
            && self.force_budget_ratio.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.high_pressure_pa > self.low_pressure_pa
            && self.low_patch_length_m > self.high_patch_length_m + 1.0e-4
            && self.low_lateral_loaded_bins >= self.high_lateral_loaded_bins
            && self.low_lateral_loaded_bins >= 2
            && self.rolling_contact_count >= 2
            && self.slipping_contact_count >= 2
            && self.slipping_sliding_count > self.rolling_sliding_count
            && self.slipping_sliding_fraction > self.rolling_sliding_fraction + 0.05
            && self.slipping_dissipated_power_w > self.rolling_dissipated_power_w + 1.0
            && self.force_budget_ratio <= 1.05
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_contact_slip_patch_outputs_finite",
                "FEM contact slip/patch report should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_slip_patch_pressure_order",
                "FEM contact patch report should compare low and high pressure tires",
                "high_minus_low_pressure_pa",
                self.high_pressure_pa - self.low_pressure_pa,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_slip_patch_low_pressure_longer",
                "low-pressure deformed FEM tire should produce a longer actual tread-node patch",
                "low_minus_high_patch_length_m",
                self.low_patch_length_m - self.high_patch_length_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_slip_patch_width_supported",
                "low-pressure tread-node support should span multiple widthwise pressure bins",
                "low_lateral_loaded_bins",
                self.low_lateral_loaded_bins as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_slip_patch_contacts_present",
                "rolling and slipping FEM contact samples should both be distributed",
                "min_contact_count",
                self.rolling_contact_count.min(self.slipping_contact_count) as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_slip_patch_sliding_count_grows",
                "sliding FEM contact-cell count should grow with commanded slip",
                "slipping_minus_rolling_sliding_count",
                self.slipping_sliding_count as f64 - self.rolling_sliding_count as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_slip_patch_sliding_fraction_grows",
                "sliding FEM contact fraction should grow with commanded slip",
                "slipping_minus_rolling_sliding_fraction",
                self.slipping_sliding_fraction - self.rolling_sliding_fraction,
                0.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_contact_slip_patch_power_grows",
                "slipping FEM contact should dissipate more brush power than rolling contact",
                "slipping_minus_rolling_power_w",
                self.slipping_dissipated_power_w - self.rolling_dissipated_power_w,
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_contact_slip_patch_friction_budget_bounded",
                "slipping FEM contact should stay inside the shared friction budget",
                "force_budget_ratio",
                self.force_budget_ratio,
                1.05,
            ),
        ]
    }
}

/// Construction sanity gate for the current tetra-FEM tire approximation.
///
/// This combines the already-separate cavity/bead and sidewall checks with a
/// loaded-shape pass that looks at belt stretch, loaded radius, contact patch
/// shape, and tread support before drive/brake force claims are trusted.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraConstructionSanityReport {
    pub low_pressure_pa: f64,
    pub high_pressure_pa: f64,
    pub low_loaded_radius_m: f64,
    pub high_loaded_radius_m: f64,
    pub low_patch_length_m: f64,
    pub high_patch_length_m: f64,
    pub low_contact_count: usize,
    pub lateral_loaded_bin_count: usize,
    pub belt_stretch: TetraTreadBeltStretchDiagnostics,
    pub max_sidewall_axial_bulge_m: f64,
    pub max_bead_mount_error_m: f64,
    pub max_sidewall_fold_drift_rad: f64,
    pub sidewall_fold_limit_rad: f64,
}

impl TetraConstructionSanityReport {
    pub fn is_finite(self) -> bool {
        self.low_pressure_pa.is_finite()
            && self.high_pressure_pa.is_finite()
            && self.low_loaded_radius_m.is_finite()
            && self.high_loaded_radius_m.is_finite()
            && self.low_patch_length_m.is_finite()
            && self.high_patch_length_m.is_finite()
            && self.belt_stretch.is_finite()
            && self.max_sidewall_axial_bulge_m.is_finite()
            && self.max_bead_mount_error_m.is_finite()
            && self.max_sidewall_fold_drift_rad.is_finite()
            && self.sidewall_fold_limit_rad.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.high_pressure_pa > self.low_pressure_pa
            && self.low_loaded_radius_m.min(self.high_loaded_radius_m) >= 0.20
            && self.low_loaded_radius_m.max(self.high_loaded_radius_m) <= 0.38
            && self.low_patch_length_m > self.high_patch_length_m + 1.0e-4
            && self.low_contact_count >= 4
            && self.lateral_loaded_bin_count >= 2
            && self.belt_stretch.mean_relative_error <= 0.22
            && self.belt_stretch.max_relative_error <= 0.45
            && self.max_sidewall_axial_bulge_m <= 0.055
            && self.max_bead_mount_error_m <= 0.050
            && self.max_sidewall_fold_drift_rad <= self.sidewall_fold_limit_rad
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "construction_pressure_order",
                "construction sanity should compare a lower-pressure and higher-pressure tire",
                "high_minus_low_pressure_pa",
                self.high_pressure_pa - self.low_pressure_pa,
                1.0,
            ),
            TireRigAcceptanceCheck::between(
                "construction_loaded_radius_bounded",
                "loaded radius should stay in a plausible finite construction envelope",
                "min_loaded_radius_m",
                self.low_loaded_radius_m.min(self.high_loaded_radius_m),
                0.20,
                0.38,
            ),
            TireRigAcceptanceCheck::at_least(
                "construction_low_pressure_patch_grows",
                "lower pressure should grow the loaded FEM footprint length",
                "low_minus_high_patch_length_m",
                self.low_patch_length_m - self.high_patch_length_m,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "construction_tread_support_multi_cell",
                "loaded tread support should span multiple contact cells",
                "low_contact_count",
                self.low_contact_count as f64,
                4.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "construction_tread_support_widthwise",
                "loaded tread support should span multiple widthwise pressure bins",
                "lateral_loaded_bin_count",
                self.lateral_loaded_bin_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "construction_belt_stretch_bounded",
                "tread-belt structural edges should keep mean stretch bounded",
                "belt_mean_relative_error",
                self.belt_stretch.mean_relative_error,
                0.22,
            ),
            TireRigAcceptanceCheck::at_most(
                "construction_belt_peak_stretch_bounded",
                "tread-belt structural edges should not hide one extreme local stretch",
                "belt_max_relative_error",
                self.belt_stretch.max_relative_error,
                0.45,
            ),
            TireRigAcceptanceCheck::at_most(
                "construction_sidewall_bulge_bounded",
                "sidewall axial bulge should remain finite and bounded before traction claims are trusted",
                "max_sidewall_axial_bulge_m",
                self.max_sidewall_axial_bulge_m,
                0.055,
            ),
            TireRigAcceptanceCheck::at_most(
                "construction_bead_mount_error_bounded",
                "bead/rim mount drift should stay inside the construction sanity envelope",
                "max_bead_mount_error_m",
                self.max_bead_mount_error_m,
                0.050,
            ),
            TireRigAcceptanceCheck::at_most(
                "construction_sidewall_fold_bounded",
                "sidewall fold drift should stay inside the construction sanity envelope",
                "max_sidewall_fold_drift_rad",
                self.max_sidewall_fold_drift_rad,
                self.sidewall_fold_limit_rad,
            ),
        ]
    }
}

fn tetra_tread_belt_stretch_diagnostics(tire: &TetraTireFem) -> TetraTreadBeltStretchDiagnostics {
    if tire.tread_belt_edges.is_empty() {
        return TetraTreadBeltStretchDiagnostics::default();
    }
    let mut sum = 0.0_f64;
    let mut max_error = 0.0_f64;
    let mut count = 0usize;
    for edge in &tire.tread_belt_edges {
        let [a, b] = edge.ids;
        let delta = sub(tire.pos[b], tire.pos[a]);
        let len = dot(delta, delta).sqrt();
        let error = (len - edge.rest_length).abs() / edge.rest_length.max(1.0e-9);
        if error.is_finite() {
            sum += error;
            max_error = max_error.max(error);
            count += 1;
        }
    }
    if count == 0 {
        return TetraTreadBeltStretchDiagnostics {
            edge_count: tire.tread_belt_edges.len(),
            ..Default::default()
        };
    }
    TetraTreadBeltStretchDiagnostics {
        edge_count: count,
        mean_relative_error: sum / count as f64,
        max_relative_error: max_error,
    }
}

fn tetra_sidewall_axial_bulge_m(tire: &TetraTireFem) -> f64 {
    let n = tire.params.segments.max(1);
    let nr = tire.params.radial_layers.max(2);
    let nw = tire.params.width_layers.max(2);
    let idx = |i: usize, r: usize, w: usize| -> usize { ((i % n) * nr + r) * nw + w };
    let mut max_bulge = 0.0_f64;
    for i in 0..n {
        for r in 1..nr.saturating_sub(1).max(1) {
            for &w in &[0, nw - 1] {
                let id = idx(i, r.min(nr - 1), w);
                let rest_abs_z = tire.rest[id][2].abs();
                let pos_abs_z = tire.pos[id][2].abs();
                max_bulge = max_bulge.max((pos_abs_z - rest_abs_z).max(0.0));
            }
        }
    }
    max_bulge
}

pub fn tetra_construction_sanity_report() -> TetraConstructionSanityReport {
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let material = SurfaceMaterial::dry_asphalt();
    let load_n = 4_200.0;
    let make_tire = |pressure_pa: f64| {
        let params = TetraTireFemParams::realtime_viewer()
            .with_resolution(12, 4, 5)
            .with_pressure_and_load(pressure_pa, load_n);
        TetraTireFem::new(0.0, params.pressure_pa, params)
    };
    let settle = |tire: &mut TetraTireFem| {
        for _ in 0..60 {
            tire.step_on_terrain(&terrain, 1.0 / 240.0);
        }
    };
    let mut low = make_tire(95_000.0);
    let mut high = make_tire(245_000.0);
    settle(&mut low);
    settle(&mut high);
    let contact_for = |tire: &TetraTireFem| {
        let state = RealWheelState {
            hub_pos: tire.hub(),
            hub_vel: [4.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: 4.0 / tire.outer_radius().max(1.0e-9),
        };
        tire.evaluate_tread_contact(&state, &terrain, &material)
    };
    let low_contact = contact_for(&low);
    let high_contact = contact_for(&high);
    let lateral_loaded_bin_count = low_contact
        .pressure_distribution(6, 5)
        .lateral_bins
        .iter()
        .filter(|bin| bin.contact_count > 0)
        .count();
    let low_diagnostics = low.diagnostics();
    let high_diagnostics = high.diagnostics();
    let low_sidewall = low.sidewall_laminate_sanity_report();
    let high_sidewall = high.sidewall_laminate_sanity_report();

    let low_belt = tetra_tread_belt_stretch_diagnostics(&low);
    let high_belt = tetra_tread_belt_stretch_diagnostics(&high);
    TetraConstructionSanityReport {
        low_pressure_pa: low.pressure(),
        high_pressure_pa: high.pressure(),
        low_loaded_radius_m: low.hub()[1],
        high_loaded_radius_m: high.hub()[1],
        low_patch_length_m: low_contact.patch_length,
        high_patch_length_m: high_contact.patch_length,
        low_contact_count: low_contact.contact_count,
        lateral_loaded_bin_count,
        belt_stretch: TetraTreadBeltStretchDiagnostics {
            edge_count: low_belt.edge_count.min(high_belt.edge_count),
            mean_relative_error: low_belt
                .mean_relative_error
                .max(high_belt.mean_relative_error),
            max_relative_error: low_belt
                .max_relative_error
                .max(high_belt.max_relative_error),
        },
        max_sidewall_axial_bulge_m: tetra_sidewall_axial_bulge_m(&low)
            .max(tetra_sidewall_axial_bulge_m(&high)),
        max_bead_mount_error_m: low_diagnostics
            .max_bead_mount_error_m
            .max(high_diagnostics.max_bead_mount_error_m),
        max_sidewall_fold_drift_rad: low_sidewall
            .dihedral
            .max_abs_deviation_rad
            .max(high_sidewall.dihedral.max_abs_deviation_rad),
        sidewall_fold_limit_rad: low_sidewall
            .fold_max_limit_rad
            .min(high_sidewall.fold_max_limit_rad),
    }
}

pub fn tetra_fem_deformation_force_report() -> TetraFemDeformationForceReport {
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let material = SurfaceMaterial::dry_asphalt();
    let load_n = 5_000.0;
    let settle_and_contact = |pressure_pa: f64| -> RealTireContact {
        let mut tire = TetraTireFem::new(
            0.0,
            pressure_pa,
            TetraTireFemParams {
                segments: 20,
                radial_layers: 3,
                width_layers: 3,
                pressure_pa,
                downforce: load_n,
                ..Default::default()
            },
        );
        for _ in 0..420 {
            tire.step_on_terrain(&terrain, 1.0 / 240.0);
        }
        let state = RealWheelState {
            hub_pos: tire.hub(),
            hub_vel: [5.0, 0.0, 0.8],
            heading: [1.0, 0.0, 0.0],
            omega: (5.0 + 2.0) / tire.outer_radius().max(1.0e-9),
        };
        tire.evaluate_tread_contact(&state, &terrain, &material)
    };

    let high_pressure_pa = 330_000.0;
    let low_pressure_pa = 80_000.0;
    let high = settle_and_contact(high_pressure_pa);
    let low = settle_and_contact(low_pressure_pa);
    let force_delta_n = (low.longitudinal_force - high.longitudinal_force).abs()
        + (low.lateral_force - high.lateral_force).abs();
    let normal_load_ratio = low.normal_load / high.normal_load.max(1.0e-9);
    let finite_outputs = high.is_finite() && low.is_finite() && high.in_contact && low.in_contact;

    TetraFemDeformationForceReport {
        high_pressure_pa,
        low_pressure_pa,
        high_patch_length_m: high.patch_length,
        low_patch_length_m: low.patch_length,
        high_contact_count: high.contact_count,
        low_contact_count: low.contact_count,
        high_longitudinal_force_n: high.longitudinal_force,
        low_longitudinal_force_n: low.longitudinal_force,
        high_lateral_force_n: high.lateral_force,
        low_lateral_force_n: low.lateral_force,
        high_normal_load_n: high.normal_load,
        low_normal_load_n: low.normal_load,
        force_delta_n,
        normal_load_ratio,
        finite_outputs,
    }
}

pub fn tetra_fem_contact_slip_patch_report() -> TetraFemContactSlipPatchReport {
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let material = SurfaceMaterial::dry_asphalt();
    let load_n = 4_500.0;
    let settle_tire = |pressure_pa: f64| -> TetraTireFem {
        let mut tire = TetraTireFem::new(
            0.0,
            pressure_pa,
            TetraTireFemParams::realtime_viewer()
                .with_resolution(16, 4, 5)
                .with_pressure_and_load(pressure_pa, load_n),
        );
        for _ in 0..180 {
            tire.step_on_terrain(&terrain, 1.0 / 240.0);
        }
        tire
    };
    let wheel_contact = |tire: &TetraTireFem, speed_m_per_s: f64, slip_ratio: f64| {
        let state = RealWheelState {
            hub_pos: tire.hub(),
            hub_vel: [speed_m_per_s, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip_ratio) / tire.outer_radius().max(1.0e-9),
        };
        tire.evaluate_tread_contact(&state, &terrain, &material)
    };
    let low_pressure_pa = 90_000.0;
    let high_pressure_pa = 260_000.0;
    let low = settle_tire(low_pressure_pa);
    let high = settle_tire(high_pressure_pa);
    let low_roll = wheel_contact(&low, 5.0, 0.02);
    let high_roll = wheel_contact(&high, 5.0, 0.02);
    let rolling = wheel_contact(&low, 5.0, 0.02);
    let slipping = wheel_contact(&low, 5.0, 0.34);
    let low_lateral_loaded_bins = low_roll
        .pressure_distribution(6, 5)
        .lateral_bins
        .iter()
        .filter(|bin| bin.contact_count > 0)
        .count();
    let high_lateral_loaded_bins = high_roll
        .pressure_distribution(6, 5)
        .lateral_bins
        .iter()
        .filter(|bin| bin.contact_count > 0)
        .count();
    let force_budget_ratio =
        slipping.consumed_friction_budget / slipping.available_friction_budget.max(1.0);
    let finite_outputs = low_roll.is_finite()
        && high_roll.is_finite()
        && rolling.is_finite()
        && slipping.is_finite()
        && low_roll.in_contact
        && high_roll.in_contact
        && rolling.in_contact
        && slipping.in_contact;
    TetraFemContactSlipPatchReport {
        low_pressure_pa,
        high_pressure_pa,
        low_patch_length_m: low_roll.patch_length,
        high_patch_length_m: high_roll.patch_length,
        low_lateral_loaded_bins,
        high_lateral_loaded_bins,
        rolling_contact_count: rolling.contact_count,
        slipping_contact_count: slipping.contact_count,
        rolling_sliding_count: rolling.sliding_count,
        slipping_sliding_count: slipping.sliding_count,
        rolling_sliding_fraction: rolling.sliding_fraction,
        slipping_sliding_fraction: slipping.sliding_fraction,
        rolling_dissipated_power_w: rolling.dissipated_power,
        slipping_dissipated_power_w: slipping.dissipated_power,
        force_budget_ratio,
        finite_outputs,
    }
}

/// Build a deterministic tetra-FEM resolution sweep without changing physics
/// parameters.
///
/// The sweep is a guardrail for the coarse-resolution approximation: before
/// raising default viewer density, tests and reports can verify that node/tet
/// counts scale predictably and that all generated tetrahedra/cavity volumes
/// remain finite and positive.
pub fn tire_fem_resolution_sweep(
    base: TetraTireFemParams,
    resolutions: &[(usize, usize, usize)],
) -> Vec<TetraTireFemResolutionPoint> {
    resolutions
        .iter()
        .map(|&(segments, radial_layers, width_layers)| {
            let params = base.with_resolution(segments, radial_layers, width_layers);
            let tire = TetraTireFem::new(0.0, params.pressure_pa, params);
            let mut min_volume = f64::INFINITY;
            let mut max_volume = 0.0_f64;
            let mut sum_volume = 0.0_f64;
            for tet in &tire.tets {
                min_volume = min_volume.min(tet.volume);
                max_volume = max_volume.max(tet.volume);
                sum_volume += tet.volume;
            }
            let mean_volume = if tire.tets.is_empty() {
                0.0
            } else {
                sum_volume / tire.tets.len() as f64
            };
            TetraTireFemResolutionPoint {
                segments: tire.params.segments,
                radial_layers: tire.params.radial_layers,
                width_layers: tire.params.width_layers,
                node_count: tire.pos.len(),
                tetra_count: tire.tets.len(),
                tread_node_count: tire.tread_nodes.len(),
                bead_node_count: tire.bead_offsets.len(),
                rest_cavity_volume_m3: tire.cavity_volume_rest,
                min_tet_volume_m3: min_volume,
                max_tet_volume_m3: max_volume,
                mean_tet_volume_m3: mean_volume,
            }
        })
        .collect()
}

/// Build a deterministic cavity/bead sanity report for the simplified
/// polytropic cavity + XPBD bead/rim guardrail.
pub fn tetra_cavity_bead_sanity_report() -> TetraCavityBeadSanityReport {
    let base = TetraTireFemParams::realtime_viewer()
        .with_resolution(16, 4, 4)
        .with_pressure_and_load(155_000.0, 4_500.0);
    let low_pressure_pa = 90_000.0;
    let high_pressure_pa = 280_000.0;
    let low = TetraTireFem::new(
        0.0,
        low_pressure_pa,
        TetraTireFemParams {
            pressure_pa: low_pressure_pa,
            ..base
        },
    )
    .diagnostics();
    let high = TetraTireFem::new(
        0.0,
        high_pressure_pa,
        TetraTireFemParams {
            pressure_pa: high_pressure_pa,
            ..base
        },
    )
    .diagnostics();
    let mut loaded = TetraTireFem::new(0.0, base.pressure_pa, base);
    let terrain = FlatTerrain::new(0.0, 1.0);
    let dt = 1.0 / 240.0;
    for _ in 0..8 {
        loaded.step_on_terrain(&terrain, dt);
    }
    let loaded_diag = loaded.diagnostics();
    let bead_seat = loaded.bead_seat_diagnostics();
    TetraCavityBeadSanityReport {
        low_pressure_pa,
        high_pressure_pa,
        low_radial_force_per_bead_n: low.cavity_radial_force_per_bead_n,
        high_radial_force_per_bead_n: high.cavity_radial_force_per_bead_n,
        loaded_cavity_volume_ratio: loaded_diag.cavity_volume_ratio,
        loaded_effective_pressure_pa: loaded_diag.effective_cavity_pressure_pa,
        cavity_carcass_coupling: loaded_diag.cavity_carcass_coupling,
        mean_bead_mount_error_m: loaded_diag.mean_bead_mount_error_m,
        max_bead_mount_error_m: loaded_diag.max_bead_mount_error_m,
        bead_seat,
        last_xpbd_bead_correction_m: loaded_diag.last_xpbd_bead_correction_m,
    }
}

pub fn tetra_cavity_coupling_effect_report() -> TetraCavityCouplingEffectReport {
    let base = TetraTireFemParams::realtime_viewer()
        .with_resolution(16, 4, 4)
        .with_pressure_and_load(180_000.0, 4_200.0);
    let terrain = FlatTerrain::new(0.0, 1.0);
    let couplings = [0.0_f64, base.cavity_carcass_coupling, 0.36_f64];
    let mut samples = Vec::with_capacity(couplings.len());
    let mut finite_outputs = true;
    let mut max_bead_mount_error_m = 0.0_f64;

    for coupling in couplings {
        let params = TetraTireFemParams {
            cavity_carcass_coupling: coupling,
            ..base
        };
        let mut tire = TetraTireFem::new(0.0, params.pressure_pa, params);
        for _ in 0..6 {
            tire.step_on_terrain(&terrain, 1.0 / 480.0);
        }
        let diagnostics = tire.diagnostics();
        finite_outputs &= diagnostics.is_finite()
            && tire_nodes_finite(&tire)
            && (diagnostics.cavity_carcass_coupling - coupling).abs() < 1.0e-12;
        max_bead_mount_error_m = max_bead_mount_error_m.max(diagnostics.max_bead_mount_error_m);
        samples.push((coupling, diagnostics.cavity_radial_force_per_bead_n));
    }

    let (min_coupling, zero_force_per_bead_n) = samples.first().copied().expect("fixed samples");
    let (nominal_coupling, nominal_force_per_bead_n) =
        samples.get(1).copied().expect("fixed samples");
    let (max_coupling, high_force_per_bead_n) = samples.last().copied().expect("fixed samples");

    TetraCavityCouplingEffectReport {
        sample_count: samples.len(),
        min_coupling,
        nominal_coupling,
        max_coupling,
        zero_force_per_bead_n,
        nominal_force_per_bead_n,
        high_force_per_bead_n,
        high_over_nominal_force: high_force_per_bead_n / nominal_force_per_bead_n.max(1.0e-9),
        max_bead_mount_error_m,
        finite_outputs,
    }
}

fn tetra_bead_seat_load_case_sample(
    pressure_pa: f64,
    downforce_n: f64,
    lateral_speed_m_per_s: f64,
    slip_ratio: f64,
) -> (
    TetraTireFemDiagnostics,
    TetraBeadSeatDiagnostics,
    RealTireContact,
) {
    let params = TetraTireFemParams::realtime_viewer()
        .with_resolution(16, 4, 4)
        .with_pressure_and_load(pressure_pa, downforce_n);
    let mut tire = TetraTireFem::new(0.0, pressure_pa, params);
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let dt = 1.0 / 240.0;
    let speed_m_per_s = 4.0;
    let slip_ratio = finite_or(slip_ratio, 0.06).clamp(-0.60, 0.60);
    let mut contact = RealTireContact::empty();
    for _ in 0..18 {
        tire.step_on_terrain(&terrain, dt);
        let state = RealWheelState {
            hub_pos: tire.hub(),
            hub_vel: [speed_m_per_s, 0.0, lateral_speed_m_per_s],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip_ratio) / tire.outer_radius().max(1.0e-6),
        };
        contact = tire.apply_tread_contact_wrench(&state, &terrain, &material, dt);
    }
    (tire.diagnostics(), tire.bead_seat_diagnostics(), contact)
}

/// Build a deterministic bead/rim load-case validation report.
pub fn tetra_bead_seat_load_case_report() -> TetraBeadSeatLoadCaseReport {
    let (static_diag, static_seat, static_contact) =
        tetra_bead_seat_load_case_sample(155_000.0, 0.0, 0.0, 0.06);
    let (loaded_diag, loaded_seat, loaded_contact) =
        tetra_bead_seat_load_case_sample(155_000.0, 4_500.0, 0.0, 0.06);
    let (lateral_diag, lateral_seat, lateral_contact) =
        tetra_bead_seat_load_case_sample(155_000.0, 4_500.0, 2.2, 0.06);
    let (torque_diag, torque_seat, torque_contact) =
        tetra_bead_seat_load_case_sample(155_000.0, 4_500.0, 0.0, 0.32);
    let (high_diag, high_seat, high_contact) =
        tetra_bead_seat_load_case_sample(320_000.0, 4_500.0, 0.0, 0.06);
    let finite_outputs = static_diag.is_finite()
        && loaded_diag.is_finite()
        && lateral_diag.is_finite()
        && torque_diag.is_finite()
        && high_diag.is_finite()
        && static_seat.is_finite()
        && loaded_seat.is_finite()
        && lateral_seat.is_finite()
        && torque_seat.is_finite()
        && high_seat.is_finite()
        && static_contact.is_finite()
        && loaded_contact.is_finite()
        && lateral_contact.is_finite()
        && torque_contact.is_finite()
        && high_contact.is_finite();
    TetraBeadSeatLoadCaseReport {
        bead_node_count: lateral_seat
            .bead_node_count
            .max(torque_seat.bead_node_count)
            .max(loaded_seat.bead_node_count)
            .max(static_seat.bead_node_count)
            .max(high_seat.bead_node_count),
        static_max_mount_error_m: static_diag.max_bead_mount_error_m,
        loaded_max_mount_error_m: loaded_diag.max_bead_mount_error_m,
        lateral_max_mount_error_m: lateral_diag.max_bead_mount_error_m,
        torque_max_mount_error_m: torque_diag.max_bead_mount_error_m,
        high_pressure_max_mount_error_m: high_diag.max_bead_mount_error_m,
        lateral_max_abs_radial_error_m: lateral_seat.max_abs_radial_error_m,
        lateral_max_abs_axial_error_m: lateral_seat.max_abs_axial_error_m,
        lateral_max_tangential_error_m: lateral_seat.max_tangential_error_m,
        torque_max_tangential_error_m: torque_seat.max_tangential_error_m,
        lateral_contact_force_n: lateral_contact.lateral_force.abs(),
        torque_contact_force_n: torque_contact.longitudinal_force.abs(),
        high_pressure_effective_pressure_pa: high_diag.effective_cavity_pressure_pa,
        finite_outputs,
    }
}

fn sidewall_dynamic_load_case(
    lateral_speed_m_per_s: f64,
    heading_yaw_rad: f64,
) -> (TetraSidewallLaminateSanityReport, RealTireContact) {
    let params = TetraTireFemParams::realtime_viewer()
        .with_resolution(20, 5, 5)
        .with_pressure_and_load(155_000.0, 4_500.0);
    let mut tire = TetraTireFem::new(0.0, params.pressure_pa, params);
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let speed_m_per_s = 4.5;
    let heading_yaw_rad = finite_or(heading_yaw_rad, 0.0).clamp(-0.7, 0.7);
    let heading = [heading_yaw_rad.cos(), 0.0, heading_yaw_rad.sin()];
    let dt = 1.0 / 240.0;
    let mut contact = RealTireContact::empty();
    for _ in 0..24 {
        tire.step_on_terrain(&terrain, dt);
        let state = RealWheelState {
            hub_pos: tire.hub(),
            hub_vel: [speed_m_per_s, 0.0, lateral_speed_m_per_s],
            heading,
            omega: speed_m_per_s * 1.04 / tire.outer_radius().max(1.0e-6),
        };
        contact = tire.apply_tread_contact_wrench(&state, &terrain, &material, dt);
    }
    (tire.sidewall_laminate_sanity_report(), contact)
}

/// Build a deterministic vertical/lateral sidewall dynamic-load sanity report.
pub fn tetra_sidewall_dynamic_load_report() -> TetraSidewallDynamicLoadReport {
    let rest_params = TetraTireFemParams::realtime_viewer()
        .with_resolution(20, 5, 5)
        .with_pressure_and_load(155_000.0, 4_500.0);
    let rest = TetraTireFem::new(0.0, rest_params.pressure_pa, rest_params)
        .sidewall_laminate_sanity_report();
    let (vertical, vertical_contact) = sidewall_dynamic_load_case(0.0, 0.0);
    let (lateral, lateral_contact) = sidewall_dynamic_load_case(2.5, 0.0);
    let (steer, steer_contact) = sidewall_dynamic_load_case(0.0, 10.0_f64.to_radians());
    let vertical_max_ply_strain = vertical
        .stretch
        .max_tensile_strain
        .max(vertical.stretch.max_compressive_strain);
    let lateral_max_ply_strain = lateral
        .stretch
        .max_tensile_strain
        .max(lateral.stretch.max_compressive_strain);
    let steer_max_ply_strain = steer
        .stretch
        .max_tensile_strain
        .max(steer.stretch.max_compressive_strain);
    let finite_outputs = rest.is_finite()
        && vertical.is_finite()
        && lateral.is_finite()
        && steer.is_finite()
        && vertical_contact.is_finite()
        && lateral_contact.is_finite()
        && steer_contact.is_finite();
    TetraSidewallDynamicLoadReport {
        rest_report: rest,
        vertical_report: vertical,
        lateral_report: lateral,
        steer_report: steer,
        lateral_contact_force_n: lateral_contact.lateral_force.abs(),
        steer_contact_force_n: steer_contact.lateral_force.abs(),
        steer_aligning_moment_nm: steer_contact.aligning_moment_z,
        vertical_max_fold_drift_rad: vertical.dihedral.max_abs_deviation_rad,
        lateral_max_fold_drift_rad: lateral.dihedral.max_abs_deviation_rad,
        steer_max_fold_drift_rad: steer.dihedral.max_abs_deviation_rad,
        vertical_max_ply_strain,
        lateral_max_ply_strain,
        steer_max_ply_strain,
        finite_outputs,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SidewallLaminateEnergyBreakdown {
    element_count: usize,
    bias_a_energy_j: f64,
    bias_b_energy_j: f64,
    radial_helper_energy_j: f64,
    axial_helper_energy_j: f64,
}

impl SidewallLaminateEnergyBreakdown {
    fn total_energy_j(self) -> f64 {
        self.bias_a_energy_j
            + self.bias_b_energy_j
            + self.radial_helper_energy_j
            + self.axial_helper_energy_j
    }

    fn is_finite(self) -> bool {
        self.bias_a_energy_j.is_finite()
            && self.bias_b_energy_j.is_finite()
            && self.radial_helper_energy_j.is_finite()
            && self.axial_helper_energy_j.is_finite()
    }
}

fn sidewall_laminate_energy_breakdown(tire: &TetraTireFem) -> SidewallLaminateEnergyBreakdown {
    let mut out = SidewallLaminateEnergyBreakdown::default();
    for tet in &tire.tets {
        if tet.fiber_k1 <= 0.0
            || tet.fiber_k1_b <= 0.0
            || tet.fiber_k1_c <= 0.0
            || tet.fiber_k1_d <= 0.0
        {
            continue;
        }
        let centroid = tet
            .ids
            .iter()
            .fold([0.0; 3], |acc, &id| add(acc, scale(tire.rest[id], 0.25)));
        let radial_raw = [centroid[0] - tire.hub[0], centroid[1] - tire.hub[1], 0.0];
        let radial_len = dot(radial_raw, radial_raw).sqrt();
        if !radial_len.is_finite() || radial_len <= 1.0e-9 {
            continue;
        }
        let radial = scale(radial_raw, 1.0 / radial_len);
        let axial = [0.0, 0.0, 1.0];
        let radial_helper = dot(tet.fiber_dir_c, radial).abs();
        let axial_helper = dot(tet.fiber_dir_d, axial).abs();
        if radial_helper < 0.85 || axial_helper < 0.85 {
            continue;
        }
        let mut f_def = [[0.0; 3]; 3];
        for a in 0..4 {
            let id = tet.ids[a];
            for row in 0..3 {
                for col in 0..3 {
                    f_def[row][col] += tire.pos[id][row] * tet.grads[a][col];
                }
            }
        }
        if det3(f_def).abs() <= 1.0e-12 {
            continue;
        }
        out.element_count += 1;
        let ply_energy = |dir: [f64; 3], k1: f64| -> f64 {
            let fa = mat_vec(f_def, dir);
            let strain = dot(fa, fa).sqrt() - 1.0;
            if strain.is_finite() && strain > 0.0 {
                0.5 * k1.max(0.0) * tet.volume * strain * strain
            } else {
                0.0
            }
        };
        out.bias_a_energy_j += ply_energy(tet.fiber_dir, tet.fiber_k1);
        out.bias_b_energy_j += ply_energy(tet.fiber_dir_b, tet.fiber_k1_b);
        out.radial_helper_energy_j += ply_energy(tet.fiber_dir_c, tet.fiber_k1_c);
        out.axial_helper_energy_j += ply_energy(tet.fiber_dir_d, tet.fiber_k1_d);
    }
    out
}

pub(crate) fn tet_is_sidewall_laminate(
    tet: &TetraElement,
    rest: &[[f64; 3]],
    hub: [f64; 3],
) -> bool {
    if tet.fiber_k1 <= 0.0
        || tet.fiber_k1_b <= 0.0
        || tet.fiber_k1_c <= 0.0
        || tet.fiber_k1_d <= 0.0
    {
        return false;
    }
    let centroid = tet
        .ids
        .iter()
        .fold([0.0; 3], |acc, &id| add(acc, scale(rest[id], 0.25)));
    let radial_raw = [centroid[0] - hub[0], centroid[1] - hub[1], 0.0];
    let radial_len = dot(radial_raw, radial_raw).sqrt();
    if !radial_len.is_finite() || radial_len <= 1.0e-9 {
        return false;
    }
    let radial = scale(radial_raw, 1.0 / radial_len);
    let axial = [0.0, 0.0, 1.0];
    dot(tet.fiber_dir_c, radial).abs() >= 0.85 && dot(tet.fiber_dir_d, axial).abs() >= 0.85
}

fn bulge_tire_radially(tire: &mut TetraTireFem, meters: f64) {
    for id in 0..tire.pos.len() {
        let radial_raw = [
            tire.rest[id][0] - tire.hub[0],
            tire.rest[id][1] - tire.hub[1],
            0.0,
        ];
        let len = dot(radial_raw, radial_raw).sqrt();
        if len > 1.0e-9 && len.is_finite() {
            let dir = scale(radial_raw, 1.0 / len);
            tire.pos[id][0] += meters * dir[0];
            tire.pos[id][1] += meters * dir[1];
        }
    }
}

fn stretch_tire_widthwise(tire: &mut TetraTireFem, stretch: f64) {
    let factor = (1.0 + finite_or(stretch, 0.0)).max(0.25);
    for id in 0..tire.pos.len() {
        tire.pos[id][2] = tire.hub[2] + (tire.pos[id][2] - tire.hub[2]) * factor;
    }
}

fn fold_sidewall_fixture(tire: &mut TetraTireFem) {
    let bend_edges = tire.sidewall_bend_edges.clone();
    for (k, edge) in bend_edges.iter().enumerate() {
        if k % 7 == 0 {
            let id = edge.ids[0];
            tire.pos[id][0] += 0.050;
            tire.pos[id][1] -= 0.035;
            tire.pos[id][2] += tire.pos[id][2].signum().max(1.0) * 0.030;
        }
    }
}

/// Build a deterministic sidewall laminate coupling report.
pub fn tetra_sidewall_laminate_coupling_report() -> TetraSidewallLaminateCouplingReport {
    let params = TetraTireFemParams {
        xpbd_sidewall_bend_compliance: f64::INFINITY,
        xpbd_sidewall_dihedral_compliance: 0.0,
        xpbd_structural_iterations: 8,
        ..TetraTireFemParams::realtime_viewer()
            .with_resolution(20, 5, 5)
            .with_pressure_and_load(180_000.0, 3_000.0)
    };
    let base = TetraTireFem::new(0.0, params.pressure_pa, params);

    let mut radial = base.clone();
    bulge_tire_radially(&mut radial, 0.018);
    let radial_energy = sidewall_laminate_energy_breakdown(&radial);

    let mut axial = base.clone();
    stretch_tire_widthwise(&mut axial, 0.08);
    let axial_energy = sidewall_laminate_energy_breakdown(&axial);

    let mut folded = base.clone();
    fold_sidewall_fixture(&mut folded);
    let folded_energy = sidewall_laminate_energy_breakdown(&folded);
    let folded_diag = folded.sidewall_dihedral_diagnostics();

    let mut projected = folded.clone();
    projected.project_structural_constraints(1.0 / 240.0, &FlatTerrain::new(-10.0, 1.0));
    let projected_diag = projected.sidewall_dihedral_diagnostics();

    let mean_bias_energy =
        0.5 * (radial_energy.bias_a_energy_j + radial_energy.bias_b_energy_j).max(1.0e-12);
    let crossed_pair_relative_imbalance =
        (radial_energy.bias_a_energy_j - radial_energy.bias_b_energy_j).abs() / mean_bias_energy;
    let folded_max = folded_diag.max_abs_deviation_rad;
    let projected_max = projected_diag.max_abs_deviation_rad;
    let finite_outputs = radial_energy.is_finite()
        && axial_energy.is_finite()
        && folded_energy.is_finite()
        && folded_diag.is_finite()
        && projected_diag.is_finite();

    TetraSidewallLaminateCouplingReport {
        sidewall_element_count: radial_energy
            .element_count
            .min(axial_energy.element_count)
            .min(folded_energy.element_count),
        balanced_bias_a_energy_j: radial_energy.bias_a_energy_j,
        balanced_bias_b_energy_j: radial_energy.bias_b_energy_j,
        crossed_pair_relative_imbalance,
        radial_helper_energy_j: radial_energy.radial_helper_energy_j,
        axial_helper_energy_j: axial_energy.axial_helper_energy_j,
        folded_ply_energy_j: folded_energy.total_energy_j(),
        folded_max_fold_drift_rad: folded_max,
        projected_max_fold_drift_rad: projected_max,
        projected_fold_ratio: projected_max / folded_max.max(1.0e-12),
        finite_outputs,
    }
}

/// Build a deterministic sidewall ply-strain XPBD effect report.
pub fn tetra_sidewall_ply_strain_xpbd_report() -> TetraSidewallPlyStrainXpbdReport {
    let params = TetraTireFemParams {
        xpbd_sidewall_bend_compliance: f64::INFINITY,
        xpbd_sidewall_dihedral_compliance: f64::INFINITY,
        xpbd_structural_iterations: 0,
        ..TetraTireFemParams::realtime_viewer()
            .with_resolution(20, 5, 5)
            .with_pressure_and_load(180_000.0, 3_000.0)
    };
    let mut tire = TetraTireFem::new(0.0, params.pressure_pa, params);
    bulge_tire_radially(&mut tire, 0.080);
    stretch_tire_widthwise(&mut tire, 0.42);
    fold_sidewall_fixture(&mut tire);
    let pre = tire.sidewall_laminate_stretch_diagnostics();
    let stretch_limit = 1.20;
    let mut max_projection_correction_m: f64 = 0.0;
    for _ in 0..10 {
        max_projection_correction_m =
            max_projection_correction_m.max(project_sidewall_ply_strain_constraints(
                &mut tire.pos,
                &mut tire.vel,
                &tire.mass,
                &tire.tets,
                &tire.rest,
                tire.hub,
                stretch_limit,
                0.0,
                1.0 / 240.0,
            ));
    }
    let post = tire.sidewall_laminate_stretch_diagnostics();
    let finite_outputs =
        pre.is_finite() && post.is_finite() && max_projection_correction_m.is_finite();
    TetraSidewallPlyStrainXpbdReport {
        sidewall_element_count: pre.sidewall_element_count.min(post.sidewall_element_count),
        stretch_limit,
        pre_max_tensile_strain: pre.max_tensile_strain,
        post_max_tensile_strain: post.max_tensile_strain,
        strain_reduction_ratio: post.max_tensile_strain / pre.max_tensile_strain.max(1.0e-12),
        max_projection_correction_m,
        finite_outputs,
    }
}

/// Build a deterministic sidewall dihedral-XPBD effect report.
pub fn tetra_sidewall_dihedral_xpbd_effect_report() -> TetraSidewallDihedralXpbdEffectReport {
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
    tire.project_structural_constraints(1.0 / 240.0, &FlatTerrain::new(-10.0, 1.0));
    let stabilized = tire.sidewall_dihedral_diagnostics();
    let folded_max = folded.max_abs_deviation_rad;
    let stabilized_max = stabilized.max_abs_deviation_rad;
    TetraSidewallDihedralXpbdEffectReport {
        quad_count: folded.quad_count,
        folded_max_abs_deviation_rad: folded_max,
        stabilized_max_abs_deviation_rad: stabilized_max,
        reduction_ratio: stabilized_max / folded_max.max(1.0e-12),
        finite_outputs: rest.is_finite()
            && folded.is_finite()
            && stabilized.is_finite()
            && tire_nodes_finite(&tire),
    }
}
