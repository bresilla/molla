//! `TetraTireFemDiagnostics`, `TetraTireFemResolutionPoint`,
//! `TetraBeadSeat*`, `TetraCavity*Report`, `TetraBeadSeatLoadCaseReport`
//! diagnostic types and builders.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::super::TireRigAcceptanceCheck;
use super::super::internal::bool_metric;

/// Lightweight runtime diagnostics for the volumetric tire FEM.
///
/// These are intentionally model-internal quantities, not calibration targets:
/// they expose whether pressure/cavity/XPBD state is doing sensible work while
/// the viewer and regression rigs are tuned.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraTireFemDiagnostics {
    pub pressure_pa: f64,
    pub cavity_volume_m3: f64,
    pub cavity_rest_volume_m3: f64,
    pub cavity_volume_ratio: f64,
    pub effective_cavity_pressure_pa: f64,
    pub cavity_carcass_coupling: f64,
    pub cavity_radial_force_per_bead_n: f64,
    pub max_tet_volume_error_m3: f64,
    pub mean_bead_mount_error_m: f64,
    pub max_bead_mount_error_m: f64,
    pub last_xpbd_volume_correction_m: f64,
    pub last_xpbd_bead_correction_m: f64,
    pub last_xpbd_structural_correction_m: f64,
}

impl TetraTireFemDiagnostics {
    pub fn is_finite(self) -> bool {
        self.pressure_pa.is_finite()
            && self.cavity_volume_m3.is_finite()
            && self.cavity_rest_volume_m3.is_finite()
            && self.cavity_volume_ratio.is_finite()
            && self.effective_cavity_pressure_pa.is_finite()
            && self.cavity_carcass_coupling.is_finite()
            && self.cavity_radial_force_per_bead_n.is_finite()
            && self.max_tet_volume_error_m3.is_finite()
            && self.mean_bead_mount_error_m.is_finite()
            && self.max_bead_mount_error_m.is_finite()
            && self.last_xpbd_volume_correction_m.is_finite()
            && self.last_xpbd_bead_correction_m.is_finite()
            && self.last_xpbd_structural_correction_m.is_finite()
    }
}

/// One point in a tetra-FEM mesh-resolution sweep.
///
/// This is non-calibration evidence for the coarse-mesh approximation ledger:
/// it records topology size and volume sanity before changing viewer defaults
/// or increasing mesh density. It deliberately avoids target tire data.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraTireFemResolutionPoint {
    pub segments: usize,
    pub radial_layers: usize,
    pub width_layers: usize,
    pub node_count: usize,
    pub tetra_count: usize,
    pub tread_node_count: usize,
    pub bead_node_count: usize,
    pub rest_cavity_volume_m3: f64,
    pub min_tet_volume_m3: f64,
    pub max_tet_volume_m3: f64,
    pub mean_tet_volume_m3: f64,
}

impl TetraTireFemResolutionPoint {
    pub fn is_finite(self) -> bool {
        self.rest_cavity_volume_m3.is_finite()
            && self.min_tet_volume_m3.is_finite()
            && self.max_tet_volume_m3.is_finite()
            && self.mean_tet_volume_m3.is_finite()
    }
}

/// Reduced rim-seat/bead diagnostics for the current XPBD bead guardrail.
///
/// This is not yet a full rim-seat contact solve. It decomposes bead drift into
/// radial, axial, and tangential components in the tire's rest bead frame, so
/// the cavity/bead sanity gate can tell *how* the bead is drifting instead of
/// only reporting one scalar mount error.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraBeadSeatDiagnostics {
    pub bead_node_count: usize,
    pub mean_abs_radial_error_m: f64,
    pub max_abs_radial_error_m: f64,
    pub mean_abs_axial_error_m: f64,
    pub max_abs_axial_error_m: f64,
    pub max_tangential_error_m: f64,
}

impl TetraBeadSeatDiagnostics {
    pub fn is_finite(self) -> bool {
        self.mean_abs_radial_error_m.is_finite()
            && self.max_abs_radial_error_m.is_finite()
            && self.mean_abs_axial_error_m.is_finite()
            && self.max_abs_axial_error_m.is_finite()
            && self.max_tangential_error_m.is_finite()
    }
}

/// Non-calibration sanity report for the simplified cavity / bead mount model.
///
/// This is deliberately **not** external bead-seat calibration. It checks the
/// internal invariants that should hold before the viewer calls the cavity
/// model believable: inflation pressure should create stronger cavity bead
/// loading, the sealed cavity should remain finite under load, and XPBD bead
/// mount projection should keep the bead/rim error bounded.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraCavityBeadSanityReport {
    pub low_pressure_pa: f64,
    pub high_pressure_pa: f64,
    pub low_radial_force_per_bead_n: f64,
    pub high_radial_force_per_bead_n: f64,
    pub loaded_cavity_volume_ratio: f64,
    pub loaded_effective_pressure_pa: f64,
    pub cavity_carcass_coupling: f64,
    pub mean_bead_mount_error_m: f64,
    pub max_bead_mount_error_m: f64,
    pub bead_seat: TetraBeadSeatDiagnostics,
    pub last_xpbd_bead_correction_m: f64,
}

impl TetraCavityBeadSanityReport {
    pub fn is_finite(self) -> bool {
        self.low_pressure_pa.is_finite()
            && self.high_pressure_pa.is_finite()
            && self.low_radial_force_per_bead_n.is_finite()
            && self.high_radial_force_per_bead_n.is_finite()
            && self.loaded_cavity_volume_ratio.is_finite()
            && self.loaded_effective_pressure_pa.is_finite()
            && self.cavity_carcass_coupling.is_finite()
            && self.mean_bead_mount_error_m.is_finite()
            && self.max_bead_mount_error_m.is_finite()
            && self.bead_seat.is_finite()
            && self.last_xpbd_bead_correction_m.is_finite()
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "cavity_pressure_scales_bead_load",
                "higher inflation should increase cavity bead load without fitted data",
                "high_over_low_bead_load",
                self.high_radial_force_per_bead_n
                    / self.low_radial_force_per_bead_n.abs().max(1.0e-9),
                2.0,
            ),
            TireRigAcceptanceCheck::between(
                "loaded_cavity_volume_stays_bounded",
                "loaded cavity should stay finite and not collapse or balloon",
                "loaded_cavity_volume_ratio",
                self.loaded_cavity_volume_ratio,
                0.35,
                2.50,
            ),
            TireRigAcceptanceCheck::between(
                "cavity_carcass_coupling_visible_and_bounded",
                "pressure-to-carcass coupling should be an audited preset parameter, not a hidden solver constant",
                "cavity_carcass_coupling",
                self.cavity_carcass_coupling,
                0.02,
                0.75,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_mount_error_bounded",
                "XPBD bead/rim guardrail should keep bead mount drift bounded",
                "max_bead_mount_error_m",
                self.max_bead_mount_error_m,
                0.050,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_seat_radial_error_bounded",
                "bead/rim seat diagnostics should keep radial drift bounded separately from axial drift",
                "max_abs_radial_error_m",
                self.bead_seat.max_abs_radial_error_m,
                0.035,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_seat_axial_error_bounded",
                "bead/rim seat diagnostics should keep axial seat drift bounded separately from radial drift",
                "max_abs_axial_error_m",
                self.bead_seat.max_abs_axial_error_m,
                0.035,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_seat_tangential_error_bounded",
                "bead/rim seat diagnostics should expose circumferential slip around the rim seat",
                "max_tangential_error_m",
                self.bead_seat.max_tangential_error_m,
                0.035,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_projection_correction_bounded",
                "bead projection should stay finite and small relative to tire radius",
                "last_xpbd_bead_correction_m",
                self.last_xpbd_bead_correction_m,
                0.050,
            ),
        ]
    }
}

/// Non-calibration effect check for the cavity-to-carcass coupling proxy.
///
/// The current model intentionally keeps `cavity_carcass_coupling` as a named
/// preset parameter until a deeper bead/rim/laminate pressure-support solve is
/// implemented. This report makes that proxy auditable: zero coupling should
/// remove cavity traction, nominal coupling should produce finite bead load,
/// and stronger coupling should increase the same gas-law pressure path
/// without causing bead drift to explode.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraCavityCouplingEffectReport {
    pub sample_count: usize,
    pub min_coupling: f64,
    pub nominal_coupling: f64,
    pub max_coupling: f64,
    pub zero_force_per_bead_n: f64,
    pub nominal_force_per_bead_n: f64,
    pub high_force_per_bead_n: f64,
    pub high_over_nominal_force: f64,
    pub max_bead_mount_error_m: f64,
    pub finite_outputs: bool,
}

impl TetraCavityCouplingEffectReport {
    pub fn is_finite(self) -> bool {
        self.min_coupling.is_finite()
            && self.nominal_coupling.is_finite()
            && self.max_coupling.is_finite()
            && self.zero_force_per_bead_n.is_finite()
            && self.nominal_force_per_bead_n.is_finite()
            && self.high_force_per_bead_n.is_finite()
            && self.high_over_nominal_force.is_finite()
            && self.max_bead_mount_error_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 3
            && self.max_coupling > self.nominal_coupling
            && self.nominal_coupling > self.min_coupling
            && self.zero_force_per_bead_n <= 1.0e-6
            && self.nominal_force_per_bead_n > 1.0
            && self.high_over_nominal_force >= 1.45
            && self.max_bead_mount_error_m <= 0.050
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "cavity_coupling_effect_has_samples",
                "cavity coupling effect should compare several named coupling levels",
                "sample_count",
                self.sample_count as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "cavity_coupling_effect_refines_coupling",
                "cavity coupling effect should sweep from zero to a stronger coupling",
                "max_minus_min_coupling",
                self.max_coupling - self.min_coupling,
                0.30,
            ),
            TireRigAcceptanceCheck::at_least(
                "cavity_coupling_effect_outputs_finite",
                "cavity coupling effect outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "cavity_coupling_zero_removes_bead_load",
                "zero cavity-to-carcass coupling should remove cavity traction instead of hiding a residual pressure force",
                "zero_force_per_bead_n",
                self.zero_force_per_bead_n,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "cavity_coupling_nominal_loads_bead",
                "nominal cavity-to-carcass coupling should produce finite pressure load on the bead/carcass path",
                "nominal_force_per_bead_n",
                self.nominal_force_per_bead_n,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "cavity_coupling_high_increases_load",
                "stronger cavity-to-carcass coupling should increase the same gas-law bead load path",
                "high_over_nominal_force",
                self.high_over_nominal_force,
                1.45,
            ),
            TireRigAcceptanceCheck::at_most(
                "cavity_coupling_bead_drift_bounded",
                "cavity coupling sweep should keep bead/rim drift bounded while the proxy remains in use",
                "max_bead_mount_error_m",
                self.max_bead_mount_error_m,
                0.050,
            ),
        ]
    }
}

/// Deterministic bead/rim load-case sanity report.
///
/// This is still not an external bead-seat calibration target or a full
/// contact solve. It is the first reusable non-calibration guard for the open
/// rim-seat validation gap: the same XPBD bead/rim guardrail is exercised under
/// unloaded, loaded, lateral-slip, and high-pressure cases so bead drift cannot
/// look acceptable only in the static inflation snapshot.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraBeadSeatLoadCaseReport {
    pub bead_node_count: usize,
    pub static_max_mount_error_m: f64,
    pub loaded_max_mount_error_m: f64,
    pub lateral_max_mount_error_m: f64,
    pub torque_max_mount_error_m: f64,
    pub high_pressure_max_mount_error_m: f64,
    pub lateral_max_abs_radial_error_m: f64,
    pub lateral_max_abs_axial_error_m: f64,
    pub lateral_max_tangential_error_m: f64,
    pub torque_max_tangential_error_m: f64,
    pub lateral_contact_force_n: f64,
    pub torque_contact_force_n: f64,
    pub high_pressure_effective_pressure_pa: f64,
    pub finite_outputs: bool,
}

impl TetraBeadSeatLoadCaseReport {
    pub fn is_finite(self) -> bool {
        self.static_max_mount_error_m.is_finite()
            && self.loaded_max_mount_error_m.is_finite()
            && self.lateral_max_mount_error_m.is_finite()
            && self.torque_max_mount_error_m.is_finite()
            && self.high_pressure_max_mount_error_m.is_finite()
            && self.lateral_max_abs_radial_error_m.is_finite()
            && self.lateral_max_abs_axial_error_m.is_finite()
            && self.lateral_max_tangential_error_m.is_finite()
            && self.torque_max_tangential_error_m.is_finite()
            && self.lateral_contact_force_n.is_finite()
            && self.torque_contact_force_n.is_finite()
            && self.high_pressure_effective_pressure_pa.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.bead_node_count >= 16
            && self.static_max_mount_error_m <= 0.050
            && self.loaded_max_mount_error_m <= 0.055
            && self.lateral_max_mount_error_m <= 0.060
            && self.torque_max_mount_error_m <= 0.060
            && self.high_pressure_max_mount_error_m <= 0.060
            && self.lateral_max_abs_radial_error_m <= 0.040
            && self.lateral_max_abs_axial_error_m <= 0.040
            && self.lateral_max_tangential_error_m <= 0.040
            && self.torque_max_tangential_error_m <= 0.040
            && self.lateral_contact_force_n >= 10.0
            && self.torque_contact_force_n >= 10.0
            && self.high_pressure_effective_pressure_pa > 0.0
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "bead_load_cases_outputs_finite",
                "bead/rim load-case diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "bead_load_cases_nodes_present",
                "rim-seat checks should cover a distributed bead node set",
                "bead_node_count",
                self.bead_node_count as f64,
                16.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_static_mount_bounded",
                "static inflated bead mount drift should stay bounded",
                "static_max_mount_error_m",
                self.static_max_mount_error_m,
                0.050,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_loaded_mount_bounded",
                "vertical load should not detach the bead from the rim seat",
                "loaded_max_mount_error_m",
                self.loaded_max_mount_error_m,
                0.055,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_lateral_mount_bounded",
                "lateral tire force should not peel the bead away from the rim seat",
                "lateral_max_mount_error_m",
                self.lateral_max_mount_error_m,
                0.060,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_torque_mount_bounded",
                "drive/brake tire force should not detach the bead from the rim seat",
                "torque_max_mount_error_m",
                self.torque_max_mount_error_m,
                0.060,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_high_pressure_mount_bounded",
                "high inflation pressure should remain inside the bead/rim guardrail envelope",
                "high_pressure_max_mount_error_m",
                self.high_pressure_max_mount_error_m,
                0.060,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_lateral_radial_error_bounded",
                "lateral bead load should keep radial seat drift bounded",
                "lateral_max_abs_radial_error_m",
                self.lateral_max_abs_radial_error_m,
                0.040,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_lateral_axial_error_bounded",
                "lateral bead load should keep axial seat drift bounded",
                "lateral_max_abs_axial_error_m",
                self.lateral_max_abs_axial_error_m,
                0.040,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_lateral_tangential_error_bounded",
                "lateral bead load should expose and bound tangential rim-seat slip",
                "lateral_max_tangential_error_m",
                self.lateral_max_tangential_error_m,
                0.040,
            ),
            TireRigAcceptanceCheck::at_most(
                "bead_load_cases_torque_tangential_error_bounded",
                "drive/brake bead load should expose and bound tangential rim-seat slip",
                "torque_max_tangential_error_m",
                self.torque_max_tangential_error_m,
                0.040,
            ),
            TireRigAcceptanceCheck::at_least(
                "bead_load_cases_lateral_force_applied",
                "the bead-seat lateral load case should actually apply lateral tire force",
                "abs_lateral_contact_force_n",
                self.lateral_contact_force_n,
                10.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "bead_load_cases_torque_force_applied",
                "the bead-seat drive/brake load case should actually apply longitudinal tire force",
                "abs_torque_contact_force_n",
                self.torque_contact_force_n,
                10.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "bead_load_cases_high_pressure_active",
                "high-pressure load case should retain positive effective cavity pressure",
                "high_pressure_effective_pressure_pa",
                self.high_pressure_effective_pressure_pa,
                1.0,
            ),
        ]
    }
}
