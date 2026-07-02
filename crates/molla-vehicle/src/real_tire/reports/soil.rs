//! Deformable-terrain (Bekker / heightfield / DEM / MPM) feedback reports.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::{FlatTerrain, HeightfieldTerrain, Terrain};

use super::super::internal::{
    add, bool_metric, dot, finite_nonnegative, finite_or, normalize, scale,
};
use super::super::{
    RealTireContact, RealTireParams, RealWheelState, SurfaceMaterial, TireRigAcceptanceCheck,
    evaluate_brush_tire, solve_loaded_tire_hub_height, tire_rig_point_at,
};

/// Deterministic Bekker/deformable-terrain support+shear coupling report.
///
/// This promotes the 2.5-D `DeformableTerrain::apply_tire_contact` path into
/// the standard tire rig: a rolling tire must create normal support/sinkage,
/// while a driven/slipping tire must add shear rutting and rolling-loss
/// feedback through the same terrain state instead of a scene grip scalar.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireSoilNormalShearCouplingReport {
    pub rolling_contact_cells: usize,
    pub rolling_normal_load_n: f64,
    pub rolling_longitudinal_force_n: f64,
    pub rolling_normal_sinkage_m: f64,
    pub rolling_shear_sinkage_m: f64,
    pub rolling_max_sinkage_m: f64,
    pub rolling_resistance_n: f64,
    pub driven_contact_cells: usize,
    pub driven_normal_load_n: f64,
    pub driven_longitudinal_force_n: f64,
    pub driven_normal_sinkage_m: f64,
    pub driven_shear_sinkage_m: f64,
    pub driven_max_sinkage_m: f64,
    pub driven_rolling_resistance_n: f64,
    pub shear_sinkage_gain_m: f64,
    pub rut_depth_gain_m: f64,
    pub rolling_resistance_gain_n: f64,
    pub finite_outputs: bool,
}

impl TireSoilNormalShearCouplingReport {
    pub fn is_finite(self) -> bool {
        self.rolling_normal_load_n.is_finite()
            && self.rolling_longitudinal_force_n.is_finite()
            && self.rolling_normal_sinkage_m.is_finite()
            && self.rolling_shear_sinkage_m.is_finite()
            && self.rolling_max_sinkage_m.is_finite()
            && self.rolling_resistance_n.is_finite()
            && self.driven_normal_load_n.is_finite()
            && self.driven_longitudinal_force_n.is_finite()
            && self.driven_normal_sinkage_m.is_finite()
            && self.driven_shear_sinkage_m.is_finite()
            && self.driven_max_sinkage_m.is_finite()
            && self.driven_rolling_resistance_n.is_finite()
            && self.shear_sinkage_gain_m.is_finite()
            && self.rut_depth_gain_m.is_finite()
            && self.rolling_resistance_gain_n.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.rolling_contact_cells > 0
            && self.driven_contact_cells > 0
            && self.rolling_normal_load_n > 500.0
            && self.driven_normal_load_n > 500.0
            && self.rolling_normal_sinkage_m > 1.0e-5
            && self.driven_normal_sinkage_m > 1.0e-5
            && self.shear_sinkage_gain_m > 1.0e-5
            && self.rut_depth_gain_m > 1.0e-5
            && self.rolling_resistance_gain_n >= 0.0
            && self.driven_longitudinal_force_n.abs() > self.rolling_longitudinal_force_n.abs()
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "soil_normal_shear_outputs_finite",
                "deformable-soil normal/shear coupling diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "soil_normal_shear_contact_cells_active",
                "deformable-soil coupling should receive active tire contact cells",
                "driven_contact_cells",
                self.driven_contact_cells as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "soil_normal_shear_normal_support",
                "normal tire load should produce positive Bekker sinkage/support",
                "rolling_normal_sinkage_m",
                self.rolling_normal_sinkage_m,
                1.0e-5,
            ),
            TireRigAcceptanceCheck::at_least(
                "soil_normal_shear_driven_support",
                "driven tire load should keep positive normal support while applying shear",
                "driven_normal_sinkage_m",
                self.driven_normal_sinkage_m,
                1.0e-5,
            ),
            TireRigAcceptanceCheck::at_least(
                "soil_normal_shear_slip_deepens_shear_rut",
                "driven slip should deepen the shear component of the rut",
                "shear_sinkage_gain_m",
                self.shear_sinkage_gain_m,
                1.0e-5,
            ),
            TireRigAcceptanceCheck::at_least(
                "soil_normal_shear_slip_deepens_total_rut",
                "driven shear should increase total rut depth relative to free rolling",
                "rut_depth_gain_m",
                self.rut_depth_gain_m,
                1.0e-5,
            ),
            TireRigAcceptanceCheck::at_least(
                "soil_normal_shear_rolling_loss_not_reduced",
                "deeper driven rut should not reduce Bekker rolling resistance",
                "rolling_resistance_gain_n",
                self.rolling_resistance_gain_n,
                0.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "soil_normal_shear_drive_force_increases",
                "driven slip should create more longitudinal shear force than free rolling",
                "abs_driven_minus_abs_rolling_fx_n",
                self.driven_longitudinal_force_n.abs() - self.rolling_longitudinal_force_n.abs(),
                1.0,
            ),
        ]
    }
}

pub fn tire_soil_normal_shear_coupling_report() -> TireSoilNormalShearCouplingReport {
    let params = RealTireParams::offroad_soft()
        .with_geometry_and_pressure(0.34, 0.24, 95_000.0)
        .with_segments(128);
    let mut soil =
        crate::deformable::DeformableTerrain::sandy_loam_patch([-2.0, -2.0], 0.20, 21, 21, 0.0);
    soil.shear_strength = 10_000.0;
    soil.shear_sinkage_gain = 0.060;
    let points = crate::deformable::tire_soil_rut_matrix(
        &params,
        &soil,
        &SurfaceMaterial::soft_soil(),
        4_800.0,
        2.5,
        &[0.0, 1.20],
    );
    let finite_outputs = points.len() == 2 && points.iter().all(|point| point.is_finite());
    let Some(rolling) = points.first() else {
        return TireSoilNormalShearCouplingReport::default();
    };
    let Some(driven) = points.get(1) else {
        return TireSoilNormalShearCouplingReport::default();
    };
    TireSoilNormalShearCouplingReport {
        rolling_contact_cells: rolling.contact.patch.contact_count,
        rolling_normal_load_n: rolling.contact.normal_load_n,
        rolling_longitudinal_force_n: rolling.contact.longitudinal_force_n,
        rolling_normal_sinkage_m: rolling.feedback.normal_sinkage_m,
        rolling_shear_sinkage_m: rolling.feedback.shear_sinkage_m,
        rolling_max_sinkage_m: rolling.feedback.max_sinkage_m,
        rolling_resistance_n: rolling.feedback.rolling_resistance_n,
        driven_contact_cells: driven.contact.patch.contact_count,
        driven_normal_load_n: driven.contact.normal_load_n,
        driven_longitudinal_force_n: driven.contact.longitudinal_force_n,
        driven_normal_sinkage_m: driven.feedback.normal_sinkage_m,
        driven_shear_sinkage_m: driven.feedback.shear_sinkage_m,
        driven_max_sinkage_m: driven.feedback.max_sinkage_m,
        driven_rolling_resistance_n: driven.feedback.rolling_resistance_n,
        shear_sinkage_gain_m: driven.feedback.shear_sinkage_m - rolling.feedback.shear_sinkage_m,
        rut_depth_gain_m: driven.feedback.max_sinkage_m - rolling.feedback.max_sinkage_m,
        rolling_resistance_gain_n: driven.feedback.rolling_resistance_n
            - rolling.feedback.rolling_resistance_n,
        finite_outputs,
    }
}

/// Heightfield surface-velocity map sanity report.
///
/// This is the CPU terrain-side counterpart to the GPU FEM
/// `surface_velocity_map` path: local ground velocity must be sampled from a
/// row-major map and must change tire force/work through relative slip, not by
/// mutating wheel speed or injecting a viewer-only force multiplier.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireHeightfieldSurfaceVelocityMapReport {
    pub static_surface_velocity_x_m_per_s: f64,
    pub moving_surface_velocity_x_m_per_s: f64,
    pub static_longitudinal_force_n: f64,
    pub moving_longitudinal_force_n: f64,
    pub force_delta_n: f64,
    pub moving_ground_power_w: f64,
    pub surface_velocity_delta_m_per_s: f64,
    pub snapshot_velocity_exported: bool,
    pub finite_outputs: bool,
}

impl TireHeightfieldSurfaceVelocityMapReport {
    pub fn is_finite(self) -> bool {
        self.static_surface_velocity_x_m_per_s.is_finite()
            && self.moving_surface_velocity_x_m_per_s.is_finite()
            && self.static_longitudinal_force_n.is_finite()
            && self.moving_longitudinal_force_n.is_finite()
            && self.force_delta_n.is_finite()
            && self.moving_ground_power_w.is_finite()
            && self.surface_velocity_delta_m_per_s.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.static_surface_velocity_x_m_per_s.abs() < 1.0e-9
            && self.moving_surface_velocity_x_m_per_s.abs() > 1.0
            && self.force_delta_n > 500.0
            && self.moving_ground_power_w.abs() > 1.0
            && self.surface_velocity_delta_m_per_s.abs() > 1.0
            && self.snapshot_velocity_exported
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "heightfield_surface_velocity_outputs_finite",
                "heightfield surface-velocity map diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "heightfield_surface_velocity_static_zero",
                "static heightfield-map region should sample zero ground velocity",
                "abs_static_surface_velocity_x_m_per_s",
                self.static_surface_velocity_x_m_per_s.abs(),
                1.0e-9,
            ),
            TireRigAcceptanceCheck::at_least(
                "heightfield_surface_velocity_map_sampled",
                "moving heightfield-map region should sample non-zero ground velocity",
                "abs_moving_surface_velocity_x_m_per_s",
                self.moving_surface_velocity_x_m_per_s.abs(),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "heightfield_surface_velocity_changes_force",
                "spatial ground velocity should change tire force through relative slip",
                "force_delta_n",
                self.force_delta_n,
                500.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "heightfield_surface_velocity_work_nonzero",
                "mapped moving ground should report non-zero force·surface-velocity work",
                "abs_moving_ground_power_w",
                self.moving_ground_power_w.abs(),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "heightfield_surface_velocity_delta_recorded",
                "heightfield surface-velocity map should record different local ground speeds",
                "abs_surface_velocity_delta_m_per_s",
                self.surface_velocity_delta_m_per_s.abs(),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "heightfield_surface_velocity_snapshot_exported",
                "heightfield velocity-map replay snapshot should export sampled terrain velocity",
                "snapshot_velocity_exported",
                bool_metric(self.snapshot_velocity_exported),
                1.0,
            ),
        ]
    }
}

pub fn tire_heightfield_surface_velocity_map_report() -> TireHeightfieldSurfaceVelocityMapReport {
    let material = SurfaceMaterial::dry_asphalt();
    let nx = 5;
    let nz = 3;
    let heights = vec![0.0; nx * nz];
    let mut velocity_map = vec![[0.0, 0.0, 0.0]; nx * nz];
    for iz in 0..nz {
        for ix in 3..nx {
            velocity_map[iz * nx + ix] = [-4.0, 0.0, 0.0];
        }
    }
    let terrain = HeightfieldTerrain::new([-1.0, -0.5], 0.5, nx, nz, heights, material.mu_static)
        .with_surface_velocity_map(velocity_map);
    let params = RealTireParams::passenger_radial()
        .with_geometry_and_pressure(0.32, 0.185, 180_000.0)
        .with_segments(96);
    let load_n = 4_500.0;
    let speed_m_per_s = 10.0;
    let slip_ratio = 0.0;
    let static_point = tire_rig_point_at(
        &params,
        &terrain,
        &material,
        [-0.75, 0.0],
        load_n,
        speed_m_per_s,
        slip_ratio,
        0.0,
        0.0,
    );
    let moving_point = tire_rig_point_at(
        &params,
        &terrain,
        &material,
        [0.75, 0.0],
        load_n,
        speed_m_per_s,
        slip_ratio,
        0.0,
        0.0,
    );
    let static_velocity = terrain.surface_velocity(-0.75, 0.0);
    let moving_velocity = terrain.surface_velocity(0.75, 0.0);
    let moving_snapshot = moving_point.snapshot_on_terrain(
        "heightfield_surface_velocity_map",
        "passenger_radial",
        "dry_asphalt",
        "heightfield_velocity_map",
        "cpu_heightfield_surface_velocity_map",
        &terrain,
        0.75,
        0.0,
    );
    TireHeightfieldSurfaceVelocityMapReport {
        static_surface_velocity_x_m_per_s: static_velocity[0],
        moving_surface_velocity_x_m_per_s: moving_velocity[0],
        static_longitudinal_force_n: static_point.telemetry.longitudinal_force_n,
        moving_longitudinal_force_n: moving_point.telemetry.longitudinal_force_n,
        force_delta_n: (moving_point.telemetry.longitudinal_force_n
            - static_point.telemetry.longitudinal_force_n)
            .abs(),
        moving_ground_power_w: finite_or(dot(moving_point.telemetry.force_n, moving_velocity), 0.0),
        surface_velocity_delta_m_per_s: moving_velocity[0] - static_velocity[0],
        snapshot_velocity_exported: (moving_snapshot.terrain_surface_velocity_m_per_s[0]
            - moving_velocity[0])
            .abs()
            < 1.0e-9,
        finite_outputs: static_point.is_finite()
            && moving_point.is_finite()
            && moving_snapshot.is_finite(),
    }
}

/// Deterministic DEM/granular tire-contact feedback sanity report.
///
/// This is not a full multi-scene offroad ownership solution. It makes the
/// current contact-cell-to-DEM coupling executable in the standard tire rig:
/// a soft-soil tire patch must transfer normal load and shear into grains, and
/// a driven/slipping tire must inject more DEM shear/motion than a free-rolling
/// tire on the same bed.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireDemSoilFeedbackReport {
    pub rolling_active_contact_cells: usize,
    pub rolling_affected_grains: usize,
    pub rolling_normal_load_n: f64,
    pub rolling_shear_load_n: f64,
    pub rolling_horizontal_grain_speed_sum_m_per_s: f64,
    pub slipping_active_contact_cells: usize,
    pub slipping_affected_grains: usize,
    pub slipping_normal_load_n: f64,
    pub slipping_shear_load_n: f64,
    pub slipping_horizontal_grain_speed_sum_m_per_s: f64,
    pub shear_ratio_slip_over_roll: f64,
    pub grain_motion_ratio_slip_over_roll: f64,
    pub max_grain_sinkage_m: f64,
    pub rolling_resistance_n: f64,
    pub finite_outputs: bool,
}

impl TireDemSoilFeedbackReport {
    pub fn is_finite(self) -> bool {
        self.rolling_normal_load_n.is_finite()
            && self.rolling_shear_load_n.is_finite()
            && self.rolling_horizontal_grain_speed_sum_m_per_s.is_finite()
            && self.slipping_normal_load_n.is_finite()
            && self.slipping_shear_load_n.is_finite()
            && self.slipping_horizontal_grain_speed_sum_m_per_s.is_finite()
            && self.shear_ratio_slip_over_roll.is_finite()
            && self.grain_motion_ratio_slip_over_roll.is_finite()
            && self.max_grain_sinkage_m.is_finite()
            && self.rolling_resistance_n.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.rolling_active_contact_cells > 0
            && self.slipping_active_contact_cells > 0
            && self.rolling_affected_grains > 0
            && self.slipping_affected_grains > 0
            && self.slipping_normal_load_n > 250.0
            && self.slipping_shear_load_n > 10.0
            && self.shear_ratio_slip_over_roll > 1.20
            && self.grain_motion_ratio_slip_over_roll > 1.02
            && self.max_grain_sinkage_m > 0.0
            && self.rolling_resistance_n > 0.0
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_outputs_finite",
                "DEM tire-contact feedback diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_contact_cells_active",
                "DEM tire feedback should receive active tire contact cells",
                "slipping_active_contact_cells",
                self.slipping_active_contact_cells as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_grains_affected",
                "DEM tire feedback should push at least one grain",
                "slipping_affected_grains",
                self.slipping_affected_grains as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_normal_load_transferred",
                "DEM tire feedback should transfer meaningful normal load into grains",
                "slipping_normal_load_n",
                self.slipping_normal_load_n,
                250.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_shear_transferred",
                "driven DEM tire feedback should transfer shear into grains",
                "slipping_shear_load_n",
                self.slipping_shear_load_n,
                10.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_slip_increases_shear",
                "slipping tire should inject more DEM shear than free rolling",
                "shear_ratio_slip_over_roll",
                self.shear_ratio_slip_over_roll,
                1.20,
            ),
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_slip_moves_grains_more",
                "slipping tire should leave more horizontal grain motion than free rolling",
                "grain_motion_ratio_slip_over_roll",
                self.grain_motion_ratio_slip_over_roll,
                1.02,
            ),
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_sinkage_positive",
                "normal DEM tire impulse should create positive grain sinkage",
                "max_grain_sinkage_m",
                self.max_grain_sinkage_m,
                1.0e-9,
            ),
            TireRigAcceptanceCheck::at_least(
                "dem_soil_feedback_rolling_loss_positive",
                "DEM tire feedback should expose a positive rolling-loss proxy",
                "rolling_resistance_n",
                self.rolling_resistance_n,
                1.0e-9,
            ),
        ]
    }
}

/// Deterministic material-point tire-soil feedback sanity report.
///
/// This mirrors the first CPU/GPU MPM tire-feedback contract at the
/// `molla-vehicle` rig level without pulling the GPU kernel into this crate.
/// Contact cells from the real tire are applied to a small material-point plug:
/// the report checks normal compaction, shear impulse, surface-velocity proxy,
/// and that driven slip produces more shear/compaction than free rolling.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireMpmSoilFeedbackReport {
    pub rolling_active_points: usize,
    pub rolling_total_compaction: f64,
    pub rolling_normal_impulse_n_s: f64,
    pub rolling_shear_impulse_n_s: f64,
    pub rolling_surface_velocity_sum_m_per_s: f64,
    pub slipping_active_points: usize,
    pub slipping_total_compaction: f64,
    pub slipping_normal_impulse_n_s: f64,
    pub slipping_shear_impulse_n_s: f64,
    pub slipping_surface_velocity_sum_m_per_s: f64,
    pub shear_ratio_slip_over_roll: f64,
    pub compaction_ratio_slip_over_roll: f64,
    pub surface_velocity_ratio_slip_over_roll: f64,
    pub finite_outputs: bool,
}

impl TireMpmSoilFeedbackReport {
    pub fn is_finite(self) -> bool {
        self.rolling_total_compaction.is_finite()
            && self.rolling_normal_impulse_n_s.is_finite()
            && self.rolling_shear_impulse_n_s.is_finite()
            && self.rolling_surface_velocity_sum_m_per_s.is_finite()
            && self.slipping_total_compaction.is_finite()
            && self.slipping_normal_impulse_n_s.is_finite()
            && self.slipping_shear_impulse_n_s.is_finite()
            && self.slipping_surface_velocity_sum_m_per_s.is_finite()
            && self.shear_ratio_slip_over_roll.is_finite()
            && self.compaction_ratio_slip_over_roll.is_finite()
            && self.surface_velocity_ratio_slip_over_roll.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.rolling_active_points > 0
            && self.slipping_active_points > 0
            && self.slipping_total_compaction > 0.0
            && self.slipping_normal_impulse_n_s > 1.0e-4
            && self.slipping_shear_impulse_n_s > 1.0e-4
            && self.shear_ratio_slip_over_roll > 1.10
            && self.compaction_ratio_slip_over_roll >= 0.95
            && self.surface_velocity_ratio_slip_over_roll > 1.05
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "mpm_soil_feedback_outputs_finite",
                "MPM tire-soil feedback diagnostics should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "mpm_soil_feedback_points_active",
                "MPM tire feedback should affect material points",
                "slipping_active_points",
                self.slipping_active_points as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "mpm_soil_feedback_compaction_positive",
                "normal tire pressure should create positive material-point compaction",
                "slipping_total_compaction",
                self.slipping_total_compaction,
                1.0e-12,
            ),
            TireRigAcceptanceCheck::at_least(
                "mpm_soil_feedback_normal_impulse_positive",
                "MPM tire feedback should accumulate positive normal impulse",
                "slipping_normal_impulse_n_s",
                self.slipping_normal_impulse_n_s,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "mpm_soil_feedback_shear_impulse_positive",
                "driven MPM tire feedback should accumulate shear impulse",
                "slipping_shear_impulse_n_s",
                self.slipping_shear_impulse_n_s,
                1.0e-4,
            ),
            TireRigAcceptanceCheck::at_least(
                "mpm_soil_feedback_slip_increases_shear",
                "slipping tire should inject more MPM shear than free rolling",
                "shear_ratio_slip_over_roll",
                self.shear_ratio_slip_over_roll,
                1.10,
            ),
            TireRigAcceptanceCheck::at_least(
                "mpm_soil_feedback_compaction_conserved_under_slip",
                "slipping tire should keep comparable normal compaction to free rolling",
                "compaction_ratio_slip_over_roll",
                self.compaction_ratio_slip_over_roll,
                0.95,
            ),
            TireRigAcceptanceCheck::at_least(
                "mpm_soil_feedback_slip_increases_surface_velocity",
                "slipping tire should write more horizontal material-point motion for FEM terrain export",
                "surface_velocity_ratio_slip_over_roll",
                self.surface_velocity_ratio_slip_over_roll,
                1.05,
            ),
        ]
    }
}

fn dem_feedback_horizontal_speed_sum(bed: &crate::granular::GranularBed) -> f64 {
    bed.grain_velocities()
        .iter()
        .map(|v| (v[0] * v[0] + v[2] * v[2]).sqrt())
        .filter(|v| v.is_finite())
        .sum()
}

fn dem_feedback_bed() -> crate::granular::GranularBed {
    let mut bed = crate::granular::GranularBed::lattice(
        [-0.32, 0.02, -0.16],
        [0.32, 0.18, 0.16],
        0.04,
        0.65,
        0.75,
    );
    bed.jitter(0.004);
    for _ in 0..160 {
        bed.step(&[], 5.0e-4);
    }
    bed
}

/// Build a deterministic DEM tire-contact feedback sanity report.
pub fn tire_dem_soil_feedback_report() -> TireDemSoilFeedbackReport {
    let params = RealTireParams::offroad_soft().with_segments(64);
    let material = SurfaceMaterial::soft_soil();
    let load_n = 3_500.0;
    let dt = 2.0e-3;

    let mut roll_bed = dem_feedback_bed();
    let roll_y = solve_loaded_tire_hub_height(&params, &roll_bed, 0.0, 0.0, load_n);
    let roll_wheel = RealWheelState {
        hub_pos: [0.0, roll_y, 0.0],
        hub_vel: [2.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: 2.0 / params.radius.max(1.0e-9),
    };
    let rolling_contact = evaluate_brush_tire(&params, &roll_wheel, &roll_bed, &material);
    let roll_feedback =
        roll_bed.apply_tire_contact(&roll_wheel, &rolling_contact, params.radius, dt);
    let roll_motion = dem_feedback_horizontal_speed_sum(&roll_bed);

    let mut slip_bed = dem_feedback_bed();
    let slip_y = solve_loaded_tire_hub_height(&params, &slip_bed, 0.0, 0.0, load_n);
    let slip_wheel = RealWheelState {
        hub_pos: [0.0, slip_y, 0.0],
        hub_vel: [2.0, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: (2.0 + 2.2) / params.radius.max(1.0e-9),
    };
    let slipping_contact = evaluate_brush_tire(&params, &slip_wheel, &slip_bed, &material);
    let slip_feedback =
        slip_bed.apply_tire_contact(&slip_wheel, &slipping_contact, params.radius, dt);
    let slip_motion = dem_feedback_horizontal_speed_sum(&slip_bed);

    let finite_outputs = rolling_contact.is_finite()
        && slipping_contact.is_finite()
        && roll_feedback.is_finite()
        && slip_feedback.is_finite()
        && roll_motion.is_finite()
        && slip_motion.is_finite();
    TireDemSoilFeedbackReport {
        rolling_active_contact_cells: roll_feedback.active_contact_cells,
        rolling_affected_grains: roll_feedback.affected_grains,
        rolling_normal_load_n: roll_feedback.transferred_normal_load_n,
        rolling_shear_load_n: roll_feedback.transferred_shear_load_n,
        rolling_horizontal_grain_speed_sum_m_per_s: roll_motion,
        slipping_active_contact_cells: slip_feedback.active_contact_cells,
        slipping_affected_grains: slip_feedback.affected_grains,
        slipping_normal_load_n: slip_feedback.transferred_normal_load_n,
        slipping_shear_load_n: slip_feedback.transferred_shear_load_n,
        slipping_horizontal_grain_speed_sum_m_per_s: slip_motion,
        shear_ratio_slip_over_roll: slip_feedback.transferred_shear_load_n
            / roll_feedback.transferred_shear_load_n.max(1.0e-9),
        grain_motion_ratio_slip_over_roll: slip_motion / roll_motion.max(1.0e-9),
        max_grain_sinkage_m: slip_feedback
            .max_grain_sinkage_m
            .max(roll_feedback.max_grain_sinkage_m),
        rolling_resistance_n: slip_feedback
            .rolling_resistance_n
            .max(roll_feedback.rolling_resistance_n),
        finite_outputs,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct MpmFeedbackPoint {
    pos: [f64; 3],
    vel: [f64; 3],
    compaction: [f64; 4],
}

fn mpm_feedback_points() -> Vec<MpmFeedbackPoint> {
    let mut points = Vec::new();
    for ix in -8..=8 {
        for iz in -3..=3 {
            for iy in 0..=2 {
                points.push(MpmFeedbackPoint {
                    pos: [
                        ix as f64 * 0.035,
                        -0.018 - iy as f64 * 0.025,
                        iz as f64 * 0.035,
                    ],
                    vel: [0.0; 3],
                    compaction: [0.0; 4],
                });
            }
        }
    }
    points
}

fn apply_mpm_feedback_reference(
    wheel: &RealWheelState,
    contact: &RealTireContact,
    points: &mut [MpmFeedbackPoint],
    dt: f64,
) -> bool {
    let heading = normalize(wheel.heading);
    let side = [-heading[2], 0.0, heading[0]];
    let mass = 0.35_f64;
    let compaction_gain = 2.0e-7_f64;
    let max_compaction = 0.60_f64;
    let damping = 0.10_f64;
    let mut touched = false;
    for point in points {
        let mut dv = [0.0; 3];
        let mut accum = [0.0; 4];
        let mut damping_weight: f64 = 0.0;
        for cell in &contact.cells {
            let area = finite_nonnegative(cell.area).max(1.0e-6);
            let influence = (area / std::f64::consts::PI).sqrt().max(0.055);
            let cx = wheel.hub_pos[0] + heading[0] * cell.local_x + side[0] * cell.local_z;
            let cz = wheel.hub_pos[2] + heading[2] * cell.local_x + side[2] * cell.local_z;
            let dx = point.pos[0] - cx;
            let dz = point.pos[2] - cz;
            let horizontal = (dx * dx + dz * dz).sqrt();
            if horizontal > influence {
                continue;
            }
            let below_surface = -point.pos[1];
            if below_surface < -influence * 0.5 || below_surface > influence * 2.0 {
                continue;
            }
            let radial_w = (1.0 - horizontal / influence).clamp(0.0, 1.0);
            let depth_w = (1.0 - below_surface.max(0.0) / (influence * 2.0)).clamp(0.10, 1.0);
            let w = radial_w * radial_w * depth_w;
            if w <= 1.0e-12 {
                continue;
            }
            let force = add(
                add(
                    scale(cell.normal, -cell.normal_load.max(0.0)),
                    scale(heading, -cell.tangential_force),
                ),
                scale(side, -cell.lateral_force),
            );
            let shear = (cell.tangential_force * cell.tangential_force
                + cell.lateral_force * cell.lateral_force)
                .sqrt();
            let pressure = (cell.normal_load.max(0.0) / area).max((-force[1]).max(0.0));
            for k in 0..3 {
                dv[k] += force[k] * w * dt / mass;
            }
            accum[0] += pressure * w * compaction_gain * dt;
            accum[1] += (-force[1]).max(0.0) * w * dt;
            accum[2] += shear * w * dt;
            accum[3] += w;
            damping_weight = damping_weight.max(w);
        }
        let damp = (damping * damping_weight * dt).clamp(0.0, 0.95);
        for (k, dv_k) in dv.iter().enumerate() {
            point.vel[k] = (point.vel[k] + *dv_k) * (1.0 - damp);
            point.pos[k] += point.vel[k] * dt;
        }
        point.compaction[0] = (point.compaction[0] + accum[0]).min(max_compaction);
        point.compaction[1] += accum[1];
        point.compaction[2] += accum[2];
        point.compaction[3] += accum[3];
        touched |= accum[3] > 0.0;
    }
    touched
}

fn mpm_feedback_case(slip_extra_m_per_s: f64) -> (Vec<MpmFeedbackPoint>, RealTireContact, bool) {
    let params = RealTireParams::offroad_soft().with_segments(64);
    let material = SurfaceMaterial::soft_soil();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let load_n = 3_500.0;
    let speed = 2.0;
    let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
    let wheel = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [speed, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: (speed + slip_extra_m_per_s) / params.radius.max(1.0e-9),
    };
    let contact = evaluate_brush_tire(&params, &wheel, &terrain, &material);
    let mut points = mpm_feedback_points();
    let touched = apply_mpm_feedback_reference(&wheel, &contact, &mut points, 2.0e-3);
    (points, contact, touched)
}

fn mpm_feedback_summary(points: &[MpmFeedbackPoint]) -> (usize, f64, f64, f64, f64, bool) {
    let mut active = 0usize;
    let mut compaction = 0.0_f64;
    let mut normal_impulse = 0.0_f64;
    let mut shear_impulse = 0.0_f64;
    let mut horizontal_motion = 0.0_f64;
    let mut finite = true;
    for point in points {
        finite &= point.pos.iter().all(|v| v.is_finite())
            && point.vel.iter().all(|v| v.is_finite())
            && point.compaction.iter().all(|v| v.is_finite());
        compaction += point.compaction[0];
        normal_impulse += point.compaction[1];
        shear_impulse += point.compaction[2];
        horizontal_motion += (point.vel[0] * point.vel[0] + point.vel[2] * point.vel[2]).sqrt();
        if point.compaction[3] > 0.0 {
            active += 1;
        }
    }
    (
        active,
        compaction,
        normal_impulse,
        shear_impulse,
        horizontal_motion,
        finite,
    )
}

/// Build a deterministic material-point tire-soil feedback sanity report.
pub fn tire_mpm_soil_feedback_report() -> TireMpmSoilFeedbackReport {
    let (rolling_points, rolling_contact, rolling_touched) = mpm_feedback_case(0.0);
    let (slipping_points, slipping_contact, slipping_touched) = mpm_feedback_case(2.2);
    let (
        rolling_active,
        rolling_compaction,
        rolling_normal,
        rolling_shear,
        rolling_motion,
        rolling_finite,
    ) = mpm_feedback_summary(&rolling_points);
    let (
        slipping_active,
        slipping_compaction,
        slipping_normal,
        slipping_shear,
        slipping_motion,
        slipping_finite,
    ) = mpm_feedback_summary(&slipping_points);
    let finite_outputs = rolling_finite
        && slipping_finite
        && rolling_contact.is_finite()
        && slipping_contact.is_finite()
        && rolling_touched
        && slipping_touched;
    TireMpmSoilFeedbackReport {
        rolling_active_points: rolling_active,
        rolling_total_compaction: rolling_compaction,
        rolling_normal_impulse_n_s: rolling_normal,
        rolling_shear_impulse_n_s: rolling_shear,
        rolling_surface_velocity_sum_m_per_s: rolling_motion,
        slipping_active_points: slipping_active,
        slipping_total_compaction: slipping_compaction,
        slipping_normal_impulse_n_s: slipping_normal,
        slipping_shear_impulse_n_s: slipping_shear,
        slipping_surface_velocity_sum_m_per_s: slipping_motion,
        shear_ratio_slip_over_roll: slipping_shear / rolling_shear.max(1.0e-9),
        compaction_ratio_slip_over_roll: slipping_compaction / rolling_compaction.max(1.0e-12),
        surface_velocity_ratio_slip_over_roll: slipping_motion / rolling_motion.max(1.0e-9),
        finite_outputs,
    }
}
