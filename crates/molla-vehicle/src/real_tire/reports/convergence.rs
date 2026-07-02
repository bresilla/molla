//! Convergence / layer-ablation / heightfield / thermal-wear / contact-patch
//! convergence reports.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::{FlatTerrain, HeightfieldTerrain};

use super::super::internal::{bool_metric, dot, finite_vec3, sub};
use super::super::{
    RealTireContact, RealTireParams, RealTireTransientState, RealWheelOrientation, RealWheelState,
    SurfaceFilmState, SurfaceMaterial, TetraTireFem, TetraTireFemDiagnostics, TetraTireFemParams,
    TireConditionMaterialResponse, TireContactPatchConvergenceReport, TireRigAcceptanceCheck,
    TireRigEnvelopePoint, TireThermalLayers, TireWearState, evaluate_brush_tire,
    evaluate_brush_tire_transient, solve_loaded_tire_hub_height,
    tetra_cavity_coupling_effect_report, tire_dem_soil_feedback_report,
    tire_rig_heightfield_envelope_sweep, tire_rig_point,
};

/// Data-free ablation/debug report for explicit tire-model layers.
///
/// These are not alternate force sources and not calibration toggles. The
/// report intentionally isolates one named model layer at a time so rigs and
/// viewer truth labels can prove that carcass/inflation, cavity coupling,
/// bristles, terrain material, wet film, thermal/wear state, soil feedback, and
/// transient relaxation are observable through native physics paths.
#[derive(Clone, Copy, Debug)]
pub struct TireLayerAblationDebugReport {
    pub sample_count: usize,
    pub finite_outputs: bool,
    pub carcass_pressure_changes_patch: bool,
    pub cavity_coupling_changes_bead_load: bool,
    pub tread_bristles_change_force: bool,
    pub terrain_material_changes_force: bool,
    pub wet_film_changes_force: bool,
    pub thermal_wear_changes_material: bool,
    pub soil_rut_feedback_active: bool,
    pub transient_relaxation_lags_raw: bool,
}

impl TireLayerAblationDebugReport {
    pub fn is_finite(self) -> bool {
        true
    }

    pub fn passed(self) -> bool {
        self.sample_count >= 8
            && self.finite_outputs
            && self.carcass_pressure_changes_patch
            && self.cavity_coupling_changes_bead_load
            && self.tread_bristles_change_force
            && self.terrain_material_changes_force
            && self.wet_film_changes_force
            && self.thermal_wear_changes_material
            && self.soil_rut_feedback_active
            && self.transient_relaxation_lags_raw
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_debug_has_samples",
                "layer-ablation debug report should isolate all named tire layers",
                "sample_count",
                self.sample_count as f64,
                8.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_debug_outputs_finite",
                "layer-ablation debug samples should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_carcass_pressure_changes_patch",
                "carcass/inflation layer should visibly change loaded contact patch",
                "carcass_pressure_changes_patch",
                bool_metric(self.carcass_pressure_changes_patch),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_cavity_coupling_changes_bead_load",
                "cavity pressure coupling layer should be isolatable through bead-load telemetry",
                "cavity_coupling_changes_bead_load",
                bool_metric(self.cavity_coupling_changes_bead_load),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_tread_bristles_change_force",
                "tread bristle layer should be isolatable without creating a fake alternate force source",
                "tread_bristles_change_force",
                bool_metric(self.tread_bristles_change_force),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_terrain_material_changes_force",
                "terrain material layer should change force through the shared contact API",
                "terrain_material_changes_force",
                bool_metric(self.terrain_material_changes_force),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_wet_film_changes_force",
                "wet-film layer should reduce usable shear through state, not a scene grip switch",
                "wet_film_changes_force",
                bool_metric(self.wet_film_changes_force),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_thermal_wear_changes_material",
                "thermal/wear layer should change material response from state variables",
                "thermal_wear_changes_material",
                bool_metric(self.thermal_wear_changes_material),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_soil_rut_feedback_active",
                "soil/rut feedback layer should be active in the reusable tire-soil report",
                "soil_rut_feedback_active",
                bool_metric(self.soil_rut_feedback_active),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "layer_ablation_transient_relaxation_lags_raw",
                "transient relaxation layer should lag raw commanded slip on the first step",
                "transient_relaxation_lags_raw",
                bool_metric(self.transient_relaxation_lags_raw),
                1.0,
            ),
        ]
    }
}

pub fn tire_layer_ablation_debug_report() -> TireLayerAblationDebugReport {
    let base = RealTireParams::passenger_radial().with_segments(96);
    let dry = SurfaceMaterial::dry_asphalt();
    let ice = SurfaceMaterial::ice();
    let dry_terrain = FlatTerrain::new(0.0, dry.mu_static);
    let ice_terrain = FlatTerrain::new(0.0, ice.mu_static);
    let load_n = 4_000.0;
    let speed_m_per_s = 12.0;
    let slip = 0.22;

    let stiff = base.with_geometry_and_pressure(base.radius, base.width, 300_000.0);
    let soft = base.with_geometry_and_pressure(base.radius, base.width, 95_000.0);
    let stiff_point = tire_rig_point(
        &stiff,
        &dry_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );
    let soft_point = tire_rig_point(
        &soft,
        &dry_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );

    let cavity = tetra_cavity_coupling_effect_report();

    let normal_bristle = tire_rig_point(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );
    let mut no_bristle_params = base;
    no_bristle_params.tread_shear_stiffness = 0.0;
    let mut no_bristle_material = dry;
    no_bristle_material.bristle_damping = 0.0;
    no_bristle_material.cohesion = 0.0;
    let no_bristle = tire_rig_point(
        &no_bristle_params,
        &dry_terrain,
        &no_bristle_material,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );

    let dry_force = tire_rig_point(
        &base,
        &dry_terrain,
        &dry,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );
    let ice_force = tire_rig_point(
        &base,
        &ice_terrain,
        &ice,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );

    let wet_material =
        SurfaceFilmState::wet_road().apply_to(dry, speed_m_per_s, &TireWearState::default());
    let wet_force = tire_rig_point(
        &base,
        &FlatTerrain::new(0.0, wet_material.mu_static),
        &wet_material,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );

    let mut worn = TireWearState::default();
    worn.depth_m = worn.max_depth_m * 0.93;
    let hot_thermal = TireThermalLayers {
        surface_c: 185.0,
        tread_c: 155.0,
        carcass_c: 130.0,
        rim_c: 95.0,
        ..TireThermalLayers::passenger_road()
    };
    let hot_worn_response =
        TireConditionMaterialResponse::from_state(&base, &dry, &hot_thermal, &worn);
    let hot_worn_force = tire_rig_point(
        &hot_worn_response.params,
        &FlatTerrain::new(0.0, hot_worn_response.material.mu_static),
        &hot_worn_response.material,
        load_n,
        speed_m_per_s,
        slip,
        0.0,
        0.0,
    );

    let soil = tire_dem_soil_feedback_report();

    let hub_y = solve_loaded_tire_hub_height(&base, &dry_terrain, 0.0, 0.0, load_n);
    let raw_state = RealWheelState {
        hub_pos: [0.0, hub_y, 0.0],
        hub_vel: [speed_m_per_s, 0.0, 0.0],
        heading: [1.0, 0.0, 0.0],
        omega: speed_m_per_s * (1.0 + 0.35) / base.radius.max(1.0e-6),
    };
    let raw_contact = evaluate_brush_tire(&base, &raw_state, &dry_terrain, &dry);
    let mut transient = RealTireTransientState::default();
    let transient_contact = evaluate_brush_tire_transient(
        &base,
        &raw_state,
        &dry_terrain,
        &dry,
        &RealWheelOrientation::default(),
        &mut transient,
        1.0 / 240.0,
    );

    let finite_outputs = stiff_point.is_finite()
        && soft_point.is_finite()
        && cavity.is_finite()
        && normal_bristle.is_finite()
        && no_bristle.is_finite()
        && dry_force.is_finite()
        && ice_force.is_finite()
        && wet_force.is_finite()
        && hot_worn_response.is_finite()
        && hot_worn_force.is_finite()
        && soil.is_finite()
        && raw_contact.is_finite()
        && transient_contact.is_finite();

    TireLayerAblationDebugReport {
        sample_count: 8,
        finite_outputs,
        carcass_pressure_changes_patch: soft_point.telemetry.patch.patch_length_m
            > stiff_point.telemetry.patch.patch_length_m + 1.0e-4,
        cavity_coupling_changes_bead_load: cavity.passed()
            && cavity.zero_force_per_bead_n <= 1.0e-6
            && cavity.high_force_per_bead_n > cavity.nominal_force_per_bead_n,
        tread_bristles_change_force: normal_bristle.telemetry.longitudinal_force_n.abs()
            > no_bristle.telemetry.longitudinal_force_n.abs() + 100.0,
        terrain_material_changes_force: ice_force.telemetry.longitudinal_force_n.abs()
            < dry_force.telemetry.longitudinal_force_n.abs() * 0.35,
        wet_film_changes_force: wet_force.telemetry.longitudinal_force_n.abs()
            < dry_force.telemetry.longitudinal_force_n.abs() * 0.95,
        thermal_wear_changes_material: hot_worn_response.material.mu_static < dry.mu_static * 0.85
            && hot_worn_force.telemetry.longitudinal_force_n.abs()
                < dry_force.telemetry.longitudinal_force_n.abs() * 0.90,
        soil_rut_feedback_active: soil.passed()
            && soil.max_grain_sinkage_m > 0.0
            && soil.rolling_resistance_n > 0.0,
        transient_relaxation_lags_raw: transient_contact.slip_ratio.abs()
            < raw_contact.slip_ratio.abs() * 0.45
            && transient_contact.longitudinal_force.abs()
                < raw_contact.longitudinal_force.abs() * 0.85,
    }
}

pub fn tire_contact_patch_convergence_report() -> TireContactPatchConvergenceReport {
    let material = SurfaceMaterial::dry_asphalt();
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let load_n = 3_800.0;
    let speed_m_per_s = 8.0;
    let slip_ratio = 0.08;
    let lateral_speed_m_per_s = 0.65;
    let segment_counts = [24usize, 48, 96, 192];
    let mut samples = Vec::with_capacity(segment_counts.len());
    let mut finite_outputs = true;
    let mut max_normal_load_error_ratio = 0.0_f64;
    let mut max_pressure_bin_load_error_ratio = 0.0_f64;

    for segments in segment_counts {
        let params = RealTireParams::passenger_radial().with_segments(segments);
        let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
        let state = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [speed_m_per_s, 0.0, lateral_speed_m_per_s],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip_ratio) / params.radius,
        };
        let contact = evaluate_brush_tire(&params, &state, &terrain, &material);
        let pressure = contact.pressure_distribution(12, 1);
        let binned_load: f64 = pressure
            .longitudinal_bins
            .iter()
            .map(|bin| bin.normal_load_n)
            .sum();
        finite_outputs &= contact.in_contact
            && contact.force.iter().all(|v| v.is_finite())
            && contact.normal_load.is_finite()
            && contact.aligning_moment_z.is_finite()
            && contact.patch_length.is_finite()
            && pressure.is_finite();
        max_normal_load_error_ratio =
            max_normal_load_error_ratio.max((contact.normal_load - load_n).abs() / load_n.max(1.0));
        max_pressure_bin_load_error_ratio = max_pressure_bin_load_error_ratio
            .max((binned_load - contact.normal_load).abs() / contact.normal_load.max(1.0));
        samples.push((segments, params.radius, contact, pressure));
    }

    let contact_count_increases = samples
        .windows(2)
        .all(|pair| pair[1].2.contact_count >= pair[0].2.contact_count);
    let (coarse_segments, _, coarse_contact, _) = samples.first().expect("fixed samples");
    let (fine_segments, fine_radius, fine_contact, fine_pressure) =
        samples.last().expect("fixed samples");
    let (_, _, prev_contact, prev_pressure) = &samples[samples.len() - 2];
    let force_delta = sub(fine_contact.force, prev_contact.force);
    let force_delta_over_load = dot(force_delta, force_delta).sqrt() / load_n.max(1.0);
    let patch_length_delta_over_radius =
        (fine_contact.patch_length - prev_contact.patch_length).abs() / fine_radius.max(1.0e-9);
    let aligning_delta_over_load_radius =
        (fine_contact.aligning_moment_z - prev_contact.aligning_moment_z).abs()
            / (load_n.max(1.0) * fine_radius.max(1.0e-9));
    let peak_pressure_relative_delta =
        (fine_pressure.peak_normal_pressure_pa - prev_pressure.peak_normal_pressure_pa).abs()
            / fine_pressure
                .peak_normal_pressure_pa
                .abs()
                .max(prev_pressure.peak_normal_pressure_pa.abs())
                .max(1.0);

    TireContactPatchConvergenceReport {
        sample_count: samples.len(),
        min_segments: *coarse_segments,
        max_segments: *fine_segments,
        finite_outputs,
        contact_count_increases: contact_count_increases
            && fine_contact.contact_count >= coarse_contact.contact_count,
        max_normal_load_error_ratio,
        patch_length_delta_over_radius,
        force_delta_over_load,
        aligning_delta_over_load_radius,
        peak_pressure_relative_delta,
        max_pressure_bin_load_error_ratio,
    }
}

/// Data-free FEM substep convergence report for the deformable tire contact path.
///
/// The sampled brush convergence report tracks ideal contact-cell refinement.
/// This report covers the FEM side of the same PLAN_REALTIRE requirement: a
/// loaded tetra tire is advanced over the same physical interval with
/// increasingly fine explicit solver substeps, then its deformed tread contact,
/// force, hub height, and volume diagnostics are compared against the finest
/// result. It is a broad sanity envelope, not a calibrated transient target.
#[derive(Clone, Copy, Debug, Default)]
pub struct TetraFemSubstepConvergenceReport {
    pub sample_count: usize,
    pub min_substeps: usize,
    pub max_substeps: usize,
    pub finite_outputs: bool,
    pub coarse_to_fine_hub_delta_over_radius: f64,
    pub coarse_to_fine_patch_delta_over_radius: f64,
    pub coarse_to_fine_force_delta_over_load: f64,
    pub max_volume_error_m3: f64,
    pub max_contact_count: usize,
}

impl TetraFemSubstepConvergenceReport {
    pub fn is_finite(self) -> bool {
        self.coarse_to_fine_hub_delta_over_radius.is_finite()
            && self.coarse_to_fine_patch_delta_over_radius.is_finite()
            && self.coarse_to_fine_force_delta_over_load.is_finite()
            && self.max_volume_error_m3.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 3
            && self.max_substeps > self.min_substeps
            && self.coarse_to_fine_hub_delta_over_radius <= 0.08
            && self.coarse_to_fine_patch_delta_over_radius <= 0.35
            && self.coarse_to_fine_force_delta_over_load <= 0.45
            && self.max_volume_error_m3 <= 2.0e-3
            && self.max_contact_count >= 2
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "fem_substep_convergence_has_samples",
                "FEM substep convergence should compare multiple explicit substep counts",
                "sample_count",
                self.sample_count as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_substep_convergence_refines_substeps",
                "FEM substep convergence should refine from coarse to fine substeps",
                "max_minus_min_substeps",
                self.max_substeps as f64 - self.min_substeps as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_substep_convergence_outputs_finite",
                "FEM substep convergence outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_substep_convergence_hub_stable",
                "loaded hub height should not drift wildly as explicit substeps are refined",
                "coarse_to_fine_hub_delta_over_radius",
                self.coarse_to_fine_hub_delta_over_radius,
                0.08,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_substep_convergence_patch_stable",
                "deformed tread patch length should remain in the same broad envelope as substeps are refined",
                "coarse_to_fine_patch_delta_over_radius",
                self.coarse_to_fine_patch_delta_over_radius,
                0.35,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_substep_convergence_force_stable",
                "FEM-derived contact force should not teleport as substeps are refined",
                "coarse_to_fine_force_delta_over_load",
                self.coarse_to_fine_force_delta_over_load,
                0.45,
            ),
            TireRigAcceptanceCheck::at_most(
                "fem_substep_convergence_volume_bounded",
                "tetra volume error should stay bounded across the substep sweep",
                "max_volume_error_m3",
                self.max_volume_error_m3,
                2.0e-3,
            ),
            TireRigAcceptanceCheck::at_least(
                "fem_substep_convergence_keeps_contact",
                "FEM substep convergence should preserve a distributed tread contact patch",
                "max_contact_count",
                self.max_contact_count as f64,
                2.0,
            ),
        ]
    }
}

/// Data-free heightfield-resolution convergence report for terrain contact.
///
/// This complements segment and FEM-substep convergence by varying the sampled
/// terrain grid resolution under the same analytic bump. It checks that
/// height, normal, patch length, and force direction/magnitude do not change
/// wildly merely because a viewer/debug terrain was rebuilt with a different
/// heightfield density.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireHeightfieldResolutionConvergenceReport {
    pub sample_count: usize,
    pub min_grid_resolution: usize,
    pub max_grid_resolution: usize,
    pub finite_outputs: bool,
    pub height_delta_m: f64,
    pub normal_delta: f64,
    pub patch_delta_over_radius: f64,
    pub force_delta_over_load: f64,
    pub min_contact_count: usize,
}

impl TireHeightfieldResolutionConvergenceReport {
    pub fn is_finite(self) -> bool {
        self.height_delta_m.is_finite()
            && self.normal_delta.is_finite()
            && self.patch_delta_over_radius.is_finite()
            && self.force_delta_over_load.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 3
            && self.max_grid_resolution > self.min_grid_resolution
            && self.height_delta_m <= 0.020
            && self.normal_delta <= 0.35
            && self.patch_delta_over_radius <= 0.35
            && self.force_delta_over_load <= 0.45
            && self.min_contact_count >= 2
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "heightfield_resolution_convergence_has_samples",
                "heightfield resolution convergence should compare several grid densities",
                "sample_count",
                self.sample_count as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "heightfield_resolution_refines_grid",
                "heightfield resolution convergence should refine from coarse to fine terrain grids",
                "max_minus_min_grid_resolution",
                self.max_grid_resolution as f64 - self.min_grid_resolution as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "heightfield_resolution_outputs_finite",
                "heightfield resolution convergence outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "heightfield_resolution_height_stable",
                "terrain height at the contact sample should stay in the same broad envelope",
                "height_delta_m",
                self.height_delta_m,
                0.020,
            ),
            TireRigAcceptanceCheck::at_most(
                "heightfield_resolution_normal_stable",
                "terrain normal should not swing wildly when heightfield resolution is refined",
                "normal_delta",
                self.normal_delta,
                0.35,
            ),
            TireRigAcceptanceCheck::at_most(
                "heightfield_resolution_patch_stable",
                "contact patch length should stay in the same broad envelope as terrain resolution changes",
                "patch_delta_over_radius",
                self.patch_delta_over_radius,
                0.35,
            ),
            TireRigAcceptanceCheck::at_most(
                "heightfield_resolution_force_stable",
                "heightfield-derived force should not teleport when grid density changes",
                "force_delta_over_load",
                self.force_delta_over_load,
                0.45,
            ),
            TireRigAcceptanceCheck::at_least(
                "heightfield_resolution_keeps_contact",
                "heightfield resolution convergence should preserve multi-cell tire contact",
                "min_contact_count",
                self.min_contact_count as f64,
                2.0,
            ),
        ]
    }
}

pub fn tetra_fem_substep_convergence_report() -> TetraFemSubstepConvergenceReport {
    let terrain = FlatTerrain::new(0.0, SurfaceMaterial::dry_asphalt().mu_static);
    let material = SurfaceMaterial::dry_asphalt();
    let load_n = 3_200.0;
    let params = TetraTireFemParams::realtime_viewer()
        .with_resolution(10, 4, 4)
        .with_pressure_and_load(150_000.0, load_n);
    let mut settled = TetraTireFem::new(0.0, params.pressure_pa, params);
    for _ in 0..28 {
        settled.step_inner(1.0 / 12_000.0, &terrain);
    }
    let state_for = |tire: &TetraTireFem| RealWheelState {
        hub_pos: tire.hub(),
        hub_vel: [5.0, 0.0, 0.15],
        heading: [1.0, 0.0, 0.0],
        omega: 5.0 * 1.05 / tire.outer_radius().max(1.0e-9),
    };
    let total_dt = 2.5e-4;
    let substep_counts = [1usize, 2, 4, 8];
    let mut samples: Vec<(usize, [f64; 3], RealTireContact, TetraTireFemDiagnostics)> = Vec::new();
    let mut finite_outputs = true;
    let mut max_volume_error_m3 = 0.0_f64;
    let mut max_contact_count = 0usize;

    for substeps in substep_counts {
        let mut tire = settled.clone();
        let h = total_dt / substeps as f64;
        for _ in 0..substeps {
            tire.step_inner(h, &terrain);
        }
        let contact = tire.evaluate_tread_contact(&state_for(&tire), &terrain, &material);
        let diagnostics = tire.diagnostics();
        finite_outputs &= finite_vec3(tire.hub())
            && contact.is_finite()
            && diagnostics.is_finite()
            && contact.force.iter().all(|v| v.is_finite());
        max_volume_error_m3 = max_volume_error_m3.max(diagnostics.max_tet_volume_error_m3.abs());
        max_contact_count = max_contact_count.max(contact.contact_count);
        samples.push((substeps, tire.hub(), contact, diagnostics));
    }

    let (min_substeps, coarse_hub, coarse_contact, _) = samples.first().expect("fixed samples");
    let (max_substeps, fine_hub, fine_contact, _) = samples.last().expect("fixed samples");
    let radius = settled.outer_radius().max(1.0e-9);
    let force_delta = sub(coarse_contact.force, fine_contact.force);

    TetraFemSubstepConvergenceReport {
        sample_count: samples.len(),
        min_substeps: *min_substeps,
        max_substeps: *max_substeps,
        finite_outputs,
        coarse_to_fine_hub_delta_over_radius: (coarse_hub[1] - fine_hub[1]).abs() / radius,
        coarse_to_fine_patch_delta_over_radius: (coarse_contact.patch_length
            - fine_contact.patch_length)
            .abs()
            / radius,
        coarse_to_fine_force_delta_over_load: dot(force_delta, force_delta).sqrt()
            / load_n.max(1.0),
        max_volume_error_m3,
        max_contact_count,
    }
}

/// Data-free heat/wear convergence report for contact-resolution changes.
///
/// Geometry-only convergence is not enough for a tire that claims thermal and
/// wear realism. This report runs the same sustained combined-slip sample with
/// increasing brush segment counts, feeds the resulting dissipated power into
/// the layered thermal/wear state, and checks that heat/wear telemetry remains
/// finite and broadly stable as contact discretization is refined.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireThermalWearConvergenceReport {
    pub sample_count: usize,
    pub min_segments: usize,
    pub max_segments: usize,
    pub finite_outputs: bool,
    pub dissipated_power_delta_over_load_speed: f64,
    pub surface_temperature_delta_c: f64,
    pub wear_depth_delta_m: f64,
    pub max_budget_ratio: f64,
    pub min_sliding_fraction: f64,
}

impl TireThermalWearConvergenceReport {
    pub fn is_finite(self) -> bool {
        self.dissipated_power_delta_over_load_speed.is_finite()
            && self.surface_temperature_delta_c.is_finite()
            && self.wear_depth_delta_m.is_finite()
            && self.max_budget_ratio.is_finite()
            && self.min_sliding_fraction.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 3
            && self.max_segments > self.min_segments
            && self.dissipated_power_delta_over_load_speed <= 0.25
            && self.surface_temperature_delta_c <= 8.0
            && self.wear_depth_delta_m <= 8.0e-6
            && self.max_budget_ratio <= 1.05
            && self.min_sliding_fraction > 0.05
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "thermal_wear_convergence_has_samples",
                "thermal/wear convergence should compare several contact segment counts",
                "sample_count",
                self.sample_count as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_wear_convergence_refines_segments",
                "thermal/wear convergence should refine contact discretization",
                "max_minus_min_segments",
                self.max_segments as f64 - self.min_segments as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_wear_convergence_outputs_finite",
                "thermal/wear convergence outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_wear_convergence_power_stable",
                "dissipated slip power should stay in the same broad envelope under contact refinement",
                "dissipated_power_delta_over_load_speed",
                self.dissipated_power_delta_over_load_speed,
                0.25,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_wear_convergence_temperature_stable",
                "surface temperature response should not teleport under contact refinement",
                "surface_temperature_delta_c",
                self.surface_temperature_delta_c,
                8.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_wear_convergence_wear_stable",
                "wear depth response should remain bounded under contact refinement",
                "wear_depth_delta_m",
                self.wear_depth_delta_m,
                8.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "thermal_wear_convergence_budget_bounded",
                "thermal/wear convergence samples should still respect the friction budget",
                "max_budget_ratio",
                self.max_budget_ratio,
                1.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "thermal_wear_convergence_sliding_active",
                "thermal/wear convergence should exercise sliding/bristle heat generation",
                "min_sliding_fraction",
                self.min_sliding_fraction,
                0.05,
            ),
        ]
    }
}

pub fn tire_heightfield_resolution_convergence_report() -> TireHeightfieldResolutionConvergenceReport
{
    let material = SurfaceMaterial::dry_asphalt();
    let params = RealTireParams::offroad_soft()
        .with_geometry_and_pressure(0.34, 0.220, 105_000.0)
        .with_segments(96);
    let load_n = 4_100.0;
    let speed_m_per_s = 5.0;
    let slip_ratio = 0.07;
    let x_sample = -0.035;
    let z_sample = 0.0;
    let domain = 1.20_f64;
    let bump_height = 0.070_f64;
    let sigma = 0.115_f64;
    let grid_resolutions = [17usize, 33, 65];
    let mut samples: Vec<(usize, TireRigEnvelopePoint)> =
        Vec::with_capacity(grid_resolutions.len());
    let mut finite_outputs = true;
    let mut min_contact_count = usize::MAX;

    for n in grid_resolutions {
        let spacing = domain / (n - 1) as f64;
        let origin = [-0.5 * domain, -0.5 * domain];
        let mut heights = Vec::with_capacity(n * n);
        for iz in 0..n {
            let z = origin[1] + iz as f64 * spacing;
            for ix in 0..n {
                let x = origin[0] + ix as f64 * spacing;
                let r2 = x * x + z * z;
                heights.push(bump_height * (-r2 / (2.0 * sigma * sigma)).exp());
            }
        }
        let terrain = HeightfieldTerrain::new(origin, spacing, n, n, heights, material.mu_static);
        let point = tire_rig_heightfield_envelope_sweep(
            &params,
            &terrain,
            &material,
            load_n,
            speed_m_per_s,
            slip_ratio,
            &[x_sample],
            z_sample,
        )
        .into_iter()
        .next()
        .expect("single heightfield sample");
        finite_outputs &= point.point.is_finite()
            && point.terrain_height_m.is_finite()
            && point.terrain_normal.iter().all(|v| v.is_finite())
            && point.point.telemetry.force_n.iter().all(|v| v.is_finite());
        min_contact_count = min_contact_count.min(point.point.telemetry.patch.contact_count);
        samples.push((n, point));
    }

    let (coarse_n, coarse) = samples.first().expect("fixed samples");
    let (fine_n, fine) = samples.last().expect("fixed samples");
    let normal_delta = dot(
        sub(coarse.terrain_normal, fine.terrain_normal),
        sub(coarse.terrain_normal, fine.terrain_normal),
    )
    .sqrt();
    let force_delta = sub(coarse.point.telemetry.force_n, fine.point.telemetry.force_n);

    TireHeightfieldResolutionConvergenceReport {
        sample_count: samples.len(),
        min_grid_resolution: *coarse_n,
        max_grid_resolution: *fine_n,
        finite_outputs,
        height_delta_m: (coarse.terrain_height_m - fine.terrain_height_m).abs(),
        normal_delta,
        patch_delta_over_radius: (coarse.point.telemetry.patch.patch_length_m
            - fine.point.telemetry.patch.patch_length_m)
            .abs()
            / params.radius.max(1.0e-9),
        force_delta_over_load: dot(force_delta, force_delta).sqrt() / load_n.max(1.0),
        min_contact_count,
    }
}

pub fn tire_thermal_wear_convergence_report() -> TireThermalWearConvergenceReport {
    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        cohesion: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let terrain = FlatTerrain::new(0.0, material.mu_static);
    let load_n = 4_200.0;
    let speed_m_per_s = 12.0;
    let slip_ratio = 0.16;
    let lateral_speed_m_per_s = 0.9;
    let dt_s = 0.50;
    let segment_counts = [32usize, 64, 128, 192];
    let mut samples: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(segment_counts.len());
    let mut finite_outputs = true;
    let mut max_budget_ratio = 0.0_f64;
    let mut min_sliding_fraction = f64::INFINITY;

    for segments in segment_counts {
        let params = RealTireParams::passenger_radial().with_segments(segments);
        let hub_y = solve_loaded_tire_hub_height(&params, &terrain, 0.0, 0.0, load_n);
        let state = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [speed_m_per_s, 0.0, lateral_speed_m_per_s],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip_ratio) / params.radius.max(1.0e-9),
        };
        let contact = evaluate_brush_tire(&params, &state, &terrain, &material);
        let mut thermal = TireThermalLayers::passenger_road();
        let mut wear = TireWearState::default();
        thermal.step(
            contact.dissipated_power,
            contact.rolling_resistance_force.abs() * speed_m_per_s * 0.25,
            contact.rolling_resistance_force.abs() * speed_m_per_s,
            0.0,
            24.0,
            dt_s,
        );
        wear.step_with_surface_temperature(contact.dissipated_power, thermal.surface_c, dt_s);
        finite_outputs &= contact.is_finite()
            && contact.dissipated_power.is_finite()
            && thermal.is_finite()
            && wear.is_finite();
        max_budget_ratio = max_budget_ratio
            .max(contact.consumed_friction_budget / contact.available_friction_budget.max(1.0));
        min_sliding_fraction = min_sliding_fraction.min(contact.sliding_fraction);
        samples.push((
            segments,
            contact.dissipated_power,
            thermal.surface_c,
            wear.depth_m,
        ));
    }

    let (min_segments, coarse_power, coarse_surface_c, coarse_wear_m) =
        samples.first().expect("fixed samples");
    let (max_segments, fine_power, fine_surface_c, fine_wear_m) =
        samples.last().expect("fixed samples");
    TireThermalWearConvergenceReport {
        sample_count: samples.len(),
        min_segments: *min_segments,
        max_segments: *max_segments,
        finite_outputs,
        dissipated_power_delta_over_load_speed: (coarse_power - fine_power).abs()
            / (load_n * speed_m_per_s).max(1.0),
        surface_temperature_delta_c: (coarse_surface_c - fine_surface_c).abs(),
        wear_depth_delta_m: (coarse_wear_m - fine_wear_m).abs(),
        max_budget_ratio,
        min_sliding_fraction,
    }
}
