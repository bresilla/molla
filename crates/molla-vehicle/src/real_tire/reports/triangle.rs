//! Triangle-mesh terrain seam/manifold/stress/force/density reports.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::Terrain;

use super::super::internal::{bool_metric, dot, sub};
use super::super::{
    RealTireContact, RealTireParams, RealWheelState, SurfaceMaterial, TetraTireFem,
    TetraTireFemParams, TireRigAcceptanceCheck, evaluate_brush_tire, solve_loaded_tire_hub_height,
    tire_nodes_finite,
};

fn tire_two_plate_triangle_terrain() -> crate::terrain::TriangleMeshTerrain {
    crate::terrain::TriangleMeshTerrain::new(
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
    .with_triangle_friction(vec![0.45, 0.45, 1.1, 1.1])
}

fn tire_subdivided_two_plate_triangle_terrain(
    subdivisions_per_half: usize,
) -> crate::terrain::TriangleMeshTerrain {
    let n_half = subdivisions_per_half.max(1);
    let nx = 2 * n_half + 1;
    let nz = 2 * n_half + 1;
    let mut vertices = Vec::with_capacity(nx * nz);
    for iz in 0..nz {
        let z = -1.0 + 2.0 * iz as f64 / (nz - 1) as f64;
        for ix in 0..nx {
            let x = -1.0 + 2.0 * ix as f64 / (nx - 1) as f64;
            vertices.push([x, 0.0, z]);
        }
    }
    let idx = |ix: usize, iz: usize| -> usize { iz * nx + ix };
    let mut triangles = Vec::with_capacity((nx - 1) * (nz - 1) * 2);
    let mut friction = Vec::with_capacity((nx - 1) * (nz - 1) * 2);
    for iz in 0..(nz - 1) {
        for ix in 0..(nx - 1) {
            let a = idx(ix, iz);
            let b = idx(ix + 1, iz);
            let c = idx(ix, iz + 1);
            let d = idx(ix + 1, iz + 1);
            let x_center =
                0.25 * (vertices[a][0] + vertices[b][0] + vertices[c][0] + vertices[d][0]);
            let mu = if x_center < 0.0 { 0.45 } else { 1.10 };
            triangles.push([a, b, c]);
            friction.push(mu);
            triangles.push([b, d, c]);
            friction.push(mu);
        }
    }
    crate::terrain::TriangleMeshTerrain::new(vertices, triangles, 0.9)
        .with_triangle_friction(friction)
}

/// Seam/manifold sanity for finite-triangle terrain contact.
///
/// This is a compact CPU-side regression/telemetry gate for the remaining
/// surface-to-surface terrain-contact work: a tread sample straddling two
/// triangle plates with different friction must expose a multi-triangle
/// manifold, preserve material variation, and let FEM tread projection act
/// before point-only penetration collapses the seam to one triangle.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireTriangleSeamManifoldReport {
    pub manifold_patch_count: usize,
    pub has_left_plate: bool,
    pub has_right_plate: bool,
    pub min_friction: f64,
    pub max_friction: f64,
    pub min_penetration_m: f64,
    pub point_gap_before_m: f64,
    pub node_lift_after_step_m: f64,
    pub signed_gap_after_step_m: f64,
}

impl TireTriangleSeamManifoldReport {
    pub fn is_finite(self) -> bool {
        self.min_friction.is_finite()
            && self.max_friction.is_finite()
            && self.min_penetration_m.is_finite()
            && self.point_gap_before_m.is_finite()
            && self.node_lift_after_step_m.is_finite()
            && self.signed_gap_after_step_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.manifold_patch_count >= 2
            && self.has_left_plate
            && self.has_right_plate
            && self.max_friction - self.min_friction >= 0.45
            && self.min_penetration_m > 0.0
            && self.point_gap_before_m > 0.0
            && self.node_lift_after_step_m > 0.004
            && self.signed_gap_after_step_m >= -1.0e-4
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "triangle_seam_manifold_multi_patch",
                "tread sample straddling a triangle seam should expose multiple contact patches",
                "manifold_patch_count",
                self.manifold_patch_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_seam_manifold_left_plate",
                "manifold should include the lower-friction left plate",
                "has_left_plate",
                bool_metric(self.has_left_plate),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_seam_manifold_right_plate",
                "manifold should include the higher-friction right plate",
                "has_right_plate",
                bool_metric(self.has_right_plate),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_seam_material_span",
                "per-triangle friction should survive seam manifold construction",
                "friction_span",
                self.max_friction - self.min_friction,
                0.45,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_seam_positive_penetration",
                "sphere/tread contact should report positive penetration before point penetration",
                "min_penetration_m",
                self.min_penetration_m,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_seam_point_gap_before",
                "test fixture should begin above point-only penetration and rely on tread radius",
                "point_gap_before_m",
                self.point_gap_before_m,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_seam_fem_projection_lifts_node",
                "FEM tread projection should consume the seam manifold radius and lift the node",
                "node_lift_after_step_m",
                self.node_lift_after_step_m,
                0.004,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_seam_projection_no_penetration",
                "FEM tread projection should not leave the seam sample penetrating",
                "signed_gap_after_step_m",
                self.signed_gap_after_step_m,
                -1.0e-4,
            ),
        ]
    }
}

pub fn tire_triangle_seam_manifold_report() -> TireTriangleSeamManifoldReport {
    let terrain = tire_two_plate_triangle_terrain();
    let sample = [0.0, 0.08, 0.0];
    let patches = terrain.contact_manifold(sample, 0.12, 0.02, 8);
    let has_left_plate = patches.iter().any(|patch| patch.triangle_id <= 1);
    let has_right_plate = patches.iter().any(|patch| patch.triangle_id >= 2);
    let min_friction = patches
        .iter()
        .map(|patch| patch.friction)
        .fold(f64::INFINITY, f64::min);
    let max_friction = patches
        .iter()
        .map(|patch| patch.friction)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_penetration_m = patches
        .iter()
        .map(|patch| patch.penetration)
        .fold(f64::INFINITY, f64::min);

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
    let before = tire.pos[id];
    let point_gap_before_m = terrain.signed_distance_and_normal(before).0;
    tire.step_on_terrain(&terrain, 1.0 / 240.0);
    let after = tire.nodes()[id];
    let signed_gap_after_step_m = terrain.signed_distance_and_normal(after).0;

    TireTriangleSeamManifoldReport {
        manifold_patch_count: patches.len(),
        has_left_plate,
        has_right_plate,
        min_friction,
        max_friction,
        min_penetration_m,
        point_gap_before_m,
        node_lift_after_step_m: after[1] - before[1],
        signed_gap_after_step_m,
    }
}

/// Broader finite-triangle surface-contact stress sanity report.
///
/// `TireTriangleSeamManifoldReport` checks one seam sample. This report samples
/// both single-material plate interiors plus the seam so the runner can catch
/// regressions where material lookup works at one point but the manifold
/// support collapses when the sample moves across the triangle surface.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireTriangleSurfaceStressReport {
    pub sample_count: usize,
    pub seam_patch_count: usize,
    pub seam_has_left_plate: bool,
    pub seam_has_right_plate: bool,
    pub left_plate_max_friction: f64,
    pub right_plate_min_friction: f64,
    pub seam_friction_span: f64,
    pub max_patch_count: usize,
    pub min_positive_penetration_m: f64,
    pub finite_outputs: bool,
}

impl TireTriangleSurfaceStressReport {
    pub fn is_finite(self) -> bool {
        self.left_plate_max_friction.is_finite()
            && self.right_plate_min_friction.is_finite()
            && self.seam_friction_span.is_finite()
            && self.min_positive_penetration_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 3
            && self.seam_patch_count >= 2
            && self.seam_has_left_plate
            && self.seam_has_right_plate
            && self.left_plate_max_friction <= 0.60
            && self.right_plate_min_friction >= 1.00
            && self.seam_friction_span >= 0.45
            && self.max_patch_count >= self.seam_patch_count
            && self.min_positive_penetration_m > 0.0
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_stress_outputs_finite",
                "triangle-surface stress samples should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_stress_sample_count",
                "triangle-surface stress report should sample plate interiors plus seam",
                "sample_count",
                self.sample_count as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_stress_seam_multi_patch",
                "seam sample should retain a multi-triangle contact manifold",
                "seam_patch_count",
                self.seam_patch_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_stress_seam_left_plate",
                "seam stress sample should include the left low-friction plate",
                "seam_has_left_plate",
                bool_metric(self.seam_has_left_plate),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_stress_seam_right_plate",
                "seam stress sample should include the right high-friction plate",
                "seam_has_right_plate",
                bool_metric(self.seam_has_right_plate),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "triangle_surface_stress_left_material",
                "left plate interior should preserve low friction material",
                "left_plate_max_friction",
                self.left_plate_max_friction,
                0.60,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_stress_right_material",
                "right plate interior should preserve high friction material",
                "right_plate_min_friction",
                self.right_plate_min_friction,
                1.00,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_stress_seam_material_span",
                "seam manifold should preserve per-triangle material contrast",
                "seam_friction_span",
                self.seam_friction_span,
                0.45,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_stress_positive_penetration",
                "all sampled triangle contacts should report positive tread-radius penetration",
                "min_positive_penetration_m",
                self.min_positive_penetration_m,
                1.0e-6,
            ),
        ]
    }
}

pub fn tire_triangle_surface_stress_report() -> TireTriangleSurfaceStressReport {
    let terrain = tire_two_plate_triangle_terrain();
    let sample_x = [-0.42, 0.0, 0.42];
    let mut finite_outputs = true;
    let mut max_patch_count = 0usize;
    let mut min_positive_penetration_m = f64::INFINITY;
    let mut left_plate_max_friction = 0.0_f64;
    let mut right_plate_min_friction = f64::INFINITY;
    let mut seam_patch_count = 0usize;
    let mut seam_has_left_plate = false;
    let mut seam_has_right_plate = false;
    let mut seam_min_friction = f64::INFINITY;
    let mut seam_max_friction = f64::NEG_INFINITY;
    let mut sample_count = 0usize;

    for &x in &sample_x {
        let patches = terrain.contact_manifold([x, 0.08, 0.0], 0.12, 0.02, 8);
        finite_outputs &= patches.iter().all(|patch| {
            patch.point.iter().all(|v| v.is_finite())
                && patch.normal.iter().all(|v| v.is_finite())
                && patch.signed_distance.is_finite()
                && patch.penetration.is_finite()
                && patch.friction.is_finite()
        });
        if patches.is_empty() {
            finite_outputs = false;
            continue;
        }
        sample_count += 1;
        max_patch_count = max_patch_count.max(patches.len());
        for patch in &patches {
            min_positive_penetration_m = min_positive_penetration_m.min(patch.penetration);
        }
        if x < -0.1 {
            left_plate_max_friction = patches.iter().map(|p| p.friction).fold(0.0_f64, f64::max);
        } else if x > 0.1 {
            right_plate_min_friction = patches
                .iter()
                .map(|p| p.friction)
                .fold(f64::INFINITY, f64::min);
        } else {
            seam_patch_count = patches.len();
            seam_has_left_plate = patches.iter().any(|patch| patch.triangle_id <= 1);
            seam_has_right_plate = patches.iter().any(|patch| patch.triangle_id >= 2);
            seam_min_friction = patches
                .iter()
                .map(|p| p.friction)
                .fold(f64::INFINITY, f64::min);
            seam_max_friction = patches
                .iter()
                .map(|p| p.friction)
                .fold(f64::NEG_INFINITY, f64::max);
        }
    }

    TireTriangleSurfaceStressReport {
        sample_count,
        seam_patch_count,
        seam_has_left_plate,
        seam_has_right_plate,
        left_plate_max_friction,
        right_plate_min_friction,
        seam_friction_span: seam_max_friction - seam_min_friction,
        max_patch_count,
        min_positive_penetration_m,
        finite_outputs,
    }
}

/// Triangle-surface force sanity for the same finite-triangle fixture.
///
/// The seam/stress reports above prove the manifold and material lookup exist.
/// This report exercises the actual brush force path on low-friction plate,
/// high-friction plate, and the seam so triangle material/manifold data is
/// observed in tractive force and bristle/sliding diagnostics, not only in
/// geometry-only contact rows.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireTriangleSurfaceForceReport {
    pub sample_count: usize,
    pub left_abs_fx_n: f64,
    pub seam_abs_fx_n: f64,
    pub right_abs_fx_n: f64,
    pub left_contact_count: usize,
    pub seam_contact_count: usize,
    pub right_contact_count: usize,
    pub left_sliding_fraction: f64,
    pub seam_sliding_fraction: f64,
    pub right_sliding_fraction: f64,
    pub seam_manifold_patch_count: usize,
    pub seam_friction_span: f64,
    pub finite_outputs: bool,
}

impl TireTriangleSurfaceForceReport {
    pub fn is_finite(self) -> bool {
        self.left_abs_fx_n.is_finite()
            && self.seam_abs_fx_n.is_finite()
            && self.right_abs_fx_n.is_finite()
            && self.left_sliding_fraction.is_finite()
            && self.seam_sliding_fraction.is_finite()
            && self.right_sliding_fraction.is_finite()
            && self.seam_friction_span.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 3
            && self.left_contact_count >= 2
            && self.seam_contact_count >= 2
            && self.right_contact_count >= 2
            && self.seam_manifold_patch_count >= 2
            && self.seam_friction_span >= 0.45
            && self.left_sliding_fraction > 0.05
            && self.seam_sliding_fraction > 0.05
            && self.right_sliding_fraction > 0.05
            && self.right_abs_fx_n > self.left_abs_fx_n * 1.25
            && self.seam_abs_fx_n > self.left_abs_fx_n * 1.02
            && self.seam_abs_fx_n < self.right_abs_fx_n * 1.08
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_force_outputs_finite",
                "triangle-surface force samples should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_force_sample_count",
                "triangle-surface force report should sample plate interiors plus seam",
                "sample_count",
                self.sample_count as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_force_contacts_present",
                "low/seam/high triangle samples should all produce distributed contact",
                "min_contact_count",
                self.left_contact_count
                    .min(self.seam_contact_count)
                    .min(self.right_contact_count) as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_force_seam_multi_patch",
                "seam force sample should retain a multi-triangle manifold",
                "seam_manifold_patch_count",
                self.seam_manifold_patch_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_force_seam_material_span",
                "seam force sample should preserve low/high material contrast",
                "seam_friction_span",
                self.seam_friction_span,
                0.45,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_force_sliding_bristles_active",
                "triangle force samples should exercise sliding/bristle diagnostics",
                "min_sliding_fraction",
                self.left_sliding_fraction
                    .min(self.seam_sliding_fraction)
                    .min(self.right_sliding_fraction),
                0.05,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_force_high_mu_stronger_than_low",
                "high-friction triangle plate should carry more drive force than low-friction plate",
                "right_over_left_abs_fx",
                self.right_abs_fx_n / self.left_abs_fx_n.max(1.0e-9),
                1.25,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_surface_force_seam_above_low",
                "seam force should include enough high-friction support to exceed the low plate",
                "seam_over_left_abs_fx",
                self.seam_abs_fx_n / self.left_abs_fx_n.max(1.0e-9),
                1.02,
            ),
            TireRigAcceptanceCheck::at_most(
                "triangle_surface_force_seam_below_high",
                "seam force should remain bounded by the high-friction plate response",
                "seam_over_right_abs_fx",
                self.seam_abs_fx_n / self.right_abs_fx_n.max(1.0e-9),
                1.08,
            ),
        ]
    }
}

/// Data-free triangle-density convergence report for finite-triangle terrain.
///
/// This is the triangle-mesh counterpart to the segment/FEM/heightfield
/// convergence reports. It rebuilds the same two-material flat road with
/// coarse-to-fine triangle subdivisions and checks that terrain height/normal,
/// material-aware brush force, and seam manifold support remain in the same
/// broad envelope as triangle density changes.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireTriangleDensityConvergenceReport {
    pub sample_count: usize,
    pub min_triangle_count: usize,
    pub max_triangle_count: usize,
    pub finite_outputs: bool,
    pub height_delta_m: f64,
    pub normal_delta: f64,
    pub force_delta_over_load: f64,
    pub patch_delta_over_radius: f64,
    pub min_contact_count: usize,
    pub min_seam_manifold_count: usize,
    pub min_friction_span: f64,
}

impl TireTriangleDensityConvergenceReport {
    pub fn is_finite(self) -> bool {
        self.height_delta_m.is_finite()
            && self.normal_delta.is_finite()
            && self.force_delta_over_load.is_finite()
            && self.patch_delta_over_radius.is_finite()
            && self.min_friction_span.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.sample_count >= 3
            && self.max_triangle_count > self.min_triangle_count
            && self.height_delta_m <= 1.0e-6
            && self.normal_delta <= 1.0e-6
            && self.force_delta_over_load <= 0.25
            && self.patch_delta_over_radius <= 0.20
            && self.min_contact_count >= 2
            && self.min_seam_manifold_count >= 2
            && self.min_friction_span >= 0.45
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "triangle_density_convergence_has_samples",
                "triangle-density convergence should compare several mesh subdivisions",
                "sample_count",
                self.sample_count as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_density_convergence_refines_mesh",
                "triangle-density convergence should refine from coarse to fine triangle meshes",
                "max_minus_min_triangle_count",
                self.max_triangle_count as f64 - self.min_triangle_count as f64,
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_density_convergence_outputs_finite",
                "triangle-density convergence outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "triangle_density_height_stable",
                "same flat triangle road should not change sampled height as density changes",
                "height_delta_m",
                self.height_delta_m,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "triangle_density_normal_stable",
                "same flat triangle road should not change sampled normal as density changes",
                "normal_delta",
                self.normal_delta,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_most(
                "triangle_density_force_stable",
                "triangle material/manifold force should not teleport when mesh density changes",
                "force_delta_over_load",
                self.force_delta_over_load,
                0.25,
            ),
            TireRigAcceptanceCheck::at_most(
                "triangle_density_patch_stable",
                "triangle material/manifold patch length should stay in the same broad envelope",
                "patch_delta_over_radius",
                self.patch_delta_over_radius,
                0.20,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_density_keeps_contact",
                "triangle-density convergence should preserve distributed tire contact",
                "min_contact_count",
                self.min_contact_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_density_keeps_seam_manifold",
                "triangle-density convergence should preserve multi-triangle seam manifolds",
                "min_seam_manifold_count",
                self.min_seam_manifold_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_density_keeps_material_span",
                "triangle-density convergence should preserve low/high material contrast at the seam",
                "min_friction_span",
                self.min_friction_span,
                0.45,
            ),
        ]
    }
}

/// Large finite-triangle terrain sanity for BVH/manifold/tread feedback paths.
///
/// The smaller seam reports prove material contrast on a hand-sized mesh. This
/// report keeps the larger static-triangle path honest: it uses the same
/// two-material road rebuilt at high subdivision, checks that the BVH narrows
/// the seam query while preserving multi-triangle material span, then exercises
/// the FEM tread-node brush/feedback path on that terrain so the large mesh is
/// not only a geometry-only diagnostic.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireTriangleLargeManifoldReport {
    pub triangle_count: usize,
    pub bvh_node_count: usize,
    pub seam_candidate_count: usize,
    pub seam_manifold_patch_count: usize,
    pub seam_friction_span: f64,
    pub fem_contact_count: usize,
    pub fem_node_sourced_contact_count: usize,
    pub feedback_contact_count: usize,
    pub feedback_force_norm_n: f64,
    pub feedback_sliding_fraction: f64,
    pub feedback_force_cell_count: usize,
    pub feedback_bristle_force_sum_n: f64,
    pub feedback_mean_bristle_force_n: f64,
    pub feedback_max_bristle_force_n: f64,
    pub feedback_width_span_m: f64,
    pub finite_outputs: bool,
}

impl TireTriangleLargeManifoldReport {
    pub fn is_finite(self) -> bool {
        self.seam_friction_span.is_finite()
            && self.feedback_force_norm_n.is_finite()
            && self.feedback_sliding_fraction.is_finite()
            && self.feedback_bristle_force_sum_n.is_finite()
            && self.feedback_mean_bristle_force_n.is_finite()
            && self.feedback_max_bristle_force_n.is_finite()
            && self.feedback_width_span_m.is_finite()
    }

    pub fn passed(self) -> bool {
        self.is_finite()
            && self.finite_outputs
            && self.triangle_count >= 400
            && self.bvh_node_count > 1
            && self.seam_candidate_count <= 12
            && self.seam_manifold_patch_count >= 2
            && self.seam_friction_span >= 0.45
            && self.fem_contact_count >= 2
            && self.fem_node_sourced_contact_count >= 2
            && self.feedback_contact_count >= 2
            && self.feedback_force_norm_n >= 25.0
            && self.feedback_sliding_fraction > 0.0
            && self.feedback_force_cell_count >= 2
            && self.feedback_bristle_force_sum_n >= 25.0
            && self.feedback_mean_bristle_force_n > 0.0
            && self.feedback_max_bristle_force_n >= self.feedback_mean_bristle_force_n
            && self.feedback_width_span_m >= 0.01
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_triangle_count",
                "large triangle-manifold report should use a sizeable static triangle mesh",
                "triangle_count",
                self.triangle_count as f64,
                400.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_has_bvh",
                "large triangle-manifold report should build a nontrivial BVH",
                "bvh_node_count",
                self.bvh_node_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_most(
                "triangle_large_manifold_bvh_narrows_query",
                "large triangle BVH should narrow seam height/material candidates",
                "seam_candidate_count",
                self.seam_candidate_count as f64,
                12.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_multi_patch",
                "large triangle seam should preserve a multi-triangle contact manifold",
                "seam_manifold_patch_count",
                self.seam_manifold_patch_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_material_span",
                "large triangle seam should preserve low/high material contrast",
                "seam_friction_span",
                self.seam_friction_span,
                0.45,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_fem_contact",
                "large triangle terrain should produce distributed FEM tread-node contact",
                "fem_contact_count",
                self.fem_contact_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_node_sourced_contact",
                "large triangle FEM contact cells should be sourced from actual tread nodes",
                "fem_node_sourced_contact_count",
                self.fem_node_sourced_contact_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_feedback_contact",
                "large triangle feedback should keep distributed tread-node contacts",
                "feedback_contact_count",
                self.feedback_contact_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_feedback_force",
                "large triangle feedback should apply a nonzero FEM brush/bristle force",
                "feedback_force_norm_n",
                self.feedback_force_norm_n,
                25.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_sliding_bristles",
                "large triangle feedback should exercise sliding/bristle diagnostics",
                "feedback_sliding_fraction",
                self.feedback_sliding_fraction,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_force_cell_reduction",
                "large triangle feedback should reduce multiple force-producing tread cells",
                "feedback_force_cell_count",
                self.feedback_force_cell_count as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_bristle_force_sum",
                "large triangle bristle reduction should carry nonzero tangential support",
                "feedback_bristle_force_sum_n",
                self.feedback_bristle_force_sum_n,
                25.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_bristle_mean_nonzero",
                "large triangle bristle reduction should expose nonzero mean force per active cell",
                "feedback_mean_bristle_force_n",
                self.feedback_mean_bristle_force_n,
                1.0e-6,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_bristle_peak_ge_mean",
                "large triangle bristle reduction should keep peak force at least as large as mean force",
                "feedback_peak_minus_mean_bristle_force_n",
                self.feedback_max_bristle_force_n - self.feedback_mean_bristle_force_n,
                -1.0e-9,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_widthwise_support",
                "large triangle feedback should preserve widthwise tread support instead of collapsing to one centerline point",
                "feedback_width_span_m",
                self.feedback_width_span_m,
                0.01,
            ),
            TireRigAcceptanceCheck::at_least(
                "triangle_large_manifold_outputs_finite",
                "large triangle manifold/feedback outputs should stay finite",
                "finite_outputs",
                bool_metric(self.finite_outputs && self.is_finite()),
                1.0,
            ),
        ]
    }
}

pub fn tire_triangle_surface_force_report() -> TireTriangleSurfaceForceReport {
    let terrain = tire_two_plate_triangle_terrain();
    let params = RealTireParams::passenger_radial().with_segments(96);
    let material = SurfaceMaterial::dry_asphalt();
    let load_n = 4_500.0;
    let speed_m_per_s = 10.0;
    let slip_ratio = 0.22;
    let mut contacts = Vec::new();

    for x in [-0.42, 0.0, 0.42] {
        let hub_y = solve_loaded_tire_hub_height(&params, &terrain, x, 0.0, load_n);
        let state = RealWheelState {
            hub_pos: [x, hub_y, 0.0],
            hub_vel: [speed_m_per_s, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip_ratio) / params.radius.max(1.0e-6),
        };
        contacts.push(evaluate_brush_tire(&params, &state, &terrain, &material));
    }

    let left = contacts
        .first()
        .cloned()
        .unwrap_or_else(RealTireContact::empty);
    let seam = contacts
        .get(1)
        .cloned()
        .unwrap_or_else(RealTireContact::empty);
    let right = contacts
        .get(2)
        .cloned()
        .unwrap_or_else(RealTireContact::empty);
    let seam_patches = terrain.contact_manifold([0.0, 0.08, 0.0], 0.12, 0.02, 8);
    let seam_min_friction = seam_patches
        .iter()
        .map(|patch| patch.friction)
        .fold(f64::INFINITY, f64::min);
    let seam_max_friction = seam_patches
        .iter()
        .map(|patch| patch.friction)
        .fold(f64::NEG_INFINITY, f64::max);
    let finite_outputs = contacts
        .iter()
        .all(|contact| contact.is_finite() && contact.contact_count >= 2)
        && seam_patches.iter().all(|patch| patch.friction.is_finite());

    TireTriangleSurfaceForceReport {
        sample_count: contacts.len(),
        left_abs_fx_n: left.longitudinal_force.abs(),
        seam_abs_fx_n: seam.longitudinal_force.abs(),
        right_abs_fx_n: right.longitudinal_force.abs(),
        left_contact_count: left.contact_count,
        seam_contact_count: seam.contact_count,
        right_contact_count: right.contact_count,
        left_sliding_fraction: left.sliding_fraction,
        seam_sliding_fraction: seam.sliding_fraction,
        right_sliding_fraction: right.sliding_fraction,
        seam_manifold_patch_count: seam_patches.len(),
        seam_friction_span: seam_max_friction - seam_min_friction,
        finite_outputs,
    }
}

pub fn tire_triangle_density_convergence_report() -> TireTriangleDensityConvergenceReport {
    let params = RealTireParams::passenger_radial().with_segments(96);
    let material = SurfaceMaterial::dry_asphalt();
    let load_n = 4_500.0;
    let speed_m_per_s = 10.0;
    let slip_ratio = 0.20;
    let mesh_subdivisions = [1usize, 2, 4, 8];
    let mut samples: Vec<(usize, f64, [f64; 3], RealTireContact, usize, f64)> =
        Vec::with_capacity(mesh_subdivisions.len());
    let mut finite_outputs = true;
    let mut min_contact_count = usize::MAX;
    let mut min_seam_manifold_count = usize::MAX;
    let mut min_friction_span = f64::INFINITY;

    for subdiv in mesh_subdivisions {
        let terrain = tire_subdivided_two_plate_triangle_terrain(subdiv);
        let x = 0.0;
        let z = 0.0;
        let hub_y = solve_loaded_tire_hub_height(&params, &terrain, x, z, load_n);
        let state = RealWheelState {
            hub_pos: [x, hub_y, z],
            hub_vel: [speed_m_per_s, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip_ratio) / params.radius.max(1.0e-6),
        };
        let contact = evaluate_brush_tire(&params, &state, &terrain, &material);
        let patches = terrain.contact_manifold([0.0, 0.08, 0.0], 0.12, 0.02, 16);
        let min_mu = patches
            .iter()
            .map(|patch| patch.friction)
            .fold(f64::INFINITY, f64::min);
        let max_mu = patches
            .iter()
            .map(|patch| patch.friction)
            .fold(f64::NEG_INFINITY, f64::max);
        let span = max_mu - min_mu;
        finite_outputs &= contact.is_finite()
            && contact.force.iter().all(|v| v.is_finite())
            && terrain.height(x, z).is_finite()
            && terrain.normal(x, z).iter().all(|v| v.is_finite())
            && patches.iter().all(|patch| patch.friction.is_finite());
        min_contact_count = min_contact_count.min(contact.contact_count);
        min_seam_manifold_count = min_seam_manifold_count.min(patches.len());
        min_friction_span = min_friction_span.min(span);
        samples.push((
            terrain.triangles.len(),
            terrain.height(x, z),
            terrain.normal(x, z),
            contact,
            patches.len(),
            span,
        ));
    }

    let (coarse_triangles, coarse_height, coarse_normal, coarse_contact, _, _) =
        samples.first().expect("fixed samples");
    let (fine_triangles, fine_height, fine_normal, fine_contact, _, _) =
        samples.last().expect("fixed samples");
    let normal_delta = dot(
        sub(*coarse_normal, *fine_normal),
        sub(*coarse_normal, *fine_normal),
    )
    .sqrt();
    let force_delta = sub(coarse_contact.force, fine_contact.force);

    TireTriangleDensityConvergenceReport {
        sample_count: samples.len(),
        min_triangle_count: *coarse_triangles,
        max_triangle_count: *fine_triangles,
        finite_outputs,
        height_delta_m: (*coarse_height - *fine_height).abs(),
        normal_delta,
        force_delta_over_load: dot(force_delta, force_delta).sqrt() / load_n.max(1.0),
        patch_delta_over_radius: (coarse_contact.patch_length - fine_contact.patch_length).abs()
            / params.radius.max(1.0e-9),
        min_contact_count,
        min_seam_manifold_count,
        min_friction_span,
    }
}

pub fn tire_triangle_large_manifold_report() -> TireTriangleLargeManifoldReport {
    let terrain = tire_subdivided_two_plate_triangle_terrain(12);
    let seam_patches = terrain.contact_manifold([0.0, 0.08, 0.0], 0.12, 0.02, 24);
    let seam_min_friction = seam_patches
        .iter()
        .map(|patch| patch.friction)
        .fold(f64::INFINITY, f64::min);
    let seam_max_friction = seam_patches
        .iter()
        .map(|patch| patch.friction)
        .fold(f64::NEG_INFINITY, f64::max);
    let (seam_candidate_count, _) = terrain.broadphase_candidate_counts([0.0, 0.0, 0.0]);

    let material = SurfaceMaterial {
        rolling_resistance: 0.0,
        ..SurfaceMaterial::dry_asphalt()
    };
    let params = TetraTireFemParams::realtime_viewer()
        .with_resolution(16, 4, 4)
        .with_pressure_and_load(155_000.0, 4_500.0);
    let mut tire = TetraTireFem::new(0.0, params.pressure_pa, params);
    let dt = 1.0 / 240.0;
    let state = RealWheelState {
        hub_pos: tire.hub(),
        hub_vel: [4.0, 0.0, 1.5],
        heading: [1.0, 0.0, 0.0],
        omega: 4.0 * 1.65 / tire.outer_radius().max(1.0e-6),
    };
    let preview = tire.evaluate_tread_contact(&state, &terrain, &material);
    let feedback = tire.apply_tread_contact_wrench(&state, &terrain, &material, dt);
    let node_sourced = preview
        .cells
        .iter()
        .filter(|cell| cell.node_index.is_some())
        .count();
    let mut feedback_force_cell_count = 0usize;
    let mut feedback_bristle_force_sum_n = 0.0_f64;
    let mut feedback_max_bristle_force_n = 0.0_f64;
    let mut min_feedback_z = f64::INFINITY;
    let mut max_feedback_z = f64::NEG_INFINITY;
    for cell in &feedback.cells {
        let bristle_force = (cell.tangential_force * cell.tangential_force
            + cell.lateral_force * cell.lateral_force)
            .sqrt();
        if bristle_force > 1.0e-9 || cell.normal_load > 1.0e-9 {
            feedback_force_cell_count += 1;
            feedback_bristle_force_sum_n += bristle_force;
            feedback_max_bristle_force_n = feedback_max_bristle_force_n.max(bristle_force);
            min_feedback_z = min_feedback_z.min(cell.local_z);
            max_feedback_z = max_feedback_z.max(cell.local_z);
        }
    }
    let feedback_mean_bristle_force_n = if feedback_force_cell_count > 0 {
        feedback_bristle_force_sum_n / feedback_force_cell_count as f64
    } else {
        0.0
    };
    let feedback_width_span_m = if feedback_force_cell_count > 1 {
        max_feedback_z - min_feedback_z
    } else {
        0.0
    };
    let finite_outputs = seam_patches
        .iter()
        .all(|patch| patch.point.iter().all(|v| v.is_finite()) && patch.friction.is_finite())
        && preview.is_finite()
        && feedback.is_finite()
        && feedback_bristle_force_sum_n.is_finite()
        && feedback_mean_bristle_force_n.is_finite()
        && feedback_max_bristle_force_n.is_finite()
        && feedback_width_span_m.is_finite()
        && tire_nodes_finite(&tire);

    TireTriangleLargeManifoldReport {
        triangle_count: terrain.triangles.len(),
        bvh_node_count: terrain.triangle_bvh.len(),
        seam_candidate_count,
        seam_manifold_patch_count: seam_patches.len(),
        seam_friction_span: seam_max_friction - seam_min_friction,
        fem_contact_count: preview.contact_count,
        fem_node_sourced_contact_count: node_sourced,
        feedback_contact_count: feedback.contact_count,
        feedback_force_norm_n: dot(feedback.force, feedback.force).sqrt(),
        feedback_sliding_fraction: feedback.sliding_fraction,
        feedback_force_cell_count,
        feedback_bristle_force_sum_n,
        feedback_mean_bristle_force_n,
        feedback_max_bristle_force_n,
        feedback_width_span_m,
        finite_outputs,
    }
}
