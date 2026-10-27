//! Non-calibration observability + runtime production-gate reports.
//!
//! Both reports walk the entire tire stack to verify that named CPU
//! presets exist, diagnostics are properly labelled, and named GPU
//! features have CPU/GPU parity. They are checklists over the existing
//! native paths, not calibration or benchmarks.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::internal::bool_metric;
use super::{
    TireGpuParityItem, TireGpuParityStatus, TireModelProvenance, TireRigAcceptanceCheck,
    tetra_fem_substep_convergence_report, tetra_sidewall_laminate_coupling_report,
    tetra_sidewall_ply_strain_xpbd_report, tire_contact_patch_convergence_report,
    tire_frame_convention_report, tire_gpu_parity_checklist,
    tire_heightfield_resolution_convergence_report, tire_layer_ablation_debug_report,
    tire_load_combined_slip_law_report, tire_low_speed_robustness_report,
    tire_preset_governance_report, tire_realism_source_classification_report,
    tire_thermal_wear_convergence_report, tire_triangle_density_convergence_report,
};

/// Non-calibration observability audit for tire parameters and shader inputs.
///
/// This is the "no hidden constants" counterpart to the preset-governance
/// range checks: it verifies that named CPU presets exist, diagnostic sources
/// are labelled, and the GPU shader contracts expose the important tire,
/// material, thermal, transient, and FEM parameters as input rows/uniforms
/// rather than burying them as untraceable scene-side multipliers.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireParameterObservabilityReport {
    pub governance_check_count: usize,
    pub governance_passed: bool,
    pub realism_sources_passed: bool,
    pub shader_contracts_checked: usize,
    pub brush_shader_exposes_tire_and_material_inputs: bool,
    pub fem_shader_exposes_structural_and_cavity_inputs: bool,
    pub thermal_shader_exposes_heat_wear_inputs: bool,
    pub transient_shader_exposes_layer_inputs: bool,
    pub hidden_force_source_blocked: bool,
}

impl TireParameterObservabilityReport {
    pub fn is_finite(self) -> bool {
        true
    }

    pub fn passed(self) -> bool {
        self.governance_check_count >= 30
            && self.governance_passed
            && self.realism_sources_passed
            && self.shader_contracts_checked >= 4
            && self.brush_shader_exposes_tire_and_material_inputs
            && self.fem_shader_exposes_structural_and_cavity_inputs
            && self.thermal_shader_exposes_heat_wear_inputs
            && self.transient_shader_exposes_layer_inputs
            && self.hidden_force_source_blocked
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "parameter_observability_governance_rows",
                "parameter observability should include broad unit/range/provenance rows",
                "governance_check_count",
                self.governance_check_count as f64,
                30.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "parameter_observability_governance_passed",
                "named tire/surface/thermal/soil/solver presets should pass governance ranges",
                "governance_passed",
                bool_metric(self.governance_passed),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "parameter_observability_realism_sources_labelled",
                "realism source classification should prevent diagnostics from becoming force sources",
                "realism_sources_passed",
                bool_metric(self.realism_sources_passed),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "parameter_observability_brush_shader_inputs_named",
                "brush shader should expose tire/material inputs like pressure, shear, μ, damping, and rolling loss",
                "brush_shader_exposes_tire_and_material_inputs",
                bool_metric(self.brush_shader_exposes_tire_and_material_inputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "parameter_observability_fem_shader_inputs_named",
                "FEM shader should expose structural, pressure, cavity, XPBD, and semi-implicit inputs",
                "fem_shader_exposes_structural_and_cavity_inputs",
                bool_metric(self.fem_shader_exposes_structural_and_cavity_inputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "parameter_observability_thermal_shader_inputs_named",
                "thermal shader should expose heat/wear power and material-response inputs",
                "thermal_shader_exposes_heat_wear_inputs",
                bool_metric(self.thermal_shader_exposes_heat_wear_inputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "parameter_observability_transient_shader_inputs_named",
                "transient shader should expose tread/belt/carcass layer lengths",
                "transient_shader_exposes_layer_inputs",
                bool_metric(self.transient_shader_exposes_layer_inputs),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "parameter_observability_hidden_force_sources_blocked",
                "debug/reference/visual sources should remain labelled as non-force-source diagnostics",
                "hidden_force_source_blocked",
                bool_metric(self.hidden_force_source_blocked),
                1.0,
            ),
        ]
    }
}

pub fn tire_parameter_observability_report() -> TireParameterObservabilityReport {
    let governance = tire_preset_governance_report();
    let realism = tire_realism_source_classification_report();
    let brush = include_str!("../../../molla-kernels/shaders/tire_brush.wgsl");
    let fem = include_str!("../../../molla-kernels/shaders/tire_fem.wgsl");
    let thermal = include_str!("../../../molla-kernels/shaders/tire_thermal.wgsl");
    let transient = include_str!("../../../molla-kernels/shaders/tire_transient.wgsl");
    let contains_all = |source: &str, needles: &[&str]| -> bool {
        needles.iter().all(|needle| source.contains(needle))
    };

    TireParameterObservabilityReport {
        governance_check_count: governance.checks.len(),
        governance_passed: governance.passed(),
        realism_sources_passed: realism.passed(),
        shader_contracts_checked: 4,
        brush_shader_exposes_tire_and_material_inputs: contains_all(
            brush,
            &[
                "pressure_pa",
                "carcass_normal_stiffness",
                "tread_shear_stiffness",
                "mu_static",
                "mu_dynamic",
                "bristle_damping",
                "rolling_resistance",
            ],
        ),
        fem_shader_exposes_structural_and_cavity_inputs: contains_all(
            fem,
            &[
                "young_modulus",
                "poisson_ratio",
                "pressure_pa",
                "cavity_radial_force_per_bead",
                "xpbd_bead_compliance",
                "semi_implicit_stiffness_scale",
                "tet_mu",
            ],
        ),
        thermal_shader_exposes_heat_wear_inputs: contains_all(
            thermal,
            &[
                "sliding_power_w",
                "carcass_flex_power_w",
                "rolling_power_w",
                "brake_rim_power_w",
                "rubber_peak_c",
                "wear_temperature_factor",
            ],
        ),
        transient_shader_exposes_layer_inputs: contains_all(
            transient,
            &[
                "raw_kappa",
                "raw_alpha_rad",
                "tread_long_m",
                "belt_long_m",
                "carcass_long_m",
                "length_scale",
            ],
        ),
        hidden_force_source_blocked: realism.diagnostics_are_not_force_source
            && realism.items.iter().all(|item| {
                item.provenance != TireModelProvenance::VisualizationOnly || !item.can_drive_forces
            }),
    }
}

/// Data-free production runtime gate for the tire stack.
///
/// This keeps the plan's "physics and performance gates together" requirement
/// out of the viewer: broad physics sanity is checked beside runtime-readiness
/// evidence for many-wheel batching, resident GPU ownership, fused dispatches,
/// narrow readback seams, convergence/LOD reports, and explicit CPU-only/debug
/// accounting. It is a checklist over existing native paths, not a benchmark
/// number and not a calibration artifact.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireRuntimeProductionGateReport {
    pub gpu_parity_items: usize,
    pub cpu_and_gpu_items: usize,
    pub cpu_only_debug_items: usize,
    pub planned_gpu_items: usize,
    pub resident_gpu_features: usize,
    pub many_wheel_batch_features: usize,
    pub fused_dispatch_features: usize,
    pub readback_seam_features: usize,
    pub convergence_reports_passed: usize,
    pub convergence_reports_required: usize,
    pub cpu_only_debug_accounted: bool,
    pub all_gpu_items_have_proof: bool,
    pub broad_physics_gates_passed: bool,
}

impl TireRuntimeProductionGateReport {
    pub fn is_finite(self) -> bool {
        true
    }

    pub fn passed(self) -> bool {
        self.gpu_parity_items >= 8
            && self.cpu_and_gpu_items >= 6
            && self.resident_gpu_features >= 4
            && self.many_wheel_batch_features >= 3
            && self.fused_dispatch_features >= 2
            && self.readback_seam_features >= 2
            && self.convergence_reports_passed >= self.convergence_reports_required
            && self.convergence_reports_required >= 5
            && self.cpu_only_debug_accounted
            && self.all_gpu_items_have_proof
            && self.broad_physics_gates_passed
    }

    pub fn acceptance_checks(self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_gpu_matrix_has_rows",
                "runtime gate should cover the major tire GPU/CPU paths",
                "gpu_parity_items",
                self.gpu_parity_items as f64,
                8.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_cpu_gpu_paths_proven",
                "performance-relevant tire paths should have GPU proof rows",
                "cpu_and_gpu_items",
                self.cpu_and_gpu_items as f64,
                6.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_resident_gpu_buffers_tracked",
                "runtime gate should track resident GPU buffer ownership paths",
                "resident_gpu_features",
                self.resident_gpu_features as f64,
                4.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_many_wheel_batching_tracked",
                "many-wheel batching paths should be visible before calling the stack production-ready",
                "many_wheel_batch_features",
                self.many_wheel_batch_features as f64,
                3.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_fused_dispatch_tracked",
                "runtime gate should track fused/atlas dispatch paths rather than only per-tire loops",
                "fused_dispatch_features",
                self.fused_dispatch_features as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_readback_seams_tracked",
                "runtime gate should track narrow readback/no-readback seams",
                "readback_seam_features",
                self.readback_seam_features as f64,
                2.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_convergence_reports_pass",
                "LOD/convergence reports should pass before raising runtime resolution",
                "convergence_reports_passed",
                self.convergence_reports_passed as f64,
                self.convergence_reports_required as f64,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_cpu_only_debug_accounted",
                "CPU-only/debug tire paths should carry explicit reason and follow-up rows",
                "cpu_only_debug_accounted",
                bool_metric(self.cpu_only_debug_accounted),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_gpu_proofs_present",
                "CPU+GPU tire paths should carry explicit proof strings/tests",
                "all_gpu_items_have_proof",
                bool_metric(self.all_gpu_items_have_proof),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "runtime_gate_broad_physics_passed",
                "runtime readiness must stay tied to physics sanity gates, not only dispatch ownership",
                "broad_physics_gates_passed",
                bool_metric(self.broad_physics_gates_passed),
                1.0,
            ),
        ]
    }
}

pub fn tire_runtime_production_gate_report() -> TireRuntimeProductionGateReport {
    let parity = tire_gpu_parity_checklist();
    let gpu_parity_items = parity.len();
    let cpu_and_gpu_items = parity
        .iter()
        .filter(|item| item.status == TireGpuParityStatus::CpuAndGpu)
        .count();
    let cpu_only_debug_items = parity
        .iter()
        .filter(|item| item.status == TireGpuParityStatus::CpuOnlyDebug)
        .count();
    let planned_gpu_items = parity
        .iter()
        .filter(|item| item.status == TireGpuParityStatus::PlannedGpu)
        .count();
    let contains_any = |item: &&TireGpuParityItem, needles: &[&str]| {
        needles.iter().any(|needle| {
            item.cpu_path.contains(needle)
                || item.gpu_path.contains(needle)
                || item.proof.contains(needle)
                || item.follow_up.contains(needle)
        })
    };
    let resident_gpu_features = parity
        .iter()
        .filter(|item| contains_any(item, &["resident", "Resident", "no-readback"]))
        .count();
    let many_wheel_batch_features = parity
        .iter()
        .filter(|item| contains_any(item, &["batch", "Batch", "many-wheel", "multiple_tires"]))
        .count();
    let fused_dispatch_features = parity
        .iter()
        .filter(|item| contains_any(item, &["fused", "atlas", "Atlas", "one-dispatch"]))
        .count();
    let readback_seam_features = parity
        .iter()
        .filter(|item| contains_any(item, &["readback", "no-readback", "without staging"]))
        .count();
    let cpu_only_debug_accounted = parity
        .iter()
        .filter(|item| item.status.is_cpu_only_debug())
        .all(|item| !item.reason.is_empty() && !item.follow_up.is_empty());
    let all_gpu_items_have_proof = parity
        .iter()
        .filter(|item| item.status.has_gpu_path())
        .all(|item| !item.gpu_path.is_empty() && !item.proof.is_empty());

    let convergence_passes = [
        tire_contact_patch_convergence_report().passed(),
        tetra_fem_substep_convergence_report().passed(),
        tire_heightfield_resolution_convergence_report().passed(),
        tire_triangle_density_convergence_report().passed(),
        tire_thermal_wear_convergence_report().passed(),
    ];
    let convergence_reports_passed = convergence_passes.iter().filter(|&&passed| passed).count();
    let convergence_reports_required = convergence_passes.len();

    let broad_physics_gates_passed = tire_low_speed_robustness_report().passed()
        && tire_frame_convention_report().passed()
        && tire_load_combined_slip_law_report().passed()
        && tire_layer_ablation_debug_report().passed()
        && tetra_sidewall_laminate_coupling_report().passed()
        && tetra_sidewall_ply_strain_xpbd_report().passed();

    TireRuntimeProductionGateReport {
        gpu_parity_items,
        cpu_and_gpu_items,
        cpu_only_debug_items,
        planned_gpu_items,
        resident_gpu_features,
        many_wheel_batch_features,
        fused_dispatch_features,
        readback_seam_features,
        convergence_reports_passed,
        convergence_reports_required,
        cpu_only_debug_accounted,
        all_gpu_items_have_proof,
        broad_physics_gates_passed,
    }
}
