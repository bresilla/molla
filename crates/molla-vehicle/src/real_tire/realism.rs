//! Tire-model provenance + realism-source classification types.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.
//! The builder `tire_realism_source_classification_report()` still lives
//! in `mod.rs` because it walks every named preset across the brush +
//! tetra + terrain stack; it can move here once those stacks split out.

use super::TireRigAcceptanceCheck;
use super::internal::{bool_metric, csv_token};

/// Provenance tag for tire-model parameters and debug outputs.
///
/// This is deliberately lightweight: the current tire plan is not doing
/// calibration/fitting, so presets and rig snapshots need to say whether a
/// value is physical, hand-tuned, numerical, or only for visualization/debug.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireModelProvenance {
    Physical,
    HandTuned,
    NumericalStability,
    Debug,
    VisualizationOnly,
}

impl TireModelProvenance {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Physical => "physical",
            Self::HandTuned => "hand_tuned",
            Self::NumericalStability => "numerical_stability",
            Self::Debug => "debug",
            Self::VisualizationOnly => "visualization_only",
        }
    }
}

/// Source classification for one tire result/control/report entry.
///
/// This is the crate-side counterpart to the viewer truth labels: realistic
/// tire UI/report output must not blur native physics, reference diagnostics,
/// hand-tuned presets, numerical stabilizers, debug data, and visualization
/// helpers into one "realistic" bucket.
#[derive(Clone, Debug)]
pub struct TireRealismSourceItem {
    pub item_id: String,
    pub provenance: TireModelProvenance,
    pub description: String,
    pub can_drive_forces: bool,
}

impl TireRealismSourceItem {
    pub fn new(
        item_id: &str,
        provenance: TireModelProvenance,
        description: &str,
        can_drive_forces: bool,
    ) -> Self {
        Self {
            item_id: csv_token(item_id),
            provenance,
            description: csv_token(description),
            can_drive_forces,
        }
    }

    pub fn is_valid(&self) -> bool {
        !self.item_id.is_empty() && !self.description.is_empty()
    }
}

/// Broad report that enforces honest "what is this result?" labels.
#[derive(Clone, Debug)]
pub struct TireRealismSourceClassificationReport {
    pub items: Vec<TireRealismSourceItem>,
    pub has_native_physics: bool,
    pub has_diagnostic_reference: bool,
    pub has_hand_tuned_preset: bool,
    pub has_numerical_stabilizer: bool,
    pub has_visualization_helper: bool,
    pub diagnostics_are_not_force_source: bool,
}

impl TireRealismSourceClassificationReport {
    pub fn is_finite(&self) -> bool {
        self.items.iter().all(TireRealismSourceItem::is_valid)
    }

    pub fn passed(&self) -> bool {
        self.is_finite()
            && self.has_native_physics
            && self.has_diagnostic_reference
            && self.has_hand_tuned_preset
            && self.has_numerical_stabilizer
            && self.has_visualization_helper
            && self.diagnostics_are_not_force_source
    }

    pub fn acceptance_checks(&self) -> Vec<TireRigAcceptanceCheck> {
        vec![
            TireRigAcceptanceCheck::at_least(
                "realism_sources_have_items",
                "realism source classification should list concrete tire result/control classes",
                "item_count",
                self.items.len() as f64,
                6.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "realism_sources_include_native_physics",
                "realistic tire labels should identify native FEM/brush/terrain physics",
                "has_native_physics",
                bool_metric(self.has_native_physics),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "realism_sources_include_diagnostic_reference",
                "reference curves/reports should be labeled as diagnostics",
                "has_diagnostic_reference",
                bool_metric(self.has_diagnostic_reference),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "realism_sources_include_hand_tuned_preset",
                "hand-tuned tire/surface presets should be labeled as such",
                "has_hand_tuned_preset",
                bool_metric(self.has_hand_tuned_preset),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "realism_sources_include_numerical_stabilizer",
                "solver/XPBD/semi-implicit controls should be labeled as numerical stabilizers",
                "has_numerical_stabilizer",
                bool_metric(self.has_numerical_stabilizer),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "realism_sources_include_visualization_helper",
                "overlays/plots/sweeps should be labeled as visualization helpers",
                "has_visualization_helper",
                bool_metric(self.has_visualization_helper),
                1.0,
            ),
            TireRigAcceptanceCheck::at_least(
                "realism_sources_diagnostics_not_force_source",
                "diagnostic reference shapes must not be classified as force sources",
                "diagnostics_are_not_force_source",
                bool_metric(self.diagnostics_are_not_force_source),
                1.0,
            ),
        ]
    }

    pub fn to_text_report(&self) -> String {
        let mut out = format!(
            "realism_source_classification items={} passed={}",
            self.items.len(),
            self.passed()
        );
        for item in &self.items {
            out.push('\n');
            out.push_str(&format!(
                "{} provenance={} force_source={} desc={}",
                item.item_id,
                item.provenance.label(),
                item.can_drive_forces,
                item.description
            ));
        }
        out
    }
}
pub fn tire_realism_source_classification_report() -> TireRealismSourceClassificationReport {
    let items = vec![
        TireRealismSourceItem::new(
            "native_brush_fem_terrain_force",
            TireModelProvenance::Physical,
            "native brush/FEM/terrain contact path may drive tire forces",
            true,
        ),
        TireRealismSourceItem::new(
            "reference_shape_diagnostics",
            TireModelProvenance::Debug,
            "Magic-Formula/TMeasy-like or internal force-shape reports are diagnostics only",
            false,
        ),
        TireRealismSourceItem::new(
            "named_tire_surface_presets",
            TireModelProvenance::HandTuned,
            "named tire/surface/film/thermal presets are hand-tuned non-calibration inputs",
            true,
        ),
        TireRealismSourceItem::new(
            "xpbd_semi_implicit_solver_knobs",
            TireModelProvenance::NumericalStability,
            "XPBD compliance, substeps, and semi-implicit stiffness scale are numerical stabilizers",
            true,
        ),
        TireRealismSourceItem::new(
            "viewer_overlays_curves_sweeps",
            TireModelProvenance::VisualizationOnly,
            "viewer overlays, plots, and auto-sweeps visualize or falsify behavior but do not drive forces",
            false,
        ),
        TireRealismSourceItem::new(
            "cpu_gpu_delta_readouts",
            TireModelProvenance::Debug,
            "CPU/GPU deltas and parity readouts are debug/verification telemetry",
            false,
        ),
    ];
    let has_native_physics = items
        .iter()
        .any(|item| item.provenance == TireModelProvenance::Physical && item.can_drive_forces);
    let has_diagnostic_reference = items.iter().any(|item| {
        item.provenance == TireModelProvenance::Debug
            && item.item_id.contains("reference")
            && !item.can_drive_forces
    });
    let has_hand_tuned_preset = items
        .iter()
        .any(|item| item.provenance == TireModelProvenance::HandTuned);
    let has_numerical_stabilizer = items
        .iter()
        .any(|item| item.provenance == TireModelProvenance::NumericalStability);
    let has_visualization_helper = items
        .iter()
        .any(|item| item.provenance == TireModelProvenance::VisualizationOnly);
    let diagnostics_are_not_force_source = items
        .iter()
        .filter(|item| {
            item.provenance == TireModelProvenance::Debug
                || item.provenance == TireModelProvenance::VisualizationOnly
        })
        .all(|item| !item.can_drive_forces);
    TireRealismSourceClassificationReport {
        items,
        has_native_physics,
        has_diagnostic_reference,
        has_hand_tuned_preset,
        has_numerical_stabilizer,
        has_visualization_helper,
        diagnostics_are_not_force_source,
    }
}
