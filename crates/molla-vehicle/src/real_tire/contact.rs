//! Tire contact patch cell + aggregated contact output.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::{RealTireTelemetrySample, TireContactCellTelemetry, TirePressureDistributionTelemetry};

/// A single contact-patch sample.
#[derive(Clone, Copy, Debug)]
pub struct TireContactCell {
    /// FEM node that produced this contact cell, when the contact came from a
    /// deformed `TetraTireFem` tread node. Ideal sampled brush contacts leave
    /// this as `None`.
    pub node_index: Option<usize>,
    /// Local rolling coordinate in metres. Positive is forward of the wheel
    /// centre.
    pub local_x: f64,
    /// Local lateral/width coordinate in metres, at the brush-strip midpoint
    /// or the deformed FEM tread-node position.
    pub local_z: f64,
    pub penetration: f64,
    pub area: f64,
    pub normal_load: f64,
    /// Local terrain/contact normal at this contact cell.
    ///
    /// Flat floors report `[0, 1, 0]`; heightfields report their local slope.
    /// Wheel-level `RealTireContact::force` accumulates these per-cell normals
    /// instead of assuming one hub-centre normal for the whole patch.
    pub normal: [f64; 3],
    /// Longitudinal brush force for this cell, in the wheel heading
    /// direction. Kept as `tangential_force` for compatibility with the
    /// earlier longitudinal-only brush API.
    pub tangential_force: f64,
    /// Lateral brush force for this cell, in the wheel side direction.
    pub lateral_force: f64,
    pub sliding: bool,
}

/// Total contact output for one tire.
#[derive(Clone, Debug)]
pub struct RealTireContact {
    pub in_contact: bool,
    pub force: [f64; 3],
    pub normal_load: f64,
    pub longitudinal_force: f64,
    pub lateral_force: f64,
    /// Self-aligning yaw moment around the terrain normal, N·m. Positive sign
    /// follows the `heading × side` basis used by the contact patch.
    pub aligning_moment_z: f64,
    /// Camber angle in radians. Positive sign follows the wheel up-vector's
    /// projection onto the tire side direction.
    pub camber_angle_rad: f64,
    /// Contact-pressure centre in local rolling coordinates (m). Positive is
    /// forward of the wheel centre.
    pub contact_center_x: f64,
    /// Lateral-force resultant centre in local rolling coordinates (m).
    pub force_resultant_x: f64,
    /// `contact_center_x - force_resultant_x`, in metres. Positive pneumatic
    /// trail with negative `Fy` produces a restoring/self-aligning moment.
    pub pneumatic_trail_m: f64,
    pub rolling_resistance_force: f64,
    pub slip_ratio: f64,
    pub slip_speed: f64,
    pub lateral_slip_speed: f64,
    pub slip_angle_rad: f64,
    pub patch_length: f64,
    pub contact_count: usize,
    pub sliding_count: usize,
    /// Fraction of active contact cells still mostly in adhesion.
    pub adhesion_fraction: f64,
    /// Fraction of active contact cells in the smooth sliding/saturation band.
    pub sliding_fraction: f64,
    /// Sum of load-sensitive static friction budgets over the patch (N).
    pub available_friction_budget: f64,
    /// Magnitude of the actually consumed tangential friction budget (N).
    pub consumed_friction_budget: f64,
    /// Effective longitudinal tread stiffness at the current operating point
    /// (N/m). This includes load sensitivity and smooth combined-slip
    /// saturation.
    pub effective_longitudinal_stiffness: f64,
    /// Effective cornering stiffness at the current operating point (N/rad).
    pub effective_cornering_stiffness: f64,
    pub cells: Vec<TireContactCell>,
    /// Stage (j) — instantaneous power dissipated at the contact this
    /// substep (W). Equal to `|F_t · v_slip|` summed over the sliding
    /// fraction of the patch — i.e. the energy going into tread heat,
    /// rubber wear and tire body damping. Feed this into a thermal
    /// integrator (see `TireThermalState` / `TireThermalLayers`) for
    /// tire-temperature dynamics.
    pub dissipated_power: f64,
}

impl RealTireContact {
    pub(crate) fn empty() -> Self {
        Self {
            in_contact: false,
            force: [0.0; 3],
            normal_load: 0.0,
            longitudinal_force: 0.0,
            lateral_force: 0.0,
            aligning_moment_z: 0.0,
            camber_angle_rad: 0.0,
            contact_center_x: 0.0,
            force_resultant_x: 0.0,
            pneumatic_trail_m: 0.0,
            rolling_resistance_force: 0.0,
            slip_ratio: 0.0,
            slip_speed: 0.0,
            lateral_slip_speed: 0.0,
            slip_angle_rad: 0.0,
            patch_length: 0.0,
            contact_count: 0,
            sliding_count: 0,
            adhesion_fraction: 0.0,
            sliding_fraction: 0.0,
            available_friction_budget: 0.0,
            consumed_friction_budget: 0.0,
            effective_longitudinal_stiffness: 0.0,
            effective_cornering_stiffness: 0.0,
            cells: Vec::new(),
            dissipated_power: 0.0,
        }
    }

    /// Build a compact, unit-labelled telemetry sample from this contact
    /// result. This is intended for repeatable tire-rig tests and viewer
    /// diagnostics; it does not perform any calibration/fitting.
    pub fn telemetry_sample(&self, dt_s: f64) -> RealTireTelemetrySample {
        RealTireTelemetrySample::from_contact(self, dt_s)
    }

    pub fn is_finite(&self) -> bool {
        self.force.iter().all(|v| v.is_finite())
            && self.normal_load.is_finite()
            && self.longitudinal_force.is_finite()
            && self.lateral_force.is_finite()
            && self.aligning_moment_z.is_finite()
            && self.camber_angle_rad.is_finite()
            && self.contact_center_x.is_finite()
            && self.force_resultant_x.is_finite()
            && self.pneumatic_trail_m.is_finite()
            && self.rolling_resistance_force.is_finite()
            && self.slip_ratio.is_finite()
            && self.slip_speed.is_finite()
            && self.lateral_slip_speed.is_finite()
            && self.slip_angle_rad.is_finite()
            && self.patch_length.is_finite()
            && self.adhesion_fraction.is_finite()
            && self.sliding_fraction.is_finite()
            && self.available_friction_budget.is_finite()
            && self.consumed_friction_budget.is_finite()
            && self.effective_longitudinal_stiffness.is_finite()
            && self.effective_cornering_stiffness.is_finite()
            && self.dissipated_power.is_finite()
            && self
                .cells
                .iter()
                .all(|cell| TireContactCellTelemetry::from_cell(cell).is_finite())
    }

    /// Build per-contact-cell telemetry rows for logging/debug overlays.
    pub fn cell_telemetry(&self) -> Vec<TireContactCellTelemetry> {
        self.cells
            .iter()
            .map(TireContactCellTelemetry::from_cell)
            .collect()
    }

    /// Build pressure-distribution diagnostics along rolling/circumferential
    /// (`local_x`) and tread-width (`local_z`) axes.
    pub fn pressure_distribution(
        &self,
        longitudinal_bins: usize,
        lateral_bins: usize,
    ) -> TirePressureDistributionTelemetry {
        TirePressureDistributionTelemetry::from_contact(self, longitudinal_bins, lateral_bins)
    }
}
