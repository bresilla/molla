//! Unit-labelled tire-contact telemetry types and pressure-distribution
//! histograms.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::internal::{csv_token, finite_nonnegative, finite_or};
use super::{RealTireContact, TireContactCell};

/// Unit-labelled summary of the active tire contact patch.
///
/// This intentionally lives in `molla-vehicle` instead of the viewer so CPU
/// tests, GPU parity tests, and debug UI can all inspect the same accounting.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireContactPatchTelemetry {
    pub contact_count: usize,
    pub sliding_count: usize,
    pub patch_length_m: f64,
    pub contact_area_m2: f64,
    pub mean_normal_pressure_pa: f64,
    pub peak_normal_pressure_pa: f64,
    pub contact_center_x_m: f64,
    pub force_resultant_x_m: f64,
    pub pneumatic_trail_m: f64,
    pub adhesion_fraction: f64,
    pub sliding_fraction: f64,
}

impl TireContactPatchTelemetry {
    pub fn from_contact(contact: &RealTireContact) -> Self {
        let mut contact_area_m2: f64 = 0.0;
        let mut peak_normal_pressure_pa: f64 = 0.0;
        for cell in &contact.cells {
            let area = finite_nonnegative(cell.area);
            contact_area_m2 += area;
            if area > 1.0e-12 {
                peak_normal_pressure_pa =
                    peak_normal_pressure_pa.max(finite_nonnegative(cell.normal_load) / area);
            }
        }
        let mean_normal_pressure_pa = if contact_area_m2 > 1.0e-12 {
            finite_nonnegative(contact.normal_load) / contact_area_m2
        } else {
            0.0
        };

        Self {
            contact_count: contact.contact_count,
            sliding_count: contact.sliding_count,
            patch_length_m: finite_nonnegative(contact.patch_length),
            contact_area_m2,
            mean_normal_pressure_pa,
            peak_normal_pressure_pa,
            contact_center_x_m: finite_or(contact.contact_center_x, 0.0),
            force_resultant_x_m: finite_or(contact.force_resultant_x, 0.0),
            pneumatic_trail_m: finite_or(contact.pneumatic_trail_m, 0.0),
            adhesion_fraction: finite_nonnegative(contact.adhesion_fraction).min(1.0),
            sliding_fraction: finite_nonnegative(contact.sliding_fraction).min(1.0),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.patch_length_m.is_finite()
            && self.contact_area_m2.is_finite()
            && self.mean_normal_pressure_pa.is_finite()
            && self.peak_normal_pressure_pa.is_finite()
            && self.contact_center_x_m.is_finite()
            && self.force_resultant_x_m.is_finite()
            && self.pneumatic_trail_m.is_finite()
            && self.adhesion_fraction.is_finite()
            && self.sliding_fraction.is_finite()
    }
}

/// Per-contact-cell telemetry with explicit units.
#[derive(Clone, Copy, Debug)]
pub struct TireContactCellTelemetry {
    pub node_index: Option<usize>,
    pub local_x_m: f64,
    pub local_z_m: f64,
    pub penetration_m: f64,
    pub area_m2: f64,
    pub normal_load_n: f64,
    pub normal_pressure_pa: f64,
    pub normal: [f64; 3],
    pub longitudinal_force_n: f64,
    pub lateral_force_n: f64,
    pub sliding: bool,
}

impl TireContactCellTelemetry {
    pub const CSV_HEADER: &'static str = concat!(
        "sample_time_s,wheel_id,cell_index,node_index,local_x_m,local_z_m,penetration_m,",
        "area_m2,normal_load_n,normal_pressure_pa,normal_x,normal_y,normal_z,",
        "longitudinal_force_n,lateral_force_n,sliding"
    );

    pub fn from_cell(cell: &TireContactCell) -> Self {
        let area_m2 = finite_nonnegative(cell.area);
        let normal_load_n = finite_nonnegative(cell.normal_load);
        Self {
            node_index: cell.node_index,
            local_x_m: finite_or(cell.local_x, 0.0),
            local_z_m: finite_or(cell.local_z, 0.0),
            penetration_m: finite_nonnegative(cell.penetration),
            area_m2,
            normal_load_n,
            normal_pressure_pa: if area_m2 > 1.0e-12 {
                normal_load_n / area_m2
            } else {
                0.0
            },
            normal: [
                finite_or(cell.normal[0], 0.0),
                finite_or(cell.normal[1], 1.0),
                finite_or(cell.normal[2], 0.0),
            ],
            longitudinal_force_n: finite_or(cell.tangential_force, 0.0),
            lateral_force_n: finite_or(cell.lateral_force, 0.0),
            sliding: cell.sliding,
        }
    }

    pub fn is_finite(&self) -> bool {
        self.local_x_m.is_finite()
            && self.local_z_m.is_finite()
            && self.penetration_m.is_finite()
            && self.area_m2.is_finite()
            && self.normal_load_n.is_finite()
            && self.normal_pressure_pa.is_finite()
            && self.normal.iter().all(|v| v.is_finite())
            && self.longitudinal_force_n.is_finite()
            && self.lateral_force_n.is_finite()
    }

    /// CSV row for deterministic logs. `wheel_id` is sanitized to avoid
    /// commas/newlines because this is intentionally dependency-free.
    pub fn to_csv_row(&self, sample_time_s: f64, wheel_id: &str, cell_index: usize) -> String {
        format!(
            "{:.6},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.6},{:.6},{:.9},{:.9},{:.9},{:.6},{:.6},{}",
            finite_or(sample_time_s, 0.0),
            csv_token(wheel_id),
            cell_index,
            self.node_index
                .map(|id| id.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            finite_or(self.local_x_m, 0.0),
            finite_or(self.local_z_m, 0.0),
            finite_or(self.penetration_m, 0.0),
            finite_or(self.area_m2, 0.0),
            finite_or(self.normal_load_n, 0.0),
            finite_or(self.normal_pressure_pa, 0.0),
            finite_or(self.normal[0], 0.0),
            finite_or(self.normal[1], 1.0),
            finite_or(self.normal[2], 0.0),
            finite_or(self.longitudinal_force_n, 0.0),
            finite_or(self.lateral_force_n, 0.0),
            self.sliding
        )
    }
}

/// One pressure-distribution bin along either the rolling/circumferential axis
/// or the tread-width axis.
#[derive(Clone, Copy, Debug, Default)]
pub struct TirePressureDistributionBin {
    pub axis_min_m: f64,
    pub axis_max_m: f64,
    pub axis_center_m: f64,
    pub contact_count: usize,
    pub sliding_count: usize,
    pub area_m2: f64,
    pub normal_load_n: f64,
    pub mean_normal_pressure_pa: f64,
    pub peak_normal_pressure_pa: f64,
}

impl TirePressureDistributionBin {
    fn new(axis_min_m: f64, axis_max_m: f64) -> Self {
        let axis_min_m = finite_or(axis_min_m, 0.0);
        let axis_max_m = finite_or(axis_max_m, axis_min_m);
        Self {
            axis_min_m,
            axis_max_m,
            axis_center_m: 0.5 * (axis_min_m + axis_max_m),
            ..Default::default()
        }
    }

    fn add_cell(&mut self, cell: &TireContactCell) {
        let area = finite_nonnegative(cell.area);
        let load = finite_nonnegative(cell.normal_load);
        self.contact_count += 1;
        self.sliding_count += usize::from(cell.sliding);
        self.area_m2 += area;
        self.normal_load_n += load;
        if area > 1.0e-12 {
            self.peak_normal_pressure_pa = self.peak_normal_pressure_pa.max(load / area);
        }
    }

    fn finalize(&mut self) {
        self.mean_normal_pressure_pa = if self.area_m2 > 1.0e-12 {
            self.normal_load_n / self.area_m2
        } else {
            0.0
        };
    }

    pub fn sliding_fraction(&self) -> f64 {
        if self.contact_count == 0 {
            0.0
        } else {
            self.sliding_count as f64 / self.contact_count as f64
        }
    }

    pub fn is_finite(&self) -> bool {
        self.axis_min_m.is_finite()
            && self.axis_max_m.is_finite()
            && self.axis_center_m.is_finite()
            && self.area_m2.is_finite()
            && self.normal_load_n.is_finite()
            && self.mean_normal_pressure_pa.is_finite()
            && self.peak_normal_pressure_pa.is_finite()
            && self.sliding_fraction().is_finite()
    }
}

/// Contact pressure distribution along tire circumference/rolling direction
/// and tread width.
///
/// This is diagnostic/falsification data, not calibration data. It makes
/// "stiff tire = short high-pressure patch", "soft tire = spread-out patch",
/// and "FEM obstacle contact should not collapse into one invisible point"
/// inspectable from reusable crate tests and viewer logs.
#[derive(Clone, Debug, Default)]
pub struct TirePressureDistributionTelemetry {
    pub longitudinal_bins: Vec<TirePressureDistributionBin>,
    pub lateral_bins: Vec<TirePressureDistributionBin>,
    pub total_normal_load_n: f64,
    pub total_area_m2: f64,
    pub peak_normal_pressure_pa: f64,
    pub longitudinal_span_m: f64,
    pub lateral_span_m: f64,
}

impl TirePressureDistributionTelemetry {
    pub fn from_contact(
        contact: &RealTireContact,
        longitudinal_bins: usize,
        lateral_bins: usize,
    ) -> Self {
        let longitudinal_bins =
            pressure_distribution_axis_bins(&contact.cells, longitudinal_bins, |cell| {
                finite_or(cell.local_x, 0.0)
            });
        let lateral_bins = pressure_distribution_axis_bins(&contact.cells, lateral_bins, |cell| {
            finite_or(cell.local_z, 0.0)
        });
        let total_normal_load_n = contact
            .cells
            .iter()
            .map(|cell| finite_nonnegative(cell.normal_load))
            .sum();
        let total_area_m2 = contact
            .cells
            .iter()
            .map(|cell| finite_nonnegative(cell.area))
            .sum();
        let peak_normal_pressure_pa = contact
            .cells
            .iter()
            .filter_map(|cell| {
                let area = finite_nonnegative(cell.area);
                (area > 1.0e-12).then_some(finite_nonnegative(cell.normal_load) / area)
            })
            .fold(0.0, f64::max);
        let longitudinal_span_m = pressure_axis_span(&contact.cells, |cell| cell.local_x);
        let lateral_span_m = pressure_axis_span(&contact.cells, |cell| cell.local_z);
        Self {
            longitudinal_bins,
            lateral_bins,
            total_normal_load_n,
            total_area_m2,
            peak_normal_pressure_pa,
            longitudinal_span_m,
            lateral_span_m,
        }
    }

    pub fn is_finite(&self) -> bool {
        self.total_normal_load_n.is_finite()
            && self.total_area_m2.is_finite()
            && self.peak_normal_pressure_pa.is_finite()
            && self.longitudinal_span_m.is_finite()
            && self.lateral_span_m.is_finite()
            && self
                .longitudinal_bins
                .iter()
                .all(TirePressureDistributionBin::is_finite)
            && self
                .lateral_bins
                .iter()
                .all(TirePressureDistributionBin::is_finite)
    }
}

fn pressure_axis_span(
    cells: &[TireContactCell],
    coordinate: impl Fn(&TireContactCell) -> f64,
) -> f64 {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for cell in cells {
        let v = finite_or(coordinate(cell), 0.0);
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if min_v.is_finite() && max_v.is_finite() {
        (max_v - min_v).max(0.0)
    } else {
        0.0
    }
}

fn pressure_distribution_axis_bins(
    cells: &[TireContactCell],
    requested_bins: usize,
    coordinate: impl Fn(&TireContactCell) -> f64,
) -> Vec<TirePressureDistributionBin> {
    if cells.is_empty() {
        return Vec::new();
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for cell in cells {
        let v = finite_or(coordinate(cell), 0.0);
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        return Vec::new();
    }
    let span = (max_v - min_v).max(0.0);
    let count = if span <= 1.0e-9 {
        1
    } else {
        requested_bins.clamp(1, 64)
    };
    let width = if count == 1 {
        span.max(1.0e-9)
    } else {
        span / count as f64
    };
    let mut bins = (0..count)
        .map(|i| {
            let lo = if count == 1 {
                min_v
            } else {
                min_v + i as f64 * width
            };
            let hi = if count == 1 || i + 1 == count {
                max_v
            } else {
                min_v + (i + 1) as f64 * width
            };
            TirePressureDistributionBin::new(lo, hi)
        })
        .collect::<Vec<_>>();
    for cell in cells {
        let v = finite_or(coordinate(cell), 0.0);
        let idx = if count == 1 {
            0
        } else {
            (((v - min_v) / width).floor() as isize).clamp(0, count as isize - 1) as usize
        };
        bins[idx].add_cell(cell);
    }
    for bin in &mut bins {
        bin.finalize();
    }
    bins
}

/// Wheel-level telemetry for repeatable tire-rig scenarios and viewer
/// diagnostics.
///
/// Fields include units in the names to avoid mixing physical quantities with
/// numerical/debug factors. The sample is derived from model state; it is not
/// a fitted curve or external calibration record.
#[derive(Clone, Debug)]
pub struct RealTireTelemetrySample {
    pub in_contact: bool,
    pub force_n: [f64; 3],
    pub normal_load_n: f64,
    pub longitudinal_force_n: f64,
    pub lateral_force_n: f64,
    pub aligning_moment_nm: f64,
    pub rolling_resistance_force_n: f64,
    pub slip_ratio: f64,
    pub slip_speed_m_per_s: f64,
    pub lateral_slip_speed_m_per_s: f64,
    pub slip_angle_rad: f64,
    pub camber_angle_rad: f64,
    pub dissipated_power_w: f64,
    pub dissipated_energy_j: f64,
    pub available_friction_budget_n: f64,
    pub consumed_friction_budget_n: f64,
    pub effective_longitudinal_stiffness_n_per_m: f64,
    pub effective_cornering_stiffness_n_per_rad: f64,
    pub patch: TireContactPatchTelemetry,
}

impl RealTireTelemetrySample {
    pub const CSV_HEADER: &'static str = concat!(
        "sample_time_s,wheel_id,in_contact,force_x_n,force_y_n,force_z_n,",
        "normal_load_n,longitudinal_force_n,lateral_force_n,aligning_moment_nm,",
        "rolling_resistance_force_n,slip_ratio,slip_speed_m_per_s,",
        "lateral_slip_speed_m_per_s,slip_angle_rad,camber_angle_rad,",
        "dissipated_power_w,dissipated_energy_j,available_friction_budget_n,",
        "consumed_friction_budget_n,effective_longitudinal_stiffness_n_per_m,",
        "effective_cornering_stiffness_n_per_rad,contact_count,sliding_count,",
        "patch_length_m,contact_area_m2,mean_normal_pressure_pa,",
        "peak_normal_pressure_pa,contact_center_x_m,force_resultant_x_m,",
        "pneumatic_trail_m,adhesion_fraction,sliding_fraction"
    );

    pub fn from_contact(contact: &RealTireContact, dt_s: f64) -> Self {
        let dt_s = finite_nonnegative(dt_s);
        let dissipated_power_w = finite_nonnegative(contact.dissipated_power);
        Self {
            in_contact: contact.in_contact,
            force_n: [
                finite_or(contact.force[0], 0.0),
                finite_or(contact.force[1], 0.0),
                finite_or(contact.force[2], 0.0),
            ],
            normal_load_n: finite_nonnegative(contact.normal_load),
            longitudinal_force_n: finite_or(contact.longitudinal_force, 0.0),
            lateral_force_n: finite_or(contact.lateral_force, 0.0),
            aligning_moment_nm: finite_or(contact.aligning_moment_z, 0.0),
            rolling_resistance_force_n: finite_or(contact.rolling_resistance_force, 0.0),
            slip_ratio: finite_or(contact.slip_ratio, 0.0),
            slip_speed_m_per_s: finite_or(contact.slip_speed, 0.0),
            lateral_slip_speed_m_per_s: finite_or(contact.lateral_slip_speed, 0.0),
            slip_angle_rad: finite_or(contact.slip_angle_rad, 0.0),
            camber_angle_rad: finite_or(contact.camber_angle_rad, 0.0),
            dissipated_power_w,
            dissipated_energy_j: dissipated_power_w * dt_s,
            available_friction_budget_n: finite_nonnegative(contact.available_friction_budget),
            consumed_friction_budget_n: finite_nonnegative(contact.consumed_friction_budget),
            effective_longitudinal_stiffness_n_per_m: finite_nonnegative(
                contact.effective_longitudinal_stiffness,
            ),
            effective_cornering_stiffness_n_per_rad: finite_nonnegative(
                contact.effective_cornering_stiffness,
            ),
            patch: TireContactPatchTelemetry::from_contact(contact),
        }
    }

    pub fn is_finite(&self) -> bool {
        self.force_n.iter().all(|v| v.is_finite())
            && self.normal_load_n.is_finite()
            && self.longitudinal_force_n.is_finite()
            && self.lateral_force_n.is_finite()
            && self.aligning_moment_nm.is_finite()
            && self.rolling_resistance_force_n.is_finite()
            && self.slip_ratio.is_finite()
            && self.slip_speed_m_per_s.is_finite()
            && self.lateral_slip_speed_m_per_s.is_finite()
            && self.slip_angle_rad.is_finite()
            && self.camber_angle_rad.is_finite()
            && self.dissipated_power_w.is_finite()
            && self.dissipated_energy_j.is_finite()
            && self.available_friction_budget_n.is_finite()
            && self.consumed_friction_budget_n.is_finite()
            && self.effective_longitudinal_stiffness_n_per_m.is_finite()
            && self.effective_cornering_stiffness_n_per_rad.is_finite()
            && self.patch.is_finite()
    }

    /// CSV row for deterministic tire-rig/viewer telemetry logs.
    ///
    /// This intentionally avoids any external logging/CSV dependency. The
    /// header carries units in each field name so downstream scripts can stay
    /// simple and unambiguous.
    pub fn to_csv_row(&self, sample_time_s: f64, wheel_id: &str) -> String {
        format!(
            "{:.6},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.9},{:.6},{:.6},{:.9},{:.9},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{},{:.9},{:.9},{:.6},{:.6},{:.9},{:.9},{:.9},{:.6},{:.6}",
            finite_or(sample_time_s, 0.0),
            csv_token(wheel_id),
            self.in_contact,
            finite_or(self.force_n[0], 0.0),
            finite_or(self.force_n[1], 0.0),
            finite_or(self.force_n[2], 0.0),
            finite_or(self.normal_load_n, 0.0),
            finite_or(self.longitudinal_force_n, 0.0),
            finite_or(self.lateral_force_n, 0.0),
            finite_or(self.aligning_moment_nm, 0.0),
            finite_or(self.rolling_resistance_force_n, 0.0),
            finite_or(self.slip_ratio, 0.0),
            finite_or(self.slip_speed_m_per_s, 0.0),
            finite_or(self.lateral_slip_speed_m_per_s, 0.0),
            finite_or(self.slip_angle_rad, 0.0),
            finite_or(self.camber_angle_rad, 0.0),
            finite_or(self.dissipated_power_w, 0.0),
            finite_or(self.dissipated_energy_j, 0.0),
            finite_or(self.available_friction_budget_n, 0.0),
            finite_or(self.consumed_friction_budget_n, 0.0),
            finite_or(self.effective_longitudinal_stiffness_n_per_m, 0.0),
            finite_or(self.effective_cornering_stiffness_n_per_rad, 0.0),
            self.patch.contact_count,
            self.patch.sliding_count,
            finite_or(self.patch.patch_length_m, 0.0),
            finite_or(self.patch.contact_area_m2, 0.0),
            finite_or(self.patch.mean_normal_pressure_pa, 0.0),
            finite_or(self.patch.peak_normal_pressure_pa, 0.0),
            finite_or(self.patch.contact_center_x_m, 0.0),
            finite_or(self.patch.force_resultant_x_m, 0.0),
            finite_or(self.patch.pneumatic_trail_m, 0.0),
            finite_or(self.patch.adhesion_fraction, 0.0),
            finite_or(self.patch.sliding_fraction, 0.0)
        )
    }
}
