//! V5 — terramechanics: deformable terrain.
//!
//! A [`DeformableTerrain`] is a heightfield that plastically sinks
//! under wheel load: each loaded cell is pressed down by a
//! Bekker-style pressure-sinkage relation and stays down, so a driven
//! vehicle leaves a rut. The sinkage also feeds back a rolling
//! resistance onto each wheel — the "soft soil costs you" drawbar
//! effect. It implements [`Terrain`], so any [`Vehicle`] drives on it
//! unchanged.
//!
//! This is a pragmatic 2.5-D soil model. Full particle-soil coupling
//! (the vehicle two-way-coupled to molla's MPM Drucker-Prager sand) is
//! the deeper version and is left as a follow-up — see `PLAN_ENGINE.md`.

use crate::real_tire::{
    RealTireContact, RealTireParams, RealTireTelemetrySample, RealWheelState, SurfaceMaterial,
    evaluate_brush_tire, solve_loaded_tire_hub_height,
};
use crate::terrain::{FlatTerrain, Terrain};
use crate::vehicle::Vehicle;

/// Summary returned when a tire contact patch plastically deforms a soil
/// heightfield.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireSoilFeedback {
    /// Number of tire contact cells that touched/deformed terrain cells.
    pub compacted_cells: usize,
    /// Maximum plastic sinkage after this update.
    pub max_sinkage_m: f64,
    /// Normal sinkage target accumulated from Bekker pressure-sinkage.
    pub normal_sinkage_m: f64,
    /// Additional shear/excavation sinkage from tread slip.
    pub shear_sinkage_m: f64,
    /// Bekker-style motion resistance estimate from the produced rut.
    pub rolling_resistance_n: f64,
}

/// One deterministic deformable-soil tire-rig sample.
///
/// This is the deformable counterpart to the brush/material tire-rig matrix:
/// it evaluates a tire contact patch, applies that patch into a cloned Bekker
/// heightfield, and records both the contact telemetry and the resulting rut
/// feedback.
#[derive(Clone, Debug)]
pub struct TireSoilRutRigPoint {
    pub slip_ratio_command: f64,
    pub load_n: f64,
    pub speed_m_per_s: f64,
    pub hub_height_m: f64,
    pub contact: RealTireTelemetrySample,
    pub feedback: TireSoilFeedback,
}

impl TireSoilRutRigPoint {
    pub fn is_finite(&self) -> bool {
        self.slip_ratio_command.is_finite()
            && self.load_n.is_finite()
            && self.speed_m_per_s.is_finite()
            && self.hub_height_m.is_finite()
            && self.contact.is_finite()
            && self.feedback.max_sinkage_m.is_finite()
            && self.feedback.normal_sinkage_m.is_finite()
            && self.feedback.shear_sinkage_m.is_finite()
            && self.feedback.rolling_resistance_n.is_finite()
    }
}

/// A heightfield that deforms plastically under wheel loads.
#[derive(Clone, Debug)]
pub struct DeformableTerrain {
    /// World `(x, z)` of sample `(0, 0)`.
    pub origin: [f64; 2],
    /// Grid spacing.
    pub spacing: f64,
    /// Samples per row.
    pub nx: usize,
    /// Number of rows.
    pub nz: usize,
    /// Current (deformed) surface, row-major.
    pub heights: Vec<f64>,
    /// Undeformed reference surface.
    pub rest_heights: Vec<f64>,
    /// Pressure-sinkage modulus in `p = stiffness · z^n`. With
    /// `bekker_n == 1.0` (default) the curve is linear and this
    /// reduces to the legacy `p = stiffness · z` relation.
    pub stiffness: f64,
    /// Maximum plastic sinkage (m).
    pub max_sinkage: f64,
    /// Surface friction coefficient.
    pub friction: f64,
    /// Soil shear strength (Pa) used by tire-contact rut/excavation coupling.
    pub shear_strength: f64,
    /// Fraction of over-shear converted into additional plastic sinkage.
    pub shear_sinkage_gain: f64,
    /// Bekker sinkage exponent `n` in `p = stiffness · z^n`. `1.0`
    /// (default) is the pre-existing linear behaviour. Sandy soil is
    /// `n ≈ 0.7–0.9`; cohesive clay can be `> 1`. Inverted in
    /// `apply_load` to recover sinkage from pressure:
    /// `z = (p / stiffness)^(1/n)`.
    pub bekker_n: f64,
}

impl DeformableTerrain {
    /// A flat deformable patch at height `level`.
    pub fn flat(
        origin: [f64; 2],
        spacing: f64,
        nx: usize,
        nz: usize,
        level: f64,
        stiffness: f64,
        friction: f64,
    ) -> Self {
        Self {
            origin,
            spacing,
            nx,
            nz,
            heights: vec![level; nx * nz],
            rest_heights: vec![level; nx * nz],
            stiffness,
            max_sinkage: 0.12,
            friction,
            shear_strength: 18_000.0,
            shear_sinkage_gain: 0.025,
            bekker_n: 1.0,
        }
    }

    /// Hand-tuned compacted-dirt Bekker patch preset.
    pub fn compacted_dirt_patch(
        origin: [f64; 2],
        spacing: f64,
        nx: usize,
        nz: usize,
        level: f64,
    ) -> Self {
        let mut soil = Self::flat(origin, spacing, nx, nz, level, 1.6e6, 0.72);
        soil.max_sinkage = 0.08;
        soil.shear_strength = 24_000.0;
        soil.shear_sinkage_gain = 0.018;
        soil.bekker_n = 0.95;
        soil
    }

    /// Hand-tuned sandy-loam / loose-soil Bekker patch preset.
    pub fn sandy_loam_patch(
        origin: [f64; 2],
        spacing: f64,
        nx: usize,
        nz: usize,
        level: f64,
    ) -> Self {
        let mut soil = Self::flat(origin, spacing, nx, nz, level, 7.5e5, 0.58);
        soil.max_sinkage = 0.14;
        soil.shear_strength = 12_000.0;
        soil.shear_sinkage_gain = 0.045;
        soil.bekker_n = 0.82;
        soil
    }

    /// Very soft wet/clay-like validation patch. This is still the 2.5-D
    /// Bekker model, not a DEM/MPM soil.
    pub fn soft_clay_patch(
        origin: [f64; 2],
        spacing: f64,
        nx: usize,
        nz: usize,
        level: f64,
    ) -> Self {
        let mut soil = Self::flat(origin, spacing, nx, nz, level, 3.8e5, 0.42);
        soil.max_sinkage = 0.18;
        soil.shear_strength = 7_500.0;
        soil.shear_sinkage_gain = 0.065;
        soil.bekker_n = 1.15;
        soil
    }

    /// Index of the grid cell nearest `(x, z)`.
    fn cell(&self, x: f64, z: f64) -> usize {
        let ix = (((x - self.origin[0]) / self.spacing).round() as i64).clamp(0, self.nx as i64 - 1)
            as usize;
        let iz = (((z - self.origin[1]) / self.spacing).round() as i64).clamp(0, self.nz as i64 - 1)
            as usize;
        iz * self.nx + ix
    }

    /// Current plastic sinkage (m) at the cell nearest `(x, z)`.
    pub fn sinkage_at(&self, x: f64, z: f64) -> f64 {
        let c = self.cell(x, z);
        (self.rest_heights[c] - self.heights[c]).max(0.0)
    }

    /// Plastically depress the terrain under a wheel of normal load
    /// `fz` (N) at `(x, z)`. The cell ratchets down — it never recovers.
    pub fn apply_load(&mut self, x: f64, z: f64, fz: f64) {
        let area = self.spacing * self.spacing;
        let pressure = (fz / area).max(0.0);
        self.apply_pressure(x, z, pressure);
    }

    /// Plastically depress terrain from direct pressure (Pa) at `(x, z)`.
    /// This is the right primitive for tire contact cells, which already
    /// know their physical contact area and local normal load.
    pub fn apply_pressure(&mut self, x: f64, z: f64, pressure: f64) {
        let c = self.cell(x, z);
        // Bekker `p = stiffness · z^n` ⇒ `z = (p / stiffness)^(1/n)`.
        // `bekker_n == 1.0` collapses to the legacy linear formula.
        let ratio = pressure.max(0.0) / self.stiffness.max(1.0e-9);
        let n = self.bekker_n.max(1.0e-6);
        let sinkage = ratio.powf(1.0 / n).min(self.max_sinkage);
        let target = self.rest_heights[c] - sinkage;
        if self.heights[c] > target {
            self.heights[c] = target;
        }
    }

    /// Couple an already-evaluated tire contact patch into this deformable
    /// terrain.
    ///
    /// Normal pressure follows Bekker pressure-sinkage. Tangential
    /// over-shear converts a bounded fraction of contact shear into extra
    /// plastic sinkage, so a spinning/sliding tire digs a deeper rut than a
    /// purely rolling tire at the same load.
    pub fn apply_tire_contact(
        &mut self,
        wheel: &RealWheelState,
        contact: &RealTireContact,
        tire_radius: f64,
    ) -> TireSoilFeedback {
        if !contact.in_contact || contact.cells.is_empty() {
            return TireSoilFeedback::default();
        }
        let heading_len = (wheel.heading[0] * wheel.heading[0]
            + wheel.heading[2] * wheel.heading[2])
            .sqrt()
            .max(1.0e-9);
        let hx = wheel.heading[0] / heading_len;
        let hz = wheel.heading[2] / heading_len;
        let mut feedback = TireSoilFeedback::default();
        let shear_strength = self.shear_strength.max(1.0);
        for cell in &contact.cells {
            let x = wheel.hub_pos[0] + hx * cell.local_x;
            let z = wheel.hub_pos[2] + hz * cell.local_x;
            let area = cell.area.max(1.0e-6);
            let normal_pressure = (cell.normal_load / area).max(0.0);
            let before = self.sinkage_at(x, z);
            self.apply_pressure(x, z, normal_pressure);
            let after_normal = self.sinkage_at(x, z);
            let shear = (cell.tangential_force * cell.tangential_force
                + cell.lateral_force * cell.lateral_force)
                .sqrt()
                / area;
            let overshear = (shear / shear_strength - 1.0).max(0.0);
            let shear_sink = (overshear * self.shear_sinkage_gain * self.max_sinkage)
                .min(self.max_sinkage * 0.45);
            if shear_sink > 0.0 {
                let c = self.cell(x, z);
                let target =
                    self.rest_heights[c] - (after_normal + shear_sink).min(self.max_sinkage);
                if self.heights[c] > target {
                    self.heights[c] = target;
                }
            }
            let after = self.sinkage_at(x, z);
            feedback.compacted_cells += usize::from(after > before + 1.0e-9);
            feedback.normal_sinkage_m = feedback.normal_sinkage_m.max(after_normal);
            feedback.shear_sinkage_m = feedback
                .shear_sinkage_m
                .max((after - after_normal).max(0.0));
            feedback.max_sinkage_m = feedback.max_sinkage_m.max(after);
        }
        feedback.rolling_resistance_n =
            contact.normal_load.max(0.0) * feedback.max_sinkage_m / tire_radius.max(1.0e-6);
        feedback
    }

    /// Deform under every wheel of `vehicle` and update each corner's
    /// rolling resistance from the local sinkage. Call once per control
    /// tick, before [`Vehicle::step`].
    pub fn apply_wheel_loads(&mut self, vehicle: &mut Vehicle) {
        for i in 0..vehicle.corners.len() {
            let p = vehicle.corner_world_pos(i);
            let fz = vehicle.corners[i].fz;
            self.apply_load(p[0], p[2], fz);
            let sink = self.sinkage_at(p[0], p[2]);
            let radius = vehicle.corners[i].wheel_radius;
            // Bekker-style motion resistance: deeper rut, more drag.
            vehicle.corners[i].rolling_resistance = fz * sink / radius;
        }
    }

    fn sample(&self, ix: i64, iz: i64) -> f64 {
        let ix = ix.clamp(0, self.nx as i64 - 1) as usize;
        let iz = iz.clamp(0, self.nz as i64 - 1) as usize;
        self.heights[iz * self.nx + ix]
    }
}

impl Terrain for DeformableTerrain {
    fn height(&self, x: f64, z: f64) -> f64 {
        let fx = (x - self.origin[0]) / self.spacing;
        let fz = (z - self.origin[1]) / self.spacing;
        let ix = fx.floor() as i64;
        let iz = fz.floor() as i64;
        let tx = fx - ix as f64;
        let tz = fz - iz as f64;
        let h00 = self.sample(ix, iz);
        let h10 = self.sample(ix + 1, iz);
        let h01 = self.sample(ix, iz + 1);
        let h11 = self.sample(ix + 1, iz + 1);
        let h0 = h00 * (1.0 - tx) + h10 * tx;
        let h1 = h01 * (1.0 - tx) + h11 * tx;
        h0 * (1.0 - tz) + h1 * tz
    }

    fn normal(&self, x: f64, z: f64) -> [f64; 3] {
        let d = self.spacing * 0.5;
        let dhdx = (self.height(x + d, z) - self.height(x - d, z)) / (2.0 * d);
        let dhdz = (self.height(x, z + d) - self.height(x, z - d)) / (2.0 * d);
        let n = [-dhdx, 1.0, -dhdz];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        [n[0] / len, n[1] / len, n[2] / len]
    }

    fn friction(&self) -> f64 {
        self.friction
    }
}

/// Deterministic tire-rig matrix for Bekker-style deformable soil.
///
/// Every slip point starts from the same `soil_template`, so the returned
/// samples compare how rolling vs sliding contact changes the produced rut
/// under identical initial soil conditions. This is still the native
/// 2.5-D Bekker heightfield model, not DEM/MPM coupling.
pub fn tire_soil_rut_matrix(
    params: &RealTireParams,
    soil_template: &DeformableTerrain,
    material: &SurfaceMaterial,
    load_n: f64,
    speed_m_per_s: f64,
    slip_ratios: &[f64],
) -> Vec<TireSoilRutRigPoint> {
    let mut out = Vec::with_capacity(slip_ratios.len());
    let load_n = if load_n.is_finite() {
        load_n.max(0.0)
    } else {
        0.0
    };
    let speed_m_per_s = if speed_m_per_s.is_finite() {
        speed_m_per_s
    } else {
        0.0
    };
    for &slip in slip_ratios {
        let slip = if slip.is_finite() { slip } else { 0.0 };
        let mut soil = soil_template.clone();
        let rigid_reference = FlatTerrain::new(soil.height(0.0, 0.0), soil.friction());
        let hub_height_m = solve_loaded_tire_hub_height(params, &rigid_reference, 0.0, 0.0, load_n);
        let wheel = RealWheelState {
            hub_pos: [0.0, hub_height_m, 0.0],
            hub_vel: [speed_m_per_s, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: speed_m_per_s * (1.0 + slip) / params.radius.max(1.0e-6),
        };
        let contact = evaluate_brush_tire(params, &wheel, &rigid_reference, material);
        let feedback = soil.apply_tire_contact(&wheel, &contact, params.radius);
        out.push(TireSoilRutRigPoint {
            slip_ratio_command: slip,
            load_n,
            speed_m_per_s,
            hub_height_m,
            contact: contact.telemetry_sample(0.0),
            feedback,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::powertrain::{DriverInput, Powertrain};

    #[test]
    fn driven_vehicle_ruts_the_soil_and_is_slowed_by_it() {
        let mut pt = Powertrain::default();
        let go = DriverInput {
            throttle: 1.0,
            ..Default::default()
        };

        // Drive a car across a deformable soil patch.
        let mut soil = DeformableTerrain::flat([-10.0, -10.0], 0.5, 80, 80, 0.0, 6.0e5, 0.9);
        let mut car = Vehicle::car();
        for _ in 0..4000 {
            soil.apply_wheel_loads(&mut car);
            car.apply_driver(&mut pt, &go);
            car.step(&soil, 1.0e-3);
        }

        // A rut formed and every sinkage is bounded.
        let mut max_sink = 0.0_f64;
        for i in 0..soil.heights.len() {
            let s = soil.rest_heights[i] - soil.heights[i];
            assert!(
                s >= -1e-9 && s <= soil.max_sinkage + 1e-9,
                "sinkage {s} out of range"
            );
            max_sink = max_sink.max(s);
        }
        assert!(max_sink > 1.0e-3, "no rut formed (max sinkage {max_sink})");
        assert!(
            car.pos.iter().all(|c| c.is_finite()),
            "vehicle state went non-finite"
        );
        // Drawbar pull is non-zero — the car pulled itself forward.
        assert!(
            car.pos[0] > 1.0,
            "car made no forward progress on soil: x = {}",
            car.pos[0]
        );

        // The same car on rigid flat ground travels further — the soft
        // soil measurably impeded it.
        let mut flat_car = Vehicle::car();
        let flat = FlatTerrain::new(0.0, 0.9);
        for _ in 0..4000 {
            flat_car.apply_driver(&mut pt, &go);
            flat_car.step(&flat, 1.0e-3);
        }
        assert!(
            car.pos[0] < flat_car.pos[0],
            "soft soil did not impede the car: soil x={} flat x={}",
            car.pos[0],
            flat_car.pos[0]
        );
    }

    #[test]
    fn tire_contact_pressure_compacts_bekker_soil_patch() {
        let mut soil = DeformableTerrain::flat([-2.0, -2.0], 0.20, 21, 21, 0.0, 5.0e5, 0.65);
        soil.bekker_n = 0.85;
        let rigid = FlatTerrain::new(0.0, soil.friction);
        let params = RealTireParams {
            pressure_pa: 160_000.0,
            segments: 80,
            ..Default::default()
        };
        let load = 4_800.0;
        let hub_y = solve_loaded_tire_hub_height(&params, &rigid, 0.0, 0.0, load);
        let wheel = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [3.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: 3.0 / params.radius,
        };
        let contact = evaluate_brush_tire(&params, &wheel, &rigid, &SurfaceMaterial::loose_dirt());
        let feedback = soil.apply_tire_contact(&wheel, &contact, params.radius);
        assert!(feedback.compacted_cells > 0, "no tire cells compacted soil");
        assert!(
            feedback.max_sinkage_m > 1.0e-3,
            "tire contact should create a rut, feedback={feedback:?}"
        );
        assert!(
            feedback.rolling_resistance_n > 0.0,
            "rut should feed back rolling resistance estimate"
        );
    }

    #[test]
    fn sliding_tire_contact_excavates_more_than_pure_rolling() {
        let make_contact = |slip_speed: f64| {
            let rigid = FlatTerrain::new(0.0, 0.65);
            let params = RealTireParams {
                pressure_pa: 130_000.0,
                segments: 96,
                ..Default::default()
            };
            let hub_y = solve_loaded_tire_hub_height(&params, &rigid, 0.0, 0.0, 4_800.0);
            let wheel = RealWheelState {
                hub_pos: [0.0, hub_y, 0.0],
                hub_vel: [2.0, 0.0, 0.0],
                heading: [1.0, 0.0, 0.0],
                omega: (2.0 + slip_speed) / params.radius,
            };
            let contact =
                evaluate_brush_tire(&params, &wheel, &rigid, &SurfaceMaterial::loose_dirt());
            (params, wheel, contact)
        };

        let (roll_params, roll_wheel, rolling) = make_contact(0.0);
        let (slip_params, slip_wheel, sliding) = make_contact(8.0);
        let mut roll_soil = DeformableTerrain::flat([-2.0, -2.0], 0.20, 21, 21, 0.0, 2.0e6, 0.65);
        let mut slip_soil = roll_soil.clone();
        roll_soil.shear_strength = 12_000.0;
        slip_soil.shear_strength = 12_000.0;
        let roll_feedback = roll_soil.apply_tire_contact(&roll_wheel, &rolling, roll_params.radius);
        let slip_feedback = slip_soil.apply_tire_contact(&slip_wheel, &sliding, slip_params.radius);
        assert!(
            slip_feedback.max_sinkage_m > roll_feedback.max_sinkage_m + 1.0e-4,
            "sliding tire should excavate/compact deeper: rolling={roll_feedback:?} sliding={slip_feedback:?}"
        );
        assert!(
            slip_feedback.shear_sinkage_m > roll_feedback.shear_sinkage_m,
            "sliding tire should report shear sinkage"
        );
    }

    #[test]
    fn tire_soil_rut_matrix_reports_slip_dependent_rutting() {
        let params = RealTireParams::offroad_soft().with_segments(96);
        let mut soil = DeformableTerrain::flat([-2.0, -2.0], 0.20, 21, 21, 0.0, 1.6e6, 0.55);
        soil.bekker_n = 0.82;
        soil.shear_strength = 10_000.0;
        soil.shear_sinkage_gain = 0.055;
        let matrix = tire_soil_rut_matrix(
            &params,
            &soil,
            &SurfaceMaterial::soft_soil(),
            4_800.0,
            2.5,
            &[0.0, 0.35, 1.2],
        );
        assert_eq!(matrix.len(), 3);
        assert!(matrix.iter().all(TireSoilRutRigPoint::is_finite));
        let rolling = &matrix[0];
        let moderate = &matrix[1];
        let high_slip = &matrix[2];
        assert!(
            rolling.feedback.compacted_cells > 0,
            "rolling tire should compact soil: {rolling:?}"
        );
        assert!(
            high_slip.feedback.max_sinkage_m > rolling.feedback.max_sinkage_m + 1.0e-4,
            "sliding should produce a deeper rut: rolling={:?} high={:?}",
            rolling.feedback,
            high_slip.feedback
        );
        assert!(
            high_slip.feedback.shear_sinkage_m > moderate.feedback.shear_sinkage_m
                && moderate.feedback.shear_sinkage_m >= rolling.feedback.shear_sinkage_m,
            "shear sinkage should grow with slip: rolling={:?} moderate={:?} high={:?}",
            rolling.feedback,
            moderate.feedback,
            high_slip.feedback
        );
        assert!(
            high_slip.feedback.rolling_resistance_n >= rolling.feedback.rolling_resistance_n,
            "deeper rut should not reduce Bekker rolling resistance"
        );
    }

    #[test]
    fn deformable_soil_presets_are_ordered_and_finite() {
        let compact = DeformableTerrain::compacted_dirt_patch([-2.0, -2.0], 0.20, 21, 21, 0.0);
        let sandy = DeformableTerrain::sandy_loam_patch([-2.0, -2.0], 0.20, 21, 21, 0.0);
        let clay = DeformableTerrain::soft_clay_patch([-2.0, -2.0], 0.20, 21, 21, 0.0);
        for soil in [&compact, &sandy, &clay] {
            assert_eq!(soil.heights.len(), soil.nx * soil.nz);
            assert_eq!(soil.rest_heights.len(), soil.nx * soil.nz);
            assert!(soil.spacing.is_finite() && soil.spacing > 0.0);
            assert!(soil.stiffness.is_finite() && soil.stiffness > 0.0);
            assert!(soil.max_sinkage.is_finite() && soil.max_sinkage > 0.0);
            assert!(soil.friction.is_finite() && soil.friction > 0.0);
            assert!(soil.shear_strength.is_finite() && soil.shear_strength > 0.0);
            assert!(soil.shear_sinkage_gain.is_finite() && soil.shear_sinkage_gain > 0.0);
            assert!(soil.bekker_n.is_finite() && soil.bekker_n > 0.0);
        }
        assert!(compact.stiffness > sandy.stiffness);
        assert!(sandy.stiffness > clay.stiffness);
        assert!(compact.max_sinkage < sandy.max_sinkage);
        assert!(sandy.max_sinkage < clay.max_sinkage);
        assert!(compact.shear_strength > sandy.shear_strength);
        assert!(sandy.shear_strength > clay.shear_strength);
    }
}
