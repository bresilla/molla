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

use crate::terrain::Terrain;
use crate::vehicle::Vehicle;

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
            bekker_n: 1.0,
        }
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
        let c = self.cell(x, z);
        let area = self.spacing * self.spacing;
        let pressure = (fz / area).max(0.0);
        // Bekker `p = stiffness · z^n` ⇒ `z = (p / stiffness)^(1/n)`.
        // `bekker_n == 1.0` collapses to the legacy linear formula.
        let ratio = pressure / self.stiffness.max(1.0e-9);
        let n = self.bekker_n.max(1.0e-6);
        let sinkage = ratio.powf(1.0 / n).min(self.max_sinkage);
        let target = self.rest_heights[c] - sinkage;
        if self.heights[c] > target {
            self.heights[c] = target;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::powertrain::{DriverInput, Powertrain};
    use crate::terrain::FlatTerrain;

    #[test]
    fn driven_vehicle_ruts_the_soil_and_is_slowed_by_it() {
        let pt = Powertrain::default();
        let go = DriverInput {
            throttle: 1.0,
            ..Default::default()
        };

        // Drive a car across a deformable soil patch.
        let mut soil = DeformableTerrain::flat([-10.0, -10.0], 0.5, 80, 80, 0.0, 6.0e5, 0.9);
        let mut car = Vehicle::car();
        for _ in 0..4000 {
            soil.apply_wheel_loads(&mut car);
            car.apply_driver(&pt, &go);
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
            flat_car.apply_driver(&pt, &go);
            flat_car.step(&flat, 1.0e-3);
        }
        assert!(
            car.pos[0] < flat_car.pos[0],
            "soft soil did not impede the car: soil x={} flat x={}",
            car.pos[0],
            flat_car.pos[0]
        );
    }
}
