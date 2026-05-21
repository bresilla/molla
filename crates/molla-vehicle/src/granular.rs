//! V5-deep — DEM (discrete element) granular soil.
//!
//! A genuine particle soil: discrete grains with pairwise spring-dashpot
//! normal contact and Coulomb friction. Unlike continuum granular SPH
//! (see `PLAN_ENGINE.md` F4), DEM is robust — the contact is
//! unambiguous, so a column of grains reliably collapses into a heap
//! with a real angle of repose.
//!
//! [`GranularBed`] implements [`Terrain`], so an existing
//! [`Vehicle`](crate::Vehicle) drives on it unchanged via the
//! grain-surface height; in the same loop the wheels are passed as
//! [`GrainObstacle`]s that plough the grains — genuine two-way
//! vehicle ↔ particle-soil coupling.

use crate::terrain::Terrain;
use crate::vehicle::Vehicle;
use std::collections::HashMap;

/// A rigid kinematic sphere (a vehicle wheel) the grains collide with.
#[derive(Clone, Copy, Debug)]
pub struct GrainObstacle {
    /// World centre.
    pub center: [f64; 3],
    /// Sphere radius.
    pub radius: f64,
    /// Obstacle velocity (so the grains feel it ploughing).
    pub vel: [f64; 3],
}

/// A DEM granular bed: a cloud of equal grains on a flat floor.
pub struct GranularBed {
    pos: Vec<[f64; 3]>,
    vel: Vec<[f64; 3]>,
    /// Grain radius (m).
    pub radius: f64,
    /// Grain mass (kg).
    pub mass: f64,
    /// Normal contact stiffness (N/m).
    pub k_n: f64,
    /// Normal contact damping (N·s/m).
    pub damping: f64,
    /// Coulomb friction coefficient.
    pub friction: f64,
    /// Floor height (world y).
    pub floor_y: f64,
    /// Gravity (m/s²).
    pub gravity: [f64; 3],
}

impl GranularBed {
    /// A bed of grains placed on a lattice over `[min, max]`, spacing
    /// `2·radius`, at rest. Let it settle with a few [`Self::step`]s.
    pub fn lattice(min: [f64; 3], max: [f64; 3], radius: f64, mass: f64, friction: f64) -> Self {
        let s = 2.0 * radius;
        let count = |a: f64, b: f64| (((b - a) / s).floor() as i32 + 1).max(1);
        let mut pos = Vec::new();
        for ix in 0..count(min[0], max[0]) {
            for iy in 0..count(min[1], max[1]) {
                for iz in 0..count(min[2], max[2]) {
                    pos.push([
                        min[0] + ix as f64 * s,
                        min[1] + iy as f64 * s,
                        min[2] + iz as f64 * s,
                    ]);
                }
            }
        }
        let n = pos.len();
        Self {
            pos,
            vel: vec![[0.0; 3]; n],
            radius,
            mass,
            k_n: 1.0e4,
            damping: 80.0,
            friction,
            floor_y: 0.0,
            gravity: [0.0, -9.81, 0.0],
        }
    }

    /// Number of grains.
    pub fn grain_count(&self) -> usize {
        self.pos.len()
    }

    /// Perturb grain positions by up to `amount` (deterministic
    /// pseudo-random) so the pack is irregular — a perfect lattice is
    /// artificially stable and would never collapse.
    pub fn jitter(&mut self, amount: f64) {
        for (i, p) in self.pos.iter_mut().enumerate() {
            for k in 0..3 {
                let mut h = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(k as u64 + 1);
                h ^= h >> 30;
                h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                h ^= h >> 27;
                let r = (h % 20001) as f64 / 10000.0 - 1.0; // [-1, 1]
                p[k] += amount * r;
            }
        }
    }

    /// Grain positions.
    pub fn grains(&self) -> &[[f64; 3]] {
        &self.pos
    }

    /// Spring-dashpot + Coulomb contact force on body `a` from body `b`
    /// (`b → a` is repulsive). `None` when they are not overlapping.
    fn contact(
        &self,
        pa: [f64; 3],
        va: [f64; 3],
        pb: [f64; 3],
        vb: [f64; 3],
        sum_radii: f64,
    ) -> Option<[f64; 3]> {
        let d = sub(pa, pb);
        let dist = norm(d);
        let overlap = sum_radii - dist;
        if overlap <= 0.0 || dist < 1e-9 {
            return None;
        }
        let n = scale(d, 1.0 / dist);
        let v_rel = sub(va, vb);
        let v_n = dot(v_rel, n);
        let fn_mag = (self.k_n * overlap - self.damping * v_n).max(0.0);
        // Tangential Coulomb friction, regularised near zero slip.
        let v_t = sub(v_rel, scale(n, v_n));
        let vt = norm(v_t);
        let mut f = scale(n, fn_mag);
        if vt > 1e-9 {
            let ft = self.friction * fn_mag * (vt / (vt + 0.05));
            f = sub(f, scale(v_t, ft / vt));
        }
        Some(f)
    }

    /// Advance the bed by `dt`, with `obstacles` (vehicle wheels)
    /// ploughing the grains.
    pub fn step(&mut self, obstacles: &[GrainObstacle], dt: f64) {
        let n = self.pos.len();
        if n == 0 {
            return;
        }
        let cell = 2.0 * self.radius;
        let key = |p: [f64; 3]| {
            (
                (p[0] / cell).floor() as i32,
                (p[1] / cell).floor() as i32,
                (p[2] / cell).floor() as i32,
            )
        };
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (i, &p) in self.pos.iter().enumerate() {
            grid.entry(key(p)).or_default().push(i);
        }

        let mut force = vec![scale(self.gravity, self.mass); n];
        let two_r = 2.0 * self.radius;
        for i in 0..n {
            let (kx, ky, kz) = key(self.pos[i]);
            // Grain-grain contacts (each pair handled once, j > i).
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(cellv) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                            for &j in cellv {
                                if j <= i {
                                    continue;
                                }
                                if let Some(f) = self.contact(
                                    self.pos[i],
                                    self.vel[i],
                                    self.pos[j],
                                    self.vel[j],
                                    two_r,
                                ) {
                                    force[i] = add(force[i], f);
                                    force[j] = sub(force[j], f);
                                }
                            }
                        }
                    }
                }
            }
            // Grain-floor contact (floor as an infinite plane at floor_y).
            let pen = (self.floor_y + self.radius) - self.pos[i][1];
            if pen > 0.0 {
                let v_n = self.vel[i][1];
                let fn_mag = (self.k_n * pen - self.damping * v_n).max(0.0);
                force[i][1] += fn_mag;
                let v_t = [self.vel[i][0], 0.0, self.vel[i][2]];
                let vt = norm(v_t);
                if vt > 1e-9 {
                    let ft = self.friction * fn_mag * (vt / (vt + 0.05));
                    force[i][0] -= v_t[0] * ft / vt;
                    force[i][2] -= v_t[2] * ft / vt;
                }
            }
            // Grain-obstacle (wheel) contacts.
            for o in obstacles {
                if let Some(f) = self.contact(
                    self.pos[i],
                    self.vel[i],
                    o.center,
                    o.vel,
                    self.radius + o.radius,
                ) {
                    force[i] = add(force[i], f);
                }
            }
        }

        for i in 0..n {
            for k in 0..3 {
                self.vel[i][k] += force[i][k] / self.mass * dt;
                self.pos[i][k] += self.vel[i][k] * dt;
            }
        }
    }

    /// Wheel [`GrainObstacle`]s for `vehicle` — one sphere per wheel,
    /// sitting at the current grain-surface height.
    pub fn wheel_obstacles(&self, vehicle: &Vehicle) -> Vec<GrainObstacle> {
        (0..vehicle.corners.len())
            .map(|i| {
                let p = vehicle.corner_world_pos(i);
                let r = vehicle.corners[i].wheel_radius;
                GrainObstacle {
                    center: [p[0], self.height(p[0], p[2]) + r, p[2]],
                    radius: r,
                    vel: vehicle.vel,
                }
            })
            .collect()
    }

    /// Highest grain centre near `(x, z)` (within a grain radius).
    fn top_grain(&self, x: f64, z: f64) -> Option<f64> {
        let mut top: Option<f64> = None;
        for p in &self.pos {
            if (p[0] - x).abs() < self.radius && (p[2] - z).abs() < self.radius {
                top = Some(top.map_or(p[1], |t: f64| t.max(p[1])));
            }
        }
        top
    }
}

impl Terrain for GranularBed {
    fn height(&self, x: f64, z: f64) -> f64 {
        // Surface = the top of the highest local grain.
        self.top_grain(x, z)
            .map_or(self.floor_y, |t| t + self.radius)
    }

    fn normal(&self, _x: f64, _z: f64) -> [f64; 3] {
        [0.0, 1.0, 0.0]
    }

    fn friction(&self) -> f64 {
        self.friction
    }
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::powertrain::{DriverInput, Powertrain};

    #[test]
    fn dem_column_collapses_into_a_pile() {
        // A frictional DEM column collapses into a heap that keeps real
        // height; a frictionless one spreads markedly flatter. This is
        // the genuine particle-soil angle of repose.
        let r = 0.05;
        let mass = 1.0;
        let collapse = |friction: f64| {
            let mut bed =
                GranularBed::lattice([-0.20, 0.05, -0.20], [0.20, 0.65, 0.20], r, mass, friction);
            // Break the artificial lattice symmetry so it can collapse.
            bed.jitter(0.012);
            for _ in 0..4000 {
                bed.step(&[], 5.0e-4);
            }
            bed.grains()
                .iter()
                .map(|g| g[1])
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let frictional = collapse(0.8);
        let frictionless = collapse(0.0);
        assert!(frictional.is_finite() && frictionless.is_finite());
        assert!(
            frictional > frictionless + 0.05,
            "frictional DEM column did not hold a taller pile: \
             frictional={frictional} frictionless={frictionless}"
        );
    }

    #[test]
    fn vehicle_ruts_the_dem_soil() {
        // A vehicle driven over the DEM bed ploughs the grains: the bed
        // surface along the wheel path is pushed down (a rut), and the
        // grains stay finite and bounded.
        let r = 0.06;
        let mut bed = GranularBed::lattice([-0.6, 0.0, -0.6], [1.4, 0.30, 0.6], r, 1.2, 0.7);
        // Settle the freshly-laid bed.
        for _ in 0..400 {
            bed.step(&[], 5.0e-4);
        }
        let surface_before = bed.height(0.4, 0.0);

        let pt = Powertrain::default();
        let mut car = Vehicle::car();
        let go = DriverInput {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..1500 {
            let obstacles = bed.wheel_obstacles(&car);
            bed.step(&obstacles, 5.0e-4);
            car.apply_driver(&pt, &go);
            car.step(&bed, 5.0e-4);
        }

        for g in bed.grains() {
            for &c in g {
                assert!(c.is_finite(), "DEM grain went non-finite");
            }
        }
        assert!(
            car.pos.iter().all(|c| c.is_finite()),
            "vehicle state went non-finite"
        );
        // The car drove forward on the soft soil (drawbar pull).
        assert!(
            car.pos[0] > 0.2,
            "car made no progress on the soil: x={}",
            car.pos[0]
        );
        // The wheels ploughed a rut — the surface mid-path dropped.
        let surface_after = bed.height(0.4, 0.0);
        assert!(
            surface_after < surface_before - 1e-3,
            "no rut formed: surface {surface_before} → {surface_after}"
        );
    }
}
