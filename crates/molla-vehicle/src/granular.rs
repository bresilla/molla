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

use crate::real_tire::{RealTireContact, RealWheelState};
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

/// Summary returned by direct tire-contact coupling into a DEM bed.
///
/// Unlike the older spherical wheel obstacle path, this uses the tire contact
/// cells from `molla_vehicle::real_tire`: normal pressure pushes grains down
/// through the local contact normal, while longitudinal/lateral brush shear
/// transfers impulse into the local grain patch. It is still a first-pass DEM
/// coupling, but it is tire-contact-cell based instead of just a kinematic
/// obstacle sphere.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireDemFeedback {
    /// Number of tire contact cells that found grains underneath them.
    pub active_contact_cells: usize,
    /// Number of individual grain impulses applied across all cells.
    pub affected_grains: usize,
    /// Total normal load transferred to DEM grains, N.
    pub transferred_normal_load_n: f64,
    /// Total tangential/shear force magnitude transferred to grains, N.
    pub transferred_shear_load_n: f64,
    /// Total impulse magnitude applied to grains, N·s.
    pub impulse_n_s: f64,
    /// Maximum absolute vertical grain displacement caused by the immediate
    /// contact push, m.
    pub max_grain_sinkage_m: f64,
    /// Deterministic drawbar/rolling-loss proxy from displaced grains, N.
    pub rolling_resistance_n: f64,
}

impl TireDemFeedback {
    pub fn is_finite(self) -> bool {
        self.transferred_normal_load_n.is_finite()
            && self.transferred_shear_load_n.is_finite()
            && self.impulse_n_s.is_finite()
            && self.max_grain_sinkage_m.is_finite()
            && self.rolling_resistance_n.is_finite()
    }
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

    /// Grain velocities. Exposed for tire/soil regression tests and
    /// diagnostics; callers should not mutate the returned slice.
    pub fn grain_velocities(&self) -> &[[f64; 3]] {
        &self.vel
    }

    /// Replace grain positions and velocities with an externally computed DEM
    /// state, preserving material parameters. This is used by accelerated DEM
    /// owners that step the same granular bed through a sparse/GPU broadphase
    /// and then hand the state back to the terrain/viewer surface.
    pub fn replace_grain_state(
        &mut self,
        positions: &[[f64; 3]],
        velocities: &[[f64; 3]],
    ) -> Result<(), &'static str> {
        if positions.len() != self.pos.len() || velocities.len() != self.vel.len() {
            return Err("granular bed replacement state length mismatch");
        }
        if positions.iter().flatten().any(|v| !v.is_finite())
            || velocities.iter().flatten().any(|v| !v.is_finite())
        {
            return Err("granular bed replacement state contains non-finite values");
        }
        self.pos.copy_from_slice(positions);
        self.vel.copy_from_slice(velocities);
        Ok(())
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

    /// Apply an evaluated real-tire contact patch directly into the DEM bed.
    ///
    /// This complements [`Self::wheel_obstacles`]: the obstacle path gives
    /// geometric two-way collision with the wheel envelope, while this method
    /// transfers the actual tire contact-cell normal/shear loads into nearby
    /// grains. That makes slip, pressure distribution, lateral force, and
    /// per-cell normals visible to the granular soil instead of treating the
    /// tire as only one smooth sphere.
    pub fn apply_tire_contact(
        &mut self,
        wheel: &RealWheelState,
        contact: &RealTireContact,
        tire_radius: f64,
        dt: f64,
    ) -> TireDemFeedback {
        let dt = finite_nonnegative(dt).min(0.05);
        if dt <= 0.0 || !contact.in_contact || contact.cells.is_empty() || self.pos.is_empty() {
            return TireDemFeedback::default();
        }

        let heading = horizontal_unit(wheel.heading);
        let side = [-heading[2], 0.0, heading[0]];
        let mut feedback = TireDemFeedback::default();
        let mut max_vertical_push: f64 = 0.0;

        for cell in &contact.cells {
            let area = cell.area.max(1.0e-6);
            let influence_radius = (area / std::f64::consts::PI).sqrt().max(self.radius * 1.25);
            let cx = wheel.hub_pos[0] + heading[0] * cell.local_x + side[0] * cell.local_z;
            let cz = wheel.hub_pos[2] + heading[2] * cell.local_x + side[2] * cell.local_z;
            let surface_y = self.height(cx, cz);
            let normal = unit_or(cell.normal, [0.0, 1.0, 0.0]);
            let force_on_grains = add(
                add(
                    scale(normal, -cell.normal_load.max(0.0)),
                    scale(heading, -cell.tangential_force),
                ),
                scale(side, -cell.lateral_force),
            );
            let force_mag = norm(force_on_grains);
            if force_mag <= 1.0e-9 {
                continue;
            }

            let mut affected = Vec::new();
            let mut weight_sum = 0.0;
            for (i, p) in self.pos.iter().enumerate() {
                let dx = p[0] - cx;
                let dz = p[2] - cz;
                let horizontal = (dx * dx + dz * dz).sqrt();
                if horizontal > influence_radius + self.radius {
                    continue;
                }
                // Only grains in the active near-surface plug receive the tire
                // contact load. Deep grains will receive force through normal
                // DEM grain-grain contacts on subsequent steps.
                let below_surface = (surface_y + self.radius * 1.5) - p[1];
                if below_surface < -self.radius || below_surface > self.radius * 5.0 {
                    continue;
                }
                let radial_w =
                    (1.0 - horizontal / (influence_radius + self.radius)).clamp(0.0, 1.0);
                let depth_w = (1.0 - below_surface.max(0.0) / (self.radius * 5.0)).clamp(0.15, 1.0);
                let w = radial_w * radial_w * depth_w;
                if w > 1.0e-9 {
                    affected.push((i, w));
                    weight_sum += w;
                }
            }

            if affected.is_empty() || weight_sum <= 1.0e-12 {
                continue;
            }

            feedback.active_contact_cells += 1;
            feedback.transferred_normal_load_n += cell.normal_load.max(0.0);
            feedback.transferred_shear_load_n += (cell.tangential_force * cell.tangential_force
                + cell.lateral_force * cell.lateral_force)
                .sqrt();

            for (i, w) in affected {
                let share = w / weight_sum;
                let impulse = scale(force_on_grains, share * dt);
                for (k, component) in impulse.iter().enumerate() {
                    self.vel[i][k] += component / self.mass.max(1.0e-9);
                }

                // A small position-level normal push prevents high-frequency
                // tire loads from becoming purely invisible velocity spikes in
                // tiny demo timesteps. It is capped by grain size and tied to
                // the actual applied impulse, not a viewer-only rut flag.
                let normal_impulse = cell.normal_load.max(0.0) * share * dt;
                let vertical_push =
                    (normal_impulse / self.mass.max(1.0e-9) * dt * 0.25).min(self.radius * 0.35);
                self.pos[i][1] = (self.pos[i][1] - vertical_push).max(self.floor_y + self.radius);
                max_vertical_push = max_vertical_push.max(vertical_push);

                feedback.affected_grains += 1;
                feedback.impulse_n_s += norm(impulse);
            }
        }

        feedback.max_grain_sinkage_m = max_vertical_push;
        feedback.rolling_resistance_n = feedback.transferred_normal_load_n
            * feedback.max_grain_sinkage_m
            / tire_radius.max(1.0e-6);
        feedback
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
fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}
fn unit_or(v: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n > 1.0e-9 && n.is_finite() {
        scale(v, 1.0 / n)
    } else {
        fallback
    }
}
fn horizontal_unit(v: [f64; 3]) -> [f64; 3] {
    let h = [v[0], 0.0, v[2]];
    unit_or(h, [1.0, 0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::powertrain::{DriverInput, Powertrain};
    use crate::real_tire::{
        RealTireParams, RealWheelState, SurfaceMaterial, evaluate_brush_tire,
        solve_loaded_tire_hub_height,
    };

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
    fn granular_bed_accepts_external_dem_owner_state() {
        let mut bed = GranularBed::lattice([0.0, 0.05, 0.0], [0.1, 0.05, 0.0], 0.05, 1.0, 0.7);
        let positions = bed
            .grains()
            .iter()
            .map(|p| [p[0], p[1] - 0.01, p[2]])
            .collect::<Vec<_>>();
        let velocities = bed
            .grain_velocities()
            .iter()
            .map(|v| [v[0] + 0.2, v[1], v[2]])
            .collect::<Vec<_>>();
        bed.replace_grain_state(&positions, &velocities)
            .expect("valid owner state");
        assert_eq!(bed.grains(), positions.as_slice());
        assert_eq!(bed.grain_velocities(), velocities.as_slice());

        let mut bad = velocities.clone();
        bad.pop();
        assert!(
            bed.replace_grain_state(&positions, &bad).is_err(),
            "external DEM owners must not be allowed to desynchronise position/velocity arrays"
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

        let mut pt = Powertrain::default();
        let mut car = Vehicle::car();
        let go = DriverInput {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..1500 {
            let obstacles = bed.wheel_obstacles(&car);
            bed.step(&obstacles, 5.0e-4);
            car.apply_driver(&mut pt, &go);
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

    #[test]
    fn tire_contact_patch_pushes_dem_grains_and_reports_feedback() {
        let r = 0.035;
        let mut bed = GranularBed::lattice([-0.35, 0.02, -0.18], [0.35, 0.16, 0.18], r, 0.55, 0.7);
        bed.jitter(0.006);
        for _ in 0..300 {
            bed.step(&[], 5.0e-4);
        }
        let y_before = bed
            .grains()
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max);

        let params = RealTireParams::offroad_soft().with_segments(64);
        let material = SurfaceMaterial::soft_soil();
        let load_n = 3_800.0;
        let hub_y = solve_loaded_tire_hub_height(&params, &bed, 0.0, 0.0, load_n);
        let wheel = RealWheelState {
            hub_pos: [0.0, hub_y, 0.0],
            hub_vel: [2.5, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: (2.5 + 1.2) / params.radius,
        };
        let contact = evaluate_brush_tire(&params, &wheel, &bed, &material);
        let feedback = bed.apply_tire_contact(&wheel, &contact, params.radius, 2.0e-3);

        assert!(
            feedback.is_finite(),
            "DEM tire feedback must be finite: {feedback:?}"
        );
        assert!(
            feedback.active_contact_cells > 0 && feedback.affected_grains > 0,
            "tire patch should find and push grains: {feedback:?}"
        );
        assert!(
            feedback.transferred_normal_load_n > load_n * 0.20,
            "DEM patch should transfer a meaningful normal load: {feedback:?}"
        );
        assert!(
            feedback.transferred_shear_load_n > 10.0 && feedback.impulse_n_s > 0.0,
            "driven tire should transfer shear impulse into grains: {feedback:?}"
        );
        assert!(
            feedback.max_grain_sinkage_m > 0.0 && feedback.rolling_resistance_n > 0.0,
            "normal patch impulse should create finite DEM sinkage/loss: {feedback:?}"
        );

        let y_after = bed
            .grains()
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            y_after <= y_before + r,
            "DEM contact push should remain bounded: before={y_before} after={y_after}"
        );
    }

    #[test]
    fn slipping_tire_transfers_more_dem_shear_than_free_rolling_tire() {
        let r = 0.04;
        let make_bed = || {
            let mut bed =
                GranularBed::lattice([-0.32, 0.02, -0.16], [0.32, 0.18, 0.16], r, 0.65, 0.75);
            bed.jitter(0.004);
            for _ in 0..250 {
                bed.step(&[], 5.0e-4);
            }
            bed
        };
        let params = RealTireParams::offroad_soft().with_segments(64);
        let material = SurfaceMaterial::soft_soil();
        let load_n = 3_500.0;

        let mut roll_bed = make_bed();
        let roll_y = solve_loaded_tire_hub_height(&params, &roll_bed, 0.0, 0.0, load_n);
        let roll_wheel = RealWheelState {
            hub_pos: [0.0, roll_y, 0.0],
            hub_vel: [2.0, 0.0, 0.0],
            heading: [1.0, 0.0, 0.0],
            omega: 2.0 / params.radius,
        };
        let rolling = evaluate_brush_tire(&params, &roll_wheel, &roll_bed, &material);
        let roll_feedback =
            roll_bed.apply_tire_contact(&roll_wheel, &rolling, params.radius, 2.0e-3);

        let mut slip_bed = make_bed();
        let slip_y = solve_loaded_tire_hub_height(&params, &slip_bed, 0.0, 0.0, load_n);
        let slip_wheel = RealWheelState {
            omega: (2.0 + 2.2) / params.radius,
            hub_pos: [0.0, slip_y, 0.0],
            ..roll_wheel
        };
        let slipping = evaluate_brush_tire(&params, &slip_wheel, &slip_bed, &material);
        let slip_feedback =
            slip_bed.apply_tire_contact(&slip_wheel, &slipping, params.radius, 2.0e-3);

        assert!(
            slip_feedback.transferred_shear_load_n > roll_feedback.transferred_shear_load_n * 1.5,
            "slipping tire should inject more DEM shear: roll={roll_feedback:?} slip={slip_feedback:?}"
        );
        let horizontal_speed = |v: [f64; 3]| (v[0] * v[0] + v[2] * v[2]).sqrt();
        let roll_speed_sum: f64 = roll_bed
            .grain_velocities()
            .iter()
            .map(|v| horizontal_speed(*v))
            .sum();
        let slip_speed_sum: f64 = slip_bed
            .grain_velocities()
            .iter()
            .map(|v| horizontal_speed(*v))
            .sum();
        assert!(
            slip_speed_sum > roll_speed_sum * 1.10,
            "slipping tire should leave more horizontal grain motion: roll={roll_speed_sum} slip={slip_speed_sum}"
        );
    }
}
