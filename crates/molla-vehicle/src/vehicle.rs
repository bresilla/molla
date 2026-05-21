//! V3/V4 — the assembled wheeled vehicle.
//!
//! A [`Vehicle`] is a rigid chassis (heave + longitudinal + lateral +
//! yaw) carried on four suspension [`Corner`]s. Each step every corner
//! queries the terrain, the suspension spring sets the tyre normal
//! load, and a [`TireModel`] turns the wheel kinematics into the ground
//! force — assembly of parts molla already provides plus the V1/V2
//! tyre force elements. The wheels follow the terrain quasi-statically
//! (massless unsprung mass) so the model stays a compact 4-DoF chassis.
//!
//! `drive_torque`, `brake_torque` and `steer` on each corner are the
//! inputs the V4 powertrain/steering layer writes.

use crate::suspension::{Suspension, SuspensionKind};
use crate::terrain::Terrain;
use crate::tire::{TireModel, TmSimpleTire};

/// One suspension corner: a wheel on a spring-damper at a chassis mount.
pub struct Corner {
    /// Chassis-frame mount point, relative to the COM.
    pub attach: [f64; 3],
    /// The suspension spring-damper.
    pub suspension: Suspension,
    /// Wheel radius (m).
    pub wheel_radius: f64,
    /// Wheel spin rate about its axle (rad/s).
    pub spin: f64,
    /// Wheel + driveline rotational inertia (kg·m²).
    pub spin_inertia: f64,
    /// Steer angle (rad) — set by the steering layer.
    pub steer: f64,
    /// Driveline torque (N·m) — set by the powertrain layer.
    pub drive_torque: f64,
    /// Brake torque magnitude (N·m) — set by the brake layer.
    pub brake_torque: f64,
    /// Last normal load (N) — telemetry.
    pub fz: f64,
    /// Last longitudinal slip ratio — telemetry.
    pub slip_ratio: f64,
    /// Terramechanics rolling resistance (N) — a backward force set by
    /// a deformable terrain from the local sinkage (V5). `0` on rigid
    /// ground.
    pub rolling_resistance: f64,
}

/// A four-wheeled vehicle.
pub struct Vehicle {
    /// Chassis centre-of-mass world position.
    pub pos: [f64; 3],
    /// Chassis COM linear velocity.
    pub vel: [f64; 3],
    /// Chassis yaw angle (rad).
    pub yaw: f64,
    /// Chassis yaw rate (rad/s).
    pub yaw_rate: f64,
    /// Chassis mass (kg).
    pub mass: f64,
    /// Chassis yaw inertia (kg·m²).
    pub yaw_inertia: f64,
    /// Suspension corners. Indices 0,1 are front (steerable), 2,3 rear.
    pub corners: Vec<Corner>,
    /// Shared tyre force model.
    pub tire: Box<dyn TireModel>,
}

impl Vehicle {
    /// A standard passenger car, parked at its equilibrium ride height
    /// over flat ground at `y = 0`.
    pub fn car() -> Self {
        Self::car_with(Box::new(TmSimpleTire::default()))
    }

    /// A standard car using the given tyre model.
    pub fn car_with(tire: Box<dyn TireModel>) -> Self {
        let stiffness = 35_000.0;
        let mk_corner = |x: f64, z: f64| Corner {
            attach: [x, -0.2, z],
            suspension: Suspension::new(SuspensionKind::DoubleWishbone, 0.35, stiffness, 4_000.0),
            wheel_radius: 0.32,
            spin: 0.0,
            spin_inertia: 1.2,
            steer: 0.0,
            drive_torque: 0.0,
            brake_torque: 0.0,
            fz: 0.0,
            slip_ratio: 0.0,
            rolling_resistance: 0.0,
        };
        Self {
            // Equilibrium height: 4·k·(M·g/4k) balances gravity.
            pos: [0.0, 0.786, 0.0],
            vel: [0.0; 3],
            yaw: 0.0,
            yaw_rate: 0.0,
            mass: 1200.0,
            yaw_inertia: 1800.0,
            corners: vec![
                mk_corner(1.30, 0.75),
                mk_corner(1.30, -0.75),
                mk_corner(-1.30, 0.75),
                mk_corner(-1.30, -0.75),
            ],
            tire,
        }
    }

    /// Chassis COM height above `y = 0`.
    pub fn chassis_height(&self) -> f64 {
        self.pos[1]
    }

    /// Speed of the chassis along its heading (m/s).
    pub fn forward_speed(&self) -> f64 {
        self.vel[0] * self.yaw.cos() + self.vel[2] * self.yaw.sin()
    }

    /// World position of suspension corner `i`'s chassis mount.
    pub fn corner_world_pos(&self, i: usize) -> [f64; 3] {
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        let a = self.corners[i].attach;
        [
            self.pos[0] + a[0] * cy - a[2] * sy,
            self.pos[1] + a[1],
            self.pos[2] + a[0] * sy + a[2] * cy,
        ]
    }

    /// Advance the vehicle by `dt` over `terrain`.
    pub fn step(&mut self, terrain: &dyn Terrain, dt: f64) {
        let g = 9.81;
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        let rot = |v: [f64; 3]| [v[0] * cy - v[2] * sy, v[1], v[0] * sy + v[2] * cy];

        let mut force = [0.0_f64; 3];
        let mut yaw_moment = 0.0_f64;

        for c in &mut self.corners {
            let off = rot(c.attach);
            let attach_w = [
                self.pos[0] + off[0],
                self.pos[1] + off[1],
                self.pos[2] + off[2],
            ];
            // Quasi-static wheel: the hub sits one radius above ground.
            let gh = terrain.height(attach_w[0], attach_w[2]);
            let hub_y = gh + c.wheel_radius;
            // Velocity of the mount point: chassis vel + yaw_rate × off.
            let corner_vel = [
                self.vel[0] - self.yaw_rate * off[2],
                self.vel[1],
                self.vel[2] + self.yaw_rate * off[0],
            ];
            // Suspension spring sets the tyre normal load.
            let compression = c.suspension.rest_length - (attach_w[1] - hub_y);
            let comp_rate = -corner_vel[1];
            let f_sus = c.suspension.force(compression, comp_rate);
            c.suspension.travel = compression.clamp(0.0, c.suspension.max_compression);
            c.suspension.travel_rate = comp_rate;
            c.fz = f_sus;

            // Tyre contact frame on flat-ish ground: longitudinal along
            // the steered heading, lateral perpendicular.
            let head = self.yaw + c.steer;
            let long = [head.cos(), 0.0, head.sin()];
            let lat = [-head.sin(), 0.0, head.cos()];
            let v_long = corner_vel[0] * long[0] + corner_vel[2] * long[2];
            let v_lat = corner_vel[0] * lat[0] + corner_vel[2] * lat[2];
            let denom = v_long.abs().max(0.5);
            let kappa = (c.spin * c.wheel_radius - v_long) / denom;
            let alpha = (v_lat / denom).atan();
            c.slip_ratio = kappa;
            let (fx, fy, _mz) = self.tire.evaluate(kappa, alpha, f_sus, terrain.friction());

            // Accumulate chassis wrench.
            force[1] += f_sus;
            let fxw = fx * long[0] + fy * lat[0];
            let fzw = fx * long[2] + fy * lat[2];
            force[0] += fxw;
            force[2] += fzw;
            // V5 — terramechanics rolling resistance: a bounded force
            // opposing forward motion, set from the terrain sinkage.
            let res = c.rolling_resistance * v_long.tanh();
            force[0] -= res * long[0];
            force[2] -= res * long[2];
            yaw_moment += off[0] * fzw - off[2] * fxw;

            // Wheel spin: drive torque, brake (smooth, opposes spin and
            // bounded so it cannot reverse it), reaction from the tyre.
            let brake = c.brake_torque * c.spin.tanh();
            c.spin += (c.drive_torque - brake - fx * c.wheel_radius) / c.spin_inertia * dt;
        }

        force[1] -= self.mass * g;
        for k in 0..3 {
            self.vel[k] += force[k] / self.mass * dt;
            self.pos[k] += self.vel[k] * dt;
        }
        self.yaw_rate += yaw_moment / self.yaw_inertia * dt;
        self.yaw += self.yaw_rate * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::FlatTerrain;

    #[test]
    fn vehicle_settles_to_a_stable_ride_height() {
        let ground = FlatTerrain::new(0.0, 0.9);
        let mut car = Vehicle::car();
        // Drop it in from above its equilibrium height.
        car.pos[1] = 1.15;
        for _ in 0..6000 {
            car.step(&ground, 1.0e-3);
        }
        // Came to rest vertically.
        assert!(
            car.vel[1].abs() < 0.02,
            "chassis still bobbing: vy = {}",
            car.vel[1]
        );
        // Settled to a sane ride height (between the wheels and the
        // drop height), and every suspension is compressed and bounded.
        assert!(
            car.pos[1] > 0.5 && car.pos[1] < 1.0,
            "implausible ride height: {}",
            car.pos[1]
        );
        for (i, c) in car.corners.iter().enumerate() {
            assert!(
                c.suspension.travel > 0.0 && c.suspension.travel <= c.suspension.max_compression,
                "corner {i} suspension travel out of range: {}",
                c.suspension.travel
            );
            assert!(c.fz > 0.0, "corner {i} carries no load");
        }
        // Total suspension load supports the car's weight.
        let total_fz: f64 = car.corners.iter().map(|c| c.fz).sum();
        assert!(
            (total_fz - car.mass * 9.81).abs() < car.mass * 9.81 * 0.05,
            "suspension load {total_fz} does not balance weight {}",
            car.mass * 9.81
        );
    }
}
