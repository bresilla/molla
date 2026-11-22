use crate::orchard_model::{ROW_HALF_LENGTH, ROW_X, TREE_COUNT, TREE_SPACING, TREES_PER_ROW};
use molla_math::{Transform, Vec3};
use molla_solvers::airflow::{FlightReference, FlightSegment, MotionLimits};

pub const DOSE_SECONDS: f64 = 2.0;
pub const LEG_SECONDS: f64 = 4.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Travel,
    Settle,
    Spray,
    Complete,
}

pub struct Mission {
    pub phase: Phase,
    pub station: usize,
    pub dose: [f64; TREE_COUNT],
    pub max_spray_speed: f64,
    pub max_spray_error: f64,
    pub pump: bool,
    height: f64,
    elapsed: f64,
    stable: f64,
    leg: FlightSegment,
    returning: bool,
}

pub fn tree_index(station: usize) -> usize {
    if station < TREES_PER_ROW {
        station
    } else {
        TREE_COUNT - 1 - (station - TREES_PER_ROW)
    }
}

pub fn station_position(station: usize, height: f64) -> Vec3 {
    let tree = tree_index(station);
    Vec3::new(
        ROW_X[tree / TREES_PER_ROW],
        height,
        (tree % TREES_PER_ROW) as f64 * TREE_SPACING - ROW_HALF_LENGTH,
    )
}

fn segment(a: Vec3, b: Vec3) -> FlightSegment {
    let leg = FlightSegment::new(
        a,
        b,
        MotionLimits {
            speed: 1.25,
            acceleration: 1.4,
            jerk: 3.0,
        },
    )
    .unwrap();
    leg.with_duration(LEG_SECONDS.max(leg.duration())).unwrap()
}

impl Mission {
    pub fn new(start: Vec3, height: f64) -> Self {
        Self {
            phase: Phase::Travel,
            station: 0,
            dose: [0.0; TREE_COUNT],
            height,
            max_spray_speed: 0.0,
            max_spray_error: 0.0,
            pump: false,
            elapsed: 0.0,
            stable: 0.0,
            leg: segment(start, station_position(0, height)),
            returning: false,
        }
    }

    pub fn target(&self) -> Vec3 {
        station_position(if self.returning { 0 } else { self.station }, self.height)
    }

    pub fn update(
        &mut self,
        dt: f64,
        pose: Transform,
        velocity: Vec3,
        enabled: bool,
    ) -> FlightReference {
        self.pump = false;
        if !enabled {
            return FlightReference::hold(pose.position);
        }
        self.elapsed += dt;
        if self.phase == Phase::Travel {
            let r = self.leg.sample(self.elapsed).unwrap();
            if self.elapsed >= self.leg.duration() {
                self.phase = Phase::Settle;
                self.stable = 0.0;
            }
            return r;
        }
        let error = pose.position.distance(self.target());
        let speed = velocity.length();
        let settled = error < 0.10 && speed < 0.10 && (pose.rotation * Vec3::Z).y > 0.99;
        self.stable = if settled { self.stable + dt } else { 0.0 };
        if self.phase == Phase::Settle && self.stable >= 0.5 {
            self.phase = if self.returning {
                Phase::Complete
            } else {
                Phase::Spray
            };
        }
        let reference = FlightReference::hold(self.target());
        self.pump = self.phase == Phase::Spray && settled;
        reference
    }

    pub fn commit_spray(&mut self, dt: f64, pose: Transform, velocity: Vec3) {
        let error = pose.position.distance(self.target());
        let speed = velocity.length();
        self.pump &= error < 0.10 && speed < 0.10 && (pose.rotation * Vec3::Z).y > 0.99;
        if self.pump {
            self.max_spray_error = self.max_spray_error.max(error);
            self.max_spray_speed = self.max_spray_speed.max(speed);
            self.dose[tree_index(self.station)] += dt;
            if self.dose[tree_index(self.station)] + 1e-9 >= DOSE_SECONDS {
                let start = self.target();
                if self.station + 1 == TREE_COUNT {
                    self.returning = true;
                } else {
                    self.station += 1;
                }
                self.leg = segment(start, self.target());
                self.elapsed = 0.0;
                self.phase = Phase::Travel;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_solvers::airflow::{Quadrotor, QuadrotorConfig};
    #[test]
    fn mission_really_stops_and_sprays_all_sixteen_trees() {
        let start = station_position(0, 3.0);
        let mut drone = Quadrotor::new(
            start,
            QuadrotorConfig {
                mass: 5.6,
                ..Default::default()
            },
        )
        .unwrap();
        let mut mission = Mission::new(start, 3.6);
        let dt = 1.0 / 60.0;
        for _ in 0..(180.0 / dt) as usize {
            let r = mission.update(dt, drone.pose().unwrap(), drone.velocity().unwrap(), true);
            drone.track(r, Vec3::Z).unwrap();
            drone.step(dt, Vec3::new(0.5, 0.0, 0.2)).unwrap();
            mission.commit_spray(dt, drone.pose().unwrap(), drone.velocity().unwrap());
            if mission.phase == Phase::Complete {
                break;
            }
        }
        assert_eq!(mission.phase, Phase::Complete);
        assert!(
            mission
                .dose
                .iter()
                .all(|s| (*s - DOSE_SECONDS).abs() < 0.02),
            "{:?}",
            mission.dose
        );
        assert!(mission.max_spray_speed < 0.10 && mission.max_spray_error < 0.10);
        assert!(
            drone
                .pose()
                .unwrap()
                .position
                .distance(station_position(0, 3.6))
                < 0.1
        );
        assert!(!mission.pump);
    }

    #[test]
    fn pump_requires_stationary_level_arrival_and_enabled_flight() {
        let start = station_position(0, 3.0);
        let drone = Quadrotor::new(start, Default::default()).unwrap();
        let mut mission = Mission::new(start - Vec3::Y * 0.6, 3.0);
        for _ in 0..600 {
            mission.update(1.0 / 60.0, drone.pose().unwrap(), Vec3::X, true);
        }
        assert!(!mission.pump && mission.dose.iter().sum::<f64>() == 0.0);
        for _ in 0..35 {
            mission.update(1.0 / 60.0, drone.pose().unwrap(), Vec3::ZERO, true);
        }
        assert!(mission.pump);
        mission.update(1.0 / 60.0, drone.pose().unwrap(), Vec3::ZERO, false);
        assert!(!mission.pump);
    }
}
