use crate::{ag_mission::Mission, leafy_model::random};
use molla_core::Result;
use molla_math::{Transform, Vec3};
use molla_solvers::{
    airflow::{TurbulentDispersion, WaterDroplet},
    attachments::Particle,
};
use std::collections::HashMap;

pub const PRESSURE_RATIO: f64 = 27.0;
pub const NOZZLE_FLOW_RATIO: f64 = 1.0 / 3.0;
pub const DIAMETER_RANGE_M: [f64; 2] = [160e-6, 240e-6];
pub const PARCELS_S: f64 = 7200.0;
pub const CONE_HALF_ANGLE_DEG: f64 = 42.0;
pub const DOWNWARD_EXIT_GAIN: f64 = 2.0;
pub const DISPERSION_INTENSITY: f64 = 0.125;
pub const DISPERSION_CORRELATION_S: f64 = 0.12;
pub const SPREAD_DEPTHS_M: [f64; 4] = [0.25, 0.75, 1.5, 2.5];
pub const BOOM_HALF_SIDE_M: f64 = 0.28;
const CELL: f64 = 0.25;
pub const CAPACITY: usize = 262_144;

pub fn flow_l_s() -> f64 {
    0.012 * NOZZLE_FLOW_RATIO * PRESSURE_RATIO.sqrt()
}
pub fn exit_speed_m_s() -> f64 {
    3.5 * PRESSURE_RATIO.sqrt()
}

pub fn effective_cone_angle_deg() -> f64 {
    2.0 * (CONE_HALF_ANGLE_DEG.to_radians().tan() / DOWNWARD_EXIT_GAIN)
        .atan()
        .to_degrees()
}

fn nozzle_velocity(index: usize, downward_gain: f64) -> Vec3 {
    let angle = random(index, 2301) * std::f64::consts::TAU;
    let cosine = 1.0 - random(index, 2302) * (1.0 - CONE_HALF_ANGLE_DEG.to_radians().cos());
    let radial = (1.0 - cosine * cosine).sqrt();
    Vec3::new(
        radial * angle.cos(),
        radial * angle.sin(),
        -cosine * downward_gain,
    ) * exit_speed_m_s()
}

pub fn nozzle_local(index: usize) -> Vec3 {
    let h = BOOM_HALF_SIDE_M;
    [
        Vec3::new(-h, -h, -0.34),
        Vec3::new(h, -h, -0.34),
        Vec3::new(h, h, -0.34),
        Vec3::new(-h, h, -0.34),
    ][index]
}

pub struct Parcel {
    pub drop: WaterDroplet,
    pub previous: Vec3,
    pub emitted_at: Vec3,
    pub litres: f64,
    pub age: f64,
    pub dispersion: TurbulentDispersion,
    crossed_depths: u8,
}

/// One parcel landed on a leaf. `t` runs base→tip in [0, 1], `x` is the lateral
/// offset in blade lengths, `side` is the face (±normal) that was hit.
#[derive(Clone, Copy, Debug)]
pub struct Deposit {
    pub leaf: usize,
    pub t: f64,
    pub x: f64,
    pub side: f64,
    pub time: f64,
    pub litres: f64,
    pub droplets: f64,
}

/// Physical droplets represented by a parcel of `litres` at `diameter` metres.
pub fn droplet_count(litres: f64, diameter: f64) -> f64 {
    litres * 1e-3 / (std::f64::consts::PI / 6.0 * diameter.powi(3))
}

pub struct Spray {
    pub mission: Mission,
    pub parcels: Vec<Parcel>,
    pub deposits: Vec<Deposit>,
    pub time: f64,
    pub wet_litres: Vec<f64>,
    pub tank_litres: f64,
    pub emitted_litres: f64,
    pub ground_litres: f64,
    pub canopy_litres: f64,
    pub peak_parcels: usize,
    pub capacity_stops: usize,
    pub wind_samples: usize,
    pub max_sampled_air_speed: f64,
    pub max_horizontal_drift: f64,
    pub upward_volume_time: f64,
    pub airborne_volume_time: f64,
    pub max_rise_above_nozzle: f64,
    pub dispersion_intensity: f64,
    pub spread_crossings: [usize; 4],
    spread_sum: [Vec3; 4],
    spread_sum_squared: [f64; 4],
    pub on: bool,
    seed: usize,
    emission: f64,
}

impl Spray {
    pub fn new(start: Vec3, height: f64, leaves: usize) -> Self {
        Self {
            mission: Mission::new(start, height),
            parcels: Vec::new(),
            deposits: Vec::new(),
            time: 0.0,
            wet_litres: vec![0.0; leaves],
            tank_litres: 0.75,
            emitted_litres: 0.0,
            ground_litres: 0.0,
            canopy_litres: 0.0,
            peak_parcels: 0,
            capacity_stops: 0,
            wind_samples: 0,
            max_sampled_air_speed: 0.0,
            max_horizontal_drift: 0.0,
            upward_volume_time: 0.0,
            airborne_volume_time: 0.0,
            max_rise_above_nozzle: 0.0,
            dispersion_intensity: DISPERSION_INTENSITY,
            spread_crossings: [0; 4],
            spread_sum: [Vec3::ZERO; 4],
            spread_sum_squared: [0.0; 4],
            on: false,
            seed: 0,
            emission: 0.0,
        }
    }

    pub fn advance(&mut self, dt: f64, air: &[Vec3], leaves: &[Particle]) -> Result<()> {
        assert_eq!(air.len(), self.parcels.len());
        self.time += dt;
        let mut grid: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
        if !self.parcels.is_empty() {
            for (i, leaf) in leaves.iter().enumerate() {
                if let Some(blade) = leaf.leaf {
                    let r =
                        Vec3::splat(blade.length.max(blade.width) * 0.5) + leaf.velocity.abs() * dt;
                    cells(leaf.position - r, leaf.position + r, |key| {
                        grid.entry(key).or_default().push(i)
                    });
                }
            }
        }
        let mut alive = Vec::with_capacity(self.parcels.len());
        for (mut parcel, wind) in self.parcels.drain(..).zip(air) {
            let from = parcel.drop.position;
            parcel.previous = from;
            let fluctuation = parcel.dispersion.advance(
                dt,
                self.dispersion_intensity * wind.length(),
                DISPERSION_CORRELATION_S,
            )?;
            parcel.drop.advance(dt, *wind + fluctuation)?;
            parcel.age += dt;
            let to = parcel.drop.position;
            self.wind_samples += 1;
            self.max_sampled_air_speed = self.max_sampled_air_speed.max(wind.length());
            let drift = to - parcel.emitted_at;
            self.max_horizontal_drift = self.max_horizontal_drift.max(drift.x.hypot(drift.z));
            let mut first = if to.y <= 0.0 && from.y > 0.0 {
                from.y / (from.y - to.y)
            } else {
                f64::INFINITY
            };
            let mut hit = None;
            cells(from.min(to), from.max(to), |key| {
                if let Some(indices) = grid.get(&key) {
                    for &i in indices {
                        if let Some(contact) = leaf_hit(from, to, &leaves[i], dt) {
                            if contact.t < first {
                                first = contact.t;
                                hit = Some((i, contact));
                            }
                        }
                    }
                }
            });
            let end = from.lerp(to, first.min(1.0));
            let volume_time = parcel.litres * dt * first.min(1.0);
            self.airborne_volume_time += volume_time;
            if end.y > from.y {
                self.upward_volume_time += volume_time;
            }
            self.max_rise_above_nozzle =
                self.max_rise_above_nozzle.max(end.y - parcel.emitted_at.y);
            for (i, depth) in SPREAD_DEPTHS_M.iter().enumerate() {
                let plane = parcel.emitted_at.y - depth;
                if parcel.crossed_depths & (1 << i) == 0 && from.y > plane && end.y <= plane {
                    let crossing = from.lerp(end, (from.y - plane) / (from.y - end.y));
                    let mut lateral = crossing - parcel.emitted_at;
                    lateral.y = 0.0;
                    self.spread_crossings[i] += 1;
                    self.spread_sum[i] += lateral;
                    self.spread_sum_squared[i] += lateral.length_squared();
                    parcel.crossed_depths |= 1 << i;
                }
            }
            if first <= 1.0 {
                if let Some((i, contact)) = hit {
                    self.wet_litres[i] += parcel.litres;
                    self.canopy_litres += parcel.litres;
                    let length = leaves[i].leaf.map_or(1.0, |blade| blade.length);
                    self.deposits.push(Deposit {
                        leaf: i,
                        t: 0.5 + contact.along / length,
                        x: contact.lateral / length,
                        side: contact.side,
                        time: self.time,
                        litres: parcel.litres,
                        droplets: droplet_count(parcel.litres, parcel.drop.diameter()),
                    });
                } else {
                    self.ground_litres += parcel.litres;
                }
            } else {
                alive.push(parcel);
            }
        }
        self.parcels = alive;
        Ok(())
    }

    pub fn emit(&mut self, dt: f64, pose: Transform, linear: Vec3, angular: Vec3) -> Result<()> {
        self.on = self.mission.pump && self.tank_litres > 0.0;
        if !self.on {
            self.emission = 0.0;
            return Ok(());
        }
        self.emission += dt * PARCELS_S;
        while self.emission >= 1.0 && self.tank_litres > 0.0 {
            self.emission -= 1.0;
            if self.parcels.len() >= CAPACITY {
                self.capacity_stops += 1;
                self.on = false;
                continue;
            }
            let i = self.seed;
            self.seed += 1;
            let offset = pose.rotation * nozzle_local(i % 4);
            let velocity = linear
                + angular.cross(offset)
                + pose.rotation * nozzle_velocity(i, DOWNWARD_EXIT_GAIN);
            let litres = (flow_l_s() / PARCELS_S).min(self.tank_litres);
            let position = pose.position + offset;
            self.parcels.push(Parcel {
                drop: WaterDroplet::new(
                    position,
                    velocity,
                    DIAMETER_RANGE_M[0]
                        + random(i, 2303) * (DIAMETER_RANGE_M[1] - DIAMETER_RANGE_M[0]),
                )?,
                previous: position,
                emitted_at: position,
                age: 0.0,
                dispersion: TurbulentDispersion::new((i as u64).wrapping_mul(0xd1342543de82ef95)),
                crossed_depths: 0,
                litres,
            });
            self.tank_litres -= litres;
            self.emitted_litres += litres;
        }
        self.peak_parcels = self.peak_parcels.max(self.parcels.len());
        Ok(())
    }

    pub fn airborne_litres(&self) -> f64 {
        self.parcels.iter().map(|p| p.litres).sum()
    }

    pub fn spread_rms(&self) -> [f64; 4] {
        std::array::from_fn(|i| {
            let n = self.spread_crossings[i].max(1) as f64;
            (self.spread_sum_squared[i] / n - (self.spread_sum[i] / n).length_squared())
                .max(0.0)
                .sqrt()
        })
    }
}

fn cells(lo: Vec3, hi: Vec3, mut f: impl FnMut([i32; 3])) {
    let a = (lo / CELL).floor().to_array().map(|v| v as i32);
    let b = (hi / CELL).floor().to_array().map(|v| v as i32);
    for x in a[0]..=b[0] {
        for y in a[1]..=b[1] {
            for z in a[2]..=b[2] {
                f([x, y, z]);
            }
        }
    }
}

/// Swept segment/blade intersection: `t` along the step, contact point in the
/// blade frame (`along` the direction, `lateral` along direction × normal), hit face.
struct Contact {
    t: f64,
    along: f64,
    lateral: f64,
    side: f64,
}

fn leaf_hit(from: Vec3, to: Vec3, leaf: &Particle, dt: f64) -> Option<Contact> {
    let blade = leaf.leaf?;
    let n = blade.frame.normal;
    let center = leaf.position - leaf.velocity * dt;
    let delta = to - from - leaf.velocity * dt;
    let divisor = delta.dot(n);
    if divisor.abs() < 1e-12 {
        return None;
    }
    let t = (center - from).dot(n) / divisor;
    if !(0.0..=1.0).contains(&t) {
        return None;
    }
    let p = from + delta * t - center;
    let along = p.dot(blade.frame.direction);
    let lateral = p.dot(blade.frame.direction.cross(n));
    let (x, y) = (along / (blade.length * 0.5), lateral / (blade.width * 0.5));
    (x * x + y * y <= 1.0).then_some(Contact {
        t,
        along,
        lateral,
        side: if (from - center).dot(n) >= 0.0 { 1.0 } else { -1.0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_solvers::airflow::Quadrotor;
    #[test]
    fn swept_drops_deposit_on_moving_leaves_before_reaching_the_floor() {
        use molla_solvers::attachments::{BladeFrame, LeafBody};
        let mut leaf = Particle::attached(Vec3::Y);
        leaf.velocity = Vec3::Y * 0.1;
        leaf.leaf = Some(LeafBody {
            frame: BladeFrame {
                direction: Vec3::X,
                normal: Vec3::Y,
                angular_velocity: Vec3::ZERO,
            },
            length: 0.2,
            width: 0.12,
            area: 0.01,
            camber: 0.0,
        });
        assert!(leaf_hit(Vec3::Y * 1.2, Vec3::Y * 0.8, &leaf, 0.1).is_some());
        assert!(
            leaf_hit(
                Vec3::new(0.3, 1.2, 0.0),
                Vec3::new(0.3, 0.8, 0.0),
                &leaf,
                0.1
            )
            .is_none()
        );
        let mut spray = Spray::new(crate::ag_mission::station_position(0, 3.0), 3.6, 1);
        spray.parcels.push(Parcel {
            drop: WaterDroplet::new(Vec3::Y * 1.04, -Vec3::Y * 4.0, 400e-6).unwrap(),
            previous: Vec3::Y * 1.04,
            emitted_at: Vec3::Y * 1.04,
            litres: 0.001,
            age: 0.0,
            dispersion: TurbulentDispersion::new(0),
            crossed_depths: 0,
        });
        spray
            .advance(1.0 / 60.0, &[-Vec3::Y * 4.0], &[leaf])
            .unwrap();
        assert!(spray.parcels.is_empty());
        assert_eq!(spray.canopy_litres, 0.001);
        assert_eq!(spray.wet_litres, [0.001]);
        assert_eq!(spray.ground_litres, 0.0);
    }

    #[test]
    fn leaf_deposits_record_blade_coordinates_face_and_droplet_count() {
        use molla_solvers::attachments::{BladeFrame, LeafBody};
        let mut leaf = Particle::attached(Vec3::Y);
        leaf.leaf = Some(LeafBody {
            frame: BladeFrame {
                direction: Vec3::X,
                normal: Vec3::Y,
                angular_velocity: Vec3::ZERO,
            },
            length: 0.2,
            width: 0.12,
            area: 0.01,
            camber: 0.0,
        });
        let mut spray = Spray::new(Vec3::Y * 3.0, 3.6, 1);
        for (start, velocity) in [
            (Vec3::new(0.05, 1.02, 0.02), -Vec3::Y * 4.0),
            (Vec3::new(-0.04, 0.98, -0.01), Vec3::Y * 4.0),
        ] {
            spray.parcels.push(Parcel {
                drop: WaterDroplet::new(start, velocity, 200e-6).unwrap(),
                previous: start,
                emitted_at: start,
                litres: 2e-6,
                age: 0.0,
                dispersion: TurbulentDispersion::new(0),
                crossed_depths: 0,
            });
        }
        spray.dispersion_intensity = 0.0;
        spray
            .advance(1.0 / 60.0, &[-Vec3::Y * 4.0, Vec3::Y * 4.0], &[leaf])
            .unwrap();
        assert!(spray.parcels.is_empty());
        let [top, bottom] = spray.deposits[..] else {
            panic!("expected two deposits, got {:?}", spray.deposits);
        };
        assert_eq!((top.leaf, top.side, bottom.side), (0, 1.0, -1.0));
        assert!((top.t - 0.75).abs() < 0.02 && (top.x - 0.1).abs() < 0.02);
        assert!((bottom.t - 0.3).abs() < 0.02 && (bottom.x + 0.05).abs() < 0.02);
        assert!((top.time - 1.0 / 60.0).abs() < 1e-12);
        let single = std::f64::consts::PI / 6.0 * 200e-6_f64.powi(3) * 1e3;
        assert!((top.droplets - 2e-6 / single).abs() < 1e-9);
        assert!((top.droplets - 477.46).abs() < 0.01);
    }

    #[test]
    fn square_boom_emits_evenly_from_four_transformed_corners() {
        let corners = [0, 1, 2, 3].map(nozzle_local);
        assert!(
            (corners.iter().copied().sum::<Vec3>() / 4.0 - Vec3::new(0.0, 0.0, -0.34)).length()
                < 1e-12
        );
        for i in 0..4 {
            let edge = corners[(i + 1) % 4] - corners[i];
            let next = corners[(i + 2) % 4] - corners[(i + 1) % 4];
            assert!((edge.length() - 2.0 * BOOM_HALF_SIDE_M).abs() < 1e-12);
            assert!(edge.dot(next).abs() < 1e-12);
        }
        let drone = Quadrotor::new(Vec3::Y * 3.0, Default::default()).unwrap();
        let mut pose = drone.pose().unwrap();
        pose.rotation = molla_math::Quat::from_rotation_z(0.2)
            * molla_math::Quat::from_rotation_y(0.7)
            * pose.rotation;
        let mut spray = Spray::new(Vec3::Y * 3.0, 3.6, 0);
        spray.mission.pump = true;
        spray
            .emit(1.0 / 60.0, pose, Vec3::ZERO, Vec3::ZERO)
            .unwrap();
        let mut counts = [0; 4];
        for (i, parcel) in spray.parcels.iter().enumerate() {
            let expected = pose.transform_point(corners[i % 4]);
            assert!(parcel.drop.position.distance(expected) < 1e-12);
            assert!(parcel.emitted_at.distance(expected) < 1e-12);
            counts[i % 4] += 1;
        }
        assert_eq!(counts, [30; 4]);
        assert!((spray.emitted_litres - flow_l_s() / 60.0).abs() < 1e-12);
    }

    #[test]
    fn nozzle_water_is_gated_and_conserved_through_ground_contact() {
        let drone = Quadrotor::new(Vec3::Y * 3.0, Default::default()).unwrap();
        let mut spray = Spray::new(Vec3::Y * 3.0, 3.6, 0);
        spray
            .emit(1.0 / 60.0, drone.pose().unwrap(), Vec3::ZERO, Vec3::ZERO)
            .unwrap();
        assert!(spray.parcels.is_empty());
        spray.mission.pump = true;
        for _ in 0..60 {
            spray
                .emit(1.0 / 60.0, drone.pose().unwrap(), Vec3::X, Vec3::ZERO)
                .unwrap();
        }
        assert_eq!(spray.parcels.len(), 7200);
        assert!((spray.emitted_litres - flow_l_s()).abs() < 1e-12);
        assert!(spray.parcels.iter().all(|p| p.drop.velocity.y < -27.0));
        assert!(spray.parcels.iter().any(|p| p.drop.velocity.y > -28.0));
        assert!(spray.parcels.iter().enumerate().all(|(i, p)| {
            (p.drop.velocity
                - Vec3::X
                - drone.pose().unwrap().rotation * nozzle_velocity(i, DOWNWARD_EXIT_GAIN))
            .length()
                < 1e-10
        }));
        assert!(
            spray
                .parcels
                .iter()
                .all(|p| (DIAMETER_RANGE_M[0]..=DIAMETER_RANGE_M[1]).contains(&p.drop.diameter()))
        );
        for _ in 0..600 {
            spray
                .advance(
                    1.0 / 60.0,
                    &vec![Vec3::new(1.0, -1.0, 0.0); spray.parcels.len()],
                    &[],
                )
                .unwrap();
        }
        assert!(spray.parcels.is_empty());
        assert!((spray.ground_litres - spray.emitted_litres).abs() < 1e-12);
        assert!((spray.tank_litres + spray.emitted_litres - 0.75).abs() < 1e-12);
    }

    #[test]
    fn extra_nozzle_drive_doubles_only_downward_velocity() {
        for index in 0..7200 {
            let before = nozzle_velocity(index, 1.0);
            let after = nozzle_velocity(index, DOWNWARD_EXIT_GAIN);
            assert_eq!(after.x, before.x);
            assert_eq!(after.y, before.y);
            assert_eq!(after.z, 2.0 * before.z);
        }
        assert!((exit_speed_m_s() * DOWNWARD_EXIT_GAIN - 36.373066958946424).abs() < 1e-12);
        assert!(effective_cone_angle_deg() > 48.0 && effective_cone_angle_deg() < 49.0);
        let drone = Quadrotor::new(Vec3::Y * 3.0, Default::default()).unwrap();
        let pose = drone.pose().unwrap();
        let (linear, angular) = (Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 0.2, 0.3));
        let mut spray = Spray::new(Vec3::Y * 3.0, 3.6, 0);
        spray.mission.pump = true;
        spray.emit(1.0 / 60.0, pose, linear, angular).unwrap();
        for (i, parcel) in spray.parcels.iter().enumerate() {
            let relative = pose.rotation.inverse()
                * (parcel.drop.velocity
                    - linear
                    - angular.cross(pose.rotation * nozzle_local(i % 4)));
            assert!((relative - nozzle_velocity(i, DOWNWARD_EXIT_GAIN)).length() < 1e-10);
        }
    }

    #[test]
    fn increased_pressure_preserves_the_full_route_liquid_budget() {
        let litres =
            flow_l_s() * crate::ag_mission::DOSE_SECONDS * crate::orchard_model::TREE_COUNT as f64;
        assert!((PRESSURE_RATIO - (exit_speed_m_s() / 3.5).powi(2)).abs() < 1e-12);
        assert!(
            (PRESSURE_RATIO - (flow_l_s() / (0.012 * NOZZLE_FLOW_RATIO)).powi(2)).abs() < 1e-12
        );
        assert!((exit_speed_m_s() / (3.5 * 3.0_f64.sqrt()) - 3.0).abs() < 1e-12);
        assert!((litres - 0.6651075101064488).abs() < 1e-12);
        assert!(0.75 - litres > 0.08);
        assert!(PARCELS_S * 32.0 < CAPACITY as f64);
    }

    #[test]
    fn mist_follows_reversed_wind_and_downwash_without_scripted_drift() {
        let run = |wind: Vec3| {
            let mut spray = Spray::new(Vec3::Y * 3.0, 3.6, 0);
            spray.dispersion_intensity = 0.0;
            spray.parcels.push(Parcel {
                drop: WaterDroplet::new(Vec3::Y * 3.0, -Vec3::Y * exit_speed_m_s(), 45e-6).unwrap(),
                previous: Vec3::Y * 3.0,
                emitted_at: Vec3::Y * 3.0,
                litres: 0.001,
                age: 0.0,
                dispersion: TurbulentDispersion::new(0),
                crossed_depths: 0,
            });
            for _ in 0..60 {
                spray.advance(1.0 / 60.0, &[wind], &[]).unwrap();
            }
            assert_eq!(spray.wind_samples, 60);
            spray.parcels[0].drop
        };
        let calm = run(Vec3::ZERO);
        let right = run(Vec3::X * 2.0);
        let left = run(-Vec3::X * 2.0);
        let downwash = run(Vec3::new(2.0, -1.5, 0.0));
        assert!(calm.position.x.abs() < 1e-12);
        assert!(right.position.x > 1.9 && left.position.x < -1.9);
        assert!((right.position.x + left.position.x).abs() < 1e-12);
        assert!((right.velocity.x - 2.0).abs() < 1e-6);
        assert!(downwash.position.y < calm.position.y - 1.4);
        assert!(downwash.velocity.y < -1.5);
    }

    #[test]
    fn heavier_spray_resists_recirculation_without_suppressing_gusts() {
        let run = |diameters: [f64; 2]| {
            let drone = Quadrotor::new(Vec3::Y * 3.34, Default::default()).unwrap();
            let mut spray = Spray::new(Vec3::Y * 3.34, 3.34, 0);
            spray.mission.pump = true;
            spray
                .emit(0.1, drone.pose().unwrap(), Vec3::ZERO, Vec3::ZERO)
                .unwrap();
            for (i, parcel) in spray.parcels.iter_mut().enumerate() {
                parcel.drop = WaterDroplet::new(
                    parcel.drop.position,
                    parcel.drop.velocity,
                    diameters[0] + random(i, 2303) * (diameters[1] - diameters[0]),
                )
                .unwrap();
            }
            for step in 0..1800 {
                let phase = step as f64 / 60.0 * std::f64::consts::TAU;
                let air = Vec3::new(1.2, 0.35 + 0.45 * phase.sin(), 0.4 * phase.cos());
                spray
                    .advance(1.0 / 60.0, &vec![air; spray.parcels.len()], &[])
                    .unwrap();
            }
            assert!(
                (spray.airborne_litres() + spray.ground_litres - spray.emitted_litres).abs()
                    < 1e-12
            );
            (
                spray.ground_litres / spray.emitted_litres,
                spray.upward_volume_time / spray.airborne_volume_time,
                spray.max_horizontal_drift,
                spray.max_rise_above_nozzle,
            )
        };
        let before = run([30e-6, 60e-6]);
        let after = run(DIAMETER_RANGE_M);
        eprintln!(
            "recirculation (deposited, rising residence, drift, rise): before={before:?}, after={after:?}"
        );
        assert!(before.0 < 0.25 && after.0 > 0.99);
        assert!(after.1 < before.1 * 0.5);
        assert!(after.2 > 0.5 && after.2 < before.2);
        assert!(before.3 > 1.0 && after.3 < 0.1);
    }

    #[test]
    fn physical_mist_spreads_progressively_below_the_nozzles() {
        let run = |intensity: f64, dt: f64, diameters: [f64; 2]| {
            let drone = Quadrotor::new(Vec3::Y * 5.34, Default::default()).unwrap();
            let mut spray = Spray::new(Vec3::Y * 5.34, 5.34, 0);
            spray.dispersion_intensity = intensity;
            spray.mission.pump = true;
            spray
                .emit(0.1, drone.pose().unwrap(), Vec3::ZERO, Vec3::ZERO)
                .unwrap();
            for (i, parcel) in spray.parcels.iter_mut().enumerate() {
                parcel.drop = WaterDroplet::new(
                    parcel.drop.position,
                    parcel.drop.velocity,
                    diameters[0] + random(i, 2303) * (diameters[1] - diameters[0]),
                )
                .unwrap();
            }
            for _ in 0..(3.0 / dt).round() as usize {
                spray
                    .advance(
                        dt,
                        &vec![Vec3::new(0.4, -3.0, 0.0); spray.parcels.len()],
                        &[],
                    )
                    .unwrap();
            }
            assert!(spray.spread_crossings.iter().all(|n| *n > 700));
            assert!(
                (spray.airborne_litres() + spray.ground_litres - spray.emitted_litres).abs()
                    < 1e-12
            );
            spray.spread_rms()
        };
        let aerosol = [30e-6, 60e-6];
        let laminar = run(0.0, 1.0 / 60.0, aerosol);
        let spread = run(DISPERSION_INTENSITY, 1.0 / 60.0, aerosol);
        let wide = run(0.25, 1.0 / 60.0, aerosol);
        let refined = run(DISPERSION_INTENSITY, 1.0 / 120.0, aerosol);
        eprintln!(
            "mist widths laminar={laminar:?}, balanced={spread:?}, wide={wide:?}, refined={refined:?}"
        );
        assert!(spread[0] > 0.025);
        assert!(spread.windows(2).all(|pair| pair[1] > pair[0] * 1.2));
        assert!(spread[3] > laminar[3] * 3.0);
        for (balanced, previous) in spread.into_iter().zip(wide) {
            assert!(balanced < previous * 0.8);
        }
        for (a, b) in spread.into_iter().zip(refined) {
            assert!((a - b).abs() < a * 0.15);
        }
        let heavy = run(DISPERSION_INTENSITY, 1.0 / 60.0, DIAMETER_RANGE_M);
        let heavy_refined = run(DISPERSION_INTENSITY, 1.0 / 120.0, DIAMETER_RANGE_M);
        let heavy_laminar = run(0.0, 1.0 / 60.0, DIAMETER_RANGE_M);
        let heavy_wide = run(0.25, 1.0 / 60.0, DIAMETER_RANGE_M);
        assert!(heavy.windows(2).all(|pair| pair[1] > pair[0] * 1.1));
        assert!(heavy[3] > heavy_laminar[3] * 1.1 && heavy[3] < heavy_wide[3] * 0.8);
        for (a, b) in heavy.into_iter().zip(heavy_refined) {
            assert!((a - b).abs() < a * 0.15);
        }
    }
}
