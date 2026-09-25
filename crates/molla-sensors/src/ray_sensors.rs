//! Ray-based sensors on the owned scene: distance, light and radar, plus the
//! joint position sensor.
//!
//! Distance sensors cast a cone of rays and reduce them as a laser, an
//! infra-red emitter or a sonar would. Light sensors add direct light from
//! host-described sources that a ray finds unblocked, plus sky light. Radars
//! list host-named target bodies inside their field of view, range and line
//! of sight, with radial speed and radar-equation received power. All of
//! them take sensor world poses from the host and cast through a
//! [`RayQuery`]: the GPU [`SceneRays`] on a [`SensorScene`], the CPU
//! [`RigidRays`] on a live rigid scene, or the host's own.

use crate::raycast::Raycaster;
use crate::scene::SensorScene;
use molla_compute::Engine;
use molla_core::{BodyId, Error, Result};
use molla_geometry::raycast_cpu::CpuRaycaster;
use molla_math::{Real, Transform, Vec3};
use molla_sim::runtime::RigidScene;
use molla_sim::{Model, State};

/// Nearest hit of one ray: distance (infinite for a miss) and body
/// (`BodyId::NONE` for static geometry or a miss).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestHit {
    pub t: Real,
    pub body: BodyId,
}

impl NearestHit {
    pub const MISS: Self = Self {
        t: Real::INFINITY,
        body: BodyId::NONE,
    };
}

/// Ray casting for the ray sensors.
pub trait RayQuery {
    /// Nearest hits of rays from `origins` along unit `directions` within
    /// `max_distance`, one per ray.
    fn cast(&mut self, origins: &[Vec3], directions: &[Vec3], max_distance: Real) -> Result<Vec<NearestHit>>;
}

/// [`RayQuery`] on a [`SensorScene`] with a GPU [`Raycaster`]; each cast
/// waits for the device.
pub struct SceneRays<'a> {
    pub engine: &'a Engine,
    pub raycaster: &'a mut Raycaster,
    pub scene: &'a SensorScene,
    pub state: &'a State,
}

impl RayQuery for SceneRays<'_> {
    fn cast(&mut self, origins: &[Vec3], directions: &[Vec3], max_distance: Real) -> Result<Vec<NearestHit>> {
        let hits = self
            .raycaster
            .cast_scene(self.engine, self.scene, self.state, origins, directions, max_distance)?;
        Ok(hits.iter().map(|h| NearestHit { t: h.t, body: h.body_id }).collect())
    }
}

/// [`RayQuery`] on a live rigid scene on the CPU, for the few rays of
/// these sensors without a device round trip. Keep `caster` between calls
/// so prepared collider geometry is reused.
pub struct RigidRays<'a> {
    pub scene: &'a RigidScene,
    pub caster: &'a mut CpuRaycaster,
}

impl RayQuery for RigidRays<'_> {
    fn cast(&mut self, origins: &[Vec3], directions: &[Vec3], max_distance: Real) -> Result<Vec<NearestHit>> {
        origins
            .iter()
            .zip(directions)
            .map(|(&origin, &direction)| {
                let hit = self.caster.cast_ray(self.scene, origin, direction, max_distance)?;
                Ok(hit.map_or(NearestHit::MISS, |h| NearestHit {
                    t: h.distance,
                    body: h.body.and_then(|b| self.scene.body_index(b)).unwrap_or(BodyId::NONE),
                }))
            })
            .collect()
    }
}

/// Runs `f` with a [`SceneRays`] on a sensor's own raycaster, which the
/// host-only constructors leave out.
fn with_scene_rays<T>(
    raycaster: &mut Option<Raycaster>,
    engine: &Engine,
    scene: &SensorScene,
    state: &State,
    f: impl FnOnce(&mut SceneRays<'_>) -> Result<T>,
) -> Result<T> {
    let raycaster = raycaster
        .as_mut()
        .ok_or_else(|| Error::Build("ray sensor built for a host ray query has no GPU raycaster".into()))?;
    f(&mut SceneRays {
        engine,
        raycaster,
        scene,
        state,
    })
}

/// Piecewise-linear map from a measurement to an output value through
/// `(input, output)` rows sorted by input; clamps outside the table and
/// returns the input unchanged for an empty table.
pub fn lookup(table: &[[Real; 2]], input: Real) -> Real {
    let (Some(first), Some(last)) = (table.first(), table.last()) else {
        return input;
    };
    if !input.is_finite() || input >= last[0] {
        return last[1];
    }
    if input <= first[0] {
        return first[1];
    }
    let i = table.partition_point(|row| row[0] <= input);
    let ([x0, y0], [x1, y1]) = (table[i - 1], table[i]);
    y0 + (y1 - y0) * (input - x0) / (x1 - x0)
}

/// How a distance sensor reduces its rays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceKind {
    /// The axis ray only.
    #[default]
    Laser,
    /// Gaussian-weighted mean over the cone.
    InfraRed,
    /// Nearest echo in the cone.
    Sonar,
}

/// A distance sensor looking along its frame's +X.
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceSpec {
    pub kind: DistanceKind,
    pub rays: u32,
    /// Full cone angle (rad).
    pub aperture: Real,
    pub max_range: Real,
    /// Infra-red weight width as a fraction of the aperture.
    pub gaussian_width: Real,
    /// Distance to output value; empty reports metres.
    pub lookup: Vec<[Real; 2]>,
}

impl Default for DistanceSpec {
    fn default() -> Self {
        Self {
            kind: DistanceKind::Laser,
            rays: 1,
            aperture: 0.0,
            max_range: 10.0,
            gaussian_width: 1.0,
            lookup: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceReading {
    /// Metres to the reduced echo; infinite when nothing is in range.
    pub distance: Real,
    /// `distance` through the lookup table (misses use `max_range`).
    pub value: Real,
    /// Body of the axis or nearest hit; `BodyId::NONE` for static geometry
    /// or a miss.
    pub body: BodyId,
}

/// Sensor-frame ray directions: the axis, then the rest on a ring at half
/// the aperture.
pub fn cone_rays(rays: u32, aperture: Real) -> Vec<Vec3> {
    let mut out = vec![Vec3::X];
    let ring = rays.saturating_sub(1);
    let half = 0.5 * aperture;
    for i in 0..ring {
        let around = std::f64::consts::TAU * i as Real / ring as Real;
        out.push(Vec3::new(half.cos(), half.sin() * around.cos(), half.sin() * around.sin()));
    }
    out
}

/// A batch of distance sensors.
pub struct DistanceSensors {
    specs: Vec<DistanceSpec>,
    rays: Vec<Vec<Vec3>>,
    raycaster: Option<Raycaster>,
}

impl DistanceSensors {
    /// Sensors that cast on the GPU sensor scene with [`Self::update`].
    pub fn new(engine: &Engine, specs: Vec<DistanceSpec>) -> Result<Self> {
        let mut sensors = Self::host(specs)?;
        let total = sensors.rays.iter().map(Vec::len).sum::<usize>().max(1);
        sensors.raycaster = Some(Raycaster::new(engine, total as u32)?);
        Ok(sensors)
    }

    /// Sensors that cast through a host [`RayQuery`] with [`Self::update_with`].
    pub fn host(specs: Vec<DistanceSpec>) -> Result<Self> {
        if specs.iter().any(|s| s.rays == 0 || !(s.max_range > 0.0) || !s.aperture.is_finite()) {
            return Err(Error::Build("distance sensor: rays, range or aperture invalid".into()));
        }
        let rays = specs.iter().map(|s| cone_rays(s.rays, s.aperture)).collect();
        Ok(Self {
            specs,
            rays,
            raycaster: None,
        })
    }

    pub fn specs(&self) -> &[DistanceSpec] {
        &self.specs
    }

    /// Readings for sensors at `poses` (world, one per spec) on the GPU.
    pub fn update(
        &mut self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
        poses: &[Transform],
    ) -> Result<Vec<DistanceReading>> {
        let mut raycaster = self.raycaster.take();
        let out = with_scene_rays(&mut raycaster, engine, scene, state, |rays| self.update_with(rays, poses));
        self.raycaster = raycaster;
        out
    }

    /// Readings for sensors at `poses` (world, one per spec).
    pub fn update_with(&self, query: &mut dyn RayQuery, poses: &[Transform]) -> Result<Vec<DistanceReading>> {
        if poses.len() != self.specs.len() {
            return Err(Error::Build("distance sensor: one pose per sensor".into()));
        }
        let (mut origins, mut directions) = (Vec::new(), Vec::new());
        for (pose, rays) in poses.iter().zip(&self.rays) {
            for ray in rays {
                origins.push(pose.position);
                directions.push(pose.rotation * *ray);
            }
        }
        let range = self.specs.iter().map(|s| s.max_range).fold(0.0, Real::max);
        let hits = query.cast(&origins, &directions, range)?;
        let mut offset = 0;
        let mut out = Vec::with_capacity(self.specs.len());
        for (spec, rays) in self.specs.iter().zip(&self.rays) {
            let own = &hits[offset..offset + rays.len()];
            offset += rays.len();
            let in_range = |t: Real| if t <= spec.max_range { t } else { Real::INFINITY };
            let (distance, body) = match spec.kind {
                DistanceKind::Laser => (in_range(own[0].t), own[0].body),
                DistanceKind::Sonar => own
                    .iter()
                    .map(|h| (in_range(h.t), h.body))
                    .fold((Real::INFINITY, BodyId::NONE), |a, b| if b.0 < a.0 { b } else { a }),
                DistanceKind::InfraRed => {
                    let width = (spec.gaussian_width * spec.aperture).max(1e-9);
                    let (mut sum, mut weights) = (0.0, 0.0);
                    for (hit, ray) in own.iter().zip(rays) {
                        let w = (-(ray.angle_between(Vec3::X) / width).powi(2)).exp();
                        sum += w * in_range(hit.t).min(spec.max_range);
                        weights += w;
                    }
                    let mean = sum / weights;
                    (if mean < spec.max_range { mean } else { Real::INFINITY }, own[0].body)
                }
            };
            out.push(DistanceReading {
                distance,
                value: lookup(&spec.lookup, distance.min(spec.max_range)),
                body,
            });
        }
        Ok(out)
    }
}

/// A light the host places in the world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightSource {
    /// Parallel light (e.g. the sun): unit direction towards the light and
    /// irradiance on a surface facing it (W/m²).
    Directional { towards: Vec3, irradiance: Real },
    /// Point light: position and radiant intensity (W/sr).
    Point { position: Vec3, intensity: Real },
}

/// Lights plus diffuse sky irradiance on an upward surface (W/m²).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LightEnvironment {
    pub sources: Vec<LightSource>,
    pub sky: Real,
    pub up: Vec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LightReading {
    /// Total irradiance on the sensor's +X-facing surface (W/m²).
    pub irradiance: Real,
    pub direct: Real,
    pub sky: Real,
    /// Sources with a clear path to the sensor.
    pub visible: u32,
    /// `irradiance` through the sensor's lookup table.
    pub value: Real,
}

/// A batch of light sensors facing their frames' +X.
pub struct LightSensors {
    lookups: Vec<Vec<[Real; 2]>>,
    raycaster: Option<Raycaster>,
    capacity: usize,
}

/// How far a directional light's shadow ray looks for occluders (m).
const SHADOW_REACH: Real = 10_000.0;

impl LightSensors {
    /// Sensors that cast on the GPU sensor scene with [`Self::update`];
    /// `lookups` holds one irradiance-to-value table per sensor and
    /// `max_sources` bounds the lights per update.
    pub fn new(engine: &Engine, lookups: Vec<Vec<[Real; 2]>>, max_sources: usize) -> Result<Self> {
        let capacity = (lookups.len() * max_sources).max(1);
        Ok(Self {
            raycaster: Some(Raycaster::new(engine, capacity as u32)?),
            lookups,
            capacity,
        })
    }

    /// Sensors that cast through a host [`RayQuery`] with [`Self::update_with`].
    pub fn host(lookups: Vec<Vec<[Real; 2]>>) -> Self {
        Self {
            lookups,
            raycaster: None,
            capacity: usize::MAX,
        }
    }

    /// Readings on the GPU.
    pub fn update(
        &mut self,
        engine: &Engine,
        scene: &SensorScene,
        state: &State,
        poses: &[Transform],
        environment: &LightEnvironment,
    ) -> Result<Vec<LightReading>> {
        let mut raycaster = self.raycaster.take();
        let out = with_scene_rays(&mut raycaster, engine, scene, state, |rays| {
            self.update_with(rays, poses, environment)
        });
        self.raycaster = raycaster;
        out
    }

    /// Readings for sensors at `poses` (world, one per table) under `environment`.
    pub fn update_with(
        &self,
        query: &mut dyn RayQuery,
        poses: &[Transform],
        environment: &LightEnvironment,
    ) -> Result<Vec<LightReading>> {
        if poses.len() != self.lookups.len()
            || poses.len() * environment.sources.len() > self.capacity
        {
            return Err(Error::Build("light sensor: poses or sources exceed the batch".into()));
        }
        let (mut origins, mut directions, mut reach) = (Vec::new(), Vec::new(), Vec::new());
        for pose in poses {
            let normal = pose.rotation * Vec3::X;
            for source in &environment.sources {
                let (dir, distance) = match *source {
                    LightSource::Directional { towards, .. } => (towards.normalize(), SHADOW_REACH),
                    LightSource::Point { position, .. } => {
                        let offset = position - pose.position;
                        (offset.normalize_or_zero(), offset.length())
                    }
                };
                origins.push(pose.position + normal * 1e-3);
                directions.push(if dir == Vec3::ZERO { normal } else { dir });
                reach.push(distance);
            }
        }
        let hits = if origins.is_empty() {
            Vec::new()
        } else {
            query.cast(&origins, &directions, SHADOW_REACH)?
        };
        let up = environment.up.normalize_or_zero();
        let mut out = Vec::with_capacity(poses.len());
        for (index, (pose, table)) in poses.iter().zip(&self.lookups).enumerate() {
            let normal = pose.rotation * Vec3::X;
            let (mut direct, mut visible) = (0.0, 0u32);
            for (k, source) in environment.sources.iter().enumerate() {
                let ray = index * environment.sources.len() + k;
                if hits[ray].t < reach[ray] {
                    continue;
                }
                let cos = normal.dot(directions[ray]).max(0.0);
                direct += cos
                    * match *source {
                        LightSource::Directional { irradiance, .. } => irradiance,
                        LightSource::Point { intensity, .. } => intensity / reach[ray].max(1e-3).powi(2),
                    };
                visible += 1;
            }
            let sky = environment.sky * 0.5 * (1.0 + normal.dot(up));
            let irradiance = direct + sky;
            out.push(LightReading {
                irradiance,
                direct,
                sky,
                visible,
                value: lookup(table, irradiance),
            });
        }
        Ok(out)
    }
}

/// A radar looking along its frame's +X.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarSpec {
    pub min_range: Real,
    pub max_range: Real,
    pub horizontal_fov: Real,
    pub vertical_fov: Real,
    /// Targets slower than this (absolute radial speed, m/s) are dropped.
    pub min_radial_speed: Real,
    pub frequency_ghz: Real,
    pub transmitted_power_dbm: Real,
    /// Drop targets whose line of sight is blocked.
    pub occlusion: bool,
}

impl Default for RadarSpec {
    fn default() -> Self {
        Self {
            min_range: 1.0,
            max_range: 50.0,
            horizontal_fov: 0.78,
            vertical_fov: 0.1,
            min_radial_speed: 0.0,
            frequency_ghz: 24.0,
            transmitted_power_dbm: 1.0,
            occlusion: true,
        }
    }
}

/// A body a radar can detect and its radar cross-section (m²).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarTarget {
    pub body: BodyId,
    pub cross_section: Real,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadarDetection {
    pub body: BodyId,
    pub distance: Real,
    /// Angle left of +X in the sensor's XY plane (rad).
    pub azimuth: Real,
    /// Angle above the sensor's XY plane (rad).
    pub elevation: Real,
    /// Rate at which the target moves away (m/s).
    pub radial_speed: Real,
    pub received_power_dbm: Real,
}

/// Radar target lists against the owned scene.
pub struct Radars {
    raycaster: Option<Raycaster>,
    capacity: usize,
}

fn host_arrays(model: &Model, state: &State) -> Result<(Vec<Transform>, Vec<molla_math::SpatialVector>, Vec<Vec3>)> {
    Ok((
        state.body_q.host()?.to_vec(),
        state.body_qd.host()?.to_vec(),
        model.body_com.host()?.to_vec(),
    ))
}

impl Radars {
    /// Radars that cast on the GPU sensor scene with [`Self::update`];
    /// `max_targets` bounds the candidates per update.
    pub fn new(engine: &Engine, max_targets: usize) -> Result<Self> {
        Ok(Self {
            raycaster: Some(Raycaster::new(engine, max_targets.max(1) as u32)?),
            capacity: max_targets.max(1),
        })
    }

    /// Radars that cast through a host [`RayQuery`] with [`Self::update_with`].
    pub fn host() -> Self {
        Self {
            raycaster: None,
            capacity: usize::MAX,
        }
    }

    /// Detections on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        engine: &Engine,
        scene: &SensorScene,
        model: &Model,
        state: &State,
        spec: &RadarSpec,
        pose: Transform,
        mount: BodyId,
        targets: &[RadarTarget],
    ) -> Result<Vec<RadarDetection>> {
        let mut raycaster = self.raycaster.take();
        let out = with_scene_rays(&mut raycaster, engine, scene, state, |rays| {
            self.update_with(rays, model, state, spec, pose, mount, targets)
        });
        self.raycaster = raycaster;
        out
    }

    /// Detections of `targets` for a radar at `pose` riding on `mount`
    /// (`BodyId::NONE` for a fixed radar), nearest first. Reads body poses
    /// and velocities from host-resident `state` and `model`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_with(
        &self,
        query: &mut dyn RayQuery,
        model: &Model,
        state: &State,
        spec: &RadarSpec,
        pose: Transform,
        mount: BodyId,
        targets: &[RadarTarget],
    ) -> Result<Vec<RadarDetection>> {
        if targets.len() > self.capacity {
            return Err(Error::Build("radar: more targets than the batch holds".into()));
        }
        let (body_q, body_qd, com) = host_arrays(model, state)?;
        let point_velocity = |body: BodyId, at: Vec3| -> Vec3 {
            let i = body.index();
            match (body_q.get(i), body_qd.get(i), com.get(i)) {
                (Some(q), Some(qd), Some(c)) if body != BodyId::NONE => {
                    qd.linear + qd.angular.cross(at - q.transform_point(*c))
                }
                _ => Vec3::ZERO,
            }
        };
        let sensor_velocity = point_velocity(mount, pose.position);
        let inverse = pose.rotation.inverse();
        let lambda = 0.299_792_458 / spec.frequency_ghz.max(1e-6);
        let mut candidates = Vec::new();
        for target in targets {
            let i = target.body.index();
            let (Some(q), Some(c)) = (body_q.get(i), com.get(i)) else {
                continue;
            };
            let centre = q.transform_point(*c);
            let offset = centre - pose.position;
            let distance = offset.length();
            if !(spec.min_range..=spec.max_range).contains(&distance) || distance == 0.0 {
                continue;
            }
            let local = inverse * offset;
            let azimuth = local.y.atan2(local.x);
            let elevation = local.z.atan2(local.x.hypot(local.y));
            if azimuth.abs() > 0.5 * spec.horizontal_fov || elevation.abs() > 0.5 * spec.vertical_fov {
                continue;
            }
            let direction = offset / distance;
            let radial_speed = (point_velocity(target.body, centre) - sensor_velocity).dot(direction);
            if radial_speed.abs() < spec.min_radial_speed {
                continue;
            }
            let gain = lambda * lambda * target.cross_section.max(0.0)
                / ((4.0 * std::f64::consts::PI).powi(3) * distance.powi(4));
            candidates.push(RadarDetection {
                body: target.body,
                distance,
                azimuth,
                elevation,
                radial_speed,
                received_power_dbm: spec.transmitted_power_dbm + 10.0 * gain.max(1e-300).log10(),
            });
        }
        if spec.occlusion && !candidates.is_empty() {
            let origins = vec![pose.position; candidates.len()];
            let directions: Vec<Vec3> = candidates
                .iter()
                .map(|d| {
                    let i = d.body.index();
                    (body_q[i].transform_point(com[i]) - pose.position) / d.distance
                })
                .collect();
            let hits = query.cast(&origins, &directions, spec.max_range)?;
            let keep: Vec<bool> = candidates
                .iter()
                .zip(&hits)
                .map(|(d, h)| h.body == d.body || h.t >= d.distance - 1e-3)
                .collect();
            let mut kept = keep.into_iter();
            candidates.retain(|_| kept.next().unwrap_or(false));
        }
        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        Ok(candidates)
    }
}

/// Position and velocity of a one-degree-of-freedom joint (rad or m).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct JointReading {
    pub position: Real,
    pub velocity: Real,
}

/// Position sensor: the first coordinate and degree of freedom of `joint`
/// from host-resident `state`.
pub fn joint_reading(model: &Model, state: &State, joint: usize) -> Result<JointReading> {
    if joint >= model.joint_count {
        return Err(Error::Build(format!("position sensor: no joint {joint}")));
    }
    let coord = model.joint_coord_offset.host()?;
    let dof = model.joint_dof_offset.host()?;
    let (q, qd) = (state.joint_q.host()?, state.joint_qd.host()?);
    let has_coord = coord[joint + 1] > coord[joint];
    let has_dof = dof[joint + 1] > dof[joint];
    Ok(JointReading {
        position: if has_coord { q[coord[joint] as usize] } else { 0.0 },
        velocity: if has_dof { qd[dof[joint] as usize] } else { 0.0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wall slab at x = 5 and a ball at (10, 5, 0) moving along +X, cast
    /// on the CPU: the distance sensor ranges the wall, the wall shadows a
    /// low sun but not a high one, and the radar sees the ball past the wall.
    #[test]
    fn host_ray_queries_drive_the_sensor_models() {
        use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry};
        let mut scene = RigidScene::new().unwrap();
        let mut wall = ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::new(0.1, 2.0, 2.0),
        });
        wall.pose = Transform::from_translation(Vec3::new(5.1, 0.0, 0.0));
        scene.insert_collider(wall).unwrap();
        let mut body = BodyDesc::default();
        body.params.pose = Transform::from_translation(Vec3::new(10.0, 5.0, 0.0));
        body.velocity = molla_math::SpatialVector::new(Vec3::X, Vec3::ZERO);
        let handle = scene.insert_body(body).unwrap();
        let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
        ball.parent = Some(handle);
        scene.insert_collider(ball).unwrap();
        let ball = scene.body_index(handle).unwrap();
        let mut caster = CpuRaycaster::default();
        let mut rays = RigidRays {
            scene: &scene,
            caster: &mut caster,
        };
        let at = Transform::IDENTITY;

        let distance = DistanceSensors::host(vec![DistanceSpec::default()]).unwrap();
        let reading = distance.update_with(&mut rays, &[at]).unwrap()[0];
        assert!((reading.distance - 5.0).abs() < 1e-9 && reading.body == BodyId::NONE, "{reading:?}");

        let light = LightSensors::host(vec![Vec::new()]);
        let sun = |towards: Vec3| LightEnvironment {
            sources: vec![LightSource::Directional {
                towards: towards.normalize(),
                irradiance: 100.0,
            }],
            sky: 0.0,
            up: Vec3::Z,
        };
        let shaded = light.update_with(&mut rays, &[at], &sun(Vec3::new(1.0, 0.0, 0.1))).unwrap()[0];
        assert_eq!((shaded.visible, shaded.direct), (0, 0.0));
        let lit = light.update_with(&mut rays, &[at], &sun(Vec3::new(1.0, 0.0, 1.0))).unwrap()[0];
        assert!(lit.visible == 1 && (lit.direct - 100.0 * 0.5f64.sqrt()).abs() < 1e-9, "{lit:?}");

        let radar = RadarSpec {
            horizontal_fov: 1.2,
            ..Default::default()
        };
        let target = [RadarTarget {
            body: ball,
            cross_section: 1.0,
        }];
        let seen = Radars::host()
            .update_with(&mut rays, scene.model(), scene.state(), &radar, at, BodyId::NONE, &target)
            .unwrap();
        let d = 125f64.sqrt();
        assert_eq!(seen.len(), 1, "{seen:?}");
        assert!((seen[0].distance - d).abs() < 1e-9 && (seen[0].radial_speed - 10.0 / d).abs() < 1e-9);
    }

    #[test]
    fn lookup_interpolates_and_clamps() {
        let table = [[0.0, 1000.0], [0.1, 1000.0], [0.2, 400.0], [0.3, 50.0], [0.37, 30.0]];
        assert_eq!(lookup(&table, 0.05), 1000.0);
        assert!((lookup(&table, 0.15) - 700.0).abs() < 1e-9);
        assert_eq!(lookup(&table, 5.0), 30.0);
        assert_eq!(lookup(&table, Real::INFINITY), 30.0);
        assert_eq!(lookup(&[], 2.5), 2.5);
    }

    #[test]
    fn cone_rays_start_on_the_axis_and_ring_at_half_aperture() {
        let rays = cone_rays(5, 0.4);
        assert_eq!(rays.len(), 5);
        assert_eq!(rays[0], Vec3::X);
        for ray in &rays[1..] {
            assert!((ray.angle_between(Vec3::X) - 0.2).abs() < 1e-12);
            assert!((ray.length() - 1.0).abs() < 1e-12);
        }
        assert_eq!(cone_rays(1, 0.4), vec![Vec3::X]);
    }
}

/// Range finder (depth camera): a tiled camera that traces only depth,
/// metres along each pixel ray.
pub fn range_finder(
    width: u32,
    height: u32,
    fov_y: Real,
    max_range: Real,
) -> crate::tiled_camera::TiledCameraConfig {
    crate::tiled_camera::TiledCameraConfig {
        width,
        height,
        fov_y_radians: fov_y as f32,
        channels: crate::tiled_camera::CameraChannels::DEPTH,
        max_distance: max_range as f32,
        ..Default::default()
    }
}
