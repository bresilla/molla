//! Realistic spacecraft navigation sensor models & closed-loop fusion filter:
//!
//! - **6-DOF Space IMU** ([`ImuSpec`] / [`ImuReading`]): gyroscope Angle Random
//!   Walk (ARW), Rate/Bias Random Walk (RRW), deterministic bias, quantization,
//!   accelerometer specific force (zero in orbital free-fall!), lever-arm
//!   kinematics, sample rate, and transport latency.
//! - **Star Tracker** ([`StarTrackerSpec`] / [`StarTrackerReading`]):
//!   anisotropic cross-boresight vs boresight-roll attitude noise, update rate,
//!   FIFO transport latency, Sun exclusion blinding, and slew-rate blinding.
//! - **Rendezvous LiDAR** ([`LidarSpec`] / [`LidarReading`]): relative range,
//!   azimuth, elevation, range-rate, and close-range 6-DOF pose with
//!   range-dependent noise, FOV / range dropout, sample rate, and latency.
//! - **Complementary Navigation Filter** ([`NavigationFilter`]): fuses noisy,
//!   delayed IMU + Star Tracker + LiDAR measurements with online gyro-bias
//!   estimation so spacecraft GNC can close the loop on realistic hardware inputs.

use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::TAU;

use molla_core::{Error, Result};
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sim::runtime::{BodyHandle, RigidScene};

use super::{Orbital, RocketStructure, SpaceEnvironment};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImuId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StarTrackerId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LidarId(pub u64);

/// Deterministic SplitMix64 + Box–Muller Gaussian PRNG (reproducible across runs).
#[derive(Clone, Copy, Debug)]
struct SensorRng {
    state: u64,
    spare: Option<Real>,
}

impl SensorRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_uniform(&mut self) -> Real {
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) * (1.0 / ((1u64 << 53) as f64))
    }

    fn next_normal(&mut self) -> Real {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = self.next_uniform().max(1e-15);
        let u2 = self.next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }

    fn next_normal_vec3(&mut self) -> Vec3 {
        Vec3::new(self.next_normal(), self.next_normal(), self.next_normal())
    }
}

fn quantize(value: Real, step: Real) -> Real {
    if step > 0.0 {
        (value / step).round() * step
    } else {
        value
    }
}

fn quantize_vec3(v: Vec3, step: Real) -> Vec3 {
    Vec3::new(quantize(v.x, step), quantize(v.y, step), quantize(v.z, step))
}

/// 6-DOF spacecraft IMU specification (gyroscope + accelerometer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuSpec {
    pub body: BodyHandle,
    /// Mounting pose in body frame (`position` adds real lever-arm kinematics).
    pub mount_frame: Transform,
    /// Output sample rate (Hz); `0.0` samples every physics step.
    pub sample_rate_hz: Real,
    /// Measurement transport latency (s).
    pub latency_s: Real,
    /// Initial gyroscope bias in sensor frame (rad/s).
    pub gyro_bias_init: Vec3,
    /// Gyroscope Angle Random Walk white noise density `σ_arw` (`rad / √s`).
    pub gyro_arw: Real,
    /// Gyroscope Rate / Bias Random Walk density `σ_rrw` (`rad / s^(3/2)`).
    pub gyro_rrw: Real,
    /// Gyroscope ADC quantization step (rad/s); `0.0` disables quantization.
    pub gyro_quantization: Real,
    /// Initial accelerometer bias in sensor frame (m/s²).
    pub accel_bias_init: Vec3,
    /// Accelerometer Velocity Random Walk white noise density `σ_vrw` (`m / s^(3/2)`).
    pub accel_vrw: Real,
    /// Accelerometer Bias Random Walk density `σ_arrw` (`m / s^(5/2)`).
    pub accel_rrw: Real,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl ImuSpec {
    /// Tactical-grade fiber-optic spacecraft IMU (LN-200S / MIMU class).
    pub fn tactical_fog(body: BodyHandle, seed: u64) -> Self {
        Self {
            body,
            mount_frame: Transform::IDENTITY,
            sample_rate_hz: 100.0,
            latency_s: 0.005,
            gyro_bias_init: Vec3::new(4.8e-5, -3.5e-5, 2.9e-5), // ~10°/hr turn-on bias
            gyro_arw: 3.5e-5,                                    // ~0.12°/√hr ARW
            gyro_rrw: 5.0e-7,                                    // Bias instability walk
            gyro_quantization: 1.0e-6,
            accel_bias_init: Vec3::new(8.0e-4, -6.0e-4, 5.0e-4), // ~80 µg
            accel_vrw: 2.5e-4,
            accel_rrw: 5.0e-6,
            seed,
        }
    }

    /// Space-grade MEMS IMU with higher noise/bias for aggressive GNC stress testing.
    pub fn space_mems(body: BodyHandle, seed: u64) -> Self {
        Self {
            body,
            mount_frame: Transform::IDENTITY,
            sample_rate_hz: 100.0,
            latency_s: 0.010,
            gyro_bias_init: Vec3::new(3.0e-4, -2.5e-4, 1.8e-4), // ~60°/hr
            gyro_arw: 2.0e-4,
            gyro_rrw: 4.0e-6,
            gyro_quantization: 5.0e-6,
            accel_bias_init: Vec3::new(3.0e-3, -2.0e-3, 2.5e-3),
            accel_vrw: 1.5e-3,
            accel_rrw: 3.0e-5,
            seed,
        }
    }
}

/// Latest delivered 6-DOF IMU measurement (in sensor mount frame).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImuReading {
    /// Measured angular velocity in sensor frame (rad/s).
    pub angular_velocity: Vec3,
    /// Measured non-gravitational specific force in sensor frame (m/s²).
    pub specific_force: Vec3,
    /// Ground-truth internal gyro bias state (for telemetry / estimator verification).
    pub true_gyro_bias: Vec3,
    /// Ground-truth internal accelerometer bias state.
    pub true_accel_bias: Vec3,
    /// Simulation time (s) when the measurement was physically sampled.
    pub timestamp_s: Real,
    /// `true` on steps where a new sample popped out of the latency pipeline.
    pub fresh: bool,
}

/// Optical Star Tracker attitude sensor specification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarTrackerSpec {
    pub body: BodyHandle,
    /// Optical mounting frame (`+X` is optical boresight).
    pub mount_frame: Transform,
    /// Solution update rate (Hz, typically 4–10 Hz).
    pub sample_rate_hz: Real,
    /// Centroiding + catalog matching transport latency (s, typically 0.05–0.15 s).
    pub latency_s: Real,
    /// 1σ cross-boresight pitch/yaw noise (rad; e.g. 5 arcsec ≈ `2.42e-5 rad`).
    pub cross_axis_noise_rad: Real,
    /// 1σ boresight roll noise (rad; typically 5–10× cross-axis, e.g. 35 arcsec ≈ `1.7e-4 rad`).
    pub boresight_noise_rad: Real,
    /// Sun exclusion cone half-angle (rad; `0.0` disables sun blinding).
    pub sun_exclusion_rad: Real,
    /// Maximum trackable body angular rate (rad/s; `0.0` disables rate blinding).
    pub max_slew_rate_rad_s: Real,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl StarTrackerSpec {
    /// Typical LEO flight star tracker (Sodern Hydra / Ball CT-2020 class).
    pub fn leo_flight(body: BodyHandle, mount_frame: Transform, seed: u64) -> Self {
        Self {
            body,
            mount_frame,
            sample_rate_hz: 5.0,
            latency_s: 0.08,
            cross_axis_noise_rad: 5.0_f64.to_radians() / 3600.0, // 5 arcsec
            boresight_noise_rad: 35.0_f64.to_radians() / 3600.0, // 35 arcsec
            sun_exclusion_rad: 30.0_f64.to_radians(),
            max_slew_rate_rad_s: 0.25,
            seed,
        }
    }
}

/// Latest delivered Star Tracker attitude solution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarTrackerReading {
    /// Estimated body-to-world orientation quaternion `q_wb`.
    pub orientation: Quat,
    /// `true` when star solution is valid (neither sun-blinded nor rate-blinded).
    pub valid: bool,
    pub sun_blinded: bool,
    pub rate_blinded: bool,
    /// Simulation time (s) when the star exposure occurred.
    pub timestamp_s: Real,
    /// `true` on steps when a new solution emerged from the latency queue.
    pub fresh: bool,
}

impl Default for StarTrackerReading {
    fn default() -> Self {
        Self {
            orientation: Quat::IDENTITY,
            valid: false,
            sun_blinded: false,
            rate_blinded: false,
            timestamp_s: 0.0,
            fresh: false,
        }
    }
}

/// Rendezvous scanning LiDAR / optical relative navigation sensor specification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LidarSpec {
    /// Host (chaser) body carrying the LiDAR.
    pub body: BodyHandle,
    /// Tracked target (station) body.
    pub target_body: BodyHandle,
    /// Sensor mount frame on host body (`+X` is optical boresight).
    pub mount_frame: Transform,
    /// Tracked retroreflector / docking-port target point in `target_body` local frame (m).
    pub target_point_local: Vec3,
    /// Output update rate (Hz, typically 5–20 Hz).
    pub sample_rate_hz: Real,
    /// Point-cloud / ICP processing latency (s, typically 0.02–0.08 s).
    pub latency_s: Real,
    /// Minimum valid range (m).
    pub min_range_m: Real,
    /// Maximum valid range (m).
    pub max_range_m: Real,
    /// Conical field-of-view half-angle around boresight `+X` (rad).
    pub fov_half_angle_rad: Real,
    /// Fixed 1σ range noise floor (m).
    pub range_noise_m: Real,
    /// Range-proportional 1σ noise (`m / m`, e.g. `5e-4` = 0.05% of range).
    pub range_noise_scale: Real,
    /// 1σ azimuth & elevation bearing noise (rad).
    pub bearing_noise_rad: Real,
    /// 1σ Doppler / differentiated range-rate noise (m/s).
    pub range_rate_noise_mps: Real,
    /// 1σ 6-DOF ICP relative-attitude noise (rad).
    pub pose_att_noise_rad: Real,
    /// Deterministic PRNG seed.
    pub seed: u64,
}

impl LidarSpec {
    /// DragonEye / TriDAR-class rendezvous proximity LiDAR.
    pub fn rendezvous_proximity(
        body: BodyHandle,
        target_body: BodyHandle,
        mount_frame: Transform,
        target_point_local: Vec3,
        seed: u64,
    ) -> Self {
        Self {
            body,
            target_body,
            mount_frame,
            target_point_local,
            sample_rate_hz: 10.0,
            latency_s: 0.04,
            min_range_m: 0.05,
            max_range_m: 150.0,
            fov_half_angle_rad: 55.0_f64.to_radians(),
            range_noise_m: 0.004,
            range_noise_scale: 4.0e-4,
            bearing_noise_rad: 0.0008, // ~0.046°
            range_rate_noise_mps: 0.006,
            pose_att_noise_rad: 0.0035, // ~0.20°
            seed,
        }
    }
}

/// Latest delivered rendezvous LiDAR measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LidarReading {
    /// Measured target range `ρ` (m).
    pub range_m: Real,
    /// Measured target azimuth `atan2(y_s, x_s)` in sensor frame (rad).
    pub azimuth_rad: Real,
    /// Measured target elevation `atan2(z_s, hypot(x_s, y_s))` in sensor frame (rad).
    pub elevation_rad: Real,
    /// Measured line-of-sight range rate `ρ̇` (m/s).
    pub range_rate_mps: Real,
    /// Reconstructed noisy target point position in sensor frame (m).
    pub relative_position_sensor: Vec3,
    /// Reconstructed noisy target orientation relative to host body (`q_host⁻¹ q_target`).
    pub relative_orientation: Quat,
    /// `true` when target is within `[min_range_m, max_range_m]` and FOV cone.
    pub valid: bool,
    /// Simulation time (s) when the LiDAR scan occurred.
    pub timestamp_s: Real,
    /// `true` on steps when a new scan emerged from the latency queue.
    pub fresh: bool,
}

impl Default for LidarReading {
    fn default() -> Self {
        Self {
            range_m: 0.0,
            azimuth_rad: 0.0,
            elevation_rad: 0.0,
            range_rate_mps: 0.0,
            relative_position_sensor: Vec3::ZERO,
            relative_orientation: Quat::IDENTITY,
            valid: false,
            timestamp_s: 0.0,
            fresh: false,
        }
    }
}

struct ImuState {
    spec: ImuSpec,
    rng: SensorRng,
    gyro_bias: Vec3,
    accel_bias: Vec3,
    prev_vel_world: Option<Vec3>,
    prev_omega_world: Option<Vec3>,
    accum_s: Real,
    queue: VecDeque<(Real, ImuReading)>,
    reading: ImuReading,
}

struct StarTrackerState {
    spec: StarTrackerSpec,
    rng: SensorRng,
    accum_s: Real,
    queue: VecDeque<(Real, StarTrackerReading)>,
    reading: StarTrackerReading,
}

struct LidarState {
    spec: LidarSpec,
    rng: SensorRng,
    accum_s: Real,
    queue: VecDeque<(Real, LidarReading)>,
    reading: LidarReading,
}

/// Hardware-fidelity spacecraft sensor suite stepped automatically by [`RigidWorld`].
#[derive(Default)]
pub struct SpaceSensors {
    next_id: u64,
    time_s: Real,
    imus: BTreeMap<ImuId, ImuState>,
    star_trackers: BTreeMap<StarTrackerId, StarTrackerState>,
    lidars: BTreeMap<LidarId, LidarState>,
}

impl SpaceSensors {
    pub(super) fn is_empty(&self) -> bool {
        self.imus.is_empty() && self.star_trackers.is_empty() && self.lidars.is_empty()
    }

    pub fn sim_time_s(&self) -> Real {
        self.time_s
    }

    pub fn insert_imu(&mut self, scene: &RigidScene, spec: ImuSpec) -> Result<ImuId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("IMU body is missing".into()));
        }
        if !spec.sample_rate_hz.is_finite()
            || spec.sample_rate_hz < 0.0
            || !spec.latency_s.is_finite()
            || spec.latency_s < 0.0
            || !spec.gyro_bias_init.is_finite()
            || !spec.accel_bias_init.is_finite()
            || !spec.gyro_arw.is_finite()
            || spec.gyro_arw < 0.0
            || !spec.gyro_rrw.is_finite()
            || spec.gyro_rrw < 0.0
            || !spec.gyro_quantization.is_finite()
            || spec.gyro_quantization < 0.0
            || !spec.accel_vrw.is_finite()
            || spec.accel_vrw < 0.0
            || !spec.accel_rrw.is_finite()
            || spec.accel_rrw < 0.0
        {
            return Err(Error::Build("invalid IMU specification".into()));
        }
        self.next_id += 1;
        let id = ImuId(self.next_id);
        let init_vel = scene.body_velocity(spec.body).ok();
        self.imus.insert(
            id,
            ImuState {
                spec,
                rng: SensorRng::new(spec.seed),
                gyro_bias: spec.gyro_bias_init,
                accel_bias: spec.accel_bias_init,
                prev_vel_world: init_vel.map(|v| v.linear),
                prev_omega_world: init_vel.map(|v| v.angular),
                accum_s: if spec.sample_rate_hz > 0.0 {
                    1.0 / spec.sample_rate_hz
                } else {
                    0.0
                },
                queue: VecDeque::new(),
                reading: ImuReading {
                    true_gyro_bias: spec.gyro_bias_init,
                    true_accel_bias: spec.accel_bias_init,
                    ..ImuReading::default()
                },
            },
        );
        Ok(id)
    }

    pub fn insert_star_tracker(
        &mut self,
        scene: &RigidScene,
        spec: StarTrackerSpec,
    ) -> Result<StarTrackerId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("star tracker body is missing".into()));
        }
        if !spec.sample_rate_hz.is_finite()
            || spec.sample_rate_hz <= 0.0
            || !spec.latency_s.is_finite()
            || spec.latency_s < 0.0
            || !spec.cross_axis_noise_rad.is_finite()
            || spec.cross_axis_noise_rad < 0.0
            || !spec.boresight_noise_rad.is_finite()
            || spec.boresight_noise_rad < 0.0
            || !spec.sun_exclusion_rad.is_finite()
            || spec.sun_exclusion_rad < 0.0
            || !spec.max_slew_rate_rad_s.is_finite()
            || spec.max_slew_rate_rad_s < 0.0
        {
            return Err(Error::Build("invalid star tracker specification".into()));
        }
        self.next_id += 1;
        let id = StarTrackerId(self.next_id);
        let init_q = scene
            .body_pose(spec.body)
            .map(|p| p.rotation)
            .unwrap_or(Quat::IDENTITY);
        self.star_trackers.insert(
            id,
            StarTrackerState {
                spec,
                rng: SensorRng::new(spec.seed),
                accum_s: 1.0 / spec.sample_rate_hz,
                queue: VecDeque::new(),
                reading: StarTrackerReading {
                    orientation: init_q,
                    valid: true,
                    ..StarTrackerReading::default()
                },
            },
        );
        Ok(id)
    }

    pub fn insert_lidar(&mut self, scene: &RigidScene, spec: LidarSpec) -> Result<LidarId> {
        if scene.body(spec.body).is_none() || scene.body(spec.target_body).is_none() {
            return Err(Error::Build("LiDAR host or target body is missing".into()));
        }
        if !spec.sample_rate_hz.is_finite()
            || spec.sample_rate_hz <= 0.0
            || !spec.latency_s.is_finite()
            || spec.latency_s < 0.0
            || !spec.min_range_m.is_finite()
            || spec.min_range_m < 0.0
            || !spec.max_range_m.is_finite()
            || spec.max_range_m <= spec.min_range_m
            || !spec.fov_half_angle_rad.is_finite()
            || spec.fov_half_angle_rad <= 0.0
            || !spec.range_noise_m.is_finite()
            || spec.range_noise_m < 0.0
            || !spec.range_noise_scale.is_finite()
            || spec.range_noise_scale < 0.0
            || !spec.bearing_noise_rad.is_finite()
            || spec.bearing_noise_rad < 0.0
            || !spec.range_rate_noise_mps.is_finite()
            || spec.range_rate_noise_mps < 0.0
            || !spec.pose_att_noise_rad.is_finite()
            || spec.pose_att_noise_rad < 0.0
        {
            return Err(Error::Build("invalid LiDAR specification".into()));
        }
        self.next_id += 1;
        let id = LidarId(self.next_id);
        self.lidars.insert(
            id,
            LidarState {
                spec,
                rng: SensorRng::new(spec.seed),
                accum_s: 1.0 / spec.sample_rate_hz,
                queue: VecDeque::new(),
                reading: LidarReading::default(),
            },
        );
        Ok(id)
    }

    pub fn imu_reading(&self, id: ImuId) -> Option<ImuReading> {
        self.imus.get(&id).map(|s| s.reading)
    }

    pub fn star_tracker_reading(&self, id: StarTrackerId) -> Option<StarTrackerReading> {
        self.star_trackers.get(&id).map(|s| s.reading)
    }

    pub fn lidar_reading(&self, id: LidarId) -> Option<LidarReading> {
        self.lidars.get(&id).map(|s| s.reading)
    }

    pub(crate) fn after_step(
        &mut self,
        scene: &RigidScene,
        orbital: &Orbital,
        space_env: &SpaceEnvironment,
        rocket_structure: &RocketStructure,
        dt: Real,
    ) -> Result<()> {
        if self.is_empty() || !(dt > 0.0) {
            return Ok(());
        }
        self.time_s += dt;
        let now = self.time_s;
        let gravity_world = scene.gravity();
        let sun_dir_world = space_env.sun_direction();

        // 1. Update IMUs
        for imu in self.imus.values_mut() {
            imu.reading.fresh = false;
            let (Ok(pose), Ok(vel), Ok(mass_props)) = (
                scene.body_pose(imu.spec.body),
                scene.body_velocity(imu.spec.body),
                scene.body_mass_properties(imu.spec.body),
            ) else {
                continue;
            };
            // Bias random walks evolve continuously with sqrt(dt).
            let sqrt_dt = dt.sqrt();
            if imu.spec.gyro_rrw > 0.0 {
                imu.gyro_bias += imu.rng.next_normal_vec3() * (imu.spec.gyro_rrw * sqrt_dt);
            }
            if imu.spec.accel_rrw > 0.0 {
                imu.accel_bias += imu.rng.next_normal_vec3() * (imu.spec.accel_rrw * sqrt_dt);
            }

            let a_com_world = imu
                .prev_vel_world
                .map_or(Vec3::ZERO, |v0| (vel.linear - v0) / dt);
            let alpha_world = imu
                .prev_omega_world
                .map_or(Vec3::ZERO, |w0| (vel.angular - w0) / dt);
            imu.prev_vel_world = Some(vel.linear);
            imu.prev_omega_world = Some(vel.angular);

            let period = if imu.spec.sample_rate_hz > 0.0 {
                1.0 / imu.spec.sample_rate_hz
            } else {
                0.0
            };
            imu.accum_s += dt;
            if period == 0.0 || imu.accum_s + 1e-12 >= period {
                let sample_dt = if period > 0.0 {
                    imu.accum_s = (imu.accum_s - period).max(0.0).min(period);
                    period
                } else {
                    dt
                };
                let sensor_pose = pose.compose(&imu.spec.mount_frame);
                let com_world = pose.transform_point(mass_props.local_com);
                let lever = sensor_pose.position - com_world;
                let a_sensor_world = a_com_world
                    + alpha_world.cross(lever)
                    + vel.angular.cross(vel.angular.cross(lever));
                let orb_acc = orbital
                    .reading(imu.spec.body)
                    .map_or(Vec3::ZERO, |r| r.acceleration);
                // Specific force = non-gravitational acceleration (zero in pure free-fall orbit!).
                let spec_force_world = a_sensor_world - gravity_world - orb_acc;
                let inv_rot = sensor_pose.rotation.conjugate();
                // Automatically include physical elastic fuselage bending feedthrough when mounted
                // on a flexible stage registered in `RocketStructure` (NASA SP-8036 CSI!).
                let flex = rocket_structure.reading_for_body(imu.spec.body);
                let inv_mount = imu.spec.mount_frame.rotation.conjugate();
                let flex_omega_s =
                    flex.map_or(Vec3::ZERO, |r| inv_mount * r.imu_bending_rate_body);
                let flex_acc_s =
                    flex.map_or(Vec3::ZERO, |r| inv_mount * r.imu_bending_accel_body);
                let true_omega_s = inv_rot * vel.angular + flex_omega_s;
                let true_sf_s = inv_rot * spec_force_world + flex_acc_s;

                let inv_sqrt_dt = 1.0 / sample_dt.max(1e-9).sqrt();
                let gyro_noise = imu.rng.next_normal_vec3() * (imu.spec.gyro_arw * inv_sqrt_dt);
                let accel_noise = imu.rng.next_normal_vec3() * (imu.spec.accel_vrw * inv_sqrt_dt);

                let meas_omega = quantize_vec3(
                    true_omega_s + imu.gyro_bias + gyro_noise,
                    imu.spec.gyro_quantization,
                );
                let meas_sf = true_sf_s + imu.accel_bias + accel_noise;

                let sample = ImuReading {
                    angular_velocity: meas_omega,
                    specific_force: meas_sf,
                    true_gyro_bias: imu.gyro_bias,
                    true_accel_bias: imu.accel_bias,
                    timestamp_s: now,
                    fresh: true,
                };
                imu.queue.push_back((now + imu.spec.latency_s, sample));
            }
            while imu.queue.front().is_some_and(|(t, _)| *t <= now + 1e-12) {
                if let Some((_, s)) = imu.queue.pop_front() {
                    imu.reading = s;
                }
            }
        }

        // 2. Update Star Trackers
        for st in self.star_trackers.values_mut() {
            st.reading.fresh = false;
            let (Ok(pose), Ok(vel)) = (scene.body_pose(st.spec.body), scene.body_velocity(st.spec.body))
            else {
                continue;
            };
            let period = 1.0 / st.spec.sample_rate_hz;
            st.accum_s += dt;
            if st.accum_s + 1e-12 >= period {
                st.accum_s = (st.accum_s - period).max(0.0).min(period);
                let sensor_pose = pose.compose(&st.spec.mount_frame);
                let boresight_w = (sensor_pose.rotation * Vec3::X).normalize();
                let sun_blinded = st.spec.sun_exclusion_rad > 0.0
                    && sun_dir_world.is_some_and(|sun| {
                        boresight_w.dot(sun).clamp(-1.0, 1.0).acos() < st.spec.sun_exclusion_rad
                    });
                let rate_blinded = st.spec.max_slew_rate_rad_s > 0.0
                    && vel.angular.length() > st.spec.max_slew_rate_rad_s;
                let valid = !sun_blinded && !rate_blinded;

                let orientation = if valid {
                    // Anisotropic noise in sensor frame: boresight = X, cross-axes = Y, Z.
                    let d_theta_sensor = Vec3::new(
                        st.rng.next_normal() * st.spec.boresight_noise_rad,
                        st.rng.next_normal() * st.spec.cross_axis_noise_rad,
                        st.rng.next_normal() * st.spec.cross_axis_noise_rad,
                    );
                    let ang = d_theta_sensor.length();
                    let dq_sensor = if ang > 1e-14 {
                        Quat::from_axis_angle(d_theta_sensor / ang, ang)
                    } else {
                        Quat::IDENTITY
                    };
                    let noisy_sensor_rot = (sensor_pose.rotation * dq_sensor).normalize();
                    (noisy_sensor_rot * st.spec.mount_frame.rotation.conjugate()).normalize()
                } else {
                    st.reading.orientation
                };

                let sample = StarTrackerReading {
                    orientation,
                    valid,
                    sun_blinded,
                    rate_blinded,
                    timestamp_s: now,
                    fresh: true,
                };
                st.queue.push_back((now + st.spec.latency_s, sample));
            }
            while st.queue.front().is_some_and(|(t, _)| *t <= now + 1e-12) {
                if let Some((_, s)) = st.queue.pop_front() {
                    st.reading = s;
                }
            }
        }

        // 3. Update Rendezvous LiDARs
        for lidar in self.lidars.values_mut() {
            lidar.reading.fresh = false;
            let (Ok(host_pose), Ok(host_vel), Ok(host_mp), Ok(tgt_pose), Ok(tgt_vel), Ok(tgt_mp)) = (
                scene.body_pose(lidar.spec.body),
                scene.body_velocity(lidar.spec.body),
                scene.body_mass_properties(lidar.spec.body),
                scene.body_pose(lidar.spec.target_body),
                scene.body_velocity(lidar.spec.target_body),
                scene.body_mass_properties(lidar.spec.target_body),
            ) else {
                continue;
            };
            let period = 1.0 / lidar.spec.sample_rate_hz;
            lidar.accum_s += dt;
            if lidar.accum_s + 1e-12 >= period {
                lidar.accum_s = (lidar.accum_s - period).max(0.0).min(period);
                let sensor_pose = host_pose.compose(&lidar.spec.mount_frame);
                let tgt_pt_w = tgt_pose.transform_point(lidar.spec.target_point_local);
                let rel_w = tgt_pt_w - sensor_pose.position;
                let range_true = rel_w.length();
                let rel_sensor = sensor_pose.rotation.conjugate() * rel_w;
                let off_bore = if range_true > 1e-9 {
                    (rel_sensor.x / range_true).clamp(-1.0, 1.0).acos()
                } else {
                    0.0
                };
                let valid = range_true >= lidar.spec.min_range_m
                    && range_true <= lidar.spec.max_range_m
                    && off_bore <= lidar.spec.fov_half_angle_rad;

                let sample = if valid {
                    let host_com_w = host_pose.transform_point(host_mp.local_com);
                    let tgt_com_w = tgt_pose.transform_point(tgt_mp.local_com);
                    let v_sensor_w =
                        host_vel.linear + host_vel.angular.cross(sensor_pose.position - host_com_w);
                    let v_tgt_w =
                        tgt_vel.linear + tgt_vel.angular.cross(tgt_pt_w - tgt_com_w);
                    let los_w = rel_w / range_true;
                    let range_rate_true = (v_tgt_w - v_sensor_w).dot(los_w);

                    let az_true = rel_sensor.y.atan2(rel_sensor.x);
                    let el_true = rel_sensor
                        .z
                        .atan2((rel_sensor.x * rel_sensor.x + rel_sensor.y * rel_sensor.y).sqrt());

                    let sigma_r =
                        lidar.spec.range_noise_m + lidar.spec.range_noise_scale * range_true;
                    let range_m = (range_true + lidar.rng.next_normal() * sigma_r).max(0.01);
                    let azimuth_rad =
                        az_true + lidar.rng.next_normal() * lidar.spec.bearing_noise_rad;
                    let elevation_rad =
                        el_true + lidar.rng.next_normal() * lidar.spec.bearing_noise_rad;
                    let range_rate_mps = range_rate_true
                        + lidar.rng.next_normal() * lidar.spec.range_rate_noise_mps;

                    let cos_el = elevation_rad.cos();
                    let relative_position_sensor = Vec3::new(
                        range_m * cos_el * azimuth_rad.cos(),
                        range_m * cos_el * azimuth_rad.sin(),
                        range_m * elevation_rad.sin(),
                    );
                    let d_att = lidar.rng.next_normal_vec3() * lidar.spec.pose_att_noise_rad;
                    let d_ang = d_att.length();
                    let dq = if d_ang > 1e-14 {
                        Quat::from_axis_angle(d_att / d_ang, d_ang)
                    } else {
                        Quat::IDENTITY
                    };
                    let true_rel_q = (host_pose.rotation.conjugate() * tgt_pose.rotation).normalize();
                    let relative_orientation = (true_rel_q * dq).normalize();

                    LidarReading {
                        range_m,
                        azimuth_rad,
                        elevation_rad,
                        range_rate_mps,
                        relative_position_sensor,
                        relative_orientation,
                        valid: true,
                        timestamp_s: now,
                        fresh: true,
                    }
                } else {
                    LidarReading {
                        valid: false,
                        timestamp_s: now,
                        fresh: true,
                        ..lidar.reading
                    }
                };
                lidar.queue.push_back((now + lidar.spec.latency_s, sample));
            }
            while lidar.queue.front().is_some_and(|(t, _)| *t <= now + 1e-12) {
                if let Some((_, s)) = lidar.queue.pop_front() {
                    lidar.reading = s;
                }
            }
        }
        Ok(())
    }
}

/// Reusable nonlinear complementary attitude + gyro-bias + relative-pose
/// navigation filter fusing noisy, delayed IMU, Star Tracker, and LiDAR data.
#[derive(Clone, Copy, Debug)]
pub struct NavigationFilter {
    /// Estimated host body orientation in world frame.
    pub orientation: Quat,
    /// Estimated host body angular velocity in body frame (bias-corrected).
    pub omega_body: Vec3,
    /// Online estimated gyroscope bias in body frame (rad/s).
    pub gyro_bias_est: Vec3,
    /// Estimated target-port position in host body frame (m).
    pub rel_target_pos_body: Vec3,
    /// Estimated target-port velocity in host body frame (m/s).
    pub rel_target_vel_body: Vec3,
    /// Estimated relative target orientation (`q_host⁻¹ q_target`).
    pub rel_target_rot: Quat,
    /// Proportional attitude correction gain (1/s) on star tracker updates.
    pub kp_att: Real,
    /// Integral gyro-bias estimation gain (1/s²) on star tracker updates.
    pub ki_bias: Real,
    /// Position innovation gain `α` on LiDAR updates.
    pub alpha_pos: Real,
    /// Velocity innovation gain `β` on LiDAR updates.
    pub beta_vel: Real,
    /// Relative-attitude blending gain on LiDAR updates.
    pub alpha_rel_att: Real,
    /// `true` once initialized with a valid LiDAR measurement.
    pub lidar_locked: bool,
    lidar_samples: u32,
    elapsed_s: Real,
    last_st_t: Real,
    last_lidar_t: Real,
}

impl NavigationFilter {
    pub fn new(initial_orientation: Quat) -> Self {
        Self {
            orientation: initial_orientation.normalize(),
            omega_body: Vec3::ZERO,
            gyro_bias_est: Vec3::ZERO,
            rel_target_pos_body: Vec3::ZERO,
            rel_target_vel_body: Vec3::ZERO,
            rel_target_rot: Quat::IDENTITY,
            kp_att: 2.8,
            ki_bias: 1.4,
            alpha_pos: 0.55,
            beta_vel: 0.22,
            alpha_rel_att: 0.50,
            lidar_locked: false,
            lidar_samples: 0,
            elapsed_s: 0.0,
            last_st_t: -1.0,
            last_lidar_t: -1.0,
        }
    }

    /// Propagate attitude with IMU gyro (`imu` mounted at identity or rotated into body frame)
    /// and fuse Star Tracker + LiDAR updates whenever fresh valid samples arrive.
    pub fn update(
        &mut self,
        dt: Real,
        imu: &ImuReading,
        star_tracker: Option<&StarTrackerReading>,
        lidar: Option<&LidarReading>,
        lidar_mount: Transform,
    ) {
        if !(dt > 0.0) {
            return;
        }
        self.elapsed_s += dt;
        // 1. High-rate gyro prediction with online bias subtraction.
        let omega_corr = imu.angular_velocity - self.gyro_bias_est;
        self.omega_body = omega_corr;
        let dtheta = omega_corr * dt;
        let ang = dtheta.length();
        if ang > 1e-14 {
            let dq = Quat::from_axis_angle(dtheta / ang, ang);
            self.orientation = (self.orientation * dq).normalize();
        }

        // 2. Star Tracker multiplicative quaternion innovation (with transport latency
        //    compensation via gyro rate!) + gyro bias integral update.
        if let Some(st) = star_tracker
            && st.fresh
            && st.valid
        {
            let dt_st = if self.last_st_t >= 0.0 {
                (st.timestamp_s - self.last_st_t).clamp(0.02, 0.5)
            } else {
                0.1
            };
            self.last_st_t = st.timestamp_s;
            let st_lat = (self.elapsed_s - st.timestamp_s).clamp(0.0, 0.25);
            let dtheta_lat = self.omega_body * st_lat;
            let lat_ang = dtheta_lat.length();
            let st_q_now = if lat_ang > 1e-12 {
                (st.orientation * Quat::from_axis_angle(dtheta_lat / lat_ang, lat_ang)).normalize()
            } else {
                st.orientation
            };
            let mut q_err = (self.orientation.conjugate() * st_q_now).normalize();
            if q_err.w < 0.0 {
                q_err = -q_err;
            }
            let v = Vec3::new(q_err.x, q_err.y, q_err.z);
            let s = v.length();
            let err_body = if s > 1e-12 {
                v * (2.0 * s.atan2(q_err.w) / s)
            } else {
                v * 2.0
            };
            // Attitude pull + gyro-bias learning (negative sign because positive bias
            // makes predicted attitude run ahead of star tracker).
            let corr = err_body * (self.kp_att * dt_st).clamp(0.0, 0.85);
            let c_ang = corr.length();
            if c_ang > 1e-14 {
                self.orientation =
                    (self.orientation * Quat::from_axis_angle(corr / c_ang, c_ang)).normalize();
            }
            self.gyro_bias_est -= err_body * (self.ki_bias * dt_st);
            self.omega_body = imu.angular_velocity - self.gyro_bias_est;
        }

        // 3. Relative-pose prediction in rotating host body frame (`ṗ_b = v_b - ω × p_b`).
        if self.lidar_locked {
            self.rel_target_pos_body +=
                (self.rel_target_vel_body - self.omega_body.cross(self.rel_target_pos_body)) * dt;
        }

        // 4. LiDAR α–β relative position/velocity + relative attitude update
        //    (cleanly drops lock on FOV/range loss and fast-seeds velocity on re-acquisition).
        if let Some(ld) = lidar
            && ld.fresh
        {
            if !ld.valid {
                self.lidar_locked = false;
                self.lidar_samples = 0;
            } else {
                let meas_pos_body = lidar_mount.transform_point(ld.relative_position_sensor);
                if !self.lidar_locked {
                    self.rel_target_pos_body = meas_pos_body;
                    self.rel_target_vel_body = Vec3::ZERO;
                    self.rel_target_rot = ld.relative_orientation;
                    self.lidar_locked = true;
                    self.lidar_samples = 1;
                    self.last_lidar_t = ld.timestamp_s;
                } else {
                    let dt_ld = (ld.timestamp_s - self.last_lidar_t).max(0.02);
                    self.last_lidar_t = ld.timestamp_s;
                    if self.lidar_samples == 1 {
                        // Fast two-point velocity seeding on re-acquisition (avoids slow ramp-up!).
                        let p_dot_b = (meas_pos_body - self.rel_target_pos_body) / dt_ld;
                        self.rel_target_vel_body =
                            p_dot_b + self.omega_body.cross(meas_pos_body);
                        self.rel_target_pos_body = meas_pos_body;
                    } else {
                        let resid = meas_pos_body - self.rel_target_pos_body;
                        self.rel_target_pos_body += resid * self.alpha_pos;
                        self.rel_target_vel_body += resid * (self.beta_vel / dt_ld);
                    }
                    self.lidar_samples = self.lidar_samples.saturating_add(1);
                    let mut q_target = ld.relative_orientation;
                    if self.rel_target_rot.dot(q_target) < 0.0 {
                        q_target = -q_target;
                    }
                    self.rel_target_rot = self
                        .rel_target_rot
                        .slerp(q_target, self.alpha_rel_att)
                        .normalize();
                }
            }
        }
    }
}
