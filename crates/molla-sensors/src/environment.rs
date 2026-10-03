//! Sensors that read the world's orientation and the planet: GPS, compass
//! and inertial unit.
//!
//! Molla has no notion of north, up or the planet; the host describes them
//! with a [`WorldFrame`] (which world axes point up and north) and, for GPS,
//! a [`GeoReference`] (where the world origin sits on the WGS84 ellipsoid).
//! Every model here is a pure function of a sensor's world pose, so any
//! physics backend that can report poses can drive them.

use molla_math::{Mat3, Quat, Real, Vec3};

/// Which world axes point up and towards geographic north. East completes
/// the right-handed east-north-up basis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldFrame {
    pub up: Vec3,
    pub north: Vec3,
}

impl WorldFrame {
    /// Z-up worlds with +Y north (ROS REP-103 ENU).
    pub const Z_UP_Y_NORTH: Self = Self {
        up: Vec3::Z,
        north: Vec3::Y,
    };
    /// Y-up worlds with -Z north (Bevy/glTF convention: +X east).
    pub const Y_UP_NEG_Z_NORTH: Self = Self {
        up: Vec3::Y,
        north: Vec3::NEG_Z,
    };

    /// Orthonormal east, north and up, with north made perpendicular to up.
    pub fn enu(&self) -> [Vec3; 3] {
        let up = self.up.normalize();
        let north = (self.north - up * self.north.dot(up)).normalize();
        [north.cross(up), north, up]
    }

    /// Components of a world vector along east, north and up.
    pub fn to_enu(&self, world: Vec3) -> Vec3 {
        let [e, n, u] = self.enu();
        Vec3::new(world.dot(e), world.dot(n), world.dot(u))
    }

    /// Rotation taking east-north-up components to world axes.
    pub fn world_from_enu(&self) -> Quat {
        let [e, n, u] = self.enu();
        Quat::from_mat3(&Mat3::from_cols(e, n, u))
    }
}

/// A point on the WGS84 ellipsoid: degrees and metres above the ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Geodetic {
    pub latitude_deg: Real,
    pub longitude_deg: Real,
    pub altitude_m: Real,
}

/// The world origin on the planet: world positions are east-north-up
/// offsets in the local tangent plane at this point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GeoReference {
    pub origin: Geodetic,
}

const WGS84_A: Real = 6_378_137.0;
const WGS84_F: Real = 1.0 / 298.257_223_563;
const WGS84_E2: Real = WGS84_F * (2.0 - WGS84_F);

/// Earth-centred, earth-fixed coordinates of a geodetic point.
pub fn geodetic_to_ecef(point: Geodetic) -> Vec3 {
    let (lat, lon) = (
        point.latitude_deg.to_radians(),
        point.longitude_deg.to_radians(),
    );
    let n = WGS84_A / (1.0 - WGS84_E2 * lat.sin().powi(2)).sqrt();
    Vec3::new(
        (n + point.altitude_m) * lat.cos() * lon.cos(),
        (n + point.altitude_m) * lat.cos() * lon.sin(),
        (n * (1.0 - WGS84_E2) + point.altitude_m) * lat.sin(),
    )
}

/// Geodetic point of earth-centred coordinates (Bowring's method, refined).
pub fn ecef_to_geodetic(ecef: Vec3) -> Geodetic {
    let p = (ecef.x * ecef.x + ecef.y * ecef.y).sqrt();
    let lon = ecef.y.atan2(ecef.x);
    let mut lat = ecef.z.atan2(p * (1.0 - WGS84_E2));
    let mut altitude = 0.0;
    for _ in 0..5 {
        let n = WGS84_A / (1.0 - WGS84_E2 * lat.sin().powi(2)).sqrt();
        altitude = if lat.cos().abs() > 1e-12 {
            p / lat.cos() - n
        } else {
            ecef.z.abs() - n * (1.0 - WGS84_E2)
        };
        lat = ecef.z.atan2(p * (1.0 - WGS84_E2 * n / (n + altitude)));
    }
    Geodetic {
        latitude_deg: lat.to_degrees(),
        longitude_deg: lon.to_degrees(),
        altitude_m: altitude,
    }
}

impl GeoReference {
    /// Geodetic point of an east-north-up offset from the origin.
    pub fn enu_to_geodetic(&self, enu: Vec3) -> Geodetic {
        let (lat, lon) = (
            self.origin.latitude_deg.to_radians(),
            self.origin.longitude_deg.to_radians(),
        );
        let (sl, cl, sp, cp) = (lon.sin(), lon.cos(), lat.sin(), lat.cos());
        let delta = Vec3::new(
            -sl * enu.x - sp * cl * enu.y + cp * cl * enu.z,
            cl * enu.x - sp * sl * enu.y + cp * sl * enu.z,
            cp * enu.y + sp * enu.z,
        );
        ecef_to_geodetic(geodetic_to_ecef(self.origin) + delta)
    }

    /// East-north-up offset from the origin of a geodetic point.
    pub fn geodetic_to_enu(&self, point: Geodetic) -> Vec3 {
        let (lat, lon) = (
            self.origin.latitude_deg.to_radians(),
            self.origin.longitude_deg.to_radians(),
        );
        let (sl, cl, sp, cp) = (lon.sin(), lon.cos(), lat.sin(), lat.cos());
        let d = geodetic_to_ecef(point) - geodetic_to_ecef(self.origin);
        Vec3::new(
            -sl * d.x + cl * d.y,
            -sp * cl * d.x - sp * sl * d.y + cp * d.z,
            cp * cl * d.x + cp * sl * d.y + sp * d.z,
        )
    }
}

/// One GPS fix.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GpsReading {
    pub geodetic: Geodetic,
    /// Sensor position as east, north and up metres from the world origin.
    pub enu: Vec3,
    /// East-north-up velocity (m/s) from the previous fix; zero on the first.
    pub velocity_enu: Vec3,
    pub speed: Real,
}

/// GPS receiver: sensor positions to WGS84 fixes, with speed from
/// consecutive fixes as a real receiver derives it.
#[derive(Debug, Clone)]
pub struct Gps {
    pub frame: WorldFrame,
    pub reference: GeoReference,
    previous: Option<(Vec3, Real)>,
}

impl Gps {
    pub fn new(frame: WorldFrame, reference: GeoReference) -> Self {
        Self {
            frame,
            reference,
            previous: None,
        }
    }

    /// Fix at simulated time `time_s` for a sensor at `world_position`.
    pub fn sample(&mut self, world_position: Vec3, time_s: Real) -> GpsReading {
        let enu = self.frame.to_enu(world_position);
        let velocity_enu = match self.previous {
            Some((previous, t)) if time_s > t => (enu - previous) / (time_s - t),
            _ => Vec3::ZERO,
        };
        self.previous = Some((enu, time_s));
        GpsReading {
            geodetic: self.reference.enu_to_geodetic(enu),
            enu,
            velocity_enu,
            speed: velocity_enu.length(),
        }
    }

    /// Forget the previous fix, e.g. after a teleport or reset.
    pub fn reset(&mut self) {
        self.previous = None;
    }
}

/// North in the sensor frame and the sensor's heading.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CompassReading {
    /// Unit vector towards north expressed in sensor axes.
    pub north: Vec3,
    /// Clockwise angle from north to the sensor's +X projected on the
    /// horizontal plane, in `[0, 2π)`.
    pub heading_rad: Real,
}

/// Magnetic compass aligned with geographic north.
pub fn compass(frame: &WorldFrame, world_rotation: Quat) -> CompassReading {
    let [east, north, _] = frame.enu();
    let forward = world_rotation * Vec3::X;
    let heading = forward.dot(east).atan2(forward.dot(north));
    CompassReading {
        north: world_rotation.inverse() * north,
        heading_rad: heading.rem_euclid(std::f64::consts::TAU),
    }
}

/// Orientation of a sensor in the world's east-north-up frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InertialReading {
    /// Rotation from sensor axes to east-north-up.
    pub orientation: Quat,
    /// Roll about +X, pitch about +Y, yaw about +Z (east-north-up, yaw zero
    /// facing east and positive counter-clockwise), applied yaw-pitch-roll.
    pub roll: Real,
    pub pitch: Real,
    pub yaw: Real,
}

/// Inertial unit: sensor attitude relative to the east-north-up frame.
pub fn inertial_unit(frame: &WorldFrame, world_rotation: Quat) -> InertialReading {
    let orientation = (frame.world_from_enu().inverse() * world_rotation).normalize();
    let Quat { x, y, z, w } = orientation;
    let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
    let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
    let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
    InertialReading {
        orientation,
        roll,
        pitch,
        yaw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn near(a: Real, b: Real, tolerance: Real) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn enu_offsets_round_trip_through_wgs84() {
        let reference = GeoReference {
            origin: Geodetic {
                latitude_deg: 51.9851,
                longitude_deg: 5.6637,
                altitude_m: 12.0,
            },
        };
        for enu in [
            Vec3::ZERO,
            Vec3::new(1000.0, 0.0, 0.0),
            Vec3::new(-2500.0, 4200.0, 35.0),
            Vec3::new(0.0, 0.0, 1000.0),
        ] {
            let fix = reference.enu_to_geodetic(enu);
            assert!(
                (reference.geodetic_to_enu(fix) - enu).length() < 1e-6,
                "{enu:?}"
            );
        }
        let north = reference.enu_to_geodetic(Vec3::new(0.0, 1000.0, 0.0));
        // One kilometre north is about 0.00899° of latitude at 52°N.
        assert!(near(north.latitude_deg - 51.9851, 0.008_99, 2e-5));
        assert!(near(north.longitude_deg, 5.6637, 1e-9));
        let east = reference.enu_to_geodetic(Vec3::new(1000.0, 0.0, 0.0));
        // One kilometre east is 1000 / (N cos φ) radians of longitude.
        assert!(near(east.longitude_deg - 5.6637, 0.014_56, 5e-6));
        assert!(near(east.altitude_m, 12.0 + 0.0785, 2e-3));
    }

    #[test]
    fn ecef_matches_known_points() {
        let equator = geodetic_to_ecef(Geodetic::default());
        assert!((equator - Vec3::new(WGS84_A, 0.0, 0.0)).length() < 1e-6);
        let pole = ecef_to_geodetic(Vec3::new(0.0, 0.0, 6_356_752.314_245 + 100.0));
        assert!(near(pole.latitude_deg, 90.0, 1e-9));
        assert!(near(pole.altitude_m, 100.0, 1e-4));
    }

    #[test]
    fn gps_reports_enu_position_and_finite_difference_velocity() {
        let mut gps = Gps::new(WorldFrame::Y_UP_NEG_Z_NORTH, GeoReference::default());
        let first = gps.sample(Vec3::new(3.0, 2.0, -4.0), 1.0);
        assert!((first.enu - Vec3::new(3.0, 4.0, 2.0)).length() < 1e-12);
        assert_eq!(first.speed, 0.0);
        let second = gps.sample(Vec3::new(3.0, 2.0, -6.0), 1.5);
        assert!((second.velocity_enu - Vec3::new(0.0, 4.0, 0.0)).length() < 1e-12);
        assert!(near(second.speed, 4.0, 1e-12));
        gps.reset();
        assert_eq!(gps.sample(Vec3::ZERO, 2.0).speed, 0.0);
    }

    #[test]
    fn compass_and_inertial_unit_follow_the_world_frame() {
        let frame = WorldFrame::Z_UP_Y_NORTH;
        let facing_north = Quat::from_rotation_z(FRAC_PI_2);
        let reading = compass(&frame, facing_north);
        assert!((reading.north - Vec3::X).length() < 1e-12);
        assert!(near(reading.heading_rad, 0.0, 1e-12));
        let facing_west = Quat::from_rotation_z(PI);
        assert!(near(
            compass(&frame, facing_west).heading_rad,
            1.5 * PI,
            1e-12
        ));

        let attitude =
            Quat::from_rotation_z(0.3) * Quat::from_rotation_y(0.2) * Quat::from_rotation_x(0.1);
        let unit = inertial_unit(&frame, attitude);
        assert!(
            near(unit.yaw, 0.3, 1e-12)
                && near(unit.pitch, 0.2, 1e-12)
                && near(unit.roll, 0.1, 1e-12)
        );

        let bevy = WorldFrame::Y_UP_NEG_Z_NORTH;
        // Sensor +X along -Z (north) with +Z up along +Y: yaw 90° in ENU.
        let north_up = Quat::from_mat3(&Mat3::from_cols(Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y));
        let unit = inertial_unit(&bevy, north_up);
        assert!(
            near(unit.yaw, FRAC_PI_2, 1e-12)
                && near(unit.pitch, 0.0, 1e-12)
                && near(unit.roll, 0.0, 1e-12)
        );
        assert!(near(compass(&bevy, north_up).heading_rad, 0.0, 1e-12));
    }
}
