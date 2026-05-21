//! H13 — Solar-position model. Computes the world-space direction
//! toward the sun (and a few related derived quantities) from a
//! latitude / longitude / date-time / UTC-offset tuple.
//!
//! This is a pure-CPU `f64` port of the standard NOAA Solar Position
//! Algorithm (NREL SPA, Reda & Andreas 2003) in its General-Solar-
//! Position simplified form — accurate to within a few hundredths of
//! a degree across the year. The reference layout mirrors helios's
//! `plugins/solarposition/include/SolarPosition.h`.
//!
//! Convention: outdoor scenes typically use `+Y` up; the returned
//! `sun_direction` is a unit vector in that local-tangent-plane (LTP)
//! frame, with `+X` pointing **east**, `+Y` pointing **up**, and
//! `+Z` pointing **north**. Vehicle / lidar / camera scenes can use
//! this directly as the direction toward the sun without worrying
//! about ECEF / quaternion conversions.

use std::f64::consts::PI;

/// Civil date-time used as the solar-position input. All fields are
/// in **civil local time** at the location's `utc_offset_hours`; the
/// solver internally converts to UTC and Julian day.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CivilDateTime {
    pub year: i32,
    pub month: u32,  // 1..=12
    pub day: u32,    // 1..=31
    pub hour: u32,   // 0..=23
    pub minute: u32, // 0..=59
    pub second: f64, // 0..60 (allow fractional)
}

impl CivilDateTime {
    pub fn new(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: f64) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    /// Decimal hours from `00:00` (civil local time).
    pub fn decimal_hour(&self) -> f64 {
        self.hour as f64 + self.minute as f64 / 60.0 + self.second / 3600.0
    }
}

/// Output of a solar-position evaluation. All angles in radians, the
/// direction in the local-tangent-plane (`+X` east, `+Y` up, `+Z`
/// north) frame.
#[derive(Debug, Clone, Copy)]
pub struct SolarState {
    /// Unit vector pointing from the observer toward the sun.
    pub sun_direction: [f64; 3],
    /// Solar elevation angle (rad) — π/2 = directly overhead,
    /// 0 = at the horizon, negative = below the horizon.
    pub elevation: f64,
    /// Solar zenith angle (rad) — `π/2 − elevation`.
    pub zenith: f64,
    /// Solar azimuth (rad), measured **clockwise from north**,
    /// in [0, 2π).
    pub azimuth: f64,
    /// Solar declination (rad).
    pub declination: f64,
    /// Solar hour angle (rad) — the difference between local solar
    /// noon and the current solar time. Negative before noon,
    /// positive after.
    pub hour_angle: f64,
    /// Equation of time (minutes) — the difference between mean and
    /// apparent solar time.
    pub equation_of_time_minutes: f64,
}

/// Compute the solar state at a location and time.
///
/// `lat` and `lon` in degrees (`+N` / `+E`). `utc_offset_hours` is the
/// civil offset of `t` relative to UTC (e.g. `−8` for Pacific
/// Standard Time, `+1` for Central European Time).
///
/// Inputs are **clamped** for robustness against unusual call sites
/// (sim configs, JSON-loaded scenarios): `lat_deg` is clamped to
/// `[-90, 90]`, `lon_deg` to `[-180, 180]` (wrapped), and
/// `utc_offset_hours` to `[-14, 14]` (the union of every IANA tz
/// offset). Out-of-range hour / minute / second values are wrapped
/// rather than aborting — a caller can still get a sensible answer
/// from a slightly malformed timestamp.
pub fn solar_position(
    lat_deg: f64,
    lon_deg: f64,
    t: CivilDateTime,
    utc_offset_hours: f64,
) -> SolarState {
    let lat_deg = lat_deg.clamp(-90.0, 90.0);
    let lon_deg = lon_deg.rem_euclid(360.0);
    let lon_deg = if lon_deg > 180.0 {
        lon_deg - 360.0
    } else {
        lon_deg
    };
    let utc_offset_hours = utc_offset_hours.clamp(-14.0, 14.0);
    let lat = lat_deg.to_radians();
    let jd = julian_day_utc(t, utc_offset_hours);
    // Julian centuries since J2000.0.
    let jc = (jd - 2_451_545.0) / 36_525.0;
    // Geometric mean longitude of the sun, in degrees.
    let l0 = wrap_deg(280.466_46 + 36_000.769_83 * jc + 0.000_303_2 * jc * jc);
    // Geometric mean anomaly of the sun, in degrees.
    let m = 357.529_11 + 35_999.052_9 * jc - 0.000_153_7 * jc * jc;
    let m_rad = m.to_radians();
    // Eccentricity of the Earth's orbit.
    let e = 0.016_708_634 - 0.000_042_037 * jc - 0.000_000_126_7 * jc * jc;
    // Sun's equation of centre, in degrees.
    let c = (1.914_602 - 0.004_817 * jc - 0.000_014 * jc * jc) * m_rad.sin()
        + (0.019_993 - 0.000_101 * jc) * (2.0 * m_rad).sin()
        + 0.000_289 * (3.0 * m_rad).sin();
    let true_long = l0 + c;
    let omega = (125.04 - 1934.136 * jc).to_radians();
    let lambda = (true_long - 0.005_69 - 0.004_78 * omega.sin()).to_radians();
    // Mean obliquity of the ecliptic.
    let epsilon0 = 23.439_291_111 - (46.815 / 3600.0) * jc - (0.000_59 / 3600.0) * jc * jc
        + (0.001_813 / 3600.0) * jc * jc * jc;
    let epsilon = (epsilon0 + 0.002_56 * omega.cos()).to_radians();
    let declination = (epsilon.sin() * lambda.sin()).asin();

    // Equation of time, in minutes.
    let y = (epsilon / 2.0).tan().powi(2);
    let l0_rad = l0.to_radians();
    let eot_rad = y * (2.0 * l0_rad).sin() - 2.0 * e * m_rad.sin()
        + 4.0 * e * y * m_rad.sin() * (2.0 * l0_rad).cos()
        - 0.5 * y * y * (4.0 * l0_rad).sin()
        - 1.25 * e * e * (2.0 * m_rad).sin();
    let equation_of_time_minutes = eot_rad.to_degrees() * 4.0;

    // Local true solar time, in hours.
    let civil_hours = t.decimal_hour();
    let solar_time_hours =
        civil_hours - utc_offset_hours + lon_deg / 15.0 + equation_of_time_minutes / 60.0;
    let hour_angle = ((solar_time_hours - 12.0) * 15.0).to_radians();

    // Standard sphere-trig solar elevation + azimuth.
    let sin_alt = lat.sin() * declination.sin() + lat.cos() * declination.cos() * hour_angle.cos();
    let sin_alt = sin_alt.clamp(-1.0, 1.0);
    let elevation = sin_alt.asin();
    let cos_alt = (1.0 - sin_alt * sin_alt).sqrt().max(1.0e-12);
    // Azimuth measured clockwise from north (NOAA convention).
    let cos_az = (declination.sin() - sin_alt * lat.sin()) / (cos_alt * lat.cos().max(1.0e-12));
    let cos_az = cos_az.clamp(-1.0, 1.0);
    let az_from_n = cos_az.acos();
    let azimuth = if hour_angle.sin() > 0.0 {
        2.0 * PI - az_from_n
    } else {
        az_from_n
    };

    // Local-tangent-plane direction: +X east, +Y up, +Z north.
    let zenith = std::f64::consts::FRAC_PI_2 - elevation;
    let dx = zenith.sin() * azimuth.sin();
    let dz = zenith.sin() * azimuth.cos();
    let dy = zenith.cos();
    let inv = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-12).recip();
    SolarState {
        sun_direction: [dx * inv, dy * inv, dz * inv],
        elevation,
        zenith,
        azimuth: azimuth.rem_euclid(2.0 * PI),
        declination,
        hour_angle,
        equation_of_time_minutes,
    }
}

/// Compute the Julian day number (with fractional time-of-day) for a
/// civil date/time. `utc_offset_hours` is subtracted so the JD is in
/// UTC. Algorithm from Meeus, "Astronomical Algorithms" Ch. 7.
pub fn julian_day_utc(t: CivilDateTime, utc_offset_hours: f64) -> f64 {
    let mut year = t.year;
    let mut month = t.month as i32;
    if month <= 2 {
        year -= 1;
        month += 12;
    }
    let a = (year as f64 / 100.0).floor();
    // Gregorian-calendar correction (valid for dates on/after 1582-10-15).
    let b = 2.0 - a + (a / 4.0).floor();
    let day = t.day as f64
        + (t.hour as f64 - utc_offset_hours + t.minute as f64 / 60.0 + t.second / 3600.0) / 24.0;
    (365.25 * (year as f64 + 4716.0)).floor() + (30.600_1 * (month + 1) as f64).floor() + day + b
        - 1524.5
}

fn wrap_deg(d: f64) -> f64 {
    let r = d.rem_euclid(360.0);
    if r < 0.0 { r + 360.0 } else { r }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_day_handles_leap_year_dates() {
        // Feb 29 in a leap year (2024) must produce a JD distinct
        // from Mar 1 (one day apart).
        let feb29 = CivilDateTime::new(2024, 2, 29, 12, 0, 0.0);
        let mar1 = CivilDateTime::new(2024, 3, 1, 12, 0, 0.0);
        let jd_feb = julian_day_utc(feb29, 0.0);
        let jd_mar = julian_day_utc(mar1, 0.0);
        assert!((jd_mar - jd_feb - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn julian_day_is_monotone_across_hour_boundary() {
        // JD must strictly increase across the midnight boundary.
        let pre = CivilDateTime::new(2024, 6, 1, 23, 59, 59.0);
        let post = CivilDateTime::new(2024, 6, 2, 0, 0, 1.0);
        assert!(julian_day_utc(post, 0.0) > julian_day_utc(pre, 0.0));
    }

    /// J2000.0 epoch: 2000-01-01 12:00 UTC → JD 2451545.0 by definition.
    #[test]
    fn julian_day_matches_j2000() {
        let t = CivilDateTime::new(2000, 1, 1, 12, 0, 0.0);
        let jd = julian_day_utc(t, 0.0);
        assert!(
            (jd - 2_451_545.0).abs() < 1.0e-6,
            "JD(J2000) = {jd}, expected 2451545.0"
        );
    }

    /// Reference point: summer solstice 2024-06-20 at 12:00 UTC on
    /// the equator (lat 0, lon 0) — the sun is very high (~66.5° if
    /// projected to the meridian; the local elevation depends on the
    /// hour angle but should be above ~60°).
    #[test]
    fn equator_noon_summer_solstice_has_high_elevation() {
        let t = CivilDateTime::new(2024, 6, 20, 12, 0, 0.0);
        let s = solar_position(0.0, 0.0, t, 0.0);
        let el_deg = s.elevation.to_degrees();
        assert!(
            el_deg > 60.0,
            "expected elevation > 60° at equator noon summer solstice, got {el_deg}°"
        );
    }

    /// The sun-direction y-component (up) is the sine of the
    /// elevation — the LTP frame is consistent with the angle.
    #[test]
    fn sun_direction_up_component_equals_sin_elevation() {
        let t = CivilDateTime::new(2024, 3, 21, 12, 0, 0.0);
        let s = solar_position(45.0, 0.0, t, 0.0);
        let sin_el = s.elevation.sin();
        assert!(
            (s.sun_direction[1] - sin_el).abs() < 1.0e-9,
            "dir.y {} ≠ sin(el) {}",
            s.sun_direction[1],
            sin_el
        );
    }

    /// Approximate equinox check: on 2024-03-21 at 12:00 UTC the sun
    /// is approximately overhead the equator, so |declination| is
    /// small (< 1°).
    #[test]
    fn equinox_declination_is_small() {
        let t = CivilDateTime::new(2024, 3, 21, 12, 0, 0.0);
        let s = solar_position(0.0, 0.0, t, 0.0);
        let dec_deg = s.declination.to_degrees().abs();
        assert!(
            dec_deg < 1.5,
            "expected |declination| < 1.5° at equinox, got {dec_deg}°"
        );
    }

    #[test]
    fn extreme_latitude_inputs_are_clamped_rather_than_panicking() {
        let t = CivilDateTime::new(2024, 6, 1, 12, 0, 0.0);
        let s_north = solar_position(95.0, 0.0, t, 0.0);
        let s_south = solar_position(-95.0, 0.0, t, 0.0);
        // Clamping should give a finite, plausible elevation either
        // way; the test merely confirms no NaN / panic occurs.
        assert!(s_north.elevation.is_finite());
        assert!(s_south.elevation.is_finite());
        // North-pole-clamped (90°N) summer noon → sun barely above
        // horizon; south-pole-clamped → below.
        assert!(s_north.elevation > s_south.elevation);
    }

    #[test]
    fn longitude_wraps_to_180_signed_range() {
        let t = CivilDateTime::new(2024, 6, 1, 12, 0, 0.0);
        // Equivalent longitudes 270°E and -90°E should agree.
        let a = solar_position(0.0, 270.0, t, 0.0);
        let b = solar_position(0.0, -90.0, t, 0.0);
        assert!((a.elevation - b.elevation).abs() < 1.0e-9);
        assert!((a.azimuth - b.azimuth).abs() < 1.0e-9);
    }

    #[test]
    fn utc_offset_outside_band_is_clamped() {
        let t = CivilDateTime::new(2024, 6, 1, 12, 0, 0.0);
        let extreme = solar_position(0.0, 0.0, t, 50.0);
        let capped = solar_position(0.0, 0.0, t, 14.0);
        assert!((extreme.elevation - capped.elevation).abs() < 1.0e-9);
    }

    /// The sun is below the horizon at midnight local time at high
    /// latitude (and on a non-polar-day date).
    #[test]
    fn midnight_at_paris_in_january_is_below_horizon() {
        // Paris: 48.85°N, 2.35°E; January 15 at 00:00 CET (UTC+1).
        let t = CivilDateTime::new(2024, 1, 15, 0, 0, 0.0);
        let s = solar_position(48.85, 2.35, t, 1.0);
        assert!(
            s.elevation < 0.0,
            "expected sun below horizon at midnight, got elevation = {}°",
            s.elevation.to_degrees()
        );
    }

    /// Property: in the morning the sun is east of the meridian
    /// (azimuth in the first or second quadrant, roughly 0° to 180°
    /// in NOAA "clockwise from north" convention). At equinox noon
    /// the sun is in the south at mid-latitudes.
    #[test]
    fn sun_is_east_of_meridian_in_morning_at_mid_latitude() {
        // Paris equinox 09:00 local (UTC+2 summer).
        let t = CivilDateTime::new(2024, 3, 21, 9, 0, 0.0);
        let s = solar_position(48.85, 2.35, t, 1.0);
        let az_deg = s.azimuth.to_degrees();
        assert!(
            s.elevation > 0.0,
            "sun should be up at 09:00, elevation = {}°",
            s.elevation.to_degrees()
        );
        // Morning sun: azimuth between 60° (NE) and 180° (S).
        assert!(
            (60.0..=180.0).contains(&az_deg),
            "expected morning azimuth in (60°, 180°), got {az_deg}°"
        );
    }

    #[test]
    fn sun_is_west_of_meridian_in_afternoon_at_mid_latitude() {
        // Paris equinox 16:00 local (UTC+1 standard).
        let t = CivilDateTime::new(2024, 3, 21, 16, 0, 0.0);
        let s = solar_position(48.85, 2.35, t, 1.0);
        let az_deg = s.azimuth.to_degrees();
        assert!(s.elevation > 0.0);
        // Afternoon sun: azimuth between 180° (S) and 300° (NW).
        assert!(
            (180.0..=300.0).contains(&az_deg),
            "expected afternoon azimuth in (180°, 300°), got {az_deg}°"
        );
    }

    /// Determinism: the same inputs must always produce the same
    /// outputs (bit-identical). Important for repro of training
    /// runs and golden-image tests downstream.
    #[test]
    fn solar_position_is_deterministic() {
        let t = CivilDateTime::new(2024, 7, 4, 13, 47, 12.5);
        let a = solar_position(40.7, -74.0, t, -4.0);
        let b = solar_position(40.7, -74.0, t, -4.0);
        assert_eq!(a.elevation, b.elevation);
        assert_eq!(a.azimuth, b.azimuth);
        assert_eq!(a.sun_direction, b.sun_direction);
    }

    /// `sun_direction` is unit length within fp tolerance.
    #[test]
    fn sun_direction_is_unit_length() {
        for hour in 0..24u32 {
            let t = CivilDateTime::new(2024, 7, 4, hour, 0, 0.0);
            let s = solar_position(40.7, -74.0, t, -4.0);
            let d = s.sun_direction;
            let m2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            assert!(
                (m2 - 1.0).abs() < 1.0e-9,
                "sun_direction at hour {hour}: |d|² = {m2}, expected 1"
            );
        }
    }
}
