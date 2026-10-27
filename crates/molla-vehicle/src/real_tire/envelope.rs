/// Compact contact-envelope geometry in a rolling/up/axle frame, in metres.
/// Sidewall bulge is a geometric approximation, not a solved FEM displacement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoadedTireEnvelope {
    pub radius: f64,
    pub bead_radius: f64,
    pub width: f64,
    pub loaded_radius: f64,
}

impl LoadedTireEnvelope {
    /// Maps reference rubber geometry onto the loaded contact envelope.
    pub fn deform(self, point: [f64; 3]) -> [f64; 3] {
        let [x, y, z] = point;
        if ![
            self.radius,
            self.bead_radius,
            self.width,
            self.loaded_radius,
            x,
            y,
            z,
        ]
        .iter()
        .all(|v| v.is_finite())
            || self.radius <= 0.0
            || self.bead_radius <= 0.0
            || self.bead_radius >= self.radius
            || self.width <= 0.0
        {
            return point;
        }
        let loaded = self.loaded_radius.clamp(self.bead_radius, self.radius);
        let compression = self.radius - loaded;
        let radial = x.hypot(y);
        if compression == 0.0 || radial <= self.bead_radius || y >= 0.0 {
            return point;
        }
        let sidewall =
            ((radial - self.bead_radius) / (self.radius - self.bead_radius)).clamp(0.0, 1.0);
        let shoulder = (std::f64::consts::PI * sidewall).sin();
        let downward = (-y / radial).clamp(0.0, 1.0);
        let lower_sector = downward.powi(4);
        let lateral = (z / (0.25 * self.width)).clamp(-1.0, 1.0);
        let bulge = 0.65 * compression * shoulder * lower_sector * lateral;
        let radial_weight = sidewall * sidewall * (3.0 - 2.0 * sidewall);
        let tread_lift = (self.radius * downward - loaded).max(0.0);
        let compressed_y = y + radial_weight * tread_lift;
        [x, compressed_y.max(-loaded), z + bulge]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(loaded_radius: f64) -> LoadedTireEnvelope {
        LoadedTireEnvelope {
            radius: 1.0,
            bead_radius: 0.5,
            width: 0.4,
            loaded_radius,
        }
    }

    #[test]
    fn unloaded_bead_and_upper_sidewall_are_unchanged() {
        for p in [[0.0, -1.0, 0.2], [0.4, -0.7, -0.2]] {
            assert_eq!(envelope(1.0).deform(p), p);
        }
        for p in [[0.0, -0.5, 0.2], [0.0, 0.8, 0.2]] {
            assert_eq!(envelope(0.8).deform(p), p);
        }
    }

    #[test]
    fn tread_meets_loaded_plane_and_sidewall_bulges_symmetrically() {
        let soft = envelope(0.8);
        assert_eq!(soft.deform([0.0, -1.0, 0.2])[1], -0.8);
        let left = soft.deform([0.0, -0.75, 0.2]);
        let right = soft.deform([0.0, -0.75, -0.2]);
        assert!(left[2] > envelope(0.95).deform([0.0, -0.75, 0.2])[2]);
        assert_eq!(left[2], -right[2]);
        assert_eq!(soft.deform([0.0, -1.0, 0.0])[2], 0.0);
    }

    #[test]
    fn invalid_geometry_is_not_deformed() {
        let point = [0.0, -0.8, 0.2];
        assert_eq!(envelope(f64::NAN).deform(point), point);
        assert_eq!(
            LoadedTireEnvelope {
                width: 0.0,
                ..envelope(0.8)
            }
            .deform(point),
            point
        );
    }

    #[test]
    fn lower_sidewall_compresses_before_it_reaches_the_ground() {
        let point = [0.0, -0.75, 0.2];
        let low = envelope(0.8).deform(point);
        let high = envelope(0.95).deform(point);
        assert!((low[1] + 0.65).abs() < 1e-12);
        assert!(low[1] > high[1] && high[1] > point[1]);
        for loaded in [0.55, 0.8, 0.95] {
            for radial in [0.5, 0.55, 0.75, 1.0, 1.05] {
                for angle in 0..=180 {
                    let theta = (angle as f64).to_radians();
                    let p = [radial * theta.cos(), -radial * theta.sin(), 0.2];
                    assert!(envelope(loaded).deform(p)[1] >= -loaded - 1e-12);
                }
            }
        }
    }
}
