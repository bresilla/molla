//! Geodesic swing coordinates and angular constraint Jacobians.

use crate::joint_control::SphericalConeLimits;
use molla_core::{Error, Result};
use molla_math::{Quat, Real, Vec3};

pub const ANTIPODAL_SINE_FLOOR: Real = 1e-6;

#[derive(Clone, Copy, Debug, Default)]
pub struct ConeRow {
    pub error: Real,
    /// Angular derivative in the parent joint frame; one-sided at the antipodal cut.
    pub angular: Vec3,
    pub bilateral: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ConeGeometry {
    pub swing: [Real; 2],
    pub angle: Real,
    pub violation: Real,
    pub rows: [ConeRow; 3],
    pub row_count: usize,
}

/// Primary axis and right-handed transverse basis; positive cardinal axes use cyclic order.
pub fn basis(axis: Vec3) -> Result<[Vec3; 3]> {
    let length = axis.length();
    if !length.is_finite() || length <= 1e-12 {
        return Err(Error::Build("invalid spherical cone axis".into()));
    }
    let p = axis / length;
    let mut dominant = 0;
    for i in 1..3 {
        if p[i].abs() > p[dominant].abs() {
            dominant = i;
        }
    }
    let reference = [Vec3::X, Vec3::Y, Vec3::Z][(dominant + 1) % 3];
    let b0 = (reference - p * p.dot(reference)).normalize();
    Ok([p, b0, p.cross(b0)])
}

fn antipodal_geometry(
    limits: SphericalConeLimits,
    basis: [Vec3; 3],
    n: Vec3,
    angle: Real,
) -> ConeGeometry {
    let [p, b0, b1] = basis;
    let extents = [limits.angle0, limits.angle1];
    let radius = extents.map(|a| a.unwrap_or(std::f64::consts::PI));
    let selected = usize::from(radius[1] > radius[0]);
    let transverse = [b0, b1];
    let mut geometry = ConeGeometry {
        angle,
        ..Default::default()
    };
    geometry.swing[selected] = angle;
    for i in 0..2 {
        if i != selected && extents[i] == Some(0.) {
            let component = n.dot(transverse[i]).clamp(-1., 1.);
            let error = component.asin();
            geometry.rows[geometry.row_count] = ConeRow {
                error,
                angular: n.cross(transverse[i]) / (1. - component * component).sqrt(),
                bilateral: true,
            };
            geometry.row_count += 1;
            geometry.violation = geometry.violation.max(error.abs());
        }
    }
    if radius[selected] < std::f64::consts::PI {
        let radial = p.cross(transverse[selected]);
        let angular = (radial - n * radial.dot(n)).normalize();
        let error = angle - radius[selected];
        geometry.rows[geometry.row_count] = ConeRow {
            error,
            angular,
            bilateral: radius[selected] == 0.,
        };
        geometry.row_count += 1;
        geometry.violation = geometry.violation.max(error.max(0.));
    }
    geometry
}

/// Evaluate the disk-log ellipse with a deterministic antipodal limiting chart.
pub fn evaluate(limits: SphericalConeLimits, axis: Vec3, relative: Quat) -> Result<ConeGeometry> {
    limits.validate()?;
    let [p, b0, b1] = basis(axis)?;
    let norm = relative.length();
    if !norm.is_finite() || norm <= 1e-12 {
        return Err(Error::Build("invalid spherical cone orientation".into()));
    }
    let n = relative.normalize() * p;
    let c = p.dot(n).clamp(-1.0, 1.0);
    let t = n - p * c;
    let s = t.length();
    let angle = s.atan2(c);
    let mut geometry = ConeGeometry {
        angle,
        ..Default::default()
    };
    if c < 0.0 && s <= ANTIPODAL_SINE_FLOOR {
        if limits.is_unrestricted() {
            geometry.swing = [angle, 0.0];
            return Ok(geometry);
        }
        return Ok(antipodal_geometry(limits, [p, b0, b1], n, angle));
    }
    let (k, q) = if angle < 1e-3 {
        let a2 = angle * angle;
        (
            1.0 + a2 / 6.0 + 7.0 * a2 * a2 / 360.0,
            -2.0 / 3.0 - a2 / 5.0,
        )
    } else {
        (angle / s, (c * s - angle) / (s * s * s))
    };
    let components = [b0.dot(t), b1.dot(t)];
    geometry.swing = components.map(|value| k * value);
    if limits.is_unrestricted() {
        return Ok(geometry);
    }
    let derivatives = [b0, b1].map(|b| n.cross(b * k + (t * q - p) * b.dot(t)));
    let angles = [limits.angle0, limits.angle1];
    for i in 0..2 {
        if angles[i] == Some(0.0) {
            geometry.rows[geometry.row_count] = ConeRow {
                error: geometry.swing[i],
                angular: derivatives[i],
                bilateral: true,
            };
            geometry.row_count += 1;
            geometry.violation = geometry.violation.max(geometry.swing[i].abs());
        }
    }
    if let Some(scale) = angles
        .into_iter()
        .flatten()
        .filter(|&a| a > 0.0)
        .reduce(Real::min)
    {
        if scale * scale == 0.0 {
            return Err(Error::Solver(
                "spherical cone extent exceeds numerical precision".into(),
            ));
        }
        let mut row = ConeRow {
            error: -scale * scale,
            ..Default::default()
        };
        let mut squared = 0.0;
        for i in 0..2 {
            if let Some(a) = angles[i].filter(|&a| a > 0.0) {
                let weight = (scale / a).powi(2);
                if weight == 0.0 {
                    return Err(Error::Solver(
                        "spherical cone aspect exceeds numerical precision".into(),
                    ));
                }
                row.error += weight * geometry.swing[i] * geometry.swing[i];
                row.angular += derivatives[i] * (2.0 * weight * geometry.swing[i]);
                squared += (geometry.swing[i] / a).powi(2);
            }
        }
        if squared > 1.0 {
            geometry.violation = geometry.violation.max(angle * (1.0 - 1.0 / squared.sqrt()));
        }
        geometry.rows[geometry.row_count] = row;
        geometry.row_count += 1;
    }
    if angles
        .into_iter()
        .any(|a| a.is_none_or(|a| a == std::f64::consts::PI))
        && std::f64::consts::PI - angle < geometry.violation
    {
        let error = std::f64::consts::PI - angle;
        geometry.rows = [ConeRow::default(); 3];
        geometry.rows[0] = ConeRow {
            error,
            angular: -p.cross(t / s),
            bilateral: false,
        };
        geometry.row_count = 1;
        geometry.violation = error;
    }
    if geometry
        .rows
        .iter()
        .any(|row| !row.error.is_finite() || !row.angular.is_finite())
        || !geometry.violation.is_finite()
    {
        return Err(Error::Solver("spherical cone geometry overflow".into()));
    }
    Ok(geometry)
}
