use super::invalid;
use molla_core::Result;
use molla_geometry::convex_cut::CutPlane;
use molla_math::{Mat3, Real, Transform, Vec3};

/// Corresponding convex interface polygons in the two material frames.
#[derive(Clone, Debug)]
pub struct BondPatch {
    pub points: Vec<[Vec3; 2]>,
}

pub(super) struct PatchMeasure {
    pub area: Real,
    pub center: Vec3,
    pub moments: Vec3,
}

impl BondPatch {
    pub fn new(points: Vec<[Vec3; 2]>) -> Result<Self> {
        let patch = Self { points };
        patch.measure(0)?;
        patch.measure(1)?;
        Ok(patch)
    }

    pub(super) fn measure(&self, side: usize) -> Result<PatchMeasure> {
        if self.points.len() < 3 || self.points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("invalid interface polygon"));
        }
        let origin = self.points[0][side];
        let scale = self
            .points
            .iter()
            .map(|p| p[side].distance(origin))
            .fold(0.0, Real::max);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid("degenerate interface polygon"));
        }
        let p: Vec<_> = self
            .points
            .iter()
            .map(|p| (p[side] - origin) / scale)
            .collect();
        let normal = (1..p.len() - 1)
            .map(|i| p[i].cross(p[i + 1]))
            .sum::<Vec3>()
            .normalize_or_zero();
        if normal.length_squared() < 0.5 || p.iter().any(|v| normal.dot(*v).abs() > 1e-10) {
            return Err(invalid("nonplanar interface polygon"));
        }
        for i in 0..p.len() {
            let edge = p[(i + 1) % p.len()] - p[i];
            if p.iter().any(|v| normal.dot(edge.cross(*v - p[i])) < -1e-12) {
                return Err(invalid("nonconvex interface polygon"));
            }
        }
        let outer = |v: Vec3| Mat3::from_cols(v * v.x, v * v.y, v * v.z);
        let mut area = 0.0;
        let mut first = Vec3::ZERO;
        let mut second = Mat3::ZERO;
        for i in 1..p.len() - 1 {
            let a = p[i];
            let b = p[i + 1];
            let tri_area = 0.5 * normal.dot(a.cross(b));
            area += tri_area;
            first += (a + b) * (tri_area / 3.0);
            second += (outer(a) + outer(b) + outer(a + b)) * (tri_area / 12.0);
        }
        if area < 1e-14 {
            return Err(invalid("unresolved interface area"));
        }
        let center = first / area;
        let covariance = second - outer(center) * area;
        let diagonal = Vec3::new(
            covariance.x_axis.x,
            covariance.y_axis.y,
            covariance.z_axis.z,
        );
        let moments = (Vec3::splat(diagonal.element_sum()) - diagonal) * scale.powi(4);
        let measure = PatchMeasure {
            area: area * scale * scale,
            center: origin + center * scale,
            moments,
        };
        if !measure.area.is_finite()
            || !measure.center.is_finite()
            || !moments.is_finite()
            || moments.min_element() <= 0.0
        {
            return Err(invalid("interface moment overflow or degeneracy"));
        }
        Ok(measure)
    }

    pub(super) fn clip_sides(
        &self,
        planes: [CutPlane; 2],
        poses: [Transform; 2],
        positive: bool,
    ) -> Result<Option<Self>> {
        let scale = self
            .points
            .iter()
            .map(|p| p[0].distance(self.points[0][0]))
            .fold(0.0, Real::max);
        let tolerance = scale * 64.0 * Real::EPSILON;
        let direction = if positive { -1.0 } else { 1.0 };
        let distance = |p: [Vec3; 2], side: usize| {
            let d = direction * planes[side].distance(poses[side].transform_point(p[side]));
            if d.abs() < tolerance { 0.0 } else { d }
        };
        let mut points = self.points.clone();
        for side in 0..2 {
            let mut out = Vec::new();
            for i in 0..points.len() {
                let a = points[i];
                let b = points[(i + 1) % points.len()];
                let da = distance(a, side);
                let db = distance(b, side);
                if da <= 0.0 {
                    out.push(a);
                }
                if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                    let t = da / (da - db);
                    out.push([a[0].lerp(b[0], t), a[1].lerp(b[1], t)]);
                }
            }
            out.dedup_by(|a, b| a[0].distance(b[0]) <= tolerance);
            if out.len() > 1 && out[0][0].distance(out.last().unwrap()[0]) <= tolerance {
                out.pop();
            }
            points = out;
            if points.len() < 3 {
                return Ok(None);
            }
        }
        let first = points[0][0];
        let area_vector = (1..points.len() - 1)
            .map(|i| (points[i][0] - first).cross(points[i + 1][0] - first))
            .sum::<Vec3>();
        if area_vector.length() < scale * scale * 1e-14 {
            return Ok(None);
        }
        let candidate = Self { points };
        let center = [candidate.measure(0)?.center, candidate.measure(1)?.center];
        if (0..2).any(|side| distance(center, side) >= -tolerance) {
            return Ok(None);
        }
        Ok(Some(candidate))
    }
}
