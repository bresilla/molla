use super::{MaterialResponse, deformation, invalid};
use crate::{Mat3, Real, Vec3, dense3::svd3};
use molla_core::Result;

/// Principal-stretch cap and unit-stiffness PPF cubic strain potential.
#[derive(Clone, Copy, Debug)]
pub struct MembraneStrainLimit {
    /// Authored maximum strain, measured as singular value minus one.
    pub strain: Real,
    /// Rest-axis shrink factors used only to correct the hard cap.
    pub shrink: [Real; 2],
}

impl MembraneStrainLimit {
    pub fn maximum_stretch(&self) -> Result<Real> {
        if !self.strain.is_finite()
            || self.strain <= 0.0
            || self.shrink.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid("membrane strain limit"));
        }
        let maximum = (1.0 + self.strain) / self.shrink[0].min(self.shrink[1]);
        if !maximum.is_finite() || maximum <= 1.0 {
            return Err(invalid("effective membrane strain limit"));
        }
        Ok(maximum)
    }

    /// Largest singular value of the first two columns, including rank-zero inputs.
    pub fn largest_stretch(f: Mat3) -> Result<Real> {
        if !f.x_axis.is_finite() || !f.y_axis.is_finite() {
            return Err(invalid("membrane deformation"));
        }
        let scale = f.x_axis.abs().max(f.y_axis.abs()).max_element();
        if scale == 0.0 {
            return Ok(0.0);
        }
        let a = f.x_axis / scale;
        let b = f.y_axis / scale;
        let aa = a.length_squared();
        let bb = b.length_squared();
        let ab = a.dot(b);
        let largest = (0.5 * (aa + bb + (aa - bb).hypot(2.0 * ab))).sqrt() * scale;
        if !largest.is_finite() {
            return Err(invalid("membrane spectrum"));
        }
        Ok(largest)
    }

    pub fn check(&self, f: Mat3) -> Result<()> {
        if Self::largest_stretch(f)? >= self.maximum_stretch()? {
            return Err(invalid("membrane principal-stretch cap"));
        }
        Ok(())
    }

    /// Safe prefix of an affine deformation path inside the spectral-norm ball.
    pub fn step_bound(&self, f: Mat3, df: Mat3, max_t: Real) -> Result<Real> {
        self.check(f)?;
        if !df.is_finite() || !max_t.is_finite() || !(0.0..=1.0).contains(&max_t) {
            return Err(invalid("membrane strain sweep"));
        }
        let limit = self.maximum_stretch()?;
        let inside = |t| Self::largest_stretch(f + df * t).is_ok_and(|s| s < limit);
        if inside(max_t) {
            return Ok(max_t);
        }
        let (mut lower, mut upper) = (0.0, max_t);
        for _ in 0..64 {
            let middle = 0.5 * (lower + upper);
            if middle == lower || middle == upper {
                break;
            }
            if inside(middle) {
                lower = middle;
            } else {
                upper = middle;
            }
        }
        Ok(0.9 * lower)
    }

    /// Cubic potential before elasticity/inertia-dependent stiffness scaling.
    pub fn energy(&self, f: Mat3) -> Result<Real> {
        self.check(f)?;
        let (f, scale) = deformation(f, 2)?;
        let largest = Self::largest_stretch(f)?;
        let smallest =
            (f.x_axis / scale).cross(f.y_axis / scale).length() / (largest / scale) * scale;
        let a = (largest - 1.0).max(0.0);
        let b = (smallest - 1.0).max(0.0);
        let energy = (2.0 / 3.0) * (a.powi(3) + b.powi(3)) / self.strain;
        if !energy.is_finite() {
            return Err(invalid("strain energy"));
        }
        Ok(energy)
    }

    /// Exact deformation-space derivatives; the cubic spectral potential is convex.
    pub fn evaluate(&self, f: Mat3) -> Result<MaterialResponse> {
        self.check(f)?;
        let (f, scale) = deformation(f, 2)?;
        let (mut u, sigma, v) = svd3(Mat3::from_cols(
            f.x_axis / scale,
            f.y_axis / scale,
            Vec3::ZERO,
        ));
        let s = sigma * scale;
        if !s.x.is_finite() || !s.y.is_finite() || s.x <= 0.0 || s.y <= 0.0 {
            return Err(invalid("strain spectrum"));
        }
        u.z_axis = u.x_axis.cross(u.y_axis).normalize();
        let a = (s.x - 1.0).max(0.0);
        let b = (s.y - 1.0).max(0.0);
        let g = Vec3::new(2.0 * a * a / self.strain, 2.0 * b * b / self.strain, 0.0);
        let normal = Vec3::new(4.0 * a / self.strain, 4.0 * b / self.strain, 0.0);
        let symmetric = if a > 0.0 && b > 0.0 {
            2.0 * (a + b) / self.strain
        } else if s.x != s.y {
            (g.x - g.y) / (s.x - s.y)
        } else {
            normal.x
        };
        let skew = (g.x + g.y) / (s.x + s.y);
        let mut hessian = [[0.0; 9]; 9];
        for basis in 0..6 {
            let mut d = [0.0; 9];
            d[basis] = 1.0;
            let q = u.transpose() * Mat3::from_cols_array(&d) * v;
            let sum = 0.5 * symmetric * (q.y_axis.x + q.x_axis.y);
            let diff = 0.5 * skew * (q.y_axis.x - q.x_axis.y);
            let z = Mat3::from_cols(
                Vec3::new(normal.x * q.x_axis.x, sum - diff, g.x / s.x * q.x_axis.z),
                Vec3::new(sum + diff, normal.y * q.y_axis.y, g.y / s.y * q.y_axis.z),
                Vec3::ZERO,
            );
            let column = (u * z * v.transpose()).to_cols_array();
            for row in 0..6 {
                hessian[row][basis] = column[row];
            }
        }
        let result = MaterialResponse {
            energy: (2.0 / 3.0) * (a.powi(3) + b.powi(3)) / self.strain,
            gradient: u * Mat3::from_diagonal(g) * v.transpose(),
            hessian,
        };
        if !result.energy.is_finite()
            || !result.gradient.is_finite()
            || result.hessian.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid("strain response"));
        }
        Ok(result)
    }
}
