//! Deformation-gradient material energies and coupled tangents.

use crate::{Mat3, Real, Vec3, dense3::svd3};
use molla_core::{Error, Result};

/// Material selectors matching the pinned PPF energy functions.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElasticModel {
    /// `mu * sum((sigma - 1)^2) + lambda/2 * sum(sigma - 1)^2` (PPF Arap).
    Corotated = 0,
    /// `mu/4 * sum((sigma - 1)^4) + lambda/2 * (J - 1)^2` (PPF StVK).
    QuarticStretch = 1,
    /// `mu/2 * (sum(sigma^2) - n) - mu*(J-1) + lambda/2*(J-1)^2`.
    StableNeoHookean = 2,
    /// Directional column-length stretch and column-dot-product shear.
    BaraffWitkin = 3,
}

#[derive(Clone, Copy, Debug)]
pub struct ElasticMaterial {
    pub model: ElasticModel,
    pub mu: Real,
    pub lambda: Real,
}

/// Hessian indexed by column-major deformation-gradient coordinates.
pub type MaterialHessian = [[Real; 9]; 9];

#[derive(Clone, Debug)]
pub struct MaterialResponse {
    pub energy: Real,
    pub gradient: Mat3,
    pub hessian: MaterialHessian,
}

impl MaterialResponse {
    pub fn hessian_mul(&self, direction: Mat3) -> Mat3 {
        let d = direction.to_cols_array();
        let mut out = [0.0; 9];
        for (i, row) in self.hessian.iter().enumerate() {
            out[i] = row.iter().zip(d).map(|(a, b)| a * b).sum();
        }
        Mat3::from_cols_array(&out)
    }
}

fn invalid(detail: &str) -> Error {
    Error::Solver(format!("constitutive: invalid {detail}"))
}

fn deformation(mut f: Mat3, dimensions: usize) -> Result<(Mat3, Real)> {
    if dimensions == 2 {
        f.z_axis = Vec3::ZERO;
    }
    if !f.is_finite() {
        return Err(invalid("deformation"));
    }
    let scale = f
        .to_cols_array()
        .iter()
        .map(|v| v.abs())
        .fold(0.0, Real::max);
    if scale == 0.0 {
        return Err(invalid("singular deformation"));
    }
    let normalized = Mat3::from_cols(f.x_axis / scale, f.y_axis / scale, f.z_axis / scale);
    if (dimensions == 3 && normalized.determinant() <= 0.0)
        || (dimensions == 2 && normalized.x_axis.cross(normalized.y_axis).length_squared() == 0.0)
    {
        return Err(invalid("singular or inverted deformation"));
    }
    Ok((f, scale))
}

impl ElasticMaterial {
    pub fn validate(&self, dimensions: usize) -> Result<()> {
        if !(2..=3).contains(&dimensions)
            || (dimensions == 3 && self.model == ElasticModel::BaraffWitkin)
        {
            return Err(invalid("material dimension"));
        }
        if !self.mu.is_finite() || self.mu < 0.0 || !self.lambda.is_finite() {
            return Err(invalid("material coefficients"));
        }
        Ok(())
    }

    /// Evaluate a 3x2 membrane or 3x3 solid gradient; shell column 2 is ignored.
    /// `project` clamps spectral eigenvalues, or stretch/shear terms separately.
    pub fn evaluate(&self, f: Mat3, dimensions: usize, project: bool) -> Result<MaterialResponse> {
        self.validate(dimensions)?;
        let (f, scale) = deformation(f, dimensions)?;
        let response = if self.model == ElasticModel::BaraffWitkin {
            self.baraff_witkin(f, project)?
        } else {
            self.spectral(f, scale, dimensions, project)?
        };
        if !response.energy.is_finite()
            || !response.gradient.is_finite()
            || response.hessian.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid("nonfinite response"));
        }
        Ok(response)
    }

    /// Evaluate energy without forming a gradient or tangent.
    pub fn energy(&self, f: Mat3, dimensions: usize) -> Result<Real> {
        self.validate(dimensions)?;
        let (f, scale) = deformation(f, dimensions)?;
        let energy = match self.model {
            ElasticModel::BaraffWitkin => {
                0.5 * self.mu
                    * ((f.x_axis.length() - 1.0).powi(2) + (f.y_axis.length() - 1.0).powi(2))
                    + 0.5 * self.lambda * f.x_axis.dot(f.y_axis).powi(2)
            }
            ElasticModel::StableNeoHookean => {
                let j = if dimensions == 3 {
                    f.determinant()
                } else {
                    f.x_axis.cross(f.y_axis).length()
                };
                let i2: Real = (0..dimensions).map(|i| f.col(i).length_squared()).sum();
                0.5 * self.mu * (i2 - dimensions as Real) - self.mu * (j - 1.0)
                    + 0.5 * self.lambda * (j - 1.0).powi(2)
            }
            _ => {
                let (_, s, _) = svd3(Mat3::from_cols(
                    f.x_axis / scale,
                    f.y_axis / scale,
                    f.z_axis / scale,
                ));
                let s = s * scale;
                let trace: Real = (0..dimensions).map(|i| s[i] - 1.0).sum();
                if self.model == ElasticModel::Corotated {
                    self.mu * (0..dimensions).map(|i| (s[i] - 1.0).powi(2)).sum::<Real>()
                        + 0.5 * self.lambda * trace * trace
                } else {
                    let j: Real = (0..dimensions).map(|i| s[i]).product();
                    0.25 * self.mu * (0..dimensions).map(|i| (s[i] - 1.0).powi(4)).sum::<Real>()
                        + 0.5 * self.lambda * (j - 1.0).powi(2)
                }
            }
        };
        if !energy.is_finite() {
            return Err(invalid("nonfinite energy"));
        }
        Ok(energy)
    }

    fn spectral(&self, f: Mat3, scale: Real, n: usize, project: bool) -> Result<MaterialResponse> {
        let (mut u, singular, v) = svd3(Mat3::from_cols(
            f.x_axis / scale,
            f.y_axis / scale,
            f.z_axis / scale,
        ));
        let s = singular * scale;
        if (0..n).any(|i| !s[i].is_finite() || s[i] <= 0.0) {
            return Err(invalid("singular spectrum"));
        }
        if n == 2 {
            u.z_axis = u.x_axis.cross(u.y_axis).normalize();
        }
        let j: Real = (0..n).map(|i| s[i]).product();
        let trace: Real = (0..n).map(|i| s[i] - 1.0).sum();
        let mut energy = match self.model {
            ElasticModel::Corotated => 0.5 * self.lambda * trace * trace,
            _ => 0.5 * self.lambda * (j - 1.0) * (j - 1.0),
        };
        let mut g = Vec3::ZERO;
        let mut normal = [[0.0; 9]; 9];
        for i in 0..n {
            let p: Real = (0..n).filter(|&k| k != i).map(|k| s[k]).product();
            let a = s[i] - 1.0;
            match self.model {
                ElasticModel::Corotated => {
                    energy += self.mu * a * a;
                    g[i] = 2.0 * self.mu * a + self.lambda * trace;
                    normal[i][i] = 2.0 * self.mu;
                }
                ElasticModel::QuarticStretch => {
                    energy += 0.25 * self.mu * a.powi(4);
                    g[i] = self.mu * a.powi(3) + self.lambda * (j - 1.0) * p;
                    normal[i][i] = 3.0 * self.mu * a * a;
                }
                ElasticModel::StableNeoHookean => {
                    energy += 0.5 * self.mu * (s[i] * s[i] - 1.0);
                    g[i] = self.mu * s[i] + (self.lambda * (j - 1.0) - self.mu) * p;
                    normal[i][i] = self.mu;
                }
                _ => unreachable!(),
            }
            for k in 0..n {
                let pk: Real = (0..n).filter(|&l| l != k).map(|l| s[l]).product();
                normal[i][k] += if self.model == ElasticModel::Corotated {
                    self.lambda
                } else {
                    self.lambda * p * pk
                };
                if i != k && self.model != ElasticModel::Corotated {
                    let r: Real = (0..n).filter(|&l| l != i && l != k).map(|l| s[l]).product();
                    let t = self.lambda * (j - 1.0)
                        - if self.model == ElasticModel::StableNeoHookean {
                            self.mu
                        } else {
                            0.0
                        };
                    normal[i][k] += t * r;
                }
            }
        }
        if self.model == ElasticModel::StableNeoHookean {
            energy -= self.mu * (j - 1.0);
        }
        if project {
            normal = project_psd(normal, n)?;
        }
        let clamp = |x: Real| if project { x.max(0.0) } else { x };
        let mut hessian = [[0.0; 9]; 9];
        for basis in 0..3 * n {
            let mut entries = [0.0; 9];
            entries[basis] = 1.0;
            let q = u.transpose() * Mat3::from_cols_array(&entries) * v;
            let mut z = [[0.0; 3]; 3];
            for i in 0..n {
                for k in 0..n {
                    z[i][i] += normal[i][k] * q.col(k)[k];
                }
                for k in i + 1..n {
                    let r: Real = (0..n).filter(|&l| l != i && l != k).map(|l| s[l]).product();
                    let symmetric = match self.model {
                        ElasticModel::Corotated => 2.0 * self.mu,
                        ElasticModel::QuarticStretch => {
                            let a = s[i] - 1.0;
                            let b = s[k] - 1.0;
                            self.mu * (a * a + a * b + b * b) - self.lambda * (j - 1.0) * r
                        }
                        _ => self.mu - (self.lambda * (j - 1.0) - self.mu) * r,
                    };
                    let skew = (g[i] + g[k]) / (s[i] + s[k]);
                    let sum = clamp(symmetric) * (q.col(k)[i] + q.col(i)[k]) * 0.5;
                    let diff = clamp(skew) * (q.col(k)[i] - q.col(i)[k]) * 0.5;
                    z[i][k] = sum + diff;
                    z[k][i] = sum - diff;
                }
                if n == 2 {
                    z[2][i] = clamp(g[i] / s[i]) * q.col(i)[2];
                }
            }
            let z = Mat3::from_cols(
                Vec3::new(z[0][0], z[1][0], z[2][0]),
                Vec3::new(z[0][1], z[1][1], z[2][1]),
                Vec3::new(z[0][2], z[1][2], z[2][2]),
            );
            let out = (u * z * v.transpose()).to_cols_array();
            for i in 0..3 * n {
                hessian[i][basis] = out[i];
            }
        }
        Ok(MaterialResponse {
            energy,
            gradient: u * Mat3::from_diagonal(g) * v.transpose(),
            hessian,
        })
    }

    fn baraff_witkin(&self, f: Mat3, project: bool) -> Result<MaterialResponse> {
        let a = f.x_axis;
        let b = f.y_axis;
        let lengths = [a.length(), b.length()];
        let shear = a.dot(b);
        let mut gradient = Mat3::ZERO;
        gradient.x_axis = a * (self.mu * (1.0 - 1.0 / lengths[0])) + b * (self.lambda * shear);
        gradient.y_axis = b * (self.mu * (1.0 - 1.0 / lengths[1])) + a * (self.lambda * shear);
        let mut stretch_h = [[0.0; 9]; 9];
        let mut shear_h = [[0.0; 9]; 9];
        for c in 0..2 {
            let dir = f.col(c) / lengths[c];
            let t = 1.0 - 1.0 / lengths[c];
            let t = if project { t.max(0.0) } else { t };
            for i in 0..3 {
                for k in 0..3 {
                    let identity = if i == k { 1.0 } else { 0.0 };
                    stretch_h[3 * c + i][3 * c + k] =
                        self.mu * (t * identity + (1.0 - t) * dir[i] * dir[k]);
                    shear_h[3 * c + i][3 * c + k] = self.lambda * f.col(1 - c)[i] * f.col(1 - c)[k];
                    shear_h[i][3 + k] = self.lambda * (b[i] * a[k] + shear * identity);
                    shear_h[3 + k][i] = shear_h[i][3 + k];
                }
            }
        }
        if project {
            shear_h = project_psd(shear_h, 6)?;
        }
        for i in 0..6 {
            for k in 0..6 {
                stretch_h[i][k] += shear_h[i][k];
            }
        }
        Ok(MaterialResponse {
            energy: 0.5 * self.mu * ((lengths[0] - 1.0).powi(2) + (lengths[1] - 1.0).powi(2))
                + 0.5 * self.lambda * shear * shear,
            gradient,
            hessian: stretch_h,
        })
    }
}

fn project_psd(mut a: MaterialHessian, n: usize) -> Result<MaterialHessian> {
    let scale = a.iter().flatten().map(|v| v.abs()).fold(0.0, Real::max);
    if !scale.is_finite() || a.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("tangent"));
    }
    if scale == 0.0 {
        return Ok(a);
    }
    let mut v = [[0.0; 9]; 9];
    for i in 0..n {
        v[i][i] = 1.0;
        for k in 0..n {
            a[i][k] /= scale;
        }
    }
    for _ in 0..32 {
        let mut largest: Real = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                let off = a[p][q];
                largest = largest.max(off.abs());
                if off.abs() <= 1.0e-14 {
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * off);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let t = if tau == 0.0 { 1.0 } else { t };
                let c = (1.0 + t * t).sqrt().recip();
                let s = t * c;
                a[p][p] -= t * off;
                a[q][q] += t * off;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for k in 0..n {
                    if k != p && k != q {
                        let ap = a[k][p];
                        let aq = a[k][q];
                        a[k][p] = c * ap - s * aq;
                        a[p][k] = a[k][p];
                        a[k][q] = s * ap + c * aq;
                        a[q][k] = a[k][q];
                    }
                    let vp = v[k][p];
                    let vq = v[k][q];
                    v[k][p] = c * vp - s * vq;
                    v[k][q] = s * vp + c * vq;
                }
            }
        }
        if largest <= 1.0e-14 {
            break;
        }
    }
    if (0..n).any(|i| (i + 1..n).any(|k| a[i][k].abs() > 1.0e-10)) {
        return Err(invalid("unconverged eigensystem"));
    }
    let mut out = [[0.0; 9]; 9];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                out[i][k] += v[i][j] * a[j][j].max(0.0) * v[k][j] * scale;
            }
        }
    }
    Ok(out)
}
