//! Membrane and tetrahedral material coordinates.

use molla_core::{Error, Result};
use molla_math::{
    Mat3, Real, Vec3,
    constitutive::{ElasticMaterial, MaterialResponse, MembraneStrainLimit},
};

#[derive(Clone, Debug)]
pub struct ElasticElement {
    vertices: [u32; 4],
    dimensions: usize,
    inverse_rest: Mat3,
    measure: Real,
    material: ElasticMaterial,
    strain_limit: Option<MembraneStrainLimit>,
    strain_mass: Real,
}

fn invalid(detail: &str) -> Error {
    Error::Solver(format!("elastic element: invalid {detail}"))
}

impl ElasticElement {
    /// Authored inverse rest basis and reference-volume energy weight.
    pub fn from_rest_coordinates(
        vertices: &[u32],
        inverse_rest: Mat3,
        measure: Real,
        material: ElasticMaterial,
    ) -> Result<Self> {
        if !(3..=4).contains(&vertices.len())
            || vertices
                .iter()
                .enumerate()
                .any(|(i, v)| vertices[..i].contains(v))
            || !inverse_rest.is_finite()
            || !measure.is_finite()
            || measure <= 0.0
        {
            return Err(invalid("rest coordinates"));
        }
        let dimensions = vertices.len() - 1;
        material.validate(dimensions)?;
        let determinant = if dimensions == 3 {
            inverse_rest.determinant()
        } else {
            if inverse_rest.z_axis != Vec3::ZERO
                || inverse_rest.x_axis.z != 0.0
                || inverse_rest.y_axis.z != 0.0
            {
                return Err(invalid("membrane rest basis"));
            }
            inverse_rest.x_axis.x * inverse_rest.y_axis.y
                - inverse_rest.x_axis.y * inverse_rest.y_axis.x
        };
        if !determinant.is_finite() || determinant == 0.0 {
            return Err(invalid("singular rest basis"));
        }
        let mut ids = [0; 4];
        ids[..vertices.len()].copy_from_slice(vertices);
        Ok(Self {
            vertices: ids,
            dimensions,
            inverse_rest,
            measure,
            material,
            strain_limit: None,
            strain_mass: 0.0,
        })
    }

    pub fn with_material(mut self, material: ElasticMaterial) -> Result<Self> {
        material.validate(self.dimensions)?;
        self.material = material;
        Ok(self)
    }

    /// Attach the cubic strain barrier and cap with the face's physical mass.
    pub fn with_strain_limit(mut self, limit: MembraneStrainLimit, mass: Real) -> Result<Self> {
        if self.dimensions != 2 {
            return Err(invalid("strain limit on solid element"));
        }
        limit.maximum_stretch()?;
        if !mass.is_finite() || mass <= 0.0 { return Err(invalid("strain face mass")); }
        self.strain_limit = Some(limit);
        self.strain_mass = mass;
        Ok(self)
    }

    pub fn strain_limit(&self) -> Option<MembraneStrainLimit> {
        self.strain_limit
    }

    pub fn strain_mass(&self) -> Real { self.strain_mass }

    /// Rest-area times thickness weights the membrane energy density.
    pub fn membrane(
        rest: &[Vec3],
        vertices: [u32; 3],
        thickness: Real,
        material: ElasticMaterial,
    ) -> Result<Self> {
        if !thickness.is_finite() || thickness <= 0.0 {
            return Err(invalid("thickness"));
        }
        Self::indices(rest, &vertices)?;
        material.validate(2)?;
        let a = rest[vertices[1] as usize] - rest[vertices[0] as usize];
        let b = rest[vertices[2] as usize] - rest[vertices[0] as usize];
        let length = a.length();
        let area2 = a.cross(b).length();
        if !length.is_finite() || !area2.is_finite() || length <= 0.0 || area2 <= 0.0 {
            return Err(invalid("rest triangle"));
        }
        let along = b.dot(a / length);
        let height = area2 / length;
        let inverse_rest = Mat3::from_cols(
            Vec3::new(1.0 / length, 0.0, 0.0),
            Vec3::new(-along / (length * height), 1.0 / height, 0.0),
            Vec3::ZERO,
        );
        let element = Self {
            vertices: [vertices[0], vertices[1], vertices[2], 0],
            dimensions: 2,
            inverse_rest,
            measure: area2 * 0.5 * thickness,
            material,
            strain_limit: None,
            strain_mass: 0.0,
        };
        element.validate(rest)?;
        Ok(element)
    }

    /// Rest volume weights the solid energy density.
    pub fn tetrahedron(
        rest: &[Vec3],
        vertices: [u32; 4],
        material: ElasticMaterial,
    ) -> Result<Self> {
        Self::indices(rest, &vertices)?;
        material.validate(3)?;
        let origin = rest[vertices[0] as usize];
        let dm = Mat3::from_cols(
            rest[vertices[1] as usize] - origin,
            rest[vertices[2] as usize] - origin,
            rest[vertices[3] as usize] - origin,
        );
        let det = dm.determinant();
        if !det.is_finite() || det == 0.0 {
            return Err(invalid("rest tet"));
        }
        let element = Self {
            vertices,
            dimensions: 3,
            inverse_rest: dm.inverse(),
            measure: det.abs() / 6.0,
            material,
            strain_limit: None,
            strain_mass: 0.0,
        };
        element.validate(rest)?;
        Ok(element)
    }

    fn indices(positions: &[Vec3], ids: &[u32]) -> Result<()> {
        for (i, &id) in ids.iter().enumerate() {
            if id as usize >= positions.len()
                || ids[..i].contains(&id)
                || !positions[id as usize].is_finite()
            {
                return Err(invalid("vertices"));
            }
        }
        Ok(())
    }

    pub fn vertices(&self) -> &[u32] {
        &self.vertices[..self.dimensions + 1]
    }
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn inverse_rest(&self) -> Mat3 {
        self.inverse_rest
    }
    pub fn measure(&self) -> Real {
        self.measure
    }
    pub fn material(&self) -> ElasticMaterial {
        self.material
    }

    pub fn shape_gradients(&self) -> [Vec3; 4] {
        let mut g = [Vec3::ZERO; 4];
        for i in 0..self.dimensions {
            g[i + 1] = self.inverse_rest.row(i);
            g[0] -= g[i + 1];
        }
        g
    }

    pub fn deformation(&self, positions: &[Vec3]) -> Result<Mat3> {
        Self::indices(positions, self.vertices())?;
        let origin = positions[self.vertices[0] as usize];
        let a = positions[self.vertices[1] as usize] - origin;
        let b = positions[self.vertices[2] as usize] - origin;
        let c = if self.dimensions == 3 {
            positions[self.vertices[3] as usize] - origin
        } else {
            Vec3::ZERO
        };
        let f = Mat3::from_cols(a, b, c) * self.inverse_rest;
        if !f.is_finite() {
            return Err(invalid("nonfinite deformation"));
        }
        Ok(f)
    }

    pub fn validate(&self, positions: &[Vec3]) -> Result<()> {
        self.material.validate(self.dimensions)?;
        if !self.inverse_rest.is_finite() || !self.measure.is_finite() || self.measure <= 0.0 {
            return Err(invalid("rest coordinates or measure"));
        }
        let f = self.deformation(positions)?;
        let measure = if self.dimensions == 3 {
            f.determinant()
        } else {
            f.x_axis.cross(f.y_axis).length_squared()
        };
        if !f.is_finite() || !measure.is_finite() || measure <= 0.0 {
            return Err(invalid("singular or inverted deformation"));
        }
        if let Some(limit) = self.strain_limit {
            limit.check(f)?;
        }
        Ok(())
    }

    pub fn energy(&self, positions: &[Vec3]) -> Result<Real> {
        let f = self.deformation(positions)?;
        if let Some(limit) = self.strain_limit {
            limit.check(f)?;
        }
        let energy = self.material.energy(f, self.dimensions)? * self.measure;
        if !energy.is_finite() {
            return Err(invalid("nonfinite energy"));
        }
        Ok(energy)
    }

    /// Conservative rank-preserving step bound along a vertex displacement.
    pub fn step_bound(&self, positions: &[Vec3], direction: &[Vec3]) -> Result<Real> {
        let f = self.deformation(positions)?;
        let scale = f
            .to_cols_array()
            .iter()
            .map(|v| v.abs())
            .fold(0.0, Real::max);
        if scale == 0.0 || !scale.is_finite() {
            return Err(invalid("deformation"));
        }
        let (_, s, _) = molla_math::dense3::svd3(Mat3::from_cols(
            f.x_axis / scale,
            f.y_axis / scale,
            f.z_axis / scale,
        ));
        let sigma = (0..self.dimensions)
            .map(|i| s[i] * scale)
            .fold(Real::INFINITY, Real::min);
        let df = self.deformation(direction)?;
        let norm = df
            .to_cols_array()
            .iter()
            .map(|v| v * v)
            .sum::<Real>()
            .sqrt();
        if !norm.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return Err(invalid("step bound"));
        }
        let alpha = if norm == 0.0 {
            1.0
        } else {
            (0.9 * sigma / norm).min(1.0)
        };
        if let Some(limit) = self.strain_limit {
            limit.step_bound(f, df, alpha)
        } else {
            Ok(alpha)
        }
    }

    pub fn response(&self, positions: &[Vec3], projected: bool) -> Result<MaterialResponse> {
        let f = self.deformation(positions)?;
        if let Some(limit) = self.strain_limit {
            limit.check(f)?;
        }
        let mut r = self.material.evaluate(f, self.dimensions, projected)?;
        r.energy *= self.measure;
        r.gradient *= self.measure;
        for value in r.hessian.iter_mut().flatten() {
            *value *= self.measure;
        }
        if !r.energy.is_finite()
            || !r.gradient.is_finite()
            || r.hessian.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(invalid("nonfinite response"));
        }
        Ok(r)
    }
}
