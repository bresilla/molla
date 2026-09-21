//! Periodic tetrahedral belt geometry for volumetric elasticity solvers.

use crate::{ModelBuilder, TetMeshId};
use molla_core::{Error, Result, WorldId};
use molla_math::{Mat3, Real, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct RubberBeltParams {
    pub radius: Real,
    pub width: Real,
    pub thickness: Real,
    pub segments: usize,
    pub width_cells: usize,
    pub thickness_cells: usize,
    pub density: Real,
    pub youngs_modulus: Real,
    pub poisson_ratio: Real,
}

impl Default for RubberBeltParams {
    fn default() -> Self {
        Self {
            radius: 0.5,
            width: 0.18,
            thickness: 0.02,
            segments: 64,
            width_cells: 4,
            thickness_cells: 2,
            density: 1100.0,
            youngs_modulus: 2.0e6,
            poisson_ratio: 0.45,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RubberBeltMesh {
    pub positions: Vec<Vec3>,
    pub tets: Vec<[u32; 4]>,
    pub masses: Vec<Real>,
    pub volumes: Vec<Real>,
    params: RubberBeltParams,
}

impl RubberBeltMesh {
    pub fn circular(params: RubberBeltParams) -> Result<Self> {
        let p = params;
        let nodes = p
            .segments
            .checked_mul(p.width_cells.saturating_add(1))
            .and_then(|n| n.checked_mul(p.thickness_cells.saturating_add(1)));
        let cells = p
            .segments
            .checked_mul(p.width_cells)
            .and_then(|n| n.checked_mul(p.thickness_cells));
        if [p.radius, p.width, p.thickness, p.density, p.youngs_modulus]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || !p.poisson_ratio.is_finite()
            || !(0.0..0.5).contains(&p.poisson_ratio)
            || p.thickness >= 2.0 * p.radius
            || p.segments < 8
            || p.width_cells == 0
            || p.thickness_cells == 0
            || nodes.is_none_or(|n| n > 1_000_000)
            || cells.is_none_or(|n| n > 1_000_000)
        {
            return Err(Error::Build(
                "invalid rubber belt dimensions, resolution or material".into(),
            ));
        }
        let mut positions = Vec::with_capacity(nodes.unwrap());
        let index = |s: usize, w: usize, t: usize| -> u32 {
            (((s % p.segments) * (p.width_cells + 1) + w) * (p.thickness_cells + 1) + t) as u32
        };
        for s in 0..p.segments {
            let angle = std::f64::consts::TAU * s as Real / p.segments as Real;
            for w in 0..=p.width_cells {
                for t in 0..=p.thickness_cells {
                    let radius =
                        p.radius + p.thickness * (t as Real / p.thickness_cells as Real - 0.5);
                    positions.push(Vec3::new(
                        p.width * (w as Real / p.width_cells as Real - 0.5),
                        radius * angle.cos(),
                        radius * angle.sin(),
                    ));
                }
            }
        }
        let mut tets = Vec::with_capacity(cells.unwrap() * 6);
        let mut volumes = Vec::with_capacity(cells.unwrap() * 6);
        let mut masses = vec![0.0; positions.len()];
        for s in 0..p.segments {
            for w in 0..p.width_cells {
                for t in 0..p.thickness_cells {
                    let [a, b, c, d, e, f, g, h] = [
                        index(s, w, t),
                        index(s + 1, w, t),
                        index(s, w + 1, t),
                        index(s + 1, w + 1, t),
                        index(s, w, t + 1),
                        index(s + 1, w, t + 1),
                        index(s, w + 1, t + 1),
                        index(s + 1, w + 1, t + 1),
                    ];
                    for mut tet in [
                        [a, b, d, h],
                        [a, d, c, h],
                        [a, c, g, h],
                        [a, g, e, h],
                        [a, e, f, h],
                        [a, f, b, h],
                    ] {
                        let mut determinant = tet_basis(&positions, tet).determinant();
                        if determinant < 0.0 {
                            tet.swap(1, 2);
                            determinant = -determinant;
                        }
                        if !determinant.is_finite() || determinant <= 1e-18 {
                            return Err(Error::Build("degenerate rubber belt tetrahedron".into()));
                        }
                        let volume = determinant / 6.0;
                        for node in tet {
                            masses[node as usize] += p.density * volume / 4.0;
                        }
                        tets.push(tet);
                        volumes.push(volume);
                    }
                }
            }
        }
        if masses.iter().any(|m| !m.is_finite() || *m <= 0.0) {
            return Err(Error::Build("invalid rubber belt nodal mass".into()));
        }
        Ok(Self {
            positions,
            tets,
            masses,
            volumes,
            params,
        })
    }

    pub fn append_to(
        &self,
        builder: &mut ModelBuilder,
        world: WorldId,
        center: Vec3,
    ) -> Result<(TetMeshId, Vec<u32>)> {
        if !center.is_finite() {
            return Err(Error::Build("non-finite rubber belt position".into()));
        }
        let nodes: Vec<_> = self
            .positions
            .iter()
            .zip(&self.masses)
            .map(|(&q, &mass)| builder.add_particle(center + q, mass, 0.0, world))
            .collect();
        let tets: Vec<_> = self
            .tets
            .iter()
            .map(|tet| tet.map(|v| nodes[v as usize]))
            .collect();
        let mesh = builder.add_tet_mesh(
            &tets,
            self.params.youngs_modulus,
            self.params.poisson_ratio,
            None,
        )?;
        Ok((mesh, nodes))
    }

    pub fn minimum_volume_ratio(&self, positions: &[Vec3]) -> Result<Real> {
        if positions.len() != self.positions.len() || positions.iter().any(|v| !v.is_finite()) {
            return Err(Error::Build("invalid rubber belt state".into()));
        }
        Ok(self
            .tets
            .iter()
            .zip(&self.volumes)
            .map(|(&tet, &volume)| tet_basis(positions, tet).determinant() / (6.0 * volume))
            .fold(Real::INFINITY, Real::min))
    }
}

fn tet_basis(q: &[Vec3], tet: [u32; 4]) -> Mat3 {
    let [a, b, c, d] = tet.map(|v| q[v as usize]);
    Mat3::from_cols(b - a, c - a, d - a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn closed_belt_has_positive_volume_and_watertight_surface() {
        let p = RubberBeltParams::default();
        let mesh = RubberBeltMesh::circular(p).unwrap();
        let expected = p.segments as Real
            * (std::f64::consts::TAU / p.segments as Real).sin()
            * p.radius
            * p.thickness
            * p.width;
        let volume: Real = mesh.volumes.iter().sum();
        assert!((volume - expected).abs() < 1e-12);
        assert!((mesh.masses.iter().sum::<Real>() - p.density * volume).abs() < 1e-10);
        assert!((mesh.minimum_volume_ratio(&mesh.positions).unwrap() - 1.0).abs() < 1e-12);
        let surface = crate::compute_tet_surface_triangles(&mesh.tets);
        let mut edges = HashMap::new();
        for tri in surface {
            for mut edge in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] {
                edge.sort();
                *edges.entry(edge).or_insert(0) += 1;
            }
        }
        assert!(edges.values().all(|&n| n == 2));
    }

    #[test]
    fn invalid_resolution_and_material_are_rejected() {
        for p in [
            RubberBeltParams {
                segments: usize::MAX,
                ..Default::default()
            },
            RubberBeltParams {
                poisson_ratio: 0.5,
                ..Default::default()
            },
            RubberBeltParams {
                thickness: 1.0,
                ..Default::default()
            },
            RubberBeltParams {
                width: Real::NAN,
                ..Default::default()
            },
        ] {
            assert!(RubberBeltMesh::circular(p).is_err());
        }
    }
}
