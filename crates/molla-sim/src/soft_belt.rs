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

/// Longitudinal EA (N), additional reference line density (kg/m), and installed engineering prestrain.
#[derive(Clone, Copy, Debug)]
pub struct RubberCordParams {
    pub axial_rigidity: Real,
    pub linear_density: Real,
    pub prestrain: Real,
}

impl Default for RubberCordParams {
    fn default() -> Self {
        Self {
            axial_rigidity: 2e5,
            linear_density: 0.1,
            prestrain: 0.0,
        }
    }
}

pub struct ReinforcedBelt {
    pub mesh: TetMeshId,
    pub nodes: Vec<u32>,
    pub cords: Vec<usize>,
}

impl RubberBeltMesh {
    /// Add midpoint-layer, perfectly bonded axial cords sharing the rubber nodes.
    pub fn append_reinforced_to(
        &self,
        builder: &mut ModelBuilder,
        world: WorldId,
        center: Vec3,
        cord: RubberCordParams,
    ) -> Result<ReinforcedBelt> {
        let p = self.params;
        if !center.is_finite()
            || p.thickness_cells % 2 != 0
            || !cord.axial_rigidity.is_finite()
            || cord.axial_rigidity <= 0.0
            || !cord.linear_density.is_finite()
            || cord.linear_density < 0.0
            || !cord.prestrain.is_finite()
            || !(0.0..=0.1).contains(&cord.prestrain)
        {
            return Err(Error::Build(
                "invalid rubber cord parameters or midpoint layer".into(),
            ));
        }
        let index = |s: usize, w: usize| {
            ((s % p.segments) * (p.width_cells + 1) + w) * (p.thickness_cells + 1)
                + p.thickness_cells / 2
        };
        let mut masses = self.masses.clone();
        let mut elements = Vec::with_capacity(p.segments * (p.width_cells + 1));
        for s in 0..p.segments {
            for w in 0..=p.width_cells {
                let a = index(s, w);
                let b = index(s + 1, w);
                let weight = if w == 0 || w == p.width_cells {
                    0.5
                } else {
                    1.0
                } / p.width_cells as Real;
                let rest =
                    (self.positions[b] - self.positions[a]).length() / (1.0 + cord.prestrain);
                let stiffness = cord.axial_rigidity * weight / rest;
                let mass = cord.linear_density * weight * rest;
                if !mass.is_finite()
                    || !stiffness.is_finite()
                    || !(stiffness as f32).is_finite()
                    || stiffness <= 0.0
                    || !((1.0 / stiffness) as f32).is_finite()
                    || ((1.0 / stiffness) as f32) < f32::MIN_POSITIVE
                    || rest as f32 <= 0.0
                {
                    return Err(Error::Build("unrepresentable rubber cord element".into()));
                }
                masses[a] += mass * 0.5;
                masses[b] += mass * 0.5;
                elements.push((a, b, rest, stiffness));
            }
        }
        if masses
            .iter()
            .any(|m| !m.is_finite() || !(*m as f32).is_finite())
        {
            return Err(Error::Build("unrepresentable reinforced belt mass".into()));
        }
        let nodes: Vec<_> = self
            .positions
            .iter()
            .zip(masses)
            .map(|(&q, m)| builder.add_particle(center + q, m, 0.0, world))
            .collect();
        let tets: Vec<_> = self
            .tets
            .iter()
            .map(|t| t.map(|v| nodes[v as usize]))
            .collect();
        let mesh = builder.add_tet_mesh(&tets, p.youngs_modulus, p.poisson_ratio, None)?;
        let cords = elements
            .into_iter()
            .map(|(a, b, rest, k)| builder.add_tension_element(nodes[a], nodes[b], rest, k))
            .collect::<Result<Vec<_>>>()?;
        Ok(ReinforcedBelt { mesh, nodes, cords })
    }

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
    fn closed_cords_preserve_axial_rigidity_mass_and_prestrain_under_refinement() {
        for (segments, width_cells) in [(16, 1), (32, 2), (64, 4)] {
            for prestrain in [0.0, 0.01] {
                let p = RubberBeltParams {
                    segments,
                    width_cells,
                    ..Default::default()
                };
                let mesh = RubberBeltMesh::circular(p).unwrap();
                let cord = RubberCordParams {
                    prestrain,
                    ..Default::default()
                };
                let mut builder = ModelBuilder::new();
                let handles = mesh
                    .append_reinforced_to(&mut builder, WorldId(0), Vec3::ZERO, cord)
                    .unwrap();
                let model = builder.build().unwrap();
                assert_eq!(handles.cords.len(), segments * (width_cells + 1));
                assert!(
                    model
                        .spring_tension_only
                        .host()
                        .unwrap()
                        .iter()
                        .all(|&v| v == 1)
                );
                assert!(
                    model
                        .spring_max_stretch
                        .host()
                        .unwrap()
                        .iter()
                        .all(|&v| v == 1.0)
                );
                let length = 2.0
                    * p.radius
                    * (std::f64::consts::PI / segments as Real).sin()
                    * segments as Real
                    / (1.0 + prestrain);
                let added_mass = model.particle_mass.host().unwrap().iter().sum::<Real>()
                    - mesh.masses.iter().sum::<Real>();
                assert!((added_mass - cord.linear_density * length).abs() < 1e-10);
                let mut energy = 0.0;
                let mut valence = HashMap::new();
                for s in handles.cords {
                    let a = model.spring_a.host().unwrap()[s] as usize;
                    let b = model.spring_b.host().unwrap()[s] as usize;
                    *valence.entry(a).or_insert(0) += 1;
                    *valence.entry(b).or_insert(0) += 1;
                    let current_length = (mesh.positions[a] - mesh.positions[b]).length() * 1.02;
                    let extension = current_length - model.spring_rest_length.host().unwrap()[s];
                    energy +=
                        0.5 * extension * extension / model.spring_compliance.host().unwrap()[s];
                }
                let strain = 1.02 * (1.0 + prestrain) - 1.0;
                assert!(
                    (energy - 0.5 * cord.axial_rigidity * length * strain * strain).abs() < 1e-8
                );
                assert!(valence.values().all(|&v| v == 2));
            }
        }
    }

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

    #[test]
    fn invalid_cords_are_rejected_before_adding_particles() {
        let mesh = RubberBeltMesh::circular(RubberBeltParams::default()).unwrap();
        for cord in [
            RubberCordParams {
                axial_rigidity: 0.0,
                ..Default::default()
            },
            RubberCordParams {
                linear_density: -1.0,
                ..Default::default()
            },
            RubberCordParams {
                prestrain: Real::NAN,
                ..Default::default()
            },
            RubberCordParams {
                prestrain: 0.2,
                ..Default::default()
            },
            RubberCordParams {
                axial_rigidity: 1e-300,
                ..Default::default()
            },
        ] {
            let mut b = ModelBuilder::new();
            b.add_particle(Vec3::X, 1.0, 0.0, WorldId(0));
            assert!(
                mesh.append_reinforced_to(&mut b, WorldId(0), Vec3::ZERO, cord)
                    .is_err()
            );
            assert_eq!(b.build().unwrap().particle_count, 1);
        }
    }
}
