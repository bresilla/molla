//! Periodic tetrahedral belt geometry for volumetric elasticity solvers.

use crate::{ModelBuilder, TetMeshId};
use molla_core::{Error, Result, WorldId};
use molla_math::{Mat3, Real, Vec3};

pub mod path;

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

#[derive(Clone, Copy, Debug)]
pub struct RubberLugParams {
    pub count: usize,
    pub segment_span: usize,
    pub width_start: usize,
    pub width_cells: usize,
    pub depth: Real,
    pub depth_cells: usize,
}

impl RubberBeltMesh {
    fn recompute_volumes_and_masses(&mut self) -> Result<()> {
        self.masses.fill(0.0);
        for (tet, volume) in self.tets.iter().zip(&mut self.volumes) {
            *volume = tet_basis(&self.positions, *tet).determinant() / 6.0;
            if !volume.is_finite() || *volume <= 1e-18 {
                return Err(Error::Build("degenerate graded belt tetrahedron".into()));
            }
            for &node in tet {
                self.masses[node as usize] += *volume * self.params.density / 4.0;
            }
        }
        Ok(())
    }

    /// Redistribute the base mesh's width columns with unchanged total width and density.
    pub fn with_width_stations(&self, stations: &[Real]) -> Result<Self> {
        let p = self.params;
        if stations.len() != p.width_cells + 1
            || self.positions.len() != p.segments * (p.width_cells + 1) * (p.thickness_cells + 1)
            || stations.iter().any(|v| !v.is_finite())
            || stations[0] != -p.width * 0.5
            || stations[p.width_cells] != p.width * 0.5
            || stations.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(Error::Build(
                "invalid belt width stations or non-base mesh".into(),
            ));
        }
        let mut mesh = self.clone();
        for (i, point) in mesh.positions.iter_mut().enumerate() {
            point.x = stations[(i / (p.thickness_cells + 1)) % (p.width_cells + 1)];
        }
        mesh.recompute_volumes_and_masses()?;
        Ok(mesh)
    }

    /// Shift outer-surface angular stations in metres along the neutral circumference.
    /// Offsets are width-major then pitch-node-major; the inner half and cords stay fixed.
    pub fn with_outer_phase_offsets(&self, count: usize, offsets: &[Real]) -> Result<Self> {
        let p = self.params;
        let invalid = || Error::Build("invalid outer belt phase grid".into());
        if count < 2
            || !p.segments.is_multiple_of(count)
            || !p.thickness_cells.is_multiple_of(2)
            || self.positions.len() != p.segments * (p.width_cells + 1) * (p.thickness_cells + 1)
        {
            return Err(invalid());
        }
        let span = p.segments / count;
        let step = std::f64::consts::TAU * p.radius / p.segments as Real;
        if offsets.len() != (p.width_cells + 1) * span || offsets.iter().any(|v| !v.is_finite()) {
            return Err(invalid());
        }
        for row in offsets.chunks_exact(span) {
            for i in 0..span {
                if step + row[(i + 1) % span] - row[i] <= 0.0 {
                    return Err(invalid());
                }
            }
        }
        let mut mesh = self.clone();
        for s in 0..p.segments {
            for w in 0..=p.width_cells {
                for r in p.thickness_cells / 2 + 1..=p.thickness_cells {
                    let i = (s * (p.width_cells + 1) + w) * (p.thickness_cells + 1) + r;
                    let blend = 2.0 * r as Real / p.thickness_cells as Real - 1.0;
                    let angle = std::f64::consts::TAU * s as Real / p.segments as Real
                        + offsets[w * span + s % span] * blend / p.radius;
                    let radius = mesh.positions[i].y.hypot(mesh.positions[i].z);
                    mesh.positions[i].y = radius * angle.cos();
                    mesh.positions[i].z = radius * angle.sin();
                }
            }
        }
        mesh.recompute_volumes_and_masses()?;
        Ok(mesh)
    }

    /// Generate initial installed positions while retaining the circular elastic reference mesh.
    pub fn installed_positions(
        &self,
        path: &path::BeltPath,
        phase: Real,
        center: Vec3,
    ) -> Result<Vec<Vec3>> {
        if !center.is_finite() || !phase.is_finite() {
            return Err(Error::Build("invalid installed belt transform".into()));
        }
        let positions = self
            .positions
            .iter()
            .map(|q| {
                let s = q.z.atan2(q.y).rem_euclid(std::f64::consts::TAU) * path.length()
                    / std::f64::consts::TAU
                    + phase;
                let frame = path.sample(s)?;
                let offset = q.y.hypot(q.z) - self.params.radius;
                Ok(center + frame.position + Vec3::X * q.x + frame.outward * offset)
            })
            .collect::<Result<Vec<_>>>()?;
        if self.minimum_volume_ratio(&positions)? <= 0.0 {
            return Err(Error::Build(
                "installed belt contains inverted tetrahedra".into(),
            ));
        }
        Ok(positions)
    }

    /// Add volumetric inner lugs whose base faces share the belt's tetrahedral nodes.
    pub fn with_drive_lugs(&self, lug: RubberLugParams) -> Result<Self> {
        self.with_drive_lug_rows(&[lug])
    }

    /// Attach separated inner lug rows to shared base faces on the width grid.
    pub fn with_drive_lug_rows(&self, lugs: &[RubberLugParams]) -> Result<Self> {
        self.with_surface_lug_rows(lugs, &[])
    }

    /// Bond inner guide rows and outer tread rows to the corresponding base faces.
    pub fn with_surface_lug_rows(
        &self,
        inner: &[RubberLugParams],
        outer: &[RubberLugParams],
    ) -> Result<Self> {
        let p = self.params;
        let original = p.segments * (p.width_cells + 1) * (p.thickness_cells + 1);
        if (inner.is_empty() && outer.is_empty()) || self.positions.len() != original {
            return Err(Error::Build("empty belt lugs or non-base mesh".into()));
        }
        for lugs in [inner, outer] {
            if lugs.iter().enumerate().any(|(i, a)| {
                lugs[..i].iter().any(|b| {
                    !(a.width_start.saturating_add(a.width_cells) < b.width_start
                        || b.width_start.saturating_add(b.width_cells) < a.width_start)
                })
            }) {
                return Err(Error::Build("overlapping belt lug rows".into()));
            }
        }
        let mut mesh = self.clone();
        for (outward, lugs) in [(false, inner), (true, outer)] {
            for &lug in lugs {
                let extra = lug
                    .count
                    .saturating_mul(lug.segment_span.saturating_add(1))
                    .saturating_mul(lug.width_cells.saturating_add(1))
                    .saturating_mul(lug.depth_cells);
                if self.positions.len() != original
                    || lug.count < 2
                    || !p.segments.is_multiple_of(lug.count)
                    || lug.segment_span == 0
                    || lug.segment_span >= p.segments / lug.count
                    || lug.width_cells == 0
                    || lug.width_start.saturating_add(lug.width_cells) > p.width_cells
                    || lug.depth_cells == 0
                    || mesh.positions.len().saturating_add(extra) > 1_000_000
                    || !lug.depth.is_finite()
                    || lug.depth <= 0.0
                    || (!outward && lug.depth >= p.radius - p.thickness * 0.5)
                {
                    return Err(Error::Build("invalid rubber drive lug geometry".into()));
                }
                for tooth in 0..lug.count {
                    let start = tooth * (p.segments / lug.count);
                    let mut grid = Vec::new();
                    for r in 0..=lug.depth_cells {
                        for s in 0..=lug.segment_span {
                            for w in 0..=lug.width_cells {
                                let segment = (start + s) % p.segments;
                                let width = lug.width_start + w;
                                let base = (segment * (p.width_cells + 1) + width)
                                    * (p.thickness_cells + 1)
                                    + if outward { p.thickness_cells } else { 0 };
                                if (outward && r == 0) || (!outward && r == lug.depth_cells) {
                                    grid.push(base as u32);
                                } else {
                                    let point = self.positions[base];
                                    let angle = point.z.atan2(point.y);
                                    let radius = point.y.hypot(point.z)
                                        + lug.depth
                                            * (r as Real / lug.depth_cells as Real
                                                - if outward { 0.0 } else { 1.0 });
                                    grid.push(mesh.positions.len() as u32);
                                    mesh.positions.push(Vec3::new(
                                        point.x,
                                        radius * angle.cos(),
                                        radius * angle.sin(),
                                    ));
                                    mesh.masses.push(0.0);
                                }
                            }
                        }
                    }
                    let index = |r: usize, s: usize, w: usize| {
                        grid[(r * (lug.segment_span + 1) + s) * (lug.width_cells + 1) + w]
                    };
                    for r in 0..lug.depth_cells {
                        for s in 0..lug.segment_span {
                            for w in 0..lug.width_cells {
                                let [a, b, c, d, e, f, g, h] = [
                                    index(r, s, w),
                                    index(r, s + 1, w),
                                    index(r, s, w + 1),
                                    index(r, s + 1, w + 1),
                                    index(r + 1, s, w),
                                    index(r + 1, s + 1, w),
                                    index(r + 1, s, w + 1),
                                    index(r + 1, s + 1, w + 1),
                                ];
                                for mut tet in [
                                    [a, b, d, h],
                                    [a, d, c, h],
                                    [a, c, g, h],
                                    [a, g, e, h],
                                    [a, e, f, h],
                                    [a, f, b, h],
                                ] {
                                    let mut determinant =
                                        tet_basis(&mesh.positions, tet).determinant();
                                    if determinant < 0.0 {
                                        tet.swap(1, 2);
                                        determinant = -determinant;
                                    }
                                    if !determinant.is_finite() || determinant <= 1e-18 {
                                        return Err(Error::Build(
                                            "degenerate drive lug tetrahedron".into(),
                                        ));
                                    }
                                    let volume = determinant / 6.0;
                                    for node in tet {
                                        mesh.masses[node as usize] += volume * p.density / 4.0;
                                    }
                                    mesh.tets.push(tet);
                                    mesh.volumes.push(volume);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(mesh)
    }

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
            || !p.thickness_cells.is_multiple_of(2)
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
                let left = self.positions[index(s, w.saturating_sub(1))].x;
                let right = self.positions[index(s, (w + 1).min(p.width_cells))].x;
                let weight = (right - left) / (2.0 * p.width);
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
    fn installed_lugged_belt_is_connected_watertight_and_noninverted() {
        use super::path::{BeltPath, BeltPulley};
        let path = BeltPath::around_pulleys(&[
            BeltPulley {
                center: Vec3::new(0.0, -0.52, -0.11),
                radius: 0.144,
            },
            BeltPulley {
                center: Vec3::new(0.0, 0.54, -0.11),
                radius: 0.144,
            },
            BeltPulley {
                center: Vec3::new(0.0, 0.405, 0.194),
                radius: 0.133,
            },
        ])
        .unwrap();
        let base = RubberBeltMesh::circular(RubberBeltParams {
            radius: path.length() / std::f64::consts::TAU,
            segments: 112,
            width_cells: 4,
            ..Default::default()
        })
        .unwrap();
        let mesh = base
            .with_drive_lugs(RubberLugParams {
                count: 56,
                segment_span: 1,
                width_start: 1,
                width_cells: 2,
                depth: 0.012,
                depth_cells: 1,
            })
            .unwrap();
        assert!(mesh.positions.len() > base.positions.len());
        assert!(mesh.tets.len() > base.tets.len());
        let mut adjacency = vec![Vec::new(); mesh.positions.len()];
        for tet in &mesh.tets {
            for &a in tet {
                for &b in tet {
                    adjacency[a as usize].push(b);
                }
            }
        }
        let mut visited = vec![false; mesh.positions.len()];
        let mut stack = vec![0u32];
        while let Some(i) = stack.pop() {
            if visited[i as usize] {
                continue;
            }
            visited[i as usize] = true;
            stack.extend(&adjacency[i as usize]);
        }
        assert!(visited.iter().all(|v| *v));
        let mut edges = HashMap::new();
        for tri in crate::compute_tet_surface_triangles(&mesh.tets) {
            for mut edge in [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]] {
                edge.sort();
                *edges.entry(edge).or_insert(0) += 1;
            }
        }
        assert!(
            edges.values().all(|&v| v == 2),
            "non-manifold lug/base seam"
        );
        let volume: Real = mesh.volumes.iter().sum();
        assert!((mesh.masses.iter().sum::<Real>() - volume * base.params.density).abs() < 1e-10);
        for phase in [0.0, 0.015, path.length() * 0.37] {
            let installed = mesh
                .installed_positions(&path, phase, Vec3::new(0.4, 0.12, 0.3))
                .unwrap();
            let ratio = mesh.minimum_volume_ratio(&installed).unwrap();
            assert!(ratio > 0.75, "installed element collapse {ratio}");
        }
        let mut builder = ModelBuilder::new();
        let handles = mesh
            .append_reinforced_to(
                &mut builder,
                WorldId(0),
                Vec3::ZERO,
                RubberCordParams::default(),
            )
            .unwrap();
        let model = builder.build().unwrap();
        assert_eq!(handles.nodes.len(), mesh.positions.len());
        assert!(model.particle_mass.host().unwrap().iter().all(|m| *m > 0.0));
        assert!(
            base.with_drive_lugs(RubberLugParams {
                count: usize::MAX,
                segment_span: 1,
                width_start: 0,
                width_cells: 1,
                depth: 0.01,
                depth_cells: 1
            })
            .is_err()
        );
    }

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
