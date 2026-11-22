//! Closed convex fiber cells and plane-aligned cut surfaces.

use molla_core::{Error, Result};
use molla_math::{Mat3, Real, Transform, Vec3};
use molla_sim::MassProperties;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceOrigin {
    Exterior,
    Interface(u64),
    Cut { id: u64, positive_side: bool },
    Fracture { id: u64, positive_side: bool },
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellFace {
    pub vertices: Vec<u32>,
    pub origin: FaceOrigin,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvexCell {
    vertices: Vec<Vec3>,
    faces: Vec<CellFace>,
}

#[derive(Clone, Copy, Debug)]
pub struct CutPlane {
    point: Vec3,
    normal: Vec3,
}

fn invalid(message: &str) -> Error {
    Error::Build(format!("convex cut: {message}"))
}

impl CutPlane {
    pub fn new(point: Vec3, normal: Vec3) -> Result<Self> {
        let scale = normal.abs().max_element();
        if !point.is_finite() || !normal.is_finite() || scale == 0.0 {
            return Err(invalid("nonfinite or zero plane"));
        }
        Ok(Self {
            point,
            normal: (normal / scale).normalize(),
        })
    }

    pub fn point(self) -> Vec3 {
        self.point
    }
    pub fn normal(self) -> Vec3 {
        self.normal
    }
    pub fn distance(self, p: Vec3) -> Real {
        (p - self.point).dot(self.normal)
    }

    pub fn to_local(self, pose: Transform) -> Result<Self> {
        validate_pose(pose)?;
        let inverse = pose.rotation.conjugate();
        Self::new(
            inverse * (self.point - pose.position),
            inverse * self.normal,
        )
    }
}

fn validate_pose(pose: Transform) -> Result<()> {
    if !pose.position.is_finite()
        || !pose.rotation.is_finite()
        || (pose.rotation.length_squared() - 1.0).abs() > 1.0e-10
    {
        return Err(invalid("invalid rigid pose"));
    }
    Ok(())
}

impl ConvexCell {
    pub fn new(vertices: Vec<Vec3>, faces: Vec<CellFace>) -> Result<Self> {
        let cell = Self { vertices, faces };
        cell.validate()?;
        Ok(cell)
    }

    /// Regular polygonal fiber volume along local Z, centered at the origin.
    pub fn prism(radius: Real, length: Real, sides: usize) -> Result<Self> {
        if !radius.is_finite()
            || !length.is_finite()
            || radius <= 0.0
            || length <= 0.0
            || !(3..=256).contains(&sides)
        {
            return Err(invalid("invalid fiber dimensions"));
        }
        let mut vertices = Vec::with_capacity(2 * sides);
        for z in [-0.5 * length, 0.5 * length] {
            for i in 0..sides {
                let angle = std::f64::consts::TAU * i as Real / sides as Real;
                vertices.push(Vec3::new(radius * angle.cos(), radius * angle.sin(), z));
            }
        }
        let face = |indices| CellFace {
            vertices: indices,
            origin: FaceOrigin::Exterior,
        };
        let mut faces = vec![
            face((0..sides as u32).rev().collect()),
            face((sides as u32..2 * sides as u32).collect()),
        ];
        for i in 0..sides {
            let j = (i + 1) % sides;
            faces.push(face(vec![
                i as u32,
                j as u32,
                (j + sides) as u32,
                (i + sides) as u32,
            ]));
        }
        Self::new(vertices, faces)
    }

    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }
    pub fn faces(&self) -> &[CellFace] {
        &self.faces
    }

    pub fn triangles(&self) -> Vec<[u32; 3]> {
        self.faces
            .iter()
            .flat_map(|f| {
                (1..f.vertices.len() - 1).map(|i| [f.vertices[0], f.vertices[i], f.vertices[i + 1]])
            })
            .collect()
    }

    fn scale(&self) -> Real {
        let lo = self
            .vertices
            .iter()
            .copied()
            .fold(Vec3::splat(Real::INFINITY), Vec3::min);
        let hi = self
            .vertices
            .iter()
            .copied()
            .fold(Vec3::splat(Real::NEG_INFINITY), Vec3::max);
        (hi - lo).max_element()
    }

    pub fn validate(&self) -> Result<()> {
        if self.vertices.len() < 4
            || self.vertices.len() > u32::MAX as usize
            || self.faces.len() < 4
            || self.vertices.iter().any(|p| !p.is_finite())
        {
            return Err(invalid("invalid cell size or vertices"));
        }
        let scale = self.scale();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid("degenerate cell extent"));
        }
        let origin = self.vertices[0];
        let points: Vec<_> = self
            .vertices
            .iter()
            .map(|p| (*p - origin) / scale)
            .collect();
        let center = points.iter().copied().sum::<Vec3>() / points.len() as Real;
        let mut edges: BTreeMap<(u32, u32), (usize, i32)> = BTreeMap::new();
        let mut used = BTreeSet::new();
        let mut volume = 0.0;
        for face in &self.faces {
            let ids = &face.vertices;
            if ids.len() < 3
                || ids.iter().any(|&i| i as usize >= points.len())
                || ids.iter().copied().collect::<BTreeSet<_>>().len() != ids.len()
            {
                return Err(invalid("invalid face indices"));
            }
            let a = points[ids[0] as usize];
            let area = (1..ids.len() - 1)
                .map(|i| (points[ids[i] as usize] - a).cross(points[ids[i + 1] as usize] - a))
                .sum::<Vec3>();
            if !area.is_finite() || area.length() <= 1.0e-24 {
                return Err(invalid("degenerate face"));
            }
            let normal = area.normalize();
            if points.iter().any(|p| normal.dot(*p - a) > 2.0e-10)
                || ids
                    .iter()
                    .any(|&i| normal.dot(points[i as usize] - a).abs() > 2.0e-10)
            {
                return Err(invalid("nonconvex, nonplanar or inward face"));
            }
            for i in 0..ids.len() {
                let (u, v, w) = (ids[i], ids[(i + 1) % ids.len()], ids[(i + 2) % ids.len()]);
                let edge = points[v as usize] - points[u as usize];
                if edge.length_squared() < 1e-28
                    || ids.iter().any(|&p| {
                        normal.dot(edge.cross(points[p as usize] - points[u as usize])) < -2e-12
                    })
                {
                    return Err(invalid("degenerate or crossing face boundary"));
                }
                if normal.dot(
                    (points[v as usize] - points[u as usize])
                        .cross(points[w as usize] - points[v as usize]),
                ) < -2.0e-12
                {
                    return Err(invalid("nonconvex face boundary"));
                }
                let entry = edges.entry((u.min(v), u.max(v))).or_default();
                entry.0 += 1;
                entry.1 += if u < v { 1 } else { -1 };
                used.insert(u);
            }
            for i in 1..ids.len() - 1 {
                volume += (a - center).dot(
                    (points[ids[i] as usize] - center).cross(points[ids[i + 1] as usize] - center),
                ) / 6.0;
            }
        }
        if edges
            .values()
            .any(|&(count, winding)| count != 2 || winding != 0)
            || used.len() != points.len()
            || !volume.is_finite()
            || volume <= 1.0e-14
        {
            return Err(invalid("cell is not a closed positive-volume solid"));
        }
        Ok(())
    }

    pub fn mass_properties(&self, density: Real) -> Result<MassProperties> {
        if !density.is_finite() || density <= 0.0 {
            return Err(invalid("invalid density"));
        }
        let scale = self.scale();
        let origin = self.vertices[0];
        let center = self
            .vertices
            .iter()
            .map(|p| (*p - origin) / scale)
            .sum::<Vec3>()
            / self.vertices.len() as Real;
        let points: Vec<_> = self
            .vertices
            .iter()
            .map(|p| (*p - origin) / scale - center)
            .collect();
        let outer = |p: Vec3| Mat3::from_cols(p * p.x, p * p.y, p * p.z);
        let mut volume = 0.0;
        let mut first = Vec3::ZERO;
        let mut second = Mat3::ZERO;
        for [i, j, k] in self.triangles() {
            let (a, b, c) = (points[i as usize], points[j as usize], points[k as usize]);
            let v = a.dot(b.cross(c)) / 6.0;
            let sum = a + b + c;
            volume += v;
            first += sum * (v / 4.0);
            second += (outer(a) + outer(b) + outer(c) + outer(sum)) * (v / 20.0);
        }
        let com = first / volume;
        let covariance = second - outer(com) * volume;
        let trace = covariance.x_axis.x + covariance.y_axis.y + covariance.z_axis.z;
        let properties = MassProperties {
            mass: density * volume * scale.powi(3),
            local_com: origin + (center + com) * scale,
            inertia: (Mat3::from_diagonal(Vec3::splat(trace)) - covariance)
                * (density * scale.powi(5)),
        };
        properties.validate()?;
        if properties.mass <= 0.0 {
            return Err(invalid("nonpositive cell mass"));
        }
        Ok(properties)
    }

    /// Return negative/positive half-cells; tangent or disjoint planes return None.
    pub fn split(&self, plane: CutPlane, cut_id: u64) -> Result<Option<[Self; 2]>> {
        let mut distances: Vec<_> = self.vertices.iter().map(|&p| plane.distance(p)).collect();
        if distances.iter().any(|d| !d.is_finite()) {
            return Err(invalid("plane distance overflow"));
        }
        let mut vertices = self.vertices.clone();
        let roundoff = self.scale() * 64.0 * Real::EPSILON;
        for (point, distance) in vertices.iter_mut().zip(&mut distances) {
            if distance.abs() <= roundoff {
                *point -= plane.normal * *distance;
                *distance = 0.0;
            }
        }
        let lo = distances.iter().copied().fold(Real::INFINITY, Real::min);
        let hi = distances
            .iter()
            .copied()
            .fold(Real::NEG_INFINITY, Real::max);
        if lo >= 0.0 || hi <= 0.0 {
            return Ok(None);
        }
        let tolerance = self.scale() * 1.0e-10;
        if -lo < tolerance || hi < tolerance {
            return Err(invalid("cut creates an unresolved sliver"));
        }
        let mut crossings = BTreeMap::new();
        let mut ring: BTreeSet<u32> = distances
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == 0.0)
            .map(|(i, _)| i as u32)
            .collect();
        for face in &self.faces {
            for i in 0..face.vertices.len() {
                let a = face.vertices[i];
                let b = face.vertices[(i + 1) % face.vertices.len()];
                let da = distances[a as usize];
                let db = distances[b as usize];
                if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                    let key = (a.min(b), a.max(b));
                    if crossings.contains_key(&key) {
                        continue;
                    }
                    let t = da / (da - db);
                    let p = self.vertices[a as usize] * (1.0 - t) + self.vertices[b as usize] * t;
                    let p = p - plane.normal * plane.distance(p);
                    let index =
                        u32::try_from(vertices.len()).map_err(|_| invalid("cut index overflow"))?;
                    vertices.push(p);
                    crossings.insert(key, index);
                    ring.insert(index);
                }
            }
        }
        let mut ring: Vec<_> = ring.into_iter().collect();
        let center = ring
            .iter()
            .map(|&i| vertices[i as usize] / ring.len() as Real)
            .sum::<Vec3>();
        let helper = if plane.normal.x.abs() < 0.8 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let u = plane.normal.cross(helper).normalize();
        let v = plane.normal.cross(u);
        ring.sort_by(|&a, &b| {
            let angle = |i| {
                let p = vertices[i as usize] - center;
                p.dot(v).atan2(p.dot(u))
            };
            angle(a).total_cmp(&angle(b))
        });
        let clip = |positive_side: bool| -> Result<Self> {
            let mut faces = Vec::new();
            for face in &self.faces {
                let mut ids = Vec::new();
                for i in 0..face.vertices.len() {
                    let a = face.vertices[i];
                    let b = face.vertices[(i + 1) % face.vertices.len()];
                    let da = distances[a as usize];
                    if if positive_side { da >= 0.0 } else { da <= 0.0 } {
                        ids.push(a);
                    }
                    if let Some(&index) = crossings.get(&(a.min(b), a.max(b))) {
                        ids.push(index);
                    }
                }
                ids.dedup();
                if ids.len() > 1 && ids.first() == ids.last() {
                    ids.pop();
                }
                if ids.len() >= 3 {
                    faces.push(CellFace {
                        vertices: ids,
                        origin: face.origin,
                    });
                }
            }
            let mut cap = ring.clone();
            if positive_side {
                cap.reverse();
            }
            faces.push(CellFace {
                vertices: cap,
                origin: FaceOrigin::Cut {
                    id: cut_id,
                    positive_side,
                },
            });
            let used: BTreeSet<_> = faces
                .iter()
                .flat_map(|f| f.vertices.iter().copied())
                .collect();
            let map: BTreeMap<_, _> = used
                .iter()
                .enumerate()
                .map(|(new, &old)| (old, new as u32))
                .collect();
            for face in &mut faces {
                for id in &mut face.vertices {
                    *id = map[id];
                }
            }
            Self::new(used.iter().map(|&i| vertices[i as usize]).collect(), faces)
        };
        Ok(Some([clip(false)?, clip(true)?]))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RigidCellMotion {
    /// The pose of the cell's authored frame, not necessarily its center of mass.
    pub pose: Transform,
    /// World-space linear velocity of the center of mass.
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
}

impl RigidCellMotion {
    /// Restrict the parent's rigid velocity field to a child's center of mass.
    pub fn restricted(self, parent: MassProperties, child: MassProperties) -> Result<Self> {
        validate_pose(self.pose)?;
        parent.validate()?;
        child.validate()?;
        if !self.linear_velocity.is_finite() || !self.angular_velocity.is_finite() {
            return Err(invalid("nonfinite cut motion"));
        }
        let linear_velocity = self.linear_velocity
            + self
                .angular_velocity
                .cross(self.pose.rotation * (child.local_com - parent.local_com));
        if !linear_velocity.is_finite() {
            return Err(invalid("cut velocity overflow"));
        }
        Ok(Self {
            linear_velocity,
            ..self
        })
    }

    pub fn momentum(self, properties: MassProperties) -> Result<(Vec3, Vec3, Real)> {
        validate_pose(self.pose)?;
        properties.validate()?;
        let rotation = Mat3::from_quat(self.pose.rotation);
        let center = self.pose.position + self.pose.rotation * properties.local_com;
        let p = self.linear_velocity * properties.mass;
        let spin = rotation * properties.inertia * rotation.transpose() * self.angular_velocity;
        let l = center.cross(p) + spin;
        let kinetic = 0.5
            * (properties.mass * self.linear_velocity.length_squared()
                + self.angular_velocity.dot(spin));
        if !p.is_finite() || !l.is_finite() || !kinetic.is_finite() {
            return Err(invalid("nonfinite cell momentum"));
        }
        Ok((p, l, kinetic))
    }
}
