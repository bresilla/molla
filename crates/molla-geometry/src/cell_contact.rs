//! Discrete convex-cell SAT and clipped, four-point contact manifolds.

use crate::{contact_reduction, convex_cut::ConvexCell};
use molla_core::{Error, Result};
use molla_math::{Real, Transform, Vec3};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug)]
pub struct CellContactPoint {
    pub a: Vec3,
    pub b: Vec3,
}

#[derive(Clone, Debug)]
pub struct CellManifold {
    /// World-space unit normal from A towards B.
    pub normal: Vec3,
    /// World-space surface witnesses, reduced to at most four points.
    pub points: Vec<CellContactPoint>,
}

pub fn face_normal(cell: &ConvexCell, face: usize) -> Vec3 {
    let indices = &cell.faces()[face].vertices;
    let origin = cell.vertices()[indices[0] as usize];
    let sum: Vec3 = (1..indices.len() - 1)
        .map(|i| {
            (cell.vertices()[indices[i] as usize] - origin)
                .cross(cell.vertices()[indices[i + 1] as usize] - origin)
        })
        .sum();
    sum.normalize()
}

pub fn edges(cell: &ConvexCell) -> Vec<[u32; 2]> {
    let mut edges = BTreeSet::new();
    for face in cell.faces() {
        for i in 0..face.vertices.len() {
            let a = face.vertices[i];
            let b = face.vertices[(i + 1) % face.vertices.len()];
            edges.insert([a.min(b), a.max(b)]);
        }
    }
    edges.into_iter().collect()
}

fn interval(points: &[Vec3], n: Vec3) -> (Real, Real) {
    points
        .iter()
        .fold((Real::INFINITY, Real::NEG_INFINITY), |(lo, hi), p| {
            let d = p.dot(n);
            (lo.min(d), hi.max(d))
        })
}

fn clip(points: &[Vec3], origin: Vec3, outward: Vec3) -> Vec<Vec3> {
    let mut result = Vec::new();
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        let da = (a - origin).dot(outward);
        let db = (b - origin).dot(outward);
        if da <= 0.0 {
            result.push(a);
        }
        if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
            result.push(a.lerp(b, da / (da - db)));
        }
    }
    result
}

fn support_segment(points: &[Vec3], n: Vec3, tolerance: Real) -> [Vec3; 2] {
    let (_, extreme) = interval(points, n);
    let candidates: Vec<_> = points
        .iter()
        .copied()
        .filter(|p| extreme - p.dot(n) <= tolerance)
        .collect();
    let mut segment = [candidates[0]; 2];
    let mut length = 0.0;
    for &a in &candidates {
        for &b in &candidates {
            let distance = a.distance_squared(b);
            if distance > length {
                length = distance;
                segment = [a, b];
            }
        }
    }
    segment
}

fn segment_witnesses(a: [Vec3; 2], b: [Vec3; 2]) -> CellContactPoint {
    let u = a[1] - a[0];
    let v = b[1] - b[0];
    let r = a[0] - b[0];
    let aa = u.length_squared();
    let bb = v.length_squared();
    let ab = u.dot(v);
    let ar = u.dot(r);
    let br = v.dot(r);
    let determinant = aa * bb - ab * ab;
    let mut s = if determinant > 1e-12 * aa * bb {
        ((ab * br - ar * bb) / determinant).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = if bb > 0.0 { (ab * s + br) / bb } else { 0.0 };
    if t < 0.0 {
        t = 0.0;
        s = if aa > 0.0 {
            (-ar / aa).clamp(0.0, 1.0)
        } else {
            0.0
        };
    } else if t > 1.0 {
        t = 1.0;
        s = if aa > 0.0 {
            ((ab - ar) / aa).clamp(0.0, 1.0)
        } else {
            0.0
        };
    } else if bb == 0.0 && aa > 0.0 {
        s = (-ar / aa).clamp(0.0, 1.0);
    }
    CellContactPoint {
        a: a[0] + u * s,
        b: b[0] + v * t,
    }
}

/// Discrete surface contact. `margin` admits nearby witnesses without inflating either cell.
pub fn cell_manifold(
    a: &ConvexCell,
    pose_a: Transform,
    b: &ConvexCell,
    pose_b: Transform,
    margin: Real,
) -> Result<Option<CellManifold>> {
    for pose in [pose_a, pose_b] {
        if !pose.position.is_finite()
            || !pose.rotation.is_finite()
            || (pose.rotation.length_squared() - 1.0).abs() > 1e-8
        {
            return Err(Error::Build("cell contact: invalid pose".into()));
        }
    }
    if !margin.is_finite() || margin < 0.0 {
        return Err(Error::Build("cell contact: invalid margin".into()));
    }
    let cells = [a, b];
    let poses = [pose_a, pose_b];
    let origin = pose_a.position;
    let vertices = [0, 1].map(|side| {
        cells[side]
            .vertices()
            .iter()
            .map(|&v| poses[side].rotation * v + (poses[side].position - origin))
            .collect::<Vec<_>>()
    });
    let scale = vertices
        .iter()
        .flatten()
        .map(|v| v.length())
        .fold(0.0, Real::max);
    let tolerance = 1e-6 * scale;
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        let (alo, ahi) = interval(&vertices[0], axis);
        let (blo, bhi) = interval(&vertices[1], axis);
        if blo - ahi > margin || alo - bhi > margin {
            return Ok(None);
        }
    }
    let mut best = Real::NEG_INFINITY;
    let mut normal = Vec3::ZERO;
    let mut feature = None;
    for side in 0..2 {
        for face in 0..cells[side].faces().len() {
            let n = poses[side].rotation * face_normal(cells[side], face);
            let p = vertices[side][cells[side].faces()[face].vertices[0] as usize];
            let (lo, _) = interval(&vertices[1 - side], n);
            let separation = lo - n.dot(p);
            if separation > margin {
                return Ok(None);
            }
            if separation > best + tolerance {
                best = separation;
                normal = if side == 0 { n } else { -n };
                feature = Some((side, face));
            }
        }
    }
    for ea in edges(a) {
        let u = (vertices[0][ea[1] as usize] - vertices[0][ea[0] as usize]).normalize();
        for eb in edges(b) {
            let v = (vertices[1][eb[1] as usize] - vertices[1][eb[0] as usize]).normalize();
            let cross = u.cross(v);
            if cross.length_squared() < 1e-12 {
                continue;
            }
            let mut n = cross.normalize();
            let (alo, ahi) = interval(&vertices[0], n);
            let (blo, bhi) = interval(&vertices[1], n);
            let mut separation = blo - ahi;
            if alo - bhi > separation {
                separation = alo - bhi;
                n = -n;
            }
            if separation > margin {
                return Ok(None);
            }
            if separation > best + tolerance {
                best = separation;
                normal = n;
                feature = None;
            }
        }
    }
    let mut points = Vec::new();
    if let Some((side, face)) = feature {
        let n = if side == 0 { normal } else { -normal };
        let incident = (0..cells[1 - side].faces().len())
            .min_by(|&i, &j| {
                let d = |f| (poses[1 - side].rotation * face_normal(cells[1 - side], f)).dot(n);
                d(i).total_cmp(&d(j))
            })
            .unwrap();
        let reference: Vec<_> = cells[side].faces()[face]
            .vertices
            .iter()
            .map(|&i| vertices[side][i as usize])
            .collect();
        let mut polygon: Vec<_> = cells[1 - side].faces()[incident]
            .vertices
            .iter()
            .map(|&i| vertices[1 - side][i as usize])
            .collect();
        for i in 0..reference.len() {
            let p = reference[i];
            let outward = (reference[(i + 1) % reference.len()] - p).cross(n);
            polygon = clip(&polygon, p, outward);
        }
        for p in polygon {
            let distance = (p - reference[0]).dot(n);
            if distance <= margin {
                let q = p - distance * n;
                points.push(if side == 0 {
                    CellContactPoint { a: q, b: p }
                } else {
                    CellContactPoint { a: p, b: q }
                });
            }
        }
    } else {
        points.push(segment_witnesses(
            support_segment(&vertices[0], normal, tolerance),
            support_segment(&vertices[1], -normal, tolerance),
        ));
    }
    if points.is_empty() {
        return Ok(None);
    }
    let contacts: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(i, p)| contact_reduction::Contact {
            a_shape: 0,
            b_shape: 1,
            id: i as u64,
            point: (p.a + p.b) * 0.5,
            normal,
            depth: (p.a - p.b).dot(normal),
        })
        .collect();
    let reduced = contact_reduction::reduce_contacts_tolerant(&contacts, 4, Some(tolerance));
    let points = reduced
        .into_iter()
        .map(|c| {
            let p = points[c.id as usize];
            CellContactPoint {
                a: p.a + origin,
                b: p.b + origin,
            }
        })
        .collect();
    Ok(Some(CellManifold { normal, points }))
}
