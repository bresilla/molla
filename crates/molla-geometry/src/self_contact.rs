//! CPU self-contact candidate generation for surface meshes.
//!
//! The detector is deliberately conservative and simple: every vertex
//! is tested against every non-adjacent triangle, and candidates are
//! emitted when the vertex lies within a caller-supplied thickness
//! shell. Higher-level solvers can feed the emitted plane/contact data
//! into Signorini or penalty contact.

use molla_math::{Real, Vec3};

use crate::sdf_bake::closest_point_on_triangle;

/// Vertex-vs-triangle self-contact candidate.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SelfContactCandidate {
    /// Slave vertex index.
    pub vertex: u32,
    /// Master triangle vertex indices.
    pub triangle: [u32; 3],
    /// Closest point on the master triangle.
    pub point: Vec3,
    /// Unit normal pointing from the master triangle toward the slave
    /// vertex. If the distance is numerically zero, uses the triangle
    /// geometric normal.
    pub normal: Vec3,
    /// Signed shell gap: `distance - thickness`. Negative means the
    /// vertex is inside the contact shell.
    pub gap: Real,
    /// Euclidean vertex-to-triangle distance.
    pub distance: Real,
}

/// Generate vertex-triangle self-contact candidates.
///
/// Adjacent vertex/triangle pairs are ignored, so a mesh does not
/// collide with its own incident faces. The routine is `O(VT)` and is
/// intended as the CPU oracle / small-mesh path; broad-phase
/// acceleration can layer on top without changing the output contract.
#[must_use]
pub fn detect_vertex_triangle_self_contacts(
    vertices: &[Vec3],
    triangles: &[[u32; 3]],
    thickness: Real,
) -> Vec<SelfContactCandidate> {
    if vertices.is_empty() || triangles.is_empty() || thickness < 0.0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let shell2 = thickness * thickness;
    for (vi, &p) in vertices.iter().enumerate() {
        let vi_u32 = vi as u32;
        for &tri in triangles {
            if tri.contains(&vi_u32) {
                continue;
            }
            let a = vertices[tri[0] as usize];
            let b = vertices[tri[1] as usize];
            let c = vertices[tri[2] as usize];
            let (d2, q) = closest_point_on_triangle(p, a, b, c);
            if d2 > shell2 {
                continue;
            }
            let distance = d2.sqrt();
            let face_n = (b - a).cross(c - a);
            let normal = if distance > 1.0e-12 {
                (p - q) / distance
            } else {
                let len = face_n.length();
                if len > 1.0e-12 { face_n / len } else { Vec3::Z }
            };
            out.push(SelfContactCandidate {
                vertex: vi_u32,
                triangle: tri,
                point: q,
                normal,
                gap: distance - thickness,
                distance,
            });
        }
    }
    out.sort_by(|a, b| {
        a.vertex
            .cmp(&b.vertex)
            .then_with(|| a.triangle.cmp(&b.triangle))
            .then_with(|| a.distance.partial_cmp(&b.distance).unwrap())
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_incident_faces_and_finds_nearby_remote_triangle() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.25, 0.25, 0.05),
        ];
        let triangles = vec![[0, 1, 2], [0, 1, 3]];
        let contacts = detect_vertex_triangle_self_contacts(&vertices, &triangles, 0.1);
        assert!(
            contacts
                .iter()
                .any(|c| c.vertex == 3 && c.triangle == [0, 1, 2])
        );
        assert!(
            !contacts
                .iter()
                .any(|c| c.vertex == 3 && c.triangle == [0, 1, 3])
        );
    }
}
