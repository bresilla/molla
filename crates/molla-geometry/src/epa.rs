//! Expanding Polytope Algorithm — given a 4-vertex GJK simplex that
//! contains the origin, EPA grows the simplex by repeatedly finding
//! the polytope face closest to the origin and pushing a new support
//! point past it. Termination yields the penetration depth + contact
//! normal.
//!
//! Reference: van den Bergen, *Proximity Queries and Penetration
//! Depth Computation on 3D Game Objects* (2001).

use molla_math::{Real, Transform, Vec3};
use molla_sim::ShapeGeometry;

use crate::gjk::{GjkSimplex, SupportPoint};
use crate::support::{support, support_with_hulls};

/// Result of EPA: the penetration depth and the contact normal
/// (pointing from A into B), plus closest points on each shape.
#[derive(Copy, Clone, Debug)]
pub struct EpaContact {
    /// Closest point on shape A (world frame).
    pub point_a: Vec3,
    /// Closest point on shape B (world frame).
    pub point_b: Vec3,
    /// Normal pointing from A into B (unit vector). XPBD's
    /// "contact normal a→b" convention.
    pub normal: Vec3,
    /// Penetration depth (positive).
    pub depth: Real,
}

/// One face of the EPA polytope: indices into `vertices` plus the
/// face plane's outward normal and distance from origin.
#[derive(Copy, Clone, Debug)]
struct Face {
    /// Vertex indices, oriented so `(b−a)×(c−a)` points away from
    /// the polytope interior.
    indices: [usize; 3],
    /// Outward unit normal.
    normal: Vec3,
    /// Distance from origin to the plane along `normal`.
    distance: Real,
}

/// Run EPA. Requires the GJK simplex to be a 4-vertex tetra
/// containing the origin.
pub fn epa(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    simplex: &GjkSimplex,
) -> Option<EpaContact> {
    epa_with_hulls(geom_a, xf_a, geom_b, xf_b, simplex, &[])
}

/// `epa` variant with access to a hull table.
pub fn epa_with_hulls(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    simplex: &GjkSimplex,
    hulls: &[Vec<Vec3>],
) -> Option<EpaContact> {
    let mut vertices: Vec<SupportPoint> = simplex.points.clone();
    // GJK may terminate with a 1, 2, or 3-vertex simplex when the
    // Minkowski difference's closest feature passes through the
    // origin (radially-symmetric shapes do this). EPA needs a 3D
    // tetra strictly containing the origin. The simplex-padding
    // strategy can fail on degenerate symmetries (e.g. two
    // co-linear sphere centres → simplex on the X axis), so when
    // the polytope can't be built from the simplex alone we fall
    // back to a fresh 4-vertex tetra built from cardinal-axis
    // support points.
    let faces = build_initial_faces_or_fallback(&mut vertices, geom_a, xf_a, geom_b, xf_b, hulls)?;
    let mut faces = faces;

    const MAX_ITERS: usize = 64;
    const TOL: Real = 1e-6;

    for _ in 0..MAX_ITERS {
        // Pick the face nearest to the origin.
        let (face_idx, face) = nearest_face(&faces);

        // Get a new support point past that face.
        let new = compute_support(geom_a, xf_a, geom_b, xf_b, hulls, face.normal);

        // Distance from origin to the new support point along the
        // face normal — i.e. how much further the polytope can
        // extend in that direction.
        let dist_new = new.w.dot(face.normal);
        if dist_new - face.distance < TOL {
            // Converged — the face is on the polytope boundary.
            return Some(extract_contact(&vertices, &faces[face_idx]));
        }

        // Expand: remove all faces whose outward normal can "see"
        // the new vertex (i.e. new is in front of them), collect
        // the silhouette edges, and re-stitch a fan from the new
        // vertex.
        let new_idx = vertices.len();
        vertices.push(new);

        let mut visible_faces: Vec<usize> = Vec::new();
        for (i, f) in faces.iter().enumerate() {
            let v = new.w - vertices[f.indices[0]].w;
            if f.normal.dot(v) > 0.0 {
                visible_faces.push(i);
            }
        }
        if visible_faces.is_empty() {
            // Numerical issue — return what we have.
            return Some(extract_contact(&vertices, &faces[face_idx]));
        }

        // Collect silhouette as DIRECTED edges from visible faces.
        // An edge (a→b) is in exactly one visible face iff its
        // canonical form doesn't appear elsewhere.
        let mut directed: Vec<(usize, usize)> = Vec::new();
        for &fi in &visible_faces {
            let f = &faces[fi];
            for k in 0..3 {
                let a = f.indices[k];
                let b = f.indices[(k + 1) % 3];
                // Cancel against any reverse edge from an adjacent
                // visible face (b→a means the two faces shared this
                // edge — interior to the visible region, not silhouette).
                if let Some(p) = directed.iter().position(|&(x, y)| x == b && y == a) {
                    directed.remove(p);
                } else {
                    directed.push((a, b));
                }
            }
        }

        // Remove visible faces (back-to-front so indices stay valid).
        let mut visible_sorted = visible_faces.clone();
        visible_sorted.sort_unstable_by(|a, b| b.cmp(a));
        for i in visible_sorted {
            faces.swap_remove(i);
        }

        // Add new fan faces. Each silhouette edge (a→b) becomes a
        // new triangle (a, b, new_idx) — the original face's outward
        // winding plus the new vertex naturally orients the new
        // triangle outward.
        for &(a, b) in &directed {
            let va = vertices[a].w;
            let vb = vertices[b].w;
            let v_new = vertices[new_idx].w;
            let n_raw = (vb - va).cross(v_new - va);
            let len = n_raw.length();
            if len < 1e-12 {
                continue; // degenerate
            }
            let n = n_raw / len;
            let dist = va.dot(n).max(0.0);
            faces.push(Face {
                indices: [a, b, new_idx],
                normal: n,
                distance: dist,
            });
        }
    }

    // Iter cap: return current best face.
    let (idx, _) = nearest_face(&faces);
    Some(extract_contact(&vertices, &faces[idx]))
}

/// Build EPA's initial polytope, falling back to a fresh
/// cardinal-axis tetra if the simplex pad fails or produces a
/// degenerate (origin-on-face) configuration.
fn build_initial_faces_or_fallback(
    vertices: &mut Vec<SupportPoint>,
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    hulls: &[Vec<Vec3>],
) -> Option<Vec<Face>> {
    if pad_to_tetra(vertices, geom_a, xf_a, geom_b, xf_b, hulls)
        && let Some(f) = build_initial_faces(vertices)
    {
        // Reject if origin lies on any face (face.distance == 0).
        // That signals a degenerate / coplanar tetra that EPA
        // can't escape from.
        if f.iter().all(|face| face.distance > 1e-6) {
            return Some(f);
        }
    }
    // Fallback: 4 cardinal-axis support points.
    vertices.clear();
    let dirs = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
    ];
    for d in dirs {
        vertices.push(compute_support(geom_a, xf_a, geom_b, xf_b, hulls, d));
    }
    build_initial_faces(vertices)
}

/// Pad a 1, 2, or 3-vertex simplex up to a 4-vertex tetrahedron by
/// querying support points along axes orthogonal to the existing
/// simplex's affine span. Returns `false` if the shapes are too
/// degenerate for EPA to make progress (rare — flat shapes /
/// zero-volume Minkowski difference).
fn pad_to_tetra(
    vs: &mut Vec<SupportPoint>,
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    hulls: &[Vec<Vec3>],
) -> bool {
    if vs.is_empty() {
        return false;
    }
    if vs.len() == 1 {
        // Pick an arbitrary direction and find a second point on the
        // opposite side of the origin.
        let axes = [Vec3::X, Vec3::Y, Vec3::Z];
        for d in axes {
            let p = compute_support(geom_a, xf_a, geom_b, xf_b, hulls, d);
            if (p.w - vs[0].w).length_squared() > 1e-12 {
                vs.push(p);
                break;
            }
        }
        if vs.len() < 2 {
            return false;
        }
    }
    if vs.len() == 2 {
        // The segment passes through origin (GJK guarantee for the
        // 2-vertex overlap case). Find a direction perpendicular to
        // it; query support both ways and pick the side whose
        // support projects further in that direction (so the
        // resulting triangle has finite area in the perpendicular
        // direction).
        let seg = vs[1].w - vs[0].w;
        let len = seg.length();
        if len < 1e-12 {
            return false;
        }
        let s = seg / len;
        let (ax, ay, az) = (s.x.abs(), s.y.abs(), s.z.abs());
        let other = if ax <= ay && ax <= az {
            Vec3::X
        } else if ay <= az {
            Vec3::Y
        } else {
            Vec3::Z
        };
        let perp = s.cross(other).normalize_or_zero();
        if perp.length_squared() < 1e-12 {
            return false;
        }
        // For the 3-vertex triangle to "contain" the origin's
        // projection in 2D, both perpendicular sides matter — but
        // only one fits in the simplex. Pick the side whose support
        // is further along that direction; the 4th-vertex padding
        // (next branch) puts a point on the other side perpendicular
        // to the triangle plane, which together usually wraps the
        // origin.
        let p_pos = compute_support(geom_a, xf_a, geom_b, xf_b, hulls, perp);
        let p_neg = compute_support(geom_a, xf_a, geom_b, xf_b, hulls, -perp);
        let proj_pos = (p_pos.w - vs[0].w).dot(perp);
        let proj_neg = (p_neg.w - vs[0].w).dot(-perp);
        if proj_pos >= proj_neg {
            vs.push(p_pos);
        } else {
            vs.push(p_neg);
        }
        if (vs[2].w - vs[0].w).cross(seg).length() < 1e-9 {
            return false;
        }
    }
    if vs.len() == 3 {
        // Find direction perpendicular to the triangle plane and
        // pick the support point on the side that puts the origin
        // INSIDE the resulting tetra. If the origin's signed
        // distance from the triangle plane is `s`, the 4th vertex
        // must be on the opposite-`s` side.
        let n_raw = (vs[1].w - vs[0].w).cross(vs[2].w - vs[0].w);
        let len = n_raw.length();
        if len < 1e-12 {
            return false;
        }
        let n = n_raw / len;
        // Origin's signed distance from triangle plane:
        //   d_origin = (0 - vs[0].w) · n = -vs[0].w · n
        let d_origin = -vs[0].w.dot(n);
        // Pick a search direction whose support places v3 on the
        // OPPOSITE side of the triangle from the origin.
        let pick = if d_origin > 0.0 { -n } else { n };
        let p = compute_support(geom_a, xf_a, geom_b, xf_b, hulls, pick);
        // Verify v3 lies on the expected side (Minkowski-diff might
        // not be thick enough — bail if it doesn't).
        let v3_signed = (p.w - vs[0].w).dot(n);
        let opposite_to_origin = if d_origin > 0.0 {
            v3_signed < -1e-9
        } else {
            v3_signed > 1e-9
        };
        if !opposite_to_origin {
            // Try the other direction as a last resort — origin may
            // happen to lie exactly on the triangle plane, in which
            // case either side works.
            let p_alt = compute_support(geom_a, xf_a, geom_b, xf_b, hulls, -pick);
            let alt_signed = (p_alt.w - vs[0].w).dot(n);
            if alt_signed.abs() < 1e-9 {
                return false;
            }
            vs.push(p_alt);
        } else {
            vs.push(p);
        }
    }
    true
}

fn nearest_face(faces: &[Face]) -> (usize, Face) {
    let mut best_idx = 0;
    let mut best_dist = Real::INFINITY;
    for (i, f) in faces.iter().enumerate() {
        if f.distance < best_dist {
            best_dist = f.distance;
            best_idx = i;
        }
    }
    (best_idx, faces[best_idx])
}

#[inline]
fn compute_support(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    hulls: &[Vec<Vec3>],
    dir: Vec3,
) -> SupportPoint {
    let a = support_with_hulls(geom_a, xf_a, dir, hulls);
    let b = support_with_hulls(geom_b, xf_b, -dir, hulls);
    SupportPoint { w: a - b, a, b }
}

#[allow(dead_code)]
fn _suppress_unused_support_warning(g: ShapeGeometry, xf: &Transform, d: Vec3) -> Vec3 {
    support(g, xf, d)
}

/// Build the initial 4-face polytope from a tetra simplex,
/// orienting each face so its outward normal points away from the
/// 4th vertex (i.e., away from the polytope interior). Returns
/// `None` if any face's outward plane contains the origin (with the
/// same orientation), which indicates a degenerate polytope EPA
/// can't escape from.
fn build_initial_faces(vertices: &[SupportPoint]) -> Option<Vec<Face>> {
    if vertices.len() < 4 {
        return None;
    }
    let combos: [([usize; 3], usize); 4] = [
        ([0, 1, 2], 3),
        ([0, 2, 3], 1),
        ([0, 3, 1], 2),
        ([1, 3, 2], 0),
    ];
    let mut faces = Vec::with_capacity(4);
    for (mut idx, opp) in combos {
        let v0 = vertices[idx[0]].w;
        let v1 = vertices[idx[1]].w;
        let v2 = vertices[idx[2]].w;
        let v_opp = vertices[opp].w;
        let mut n = (v1 - v0).cross(v2 - v0);
        let len = n.length();
        if len < 1e-12 {
            return None;
        }
        n /= len;
        if n.dot(v_opp - v0) > 0.0 {
            n = -n;
            idx.swap(1, 2);
        }
        let distance = v0.dot(n);
        faces.push(Face {
            indices: idx,
            normal: n,
            distance: distance.max(0.0),
        });
    }
    Some(faces)
}

/// Extract closest-points on each shape from the EPA face's
/// barycentrics, plus the contact normal + depth.
fn extract_contact(vertices: &[SupportPoint], face: &Face) -> EpaContact {
    // Project the origin onto the face plane to get barycentric
    // weights, then mix `point_a` and `point_b` by those weights.
    let v0 = vertices[face.indices[0]];
    let v1 = vertices[face.indices[1]];
    let v2 = vertices[face.indices[2]];
    let proj = face.normal * face.distance;
    let (u, v, w) = barycentric(v0.w, v1.w, v2.w, proj);
    let pa = v0.a * u + v1.a * v + v2.a * w;
    let pb = v0.b * u + v1.b * v + v2.b * w;
    EpaContact {
        point_a: pa,
        point_b: pb,
        // EPA's face normal points OUT of the polytope, which in
        // Minkowski space `A − B` is `A → B` direction. Matches
        // molla's contact-normal convention.
        normal: face.normal,
        depth: face.distance.max(0.0),
    }
}

fn barycentric(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> (Real, Real, Real) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-24 {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

#[cfg(test)]
mod tests {
    use super::super::gjk::{GjkResult, gjk};
    use super::*;

    #[test]
    fn epa_two_overlapping_spheres() {
        // A: r=1 at origin, B: r=1 at (1.5, 0, 0). Centres 1.5 apart,
        // sum radii 2 → overlap = 0.5 along +X. Normal a→b = +X.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(1.5, 0.0, 0.0));
        let res = gjk(
            ShapeGeometry::Sphere { radius: 1.0 },
            &xf_a,
            ShapeGeometry::Sphere { radius: 1.0 },
            &xf_b,
        );
        let simplex = match res {
            GjkResult::Overlap(s) => s,
            _ => panic!("expected overlap"),
        };
        let contact = epa(
            ShapeGeometry::Sphere { radius: 1.0 },
            &xf_a,
            ShapeGeometry::Sphere { radius: 1.0 },
            &xf_b,
            &simplex,
        )
        .expect("epa");
        assert!(
            (contact.depth - 0.5).abs() < 5e-2,
            "depth: got {} expected 0.5",
            contact.depth
        );
        assert!(
            (contact.normal - Vec3::X).length() < 5e-2,
            "normal: got {:?}",
            contact.normal
        );
    }

    #[test]
    fn epa_box_box_overlap_axis_aligned() {
        // Box A at origin half=1, box B at (1.5, 0, 0) half=1.
        // Overlap along +X = 0.5. Normal a→b = +X.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(1.5, 0.0, 0.0));
        let res = gjk(
            ShapeGeometry::Box {
                half_extents: Vec3::splat(1.0),
            },
            &xf_a,
            ShapeGeometry::Box {
                half_extents: Vec3::splat(1.0),
            },
            &xf_b,
        );
        let simplex = match res {
            GjkResult::Overlap(s) => s,
            _ => panic!("expected overlap"),
        };
        let contact = epa(
            ShapeGeometry::Box {
                half_extents: Vec3::splat(1.0),
            },
            &xf_a,
            ShapeGeometry::Box {
                half_extents: Vec3::splat(1.0),
            },
            &xf_b,
            &simplex,
        )
        .expect("epa");
        assert!(
            (contact.depth - 0.5).abs() < 5e-2,
            "depth: got {}",
            contact.depth
        );
        // Normal should be along ±X (face axis).
        assert!(
            contact.normal.x.abs() > 0.95,
            "normal: got {:?}",
            contact.normal
        );
    }

    #[test]
    fn epa_capsule_box_overlap() {
        // Capsule along Y at origin, half_h=0.5 r=0.4. Box at (0.5,0,0)
        // half=0.5. Capsule reaches +X to x=0.4; box -X face at x=0.
        // Overlap along +X = 0.4.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(0.5, 0.0, 0.0));
        let geom_a = ShapeGeometry::Capsule {
            half_height: 0.5,
            radius: 0.4,
        };
        let geom_b = ShapeGeometry::Box {
            half_extents: Vec3::splat(0.5),
        };
        let res = gjk(geom_a, &xf_a, geom_b, &xf_b);
        let simplex = match res {
            GjkResult::Overlap(s) => s,
            _ => panic!("expected overlap"),
        };
        let contact = epa(geom_a, &xf_a, geom_b, &xf_b, &simplex).expect("epa");
        assert!(
            (contact.depth - 0.4).abs() < 5e-2,
            "depth: got {} expected ~0.4",
            contact.depth
        );
        assert!(
            contact.normal.x > 0.9,
            "normal should point +X, got {:?}",
            contact.normal
        );
    }
}
