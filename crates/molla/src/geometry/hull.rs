//! 3D convex hull (QuickHull) for mesh-authoring tools.
//!
//! Stage-21 deliverable per `PLAN.md` §21 / `GAP.md` §4. Lets a user
//! pass an arbitrary point cloud (e.g. URDF mesh-collision vertices)
//! and get back a hull suitable for the existing
//! [`crate::sim::ShapeGeometry::ConvexHull`] shape.
//!
//! The implementation is the classic incremental QuickHull:
//!
//! 1. Pick four extreme points to seed an initial tetrahedron.
//! 2. Partition the remaining points by the face they sit "outside" of.
//! 3. Repeatedly pick the farthest point of any face's outside set,
//!    find every face that point can "see" (its `horizon`), delete
//!    them, and stitch the silhouette to the new apex.
//! 4. When every outside set is empty, the surviving faces form the
//!    hull.
//!
//! This is CPU-only authoring code that runs once at asset load (per
//! `feedback_gpu_first`'s carve-out for tools), not in the per-frame
//! hot path. Numerical robustness uses an ε scaled to the input
//! diameter — degenerate inputs (coplanar / colinear points) fall
//! back to a 2D hull projected on the dominant plane, or a zero-vol
//! line / point as a last resort.

use crate::math::{Real, Vec3};

/// Compute the convex hull of `points` and return the hull's
/// vertex list. Vertex ordering is not specified beyond "every
/// returned vertex sits on the hull surface and no interior point of
/// the input is included".
///
/// The result is suitable to pass directly to
/// [`crate::sim::ModelBuilder::add_hull`].
///
/// Empty / single-point inputs round-trip; two-point and degenerate
/// (colinear / coplanar) clouds fall back to whatever subset the
/// algorithm can recover (an interval, a 2D hull). Callers that need
/// to detect degeneracy should compare `points.len()` to the returned
/// length.
pub fn convex_hull(points: &[Vec3]) -> Vec<Vec3> {
    let n = points.len();
    if n < 4 {
        // <4 points → not enough for a tetrahedron. Return the unique
        // input vertices verbatim; the caller decides what to do.
        let mut out = Vec::with_capacity(n);
        for &p in points {
            if !out.iter().any(|q: &Vec3| (p - *q).length_squared() < 1e-24) {
                out.push(p);
            }
        }
        return out;
    }

    // ── Diameter-scaled tolerance ─────────────────────────────────
    let (mn, mx) = points.iter().fold(
        (Vec3::splat(Real::INFINITY), Vec3::splat(-Real::INFINITY)),
        |(a, b), p| (a.min(*p), b.max(*p)),
    );
    let diag = (mx - mn).length();
    let eps = (diag.max(1.0)) * 1.0e-10;

    // ── Seed tetrahedron ──────────────────────────────────────────
    let Some(seed) = seed_tetrahedron(points, eps) else {
        // Degenerate (coplanar). Fall back to a 2D hull on the
        // dominant axis-aligned projection so callers still get
        // *something* sensible.
        return planar_fallback(points, eps);
    };
    // Build initial faces (4 of them, one per omitted vertex). Wind
    // each so its normal points away from the centroid → outward.
    let centroid = (points[seed[0]] + points[seed[1]] + points[seed[2]] + points[seed[3]]) / 4.0;
    let mut faces: Vec<Face> = Vec::with_capacity(64);
    for &(a, b, c) in &[
        (seed[0], seed[1], seed[2]),
        (seed[0], seed[1], seed[3]),
        (seed[0], seed[2], seed[3]),
        (seed[1], seed[2], seed[3]),
    ] {
        faces.push(make_face(points, a, b, c, centroid));
    }

    // Partition remaining points among the four seed faces.
    for (i, p) in points.iter().enumerate() {
        if seed.contains(&i) {
            continue;
        }
        // Assign to the face whose outward halfspace this point sits
        // in. Points strictly inside the tetrahedron belong to no
        // face and are ignored.
        let mut best: Option<usize> = None;
        let mut best_d = eps;
        for (fi, f) in faces.iter().enumerate() {
            let d = (*p - points[f.a]).dot(f.normal);
            if d > best_d {
                best_d = d;
                best = Some(fi);
            }
        }
        if let Some(fi) = best {
            faces[fi].outside.push(i);
        }
    }

    // ── Iteratively expand. Each pass picks one face with a
    // non-empty outside set, finds its farthest point, computes the
    // horizon, replaces visible faces with a fan of new faces from
    // the apex.
    let max_iters = 1_000_000usize; // cheap safety net; QuickHull is finite for finite inputs
    let mut iter = 0;
    while iter < max_iters {
        iter += 1;
        let face_idx = match faces
            .iter()
            .position(|f| !f.outside.is_empty() && !f.deleted)
        {
            Some(i) => i,
            None => break, // hull complete
        };
        // Pick the farthest outside point of this face.
        let apex_pt = {
            let f = &faces[face_idx];
            let mut best_d = Real::NEG_INFINITY;
            let mut apex = f.outside[0];
            for &i in &f.outside {
                let d = (points[i] - points[f.a]).dot(f.normal);
                if d > best_d {
                    best_d = d;
                    apex = i;
                }
            }
            apex
        };

        // Find all faces visible from `apex_pt` via flood-fill from
        // `face_idx` over face adjacency (any face sharing an edge
        // with a visible face whose outward normal also sees the
        // apex is added).
        //
        // We don't keep a true adjacency graph — for the modest
        // hull sizes this code is intended for (≤ a few hundred
        // vertices) the O(F) "any face sharing 2 vertices with a
        // visible one" sweep is fast enough.
        let visible = visible_set(&faces, points, apex_pt, eps);

        // Collect all edges of visible faces, then any edge that
        // appears exactly once is on the horizon.
        let horizon = horizon_edges(&faces, &visible);

        // Mark every visible face as deleted, and gather their
        // outside-point sets (minus the apex itself) for
        // re-partitioning.
        let mut orphans: Vec<usize> = Vec::new();
        for &fi in &visible {
            for &p in &faces[fi].outside {
                if p != apex_pt {
                    orphans.push(p);
                }
            }
            faces[fi].outside.clear();
            faces[fi].deleted = true;
        }

        // Build the new fan: one face per horizon edge with `apex_pt`
        // as the third vertex.
        let mut new_face_idxs: Vec<usize> = Vec::with_capacity(horizon.len());
        for &(u, v) in &horizon {
            let f = make_face(points, u, v, apex_pt, centroid);
            new_face_idxs.push(faces.len());
            faces.push(f);
        }

        // Re-partition orphans onto the new faces.
        for p in orphans {
            let pp = points[p];
            let mut best: Option<usize> = None;
            let mut best_d = eps;
            for &fi in &new_face_idxs {
                let f = &faces[fi];
                let d = (pp - points[f.a]).dot(f.normal);
                if d > best_d {
                    best_d = d;
                    best = Some(fi);
                }
            }
            if let Some(fi) = best {
                faces[fi].outside.push(p);
            }
        }
    }

    // Collect every unique vertex on a surviving face — those are the
    // hull vertices.
    let mut on_hull: Vec<bool> = vec![false; n];
    for f in &faces {
        if f.deleted {
            continue;
        }
        on_hull[f.a] = true;
        on_hull[f.b] = true;
        on_hull[f.c] = true;
    }
    let mut out = Vec::new();
    for (i, &flag) in on_hull.iter().enumerate() {
        if flag {
            out.push(points[i]);
        }
    }
    out
}

#[derive(Debug, Clone)]
struct Face {
    a: usize,
    b: usize,
    c: usize,
    normal: Vec3,
    outside: Vec<usize>,
    deleted: bool,
}

/// Build a face with outward-facing normal (i.e. centroid sits on the
/// negative side of the plane).
fn make_face(points: &[Vec3], a: usize, b: usize, c: usize, centroid: Vec3) -> Face {
    let pa = points[a];
    let pb = points[b];
    let pc = points[c];
    let mut n = (pb - pa).cross(pc - pa);
    // Avoid zero-area face exploding to NaN when normalising — keep
    // the raw normal (zero) and let downstream `dot` checks treat it
    // as never-visible.
    let len = n.length();
    if len > 0.0 {
        n /= len;
    }
    // Flip if centroid is on the positive side (face would be
    // inward-pointing).
    if (centroid - pa).dot(n) > 0.0 {
        n = -n;
        return Face {
            a,
            b: c,
            c: b,
            normal: n,
            outside: Vec::new(),
            deleted: false,
        };
    }
    Face {
        a,
        b,
        c,
        normal: n,
        outside: Vec::new(),
        deleted: false,
    }
}

/// Find a non-degenerate seed tetrahedron. Returns 4 distinct point
/// indices, or `None` if the input is coplanar / colinear / a single
/// point.
fn seed_tetrahedron(points: &[Vec3], eps: Real) -> Option<[usize; 4]> {
    // Extreme indices along ±x / ±y / ±z. The two farthest of these
    // become the first edge.
    let mut ext = [0usize; 6];
    for axis in 0..3 {
        let mut lo = 0usize;
        let mut hi = 0usize;
        for (i, p) in points.iter().enumerate() {
            let v = component(*p, axis);
            if v < component(points[lo], axis) {
                lo = i;
            }
            if v > component(points[hi], axis) {
                hi = i;
            }
        }
        ext[axis * 2] = lo;
        ext[axis * 2 + 1] = hi;
    }
    // Pick the pair of extremes with the largest separation.
    let mut a = ext[0];
    let mut b = ext[1];
    let mut best = (points[a] - points[b]).length_squared();
    for i in 0..6 {
        for j in (i + 1)..6 {
            let d = (points[ext[i]] - points[ext[j]]).length_squared();
            if d > best {
                best = d;
                a = ext[i];
                b = ext[j];
            }
        }
    }
    if best < eps * eps {
        return None;
    }
    // Pick the input point farthest from line `ab`.
    let pa = points[a];
    let pb = points[b];
    let edge = pb - pa;
    let edge_len2 = edge.length_squared();
    let mut c = usize::MAX;
    let mut best_d2 = eps * eps;
    for (i, p) in points.iter().enumerate() {
        if i == a || i == b {
            continue;
        }
        // Distance² from line.
        let cross = edge.cross(*p - pa);
        let d2 = cross.length_squared() / edge_len2;
        if d2 > best_d2 {
            best_d2 = d2;
            c = i;
        }
    }
    if c == usize::MAX {
        return None;
    }
    // Pick the input point farthest from plane `abc`.
    let pc = points[c];
    let normal = (pb - pa).cross(pc - pa);
    let n_len = normal.length();
    if n_len < eps {
        return None;
    }
    let n_unit = normal / n_len;
    let mut d = usize::MAX;
    let mut best_d = eps;
    for (i, p) in points.iter().enumerate() {
        if i == a || i == b || i == c {
            continue;
        }
        let dist = (*p - pa).dot(n_unit).abs();
        if dist > best_d {
            best_d = dist;
            d = i;
        }
    }
    if d == usize::MAX {
        return None;
    }
    Some([a, b, c, d])
}

/// Coplanar fallback — project onto the dominant axis-aligned plane,
/// run a 2D Andrew's monotone-chain hull, lift back to 3D. The
/// returned vertices are still hull points of the *original* input
/// (we just drop the degenerate axis for the inside-test).
fn planar_fallback(points: &[Vec3], _eps: Real) -> Vec<Vec3> {
    if points.len() < 3 {
        // <3 points: just return the unique inputs.
        let mut out = Vec::new();
        for &p in points {
            if !out.iter().any(|q: &Vec3| (p - *q).length_squared() < 1e-24) {
                out.push(p);
            }
        }
        return out;
    }
    // Pick the axis with the *least* spread → drop it.
    let (mn, mx) = points.iter().fold(
        (Vec3::splat(Real::INFINITY), Vec3::splat(-Real::INFINITY)),
        |(a, b), p| (a.min(*p), b.max(*p)),
    );
    let extent = mx - mn;
    let drop_axis = if extent.x <= extent.y && extent.x <= extent.z {
        0
    } else if extent.y <= extent.z {
        1
    } else {
        2
    };
    // 2D coordinates per index.
    let mut pts2d: Vec<(usize, Real, Real)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (u, v) = match drop_axis {
                0 => (p.y, p.z),
                1 => (p.x, p.z),
                _ => (p.x, p.y),
            };
            (i, u, v)
        })
        .collect();
    pts2d.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    // Andrew's monotone chain.
    let cross = |o: (Real, Real), a: (Real, Real), b: (Real, Real)| -> Real {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(usize, Real, Real)> = Vec::new();
    for p in &pts2d {
        while lower.len() >= 2 {
            let n = lower.len();
            if cross(
                (lower[n - 2].1, lower[n - 2].2),
                (lower[n - 1].1, lower[n - 1].2),
                (p.1, p.2),
            ) <= 0.0
            {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(*p);
    }
    let mut upper: Vec<(usize, Real, Real)> = Vec::new();
    for p in pts2d.iter().rev() {
        while upper.len() >= 2 {
            let n = upper.len();
            if cross(
                (upper[n - 2].1, upper[n - 2].2),
                (upper[n - 1].1, upper[n - 1].2),
                (p.1, p.2),
            ) <= 0.0
            {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    let mut idxs: Vec<usize> = lower.into_iter().map(|t| t.0).collect();
    idxs.extend(upper.into_iter().map(|t| t.0));
    idxs.sort();
    idxs.dedup();
    idxs.into_iter().map(|i| points[i]).collect()
}

fn visible_set(faces: &[Face], points: &[Vec3], apex: usize, eps: Real) -> Vec<usize> {
    // Naïve "every undeleted face whose plane the apex sees" sweep.
    // For the hull sizes this code targets (≤ a few hundred vertices)
    // the cost is dominated by the dot products, not graph traversal.
    let p = points[apex];
    let mut out = Vec::new();
    for (i, f) in faces.iter().enumerate() {
        if f.deleted {
            continue;
        }
        let d = (p - points[f.a]).dot(f.normal);
        if d > eps {
            out.push(i);
        }
    }
    out
}

fn horizon_edges(faces: &[Face], visible: &[usize]) -> Vec<(usize, usize)> {
    use std::collections::HashMap;
    // Each visible face contributes 3 directed edges. An edge that
    // appears exactly once across the visible set (i.e. its mirror
    // doesn't appear) lies on the boundary between visible and
    // hidden faces — that's the horizon.
    let mut seen: HashMap<(usize, usize), u32> = HashMap::new();
    for &fi in visible {
        let f = &faces[fi];
        for &(u, v) in &[(f.a, f.b), (f.b, f.c), (f.c, f.a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            *seen.entry(key).or_insert(0) += 1;
        }
    }
    let mut horizon: Vec<(usize, usize)> = Vec::new();
    for (key, count) in seen {
        if count == 1 {
            horizon.push(key);
        }
    }
    // Order doesn't matter for fan construction; each horizon edge
    // becomes one face with the apex.
    horizon
}

fn component(v: Vec3, axis: usize) -> Real {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_cube_corners_returns_eight_vertices() {
        let v: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -1.0 } else { 1.0 },
                    if i & 2 == 0 { -1.0 } else { 1.0 },
                    if i & 4 == 0 { -1.0 } else { 1.0 },
                )
            })
            .collect();
        let hull = convex_hull(&v);
        assert_eq!(hull.len(), 8, "hull of 8 cube corners must yield 8 vertices, got {hull:?}");
    }

    #[test]
    fn hull_drops_interior_points() {
        let mut v: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -1.0 } else { 1.0 },
                    if i & 2 == 0 { -1.0 } else { 1.0 },
                    if i & 4 == 0 { -1.0 } else { 1.0 },
                )
            })
            .collect();
        // Add 50 interior points (all strictly inside the cube).
        let mut s: u32 = 0xCAFE_F00D;
        let mut next = || {
            s = s.wrapping_mul(2654435761).wrapping_add(0x12345678);
            (s as f64 / u32::MAX as f64) * 1.6 - 0.8
        };
        for _ in 0..50 {
            v.push(Vec3::new(next(), next(), next()));
        }
        let hull = convex_hull(&v);
        assert_eq!(
            hull.len(),
            8,
            "hull must be just the 8 corners — got {} vertices",
            hull.len()
        );
        // Each hull vertex should sit at a corner of the cube.
        for h in &hull {
            assert!((h.x.abs() - 1.0).abs() < 1e-9);
            assert!((h.y.abs() - 1.0).abs() < 1e-9);
            assert!((h.z.abs() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn hull_of_tetrahedron_is_itself() {
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let hull = convex_hull(&v);
        assert_eq!(hull.len(), 4);
    }
}
