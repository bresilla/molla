//! Stage 21 — mesh-utility integration tests.
//!
//! All tests are CPU-only (no GPU device required) and must pass
//! without `#[ignore]`. They exercise the public APIs only — no
//! internal-helper poking.

use molla_geometry::{
    SdfGrid, SpatialHashGrid, bake_mesh_to_sdf, convex_hull, decimate, marching_cubes,
};
use molla_math::{Real, Vec3};
use molla_sim::TriMesh;

// ── Shared mesh helpers ────────────────────────────────────────────

/// Build an icosphere by recursive subdivision of a 20-face icosahedron.
/// `subdivisions = 0` → 20 tris, 1 → 80, 2 → 320, 3 → 1280, ...
fn icosphere(subdivisions: u32, radius: Real) -> TriMesh {
    // Initial 12 vertices of a regular icosahedron (golden-ratio
    // coordinate set), then normalised onto the unit sphere.
    let t: Real = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let raw = [
        (-1.0, t, 0.0),
        (1.0, t, 0.0),
        (-1.0, -t, 0.0),
        (1.0, -t, 0.0),
        (0.0, -1.0, t),
        (0.0, 1.0, t),
        (0.0, -1.0, -t),
        (0.0, 1.0, -t),
        (t, 0.0, -1.0),
        (t, 0.0, 1.0),
        (-t, 0.0, -1.0),
        (-t, 0.0, 1.0),
    ];
    let mut verts: Vec<Vec3> = raw
        .iter()
        .map(|(x, y, z)| {
            let v = Vec3::new(*x, *y, *z);
            v / v.length()
        })
        .collect();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        // Subdivide each tri into 4 by inserting midpoints on each edge.
        use std::collections::HashMap;
        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&i) = midpoint_cache.get(&key) {
                return i;
            }
            let mid = (verts[a as usize] + verts[b as usize]) * 0.5;
            let mid = mid / mid.length();
            verts.push(mid);
            let idx = (verts.len() - 1) as u32;
            midpoint_cache.insert(key, idx);
            idx
        };
        let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(faces.len() * 4);
        for tri in &faces {
            let a = midpoint(tri[0], tri[1], &mut verts);
            let b = midpoint(tri[1], tri[2], &mut verts);
            let c = midpoint(tri[2], tri[0], &mut verts);
            new_faces.push([tri[0], a, c]);
            new_faces.push([tri[1], b, a]);
            new_faces.push([tri[2], c, b]);
            new_faces.push([a, b, c]);
        }
        faces = new_faces;
    }
    // Scale by radius.
    for v in &mut verts {
        *v *= radius;
    }
    let mut mesh = TriMesh {
        vertices: verts,
        indices: faces,
        ..Default::default()
    };
    mesh.build_bvh(4);
    mesh
}

// ── Tests ──────────────────────────────────────────────────────────

#[test]
fn hull_of_cube_corners_plus_interior_returns_eight_vertices() {
    // Eight cube corners.
    let mut points: Vec<Vec3> = Vec::new();
    for i in 0..8 {
        points.push(Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        ));
    }
    // Plus 200 deterministically-random interior points.
    let mut s: u32 = 0xDEAD_BEEF;
    let mut next = || {
        s = s.wrapping_mul(2654435761).wrapping_add(0x12345678);
        ((s as f64) / (u32::MAX as f64)) * 1.6 - 0.8
    };
    for _ in 0..200 {
        points.push(Vec3::new(next(), next(), next()));
    }
    let hull = convex_hull(&points);
    assert_eq!(
        hull.vertices.len(),
        8,
        "expected exactly 8 hull vertices (the 8 cube corners), got {}: {:?}",
        hull.vertices.len(),
        hull.vertices,
    );
    // Every vertex must be at a corner.
    for h in &hull.vertices {
        assert!((h.x.abs() - 1.0).abs() < 1.0e-9, "non-corner x at {h:?}");
        assert!((h.y.abs() - 1.0).abs() < 1.0e-9, "non-corner y at {h:?}");
        assert!((h.z.abs() - 1.0).abs() < 1.0e-9, "non-corner z at {h:?}");
    }
}

#[test]
fn decimate_sphere_keeps_hausdorff_under_5pct() {
    let m = icosphere(3, 1.0); // 1280 tris, 642 verts
    assert_eq!(m.indices.len(), 1280);
    let target = 320;
    let d = decimate(&m, target);
    assert!(
        d.indices.len() <= target + (target / 4),
        "decimate should hit roughly target_tri_count, got {} (target {})",
        d.indices.len(),
        target,
    );

    // One-sided Hausdorff distance from the *decimated* surface to the
    // original surface — for an analytic ground-truth (the unit
    // sphere) this is just `max over decimated vertices of
    // |length - 1|`. The decimator with QEM tends to leave
    // vertices on the original surface (the optimal-position solve
    // collapses to the local quadric minimum, which sits on the
    // sphere for uniform-curvature inputs), so this should be ≪ 5 %.
    let radius: Real = 1.0;
    let mut max_err: Real = 0.0;
    for v in &d.vertices {
        let err = (v.length() - radius).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < radius * 0.05,
        "Hausdorff distance from decimated vertices to sphere surface \
         {max_err} > 5 % of radius (decimated vertex count {})",
        d.vertices.len(),
    );
}

#[test]
fn sdf_bake_then_marching_cubes_recovers_sphere_surface() {
    let m = icosphere(2, 1.0); // 320 tris, 162 verts — keeps bake fast
    let res = 24;
    let sdf: SdfGrid = bake_mesh_to_sdf(&m, res, 0.5);
    let extracted = marching_cubes(&sdf, 0.0);
    assert!(
        !extracted.indices.is_empty(),
        "marching cubes produced no triangles for a sphere SDF"
    );
    // Every extracted vertex should sit on the sphere surface within
    // ~one voxel: distance from origin ≈ 1.0 ± cell_size.
    let cell = sdf.cell_size;
    let mut max_err: Real = 0.0;
    for v in &extracted.vertices {
        let r = v.length();
        let err = (r - 1.0).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // Allow ~2 cells of slack: voxelisation rounds + MC interpolation.
    assert!(
        max_err < cell * 2.0,
        "extracted surface deviates {max_err} from r=1 (cell {cell})",
    );
}

#[test]
fn spatial_hash_query_matches_brute_force() {
    let mut g = SpatialHashGrid::new(0.25);
    let mut points: Vec<Vec3> = Vec::new();
    let mut s: u32 = 0xC0FFEE;
    let mut next = || {
        s = s.wrapping_mul(2654435761).wrapping_add(0x12345678);
        ((s as f64) / (u32::MAX as f64)) * 4.0 - 2.0
    };
    for i in 0..400 {
        let p = Vec3::new(next(), next(), next());
        points.push(p);
        g.insert(i as u32, p);
    }
    // 20 random AABB queries — grid result must include every brute-
    // force ground-truth hit (the grid query is a conservative
    // superset since cell-AABB > query-AABB at the cell boundaries).
    for _ in 0..20 {
        let cx = next() * 4.0 - 2.0;
        let cy = next() * 4.0 - 2.0;
        let cz = next() * 4.0 - 2.0;
        let half = next().abs() * 0.6 + 0.1;
        let lo = Vec3::new(cx - half, cy - half, cz - half);
        let hi = Vec3::new(cx + half, cy + half, cz + half);
        let mut from_grid: Vec<u32> = g.query_aabb(lo, hi).collect();
        from_grid.sort();
        from_grid.dedup();
        let mut bf: Vec<u32> = (0..points.len() as u32)
            .filter(|i| {
                let p = points[*i as usize];
                p.x >= lo.x
                    && p.x <= hi.x
                    && p.y >= lo.y
                    && p.y <= hi.y
                    && p.z >= lo.z
                    && p.z <= hi.z
            })
            .collect();
        bf.sort();
        for id in &bf {
            assert!(
                from_grid.contains(id),
                "grid query missed brute-force hit {id}"
            );
        }
    }
}
