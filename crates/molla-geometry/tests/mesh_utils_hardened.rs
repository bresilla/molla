//! Stage 21.1 — hardened mesh-utility integration tests.
//!
//! Covers the deliverables added in Stage 21.1:
//!   * rayon-parallel `decimate` + `bake_mesh_to_sdf` performance,
//!   * `convex_hull` face-data output (real wireframe) with outward
//!     normals,
//!   * `decimate_lod_chain` shared-priority-queue LOD output, and
//!   * determinism: rayon parallelism must not perturb output.
//!
//! All tests are CPU-only (no GPU device required) and must pass
//! without `#[ignore]`. Timing thresholds carry a 2x headroom over the
//! PLAN's nominal targets to absorb CI runner variance.

use molla_geometry::{
    ConvexHullMesh, SdfGrid, bake_mesh_to_sdf, convex_hull, decimate, decimate_lod_chain,
};
use molla_math::{Real, Vec3};
use molla_sim::TriMesh;
use std::time::Instant;

// ── Mesh helpers ───────────────────────────────────────────────────

/// Recursive icosphere subdivision. `subdivisions = 0` → 20 tris,
/// 1 → 80, 2 → 320, ..., 7 → 327680. Same routine as
/// `tests/mesh_utils.rs` — duplicated here so each test file is
/// self-contained.
fn icosphere(subdivisions: u32, radius: Real) -> TriMesh {
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
    for v in &mut verts {
        *v *= radius;
    }
    let mut mesh = TriMesh {
        vertices: verts,
        indices: faces,
    };
    mesh.build_bvh(4);
    mesh
}

/// Eight-corner unit cube as `[Vec3; 8]`.
fn cube_corners() -> Vec<Vec3> {
    (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.0 } else { 1.0 },
            )
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────

/// Returns the wall-clock budget for the perf tests. Release mode
/// uses the PLAN's 2x-headroom target; debug builds get a 6x slack
/// because the workspace dev profile leaves opt-level=0 in place
/// (the BVH / heap inner loops measure ~5-8x slower without LLVM
/// optimisations).
fn perf_budget(release_secs: f64) -> f64 {
    if cfg!(debug_assertions) {
        release_secs * 6.0
    } else {
        release_secs
    }
}

/// PLAN target: 50k-tri decimate < 1 s (release). 2x headroom over
/// that nominal target gives 2 s; debug builds are scaled up by
/// `perf_budget`.
#[test]
fn decimate_50k_tri_completes_under_budget() {
    // subdivisions = 6 → 81920 tris > 50k: cap the work by halving
    // the icosphere's face count first via a quick decimate (not
    // counted toward the timed run).
    let big = icosphere(6, 1.0);
    assert!(big.indices.len() >= 50_000, "got {}", big.indices.len());
    // Trim down to 50k for the timed run by selecting the first 50k
    // triangles — the resulting partial mesh is still manifold-ish
    // (icosphere subdivision is regular) which is good enough for
    // a perf test.
    let mut trimmed = TriMesh {
        vertices: big.vertices.clone(),
        indices: big.indices.iter().take(50_000).copied().collect(),
    };
    trimmed.build_bvh(4);
    assert_eq!(trimmed.indices.len(), 50_000);

    let budget = perf_budget(2.0);
    let t0 = Instant::now();
    let dec = decimate(&trimmed, 25_000); // ~50 % LOD
    let elapsed = t0.elapsed();
    println!(
        "decimate 50k → ~25k tris took {elapsed:?} (release-budget 2s, this-build budget {budget}s)",
    );
    assert!(
        elapsed.as_secs_f64() < budget,
        "50k-tri decimation took {:?}, > {budget}s budget",
        elapsed,
    );
    // Sanity: result should be roughly half the input.
    assert!(
        dec.indices.len() <= 25_000 + 5_000,
        "decimate didn't hit target ({} vs 25k)",
        dec.indices.len(),
    );
}

/// PLAN target: 64³ SDF bake of a 10k-tri mesh < 500 ms (release).
/// 2x headroom = 1 s; debug builds scaled by `perf_budget`.
#[test]
fn bake_64_cubed_10k_tri_under_budget() {
    // subdivisions = 5 → 20480 tris; trim to ~10k for the test.
    let big = icosphere(5, 1.0);
    assert!(big.indices.len() >= 10_000);
    let mut mesh = TriMesh {
        vertices: big.vertices.clone(),
        indices: big.indices.iter().take(10_000).copied().collect(),
    };
    mesh.build_bvh(4);

    let budget = perf_budget(1.0);
    let t0 = Instant::now();
    let sdf = bake_mesh_to_sdf(&mesh, 64, 0.1);
    let elapsed = t0.elapsed();
    println!(
        "bake 64³ × 10k tris took {elapsed:?} (release-budget 1s, this-build budget {budget}s)",
    );
    assert_eq!(sdf.nx, 64);
    assert!(
        elapsed.as_secs_f64() < budget,
        "64³ SDF bake took {:?}, > {budget}s budget",
        elapsed,
    );
}

/// `convex_hull` of a cube returns a `ConvexHullMesh` with 12
/// triangle faces (two per cube face), and every face's normal points
/// outward (test: face_normal · (face_centre - hull_centroid) > 0).
#[test]
fn hull_of_cube_returns_twelve_outward_faces() {
    let pts = cube_corners();
    let hull: ConvexHullMesh = convex_hull(&pts);
    assert_eq!(hull.vertices.len(), 8, "cube hull has 8 vertices");
    assert_eq!(
        hull.faces.len(),
        12,
        "cube hull should have 12 triangle faces, got {}",
        hull.faces.len(),
    );
    // Hull centroid (mean of vertices).
    let mut centroid = Vec3::ZERO;
    for v in &hull.vertices {
        centroid += *v;
    }
    centroid /= hull.vertices.len() as Real;

    for (fi, face) in hull.faces.iter().enumerate() {
        let v0 = hull.vertices[face[0] as usize];
        let v1 = hull.vertices[face[1] as usize];
        let v2 = hull.vertices[face[2] as usize];
        let normal = (v1 - v0).cross(v2 - v0);
        assert!(
            normal.length_squared() > 1.0e-12,
            "face {fi} has zero-area normal",
        );
        let face_centre = (v0 + v1 + v2) / 3.0;
        let outward = face_centre - centroid;
        assert!(
            normal.dot(outward) > 0.0,
            "face {fi} normal {normal:?} not outward (centre {face_centre:?}, centroid {centroid:?}, dot={})",
            normal.dot(outward),
        );
    }
}

/// `decimate_lod_chain([0.5, 0.25, 0.1])` returns 3 meshes with
/// monotonically decreasing triangle counts; each mesh's vertex set
/// is a subset of the parent's vertex set (post-collapse positions
/// move, so we test vertex *count* monotonicity and that the deepest
/// LOD's positions all lie within the previous LOD's bounding box).
#[test]
fn decimate_lod_chain_monotonic_and_subset() {
    let m = icosphere(4, 1.0); // 5120 tris
    let factors = [0.5, 0.25, 0.1];
    let chain = decimate_lod_chain(&m, &factors);
    assert_eq!(chain.len(), 3, "chain length must match factors");
    // Triangle counts strictly decreasing.
    let counts: Vec<usize> = chain.iter().map(|m| m.indices.len()).collect();
    println!("LOD chain triangle counts: {counts:?}");
    assert!(
        counts[0] > counts[1] && counts[1] > counts[2],
        "LOD triangle counts must be strictly decreasing, got {counts:?}",
    );
    // Triangle counts roughly track the factors (within 25 % slack).
    for (k, &f) in factors.iter().enumerate() {
        let expected = (m.indices.len() as f64 * f).round() as usize;
        let slack = (expected / 4).max(50);
        assert!(
            counts[k] <= expected + slack,
            "LOD {k} count {} > expected {} + slack {}",
            counts[k],
            expected,
            slack,
        );
    }
    // Vertex counts must also be monotonically non-increasing. QEM
    // collapses move surviving vertices to optimal positions, so the
    // PLAN's literal "subset" wording can't hold positionally — we
    // assert the cardinality interpretation (#verts(LOD_{k+1}) ≤
    // #verts(LOD_k)) and back it up with a 3D-bounding-box
    // containment check (every deeper-LOD vertex sits inside the
    // previous LOD's AABB inflated by `eps`).
    let vert_counts: Vec<usize> = chain.iter().map(|m| m.vertices.len()).collect();
    println!("LOD chain vertex counts: {vert_counts:?}");
    for k in 0..(chain.len() - 1) {
        assert!(
            vert_counts[k + 1] <= vert_counts[k],
            "LOD {} has more vertices ({}) than LOD {} ({})",
            k + 1,
            vert_counts[k + 1],
            k,
            vert_counts[k],
        );
        let (mn, mx) = chain[k].vertices.iter().fold(
            (Vec3::splat(Real::INFINITY), Vec3::splat(-Real::INFINITY)),
            |(a, b), v| (a.min(*v), b.max(*v)),
        );
        let eps = Vec3::splat(0.05);
        for v in &chain[k + 1].vertices {
            assert!(
                v.x >= mn.x - eps.x
                    && v.x <= mx.x + eps.x
                    && v.y >= mn.y - eps.y
                    && v.y <= mx.y + eps.y
                    && v.z >= mn.z - eps.z
                    && v.z <= mx.z + eps.z,
                "LOD {} vertex {:?} outside LOD {}'s padded AABB ({mn:?}..{mx:?})",
                k + 1,
                v,
                k,
            );
        }
    }
}

/// Determinism — same inputs across 10 runs produce byte-identical
/// outputs from `decimate`. Regression guard against rayon
/// non-determinism (parallel float reductions, hash-set iteration
/// order).
#[test]
fn decimate_is_deterministic_across_runs() {
    let m = icosphere(4, 1.0); // 5120 tris
    let target = 1280;
    let baseline = decimate(&m, target);
    for run in 1..10 {
        let again = decimate(&m, target);
        assert_eq!(
            again.vertices.len(),
            baseline.vertices.len(),
            "run {run}: vertex count drift",
        );
        assert_eq!(
            again.indices.len(),
            baseline.indices.len(),
            "run {run}: index count drift",
        );
        // Byte-identical positions (Vec3 derives PartialEq via
        // f64 == — only safe because we want bit-identical output).
        for (i, (a, b)) in baseline.vertices.iter().zip(&again.vertices).enumerate() {
            assert_eq!(a.to_array(), b.to_array(), "run {run}: vertex {i} drift");
        }
        for (i, (a, b)) in baseline.indices.iter().zip(&again.indices).enumerate() {
            assert_eq!(a, b, "run {run}: index {i} drift");
        }
    }
}

/// Determinism — same inputs across 10 runs produce byte-identical
/// outputs from `bake_mesh_to_sdf`.
#[test]
fn sdf_bake_is_deterministic_across_runs() {
    let m = icosphere(2, 1.0); // 320 tris — fast bake
    let baseline: SdfGrid = bake_mesh_to_sdf(&m, 24, 0.25);
    for run in 1..10 {
        let again = bake_mesh_to_sdf(&m, 24, 0.25);
        assert_eq!(
            again.data.len(),
            baseline.data.len(),
            "run {run}: len drift"
        );
        for (i, (a, b)) in baseline.data.iter().zip(&again.data).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "run {run}: voxel {i} drift {a} vs {b}",
            );
        }
    }
}

/// Determinism — `convex_hull` output is byte-identical across runs.
#[test]
fn convex_hull_is_deterministic_across_runs() {
    let pts = cube_corners();
    let baseline = convex_hull(&pts);
    for run in 1..10 {
        let again = convex_hull(&pts);
        assert_eq!(
            again.vertices.len(),
            baseline.vertices.len(),
            "run {run}: vertex count drift",
        );
        assert_eq!(
            again.faces.len(),
            baseline.faces.len(),
            "run {run}: face count drift",
        );
        for (i, (a, b)) in baseline.vertices.iter().zip(&again.vertices).enumerate() {
            assert_eq!(a.to_array(), b.to_array(), "run {run}: vertex {i} drift");
        }
        for (i, (a, b)) in baseline.faces.iter().zip(&again.faces).enumerate() {
            assert_eq!(a, b, "run {run}: face {i} drift");
        }
    }
}
