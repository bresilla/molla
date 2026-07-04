//! Verifies the per-mesh BVH actually accelerates shape-vs-trimesh
//! narrow phase. Builds a 1024-triangle grid mesh and a sphere
//! sitting near one corner; with BVH on, the contact search visits
//! only a few triangles. Without BVH (legacy meshes have no nodes)
//! it would visit all 1024.
//!
//! Asserts that wall-time of contact generation is well under what
//! the old O(N) iteration would cost — modest budget, just enough
//! to catch regressions.

use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, ModelBuilder, TriMesh};

fn sphere_inertia(mass: f64, r: f64) -> Mat3 {
    let i = (2.0 / 5.0) * mass * r * r;
    Mat3::from_diagonal(V3::new(i, i, i))
}

fn build_grid_mesh(n: usize) -> TriMesh {
    // n×n quads = 2·n² triangles, all in the y=0 plane.
    let mut mesh = TriMesh::new();
    for j in 0..=n {
        for i in 0..=n {
            mesh.vertices.push(V3::new(
                (i as f64 - n as f64 * 0.5) * 0.1,
                0.0,
                (j as f64 - n as f64 * 0.5) * 0.1,
            ));
        }
    }
    let row = (n + 1) as u32;
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let v00 = j * row + i;
            let v10 = j * row + i + 1;
            let v01 = (j + 1) * row + i;
            let v11 = (j + 1) * row + i + 1;
            mesh.indices.push([v00, v10, v11]);
            mesh.indices.push([v00, v11, v01]);
        }
    }
    mesh
}

#[test]
fn bvh_accelerates_sphere_on_grid_mesh() {
    let n = 32; // 32×32 quads = 2048 triangles
    let mesh = build_grid_mesh(n);
    assert_eq!(mesh.indices.len(), 2 * n * n);

    let mut b = ModelBuilder::new();
    let mass = 1.0_f64;
    let radius = 0.05_f64;
    let inertia = sphere_inertia(mass, radius);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia,
            })
            // Place the sphere near one corner so it overlaps few triangles.
            .with_position(V3::new(-1.5, radius, -1.5)),
    );
    let _ = b
        .add_shape_sphere(body, radius, Transform::IDENTITY)
        .unwrap();
    let mesh_id = b.add_mesh(mesh);
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let state = model.state().unwrap();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();

    // Time 1000 collide() calls.
    let t0 = std::time::Instant::now();
    let iters = 1000;
    for _ in 0..iters {
        contacts.clear();
        pipeline.collide(&model, &state, &mut contacts).unwrap();
    }
    let dt_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    eprintln!(
        "32x32 mesh + 1 sphere narrow-phase: {dt_us:.1} µs / call ({} contacts)",
        contacts.count
    );
    // Sanity: each call should produce at most a small handful of
    // contacts — the sphere only touches 1-2 triangles.
    assert!(
        contacts.count <= 4,
        "expected ≤4 contacts, got {}",
        contacts.count
    );
    // Performance gate: with BVH wired in, collide should be well
    // under 200 µs per call in release. Debug builds are much noisier
    // on this path (disabled inlining + bounds checks + unoptimised
    // BVH traversal), so keep a wide debug budget and leave the tight
    // regression signal to release/bench runs rather than making the
    // default `make test` lane fail on machine-dependent wall time.
    // The debug budget only needs to catch "BVH not wired at all"
    // (a ~100× regression, tens of ms) — 6 ms proved too tight for
    // shared CI runners (measured 7.5 ms on a hosted GitHub runner
    // that passes comfortably on dev machines).
    let budget_us = if cfg!(debug_assertions) {
        20_000.0
    } else {
        200.0
    };
    assert!(
        dt_us < budget_us,
        "narrow phase too slow ({dt_us:.1} µs > {budget_us} µs); BVH may not be wired in"
    );
}
