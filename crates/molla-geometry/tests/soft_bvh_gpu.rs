//! GPU soft-BVH broad-phase query vs the CPU `SoftBvh::query_point_swept`.
//!
//! Run with: `make test-gpu P=molla-geometry T=soft_bvh_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_geometry::{SoftBvh, soft_bvh_query_points_swept_gpu};
use molla_math::Vec3;
use molla_sim::ModelBuilder;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_soft_bvh_query_matches_cpu() {
    // Two tets sharing a face, both wound CCW (positive rest volume).
    let verts = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    let mut b = ModelBuilder::new();
    let mut ids = [0u32; 5];
    for (i, v) in verts.iter().enumerate() {
        ids[i] = b.add_particle(*v, 0.05, 0.05, WorldId(0));
    }
    let tets = [
        [ids[0], ids[1], ids[2], ids[3]],
        [ids[1], ids[2], ids[3], ids[4]],
    ];
    let mesh = b.add_tet_mesh(&tets, 1.0e5, 0.3, None).expect("tet mesh");
    b.set_tet_mesh_self_collision(mesh, true);
    let model = b.build().unwrap();
    let state = model.state().unwrap();
    let bvh = SoftBvh::build(&model, &state);
    assert!(!bvh.tri_aabbs.is_empty(), "bvh built no tris");

    let queries = vec![
        (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 0.1),
        (Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.5, 0.5, 0.5), 0.05),
        (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), 0.05),
        (Vec3::new(5.0, 5.0, 5.0), Vec3::new(5.0, 5.0, 5.0), 0.1),
        (Vec3::new(-0.5, -0.5, 0.0), Vec3::new(0.5, 0.5, 0.0), 0.02),
    ];

    let engine = Engine::new_gpu().expect("gpu");
    let gpu = soft_bvh_query_points_swept_gpu(&engine, &bvh, &queries).expect("gpu query");
    assert_eq!(gpu.len(), queries.len());

    for (i, (p0, p1, margin)) in queries.iter().enumerate() {
        let mut cpu = bvh.query_point_swept(*p0, *p1, *margin);
        cpu.sort_unstable();
        let mut g = gpu[i].clone();
        g.sort_unstable();
        assert_eq!(g, cpu, "query {i}: candidate set differs");
    }
}
