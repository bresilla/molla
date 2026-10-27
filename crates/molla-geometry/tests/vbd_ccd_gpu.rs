//! GPU VBD CCD pass (broad + narrow, composed) vs a CPU reference built
//! from the same `SoftBvh::query_point_swept` + `vt_ccd` primitives.
//!
//! Run with: `make test-gpu P=molla-geometry T=vbd_ccd_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_geometry::{SoftBvh, vbd_ccd_earliest_toi_gpu, vt_ccd};
use molla_math::Vec3;
use molla_sim::ModelBuilder;

// CPU reference: the same two-phase logic the GPU path composes.
fn cpu_ref(
    bvh: &SoftBvh,
    verts_t0: &[Vec3],
    verts_t1: &[Vec3],
    tris: &[[u32; 3]],
    query_vertices: &[u32],
    margin: f64,
) -> Vec<Option<f64>> {
    query_vertices
        .iter()
        .map(|&vi| {
            let mut best: Option<f64> = None;
            for t in bvh.query_point_swept(verts_t0[vi as usize], verts_t1[vi as usize], margin) {
                let tri = tris[t as usize];
                if tri.contains(&vi) {
                    continue;
                }
                let toi = vt_ccd(
                    verts_t0[vi as usize],
                    verts_t1[vi as usize],
                    [
                        verts_t0[tri[0] as usize],
                        verts_t0[tri[1] as usize],
                        verts_t0[tri[2] as usize],
                    ],
                    [
                        verts_t1[tri[0] as usize],
                        verts_t1[tri[1] as usize],
                        verts_t1[tri[2] as usize],
                    ],
                );
                if let Some(x) = toi {
                    best = Some(best.map_or(x, |e| e.min(x)));
                }
            }
            best
        })
        .collect()
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_vbd_ccd_matches_cpu() {
    // Positive-volume tet; vertex 0 sweeps from the origin through the
    // opposite face (the plane x+y+z=1) to (0.5,0.5,0.5): TOI = 2/3.
    let verts_t0 = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let verts_t1 = [
        Vec3::new(0.5, 0.5, 0.5),
        verts_t0[1],
        verts_t0[2],
        verts_t0[3],
    ];

    let mut b = ModelBuilder::new();
    let mut ids = [0u32; 4];
    for (i, v) in verts_t0.iter().enumerate() {
        ids[i] = b.add_particle(*v, 0.05, 0.05, WorldId(0));
    }
    let mesh = b
        .add_tet_mesh(&[[ids[0], ids[1], ids[2], ids[3]]], 1.0e5, 0.3, None)
        .expect("tet");
    b.set_tet_mesh_self_collision(mesh, true);
    let model = b.build().unwrap();
    let state = model.state().unwrap();
    let mut bvh = SoftBvh::build(&model, &state);

    let t0: Vec<Vec3> = verts_t0.to_vec();
    let t1: Vec<Vec3> = verts_t1.to_vec();
    let tris = &model.soft_surface_tris;
    bvh.refit_swept_arrays(&t0, &t1, tris);

    let query = vec![0u32];
    let margin = 0.05;

    let cpu = cpu_ref(&bvh, &t0, &t1, tris, &query, margin);
    let engine = Engine::new_gpu().expect("gpu");
    let gpu = vbd_ccd_earliest_toi_gpu(&engine, &bvh, &t0, &t1, tris, &query, margin).expect("gpu");

    assert_eq!(gpu.len(), 1);
    match (cpu[0], gpu[0]) {
        (Some(a), Some(b)) => assert!((a - b).abs() < 1e-3, "TOI cpu={a} gpu={b}"),
        (a, b) => panic!("hit/no-hit disagreement cpu={a:?} gpu={b:?}"),
    }
    // Non-degenerate: the vertex actually crosses (TOI ≈ 2/3).
    assert!(cpu[0].is_some(), "expected a crossing");
    assert!(
        (cpu[0].unwrap() - 2.0 / 3.0).abs() < 1e-2,
        "TOI ≈ 2/3, got {:?}",
        cpu[0]
    );
}
