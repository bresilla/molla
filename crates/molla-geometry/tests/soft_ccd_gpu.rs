//! GPU batched VT-CCD vs the CPU `vt_ccd` oracle.
//!
//! Run with: `make test-gpu P=molla-geometry T=soft_ccd_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_geometry::{VtCcdQuery, vt_ccd, vt_ccd_batch_gpu};
use molla_math::Vec3;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_vt_ccd_matches_cpu() {
    let flat_tri = [
        Vec3::new(-1.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let small_tri = [
        Vec3::new(-0.1, 0.5, -0.1),
        Vec3::new(0.1, 0.5, -0.1),
        Vec3::new(0.0, 0.5, 0.1),
    ];
    let unit_tri = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];

    let queries = vec![
        // Pass-through at mid-step: TOI ≈ 0.5.
        VtCcdQuery {
            v_at_t0: Vec3::new(0.0, -0.5, 0.0),
            v_at_t1: Vec3::new(0.0, 0.5, 0.0),
            tri_at_t0: flat_tri,
            tri_at_t1: flat_tri,
        },
        // Stays below the plane: no hit.
        VtCcdQuery {
            v_at_t0: Vec3::new(0.0, -0.5, 0.0),
            v_at_t1: Vec3::new(0.0, -0.1, 0.0),
            tri_at_t0: flat_tri,
            tri_at_t1: flat_tri,
        },
        // Crosses the plane but outside the triangle: no hit.
        VtCcdQuery {
            v_at_t0: Vec3::new(5.0, -0.5, 0.0),
            v_at_t1: Vec3::new(5.0, 0.5, 0.0),
            tri_at_t0: unit_tri,
            tri_at_t1: unit_tri,
        },
        // Realistic cube-through-cloth crossing near y = 0.5.
        VtCcdQuery {
            v_at_t0: Vec3::new(0.0, 0.521, 0.0),
            v_at_t1: Vec3::new(0.0, 0.479, 0.0),
            tri_at_t0: small_tri,
            tri_at_t1: small_tri,
        },
        // Quarter-step crossing: TOI ≈ 0.25.
        VtCcdQuery {
            v_at_t0: Vec3::new(0.0, -0.25, 0.0),
            v_at_t1: Vec3::new(0.0, 0.75, 0.0),
            tri_at_t0: flat_tri,
            tri_at_t1: flat_tri,
        },
    ];

    let engine = Engine::new_gpu().expect("gpu");
    let gpu = vt_ccd_batch_gpu(&engine, &queries).expect("gpu ccd");
    assert_eq!(gpu.len(), queries.len());

    for (i, q) in queries.iter().enumerate() {
        let cpu = vt_ccd(q.v_at_t0, q.v_at_t1, q.tri_at_t0, q.tri_at_t1);
        match (cpu, gpu[i]) {
            (None, None) => {}
            (Some(a), Some(b)) => assert!(
                (a - b).abs() < 1e-3,
                "query {i}: TOI mismatch cpu={a} gpu={b}",
            ),
            (a, b) => panic!("query {i}: hit/no-hit disagreement cpu={a:?} gpu={b:?}"),
        }
    }
}
