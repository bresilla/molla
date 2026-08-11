//! GPU SDF bake vs the CPU `bake_mesh_to_sdf` oracle.
//!
//! Run with: `make test-gpu P=molla-geometry T=sdf_bake_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_geometry::{bake_mesh_to_sdf, bake_mesh_to_sdf_gpu};
use molla_math::Vec3;
use molla_sim::TriMesh;

fn unit_cube_mesh() -> TriMesh {
    let v = vec![
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
    ];
    let i = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 6, 2],
        [3, 7, 6],
        [0, 7, 3],
        [0, 4, 7],
        [1, 2, 6],
        [1, 6, 5],
    ];
    TriMesh {
        vertices: v,
        indices: i,
    }
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_sdf_bake_matches_cpu_on_a_cube() {
    let mesh = unit_cube_mesh();
    let resolution = 16;
    let padding = 0.25;

    let cpu = bake_mesh_to_sdf(&mesh, resolution, padding);
    let engine = Engine::new_gpu().expect("gpu");
    let gpu = bake_mesh_to_sdf_gpu(&engine, &mesh, resolution, padding).expect("bake");

    // Grids must be sample aligned.
    assert_eq!((gpu.nx, gpu.ny, gpu.nz), (cpu.nx, cpu.ny, cpu.nz));
    assert!((gpu.cell_size - cpu.cell_size).abs() < 1e-5);
    assert!((gpu.origin - cpu.origin).length() < 1e-5);
    assert_eq!(gpu.data.len(), cpu.data.len());

    let cell = cpu.cell_size;
    let dist_tol = 2.0e-3;
    let sign_band = 2.0 * cell;
    let mut sign_mismatches = 0usize;
    for (g, c) in gpu.data.iter().zip(cpu.data.iter()) {
        // Unsigned distance must agree everywhere (the parallel core).
        assert!(
            (g.abs() - c.abs()).abs() < dist_tol,
            "distance mismatch: gpu={g} cpu={c}",
        );
        // Sign must agree away from the surface, where it is unambiguous.
        if c.abs() > sign_band && g.signum() != c.signum() {
            sign_mismatches += 1;
        }
    }
    assert_eq!(sign_mismatches, 0, "sign disagreements away from surface");

    // Spot check the analytic expectation: centre inside, corner outside.
    let (cx, cy, cz) = (gpu.nx / 2, gpu.ny / 2, gpu.nz / 2);
    assert!(gpu.at(cx, cy, cz) < 0.0, "cube centre should be inside");
    assert!(gpu.at(0, 0, 0) > 0.0, "grid corner should be outside");
}
