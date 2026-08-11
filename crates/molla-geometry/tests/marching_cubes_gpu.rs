//! GPU marching cubes vs the CPU `marching_cubes` oracle.
//!
//! Run with: `make test-gpu P=molla-geometry T=marching_cubes_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_geometry::{bake_mesh_to_sdf, marching_cubes, marching_cubes_gpu};
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
fn gpu_marching_cubes_matches_cpu() {
    let sdf = bake_mesh_to_sdf(&unit_cube_mesh(), 20, 0.3);
    let cpu = marching_cubes(&sdf, 0.0);

    let engine = Engine::new_gpu().expect("gpu");
    let gpu = marching_cubes_gpu(&engine, &sdf, 0.0).expect("gpu march");

    // The surface must be non-trivial.
    assert!(!cpu.vertices.is_empty(), "cpu produced no surface");
    // Same triangle sequence (cell order is identical on both paths).
    assert_eq!(
        gpu.vertices.len(),
        cpu.vertices.len(),
        "vertex count differs"
    );
    assert_eq!(
        gpu.indices.len(),
        cpu.indices.len(),
        "triangle count differs"
    );
    assert_eq!(gpu.indices, cpu.indices, "index topology differs");
    for (i, (g, c)) in gpu.vertices.iter().zip(cpu.vertices.iter()).enumerate() {
        assert!(
            (*g - *c).length() < 1e-3,
            "vertex {i} mismatch: gpu={g:?} cpu={c:?}",
        );
    }
}
