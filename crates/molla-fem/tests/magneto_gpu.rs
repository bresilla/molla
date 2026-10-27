//! GPU Nédélec curl recovery vs the CPU `compute_curl_per_cell`.
//!
//! Run with: `make test-gpu P=molla-fem T=magneto_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_fem::magneto::compute_curl_per_cell;
use molla_fem::magneto_gpu::compute_curl_per_cell_gpu;
use molla_fem::mesh::FemMesh;
use molla_fem::nedelec_element::NedelecDofMap;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_magneto_curl_matches_cpu() {
    let mesh = FemMesh::tet_grid_3d(3, 2, 2, 2.0, 1.0, 1.0);
    let map = NedelecDofMap::build(&mesh);

    // Deterministic synthetic edge-DoF vector potential.
    let a_global: Vec<f64> = (0..map.ndof())
        .map(|i| ((i as f64) * 0.019 + 0.2).sin() * 0.5)
        .collect();

    let cpu = compute_curl_per_cell(&mesh, &map, &a_global);

    let engine = Engine::new_gpu().expect("gpu");
    let gpu = compute_curl_per_cell_gpu(&engine, &mesh, &map, &a_global).expect("gpu curl");

    assert_eq!(gpu.len(), cpu.len());
    let mut max_err = 0.0f64;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        max_err = max_err.max((*g - *c).length());
    }
    assert!(max_err < 1.0e-3, "curl max_err = {max_err}");
    // Non-degenerate: the field actually curls somewhere.
    assert!(
        cpu.iter().map(|b| b.length()).fold(0.0, f64::max) > 1.0e-3,
        "degenerate test: curl ~ 0 everywhere",
    );
}
