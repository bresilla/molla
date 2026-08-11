//! GPU phase-field damage Jacobi solve vs the CPU oracle.
//!
//! Run with: `make test-gpu P=molla-fem T=phase_field_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_fem::mesh::FemMesh;
use molla_fem::phase_field::{PhaseFieldFractureParams, solve_phase_field_fracture_jacobi};
use molla_fem::phase_field_gpu::solve_phase_field_fracture_jacobi_gpu;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_phase_field_jacobi_matches_cpu() {
    let mesh = FemMesh::tri_grid_2d(6, 6, 1.0, 1.0);
    let n = mesh.vert_count();
    let params = PhaseFieldFractureParams::default();

    // A driving-energy bump near the centre so damage is non-trivial.
    let history: Vec<f64> = mesh
        .vertices
        .iter()
        .map(|p| {
            let dx = p.x - 0.5;
            let dy = p.y - 0.5;
            10.0 * (-(dx * dx + dy * dy) * 20.0).exp()
        })
        .collect();
    let damage_old = vec![0.0; n];

    // Fixed sweep count, well short of machine convergence, so both
    // paths run exactly the same number of contractive Jacobi steps.
    let sweeps = 30;
    let cpu = solve_phase_field_fracture_jacobi(&mesh, &damage_old, &history, params, sweeps, 0.0)
        .expect("cpu solve");

    let engine = Engine::new_gpu().expect("gpu");
    let gpu = solve_phase_field_fracture_jacobi_gpu(
        &engine,
        &mesh,
        &damage_old,
        &history,
        params,
        sweeps,
    )
    .expect("gpu solve");

    assert_eq!(gpu.len(), cpu.damage.len());
    let mut max_err = 0.0f64;
    for (g, c) in gpu.iter().zip(cpu.damage.iter()) {
        max_err = max_err.max((g - c).abs());
    }
    assert!(max_err < 2.0e-3, "damage mismatch, max_err = {max_err}");
    // Sanity: damage actually grew somewhere (non-degenerate test).
    assert!(
        cpu.damage.iter().cloned().fold(0.0, f64::max) > 1.0e-3,
        "degenerate test: no damage developed",
    );
}
