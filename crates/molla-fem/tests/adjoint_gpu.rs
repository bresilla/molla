//! GPU adjoint Young's-modulus gradient vs the CPU `compute_grad_youngs`.
//!
//! Run with: `make test-gpu P=molla-fem T=adjoint_gpu`.
//!
//! The kernel evaluates `λᵀ K u` for arbitrary fields, so synthetic
//! `u`/`λ` exercise the numerics without running the forward+adjoint
//! solve — the gradient value need not be physical, only match the CPU.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_fem::adjoint::compute_grad_youngs;
use molla_fem::adjoint_gpu::compute_grad_youngs_gpu;
use molla_fem::mesh::FemMesh;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_adjoint_grad_youngs_matches_cpu() {
    let mesh = FemMesh::tet_grid_3d(3, 2, 2, 2.0, 1.0, 1.0);
    let n = mesh.vertices.len() * 3;

    // Deterministic synthetic displacement / adjoint fields.
    let u: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.017).sin() * 0.01).collect();
    let lambda: Vec<f64> = (0..n)
        .map(|i| ((i as f64) * 0.011 + 0.3).cos() * 0.02)
        .collect();

    let poisson = 0.3;
    let youngs = 1.0e6;

    let cpu = compute_grad_youngs(&mesh, &mesh.vertices, &u, &lambda, poisson, youngs);

    let engine = Engine::new_gpu().expect("gpu");
    let gpu = compute_grad_youngs_gpu(&engine, &mesh, &mesh.vertices, &u, &lambda, poisson, youngs)
        .expect("gpu grad");

    let denom = cpu.abs().max(1.0e-6);
    assert!(
        (gpu - cpu).abs() / denom < 2.0e-3,
        "grad mismatch: cpu={cpu} gpu={gpu} (rel {})",
        (gpu - cpu).abs() / denom,
    );
    // Sanity: the field is non-trivial, so the gradient is non-zero.
    assert!(cpu.abs() > 1.0e-9, "degenerate test: cpu grad ~ 0");
}
