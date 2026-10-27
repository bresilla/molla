//! GPU level-set curvature + CSF force vs the CPU oracles.
//!
//! Run with: `make test-gpu P=molla-fem T=multiphase_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_fem::multiphase::{csf_surface_tension_force_2d, level_set_curvature_2d};
use molla_fem::multiphase_gpu::{csf_surface_tension_force_2d_gpu, level_set_curvature_2d_gpu};

fn circle_sdf(n: usize) -> (Vec<f64>, f64) {
    let dx = 1.0 / n as f64;
    let r = 0.2;
    let mut phi = vec![0.0; n * n];
    for j in 0..n {
        let y = (j as f64 + 0.5) * dx - 0.5;
        for i in 0..n {
            let x = (i as f64 + 0.5) * dx - 0.5;
            phi[j * n + i] = (x * x + y * y).sqrt() - r;
        }
    }
    (phi, dx)
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_multiphase_matches_cpu() {
    let n = 48usize;
    let (phi, dx) = circle_sdf(n);
    let sigma = 0.7;

    let cpu_k = level_set_curvature_2d(&phi, n, n, dx).unwrap();
    let cpu_f = csf_surface_tension_force_2d(&phi, n, n, dx, sigma).unwrap();

    let engine = Engine::new_gpu().expect("gpu");
    let gpu_k = level_set_curvature_2d_gpu(&engine, &phi, n, n, dx).expect("gpu curvature");
    let gpu_f = csf_surface_tension_force_2d_gpu(&engine, &phi, n, n, dx, sigma).expect("gpu csf");

    // Curvature is only well-defined near the interface — the SDF is
    // singular at the circle centre (∇φ → 0), where both paths hit the
    // 1e-12 clamp and produce (different) garbage. Compare the interface
    // band, which is exactly where the CSF force is non-zero.
    let mut max_k = 0.0f64;
    for ((g, c), p) in gpu_k.iter().zip(cpu_k.iter()).zip(phi.iter()) {
        if p.abs() < 3.0 * dx {
            max_k = max_k.max((g - c).abs());
        }
    }
    assert!(max_k < 1.0e-3, "curvature max_err = {max_k}");

    let mut max_f = 0.0f64;
    for (g, c) in gpu_f.iter().zip(cpu_f.iter()) {
        max_f = max_f.max((g[0] - c[0]).abs()).max((g[1] - c[1]).abs());
    }
    assert!(max_f < 1.0e-3, "csf force max_err = {max_f}");

    // Non-degenerate: curvature near the interface is ~ 1/r.
    let near: Vec<f64> = phi
        .iter()
        .zip(cpu_k.iter())
        .filter(|(p, _)| p.abs() < dx)
        .map(|(_, k)| *k)
        .collect();
    let avg = near.iter().sum::<f64>() / near.len() as f64;
    assert!(
        (avg - 1.0 / 0.2).abs() / (1.0 / 0.2) < 0.15,
        "avg κ = {avg}"
    );
}
