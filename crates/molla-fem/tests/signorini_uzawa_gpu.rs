//! GPU Signorini Uzawa solve (inner CG on device) vs the CPU
//! `solve_augmented_system`.
//!
//! Run with: `make test-gpu P=molla-fem T=signorini_uzawa_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_fem::signorini::solve_augmented_system;
use molla_fem::signorini_gpu::solve_augmented_system_gpu;
use molla_math::sparse::BsrMatrix;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_signorini_uzawa_matches_cpu() {
    // SPD stiffness: 1-D Laplacian tridiag(2, -1).
    let n = 8usize;
    let mut kt: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        kt.push((i, i, 2.0));
        if i + 1 < n {
            kt.push((i, i + 1, -1.0));
            kt.push((i + 1, i, -1.0));
        }
    }
    let k = BsrMatrix::from_scalar_triplets(n, n, &kt).unwrap();
    let f = vec![1.0; n];

    // One inequality constraint on the middle DoF, target above its
    // unconstrained value so the constraint is active (λ > 0).
    let c = BsrMatrix::from_scalar_triplets(1, n, &[(0, 4, 1.0)]).unwrap();
    let g = vec![15.0]; // above the unconstrained peak (~10) → constraint active

    let cpu = solve_augmented_system(&k, &c, &f, &g, 800, 1.0e-9).unwrap();

    let engine = Engine::new_gpu().expect("gpu");
    let gpu = solve_augmented_system_gpu(&engine, &k, &c, &f, &g, 800, 1.0e-9).expect("gpu");

    let mut du = 0.0f64;
    for (a, b) in gpu.u.iter().zip(cpu.u.iter()) {
        du = du.max((a - b).abs());
    }
    assert!(du < 1.0e-3, "u mismatch, max_err = {du}");
    let dl = (gpu.lambda[0] - cpu.lambda[0]).abs();
    assert!(
        dl < 1.0e-3,
        "λ mismatch: cpu={} gpu={}",
        cpu.lambda[0],
        gpu.lambda[0]
    );

    // Non-degenerate: the constraint is genuinely active.
    assert!(
        cpu.lambda[0] > 1.0e-3,
        "constraint should be active, λ = {}",
        cpu.lambda[0]
    );
}
