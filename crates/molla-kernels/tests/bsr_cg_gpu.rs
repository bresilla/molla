//! GPU BSR conjugate-gradient solve vs the host `cg_solve` oracle.
//!
//! Run with: `make test-gpu P=molla-kernels T=bsr_cg_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{BsrAlgebraKernel, BsrGpu, bsr_cg_gpu};
use molla_math::iterative::cg_solve;
use molla_math::sparse::BsrMatrix;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_bsr_cg_matches_cpu() {
    // SPD 1-D Laplacian: tridiagonal diag 2, off-diag -1.
    let n = 64usize;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0));
        if i + 1 < n {
            triplets.push((i, i + 1, -1.0));
            triplets.push((i + 1, i, -1.0));
        }
    }
    let a = BsrMatrix::from_scalar_triplets(n, n, &triplets).unwrap();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * (i as f64).sin()).collect();

    let tol = 1.0e-6;
    let max_iters = 500;
    let cpu = cg_solve(
        |x| {
            let mut y = vec![0.0; n];
            a.mv(x, &mut y).unwrap();
            y
        },
        &b,
        &vec![0.0; n],
        tol,
        max_iters,
    );

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let a_gpu = BsrGpu::upload(&device, &queue, &a).unwrap();
    let kernel = BsrAlgebraKernel::new(device, queue);

    let gpu = bsr_cg_gpu(&kernel, &a_gpu, &b, tol, max_iters).expect("gpu cg");

    // Both should converge to the same solution.
    assert!(
        gpu.residual < 1.0e-3,
        "gpu residual {} too large",
        gpu.residual
    );
    let mut max_err = 0.0f64;
    for (g, c) in gpu.x.iter().zip(cpu.x.iter()) {
        max_err = max_err.max((g - c).abs());
    }
    assert!(max_err < 1.0e-3, "solution mismatch, max_err = {max_err}");
    // Non-degenerate: the solution is non-trivial.
    assert!(
        cpu.x.iter().cloned().fold(0.0, f64::max) > 1.0,
        "degenerate solve"
    );
}
