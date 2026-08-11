//! GPU BSR GMRES solve vs the host `gmres` oracle (non-symmetric A).
//!
//! Run with: `make test-gpu P=molla-kernels T=bsr_gmres_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{BsrAlgebraKernel, BsrGpu, bsr_gmres_gpu};
use molla_math::sparse::{BsrMatrix, IterConfig, gmres};

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_bsr_gmres_matches_cpu() {
    // Non-symmetric, diagonally dominant: diag 2, upper -1, lower -0.5.
    let n = 32usize;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0));
        if i + 1 < n {
            triplets.push((i, i + 1, -1.0));
        }
        if i > 0 {
            triplets.push((i, i - 1, -0.5));
        }
    }
    let a = BsrMatrix::from_scalar_triplets(n, n, &triplets).unwrap();
    let b: Vec<f64> = (0..n).map(|i| 1.0 + 0.2 * (i as f64 * 0.3).cos()).collect();

    let cfg = IterConfig {
        max_iter: 200,
        tol: 1.0e-8,
        restart: 20,
    };
    let mut x_cpu = vec![0.0; n];
    let cpu = gmres(&a, &b, &mut x_cpu, None, &cfg).unwrap();
    assert!(cpu.converged, "cpu gmres did not converge");

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let a_gpu = BsrGpu::upload(&device, &queue, &a).unwrap();
    let kernel = BsrAlgebraKernel::new(device, queue);

    let gpu =
        bsr_gmres_gpu(&kernel, &a_gpu, &b, cfg.tol, cfg.max_iter, cfg.restart).expect("gpu gmres");
    assert!(
        gpu.converged,
        "gpu gmres did not converge (res {})",
        gpu.residual
    );

    let mut max_err = 0.0f64;
    for (g, c) in gpu.x.iter().zip(x_cpu.iter()) {
        max_err = max_err.max((g - c).abs());
    }
    assert!(max_err < 1.0e-3, "solution mismatch, max_err = {max_err}");

    // Independently verify the GPU solution actually solves A x = b.
    let mut ax = vec![0.0; n];
    a.mv(&gpu.x, &mut ax).unwrap();
    let res: f64 = ax
        .iter()
        .zip(b.iter())
        .map(|(v, bi)| (v - bi).abs())
        .fold(0.0, f64::max);
    assert!(res < 1.0e-3, "residual ||Ax-b||_inf = {res}");
}
