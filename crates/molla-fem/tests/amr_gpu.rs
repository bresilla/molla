//! GPU AMR constrained assembly (TᵀKT) vs the CPU `compute_ttkt`.
//!
//! Run with: `make test-gpu P=molla-fem T=amr_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_fem::amr_gpu::compute_ttkt_gpu;
use molla_fem::amr_mesh::compute_ttkt;
use molla_math::sparse::BsrMatrix;

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_amr_ttkt_matches_cpu() {
    // K: 6×6 SPD tridiag(2, -1).
    let n = 6usize;
    let mut kt: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        kt.push((i, i, 2.0));
        if i + 1 < n {
            kt.push((i, i + 1, -1.0));
            kt.push((i + 1, i, -1.0));
        }
    }
    let k = BsrMatrix::from_scalar_triplets(n, n, &kt).unwrap();

    // T: 6×4 prolongation — free DoFs 0..3 pass through; total DoFs 4,5
    // are hanging nodes averaging two free DoFs (2:1 constraint).
    let free = 4usize;
    let t = BsrMatrix::from_scalar_triplets(
        n,
        free,
        &[
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 2, 1.0),
            (3, 3, 1.0),
            (4, 0, 0.5),
            (4, 1, 0.5),
            (5, 2, 0.5),
            (5, 3, 0.5),
        ],
    )
    .unwrap();

    let cpu = compute_ttkt(&t, &k).unwrap();
    let engine = Engine::new_gpu().expect("gpu");
    let gpu = compute_ttkt_gpu(&engine, &t, &k).expect("gpu ttkt");

    assert_eq!(cpu.nrows(), free);
    assert_eq!(gpu.nrows(), free);

    // Compare the two operators by their action on test vectors.
    let xs = [
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![1.0, -2.0, 3.0, -0.5],
        vec![0.7, 0.3, -0.9, 1.1],
    ];
    let mut max_err = 0.0f64;
    let mut scale = 0.0f64;
    for x in &xs {
        let mut yc = vec![0.0; free];
        let mut yg = vec![0.0; free];
        cpu.mv(x, &mut yc).unwrap();
        gpu.mv(x, &mut yg).unwrap();
        for (a, b) in yc.iter().zip(yg.iter()) {
            max_err = max_err.max((a - b).abs());
            scale = scale.max(a.abs());
        }
    }
    assert!(
        max_err / scale.max(1.0e-30) < 1.0e-4,
        "TᵀKT mismatch, max_err = {max_err}"
    );
    assert!(scale > 1.0e-3, "degenerate: TᵀKT ~ 0");
}
