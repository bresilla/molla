//! Stage 39F — full-GPU SpGEMM parity tests.
//!
//! Compares the new GPU-symbolic path (`BsrAlgebraKernel::spgemm`,
//! which routes to `spgemm_gpu_symbolic` for `n ≥ 64`) against the
//! Stage 38a host-symbolic path (`spgemm_host_symbolic`). Three sizes:
//! 64×64, 256×256, 1024×1024.
//!
//! Run with:
//! ```sh
//! make test-spgemm-symbolic-gpu
//! ```

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_kernels::{BsrAlgebraKernel, BsrGpu, SPGEMM_GPU_SYMBOLIC_CUTOVER};
use molla_math::Real;
use molla_math::sparse::BsrMatrix;

static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Cheap deterministic LCG so tests don't need a `rand` dep.
fn lcg(seed: u64) -> impl FnMut() -> Real {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 32) as u32) as Real) / (u32::MAX as Real) - 0.5
    }
}

/// Random `n × n` BSR matrix with a ~7-banded sparsity (matches the
/// Stage 38a parity test's `random_banded`). Symmetric pattern keeps
/// the C-row width predictable (~49 unique cols).
fn random_banded(n: usize, seed: u64) -> BsrMatrix {
    let mut rng = lcg(seed);
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0 + rng().abs()));
    }
    let band = 3usize;
    for i in 0..n {
        for d in 1..=band {
            let j = (i + d) % n;
            if j == i {
                continue;
            }
            let v = (rng()) * 0.1;
            triplets.push((i, j, v));
            triplets.push((j, i, v));
        }
    }
    BsrMatrix::from_scalar_triplets(n, n, &triplets).expect("triplets")
}

fn compare_csr(label: &str, n: usize, gpu_sym: &BsrMatrix, host_sym: &BsrMatrix) -> f32 {
    assert_eq!(gpu_sym.block_rows, host_sym.block_rows, "{label}: nrows");
    assert_eq!(gpu_sym.block_cols, host_sym.block_cols, "{label}: ncols");
    assert_eq!(
        gpu_sym.row_offsets, host_sym.row_offsets,
        "{label}: row_offsets mismatch",
    );
    assert_eq!(
        gpu_sym.col_idx, host_sym.col_idx,
        "{label}: col_idx mismatch (sortedness or pattern)",
    );
    assert_eq!(
        gpu_sym.values.len(),
        host_sym.values.len(),
        "{label}: values length",
    );
    let mut max_err: f32 = 0.0;
    for i in 0..gpu_sym.values.len() {
        let err = (gpu_sym.values[i] - host_sym.values[i]).abs() as f32;
        max_err = max_err.max(err);
    }
    // Loose tolerance: numeric pass uses CAS-add f32 — order of
    // accumulation is nondeterministic, so we tolerate
    // `~ulp(max_value) · max_accum_depth`. For our banded matrices
    // the max C-value is ~10 and depth ≤ 49 → ULP ~5e-6.
    assert!(
        max_err < 5e-3,
        "{label}: numeric mismatch — max_err={:.3e} (n={n})",
        max_err
    );
    max_err
}

fn run_one(n: usize, seed_a: u64, seed_b: u64) -> f32 {
    let a_cpu = random_banded(n, seed_a);
    let b_cpu = random_banded(n, seed_b);

    let engine = Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = BsrAlgebraKernel::new(device.clone(), queue.clone());
    let a_gpu = BsrGpu::upload(&device, &queue, &a_cpu).expect("a upload");
    let b_gpu = BsrGpu::upload(&device, &queue, &b_cpu).expect("b upload");

    // GPU symbolic path (dispatched automatically for n ≥ cutover).
    assert!(
        (n as u32) >= SPGEMM_GPU_SYMBOLIC_CUTOVER,
        "test n={n} must be ≥ SPGEMM_GPU_SYMBOLIC_CUTOVER={} to exercise the new path",
        SPGEMM_GPU_SYMBOLIC_CUTOVER
    );
    let c_gpu = kernel.spgemm(&a_gpu, &b_gpu).expect("spgemm gpu-symbolic");
    let c_gpu_host = c_gpu.read_back(&device, &queue).expect("readback gpu_sym");

    // Host-symbolic reference path.
    let c_ref = kernel
        .spgemm_host_symbolic(&a_gpu, &b_gpu)
        .expect("spgemm host-symbolic");
    let c_ref_host = c_ref.read_back(&device, &queue).expect("readback host_sym");

    let max_err = compare_csr(
        &format!("spgemm_symbolic_gpu n={n}"),
        n,
        &c_gpu_host,
        &c_ref_host,
    );
    eprintln!(
        "spgemm_symbolic_gpu n={n}: c_nnz={}, max_err={:.3e}",
        c_gpu_host.values.len(),
        max_err
    );
    max_err
}

#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn spgemm_symbolic_gpu_64() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    run_one(64, 0xAAA1, 0xBBB1);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn spgemm_symbolic_gpu_256() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    run_one(256, 0xAAA2, 0xBBB2);
}

#[test]
#[ignore = "needs a real wgpu adapter — run via --ignored"]
fn spgemm_symbolic_gpu_1024() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    run_one(1024, 0xAAA3, 0xBBB3);
}
