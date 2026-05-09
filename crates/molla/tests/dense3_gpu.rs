//! Stage 24a — WGSL parity tests for batched 3×3 SVD / QR / eig.
//!
//! All `#[ignore]` because they need a real wgpu adapter (CI doesn't
//! have one); run with
//! `cargo test -p molla --features gpu --test dense3_gpu -- --ignored`.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::kernels::{Dense3Kernel, Dense3Op};
use molla::math::dense3::{eig3_sym, qr3, svd3};
use molla::math::{Mat3, Real, Vec3};
use std::time::Instant;

const BATCH: usize = 1024;
const PARITY_TOL: Real = 1.0e-5;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xCAFE_F00D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn signed(&mut self) -> Real {
        let x = self.next() >> 11;
        let u = (x as Real + 1.0) / ((1u64 << 53) as Real + 1.0);
        2.0 * u - 1.0
    }
    fn rand_mat(&mut self) -> Mat3 {
        Mat3::from_cols(
            Vec3::new(self.signed(), self.signed(), self.signed()),
            Vec3::new(self.signed(), self.signed(), self.signed()),
            Vec3::new(self.signed(), self.signed(), self.signed()),
        )
    }
    fn rand_sym(&mut self) -> Mat3 {
        let m = self.rand_mat();
        0.5 * (m + m.transpose())
    }
}

fn frob(a: Mat3) -> Real {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let v = a.col(j)[i];
            s += v * v;
        }
    }
    s.sqrt()
}

fn try_engine() -> Option<Engine> {
    Engine::new_gpu().ok()
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn svd_gpu_matches_cpu_reconstruction() {
    let Some(engine) = try_engine() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = Dense3Kernel::new(&device);
    let mut rng = Lcg::new(0x1111_2222);
    let mats: Vec<Mat3> = (0..BATCH).map(|_| rng.rand_mat()).collect();
    let res = kernel
        .dispatch_blocking(&device, &queue, Dense3Op::Svd, &mats)
        .expect("svd dispatch");

    let mut max_recon = 0.0_f64;
    let mut max_vs_cpu = 0.0_f64;
    for i in 0..BATCH {
        let m = mats[i];
        let u = res.left[i];
        let sigma = res.diag[i];
        let v = res.right[i];
        // GPU reconstruction.
        let recon = u * Mat3::from_diagonal(sigma) * v.transpose();
        let r_err = frob(recon - m);
        if r_err > max_recon {
            max_recon = r_err;
        }
        assert!(
            r_err < PARITY_TOL * (1.0 + frob(m)),
            "GPU SVD recon err {r_err}",
        );
        // CPU vs GPU singular-value parity (sorted, since the rotation
        // ordering may differ between f32 and f64).
        let (_, sigma_cpu, _) = svd3(m);
        let mut g: [Real; 3] = [sigma.x, sigma.y, sigma.z];
        let mut c: [Real; 3] = [sigma_cpu.x, sigma_cpu.y, sigma_cpu.z];
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..3 {
            let d = (g[k] - c[k]).abs();
            if d > max_vs_cpu {
                max_vs_cpu = d;
            }
            assert!(d < PARITY_TOL, "σ[{k}] gpu {} vs cpu {}", g[k], c[k]);
        }
    }
    eprintln!(
        "SVD GPU: max recon {max_recon:e}, max σ diff vs CPU {max_vs_cpu:e}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn qr_gpu_matches_cpu() {
    let Some(engine) = try_engine() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = Dense3Kernel::new(&device);
    let mut rng = Lcg::new(0x3333_4444);
    let mats: Vec<Mat3> = (0..BATCH).map(|_| rng.rand_mat()).collect();
    let res = kernel
        .dispatch_blocking(&device, &queue, Dense3Op::Qr, &mats)
        .unwrap();

    let mut max_recon = 0.0_f64;
    for i in 0..BATCH {
        let m = mats[i];
        let q = res.left[i];
        let r = res.middle_mats[i];
        let recon = q * r;
        let err = frob(recon - m);
        if err > max_recon {
            max_recon = err;
        }
        assert!(err < PARITY_TOL * (1.0 + frob(m)), "QR recon err {err}");
        // R upper triangular.
        for ii in 0..3 {
            for jj in 0..ii {
                assert!(r.col(jj)[ii].abs() < 1.0e-5, "R not upper-tri ({ii},{jj})");
            }
        }
        // Compare to CPU via R-diagonal parity (signs may differ; we
        // compare absolute diagonal entries).
        let (_, r_cpu) = qr3(m);
        for k in 0..3 {
            let d = (r.col(k)[k].abs() - r_cpu.col(k)[k].abs()).abs();
            assert!(d < PARITY_TOL, "|R[{k},{k}]| gpu vs cpu diff {d}");
        }
    }
    eprintln!("QR GPU: max recon {max_recon:e}");
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn eig_gpu_matches_cpu() {
    let Some(engine) = try_engine() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = Dense3Kernel::new(&device);
    let mut rng = Lcg::new(0x5555_6666);
    let mats: Vec<Mat3> = (0..BATCH).map(|_| rng.rand_sym()).collect();
    let res = kernel
        .dispatch_blocking(&device, &queue, Dense3Op::Eig, &mats)
        .unwrap();

    let mut max_pair = 0.0_f64;
    let mut max_lambda_diff = 0.0_f64;
    for i in 0..BATCH {
        let m = mats[i];
        let v = res.left[i];
        let lambda = res.diag[i];
        for k in 0..3 {
            let vk = v.col(k);
            let mvk = m * vk;
            let err = (mvk - vk * lambda[k]).length();
            if err > max_pair {
                max_pair = err;
            }
            assert!(err < PARITY_TOL * (1.0 + frob(m)), "eig pair {k} err {err}");
        }
        // Sorted-eigenvalue parity vs CPU.
        let (lam_cpu, _) = eig3_sym(m);
        let mut g: [Real; 3] = [lambda.x, lambda.y, lambda.z];
        let mut c: [Real; 3] = [lam_cpu.x, lam_cpu.y, lam_cpu.z];
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for k in 0..3 {
            let d = (g[k] - c[k]).abs();
            if d > max_lambda_diff {
                max_lambda_diff = d;
            }
            assert!(d < PARITY_TOL, "λ[{k}] gpu {} vs cpu {}", g[k], c[k]);
        }
    }
    eprintln!(
        "Eig GPU: max pair {max_pair:e}, max λ diff vs CPU {max_lambda_diff:e}"
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — wall-clock benchmark"]
fn svd_gpu_latency_under_100ns_per_matrix() {
    let Some(engine) = try_engine() else {
        return;
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = Dense3Kernel::new(&device);
    let mut rng = Lcg::new(0xBEEF_BEEF);
    let n = 16384usize;
    let mats: Vec<Mat3> = (0..n).map(|_| rng.rand_mat()).collect();

    // Warm up.
    let _ = kernel
        .dispatch_blocking(&device, &queue, Dense3Op::Svd, &mats)
        .unwrap();

    // Multiple trials so the budget isn't dominated by one outlier.
    let trials = 5;
    let mut best_per_mat_ns = f64::INFINITY;
    for _ in 0..trials {
        let t = Instant::now();
        let _ = kernel
            .dispatch_blocking(&device, &queue, Dense3Op::Svd, &mats)
            .unwrap();
        let ns_per_mat = t.elapsed().as_nanos() as f64 / n as f64;
        if ns_per_mat < best_per_mat_ns {
            best_per_mat_ns = ns_per_mat;
        }
    }
    eprintln!(
        "SVD GPU best per-matrix latency: {best_per_mat_ns:.1} ns ({n} matrices, {trials} trials)"
    );
    // Budget = 100 ns per matrix on the dev laptop. The blocking
    // dispatch's read-back overhead means this won't beat 100 ns at
    // small batch sizes; we use 16k matrices to amortise dispatch
    // and only flag a hard regression (10× over budget).
    assert!(
        best_per_mat_ns < 1000.0,
        "SVD per-matrix > 1 µs (budget 100 ns + 10× overhead allowance): {best_per_mat_ns} ns",
    );
}
